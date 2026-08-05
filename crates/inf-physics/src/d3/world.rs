//! [`PhysicsWorld3D`]: the fixed-step 3D rigid-body world. The `d3` mirror of
//! [`crate::d2::PhysicsWorld2D`].

use glam::{DQuat, DVec2, DVec3};
use parry3d_f64::shape::{HeightField, HeightFieldCellStatus, HeightFieldFlags, TriMeshFlags};
use parry3d_f64::utils::Array2;
use rapier3d_f64::dynamics::CoefficientCombineRule;
use rapier3d_f64::geometry::{Group, InteractionGroups, InteractionTestMode};
use rapier3d_f64::prelude::{
    Aabb, ActiveCollisionTypes, ActiveEvents, BroadPhaseBvh, CCDSolver, ColliderBuilder,
    ColliderSet, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
    NarrowPhase, PhysicsPipeline, QueryFilter, QueryPipeline, Ray, RigidBodyBuilder, RigidBodySet,
    RigidBodyType, SharedShape,
};

use super::character::{CharacterMove3D, CharacterMover3D};
use super::events::{ContactEvent3D, EventCollector};
use super::joint::{JointDesc3D, JointId3D};
use super::query::RayHit3D;
use super::{BodyId3D, ColliderId3D};
use crate::filtering::{CollisionLayers, CombineRule};

/// Map facade collision layers onto rapier's symmetric interaction groups.
pub(crate) fn to_interaction_groups(layers: CollisionLayers) -> InteractionGroups {
    InteractionGroups::new(
        Group::from_bits_truncate(layers.memberships),
        Group::from_bits_truncate(layers.filter),
        InteractionTestMode::And,
    )
}

/// Map a facade combine rule onto rapier's `CoefficientCombineRule`.
pub(crate) fn to_combine_rule(rule: CombineRule) -> CoefficientCombineRule {
    match rule {
        CombineRule::Average => CoefficientCombineRule::Average,
        CombineRule::Min => CoefficientCombineRule::Min,
        CombineRule::Multiply => CoefficientCombineRule::Multiply,
        CombineRule::Max => CoefficientCombineRule::Max,
    }
}

/// The kind of a rigid body. Mirrors [`crate::d2::BodyKind`] at 3D.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind3D {
    /// Never moved by the solver; infinite mass (floors, walls, level geometry).
    Static,
    /// Moved only by the facade (`set_body_translation`/`_rotation`), pushes
    /// dynamic bodies but is not pushed back. Position-based kinematic — the kind
    /// a character mover or moving platform uses.
    Kinematic,
    /// Fully simulated: gravity, forces, impulses, and contacts move it.
    Dynamic,
}

impl BodyKind3D {
    fn to_rapier(self) -> RigidBodyType {
        match self {
            BodyKind3D::Static => RigidBodyType::Fixed,
            BodyKind3D::Kinematic => RigidBodyType::KinematicPositionBased,
            BodyKind3D::Dynamic => RigidBodyType::Dynamic,
        }
    }
}

/// A 3D collider shape. Half-extents / radii are in world (f64) units.
///
/// Unlike [`crate::d2::ColliderShape2D`] this is `Clone`, not `Copy`, because the
/// [`Trimesh`](ColliderShape3D::Trimesh) variant owns its vertex/index buffers —
/// the seam static-mesh colliders (P12) plug into.
#[derive(Clone, Debug, PartialEq)]
pub enum ColliderShape3D {
    /// A sphere of the given radius.
    Sphere { radius: f64 },
    /// An axis-aligned box given by its half-extents (x, y, z).
    Box { half_extents: DVec3 },
    /// A vertical capsule: a segment of length `2 * half_height` along local Y,
    /// swept by `radius`.
    Capsule { half_height: f64, radius: f64 },
    /// A triangle mesh built from a vertex buffer + triangle index buffer — the
    /// hook for later mesh colliders. Static/kinematic use only in practice
    /// (rapier cannot give a trimesh a well-defined mass).
    Trimesh {
        vertices: Vec<DVec3>,
        indices: Vec<[u32; 3]>,
    },
    /// The **convex hull** of a point cloud (P22.2): the solid body of the
    /// smallest convex shape containing every point. The points need not already
    /// be a hull — rapier computes it — and interior points are simply ignored.
    ///
    /// # Why this variant exists at all, given [`Trimesh`](Self::Trimesh)
    ///
    /// Because **a convex hull has a well-defined mass and a trimesh does not.**
    /// A triangle soup is a *surface*: rapier can raycast and collide against it,
    /// but it cannot integrate a volume over it, so its mass properties come back
    /// zero and a dynamic body built from one has no inertia to simulate. That is
    /// why the `Trimesh` doc above says "static/kinematic use only in practice",
    /// and it is exactly the wall the fracture pipeline would hit: a chunk of a
    /// shattered wall is the archetypal *dynamic* body, and it needs to weigh
    /// something.
    ///
    /// A hull is a *solid*. rapier knows its volume, centre of mass and inertia
    /// tensor exactly (see [`volume_m3`](Self::volume_m3)), so
    /// `density × volume` is a real mass and a chunk falls, tumbles and rests
    /// like an object rather than like a ghost.
    ///
    /// The trade is stated: a hull cannot be concave, so a chunk with a hollow or
    /// a re-entrant notch collides as if it were filled in. For pre-fractured
    /// Voronoi chunks that is not an approximation at all — a Voronoi cell
    /// intersected with a convex hull **is** convex — which is why the P22.2 cook
    /// emits hull point sets in the first place.
    ///
    /// A degenerate point set (fewer than four points, or points that are
    /// collinear/coplanar and so bound no volume) is **refused**, not panicked
    /// on: the shape builder returns `None` and
    /// [`PhysicsWorld3D::add_collider`] skips the collider, leaving the body
    /// intact. Producers should ask [`convex_hull_is_buildable`] first — it is
    /// the same door — rather than discover the refusal at spawn time.
    ConvexHull { points: Vec<DVec3> },
    /// A **regular-grid height field** on the local XZ plane, with per-cell
    /// removal for holes (P22.3). The shape a terrain tile becomes.
    ///
    /// # Why this variant and not a [`Trimesh`](Self::Trimesh) per tile
    ///
    /// A terrain tile is 256 × 256 samples. As a trimesh that is **130 050
    /// triangles** plus a BVH over them, per tile, rebuilt whenever a sculpt or a
    /// runtime carve moves a sample — and a broad-phase query then walks a tree to
    /// find the one quad under a body's foot. As a height field it is the same
    /// 65 536 numbers with the topology *implied*: parry indexes the cell under a
    /// point arithmetically, so a contact query is O(1) in the sample count and
    /// the memory is the heights and nothing else. It is also the only one of the
    /// two that can express a hole without re-triangulating: a removed cell is two
    /// status bits.
    ///
    /// A height field is a **surface**, so — exactly like `Trimesh` —
    /// [`volume_m3`](Self::volume_m3) is `None` and it is static/kinematic-only in
    /// practice. That costs nothing here: ground does not fall.
    ///
    /// # The local frame is CENTRED, and the heights are metres
    ///
    /// parry's height field spans `[-span.x/2, +span.x/2] × [-span.y/2, +span.y/2]`
    /// (x and z) about its **own origin**, so the collider must be placed at the
    /// tile's **centre**, not at its corner sample. The Y scale is fixed at `1`, so
    /// [`heights`](Self::Heightfield::heights) are used as local metres directly and a
    /// body placed at the tile's `origin.y` puts sample `h` at world `origin.y + h`
    /// — which is exactly `TerrainTile::world_height`. Nothing is rescaled, so the
    /// surface a body stands on and the surface `height_at` reports are the same
    /// arithmetic.
    Heightfield {
        /// Height samples along local **X** (columns). Must be ≥ 2.
        samples_x: u32,
        /// Height samples along local **Z** (rows). Must be ≥ 2.
        samples_z: u32,
        /// `samples_x · samples_z` heights in **metres**, row-major in Z:
        /// index `j · samples_x + i` is the sample `i` steps along X and `j` steps
        /// along Z. This is `TerrainTile`'s own layout, kept verbatim so the
        /// descriptor is a clone rather than a transform (parry wants the
        /// transpose; the shape builder does that conversion in one place).
        heights: Vec<f32>,
        /// `(samples_x − 1) · (samples_z − 1)` bits, row-major in Z: `true` means
        /// the **cell** has no surface and both of its triangles are removed.
        ///
        /// **Empty means "no cell is removed"** — the sparse default, so an
        /// un-holed tile costs nothing here (the `TerrainTile::holes` convention,
        /// one container down).
        removed_cells: Vec<bool>,
        /// World size of the field along local X and Z, in metres.
        span: DVec2,
    },
}

impl ColliderShape3D {
    /// Build the rapier shape. Returns `None` when the shape's buffers cannot
    /// make a collider — a degenerate `Trimesh` (empty / non-manifold) or a
    /// degenerate `ConvexHull` (see [`convex_hull_is_buildable`]) — so
    /// [`PhysicsWorld3D::add_collider`] can refuse it rather than panic.
    pub(crate) fn to_shared(&self) -> Option<SharedShape> {
        Some(match self {
            ColliderShape3D::Sphere { radius } => SharedShape::ball(*radius),
            ColliderShape3D::Box { half_extents } => {
                SharedShape::cuboid(half_extents.x, half_extents.y, half_extents.z)
            }
            ColliderShape3D::Capsule {
                half_height,
                radius,
            } => SharedShape::capsule_y(*half_height, *radius),
            ColliderShape3D::Trimesh { vertices, indices } => {
                if vertices.is_empty() || indices.is_empty() {
                    return None;
                }
                let verts: Vec<DVec3> = vertices.clone();
                // The same internal-edge fix the height field above needs, and
                // for the same reason — a trimesh is also "one triangle at a
                // time" to the narrow phase. This is the surface a P21 voxel
                // cave FLOOR is made of, which is exactly where fracture debris
                // that falls through a hole in the terrain lands, so debris
                // skittering on a flat cave floor is the same defect one crate
                // over. `FIX_INTERNAL_EDGES` implies `MERGE_DUPLICATE_VERTICES`
                // (parry folds it in), which is harmless here: the Surface-Nets
                // mesher already emits one vertex per cell.
                SharedShape::trimesh_with_flags(
                    verts,
                    indices.clone(),
                    TriMeshFlags::FIX_INTERNAL_EDGES,
                )
                .ok()?
            }
            ColliderShape3D::ConvexHull { points } => hull_shape(points)?,
            ColliderShape3D::Heightfield {
                samples_x,
                samples_z,
                heights,
                removed_cells,
                span,
            } => heightfield_shape(*samples_x, *samples_z, heights, removed_cells, *span)?,
        })
    }

    /// The shape's **volume in m³**, or `None` for a shape that has no
    /// well-defined one.
    ///
    /// Read off the *same* mass properties the solver integrates (at unit
    /// density, where mass and volume are numerically equal) rather than from a
    /// second hand-written per-shape volume formula that could drift from it —
    /// the P20.2 buoyancy precedent, where the displaced volume is likewise taken
    /// from rapier's own numbers.
    ///
    /// [`Trimesh`](Self::Trimesh) returns `None`, and that is the whole reason
    /// [`ConvexHull`](Self::ConvexHull) exists: a surface has no volume, so a
    /// trimesh body's mass would be zero. Callers giving a chunk its mass
    /// (`density_kg_m3 × volume`) must treat `None` as "this cannot be a dynamic
    /// body".
    pub fn volume_m3(&self) -> Option<f64> {
        if matches!(
            self,
            ColliderShape3D::Trimesh { .. } | ColliderShape3D::Heightfield { .. }
        ) {
            return None;
        }
        let shape = self.to_shared()?;
        // Unit density ⇒ the reported mass IS the volume, in m³.
        let props: parry3d_f64::mass_properties::MassProperties = shape.mass_properties(1.0);
        let v = props.mass();
        (v.is_finite() && v > 0.0).then_some(v)
    }
}

/// The smallest volume, m³, a convex hull must bound to be accepted as a
/// collider — a cubic micrometre.
///
/// Not zero, and the gap matters. `parry`'s hull builder happily accepts a
/// **coplanar** cloud and returns a polyhedron of zero thickness (measured, not
/// assumed — `degenerate_point_sets_refuse_cleanly` is what found this), and a
/// *near*-coplanar cloud returns one whose volume is f64 noise. Either would
/// insert a collider that a dynamic body then has to be given a mass from, and
/// `density × 0` is the failure this constant exists to make impossible.
///
/// The floor is absolute rather than relative because the shapes are in metres
/// (architecture rule 6): 1e-12 m³ is a thousandth of a cubic millimetre —
/// below anything that can be a game object, and many orders above the rounding
/// noise of a hull built from metre-scale coordinates.
pub const MIN_HULL_VOLUME_M3: f64 = 1e-12;

/// Build a convex-hull shape from `points`, refusing anything that does not
/// bound a real volume. The single door both [`ColliderShape3D::to_shared`] and
/// [`convex_hull_is_buildable`] go through, so the pre-check and the build can
/// never disagree.
fn hull_shape(points: &[DVec3]) -> Option<SharedShape> {
    // `parry3d_f64::shape::ConvexPolyhedron::from_convex_hull` is what this
    // calls; naming it here is why the crate is a direct dependency.
    let shape = SharedShape::convex_hull(points)?;
    let props: parry3d_f64::mass_properties::MassProperties = shape.mass_properties(1.0);
    let volume = props.mass(); // unit density ⇒ mass == volume
    (volume.is_finite() && volume > MIN_HULL_VOLUME_M3).then_some(shape)
}

/// Build a parry height field from a [`ColliderShape3D::Heightfield`]'s buffers,
/// refusing anything that cannot make a surface.
///
/// # The two index conversions, stated once
///
/// 1. **Transpose.** Our buffer is row-major in Z (`j · samples_x + i`, the
///    `TerrainTile` layout). parry's [`Array2`] is column-major with the *row*
///    index advancing Z and the *column* index advancing X, i.e. `i + j · nrows`
///    where `i` is the Z index and `j` the X index. So `nrows = samples_z`,
///    `ncols = samples_x`, and the two layouts are transposes of each other.
///    Doing that here, once, is why the descriptor can stay a clone of the tile's
///    own `heights`.
/// 2. **Cell status.** parry's cell `(i, j)` is the quad between samples
///    `(i, j) .. (i+1, j+1)` — `i` in Z, `j` in X — so our row-major-in-Z cell
///    index `jz · (samples_x − 1) + ix` maps to `set_cell_status(jz, ix, …)`.
///
/// Refusals (returning `None`, so [`PhysicsWorld3D::add_collider`] skips the
/// collider and leaves the body intact) are: fewer than two samples on either
/// axis, a `heights` buffer that is not exactly `samples_x · samples_z`, a
/// `removed_cells` buffer that is neither empty nor exactly the cell count, a
/// non-finite or non-positive span, and a non-finite height. Every one of them is
/// a producer bug that would otherwise be an assert inside parry.
fn heightfield_shape(
    samples_x: u32,
    samples_z: u32,
    heights: &[f32],
    removed_cells: &[bool],
    span: DVec2,
) -> Option<SharedShape> {
    if samples_x < 2 || samples_z < 2 {
        return None;
    }
    let nx = samples_x as usize;
    let nz = samples_z as usize;
    if heights.len() != nx * nz {
        return None;
    }
    let cells = (nx - 1) * (nz - 1);
    if !removed_cells.is_empty() && removed_cells.len() != cells {
        return None;
    }
    if !span.x.is_finite() || !span.y.is_finite() || span.x <= 0.0 || span.y <= 0.0 {
        return None;
    }
    if heights.iter().any(|h| !h.is_finite()) {
        return None;
    }
    // Transpose into parry's (rows = Z, cols = X) column-major array.
    let mut data = vec![0.0_f64; nx * nz];
    for jz in 0..nz {
        for ix in 0..nx {
            data[jz + ix * nz] = heights[jz * nx + ix] as f64;
        }
    }
    let array = Array2::new(nz, nx, data);
    // Y scale 1: the heights ARE metres (see the variant docs).
    //
    // ── FIX_INTERNAL_EDGES IS NOT OPTIONAL ──────────────────────────────────
    //
    // A height field is two triangles per cell, and without this flag the narrow
    // phase resolves a contact against **one triangle's** face normal with no
    // knowledge of its neighbours. A body sliding across a cell boundary
    // therefore hits the *edge* of the next triangle and is answered with that
    // edge's normal, which points partly upward even on ground that is dead flat.
    //
    // Measured, not feared: a 0.25 m sphere sliding at 12 m/s across FLAT terrain
    // takes **5 upward kicks, peaking at 0.105 m/s**, over 300 steps of contact —
    // upward, on flat ground, from nothing but triangulation. With the flag: zero.
    // (`a_sphere_sliding_on_flat_ground_is_never_kicked_upward` is where those
    // numbers come from and where they are re-measured.) Every character, prop and
    // piece of debris that touches terrain is affected, which is everything this
    // phase exists to make land.
    //
    // The cost is one O(n) pseudo-normal pass at build time, paid on the same
    // change-stamped path that already refuses to rebuild an unsculpted tile.
    let mut field = HeightField::with_flags(
        array,
        DVec3::new(span.x, 1.0, span.y),
        HeightFieldFlags::FIX_INTERNAL_EDGES,
    );
    if !removed_cells.is_empty() {
        for jz in 0..(nz - 1) {
            for ix in 0..(nx - 1) {
                if removed_cells[jz * (nx - 1) + ix] {
                    field.set_cell_status(jz, ix, HeightFieldCellStatus::CELL_REMOVED);
                }
            }
        }
    }
    Some(SharedShape::new(field))
}

/// Whether `points` can become a [`ColliderShape3D::ConvexHull`] — i.e. whether
/// they bound a volume of at least `MIN_HULL_VOLUME_M3`.
///
/// Refuses fewer than four points, collinear points, and **coplanar** points: a
/// flat slab of vertices has no interior and is not a solid. Note that the last
/// of those is *our* refusal, not parry's — parry's builder returns a zero-
/// thickness polyhedron for a coplanar cloud rather than `None`, so a producer
/// that only checked "did the builder succeed" would ship massless chunks.
///
/// Exposed so a *producer* of hull point sets (the P22.2 fracture cook, which
/// must not ship a chunk nothing can collide with) can check before it writes
/// bytes, instead of discovering it in a player. It runs the same
/// `hull_shape` the collider build runs, so a "yes" here is a guarantee rather
/// than an estimate.
pub fn convex_hull_is_buildable(points: &[DVec3]) -> bool {
    hull_shape(points).is_some()
}

/// Description of a collider to attach to a body. Build with [`ColliderDesc3D::new`]
/// and the fluent setters. Every collider the facade creates has collision-event
/// reporting enabled so the [`PhysicsWorld3D::drain_contact_events`] drain sees it.
///
/// `Clone` (not `Copy`) because the shape may own a trimesh buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct ColliderDesc3D {
    /// The shape.
    pub shape: ColliderShape3D,
    /// Coulomb friction coefficient.
    pub friction: f64,
    /// Bounciness in `[0, 1]`.
    pub restitution: f64,
    /// Mass density (drives a dynamic body's mass/inertia).
    pub density: f64,
    /// When `true`, the collider detects overlaps but generates no contact
    /// forces — a trigger volume.
    pub sensor: bool,
    /// Offset of the collider from its parent body's origin, in the body frame.
    pub local_translation: DVec3,
    /// Bitmask collision layers (P12.1). Default = interact with everything.
    pub layers: CollisionLayers,
    /// How this collider's friction combines with a contacting collider's (P12.1).
    pub friction_combine: CombineRule,
    /// How this collider's restitution combines with a contacting collider's.
    pub restitution_combine: CombineRule,
}

impl ColliderDesc3D {
    /// A solid collider with engine-default material (friction 0.5, no
    /// restitution, unit density, all layers, `Average` combine rules).
    pub fn new(shape: ColliderShape3D) -> Self {
        Self {
            shape,
            friction: 0.5,
            restitution: 0.0,
            density: 1.0,
            sensor: false,
            local_translation: DVec3::ZERO,
            layers: CollisionLayers::default(),
            friction_combine: CombineRule::default(),
            restitution_combine: CombineRule::default(),
        }
    }

    /// Set the friction coefficient.
    pub fn friction(mut self, friction: f64) -> Self {
        self.friction = friction;
        self
    }
    /// Set the restitution (bounciness).
    pub fn restitution(mut self, restitution: f64) -> Self {
        self.restitution = restitution;
        self
    }
    /// Set the mass density.
    pub fn density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }
    /// Mark this collider a sensor (trigger).
    pub fn sensor(mut self, sensor: bool) -> Self {
        self.sensor = sensor;
        self
    }
    /// Offset the collider from its body's origin, in the body frame.
    pub fn local_translation(mut self, offset: DVec3) -> Self {
        self.local_translation = offset;
        self
    }
    /// Set the collision layers (membership + filter masks).
    pub fn layers(mut self, layers: CollisionLayers) -> Self {
        self.layers = layers;
        self
    }
    /// Set the friction combine rule.
    pub fn friction_combine(mut self, rule: CombineRule) -> Self {
        self.friction_combine = rule;
        self
    }
    /// Set the restitution combine rule.
    pub fn restitution_combine(mut self, rule: CombineRule) -> Self {
        self.restitution_combine = rule;
        self
    }
}

/// A fixed-step 3D physics world wrapping `rapier3d-f64`.
///
/// Simulation is advanced only by [`step`](Self::step), which takes the timestep
/// as an argument — the world never reads a wall clock, so replaying the same
/// calls reproduces the same result bit-for-bit (see the crate-level determinism
/// note). Contact events accumulate across steps and are read with
/// [`drain_contact_events`](Self::drain_contact_events); scene queries and the
/// character mover run against the post-step state.
pub struct PhysicsWorld3D {
    gravity: DVec3,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,

    /// A broad-phase BVH kept purely for scene queries / the character mover,
    /// rebuilt lazily from the current colliders whenever the world changed.
    query_bvh: BroadPhaseBvh,
    query_dirty: bool,

    pending_contacts: Vec<ContactEvent3D>,
}

impl PhysicsWorld3D {
    /// A new, empty world with the given gravity (world units / s²). A typical
    /// 3D world uses `DVec3::new(0.0, -9.81, 0.0)`.
    pub fn new(gravity: DVec3) -> Self {
        Self {
            gravity,
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_bvh: BroadPhaseBvh::new(),
            query_dirty: false,
            pending_contacts: Vec::new(),
        }
    }

    /// The current gravity vector.
    pub fn gravity(&self) -> DVec3 {
        self.gravity
    }

    /// Replace the gravity vector (takes effect on the next [`step`](Self::step)).
    pub fn set_gravity(&mut self, gravity: DVec3) {
        self.gravity = gravity;
    }

    // ── Simulation ──────────────────────────────────────────────────────────

    /// Advance the simulation by `dt` seconds. `dt` must be the caller's fixed
    /// timestep (see [`FixedStepper`](crate::FixedStepper)); passing a wall-clock
    /// delta here would destroy determinism.
    ///
    /// Contact events produced this step are appended to the internal buffer;
    /// read them with [`drain_contact_events`](Self::drain_contact_events).
    pub fn step(&mut self, dt: f64) {
        self.integration_parameters.dt = dt;
        let collector = EventCollector::default();
        self.physics_pipeline.step(
            self.gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(), // no physics hooks
            &collector,
        );
        collector.append_into(&mut self.pending_contacts);
        self.query_dirty = true;
    }

    /// Remove and return all contact events accumulated since the last drain,
    /// sorted deterministically by `(collider_a, collider_b, phase)`. Pair member
    /// order is canonicalized (`collider_a <= collider_b`) so a given pair always
    /// reports with the same orientation. Sensor overlaps carry `sensor == true`.
    pub fn drain_contact_events(&mut self) -> Vec<ContactEvent3D> {
        let mut out = std::mem::take(&mut self.pending_contacts);
        out.sort_unstable();
        out
    }

    // ── Bodies ──────────────────────────────────────────────────────────────

    /// Create a rigid body at `position` with `rotation`. Returns its stable
    /// handle.
    pub fn add_body(&mut self, kind: BodyKind3D, position: DVec3, rotation: DQuat) -> BodyId3D {
        let rb = RigidBodyBuilder::new(kind.to_rapier())
            .translation(position)
            .rotation(rotation.to_scaled_axis())
            .build();
        let handle = self.bodies.insert(rb);
        self.query_dirty = true;
        BodyId3D(handle)
    }

    /// Destroy a body and all colliders attached to it. Returns `false` if the
    /// handle was already invalid.
    pub fn remove_body(&mut self, body: BodyId3D) -> bool {
        let removed = self
            .bodies
            .remove(
                body.0,
                &mut self.islands,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                true,
            )
            .is_some();
        if removed {
            self.query_dirty = true;
        }
        removed
    }

    /// Does this body still exist?
    pub fn contains_body(&self, body: BodyId3D) -> bool {
        self.bodies.contains(body.0)
    }

    /// Every live body handle, sorted deterministically. Handy for snapshotting
    /// the world (e.g. the determinism harness).
    pub fn body_ids(&self) -> Vec<BodyId3D> {
        let mut ids: Vec<BodyId3D> = self.bodies.iter().map(|(h, _)| BodyId3D(h)).collect();
        ids.sort_unstable();
        ids
    }

    /// The body's world-space translation.
    pub fn body_translation(&self, body: BodyId3D) -> Option<DVec3> {
        self.bodies.get(body.0).map(|rb| rb.translation())
    }

    /// The body's world-space orientation.
    pub fn body_rotation(&self, body: BodyId3D) -> Option<DQuat> {
        self.bodies.get(body.0).map(|rb| *rb.rotation())
    }

    /// Teleport the body's translation (wakes it).
    pub fn set_body_translation(&mut self, body: BodyId3D, translation: DVec3) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_translation(translation, true);
            self.query_dirty = true;
            true
        } else {
            false
        }
    }

    /// Set the body's orientation (wakes it).
    pub fn set_body_rotation(&mut self, body: BodyId3D, rotation: DQuat) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_rotation(rotation, true);
            self.query_dirty = true;
            true
        } else {
            false
        }
    }

    /// The body's linear velocity.
    pub fn body_linvel(&self, body: BodyId3D) -> Option<DVec3> {
        self.bodies.get(body.0).map(|rb| rb.linvel())
    }

    /// The body's angular velocity (rad/s, about each world axis).
    pub fn body_angvel(&self, body: BodyId3D) -> Option<DVec3> {
        self.bodies.get(body.0).map(|rb| rb.angvel())
    }

    /// Set the body's linear velocity.
    pub fn set_body_linvel(&mut self, body: BodyId3D, linvel: DVec3) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_linvel(linvel, true);
            true
        } else {
            false
        }
    }

    /// Set the body's angular velocity (rad/s about each world axis).
    pub fn set_body_angvel(&mut self, body: BodyId3D, angvel: DVec3) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_angvel(angvel, true);
            true
        } else {
            false
        }
    }

    /// The body's mass, kg — what rapier derived from its colliders' shapes and
    /// densities. `0` for a static body (infinite mass is reported as zero
    /// inverse mass) and for a massless one.
    ///
    /// Exposed for P20.2's buoyancy: the displaced volume of a floating body is
    /// `mass / density`, which reads rapier's own **exact per-shape** volume
    /// rather than a second, hand-written volume table beside it. A cuboid, a
    /// ball and a capsule all have closed-form volumes in rapier already, and two
    /// copies of a formula are two chances to disagree.
    pub fn body_mass(&self, body: BodyId3D) -> Option<f64> {
        self.bodies.get(body.0).map(|rb| rb.mass())
    }

    /// Add a force applied at `point` (world space). A force off the centre of
    /// mass produces a torque as well as an acceleration, which is the whole
    /// reason P20.2's buoyancy is sampled at several points rather than one: a
    /// body tipped on a wave has more of itself under water on one side, and that
    /// difference is the righting moment.
    ///
    /// **TWO LAWS, both paid for once, both about rapier's force model.**
    ///
    /// 1. **A rapier force is persistent.** It keeps being applied at every step
    ///    until [`reset_forces`](Self::reset_forces) clears it — unlike an
    ///    impulse, which is consumed. A per-step force re-added each step
    ///    therefore accumulates without bound, and a floating box leaves the
    ///    atmosphere in about fifteen seconds. Anything that re-computes a force
    ///    every step must clear the previous one first.
    /// 2. **A force is not an impulse of `F · dt`, for POSITION.** rapier
    ///    substeps: it integrates gravity (and forces) once per substep, so with
    ///    `N` substeps a front-loaded impulse of `F · dt` conserves the velocity
    ///    exactly and still drifts the position by `g · dt² · (N−1) / 2N` every
    ///    step — a neutrally buoyant body that should hover rises about a
    ///    millimetre per step. Buoyancy is a **force**, and applying it as one is
    ///    what makes it cancel gravity substep for substep.
    pub fn apply_force_at_point(&mut self, body: BodyId3D, force: DVec3, point: DVec3) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.add_force_at_point(force, point, true);
            true
        } else {
            false
        }
    }

    /// Add a force applied at the center of mass. Forces accumulate and are
    /// consumed each [`step`](Self::step).
    pub fn apply_force(&mut self, body: BodyId3D, force: DVec3) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.add_force(force, true);
            true
        } else {
            false
        }
    }

    /// Apply an instantaneous linear impulse (immediately changes velocity).
    pub fn apply_impulse(&mut self, body: BodyId3D, impulse: DVec3) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.apply_impulse(impulse, true);
            true
        } else {
            false
        }
    }

    /// Add a torque (accumulates, consumed each step).
    pub fn apply_torque(&mut self, body: BodyId3D, torque: DVec3) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.add_torque(torque, true);
            true
        } else {
            false
        }
    }

    /// Apply an instantaneous angular impulse.
    pub fn apply_torque_impulse(&mut self, body: BodyId3D, torque_impulse: DVec3) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.apply_torque_impulse(torque_impulse, true);
            true
        } else {
            false
        }
    }

    /// Clear any accumulated (not-yet-integrated) forces and torques on the body.
    pub fn reset_forces(&mut self, body: BodyId3D) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.reset_forces(true);
            rb.reset_torques(true);
            true
        } else {
            false
        }
    }

    /// Change a body's kind (Static/Kinematic/Dynamic) in place, waking it.
    pub fn set_body_kind(&mut self, body: BodyId3D, kind: BodyKind3D) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_body_type(kind.to_rapier(), true);
            self.query_dirty = true;
            true
        } else {
            false
        }
    }

    /// Per-body multiplier on world gravity (dynamic bodies).
    pub fn set_body_gravity_scale(&mut self, body: BodyId3D, scale: f64) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_gravity_scale(scale, true);
            true
        } else {
            false
        }
    }

    /// Linear + angular velocity decay per second (drag).
    pub fn set_body_damping(&mut self, body: BodyId3D, linear: f64, angular: f64) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_linear_damping(linear);
            rb.set_angular_damping(angular);
            true
        } else {
            false
        }
    }

    /// Lock (or unlock) all rotation so the solver never spins the body — the
    /// usual setting for an upright character.
    pub fn set_body_locked_rotations(&mut self, body: BodyId3D, locked: bool) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.lock_rotations(locked, true);
            true
        } else {
            false
        }
    }

    /// Enable/disable Continuous Collision Detection for this body (P12.1). CCD
    /// stops a fast small body from tunnelling through a thin static wall in a
    /// single step, at extra solver cost — enable it for bullets/projectiles.
    pub fn set_body_ccd(&mut self, body: BodyId3D, enabled: bool) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.enable_ccd(enabled);
            true
        } else {
            false
        }
    }

    // ── Joints (P12.1) ────────────────────────────────────────────────────────

    /// Create a joint linking `body1` and `body2` per `desc`. Returns `None` if
    /// either body handle is invalid. Both bodies are woken.
    pub fn add_joint(
        &mut self,
        body1: BodyId3D,
        body2: BodyId3D,
        desc: JointDesc3D,
    ) -> Option<JointId3D> {
        if !self.bodies.contains(body1.0) || !self.bodies.contains(body2.0) {
            return None;
        }
        let handle = self
            .impulse_joints
            .insert(body1.0, body2.0, desc.to_generic(), true);
        Some(JointId3D(handle))
    }

    /// Destroy a joint. Returns `false` if the handle was already invalid.
    pub fn remove_joint(&mut self, joint: JointId3D) -> bool {
        self.impulse_joints.remove(joint.0, true).is_some()
    }

    /// Does this joint still exist?
    pub fn contains_joint(&self, joint: JointId3D) -> bool {
        self.impulse_joints.get(joint.0).is_some()
    }

    /// Every live joint handle, sorted deterministically by handle.
    pub fn joint_ids(&self) -> Vec<JointId3D> {
        let mut ids: Vec<JointId3D> = self
            .impulse_joints
            .iter()
            .map(|(h, _)| JointId3D(h))
            .collect();
        ids.sort_unstable();
        ids
    }

    /// The two bodies a joint connects (canonicalized `body_a <= body_b`), or
    /// `None` if the handle is invalid.
    pub fn joint_bodies(&self, joint: JointId3D) -> Option<(BodyId3D, BodyId3D)> {
        let j = self.impulse_joints.get(joint.0)?;
        let (mut a, mut b) = (BodyId3D(j.body1()), BodyId3D(j.body2()));
        if b < a {
            std::mem::swap(&mut a, &mut b);
        }
        Some((a, b))
    }

    // ── Colliders ─────────────────────────────────────────────────────────────

    /// Attach a collider to a body. Returns `None` if the body handle is invalid
    /// or the shape could not be built (a degenerate trimesh).
    pub fn add_collider(&mut self, body: BodyId3D, desc: ColliderDesc3D) -> Option<ColliderId3D> {
        if !self.bodies.contains(body.0) {
            return None;
        }
        let shape = desc.shape.to_shared()?;
        let collider = ColliderBuilder::new(shape)
            .friction(desc.friction)
            .restitution(desc.restitution)
            .density(desc.density)
            .sensor(desc.sensor)
            .translation(desc.local_translation)
            .collision_groups(to_interaction_groups(desc.layers))
            .friction_combine_rule(to_combine_rule(desc.friction_combine))
            .restitution_combine_rule(to_combine_rule(desc.restitution_combine))
            .active_events(ActiveEvents::COLLISION_EVENTS)
            // Report every body-type pairing, not just rapier's dynamic-involving
            // default — game triggers routinely involve kinematic-vs-static and
            // static-vs-static sensor overlaps, which would otherwise be silent.
            .active_collision_types(ActiveCollisionTypes::all())
            .build();
        let handle = self
            .colliders
            .insert_with_parent(collider, body.0, &mut self.bodies);
        self.query_dirty = true;
        Some(ColliderId3D(handle))
    }

    /// Destroy a collider. Returns `false` if the handle was already invalid.
    pub fn remove_collider(&mut self, collider: ColliderId3D) -> bool {
        let removed = self
            .colliders
            .remove(collider.0, &mut self.islands, &mut self.bodies, true)
            .is_some();
        if removed {
            self.query_dirty = true;
        }
        removed
    }

    /// Does this collider still exist?
    pub fn contains_collider(&self, collider: ColliderId3D) -> bool {
        self.colliders.contains(collider.0)
    }

    /// The body a collider is attached to.
    pub fn collider_parent(&self, collider: ColliderId3D) -> Option<BodyId3D> {
        self.colliders.get(collider.0)?.parent().map(BodyId3D)
    }

    // ── Scene queries ─────────────────────────────────────────────────────────

    /// Cast a ray from `origin` along `dir` (need not be normalized) up to
    /// `max_toi` world units. Returns the closest hit, or `None`.
    pub fn cast_ray(&mut self, origin: DVec3, dir: DVec3, max_toi: f64) -> Option<RayHit3D> {
        let dir = dir.normalize_or_zero();
        if dir == DVec3::ZERO {
            return None;
        }
        self.ensure_query_pipeline();
        let ray = Ray::new(origin, dir);
        let pipe = self.query_pipeline(QueryFilter::default());
        let (handle, hit) = pipe.cast_ray_and_get_normal(&ray, max_toi, true)?;
        Some(RayHit3D {
            collider: ColliderId3D(handle),
            point: ray.point_at(hit.time_of_impact),
            normal: hit.normal,
            toi: hit.time_of_impact,
        })
    }

    /// [`cast_ray`](Self::cast_ray) against **static and kinematic** geometry
    /// only, ignoring every collider in `exclude` (P22.3).
    ///
    /// # Why a filtered ray and not an AABB overlap
    ///
    /// The structural solve asks "is this chunk resting on static geometry", and
    /// the two obvious cheaper answers are both wrong here.
    /// [`intersect_aabb`](Self::intersect_aabb) is broad-phase only, so a chunk
    /// three metres above a terrain tile overlaps that tile's 255 m box and
    /// reports supported. [`intersect_point`](Self::intersect_point) needs an
    /// *interior*, and a height field has none — the ground would answer "no" for
    /// every point on it.
    ///
    /// A ray is exact against every shape in the facade, and the exclusion is
    /// what makes it usable: without it the probe from a chunk's underside hits
    /// the chunk itself (or the intact collider it is replacing) and certifies a
    /// building that is standing on nothing.
    ///
    /// The filter is rapier's own `QueryFilter::predicate`, so the broad phase
    /// still prunes and the excluded colliders are rejected before any narrow
    /// phase runs, and `QueryFilter::exclude_dynamic` rides with it — see the body
    /// for why that must happen in the filter and not at the call site.
    ///
    /// It was briefly a `skip_dynamic: bool` parameter with exactly one caller
    /// passing exactly one value. A knob nobody turns is a knob that documents a
    /// choice nobody made, so the behaviour is in the name instead.
    pub fn cast_ray_excluding(
        &mut self,
        origin: DVec3,
        dir: DVec3,
        max_toi: f64,
        exclude: &std::collections::BTreeSet<ColliderId3D>,
    ) -> Option<RayHit3D> {
        let dir = dir.normalize_or_zero();
        if dir == DVec3::ZERO {
            return None;
        }
        self.ensure_query_pipeline();
        let ray = Ray::new(origin, dir);
        let predicate = |h: rapier3d_f64::geometry::ColliderHandle,
                         _: &rapier3d_f64::geometry::Collider| {
            !exclude.contains(&ColliderId3D(h))
        };
        // `skip_dynamic` asks the BROAD PHASE to leave dynamic bodies out
        // entirely, which is not the same as "cast, then reject a dynamic hit".
        // The difference is the whole of the P22.3 audit's M4: with the check
        // downstream, a single crate resting a centimetre above the floor is the
        // NEAREST hit, the cast returns it, the caller rejects it — and the floor
        // underneath, which the caller was asking about, is never seen at all. So
        // "debris provides no support" silently became "debris HIDES support",
        // and a tower collapsed because its own rubble had landed beside it. It
        // also removes a tie-at-TOI hazard: two coincident hits, one dynamic, are
        // ordered by the BVH rather than by anything deterministic.
        let filter = QueryFilter::exclude_dynamic().predicate(&predicate);
        let pipe = self.query_pipeline(filter);
        let (handle, hit) = pipe.cast_ray_and_get_normal(&ray, max_toi, true)?;
        Some(RayHit3D {
            collider: ColliderId3D(handle),
            point: ray.point_at(hit.time_of_impact),
            normal: hit.normal,
            toi: hit.time_of_impact,
        })
    }

    /// A body's kind (static / kinematic / dynamic), or `None` for a stale
    /// handle.
    pub fn body_kind(&self, body: BodyId3D) -> Option<BodyKind3D> {
        self.bodies.get(body.0).map(|rb| match rb.body_type() {
            RigidBodyType::Fixed => BodyKind3D::Static,
            RigidBodyType::Dynamic => BodyKind3D::Dynamic,
            _ => BodyKind3D::Kinematic,
        })
    }

    /// Every collider containing `point`, sorted deterministically by handle.
    pub fn intersect_point(&mut self, point: DVec3) -> Vec<ColliderId3D> {
        self.ensure_query_pipeline();
        let pipe = self.query_pipeline(QueryFilter::default());
        let mut out: Vec<ColliderId3D> = pipe
            .intersect_point(point)
            .map(|(h, _)| ColliderId3D(h))
            .collect();
        out.sort_unstable();
        out
    }

    /// Every collider whose broad-phase AABB overlaps the given AABB, sorted
    /// deterministically. This is a conservative (AABB-level) query — a returned
    /// collider's exact shape may not overlap the box, only its bounds.
    pub fn intersect_aabb(&mut self, min: DVec3, max: DVec3) -> Vec<ColliderId3D> {
        self.ensure_query_pipeline();
        let aabb = Aabb::new(min, max);
        let pipe = self.query_pipeline(QueryFilter::default());
        let mut out: Vec<ColliderId3D> = pipe
            .intersect_aabb_conservative(aabb)
            .map(|(h, _)| ColliderId3D(h))
            .collect();
        out.sort_unstable();
        out
    }

    // ── Character mover ───────────────────────────────────────────────────────

    /// Slide `mover`'s shape from `position` by `desired_translation` through the
    /// world, resolving collisions (sliding along walls, snapping to ground, etc.
    /// per the mover's config). Returns the movement actually applied, whether the
    /// character ended grounded, and the colliders it touched.
    ///
    /// `exclude` should be the character's own collider (if it has one in this
    /// world) so it doesn't collide with itself.
    pub fn move_character(
        &mut self,
        mover: &CharacterMover3D,
        position: DVec3,
        desired_translation: DVec3,
        exclude: Option<ColliderId3D>,
    ) -> CharacterMove3D {
        self.ensure_query_pipeline();
        let dt = self.integration_parameters.dt;
        let mut filter = QueryFilter::default();
        if let Some(c) = exclude {
            filter = filter.exclude_collider(c.0);
        }
        let pipe = self.query_pipeline(filter);
        mover.solve(&pipe, dt, position, desired_translation)
    }

    // ── internal ──────────────────────────────────────────────────────────────

    /// Rebuild the query BVH from the current colliders if the world changed
    /// since the last query. Kept separate from the simulation broad-phase so a
    /// query never perturbs the deterministic step state.
    fn ensure_query_pipeline(&mut self) {
        if !self.query_dirty {
            return;
        }
        let params = self.integration_parameters;
        let mut bvh = BroadPhaseBvh::new();
        for (handle, collider) in self.colliders.iter() {
            bvh.set_aabb(&params, handle, collider.compute_aabb());
        }
        self.query_bvh = bvh;
        self.query_dirty = false;
    }

    fn query_pipeline<'a>(&'a self, filter: QueryFilter<'a>) -> QueryPipeline<'a> {
        self.query_bvh.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.bodies,
            &self.colliders,
            filter,
        )
    }
}
