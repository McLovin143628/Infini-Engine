//! [`PhysicsWorld2D`]: the fixed-step 2D rigid-body world.

use glam::DVec2;
use rapier2d_f64::dynamics::CoefficientCombineRule;
use rapier2d_f64::geometry::{Group, InteractionGroups, InteractionTestMode};
use rapier2d_f64::prelude::{
    Aabb, ActiveCollisionTypes, ActiveEvents, BroadPhaseBvh, CCDSolver, ColliderBuilder,
    ColliderSet, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
    NarrowPhase, PhysicsPipeline, QueryFilter, QueryPipeline, Ray, RigidBodyBuilder, RigidBodySet,
    RigidBodyType, Rotation, SharedShape,
};

use super::character::{CharacterMove2D, CharacterMover2D};
use super::events::{ContactEvent2D, EventCollector};
use super::joint::{JointDesc2D, JointId2D};
use super::query::RayHit2D;
use super::{BodyId, ColliderId};
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

/// The kind of a rigid body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    /// Never moved by the solver; infinite mass (floors, walls).
    Static,
    /// Moved only by the facade (`set_body_translation`/`_rotation`), pushes
    /// dynamic bodies but is not pushed back. Position-based kinematic — the kind
    /// a character mover or moving platform uses.
    Kinematic,
    /// Fully simulated: gravity, forces, impulses, and contacts move it.
    Dynamic,
}

impl BodyKind {
    fn to_rapier(self) -> RigidBodyType {
        match self {
            BodyKind::Static => RigidBodyType::Fixed,
            BodyKind::Kinematic => RigidBodyType::KinematicPositionBased,
            BodyKind::Dynamic => RigidBodyType::Dynamic,
        }
    }
}

/// A 2D collider shape. Half-extents / radii are in world (f64) units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColliderShape2D {
    /// A circle of the given radius.
    Circle { radius: f64 },
    /// An axis-aligned box given by its half-extents.
    Box { half_width: f64, half_height: f64 },
    /// A vertical capsule: a segment of length `2 * half_height` along local Y,
    /// swept by `radius`.
    Capsule { half_height: f64, radius: f64 },
}

impl ColliderShape2D {
    pub(crate) fn to_shared(self) -> SharedShape {
        match self {
            ColliderShape2D::Circle { radius } => SharedShape::ball(radius),
            ColliderShape2D::Box {
                half_width,
                half_height,
            } => SharedShape::cuboid(half_width, half_height),
            ColliderShape2D::Capsule {
                half_height,
                radius,
            } => SharedShape::capsule_y(half_height, radius),
        }
    }
}

/// Description of a collider to attach to a body. Build with [`ColliderDesc2D::new`]
/// and the fluent setters. Every collider the facade creates has collision-event
/// reporting enabled so the [`PhysicsWorld2D::drain_contact_events`] drain sees it.
#[derive(Clone, Copy, Debug)]
pub struct ColliderDesc2D {
    /// The shape.
    pub shape: ColliderShape2D,
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
    pub local_translation: DVec2,
    /// Bitmask collision layers (P12.1). Default = interact with everything.
    pub layers: CollisionLayers,
    /// How this collider's friction combines with a contacting collider's (P12.1).
    pub friction_combine: CombineRule,
    /// How this collider's restitution combines with a contacting collider's.
    pub restitution_combine: CombineRule,
}

impl ColliderDesc2D {
    /// A solid collider with engine-default material (friction 0.5, no
    /// restitution, unit density, all layers, `Average` combine rules).
    pub fn new(shape: ColliderShape2D) -> Self {
        Self {
            shape,
            friction: 0.5,
            restitution: 0.0,
            density: 1.0,
            sensor: false,
            local_translation: DVec2::ZERO,
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
    pub fn local_translation(mut self, offset: DVec2) -> Self {
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

/// A fixed-step 2D physics world wrapping `rapier2d-f64`.
///
/// Simulation is advanced only by [`step`](Self::step), which takes the timestep
/// as an argument — the world never reads a wall clock, so replaying the same
/// calls reproduces the same result bit-for-bit (see the crate-level determinism
/// note). Contact events accumulate across steps and are read with
/// [`drain_contact_events`](Self::drain_contact_events); scene queries and the
/// character mover run against the post-step state.
pub struct PhysicsWorld2D {
    gravity: DVec2,
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

    pending_contacts: Vec<ContactEvent2D>,
}

impl PhysicsWorld2D {
    /// A new, empty world with the given gravity (world units / s²). A typical
    /// top-down world uses `DVec2::ZERO`; a side-scroller uses
    /// `DVec2::new(0.0, -9.81)`.
    pub fn new(gravity: DVec2) -> Self {
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
    pub fn gravity(&self) -> DVec2 {
        self.gravity
    }

    /// Replace the gravity vector (takes effect on the next [`step`](Self::step)).
    pub fn set_gravity(&mut self, gravity: DVec2) {
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
    pub fn drain_contact_events(&mut self) -> Vec<ContactEvent2D> {
        let mut out = std::mem::take(&mut self.pending_contacts);
        out.sort_unstable();
        out
    }

    // ── Bodies ──────────────────────────────────────────────────────────────

    /// Create a rigid body at `position` with `rotation` (radians). Returns its
    /// stable handle.
    pub fn add_body(&mut self, kind: BodyKind, position: DVec2, rotation: f64) -> BodyId {
        let rb = RigidBodyBuilder::new(kind.to_rapier())
            .translation(position)
            .rotation(rotation)
            .build();
        let handle = self.bodies.insert(rb);
        self.query_dirty = true;
        BodyId(handle)
    }

    /// Destroy a body and all colliders attached to it. Returns `false` if the
    /// handle was already invalid.
    pub fn remove_body(&mut self, body: BodyId) -> bool {
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
    pub fn contains_body(&self, body: BodyId) -> bool {
        self.bodies.contains(body.0)
    }

    /// Every live body handle, sorted deterministically. Handy for snapshotting
    /// the world (e.g. the determinism harness).
    pub fn body_ids(&self) -> Vec<BodyId> {
        let mut ids: Vec<BodyId> = self.bodies.iter().map(|(h, _)| BodyId(h)).collect();
        ids.sort_unstable();
        ids
    }

    /// The body's world-space translation.
    pub fn body_translation(&self, body: BodyId) -> Option<DVec2> {
        self.bodies.get(body.0).map(|rb| rb.translation())
    }

    /// The body's world-space rotation angle (radians).
    pub fn body_rotation(&self, body: BodyId) -> Option<f64> {
        self.bodies.get(body.0).map(|rb| rb.rotation().angle())
    }

    /// Teleport the body's translation (wakes it).
    pub fn set_body_translation(&mut self, body: BodyId, translation: DVec2) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_translation(translation, true);
            self.query_dirty = true;
            true
        } else {
            false
        }
    }

    /// Set the body's rotation angle (radians; wakes it).
    pub fn set_body_rotation(&mut self, body: BodyId, angle: f64) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_rotation(Rotation::new(angle), true);
            self.query_dirty = true;
            true
        } else {
            false
        }
    }

    /// The body's linear velocity.
    pub fn body_linvel(&self, body: BodyId) -> Option<DVec2> {
        self.bodies.get(body.0).map(|rb| rb.linvel())
    }

    /// The body's angular velocity (radians/s).
    pub fn body_angvel(&self, body: BodyId) -> Option<f64> {
        self.bodies.get(body.0).map(|rb| rb.angvel())
    }

    /// Set the body's linear velocity.
    pub fn set_body_linvel(&mut self, body: BodyId, linvel: DVec2) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_linvel(linvel, true);
            true
        } else {
            false
        }
    }

    /// Set the body's angular velocity (radians/s).
    pub fn set_body_angvel(&mut self, body: BodyId, angvel: f64) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_angvel(angvel, true);
            true
        } else {
            false
        }
    }

    /// Add a force applied at the center of mass. Forces accumulate and are
    /// consumed each [`step`](Self::step).
    pub fn apply_force(&mut self, body: BodyId, force: DVec2) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.add_force(force, true);
            true
        } else {
            false
        }
    }

    /// Apply an instantaneous linear impulse (immediately changes velocity).
    pub fn apply_impulse(&mut self, body: BodyId, impulse: DVec2) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.apply_impulse(impulse, true);
            true
        } else {
            false
        }
    }

    /// Add a torque (accumulates, consumed each step).
    pub fn apply_torque(&mut self, body: BodyId, torque: f64) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.add_torque(torque, true);
            true
        } else {
            false
        }
    }

    /// Apply an instantaneous angular impulse.
    pub fn apply_torque_impulse(&mut self, body: BodyId, torque_impulse: f64) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.apply_torque_impulse(torque_impulse, true);
            true
        } else {
            false
        }
    }

    /// Clear any accumulated (not-yet-integrated) forces and torques on the body.
    pub fn reset_forces(&mut self, body: BodyId) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.reset_forces(true);
            rb.reset_torques(true);
            true
        } else {
            false
        }
    }

    /// Change a body's kind (Static/Kinematic/Dynamic) in place, waking it.
    pub fn set_body_kind(&mut self, body: BodyId, kind: BodyKind) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_body_type(kind.to_rapier(), true);
            self.query_dirty = true;
            true
        } else {
            false
        }
    }

    /// Per-body multiplier on world gravity (dynamic bodies).
    pub fn set_body_gravity_scale(&mut self, body: BodyId, scale: f64) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_gravity_scale(scale, true);
            true
        } else {
            false
        }
    }

    /// Linear + angular velocity decay per second (drag).
    pub fn set_body_damping(&mut self, body: BodyId, linear: f64, angular: f64) -> bool {
        if let Some(rb) = self.bodies.get_mut(body.0) {
            rb.set_linear_damping(linear);
            rb.set_angular_damping(angular);
            true
        } else {
            false
        }
    }

    /// Lock (or unlock) the body's rotation so the solver never spins it — the
    /// usual setting for an upright character.
    pub fn set_body_locked_rotations(&mut self, body: BodyId, locked: bool) -> bool {
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
    pub fn set_body_ccd(&mut self, body: BodyId, enabled: bool) -> bool {
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
        body1: BodyId,
        body2: BodyId,
        desc: JointDesc2D,
    ) -> Option<JointId2D> {
        if !self.bodies.contains(body1.0) || !self.bodies.contains(body2.0) {
            return None;
        }
        let handle = self
            .impulse_joints
            .insert(body1.0, body2.0, desc.to_generic(), true);
        Some(JointId2D(handle))
    }

    /// Destroy a joint. Returns `false` if the handle was already invalid.
    pub fn remove_joint(&mut self, joint: JointId2D) -> bool {
        self.impulse_joints.remove(joint.0, true).is_some()
    }

    /// Does this joint still exist?
    pub fn contains_joint(&self, joint: JointId2D) -> bool {
        self.impulse_joints.get(joint.0).is_some()
    }

    /// Every live joint handle, sorted deterministically by handle.
    pub fn joint_ids(&self) -> Vec<JointId2D> {
        let mut ids: Vec<JointId2D> = self
            .impulse_joints
            .iter()
            .map(|(h, _)| JointId2D(h))
            .collect();
        ids.sort_unstable();
        ids
    }

    /// The two bodies a joint connects (canonicalized `body_a <= body_b`), or
    /// `None` if the handle is invalid.
    pub fn joint_bodies(&self, joint: JointId2D) -> Option<(BodyId, BodyId)> {
        let j = self.impulse_joints.get(joint.0)?;
        let (mut a, mut b) = (BodyId(j.body1()), BodyId(j.body2()));
        if b < a {
            std::mem::swap(&mut a, &mut b);
        }
        Some((a, b))
    }

    // ── Colliders ─────────────────────────────────────────────────────────────

    /// Attach a collider to a body. Returns `None` if the body handle is invalid.
    pub fn add_collider(&mut self, body: BodyId, desc: ColliderDesc2D) -> Option<ColliderId> {
        if !self.bodies.contains(body.0) {
            return None;
        }
        let collider = ColliderBuilder::new(desc.shape.to_shared())
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
        Some(ColliderId(handle))
    }

    /// Destroy a collider. Returns `false` if the handle was already invalid.
    pub fn remove_collider(&mut self, collider: ColliderId) -> bool {
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
    pub fn contains_collider(&self, collider: ColliderId) -> bool {
        self.colliders.contains(collider.0)
    }

    /// The body a collider is attached to.
    pub fn collider_parent(&self, collider: ColliderId) -> Option<BodyId> {
        self.colliders.get(collider.0)?.parent().map(BodyId)
    }

    // ── Scene queries ─────────────────────────────────────────────────────────

    /// Cast a ray from `origin` along `dir` (need not be normalized) up to
    /// `max_toi` in units of `dir`'s length after normalization, i.e. world
    /// distance. Returns the closest hit, or `None`.
    pub fn cast_ray(&mut self, origin: DVec2, dir: DVec2, max_toi: f64) -> Option<RayHit2D> {
        let dir = dir.normalize_or_zero();
        if dir == DVec2::ZERO {
            return None;
        }
        self.ensure_query_pipeline();
        let ray = Ray::new(origin, dir);
        let pipe = self.query_pipeline(QueryFilter::default());
        let (handle, hit) = pipe.cast_ray_and_get_normal(&ray, max_toi, true)?;
        Some(RayHit2D {
            collider: ColliderId(handle),
            point: ray.point_at(hit.time_of_impact),
            normal: hit.normal,
            toi: hit.time_of_impact,
        })
    }

    /// Every collider containing `point`, sorted deterministically by handle.
    pub fn intersect_point(&mut self, point: DVec2) -> Vec<ColliderId> {
        self.ensure_query_pipeline();
        let pipe = self.query_pipeline(QueryFilter::default());
        let mut out: Vec<ColliderId> = pipe
            .intersect_point(point)
            .map(|(h, _)| ColliderId(h))
            .collect();
        out.sort_unstable();
        out
    }

    /// Every collider whose broad-phase AABB overlaps the given AABB, sorted
    /// deterministically. This is a conservative (AABB-level) query — a returned
    /// collider's exact shape may not overlap the box, only its bounds.
    pub fn intersect_aabb(&mut self, min: DVec2, max: DVec2) -> Vec<ColliderId> {
        self.ensure_query_pipeline();
        let aabb = Aabb::new(min, max);
        let pipe = self.query_pipeline(QueryFilter::default());
        let mut out: Vec<ColliderId> = pipe
            .intersect_aabb_conservative(aabb)
            .map(|(h, _)| ColliderId(h))
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
        mover: &CharacterMover2D,
        position: DVec2,
        desired_translation: DVec2,
        exclude: Option<ColliderId>,
    ) -> CharacterMove2D {
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
