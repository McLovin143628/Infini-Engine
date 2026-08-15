//! ECS ↔ physics bridge (3D): the `d3` mirror of [`crate::d2::PhysicsBridge2D`].
//!
//! # Two entry points: a raw snapshot and the ECS adapter
//!
//! The reconcile / spawn / update / despawn / write-back **engine** here consumes
//! a plain, facade-local snapshot ([`EntitySync3D`]) via [`sync`](PhysicsBridge3D::sync)
//! so it never has to name ECS types. On top of it,
//! [`sync_from_world`](PhysicsBridge3D::sync_from_world) reads the real
//! `inf_ecs::components::{RigidBody3D, Collider3D, Transform}` into `EntitySync3D`
//! and calls `sync`, and [`write_back_into`](PhysicsBridge3D::write_back_into)
//! copies the simulated dynamic poses back onto the entities' `Transform`s — the
//! exact `d2` shape at 3D (full translation + quaternion, not just XY + Z-euler).
//!
//! Determinism (§2.5) is preserved exactly as in d2: **every** entity pass —
//! spawn, update, despawn, write-back — runs in sorted `Guid` order (a `BTreeMap`
//! of handles + a sort of the incoming snapshot), never in ECS `Entity`-id order,
//! so two worlds built from the same scene allocate rapier handles in the same
//! sequence and step to byte-identical poses.
//!
//! Mapping rules (mirrored from d2, at 3D):
//! * `Transform` (translation + rotation quaternion) is authoritative for
//!   **static/kinematic** bodies (pushed every sync). **Dynamic** bodies are
//!   solver-owned: their pose is pushed once at spawn and flows back out through
//!   [`write_back`](PhysicsBridge3D::write_back).
//! * An entity with a body descriptor becomes a body of that kind; an entity with
//!   only a collider gets an implicit **static** body so the collider has a parent.

use std::collections::{BTreeMap, BTreeSet};

use glam::{DAffine3, DQuat, DVec2, DVec3};
use inf_ecs::components::{
    BodyKind3D as SceneBodyKind3D, Buoyancy, Collider3D, ColliderShape3DKind,
    CombineRule as SceneCombineRule, GlobalTransform, Joint3D, JointKind3D as SceneJointKind3D,
    PcgVolume, RigidBody3D, Spline, Terrain, Transform, WaterBody,
};
use inf_ecs::{EcsWorld, Vec3d};
use inf_terrain::TileKey;
use uuid::Uuid;

use super::joint::{JointDesc3D, JointId3D, JointKind3D, JointMotor3D};
use super::water::{
    self, BodyState, BuoyancyDesc3D, BuoyantMap, SampleGeometry, WaterEvent3D, WaterIndex,
    WaterProbe, WaterStamp,
};
use super::world::{BodyKind3D, ColliderDesc3D, ColliderShape3D, PhysicsWorld3D};
use super::{BodyId3D, ColliderId3D};
use crate::filtering::{CollisionLayers, CombineRule};

/// A rigid-body descriptor — the facade-local shape of the future `RigidBody3D`
/// component (see the batch report for the ready-to-paste `inf-ecs` struct). The
/// bridge maps this onto body properties every sync.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyDesc3D {
    /// Static / Kinematic / Dynamic.
    pub kind: BodyKind3D,
    /// Per-body multiplier on world gravity (dynamic bodies).
    pub gravity_scale: f64,
    /// Lock all rotation so the body never spins (typical for characters).
    pub fixed_rotation: bool,
    /// Linear velocity decay per second (drag).
    pub linear_damping: f64,
    /// Angular velocity decay per second.
    pub angular_damping: f64,
    /// Continuous Collision Detection (P12.1).
    pub ccd_enabled: bool,
}

impl Default for BodyDesc3D {
    fn default() -> Self {
        Self {
            kind: BodyKind3D::Static,
            gravity_scale: 1.0,
            fixed_rotation: false,
            linear_damping: 0.0,
            angular_damping: 0.0,
            ccd_enabled: false,
        }
    }
}

/// One entity's joint for a [`sync`](PhysicsBridge3D::sync) pass: the `Guid` of the
/// OTHER body it links to, plus the facade joint descriptor. Reconciled after all
/// bodies are spawned so the referenced body always exists.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointSync3D {
    /// The other body's entity `Guid`.
    pub other: Uuid,
    /// The joint family + anchors + params.
    pub desc: JointDesc3D,
}

/// One entity's physics state for a [`sync`](PhysicsBridge3D::sync) pass: its
/// stable `guid`, optional body + collider descriptors, and its world pose.
///
/// This is the seam the next-batch `sync_from_world` adapter fills from the real
/// ECS components; today the tests build it directly.
#[derive(Clone, Debug, PartialEq)]
pub struct EntitySync3D {
    /// Stable entity identity (keys the deterministic handle maps).
    pub guid: Uuid,
    /// The body descriptor, or `None` (collider-only → implicit static body).
    pub body: Option<BodyDesc3D>,
    /// The collider descriptor, or `None` (a bodiless-collider entity is skipped).
    pub collider: Option<ColliderDesc3D>,
    /// World-space translation (from the entity's `Transform`).
    pub translation: DVec3,
    /// World-space orientation (from the entity's `Transform`).
    pub rotation: DQuat,
    /// An optional joint to another body (P12.1), reconciled in a second pass.
    pub joint: Option<JointSync3D>,
}

/// The pose the bridge wrote back for one dynamic body — the next-batch adapter
/// copies this onto the entity's `Transform`. Sorted by `guid`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseWriteback3D {
    pub guid: Uuid,
    pub translation: DVec3,
    pub rotation: DQuat,
}

/// One tracked entity: its rapier body/collider handles plus the last-synced
/// descriptors (for cheap change detection).
struct BodyRecord {
    body: BodyId3D,
    collider: Option<ColliderId3D>,
    kind: BodyKind3D,
    rb: Option<BodyDesc3D>,
    /// The collider descriptor that is **actually attached** — never the one
    /// that was asked for (C4-30).
    ///
    /// The distinction is the whole finding. This field is the reconcile's
    /// change detector: `rec.col != snap.collider` is what decides whether to
    /// rebuild. Writing the *desired* descriptor here after `add_collider`
    /// returned `None` told the next reconcile there was nothing to do, and the
    /// entity kept a rigid body with no collider — silently, with no log and no
    /// counter, which is a player walking through a cave wall.
    col: Option<ColliderDesc3D>,
    /// The descriptor that was tried and **refused**, so the retry is bounded
    /// (C4-30).
    ///
    /// Without it, an honest `col` (left `None` after a refusal) would make
    /// every subsequent sync re-attempt the same build — and `to_shared_checked`
    /// is a pure function of the descriptor, so it would fail identically, for
    /// ever, at trimesh-topology cost per fixed step. With it the retry happens
    /// exactly when it can succeed: when the shape *changes*. Which is what a
    /// re-carve, a re-fracture or a terrain edit does.
    col_refused: Option<ColliderDesc3D>,
    /// The joint this entity owns (P12.1): its handle, the other body it was
    /// built against, and the last-synced snapshot (for change detection).
    joint: Option<JointBinding>,
}

/// A live joint binding tracked on the owning entity's record.
#[derive(Clone, Copy)]
struct JointBinding {
    id: JointId3D,
    other_body: BodyId3D,
    sync: JointSync3D,
}

/// Owns a [`PhysicsWorld3D`] and keeps it in sync with a scene snapshot.
pub struct PhysicsBridge3D {
    world: PhysicsWorld3D,
    /// `Guid` → its rapier handles. `BTreeMap` gives sorted (deterministic)
    /// iteration for the despawn + write-back passes.
    entities: BTreeMap<Uuid, BodyRecord>,
    /// Per-`PcgVolume` change stamp for its derived solids (P19.5):
    /// `guid → (structures_gen, count)`. While both match, the volume's
    /// colliders are **retained without being re-described** — see
    /// [`pcg_structure_snaps`].
    structure_stamps: BTreeMap<Uuid, (u64, usize)>,
    /// Per-voxel-chunk change stamp (P21.4): `(volume entity, chunk key) →
    /// `inf_voxel::source_key`. The `structure_stamps` twin, keyed one level
    /// finer because a runtime carve moves a *handful* of chunks and re-meshing a
    /// whole cave system for it inside a fixed step is exactly the bill the pattern
    /// exists to refuse. While a stamp matches, that chunk's collider is
    /// **retained without being re-described** — see
    /// [`gather_voxels`](Self::gather_voxels).
    ///
    /// **"A handful" is up to 27, not two.** `source_key` is a max over the 3×3×3
    /// neighbourhood, so carving one chunk moves the key of every chunk that can
    /// see it — the carved one and its 26 neighbours — and each of them is
    /// re-meshed and re-described. That is the price of correctness (a mesh really
    /// is a function of its neighbourhood; the alternative is the stale seam the
    /// M3 defect produced), and it is a real per-carve-step cost on a volume being
    /// dug continuously. It is bounded and local: 27 chunks against a cave system
    /// of hundreds, and zero on every step that digs nothing.
    ///
    /// The value is the **mesher's** key, not the chunk's own version: a mesh is a
    /// function of its 3×3×3 neighbourhood, so the chunk's own stamp cannot see a
    /// neighbour being carved or evicted (`inf_voxel::MeshSourceKey`'s docs carry
    /// the argument and the bug it hid).
    voxel_stamps: BTreeMap<(Uuid, inf_voxel::ChunkKey), inf_voxel::MeshSourceKey>,
    /// Per-terrain-tile change stamp (P22.3): `(terrain entity, level-0 tile
    /// coord) → (`[`TerrainData::tile_version`], terrain origin)`. The
    /// [`structure_stamps`](Self::structure_stamps) pattern one more time, at the
    /// granularity streaming already has.
    ///
    /// A tile's version is a **process-global monotone counter** minted on every
    /// mutating touch (and on paging a tile in), so it is runtime cache identity
    /// and never content — which is exactly what a retention stamp needs and
    /// exactly what a *serialized* one must not be. Determinism does not rest on
    /// it either way: a re-described tile produces a byte-identical
    /// [`ColliderDesc3D`], and [`sync_retaining`](Self::sync_retaining) rebuilds a
    /// collider only when the descriptor actually differs. The stamp buys the
    /// *work*, not the answer.
    ///
    /// The origin rides in the stamp because a terrain that moved has to
    /// re-place every one of its tiles even though not one sample changed.
    terrain_stamps: BTreeMap<TerrainTileKey, TerrainTileStamp>,
    /// How many terrain tile colliders were (re-)described on the most recent
    /// sync — the audit counter the budget test reads. Zero on a steady-state
    /// step is the whole point of [`terrain_stamps`](Self::terrain_stamps).
    terrain_audit: TerrainColliderAudit,
    /// Per-destructible-actor change stamp (P22.3): `entity →
    /// `[`FractureState::generation`](super::fracture::FractureState::generation).
    /// The same pattern a fourth time, keyed by ACTOR rather than by chunk —
    /// because a fracture event moves every chunk of one actor at once and
    /// nothing at all on any other, which is the opposite shape from a voxel
    /// carve.
    ///
    /// **Only broken actors appear here.** An intact destructible emits no chunk
    /// bodies at all, so a town full of destructible walls nobody has shot at
    /// costs an empty map and one `is_intact` call each.
    fracture_stamps: BTreeMap<Uuid, u64>,
    /// Reverse map `collider handle → owning entity Guid`, rebuilt at the end of
    /// every [`sync`](Self::sync) (Wave 3). The collision-event drain resolves
    /// rapier's `ContactEvent3D` collider handles back to entity `Guid`s through
    /// it — the inverse of [`collider_of`](Self::collider_of). Deterministic
    /// (`BTreeMap`, sorted keys).
    collider_to_guid: BTreeMap<ColliderId3D, Uuid>,
    /// Whether [`collider_to_guid`](Self::collider_to_guid) needs rebuilding —
    /// set by the **three** sites that can move a collider handle (a rebuilt
    /// collider on an existing entity, a new entity, a despawn) and cleared by
    /// the rebuild at the end of a sync (Hardening Wave E).
    ///
    /// The map used to be rebuilt from scratch on **every** sync, and the
    /// comment above it called that "cheap (one pass over the tracked
    /// entities)". One pass is O(colliders) `BTreeMap` inserts — the whole
    /// collider population of the level, sixty times a second, to reproduce a
    /// map that changes only when something spawns, despawns or has its shape
    /// edited. On a furnished town (the tree's own 13 000-collider figure) the
    /// steady state paid the whole rebuild for a guaranteed-identical result.
    ///
    /// A flag rather than incremental mutation on purpose: three call sites can
    /// invalidate and one place rebuilds, so a site added later that forgets to
    /// mutate the map would be a silently stale map, while a site that forgets
    /// to set this flag is the same bug with one line to find. `true` at
    /// construction, so the first sync always builds.
    collider_map_dirty: bool,
    /// **How many collider builds this bridge has refused, ever** (C4-30), and
    /// which entities it has already complained about.
    ///
    /// A counter rather than a log line alone because the failure is a *rate*
    /// question — "~10 % of meshes" is the seam-chord law's own measurement of
    /// how often `FIX_INTERNAL_EDGES` fails on Surface-Nets and Voronoi output —
    /// and because a test can assert a number. The warn set keeps the log
    /// honest: one line per entity, not one per fixed step.
    collider_refusals: u64,
    warned_colliders: BTreeSet<Uuid>,

    // ── water (P20.2) ─────────────────────────────────────────────────────
    /// The level's water, spatially indexed. Rebuilt only when
    /// [`water_stamps`](Self::water_stamps) changes — a river's arc-length
    /// resample is not a per-step cost.
    water: WaterIndex,
    /// What the index was last built from (the P19.5 change-stamp pattern).
    water_stamps: Vec<WaterStamp>,
    /// Scratch for this sync's stamps, reused so the steady state allocates
    /// nothing new.
    water_scratch: Vec<WaterStamp>,
    /// The components to build surfaces from when the stamp changed — gathered
    /// alongside the stamps in `sync_from_world`'s single entity walk.
    water_sources: Vec<(Uuid, WaterBody, Option<Spline>, DAffine3)>,
    /// `(level clock seconds, weather wind m/s)` for this step, resolved once per
    /// sync from [`inf_ecs::sky::water_environment`] — never from a wall clock.
    water_env: (f64, (f64, f64)),
    /// Buoyant bodies: their tuning and their per-step latch, in `Guid` order.
    /// **Empty for every level without a `Buoyancy` component**, which is what
    /// makes the whole water pass one branch on the off path.
    buoyant: BuoyantMap,
    /// Swim latches, keyed by character `Guid` (P20.2). Separate from
    /// [`buoyant`](Self::buoyant) because a character controller is kinematic and
    /// never floats — it swims.
    swimming: BTreeMap<Uuid, bool>,
    /// This step's crossings, drained by the host in the collision slot.
    water_events: Vec<WaterEvent3D>,
    /// The per-step entity snapshot buffer, **reused** (lens 3 P32, Hardening
    /// Wave G) — the [`water_scratch`](Self::water_scratch) pattern, applied to
    /// the biggest buffer in the file.
    ///
    /// `sync_from_world_sim` gathers one [`EntitySync3D`] per physics entity per
    /// fixed step, and that type is 408 bytes (it embeds a whole
    /// [`ColliderDesc3D`], whose trimesh and hull variants own buffers). Built
    /// into a fresh `Vec::new()`, a 13 000-entity level re-grew it from zero
    /// every step — fourteen reallocations, each copying what was there — which
    /// measured **1.75 ms per fixed step**, more than the entire reconcile it
    /// was feeding.
    ///
    /// Cleared on the way out as well as on the way in, deliberately: the
    /// descriptors hold voxel and fracture geometry, and a capacity kept alive
    /// between steps must not keep *contents* alive with it.
    snaps_scratch: Vec<EntitySync3D>,
}

impl PhysicsBridge3D {
    /// A new bridge over an empty world with the given gravity (world units/s²).
    /// A typical 3D world uses `DVec3::new(0.0, -9.81, 0.0)`.
    pub fn new(gravity: DVec3) -> Self {
        Self {
            world: PhysicsWorld3D::new(gravity),
            entities: BTreeMap::new(),
            structure_stamps: BTreeMap::new(),
            voxel_stamps: BTreeMap::new(),
            terrain_stamps: BTreeMap::new(),
            terrain_audit: TerrainColliderAudit::default(),
            fracture_stamps: BTreeMap::new(),
            collider_to_guid: BTreeMap::new(),
            collider_map_dirty: true,
            collider_refusals: 0,
            warned_colliders: BTreeSet::new(),
            water: WaterIndex::default(),
            water_stamps: Vec::new(),
            water_scratch: Vec::new(),
            water_sources: Vec::new(),
            water_env: (0.0, (0.0, 0.0)),
            buoyant: BuoyantMap::new(),
            swimming: BTreeMap::new(),
            water_events: Vec::new(),
            snaps_scratch: Vec::new(),
        }
    }

    /// **The one door every collider attach goes through** (C4-30).
    ///
    /// Returns `(handle, achieved descriptor, refused descriptor)` — the second
    /// and third are the honest record. Exactly one of them is `Some` when a
    /// collider was wanted; both are `None` when none was.
    ///
    /// A refusal is counted, and logged **once per entity** rather than once per
    /// fixed step: the same non-manifold chunk is re-described by the voxel
    /// stamp machinery whenever a neighbour moves, and a log line per step is a
    /// log nobody reads.
    fn attach_collider(
        &mut self,
        guid: Uuid,
        body: BodyId3D,
        want: Option<&ColliderDesc3D>,
    ) -> (
        Option<ColliderId3D>,
        Option<ColliderDesc3D>,
        Option<ColliderDesc3D>,
    ) {
        let Some(desc) = want else {
            return (None, None, None);
        };
        match self.world.try_add_collider(body, desc.clone()) {
            Ok(id) => (Some(id), Some(desc.clone()), None),
            Err(why) => {
                self.collider_refusals += 1;
                if self.warned_colliders.insert(guid) {
                    tracing::warn!(
                        entity = %guid,
                        reason = why,
                        "collider could not be built; this entity has a rigid body and \
                         nothing to collide with until its shape changes"
                    );
                }
                (None, None, Some(desc.clone()))
            }
        }
    }

    /// How many collider builds this bridge has refused since it was created
    /// (C4-30). Zero on a healthy level.
    pub fn collider_refusals(&self) -> u64 {
        self.collider_refusals
    }

    /// The entities that asked for a collider and have none, in `Guid` order
    /// (C4-30) — the *world's* answer, not a report of it.
    pub fn entities_missing_colliders(&self) -> Vec<Uuid> {
        self.entities
            .iter()
            .filter(|(_, r)| r.collider.is_none() && r.col_refused.is_some())
            .map(|(g, _)| *g)
            .collect()
    }

    /// The collider descriptor the bridge believes is **attached** to `guid`
    /// (C4-30).
    ///
    /// Exists so the claim can be falsified. `col_refused` is what makes the
    /// retry bounded, so a record that *also* latched `col` to the refused
    /// descriptor would behave identically through every other public path —
    /// which would leave the honesty of `col` an untestable assertion, and this
    /// campaign has already been caught by one of those. Measured: without this
    /// accessor the latch-restoring mutation passes the whole suite; with it,
    /// it fails by name.
    pub fn attached_collider_desc(&self, guid: Uuid) -> Option<&ColliderDesc3D> {
        self.entities.get(&guid).and_then(|r| r.col.as_ref())
    }

    /// The wrapped physics world (for scene queries, contact-event drain, etc.).
    pub fn world(&self) -> &PhysicsWorld3D {
        &self.world
    }

    /// Mutable access to the wrapped world (queries mutate the lazy query BVH).
    pub fn world_mut(&mut self) -> &mut PhysicsWorld3D {
        &mut self.world
    }

    /// The body handle mirroring `guid`, if it is tracked.
    /// How many entities the bridge is mirroring — real ones, `PcgVolume`
    /// solids and voxel chunks alike. A cheap, order-free handle on "the world
    /// the solver sees changed", which is what a gate needs when the change it is
    /// watching for is a collider that appeared or vanished under a carve.
    pub fn body_count(&self) -> usize {
        self.entities.len()
    }

    pub fn body_of(&self, guid: Uuid) -> Option<BodyId3D> {
        self.entities.get(&guid).map(|r| r.body)
    }

    /// The joint handle owned by `guid` (P12.1), if it has one.
    pub fn joint_of(&self, guid: Uuid) -> Option<JointId3D> {
        self.entities.get(&guid).and_then(|r| r.joint).map(|b| b.id)
    }

    /// The collider handle mirroring `guid`, if it has one.
    pub fn collider_of(&self, guid: Uuid) -> Option<ColliderId3D> {
        self.entities.get(&guid).and_then(|r| r.collider)
    }

    /// Every tracked `(guid, body, kind)`, in `Guid` order (P22.3).
    ///
    /// The seam a level-wide sweep — a radial impulse, a debug view — walks
    /// instead of reaching into rapier's own body set, because the rapier set is
    /// in handle-allocation order and the `Guid` order is the one every other
    /// pass in this bridge uses.
    pub fn tracked_bodies(&self) -> Vec<(Uuid, BodyId3D, BodyKind3D)> {
        self.entities
            .iter()
            .map(|(g, r)| (*g, r.body, r.kind))
            .collect()
    }

    /// The entity `Guid` owning `collider`, if tracked — the inverse of
    /// [`collider_of`](Self::collider_of), maintained each [`sync`](Self::sync)
    /// (Wave 3). The seam the collision-event drain uses to map a rapier
    /// [`ContactEvent3D`](crate::d3::ContactEvent3D) collider handle back to the
    /// entity it belongs to.
    pub fn guid_of_collider(&self, collider: ColliderId3D) -> Option<Uuid> {
        self.collider_to_guid.get(&collider).copied()
    }

    /// Advance the simulation by `dt` seconds (the caller's fixed step).
    pub fn step(&mut self, dt: f64) {
        self.world.step(dt);
    }

    /// Reconcile the physics world with the current ECS components: gather every
    /// entity carrying a [`RigidBody3D`] and/or [`Collider3D`] into an
    /// [`EntitySync3D`] snapshot (reading its `Transform` for the world pose —
    /// translation + rotation quaternion) and hand it to [`sync`](Self::sync),
    /// which reconciles in deterministic `Guid` order. The `d3` mirror of
    /// [`crate::d2::PhysicsBridge2D::sync_from_world`].
    pub fn sync_from_world(&mut self, world: &EcsWorld) {
        self.sync_from_world_with_voxels(world, &BTreeMap::new());
    }

    /// [`sync_from_world`](Self::sync_from_world) **plus the sim's voxel volumes**
    /// (P21.4), which become static trimesh colliders so a cave has a floor and a
    /// runtime carve is something a body can fall into.
    ///
    /// `volumes` is the *simulation's* map — `RuntimeSim::voxels` /
    /// `SimSession::voxels`, keyed by the volume's entity `Guid` — and emphatically
    /// **not** the render host's camera-paged store. A collider set that depended
    /// on where anyone was looking would put the floor under a player only while
    /// the camera happened to have paged it, which is the failure every seam in
    /// this phase is shaped to forbid.
    ///
    /// # Cost, and why the stamp is per chunk
    ///
    /// The first sync meshes every resident chunk once — a load-time cost, paid in
    /// the same place a level's colliders are always paid for. After that,
    /// [`gather_voxels`](Self::gather_voxels) re-describes **only chunks whose
    /// `VoxelData::chunk_version` moved**, which for a gameplay carve is the two or
    /// three the brush touched. `mesh_chunk` is the same deterministic Surface-Nets
    /// pass the renderer draws with, so the surface a body collides with and the
    /// surface a player sees are one extraction.
    pub fn sync_from_world_with_voxels(
        &mut self,
        world: &EcsWorld,
        volumes: &BTreeMap<Uuid, inf_voxel::VoxelData>,
    ) {
        self.sync_from_world_sim(world, volumes, &BTreeMap::new());
    }

    /// [`sync_from_world_with_voxels`](Self::sync_from_world_with_voxels) **plus
    /// the sim's fracture states** (P22.3) — the full sim-side sync both hosts
    /// call.
    ///
    /// `fractures` is the *simulation's* map (`RuntimeSim::fractures` /
    /// `SimSession::fractures`), keyed by the destructible actor's entity `Guid`.
    /// An actor whose state is still [`FractureState::is_intact`](super::fracture::FractureState::is_intact) contributes
    /// nothing here and keeps its authored collider; the first chunk to come off
    /// swaps that one collider for per-chunk bodies, atomically, in this pass.
    pub fn sync_from_world_sim(
        &mut self,
        world: &EcsWorld,
        volumes: &BTreeMap<Uuid, inf_voxel::VoxelData>,
        fractures: &BTreeMap<Uuid, super::fracture::FractureState>,
    ) {
        // **THE SWAP, and why it is decided before the walk.** An actor that has
        // started breaking must render and collide as chunks and NOT as itself —
        // never both, never neither. Both halves of that are one fact, so both
        // read one set: this one gates the collider below, and the render
        // projector's `is_intact` gates the mesh. Computing it up front is what
        // makes the swap atomic within the frame rather than an ordering
        // accident between two passes.
        let broken: BTreeSet<Uuid> = fractures
            .iter()
            .filter(|(_, s)| !s.is_intact())
            .map(|(g, _)| *g)
            .collect();
        let mut snaps: Vec<EntitySync3D> = std::mem::take(&mut self.snaps_scratch);
        snaps.clear();
        // P20.2: the water gather rides in THIS walk rather than in a second one.
        // A furnished town is 13 000 entities, and walking them twice per fixed
        // step to learn that a lake has not moved is the cost the change stamp
        // exists to avoid — spending it on the walk instead would be the same
        // mistake one level up.
        self.water_env = inf_ecs::sky::water_environment(world);
        self.water_scratch.clear();
        self.water_sources.clear();
        let mut buoyancy: Vec<(Uuid, BuoyancyDesc3D)> = Vec::new();
        for entity in world.world().iter_entities() {
            let Some(guid) = entity.get::<inf_ecs::Guid>().map(|g| g.0) else {
                continue;
            };
            if let Some(water) = entity.get::<WaterBody>() {
                let affine = entity.get::<GlobalTransform>().map(|g| g.0).unwrap_or({
                    // A water entity with no computed global transform yet falls
                    // back to its local one, then to the identity — the same
                    // tolerance the renderer's projector shows, for the same
                    // reason: an unpropagated frame is a timing artefact, not an
                    // authoring error.
                    entity
                        .get::<Transform>()
                        .map(|t| DAffine3::from_translation(t.translation.to_dvec3()))
                        .unwrap_or(DAffine3::IDENTITY)
                });
                let spline = entity.get::<Spline>().cloned();
                self.water_scratch
                    .push(WaterStamp::new(guid, *water, spline.as_ref(), affine));
                self.water_sources.push((guid, *water, spline, affine));
            }
            if let Some(b) = entity.get::<Buoyancy>() {
                if let Some(desc) = BuoyancyDesc3D::from_component(b) {
                    buoyancy.push((guid, desc));
                }
            }
            // A broken destructible's own body and collider are **gone**: the
            // chunk bodies below are what the world collides with now. Dropping
            // both makes `sync_retaining` skip the entity entirely, so the
            // despawn sweep removes the intact body — which is the swap. The
            // entity itself survives untouched in the scene document (it still
            // has a name, a transform and its components); it is only its
            // physical and visible representation that moved into the chunks.
            let broken_here = broken.contains(&guid);
            let rb = if broken_here {
                None
            } else {
                entity.get::<RigidBody3D>().copied()
            };
            let col = if broken_here {
                None
            } else {
                entity.get::<Collider3D>().copied()
            };
            let joint = entity.get::<Joint3D>().copied();
            if rb.is_none() && col.is_none() {
                continue;
            }
            let transform = entity
                .get::<Transform>()
                .copied()
                .unwrap_or(Transform::IDENTITY);
            snaps.push(EntitySync3D {
                guid,
                body: rb.map(body_desc),
                collider: col.as_ref().map(collider_desc),
                translation: transform.translation.to_dvec3(),
                rotation: transform.quat(),
                joint: joint.and_then(joint_sync),
            });
        }
        self.reconcile_water(buoyancy);
        // P19.5: a volume's derived solids. Unchanged volumes are **retained**
        // rather than re-described — see the doc on `structure_stamps`.
        let mut retained = self.gather_structures(world, &mut snaps);
        // P21.4: the sim's voxel chunks, on the same rule one level finer.
        self.gather_voxels(volumes, &mut snaps, &mut retained);
        // P22.3: the sim's terrain tiles, on the same rule again. Before this the
        // ground had NO collider at all — the heightfield answered
        // `terrain.height_at` and a character controller read it, but a dynamic
        // body (a crate, a chunk of a shattered wall) fell through the world.
        self.gather_terrain(world, &mut snaps, &mut retained);
        // P22.3: the sim's fracture chunks, on the same rule a fourth time.
        self.gather_fracture(fractures, &mut snaps, &mut retained);
        // `sync` sorts by Guid internally, so the gather order here is irrelevant.
        self.sync_retaining(&snaps, &retained);
        snaps.clear();
        self.snaps_scratch = snaps;
    }

    /// Bring the water index and the buoyant set in line with what the walk
    /// found. The index is rebuilt only when the **stamp** moved — see
    /// [`WaterStamp`].
    fn reconcile_water(&mut self, mut buoyancy: Vec<(Uuid, BuoyancyDesc3D)>) {
        // Guid order, always: the index's body indices are what a water event
        // names, and an index that depended on ECS archetype order would make the
        // events depend on spawn history.
        self.water_scratch.sort_by_key(|s| s.guid());
        if self.water_scratch != self.water_stamps {
            self.water_stamps.clear();
            self.water_stamps.extend_from_slice(&self.water_scratch);
            self.water_sources.sort_by_key(|(g, _, _, _)| *g);
            let env = self.water_env;
            let entries = self
                .water_sources
                .iter()
                .filter_map(|(guid, body, spline, affine)| {
                    water::water_surface_of(body, spline.as_ref(), affine, env)
                        .map(|s| water::water_entry(*guid, s))
                })
                .collect();
            self.water.rebuild(entries);
        }
        // The buoyant set is rebuilt every sync (it is tiny), but each entry's
        // **latch** is carried over so an enter/exit does not re-fire because a
        // drag coefficient was edited.
        buoyancy.sort_by_key(|(g, _)| *g);
        let mut next = BuoyantMap::new();
        for (guid, desc) in buoyancy {
            let state = self.buoyant.get(&guid).map(|(_, s)| *s).unwrap_or_default();
            next.insert(guid, (desc, state));
        }
        // A body that STOPPED being buoyant (its component was removed or
        // disabled) still carries last step's water force, and a rapier force
        // persists. Clearing it here is the other half of the ownership the apply
        // pass claims — without it, deleting a `Buoyancy` would leave the body
        // rising forever.
        let dropped: Vec<BodyId3D> = self
            .buoyant
            .keys()
            .filter(|g| !next.contains_key(*g))
            .filter_map(|g| self.entities.get(g).map(|r| r.body))
            .collect();
        for body in dropped {
            self.world.reset_forces(body);
        }
        self.buoyant = next;
    }

    /// Append descriptors for every `PcgVolume` whose solids **changed**, and
    /// return the guids of the ones that did not (which must survive the despawn
    /// sweep without being rebuilt).
    ///
    /// This is the whole point of the change stamp: a furnished town is ~13 000
    /// immovable boxes, and describing + sorting them at 60 Hz to learn that a
    /// wall has not moved is a per-step cost a load-time budget never sees.
    fn gather_structures(
        &mut self,
        world: &EcsWorld,
        snaps: &mut Vec<EntitySync3D>,
    ) -> BTreeSet<Uuid> {
        let mut retained: BTreeSet<Uuid> = BTreeSet::new();
        // A level with no `PcgVolume` at all — every hand-authored one — used to
        // walk every entity in the world per fixed step to establish that
        // (lens 3 P12). The stamp-map clause is what keeps the *last* volume's
        // disappearance reaching the prune below; `gather_voxels`' guard is the
        // same shape.
        if self.structure_stamps.is_empty() && !world.has_component::<PcgVolume>() {
            return retained;
        }
        let mut live_volumes: BTreeSet<Uuid> = BTreeSet::new();
        for entity in world.world().iter_entities() {
            let Some(guid) = entity.get::<inf_ecs::Guid>().map(|g| g.0) else {
                continue;
            };
            let Some(vol) = entity.get::<PcgVolume>() else {
                continue;
            };
            live_volumes.insert(guid);
            let stamp = (vol.structures_gen, vol.structures.len());
            if self.structure_stamps.get(&guid) == Some(&stamp) {
                retained.extend((0..stamp.1).map(|i| pcg_structure_guid(guid, i)));
                continue;
            }
            self.structure_stamps.insert(guid, stamp);
            snaps.extend(structure_snaps_of(guid, vol));
        }
        // A volume that disappeared drops its stamp, so a later volume reusing
        // the guid cannot inherit a stale one.
        self.structure_stamps
            .retain(|g, _| live_volumes.contains(g));
        retained
    }

    /// Append a static trimesh collider for every voxel chunk whose field
    /// **changed**, and add the unchanged ones to `retained` so the despawn sweep
    /// leaves them alone (P21.4).
    ///
    /// The [`gather_structures`](Self::gather_structures) rule, one level finer.
    /// A cave system is hundreds of chunks and a gameplay carve moves two of them;
    /// re-meshing the rest at 60 Hz to learn that the far wall has not moved is the
    /// cost the stamp exists to refuse. `VoxelData::chunk_version` is the stamp —
    /// minted by the store on every mutating touch, and *not* by paging, which is
    /// what makes "changed" mean *dug* rather than *loaded*.
    ///
    /// **A chunk that meshes to nothing gets no collider**, and its stamp is still
    /// recorded — so the empty result is reached once, not once per step. Carving a
    /// chunk hollow therefore *removes* its collider on the next sync (the key is
    /// no longer in `snaps` and not in `retained`, so the sweep takes it), which is
    /// how a runtime carve becomes a hole a body can fall through.
    ///
    /// Order is `BTreeMap` over `(entity, chunk key)`, and `sync_retaining` sorts
    /// by `Guid` again on top of that, so the handles rapier allocates are a
    /// function of the content and of nothing else.
    fn gather_voxels(
        &mut self,
        volumes: &BTreeMap<Uuid, inf_voxel::VoxelData>,
        snaps: &mut Vec<EntitySync3D>,
        retained: &mut BTreeSet<Uuid>,
    ) {
        // The fast path a level with no voxels takes: no walk, no allocation, and
        // the stale-stamp prune below is a no-op on an empty map.
        if volumes.is_empty() && self.voxel_stamps.is_empty() {
            return;
        }
        let mut live: BTreeSet<(Uuid, inf_voxel::ChunkKey)> = BTreeSet::new();
        for (&entity, data) in volumes {
            let voxel_size_m = data.voxel_size_m();
            // **The mesher's own key set, and the mesher's own stamp.** Both were
            // wrong here in the first cut of P21.4, and both were wrong in ways
            // that produce a *silent phantom wall* rather than a crash:
            //
            // * `resident_keys()` is not the surface. A cell owned by chunk `K`
            //   has corners in `K + 1`, so the surface of a volume lives on
            //   `mesh_keys_for` — the resident set **closed downward by one chunk
            //   on each axis** — which `inf_voxel::mesh` calls "a correctness
            //   requirement, not defensive padding". Iterating residency instead
            //   left 37 % of this phase's own flagship sample drawn and not
            //   collidable (10 936 rendered triangles against 6 874 collidable
            //   ones; the whole −X/−Y/−Z faces were walk-through).
            //
            // * `chunk_version` is not the mesh's identity. A mesh is a function
            //   of its 3×3×3 neighbourhood, so a doorway carved wholly inside
            //   chunk `K + 1` never moves `K`'s own stamp and `K` keeps a wall
            //   across the opening — the exact M3 defect P21.1 paid for in the
            //   renderer, re-introduced here. `source_key` is the answer that
            //   already exists: the max stamp over the neighbourhood **plus a
            //   residency mask**, so an eviction invalidates too.
            //
            // Using the mesher's own two functions is also what makes
            // `inf-physics`' "the SAME mesher the renderer draws with" true rather
            // than aspirational: same key set, same invalidation, same triangles.
            for key in inf_voxel::mesh_keys_for(data) {
                live.insert((entity, key));
                let version = inf_voxel::source_key(data, key);
                if self.voxel_stamps.get(&(entity, key)) == Some(&version) {
                    retained.insert(voxel_chunk_guid(entity, key));
                    continue;
                }
                self.voxel_stamps.insert((entity, key), version);
                let mesh = inf_voxel::mesh_chunk(data, key);
                if mesh.is_empty() {
                    continue;
                }
                // Chunk-local metres against the chunk's own `f64` world origin —
                // the floating-origin split `VoxelMesh::local_positions_m` exists
                // to make one function, so the collider surface and the drawn
                // surface cannot be a fraction of a voxel apart.
                let vertices: Vec<DVec3> = mesh
                    .local_positions_m(voxel_size_m)
                    .into_iter()
                    .map(|p| DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
                    .collect();
                let indices: Vec<[u32; 3]> = mesh
                    .indices
                    .chunks_exact(3)
                    .map(|t| [t[0], t[1], t[2]])
                    .collect();
                snaps.push(EntitySync3D {
                    guid: voxel_chunk_guid(entity, key),
                    // No body: the bridge's implicit-static-body rule gives it a
                    // static parent. Rock does not fall, and rapier cannot give a
                    // trimesh a well-defined mass anyway.
                    body: None,
                    collider: Some(ColliderDesc3D::new(ColliderShape3D::Trimesh {
                        vertices,
                        indices,
                    })),
                    translation: data.chunk_origin_world(key),
                    rotation: DQuat::IDENTITY,
                    joint: None,
                });
            }
        }
        // A chunk (or a whole volume) that went away drops its stamp, so a key that
        // comes back is re-described rather than inheriting a stale version — the
        // `structure_stamps` prune's reasoning, and the `swimming` prune's.
        self.voxel_stamps.retain(|k, _| live.contains(k));
    }

    /// Append a static height-field collider for every **sim-resident** terrain
    /// tile whose contents (or whose terrain's origin) changed, and add the
    /// unchanged ones to `retained` (P22.3).
    ///
    /// # Sim-resident, and why that phrase is the whole design
    ///
    /// The tiles walked here are the ones in the ECS [`Terrain`] component's own
    /// [`TerrainData`](inf_terrain::TerrainData). That set is **camera-free by
    /// construction**, and structurally so rather than by convention: the
    /// determinism doctrine in `inf_terrain::wants` keeps two residency drivers
    /// apart and gives them two *different* `TerrainData` instances — the sim's
    /// lives in this component and pages against `sim_wants` (entity positions +
    /// a fixed margin, loaded synchronously at the fixed-step boundary), while the
    /// camera-driven render cut lives in the streamer, outside the world entirely,
    /// with no reference from here to there. So there is no way for this gather to
    /// reach a camera-paged tile even by mistake, which is the property that keeps
    /// the collider set — and therefore every replay and every PIE-vs-shipping
    /// trace — independent of where anyone was looking. It is the same rule
    /// `sync_from_world_with_voxels` states for the voxel map, one crate over.
    ///
    /// Only **level-0** tiles get colliders. The coarse pyramid levels are a
    /// rendering LOD; two overlapping surfaces at different resolutions would
    /// double-contact a body standing on the seam, and the sim never asks a coarse
    /// tile for a height either ([`TerrainData::height_at`] reads level 0).
    ///
    /// # Holes: the cell rule, and its consistency with `height_at`
    ///
    /// **A height-field cell is removed when ANY of its four corner samples is
    /// holed** — the identical rule
    /// [`TerrainData::height_at`](inf_terrain::TerrainData::height_at) applies to
    /// decide that a bilinear cell has no surface ("THE POISON RULE", P21.2), and
    /// the identical rule `terrain.wgsl` discards a fragment on. All three are one
    /// statement: *the visible gap, the queryable gap and the walkable gap are the
    /// same gap.* A body dropped over a cave mouth falls into it, and the voxel
    /// chunk colliders from [`gather_voxels`](Self::gather_voxels) are what catch
    /// it — the P21.2 combined ground query, now with real physics under it.
    ///
    /// If this rule and `height_at`'s ever diverge the symptom is a character
    /// controller (which reads `height_at`) standing on air a metre from a
    /// dynamic body (which reads this) resting on rock, so the statement is
    /// repeated at both sites deliberately.
    ///
    /// # Cost
    ///
    /// Bounded by the sim-resident tile count, which is already budgeted by
    /// `sim_wants`' margin. Within that, a tile is re-described only when its
    /// version stamp or its terrain's origin moved — see
    /// [`terrain_stamps`](Self::terrain_stamps) — so a level whose ground nobody
    /// is sculpting pays one build at load and nothing per step.
    /// [`terrain_collider_audit`](Self::terrain_collider_audit) counts what it
    /// actually did.
    fn gather_terrain(
        &mut self,
        world: &EcsWorld,
        snaps: &mut Vec<EntitySync3D>,
        retained: &mut BTreeSet<Uuid>,
    ) {
        // An interior level, or any level with no `Terrain` component, walked
        // every entity per fixed step to find out (lens 3 P12). The audit is
        // still published — a zeroed audit is the honest answer for a world with
        // no terrain in it, and it is what the previous walk produced too.
        if self.terrain_stamps.is_empty() && !world.has_component::<Terrain>() {
            self.terrain_audit = TerrainColliderAudit::default();
            return;
        }
        let mut live: BTreeSet<(Uuid, (i32, i32))> = BTreeSet::new();
        let mut described = 0_u32;
        let mut resident = 0_u32;
        // Walk in `Guid` order rather than archetype order: the synthetic guids
        // below are sorted by `sync_retaining` anyway, but the *stamp* map's
        // insertion order is observable through `terrain_stamps` and a gate reads
        // it. Collected first because the descriptor build needs no borrow.
        let mut terrains: Vec<(Uuid, DVec3)> = Vec::new();
        for entity in world.world().iter_entities() {
            let Some(guid) = entity.get::<inf_ecs::Guid>().map(|g| g.0) else {
                continue;
            };
            let Some(t) = entity.get::<Terrain>() else {
                continue;
            };
            if t.data.is_empty() {
                continue;
            }
            // The same origin resolution `terrain.height_at` uses in both hosts:
            // the computed global transform, falling back to the local one and
            // then to the world origin. An unpropagated frame is a timing
            // artefact, not an authoring error.
            let origin = entity
                .get::<GlobalTransform>()
                .map(|g| g.translation())
                .or_else(|| entity.get::<Transform>().map(|t| t.translation.to_dvec3()))
                .unwrap_or(DVec3::ZERO);
            terrains.push((guid, origin));
        }
        terrains.sort_by_key(|(g, _)| *g);
        if terrains.is_empty() && self.terrain_stamps.is_empty() {
            self.terrain_audit = TerrainColliderAudit::default();
            return;
        }
        for (guid, origin) in terrains {
            let Some(entity) = world.entity_of(guid) else {
                continue;
            };
            let Some(terrain) = world.world().get::<Terrain>(entity) else {
                continue;
            };
            let data = &terrain.data;
            let res = data.tile_resolution();
            let span = data.tile_span();
            // The origin rides in the stamp as raw bits: a terrain that MOVED has
            // to re-place every tile, and `f64 == f64` on a translation that was
            // copied rather than recomputed is exact — but bit equality is the
            // honest comparison for a cache key, and it makes `-0.0` and `NaN`
            // behave (they compare unequal to their own re-derivation, so the
            // conservative answer is "re-describe").
            let origin_bits = [origin.x.to_bits(), origin.y.to_bits(), origin.z.to_bits()];
            for (&coord, tile) in data.tiles() {
                resident += 1;
                live.insert((guid, coord));
                let stamp = (data.tile_version(TileKey::lod0(coord)), origin_bits);
                if self.terrain_stamps.get(&(guid, coord)) == Some(&stamp) {
                    retained.insert(terrain_tile_guid(guid, coord));
                    continue;
                }
                self.terrain_stamps.insert((guid, coord), stamp);
                described += 1;
                let Some(collider) = terrain_tile_collider(tile, res, span) else {
                    // A degenerate tile (resolution < 2, a short height buffer)
                    // gets no collider and still records its stamp, so the
                    // refusal is reached once rather than once per step — the
                    // `gather_voxels` empty-mesh rule.
                    continue;
                };
                snaps.push(EntitySync3D {
                    guid: terrain_tile_guid(guid, coord),
                    // No body: the implicit-static-body rule gives it a static
                    // parent. Ground does not fall, and a height field has no
                    // well-defined mass anyway (`ColliderShape3D::volume_m3`).
                    body: None,
                    collider: Some(collider),
                    // The field is CENTRED in its own local frame, so the body
                    // goes at the tile's centre and not at its corner sample.
                    translation: origin
                        + DVec3::new(
                            tile.origin.x + span * 0.5,
                            tile.origin.y,
                            tile.origin.z + span * 0.5,
                        ),
                    rotation: DQuat::IDENTITY,
                    joint: None,
                });
            }
        }
        // A tile (or a whole terrain) that paged out drops its stamp, so a key
        // that comes back is re-described rather than inheriting a stale version
        // — the `voxel_stamps` prune's reasoning, and the `structure_stamps`
        // prune's. This is also what makes paging in and out leak nothing: the
        // guid leaves `live`, so `sync_retaining`'s despawn sweep removes the body
        // and this line removes the stamp.
        self.terrain_stamps.retain(|k, _| live.contains(k));
        self.terrain_audit = TerrainColliderAudit {
            resident_tiles: resident,
            described,
            retained: resident.saturating_sub(described),
        };
    }

    /// **Follow the world**: bring every still-intact destructible's placement in
    /// line with its actor's `GlobalTransform` (P22.3).
    ///
    /// Called by both hosts at the top of the fixed step, before
    /// [`sync_from_world_sim`](Self::sync_from_world_sim). Separate from the sync
    /// because the sync borrows the fracture map immutably (the render projector
    /// and the collider gather both read it) while this writes to it, and because
    /// "where is this actor" is a question about the *world*, answered once,
    /// rather than something the collider gather should be re-deriving.
    ///
    /// A no-op for a broken actor (its chunks are solver-owned) and for one that
    /// has not moved, so the steady state is one affine comparison per
    /// destructible.
    pub fn follow_fractures(
        world: &EcsWorld,
        fractures: &mut BTreeMap<Uuid, super::fracture::FractureState>,
    ) {
        if fractures.is_empty() {
            return;
        }
        for (guid, state) in fractures.iter_mut() {
            if !state.is_intact() {
                continue;
            }
            let Some(entity) = world.entity_of(*guid) else {
                continue;
            };
            let w = world.world();
            let placement = w
                .get::<GlobalTransform>(entity)
                .map(|g| g.0)
                .unwrap_or_else(|| {
                    w.get::<Transform>(entity)
                        .map(|t| DAffine3::from_translation(t.translation.to_dvec3()))
                        .unwrap_or(DAffine3::IDENTITY)
                });
            state.follow(placement);
        }
    }

    /// Append a collider for every live fracture chunk of every **broken**
    /// destructible, and add the unchanged actors' chunks to `retained` (P22.3).
    ///
    /// The [`gather_voxels`](Self::gather_voxels) rule again, keyed by actor
    /// rather than by chunk:
    ///
    /// * An **intact** actor emits nothing. Its authored collider is still in the
    ///   world (the walk left it there), and describing twelve chunk bodies for
    ///   every destructible wall in a town that nobody has shot at is precisely
    ///   the bill the stamp pattern exists to refuse.
    /// * A **broken** actor's chunks are described together, keyed on the state's
    ///   [`generation`](super::fracture::FractureState::generation) — which moves
    ///   when a chunk detaches or is reclaimed and **not** when one merely
    ///   tumbles, because a dynamic body's pose is solver-owned and re-describing
    ///   it every step would teleport it back to where it broke.
    /// * An **attached** chunk is a *static* body at its rest pose: the standing
    ///   half of a half-collapsed wall still stops bullets and holds up the
    ///   chunks above it.
    /// * A **detached** chunk is a *dynamic* [`ColliderShape3D::ConvexHull`] with
    ///   a real mass — `density_kg_m3 × volume_m3 × scale³` — which is the whole
    ///   reason the hull variant exists (a trimesh has no volume, so a trimesh
    ///   chunk would have no inertia to simulate).
    /// * A **reclaimed** chunk is emitted by nobody and retained by nobody, so
    ///   the despawn sweep takes it. Despawn-by-omission, the third time.
    ///
    /// Debris goes on its own collision layer ([`DEBRIS_LAYER`]) so a game can
    /// stop rubble from blocking a doorway or from tripping a trigger volume
    /// without touching anything else. CCD is **off** by default: a fracture
    /// event makes tens of bodies at once and continuous detection is the most
    /// expensive thing in the solver, so it is opt-in-by-editing rather than
    /// on-by-default (documented in `DEBRIS_LAYER`).
    fn gather_fracture(
        &mut self,
        fractures: &BTreeMap<Uuid, super::fracture::FractureState>,
        snaps: &mut Vec<EntitySync3D>,
        retained: &mut BTreeSet<Uuid>,
    ) {
        if fractures.is_empty() && self.fracture_stamps.is_empty() {
            return;
        }
        let mut live: BTreeSet<Uuid> = BTreeSet::new();
        for (&entity, state) in fractures {
            if state.is_intact() {
                continue;
            }
            live.insert(entity);
            let retain = self.fracture_stamps.get(&entity) == Some(&state.generation());
            if !retain {
                self.fracture_stamps.insert(entity, state.generation());
            }
            for (i, chunk) in state.chunks().iter().enumerate() {
                if chunk.gone {
                    continue;
                }
                let guid = super::fracture::fracture_chunk_guid(entity, i as u32);
                if retain {
                    retained.insert(guid);
                    continue;
                }
                let Some(collider) = state.chunk_collider(i, debris_layers()) else {
                    // A hull the cook's own producer gate should already have
                    // refused. Skipping it leaves a chunk that renders and does
                    // not collide, which is strictly better than a massless
                    // dynamic body — and the stamp is still recorded above, so
                    // the refusal is reached once.
                    continue;
                };
                let (translation, rotation) = state.chunk_pose(i);
                snaps.push(EntitySync3D {
                    guid,
                    body: Some(BodyDesc3D {
                        kind: if chunk.detached {
                            BodyKind3D::Dynamic
                        } else {
                            BodyKind3D::Static
                        },
                        ..BodyDesc3D::default()
                    }),
                    collider: Some(collider),
                    translation,
                    rotation,
                    joint: None,
                });
            }
        }
        // An actor that became whole again (impossible today) or vanished drops
        // its stamp, so a guid that comes back is re-described rather than
        // inheriting a stale generation — the `voxel_stamps` prune's reasoning.
        self.fracture_stamps.retain(|g, _| live.contains(g));
    }

    /// What the last `gather_terrain` pass actually did
    /// (P22.3) — the budget instrument. `described == 0` with
    /// `resident_tiles > 0` is the steady state the change stamp exists to
    /// produce.
    pub fn terrain_collider_audit(&self) -> TerrainColliderAudit {
        self.terrain_audit
    }

    /// How many terrain tiles currently hold a collider stamp — the bounded-ledger
    /// invariant made observable, so a paging soak can assert it does not grow.
    pub fn terrain_stamp_count(&self) -> usize {
        self.terrain_stamps.len()
    }

    /// Reconcile the physics world with a scene snapshot: spawn new
    /// bodies/colliders, update changed ones, and despawn bodies whose entity
    /// disappeared. Runs in `Guid` order regardless of the input order, so the
    /// result is independent of how the caller gathered the snapshot.
    pub fn sync(&mut self, entities: &[EntitySync3D]) {
        self.sync_retaining(entities, &BTreeSet::new());
    }

    /// [`sync`](Self::sync), plus a set of guids that are still alive but were
    /// deliberately **not** re-described this pass (P19.5's unchanged
    /// `PcgVolume` solids). They survive the despawn sweep untouched.
    fn sync_retaining(&mut self, entities: &[EntitySync3D], retained: &BTreeSet<Uuid>) {
        // 1. Sort the snapshot into deterministic Guid order (and drop entities
        //    with neither a body nor a collider — nothing to simulate).
        let mut live: Vec<&EntitySync3D> = entities
            .iter()
            .filter(|e| e.body.is_some() || e.collider.is_some())
            .collect();
        live.sort_by_key(|e| e.guid);

        // 2. Spawn / update. (Joints are reconciled in a second pass, below, once
        //    every body exists, so a joint can always resolve its other body.)
        //
        // `seen` is a **sorted `Vec`, not a `BTreeSet`** (Hardening Wave G):
        // `live` was just sorted by guid, so pushing is already sorted, and the
        // one consumer — the despawn sweep below — needs `contains`, which a
        // `binary_search` answers at the same complexity without one B-tree node
        // allocation per tracked entity per fixed step.
        let mut seen: Vec<Uuid> = Vec::with_capacity(live.len());
        // Only entities that HAVE a joint or that HAD one need pass 4 (P30). A
        // level with no joints — which is most of them, and all of the 13 000
        // static colliders a furnished town is made of — leaves this empty and
        // skips the pass entirely, instead of paying two `BTreeMap` probes per
        // entity to be told there is nothing to reconcile.
        let mut joint_desires: Vec<(Uuid, Option<JointSync3D>)> = Vec::new();
        for snap in live {
            seen.push(snap.guid);
            let kind = snap.body.map(|b| b.kind).unwrap_or(BodyKind3D::Static);
            let pos = snap.translation;
            let rot = snap.rotation;

            if let Some(rec) = self.entities.get(&snap.guid) {
                let rec_kind = rec.kind;
                let rec_rb = rec.rb;
                // Compared **through the borrow**, never cloned (Hardening Wave
                // G). These are `ColliderDesc3D`s, and the trimesh and hull
                // variants own their vertex and index buffers: cloning both of
                // them per tracked entity per fixed step deep-copied every voxel
                // chunk's and every terrain tile's geometry to answer a question
                // that is a comparison.
                let col_changed =
                    rec.col.as_ref() != snap.collider.as_ref() && rec.col_refused != snap.collider;
                let has_joint_work = snap.joint.is_some() || rec.joint.is_some();
                let old_collider = rec.collider;
                let body = rec.body;

                if has_joint_work {
                    joint_desires.push((snap.guid, snap.joint));
                }
                if rec_kind != kind {
                    self.world.set_body_kind(body, kind);
                    if let Some(r) = self.entities.get_mut(&snap.guid) {
                        r.kind = kind;
                    }
                }
                // Static/kinematic follow their Transform; dynamic is
                // solver-owned. The write is skipped when the body is already
                // there — see `set_body_pose_if_moved` for why the comparison is
                // against rapier's own state and what it costs not to skip.
                if kind != BodyKind3D::Dynamic {
                    self.world.set_body_pose_if_moved(body, pos, rot);
                }
                if rec_rb != snap.body {
                    if let Some(rb) = snap.body.as_ref() {
                        apply_rb_props(&mut self.world, body, rb);
                    }
                    if let Some(r) = self.entities.get_mut(&snap.guid) {
                        r.rb = snap.body;
                    }
                }
                // Rebuild the collider so shape/material edits take effect —
                // unless this exact descriptor has already been tried and
                // refused, in which case retrying is a pure function returning
                // the same answer at trimesh-topology cost per step (C4-30).
                if col_changed {
                    if let Some(old) = old_collider {
                        self.world.remove_collider(old);
                    }
                    let (new_col, achieved, refused) =
                        self.attach_collider(snap.guid, body, snap.collider.as_ref());
                    if let Some(r) = self.entities.get_mut(&snap.guid) {
                        r.collider = new_col;
                        self.collider_map_dirty = true;
                        r.col = achieved;
                        r.col_refused = refused;
                    }
                }
            } else {
                if snap.joint.is_some() {
                    joint_desires.push((snap.guid, snap.joint));
                }
                // New entity → create its body, apply props, attach a collider.
                let body = self.world.add_body(kind, pos, rot);
                if let Some(rb) = snap.body.as_ref() {
                    apply_rb_props(&mut self.world, body, rb);
                }
                let (collider, achieved, refused) =
                    self.attach_collider(snap.guid, body, snap.collider.as_ref());
                self.collider_map_dirty = true;
                self.entities.insert(
                    snap.guid,
                    BodyRecord {
                        body,
                        collider,
                        kind,
                        rb: snap.body,
                        col: achieved,
                        col_refused: refused,
                        joint: None,
                    },
                );
            }
        }

        // 3. Despawn: any tracked guid not seen this sync is gone. Removing the
        //    body drops its colliders AND any joints attached to it (rapier), so a
        //    joint whose endpoint despawns is cleaned up here; the owning record's
        //    stale handle is reconciled to `None` in pass 4.
        //
        //    A **merge**, not a probe per entity (Hardening Wave G). Both sides
        //    are already in guid order — `seen` because `live` was just sorted
        //    into it, `entities` because it is a `BTreeMap` — so one cursor
        //    walked forward answers in a linear sequential pass what a
        //    `contains` per tracked entity answered with a random descent
        //    through a 13 000-entry tree, sixty times a second, to conclude that
        //    nothing had despawned.
        let mut gone: Vec<Uuid> = Vec::new();
        let mut cursor = 0usize;
        for guid in self.entities.keys() {
            while cursor < seen.len() && seen[cursor] < *guid {
                cursor += 1;
            }
            if seen.get(cursor) == Some(guid) || retained.contains(guid) {
                continue;
            }
            gone.push(*guid);
        }
        for guid in gone {
            self.collider_map_dirty = true;
            if let Some(rec) = self.entities.remove(&guid) {
                self.world.remove_body(rec.body);
            }
        }

        // 4. Reconcile joints (P12.1), now that every body exists. In Guid
        //    order, over the entities that have a joint or had one — an entity
        //    that has never had one has nothing for this pass to reconcile and
        //    is not in the list (P30).
        for (guid, desire) in joint_desires {
            self.reconcile_joint(guid, desire);
        }

        // 5. Rebuild the reverse collider→Guid map for this step's event drain
        //    (Wave 3), when a handle moved. Always consistent with the handles
        //    just reconciled above — see `collider_map_dirty`.
        if self.collider_map_dirty {
            self.collider_to_guid = self
                .entities
                .iter()
                .filter_map(|(g, r)| r.collider.map(|c| (c, *g)))
                .collect();
            self.collider_map_dirty = false;
        }

        // 6. P20.2: a despawned character forgets it was swimming, so a later
        //    entity reusing the guid cannot inherit a stale latch — the same
        //    reasoning as the `structure_stamps` prune above.
        let mut swimming = std::mem::take(&mut self.swimming);
        swimming.retain(|g, _| self.entities.contains_key(g));
        self.swimming = swimming;
    }

    /// Bring one entity's joint in line with its desired snapshot. Resolves the
    /// other body from the tracked entity map; rebuilds the joint if the snapshot,
    /// the resolved other body, or its very existence changed; removes it if the
    /// desire is `None` or the other body is missing/despawned.
    fn reconcile_joint(&mut self, guid: Uuid, desire: Option<JointSync3D>) {
        // The self body must still exist.
        let Some(self_body) = self.entities.get(&guid).map(|r| r.body) else {
            return;
        };
        let existing = self.entities.get(&guid).and_then(|r| r.joint);
        // Resolve the desired other body (skip self-links and dangling refs).
        let target = desire.and_then(|d| {
            if d.other == guid {
                return None;
            }
            self.entities.get(&d.other).map(|r| (r.body, d))
        });

        match target {
            Some((other_body, sync)) => {
                let up_to_date = existing.is_some_and(|b| {
                    b.other_body == other_body && b.sync == sync && self.world.contains_joint(b.id)
                });
                if up_to_date {
                    return;
                }
                if let Some(b) = existing {
                    self.world.remove_joint(b.id);
                }
                let id = self.world.add_joint(self_body, other_body, sync.desc);
                if let Some(r) = self.entities.get_mut(&guid) {
                    r.joint = id.map(|id| JointBinding {
                        id,
                        other_body,
                        sync,
                    });
                }
            }
            None => {
                if let Some(b) = existing {
                    self.world.remove_joint(b.id);
                    if let Some(r) = self.entities.get_mut(&guid) {
                        r.joint = None;
                    }
                }
            }
        }
    }

    // ── water (P20.2) ─────────────────────────────────────────────────────

    /// **The water force pass.** Apply buoyancy + hydrodynamic drag to every
    /// buoyant body, arm this step's enter/exit/splash events, and return.
    ///
    /// Runs **between [`sync_from_world`](Self::sync_from_world) and
    /// [`step`](Self::step)** — after the sync because a body has to be sampled
    /// where it is, before the step because rapier clears force accumulators every
    /// step. See the [`water`](super::water) module docs for the full ordering law.
    ///
    /// Takes no world: everything it needs was gathered during the sync, which is
    /// the strongest available statement that a water force cannot depend on
    /// anything the renderer owns.
    ///
    /// **Off-path cost is one branch.** A level with no `Buoyancy` component
    /// returns immediately, without enumerating a single rigid body.
    pub fn apply_water_forces(&mut self, dt: f64) {
        self.water_events.clear();
        if self.buoyant.is_empty() {
            return;
        }
        let t = self.water_env.0;
        let gravity = self.world.gravity();
        // Phase 1 — solve, read-only, in Guid order (the bridge discipline).
        struct Plan {
            guid: Uuid,
            body: BodyId3D,
            forces: water::WaterForces,
            hysteresis_m: f64,
            up_speed: f64,
        }
        let mut plans: Vec<Plan> = Vec::new();
        // Bodies that are buoyant and dynamic but that this pass will NOT plan a
        // force for — a `Buoyancy` on a body with no collider has nothing to
        // displace, and one whose handles just went stale has nothing to push.
        // They still get their accumulator cleared, because a rapier force
        // persists: the invariant this pass maintains is **"a body that could
        // ever have felt a water force is reset every step"**, with no exceptions,
        // so a crate that loses its collider does not keep last step's lift
        // forever.
        let mut reset_only: Vec<BodyId3D> = Vec::new();
        for (guid, (desc, _)) in &self.buoyant {
            let Some(rec) = self.entities.get(guid) else {
                continue;
            };
            // Only the solver's own bodies float: a static wall does not, and a
            // kinematic platform is script-driven by definition. A character
            // controller is kinematic and gets swim mode instead. A non-dynamic
            // body needs no reset either — rapier ignores forces on it.
            if rec.kind != BodyKind3D::Dynamic {
                continue;
            }
            let Some(col) = rec.col.as_ref() else {
                reset_only.push(rec.body);
                continue;
            };
            let Some(state) = self.body_state(rec.body) else {
                reset_only.push(rec.body);
                continue;
            };
            let geo = water::sample_geometry(&col.shape, col.local_translation);
            let forces = water::solve(&self.water, t, &state, &geo, desc, gravity, dt);
            let up = if gravity.length_squared() > 0.0 {
                -gravity.normalize()
            } else {
                DVec3::Y
            };
            plans.push(Plan {
                guid: *guid,
                body: rec.body,
                hysteresis_m: water::exit_hysteresis_m(&geo),
                up_speed: state.linvel.dot(up),
                forces,
            });
        }
        // Phase 2 — apply, as real forces. Two rapier laws govern this loop and
        // both are written out on [`PhysicsWorld3D::apply_force_at_point`]: a
        // rapier force **persists** until reset (so last step's is cleared first,
        // for every buoyant body, whether or not it is wet this step), and a force
        // is **not** an impulse of `F · dt` for the position (rapier substeps, so
        // a front-loaded impulse leaves a neutrally buoyant body drifting upward
        // about a millimetre per step while its velocity stays exactly zero).
        //
        // The reset is scoped to bodies the water pass owns. Nothing else applies
        // a persistent force to them — there is no `apply_force` Blueprint node,
        // and both hosts use impulses — so the ownership is total rather than
        // shared.
        for body in reset_only {
            self.world.reset_forces(body);
        }
        for plan in plans {
            self.world.reset_forces(plan.body);
            for (force, point) in plan.forces.samples {
                if force != DVec3::ZERO {
                    self.world.apply_force_at_point(plan.body, force, point);
                }
            }
            if plan.forces.drag != DVec3::ZERO {
                self.world.apply_force(plan.body, plan.forces.drag);
            }
            if plan.forces.torque != DVec3::ZERO {
                self.world.apply_torque(plan.body, plan.forces.torque);
            }
            if let Some((_, state)) = self.buoyant.get_mut(&plan.guid) {
                water::crossing_events(
                    plan.guid,
                    state,
                    &plan.forces.probe,
                    plan.hysteresis_m,
                    plan.up_speed,
                    &mut self.water_events,
                );
            }
        }
    }

    /// Take this step's water crossings, in body-`Guid` order (`Enter`/`Exit`
    /// before the `Splash` that accompanies it). Drained in the same slot as the
    /// collision events, so the fixed step has one event point rather than two.
    pub fn drain_water_events(&mut self) -> Vec<WaterEvent3D> {
        std::mem::take(&mut self.water_events)
    }

    /// The level's water index — the seam the `water.*` Blueprint nodes and any
    /// debug view read.
    pub fn water(&self) -> &WaterIndex {
        &self.water
    }

    /// `(level clock seconds, weather wind m/s)` as of the last sync.
    pub fn water_env(&self) -> (f64, (f64, f64)) {
        self.water_env
    }

    /// The highest water surface over world `(x, z)`, or `None` where there is no
    /// water — the `water.surface_height` host seam.
    pub fn water_surface_height(&self, x: f64, z: f64) -> Option<f64> {
        self.water
            .highest_surface_at(DVec2::new(x, z), self.water_env.0)
            .map(|(_, h)| h)
    }

    /// Probe the water under a tracked entity: submerged fraction, deepest
    /// submersion, surface height and flow.
    ///
    /// A **live** query rather than a read of the force pass's cache, so it
    /// answers for any entity with a collider — a kinematic character has no
    /// `Buoyancy` and still needs to know how deep it is standing.
    ///
    /// **Instantaneous, and the events are latched.** `depth_m > 0` here is "the
    /// lowest point is under a surface *right now*", while
    /// [`WaterEventKind3D::Enter`]/[`Exit`](WaterEventKind3D::Exit) fire off the
    /// exit-hysteresis latch (5 % of the body's own height). The two disagree
    /// inside that band on purpose: a poll wants the truth now, an event wants a
    /// debounced edge. See the `water.*` node-kit docs for the same statement at
    /// the Blueprint boundary.
    pub fn water_probe(&self, guid: Uuid) -> Option<WaterProbe> {
        if self.water.is_empty() {
            return None;
        }
        let rec = self.entities.get(&guid)?;
        let col = rec.col.as_ref()?;
        let state = self.body_state(rec.body)?;
        let geo = water::sample_geometry(&col.shape, col.local_translation);
        Some(water::probe(&self.water, self.water_env.0, &state, &geo))
    }

    /// The sample layout the water pass uses for `guid`, if it is tracked and has
    /// a collider. Exposed so a test can assert against the same geometry the pass
    /// used rather than against a second copy of it.
    pub fn water_sample_geometry(&self, guid: Uuid) -> Option<SampleGeometry> {
        let rec = self.entities.get(&guid)?;
        let col = rec.col.as_ref()?;
        Some(water::sample_geometry(&col.shape, col.local_translation))
    }

    /// Whether `guid` is currently swimming (the latched state — call
    /// [`update_swim`](Self::update_swim) to advance it).
    pub fn is_swimming(&self, guid: Uuid) -> bool {
        self.swimming.get(&guid).copied().unwrap_or(false)
    }

    /// Advance `guid`'s swim latch from how submerged it is now, and return it.
    ///
    /// Both hosts call this — the editor's Simulate loop and the shipped runtime —
    /// so the threshold exists once. A per-host copy of "0.6 means swimming" is
    /// exactly the drift the projector MIRROR gate catches elsewhere, avoided here
    /// by there being nothing to mirror.
    pub fn update_swim(&mut self, guid: Uuid) -> bool {
        let fraction = self.water_probe(guid).map(|p| p.fraction).unwrap_or(0.0);
        let was = self.is_swimming(guid);
        let now = water::swim_latch(was, fraction);
        if now || was {
            self.swimming.insert(guid, now);
        }
        now
    }

    /// Transform a `move_and_slide` motion for `guid` if it is swimming, else
    /// return it untouched — the one place the mover's water behaviour lives, so
    /// the `physics3d.move_and_slide` host path in either host is a call rather
    /// than a policy.
    pub fn apply_swim_motion(&self, guid: Uuid, motion: DVec3, dt: f64) -> DVec3 {
        if !self.is_swimming(guid) {
            return motion;
        }
        let fraction = self.water_probe(guid).map(|p| p.fraction).unwrap_or(0.0);
        water::swim_motion(motion, fraction, dt)
    }

    /// A body's pose + velocities + mass, as the water solve wants them.
    fn body_state(&self, body: BodyId3D) -> Option<BodyState> {
        Some(BodyState {
            translation: self.world.body_translation(body)?,
            rotation: self.world.body_rotation(body)?,
            linvel: self.world.body_linvel(body).unwrap_or(DVec3::ZERO),
            angvel: self.world.body_angvel(body).unwrap_or(DVec3::ZERO),
            mass: self.world.body_mass(body).unwrap_or(0.0),
        })
    }

    /// The simulated **dynamic** poses, in `Guid` order, for the caller to copy
    /// back onto entity `Transform`s. Static/kinematic bodies are editor-driven
    /// and are not reported (they never moved on their own). The next-batch
    /// adapter preserves each entity's untouched components and only overwrites
    /// translation + rotation.
    pub fn write_back(&self) -> Vec<PoseWriteback3D> {
        let mut out = Vec::new();
        for (guid, rec) in &self.entities {
            if rec.kind != BodyKind3D::Dynamic {
                continue;
            }
            let (Some(translation), Some(rotation)) = (
                self.world.body_translation(rec.body),
                self.world.body_rotation(rec.body),
            ) else {
                continue;
            };
            out.push(PoseWriteback3D {
                guid: *guid,
                translation,
                rotation,
            });
        }
        out
    }

    /// Write simulated **dynamic** poses back onto the ECS `Transform`s: full
    /// world translation and orientation (extracted into the transform's euler
    /// degrees). Runs in `Guid` order and marks the world dirty so transform
    /// propagation reruns. Static/kinematic bodies are editor-driven and are not
    /// written back. The `d3` mirror of [`crate::d2::PhysicsBridge2D::write_back`]
    /// (at 3D the solver owns all three axes and the full rotation, so — unlike
    /// d2, which preserves Z translation — every component is overwritten).
    pub fn write_back_into(&mut self, world: &mut EcsWorld) {
        let mut changed = false;
        for (guid, rec) in &self.entities {
            if rec.kind != BodyKind3D::Dynamic {
                continue;
            }
            let (Some(translation), Some(rotation)) = (
                self.world.body_translation(rec.body),
                self.world.body_rotation(rec.body),
            ) else {
                continue;
            };
            let Some(entity) = world.entity_of(*guid) else {
                continue;
            };
            if let Some(mut t) = world.world_mut().get_mut::<Transform>(entity) {
                t.translation = Vec3d::from_dvec3(translation);
                t.set_quat(rotation);
                changed = true;
            }
        }
        if changed {
            world.mark_dirty();
        }
    }
}

fn apply_rb_props(world: &mut PhysicsWorld3D, body: BodyId3D, rb: &BodyDesc3D) {
    world.set_body_gravity_scale(body, rb.gravity_scale);
    world.set_body_damping(body, rb.linear_damping, rb.angular_damping);
    world.set_body_locked_rotations(body, rb.fixed_rotation);
    world.set_body_ccd(body, rb.ccd_enabled);
}

fn to_phys_kind(k: SceneBodyKind3D) -> BodyKind3D {
    match k {
        SceneBodyKind3D::Static => BodyKind3D::Static,
        SceneBodyKind3D::Kinematic => BodyKind3D::Kinematic,
        SceneBodyKind3D::Dynamic => BodyKind3D::Dynamic,
    }
}

/// Map a scene [`RigidBody3D`] onto the facade-local [`BodyDesc3D`].
fn body_desc(rb: RigidBody3D) -> BodyDesc3D {
    BodyDesc3D {
        kind: to_phys_kind(rb.kind),
        gravity_scale: rb.gravity_scale,
        fixed_rotation: rb.fixed_rotation,
        linear_damping: rb.linear_damping,
        angular_damping: rb.angular_damping,
        ccd_enabled: rb.ccd_enabled,
    }
}

fn to_phys_combine(r: SceneCombineRule) -> CombineRule {
    match r {
        SceneCombineRule::Average => CombineRule::Average,
        SceneCombineRule::Min => CombineRule::Min,
        SceneCombineRule::Multiply => CombineRule::Multiply,
        SceneCombineRule::Max => CombineRule::Max,
    }
}

/// The salt that carves the PCG structures' synthetic GUID space out of the
/// scene's own. Folded with the volume's GUID and the structure's index, it
/// makes an identity that (a) no authored entity can collide with and (b) is a
/// pure function of the content, which is what keeps the bridge's `Guid`-ordered
/// reconciliation deterministic.
const PCG_STRUCTURE_SALT: u128 = 0x7019_0500_5043_4753_b31f_60e8_9d4e_2c7a;

/// The synthetic identity of structure `index` inside volume `volume`.
///
/// Stated as one function so the derivation cannot drift: the bridge is the only
/// caller today, and a debug view or a save-game hook that ever needs to name
/// one of these must name it the same way.
pub fn pcg_structure_guid(volume: Uuid, index: usize) -> Uuid {
    // A 128-bit mix, not a XOR: XORing an index into a GUID makes two volumes
    // whose ids differ in the low bits alias each other's structures.
    let mut x = volume.as_u128() ^ PCG_STRUCTURE_SALT;
    x ^= (index as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5cec_c5c3);
    x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

/// Salt for [`voxel_chunk_guid`]. A different constant from
/// [`PCG_STRUCTURE_SALT`] so a scattered solid and a voxel chunk can never
/// collide in the bridge's one entity map.
const VOXEL_CHUNK_SALT: u128 = 0x2104_0400_564f_5845_4c43_484e_4b21_0021;

/// The synthetic identity of chunk `key` inside the volume on entity `volume`.
///
/// The [`pcg_structure_guid`] rule, with the chunk's three signed coordinates
/// folded in instead of an index — and for the same reason it is a 128-bit mix
/// rather than a XOR: two volumes whose ids differ in the low bits must not alias
/// each other's chunks. Stated as one function so a debug view or a save hook that
/// ever needs to name one of these names it the same way.
pub fn voxel_chunk_guid(volume: Uuid, key: inf_voxel::ChunkKey) -> Uuid {
    let mut x = volume.as_u128() ^ VOXEL_CHUNK_SALT;
    for (i, c) in [key.x, key.y, key.z].into_iter().enumerate() {
        // `as u32 as u128` keeps a negative coordinate's bits (two's complement)
        // rather than sign-extending them across the whole word, so −1 and a large
        // positive coordinate stay distinct inputs.
        let lane = (c as u32 as u128) | ((i as u128 + 1) << 96);
        x ^= lane.wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5cec_c5c3);
        x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    }
    Uuid::from_u128(x)
}

/// What identifies one terrain tile to the collider gather: the terrain entity's
/// `Guid` and the level-0 tile's grid coordinate.
type TerrainTileKey = (Uuid, (i32, i32));

/// What that tile's collider was last built from: the tile's own change stamp
/// plus its terrain's world origin **as raw bits**.
///
/// Bits rather than `f64` because this is a cache key, not a measurement: bit
/// equality makes `-0.0` and `NaN` behave (both compare unequal to a
/// re-derivation, so the conservative answer is "re-describe"), and it lets the
/// whole stamp derive `Eq` and `Ord` for the `BTreeMap`.
type TerrainTileStamp = (u64, [u64; 3]);

/// The collision layer bit fracture debris belongs to (P22.3).
///
/// **Layer 30**, near the top of the 32 available, so it does not collide with a
/// project's own low-numbered layers. Debris is a **member** of this layer and
/// interacts with **everything**, which is the default a designer expects (rubble
/// piles up on the floor and bounces off walls). What the layer buys is the
/// *other* direction: a trigger volume, a bullet trace or a doorway blocker can
/// clear bit 30 from its own filter and stop seeing rubble at all, without any
/// per-chunk authoring — which is impossible for geometry that has no entity to
/// author on.
///
/// See [`inf_physics::filtering::CollisionLayers`](crate::filtering::CollisionLayers)
/// for the membership/filter convention.
pub const DEBRIS_LAYER: u32 = 30;

/// The layers a debris chunk's collider is built with: a member of
/// [`DEBRIS_LAYER`] alone, interacting with everything.
fn debris_layers() -> CollisionLayers {
    CollisionLayers::new(1 << DEBRIS_LAYER, u32::MAX)
}

/// Salt for [`terrain_tile_guid`]. A third distinct constant, so a scattered
/// solid, a voxel chunk and a terrain tile can never collide in the bridge's one
/// entity map.
const TERRAIN_TILE_SALT: u128 = 0x2203_0100_5445_5252_4149_4e54_494c_4521;

/// The synthetic identity of level-0 tile `coord` inside the terrain on entity
/// `terrain`.
///
/// The [`voxel_chunk_guid`] rule with two coordinates instead of three, and a
/// 128-bit mix rather than a XOR for the same reason: two terrains whose ids
/// differ in the low bits must not alias each other's tiles. Stated as one
/// function so a debug view or a save hook that ever needs to name one of these
/// names it the same way.
pub fn terrain_tile_guid(terrain: Uuid, coord: (i32, i32)) -> Uuid {
    let mut x = terrain.as_u128() ^ TERRAIN_TILE_SALT;
    for (i, c) in [coord.0, coord.1].into_iter().enumerate() {
        // `as u32 as u128` keeps a negative coordinate's two's-complement bits
        // rather than sign-extending them across the word, so −1 and a large
        // positive coordinate stay distinct inputs.
        let lane = (c as u32 as u128) | ((i as u128 + 1) << 96);
        x ^= lane.wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5cec_c5c3);
        x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    }
    Uuid::from_u128(x)
}

/// What one `PhysicsBridge3D::gather_terrain` pass did (P22.3).
///
/// A plain counter struct on the `ScatterAudit` precedent rather than a new
/// budget ratchet: the question a gate asks here is "did the change stamp
/// actually retain, or is this rebuilding a quarter-million triangles a step",
/// and that is answered by a number, not by a threshold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainColliderAudit {
    /// Level-0 tiles resident in the sim's terrains this sync.
    pub resident_tiles: u32,
    /// Tiles whose collider was (re-)described — a new tile, a sculpted or
    /// carved one, or a terrain that moved.
    pub described: u32,
    /// Tiles kept without being re-described. `resident_tiles − described`.
    pub retained: u32,
}

/// One static [`ColliderShape3D::Heightfield`] for a level-0 terrain tile.
///
/// **Public so the consistency gate can drive it.** The first cut of
/// `the_removed_cell_rule_matches_height_at` re-derived the removal mask inside
/// the test — a THIRD hand-copy of a rule that already existed twice — and a
/// mutation flipping this function's `||` to `&&` sailed through it. A gate on a
/// copy is a gate on nothing.
///
/// `None` for a tile the shape cannot be built from (a resolution below 2 or a
/// short height buffer) — a refusal, never a panic, on the
/// `convex_hull_is_buildable` precedent.
///
/// # The hole mask, stated here and at the call site
///
/// A cell is removed when **any** of its four corner samples is holed. That is
/// [`TerrainData::height_at`](inf_terrain::TerrainData::height_at)'s poison rule
/// verbatim: one holed corner removes the surface from the whole bilinear cell,
/// so the query, the raster and the collider agree about where the ground stops.
/// A tile with no holes emits an **empty** removal buffer (the sparse default),
/// which is also what keeps an un-carved level's descriptors small.
pub fn terrain_tile_collider(
    tile: &inf_terrain::TerrainTile,
    resolution: u32,
    span: f64,
) -> Option<ColliderDesc3D> {
    if resolution < 2 {
        return None;
    }
    let n = resolution as usize;
    if tile.heights().len() != n * n {
        return None;
    }
    let removed_cells = if tile.has_holes() {
        let mut cells = vec![false; (n - 1) * (n - 1)];
        for jz in 0..(n - 1) {
            for ix in 0..(n - 1) {
                let (i0, j0) = (ix as u32, jz as u32);
                let (i1, j1) = (i0 + 1, j0 + 1);
                cells[jz * (n - 1) + ix] = tile.is_hole(resolution, i0, j0)
                    || tile.is_hole(resolution, i1, j0)
                    || tile.is_hole(resolution, i0, j1)
                    || tile.is_hole(resolution, i1, j1);
            }
        }
        cells
    } else {
        Vec::new()
    };
    Some(ColliderDesc3D::new(ColliderShape3D::Heightfield {
        samples_x: resolution,
        samples_z: resolution,
        heights: tile.heights().to_vec(),
        removed_cells,
        span: DVec2::splat(span),
    }))
}

/// One static box collider per [`PcgVolume::structures`] entry (P19.5).
///
/// **Why this exists at all.** Scattered content has always been render-only: a
/// `ScatteredInstance` is not an entity, has no `Guid`, and is invisible to the
/// bridge's world walk. That is right for a million blades of grass and wrong
/// for a building — "fully enterable" means the floor holds you up and the wall
/// stops you, which is a statement about colliders, not about geometry. So a
/// volume's *solid* half is walked here and given synthetic, content-derived
/// identities.
///
/// **Why not real entities.** Spawning one entity per wall panel would put
/// thousands of derived rows into `.inf_lvl`, need a despawn-before-re-evaluate
/// pass in two hosts, and make undo mean something new — all to express data
/// that is already a pure function of the graph and the terrain.
/// `PcgVolume::structures` is `#[serde(skip)]` derived state on the
/// `PcgVolume::evaluated` precedent, so **no schema moves in either codec
/// mirror**.
///
/// Every box is **static**: a building does not fall over, and a dynamic body
/// per wall panel would be a physics bill nobody asked for. Destruction is
/// Phase 22's, and it will want fracture chunks rather than these.
fn structure_snaps_of(guid: Uuid, vol: &PcgVolume) -> Vec<EntitySync3D> {
    vol.structures
        .iter()
        .enumerate()
        .map(|(i, solid)| EntitySync3D {
            guid: pcg_structure_guid(guid, i),
            body: None,
            collider: Some(ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: solid.half_extents,
            })),
            translation: solid.center,
            rotation: solid.rotation,
            joint: None,
        })
        .collect()
}

/// Map a scene [`Collider3D`] onto the facade-local [`ColliderDesc3D`].
fn collider_desc(col: &Collider3D) -> ColliderDesc3D {
    let shape = match col.shape_kind {
        ColliderShape3DKind::Box => ColliderShape3D::Box {
            half_extents: col.half_extents.to_dvec3(),
        },
        ColliderShape3DKind::Sphere => ColliderShape3D::Sphere { radius: col.radius },
        ColliderShape3DKind::Capsule => ColliderShape3D::Capsule {
            half_height: col.half_extents.y,
            radius: col.radius,
        },
    };
    ColliderDesc3D::new(shape)
        .friction(col.friction)
        .restitution(col.restitution)
        .density(col.density)
        .sensor(col.sensor)
        .local_translation(col.offset.to_dvec3())
        .layers(CollisionLayers::new(
            col.collision_memberships,
            col.collision_filter,
        ))
        .friction_combine(to_phys_combine(col.friction_combine))
        .restitution_combine(to_phys_combine(col.restitution_combine))
}

/// Map a scene [`Joint3D`] onto a facade [`JointSync3D`]; `None` if it is unbound
/// (no `other` entity set).
fn joint_sync(j: Joint3D) -> Option<JointSync3D> {
    let other = j.other.get()?;
    let motor = j.motor_enabled.then_some(JointMotor3D {
        target_pos: j.motor_target_pos,
        target_vel: j.motor_target_vel,
        stiffness: j.motor_stiffness,
        damping: j.motor_damping,
        max_force: j.motor_max_force,
    });
    let limits = j.limits_enabled.then_some([j.limit_min, j.limit_max]);
    let kind = match j.kind {
        SceneJointKind3D::Fixed => JointKind3D::Fixed,
        SceneJointKind3D::Revolute => JointKind3D::Revolute {
            axis: j.axis.to_dvec3(),
            limits,
            motor,
        },
        SceneJointKind3D::Prismatic => JointKind3D::Prismatic {
            axis: j.axis.to_dvec3(),
            limits,
            motor,
        },
        SceneJointKind3D::Spherical => JointKind3D::Spherical,
        SceneJointKind3D::Distance => JointKind3D::Distance {
            max_distance: j.max_distance,
        },
    };
    let desc = JointDesc3D::new(kind)
        .local_anchor1(j.local_anchor.to_dvec3())
        .local_anchor2(j.other_anchor.to_dvec3());
    Some(JointSync3D { other, desc })
}
