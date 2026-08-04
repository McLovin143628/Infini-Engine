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
    PcgVolume, RigidBody3D, Spline, Transform, WaterBody,
};
use inf_ecs::{EcsWorld, Vec3d};
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
    col: Option<ColliderDesc3D>,
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
    /// VoxelData::chunk_version`. The `structure_stamps` twin, keyed one level
    /// finer because a runtime carve moves *one chunk of one volume* and
    /// re-meshing a whole cave system for it inside a fixed step is exactly the
    /// bill the pattern exists to refuse. While a stamp matches, that chunk's
    /// collider is **retained without being re-described** — see
    /// [`gather_voxels`](Self::gather_voxels).
    voxel_stamps: BTreeMap<(Uuid, inf_voxel::ChunkKey), u64>,
    /// Reverse map `collider handle → owning entity Guid`, rebuilt at the end of
    /// every [`sync`](Self::sync) (Wave 3). The collision-event drain resolves
    /// rapier's `ContactEvent3D` collider handles back to entity `Guid`s through
    /// it — the inverse of [`collider_of`](Self::collider_of). Deterministic
    /// (`BTreeMap`, sorted keys).
    collider_to_guid: BTreeMap<ColliderId3D, Uuid>,

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
            collider_to_guid: BTreeMap::new(),
            water: WaterIndex::default(),
            water_stamps: Vec::new(),
            water_scratch: Vec::new(),
            water_sources: Vec::new(),
            water_env: (0.0, (0.0, 0.0)),
            buoyant: BuoyantMap::new(),
            swimming: BTreeMap::new(),
            water_events: Vec::new(),
        }
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
        let mut snaps: Vec<EntitySync3D> = Vec::new();
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
            let rb = entity.get::<RigidBody3D>().copied();
            let col = entity.get::<Collider3D>().copied();
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
        // `sync` sorts by Guid internally, so the gather order here is irrelevant.
        self.sync_retaining(&snaps, &retained);
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
            for key in data.resident_keys() {
                live.insert((entity, key));
                let version = data.chunk_version(key);
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
        let mut seen: BTreeSet<Uuid> = BTreeSet::new();
        let mut joint_desires: Vec<(Uuid, Option<JointSync3D>)> = Vec::new();
        for snap in live {
            seen.insert(snap.guid);
            joint_desires.push((snap.guid, snap.joint));
            let kind = snap.body.map(|b| b.kind).unwrap_or(BodyKind3D::Static);
            let pos = snap.translation;
            let rot = snap.rotation;

            if let Some(rec) = self.entities.get(&snap.guid) {
                let rec_kind = rec.kind;
                let rec_rb = rec.rb;
                let rec_col = rec.col.clone();
                let old_collider = rec.collider;
                let body = rec.body;

                if rec_kind != kind {
                    self.world.set_body_kind(body, kind);
                    if let Some(r) = self.entities.get_mut(&snap.guid) {
                        r.kind = kind;
                    }
                }
                // Static/kinematic follow their Transform; dynamic is solver-owned.
                if kind != BodyKind3D::Dynamic {
                    self.world.set_body_translation(body, pos);
                    self.world.set_body_rotation(body, rot);
                }
                if rec_rb != snap.body {
                    if let Some(rb) = snap.body.as_ref() {
                        apply_rb_props(&mut self.world, body, rb);
                    }
                    if let Some(r) = self.entities.get_mut(&snap.guid) {
                        r.rb = snap.body;
                    }
                }
                if rec_col != snap.collider {
                    // Rebuild the collider so shape/material edits take effect.
                    if let Some(old) = old_collider {
                        self.world.remove_collider(old);
                    }
                    let new_col = snap
                        .collider
                        .as_ref()
                        .and_then(|c| self.world.add_collider(body, c.clone()));
                    if let Some(r) = self.entities.get_mut(&snap.guid) {
                        r.collider = new_col;
                        r.col = snap.collider.clone();
                    }
                }
            } else {
                // New entity → create its body, apply props, attach a collider.
                let body = self.world.add_body(kind, pos, rot);
                if let Some(rb) = snap.body.as_ref() {
                    apply_rb_props(&mut self.world, body, rb);
                }
                let collider = snap
                    .collider
                    .as_ref()
                    .and_then(|c| self.world.add_collider(body, c.clone()));
                self.entities.insert(
                    snap.guid,
                    BodyRecord {
                        body,
                        collider,
                        kind,
                        rb: snap.body,
                        col: snap.collider.clone(),
                        joint: None,
                    },
                );
            }
        }

        // 3. Despawn: any tracked guid not seen this sync is gone. Removing the
        //    body drops its colliders AND any joints attached to it (rapier), so a
        //    joint whose endpoint despawns is cleaned up here; the owning record's
        //    stale handle is reconciled to `None` in pass 4.
        let gone: Vec<Uuid> = self
            .entities
            .keys()
            .filter(|g| !seen.contains(g) && !retained.contains(g))
            .copied()
            .collect();
        for guid in gone {
            if let Some(rec) = self.entities.remove(&guid) {
                self.world.remove_body(rec.body);
            }
        }

        // 4. Reconcile joints (P12.1), now that every body exists. In Guid order.
        for (guid, desire) in joint_desires {
            self.reconcile_joint(guid, desire);
        }

        // 5. Rebuild the reverse collider→Guid map for this step's event drain
        //    (Wave 3). Cheap (one pass over the tracked entities) and always
        //    consistent with the handles just reconciled above.
        self.collider_to_guid = self
            .entities
            .iter()
            .filter_map(|(g, r)| r.collider.map(|c| (c, *g)))
            .collect();

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
