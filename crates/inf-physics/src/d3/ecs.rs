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

use glam::{DQuat, DVec3};
use inf_ecs::components::{
    BodyKind3D as SceneBodyKind3D, Collider3D, ColliderShape3DKind,
    CombineRule as SceneCombineRule, Joint3D, JointKind3D as SceneJointKind3D, PcgVolume,
    RigidBody3D, Transform,
};
use inf_ecs::{EcsWorld, Vec3d};
use uuid::Uuid;

use super::joint::{JointDesc3D, JointId3D, JointKind3D, JointMotor3D};
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
    /// Reverse map `collider handle → owning entity Guid`, rebuilt at the end of
    /// every [`sync`](Self::sync) (Wave 3). The collision-event drain resolves
    /// rapier's `ContactEvent3D` collider handles back to entity `Guid`s through
    /// it — the inverse of [`collider_of`](Self::collider_of). Deterministic
    /// (`BTreeMap`, sorted keys).
    collider_to_guid: BTreeMap<ColliderId3D, Uuid>,
}

impl PhysicsBridge3D {
    /// A new bridge over an empty world with the given gravity (world units/s²).
    /// A typical 3D world uses `DVec3::new(0.0, -9.81, 0.0)`.
    pub fn new(gravity: DVec3) -> Self {
        Self {
            world: PhysicsWorld3D::new(gravity),
            entities: BTreeMap::new(),
            structure_stamps: BTreeMap::new(),
            collider_to_guid: BTreeMap::new(),
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
        let mut snaps: Vec<EntitySync3D> = Vec::new();
        for entity in world.world().iter_entities() {
            let Some(guid) = entity.get::<inf_ecs::Guid>().map(|g| g.0) else {
                continue;
            };
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
        // P19.5: a volume's derived solids. Unchanged volumes are **retained**
        // rather than re-described — see the doc on `structure_stamps`.
        let retained = self.gather_structures(world, &mut snaps);
        // `sync` sorts by Guid internally, so the gather order here is irrelevant.
        self.sync_retaining(&snaps, &retained);
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
