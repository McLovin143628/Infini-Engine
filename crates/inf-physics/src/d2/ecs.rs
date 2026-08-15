//! ECS ↔ physics bridge (ROADMAP P8.3.2): mirror scene components into a
//! [`PhysicsWorld2D`] and write the simulated poses back.
//!
//! [`PhysicsBridge2D`] owns a [`PhysicsWorld2D`] plus deterministic
//! entity↔handle maps keyed by the entity's stable [`Guid`]. Determinism (§2.5)
//! is preserved by construction: **every** entity iteration — spawn, update,
//! despawn, write-back — runs in sorted `Guid` order, never in bevy `Entity`-id
//! order (which churns as entities are recycled). Two worlds built from the same
//! scene therefore allocate rapier handles in the same sequence and step to
//! byte-identical poses.
//!
//! Dependency direction is **inf-physics → inf-ecs** (both Ring 0): physics
//! reads scene components; inf-ecs never names physics, so there is no cycle.
//!
//! Mapping rules:
//! * The XY plane is the simulation plane; an entity's `Transform.translation.z`
//!   is ignored by the sim and preserved on write-back. Rotation is the
//!   `Transform.rotation.z` euler angle (degrees ⇄ radians).
//! * An entity with a [`RigidBody2D`] becomes a body of that kind; an entity
//!   with only a [`Collider2D`] gets an implicit **static** body so the collider
//!   has a parent.
//! * `Transform` is the authoritative source for **static/kinematic** bodies
//!   (pushed to the sim every sync — a kinematic body follows its externally
//!   set transform). **Dynamic** bodies are owned by the solver: their transform
//!   is pushed only once, at spawn, and flows back out through [`write_back`].
//!
//! [`write_back`]: PhysicsBridge2D::write_back

use std::collections::BTreeMap;

use glam::DVec2;
use inf_ecs::components::{
    BodyKind2D, Collider2D, ColliderShape2DKind, CombineRule as SceneCombineRule, Joint2D,
    JointKind2D as SceneJointKind2D, RigidBody2D, Transform,
};
use inf_ecs::EcsWorld;
use uuid::Uuid;

use super::joint::{JointDesc2D, JointId2D, JointKind2D, JointMotor2D};
use super::world::{BodyKind, ColliderDesc2D, ColliderShape2D};
use super::{BodyId, ColliderId, PhysicsWorld2D};
use crate::filtering::{CollisionLayers, CombineRule};

/// A joint another entity's body is linked to (P12.1): the other body's `Guid`
/// plus the facade descriptor. Reconciled after all bodies exist.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointSync2D {
    /// The other body's entity `Guid`.
    pub other: Uuid,
    /// The joint family + anchors + params.
    pub desc: JointDesc2D,
}

/// One tracked entity: its rapier body/collider handles plus the last-synced
/// component snapshots (for cheap change detection — the component structs are
/// `Copy` + `PartialEq`).
struct BodyRecord {
    body: BodyId,
    collider: Option<ColliderId>,
    kind: BodyKind,
    rb: Option<RigidBody2D>,
    /// The collider descriptor **actually attached**, never the one asked for
    /// (C4-30). See the 3D mirror's field for the argument: this is the
    /// reconcile's change detector, so recording a desired-but-unbuilt shape
    /// here tells the next sync there is nothing to do.
    ///
    /// 2D shapes are all analytic (ball / cuboid / capsule / polygon), so a
    /// refusal here is far rarer than in 3D — but "rarer" is not "impossible",
    /// and the two bridges are read side by side.
    col: Option<Collider2D>,
    /// The joint this entity owns (P12.1): handle + other body + last snapshot.
    joint: Option<JointBinding>,
}

/// A live joint binding tracked on the owning entity's record.
#[derive(Clone, Copy)]
struct JointBinding {
    id: JointId2D,
    other_body: BodyId,
    sync: JointSync2D,
}

/// Owns a [`PhysicsWorld2D`] and keeps it in sync with an [`EcsWorld`].
pub struct PhysicsBridge2D {
    world: PhysicsWorld2D,
    /// `Guid` → its rapier handles. `BTreeMap` gives sorted (deterministic)
    /// iteration for the despawn + write-back passes.
    entities: BTreeMap<Uuid, BodyRecord>,
    /// Reverse map `collider handle → owning entity Guid`, rebuilt at the end of
    /// every [`sync_from_world`](Self::sync_from_world) (Wave 3). The
    /// collision-event drain resolves rapier's `ContactEvent2D` collider handles
    /// back to entity `Guid`s through it — the inverse of
    /// [`collider_of`](Self::collider_of). Deterministic (`BTreeMap`).
    collider_to_guid: BTreeMap<ColliderId, Uuid>,
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
}

impl PhysicsBridge2D {
    /// A new bridge over an empty world with the given gravity (world units/s²).
    /// A side-scroller uses `DVec2::new(0.0, -9.81)`; a top-down world `ZERO`.
    pub fn new(gravity: DVec2) -> Self {
        Self {
            world: PhysicsWorld2D::new(gravity),
            entities: BTreeMap::new(),
            collider_to_guid: BTreeMap::new(),
            collider_map_dirty: true,
        }
    }

    /// The wrapped physics world (for scene queries, contact-event drain, etc.).
    pub fn world(&self) -> &PhysicsWorld2D {
        &self.world
    }

    /// Mutable access to the wrapped world (queries mutate the lazy query BVH).
    pub fn world_mut(&mut self) -> &mut PhysicsWorld2D {
        &mut self.world
    }

    /// The body handle mirroring `guid`, if it is tracked.
    pub fn body_of(&self, guid: Uuid) -> Option<BodyId> {
        self.entities.get(&guid).map(|r| r.body)
    }

    /// The collider handle mirroring `guid`, if it has one.
    pub fn collider_of(&self, guid: Uuid) -> Option<ColliderId> {
        self.entities.get(&guid).and_then(|r| r.collider)
    }

    /// The entity `Guid` owning `collider`, if tracked — the inverse of
    /// [`collider_of`](Self::collider_of), maintained each
    /// [`sync_from_world`](Self::sync_from_world) (Wave 3). The seam the
    /// collision-event drain uses to map a rapier
    /// [`ContactEvent2D`](crate::d2::ContactEvent2D) collider handle back to the
    /// entity it belongs to.
    pub fn guid_of_collider(&self, collider: ColliderId) -> Option<Uuid> {
        self.collider_to_guid.get(&collider).copied()
    }

    /// The joint handle owned by `guid` (P12.1), if it has one.
    pub fn joint_of(&self, guid: Uuid) -> Option<JointId2D> {
        self.entities.get(&guid).and_then(|r| r.joint).map(|b| b.id)
    }

    /// Advance the simulation by `dt` seconds (the caller's fixed step).
    pub fn step(&mut self, dt: f64) {
        self.world.step(dt);
    }

    /// Reconcile the physics world with the current ECS components: spawn new
    /// bodies/colliders, update changed ones, and despawn bodies whose entity
    /// (or its physics components) disappeared. Runs in `Guid` order.
    pub fn sync_from_world(&mut self, world: &EcsWorld) {
        // 0. **The pure-3D level's whole cost** (lens 3 P12, Hardening Wave G).
        //    Both hosts call this unconditionally from the fixed step, so a
        //    level that has never had a 2D component walked every one of its
        //    entities — three component lookups apiece, thirteen thousand of
        //    them on a furnished town — sixty times a second, to build an empty
        //    `live` and reconcile nothing against nothing.
        //
        //    The tracked-set half of the guard is not decoration: a level whose
        //    last 2D component was just deleted must still reach the despawn
        //    sweep below, or its bodies would stay in the solver for ever.
        //    `gather_voxels`' guard is the same two-clause shape for the same
        //    reason.
        if self.entities.is_empty()
            && !world.has_component::<RigidBody2D>()
            && !world.has_component::<Collider2D>()
        {
            return;
        }

        // 1. Gather participating entities (RigidBody2D and/or Collider2D) in
        //    deterministic Guid order.
        let mut live: Vec<(Uuid, EntitySnapshot)> = Vec::new();
        for entity in world.world().iter_entities() {
            let Some(guid) = entity.get::<inf_ecs::Guid>().map(|g| g.0) else {
                continue;
            };
            let rb = entity.get::<RigidBody2D>().copied();
            let col = entity.get::<Collider2D>().copied();
            let joint = entity.get::<Joint2D>().copied();
            if rb.is_none() && col.is_none() {
                continue;
            }
            let transform = entity
                .get::<Transform>()
                .copied()
                .unwrap_or(Transform::IDENTITY);
            live.push((
                guid,
                EntitySnapshot {
                    rb,
                    col,
                    transform,
                    joint: joint.and_then(joint_sync),
                },
            ));
        }
        live.sort_by_key(|(g, _)| *g);

        // 2. Spawn / update. (Joints reconciled in a second pass, once every body
        //    exists, so a joint can always resolve its other body.)
        let mut seen: Vec<Uuid> = Vec::with_capacity(live.len());
        // The d3 mirror's P30 filter: only an entity that has a joint or had one
        // needs pass 4.
        let mut joint_desires: Vec<(Uuid, Option<JointSync2D>)> = Vec::new();
        for (guid, snap) in live {
            seen.push(guid);
            let kind = to_phys_kind(snap.rb.map(|r| r.kind).unwrap_or(BodyKind2D::Static));
            let (pos, rot) = transform_pose(&snap.transform);

            if let Some(rec) = self.entities.get(&guid) {
                let rec_kind = rec.kind;
                let rec_rb = rec.rb;
                let rec_col = rec.col;
                let old_collider = rec.collider;
                let body = rec.body;
                if snap.joint.is_some() || rec.joint.is_some() {
                    joint_desires.push((guid, snap.joint));
                }

                if rec_kind != kind {
                    self.world.set_body_kind(body, kind);
                    if let Some(r) = self.entities.get_mut(&guid) {
                        r.kind = kind;
                    }
                }
                // Static/kinematic follow their Transform; dynamic is
                // solver-owned — and the write is skipped when the body is
                // already there, against rapier's own state. See the d3
                // `set_body_pose_if_moved` for the argument and the numbers.
                if kind != BodyKind::Dynamic {
                    self.world.set_body_pose_if_moved(body, pos, rot);
                }
                if rec_rb != snap.rb {
                    if let Some(rb) = snap.rb.as_ref() {
                        apply_rb_props(&mut self.world, body, rb);
                    }
                    if let Some(r) = self.entities.get_mut(&guid) {
                        r.rb = snap.rb;
                    }
                }
                if rec_col != snap.col {
                    // Rebuild the collider so shape/material edits take effect.
                    if let Some(old) = old_collider {
                        self.world.remove_collider(old);
                    }
                    let new_col = snap
                        .col
                        .as_ref()
                        .and_then(|c| self.world.add_collider(body, collider_desc(c)));
                    if let Some(r) = self.entities.get_mut(&guid) {
                        r.collider = new_col;
                        self.collider_map_dirty = true;
                        // Only what was achieved (C4-30). A refused build leaves
                        // `col` at `None`, so the next sync sees the difference
                        // and retries — which in 2D is cheap, every shape being
                        // analytic, so this mirror needs no `col_refused` brake.
                        r.col = new_col.and(snap.col);
                    }
                }
            } else {
                if snap.joint.is_some() {
                    joint_desires.push((guid, snap.joint));
                }
                // New entity → create its body, apply props, attach a collider.
                let body = self.world.add_body(kind, pos, rot);
                if let Some(rb) = snap.rb.as_ref() {
                    apply_rb_props(&mut self.world, body, rb);
                }
                let collider = snap
                    .col
                    .as_ref()
                    .and_then(|c| self.world.add_collider(body, collider_desc(c)));
                self.collider_map_dirty = true;
                self.entities.insert(
                    guid,
                    BodyRecord {
                        body,
                        collider,
                        kind,
                        rb: snap.rb,
                        col: collider.and(snap.col),
                        joint: None,
                    },
                );
            }
        }

        // 3. Despawn: any tracked guid not seen this sync is gone. Removing the
        //    body drops its colliders AND any joints attached to it (rapier).
        //    A merge over two guid-ordered sequences, not a set probe per
        //    tracked entity — the d3 mirror's reasoning.
        let mut gone: Vec<Uuid> = Vec::new();
        let mut cursor = 0usize;
        for guid in self.entities.keys() {
            while cursor < seen.len() && seen[cursor] < *guid {
                cursor += 1;
            }
            if seen.get(cursor) == Some(guid) {
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

        // 4. Reconcile joints (P12.1), now that every body exists. In Guid order.
        for (guid, desire) in joint_desires {
            self.reconcile_joint(guid, desire);
        }

        // 5. Rebuild the reverse collider→Guid map for this step's event drain
        //    (Wave 3), consistent with the handles just reconciled above.
        if self.collider_map_dirty {
            self.collider_to_guid = self
                .entities
                .iter()
                .filter_map(|(g, r)| r.collider.map(|c| (c, *g)))
                .collect();
            self.collider_map_dirty = false;
        }
    }

    /// Bring one entity's joint in line with its desired snapshot (the `d2`
    /// mirror of [`crate::d3::PhysicsBridge3D`]'s reconcile). Resolves the other
    /// body from the tracked map; rebuilds on any change; removes when the desire
    /// is `None` or the other body is missing.
    fn reconcile_joint(&mut self, guid: Uuid, desire: Option<JointSync2D>) {
        let Some(self_body) = self.entities.get(&guid).map(|r| r.body) else {
            return;
        };
        let existing = self.entities.get(&guid).and_then(|r| r.joint);
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

    /// Write simulated **dynamic** poses back onto the ECS `Transform`s: XY
    /// translation and the Z euler rotation, **preserving** each entity's Z
    /// translation (the sim never touched it). Runs in `Guid` order and marks
    /// the world dirty so transform propagation reruns. Static/kinematic bodies
    /// are editor-driven and are not written back.
    pub fn write_back(&self, world: &mut EcsWorld) {
        let mut changed = false;
        for (guid, rec) in &self.entities {
            if rec.kind != BodyKind::Dynamic {
                continue;
            }
            let (Some(pos), Some(rot)) = (
                self.world.body_translation(rec.body),
                self.world.body_rotation(rec.body),
            ) else {
                continue;
            };
            let Some(entity) = world.entity_of(*guid) else {
                continue;
            };
            if let Some(mut t) = world.world_mut().get_mut::<Transform>(entity) {
                t.translation.x = pos.x;
                t.translation.y = pos.y;
                // Z translation preserved; Z rotation from the sim (rad → deg).
                t.rotation.z = rot.to_degrees();
                changed = true;
            }
        }
        if changed {
            world.mark_dirty();
        }
    }
}

/// A per-entity snapshot captured during a sync pass.
struct EntitySnapshot {
    rb: Option<RigidBody2D>,
    col: Option<Collider2D>,
    transform: Transform,
    joint: Option<JointSync2D>,
}

fn to_phys_kind(k: BodyKind2D) -> BodyKind {
    match k {
        BodyKind2D::Static => BodyKind::Static,
        BodyKind2D::Kinematic => BodyKind::Kinematic,
        BodyKind2D::Dynamic => BodyKind::Dynamic,
    }
}

/// XY translation + Z-euler rotation (radians) of a `Transform`.
fn transform_pose(t: &Transform) -> (DVec2, f64) {
    (
        DVec2::new(t.translation.x, t.translation.y),
        t.rotation.z.to_radians(),
    )
}

fn apply_rb_props(world: &mut PhysicsWorld2D, body: BodyId, rb: &RigidBody2D) {
    world.set_body_gravity_scale(body, rb.gravity_scale);
    world.set_body_damping(body, rb.linear_damping, rb.angular_damping);
    world.set_body_locked_rotations(body, rb.fixed_rotation);
    world.set_body_ccd(body, rb.ccd_enabled);
}

fn to_phys_combine(r: SceneCombineRule) -> CombineRule {
    match r {
        SceneCombineRule::Average => CombineRule::Average,
        SceneCombineRule::Min => CombineRule::Min,
        SceneCombineRule::Multiply => CombineRule::Multiply,
        SceneCombineRule::Max => CombineRule::Max,
    }
}

/// Map a scene [`Joint2D`] onto a facade [`JointSync2D`]; `None` if unbound.
fn joint_sync(j: Joint2D) -> Option<JointSync2D> {
    let other = j.other.get()?;
    let motor = j.motor_enabled.then_some(JointMotor2D {
        target_pos: j.motor_target_pos,
        target_vel: j.motor_target_vel,
        stiffness: j.motor_stiffness,
        damping: j.motor_damping,
        max_force: j.motor_max_force,
    });
    let limits = j.limits_enabled.then_some([j.limit_min, j.limit_max]);
    let kind = match j.kind {
        SceneJointKind2D::Fixed => JointKind2D::Fixed,
        SceneJointKind2D::Revolute => JointKind2D::Revolute { limits, motor },
        SceneJointKind2D::Prismatic => JointKind2D::Prismatic {
            axis: DVec2::new(j.axis.x, j.axis.y),
            limits,
            motor,
        },
        SceneJointKind2D::Distance => JointKind2D::Distance {
            max_distance: j.max_distance,
        },
    };
    let desc = JointDesc2D::new(kind)
        .local_anchor1(DVec2::new(j.local_anchor.x, j.local_anchor.y))
        .local_anchor2(DVec2::new(j.other_anchor.x, j.other_anchor.y));
    Some(JointSync2D { other, desc })
}

fn collider_desc(col: &Collider2D) -> ColliderDesc2D {
    let shape = match col.shape_kind {
        ColliderShape2DKind::Box => ColliderShape2D::Box {
            half_width: col.half_extents.x,
            half_height: col.half_extents.y,
        },
        ColliderShape2DKind::Circle => ColliderShape2D::Circle { radius: col.radius },
        ColliderShape2DKind::Capsule => ColliderShape2D::Capsule {
            half_height: col.half_extents.y,
            radius: col.radius,
        },
    };
    ColliderDesc2D::new(shape)
        .friction(col.friction)
        .restitution(col.restitution)
        .density(col.density)
        .sensor(col.sensor)
        .local_translation(DVec2::new(col.offset.x, col.offset.y))
        .layers(CollisionLayers::new(
            col.collision_memberships,
            col.collision_filter,
        ))
        .friction_combine(to_phys_combine(col.friction_combine))
        .restitution_combine(to_phys_combine(col.restitution_combine))
}
