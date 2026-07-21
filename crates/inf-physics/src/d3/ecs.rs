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
    BodyKind3D as SceneBodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Transform,
};
use inf_ecs::{EcsWorld, Vec3d};
use uuid::Uuid;

use super::world::{BodyKind3D, ColliderDesc3D, ColliderShape3D, PhysicsWorld3D};
use super::{BodyId3D, ColliderId3D};

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
}

impl Default for BodyDesc3D {
    fn default() -> Self {
        Self {
            kind: BodyKind3D::Static,
            gravity_scale: 1.0,
            fixed_rotation: false,
            linear_damping: 0.0,
            angular_damping: 0.0,
        }
    }
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
}

/// Owns a [`PhysicsWorld3D`] and keeps it in sync with a scene snapshot.
pub struct PhysicsBridge3D {
    world: PhysicsWorld3D,
    /// `Guid` → its rapier handles. `BTreeMap` gives sorted (deterministic)
    /// iteration for the despawn + write-back passes.
    entities: BTreeMap<Uuid, BodyRecord>,
}

impl PhysicsBridge3D {
    /// A new bridge over an empty world with the given gravity (world units/s²).
    /// A typical 3D world uses `DVec3::new(0.0, -9.81, 0.0)`.
    pub fn new(gravity: DVec3) -> Self {
        Self {
            world: PhysicsWorld3D::new(gravity),
            entities: BTreeMap::new(),
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

    /// The collider handle mirroring `guid`, if it has one.
    pub fn collider_of(&self, guid: Uuid) -> Option<ColliderId3D> {
        self.entities.get(&guid).and_then(|r| r.collider)
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
            });
        }
        // `sync` sorts by Guid internally, so the gather order here is irrelevant.
        self.sync(&snaps);
    }

    /// Reconcile the physics world with a scene snapshot: spawn new
    /// bodies/colliders, update changed ones, and despawn bodies whose entity
    /// disappeared. Runs in `Guid` order regardless of the input order, so the
    /// result is independent of how the caller gathered the snapshot.
    pub fn sync(&mut self, entities: &[EntitySync3D]) {
        // 1. Sort the snapshot into deterministic Guid order (and drop entities
        //    with neither a body nor a collider — nothing to simulate).
        let mut live: Vec<&EntitySync3D> = entities
            .iter()
            .filter(|e| e.body.is_some() || e.collider.is_some())
            .collect();
        live.sort_by_key(|e| e.guid);

        // 2. Spawn / update.
        let mut seen: BTreeSet<Uuid> = BTreeSet::new();
        for snap in live {
            seen.insert(snap.guid);
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
                    },
                );
            }
        }

        // 3. Despawn: any tracked guid not seen this sync is gone.
        let gone: Vec<Uuid> = self
            .entities
            .keys()
            .filter(|g| !seen.contains(g))
            .copied()
            .collect();
        for guid in gone {
            if let Some(rec) = self.entities.remove(&guid) {
                // Removing the body drops its colliders too.
                self.world.remove_body(rec.body);
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
    }
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
}
