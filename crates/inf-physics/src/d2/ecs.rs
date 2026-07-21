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
use inf_ecs::components::{BodyKind2D, Collider2D, ColliderShape2DKind, RigidBody2D, Transform};
use inf_ecs::EcsWorld;
use uuid::Uuid;

use super::world::{BodyKind, ColliderDesc2D, ColliderShape2D};
use super::{BodyId, ColliderId, PhysicsWorld2D};

/// One tracked entity: its rapier body/collider handles plus the last-synced
/// component snapshots (for cheap change detection — the component structs are
/// `Copy` + `PartialEq`).
struct BodyRecord {
    body: BodyId,
    collider: Option<ColliderId>,
    kind: BodyKind,
    rb: Option<RigidBody2D>,
    col: Option<Collider2D>,
}

/// Owns a [`PhysicsWorld2D`] and keeps it in sync with an [`EcsWorld`].
pub struct PhysicsBridge2D {
    world: PhysicsWorld2D,
    /// `Guid` → its rapier handles. `BTreeMap` gives sorted (deterministic)
    /// iteration for the despawn + write-back passes.
    entities: BTreeMap<Uuid, BodyRecord>,
}

impl PhysicsBridge2D {
    /// A new bridge over an empty world with the given gravity (world units/s²).
    /// A side-scroller uses `DVec2::new(0.0, -9.81)`; a top-down world `ZERO`.
    pub fn new(gravity: DVec2) -> Self {
        Self {
            world: PhysicsWorld2D::new(gravity),
            entities: BTreeMap::new(),
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

    /// Advance the simulation by `dt` seconds (the caller's fixed step).
    pub fn step(&mut self, dt: f64) {
        self.world.step(dt);
    }

    /// Reconcile the physics world with the current ECS components: spawn new
    /// bodies/colliders, update changed ones, and despawn bodies whose entity
    /// (or its physics components) disappeared. Runs in `Guid` order.
    pub fn sync_from_world(&mut self, world: &EcsWorld) {
        // 1. Gather participating entities (RigidBody2D and/or Collider2D) in
        //    deterministic Guid order.
        let mut live: Vec<(Uuid, EntitySnapshot)> = Vec::new();
        for entity in world.world().iter_entities() {
            let Some(guid) = entity.get::<inf_ecs::Guid>().map(|g| g.0) else {
                continue;
            };
            let rb = entity.get::<RigidBody2D>().copied();
            let col = entity.get::<Collider2D>().copied();
            if rb.is_none() && col.is_none() {
                continue;
            }
            let transform = entity
                .get::<Transform>()
                .copied()
                .unwrap_or(Transform::IDENTITY);
            live.push((guid, EntitySnapshot { rb, col, transform }));
        }
        live.sort_by_key(|(g, _)| *g);

        // 2. Spawn / update.
        let mut seen: Vec<Uuid> = Vec::with_capacity(live.len());
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

                if rec_kind != kind {
                    self.world.set_body_kind(body, kind);
                    if let Some(r) = self.entities.get_mut(&guid) {
                        r.kind = kind;
                    }
                }
                // Static/kinematic follow their Transform; dynamic is solver-owned.
                if kind != BodyKind::Dynamic {
                    self.world.set_body_translation(body, pos);
                    self.world.set_body_rotation(body, rot);
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
                        r.col = snap.col;
                    }
                }
            } else {
                // New entity → create its body, apply props, attach a collider.
                let body = self.world.add_body(kind, pos, rot);
                if let Some(rb) = snap.rb.as_ref() {
                    apply_rb_props(&mut self.world, body, rb);
                }
                let collider = snap
                    .col
                    .as_ref()
                    .and_then(|c| self.world.add_collider(body, collider_desc(c)));
                self.entities.insert(
                    guid,
                    BodyRecord {
                        body,
                        collider,
                        kind,
                        rb: snap.rb,
                        col: snap.col,
                    },
                );
            }
        }

        // 3. Despawn: any tracked guid not seen this sync is gone.
        let seen: std::collections::BTreeSet<Uuid> = seen.into_iter().collect();
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
}
