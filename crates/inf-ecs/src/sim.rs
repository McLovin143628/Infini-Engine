//! Runtime simulation components, resources, and systems (P9.1a, §2.5).
//!
//! This module is the **content** of the parallel sim schedule ([`crate::schedule`]).
//! Because `inf-ecs` is the sole crate that may name `bevy_ecs`, every system that
//! needs real, access-declared parallelism (so the executor can prove two systems
//! are non-conflicting and run them concurrently) must live behind this facade —
//! outside crates (`inf-runtime`, the player) compose these systems through the
//! [`crate::schedule::SimSchedule`] builder and drive the world through
//! [`crate::EcsWorld`], never touching a `Query`/`Res`/`Commands` directly.
//!
//! ## The determinism discipline (the phase's spine)
//!
//! The parallel schedule keeps the fixed-step sim **bit-deterministic** by three
//! rules, all enforced here:
//!
//! 1. **Disjoint or ordered.** Systems that touch overlapping component/resource
//!    state carry explicit ordering edges (see the `SimSet` wiring in
//!    [`crate::schedule`]); systems with disjoint access are free to run in any
//!    order — the result is identical because there is no shared mutable state.
//!    [`crate::schedule::SimSchedule::conflict_report`] asserts nothing conflicts
//!    *without* an ordering edge.
//! 2. **No within-system order dependence.** Every per-entity system
//!    ([`integrate_velocity`], [`apply_spin`], …) computes each entity's next
//!    state from *that entity's own* data, so query-iteration order never changes
//!    the outcome. Cross-entity coupling flows only through the gather→scatter
//!    resource pattern ([`accumulate_centroid`] → [`attract_to_centroid`]), whose
//!    reduction runs inside a single serial system (the executor parallelizes
//!    *across* systems, never *within* one query loop) so the floating-point sum
//!    order is fixed regardless of pool size.
//! 3. **Structural changes at sync points only.** Spawns and despawns are never
//!    applied inline; they are *requested* into [`SimControl`] and enacted by
//!    [`apply_spawns`]/[`apply_despawns`] through `Commands` (deferred), which the
//!    schedule flushes at an `ApplyDeferred` barrier before transform propagation.
//!    Two runs enqueue the same requests in the same order, so archetype evolution
//!    — and therefore query iteration order — is identical across runs and pool
//!    sizes.
//!
//! Entity ids are never hashed: the replay trace ([`EntitySimState`]) is keyed and
//! sorted by the stable [`Guid`], so recycled `Entity` indices cannot perturb it.

use bevy_ecs::prelude::*;
use serde::Serialize;
use uuid::Uuid;

use crate::components::{ComputedVisibility, GlobalTransform, Guid, Name, Transform, Visibility};
use crate::math::Vec3d;
use crate::world::EcsWorld;

// ─────────────────────────────── components ────────────────────────────────

/// Linear velocity in world units per second. Integrated into
/// [`Transform::translation`] by [`integrate_velocity`].
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Velocity(pub Vec3d);

/// Angular velocity about the world **Y** axis, in degrees per second. Applied to
/// [`Transform::rotation`]`.y` by [`apply_spin`].
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct AngularVelocity(pub f64);

/// A countdown (seconds) after which [`reap_expired`] requests the entity's
/// despawn at the next structural sync point. Aged by [`age_lifetime`].
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Lifetime {
    pub remaining: f64,
}

// ─────────────────────────────── resources ─────────────────────────────────

/// Fixed timestep (seconds) shared by every integrating system. Read-only during
/// a step, so all movers can read it concurrently.
#[derive(Resource, Clone, Copy, Debug)]
pub struct SimTime {
    pub dt: f64,
}

/// The axis-aligned bounce box: movers reflect off `min`/`max` per axis.
#[derive(Resource, Clone, Copy, Debug)]
pub struct SimBounds {
    pub min: Vec3d,
    pub max: Vec3d,
    /// Spring constant pulling each mover toward the swarm centroid (the
    /// cross-entity coupling term). `0` disables attraction.
    pub attract: f64,
}

/// Monotonic fixed-step counter, advanced once per step by [`tick_counter`].
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct SimTick {
    pub step: u64,
}

/// The swarm centroid computed each step by [`accumulate_centroid`] and consumed
/// by [`attract_to_centroid`] — the resource seam that carries cross-entity reads
/// deterministically (gather into a resource, scatter from it).
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct Centroid {
    pub sum: Vec3d,
    pub count: u64,
}

impl Centroid {
    fn mean(&self) -> Vec3d {
        if self.count == 0 {
            Vec3d::ZERO
        } else {
            let n = self.count as f64;
            Vec3d::new(self.sum.x / n, self.sum.y / n, self.sum.z / n)
        }
    }
}

/// A deferred spawn request (enacted at the structural sync point).
#[derive(Clone, Debug)]
pub struct SpawnRequest {
    pub guid: Uuid,
    pub name: String,
    pub translation: Vec3d,
    pub velocity: Vec3d,
    pub lifetime: Option<f64>,
}

/// The structural-change command queue. Systems *request* spawns/despawns here;
/// [`apply_spawns`]/[`apply_despawns`] drain it through `Commands` at the sync
/// point. Every writer of this resource is explicitly ordered (see the `SimSet`
/// wiring) so the request order — and thus determinism — is fixed.
#[derive(Resource, Default)]
pub struct SimControl {
    pub spawn: Vec<SpawnRequest>,
    pub despawn: Vec<Uuid>,
    /// Deterministic id source for director-spawned entities.
    pub next_spawn: u128,
}

/// Tuning for the standard runtime sim (the values the game loop / harness seed).
#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub fixed_dt: f64,
    pub bounds_min: Vec3d,
    pub bounds_max: Vec3d,
    pub attract: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            fixed_dt: 1.0 / 60.0,
            bounds_min: Vec3d::new(-50.0, -50.0, -50.0),
            bounds_max: Vec3d::new(50.0, 50.0, 50.0),
            attract: 0.5,
        }
    }
}

/// Install the sim resources onto a world (idempotent — overwrites). The runtime
/// calls this once before running the schedule.
pub fn install_resources(world: &mut EcsWorld, config: SimConfig) {
    let w = world.world_mut();
    w.insert_resource(SimTime {
        dt: config.fixed_dt,
    });
    w.insert_resource(SimBounds {
        min: config.bounds_min,
        max: config.bounds_max,
        attract: config.attract,
    });
    w.insert_resource(SimTick::default());
    w.insert_resource(Centroid::default());
    w.insert_resource(SimControl::default());
}

// ─────────────────────────────── systems ───────────────────────────────────
//
// Access declarations (what the executor uses to prove (non-)conflict):
//   tick_counter        : W SimTick
//   director            : R SimTick, R SimTime,  W SimControl
//   accumulate_centroid : R Transform,           W Centroid
//   attract_to_centroid : R Transform, R Centroid, R SimTime, W Velocity
//   integrate_velocity  : W Transform, R Velocity, R SimTime
//   apply_spin          : W Transform, R AngularVelocity, R SimTime
//   bounce_bounds       : W Transform, W Velocity, R SimBounds
//   age_lifetime        : W Lifetime, R SimTime
//   reap_expired        : R Guid, R Lifetime,     W SimControl
//   apply_spawns        : Commands,               W SimControl
//   apply_despawns      : Commands, R Guid,       W SimControl
//   propagate_transforms: exclusive (&mut World)  — the sync-point tail

/// Advance the fixed-step counter (the frame clock).
pub(crate) fn tick_counter(mut tick: ResMut<SimTick>) {
    tick.step += 1;
}

/// Deterministically drive structural churn: spawn a short-lived mover on a fixed
/// cadence. Exercises the spawn→despawn sync-point path in the replay harness.
pub(crate) fn director(tick: Res<SimTick>, time: Res<SimTime>, mut ctrl: ResMut<SimControl>) {
    // Spawn a transient mover every 20 steps; it lives ~40 steps, so spawns and
    // despawns interleave across a 300-step trace.
    if tick.step.is_multiple_of(20) {
        let id = ctrl.next_spawn;
        ctrl.next_spawn += 1;
        // High tag bits keep director guids clear of the seeded low-index guids.
        let guid = Uuid::from_u128(0x5EED_0000_0000_0000_0000_0000_0000_0000u128 | id);
        let s = tick.step as f64;
        ctrl.spawn.push(SpawnRequest {
            guid,
            name: format!("transient-{id}"),
            translation: Vec3d::new((tick.step % 7) as f64, 3.0, (tick.step % 5) as f64),
            velocity: Vec3d::new(1.0 + (id % 3) as f64, 0.5, -0.75 - (s * 0.0)),
            lifetime: Some(40.0 * time.dt),
        });
    }
}

/// Gather step: sum every mover's local translation into the [`Centroid`]
/// resource. Runs before any Transform writer (ordering edge) so it reads
/// pre-move positions; the sum is computed serially inside this one system, so it
/// is pool-size-invariant.
pub(crate) fn accumulate_centroid(q: Query<&Transform>, mut centroid: ResMut<Centroid>) {
    let mut sum = Vec3d::ZERO;
    let mut count = 0u64;
    for t in &q {
        sum = Vec3d::new(
            sum.x + t.translation.x,
            sum.y + t.translation.y,
            sum.z + t.translation.z,
        );
        count += 1;
    }
    centroid.sum = sum;
    centroid.count = count;
}

/// Scatter step: nudge each mover's velocity toward the swarm centroid (a spring).
/// Pure per-entity write from a shared read — deterministic under any pool size.
#[tracing::instrument(level = "trace", skip_all)]
pub(crate) fn attract_to_centroid(
    mut q: Query<(&mut Velocity, &Transform)>,
    centroid: Res<Centroid>,
    bounds: Res<SimBounds>,
    time: Res<SimTime>,
) {
    if bounds.attract == 0.0 {
        return;
    }
    let mean = centroid.mean();
    for (mut vel, t) in &mut q {
        let to_center = Vec3d::new(
            mean.x - t.translation.x,
            mean.y - t.translation.y,
            mean.z - t.translation.z,
        );
        let k = bounds.attract * time.dt;
        vel.0 = Vec3d::new(
            vel.0.x + to_center.x * k,
            vel.0.y + to_center.y * k,
            vel.0.z + to_center.z * k,
        );
    }
}

/// Integrate velocity into translation.
#[tracing::instrument(level = "trace", skip_all)]
pub(crate) fn integrate_velocity(mut q: Query<(&mut Transform, &Velocity)>, time: Res<SimTime>) {
    let dt = time.dt;
    for (mut t, vel) in &mut q {
        t.translation = Vec3d::new(
            t.translation.x + vel.0.x * dt,
            t.translation.y + vel.0.y * dt,
            t.translation.z + vel.0.z * dt,
        );
    }
}

/// Apply angular velocity about world Y.
pub(crate) fn apply_spin(mut q: Query<(&mut Transform, &AngularVelocity)>, time: Res<SimTime>) {
    let dt = time.dt;
    for (mut t, spin) in &mut q {
        let mut r = t.rotation;
        r.y += spin.0 * dt;
        t.rotation = r;
    }
}

/// Reflect movers off the bounce box and flip the corresponding velocity axis.
pub(crate) fn bounce_bounds(mut q: Query<(&mut Transform, &mut Velocity)>, bounds: Res<SimBounds>) {
    for (mut t, mut vel) in &mut q {
        let mut p = t.translation;
        let mut v = vel.0;
        // x
        if p.x > bounds.max.x {
            p.x = bounds.max.x;
            v.x = -v.x;
        } else if p.x < bounds.min.x {
            p.x = bounds.min.x;
            v.x = -v.x;
        }
        // y
        if p.y > bounds.max.y {
            p.y = bounds.max.y;
            v.y = -v.y;
        } else if p.y < bounds.min.y {
            p.y = bounds.min.y;
            v.y = -v.y;
        }
        // z
        if p.z > bounds.max.z {
            p.z = bounds.max.z;
            v.z = -v.z;
        } else if p.z < bounds.min.z {
            p.z = bounds.min.z;
            v.z = -v.z;
        }
        t.translation = p;
        vel.0 = v;
    }
}

/// Age lifetimes. Disjoint from every Transform/Velocity system, so it runs
/// concurrently with the whole motion pipeline — the schedule's parallelism
/// showcase.
#[tracing::instrument(level = "trace", skip_all)]
pub(crate) fn age_lifetime(mut q: Query<&mut Lifetime>, time: Res<SimTime>) {
    let dt = time.dt;
    for mut l in &mut q {
        l.remaining -= dt;
    }
}

/// Queue expired entities for despawn (does not despawn inline — that would be a
/// structural change outside a sync point).
pub(crate) fn reap_expired(q: Query<(&Guid, &Lifetime)>, mut ctrl: ResMut<SimControl>) {
    for (g, life) in &q {
        if life.remaining <= 0.0 {
            ctrl.despawn.push(g.0);
        }
    }
}

/// Enact queued spawns through deferred `Commands` (applied at the sync point).
#[tracing::instrument(level = "trace", skip_all)]
pub(crate) fn apply_spawns(mut commands: Commands, mut ctrl: ResMut<SimControl>) {
    let reqs = std::mem::take(&mut ctrl.spawn);
    for req in reqs {
        let mut e = commands.spawn((
            Guid(req.guid),
            Name(req.name),
            Transform::from_translation(req.translation.to_dvec3()),
            Visibility::default(),
            GlobalTransform::default(),
            ComputedVisibility::default(),
            Velocity(req.velocity),
        ));
        if let Some(remaining) = req.lifetime {
            e.insert(Lifetime { remaining });
        }
    }
}

/// Enact queued despawns through deferred `Commands`. Resolves guid→entity from a
/// read-only query so it never conflicts with a Transform/Velocity writer.
pub(crate) fn apply_despawns(
    mut commands: Commands,
    mut ctrl: ResMut<SimControl>,
    q: Query<(&Guid, Entity)>,
) {
    let ids = std::mem::take(&mut ctrl.despawn);
    if ids.is_empty() {
        return;
    }
    for guid in ids {
        if let Some((_, entity)) = q.iter().find(|(g, _)| g.0 == guid) {
            commands.entity(entity).despawn();
        }
    }
}

/// The transform-propagation sync-point tail: an exclusive system (`&mut World`)
/// so it runs alone after every Transform writer and after deferred structural
/// commands have been applied. Reuses the audited P3 DFS propagation.
///
/// Propagation is intentionally serial for now (a whole-world hierarchy walk);
/// parallelizing it per-root is a documented §2.5 follow-up — the parallel win in
/// this phase is running the independent gameplay systems concurrently.
#[tracing::instrument(level = "trace", skip_all)]
pub(crate) fn propagate_transforms(world: &mut World) {
    crate::transform::propagate(world);
}

// ─────────────────────────── replay trace state ────────────────────────────

/// A deterministic, serializable slice of one entity's sim state — the unit of
/// the replay trace. Keyed by [`Guid`] (never `Entity`), so the trace is stable
/// across entity-id recycling and pool sizes.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct EntitySimState {
    pub guid: Uuid,
    /// World-space translation (post-propagation).
    pub world_translation: [f64; 3],
    /// Local rotation, euler degrees.
    pub rotation: [f64; 3],
    pub velocity: Option<[f64; 3]>,
    pub lifetime: Option<f64>,
}

/// Snapshot the whole world's sim state, **sorted by `Guid`** for a deterministic
/// trace. This is the facade read that `inf-runtime` hashes each step (keeping
/// `bevy_ecs` — and the hashing crate — out of the runtime).
pub fn sim_snapshot(world: &mut EcsWorld) -> Vec<EntitySimState> {
    let w = world.world_mut();
    let mut out: Vec<EntitySimState> = w
        .query::<(
            &Guid,
            &Transform,
            &GlobalTransform,
            Option<&Velocity>,
            Option<&Lifetime>,
        )>()
        .iter(w)
        .map(|(g, t, gt, vel, life)| {
            let wt = gt.translation();
            EntitySimState {
                guid: g.0,
                world_translation: [wt.x, wt.y, wt.z],
                rotation: [t.rotation.x, t.rotation.y, t.rotation.z],
                velocity: vel.map(|v| [v.0.x, v.0.y, v.0.z]),
                lifetime: life.map(|l| l.remaining),
            }
        })
        .collect();
    out.sort_by_key(|s| s.guid);
    out
}

// ───────────────────────────── seeding helpers ─────────────────────────────

/// Set an entity's local translation (facade convenience for world seeding).
pub fn set_translation(world: &mut EcsWorld, entity: Entity, translation: Vec3d) {
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(entity) {
        t.translation = translation;
    }
    world.mark_dirty();
}

/// Attach linear velocity to an existing editor entity (facade convenience so the
/// runtime can build a moving world without naming `bevy_ecs`).
pub fn set_velocity(world: &mut EcsWorld, entity: Entity, velocity: Vec3d) {
    world
        .world_mut()
        .entity_mut(entity)
        .insert(Velocity(velocity));
}

/// Attach angular velocity (spin about Y, deg/s) to an entity.
pub fn set_angular_velocity(world: &mut EcsWorld, entity: Entity, deg_per_sec: f64) {
    world
        .world_mut()
        .entity_mut(entity)
        .insert(AngularVelocity(deg_per_sec));
}

/// Attach a lifetime countdown (seconds) to an entity.
pub fn set_lifetime(world: &mut EcsWorld, entity: Entity, remaining: f64) {
    world
        .world_mut()
        .entity_mut(entity)
        .insert(Lifetime { remaining });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_resources_are_present() {
        let mut w = EcsWorld::new();
        install_resources(&mut w, SimConfig::default());
        assert!(w.world().get_resource::<SimTime>().is_some());
        assert!(w.world().get_resource::<SimBounds>().is_some());
        assert!(w.world().get_resource::<SimControl>().is_some());
    }

    #[test]
    fn snapshot_is_guid_sorted() {
        let mut w = EcsWorld::new();
        // Spawn out of guid order.
        let e1 = w.spawn_with_guid(Uuid::from_u128(9), "b", None);
        let _e2 = w.spawn_with_guid(Uuid::from_u128(1), "a", None);
        set_velocity(&mut w, e1, Vec3d::new(1.0, 0.0, 0.0));
        w.propagate();
        let snap = sim_snapshot(&mut w);
        assert_eq!(snap.len(), 2);
        assert!(snap[0].guid < snap[1].guid);
    }
}
