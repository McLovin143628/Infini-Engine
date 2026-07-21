//! Transform-replication glue (ROADMAP P14.3 item 2).
//!
//! Pure functions bridging the ECS world and the sans-io [`inf_net`] snapshot
//! wire type. The **source** side ([`snapshot_world`]) reads every replicated
//! entity's Guid + local transform into a [`NetSnapshot`]; the **sink** side
//! ([`apply_snapshot`]) writes a received snapshot back onto Guid-matched entities.
//! Both are ordinary functions over [`EcsWorld`] — no sockets, no async — so the
//! transform-replication proof (`replicates_transforms_between_two_worlds`) needs
//! no real network: world A steps and snapshots, world B applies, and B's
//! transforms equal A's.
//!
//! Replication is taken at **snapshot boundaries** (after a fixed step), so it
//! never enters the deterministic step itself (§2.5): the network is IO layered on
//! top of the sim, not inside it.
//!
//! ## Facade note
//!
//! Reading/writing the `Transform` component uses inherent `bevy_ecs::World`
//! methods on the world the `inf-ecs` facade hands back via
//! [`EcsWorld::world_mut`]; `inf-runtime` names only the facade's re-exported
//! `Guid`/`Transform`/`Vec3d` types, never `bevy_ecs` itself (the three-ring rule
//! holds — `inf-runtime` has no `bevy_ecs` dependency).

use inf_ecs::{EcsWorld, Guid, Transform, Vec3d};
use inf_net::{NetId, NetSnapshot, TransformState};

use crate::GameLoop;

fn transform_to_state(t: &Transform) -> TransformState {
    TransformState {
        translation: [t.translation.x, t.translation.y, t.translation.z],
        rotation: [t.rotation.x, t.rotation.y, t.rotation.z],
        scale: [t.scale.x, t.scale.y, t.scale.z],
    }
}

fn state_into_transform(state: &TransformState, t: &mut Transform) {
    t.translation = Vec3d::new(
        state.translation[0],
        state.translation[1],
        state.translation[2],
    );
    t.rotation = Vec3d::new(state.rotation[0], state.rotation[1], state.rotation[2]);
    t.scale = Vec3d::new(state.scale[0], state.scale[1], state.scale[2]);
}

/// Read every replicated entity's (Guid → local transform) into a snapshot tagged
/// with `tick`. Deterministic (the wire snapshot is a `BTreeMap`, so byte-identical
/// worlds encode identically).
pub fn snapshot_world(world: &mut EcsWorld, tick: u64) -> NetSnapshot {
    let w = world.world_mut();
    let entries: Vec<(u128, TransformState)> = w
        .query::<(&Guid, &Transform)>()
        .iter(w)
        .map(|(g, t)| (g.0.as_u128(), transform_to_state(t)))
        .collect();
    let mut snap = NetSnapshot::new(tick);
    for (id, state) in entries {
        snap.insert(NetId(id), state);
    }
    snap
}

/// Write a received snapshot onto Guid-matched entities' local transforms. Entities
/// present in the snapshot but absent locally are ignored (spawn replication is a
/// documented follow-up — see the net memo); locally-present entities not in the
/// snapshot are left untouched. Marks the world dirty and re-propagates so world
/// transforms reflect the applied locals.
pub fn apply_snapshot(world: &mut EcsWorld, snapshot: &NetSnapshot) {
    for (id, state) in &snapshot.transforms {
        let guid = uuid::Uuid::from_u128(id.0);
        let Some(entity) = world.entity_of(guid) else {
            continue;
        };
        if let Some(mut t) = world.world_mut().get_mut::<Transform>(entity) {
            state_into_transform(state, &mut t);
        }
    }
    world.mark_dirty();
    world.propagate();
}

/// A source of replicated world state. The authoritative peer implements this to
/// hand the transport a snapshot each network tick.
pub trait ReplicationSource {
    /// A snapshot of the current replicated state.
    fn snapshot(&mut self) -> NetSnapshot;
}

impl ReplicationSource for GameLoop {
    fn snapshot(&mut self) -> NetSnapshot {
        let tick = self.steps();
        snapshot_world(self.world_mut(), tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_ecs::{sim, ScheduleMode, SimConfig};
    use uuid::Uuid;

    fn seeded() -> EcsWorld {
        let mut w = EcsWorld::new();
        for i in 0..8u128 {
            let e = w.spawn_with_guid(Uuid::from_u128(i + 1), "m", None);
            sim::set_velocity(&mut w, e, Vec3d::new(2.0, 1.0, -1.0));
            sim::set_angular_velocity(&mut w, e, 30.0);
        }
        w
    }

    #[test]
    fn replicates_transforms_between_two_worlds() {
        // Two loops seeded identically (same guids). A advances; B does not.
        let mut a = GameLoop::new(seeded(), SimConfig::default(), ScheduleMode::Serial);
        let mut b = GameLoop::new(seeded(), SimConfig::default(), ScheduleMode::Serial);

        // 15 steps (< the director's step-20 spawn cadence) keeps A's and B's
        // entity sets identical, so this isolates *transform* replication from
        // spawn replication (a documented follow-up).
        for _ in 0..15 {
            a.step_once();
        }

        // Before replication, B differs from A.
        let (ta, tb) = (a.steps(), b.steps());
        let snap_a = snapshot_world(a.world_mut(), ta);
        let snap_b_before = snapshot_world(b.world_mut(), tb);
        assert_ne!(snap_a.transforms, snap_b_before.transforms);

        // Replicate A → B.
        apply_snapshot(b.world_mut(), &snap_a);

        // Now B's replicated transforms equal A's (the transform-replication proof).
        let tb2 = b.steps();
        let snap_b_after = snapshot_world(b.world_mut(), tb2);
        assert_eq!(snap_a.transforms, snap_b_after.transforms);
        assert_eq!(snap_a.transforms.len(), 8);
    }

    #[test]
    fn snapshot_encode_is_deterministic_across_equal_worlds() {
        let mut a = GameLoop::new(seeded(), SimConfig::default(), ScheduleMode::Serial);
        let mut b = GameLoop::new(seeded(), SimConfig::default(), ScheduleMode::Serial);
        for _ in 0..15 {
            a.step_once();
            b.step_once();
        }
        // Identical deterministic sims → byte-identical snapshots.
        let (ta, tb) = (a.steps(), b.steps());
        let sa = snapshot_world(a.world_mut(), ta);
        let sb = snapshot_world(b.world_mut(), tb);
        assert_eq!(sa.encode(), sb.encode());
    }

    #[test]
    fn delta_apply_round_trips_over_the_wire_types() {
        let mut a = GameLoop::new(seeded(), SimConfig::default(), ScheduleMode::Serial);
        let t0 = a.steps();
        let baseline = snapshot_world(a.world_mut(), t0);
        for _ in 0..10 {
            a.step_once();
        }
        let t1 = a.steps();
        let next = snapshot_world(a.world_mut(), t1);

        // Only-changed delta, encoded + decoded, reconstructs the full snapshot.
        let delta = next.delta(&baseline);
        let wire = delta.encode();
        let decoded = inf_net::SnapshotDelta::decode(&wire).unwrap();
        assert_eq!(baseline.apply_delta(&decoded), next);
    }

    #[test]
    fn replication_source_trait_snapshots_the_loop() {
        let mut a = GameLoop::new(seeded(), SimConfig::default(), ScheduleMode::Serial);
        a.step_once();
        let snap = a.snapshot();
        assert_eq!(snap.transforms.len(), 8);
        assert_eq!(snap.tick, 1);
    }
}
