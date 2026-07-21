//! Socket attachments (P11.3): the post-anim-tick system that makes an entity
//! ride another entity's socket.
//!
//! [`update_attachments`] reads every [`AttachedTo`] entity, resolves its target's
//! world [`GlobalTransform`], composes the baked socket offset, and writes the
//! result onto the follower's local `Transform`. It runs **after** transform
//! propagation (so targets have fresh globals) and marks the world dirty so the
//! next propagate refreshes the followers' own globals.
//!
//! ## v1 scope
//!
//! The follower is treated as a **root** (unparented) entity, so writing the
//! composed world transform as its local `Transform` is correct — the same
//! roots-are-correct rule the gizmo write-back relies on. Attaching a *parented*
//! follower (world→local solve through the parent) and **pose-driven** socket
//! tracking (evaluating the target skeleton's animated joint, not just its root
//! `GlobalTransform`) are documented follow-ups — the pure joint math already
//! lives in [`inf_anim::socket_transform`](../../inf_anim/index.html).

use crate::components::{AttachedTo, GlobalTransform, Transform};
use crate::math::Vec3d;
use crate::world::EcsWorld;

/// Update every [`AttachedTo`] entity to ride its target's socket for this step.
///
/// Runs in the fixed tick after `propagate()`. Followers whose target is missing
/// (despawned, or a dangling GUID) are left untouched. Deterministic: the write
/// for entity A never depends on entity B's *new* value this pass (targets are
/// read from `GlobalTransform`, which propagation already settled).
pub fn update_attachments(world: &mut EcsWorld) {
    // 1. Gather (follower_entity, target_guid, offset_affine) — a read pass so we
    //    don't hold a query borrow while mutating.
    let mut work: Vec<(bevy_ecs::entity::Entity, uuid::Uuid, glam::DAffine3)> = Vec::new();
    {
        let w = world.world_mut();
        let mut q = w.query::<(bevy_ecs::entity::Entity, &AttachedTo)>();
        for (entity, att) in q.iter(w) {
            work.push((entity, att.target, att.offset_affine()));
        }
    }
    if work.is_empty() {
        return;
    }

    // 2. Resolve each target's world transform and write the follower's local.
    let mut changed = false;
    for (follower, target_guid, offset) in work {
        let Some(target) = world.entity_of(target_guid) else {
            continue;
        };
        let Some(target_global) = world.world().get::<GlobalTransform>(target).map(|g| g.0) else {
            continue;
        };
        let world_affine = target_global * offset;
        let (scale, rot, trans) = world_affine.to_scale_rotation_translation();
        if let Some(mut t) = world.world_mut().get_mut::<Transform>(follower) {
            t.translation = Vec3d::from_dvec3(trans);
            t.set_quat(rot);
            t.scale = Vec3d::from_dvec3(scale);
            changed = true;
        }
    }
    if changed {
        world.mark_dirty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{AttachedTo, Guid, Transform};
    use crate::math::Vec3d;
    use glam::DVec3;
    use uuid::Uuid;

    #[test]
    fn follower_rides_the_target_with_offset() {
        let mut world = EcsWorld::new();
        let target_guid = Uuid::from_u128(1);

        let target = world
            .world_mut()
            .spawn((
                Guid(target_guid),
                Transform::from_translation(DVec3::new(10.0, 0.0, 0.0)),
                GlobalTransform::default(),
            ))
            .id();
        let _ = target;
        world.reindex_guids();

        let follower = world
            .world_mut()
            .spawn((
                Guid(Uuid::from_u128(2)),
                Transform::IDENTITY,
                GlobalTransform::default(),
                AttachedTo::new(target_guid, "hand_r", Vec3d::new(0.0, 1.0, 0.0)),
            ))
            .id();
        world.reindex_guids();

        // Propagate so the target has a fresh global, then attach, then propagate
        // again so the follower's own global reflects its new local.
        world.mark_dirty();
        world.propagate();
        update_attachments(&mut world);
        world.propagate();

        let g = world.world().get::<GlobalTransform>(follower).unwrap();
        // Target at (10,0,0) + offset (0,1,0) → follower at (10,1,0).
        assert!((g.translation() - DVec3::new(10.0, 1.0, 0.0)).length() < 1e-9);
    }

    #[test]
    fn follower_tracks_a_moved_target() {
        let mut world = EcsWorld::new();
        let target_guid = Uuid::from_u128(1);
        let target = world
            .world_mut()
            .spawn((
                Guid(target_guid),
                Transform::from_translation(DVec3::ZERO),
                GlobalTransform::default(),
            ))
            .id();
        let follower = world
            .world_mut()
            .spawn((
                Guid(Uuid::from_u128(2)),
                Transform::IDENTITY,
                GlobalTransform::default(),
                AttachedTo::new(target_guid, "", Vec3d::ZERO),
            ))
            .id();
        world.reindex_guids();

        // Move the target, re-run the attach step.
        if let Some(mut t) = world.world_mut().get_mut::<Transform>(target) {
            t.translation = Vec3d::new(3.0, 4.0, 0.0);
        }
        world.mark_dirty();
        world.propagate();
        update_attachments(&mut world);
        world.propagate();

        let g = world.world().get::<GlobalTransform>(follower).unwrap();
        assert!((g.translation() - DVec3::new(3.0, 4.0, 0.0)).length() < 1e-9);
    }
}
