//! Socket attachments (P11.3, pose-driven since P24.1): the post-anim-tick system
//! that makes an entity ride another entity's socket.
//!
//! [`update_attachments`] reads every [`AttachedTo`] entity, resolves its target's
//! world [`GlobalTransform`], composes the **animated socket's** model-space
//! transform and the attachment's own offset, and writes the result onto the
//! follower's local `Transform`. It runs **after** transform propagation (so
//! targets have fresh globals) *and* after the pose slot
//! ([`crate::pose::step_pose_evaluation`], so the socket is this step's), and it
//! marks the world dirty so the next propagate refreshes the followers' own
//! globals.
//!
//! # The socket, and what it replaces
//!
//! Until P24.1 this composed `target.GlobalTransform · offset` and nothing else:
//! a sword "attached to `hand_r`" rode the character's **origin**, so it hung at
//! the pelvis and stayed there while the hand swung. The socket name was recorded
//! and never read; `inf_anim::socket_transform` had no runtime caller at all.
//!
//! What made it fixable is that the pose is now sim state
//! ([`crate::pose::EvaluatedPose`]) with the skeleton's socket table already
//! composed into model space by the host that owns the `.inf_skel`. So this
//! system needs no asset access of its own — it looks a name up in a map — and
//! the composition is:
//!
//! ```text
//! follower_world = target.GlobalTransform · socket_model · offset
//! ```
//!
//! # The origin fallback survives, deliberately
//!
//! Three cases keep the pre-P24.1 behaviour, and each is a real authoring state
//! rather than a failure: an attachment with an **empty** socket name (attach to
//! the target's origin — the documented meaning), a target with **no evaluated
//! pose** (no state machine, or its skeleton did not resolve), and a socket name
//! the target's skeleton does not author. Falling back is right because the
//! alternative is a weapon that vanishes to the world origin the moment a rig is
//! unbound.
//!
//! ## v1 scope
//!
//! The follower is treated as a **root** (unparented) entity, so writing the
//! composed world transform as its local `Transform` is correct — the same
//! roots-are-correct rule the gizmo write-back relies on. Attaching a *parented*
//! follower (world→local solve through the parent) is still a documented
//! follow-up. So is driving a socket from an `AnimPlayer` rather than a state
//! machine: the pose store is populated by the machine slot, so a clip-only
//! character's sockets do not move yet.

use crate::components::{AttachedTo, GlobalTransform, Transform};
use crate::math::Vec3d;
use crate::pose::evaluated_pose;
use crate::world::EcsWorld;

/// Update every [`AttachedTo`] entity to ride its target's socket for this step.
///
/// Runs in the fixed tick after `propagate()` and after the pose slot. Followers
/// whose target is missing (despawned, or a dangling GUID) are left untouched.
/// Deterministic: the write for entity A never depends on entity B's *new* value
/// this pass (targets are read from `GlobalTransform`, which propagation already
/// settled, and from the pose store, which the pose slot already published).
pub fn update_attachments(world: &mut EcsWorld) {
    // 1. Gather (follower_entity, target_guid, socket name, offset_affine) — a
    //    read pass so we don't hold a query borrow while mutating.
    let mut work: Vec<(bevy_ecs::entity::Entity, uuid::Uuid, String, glam::DAffine3)> = Vec::new();
    {
        let w = world.world_mut();
        let mut q = w.query::<(bevy_ecs::entity::Entity, &AttachedTo)>();
        for (entity, att) in q.iter(w) {
            work.push((entity, att.target, att.socket.clone(), att.offset_affine()));
        }
    }
    if work.is_empty() {
        return;
    }

    // 2. Resolve each target's world transform + animated socket, and write the
    //    follower's local.
    let mut changed = false;
    for (follower, target_guid, socket, offset) in work {
        let Some(target) = world.entity_of(target_guid) else {
            continue;
        };
        let Some(target_global) = world.world().get::<GlobalTransform>(target).map(|g| g.0) else {
            continue;
        };
        // The socket's MODEL-space transform under this step's evaluated pose, or
        // the identity when the attachment names no socket / the target is not
        // posed / the skeleton authors no such socket (see the module docs — all
        // three are the documented origin fallback, not an error).
        let socket_local = if socket.is_empty() {
            glam::DAffine3::IDENTITY
        } else {
            evaluated_pose(world, target_guid)
                .and_then(|p| p.socket(&socket))
                .map(|m| glam::DAffine3::from_mat4(m.as_dmat4()))
                .unwrap_or(glam::DAffine3::IDENTITY)
        };
        let world_affine = target_global * socket_local * offset;
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
    // ── P24.1: the socket the attachment names ──────────────────────────────

    use crate::components::{AnimStateMachine, SkeletalMesh};
    use crate::pose::step_pose_evaluation;
    use inf_anim::{
        AnimClip, ClipRef, Interpolation, Joint, JointTrack, JointTransform, QuatTrack, Skeleton,
        SkeletonAsset, SmState, SmTransition, Socket, StateMachine,
    };
    use std::collections::BTreeMap;

    const SM: Uuid = Uuid::from_u128(0x2401_0501);
    const SKEL: Uuid = Uuid::from_u128(0x2401_0502);
    const REST: ClipRef = [1; 16];
    const SWING: ClipRef = [2; 16];

    /// A 3-joint chain along +Y with a `hand_r` socket on the tip (joint 2), so
    /// the socket sits at (0, 2, 0) in the rest pose.
    fn chain_rig() -> SkeletonAsset {
        let mut joints = Vec::new();
        let mut global = glam::Mat4::IDENTITY;
        for i in 0..3 {
            let local = JointTransform::from_trs(
                if i == 0 {
                    glam::Vec3::ZERO
                } else {
                    glam::Vec3::Y
                },
                glam::Quat::IDENTITY,
                glam::Vec3::ONE,
            );
            global *= local.to_mat4();
            joints.push(Joint {
                name: format!("j{i}"),
                parent: if i == 0 { None } else { Some(i as u16 - 1) },
                inverse_bind: global.inverse().to_cols_array(),
                local_bind: local,
            });
        }
        SkeletonAsset::with_sockets(
            Skeleton::new(joints).unwrap(),
            vec![Socket::new("hand_r", 2)],
        )
    }

    /// A clip holding joint 1 at `deg` about +Z (one stepped key → an exact
    /// constant pose, so "the socket moved" is a statement about the STATE).
    fn hold(deg: f32) -> AnimClip {
        let q = glam::Quat::from_rotation_z(deg.to_radians()).to_array();
        // Through the constructor since `.inf_anim` v2 (P29.2); `duration` is
        // set afterwards because one key at t=0 derives a zero-length clip and
        // this fixture wants a 1 s state period.
        let mut clip = AnimClip::new(
            "hold",
            vec![JointTrack {
                joint: 1,
                translation: None,
                rotation: Some(QuatTrack::new(vec![0.0], vec![q], Interpolation::Step)),
                scale: None,
            }],
        );
        clip.duration = 1.0;
        clip
    }

    /// rest → swing, unconditional, so the machine leaves its entry state on the
    /// first fixed step with no actor involved.
    fn swing_machine() -> StateMachine {
        StateMachine {
            states: vec![SmState::clip("rest", REST), SmState::clip("swing", SWING)],
            transitions: vec![SmTransition::new(0, 1, 0.0)],
            entry: 0,
            ..Default::default()
        }
    }

    /// Run the pose slot once against the fixture's registries.
    fn pose_step(world: &mut EcsWorld, rig: &SkeletonAsset, clips: &BTreeMap<ClipRef, AnimClip>) {
        let sm = swing_machine();
        let machines = |g: Uuid| (g == SM).then_some(&sm);
        let skeletons = |g: Uuid| (g == SKEL).then_some(rig);
        let clip = |c: ClipRef| clips.get(&c);
        let vars = |_: Uuid| BTreeMap::new();
        step_pose_evaluation(world, 1.0 / 60.0, &machines, &skeletons, &clip, &vars);
    }

    /// A machine-driven character at `pos` with a follower attached to `socket`.
    /// Returns `(world, follower entity)`.
    fn world_with_attachment(socket: &str, pos: DVec3) -> (EcsWorld, bevy_ecs::entity::Entity) {
        let mut world = EcsWorld::new();
        let target_guid = Uuid::from_u128(1);
        world.world_mut().spawn((
            Guid(target_guid),
            Transform::from_translation(pos),
            GlobalTransform::default(),
            AnimStateMachine {
                sm: Some(SM),
                ..Default::default()
            },
            SkeletalMesh {
                mesh: Some(Uuid::from_u128(9)),
                skeleton: Some(SKEL),
            },
        ));
        let follower = world
            .world_mut()
            .spawn((
                Guid(Uuid::from_u128(2)),
                Transform::IDENTITY,
                GlobalTransform::default(),
                AttachedTo::new(target_guid, socket, Vec3d::ZERO),
            ))
            .id();
        world.reindex_guids();
        world.mark_dirty();
        world.propagate();
        (world, follower)
    }

    /// **The P24.1 headline gate for attachments**: a follower on a socket lands
    /// at the ANIMATED joint's world transform — computed independently here, out
    /// of `global_transforms`, so the test does not read back the same expression
    /// the system wrote.
    #[test]
    fn an_attachment_lands_on_the_animated_joint() {
        let rig = chain_rig();
        let clips = BTreeMap::from([(REST, hold(0.0)), (SWING, hold(90.0))]);
        let origin = DVec3::new(10.0, 0.0, 3.0);
        let (mut world, follower) = world_with_attachment("hand_r", origin);

        // Step 1 fires rest → swing, so the pose is the 90° bend.
        pose_step(&mut world, &rig, &clips);
        update_attachments(&mut world);
        world.propagate();

        // The independent expectation: rebuild the pose the machine is in, run
        // the skeleton forward, and place the socket's joint in world space.
        let mut want_pose = inf_anim::Pose::rest(&rig.skeleton);
        want_pose.locals[1].rotation = glam::Quat::from_rotation_z(90f32.to_radians()).to_array();
        let joint = inf_anim::global_transforms(&rig.skeleton, &want_pose)[2];
        let want = origin + joint.transform_point3(glam::Vec3::ZERO).as_dvec3();

        let got = world
            .world()
            .get::<GlobalTransform>(follower)
            .unwrap()
            .translation();
        assert!(
            (got - want).length() < 1e-5,
            "follower at {got:?}, the animated joint is at {want:?}"
        );
        // …and it is NOT the target's origin, which is where it used to sit.
        assert!(
            (got - origin).length() > 0.5,
            "the follower rode the target's origin — this is the P24.1 defect"
        );
    }

    /// **ANTI-VACUITY**: the attachment MOVES when the machine changes state. A
    /// follower pinned to a static socket agrees with a follower pinned to the
    /// origin on every assertion that only checks one frame.
    #[test]
    fn an_attachment_moves_when_the_machine_changes_state() {
        let rig = chain_rig();
        let clips = BTreeMap::from([(REST, hold(0.0)), (SWING, hold(90.0))]);
        let (mut world, follower) = world_with_attachment("hand_r", DVec3::ZERO);

        // Read the socket BEFORE the machine has been stepped: the store is empty,
        // so this is the origin fallback.
        update_attachments(&mut world);
        world.propagate();
        let unposed = world
            .world()
            .get::<GlobalTransform>(follower)
            .unwrap()
            .translation();
        assert!(unposed.length() < 1e-9, "unposed follows the origin");

        // Step 1: the machine enters `rest` and immediately transitions to
        // `swing`, so the very first published pose is the 90° bend.
        pose_step(&mut world, &rig, &clips);
        update_attachments(&mut world);
        world.propagate();
        let swung = world
            .world()
            .get::<GlobalTransform>(follower)
            .unwrap()
            .translation();
        assert!(
            (swung - unposed).length() > 0.5,
            "the attachment did not move when the machine posed the rig"
        );
        // The chain is 2 m long; a 90° bend at joint 1 puts the tip at (-1, 1, 0).
        assert!(
            (swung - DVec3::new(-1.0, 1.0, 0.0)).length() < 1e-4,
            "{swung:?}"
        );
    }

    /// The three documented origin fallbacks: an empty socket name, a target with
    /// no evaluated pose, and a socket the skeleton does not author.
    #[test]
    fn the_origin_fallback_survives_for_socketless_and_unposed_targets() {
        let rig = chain_rig();
        let clips = BTreeMap::from([(REST, hold(0.0)), (SWING, hold(90.0))]);
        let origin = DVec3::new(4.0, 1.0, 0.0);

        for socket in ["", "no_such_socket"] {
            let (mut world, follower) = world_with_attachment(socket, origin);
            pose_step(&mut world, &rig, &clips);
            update_attachments(&mut world);
            world.propagate();
            let got = world
                .world()
                .get::<GlobalTransform>(follower)
                .unwrap()
                .translation();
            assert!(
                (got - origin).length() < 1e-9,
                "socket {socket:?} should follow the origin, got {got:?}"
            );
        }

        // A target with a socket name but NO machine at all is never posed.
        let mut world = EcsWorld::new();
        let target_guid = Uuid::from_u128(1);
        world.world_mut().spawn((
            Guid(target_guid),
            Transform::from_translation(origin),
            GlobalTransform::default(),
        ));
        let follower = world
            .world_mut()
            .spawn((
                Guid(Uuid::from_u128(2)),
                Transform::IDENTITY,
                GlobalTransform::default(),
                AttachedTo::new(target_guid, "hand_r", Vec3d::ZERO),
            ))
            .id();
        world.reindex_guids();
        world.mark_dirty();
        world.propagate();
        update_attachments(&mut world);
        world.propagate();
        let got = world
            .world()
            .get::<GlobalTransform>(follower)
            .unwrap()
            .translation();
        assert!((got - origin).length() < 1e-9, "{got:?}");
    }

    /// The attachment's own offset still composes, **after** the socket — a
    /// muzzle 30 cm along the weapon is 30 cm from the hand, not from the pelvis.
    #[test]
    fn the_offset_composes_after_the_socket() {
        let rig = chain_rig();
        let clips = BTreeMap::from([(REST, hold(0.0)), (SWING, hold(0.0))]);
        let mut world = EcsWorld::new();
        let target_guid = Uuid::from_u128(1);
        world.world_mut().spawn((
            Guid(target_guid),
            Transform::IDENTITY,
            GlobalTransform::default(),
            AnimStateMachine {
                sm: Some(SM),
                ..Default::default()
            },
            SkeletalMesh {
                mesh: Some(Uuid::from_u128(9)),
                skeleton: Some(SKEL),
            },
        ));
        let follower = world
            .world_mut()
            .spawn((
                Guid(Uuid::from_u128(2)),
                Transform::IDENTITY,
                GlobalTransform::default(),
                AttachedTo::new(target_guid, "hand_r", Vec3d::new(0.0, 0.0, 0.3)),
            ))
            .id();
        world.reindex_guids();
        world.mark_dirty();
        world.propagate();

        pose_step(&mut world, &rig, &clips);
        update_attachments(&mut world);
        world.propagate();
        let got = world
            .world()
            .get::<GlobalTransform>(follower)
            .unwrap()
            .translation();
        // Unbent chain: the socket joint is at (0,2,0), and the offset adds +0.3 Z.
        assert!((got - DVec3::new(0.0, 2.0, 0.3)).length() < 1e-5, "{got:?}");
    }
}
