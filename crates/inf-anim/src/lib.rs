//! Animation: skeletal runtime, clips, blend spaces, state machines.
//!
//! **P11.1 skeletal foundation** — this crate is Ring 0 and **pure**: a
//! deterministic pose pipeline (skeleton → clip sample → local pose → global
//! transforms → GPU skinning palette) with no rendering, ECS, or editor
//! dependencies. Pose math is `f32` on purpose (see [`skeleton`] for the f32 vs.
//! f64 rationale). The `.inf_skel` / `.inf_anim` asset payloads live in
//! [`asset`].
//!
//! ## What landed on the P11.1 seams
//!
//! All of them, by P24.1 — the list below used to read "seams left for later" and
//! outlived every one of them:
//! * **blend spaces** ([`blend_space`]) compose over [`pose::blend_poses`];
//! * **state machines** ([`state_machine`], `.inf_sm`) drive
//!   [`pose::sample_clip`] + [`pose::blend_poses`], and since P24.1 their pose is
//!   evaluated at fixed step by `inf_ecs::pose` and is what both hosts DRAW;
//! * **sockets** ([`sockets`]) and **retarget** ([`retarget`]) both key on
//!   [`skeleton::Joint::name`], which is why it is retained per joint; a socket's
//!   transform reaches an attached entity through `inf_ecs::attach`;
//! * **template body plans** ([`template`], P24.1) generate rigs that use the
//!   [`retarget::humanoid_joint_names`] vocabulary verbatim.
//!
//! * **IK** ([`ik`], P24.2) is a post-pass over an evaluated pose, applied inside
//!   `inf_ecs::pose::step_pose_evaluation` so both hosts inherit it — and it is
//!   `sqrt`-only, because its result rides the sim trace.
//!
//! Since P24.3 the solver **reads [`template::JointLimit`]**: `solve_chain` takes
//! the skeleton's limit table and clamps each hinge as that joint's rotation is
//! written, so an elbow no longer bends backwards because a target asked it to.
//! [`ik::IkReport::clamped`] reports how often the range was load-bearing.
//!
//! Still open: **cubic tracks** — [`clip::Interpolation`] keeps `Step`/`Linear`
//! and cubic is resampled to linear on import (documented in [`clip`]); and a
//! limit free on **more than one axis** is not applied (a swing-twist cone needs
//! a decomposition this does not have — see [`ik`], and ROADMAP §12's P24 block).

pub mod asset;
pub mod blend_space;
pub mod clip;
// P24.4 secondary animation: XPBD cloth over the posed skeleton.
pub mod cloth;
// P24.4 secondary animation: strand hair, on the same solver primitives.
pub mod hair;
// P24.2 inverse kinematics: the post-pass over an evaluated pose.
pub mod ik;
// P24.5 the default locomotion set: clips and a machine derived from a body plan.
pub mod locomotion;
// P24.3 modular rigging: assembling one skeleton out of parts.
pub mod merge;
pub mod pose;
pub mod skeleton;
pub mod state_machine;
// P11.3 character tools (sockets / root motion / retarget) — new modules, each
// pure like the rest of the crate.
pub mod retarget;
pub mod root_motion;
pub mod sockets;
// P24.1 template body plans: the parametric N-pedal skeleton generator.
pub mod template;

pub use asset::{AnimClipAsset, SkeletonAsset, StateMachineAsset};
pub use blend_space::{
    blend_weights_1d, blend_weights_2d, sample_blend_space_1d, sample_blend_space_2d, BlendEntry1D,
    BlendEntry2D, BlendSpace1D, BlendSpace2D, ClipRef,
};
pub use clip::{AnimClip, Interpolation, JointTrack, QuatTrack, Vec3Track};
pub use cloth::{
    body_capsules, capsules_for, step_cloth, Capsule, ClothAsset, ClothCapsule, ClothEdge,
    ClothError, ClothMaterial, ClothState, GRAVITY_M_S2,
};
pub use hair::{
    render_mesh, ribbon_mesh, roots_for, step_hair, HairAsset, HairDetail, HairGroom, HairMaterial,
    HairRoot, HairState, HairStrand,
};
pub use ik::{
    fabrik, rotation_between, solve_chain, two_bone_positions, IkError, IkReport,
    FABRIK_ITERATIONS, MIN_BONE_LENGTH_M, REACH_TOLERANCE_M,
};
pub use locomotion::{
    build_locomotion, locomotion_machine, GaitParams, LegSummary, LocomotionError, LocomotionSet,
    MAX_KEYS_PER_CYCLE, SPEED_VAR, STATE_NAMES,
};
pub use merge::{
    merge_skeletons, mirror_joint_map, mirrored_joint_name, unmatched_sided_joints, SkeletonMerge,
    SkeletonMergeError,
};
pub use pose::{
    advance_clip_time, blend_poses, blend_poses_weighted, global_transforms, sample_clip,
    skinning_matrices, Pose,
};
pub use retarget::{humanoid_joint_names, retarget_pose, RetargetMap};
pub use root_motion::{root_delta, root_delta_world, root_joint_index, RootMotionDelta};
pub use skeleton::{Joint, JointTransform, Skeleton, SkeletonError};
pub use sockets::{find_socket, socket_transform, socket_transforms, Socket};
pub use state_machine::{
    eval_condition, eval_pose, motion_period, sample_motion, step, BlendCurve, BlendProfile, CmpOp,
    InterruptBlend, InterruptSource, JointBlendWeight, Motion, SmCompare, SmCond, SmContext,
    SmError, SmInterrupt, SmParam, SmParamKind, SmRuntime, SmSource, SmState, SmStep, SmSub,
    SmTransition, SmValue, StateMachine, MAX_COND_DEPTH, MAX_COND_NODES, MAX_PARAMS,
};
pub use template::{
    build_template, girdle_name, leg_suffix, BodyParams, BodyPlan, JointLimit, TemplateError,
    MAX_LEGS,
};

/// `v > 0.0`, written once so the **NaN-rejecting** form reads as intent rather
/// than as a negated comparison.
///
/// Every duration guard in this crate goes through it: a clip whose length is
/// not a number must behave like a zero-length one (collapse to `t = 0`) rather
/// than like a positive one, and `!(v > 0.0)` is the only spelling that does
/// that — `v <= 0.0` is **false** for a NaN and lets it through to
/// `clamp(0.0, NaN)`, which panics.
///
/// Naming it keeps the meaning visible and keeps clippy's
/// `neg_cmp_op_on_partial_ord` from being suppressed one `allow` at a time —
/// the discipline `inf_pcg::grammar::span::positive` and this crate's own
/// `ik::usable_length` already follow.
#[inline]
pub(crate) fn positive(v: f32) -> bool {
    v > 0.0
}

/// [`positive`] for the `f64` half of the same integration
/// ([`advance_clip_time`], which the runtime and editor Simulate ticks share).
#[inline]
pub(crate) fn positive64(v: f64) -> bool {
    v > 0.0
}
