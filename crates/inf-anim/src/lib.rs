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
// P24.2 inverse kinematics: the post-pass over an evaluated pose.
pub mod ik;
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
pub use ik::{
    fabrik, rotation_between, solve_chain, two_bone_positions, IkError, IkReport,
    FABRIK_ITERATIONS, MIN_BONE_LENGTH_M, REACH_TOLERANCE_M,
};
pub use pose::{
    advance_clip_time, blend_poses, global_transforms, sample_clip, skinning_matrices, Pose,
};
pub use retarget::{humanoid_joint_names, retarget_pose, RetargetMap};
pub use root_motion::{root_delta, root_joint_index, RootMotionDelta};
pub use skeleton::{Joint, JointTransform, Skeleton, SkeletonError};
pub use sockets::{find_socket, socket_transform, socket_transforms, Socket};
pub use state_machine::{
    eval_pose, sample_motion, step, CmpOp, Motion, SmCondition, SmContext, SmRuntime, SmState,
    SmTransition, StateMachine,
};
pub use template::{build_template, BodyParams, BodyPlan, JointLimit, TemplateError, MAX_LEGS};
