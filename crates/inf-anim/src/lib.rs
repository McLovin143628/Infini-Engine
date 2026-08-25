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
// P29.2 `.inf_anim` v2: the clip channel model — named curves, timed markers,
// the additive reference, root motion (Y included) and the distance track.
pub mod channels;
pub mod clip;
// P29.2 blend spaces: the deterministic triangulator that closes the P11.2
// IDW-k3 deferral.
pub mod delaunay;
// P29.5 pillar S2: what Epic's seven hand-run AnimModifiers bake, derived from
// the clip at import instead — root motion, distance, foot plants, gait curves.
pub mod derive;
// P24.4 secondary animation: XPBD cloth over the posed skeleton.
pub mod cloth;
// P24.4 secondary animation: strand hair, on the same solver primitives.
pub mod hair;
// P29.4 foot IK + foot locking: the pure half of the ground contact, and the
// lock whose slide the wave's gate measures in metres.
pub mod foot;
// SK1a the procedural drive pass: the bones a clip never authors — twist
// extraction and the IK handles' FK follow.
pub mod drive;
// P24.2 inverse kinematics: the post-pass over an evaluated pose.
pub mod ik;
// P29.2 inertialization: the quintic decay of a pose deviation, and the blender
// that makes it the default for state transitions.
pub mod inertialize;
// P29.2 blending depth: additive poses, per-bone masks and the layer stack.
pub mod layers;
// P24.5 the default locomotion set: clips and a machine derived from a body plan.
pub mod locomotion;
// SK1a the 161-bone UE5 mannequin hierarchy — `BodyPlan::Biped`'s rig.
pub mod manny;
// P24.3 modular rigging: assembling one skeleton out of parts.
pub mod merge;
pub mod pose;
// P29.5 pillar S3: a state graph proposed from a derived clip set, written as a
// normal text-diffable `.inf_sm` the author edits.
pub mod propose;
// P29.2 pose snapshot + pose matching (P29.4's get-up and landing consumers).
pub mod pose_match;
// P29.4 the ragdoll bridge's animation half: the motor drive, the face-up read
// and the blend weight that is a pure function of sim state.
pub mod ragdoll;
// SK1a the rig's side tables: what each bone IS, what drives it, where it grips.
pub mod roles;
pub mod skeleton;
pub mod state_machine;
// P29.2 sync groups: marker-phase warping so a walk↔run blend keeps its feet.
pub mod sync;
// P11.3 character tools (sockets / root motion / retarget) — new modules, each
// pure like the rest of the crate.
pub mod retarget;
pub mod root_motion;
pub mod sockets;
// P24.1 template body plans: the parametric N-pedal skeleton generator.
pub mod template;
// P29.6 pillar S1: the `.inf_sm` TEXT form — the reviewable face of a machine,
// and the substrate `phase29_gate`'s one-line-diff arm measures.
pub mod text;
// P29.4 motion warping: warp windows scaling root motion onto a runtime target,
// plus distance matching and orientation warping.
pub mod warp;

pub use asset::{AnimClipAsset, SkeletonAsset, StateMachineAsset};
pub use blend_space::{
    blend_leader, blend_weights_1d, blend_weights_2d, sample_blend_space_1d, sample_blend_space_2d,
    weights_2d, BlendEntry1D, BlendEntry2D, BlendSpace1D, BlendSpace2D, ClipRef,
};
pub use channels::{AdditiveRef, AnimMarker, CurveChannel, DistanceTrack, RootMotionTrack};
pub use clip::{AnimClip, Interpolation, JointTrack, QuatTrack, Vec3Track};
pub use cloth::{
    body_capsules, capsules_for, step_cloth, Capsule, ClothAsset, ClothCapsule, ClothEdge,
    ClothError, ClothMaterial, ClothState, GRAVITY_M_S2,
};
pub use delaunay::{barycentric, triangulate, Triangulation};
pub use derive::{
    derive_clip, foot_joints, gait_of, speed_of_gait, unbake_root_motion, DeriveError,
    DeriveOptions, DeriveReport, DerivedNames, FootPlant, VerticalPolicy, FOOTSTEP_PREFIX,
};
pub use drive::{drive_ik_follow, drive_pose, drive_twists, twist_about};
pub use foot::{
    ground_offset, interp_to, pelvis_offset, FootLock, GroundOffset, FOOT_HEIGHT_M, TRACE_ABOVE_M,
    TRACE_BELOW_M,
};
pub use hair::{
    render_mesh, ribbon_mesh, roots_for, step_hair, HairAsset, HairDetail, HairGroom, HairMaterial,
    HairRoot, HairState, HairStrand,
};
pub use ik::{
    fabrik, rotation_between, solve_chain, two_bone_positions, IkError, IkReport,
    FABRIK_ITERATIONS, MIN_BONE_LENGTH_M, REACH_TOLERANCE_M,
};
pub use inertialize::{quintic_decay, Inertializer, PoseBlender, SmBlendMode, TransitionEntry};
pub use layers::{
    additive_base_pose, additive_delta, apply_additive, apply_layer, apply_layers,
    sample_additive_clip, AnimLayer, JointMask, LayerMode,
};
pub use locomotion::{
    build_locomotion, locomotion_machine, GaitParams, LegSummary, LocomotionError, LocomotionSet,
    FOOT_SYNC_GROUP, MAX_KEYS_PER_CYCLE, SPEED_VAR, STATE_NAMES,
};
pub use manny::{build_manny, MANNY_JOINT_COUNT};
pub use merge::{
    merge_skeletons, mirror_joint_map, mirrored_joint_name, unmatched_sided_joints, SkeletonMerge,
    SkeletonMergeError,
};
pub use pose::{
    advance_clip_time, blend_poses, blend_poses_weighted, global_transforms, sample_clip,
    skinning_matrices, Pose,
};
pub use pose_match::{
    match_clip, match_clips, pose_cost, PoseMatch, PoseMatchWeights, PoseSnapshot,
};
pub use propose::{
    facts_of, propose_machine, ClipFacts, Proposal, ProposalOptions, ProposeError, SPEED_PARAM,
};
pub use ragdoll::{
    blend_weight as ragdoll_blend_weight, face_up_from_pelvis_roll, motor_stiffness, GetUp,
    RagdollPhase,
};
pub use retarget::{
    humanoid_joint_names, retarget_pose, retarget_pose_reported, RetargetMap, RetargetReport,
};
pub use roles::{
    BoneRole, BoneRoleKind, BoneSide, GripAffordance, IkFollow, RoleIndex, TwistDriver,
};
pub use root_motion::{
    bake_root_motion, root_delta, root_delta_3d, root_delta_world, root_delta_world_3d,
    root_joint_index, RootMotion3D, RootMotionDelta,
};
pub use skeleton::{Joint, JointTransform, Skeleton, SkeletonError};
pub use sockets::{find_socket, socket_transform, socket_transforms, Socket};
pub use state_machine::{
    eval_condition, eval_pose, motion_leader, motion_period, sample_motion, step, BlendCurve,
    BlendProfile, CmpOp, InterruptBlend, InterruptSource, JointBlendWeight, Motion, SmCompare,
    SmCond, SmContext, SmError, SmInterrupt, SmParam, SmParamKind, SmRuntime, SmSource, SmState,
    SmStep, SmSub, SmTransition, SmValue, StateMachine, MAX_COND_DEPTH, MAX_COND_NODES, MAX_PARAMS,
};
pub use sync::{common_group, leader_index, warped_times, SyncPhase, SyncTrack};
pub use template::{
    build_template, girdle_name, leg_suffix, BodyParams, BodyPlan, ConeLimit, JointLimit,
    TemplateError, MAX_LEGS,
};
pub use text::{cond_text, from_toml, parse_cond, to_toml, TextError};
pub use warp::{
    distance_match, height_remap, play_rate_for, warp_ease, warp_offset, warp_yaw_deg, HeightRemap,
    WarpWindow, MANTLE_HIGH_SPLIT_M,
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

/// `a > b`, named for the same reason [`positive`] is: the crate needs the
/// **NaN-rejecting** negation `!greater(a, b)` in several places, and clippy's
/// `neg_cmp_op_on_partial_ord` is right that `!(a > b)` reads badly — but wrong
/// that the fix is `partial_cmp`, because `partial_cmp` returns `None` for a NaN
/// and every one of these sites wants the NaN on the refusing side.
///
/// Naming it keeps the meaning visible and keeps the lint from being suppressed
/// one `allow` at a time, which is the discipline `positive` already follows.
#[inline]
pub(crate) fn greater(a: f32, b: f32) -> bool {
    a > b
}

/// `a <= b`, for the same reason as [`greater`]. `!at_most(a, b)` is "not
/// non-decreasing, NaN included" — the question a monotonicity check has to ask,
/// and the one `a > b` answers wrongly for a NaN.
#[inline]
pub(crate) fn at_most(a: f32, b: f32) -> bool {
    a <= b
}
