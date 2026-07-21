//! Animation: skeletal runtime, clips, blend spaces, state machines.
//!
//! **P11.1 skeletal foundation** — this crate is Ring 0 and **pure**: a
//! deterministic pose pipeline (skeleton → clip sample → local pose → global
//! transforms → GPU skinning palette) with no rendering, ECS, or editor
//! dependencies. Pose math is `f32` on purpose (see [`skeleton`] for the f32 vs.
//! f64 rationale). The `.inf_skel` / `.inf_anim` asset payloads live in
//! [`asset`].
//!
//! ## Seams left for later P11 sub-phases
//! * **Blend spaces (P11.2)** — [`pose::blend_poses`] is the two-pose primitive a
//!   1D/2D blend space composes; multi-pose weighted blends layer on top.
//! * **State machines (P11.2, `.inf_sm`)** — states hold clip refs + blend
//!   weights; the interpreter drives [`pose::sample_clip`] + [`pose::blend_poses`].
//! * **Sockets / retarget (P11.3)** — [`skeleton::Joint::name`] is retained per
//!   joint precisely so sockets attach by name and retargeting maps by name.
//! * **Cubic tracks** — [`clip::Interpolation`] keeps `Step`/`Linear`; cubic is
//!   resampled to linear on import (documented in [`clip`]).

pub mod asset;
pub mod clip;
pub mod pose;
pub mod skeleton;

pub use asset::{AnimClipAsset, SkeletonAsset};
pub use clip::{AnimClip, Interpolation, JointTrack, QuatTrack, Vec3Track};
pub use pose::{
    advance_clip_time, blend_poses, global_transforms, sample_clip, skinning_matrices, Pose,
};
pub use skeleton::{Joint, JointTransform, Skeleton, SkeletonError};
