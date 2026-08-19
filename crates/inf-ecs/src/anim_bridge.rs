//! **The animation bridge** (P29.4) — the Ring-0 doors the `anim.*` Blueprint kit
//! and the movement step go through, and the one resource they share.
//!
//! # What a bridge is for
//!
//! Before this wave the animation state machine could only be driven by the
//! actor's Blueprint **variables**: [`crate::pose::step_pose_evaluation`] resolves
//! every parameter through a `vars` closure each host builds out of its own actor
//! registry. That is a fine authoring path and a poor *gameplay* one — a system
//! that wants to say "this character just landed hard" has to know which actor
//! owns the character and what its variable is called, and a system that wants to
//! ask "is it still getting up?" has no route at all.
//!
//! So there are four questions, and they are the four nodes of the kit:
//!
//! * **set a parameter** — a named `f64` overlaid on the actor's variables;
//! * **set a trigger** — arm a declared [`inf_anim::SmParamKind::Trigger`] once,
//!   through [`inf_anim::SmRuntime::arm_trigger`] rather than by writing a level
//!   (see that method for why the difference matters);
//! * **query the state** — what the machine is in, and for how long;
//! * **consume a notify** — take one of this step's events, so exactly one
//!   consumer acts on it.
//!
//! # Why one resource and not five
//!
//! Every field here follows [`crate::pose::PoseStoreRes`]'s rules exactly — never
//! serialized, written only from the fixed step or from a gameplay door, dropped
//! by [`crate::pose::clear_poses`] — and they are all read inside the same loop of
//! the same function. Five resources would be five `get_resource` calls per step
//! and five things to remember to clear; one is one.
//!
//! It is **absent** until something writes it, so a level whose characters have no
//! bridge traffic is byte-identical to its pre-P29.4 self.
//!
//! # Determinism
//!
//! `BTreeMap` / `BTreeSet` throughout, for the reason every other sim-side map in
//! this crate is: the iteration order reaches a pose, and a pose reaches
//! `state_bytes`.

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::Resource;
use inf_anim::RootMotion3D;
use uuid::Uuid;

use crate::components::AnimStateMachine;
use crate::math::Vec3d;
use crate::world::EcsWorld;

/// One bone of a character's rig, in **world space** — a name and the segment it
/// spans (P29.4, clause 6).
///
/// Deliberately a plain value type in this crate rather than
/// `inf_physics::ragdoll::RagdollBone`: `inf-ecs` does not depend on `inf-physics`
/// (the direction is the other way round), and the whole point of the seam is
/// that neither side names the other's types. The physics side maps this onto its
/// own descriptor in one line.
#[derive(Clone, Debug, PartialEq)]
pub struct RigBone {
    /// The joint's name — what `inf_physics::ragdoll::classify` reads.
    pub name: String,
    /// The parent-facing end of the segment, world metres.
    pub head: Vec3d,
    /// The far end.
    pub tail: Vec3d,
}

/// One foot, as the pose left it (P29.4, clause 5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FootState {
    /// The joint index the foot is, so the IK pass can walk up to a chain
    /// without matching names a second time.
    pub joint: u16,
    /// Where the pose put it, world metres.
    pub world: Vec3d,
}

/// Where a foot must be put, and how much of the way (P29.4, clause 5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FootGoal {
    /// The world-space target.
    pub target: Vec3d,
    /// How much of the solve to apply, `[0, 1]` — the `Enable_FootIK` curve, so
    /// a state that wants no foot IK authors a zero and gets the pose untouched.
    pub weight: f32,
}

/// How many points a [`TraversalArc`] is resampled onto.
///
/// Odd on purpose, so the midpoint of the clip is a sample and not an
/// interpolation between two — a traversal clip's most characteristic frame (the
/// hands on the ledge) sits there far more often than anywhere else.
pub const TRAVERSAL_ARC_SAMPLES: usize = 33;

/// **The arc a one-shot clip's root motion draws**, resampled onto a fixed grid
/// of normalized clip positions (P29.5).
///
/// This is what closes P29.4's ledgered "the mantle's progress is a clock"
/// remainder. That wave's warp was *called* with the clip's `(delivered, total)`
/// pair and, with no traversal clip bound, had to synthesise them as
/// `total × progress` and `total` — a scale of exactly one, so the warp
/// degenerated to its own ease. With a clip's root motion baked at import
/// ([`inf_anim::derive`]) the pair is real, and the same call at the same site
/// scales the clip's own arc onto the runtime target.
///
/// # Why an arc and not a delta
///
/// The movement step needs "where would this clip be at *my* progress", and its
/// progress is its own clock — a mantle's duration is derived from the ledge
/// height and the play rate, not from the machine's play-head. A per-step delta
/// cannot answer that; a resampled arc can, at [`TRAVERSAL_ARC_SAMPLES`] points
/// and one lerp.
///
/// # Why only one-shots
///
/// After P29.5 *every* imported clip carries a root-motion track, so publishing
/// an arc for all of them would put 33 samples per character per fixed step on a
/// level full of walking NPCs for nothing. A gait's arc is not a traversal arc:
/// a walk cycle's root motion is consumed continuously by the movement step and
/// never warped onto a target. The pose step therefore publishes this only for a
/// state whose `looping` is **false**, which is what a traversal state is.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TraversalArc {
    /// Root translation at each grid point, clip space, metres, **Y included**.
    /// Index 0 is the clip's first frame; the last index is its last.
    pub samples: Vec<glam::Vec3>,
    /// Root yaw at each grid point, radians, index-aligned with `samples`.
    pub yaw_rad: Vec<f32>,
}

impl TraversalArc {
    /// Whether the arc has enough points to be read.
    pub fn is_usable(&self) -> bool {
        self.samples.len() >= 2 && self.samples.len() == self.yaw_rad.len()
    }

    /// The motion delivered from the clip's first frame to normalized position
    /// `alpha`, and the yaw with it. `(ZERO, 0)` for an unusable arc — a value,
    /// so a caller reads a shape it can scale rather than a `None` it has to
    /// branch on twice.
    pub fn at(&self, alpha: f64) -> (glam::Vec3, f32) {
        if !self.is_usable() {
            return (glam::Vec3::ZERO, 0.0);
        }
        let a = if alpha.is_finite() {
            alpha.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let last = self.samples.len() - 1;
        let x = a * last as f64;
        let i0 = (x.floor() as usize).min(last);
        let i1 = (i0 + 1).min(last);
        let f = (x - i0 as f64) as f32;
        let p = self.samples[i0] + (self.samples[i1] - self.samples[i0]) * f;
        let y = self.yaw_rad[i0] + (self.yaw_rad[i1] - self.yaw_rad[i0]) * f;
        (p - self.samples[0], y - self.yaw_rad[0])
    }

    /// What the whole clip produces — [`at`](Self::at) at the end.
    pub fn total(&self) -> (glam::Vec3, f32) {
        self.at(1.0)
    }
}

/// What one entity's machine is doing, published every fixed step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnimStateInfo {
    /// Index into the machine's `states`.
    pub index: usize,
    /// The state's authored name — what `anim.query_state` compares against.
    pub name: String,
    /// Seconds spent in this state.
    pub time_s: f64,
    /// Whether a transition is still blending.
    pub blending: bool,
}

/// The bridge's shared state — see the module docs.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct AnimBridgeRes {
    /// Parameters set through [`set_anim_param`], overlaid on the actor's
    /// Blueprint variables. **The overlay wins**: a name set here shadows a
    /// variable of the same name, because a gameplay system that has just said
    /// "landed hard" must not be silently outvoted by a stale variable.
    pub params: BTreeMap<Uuid, BTreeMap<String, f64>>,
    /// Triggers armed through [`set_anim_trigger`] and not yet handed to a
    /// machine. Drained by the pose step on the very next fixed step.
    pub triggers: BTreeMap<Uuid, BTreeSet<String>>,
    /// What each machine is in, published by the pose step.
    pub states: BTreeMap<Uuid, AnimStateInfo>,
    /// The root motion each machine-driven character's current state produced
    /// this step (clause 2). Published by the pose step; consumed by
    /// `inf_physics::d3::movement`.
    pub root_motion: BTreeMap<Uuid, RootMotion3D>,
    /// **The `.inf_anim` v2 curve channels** of the state each machine is in,
    /// sampled at its play-head (clause 5 and clause 7).
    ///
    /// The reason the movement step can read `Enable_FootIK_L`, `FootLock_R` and
    /// `Mask_AimOffset` at all: those are properties of a *clip*, the clip
    /// resolver is a host registry, and the pose step is the one place that has
    /// both it and the play-head. Published rather than queried, which is the
    /// command-queue shape — the physics side never reaches into the machine.
    pub curves: BTreeMap<Uuid, BTreeMap<String, f32>>,
    /// **Where the pose put each foot**, world space, published every step for
    /// every rig that has feet (clause 5).
    ///
    /// `[left, right]`. The movement step reads it to know where to probe for
    /// the ground; it cannot derive it itself, because a foot's position is a
    /// pose and the pose lives on the other side of the seam.
    pub feet: BTreeMap<Uuid, [Option<FootState>; 2]>,
    /// **Where each foot must go**, world space — the movement step's answer,
    /// consumed by the pose step's IK pass in the SAME fixed step (the movement
    /// step runs first in both hosts).
    pub foot_ik: BTreeMap<Uuid, [Option<FootGoal>; 2]>,
    /// Entities that have asked the pose step for their **ragdoll rig** and
    /// not yet been given one (clause 6).
    ///
    /// The request half of the command queue: the movement step knows a
    /// character has started ragdolling and has no skeleton; the pose step has
    /// the skeleton and no idea that it happened. Asking rather than computing
    /// every step is what keeps a level full of characters from paying for a
    /// mechanism none of them are using.
    pub ragdoll_requested: BTreeSet<Uuid>,
    /// The **world-space bone segments** the pose step derived for a requested
    /// ragdoll — the answer half.
    ///
    /// Published once per request and consumed by the physics side, which turns
    /// it into bodies and joints through `inf_physics::ragdoll::build_ragdoll`.
    /// Physics never reaches into the machine and the machine never reaches into
    /// physics; the P12 command-queue doctrine, applied to a rig.
    pub ragdoll_rig: BTreeMap<Uuid, Vec<RigBone>>,
    /// Entities whose transitions enter at the **pose-matched** frame
    /// ([`inf_anim::TransitionEntry::PoseMatched`]) rather than at zero.
    ///
    /// P29.2 shipped that entry mode off by default because it changes which
    /// frame plays, and named this wave's get-up and landing consumers as the
    /// ones that turn it on. This set is how they do.
    pub pose_matched: BTreeSet<Uuid>,
    /// **The traversal arc** of the one-shot clip each machine is playing, when
    /// it is playing one that carries baked root motion (P29.5). See
    /// [`TraversalArc`] for what it is and why only one-shots have one.
    pub traversal: BTreeMap<Uuid, TraversalArc>,
}

impl AnimBridgeRes {
    /// Whether nothing at all is stored — the condition under which the pose step
    /// removes the resource rather than writing an empty one back.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
            && self.triggers.is_empty()
            && self.states.is_empty()
            && self.root_motion.is_empty()
            && self.curves.is_empty()
            && self.feet.is_empty()
            && self.foot_ik.is_empty()
            && self.ragdoll_requested.is_empty()
            && self.ragdoll_rig.is_empty()
            && self.pose_matched.is_empty()
            && self.traversal.is_empty()
    }
}

/// Whether `guid` names an entity that carries an [`AnimStateMachine`] at all.
///
/// The question every write door asks first: writing a parameter for an entity
/// with no machine is not an error worth failing a handler over, but it is worth
/// reporting, and it is the difference between "the node did nothing" and "the
/// node did nothing **and told you**".
fn has_machine(world: &EcsWorld, guid: Uuid) -> bool {
    world
        .entity_of(guid)
        .and_then(|e| world.world().get::<AnimStateMachine>(e))
        .is_some()
}

/// Read-modify-write the bridge resource, creating it if absent.
fn with_bridge<R>(world: &mut EcsWorld, f: impl FnOnce(&mut AnimBridgeRes) -> R) -> R {
    let w = world.world_mut();
    if !w.contains_resource::<AnimBridgeRes>() {
        w.insert_resource(AnimBridgeRes::default());
    }
    let mut res = w.resource_mut::<AnimBridgeRes>();
    f(&mut res)
}

/// The bridge resource, if this world has one.
pub fn bridge(world: &EcsWorld) -> Option<&AnimBridgeRes> {
    world.world().get_resource::<AnimBridgeRes>()
}

/// **Set an animation parameter** on `guid`'s machine.
///
/// Returns `false` when the entity has no machine or the value is not finite —
/// both **values**, not failures: a NaN parameter would reach a condition compare
/// and a blend-space axis, and refusing it here is cheaper than a pose nobody can
/// explain.
pub fn set_anim_param(world: &mut EcsWorld, guid: Uuid, name: &str, value: f64) -> bool {
    if name.is_empty() || !value.is_finite() || !has_machine(world, guid) {
        return false;
    }
    with_bridge(world, |b| {
        b.params
            .entry(guid)
            .or_default()
            .insert(name.to_string(), value);
    });
    true
}

/// The parameter last set through [`set_anim_param`], if any.
///
/// Deliberately **not** "the value the machine read": the machine's own answer is
/// the overlay *or* the actor's variable, and a caller asking this question wants
/// to know what it set.
pub fn anim_param(world: &EcsWorld, guid: Uuid, name: &str) -> Option<f64> {
    bridge(world)?.params.get(&guid)?.get(name).copied()
}

/// **Arm a trigger** on `guid`'s machine, consumed by the next fixed step.
///
/// Returns `false` for an entity with no machine. It does **not** report whether
/// the machine declares a trigger by that name — that is not knowable here (the
/// `.inf_sm` lives in the host's registry) and it is knowable one step later, in
/// the pose step, which is where the arm actually happens.
pub fn set_anim_trigger(world: &mut EcsWorld, guid: Uuid, name: &str) -> bool {
    if name.is_empty() || !has_machine(world, guid) {
        return false;
    }
    with_bridge(world, |b| {
        b.triggers.entry(guid).or_default().insert(name.to_string());
    });
    true
}

/// What `guid`'s machine is in, as of the last fixed step.
pub fn anim_state(world: &EcsWorld, guid: Uuid) -> Option<&AnimStateInfo> {
    bridge(world)?.states.get(&guid)
}

/// Whether `guid`'s machine is in the state named `name`.
///
/// `false` for an entity that has never stepped, which is the conservative
/// reading: a state nothing has entered is not a state anything is in.
pub fn anim_state_is(world: &EcsWorld, guid: Uuid, name: &str) -> bool {
    anim_state(world, guid).is_some_and(|s| s.name == name)
}

/// How long `guid`'s machine has been in its current state, seconds; `0` when it
/// has never stepped.
pub fn anim_state_time(world: &EcsWorld, guid: Uuid) -> f64 {
    anim_state(world, guid).map(|s| s.time_s).unwrap_or(0.0)
}

/// **Consume one of this step's notifies.**
///
/// Returns `true` exactly once per fired name per step: the name is *removed*
/// from [`crate::pose::AnimEventsRes`], so two handlers racing for one footstep
/// get one footstep. That is the difference between this and
/// [`crate::pose::anim_events`], which reads without taking.
///
/// A name that did not fire answers `false` and changes nothing.
pub fn consume_anim_notify(world: &mut EcsWorld, guid: Uuid, name: &str) -> bool {
    let w = world.world_mut();
    let Some(mut res) = w.get_resource_mut::<crate::pose::AnimEventsRes>() else {
        return false;
    };
    let Some(list) = res.0.get_mut(&guid) else {
        return false;
    };
    let Some(i) = list.iter().position(|n| n == name) else {
        return false;
    };
    list.remove(i);
    if list.is_empty() {
        res.0.remove(&guid);
    }
    let empty = res.0.is_empty();
    if empty {
        w.remove_resource::<crate::pose::AnimEventsRes>();
    }
    true
}

/// Turn [`inf_anim::TransitionEntry::PoseMatched`] on or off for `guid`.
///
/// The door P29.2 named and did not build a caller for: "off by default because
/// it changes which frame plays, and that is a content decision; P29.4's get-up
/// and landing consumers turn it on."
pub fn set_pose_match_entry(world: &mut EcsWorld, guid: Uuid, on: bool) {
    with_bridge(world, |b| {
        if on {
            b.pose_matched.insert(guid);
        } else {
            b.pose_matched.remove(&guid);
        }
    });
}

/// The root motion `guid`'s machine produced on the **last** fixed step's pose
/// evaluation, in the character's own facing frame.
///
/// # One step of latency, stated
///
/// The pose is evaluated *after* the movement step in both hosts' fixed steps
/// (movement runs before the solver; the pose runs after the write-back), so the
/// movement step reads the root motion the pose step published one step earlier.
/// Over any interval the total displacement is the same, shifted by one step of
/// 1/60 s — the same one-beat latency P22.3's structural collapse documents, and
/// for the same structural reason. Moving the pose step in front of the movement
/// step would invert the dependency (the pose reads the mode the movement step
/// decides), which is worse.
pub fn anim_root_motion(world: &EcsWorld, guid: Uuid) -> Option<RootMotion3D> {
    bridge(world)?.root_motion.get(&guid).copied()
}

/// **The traversal arc** of the one-shot clip `guid`'s machine is playing, if it
/// is playing one that carries baked root motion (P29.5).
///
/// `None` is a value the mantle already knows what to do with: it falls back to
/// its own clock, which is exactly the pre-P29.5 behaviour, so a project with no
/// traversal content behaves as it did.
pub fn traversal_arc(world: &EcsWorld, guid: Uuid) -> Option<&TraversalArc> {
    bridge(world)?.traversal.get(&guid)
}

/// The value of `guid`'s current clip curve `name` at this step's play-head, or
/// `fallback` when the clip has no such channel.
///
/// The reference vocabulary is [`inf_anim::channels::als`]; the names are
/// free-form `String`s by Ruling 2, so a studio's own channel reads here exactly
/// like one of the twenty-five.
pub fn anim_curve(world: &EcsWorld, guid: Uuid, name: &str, fallback: f32) -> f32 {
    bridge(world)
        .and_then(|b| b.curves.get(&guid))
        .and_then(|c| c.get(name))
        .copied()
        .unwrap_or(fallback)
}

/// A footstep the animation asked for this fixed step (P29.4, clause 7).
#[derive(Clone, Debug, PartialEq)]
pub struct FootstepCue {
    /// The character whose foot landed — and, because a footstep is played on
    /// the character's own emitter, the audio source.
    pub source: Uuid,
    /// The notify's name, so a project can tell left from right (or a surface
    /// from a surface) without a second channel.
    pub name: String,
    /// How loud, `[0, 1]` — ALS's `Mask_FootstepSound`, which is a *scale* an
    /// animator authors on the clip so a crouched walk is quieter than a sprint.
    pub gain: f64,
}

/// The name a notify must start with to be a footstep.
///
/// A prefix rather than an exact match, because `footstep_l` and `footstep_r`
/// are two notifies and one sound, and a project that adds `footstep_land` gets
/// it for free.
///
/// **One spelling, not two.** P29.5 gave this prefix a *producer*
/// ([`inf_anim::derive`] writes the event markers a clip's own feet imply), and
/// a producer and a consumer that each spell a wire string in their own crate is
/// how a mechanism goes quiet without anything failing. It is defined where the
/// markers are written and re-exported here, where they are heard.
pub const FOOTSTEP_PREFIX: &str = inf_anim::derive::FOOTSTEP_PREFIX;

/// **The footstep cues this fixed step produced** — a pure function of sim state.
///
/// The P12 doctrine's sentence for audio is that the stream is a pure function of
/// the simulation, and this is where that is true for footsteps: the notifies are
/// what the pose step published (a property of the step history), the gain is a
/// curve channel on the clip that is playing, and the order is `Guid` order
/// followed by the order the machine fired them in. Two identical worlds produce
/// identical lists, which `the_footstep_cues_are_a_pure_function_of_sim_state`
/// asserts rather than describes.
///
/// The cues are **read, not taken**: a caller that wants exactly-once semantics
/// uses [`consume_anim_notify`], which is what the `anim.*` kit's node does. This
/// is the host's audio drain, and a host drains once per step by construction.
pub fn footstep_cues(world: &EcsWorld) -> Vec<FootstepCue> {
    let Some(events) = world.world().get_resource::<crate::pose::AnimEventsRes>() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (guid, names) in &events.0 {
        for name in names {
            if !name.starts_with(FOOTSTEP_PREFIX) {
                continue;
            }
            let gain = anim_curve(
                world,
                *guid,
                inf_anim::channels::als::MASK_FOOTSTEP_SOUND,
                1.0,
            ) as f64;
            out.push(FootstepCue {
                source: *guid,
                name: name.clone(),
                gain: gain.clamp(0.0, 1.0),
            });
        }
    }
    out
}

/// Where the pose put `guid`'s feet last step, world space.
pub fn feet_of(world: &EcsWorld, guid: Uuid) -> Option<[Option<FootState>; 2]> {
    bridge(world)?.feet.get(&guid).copied()
}

/// **Set this step's foot-IK goals** for `guid`, world space.
///
/// Consumed by the pose step's IK pass in the same fixed step, and replaced
/// every step: a goal is a statement about where the ground is *now*.
pub fn set_foot_ik(world: &mut EcsWorld, guid: Uuid, goals: [Option<FootGoal>; 2]) {
    with_bridge(world, |b| {
        if goals.iter().all(Option::is_none) {
            b.foot_ik.remove(&guid);
        } else {
            b.foot_ik.insert(guid, goals);
        }
    });
}

/// **Ask the pose step for `guid`'s ragdoll rig.**
///
/// The physics side calls this the step a ragdoll starts; the answer arrives on
/// the next fixed step, which is the same one beat of latency
/// [`anim_root_motion`] documents and for the same structural reason.
pub fn request_ragdoll_rig(world: &mut EcsWorld, guid: Uuid) {
    with_bridge(world, |b| {
        b.ragdoll_requested.insert(guid);
    });
}

/// The world-space rig the pose step published for `guid`, if it has.
pub fn ragdoll_rig(world: &EcsWorld, guid: Uuid) -> Option<&[RigBone]> {
    bridge(world)?.ragdoll_rig.get(&guid).map(Vec::as_slice)
}

/// **Take** the rig, so the bodies are built from it exactly once.
pub fn take_ragdoll_rig(world: &mut EcsWorld, guid: Uuid) -> Option<Vec<RigBone>> {
    let w = world.world_mut();
    let mut res = w.get_resource_mut::<AnimBridgeRes>()?;
    res.ragdoll_rig.remove(&guid)
}

/// **Forget every bridge entry.** Called by [`crate::pose::clear_poses`], which is
/// the one door that forgets a play session's animation state.
pub fn clear_anim_bridge(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<AnimBridgeRes>();
}

/// **The parameter names a character publishes into its own machine** (P29.6).
///
/// ALS's AnimInstance copies seventeen fields off the character every tick and
/// the AnimBP reads them by name; this is the same idea with the name set
/// written down once, so a proposed machine, a wizard-generated one and a
/// hand-authored one all read the same words for the same numbers.
///
/// `SPEED` is `inf_anim::locomotion::SPEED_VAR` by construction — the
/// generator has used that name since P24.5 and a second spelling would leave
/// every wizard character standing still. The arm in this module holds the two
/// together, because `inf-ecs` cannot reach for the constant without the
/// dependency edge it deliberately does not want on the *name* side.
pub mod params {
    /// Planar ground speed, m/s.
    pub const SPEED: &str = "speed";
    /// The normalized 0–3 gait scale — 0 stopped, 1 walk, 2 run, 3 sprint.
    /// ALS's `GetMappedSpeed`, and the X axis of every movement curve.
    pub const GAIT: &str = "gait";
    /// `1` when the last sweep ended on the ground, `0` otherwise.
    pub const GROUNDED: &str = "grounded";
    /// The [`crate::components::MovementMode`] discriminant, as a number — so a
    /// machine can say `mode == 4` for a slide without the engine growing a
    /// typed parameter kind for enums.
    pub const MODE: &str = "mode";
    /// The [`crate::components::MovementDirection`] discriminant.
    pub const DIRECTION: &str = "direction";
    /// Downward speed, m/s, positive falling — `0` while grounded.
    pub const FALL_SPEED: &str = "fall_speed";
    /// How close a predicted landing is, `[0, 1]`.
    pub const LAND_ALPHA: &str = "land_alpha";
    /// The **overlay id** this character is wearing — `0` is the default overlay
    /// and every other value is `crate::movement::OverlayRegistry`'s interning of
    /// the scene's own string (Ruling 4's "open interned id", given its first
    /// caller in P29.6).
    pub const OVERLAY: &str = "overlay";
    /// The flail blend `[0, 1]` a falling ragdoll gets, from
    /// [`inf_anim::ragdoll::flail_rate`].
    pub const FLAIL: &str = "flail";
}

/// **Publish a character's movement state into its machine's parameters**
/// (P29.6) — the door that makes a wizard-generated character animate with no
/// script at all.
///
/// Before this, `speed` was a parameter every generated and proposed machine
/// gated on and **nothing in the engine ever set**: the only writer was
/// `anim.set_param`, which is a Blueprint's door. So the wizard's own character
/// stood in its idle state for ever unless somebody wrote a program to tell it
/// how fast it was going — and the number it needed was already on the movement
/// runtime, one crate away.
///
/// # The precedence, and it is the other way round (P29.6 audit, A11)
///
/// This writes into the same overlay `anim.set_param` writes into, and the last
/// writer in a fixed step wins. Both hosts dispatch `EventKind::Tick` and *then*
/// run `step_character_movement`, so **this** lands last: on a character with a
/// [`CharacterMovement`](crate::components::CharacterMovement) the engine owns
/// the nine names in [`params`], and a Blueprint that sets one of them is
/// overwritten before the pose step reads it.
///
/// That is the right design — one authority for one fact, and the number the
/// engine has is the true one — and it is a real constraint: a game that wants
/// to drive `speed` itself takes the movement component off, which is exactly
/// what `phase24_wizard`'s fixture does and says. The order it rests on is
/// pinned in both hosts by `projector_mirror`'s
/// `both_fixed_steps_publish_character_params_after_the_blueprint_tick`, and the
/// consequence is pinned by `live_tuning`.
///
/// The first two write-ups of this paragraph — here and at the call site —
/// claimed a Blueprint still wins. It does not, and nothing asserted either way.
///
/// Returns whether anything was published. `false` for an entity with no
/// machine, which is most characters, and the check is the first thing that
/// happens so the cost on those is one map lookup.
pub fn publish_character_params(
    world: &mut EcsWorld,
    guid: Uuid,
    cm: &crate::components::CharacterMovement,
    overlay: u32,
) -> bool {
    if !has_machine(world, guid) {
        return false;
    }
    let rt = &cm.runtime;
    let planar = (rt.velocity.x * rt.velocity.x + rt.velocity.z * rt.velocity.z).sqrt();
    // `flail_rate` gets its first caller here (a P29.4 audit zero-caller item):
    // how much limb-waving overlay a falling body wants, from its own speed.
    let flail = f64::from(inf_anim::ragdoll::flail_rate(
        rt.velocity.to_dvec3().length(),
    ));
    let values: [(&str, f64); 9] = [
        (params::SPEED, planar),
        (params::GAIT, rt.mapped_speed),
        (params::GROUNDED, f64::from(u8::from(rt.grounded))),
        (params::MODE, cm.mode as u8 as f64),
        (params::DIRECTION, rt.direction as u8 as f64),
        (params::FALL_SPEED, (-rt.velocity.y).max(0.0)),
        (params::LAND_ALPHA, rt.land_alpha),
        (params::OVERLAY, f64::from(overlay)),
        (params::FLAIL, flail),
    ];
    with_bridge(world, |b| {
        let slot = b.params.entry(guid).or_default();
        for (name, value) in values {
            // A non-finite number is DROPPED rather than published: the machine
            // would compare against it and every comparison a NaN takes part in
            // is false, which reads as "no transition is ready" for ever.
            if value.is_finite() {
                slot.insert(name.to_string(), value);
            }
        }
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Guid;

    fn world_with_machine() -> (EcsWorld, Uuid, Uuid) {
        let mut w = EcsWorld::new();
        let e = w.spawn("rig", None);
        let plain = w.spawn("prop", None);
        let guid = w.world().get::<Guid>(e).unwrap().0;
        let other = w.world().get::<Guid>(plain).unwrap().0;
        w.world_mut().entity_mut(e).insert(AnimStateMachine {
            sm: Some(Uuid::from_u128(9)),
            ..Default::default()
        });
        (w, guid, other)
    }

    #[test]
    fn a_write_door_refuses_by_value_and_says_so() {
        let (mut w, guid, other) = world_with_machine();
        assert!(set_anim_param(&mut w, guid, "speed", 3.5));
        assert_eq!(anim_param(&w, guid, "speed"), Some(3.5));
        // No machine on that entity.
        assert!(!set_anim_param(&mut w, other, "speed", 1.0));
        assert!(!set_anim_trigger(&mut w, other, "jump"));
        // A NaN never reaches a condition compare.
        assert!(!set_anim_param(&mut w, guid, "speed", f64::NAN));
        assert_eq!(
            anim_param(&w, guid, "speed"),
            Some(3.5),
            "the NaN did not land"
        );
        // An empty name is not a name.
        assert!(!set_anim_param(&mut w, guid, "", 1.0));
        // A world nobody has written has no resource at all.
        let clean = EcsWorld::new();
        assert!(bridge(&clean).is_none());
    }

    #[test]
    fn a_trigger_is_recorded_once_and_a_repeat_is_idempotent() {
        let (mut w, guid, _) = world_with_machine();
        assert!(set_anim_trigger(&mut w, guid, "jump"));
        assert!(set_anim_trigger(&mut w, guid, "jump"));
        assert!(set_anim_trigger(&mut w, guid, "roll"));
        let b = bridge(&w).unwrap();
        let names: Vec<&str> = b.triggers[&guid].iter().map(String::as_str).collect();
        assert_eq!(names, ["jump", "roll"], "sorted, because a BTreeSet is");
    }

    #[test]
    fn a_notify_is_consumed_exactly_once() {
        let (mut w, guid, _) = world_with_machine();
        w.world_mut().insert_resource(crate::pose::AnimEventsRes(
            [(guid, vec!["footstep_l".to_string(), "land".to_string()])]
                .into_iter()
                .collect(),
        ));
        assert!(consume_anim_notify(&mut w, guid, "footstep_l"));
        assert!(
            !consume_anim_notify(&mut w, guid, "footstep_l"),
            "twice is once"
        );
        assert_eq!(crate::pose::anim_events(&w, guid), ["land"]);
        assert!(consume_anim_notify(&mut w, guid, "land"));
        // The last one out removes the resource, so "nothing fired" has exactly
        // one representation.
        assert!(w
            .world()
            .get_resource::<crate::pose::AnimEventsRes>()
            .is_none());
        assert!(!consume_anim_notify(&mut w, guid, "land"));
    }

    #[test]
    fn the_pose_match_door_toggles_and_clearing_forgets_everything() {
        let (mut w, guid, _) = world_with_machine();
        set_pose_match_entry(&mut w, guid, true);
        assert!(bridge(&w).unwrap().pose_matched.contains(&guid));
        set_pose_match_entry(&mut w, guid, false);
        assert!(!bridge(&w).unwrap().pose_matched.contains(&guid));
        set_anim_param(&mut w, guid, "x", 1.0);
        clear_anim_bridge(&mut w);
        assert!(bridge(&w).is_none());
    }

    /// **`speed` is the name the generator has used since P24.5**, and a second
    /// spelling would leave every wizard character standing in its idle state.
    ///
    /// `inf-anim` owns the constant and `inf-ecs` owns the publisher, and the
    /// two are held together here because this crate can name both. It is an
    /// assertion rather than a `pub use` on purpose: the publisher's vocabulary
    /// is the ENGINE's, and a machine that a project authors by hand reads these
    /// words whether or not `inf_anim::locomotion` exists.
    #[test]
    fn the_published_speed_is_the_name_every_generated_machine_reads() {
        assert_eq!(params::SPEED, inf_anim::locomotion::SPEED_VAR);
    }

    /// The publisher writes every name it declares, and refuses an entity with
    /// no machine — the cheap path every character in every committed level
    /// before P29.6 takes.
    #[test]
    fn a_character_publishes_its_state_into_its_own_machine() {
        use crate::components::{AnimStateMachine, CharacterMovement, MovementMode};
        let mut w = EcsWorld::new();
        let guid = Uuid::from_u128(0x2906_1001);
        let bare = Uuid::from_u128(0x2906_1002);
        let e = w.spawn_with_guid(guid, "Hero", None);
        w.world_mut().entity_mut(e).insert(AnimStateMachine {
            sm: Some(Uuid::from_u128(9)),
            ..Default::default()
        });
        w.spawn_with_guid(bare, "Prop", None);

        let mut cm = CharacterMovement {
            mode: MovementMode::Crouch,
            ..Default::default()
        };
        cm.runtime.velocity = Vec3d::new(3.0, -4.0, 4.0);
        cm.runtime.mapped_speed = 1.75;
        cm.runtime.grounded = true;

        assert!(
            !publish_character_params(&mut w, bare, &cm, 0),
            "no machine"
        );
        assert!(publish_character_params(&mut w, guid, &cm, 0));
        // 3-4-5: the planar speed is 5, and the vertical is NOT in it.
        assert_eq!(anim_param(&w, guid, params::SPEED), Some(5.0));
        assert_eq!(anim_param(&w, guid, params::GAIT), Some(1.75));
        assert_eq!(anim_param(&w, guid, params::GROUNDED), Some(1.0));
        assert_eq!(
            anim_param(&w, guid, params::MODE),
            Some(MovementMode::Crouch as u8 as f64)
        );
        assert_eq!(anim_param(&w, guid, params::FALL_SPEED), Some(4.0));
        // Every declared name is written — a name that is declared and never
        // published is a machine gating on a parameter that stays at its default.
        for name in [
            params::SPEED,
            params::GAIT,
            params::GROUNDED,
            params::MODE,
            params::DIRECTION,
            params::FALL_SPEED,
            params::LAND_ALPHA,
            params::OVERLAY,
            params::FLAIL,
        ] {
            assert!(
                anim_param(&w, guid, name).is_some(),
                "`{name}` is declared and never published"
            );
        }
        // A Blueprint still wins: the kit writes into the same overlay.
        assert!(set_anim_param(&mut w, guid, params::SPEED, 99.0));
        assert_eq!(anim_param(&w, guid, params::SPEED), Some(99.0));
    }

    /// A non-finite value is **dropped**, not published: a machine comparing
    /// against a NaN reads every transition as not-ready, for ever.
    #[test]
    fn a_non_finite_movement_value_is_not_published() {
        use crate::components::{AnimStateMachine, CharacterMovement};
        let mut w = EcsWorld::new();
        let guid = Uuid::from_u128(0x2906_1003);
        let e = w.spawn_with_guid(guid, "Hero", None);
        w.world_mut().entity_mut(e).insert(AnimStateMachine {
            sm: Some(Uuid::from_u128(9)),
            ..Default::default()
        });
        let mut cm = CharacterMovement::default();
        cm.runtime.velocity = Vec3d::new(2.0, 0.0, 0.0);
        assert!(publish_character_params(&mut w, guid, &cm, 0));
        assert_eq!(anim_param(&w, guid, params::SPEED), Some(2.0));
        cm.runtime.velocity = Vec3d::new(f64::NAN, 0.0, 0.0);
        assert!(publish_character_params(&mut w, guid, &cm, 0));
        assert_eq!(
            anim_param(&w, guid, params::SPEED),
            Some(2.0),
            "a NaN overwrote the last good speed"
        );
    }
}
