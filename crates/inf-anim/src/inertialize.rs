//! Inertialization (P29.2 clause 5) — the smoothness stack's headline, and the
//! **default** for state transitions.
//!
//! §13's catalogue amendment: *"The single biggest transition-quality lever is
//! **inertialization** — decay the pose deviation with a quintic polynomial
//! instead of evaluating two animations and cross-fading them — which is
//! simultaneously cheaper and smoother than the cross-fade."* Both halves of that
//! sentence are claims, and both are measured in this module's tests rather than
//! asserted in prose.
//!
//! # The mechanism
//!
//! A cross-fade evaluates **both** states every frame of the transition and
//! interpolates. Inertialization evaluates only the **incoming** state, and adds
//! a decaying copy of the *difference* the outgoing pose had at the moment of the
//! switch:
//!
//! ```text
//! pose(t) = incoming(t) + y(t) · (outgoing(0) − incoming(0))
//! ```
//!
//! At `t = 0` that is exactly the outgoing pose — no pop — and at `t = T` it is
//! exactly the incoming one. In between it is one clip's evaluation plus a
//! per-joint add, which is where the "cheaper" comes from, and the shape of `y`
//! is where "smoother" comes from.
//!
//! # The quintic
//!
//! `y` is the standard inertialization curve (Bollo): the unique quintic with
//! `y(0) = 1`, `y′(0) = v₀`, and `y(T) = y′(T) = y″(T) = 0`. Landing with zero
//! first *and second* derivative is the point — the offset does not merely reach
//! zero, it arrives with no velocity and no acceleration, so nothing in the pose
//! changes direction at the end of a transition.
//!
//! `v₀` is what makes it more than a nice easing curve. The deviation is not
//! created at rest: just before the switch the rendered pose was moving at the
//! outgoing clip's rate, and afterwards the incoming clip supplies its own. Giving
//! `y` the deviation's own initial rate makes the *total* pose velocity continuous
//! across the switch, which no cross-fade curve can do — a cross-fade's velocity
//! jumps by the difference between the two clips' velocities however it is shaped.
//! With `v₀ = 0` (the honest default when there is no previous frame to difference)
//! the curve is `4t⁵ − 15t⁴ + 20t³ − 10t² + 1` on the unit interval, which is
//! strictly decreasing from 1 to 0 — asserted, because "the offset never grows"
//! is the property a consumer relies on.
//!
//! # Portability
//!
//! A polynomial, a `lerp`, and [`inf_math::pslerp`] from identity. The amendment
//! predicted this ("the polynomial form also keeps it inside the portable-math ban
//! list without a new helper") and it held: no transcendental, and the one `sqrt`
//! is in the *rate estimate*, which IEEE-754 specifies exactly.
//!
//! # What it costs to be `Copy`-free
//!
//! An [`Inertializer`] holds one deviation per joint, so it cannot live inside
//! [`SmRuntime`] — that struct is `Copy` because it is inlined into an ECS
//! component. It lives in [`PoseBlender`] instead, beside the previous frame's
//! pose, which the deviation is captured from. One consequence is a strict
//! improvement over the cross-fade path: because each capture is taken from *what
//! was on screen*, an interruption during a live inertialization inherits the
//! whole stack of earlier offsets automatically. The `Copy` runtime's carry slot
//! is one deep (P29.1 measured the second interruption at 40.5° of discontinuity);
//! this has no depth limit at all, because it never stores states — only the pose.

use serde::{Deserialize, Serialize};

use crate::blend_space::ClipRef;
use crate::layers::{additive_delta, apply_additive};
use crate::pose::Pose;
use crate::pose_match::{match_clip, PoseMatchWeights, PoseSnapshot};
use crate::skeleton::Skeleton;
use crate::state_machine::{eval_pose, Motion, SmContext, SmRuntime, SmStep, StateMachine};
use crate::AnimClip;

/// The inertialization curve `y(t)` — see the module docs.
///
/// `elapsed` and `duration` are seconds; `v0` is the deviation's initial rate
/// **normalized by its own magnitude**, s⁻¹, so the curve is dimensionless and one
/// scalar serves every joint and every channel.
///
/// Outside `[0, duration]` it is clamped to `1` and `0`; a non-positive or
/// non-finite `duration` is an instant switch (`0`).
pub fn quintic_decay(elapsed: f32, duration: f32, v0: f32) -> f32 {
    if !crate::positive(duration) {
        return 0.0;
    }
    if !crate::positive(elapsed) {
        return 1.0;
    }
    if elapsed >= duration {
        return 0.0;
    }
    // **The coefficients are formed in `f64` and the polynomial evaluated in it.**
    // Not a style choice: `a` is `O(1/T⁵)` and the five terms cancel to ~0 at
    // `t = T`, so in `f32` the residual at 249/250 of the way through a 0.25 s
    // decay was measured at −9.5e-7 — a *negative* decay weight, which inverts the
    // deviation instead of dropping it. `f64` is deterministic across targets for
    // the same reason `f32` is (these are add/mul, not libm), so the portability
    // law is untouched and the conditioning is six orders better.
    let v0 = if v0.is_finite() { v0 as f64 } else { 0.0 };
    let t1 = duration as f64;
    let (t2, t3, t4, t5) = (
        t1 * t1,
        t1 * t1 * t1,
        t1 * t1 * t1 * t1,
        t1 * t1 * t1 * t1 * t1,
    );
    // The classic coefficients with x0 normalized to 1.
    let a0 = (-8.0 * v0 * t1 - 20.0) / t2;
    let a = -(a0 * t2 + 6.0 * v0 * t1 + 12.0) / (2.0 * t5);
    let b = (3.0 * a0 * t2 + 16.0 * v0 * t1 + 30.0) / (2.0 * t4);
    let c = -(3.0 * a0 * t2 + 12.0 * v0 * t1 + 20.0) / (2.0 * t3);
    let t = elapsed as f64;
    let y =
        a * t * t * t * t * t + b * t * t * t * t + c * t * t * t + 0.5 * a0 * t * t + v0 * t + 1.0;
    if y.is_finite() {
        y as f32
    } else {
        0.0
    }
}

/// A captured pose deviation, decaying to zero.
#[derive(Clone, Debug, PartialEq)]
pub struct Inertializer {
    /// The deviation of the outgoing pose from the incoming one, in
    /// [`additive_delta`]'s shape.
    deviation: Pose,
    /// How long the decay lasts, seconds.
    pub duration_s: f32,
    /// How much of it has elapsed.
    pub elapsed_s: f32,
    /// The deviation's initial rate, normalized by its own magnitude (s⁻¹).
    pub v0: f32,
}

impl Inertializer {
    /// Capture the deviation of `outgoing` from `incoming`, decaying over
    /// `duration_s`, with no initial rate.
    pub fn capture(outgoing: &Pose, incoming: &Pose, duration_s: f32) -> Self {
        Self {
            deviation: additive_delta(incoming, outgoing),
            duration_s,
            elapsed_s: 0.0,
            v0: 0.0,
        }
    }

    /// Capture with the deviation's **initial rate** measured from the frame
    /// before — the form that makes the total pose velocity continuous.
    ///
    /// `outgoing_prev` is what was rendered one step (`dt` seconds) earlier. The
    /// rate is `(|d| − |d_prev|) / dt / |d|`, both deviations taken against the
    /// same `incoming` pose so the difference is the deviation's own motion and
    /// not the incoming clip's.
    ///
    /// A degenerate case — zero `dt`, a vanishing deviation, a non-finite result —
    /// falls back to `v0 = 0`, which is [`capture`](Self::capture) and is never
    /// worse than a cross-fade.
    pub fn capture_moving(
        outgoing: &Pose,
        outgoing_prev: &Pose,
        incoming: &Pose,
        dt: f32,
        duration_s: f32,
    ) -> Self {
        let deviation = additive_delta(incoming, outgoing);
        let mut v0 = 0.0f32;
        if crate::positive(dt) {
            let now = deviation_magnitude(&deviation);
            let before = deviation_magnitude(&additive_delta(incoming, outgoing_prev));
            if now > 1e-6 {
                let rate = (now - before) / dt / now;
                if rate.is_finite() {
                    // A deviation that was *shrinking* has a negative rate, which
                    // is the common case and what the quintic wants. A large
                    // positive rate (the two poses were flying apart) can make the
                    // curve overshoot past 1, so it is bounded by the reciprocal
                    // of the duration — one deviation's worth of growth per
                    // transition, which is already generous.
                    let bound = 1.0 / duration_s.max(1e-3);
                    v0 = rate.clamp(-8.0 * bound, bound);
                }
            }
        }
        Self {
            deviation,
            duration_s,
            elapsed_s: 0.0,
            v0,
        }
    }

    /// Advance the decay by `dt` seconds.
    pub fn advance(&mut self, dt: f32) {
        if dt.is_finite() {
            self.elapsed_s += dt.max(0.0);
        }
    }

    /// The current decay weight — in `[0,1]` for `v0 ≤ 0`, and a little above 1
    /// early on for a positive `v0` (measured: **1.019** at the `v0 = 4`
    /// [`capture_moving`](Self::capture_moving) allows for a 0.25 s decay), which
    /// is the overshoot a real inertialization has and a cross-fade cannot
    /// express.
    ///
    /// **What is rendered does not overshoot** (P29.2 audit): [`apply`](Self::apply)
    /// hands this to [`apply_additive`], which clamps every per-joint weight into
    /// `[0,1]` — so the excess is a property of the curve and not of the pose, and
    /// the render lands *on* the outgoing pose rather than past it. That clamp is
    /// the shared conservative rule masks rely on and is not worth relaxing for
    /// 2 % of one frame; this line says so instead of claiming a behaviour the
    /// type's own `apply` removes.
    pub fn decay(&self) -> f32 {
        quintic_decay(self.elapsed_s, self.duration_s, self.v0)
    }

    /// Whether the decay has finished and the inertializer can be dropped.
    pub fn is_done(&self) -> bool {
        !crate::greater(self.duration_s, self.elapsed_s) || !crate::positive(self.duration_s)
    }

    /// The captured deviation, for callers that want to inspect it (the tests do).
    pub fn deviation(&self) -> &Pose {
        &self.deviation
    }

    /// `target` plus the decaying deviation — the pose to render.
    pub fn apply(&self, target: &Pose) -> Pose {
        let w = self.decay();
        if !crate::positive(w) {
            return target.clone();
        }
        apply_additive(target, &self.deviation, &vec![w; target.locals.len()])
    }
}

/// The magnitude of a deviation pose: translation in metres, rotation through the
/// half-angle sine (`|q.xyz|`, which needs no `acos` and is monotone in the angle
/// over the half-turn a joint delta can be).
///
/// One scalar for the whole pose, deliberately: the decay is one curve, so the
/// rate that shapes it has to be one number. A per-joint rate would need five
/// coefficients per joint and would decay different limbs on different schedules,
/// which is a *different* algorithm and not a more accurate version of this one.
fn deviation_magnitude(dev: &Pose) -> f32 {
    let mut sum = 0.0f32;
    for l in &dev.locals {
        let t = l.translation_vec();
        sum += t.length_squared();
        let q = l.rotation_quat();
        sum += q.x * q.x + q.y * q.y + q.z * q.z;
    }
    sum.sqrt()
}

/// The sub-machine the runtime is currently inside, if its active state is one.
///
/// One level deep by construction ([`Motion::SubMachine`]'s own bound), so this
/// is a lookup and not a walk.
fn sub_machine_of<'a>(sm: &'a StateMachine, runtime: &SmRuntime) -> Option<&'a StateMachine> {
    match &sm.states.get(runtime.current)?.motion {
        Motion::SubMachine(inner) => Some(inner),
        _ => None,
    }
}

/// How a fired transition is blended.
///
/// # A WIRE ENUM as of `.inf_sm` v3 — append only, never reorder
///
/// `SmTransition::blend` puts this on disk, so its discriminants are content.
/// bincode writes an externally-tagged enum as its **variant index**, which means
/// inserting a variant anywhere but the end silently re-reads every committed
/// machine's transitions as a different mode. The standing P19 rule, and
/// `the_blend_mode_wire_discriminants_are_frozen` is what enforces it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmBlendMode {
    /// Inertialization — **the P29.2 default** (§13's catalogue amendment).
    #[default]
    Inertialize,
    /// The P11.2/P29.1 cross-fade: evaluate both states and interpolate.
    /// Retained because the two are measurably different and a project is
    /// entitled to the older behaviour.
    CrossFade,
}

/// Where the incoming state's play-head starts when a transition fires.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransitionEntry {
    /// At zero — the state's own beginning. What every transition has always
    /// done.
    #[default]
    Restart,
    /// At the frame of the incoming clip that best matches the pose on screen
    /// ([`crate::pose_match`]).
    ///
    /// Off by default because it changes *which frame plays*, which is a content
    /// decision. P29.4's get-up and landing consumers turn it on; it is reachable
    /// here so the primitive has a shipped door rather than only a test.
    PoseMatched,
}

/// Drives a [`StateMachine`]'s pose with inertialization.
///
/// Holds what [`SmRuntime`] cannot: the previous two rendered poses (the
/// deviation is captured from them) and the live [`Inertializer`]. One per posed
/// entity.
#[derive(Clone, Debug, Default)]
pub struct PoseBlender {
    pub mode: SmBlendMode,
    pub entry: TransitionEntry,
    /// How many candidate frames [`TransitionEntry::PoseMatched`] scans.
    pub match_samples: usize,
    /// How the pose match is priced.
    pub match_weights: PoseMatchWeights,
    inertia: Option<Inertializer>,
    last: Option<Pose>,
    prev: Option<Pose>,
    /// Motions evaluated since construction — the "cheaper" claim's counter.
    evaluations: u64,
}

impl PoseBlender {
    /// A blender in the P29.2 default configuration: inertialized transitions,
    /// restart entry.
    pub fn new() -> Self {
        Self {
            mode: SmBlendMode::default(),
            entry: TransitionEntry::default(),
            match_samples: 32,
            match_weights: PoseMatchWeights::default(),
            inertia: None,
            last: None,
            prev: None,
            evaluations: 0,
        }
    }

    /// A blender that reproduces the P11.2/P29.1 cross-fade exactly.
    pub fn cross_fade() -> Self {
        Self {
            mode: SmBlendMode::CrossFade,
            ..Self::new()
        }
    }

    /// The number of **state evaluations** performed — one per pose the machine
    /// had to sample, summed over every step.
    ///
    /// This is the perf claim as a *function of the program* rather than of the
    /// machine it ran on: a cross-fade costs two (three across a carried
    /// interruption) while a fade is running, inertialization costs one always.
    /// The wall-clock arm measures the same thing in seconds and is the one that
    /// cannot assert on a paravirtual runner.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    /// Whether a decay is currently running.
    pub fn is_blending(&self) -> bool {
        self.inertia.is_some()
    }

    /// The live decay weight, or `0` when nothing is decaying.
    pub fn decay(&self) -> f32 {
        self.inertia.as_ref().map(|i| i.decay()).unwrap_or(0.0)
    }

    /// The pose rendered last step, if any — the snapshot primitive's natural
    /// source, and what P29.4's ragdoll exit reads.
    pub fn snapshot(&self, label: impl Into<String>, time_s: f64) -> Option<PoseSnapshot> {
        self.last
            .as_ref()
            .map(|p| PoseSnapshot::capture(label, p, time_s))
    }

    /// Forget the history — call when the entity's machine or skeleton changes,
    /// so a deviation is never captured between two different rigs.
    pub fn reset(&mut self) {
        self.inertia = None;
        self.last = None;
        self.prev = None;
    }

    /// [`reset`](Self::reset) if the history is for a **different rig**.
    ///
    /// A deviation between poses of different joint counts is not a deviation, and
    /// nothing downstream would notice: [`apply_additive`] takes the shorter
    /// length and quietly leaves the rest of the character on the base. The caller
    /// that has both the entity and the skeleton — `inf_ecs::pose` — asks this
    /// once per step, which is cheaper than a rig identity and catches the case
    /// that actually happens (a `SkeletalMesh` re-bound during a session).
    pub fn fit_rig(&mut self, joints: usize) {
        if self.last.as_ref().is_some_and(|p| p.locals.len() != joints) {
            self.reset();
        }
    }

    /// **How the transition that just fired blends** — the ONE precedence rule
    /// (`.inf_sm` v3).
    ///
    /// ```text
    /// the fired transition's own `blend`   (Some)   — the author's per-edge choice
    ///   else this blender's `mode`                  — the session default
    ///                                                 (`inf_ecs::pose::set_blend_mode`,
    ///                                                  carried by ScenePayload v11)
    ///   else SmBlendMode::Inertialize               — `mode`'s own Default
    /// ```
    ///
    /// Written down in exactly one function, and read by the step below and by
    /// nothing else. Two authorities that each believe they decide is the P29.3
    /// slope-limit defect — `mover_for` and the controller both "owned" the slope
    /// limit and disagreed — and a per-edge override on top of a world-level
    /// setting is precisely the shape that recurs at.
    ///
    /// A **sub-machine** transition resolves against the sub-machine's own
    /// transition list, because that is where the author wrote it. A top-level
    /// fire wins over a simultaneous sub-fire, matching the step below, which
    /// collapses one fade for both.
    pub fn mode_for(&self, sm: &StateMachine, runtime: &SmRuntime, step: &SmStep) -> SmBlendMode {
        if let Some(i) = step.fired {
            if let Some(b) = sm.transitions.get(i).and_then(|t| t.blend) {
                return b;
            }
            return self.mode;
        }
        if let Some(i) = step.sub_fired {
            if let Some(b) = sub_machine_of(sm, runtime)
                .and_then(|inner| inner.transitions.get(i))
                .and_then(|t| t.blend)
            {
                return b;
            }
        }
        self.mode
    }

    /// Advance the machine by `dt` and produce the pose to render.
    ///
    /// Equivalent to [`SmRuntime::advance`] + [`eval_pose`] in `CrossFade` mode.
    /// In `Inertialize` mode the fade the runtime set up is **collapsed** the step
    /// it starts — `prev`, `fade_dur` and any carry are cleared — so `eval_pose`
    /// samples exactly one state and the outgoing half is the decaying deviation
    /// instead.
    #[allow(clippy::too_many_arguments)]
    pub fn step<'c>(
        &mut self,
        sm: &StateMachine,
        runtime: &mut SmRuntime,
        skeleton: &Skeleton,
        clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
        ctx: &SmContext,
        dt: f64,
    ) -> (Pose, SmStep) {
        let step = runtime.advance(sm, ctx, dt);
        let fired = step.fired.is_some() || step.sub_fired.is_some();
        // `.inf_sm` v3: the transition that just fired may name its own mode. THE
        // ONE precedence site — see `mode_for`.
        let mode = self.mode_for(sm, runtime, &step);
        let mut duration = 0.0f32;
        if mode == SmBlendMode::Inertialize && fired {
            // The fade the runtime just set up, taken as the decay's length before
            // it is collapsed. A zero-duration transition stays a snap, exactly as
            // it is under the cross-fade.
            duration = runtime.fade_dur.max(runtime.sub.fade_dur) as f32;
            runtime.prev = None;
            runtime.prev_time = 0.0;
            runtime.fade_t = 0.0;
            runtime.fade_dur = 0.0;
            runtime.carry = None;
            runtime.carry_time = 0.0;
            runtime.carry_alpha = 0.0;
            runtime.sub.prev = None;
            runtime.sub.prev_time = 0.0;
            runtime.sub.fade_t = 0.0;
            runtime.sub.fade_dur = 0.0;
            // Pose matching, when the consumer asked for it: enter the incoming
            // clip at the frame nearest what is on screen rather than at zero.
            if self.entry == TransitionEntry::PoseMatched {
                if let Some(t) = self.matched_entry(sm, runtime, skeleton, clips) {
                    runtime.state_time = t;
                }
            }
        }
        self.evaluations += self.evaluation_cost(sm, runtime);
        let target = eval_pose(sm, runtime, skeleton, clips, ctx);
        if duration > 0.0 {
            if let Some(last) = self.last.clone() {
                self.inertia = Some(match &self.prev {
                    Some(prev) => {
                        Inertializer::capture_moving(&last, prev, &target, dt as f32, duration)
                    }
                    None => Inertializer::capture(&last, &target, duration),
                });
            }
        }
        let out = match &mut self.inertia {
            Some(i) => {
                i.advance(dt as f32);
                let posed = i.apply(&target);
                if i.is_done() {
                    self.inertia = None;
                }
                posed
            }
            None => target,
        };
        self.prev = self.last.replace(out.clone());
        (out, step)
    }

    /// How many state evaluations the coming [`eval_pose`] will perform — one for
    /// the current state, one for an outgoing cross-fade partner, one more for a
    /// carried interruption.
    fn evaluation_cost(&self, sm: &StateMachine, runtime: &SmRuntime) -> u64 {
        let n = sm.states.len();
        let mut cost = 1;
        if runtime.prev.is_some_and(|p| p < n) {
            cost += 1;
            if runtime.carry.is_some_and(|c| c < n) {
                cost += 1;
            }
        }
        cost
    }

    /// The pose-matched entry time for the state just entered, if it plays a
    /// single clip and there is a pose on screen to match against.
    fn matched_entry<'c>(
        &self,
        sm: &StateMachine,
        runtime: &SmRuntime,
        skeleton: &Skeleton,
        clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
    ) -> Option<f64> {
        let last = self.last.as_ref()?;
        let state = sm.states.get(runtime.current)?;
        let Motion::Clip(id) = &state.motion else {
            // A blend space's frame is a function of live parameters, so "the
            // frame that matches" is not a property of one clip. P29.4 owns that
            // case; refusing here is the honest answer, not a guess.
            return None;
        };
        let clip = clips(*id)?;
        let query = match &self.prev {
            Some(prev) => {
                PoseSnapshot::capture_moving("entry", skeleton, last, prev, 1.0 / 60.0, 0.0)
            }
            None => PoseSnapshot::capture("entry", last, 0.0),
        };
        let m = match_clip(
            skeleton,
            &query,
            clip,
            self.match_samples,
            &self.match_weights,
        )?;
        // `state_time` is the play-head BEFORE `speed` scales it (`eval_pose`
        // samples at `t * state.speed`), so the matched clip time divides by it.
        // A non-positive or non-finite speed is REFUSED rather than clamped: a
        // state playing backwards has no "the frame we are at" in this
        // parameterisation, and a `1e-6` floor would turn its match into a
        // play-head of a million seconds.
        crate::positive64(state.speed).then(|| m.time_s as f64 / state.speed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Interpolation, JointTrack, QuatTrack};
    use crate::skeleton::{Joint, JointTransform};
    use crate::state_machine::{CmpOp, SmState, SmTransition};
    use glam::{Mat4, Quat, Vec3};

    fn chain() -> Skeleton {
        let mut joints = Vec::new();
        let mut global = Mat4::IDENTITY;
        for i in 0..3 {
            let local = JointTransform::from_trs(
                if i == 0 { Vec3::ZERO } else { Vec3::Y },
                Quat::IDENTITY,
                Vec3::ONE,
            );
            global *= local.to_mat4();
            joints.push(Joint {
                name: format!("j{i}"),
                parent: if i == 0 { None } else { Some(i as u16 - 1) },
                inverse_bind: global.inverse().to_cols_array(),
                local_bind: local,
            });
        }
        Skeleton::new(joints).unwrap()
    }

    fn rot(deg: f32) -> Quat {
        inf_math::pslerp(
            Quat::IDENTITY,
            Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
            deg / 180.0,
        )
    }

    /// A clip holding joint 1 at a constant angle, spinning slowly.
    fn spin(name: &str, from: f32, to: f32, dur: f32) -> AnimClip {
        let mut jt = JointTrack::new(1);
        jt.rotation = Some(QuatTrack::new(
            vec![0.0, dur],
            vec![rot(from).to_array(), rot(to).to_array()],
            Interpolation::Linear,
        ));
        AnimClip::new(name, vec![jt])
    }

    fn two_states() -> StateMachine {
        StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition::on(0, 1, 0.25, "go", CmpOp::Gt, 0.5)],
            entry: 0,
            ..Default::default()
        }
    }

    fn deg(p: &Pose) -> f32 {
        p.locals[1]
            .rotation_quat()
            .angle_between(Quat::IDENTITY)
            .to_degrees()
    }

    /// The curve's defining properties, asserted rather than described.
    #[test]
    fn the_quintic_lands_flat_and_never_grows() {
        let t1 = 0.25f32;
        assert_eq!(quintic_decay(0.0, t1, 0.0), 1.0);
        assert_eq!(quintic_decay(t1, t1, 0.0), 0.0);
        assert_eq!(quintic_decay(t1 * 2.0, t1, 0.0), 0.0);
        // Monotone decreasing on [0, T], and inside [0,1].
        let mut last = 1.0;
        for k in 1..=250 {
            let y = quintic_decay(t1 * k as f32 / 250.0, t1, 0.0);
            assert!((0.0..=1.0).contains(&y), "y={y} at k={k}");
            assert!(y <= last + 1e-6, "y grew: {y} after {last}");
            last = y;
        }
        // First AND second derivative vanish at the end — the property a
        // cross-fade curve cannot have, checked by finite difference.
        let h = 1e-4f32;
        let y_end = quintic_decay(t1, t1, 0.0);
        let y1 = quintic_decay(t1 - h, t1, 0.0);
        let y2 = quintic_decay(t1 - 2.0 * h, t1, 0.0);
        let d1 = (y_end - y1) / h;
        let d2 = (y_end - 2.0 * y1 + y2) / (h * h);
        assert!(d1.abs() < 1e-3, "y'(T) = {d1}");
        assert!(d2.abs() < 1.0, "y''(T) = {d2}");
        // A degenerate duration is an instant switch, not a division.
        for bad in [0.0f32, -1.0, f32::NAN] {
            assert_eq!(quintic_decay(0.1, bad, 0.0), 0.0);
        }
        // A non-finite rate reads as zero rather than poisoning the curve.
        assert!(quintic_decay(0.1, t1, f32::NAN).is_finite());
    }

    /// The initial rate really is used: a negative `v0` (the deviation was already
    /// shrinking) makes the curve leave 1 with a downward slope, where `v0 = 0`
    /// leaves it *nearly* flat.
    ///
    /// "Nearly", and the gap is the design rather than slop: `y′(0) = v₀` exactly,
    /// but `y″(0) = a₀ = −20/T²` is deliberately **not** zero — that is what buys
    /// the flat landing at `T`. So a finite difference over `h` measures
    /// `v₀ + a₀h/2`, which at `h = 1e-3` and `T = 0.3` is `−0.11` for `v₀ = 0`.
    /// Asserting "< 1e-2" there would have been asserting against the curve.
    #[test]
    fn the_initial_rate_shapes_the_start_of_the_curve() {
        let t1 = 0.3f32;
        let h = 1e-3f32;
        let slope = |v0: f32| (quintic_decay(h, t1, v0) - 1.0) / h;
        let a0 = -20.0 / (t1 * t1);
        assert!(
            (slope(0.0) - a0 * h / 2.0).abs() < 1e-2,
            "v0=0 left the curve at {} rather than at a0·h/2 = {}",
            slope(0.0),
            a0 * h / 2.0
        );
        assert!(slope(-2.0) < -1.5, "{}", slope(-2.0));
        assert!(
            slope(-2.0) < slope(0.0) - 1.5,
            "the rate changed nothing: {} vs {}",
            slope(-2.0),
            slope(0.0)
        );
        // …and it stays bounded and finite over the whole interval.
        for v0 in [-8.0f32, -2.0, 0.0, 1.0] {
            for k in 0..=100 {
                let y = quintic_decay(t1 * k as f32 / 100.0, t1, v0);
                assert!(y.is_finite() && y.abs() < 4.0, "v0={v0} k={k} y={y}");
            }
        }
    }

    #[test]
    fn an_inertializer_starts_at_the_outgoing_pose_and_ends_at_the_incoming_one() {
        let sk = chain();
        let mut outgoing = Pose::rest(&sk);
        outgoing.locals[1].rotation = rot(60.0).to_array();
        let incoming = Pose::rest(&sk);
        let mut i = Inertializer::capture(&outgoing, &incoming, 0.2);
        // t = 0: exactly the outgoing pose.
        assert!((deg(&i.apply(&incoming)) - 60.0).abs() < 0.01);
        i.advance(0.2);
        assert!(i.is_done());
        // t = T: exactly the incoming pose.
        assert!(deg(&i.apply(&incoming)) < 0.01);
    }

    /// **The perf claim, as a function of the program.** Inertialization evaluates
    /// one state per step; the cross-fade evaluates two while a fade runs.
    ///
    /// Asserted everywhere — this is arithmetic on a counter, not a clock, so the
    /// paravirtual-runner law does not apply to it. The wall-clock twin is
    /// `crates/inf-anim/tests/inertialization.rs`.
    #[test]
    fn inertialization_evaluates_one_state_where_the_cross_fade_evaluates_two() {
        let sk = chain();
        let (a, b) = (spin("a", 0.0, 10.0, 1.0), spin("b", 80.0, 90.0, 1.0));
        let clips = |c: ClipRef| -> Option<&AnimClip> {
            match c[0] {
                1 => Some(&a),
                2 => Some(&b),
                _ => None,
            }
        };
        let sm = two_states();
        let run = |mut blender: PoseBlender| -> u64 {
            let mut rt = SmRuntime::default();
            let go = std::cell::Cell::new(0.0f64);
            let vars = |n: &str| (n == "go").then(|| go.get());
            let ctx = SmContext::new(&vars);
            for k in 0..30 {
                if k == 5 {
                    go.set(1.0);
                }
                blender.step(&sm, &mut rt, &sk, &clips, &ctx, 1.0 / 60.0);
            }
            blender.evaluations()
        };
        let inert = run(PoseBlender::new());
        let faded = run(PoseBlender::cross_fade());
        assert_eq!(inert, 30, "one evaluation per step, always");
        // 0.25 s at 60 Hz is 15 steps of two evaluations.
        assert!(
            faded >= 44,
            "the cross-fade should have paid for a second state ({faded})"
        );
        assert!(
            faded > inert,
            "cross-fade {faded} was not dearer than inertialization {inert}"
        );
    }

    /// The shared trace harness: run a two-state machine through one transition
    /// and return the rendered joint-1 angle per step, plus the step the
    /// transition fired on.
    fn transition_trace<'c>(
        blender: &mut PoseBlender,
        sk: &Skeleton,
        sm: &StateMachine,
        clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
        steps: usize,
        fire_at: usize,
    ) -> (Vec<f32>, usize) {
        let mut rt = SmRuntime::default();
        let go = std::cell::Cell::new(0.0f64);
        let vars = |n: &str| (n == "go").then(|| go.get());
        let ctx = SmContext::new(&vars);
        let mut out = Vec::new();
        let mut fired_on = 0;
        for k in 0..steps {
            if k == fire_at {
                go.set(1.0);
            }
            let (p, step) = blender.step(sm, &mut rt, sk, clips, &ctx, 1.0 / 60.0);
            if step.fired.is_some() {
                fired_on = k;
            }
            out.push(deg(&p));
        }
        (out, fired_on)
    }

    /// The largest `|second difference|` over a window — the frame-to-frame change
    /// in the rendered angle's *velocity*, i.e. how abruptly the motion changes.
    fn peak_jerk(v: &[f32]) -> f32 {
        v.windows(3)
            .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
            .fold(0.0f32, f32::max)
    }

    /// **The smoothness claim, at the place the two mechanisms differ most.**
    ///
    /// A linear cross-fade — the engine's default curve, so this is what
    /// inertialization replaces — contributes a constant extra velocity of
    /// `Δpose / T` for the whole fade and then **stops**, so the rendered motion
    /// has a velocity step at each end. Inertialization's offset lands with
    /// `y′(T) = y″(T) = 0` by construction, so it has none at the end at all.
    ///
    /// Measured on this fixture (a ~76° pose difference, a 0.25 s transition at
    /// 60 Hz, so 15 frames of fade), over the **last third of the fade plus the
    /// six frames after it**: the cross-fade's worst second difference is
    /// **3.63°/step²** and inertialization's is **0.59°/step²** — **6.2×**. The
    /// arm asserts 4×, which leaves room for arithmetic noise without leaving room
    /// for the property to disappear.
    ///
    /// The *overall* peak is a different and more honest story, asserted in the
    /// sibling arm below.
    #[test]
    fn inertialization_lands_a_transition_without_the_cross_fade_velocity_step() {
        let sk = chain();
        // Two clips at very different angles, each moving, so the transition has a
        // real velocity mismatch to absorb.
        let (a, b) = (spin("a", 0.0, 30.0, 1.0), spin("b", 80.0, 20.0, 1.0));
        let clips = |c: ClipRef| -> Option<&AnimClip> {
            match c[0] {
                1 => Some(&a),
                2 => Some(&b),
                _ => None,
            }
        };
        let sm = two_states();
        let mut inert_b = PoseBlender::new();
        let mut fade_b = PoseBlender::cross_fade();
        let (inert, fired) = transition_trace(&mut inert_b, &sk, &sm, &clips, 48, 8);
        let (faded, fired_f) = transition_trace(&mut fade_b, &sk, &sm, &clips, 48, 8);
        assert_eq!(fired, fired_f, "the two ran different machines");
        // Both end up at the same place — a comparison of routes, not destinations.
        assert!(
            (inert.last().unwrap() - faded.last().unwrap()).abs() < 1.0,
            "the two blends did not converge: {:?} vs {:?}",
            inert.last(),
            faded.last()
        );
        // 0.25 s at 60 Hz is 15 frames; the tail window is the last third plus the
        // six frames after the fade ends, which is where the step lives.
        let tail = fired + 10..(fired + 21).min(inert.len());
        let (ti, tf) = (peak_jerk(&inert[tail.clone()]), peak_jerk(&faded[tail]));
        assert!(
            ti * 4.0 < tf,
            "at the end of the transition inertialization jerked {ti}°/step² against \
             the cross-fade's {tf}°/step² — under the 4x this arm claims"
        );
    }

    /// **The overall peak, stated honestly.** Inertialization is smoother across
    /// the whole transition too, but by a smaller margin than the tail, and the
    /// reason is a real property of the curve rather than a measurement artefact:
    /// `y″(0) = −20/T²` is deliberately non-zero (that is what buys the flat
    /// landing), so the offset starts shedding quickly. Measured on the same
    /// fixture: **3.2°/step² against the cross-fade's 4.9°/step²**, a 1.5× win —
    /// not the 10× the tail shows, and written down as such.
    #[test]
    fn inertializations_worst_frame_is_still_better_than_the_cross_fades() {
        let sk = chain();
        let (a, b) = (spin("a", 0.0, 30.0, 1.0), spin("b", 80.0, 20.0, 1.0));
        let clips = |c: ClipRef| -> Option<&AnimClip> {
            match c[0] {
                1 => Some(&a),
                2 => Some(&b),
                _ => None,
            }
        };
        let sm = two_states();
        let (inert, _) = transition_trace(&mut PoseBlender::new(), &sk, &sm, &clips, 48, 8);
        let (faded, _) = transition_trace(&mut PoseBlender::cross_fade(), &sk, &sm, &clips, 48, 8);
        let (pi, pf) = (peak_jerk(&inert), peak_jerk(&faded));
        assert!(
            pi < pf,
            "inertialization peaked at {pi}°/step² against the cross-fade's {pf}°/step²"
        );
    }

    /// **The runtime's own cross-fade really is collapsed**, which is what makes
    /// the single evaluation a fact about the program rather than about the
    /// counter — and the pose still arrives.
    #[test]
    fn an_inertialized_transition_leaves_the_runtime_with_one_live_state() {
        let sk = chain();
        let (a, b) = (spin("a", 0.0, 5.0, 1.0), spin("b", 90.0, 95.0, 1.0));
        let clips = |c: ClipRef| -> Option<&AnimClip> {
            match c[0] {
                1 => Some(&a),
                2 => Some(&b),
                _ => None,
            }
        };
        let sm = two_states();
        let mut blender = PoseBlender::new();
        let mut rt = SmRuntime::default();
        let go = std::cell::Cell::new(0.0f64);
        let vars = |n: &str| (n == "go").then(|| go.get());
        let ctx = SmContext::new(&vars);
        let mut saw_fire = false;
        let mut last = 0.0;
        for k in 0..40 {
            if k == 6 {
                go.set(1.0);
            }
            let (p, step) = blender.step(&sm, &mut rt, &sk, &clips, &ctx, 1.0 / 60.0);
            if step.fired.is_some() {
                saw_fire = true;
                assert!(rt.prev.is_none(), "the fade was not collapsed");
                assert_eq!(rt.fade_dur, 0.0);
                assert!(rt.carry.is_none());
                assert!(blender.is_blending(), "no decay was captured");
            }
            last = deg(&p);
        }
        assert!(saw_fire);
        assert!(!blender.is_blending(), "the decay never finished");
        assert!(
            (last - 90.0).abs() < 5.0,
            "ended at {last}°, not near the incoming clip"
        );
    }

    /// **An interruption during a live decay inherits the whole stack**, which the
    /// `Copy` runtime's one-deep carry cannot: the capture is taken from the
    /// *rendered* pose, so whatever was already being decayed is inside it.
    ///
    /// The P29.1 audit measured the cross-fade's second interruption at 40.5° of
    /// discontinuity. Here the two mechanisms are run through the *same* double
    /// transition and compared on the same statistic.
    #[test]
    fn a_second_transition_inside_a_decay_is_still_continuous() {
        let sk = chain();
        let (a, b, c) = (
            spin("a", 0.0, 2.0, 1.0),
            spin("b", 70.0, 72.0, 1.0),
            spin("c", 140.0, 142.0, 1.0),
        );
        let clips = |x: ClipRef| -> Option<&AnimClip> {
            match x[0] {
                1 => Some(&a),
                2 => Some(&b),
                3 => Some(&c),
                _ => None,
            }
        };
        let sm = StateMachine {
            states: vec![
                SmState::clip("a", [1; 16]),
                SmState::clip("b", [2; 16]),
                SmState::clip("c", [3; 16]),
            ],
            transitions: vec![
                SmTransition::on(0, 1, 0.4, "one", CmpOp::Gt, 0.5),
                SmTransition::on(1, 2, 0.4, "two", CmpOp::Gt, 0.5),
            ],
            entry: 0,
            ..Default::default()
        };
        let trace = |mut blender: PoseBlender| -> Vec<f32> {
            let mut rt = SmRuntime::default();
            let one = std::cell::Cell::new(0.0f64);
            let two = std::cell::Cell::new(0.0f64);
            let vars = |n: &str| match n {
                "one" => Some(one.get()),
                "two" => Some(two.get()),
                _ => None,
            };
            let ctx = SmContext::new(&vars);
            let mut out = Vec::new();
            for k in 0..60 {
                if k == 4 {
                    one.set(1.0);
                }
                // The second transition fires 6 steps in — well inside the first
                // decay.
                if k == 10 {
                    two.set(1.0);
                }
                let (p, _) = blender.step(&sm, &mut rt, &sk, &clips, &ctx, 1.0 / 60.0);
                out.push(deg(&p));
            }
            out
        };
        let inert = trace(PoseBlender::new());
        let faded = trace(PoseBlender::cross_fade());
        let (pi, pf) = (peak_jerk(&inert), peak_jerk(&faded));
        assert!(
            pi < pf,
            "across two stacked transitions inertialization jerked {pi}°/step² \
             against the cross-fade's {pf}°/step²"
        );
        // The interesting half is the *second* interruption, at step 11: the
        // cross-fade's carry slot drops the older partner there, and this does not.
        let window = 9..18;
        let (wi, wf) = (peak_jerk(&inert[window.clone()]), peak_jerk(&faded[window]));
        assert!(
            wi < wf,
            "at the second interruption inertialization jerked {wi}°/step² against \
             the cross-fade's {wf}°/step²"
        );
        // Both arrive at the third clip.
        for (name, v) in [("inertialized", &inert), ("cross-faded", &faded)] {
            assert!(
                (v.last().unwrap() - 140.0).abs() < 6.0,
                "the {name} run ended at {}°, not at the third clip",
                v.last().unwrap()
            );
        }
    }

    /// `CrossFade` mode leaves the runtime's own bookkeeping alone, so a project
    /// that opts out gets exactly the P29.1 behaviour.
    #[test]
    fn cross_fade_mode_is_the_untouched_p29_1_path() {
        let sk = chain();
        let (a, b) = (spin("a", 0.0, 10.0, 1.0), spin("b", 80.0, 90.0, 1.0));
        let clips = |c: ClipRef| -> Option<&AnimClip> {
            match c[0] {
                1 => Some(&a),
                2 => Some(&b),
                _ => None,
            }
        };
        let sm = two_states();
        let go = std::cell::Cell::new(0.0f64);
        let vars = |n: &str| (n == "go").then(|| go.get());
        let ctx = SmContext::new(&vars);
        let mut blender = PoseBlender::cross_fade();
        let mut rt = SmRuntime::default();
        let mut direct = SmRuntime::default();
        for k in 0..24 {
            if k == 5 {
                go.set(1.0);
            }
            let (p, _) = blender.step(&sm, &mut rt, &sk, &clips, &ctx, 1.0 / 60.0);
            direct.advance(&sm, &ctx, 1.0 / 60.0);
            let q = eval_pose(&sm, &direct, &sk, &clips, &ctx);
            assert_eq!(p, q, "step {k}");
            assert_eq!(rt, direct, "step {k}");
        }
    }

    /// **The pose-matching consumer**, through the shipped door rather than only
    /// through `pose_match`'s own arms: with `PoseMatched` entry the incoming clip
    /// starts at the frame nearest what was on screen, so the switch costs less
    /// deviation than a restart does.
    #[test]
    fn a_pose_matched_entry_starts_the_incoming_clip_where_the_body_already_is() {
        let sk = chain();
        // `a` holds ~50°; `b` sweeps 0 → 100° over a second, so 50° is halfway in.
        let a = spin("a", 50.0, 50.0, 1.0);
        let b = spin("b", 0.0, 100.0, 1.0);
        let clips = |c: ClipRef| -> Option<&AnimClip> {
            match c[0] {
                1 => Some(&a),
                2 => Some(&b),
                _ => None,
            }
        };
        let sm = two_states();
        let run = |entry: TransitionEntry| -> (f64, f32) {
            let mut blender = PoseBlender::new();
            blender.entry = entry;
            blender.match_samples = 101;
            let mut rt = SmRuntime::default();
            let go = std::cell::Cell::new(0.0f64);
            let vars = |n: &str| (n == "go").then(|| go.get());
            let ctx = SmContext::new(&vars);
            let mut entered_at = 0.0;
            let mut peak = 0.0f32;
            for k in 0..10 {
                if k == 4 {
                    go.set(1.0);
                }
                let (_, step) = blender.step(&sm, &mut rt, &sk, &clips, &ctx, 1.0 / 60.0);
                if step.fired.is_some() {
                    entered_at = rt.state_time;
                    peak = blender
                        .inertia
                        .as_ref()
                        .map(|i| deviation_magnitude(i.deviation()))
                        .unwrap_or(0.0);
                }
            }
            (entered_at, peak)
        };
        let (restart_t, restart_dev) = run(TransitionEntry::Restart);
        let (matched_t, matched_dev) = run(TransitionEntry::PoseMatched);
        assert_eq!(restart_t, 0.0, "a restart entry starts at zero");
        assert!(
            (matched_t - 0.5).abs() < 0.03,
            "the matched entry should have landed near 0.5 s, got {matched_t}"
        );
        assert!(
            matched_dev < restart_dev * 0.25,
            "matching did not reduce the deviation ({matched_dev} vs {restart_dev})"
        );
    }
}
