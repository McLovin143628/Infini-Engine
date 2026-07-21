//! Animation state machines (P11.2, `.inf_sm`).
//!
//! A state machine is **not** a dataflow DAG (unlike Blueprints / materials /
//! PCG, which ride the `inf-graph` substrate): its nodes are *states* and its
//! edges are *transitions*. So it is a plain, typed serde model — a `Vec` of
//! [`SmState`]s and a `Vec` of [`SmTransition`]s — with canvas `position`s for
//! layout. The @xyflow editor renders that model directly; there is no graph
//! compile step to lower.
//!
//! # Evaluation
//!
//! Each state plays a [`Motion`] (a single clip or a 1D/2D blend space). Each
//! fixed step [`SmRuntime::advance`] evaluates the outgoing transitions of the
//! current state — a transition fires when **all** its [`SmCondition`]s hold
//! (AND semantics; an OR is authored as two parallel transitions) and its
//! optional normalized `exit_time` gate is satisfied — then cross-fades to the
//! target over `duration` seconds. [`eval_pose`] samples the current (and, mid
//! cross-fade, the previous) state into a [`Pose`]. Everything is pure and
//! deterministic given `(sm, runtime, ctx, dt)`.
//!
//! # The condition/param seam ([`SmContext`])
//!
//! Conditions and blend-space parameters both read **named f64 values**. The
//! machine never reaches into an interpreter directly; it reads through
//! [`SmContext`], a thin `name → f64` (and `state → loop-period`) lookup. The
//! editor Simulate loop and the shipped runtime back it with the actor's
//! Blueprint variables; tests back it with a closure. This is the whole coupling
//! surface between the animation layer and gameplay state.

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::blend_space::{
    sample_blend_space_1d, sample_blend_space_2d, BlendSpace1D, BlendSpace2D, ClipRef,
};
use crate::clip::AnimClip;
use crate::pose::{blend_poses, sample_clip, Pose};
use crate::skeleton::Skeleton;

/// A comparison operator for a transition condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp {
    /// `>`
    Gt,
    /// `<`
    Lt,
    /// `>=`
    Ge,
    /// `<=`
    Le,
    /// `==` (within a small epsilon — these are floats)
    Eq,
    /// `!=`
    Ne,
}

impl CmpOp {
    /// Evaluate `lhs <op> rhs`. `Eq`/`Ne` use a small epsilon so authored
    /// integer-ish thresholds compare cleanly.
    pub fn eval(self, lhs: f64, rhs: f64) -> bool {
        const EPS: f64 = 1e-9;
        match self {
            CmpOp::Gt => lhs > rhs,
            CmpOp::Lt => lhs < rhs,
            CmpOp::Ge => lhs >= rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Eq => (lhs - rhs).abs() <= EPS,
            CmpOp::Ne => (lhs - rhs).abs() > EPS,
        }
    }
}

/// One condition on a transition: `var <op> value`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmCondition {
    /// The variable name looked up through [`SmContext`] (a missing variable
    /// reads as `0.0`).
    pub var: String,
    pub op: CmpOp,
    pub value: f64,
}

impl SmCondition {
    /// Whether this condition holds under `ctx`.
    pub fn eval(&self, ctx: &SmContext) -> bool {
        let v = ctx.var(&self.var).unwrap_or(0.0);
        self.op.eval(v, self.value)
    }
}

/// What a state plays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Motion {
    /// A single animation clip (raw GUID).
    Clip(ClipRef),
    /// A 1D blend space.
    Blend1D(BlendSpace1D),
    /// A 2D blend space.
    Blend2D(BlendSpace2D),
}

/// One state of the machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmState {
    /// Display name (unique within the machine, by editor convention).
    pub name: String,
    /// What this state plays.
    pub motion: Motion,
    /// Wrap the motion at its end (`true`) or hold the last pose (`false`).
    pub looping: bool,
    /// Play-rate multiplier for this state's motion.
    pub speed: f64,
    /// Canvas layout position (editor-only; ignored by evaluation).
    pub position: (f32, f32),
}

impl SmState {
    /// A looping single-clip state at unit speed.
    pub fn clip(name: impl Into<String>, clip: ClipRef) -> Self {
        Self {
            name: name.into(),
            motion: Motion::Clip(clip),
            looping: true,
            speed: 1.0,
            position: (0.0, 0.0),
        }
    }
}

/// One transition edge between two states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmTransition {
    /// Source state index.
    pub from: usize,
    /// Destination state index.
    pub to: usize,
    /// Cross-fade duration in seconds (`0` = a hard cut).
    pub duration: f64,
    /// Conditions, ANDed. Empty = unconditional (subject only to `exit_time`).
    pub conditions: Vec<SmCondition>,
    /// Optional normalized (`[0,1]`) exit-time gate against the source state's
    /// motion loop: the transition may only fire once the source has played at
    /// least this fraction of a loop. `None` = fireable any time.
    pub exit_time: Option<f64>,
}

impl SmTransition {
    /// Whether this transition is ready to fire: all conditions hold **and** the
    /// exit-time gate is satisfied. `state_time` is seconds spent in the source
    /// state; `period` is the source state's motion loop length (see
    /// [`SmContext::period`]) — `None`/`0` means "period unknown", in which case
    /// the exit-time gate is treated as satisfied (documented v1 fallback, so an
    /// unresolved-duration machine never deadlocks).
    pub fn ready(&self, ctx: &SmContext, state_time: f64, period: Option<f64>) -> bool {
        if !self.conditions.iter().all(|c| c.eval(ctx)) {
            return false;
        }
        match (self.exit_time, period) {
            (Some(frac), Some(p)) if p > 0.0 => state_time >= frac * p,
            _ => true,
        }
    }
}

/// A whole state machine: states, transitions, and the entry state index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateMachine {
    pub states: Vec<SmState>,
    pub transitions: Vec<SmTransition>,
    /// Index of the state entered on start.
    pub entry: usize,
}

impl StateMachine {
    /// An empty machine (no states) with entry 0.
    pub fn empty() -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            entry: 0,
        }
    }
}

/// The live play state of a machine on one entity. Small + `Copy` so it can live
/// inside an ECS component; **not** serialized (rebuilt each play session).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmRuntime {
    /// The active state index.
    pub current: usize,
    /// The state being cross-faded *out of*, if a fade is in progress.
    pub prev: Option<usize>,
    /// The `prev` state's play-head, frozen at hand-off (v1: the outgoing pose is
    /// held static through the fade; a continuously-advancing outgoing pose is a
    /// documented follow-up).
    pub prev_time: f64,
    /// Elapsed cross-fade time (seconds).
    pub fade_t: f64,
    /// Total cross-fade duration (seconds).
    pub fade_dur: f64,
    /// Seconds spent in the current state.
    pub state_time: f64,
    /// Whether the runtime has been initialized to the machine's entry state.
    pub started: bool,
}

impl Default for SmRuntime {
    fn default() -> Self {
        Self {
            current: 0,
            prev: None,
            prev_time: 0.0,
            fade_t: 0.0,
            fade_dur: 0.0,
            state_time: 0.0,
            started: false,
        }
    }
}

impl SmRuntime {
    /// The cross-fade weight of the current state in `[0,1]`: `0` = fully the
    /// outgoing (`prev`) pose, `1` = fully the current pose (or no fade running).
    pub fn fade_alpha(&self) -> f64 {
        if self.prev.is_some() && self.fade_dur > 0.0 {
            (self.fade_t / self.fade_dur).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Advance the machine by one fixed step of `dt` seconds: lazily enter the
    /// machine's `entry` state, age any running cross-fade, integrate the current
    /// state's time, then fire the **first** ready outgoing transition (in
    /// declaration order — deterministic). Pure given `(sm, ctx, dt)`.
    pub fn advance(&mut self, sm: &StateMachine, ctx: &SmContext, dt: f64) {
        if sm.states.is_empty() {
            return;
        }
        if !self.started {
            self.current = sm.entry.min(sm.states.len() - 1);
            self.prev = None;
            self.fade_t = 0.0;
            self.fade_dur = 0.0;
            self.state_time = 0.0;
            self.started = true;
        }

        // Age an in-progress cross-fade; retire it when complete.
        if self.prev.is_some() {
            self.fade_t += dt;
            if self.fade_dur <= 0.0 || self.fade_t >= self.fade_dur {
                self.prev = None;
                self.fade_t = 0.0;
                self.fade_dur = 0.0;
            }
        }

        self.state_time += dt;

        // Fire the first ready transition out of the current state.
        let period = ctx.period(self.current);
        for tr in sm
            .transitions
            .iter()
            .filter(|tr| tr.from == self.current && tr.to < sm.states.len())
        {
            if tr.ready(ctx, self.state_time, period) {
                // Interrupting a fade snaps: the new fade starts from the state
                // being left; the older fade partner is dropped (v1).
                self.prev = Some(self.current);
                self.prev_time = self.state_time;
                self.current = tr.to;
                self.fade_dur = tr.duration.max(0.0);
                self.fade_t = 0.0;
                self.state_time = 0.0;
                break;
            }
        }
    }
}

/// The `name → f64` (and `state → loop-period`) lookup a machine evaluates
/// against. See the module docs: this is the whole seam between animation and
/// gameplay state. Build it from an actor's Blueprint variables, or (in tests)
/// from a closure.
pub struct SmContext<'a> {
    vars: &'a dyn Fn(&str) -> Option<f64>,
    period: Option<&'a dyn Fn(usize) -> Option<f64>>,
}

impl<'a> SmContext<'a> {
    /// A context that resolves variables via `vars` and has **no** loop-period
    /// resolver — so every `exit_time` gate is treated as satisfied (the v1
    /// no-deadlock fallback). Sufficient for condition-driven machines.
    pub fn new(vars: &'a dyn Fn(&str) -> Option<f64>) -> Self {
        Self { vars, period: None }
    }

    /// A context that also resolves each state's motion loop period (seconds),
    /// enabling precise `exit_time` gating.
    pub fn with_period(
        vars: &'a dyn Fn(&str) -> Option<f64>,
        period: &'a dyn Fn(usize) -> Option<f64>,
    ) -> Self {
        Self {
            vars,
            period: Some(period),
        }
    }

    /// Look up a variable (blend-space param or condition operand).
    pub fn var(&self, name: &str) -> Option<f64> {
        (self.vars)(name)
    }

    /// The loop period (seconds) of state `index`, if a resolver was supplied.
    pub fn period(&self, index: usize) -> Option<f64> {
        self.period.and_then(|p| p(index))
    }
}

/// Sample a [`Motion`] into a [`Pose`] at play-head `t` (seconds), reading any
/// blend-space parameters from `ctx`.
pub fn sample_motion<'c>(
    motion: &Motion,
    skeleton: &Skeleton,
    clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
    ctx: &SmContext,
    looping: bool,
    t: f64,
) -> Pose {
    match motion {
        Motion::Clip(id) => match clips(*id) {
            Some(c) => sample_clip(skeleton, c, t as f32, looping),
            None => Pose::rest(skeleton),
        },
        Motion::Blend1D(space) => {
            let p = ctx.var(&space.param).unwrap_or(0.0);
            sample_blend_space_1d(space, skeleton, clips, p, t)
        }
        Motion::Blend2D(space) => {
            let x = ctx.var(&space.params.0).unwrap_or(0.0);
            let y = ctx.var(&space.params.1).unwrap_or(0.0);
            sample_blend_space_2d(space, skeleton, clips, DVec2::new(x, y), t)
        }
    }
}

/// Evaluate the machine into a [`Pose`] for the current runtime state, blending
/// the outgoing state in during a cross-fade. Does **not** advance the runtime
/// (call [`SmRuntime::advance`] first, or use [`step`]).
pub fn eval_pose<'c>(
    sm: &StateMachine,
    runtime: &SmRuntime,
    skeleton: &Skeleton,
    clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
    ctx: &SmContext,
) -> Pose {
    if sm.states.is_empty() {
        return Pose::rest(skeleton);
    }
    let cur = &sm.states[runtime.current.min(sm.states.len() - 1)];
    let cur_pose = sample_motion(
        &cur.motion,
        skeleton,
        clips,
        ctx,
        cur.looping,
        runtime.state_time * cur.speed,
    );
    match runtime.prev {
        Some(pi) if pi < sm.states.len() => {
            let prev = &sm.states[pi];
            let prev_pose = sample_motion(
                &prev.motion,
                skeleton,
                clips,
                ctx,
                prev.looping,
                runtime.prev_time * prev.speed,
            );
            blend_poses(&prev_pose, &cur_pose, runtime.fade_alpha() as f32)
        }
        _ => cur_pose,
    }
}

/// Advance the machine by `dt` and evaluate the resulting [`Pose`] — the
/// combined per-frame entry point. Equivalent to [`SmRuntime::advance`] followed
/// by [`eval_pose`].
#[allow(clippy::too_many_arguments)]
pub fn step<'c>(
    sm: &StateMachine,
    runtime: &mut SmRuntime,
    skeleton: &Skeleton,
    clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
    ctx: &SmContext,
    dt: f64,
) -> Pose {
    runtime.advance(sm, ctx, dt);
    eval_pose(sm, runtime, skeleton, clips, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blend_space::BlendEntry1D;
    use std::cell::Cell;

    fn two_state_machine() -> StateMachine {
        StateMachine {
            states: vec![
                SmState::clip("idle", [1; 16]),
                SmState::clip("walk", [2; 16]),
            ],
            transitions: vec![
                SmTransition {
                    from: 0,
                    to: 1,
                    duration: 0.2,
                    conditions: vec![SmCondition {
                        var: "moving".into(),
                        op: CmpOp::Gt,
                        value: 0.5,
                    }],
                    exit_time: None,
                },
                SmTransition {
                    from: 1,
                    to: 0,
                    duration: 0.2,
                    conditions: vec![SmCondition {
                        var: "moving".into(),
                        op: CmpOp::Lt,
                        value: 0.5,
                    }],
                    exit_time: None,
                },
            ],
            entry: 0,
        }
    }

    #[test]
    fn cmp_ops_evaluate() {
        assert!(CmpOp::Gt.eval(2.0, 1.0));
        assert!(!CmpOp::Gt.eval(1.0, 1.0));
        assert!(CmpOp::Ge.eval(1.0, 1.0));
        assert!(CmpOp::Lt.eval(0.0, 1.0));
        assert!(CmpOp::Le.eval(1.0, 1.0));
        assert!(CmpOp::Eq.eval(1.0, 1.0));
        assert!(CmpOp::Ne.eval(1.0, 2.0));
        assert!(!CmpOp::Ne.eval(1.0, 1.0));
    }

    #[test]
    fn missing_variable_reads_as_zero() {
        let vars = |_: &str| None;
        let ctx = SmContext::new(&vars);
        let c = SmCondition {
            var: "absent".into(),
            op: CmpOp::Eq,
            value: 0.0,
        };
        assert!(c.eval(&ctx));
    }

    #[test]
    fn transition_fires_once_and_lands_in_target() {
        let sm = two_state_machine();
        let mut rt = SmRuntime::default();
        let moving = Cell::new(0.0f64);
        let vars = |n: &str| {
            if n == "moving" {
                Some(moving.get())
            } else {
                None
            }
        };
        let ctx = SmContext::new(&vars);

        // Not moving → stays in idle.
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 0);
        assert!(rt.prev.is_none());

        // Start moving → transitions to walk exactly once.
        moving.set(1.0);
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1);
        assert_eq!(rt.prev, Some(0));
        // A second step while still moving: no re-fire (already in walk), and the
        // fade retires after its duration.
        rt.advance(&sm, &ctx, 0.2);
        assert_eq!(rt.current, 1);
        assert!(rt.prev.is_none(), "fade should have retired");
    }

    #[test]
    fn crossfade_midpoint_weight_is_half() {
        let sm = two_state_machine();
        let mut rt = SmRuntime::default();
        let vars = |n: &str| if n == "moving" { Some(1.0) } else { None };
        let ctx = SmContext::new(&vars);
        // Fire the transition (fade_dur 0.2), fade_t starts at 0.
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1);
        assert!((rt.fade_alpha() - 0.0).abs() < 1e-9);
        // Advance half the fade: alpha ≈ 0.5. (moving still true → no re-fire out
        // of walk; the reverse transition needs moving<0.5.)
        rt.advance(&sm, &ctx, 0.1);
        assert!((rt.fade_alpha() - 0.5).abs() < 1e-9, "{}", rt.fade_alpha());
    }

    #[test]
    fn exit_time_gates_the_transition() {
        // A single unconditional transition gated at 80% of a 1 s loop.
        let sm = StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition {
                from: 0,
                to: 1,
                duration: 0.0,
                conditions: vec![],
                exit_time: Some(0.8),
            }],
            entry: 0,
        };
        let mut rt = SmRuntime::default();
        let vars = |_: &str| None;
        let period = |_: usize| Some(1.0); // 1 s loop
        let ctx = SmContext::with_period(&vars, &period);

        // 0.5 s in → gate (0.8 s) not met.
        rt.advance(&sm, &ctx, 0.5);
        assert_eq!(rt.current, 0);
        // Cross 0.8 s → fires.
        rt.advance(&sm, &ctx, 0.5);
        assert_eq!(rt.current, 1);
    }

    #[test]
    fn exit_time_without_period_is_satisfied() {
        // Same machine, but no period resolver → the gate is treated as met (the
        // v1 no-deadlock fallback), so it fires on the first step.
        let sm = StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition {
                from: 0,
                to: 1,
                duration: 0.0,
                conditions: vec![],
                exit_time: Some(0.8),
            }],
            entry: 0,
        };
        let mut rt = SmRuntime::default();
        let vars = |_: &str| None;
        let ctx = SmContext::new(&vars);
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1);
    }

    #[test]
    fn two_state_ping_pong_under_a_toggling_var() {
        let sm = two_state_machine();
        let mut rt = SmRuntime::default();
        let moving = Cell::new(1.0f64);
        let vars = |n: &str| {
            if n == "moving" {
                Some(moving.get())
            } else {
                None
            }
        };
        let ctx = SmContext::new(&vars);

        // moving → walk
        rt.advance(&sm, &ctx, 0.3);
        assert_eq!(rt.current, 1);
        // stop → idle
        moving.set(0.0);
        rt.advance(&sm, &ctx, 0.3);
        assert_eq!(rt.current, 0);
        // move again → walk
        moving.set(1.0);
        rt.advance(&sm, &ctx, 0.3);
        assert_eq!(rt.current, 1);
    }

    #[test]
    fn blend1d_motion_reads_param_from_context() {
        // A blend-1D state whose param drives the weights; here we only assert the
        // machine advances and the pose evaluates without panicking (pose values
        // are covered in blend_space tests).
        use crate::skeleton::{Joint, JointTransform, Skeleton};
        use glam::Mat4;
        let sk = Skeleton::new(vec![Joint {
            name: "r".into(),
            parent: None,
            inverse_bind: Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::IDENTITY,
        }])
        .unwrap();
        let sm = StateMachine {
            states: vec![SmState {
                name: "locomotion".into(),
                motion: Motion::Blend1D(BlendSpace1D::new(
                    "speed",
                    vec![
                        BlendEntry1D {
                            pos: 0.0,
                            clip: [1; 16],
                        },
                        BlendEntry1D {
                            pos: 1.0,
                            clip: [2; 16],
                        },
                    ],
                )),
                looping: true,
                speed: 1.0,
                position: (0.0, 0.0),
            }],
            transitions: vec![],
            entry: 0,
        };
        let mut rt = SmRuntime::default();
        let vars = |n: &str| if n == "speed" { Some(0.5) } else { None };
        let ctx = SmContext::new(&vars);
        let clips = |_: ClipRef| -> Option<&AnimClip> { None };
        let pose = step(&sm, &mut rt, &sk, &clips, &ctx, 0.1);
        // No clips resolve → rest pose, but the path ran end-to-end.
        assert_eq!(pose, Pose::rest(&sk));
    }
}
