//! Animation state machines (`.inf_sm` — P11.2 v1, **P29.1 v2**).
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
//! Each state plays a [`Motion`] (a clip, a 1D/2D blend space, or — since v2 — a
//! nested [`StateMachine`]). Each fixed step [`SmRuntime::advance`] samples the
//! declared [`SmParamKind::Trigger`] parameters for rising edges, ages any
//! running cross-fade, integrates the current state's time, then fires the
//! **highest-priority ready** outgoing transition and cross-fades to its target
//! over `duration` seconds. [`eval_pose`] samples the current state (and, mid
//! cross-fade, the outgoing one — and, across an interruption, the one before
//! *that*) into a [`Pose`]. Everything is pure and deterministic given
//! `(sm, runtime, ctx, dt)`.
//!
//! # What v2 changed, and why each one was a defect rather than a wish
//!
//! * **Typed parameters** ([`SmParam`]). v1 compared everything as `f64` with an
//!   epsilon, so `state == 3` on an integer selector was a floating-point
//!   question and a boolean flag had no spelling of its own. A declared
//!   [`SmParamKind`] decides how a compare reads: `Int` is exact, `Bool` is a
//!   threshold at `0.5`, `Float` keeps v1's epsilon, and `Trigger` is an **edge**
//!   rather than a level (below).
//! * **Condition trees** ([`SmCond`]). v1 ANDed a flat list, so an OR had to be
//!   authored as two parallel transitions — which duplicates the duration, the
//!   curve and the exit gate, and which is exactly the kind of hand-copy this
//!   repository keeps having to repair. `And`/`Or`/`Not` over typed compares is
//!   one edge again.
//! * **Priority** ([`SmTransition::priority`]). v1's tie-break was declaration
//!   order and nothing else, so "jump beats run" was expressed by *where the
//!   transition sat in a `Vec`* — invisible in the editor and destroyed by any
//!   re-ordering. Priority is explicit; declaration order is still the tie-break,
//!   so a machine that sets no priorities behaves exactly as v1 did.
//! * **Interruption** ([`SmInterrupt`]). v1 fired a transition mid-cross-fade by
//!   *snapping*: the older fade partner was dropped and the new fade started from
//!   the incoming state's pose, which is a visible pop precisely when the machine
//!   is changing its mind quickly. v2 names both halves — which fades a
//!   transition may cut into ([`InterruptSource`]) and what the outgoing pose is
//!   when it does ([`InterruptBlend`]) — and [`InterruptBlend::Carry`] keeps the
//!   pose continuous by carrying the interrupted partner for one more fade.
//! * **The outgoing pose ADVANCES** ([`SmRuntime::prev_time`]). v1 froze it at
//!   hand-off, so a 0.3 s cross-fade out of a run cycle blended against a single
//!   held frame of that run — the feet stopped moving on the outgoing half while
//!   the incoming half walked. This was written down in v1's own field docs as a
//!   follow-up and stayed one for four phases.
//! * **`exit_time` is live** ([`SmContext::with_clip_lengths`]). v1 had the field
//!   and no resolver in production, so every gate was treated as satisfied. The
//!   context now resolves **clip lengths**, from which the machine derives any
//!   state's period itself — at any nesting depth, which a per-state-index
//!   resolver could not have done.
//! * **Any-state transitions** ([`SmSource::Any`]) and **sub-machines**
//!   ([`Motion::SubMachine`]), the two structural pieces v1 had no spelling for.
//! * **State enter/exit events** ([`SmState::on_enter`] / [`SmState::on_exit`],
//!   reported through [`SmStep`]) — the notify seam P29.4's `anim.*` kit consumes.
//!
//! # The condition/param seam ([`SmContext`])
//!
//! Conditions and blend-space parameters both read **named `f64` values**. The
//! machine never reaches into an interpreter directly; it reads through
//! [`SmContext`], a thin `name → f64` (and, since v2, `clip → seconds`) lookup.
//! The editor Simulate loop and the shipped runtime back it with the actor's
//! Blueprint variables; tests back it with a closure. This is the whole coupling
//! surface between the animation layer and gameplay state — and v2 deliberately
//! did **not** widen it: typing happens against the machine's own declared
//! parameter table, so neither host had to learn a new value type.
//!
//! # Portable math
//!
//! This file is on `tests/portable_pose.rs`' `SIM_PATH` list: an evaluated pose
//! is folded into `state_bytes` and compared between the editor's Simulate and
//! the shipped player. Every v2 blend curve is therefore a **polynomial** —
//! there is no `powf` in an ease, and there never can be.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use glam::DVec2;

use crate::blend_space::{
    blend_weights_1d, blend_weights_2d, sample_blend_space_1d, sample_blend_space_2d, BlendSpace1D,
    BlendSpace2D, ClipRef,
};
use crate::clip::AnimClip;
use crate::pose::{blend_poses, blend_poses_weighted, sample_clip, Pose};
use crate::skeleton::Skeleton;

/// The most parameters one machine may declare.
///
/// The bound is not arbitrary: a [`SmParamKind::Trigger`]'s armed state lives in
/// a `u64` bitmask on [`SmRuntime`], which is `Copy` because it rides an ECS
/// component. One bit per declared parameter (not per trigger) keeps the index
/// arithmetic a plain `params` index with nothing to keep in step.
pub const MAX_PARAMS: usize = 64;

/// The deepest a [`SmCond`] tree may nest.
///
/// Evaluation is recursive and a `.inf_sm` is a file, so a hostile or corrupt one
/// could otherwise blow the stack in the fixed step — the P19 parser-depth law,
/// applied to the one recursive structure this format has. Enforced at decode by
/// [`StateMachine::validate`] **and** in [`eval_condition`], because a machine
/// built in memory never passes through a decoder.
pub const MAX_COND_DEPTH: usize = 16;

/// The most nodes one [`SmCond`] tree may contain (a cost bound rather than a
/// safety one: this is evaluated per candidate transition per fixed step).
pub const MAX_COND_NODES: usize = 256;

/// The deepest a [`Motion::SubMachine`] chain may nest **at decode**.
///
/// [`StateMachine::validate`] refuses a sub-machine inside a sub-machine, so a
/// legal file never reaches 2. The decoder's own bound is one step looser than
/// the model's so that a doubly-nested machine still gets `validate`'s named
/// [`SmError::NestedSubMachine`] rather than a decoder error — and anything
/// deeper never gets built at all.
const MAX_SUB_DEPTH_AT_DECODE: usize = 2;

/// The deepest a [`SmCond`] tree may nest **at decode**.
///
/// Deliberately far above [`MAX_COND_DEPTH`], and the gap is the design: these
/// two bounds answer different questions. `MAX_COND_DEPTH` is the **model's**
/// rule and [`StateMachine::validate`] owns it, reporting
/// [`SmError::ConditionTooDeep`] with the offending transition and the actual
/// depth — a message an author can act on. This one exists only to keep the
/// decoder off the bottom of the stack, so it has to sit above every depth
/// `validate` is willing to describe; a few dozen frames of `Deserialize` cost
/// nothing and a few thousand end the process.
const MAX_COND_DEPTH_AT_DECODE: usize = 64;

// ── the decode-time recursion guards ────────────────────────────────────────
//
// **A `.inf_sm` is a file, and v2 gave it two RECURSIVE shapes** — `SmCond`'s
// `Not`/`And`/`Or` and `Motion::SubMachine` — where v1 had a flat `Vec` of
// compares and three flat motions. serde's derived decoder descends one stack
// frame per level with no bound of its own, so `validate`'s depth check (which
// is the P19 parser-depth law, and is real) runs on a tree the decoder has
// already built: measured, a **7.9 KB** payload of nothing but `Not` tags, and a
// **54 KB** one of nested sub-machines, each take the process down with
// `STATUS_STACK_OVERFLOW` before `migrate` is reached. That is not a refusal —
// it is an abort, in the editor's Content-Drawer scan of every loose file under
// a project root, and in the shipped player reading a cooked pack.
//
// So the bound is enforced HERE, at the only place that is in front of the
// stack: a thread-local descent counter, checked as each recursive field is
// entered, reported as a typed `de::Error` naming the limit. `deserialize_with`
// delegates to the ordinary `Deserialize` impl, so **the wire is unchanged** —
// the same bytes decode to the same value, and only the ones that would have
// aborted now come back as an error.

thread_local! {
    static COND_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SUB_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Decrements its counter however the decode leaves — including the error path,
/// which is the whole reason this is a guard and not a pair of statements.
struct DepthGuard(&'static std::thread::LocalKey<std::cell::Cell<usize>>);

impl Drop for DepthGuard {
    fn drop(&mut self) {
        self.0.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

fn enter_depth<E: serde::de::Error>(
    key: &'static std::thread::LocalKey<std::cell::Cell<usize>>,
    limit: usize,
    what: &str,
) -> Result<DepthGuard, E> {
    let depth = key.with(|c| {
        let d = c.get() + 1;
        c.set(d);
        d
    });
    let guard = DepthGuard(key);
    if depth > limit {
        return Err(E::custom(format!(
            "{what} nests more than {limit} deep — refused at the door, because the \
             decoder would otherwise reach the bottom of the stack before \
             `validate` ever saw the tree"
        )));
    }
    Ok(guard)
}

fn de_cond_box<'de, D>(d: D) -> Result<Box<SmCond>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let _guard =
        enter_depth::<D::Error>(&COND_DEPTH, MAX_COND_DEPTH_AT_DECODE, "a condition tree")?;
    Ok(Box::new(SmCond::deserialize(d)?))
}

fn de_cond_vec<'de, D>(d: D) -> Result<Vec<SmCond>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let _guard =
        enter_depth::<D::Error>(&COND_DEPTH, MAX_COND_DEPTH_AT_DECODE, "a condition tree")?;
    Vec::<SmCond>::deserialize(d)
}

fn de_sub_machine<'de, D>(d: D) -> Result<Box<StateMachine>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let _guard = enter_depth::<D::Error>(
        &SUB_DEPTH,
        MAX_SUB_DEPTH_AT_DECODE,
        "a nested state machine",
    )?;
    Ok(Box::new(StateMachine::deserialize(d)?))
}

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
    /// `==` (within a small epsilon for `Float`; **exact** for `Int` / `Bool`)
    Eq,
    /// `!=`
    Ne,
}

impl CmpOp {
    /// Evaluate `lhs <op> rhs` as **floats**. `Eq`/`Ne` use a small epsilon so
    /// authored integer-ish thresholds compare cleanly.
    ///
    /// This is v1's rule, kept verbatim for [`SmParamKind::Float`] (and for an
    /// undeclared parameter, which reads as `Float` — that is what makes a v1
    /// machine's conditions mean in v2 exactly what they meant in v1).
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

    /// Evaluate `lhs <op> rhs` as **integers** — exactly, with no epsilon.
    ///
    /// The reason [`SmParamKind::Int`] exists: `combo_step == 3` under
    /// [`eval`](Self::eval) is a question about floating-point neighbourhoods,
    /// and a selector that is conceptually an enum should not have one.
    pub fn eval_int(self, lhs: i64, rhs: i64) -> bool {
        match self {
            CmpOp::Gt => lhs > rhs,
            CmpOp::Lt => lhs < rhs,
            CmpOp::Ge => lhs >= rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Eq => lhs == rhs,
            CmpOp::Ne => lhs != rhs,
        }
    }

    /// Evaluate `lhs <op> rhs` as **booleans** (`false < true`, so the ordering
    /// operators are defined rather than refused — an author who writes
    /// `grounded >= true` gets the obvious answer instead of a silent `false`).
    pub fn eval_bool(self, lhs: bool, rhs: bool) -> bool {
        self.eval_int(lhs as i64, rhs as i64)
    }
}

// ── parameters (v2) ─────────────────────────────────────────────────────────

/// What a declared parameter **is**, which is what decides how a compare against
/// it reads.
///
/// The wire discriminants are frozen (`freeze_pins` in this module's tests):
/// bincode writes a unit enum as its variant index, so re-ordering these renames
/// every parameter in every committed `.inf_sm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SmParamKind {
    /// A flag. The host's `f64` reads as `true` above `0.5`.
    Bool,
    /// A whole number (a selector, a combo step, a stance index). Compares are
    /// exact; the host's `f64` is truncated toward zero.
    Int,
    /// A continuous value — v1's only kind, and still the default so an
    /// undeclared parameter behaves exactly as it did.
    #[default]
    Float,
    /// A one-shot **edge**, not a level: see [`SmRuntime::triggers`] for the arm
    /// and consume rules.
    Trigger,
}

/// A value a condition compares against, or a parameter's default.
///
/// Externally tagged (serde's default representation) because bincode cannot
/// decode an internally-tagged enum — the crate-wide law recorded on
/// [`crate::asset::AnimClipAsset`]'s siblings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SmValue {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl SmValue {
    /// The value as an `f64` (`Bool` → `0`/`1`).
    pub fn as_f64(self) -> f64 {
        match self {
            SmValue::Bool(b) => b as i64 as f64,
            SmValue::Int(i) => i as f64,
            SmValue::Float(f) => f,
        }
    }

    /// The value as an `i64`. A `Float` truncates toward zero; a non-finite one
    /// saturates (Rust's `as` is defined for this, so it is deterministic rather
    /// than merely unsurprising).
    pub fn as_i64(self) -> i64 {
        match self {
            SmValue::Bool(b) => b as i64,
            SmValue::Int(i) => i,
            SmValue::Float(f) => f as i64,
        }
    }

    /// The value as a `bool` (`> 0.5` for the numeric kinds, so it agrees with
    /// how a host's `f64` is read).
    pub fn as_bool(self) -> bool {
        match self {
            SmValue::Bool(b) => b,
            SmValue::Int(i) => i != 0,
            SmValue::Float(f) => f > 0.5,
        }
    }

    /// Whether this value is representable — a `Float` that is not finite is not.
    /// [`StateMachine::validate`] refuses those at the door (the C4-4 law: a NaN
    /// that reaches a comparison makes **every** ordering false, which reads as a
    /// machine that simply never transitions).
    pub fn is_finite(self) -> bool {
        match self {
            SmValue::Float(f) => f.is_finite(),
            _ => true,
        }
    }
}

/// One declared parameter: a name, what it is, and what it reads as when the host
/// does not have it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmParam {
    /// The name looked up through [`SmContext`] — the actor's Blueprint variable.
    pub name: String,
    pub kind: SmParamKind,
    /// The value used when the host has no variable of this name. v1's rule was
    /// a hard-coded `0.0`; `Float(0.0)` reproduces it, and is this field's
    /// default.
    pub default: SmValue,
}

impl SmParam {
    /// A declared parameter with the zero default for its kind.
    pub fn new(name: impl Into<String>, kind: SmParamKind) -> Self {
        let default = match kind {
            SmParamKind::Bool | SmParamKind::Trigger => SmValue::Bool(false),
            SmParamKind::Int => SmValue::Int(0),
            SmParamKind::Float => SmValue::Float(0.0),
        };
        Self {
            name: name.into(),
            kind,
            default,
        }
    }

    /// A `Float` parameter — the v1 shape, named.
    pub fn float(name: impl Into<String>) -> Self {
        Self::new(name, SmParamKind::Float)
    }

    /// A `Trigger` parameter.
    pub fn trigger(name: impl Into<String>) -> Self {
        Self::new(name, SmParamKind::Trigger)
    }
}

// ── conditions (v2) ─────────────────────────────────────────────────────────

/// One typed compare: `param <op> value`, read through the parameter's declared
/// [`SmParamKind`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmCompare {
    /// The parameter name looked up through [`SmContext`].
    pub param: String,
    pub op: CmpOp,
    pub value: SmValue,
}

impl SmCompare {
    /// A float compare — v1's `SmCondition`, which is what nearly every authored
    /// condition still is.
    pub fn float(param: impl Into<String>, op: CmpOp, value: f64) -> Self {
        Self {
            param: param.into(),
            op,
            value: SmValue::Float(value),
        }
    }
}

/// A transition's condition: a **tree**, not a list.
///
/// `And(vec![])` and [`SmCond::Always`] are both "no condition"; `Always` is the
/// canonical spelling and is what an unconditional transition carries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum SmCond {
    /// Always true — an unconditional transition (subject only to `exit_time`).
    #[default]
    Always,
    /// A typed compare.
    Compare(SmCompare),
    /// A [`SmParamKind::Trigger`] parameter that is currently **armed**. Reading
    /// it `true` inside the transition that fires **consumes** it.
    Trigger(String),
    /// Every term holds (an empty `And` is true — v1's empty condition list).
    And(#[serde(deserialize_with = "de_cond_vec")] Vec<SmCond>),
    /// Some term holds (an empty `Or` is **false**: nothing holds).
    Or(#[serde(deserialize_with = "de_cond_vec")] Vec<SmCond>),
    Not(#[serde(deserialize_with = "de_cond_box")] Box<SmCond>),
}

impl SmCond {
    /// A float compare, the common leaf — `SmCond::float("speed", CmpOp::Gt, 0.1)`
    /// is v1's whole condition vocabulary in one call.
    pub fn float(param: impl Into<String>, op: CmpOp, value: f64) -> Self {
        SmCond::Compare(SmCompare::float(param, op, value))
    }

    /// An `Int` compare.
    pub fn int(param: impl Into<String>, op: CmpOp, value: i64) -> Self {
        SmCond::Compare(SmCompare {
            param: param.into(),
            op,
            value: SmValue::Int(value),
        })
    }

    /// A `Bool` compare (`param == value`).
    pub fn bool(param: impl Into<String>, value: bool) -> Self {
        SmCond::Compare(SmCompare {
            param: param.into(),
            op: CmpOp::Eq,
            value: SmValue::Bool(value),
        })
    }

    /// The tree's depth (a leaf is 1).
    pub fn depth(&self) -> usize {
        match self {
            SmCond::Always | SmCond::Compare(_) | SmCond::Trigger(_) => 1,
            SmCond::Not(inner) => 1 + inner.depth(),
            SmCond::And(terms) | SmCond::Or(terms) => {
                1 + terms.iter().map(SmCond::depth).max().unwrap_or(0)
            }
        }
    }

    /// The number of nodes in the tree.
    pub fn nodes(&self) -> usize {
        match self {
            SmCond::Always | SmCond::Compare(_) | SmCond::Trigger(_) => 1,
            SmCond::Not(inner) => 1 + inner.nodes(),
            SmCond::And(terms) | SmCond::Or(terms) => {
                1 + terms.iter().map(SmCond::nodes).sum::<usize>()
            }
        }
    }

    /// Every parameter name this tree reads, in walk order (duplicates kept —
    /// the caller decides whether it cares).
    pub fn params(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_params(&mut out);
        out
    }

    fn collect_params<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            SmCond::Always => {}
            SmCond::Compare(c) => out.push(&c.param),
            SmCond::Trigger(name) => out.push(name),
            SmCond::Not(inner) => inner.collect_params(out),
            SmCond::And(terms) | SmCond::Or(terms) => {
                for t in terms {
                    t.collect_params(out);
                }
            }
        }
    }

    /// The **flat AND of float compares** this tree is, if it is one.
    ///
    /// The editor's v1 condition inspector edits a list of `var op value` rows,
    /// and the overwhelming majority of authored machines are exactly that. This
    /// is the door the Ring-2 DTO uses to decide whether the flat view is lossless
    /// — returning `None` is what stops a v2 tree being silently flattened (and
    /// therefore destroyed) by a save from a UI that cannot draw it.
    pub fn as_flat_and(&self) -> Option<Vec<SmCompare>> {
        fn leaf(c: &SmCond) -> Option<SmCompare> {
            match c {
                SmCond::Compare(cmp) if matches!(cmp.value, SmValue::Float(_)) => Some(cmp.clone()),
                _ => None,
            }
        }
        match self {
            SmCond::Always => Some(Vec::new()),
            SmCond::And(terms) => terms.iter().map(leaf).collect(),
            other => leaf(other).map(|c| vec![c]),
        }
    }

    /// The tree a flat list of compares lowers to (the inverse of
    /// [`as_flat_and`](Self::as_flat_and) for the shapes it accepts).
    pub fn from_flat_and(terms: Vec<SmCompare>) -> Self {
        match terms.len() {
            0 => SmCond::Always,
            1 => SmCond::Compare(terms.into_iter().next().expect("len == 1")),
            _ => SmCond::And(terms.into_iter().map(SmCond::Compare).collect()),
        }
    }
}

// ── blend curves & profiles (v2) ────────────────────────────────────────────

/// How a cross-fade's linear progress is shaped.
///
/// **Every one of these is a polynomial**, and that is a rule rather than a
/// coincidence: this file is on the portable-math ban list (see the module docs),
/// so an ease written as `t.powf(2.0)` would fail the gate — and would be a real
/// cross-platform trace divergence, because a pose blended at a different alpha
/// is different bytes.
///
/// Wire discriminants are frozen (`freeze_pins`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BlendCurve {
    /// `t` — v1's behaviour, and still the default.
    #[default]
    Linear,
    /// `t²` — slow to leave the outgoing pose.
    EaseIn,
    /// `1 − (1 − t)²` — quick to leave, slow to arrive.
    EaseOut,
    /// `t²(3 − 2t)` (smoothstep) — zero slope at both ends, which is what makes a
    /// short fade read as deliberate rather than as a lurch.
    EaseInOut,
    /// Hold the outgoing pose, then cut at the end of the duration. The honest
    /// spelling of "no blend, but wait first" — a hard cut is `duration = 0`.
    Step,
}

impl BlendCurve {
    /// Shape a linear `t ∈ [0,1]` (values outside are clamped first).
    pub fn apply(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            BlendCurve::Linear => t,
            BlendCurve::EaseIn => t * t,
            BlendCurve::EaseOut => {
                let u = 1.0 - t;
                1.0 - u * u
            }
            BlendCurve::EaseInOut => t * t * (3.0 - 2.0 * t),
            BlendCurve::Step => {
                if t >= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// One joint's share of a cross-fade (a **blend profile** entry).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointBlendWeight {
    /// Skeleton joint index.
    pub joint: u16,
    /// Multiplier on the transition's alpha for this joint, `0..=1`. `0` holds
    /// the outgoing pose on that joint for the whole fade; `1` is the unmasked
    /// default.
    pub weight: f32,
}

/// A named per-joint scale on a cross-fade's weight.
///
/// This is what makes "the upper body snaps to the aim pose while the legs keep
/// walking" one transition instead of a layer stack — and it is the piece a
/// per-transition `duration` alone cannot express, because the difference between
/// those two halves is spatial, not temporal.
///
/// A joint **absent** from `weights` blends at the transition's own alpha. That,
/// and not a full row of `1.0`, is the meaningful default: a profile authored
/// against one rig then names only the joints it actually masks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlendProfile {
    pub name: String,
    pub weights: Vec<JointBlendWeight>,
}

impl BlendProfile {
    /// A profile that scales the named joints and leaves the rest alone.
    pub fn new(name: impl Into<String>, weights: Vec<JointBlendWeight>) -> Self {
        Self {
            name: name.into(),
            weights,
        }
    }

    /// This profile's per-joint alphas for a skeleton of `joints` joints, given
    /// the transition's own `alpha`.
    pub fn alphas(&self, joints: usize, alpha: f32) -> Vec<f32> {
        let mut out = vec![alpha; joints];
        for w in &self.weights {
            if let Some(slot) = out.get_mut(w.joint as usize) {
                *slot = alpha * w.weight.clamp(0.0, 1.0);
            }
        }
        out
    }
}

// ── interruption (v2) ───────────────────────────────────────────────────────

/// Which in-progress cross-fade a transition may cut into.
///
/// Wire discriminants are frozen (`freeze_pins`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum InterruptSource {
    /// Not at all: while a cross-fade is running this transition cannot fire. The
    /// spelling for "let the blend finish" — which v1 could not say, so a rapid
    /// input produced a chain of half-finished fades.
    None,
    /// When the machine is settled, or out of the state being faded **into**.
    /// v1's rule, and the default.
    #[default]
    Destination,
    /// …and also out of the state being faded **out of**, which is what lets a
    /// machine change its mind about a transition already underway.
    SourceOrDestination,
}

/// What the outgoing pose is when a transition interrupts a running cross-fade.
///
/// Wire discriminants are frozen (`freeze_pins`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum InterruptBlend {
    /// The new fade starts from the incoming state's pose alone — **v1's snap**,
    /// kept because it is occasionally what an author wants (and because naming
    /// it is what makes the alternative a choice).
    Snap,
    /// The new fade starts from the pose that was actually being rendered: the
    /// interrupted partner is carried for one more fade at the alpha it had
    /// reached. The default, because pose continuity is the thing v1 got wrong.
    #[default]
    Carry,
}

/// A transition's interruption policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SmInterrupt {
    pub source: InterruptSource,
    pub blend: InterruptBlend,
}

impl SmInterrupt {
    /// v1's exact behaviour: fires out of the destination state, and snaps.
    pub fn v1() -> Self {
        Self {
            source: InterruptSource::Destination,
            blend: InterruptBlend::Snap,
        }
    }
}

// ── the model ───────────────────────────────────────────────────────────────

/// What a state plays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Motion {
    /// A single animation clip (raw GUID).
    Clip(ClipRef),
    /// A 1D blend space.
    Blend1D(BlendSpace1D),
    /// A 2D blend space.
    Blend2D(BlendSpace2D),
    /// A **nested machine** (v2), sharing the parent's parameters.
    ///
    /// One level: a sub-machine's own states may not themselves be sub-machines,
    /// and [`StateMachine::validate`] refuses that. The bound is not aesthetic —
    /// [`SmRuntime`] is `Copy` because it lives inside an ECS component, so the
    /// nested play state is a fixed inline [`SmSub`] rather than a `Box`, and one
    /// slot is what "fixed" means.
    SubMachine(#[serde(deserialize_with = "de_sub_machine")] Box<StateMachine>),
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
    /// Event names emitted the step this state is **entered** (v2).
    ///
    /// Reported through [`SmStep::entered_events`] rather than dispatched: this
    /// crate is Ring 0 and pure, and "what a notify does" is a gameplay question
    /// that belongs to P29.4's `anim.*` kit. What lives here is the only half that
    /// can be deterministic — *which* events fired, in which fixed step.
    #[serde(default)]
    pub on_enter: Vec<String>,
    /// Event names emitted the step this state is **exited** (v2).
    #[serde(default)]
    pub on_exit: Vec<String>,
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
            on_enter: Vec::new(),
            on_exit: Vec::new(),
        }
    }

    /// The same, placed on the canvas.
    pub fn clip_at(name: impl Into<String>, clip: ClipRef, position: (f32, f32)) -> Self {
        Self {
            position,
            ..Self::clip(name, clip)
        }
    }

    /// A state playing a nested machine.
    pub fn sub_machine(name: impl Into<String>, machine: StateMachine) -> Self {
        Self {
            motion: Motion::SubMachine(Box::new(machine)),
            ..Self::clip(name, [0u8; 16])
        }
    }
}

/// Where a transition leaves **from**.
///
/// Wire discriminants are frozen (`freeze_pins`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmSource {
    /// One state, by index — v1's only spelling.
    State(usize),
    /// **Any** state (UE's "any state" node): the escape hatch that turns N
    /// hand-copied edges into one. `exclude_self` stops the machine re-entering
    /// the state it is already in, which is what an author means almost always —
    /// so it is `true` in [`SmTransition::any`].
    Any { exclude_self: bool },
}

impl SmSource {
    /// Whether this source matches a machine currently in `state`.
    pub fn matches(self, state: usize, to: usize) -> bool {
        match self {
            SmSource::State(i) => i == state,
            SmSource::Any { exclude_self } => !(exclude_self && to == state),
        }
    }

    /// The state index this leaves, if it names one.
    pub fn state(self) -> Option<usize> {
        match self {
            SmSource::State(i) => Some(i),
            SmSource::Any { .. } => None,
        }
    }
}

/// One transition edge between two states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmTransition {
    /// Source state, or **any** state (v2).
    pub from: SmSource,
    /// Destination state index.
    pub to: usize,
    /// Cross-fade duration in seconds (`0` = a hard cut).
    pub duration: f64,
    /// The condition tree. [`SmCond::Always`] = unconditional (subject only to
    /// `exit_time`).
    pub condition: SmCond,
    /// Optional normalized (`[0,1]`) exit-time gate against the source state's
    /// motion period: the transition may only fire once the source has played at
    /// least this fraction of a loop. `None` = fireable any time.
    ///
    /// **Live since v2.** v1 shipped the field with no production period
    /// resolver, so every gate read as satisfied; a [`SmContext`] built with
    /// [`with_clip_lengths`](SmContext::with_clip_lengths) makes it mean what it
    /// says. A context that still cannot resolve lengths keeps the v1 fallback
    /// (gate satisfied), so an unresolved machine never deadlocks.
    pub exit_time: Option<f64>,
    /// Higher fires first (v2). Ties break by **declaration order**, which is v1's
    /// whole rule — so a machine that leaves every priority at `0` behaves exactly
    /// as it did.
    #[serde(default)]
    pub priority: i32,
    /// What this transition may interrupt, and how (v2).
    #[serde(default)]
    pub interrupt: SmInterrupt,
    /// How this transition's cross-fade is shaped (v2).
    #[serde(default)]
    pub curve: BlendCurve,
    /// Index into [`StateMachine::profiles`] — a per-joint mask on this fade (v2).
    #[serde(default)]
    pub profile: Option<usize>,
}

impl SmTransition {
    /// An unconditional transition out of one state.
    pub fn new(from: usize, to: usize, duration: f64) -> Self {
        Self {
            from: SmSource::State(from),
            to,
            duration,
            condition: SmCond::Always,
            exit_time: None,
            priority: 0,
            interrupt: SmInterrupt::default(),
            curve: BlendCurve::default(),
            profile: None,
        }
    }

    /// A transition on one float compare — v1's shape, which is what nearly every
    /// authored edge in this repository still is.
    pub fn on(from: usize, to: usize, duration: f64, param: &str, op: CmpOp, value: f64) -> Self {
        Self {
            condition: SmCond::float(param, op, value),
            ..Self::new(from, to, duration)
        }
    }

    /// An **any-state** transition (excluding the state it targets).
    pub fn any(to: usize, duration: f64) -> Self {
        Self {
            from: SmSource::Any { exclude_self: true },
            ..Self::new(0, to, duration)
        }
    }

    /// Builder: set the condition tree.
    pub fn when(mut self, condition: SmCond) -> Self {
        self.condition = condition;
        self
    }

    /// Builder: set the priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Builder: set the blend curve.
    pub fn with_curve(mut self, curve: BlendCurve) -> Self {
        self.curve = curve;
        self
    }

    /// Builder: set the exit-time gate.
    pub fn with_exit_time(mut self, exit_time: f64) -> Self {
        self.exit_time = Some(exit_time);
        self
    }

    /// Builder: set the interruption policy.
    pub fn with_interrupt(mut self, interrupt: SmInterrupt) -> Self {
        self.interrupt = interrupt;
        self
    }

    /// Builder: set the blend profile index.
    pub fn with_profile(mut self, profile: usize) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Whether this transition is ready to fire: its condition tree holds **and**
    /// the exit-time gate is satisfied.
    ///
    /// `state_time` is seconds spent in the source state; `period` is that state's
    /// motion period (see [`motion_period`]) — `None`/non-positive means "period
    /// unknown", in which case the exit-time gate is treated as satisfied (the
    /// documented v1 fallback, so an unresolved-duration machine never deadlocks).
    /// `sm` is the machine whose **parameter table** applies -- the root machine,
    /// for a transition inside a sub-machine.
    pub fn ready(
        &self,
        sm: &StateMachine,
        ctx: &SmContext,
        triggers: u64,
        state_time: f64,
        period: Option<f64>,
    ) -> bool {
        if !eval_condition(&self.condition, sm, ctx, triggers) {
            return false;
        }
        match (self.exit_time, period) {
            (Some(frac), Some(p)) if p > 0.0 => state_time >= frac * p,
            _ => true,
        }
    }
}

/// A whole state machine: states, transitions, the entry state, and — since v2 —
/// the declared parameter table and blend profiles.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct StateMachine {
    pub states: Vec<SmState>,
    pub transitions: Vec<SmTransition>,
    /// Index of the state entered on start.
    pub entry: usize,
    /// Declared parameters (v2). A condition may name a parameter that is **not**
    /// declared; it then reads as [`SmParamKind::Float`] defaulting to `0.0`,
    /// which is v1's rule exactly. Declaring one is what buys typing.
    #[serde(default)]
    pub params: Vec<SmParam>,
    /// Named per-joint blend masks a transition may point at (v2).
    #[serde(default)]
    pub profiles: Vec<BlendProfile>,
}

/// Why a [`StateMachine`] is not usable.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SmError {
    #[error("entry state {entry} is out of range ({states} states)")]
    EntryOutOfRange { entry: usize, states: usize },
    #[error("transition {index} targets state {to}, which does not exist ({states} states)")]
    TargetOutOfRange {
        index: usize,
        to: usize,
        states: usize,
    },
    #[error("transition {index} leaves state {from}, which does not exist ({states} states)")]
    SourceOutOfRange {
        index: usize,
        from: usize,
        states: usize,
    },
    #[error("transition {index} names blend profile {profile}, which does not exist ({profiles} profiles)")]
    ProfileOutOfRange {
        index: usize,
        profile: usize,
        profiles: usize,
    },
    #[error("transition {index} has a non-finite or negative duration ({duration})")]
    BadDuration { index: usize, duration: f64 },
    #[error("transition {index} has an exit time of {exit_time}, which is not in [0,1]")]
    BadExitTime { index: usize, exit_time: f64 },
    #[error("transition {index}'s condition nests {depth} deep (the limit is {MAX_COND_DEPTH})")]
    ConditionTooDeep { index: usize, depth: usize },
    #[error("transition {index}'s condition has {nodes} nodes (the limit is {MAX_COND_NODES})")]
    ConditionTooLarge { index: usize, nodes: usize },
    #[error("transition {index}'s condition compares against a non-finite value")]
    NonFiniteCompare { index: usize },
    #[error("state {index} ({name}) has a non-finite play speed ({speed})")]
    BadSpeed {
        index: usize,
        name: String,
        speed: f64,
    },
    #[error("{count} parameters declared (the limit is {MAX_PARAMS})")]
    TooManyParams { count: usize },
    #[error("parameter {index} has an empty name")]
    EmptyParamName { index: usize },
    #[error("parameter `{name}` is declared twice")]
    DuplicateParam { name: String },
    #[error("parameter `{name}` has a non-finite default")]
    NonFiniteParamDefault { name: String },
    #[error("blend profile {index} weights joint {joint} at {weight}, which is not in [0,1]")]
    BadProfileWeight {
        index: usize,
        joint: u16,
        weight: f32,
    },
    #[error("state {index} ({name}) nests a sub-machine inside a sub-machine, which v2 does not support")]
    NestedSubMachine { index: usize, name: String },
    #[error("a sub-machine declares {count} parameters of its own; a nested machine shares its parent's table")]
    SubMachineParams { count: usize },
    #[error("state {index} ({name})'s sub-machine is invalid: {source}")]
    SubMachine {
        index: usize,
        name: String,
        #[source]
        source: Box<SmError>,
    },
}

impl StateMachine {
    /// An empty machine (no states) with entry 0.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Look a declared parameter up by name, with its index (the bit position its
    /// trigger state occupies in [`SmRuntime::triggers`]).
    ///
    /// **Bounded by [`MAX_PARAMS`]**, and that is not belt-and-braces: the index
    /// this returns is shifted into a `u64` by `trigger_armed` and
    /// `collect_true_triggers`, and `1u64 << 64` is a debug panic (`attempt to
    /// shift left with overflow`) and a *silent wrap to bit 0* in release.
    /// [`StateMachine::validate`] refuses a table past the limit, so no decoded
    /// machine can get here — but the editor builds machines in memory that never
    /// pass a decoder, which is the same argument that put the depth bound in
    /// [`eval_condition`] as well as in `validate`. A parameter past the limit is
    /// simply not found, so it reads as an undeclared `Float` at `0.0` (v1's
    /// rule) rather than taking the fixed step down.
    pub fn param(&self, name: &str) -> Option<(usize, &SmParam)> {
        self.params
            .iter()
            .enumerate()
            .take(MAX_PARAMS)
            .find(|(_, p)| p.name == name)
    }

    /// **The structural questions a decoded machine must answer** — asked by
    /// `crate::asset::StateMachineAsset`'s `migrate`, so a corrupt or hostile
    /// `.inf_sm` is refused at the door rather than in the fixed step.
    ///
    /// The campaign's U6 standard: a `migrate` that only compares a version number
    /// certifies nothing. What is checked here is everything the evaluator would
    /// otherwise have to trust — indices in range, floats finite, the recursive
    /// structure bounded — plus the one bound that is a property of *this*
    /// runtime rather than of the data (a sub-machine inside a sub-machine has
    /// nowhere to keep its play state).
    ///
    /// An **empty** machine is valid: it is what a freshly created document is,
    /// and [`SmRuntime::advance`] returns immediately on one.
    pub fn validate(&self) -> Result<(), SmError> {
        self.validate_at_depth(0)
    }

    fn validate_at_depth(&self, depth: usize) -> Result<(), SmError> {
        // A sub-machine evaluates against its PARENT's parameter table (see
        // `advance_play`'s `root`), so a table of its own would be data that
        // nothing reads -- and the silent kind, because every name in it would
        // still resolve, as an undeclared `Float`.
        if depth > 0 && !self.params.is_empty() {
            return Err(SmError::SubMachineParams {
                count: self.params.len(),
            });
        }
        if self.params.len() > MAX_PARAMS {
            return Err(SmError::TooManyParams {
                count: self.params.len(),
            });
        }
        for (i, p) in self.params.iter().enumerate() {
            if p.name.is_empty() {
                return Err(SmError::EmptyParamName { index: i });
            }
            if self.params[..i].iter().any(|q| q.name == p.name) {
                return Err(SmError::DuplicateParam {
                    name: p.name.clone(),
                });
            }
            if !p.default.is_finite() {
                return Err(SmError::NonFiniteParamDefault {
                    name: p.name.clone(),
                });
            }
        }
        for (i, prof) in self.profiles.iter().enumerate() {
            for w in &prof.weights {
                if !(w.weight.is_finite() && (0.0..=1.0).contains(&w.weight)) {
                    return Err(SmError::BadProfileWeight {
                        index: i,
                        joint: w.joint,
                        weight: w.weight,
                    });
                }
            }
        }
        if self.states.is_empty() {
            // Nothing else can be out of range, and an empty machine is a legal
            // (inert) document.
            return Ok(());
        }
        if self.entry >= self.states.len() {
            return Err(SmError::EntryOutOfRange {
                entry: self.entry,
                states: self.states.len(),
            });
        }
        for (i, s) in self.states.iter().enumerate() {
            if !s.speed.is_finite() {
                return Err(SmError::BadSpeed {
                    index: i,
                    name: s.name.clone(),
                    speed: s.speed,
                });
            }
            if let Motion::SubMachine(inner) = &s.motion {
                if depth >= 1 {
                    return Err(SmError::NestedSubMachine {
                        index: i,
                        name: s.name.clone(),
                    });
                }
                inner
                    .validate_at_depth(depth + 1)
                    .map_err(|e| SmError::SubMachine {
                        index: i,
                        name: s.name.clone(),
                        source: Box::new(e),
                    })?;
            }
        }
        for (i, t) in self.transitions.iter().enumerate() {
            if t.to >= self.states.len() {
                return Err(SmError::TargetOutOfRange {
                    index: i,
                    to: t.to,
                    states: self.states.len(),
                });
            }
            if let Some(from) = t.from.state() {
                if from >= self.states.len() {
                    return Err(SmError::SourceOutOfRange {
                        index: i,
                        from,
                        states: self.states.len(),
                    });
                }
            }
            if let Some(p) = t.profile {
                if p >= self.profiles.len() {
                    return Err(SmError::ProfileOutOfRange {
                        index: i,
                        profile: p,
                        profiles: self.profiles.len(),
                    });
                }
            }
            if !(t.duration.is_finite() && t.duration >= 0.0) {
                return Err(SmError::BadDuration {
                    index: i,
                    duration: t.duration,
                });
            }
            if let Some(x) = t.exit_time {
                if !(x.is_finite() && (0.0..=1.0).contains(&x)) {
                    return Err(SmError::BadExitTime {
                        index: i,
                        exit_time: x,
                    });
                }
            }
            let depth = t.condition.depth();
            if depth > MAX_COND_DEPTH {
                return Err(SmError::ConditionTooDeep { index: i, depth });
            }
            let nodes = t.condition.nodes();
            if nodes > MAX_COND_NODES {
                return Err(SmError::ConditionTooLarge { index: i, nodes });
            }
            if !condition_values_finite(&t.condition) {
                return Err(SmError::NonFiniteCompare { index: i });
            }
        }
        Ok(())
    }
}

fn condition_values_finite(c: &SmCond) -> bool {
    match c {
        SmCond::Always | SmCond::Trigger(_) => true,
        SmCond::Compare(cmp) => cmp.value.is_finite(),
        SmCond::Not(inner) => condition_values_finite(inner),
        SmCond::And(terms) | SmCond::Or(terms) => terms.iter().all(condition_values_finite),
    }
}

// ── the live runtime ────────────────────────────────────────────────────────

/// The nested play state of a [`Motion::SubMachine`] state (v2).
///
/// A flat mirror of the parent's play fields, minus the carry: an interruption
/// **inside** a sub-machine always snaps. That is a bound rather than a design —
/// carrying costs a third state index and this struct is inlined into a `Copy`
/// ECS component — and it is written here so it is a decision.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SmSub {
    /// The parent state index this play state belongs to.
    pub owner: usize,
    pub current: usize,
    pub prev: Option<usize>,
    pub prev_time: f64,
    pub fade_t: f64,
    pub fade_dur: f64,
    pub state_time: f64,
    pub started: bool,
    pub curve: BlendCurve,
}

/// The play fields shared by the machine and its sub-machine, so the advance rule
/// is written **once** (the mirror law, applied inside one file).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Play {
    current: usize,
    prev: Option<usize>,
    prev_time: f64,
    fade_t: f64,
    fade_dur: f64,
    state_time: f64,
    started: bool,
    carry: Option<usize>,
    carry_time: f64,
    carry_alpha: f64,
    curve: BlendCurve,
    profile: Option<usize>,
}

impl Play {
    /// The **raw** (uncurved) cross-fade progress in `[0,1]`.
    fn alpha(&self) -> f64 {
        if self.prev.is_some() && self.fade_dur > 0.0 {
            (self.fade_t / self.fade_dur).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// The curved cross-fade weight actually applied to the pose.
    fn curved_alpha(&self) -> f64 {
        self.curve.apply(self.alpha())
    }
}

/// What one [`SmRuntime::advance`] did — the deterministic report both the notify
/// seam (P29.4) and the tests read.
///
/// Returned rather than dispatched: this crate is pure, and the only half of a
/// notify that can be a function of the step history is *which* events fired.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SmStep {
    /// Index into [`StateMachine::transitions`] of the transition that fired.
    pub fired: Option<usize>,
    /// The state left this step (`None` if none was).
    pub exited: Option<usize>,
    /// The state entered this step — including the **entry** state on the first
    /// step, which v1 reported to nobody.
    pub entered: Option<usize>,
    /// The exited state's [`SmState::on_exit`] names, then the entered state's
    /// [`SmState::on_enter`] names — in that order, because an exit happens before
    /// the entry it causes.
    pub events: Vec<String>,
    /// The same for the sub-machine, if one stepped.
    pub sub_fired: Option<usize>,
    pub sub_exited: Option<usize>,
    pub sub_entered: Option<usize>,
    /// Trigger parameters **consumed** this step, by index into
    /// [`StateMachine::params`].
    pub consumed: u64,
}

impl SmStep {
    /// Whether anything at all happened (a cheap early-out for callers that only
    /// care about change).
    pub fn is_quiet(&self) -> bool {
        self.fired.is_none()
            && self.entered.is_none()
            && self.exited.is_none()
            && self.sub_fired.is_none()
    }

    /// The entered state's events only (the half a P29.4 notify handler wants when
    /// it is looking for "a state began").
    pub fn entered_events<'a>(&'a self, sm: &'a StateMachine) -> &'a [String] {
        match self.entered.and_then(|i| sm.states.get(i)) {
            Some(s) => &s.on_enter,
            None => &[],
        }
    }
}

/// The live play state of a machine on one entity.
///
/// Small + `Copy` so it can live inside an ECS component
/// (`inf_ecs::components::AnimStateMachine`); **not** serialized — it is rebuilt
/// each play session, which is why v2 could grow it without touching the scene
/// schema.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmRuntime {
    /// The active state index.
    pub current: usize,
    /// The state being cross-faded *out of*, if a fade is in progress.
    pub prev: Option<usize>,
    /// The `prev` state's play-head. **Advances with the fade since v2** — v1
    /// froze it at hand-off, so the outgoing half of every cross-fade was a single
    /// held frame.
    pub prev_time: f64,
    /// Elapsed cross-fade time (seconds).
    pub fade_t: f64,
    /// Total cross-fade duration (seconds).
    pub fade_dur: f64,
    /// Seconds spent in the current state.
    pub state_time: f64,
    /// Whether the runtime has been initialized to the machine's entry state.
    pub started: bool,
    /// The partner of an **interrupted** fade, carried so the new fade starts from
    /// the pose that was being rendered ([`InterruptBlend::Carry`]).
    ///
    /// **One deep**, which is the bound a `Copy` runtime buys: a second
    /// interruption while a carry is live replaces it with the partner of the
    /// fade just cut, and the oldest pose stops contributing.
    ///
    /// This block used to say the machine "snaps" there. The P29.1 audit
    /// measured it: **40.5°** of discontinuity across the second interruption,
    /// against **63°** for [`InterruptBlend::Snap`] on the same fixture — so it
    /// is not a snap, and it is not free either. (The first interruption, which
    /// the slot *does* serve, costs 0.06°.) A bound worth writing down is worth
    /// measuring, and both numbers live in
    /// `interrupting_an_interruption_degrades_toward_the_newer_partner`. A
    /// machine that changes its mind twice inside one fade is the case a second
    /// slot would buy, and it is not bought.
    pub carry: Option<usize>,
    /// The carried state's play-head.
    pub carry_time: f64,
    /// The curved alpha the interrupted fade had reached, frozen.
    pub carry_alpha: f64,
    /// The running fade's curve (copied off the transition that started it, so a
    /// live fade is unaffected by a later edit to the machine).
    pub curve: BlendCurve,
    /// The running fade's blend profile index.
    pub profile: Option<usize>,
    /// **Armed** trigger parameters, one bit per [`StateMachine::params`] index.
    ///
    /// # The read point, precisely
    ///
    /// 1. At the **top** of [`advance`](Self::advance), before any transition is
    ///    evaluated, every declared [`SmParamKind::Trigger`] parameter's level is
    ///    sampled once. A **rising** edge (`<= 0.5` last step, `> 0.5` now) arms
    ///    its bit. This is the only place a trigger becomes set, so a trigger
    ///    raised and cleared within one fixed step is still seen exactly once.
    /// 2. A [`SmCond::Trigger`] leaf (and a [`SmCompare`] against a trigger
    ///    parameter) reads the **armed bit**, never the raw level. Evaluation is
    ///    therefore side-effect free, which is what lets every candidate
    ///    transition be scanned in priority order without the answer depending on
    ///    how many were scanned first.
    /// 3. When a transition fires, every trigger its condition tree read as
    ///    **true** is disarmed — and only those. A trigger under a `Not`, or in
    ///    the untaken half of an `Or`, is untouched, so one event cannot be eaten
    ///    by a transition that did not act on it.
    /// 4. An armed trigger nothing consumes **stays armed**, which is the whole
    ///    point: a jump pressed one step before the machine could act on it still
    ///    lands. Re-arming needs a fresh rising edge.
    pub triggers: u64,
    /// The trigger levels seen last step, for the edge detection in rule 1.
    pub trigger_levels: u64,
    /// The nested play state of a [`Motion::SubMachine`] state.
    pub sub: SmSub,
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
            carry: None,
            carry_time: 0.0,
            carry_alpha: 0.0,
            curve: BlendCurve::Linear,
            profile: None,
            triggers: 0,
            trigger_levels: 0,
            sub: SmSub::default(),
        }
    }
}

impl SmRuntime {
    fn play(&self) -> Play {
        Play {
            current: self.current,
            prev: self.prev,
            prev_time: self.prev_time,
            fade_t: self.fade_t,
            fade_dur: self.fade_dur,
            state_time: self.state_time,
            started: self.started,
            carry: self.carry,
            carry_time: self.carry_time,
            carry_alpha: self.carry_alpha,
            curve: self.curve,
            profile: self.profile,
        }
    }

    fn set_play(&mut self, p: Play) {
        self.current = p.current;
        self.prev = p.prev;
        self.prev_time = p.prev_time;
        self.fade_t = p.fade_t;
        self.fade_dur = p.fade_dur;
        self.state_time = p.state_time;
        self.started = p.started;
        self.carry = p.carry;
        self.carry_time = p.carry_time;
        self.carry_alpha = p.carry_alpha;
        self.curve = p.curve;
        self.profile = p.profile;
    }

    /// The cross-fade weight of the current state in `[0,1]`: `0` = fully the
    /// outgoing (`prev`) pose, `1` = fully the current pose (or no fade running).
    ///
    /// **Raw** — the transition's [`BlendCurve`] is applied by
    /// [`curved_alpha`](Self::curved_alpha), which is what
    /// [`eval_pose`] blends with. Kept linear here because it is also the fade's
    /// *progress*, which is what a UI draws.
    pub fn fade_alpha(&self) -> f64 {
        self.play().alpha()
    }

    /// The cross-fade weight actually applied to the pose: `fade_alpha` shaped by
    /// the running transition's [`BlendCurve`].
    pub fn curved_alpha(&self) -> f64 {
        self.play().curved_alpha()
    }

    /// Advance the machine by one fixed step of `dt` seconds. Pure given
    /// `(sm, ctx, dt)`; returns what it did ([`SmStep`]).
    ///
    /// In order:
    ///  1. lazily enter the machine's `entry` state (reported, since v2);
    ///  2. sample declared triggers for rising edges (see [`triggers`](Self::triggers));
    ///  3. age any running cross-fade — **including the outgoing play-head**, and
    ///     any carried interruption partner;
    ///  4. integrate the current state's time, and step a sub-machine if the
    ///     current state is one;
    ///  5. fire the **highest-priority ready** outgoing transition (ties by
    ///     declaration order — v1's whole rule, kept as the tie-break), honouring
    ///     its [`SmInterrupt`] policy;
    ///  6. consume the triggers that transition read as true.
    pub fn advance(&mut self, sm: &StateMachine, ctx: &SmContext, dt: f64) -> SmStep {
        let mut step = SmStep::default();
        if sm.states.is_empty() {
            return step;
        }

        // 2. Trigger edges, before anything reads them.
        self.sample_triggers(sm, ctx);

        // **One snapshot of the armed set for the whole step.** Every condition
        // evaluated below -- the machine's and its sub-machine's -- reads THIS
        // value, and the consumption at the end clears against it. Consuming as
        // we went would make the parent's answer depend on whether the
        // sub-machine had already eaten a trigger, which is an ordering nobody
        // authoring a machine can see.
        let armed = self.triggers;

        let mut play = self.play();
        let entered_now = advance_play(&mut play, sm, sm, ctx, dt, armed, true, &mut step);

        // 4b. The sub-machine of whatever state we are now in.
        self.set_play(play);
        let sub_consumed = self.step_sub_machine(sm, ctx, dt, entered_now, armed, &mut step);

        // 6. Consumption, from the transitions that actually fired -- the
        //    machine's own, and the sub-machine's, which reads the same
        //    parameters and must therefore consume from the same set.
        let mut consumed = sub_consumed;
        if let Some(i) = step.fired {
            collect_true_triggers(
                &sm.transitions[i].condition,
                sm,
                ctx,
                armed,
                true,
                &mut consumed,
            );
        }
        self.triggers &= !consumed;
        step.consumed = consumed;
        step
    }

    /// Rule 1 of the trigger contract: rising edges arm, and nothing else does.
    fn sample_triggers(&mut self, sm: &StateMachine, ctx: &SmContext) {
        let mut levels = 0u64;
        for (i, p) in sm.params.iter().enumerate().take(MAX_PARAMS) {
            if p.kind != SmParamKind::Trigger {
                continue;
            }
            let raw = ctx.var(&p.name).unwrap_or_else(|| p.default.as_f64());
            if raw > 0.5 {
                levels |= 1 << i;
            }
        }
        let rising = levels & !self.trigger_levels;
        self.triggers |= rising;
        self.trigger_levels = levels;
    }

    /// Step the nested machine of the current state, entering it when the parent
    /// arrives on a [`Motion::SubMachine`] state.
    ///
    /// The slot **follows `current`**, and is left alone otherwise — which is what
    /// keeps a sub-machine posing while it is being faded *out of*. Fading from
    /// one sub-machine state into another is the case one slot cannot serve: the
    /// outgoing one poses at rest for the length of that fade. Written down rather
    /// than discovered.
    fn step_sub_machine(
        &mut self,
        sm: &StateMachine,
        ctx: &SmContext,
        dt: f64,
        entered_now: bool,
        armed: u64,
        step: &mut SmStep,
    ) -> u64 {
        let Some(state) = sm.states.get(self.current) else {
            return 0;
        };
        let Motion::SubMachine(inner) = &state.motion else {
            return 0;
        };
        if entered_now || self.sub.owner != self.current || !self.sub.started {
            self.sub = SmSub {
                owner: self.current,
                ..SmSub::default()
            };
        }
        let mut play = Play {
            current: self.sub.current,
            prev: self.sub.prev,
            prev_time: self.sub.prev_time,
            fade_t: self.sub.fade_t,
            fade_dur: self.sub.fade_dur,
            state_time: self.sub.state_time,
            started: self.sub.started,
            curve: self.sub.curve,
            ..Play::default()
        };
        let mut sub_step = SmStep::default();
        // `allow_carry = false`: a sub-machine's interruption always snaps.
        // `sm` (the parent) is the PARAMETER table -- see `advance_play`.
        advance_play(&mut play, inner, sm, ctx, dt, armed, false, &mut sub_step);
        self.sub = SmSub {
            owner: self.current,
            current: play.current,
            prev: play.prev,
            prev_time: play.prev_time,
            fade_t: play.fade_t,
            fade_dur: play.fade_dur,
            state_time: play.state_time,
            started: play.started,
            curve: play.curve,
        };
        step.sub_fired = sub_step.fired;
        step.sub_exited = sub_step.exited;
        step.sub_entered = sub_step.entered;
        step.events.extend(sub_step.events);
        // A trigger a nested transition read is consumed exactly like one the
        // parent read -- they are the same parameter, out of the same table.
        let mut consumed = 0u64;
        if let Some(j) = sub_step.fired {
            collect_true_triggers(
                &inner.transitions[j].condition,
                sm,
                ctx,
                armed,
                true,
                &mut consumed,
            );
        }
        consumed
    }
}

/// The one advance rule, over the play fields the machine and its sub-machine
/// share. Returns whether a state was **entered** this step.
///
/// `sm` supplies the **states, transitions and profiles**; `root` supplies the
/// **parameter table**. For a top-level machine they are the same value. For a
/// sub-machine they are not, and that is the whole point: a nested machine
/// shares its parent's parameters (that is what "shares the parent's parameters"
/// in [`Motion::SubMachine`] means), so resolving `stance` inside it against its
/// own empty table would silently read every typed parameter as an undeclared
/// `Float` and every trigger as unarmed. [`StateMachine::validate`] refuses a
/// sub-machine that declares parameters of its own, so there is exactly one
/// table and no question of which one wins.
#[allow(clippy::too_many_arguments)]
fn advance_play(
    play: &mut Play,
    sm: &StateMachine,
    root: &StateMachine,
    ctx: &SmContext,
    dt: f64,
    triggers: u64,
    allow_carry: bool,
    step: &mut SmStep,
) -> bool {
    let mut entered_now = false;
    if !play.started {
        *play = Play {
            current: sm.entry.min(sm.states.len() - 1),
            started: true,
            ..Play::default()
        };
        step.entered = Some(play.current);
        push_events(step, sm, None, Some(play.current));
        entered_now = true;
    }

    // 3. Age an in-progress cross-fade; retire it when complete.
    if play.prev.is_some() {
        play.fade_t += dt;
        // **The outgoing pose advances** — the v1 frozen frame.
        play.prev_time += dt;
        if play.carry.is_some() {
            play.carry_time += dt;
        }
        if play.fade_dur <= 0.0 || play.fade_t >= play.fade_dur {
            play.prev = None;
            play.carry = None;
            play.fade_t = 0.0;
            play.fade_dur = 0.0;
            play.carry_alpha = 0.0;
        }
    }

    play.state_time += dt;

    // 5. Fire the highest-priority ready transition.
    let fading = play.prev.is_some();
    let mut best: Option<(usize, i32)> = None;
    for (i, tr) in sm.transitions.iter().enumerate() {
        if tr.to >= sm.states.len() {
            continue;
        }
        // Eligibility: does this transition leave where the machine is, and — if a
        // cross-fade is running — is it allowed to cut into it?
        let eligible = if !fading {
            tr.from.matches(play.current, tr.to)
        } else {
            match tr.interrupt.source {
                InterruptSource::None => false,
                InterruptSource::Destination => tr.from.matches(play.current, tr.to),
                // Either end of the running fade may be left. `Any` is already
                // covered by the first arm (it matches every state), so the
                // second only has to add the named-source case.
                InterruptSource::SourceOrDestination => {
                    tr.from.matches(play.current, tr.to)
                        || play.prev.is_some_and(|p| tr.from == SmSource::State(p))
                }
            }
        };
        if !eligible {
            continue;
        }
        // **The exit-time gate is always about the state the machine is IN.**
        // `state_time` is the only elapsed clock the runtime keeps, and it is the
        // current state's; a `SourceOrDestination` transition leaving `prev`
        // therefore gates on `current`'s phase. Stated rather than left to be
        // inferred from which index happens to be passed here.
        let period = sm
            .states
            .get(play.current)
            .and_then(|s| motion_period(&s.motion, ctx));
        if !tr.ready(root, ctx, triggers, play.state_time, period) {
            continue;
        }
        // Higher priority wins; declaration order breaks the tie (v1's rule).
        if best.is_none_or(|(_, p)| tr.priority > p) {
            best = Some((i, tr.priority));
        }
    }

    if let Some((i, _)) = best {
        let tr = &sm.transitions[i];
        let left = play.current;
        // The interruption: what the outgoing pose of the NEW fade is.
        let (carry, carry_time, carry_alpha) = match (play.prev, allow_carry, tr.interrupt.blend) {
            (Some(p), true, InterruptBlend::Carry) => {
                (Some(p), play.prev_time, play.curved_alpha())
            }
            _ => (None, 0.0, 0.0),
        };
        *play = Play {
            current: tr.to,
            prev: Some(left),
            prev_time: play.state_time,
            fade_t: 0.0,
            fade_dur: tr.duration.max(0.0),
            state_time: 0.0,
            started: true,
            carry,
            carry_time,
            carry_alpha,
            curve: tr.curve,
            profile: tr.profile,
        };
        step.fired = Some(i);
        step.exited = Some(left);
        step.entered = Some(tr.to);
        push_events(step, sm, Some(left), Some(tr.to));
        entered_now = true;
    }
    entered_now
}

fn push_events(
    step: &mut SmStep,
    sm: &StateMachine,
    exited: Option<usize>,
    entered: Option<usize>,
) {
    if let Some(s) = exited.and_then(|i| sm.states.get(i)) {
        step.events.extend(s.on_exit.iter().cloned());
    }
    if let Some(s) = entered.and_then(|i| sm.states.get(i)) {
        step.events.extend(s.on_enter.iter().cloned());
    }
}

// ── condition evaluation ────────────────────────────────────────────────────

/// The `f64` a parameter currently reads as, through the context and its declared
/// default.
fn param_value(sm: &StateMachine, ctx: &SmContext, name: &str) -> f64 {
    match ctx.var(name) {
        Some(v) => v,
        // An undeclared parameter reads `0.0` — v1's rule exactly.
        None => sm
            .param(name)
            .map(|(_, p)| p.default.as_f64())
            .unwrap_or(0.0),
    }
}

/// Whether the named parameter is an **armed** trigger.
fn trigger_armed(sm: &StateMachine, name: &str, triggers: u64) -> bool {
    match sm.param(name) {
        Some((i, p)) if p.kind == SmParamKind::Trigger => triggers & (1 << i) != 0,
        _ => false,
    }
}

/// Evaluate a condition tree.
///
/// `sm` is the machine whose **parameter table** applies. For a condition inside
/// a [`Motion::SubMachine`] that is the **root** machine, not the nested one:
/// a sub-machine shares its parent's parameters, and
/// [`StateMachine::validate`] refuses one that declares its own.
///
/// `triggers` is [`SmRuntime::triggers`] — the armed set. **Side-effect free**:
/// consumption is decided by `collect_true_triggers` on the transition that
/// actually fires, which is what keeps the answer independent of how many
/// candidates were scanned first.
pub fn eval_condition(cond: &SmCond, sm: &StateMachine, ctx: &SmContext, triggers: u64) -> bool {
    // **Unreadable is `false` for the WHOLE tree, not for the subtree.** The
    // first cut returned `false` from the over-deep frame and let the answer keep
    // propagating — and `Not` inverts it, so a tree that nested an *even* number
    // of `Not`s past the bound came back **true**: "a condition that cannot be
    // read" was treated as satisfied, which is the precise thing the bound
    // exists to prevent. Measured before the fix: 18, 20 and 32 `Not`s over a
    // false leaf all evaluated `true`. `None` here means "this tree cannot be
    // read", it propagates through every combinator, and it lands as `false`
    // exactly once.
    eval_at(cond, sm, ctx, triggers, 0).unwrap_or(false)
}

fn eval_at(
    cond: &SmCond,
    sm: &StateMachine,
    ctx: &SmContext,
    triggers: u64,
    depth: usize,
) -> Option<bool> {
    if depth > MAX_COND_DEPTH {
        // A machine built in memory never passed a decoder, so the depth bound is
        // enforced here too — as "unreadable", which the caller turns into
        // `false` after every enclosing `Not` has had its say.
        return None;
    }
    Some(match cond {
        SmCond::Always => true,
        SmCond::Trigger(name) => trigger_armed(sm, name, triggers),
        SmCond::Compare(cmp) => match sm.param(&cmp.param).map(|(_, p)| p.kind) {
            Some(SmParamKind::Trigger) => cmp
                .op
                .eval_bool(trigger_armed(sm, &cmp.param, triggers), cmp.value.as_bool()),
            Some(SmParamKind::Bool) => cmp
                .op
                .eval_bool(param_value(sm, ctx, &cmp.param) > 0.5, cmp.value.as_bool()),
            Some(SmParamKind::Int) => cmp
                .op
                .eval_int(param_value(sm, ctx, &cmp.param) as i64, cmp.value.as_i64()),
            // `Float`, and every *undeclared* parameter: v1's comparison.
            _ => cmp
                .op
                .eval(param_value(sm, ctx, &cmp.param), cmp.value.as_f64()),
        },
        SmCond::Not(inner) => !eval_at(inner, sm, ctx, triggers, depth + 1)?,
        SmCond::And(terms) => {
            let mut all = true;
            for t in terms {
                // Every term is asked, even after one is false: an unreadable
                // term must poison the tree rather than be short-circuited past.
                all &= eval_at(t, sm, ctx, triggers, depth + 1)?;
            }
            all
        }
        SmCond::Or(terms) => {
            let mut any = false;
            for t in terms {
                any |= eval_at(t, sm, ctx, triggers, depth + 1)?;
            }
            any
        }
    })
}

/// Rule 3 of the trigger contract: the triggers a tree read as **true**.
///
/// # `Not` flips the sense, it does not end the walk
///
/// The rule is "a trigger the fired tree read as TRUE is consumed", and `Not`
/// decides what "read as true" means for everything under it. The first cut
/// simply stopped at a `Not` — which is right for `Not(Trigger(x))`, where the
/// transition fired *because* `x` was unset and has nothing to consume, and
/// wrong for `Not(Not(Trigger(x)))`, where it fired **because `x` was armed** and
/// then left it armed for ever. Measured before the fix: that tree fires,
/// reports `consumed = 0`, and re-fires on the next step off the same press.
///
/// So the walk carries `positive` — whether an even number of `Not`s stands
/// between here and the root — and only collects where it is `true`. A subtree
/// under an odd number is where the transition fired on something being
/// **false**, and nothing there is consumed, which is the original rule
/// preserved exactly. An `Or`'s untaken terms are still not consumed either,
/// which is what stops one press being eaten by an unrelated branch.
fn collect_true_triggers(
    cond: &SmCond,
    sm: &StateMachine,
    ctx: &SmContext,
    triggers: u64,
    positive: bool,
    out: &mut u64,
) {
    // `Not` is handled first, because it is the one node that means something
    // under an odd count as well as an even one.
    if let SmCond::Not(inner) = cond {
        collect_true_triggers(inner, sm, ctx, triggers, !positive, out);
        return;
    }
    if !positive {
        // An odd number of `Not`s: whatever is here was read as FALSE on the
        // path that fired, so there is no armed bit to spend.
        return;
    }
    match cond {
        SmCond::Always | SmCond::Not(_) => {}
        SmCond::Trigger(name) => {
            if let Some((i, p)) = sm.param(name) {
                if p.kind == SmParamKind::Trigger && triggers & (1 << i) != 0 {
                    *out |= 1 << i;
                }
            }
        }
        SmCond::Compare(cmp) => {
            if let Some((i, p)) = sm.param(&cmp.param) {
                if p.kind == SmParamKind::Trigger
                    && triggers & (1 << i) != 0
                    && eval_condition(cond, sm, ctx, triggers)
                {
                    *out |= 1 << i;
                }
            }
        }
        SmCond::And(terms) => {
            for t in terms {
                collect_true_triggers(t, sm, ctx, triggers, positive, out);
            }
        }
        SmCond::Or(terms) => {
            for t in terms {
                if eval_condition(t, sm, ctx, triggers) {
                    collect_true_triggers(t, sm, ctx, triggers, positive, out);
                }
            }
        }
    }
}

// ── the context ─────────────────────────────────────────────────────────────

/// The `name → f64` (and, since v2, `clip → seconds`) lookup a machine evaluates
/// against. See the module docs: this is the whole seam between animation and
/// gameplay state. Build it from an actor's Blueprint variables, or (in tests)
/// from a closure.
pub struct SmContext<'a> {
    vars: &'a dyn Fn(&str) -> Option<f64>,
    clip_len: Option<&'a dyn Fn(ClipRef) -> Option<f64>>,
}

impl<'a> SmContext<'a> {
    /// A context that resolves variables via `vars` and has **no** clip-length
    /// resolver — so every `exit_time` gate is treated as satisfied (the v1
    /// no-deadlock fallback). Sufficient for condition-driven machines.
    pub fn new(vars: &'a dyn Fn(&str) -> Option<f64>) -> Self {
        Self {
            vars,
            clip_len: None,
        }
    }

    /// A context that also resolves **clip lengths** (seconds), from which the
    /// machine derives any state's motion period itself — which is what makes
    /// `exit_time` live.
    ///
    /// Clip lengths rather than a `state index → period` resolver, deliberately: a
    /// per-index resolver cannot answer for a state **inside a sub-machine**, and
    /// a resolver that silently answered the wrong machine's index would be worse
    /// than the v1 fallback it replaced.
    pub fn with_clip_lengths(
        vars: &'a dyn Fn(&str) -> Option<f64>,
        clip_len: &'a dyn Fn(ClipRef) -> Option<f64>,
    ) -> Self {
        Self {
            vars,
            clip_len: Some(clip_len),
        }
    }

    /// Look up a variable (blend-space param or condition operand).
    pub fn var(&self, name: &str) -> Option<f64> {
        (self.vars)(name)
    }

    /// The length of a clip in seconds, if a resolver was supplied.
    pub fn clip_len(&self, clip: ClipRef) -> Option<f64> {
        self.clip_len.and_then(|f| f(clip))
    }

    /// Whether this context can resolve clip lengths at all — the difference
    /// between a live `exit_time` and the v1 fallback.
    pub fn resolves_periods(&self) -> bool {
        self.clip_len.is_some()
    }
}

/// How long one loop of a [`Motion`] lasts, in seconds — the denominator an
/// `exit_time` gate is a fraction of.
///
/// * a clip is its own duration;
/// * a blend space is the **weight-blended** duration of its contributing clips,
///   which is exactly the denominator `blend_space`'s sampler already normalizes
///   its phase against, so a gate at `0.8` means the same 80% the feet are at;
/// * a sub-machine has none (its states have their own, and which one is a
///   question about live state rather than about the motion) — so an `exit_time`
///   on a transition out of a sub-machine state reads as satisfied, the v1
///   fallback, and is documented as such rather than guessed at.
///
/// `None` whenever the context cannot resolve lengths.
pub fn motion_period(motion: &Motion, ctx: &SmContext) -> Option<f64> {
    if !ctx.resolves_periods() {
        return None;
    }
    match motion {
        Motion::Clip(id) => ctx.clip_len(*id).filter(|d| *d > 0.0),
        Motion::Blend1D(space) => {
            let p = ctx.var(&space.param).unwrap_or(0.0);
            let weights = blend_weights_1d(space, p);
            weighted_period(&weights, ctx, |i| space.entries[i].clip)
        }
        Motion::Blend2D(space) => {
            let x = ctx.var(&space.params.0).unwrap_or(0.0);
            let y = ctx.var(&space.params.1).unwrap_or(0.0);
            let weights = blend_weights_2d(space, DVec2::new(x, y));
            weighted_period(&weights, ctx, |i| space.entries[i].clip)
        }
        Motion::SubMachine(_) => None,
    }
}

fn weighted_period(
    weights: &[(usize, f64)],
    ctx: &SmContext,
    clip_of: impl Fn(usize) -> ClipRef,
) -> Option<f64> {
    let mut total = 0.0;
    let mut any = false;
    for &(i, w) in weights {
        if let Some(d) = ctx.clip_len(clip_of(i)) {
            total += d * w;
            any = true;
        }
    }
    (any && total > 0.0).then_some(total)
}

/// Every clip one [`Motion`] plays: a single clip, every entry of a 1D/2D blend
/// space, or — since v2 — everything a nested machine plays. **Private** —
/// [`StateMachine::clip_refs`] is the door, and a second public spelling of the
/// same walk is what the P24.1 audit asked to be closed rather than left as a
/// `pub fn` with no callers.
fn motion_clip_refs(motion: &Motion, out: &mut Vec<ClipRef>) {
    match motion {
        Motion::Clip(c) => out.push(*c),
        Motion::Blend1D(space) => out.extend(space.entries.iter().map(|e| e.clip)),
        Motion::Blend2D(space) => out.extend(space.entries.iter().map(|e| e.clip)),
        Motion::SubMachine(inner) => {
            for s in &inner.states {
                motion_clip_refs(&s.motion, out);
            }
        }
    }
}

impl StateMachine {
    /// Every clip GUID this machine's states play, deduplicated and in ascending
    /// byte order (deterministic).
    ///
    /// **The one walk.** A machine names its clips *inside its own payload* and no
    /// component references them, so every consumer that has to close the
    /// `state machine → clip` edge closes it from here: the cook's dependency
    /// closure (`inf_packager::cook::asset_deps`), and — through
    /// `inf_editor_core::simulate::machine_clip_refs`, which is this in `Uuid`
    /// spelling — the PIE payload builder and the editor Simulate resolver. It was
    /// written twice before P24.1: the cook closed the edge and the PIE payload
    /// did not, which was invisible for as long as nothing evaluated the machine's
    /// pose, and became "the character animates in the shipped build and stands
    /// still in the preview" the moment something did.
    ///
    /// **Since v2 it descends into sub-machines**, for the same reason: a clip a
    /// nested machine plays that the cook did not close over is a clip the shipped
    /// pack does not contain.
    pub fn clip_refs(&self) -> std::collections::BTreeSet<ClipRef> {
        let mut out = Vec::new();
        for s in &self.states {
            motion_clip_refs(&s.motion, &mut out);
        }
        out.into_iter().collect()
    }
}

/// Sample a [`Motion`] into a [`Pose`] at play-head `t` (seconds), reading any
/// blend-space parameters from `ctx`.
///
/// A [`Motion::SubMachine`] samples as the **rest pose**: a nested machine's pose
/// is a function of its own live play state, which this signature does not carry.
/// [`eval_pose`] is the door that has it.
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
        Motion::SubMachine(_) => Pose::rest(skeleton),
    }
}

/// Evaluate the machine into a [`Pose`] for the current runtime state, blending
/// the outgoing state in during a cross-fade. Does **not** advance the runtime
/// (call [`SmRuntime::advance`] first, or use [`step`]).
///
/// Up to **three** poses since v2: the incoming state, the outgoing one, and —
/// across an [`InterruptBlend::Carry`] interruption — the partner of the fade
/// that was cut into. The blend is shaped by the running transition's
/// [`BlendCurve`] and masked by its [`BlendProfile`], if it named one.
pub fn eval_pose<'c>(
    sm: &StateMachine,
    runtime: &SmRuntime,
    skeleton: &Skeleton,
    clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
    ctx: &SmContext,
) -> Pose {
    eval_play_pose(
        sm,
        &runtime.play(),
        Some(&runtime.sub),
        skeleton,
        clips,
        ctx,
    )
}

fn eval_play_pose<'c>(
    sm: &StateMachine,
    play: &Play,
    sub: Option<&SmSub>,
    skeleton: &Skeleton,
    clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
    ctx: &SmContext,
) -> Pose {
    if sm.states.is_empty() {
        return Pose::rest(skeleton);
    }
    let pose_of = |index: usize, t: f64| -> Pose {
        let Some(state) = sm.states.get(index) else {
            return Pose::rest(skeleton);
        };
        match &state.motion {
            Motion::SubMachine(inner) => match sub {
                // The slot follows `current`, and lingers on the state being faded
                // out of — so a sub-machine keeps posing through its own exit.
                Some(s) if s.owner == index && s.started => {
                    let inner_play = Play {
                        current: s.current,
                        prev: s.prev,
                        prev_time: s.prev_time,
                        fade_t: s.fade_t,
                        fade_dur: s.fade_dur,
                        state_time: s.state_time,
                        started: s.started,
                        curve: s.curve,
                        ..Play::default()
                    };
                    eval_play_pose(inner, &inner_play, None, skeleton, clips, ctx)
                }
                _ => Pose::rest(skeleton),
            },
            m => sample_motion(m, skeleton, clips, ctx, state.looping, t * state.speed),
        }
    };

    let cur_pose = pose_of(play.current.min(sm.states.len() - 1), play.state_time);
    let Some(pi) = play.prev.filter(|p| *p < sm.states.len()) else {
        return cur_pose;
    };
    let prev_pose = pose_of(pi, play.prev_time);
    // The interrupted partner, if this fade carries one.
    let outgoing = match play.carry.filter(|c| *c < sm.states.len()) {
        Some(ci) => {
            let carry_pose = pose_of(ci, play.carry_time);
            blend_poses(&carry_pose, &prev_pose, play.carry_alpha as f32)
        }
        None => prev_pose,
    };
    let alpha = play.curved_alpha() as f32;
    match play.profile.and_then(|p| sm.profiles.get(p)) {
        Some(profile) => {
            let alphas = profile.alphas(cur_pose.len().max(outgoing.len()), alpha);
            blend_poses_weighted(&outgoing, &cur_pose, &alphas)
        }
        None => blend_poses(&outgoing, &cur_pose, alpha),
    }
}

/// Advance the machine by `dt` and evaluate the resulting [`Pose`] — the
/// combined per-frame entry point. Equivalent to [`SmRuntime::advance`] followed
/// by [`eval_pose`]; the [`SmStep`] is discarded (use the two calls when the
/// notifies matter).
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
    use crate::clip::{Interpolation, JointTrack, QuatTrack};
    use crate::skeleton::{Joint, JointTransform};
    use glam::{Mat4, Quat, Vec3};
    use std::cell::Cell;

    fn two_state_machine() -> StateMachine {
        StateMachine {
            states: vec![
                SmState::clip("idle", [1; 16]),
                SmState::clip("walk", [2; 16]),
            ],
            transitions: vec![
                SmTransition::on(0, 1, 0.2, "moving", CmpOp::Gt, 0.5),
                SmTransition::on(1, 0, 0.2, "moving", CmpOp::Lt, 0.5),
            ],
            entry: 0,
            ..Default::default()
        }
    }

    /// A straight 2-joint chain, enough to tell two poses apart.
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

    /// A clip that rotates joint `j` from 0 to `deg` over `secs`.
    fn sweep(name: &str, j: u16, deg: f32, secs: f32) -> AnimClip {
        let mut jt = JointTrack::new(j);
        jt.rotation = Some(QuatTrack::new(
            vec![0.0, secs],
            vec![
                Quat::IDENTITY.to_array(),
                Quat::from_rotation_z(deg.to_radians()).to_array(),
            ],
            Interpolation::Linear,
        ));
        AnimClip::new(name, vec![jt])
    }

    // ── v1 behaviour that must not have moved ───────────────────────────────

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
        let sm = StateMachine::default();
        let vars = |_: &str| None;
        let ctx = SmContext::new(&vars);
        let c = SmCond::float("absent", CmpOp::Eq, 0.0);
        assert!(eval_condition(&c, &sm, &ctx, 0));
    }

    #[test]
    fn transition_fires_once_and_lands_in_target() {
        let sm = two_state_machine();
        let mut rt = SmRuntime::default();
        let moving = Cell::new(0.0f64);
        let vars = |n: &str| (n == "moving").then(|| moving.get());
        let ctx = SmContext::new(&vars);

        let step = rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 0);
        assert!(rt.prev.is_none());
        // The entry state is REPORTED now, which v1 told nobody.
        assert_eq!(step.entered, Some(0));

        moving.set(1.0);
        let step = rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1);
        assert_eq!(rt.prev, Some(0));
        assert_eq!(
            (step.fired, step.exited, step.entered),
            (Some(0), Some(0), Some(1))
        );

        let step = rt.advance(&sm, &ctx, 0.2);
        assert_eq!(rt.current, 1);
        assert!(rt.prev.is_none(), "fade should have retired");
        assert!(step.is_quiet());
    }

    #[test]
    fn crossfade_midpoint_weight_is_half() {
        let sm = two_state_machine();
        let mut rt = SmRuntime::default();
        let vars = |n: &str| (n == "moving").then_some(1.0);
        let ctx = SmContext::new(&vars);
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1);
        assert!((rt.fade_alpha() - 0.0).abs() < 1e-9);
        rt.advance(&sm, &ctx, 0.1);
        assert!((rt.fade_alpha() - 0.5).abs() < 1e-9, "{}", rt.fade_alpha());
        // Linear is the default curve, so the applied weight is the raw one.
        assert!((rt.curved_alpha() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn two_state_ping_pong_under_a_toggling_var() {
        let sm = two_state_machine();
        let mut rt = SmRuntime::default();
        let moving = Cell::new(1.0f64);
        let vars = |n: &str| (n == "moving").then(|| moving.get());
        let ctx = SmContext::new(&vars);

        rt.advance(&sm, &ctx, 0.3);
        assert_eq!(rt.current, 1);
        moving.set(0.0);
        rt.advance(&sm, &ctx, 0.3);
        assert_eq!(rt.current, 0);
        moving.set(1.0);
        rt.advance(&sm, &ctx, 0.3);
        assert_eq!(rt.current, 1);
    }

    #[test]
    fn clip_refs_cover_single_clips_blend_entries_and_sub_machines() {
        use crate::blend_space::BlendSpace1D;
        let inner = StateMachine {
            states: vec![SmState::clip("crouch", [3; 16])],
            entry: 0,
            ..Default::default()
        };
        let sm = StateMachine {
            states: vec![
                SmState::clip("idle", [1; 16]),
                SmState {
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
                    ..SmState::clip("locomotion", [0; 16])
                },
                SmState::sub_machine("stance", inner),
            ],
            entry: 0,
            ..Default::default()
        };
        let refs: Vec<ClipRef> = sm.clip_refs().into_iter().collect();
        // Deduplicated (idle's clip is also a blend entry), sorted, and the
        // sub-machine's clip is in the closure — the cook ships exactly this. A
        // sub-machine state contributes no clip of its own, only its states'.
        assert_eq!(refs, vec![[1u8; 16], [2u8; 16], [3u8; 16]]);
        assert!(StateMachine::empty().clip_refs().is_empty());
    }

    #[test]
    fn blend1d_motion_reads_param_from_context() {
        use crate::blend_space::BlendSpace1D;
        let sk = chain();
        let sm = StateMachine {
            states: vec![SmState {
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
                ..SmState::clip("locomotion", [0; 16])
            }],
            entry: 0,
            ..Default::default()
        };
        let mut rt = SmRuntime::default();
        let vars = |n: &str| (n == "speed").then_some(0.5);
        let ctx = SmContext::new(&vars);
        let clips = |_: ClipRef| -> Option<&AnimClip> { None };
        let pose = step(&sm, &mut rt, &sk, &clips, &ctx, 0.1);
        assert_eq!(pose, Pose::rest(&sk));
    }

    // ── v2: exit_time is live ───────────────────────────────────────────────

    #[test]
    fn exit_time_gates_the_transition_against_a_resolved_clip_length() {
        let sm = StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition::new(0, 1, 0.0).with_exit_time(0.8)],
            entry: 0,
            ..Default::default()
        };
        let mut rt = SmRuntime::default();
        let vars = |_: &str| None;
        let lens = |_: ClipRef| Some(1.0); // a 1 s loop
        let ctx = SmContext::with_clip_lengths(&vars, &lens);

        rt.advance(&sm, &ctx, 0.5);
        assert_eq!(rt.current, 0, "0.5 s in, the 0.8 s gate is not met");
        rt.advance(&sm, &ctx, 0.5);
        assert_eq!(rt.current, 1, "crossing 0.8 s fires it");
    }

    /// The v1 fallback survives **exactly** where it was: a context with no
    /// length resolver treats every gate as satisfied, so an unresolved machine
    /// cannot deadlock.
    #[test]
    fn exit_time_without_a_length_resolver_is_satisfied() {
        let sm = StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition::new(0, 1, 0.0).with_exit_time(0.8)],
            entry: 0,
            ..Default::default()
        };
        let mut rt = SmRuntime::default();
        let vars = |_: &str| None;
        let ctx = SmContext::new(&vars);
        assert!(!ctx.resolves_periods());
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1);
    }

    /// …and so does a machine whose clips the host **has** a resolver for but
    /// cannot resolve (an unloaded pack): the gate is unknown, not infinite.
    #[test]
    fn exit_time_against_an_unresolvable_clip_is_satisfied() {
        let sm = StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition::new(0, 1, 0.0).with_exit_time(0.8)],
            entry: 0,
            ..Default::default()
        };
        let mut rt = SmRuntime::default();
        let vars = |_: &str| None;
        let lens = |_: ClipRef| None;
        let ctx = SmContext::with_clip_lengths(&vars, &lens);
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1);
    }

    /// A blend space's period is the **weight-blended** duration — the same
    /// denominator its own sampler normalizes phase against, so a gate at 0.8 is
    /// the same 80% the feet are at.
    #[test]
    fn a_blend_spaces_period_is_the_weight_blended_duration() {
        use crate::blend_space::BlendSpace1D;
        let space = BlendSpace1D::new(
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
        );
        let motion = Motion::Blend1D(space);
        let lens = |c: ClipRef| Some(if c == [1u8; 16] { 1.0 } else { 0.5 });
        let at = |p: f64| {
            let vars = move |n: &str| (n == "speed").then_some(p);
            // The closure has to outlive the context, so the period is taken here.
            motion_period(&motion, &SmContext::with_clip_lengths(&vars, &lens))
        };
        assert_eq!(at(0.0), Some(1.0), "fully the 1 s clip");
        assert_eq!(at(1.0), Some(0.5), "fully the 0.5 s clip");
        let mid = at(0.5).unwrap();
        assert!((mid - 0.75).abs() < 1e-9, "halfway is 0.75 s, got {mid}");
    }

    // ── v2: the outgoing pose advances ──────────────────────────────────────

    /// **The frozen-frame defect.** v1 held `prev_time` at hand-off, so the
    /// outgoing half of a cross-fade was one still frame. The measurement is the
    /// outgoing pose at two points inside the same fade: identical under v1's
    /// rule, different under v2's.
    #[test]
    fn the_outgoing_pose_advances_through_the_crossfade() {
        let sk = chain();
        let run = sweep("run", 1, 90.0, 1.0);
        let idle = sweep("idle", 2, 0.0, 1.0);
        let clips = |c: ClipRef| -> Option<&AnimClip> {
            match c {
                x if x == [1u8; 16] => Some(&run),
                x if x == [2u8; 16] => Some(&idle),
                _ => None,
            }
        };
        let sm = StateMachine {
            states: vec![
                SmState::clip("run", [1; 16]),
                SmState::clip("idle", [2; 16]),
            ],
            // A 1 s fade, so the samples below sit well inside it.
            transitions: vec![SmTransition::on(0, 1, 1.0, "stop", CmpOp::Gt, 0.5)],
            entry: 0,
            ..Default::default()
        };
        let stop = Cell::new(0.0f64);
        let vars = |n: &str| (n == "stop").then(|| stop.get());
        let ctx = SmContext::new(&vars);

        let mut rt = SmRuntime::default();
        rt.advance(&sm, &ctx, 0.1);
        stop.set(1.0);
        rt.advance(&sm, &ctx, 0.1); // fires; prev = run at t = 0.2
        let t0 = rt.prev_time;
        rt.advance(&sm, &ctx, 0.3);
        let t1 = rt.prev_time;
        assert!(
            (t1 - t0 - 0.3).abs() < 1e-9,
            "the outgoing play-head must have moved by dt: {t0} -> {t1}"
        );
        // …and it is VISIBLE through the real evaluator. `frozen` is v1's rule
        // reconstructed exactly — the same runtime with `prev_time` held at
        // hand-off — so the two poses differ by nothing except the defect.
        let mut frozen = rt;
        frozen.prev_time = t0;
        let v2 = eval_pose(&sm, &rt, &sk, &clips, &ctx);
        let v1 = eval_pose(&sm, &frozen, &sk, &clips, &ctx);
        let ang = v1.locals[1]
            .rotation_quat()
            .angle_between(v2.locals[1].rotation_quat())
            .to_degrees();
        assert!(
            ang > 5.0,
            "the evaluated pose is the same whether the outgoing play-head advanced \
             or not ({ang} deg apart) — the v1 freeze is still in place"
        );
    }

    // ── v2: priority, any-state, interruption ───────────────────────────────

    #[test]
    fn priority_beats_declaration_order_and_ties_fall_back_to_it() {
        // Two transitions out of state 0, both ready. Declared low-first.
        let sm = StateMachine {
            states: vec![
                SmState::clip("idle", [1; 16]),
                SmState::clip("run", [2; 16]),
                SmState::clip("jump", [3; 16]),
            ],
            transitions: vec![
                SmTransition::new(0, 1, 0.0),
                SmTransition::new(0, 2, 0.0).with_priority(10),
            ],
            entry: 0,
            ..Default::default()
        };
        let vars = |_: &str| None;
        let ctx = SmContext::new(&vars);
        let mut rt = SmRuntime::default();
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(
            rt.current, 2,
            "the higher priority won despite coming second"
        );

        // With the priorities equal, declaration order decides — v1's rule.
        let mut flat = sm.clone();
        flat.transitions[1].priority = 0;
        let mut rt = SmRuntime::default();
        rt.advance(&flat, &ctx, 0.1);
        assert_eq!(rt.current, 1);
    }

    #[test]
    fn an_any_state_transition_fires_from_every_state_but_its_own_target() {
        let sm = StateMachine {
            states: vec![
                SmState::clip("idle", [1; 16]),
                SmState::clip("run", [2; 16]),
                SmState::clip("hit", [3; 16]),
            ],
            transitions: vec![
                SmTransition::any(2, 0.0).when(SmCond::float("hit", CmpOp::Gt, 0.5)),
                SmTransition::on(0, 1, 0.0, "speed", CmpOp::Gt, 0.5),
            ],
            entry: 0,
            ..Default::default()
        };
        let hit = Cell::new(0.0f64);
        let speed = Cell::new(1.0f64);
        let vars = |n: &str| match n {
            "hit" => Some(hit.get()),
            "speed" => Some(speed.get()),
            _ => None,
        };
        let ctx = SmContext::new(&vars);

        let mut rt = SmRuntime::default();
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1, "idle -> run");
        hit.set(1.0);
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 2, "any -> hit, from run");
        // `exclude_self`: it does not re-enter itself while the flag is still up.
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 2);
        assert!(rt.prev.is_none(), "no self-transition was taken");
    }

    #[test]
    fn interrupt_none_lets_the_fade_finish() {
        let sm = StateMachine {
            states: vec![
                SmState::clip("a", [1; 16]),
                SmState::clip("b", [2; 16]),
                SmState::clip("c", [3; 16]),
            ],
            transitions: vec![
                SmTransition::new(0, 1, 1.0),
                SmTransition::new(1, 2, 0.0).with_interrupt(SmInterrupt {
                    source: InterruptSource::None,
                    ..Default::default()
                }),
            ],
            entry: 0,
            ..Default::default()
        };
        let vars = |_: &str| None;
        let ctx = SmContext::new(&vars);
        let mut rt = SmRuntime::default();
        rt.advance(&sm, &ctx, 0.1); // a -> b, 1 s fade running
        assert_eq!(rt.current, 1);
        rt.advance(&sm, &ctx, 0.1); // b -> c is blocked while fading
        assert_eq!(rt.current, 1, "the fade was cut into anyway");
        rt.advance(&sm, &ctx, 1.0); // fade retires this step…
        rt.advance(&sm, &ctx, 0.1); // …and now it may fire
        assert_eq!(rt.current, 2);
    }

    /// `SourceOrDestination` lets the machine change its mind about a transition
    /// **already underway** — a transition out of the state being faded *away
    /// from* becomes eligible. Under `Destination` (v1's rule) it is not, which is
    /// what makes this arm a comparison rather than an assertion.
    #[test]
    fn source_interruption_lets_a_transition_out_of_the_outgoing_state_fire() {
        let machine = |source: InterruptSource| StateMachine {
            states: vec![
                SmState::clip("a", [1; 16]),
                SmState::clip("b", [2; 16]),
                SmState::clip("c", [3; 16]),
            ],
            transitions: vec![
                SmTransition::new(0, 1, 1.0),
                // Leaves state 0 — which, mid-fade, is `prev`, not `current`.
                SmTransition::on(0, 2, 0.0, "recover", CmpOp::Gt, 0.5).with_interrupt(
                    SmInterrupt {
                        source,
                        blend: InterruptBlend::Carry,
                    },
                ),
            ],
            entry: 0,
            ..Default::default()
        };
        let recover = Cell::new(0.0f64);
        let vars = |n: &str| (n == "recover").then(|| recover.get());
        let ctx = SmContext::new(&vars);

        let land = |source: InterruptSource| -> usize {
            let sm = machine(source);
            let mut rt = SmRuntime::default();
            recover.set(0.0);
            rt.advance(&sm, &ctx, 0.1); // a -> b, 1 s fade
            assert_eq!(rt.prev, Some(0), "the fade is running out of state 0");
            recover.set(1.0);
            rt.advance(&sm, &ctx, 0.1);
            rt.current
        };
        assert_eq!(
            land(InterruptSource::Destination),
            1,
            "v1's rule only ever looks at the state being faded INTO"
        );
        assert_eq!(
            land(InterruptSource::SourceOrDestination),
            2,
            "the source end of the running fade must be leavable"
        );
    }

    /// **The interruption measurement.** v1 snapped: the new fade's outgoing pose
    /// was the incoming state at full weight, discarding the blend actually on
    /// screen. `Carry` keeps it, and the difference is measured in degrees of
    /// pose discontinuity across the interrupting step.
    #[test]
    fn carrying_an_interrupted_fade_is_measurably_more_continuous_than_snapping() {
        let sk = chain();
        // Three clips that put joint 1 in three very different places.
        let a = sweep("a", 1, -90.0, 1.0);
        let b = sweep("b", 1, 90.0, 1.0);
        let c = sweep("c", 1, 0.0, 1.0);
        let clips = |g: ClipRef| -> Option<&AnimClip> {
            match g[0] {
                1 => Some(&a),
                2 => Some(&b),
                3 => Some(&c),
                _ => None,
            }
        };
        let machine = |blend: InterruptBlend| StateMachine {
            states: vec![
                SmState::clip("a", [1; 16]),
                SmState::clip("b", [2; 16]),
                SmState::clip("c", [3; 16]),
            ],
            transitions: vec![
                SmTransition::new(0, 1, 1.0),
                // Gated, so the interruption happens at a chosen moment rather
                // than on the first step it is eligible for.
                SmTransition::on(1, 2, 1.0, "cut", CmpOp::Gt, 0.5).with_interrupt(SmInterrupt {
                    source: InterruptSource::Destination,
                    blend,
                }),
            ],
            entry: 0,
            ..Default::default()
        };
        let cut = Cell::new(0.0f64);
        let vars = |n: &str| (n == "cut").then(|| cut.get());
        let ctx = SmContext::new(&vars);

        let jump_for = |blend: InterruptBlend| -> f32 {
            let sm = machine(blend);
            let mut rt = SmRuntime::default();
            cut.set(0.0);
            rt.advance(&sm, &ctx, 0.1); // a -> b (1 s fade)
            rt.advance(&sm, &ctx, 0.4); // 40% through it
            let before = eval_pose(&sm, &rt, &sk, &clips, &ctx);
            cut.set(1.0);
            rt.advance(&sm, &ctx, 0.001); // b -> c interrupts, alpha ~= 0
            assert_eq!(rt.current, 2, "the interruption did not happen");
            let after = eval_pose(&sm, &rt, &sk, &clips, &ctx);
            before.locals[1]
                .rotation_quat()
                .angle_between(after.locals[1].rotation_quat())
                .to_degrees()
        };

        let snap = jump_for(InterruptBlend::Snap);
        let carry = jump_for(InterruptBlend::Carry);
        assert!(
            snap > 20.0,
            "the snap is supposed to be the discontinuity this test measures ({snap} deg)"
        );
        assert!(
            carry < 1.0,
            "carrying the interrupted partner must keep the pose continuous, got {carry} deg \
             against the snap's {snap}"
        );
    }

    /// **The carry is one deep, and what happens at the second interruption is a
    /// measurement rather than a sentence** (P29.1 audit, finding A8).
    ///
    /// [`SmRuntime::carry`] used to say a second interruption "drops it (and the
    /// machine snaps)". It does drop the oldest partner — there is one slot,
    /// because the runtime is `Copy` and rides an ECS component — but it does
    /// **not** snap: the newer partner is carried at the alpha the cut fade had
    /// reached, so the pose falls back one fade rather than to the incoming
    /// state. Naming the bound wrongly is how a bound stops being checked, so
    /// the three cases are measured side by side and ordered.
    #[test]
    fn interrupting_an_interruption_degrades_toward_the_newer_partner() {
        let sk = chain();
        let a = sweep("a", 1, -90.0, 1.0);
        let b = sweep("b", 1, 90.0, 1.0);
        let c = sweep("c", 1, 0.0, 1.0);
        let d = sweep("d", 1, 45.0, 1.0);
        let clips = |g: ClipRef| -> Option<&AnimClip> {
            match g[0] {
                1 => Some(&a),
                2 => Some(&b),
                3 => Some(&c),
                4 => Some(&d),
                _ => None,
            }
        };
        let machine = |blend: InterruptBlend| StateMachine {
            states: vec![
                SmState::clip("a", [1; 16]),
                SmState::clip("b", [2; 16]),
                SmState::clip("c", [3; 16]),
                SmState::clip("d", [4; 16]),
            ],
            transitions: vec![
                SmTransition::new(0, 1, 1.0),
                SmTransition::on(1, 2, 1.0, "cut1", CmpOp::Gt, 0.5).with_interrupt(SmInterrupt {
                    source: InterruptSource::Destination,
                    blend,
                }),
                SmTransition::on(2, 3, 1.0, "cut2", CmpOp::Gt, 0.5).with_interrupt(SmInterrupt {
                    source: InterruptSource::Destination,
                    blend,
                }),
            ],
            entry: 0,
            ..Default::default()
        };
        let c1 = Cell::new(0.0f64);
        let c2 = Cell::new(0.0f64);
        let vars = |n: &str| match n {
            "cut1" => Some(c1.get()),
            "cut2" => Some(c2.get()),
            _ => None,
        };
        let ctx = SmContext::new(&vars);

        // The jump across the SECOND interruption, which is the one the carry
        // slot cannot fully serve.
        let second_jump = |blend: InterruptBlend| -> (f32, Option<usize>) {
            let sm = machine(blend);
            let mut rt = SmRuntime::default();
            c1.set(0.0);
            c2.set(0.0);
            rt.advance(&sm, &ctx, 0.1); // a -> b
            rt.advance(&sm, &ctx, 0.4); // 40% through
            c1.set(1.0);
            rt.advance(&sm, &ctx, 0.3); // b -> c interrupts, carrying a
            assert_eq!(rt.current, 2, "the first interruption did not happen");
            let before = eval_pose(&sm, &rt, &sk, &clips, &ctx);
            c2.set(1.0);
            rt.advance(&sm, &ctx, 0.001); // c -> d interrupts again
            assert_eq!(rt.current, 3, "the second interruption did not happen");
            let after = eval_pose(&sm, &rt, &sk, &clips, &ctx);
            let deg = before.locals[1]
                .rotation_quat()
                .angle_between(after.locals[1].rotation_quat())
                .to_degrees();
            (deg, rt.carry)
        };

        let (carry_jump, carried) = second_jump(InterruptBlend::Carry);
        let (snap_jump, snapped) = second_jump(InterruptBlend::Snap);
        // `prev` is now `c`; the one slot holds `b`, the partner of the fade
        // that was just cut — and `a`, which the first interruption had carried,
        // is what falls out. That is the bound, exactly.
        assert_eq!(
            carried,
            Some(1),
            "the slot must hold the partner of the fade just cut, and drop the older one"
        );
        assert_eq!(snapped, None, "Snap keeps no partner at all");
        // The bound, stated as an inequality rather than a story: the second
        // interruption costs something, and it costs strictly less than the snap
        // the doc used to claim it becomes.
        assert!(
            carry_jump < snap_jump,
            "carrying the newer partner ({carry_jump} deg) is supposed to beat \
             snapping ({snap_jump} deg) even at the second interruption"
        );
        // Measured on this fixture: **40.5°** against the snap's **63°**. That
        // is not "small" — the audit's first draft of the doc above said so and
        // was wrong twice over — but it is a third less, and it is the price of
        // one slot rather than of two. Banded rather than pinned to a constant,
        // because the number is a property of these four clips.
        assert!(
            carry_jump > 20.0,
            "the one-deep bound is supposed to COST something ({carry_jump} deg) \
             — if it is nearly free, the slot is not the limit this claims it is, \
             and the first interruption's 0.06° is what free looks like"
        );
    }

    // ── v2: typed parameters and triggers ───────────────────────────────────

    #[test]
    fn an_int_parameter_compares_exactly_where_a_float_would_not() {
        let sm = StateMachine {
            params: vec![SmParam::new("stance", SmParamKind::Int)],
            ..Default::default()
        };
        let float_sm = StateMachine::default(); // `stance` undeclared -> Float
                                                // 2.0000000001 is `== 2` under the float epsilon and is not an Int 2…
        let vars = |n: &str| (n == "stance").then_some(2.000_000_000_1);
        let ctx = SmContext::new(&vars);
        let cond_i = SmCond::int("stance", CmpOp::Eq, 2);
        let cond_f = SmCond::float("stance", CmpOp::Eq, 2.0);
        assert!(eval_condition(&cond_i, &sm, &ctx, 0), "Int truncates to 2");
        assert!(eval_condition(&cond_f, &float_sm, &ctx, 0), "Float epsilon");
        // …and 2.9 is Int 2 but is NOT float-equal to 2.
        let vars = |n: &str| (n == "stance").then_some(2.9);
        let ctx = SmContext::new(&vars);
        assert!(eval_condition(&cond_i, &sm, &ctx, 0));
        assert!(!eval_condition(&cond_f, &float_sm, &ctx, 0));
    }

    #[test]
    fn a_bool_parameter_reads_at_the_half_threshold() {
        let sm = StateMachine {
            params: vec![SmParam::new("grounded", SmParamKind::Bool)],
            ..Default::default()
        };
        let on = SmCond::bool("grounded", true);
        for (v, want) in [(0.0, false), (0.4, false), (0.6, true), (1.0, true)] {
            let vars = |n: &str| (n == "grounded").then_some(v);
            let ctx = SmContext::new(&vars);
            assert_eq!(eval_condition(&on, &sm, &ctx, 0), want, "at {v}");
        }
    }

    /// **The trigger contract, all four rules.**
    #[test]
    fn a_trigger_arms_on_a_rising_edge_and_is_consumed_by_the_transition_that_read_it() {
        let sm = StateMachine {
            states: vec![
                SmState::clip("idle", [1; 16]),
                SmState::clip("jump", [2; 16]),
                SmState::clip("land", [3; 16]),
            ],
            transitions: vec![
                SmTransition::new(0, 1, 0.0).when(SmCond::Trigger("jump".into())),
                SmTransition::new(1, 2, 0.0).when(SmCond::Trigger("jump".into())),
            ],
            entry: 0,
            params: vec![SmParam::trigger("jump")],
            ..Default::default()
        };
        let pressed = Cell::new(0.0f64);
        let vars = |n: &str| (n == "jump").then(|| pressed.get());
        let ctx = SmContext::new(&vars);
        let mut rt = SmRuntime::default();

        // Rule 1: no edge yet.
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 0);
        assert_eq!(rt.triggers, 0);

        // Rising edge arms it and the transition fires and consumes it…
        pressed.set(1.0);
        let step = rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1);
        assert_eq!(step.consumed, 1, "the jump trigger was consumed");
        assert_eq!(rt.triggers, 0, "and is no longer armed");

        // …so HOLDING the button does not fire the second transition: a level is
        // not an edge, which is the whole difference between a Trigger and a Bool.
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1, "a held trigger re-fired");

        // Releasing and pressing again is a fresh edge.
        pressed.set(0.0);
        rt.advance(&sm, &ctx, 0.1);
        pressed.set(1.0);
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 2);
    }

    /// Rule 4: an armed trigger nothing consumed **stays armed**, so an input one
    /// step early is not lost.
    #[test]
    fn an_unconsumed_trigger_stays_armed() {
        let sm = StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition::new(0, 1, 0.0).when(SmCond::And(vec![
                SmCond::Trigger("go".into()),
                SmCond::float("ready", CmpOp::Gt, 0.5),
            ]))],
            entry: 0,
            params: vec![SmParam::trigger("go")],
            ..Default::default()
        };
        let ready = Cell::new(0.0f64);
        let go = Cell::new(0.0f64);
        let vars = |n: &str| match n {
            "go" => Some(go.get()),
            "ready" => Some(ready.get()),
            _ => None,
        };
        let ctx = SmContext::new(&vars);
        let mut rt = SmRuntime::default();
        rt.advance(&sm, &ctx, 0.1);

        // Pressed one step before the gate opens, and released immediately.
        go.set(1.0);
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 0, "not ready yet");
        assert_eq!(rt.triggers, 1, "but the press was NOT lost");
        go.set(0.0);
        ready.set(1.0);
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1, "the remembered press fired it");
    }

    /// Rule 3: an `Or` branch that evaluated **false** keeps its trigger, and a
    /// `Not` never consumes. One press cannot be eaten by a branch that did not
    /// act on it.
    ///
    /// Both triggers are armed; the right-hand branch is false for a reason that
    /// has nothing to do with its trigger (`enabled` is 0), which is exactly the
    /// shape a "consume everything the tree mentions" rule would get wrong.
    #[test]
    fn only_the_triggers_a_transition_read_as_true_are_consumed() {
        let sm = StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition::new(0, 1, 0.0).when(SmCond::Or(vec![
                SmCond::Trigger("left".into()),
                SmCond::And(vec![
                    SmCond::Trigger("right".into()),
                    SmCond::float("enabled", CmpOp::Gt, 0.5),
                ]),
                // Armed, so this reads `Not(true)` = false — the branch is not
                // taken and its trigger must survive.
                SmCond::Not(Box::new(SmCond::Trigger("never".into()))),
            ]))],
            entry: 0,
            params: vec![
                SmParam::trigger("left"),
                SmParam::trigger("right"),
                SmParam::trigger("never"),
            ],
            ..Default::default()
        };
        let vars = |_: &str| None; // `enabled` reads 0 — the middle branch is false
        let ctx = SmContext::new(&vars);
        // All three armed before the first step: `left` is what fires it, the
        // middle branch is false for a reason unrelated to its trigger, and
        // `never` is armed so the `Not` over it reads FALSE (which is what makes
        // that branch a real test rather than a vacuous one).
        let mut rt = SmRuntime {
            triggers: 0b111,
            ..Default::default()
        };
        let step = rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1);
        assert_eq!(
            step.consumed, 0b001,
            "only the branch that read TRUE consumed its trigger"
        );
        assert_eq!(
            rt.triggers, 0b110,
            "`right` (false branch) and `never` (under a Not) are still armed"
        );
    }

    // ── v2: curves, profiles, sub-machines, events ──────────────────────────

    #[test]
    fn every_blend_curve_is_a_polynomial_through_the_unit_square() {
        for c in [
            BlendCurve::Linear,
            BlendCurve::EaseIn,
            BlendCurve::EaseOut,
            BlendCurve::EaseInOut,
            BlendCurve::Step,
        ] {
            assert_eq!(c.apply(0.0), 0.0, "{c:?} does not start at 0");
            assert_eq!(c.apply(1.0), 1.0, "{c:?} does not end at 1");
            // Clamped outside, monotone inside.
            assert_eq!(c.apply(-5.0), 0.0);
            assert_eq!(c.apply(5.0), 1.0);
            let mut last = 0.0;
            for i in 0..=20 {
                let v = c.apply(i as f64 / 20.0);
                assert!((0.0..=1.0).contains(&v), "{c:?} left the unit square: {v}");
                assert!(v >= last - 1e-12, "{c:?} is not monotone at {i}");
                last = v;
            }
        }
        // The shapes are actually different from each other at the midpoint.
        assert!(BlendCurve::EaseIn.apply(0.5) < 0.5);
        assert!(BlendCurve::EaseOut.apply(0.5) > 0.5);
        assert_eq!(BlendCurve::EaseInOut.apply(0.5), 0.5);
        assert_eq!(BlendCurve::Step.apply(0.99), 0.0);
    }

    #[test]
    fn a_blend_profile_masks_the_fade_per_joint() {
        let sk = chain();
        let a = sweep("a", 1, 0.0, 1.0);
        let b = sweep("b", 1, 90.0, 1.0);
        let clips = |g: ClipRef| -> Option<&AnimClip> {
            match g[0] {
                1 => Some(&a),
                _ => Some(&b),
            }
        };
        let sm = StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition::new(0, 1, 1.0).with_profile(0)],
            entry: 0,
            profiles: vec![BlendProfile::new(
                "legs-only",
                // Joint 1 is pinned to the OUTGOING pose for the whole fade.
                vec![JointBlendWeight {
                    joint: 1,
                    weight: 0.0,
                }],
            )],
            ..Default::default()
        };
        let vars = |_: &str| None;
        let ctx = SmContext::new(&vars);
        let mut rt = SmRuntime::default();
        rt.advance(&sm, &ctx, 0.1);
        rt.advance(&sm, &ctx, 0.5); // half-way through the fade
        let posed = eval_pose(&sm, &rt, &sk, &clips, &ctx);
        // The b clip drives joint 1 to 90 deg; the mask holds it at the outgoing
        // pose, so the masked joint has not moved with the blend.
        let outgoing = sample_clip(&sk, &a, rt.prev_time as f32, true);
        let ang = posed.locals[1]
            .rotation_quat()
            .angle_between(outgoing.locals[1].rotation_quat())
            .to_degrees();
        assert!(ang < 1e-3, "the masked joint blended anyway ({ang} deg)");

        // Without the profile the same fade DOES move it — otherwise the arm
        // above is vacuous.
        let mut unmasked = sm.clone();
        unmasked.transitions[0].profile = None;
        let mut rt = SmRuntime::default();
        rt.advance(&unmasked, &ctx, 0.1);
        rt.advance(&unmasked, &ctx, 0.5);
        let posed = eval_pose(&unmasked, &rt, &sk, &clips, &ctx);
        let ang = posed.locals[1]
            .rotation_quat()
            .angle_between(outgoing.locals[1].rotation_quat())
            .to_degrees();
        assert!(
            ang > 10.0,
            "the unmasked fade did not move the joint either"
        );
    }

    /// **A sub-machine evaluates against its PARENT's parameter table.**
    ///
    /// The nested machine declares nothing (`validate` refuses it if it tries),
    /// so resolving a typed parameter inside it against its own table would read
    /// every `Int` as a `Float` and every `Trigger` as permanently unarmed — the
    /// silent kind of wrong, because every name still resolves. And a trigger a
    /// nested transition read is consumed exactly like one the parent read: they
    /// are the same parameter.
    #[test]
    fn a_sub_machine_reads_and_consumes_the_parents_parameters() {
        let inner = StateMachine {
            states: vec![
                SmState::clip("stand", [2; 16]),
                SmState::clip("crouch", [3; 16]),
            ],
            transitions: vec![
                // A TYPED compare and a TRIGGER, both of which are meaningless
                // unless the parent's table is what resolves them.
                SmTransition::new(0, 1, 0.0).when(SmCond::And(vec![
                    SmCond::int("stance", CmpOp::Eq, 2),
                    SmCond::Trigger("duck".into()),
                ])),
            ],
            entry: 0,
            ..Default::default()
        };
        let sm = StateMachine {
            states: vec![SmState::sub_machine("ground", inner)],
            entry: 0,
            params: vec![
                SmParam::new("stance", SmParamKind::Int),
                SmParam::trigger("duck"),
            ],
            ..Default::default()
        };
        sm.validate()
            .expect("a machine with one nested level is valid");

        let stance = Cell::new(0.0f64);
        let duck = Cell::new(0.0f64);
        let vars = |n: &str| match n {
            "stance" => Some(stance.get()),
            "duck" => Some(duck.get()),
            _ => None,
        };
        let ctx = SmContext::new(&vars);
        let mut rt = SmRuntime::default();
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.sub.current, 0, "still standing");

        // `stance` is an Int: 2.4 truncates to 2 and satisfies `== 2`, which a
        // Float compare with an epsilon would NOT.
        stance.set(2.4);
        duck.set(1.0);
        let step = rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.sub.current, 1, "the nested transition never fired");
        assert_eq!(step.sub_fired, Some(0));
        assert_eq!(
            step.consumed, 0b10,
            "the nested read must consume the parent's trigger"
        );
        assert_eq!(rt.triggers, 0, "and disarm it");
    }

    /// A sub-machine that declares parameters of its own is refused, because
    /// nothing would read them.
    #[test]
    fn a_sub_machine_may_not_declare_its_own_parameters() {
        let inner = StateMachine {
            states: vec![SmState::clip("x", [1; 16])],
            params: vec![SmParam::float("mine")],
            ..Default::default()
        };
        let sm = StateMachine {
            states: vec![SmState::sub_machine("s", inner)],
            ..Default::default()
        };
        let e = sm.validate().unwrap_err();
        assert!(
            e.to_string().contains("shares its parent"),
            "the refusal must say why: {e}"
        );
    }

    #[test]
    fn a_sub_machine_steps_and_poses_from_its_own_state() {
        let sk = chain();
        let stand = sweep("stand", 1, 0.0, 1.0);
        let crouch = sweep("crouch", 1, 60.0, 1.0);
        let clips = |g: ClipRef| -> Option<&AnimClip> {
            match g[0] {
                2 => Some(&stand),
                3 => Some(&crouch),
                _ => None,
            }
        };
        let inner = StateMachine {
            states: vec![
                SmState::clip("stand", [2; 16]),
                SmState::clip("crouch", [3; 16]),
            ],
            transitions: vec![SmTransition::on(0, 1, 0.0, "crouching", CmpOp::Gt, 0.5)],
            entry: 0,
            ..Default::default()
        };
        let sm = StateMachine {
            states: vec![
                SmState::clip("air", [1; 16]),
                SmState::sub_machine("ground", inner),
            ],
            transitions: vec![SmTransition::on(0, 1, 0.0, "grounded", CmpOp::Gt, 0.5)],
            entry: 0,
            ..Default::default()
        };
        let grounded = Cell::new(0.0f64);
        let crouching = Cell::new(0.0f64);
        let vars = |n: &str| match n {
            "grounded" => Some(grounded.get()),
            "crouching" => Some(crouching.get()),
            _ => None,
        };
        let ctx = SmContext::new(&vars);
        let mut rt = SmRuntime::default();

        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 0);
        assert!(!rt.sub.started, "no sub-machine on a clip state");

        grounded.set(1.0);
        rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.current, 1);
        assert_eq!((rt.sub.owner, rt.sub.current), (1, 0), "entered `stand`");
        rt.advance(&sm, &ctx, 0.3);
        let standing = eval_pose(&sm, &rt, &sk, &clips, &ctx);

        crouching.set(1.0);
        let step = rt.advance(&sm, &ctx, 0.1);
        assert_eq!(rt.sub.current, 1, "the sub-machine transitioned");
        assert_eq!(step.sub_fired, Some(0));
        // Let the crouch clip play a little, so the comparison is against a pose
        // and not against both clips' shared first keyframe.
        rt.advance(&sm, &ctx, 0.3);
        let crouched = eval_pose(&sm, &rt, &sk, &clips, &ctx);
        let ang = standing.locals[1]
            .rotation_quat()
            .angle_between(crouched.locals[1].rotation_quat())
            .to_degrees();
        assert!(
            ang > 10.0,
            "the parent posed the same thing before and after the sub-machine moved ({ang} deg)"
        );
    }

    #[test]
    fn enter_and_exit_events_are_reported_in_order() {
        let mut idle = SmState::clip("idle", [1; 16]);
        idle.on_enter = vec!["idle_begin".into()];
        idle.on_exit = vec!["idle_end".into()];
        let mut run = SmState::clip("run", [2; 16]);
        run.on_enter = vec!["footstep_start".into()];
        let sm = StateMachine {
            states: vec![idle, run],
            transitions: vec![SmTransition::on(0, 1, 0.0, "speed", CmpOp::Gt, 0.5)],
            entry: 0,
            ..Default::default()
        };
        let speed = Cell::new(0.0f64);
        let vars = |n: &str| (n == "speed").then(|| speed.get());
        let ctx = SmContext::new(&vars);
        let mut rt = SmRuntime::default();

        let step = rt.advance(&sm, &ctx, 0.1);
        assert_eq!(step.events, vec!["idle_begin".to_string()]);
        assert_eq!(step.entered_events(&sm), ["idle_begin".to_string()]);

        speed.set(1.0);
        let step = rt.advance(&sm, &ctx, 0.1);
        // Exit first, then the entry it caused.
        assert_eq!(
            step.events,
            vec!["idle_end".to_string(), "footstep_start".to_string()]
        );
    }

    // ── validation ──────────────────────────────────────────────────────────

    #[test]
    fn validate_refuses_what_the_evaluator_would_have_to_trust() {
        let ok = two_state_machine();
        ok.validate().expect("a healthy machine must validate");
        StateMachine::empty()
            .validate()
            .expect("an empty machine is a legal document");

        let bad = |f: &dyn Fn(&mut StateMachine)| {
            let mut sm = two_state_machine();
            f(&mut sm);
            sm.validate().expect_err("this machine should be refused")
        };

        assert!(matches!(
            bad(&|sm| sm.entry = 9),
            SmError::EntryOutOfRange { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.transitions[0].to = 9),
            SmError::TargetOutOfRange { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.transitions[0].from = SmSource::State(9)),
            SmError::SourceOutOfRange { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.transitions[0].profile = Some(0)),
            SmError::ProfileOutOfRange { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.transitions[0].duration = f64::NAN),
            SmError::BadDuration { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.transitions[0].duration = -1.0),
            SmError::BadDuration { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.transitions[0].exit_time = Some(1.5)),
            SmError::BadExitTime { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.transitions[0].exit_time = Some(f64::NAN)),
            SmError::BadExitTime { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.states[0].speed = f64::INFINITY),
            SmError::BadSpeed { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.transitions[0].condition = SmCond::float("x", CmpOp::Gt, f64::NAN)),
            SmError::NonFiniteCompare { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.params = vec![SmParam::float("a"), SmParam::float("a")]),
            SmError::DuplicateParam { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.params = vec![SmParam::float("")]),
            SmError::EmptyParamName { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.params = (0..MAX_PARAMS + 1)
                .map(|i| SmParam::float(format!("p{i}")))
                .collect()),
            SmError::TooManyParams { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.profiles = vec![BlendProfile::new(
                "bad",
                vec![JointBlendWeight {
                    joint: 0,
                    weight: 2.0
                }]
            )]),
            SmError::BadProfileWeight { .. }
        ));

        // The recursive bounds.
        let deep = {
            let mut c = SmCond::Always;
            for _ in 0..MAX_COND_DEPTH + 1 {
                c = SmCond::Not(Box::new(c));
            }
            c
        };
        assert!(matches!(
            bad(&|sm| sm.transitions[0].condition = deep.clone()),
            SmError::ConditionTooDeep { .. }
        ));
        assert!(matches!(
            bad(&|sm| sm.transitions[0].condition =
                SmCond::And((0..MAX_COND_NODES).map(|_| SmCond::Always).collect())),
            SmError::ConditionTooLarge { .. }
        ));

        // Two levels of sub-machine.
        let inner = StateMachine {
            states: vec![SmState::clip("x", [1; 16])],
            ..Default::default()
        };
        let mid = StateMachine {
            states: vec![SmState::sub_machine("mid", inner)],
            ..Default::default()
        };
        let outer = StateMachine {
            states: vec![SmState::sub_machine("outer", mid)],
            ..Default::default()
        };
        assert!(matches!(
            outer.validate().unwrap_err(),
            SmError::SubMachine { .. }
        ));
        // …and a sub-machine's own invalidity is named through its parent.
        let broken = StateMachine {
            states: vec![SmState::sub_machine(
                "s",
                StateMachine {
                    states: vec![SmState::clip("x", [1; 16])],
                    entry: 7,
                    ..Default::default()
                },
            )],
            ..Default::default()
        };
        let e = broken.validate().unwrap_err();
        assert!(e.to_string().contains("sub-machine is invalid"), "{e}");
    }

    /// The depth bound is enforced in **evaluation** too, because a machine built
    /// in memory never passed through a decoder.
    #[test]
    fn an_over_deep_condition_evaluates_false_rather_than_recursing() {
        let sm = StateMachine::default();
        let vars = |_: &str| None;
        let ctx = SmContext::new(&vars);
        let mut c = SmCond::Always;
        for _ in 0..MAX_COND_DEPTH + 4 {
            c = SmCond::And(vec![c]);
        }
        assert!(!eval_condition(&c, &sm, &ctx, 0));
    }

    /// **…and `Not` does not turn "unreadable" back into "satisfied"** (P29.1
    /// audit, finding A3).
    ///
    /// The bound's whole justification is that "a condition that cannot be read
    /// must not be treated as satisfied". It returned `false` from the over-deep
    /// frame and let that answer keep propagating, so every enclosing `Not`
    /// inverted it: a tree nesting an **even** number of `Not`s past the bound
    /// came back `true`. The arm above could not see it — an `And` chain has no
    /// inversion in it, which is exactly why the defect survived one.
    ///
    /// Measured before the fix: 18, 20 and 32 `Not`s over a **false** leaf all
    /// evaluated `true`.
    #[test]
    fn an_over_deep_condition_is_false_through_a_not_chain_of_either_parity() {
        let sm = StateMachine::default();
        let vars = |_: &str| Some(0.0);
        let ctx = SmContext::new(&vars);
        // A leaf that is plainly FALSE, so an even chain's honest answer is
        // `false` and an odd chain's is `true` — and the bound must make BOTH
        // read `false`, because neither was read at all.
        let leaf = || SmCond::float("speed", CmpOp::Gt, 100.0);
        assert!(!eval_condition(&leaf(), &sm, &ctx, 0), "the leaf is false");

        for n in [MAX_COND_DEPTH + 1, MAX_COND_DEPTH + 2, 31, 32] {
            let mut c = leaf();
            for _ in 0..n {
                c = SmCond::Not(Box::new(c));
            }
            assert!(
                !eval_condition(&c, &sm, &ctx, 0),
                "{n} `Not`s past the depth bound evaluated TRUE — an unreadable \
                 condition was treated as satisfied"
            );
        }
        // NOT VACUOUS: inside the bound, the parity still decides.
        let mut even = leaf();
        let mut odd = leaf();
        for _ in 0..2 {
            even = SmCond::Not(Box::new(even));
        }
        odd = SmCond::Not(Box::new(odd));
        assert!(
            !eval_condition(&even, &sm, &ctx, 0),
            "Not(Not(false)) is false"
        );
        assert!(eval_condition(&odd, &sm, &ctx, 0), "Not(false) is true");
    }

    /// **A parameter past the trigger bitmask is not found, and does not shift**
    /// (P29.1 audit, finding A2).
    ///
    /// `SmRuntime::triggers` is a `u64`, so a parameter's index is a bit
    /// position. `sample_triggers` already stopped at [`MAX_PARAMS`]; the two
    /// **readers** did not, and `1u64 << 64` is `attempt to shift left with
    /// overflow` in debug and a silent wrap to bit 0 in release — the second
    /// being the worse one, because it consumes an unrelated trigger.
    /// `validate` refuses an over-long table at decode, but the editor builds
    /// machines in memory that never pass a decoder, which is the same argument
    /// that put the depth bound in `eval_condition` as well as in `validate`.
    #[test]
    fn a_parameter_past_the_trigger_bitmask_is_not_found_rather_than_shifting() {
        let mut sm = StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition::new(0, 1, 0.0).when(SmCond::Trigger("t70".into()))],
            entry: 0,
            ..Default::default()
        };
        sm.params = (0..80).map(|i| SmParam::trigger(format!("t{i}"))).collect();
        // In range, so the machine still works for the first 64.
        assert_eq!(sm.param("t0").map(|(i, _)| i), Some(0));
        assert_eq!(sm.param("t63").map(|(i, _)| i), Some(63));
        assert_eq!(
            sm.param("t70"),
            None,
            "a parameter past the bitmask must not be resolvable at all"
        );

        // The fixed step over it: this used to panic.
        let vars = |_: &str| Some(1.0);
        let ctx = SmContext::new(&vars);
        let mut rt = SmRuntime::default();
        let step = rt.advance(&sm, &ctx, 1.0 / 60.0);
        assert_eq!(step.fired, None, "an unresolvable trigger reads as unarmed");
        assert_eq!(step.consumed, 0);
        // …and the in-range half is untouched, so this is a bound and not a
        // blanket refusal.
        sm.transitions[0].condition = SmCond::Trigger("t5".into());
        let mut rt = SmRuntime::default();
        let step = rt.advance(&sm, &ctx, 1.0 / 60.0);
        assert_eq!(step.fired, Some(0));
        assert_eq!(step.consumed, 1 << 5);
    }

    /// **A trigger read through an EVEN number of `Not`s is consumed** (P29.1
    /// audit, finding A4).
    ///
    /// Rule 3 is "every trigger the fired tree read as **true** is disarmed".
    /// The walk stopped dead at a `Not`, which is right for `Not(Trigger(x))` —
    /// the transition fired because `x` was unset, and has nothing to spend — and
    /// wrong for `Not(Not(Trigger(x)))`, which fires *because `x` is armed* and
    /// then left it armed for ever, so the same press re-fired on the next step.
    /// Both parities are asserted here, because a fix that consumed under an odd
    /// count would break the rule it was written to keep.
    #[test]
    fn a_trigger_is_consumed_through_an_even_not_chain_and_survives_an_odd_one() {
        let machine = |cond: SmCond| StateMachine {
            states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
            transitions: vec![SmTransition::new(0, 1, 0.0).when(cond)],
            entry: 0,
            params: vec![SmParam::trigger("t"), SmParam::float("f")],
            ..Default::default()
        };
        let not = |c: SmCond| SmCond::Not(Box::new(c));
        let run = |cond: SmCond, t: f64| -> (Option<usize>, u64, u64) {
            let sm = machine(cond);
            let vars = move |n: &str| Some(if n == "t" { t } else { 0.0 });
            let ctx = SmContext::new(&vars);
            let mut rt = SmRuntime::default();
            let step = rt.advance(&sm, &ctx, 0.1);
            (step.fired, step.consumed, rt.triggers)
        };

        // Even: the tree fired BECAUSE the trigger was armed. It must be spent.
        let (fired, consumed, left) = run(not(not(SmCond::Trigger("t".into()))), 1.0);
        assert_eq!(fired, Some(0), "Not(Not(armed)) must fire");
        assert_eq!(
            consumed, 1,
            "the press the transition acted on was not spent"
        );
        assert_eq!(left, 0, "and it must not stay armed to re-fire next step");

        // Odd: the tree fired because the trigger was UNSET. Nothing to spend,
        // and — the case the original rule exists for — a later press must still
        // be there for whoever wanted it.
        let (fired, consumed, left) = run(not(SmCond::Trigger("t".into())), 0.0);
        assert_eq!(fired, Some(0), "Not(unarmed) must fire");
        assert_eq!((consumed, left), (0, 0), "there was nothing armed to spend");

        // Odd, with the trigger armed but the transition firing on something
        // else being false: the arm stays.
        let cond = not(SmCond::And(vec![
            SmCond::Trigger("t".into()),
            SmCond::float("f", CmpOp::Gt, 100.0),
        ]));
        let (fired, consumed, left) = run(cond, 1.0);
        assert_eq!(fired, Some(0));
        assert_eq!(
            consumed, 0,
            "a trigger under an odd `Not` must not be eaten"
        );
        assert_eq!(left, 1, "it stays armed for the transition that wants it");
    }

    // ── the flat view the editor DTO round-trips through ────────────────────

    #[test]
    fn the_flat_and_view_is_lossless_exactly_when_it_is_flat() {
        let flat = SmCond::And(vec![
            SmCond::float("speed", CmpOp::Gt, 0.1),
            SmCond::float("grounded", CmpOp::Ge, 1.0),
        ]);
        let view = flat.as_flat_and().expect("a flat AND of float compares");
        assert_eq!(view.len(), 2);
        assert_eq!(SmCond::from_flat_and(view), flat);

        // A single leaf, and the empty condition, both round-trip.
        let one = SmCond::float("speed", CmpOp::Gt, 0.1);
        assert_eq!(
            SmCond::from_flat_and(one.as_flat_and().unwrap()),
            one,
            "a lone compare"
        );
        assert_eq!(SmCond::Always.as_flat_and().unwrap(), Vec::new());
        assert_eq!(SmCond::from_flat_and(Vec::new()), SmCond::Always);

        // …and everything that is NOT flat says so, which is what stops a v1 UI
        // silently destroying a v2 tree on save.
        for tree in [
            SmCond::Or(vec![SmCond::float("a", CmpOp::Gt, 0.0)]),
            SmCond::Not(Box::new(SmCond::float("a", CmpOp::Gt, 0.0))),
            SmCond::Trigger("jump".into()),
            SmCond::int("stance", CmpOp::Eq, 1),
            SmCond::bool("grounded", true),
            SmCond::And(vec![
                SmCond::float("a", CmpOp::Gt, 0.0),
                SmCond::Trigger("jump".into()),
            ]),
        ] {
            assert!(
                tree.as_flat_and().is_none(),
                "{tree:?} claimed to be a flat float AND"
            );
        }
    }

    /// **The wire discriminants of every enum in this model are frozen.**
    ///
    /// bincode writes a unit variant as its index, so re-ordering any of these
    /// silently re-interprets every committed `.inf_sm`: an `EaseOut` becomes an
    /// `EaseIn`, a `Destination` interruption becomes a `None` one. The freeze is
    /// the P19 wire-enum law; this is its `.inf_sm` copy.
    #[test]
    fn freeze_pins() {
        // The **real** wire config, not a plausible one: `inf_asset::encode` is
        // what writes a `.inf_sm`, so pinning against `config::standard()` would
        // pin a format nothing uses.
        fn idx<T: Serialize>(v: &T) -> u32 {
            let bytes = bincode::serde::encode_to_vec(v, inf_asset::bincode_config()).unwrap();
            bytes[0] as u32
        }
        assert_eq!(
            [
                idx(&CmpOp::Gt),
                idx(&CmpOp::Lt),
                idx(&CmpOp::Ge),
                idx(&CmpOp::Le),
                idx(&CmpOp::Eq),
                idx(&CmpOp::Ne)
            ],
            [0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            [
                idx(&SmParamKind::Bool),
                idx(&SmParamKind::Int),
                idx(&SmParamKind::Float),
                idx(&SmParamKind::Trigger)
            ],
            [0, 1, 2, 3]
        );
        assert_eq!(
            [
                idx(&BlendCurve::Linear),
                idx(&BlendCurve::EaseIn),
                idx(&BlendCurve::EaseOut),
                idx(&BlendCurve::EaseInOut),
                idx(&BlendCurve::Step)
            ],
            [0, 1, 2, 3, 4]
        );
        assert_eq!(
            [
                idx(&InterruptSource::None),
                idx(&InterruptSource::Destination),
                idx(&InterruptSource::SourceOrDestination)
            ],
            [0, 1, 2]
        );
        assert_eq!(
            [idx(&InterruptBlend::Snap), idx(&InterruptBlend::Carry)],
            [0, 1]
        );
        assert_eq!(
            [
                idx(&SmValue::Bool(false)),
                idx(&SmValue::Int(0)),
                idx(&SmValue::Float(0.0))
            ],
            [0, 1, 2]
        );
        assert_eq!(
            [
                idx(&SmCond::Always),
                idx(&SmCond::float("a", CmpOp::Gt, 0.0)),
                idx(&SmCond::Trigger("a".into())),
                idx(&SmCond::And(vec![])),
                idx(&SmCond::Or(vec![])),
                idx(&SmCond::Not(Box::new(SmCond::Always)))
            ],
            [0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            [
                idx(&Motion::Clip([0; 16])),
                idx(&Motion::SubMachine(Box::default()))
            ],
            [0, 3]
        );
        assert_eq!(
            [
                idx(&SmSource::State(0)),
                idx(&SmSource::Any { exclude_self: true })
            ],
            [0, 1]
        );
        // …and the defaults, which are what an unset field decodes as.
        assert_eq!(BlendCurve::default(), BlendCurve::Linear);
        assert_eq!(SmParamKind::default(), SmParamKind::Float);
        assert_eq!(SmCond::default(), SmCond::Always);
        assert_eq!(
            SmInterrupt::default(),
            SmInterrupt {
                source: InterruptSource::Destination,
                blend: InterruptBlend::Carry
            }
        );
    }
}
