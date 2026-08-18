//! **The `.inf_sm` text form** — pillar S1's substrate (P29.6).
//!
//! §13's opening finding is that the UE5 reference project's author never opened
//! its animation graphs, and that the reason is the file format: *"an
//! `AnimBlueprint` is a binary `.uasset`, so it cannot be diffed, reviewed,
//! merged or edited outside the editor, and the cost of understanding somebody
//! else's is high enough that the stock one gets shipped as delivered."* S1's
//! answer is that a machine is **text**. P29.1 shipped the model and said so;
//! P29.5's [`crate::propose`] docs said so; nothing in the tree could actually
//! produce the text. This module is it, and `phase29_gate`'s one-line-diff arm is
//! its acceptance test.
//!
//! # What it is
//!
//! A deterministic TOML **projection** of a [`StateMachine`], and a reader that
//! takes it back. The bincode payload stays the machine face; this is the human
//! one, exactly as the asset sidecar is for metadata. Two properties are asserted
//! rather than hoped for:
//!
//! * **Lossless.** `from_toml(to_toml(m))? == m` for every v2 shape — typed
//!   parameters, condition trees, priority, interruption, curves, per-joint
//!   profiles, `exit_time`, any-state edges, blend spaces and one level of nested
//!   sub-machine.
//! * **One value, one line.** Every number an author would reach for — a blend
//!   duration, a threshold, a play rate, a priority — is its own `key = value`
//!   line, so changing one is a one-line diff. That is the property the gate
//!   measures, and it is why the writer emits standard tables rather than the
//!   inline tables TOML would also accept.
//!
//! # Why the condition is an expression and not a tree of tables
//!
//! `SmCond` is recursive, and TOML's spelling for a recursive sum type is nested
//! arrays-of-tables: an `And` of two compares is six header lines and eight keys.
//! That is representable and unreviewable, and it would put a threshold's number
//! three levels deep in a structure whose *shape* changes when the author adds a
//! term. So a condition is **one line of a tiny expression language** instead:
//!
//! ```text
//! always                     SmCond::Always
//! speed > 2.6                a float compare
//! tier == 2                  an int compare
//! aiming == true             a bool compare
//! trigger jump               a trigger read
//! !(speed > 2.6)             Not
//! a > 1 && b < 2             And   (n-ary, flattened)
//! a > 1 || b < 2             Or
//! ```
//!
//! `&&` binds tighter than `||`, `!` tighter than both, and parentheses group.
//! The parser is recursive descent with an explicit depth guard (the P19 law) at
//! [`MAX_COND_PARSE_DEPTH`] — the **decoder's** bound, so the text door and the
//! binary door refuse at the same depth; the *model's* narrower one stays
//! `StateMachine::validate`'s, which the caller runs either way.
//!
//! # Names, not indices
//!
//! `entry`, `from` and `to` are written as **state names**, and `profile` as a
//! **profile name**, because an index is the one thing a reviewer cannot check.
//! A machine whose names are ambiguous or empty falls back to the integer index
//! for the affected reference, and the reader accepts either — so the projection
//! is total rather than merely usual.

use std::fmt::Write as _;

use crate::blend_space::{BlendEntry1D, BlendEntry2D, BlendSpace1D, BlendSpace2D, ClipRef};
use crate::state_machine::{
    BlendCurve, BlendProfile, CmpOp, InterruptBlend, InterruptSource, JointBlendWeight, Motion,
    SmCompare, SmCond, SmInterrupt, SmParam, SmParamKind, SmSource, SmState, SmTransition, SmValue,
    StateMachine,
};

/// How deep the condition parser may recurse before it refuses.
///
/// The **parser's** bound, not the model's, and the two answer different
/// questions — exactly as `state_machine.rs` splits its own pair. The model's
/// [`crate::state_machine::MAX_COND_DEPTH`] (16) is what `validate` holds an
/// author to; this one exists so that a hostile or generated file cannot walk
/// the stack off the end *before* anything gets to have an opinion, which is
/// P29.1's A1 lesson (a bound applied after serde has already built the tree is
/// a bound applied one stack frame too late). It is set to the decoder's own
/// 64 so the text door and the binary door refuse at the same depth.
pub const MAX_COND_PARSE_DEPTH: usize = 64;

/// Why a text machine could not be read. A **value**, like every refusal in this
/// crate — a malformed sidecar is an author's typo, not a reason to take a
/// process down.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TextError {
    #[error("not valid TOML: {0}")]
    Toml(String),
    #[error("{path}: expected {want}")]
    Expected { path: String, want: String },
    #[error("{path}: no state is called `{name}`")]
    NoSuchState { path: String, name: String },
    #[error("{path}: no blend profile is called `{name}`")]
    NoSuchProfile { path: String, name: String },
    #[error("{path}: `{text}` is not a clip id (expected 32 hex digits, hyphens optional)")]
    BadClipRef { path: String, text: String },
    #[error("{path}: `{value}` is not one of {allowed}")]
    BadEnum {
        path: String,
        value: String,
        allowed: String,
    },
    #[error("condition `{text}`: {message} at byte {at}")]
    BadCondition {
        text: String,
        message: String,
        at: usize,
    },
    #[error("condition nests deeper than {MAX_COND_PARSE_DEPTH} while parsing")]
    ConditionTooDeep,
    #[error("a state may play exactly one of `clip`, `blend1d`, `blend2d` or `machine` ({path})")]
    MotionAmbiguous { path: String },
    #[error("sub-machines are one level deep; `{path}` nests another")]
    SubMachineTooDeep { path: String },
}

// ── writing ─────────────────────────────────────────────────────────────────

/// **Project a machine into its text form.** Deterministic: the same machine
/// produces the same bytes, on every platform and in every process.
pub fn to_toml(machine: &StateMachine) -> String {
    let mut out = String::new();
    out.push_str(
        "# An Infinity Engine state machine, as text (pillar S1).\n\
         #\n\
         # This is the reviewable face of a `.inf_sm`; the binary payload beside it is\n\
         # the machine face. Every authorable value is its own line, so a tweak is a\n\
         # one-line diff. Conditions are expressions: `&&` binds tighter than `||`,\n\
         # `!` tighter than both, and `trigger <name>` reads a trigger parameter.\n",
    );
    write_machine(&mut out, machine, "");
    out
}

/// Emit `machine` at `prefix` (`""` at the top level, `"states.machine."`
/// inside a sub-machine — the one level [`Motion::SubMachine`] allows).
fn write_machine(out: &mut String, m: &StateMachine, prefix: &str) {
    let entry = state_ref(m, m.entry);
    let _ = write!(out, "\nentry = {entry}\n");

    for p in &m.params {
        let _ = write!(out, "\n[[{prefix}params]]\n");
        let _ = write!(out, "name = {}\n", quote(&p.name));
        let _ = write!(out, "kind = {}\n", quote(param_kind_name(p.kind)));
        let _ = write!(out, "default = {}\n", value_text(p.default));
    }

    for p in &m.profiles {
        let _ = write!(out, "\n[[{prefix}profiles]]\n");
        let _ = write!(out, "name = {}\n", quote(&p.name));
        let _ = write!(out, "weights = [");
        for (i, w) in p.weights.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(
                out,
                "{{ joint = {}, weight = {} }}",
                w.joint,
                f32num(w.weight)
            );
        }
        out.push_str("]\n");
    }

    for s in &m.states {
        let _ = write!(out, "\n[[{prefix}states]]\n");
        let _ = write!(out, "name = {}\n", quote(&s.name));
        let _ = write!(out, "looping = {}\n", s.looping);
        let _ = write!(out, "speed = {}\n", num(s.speed));
        let _ = write!(
            out,
            "position = [{}, {}]\n",
            f32num(s.position.0),
            f32num(s.position.1)
        );
        if !s.on_enter.is_empty() {
            let _ = write!(out, "on_enter = {}\n", string_list(&s.on_enter));
        }
        if !s.on_exit.is_empty() {
            let _ = write!(out, "on_exit = {}\n", string_list(&s.on_exit));
        }
        match &s.motion {
            Motion::Clip(c) => {
                let _ = write!(out, "clip = {}\n", quote(&clip_text(*c)));
            }
            Motion::Blend1D(b) => {
                let _ = write!(out, "blend1d = {{ param = {}, entries = [", quote(&b.param));
                for (i, e) in b.entries.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(
                        out,
                        "{{ at = {}, clip = {} }}",
                        num(e.pos),
                        quote(&clip_text(e.clip))
                    );
                }
                out.push_str("] }\n");
            }
            Motion::Blend2D(b) => {
                let _ = write!(
                    out,
                    "blend2d = {{ params = [{}, {}], entries = [",
                    quote(&b.params.0),
                    quote(&b.params.1)
                );
                for (i, e) in b.entries.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(
                        out,
                        "{{ at = [{}, {}], clip = {} }}",
                        num(e.pos[0]),
                        num(e.pos[1]),
                        quote(&clip_text(e.clip))
                    );
                }
                out.push_str("] }\n");
            }
            Motion::SubMachine(sub) => {
                // One level, so the prefix is known and this recursion terminates
                // by the same rule `validate` enforces.
                let _ = write!(out, "\n[{prefix}states.machine]");
                write_machine(out, sub, &format!("{prefix}states.machine."));
            }
        }
    }

    for t in &m.transitions {
        let _ = write!(out, "\n[[{prefix}transitions]]\n");
        match t.from {
            SmSource::State(i) => {
                let _ = write!(out, "from = {}\n", state_ref(m, i));
            }
            SmSource::Any { exclude_self } => {
                let _ = write!(out, "from = \"any\"\n");
                if !exclude_self {
                    out.push_str("any_may_re_enter = true\n");
                }
            }
        }
        let _ = write!(out, "to = {}\n", state_ref(m, t.to));
        let _ = write!(out, "duration = {}\n", num(t.duration));
        let _ = write!(out, "condition = {}\n", quote(&cond_text(&t.condition)));
        if let Some(x) = t.exit_time {
            let _ = write!(out, "exit_time = {}\n", num(x));
        }
        if t.priority != 0 {
            let _ = write!(out, "priority = {}\n", t.priority);
        }
        let _ = write!(out, "curve = {}\n", quote(curve_name(t.curve)));
        let _ = write!(
            out,
            "interrupt = {}\n",
            quote(&format!(
                "{}/{}",
                interrupt_source_name(t.interrupt.source),
                interrupt_blend_name(t.interrupt.blend)
            ))
        );
        if let Some(p) = t.profile {
            let _ = write!(out, "profile = {}\n", profile_ref(m, p));
        }
    }
}

/// A state reference: its name when that names it uniquely, else its index.
fn state_ref(m: &StateMachine, index: usize) -> String {
    match m.states.get(index) {
        Some(s)
            if !s.name.is_empty()
                && s.name != "any"
                && m.states.iter().filter(|o| o.name == s.name).count() == 1 =>
        {
            quote(&s.name)
        }
        _ => index.to_string(),
    }
}

fn profile_ref(m: &StateMachine, index: usize) -> String {
    match m.profiles.get(index) {
        Some(p)
            if !p.name.is_empty()
                && m.profiles.iter().filter(|o| o.name == p.name).count() == 1 =>
        {
            quote(&p.name)
        }
        _ => index.to_string(),
    }
}

/// Hyphenated hex, the shape every other id in this engine is written in.
fn clip_text(c: ClipRef) -> String {
    let h: String = c.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

/// TOML basic-string quoting — the escapes the spec requires, and nothing else.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\u{:04X}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn string_list(items: &[String]) -> String {
    let mut out = String::from("[");
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&quote(s));
    }
    out.push(']');
    out
}

/// A float, round-trippably, in TOML's spelling for the three non-finite cases.
fn num(v: f64) -> String {
    if v.is_nan() {
        "nan".to_string()
    } else if v == f64::INFINITY {
        "inf".to_string()
    } else if v == f64::NEG_INFINITY {
        "-inf".to_string()
    } else {
        // `{:?}` is the shortest representation that parses back to the same
        // bits, and it always carries a `.` or an exponent, so TOML reads it as a
        // float rather than an integer.
        format!("{v:?}")
    }
}

fn f32num(v: f32) -> String {
    if v.is_nan() {
        "nan".to_string()
    } else if v == f32::INFINITY {
        "inf".to_string()
    } else if v == f32::NEG_INFINITY {
        "-inf".to_string()
    } else {
        format!("{v:?}")
    }
}

fn param_kind_name(k: SmParamKind) -> &'static str {
    match k {
        SmParamKind::Bool => "bool",
        SmParamKind::Int => "int",
        SmParamKind::Float => "float",
        SmParamKind::Trigger => "trigger",
    }
}

fn curve_name(c: BlendCurve) -> &'static str {
    match c {
        BlendCurve::Linear => "linear",
        BlendCurve::EaseIn => "ease_in",
        BlendCurve::EaseOut => "ease_out",
        BlendCurve::EaseInOut => "ease_in_out",
        BlendCurve::Step => "step",
    }
}

fn interrupt_source_name(s: InterruptSource) -> &'static str {
    match s {
        InterruptSource::None => "none",
        InterruptSource::Destination => "destination",
        InterruptSource::SourceOrDestination => "source_or_destination",
    }
}

fn interrupt_blend_name(b: InterruptBlend) -> &'static str {
    match b {
        InterruptBlend::Snap => "snap",
        InterruptBlend::Carry => "carry",
    }
}

/// A parameter default, in the expression language's own vocabulary — so the
/// same three spellings mean the same three types here and in a condition.
fn value_text(v: SmValue) -> String {
    match v {
        SmValue::Bool(b) => b.to_string(),
        SmValue::Int(i) => i.to_string(),
        SmValue::Float(f) => num(f),
    }
}

fn op_text(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Gt => ">",
        CmpOp::Lt => "<",
        CmpOp::Ge => ">=",
        CmpOp::Le => "<=",
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
    }
}

/// A condition, as one line. Parentheses only where precedence needs them.
pub fn cond_text(c: &SmCond) -> String {
    fn go(c: &SmCond, out: &mut String, parent: u8) {
        // Precedence: 0 = `||`, 1 = `&&`, 2 = unary/atom.
        match c {
            SmCond::Always => out.push_str("always"),
            SmCond::Trigger(name) => {
                let _ = write!(out, "trigger {name}");
            }
            SmCond::Compare(SmCompare { param, op, value }) => {
                let _ = write!(out, "{param} {} {}", op_text(*op), value_text(*value));
            }
            SmCond::Not(inner) => {
                out.push('!');
                let mut body = String::new();
                go(inner, &mut body, 2);
                if needs_parens(inner) {
                    let _ = write!(out, "({body})");
                } else {
                    out.push_str(&body);
                }
            }
            SmCond::And(terms) | SmCond::Or(terms) => {
                let (sep, me) = if matches!(c, SmCond::And(_)) {
                    (" && ", 1)
                } else {
                    (" || ", 0)
                };
                // An empty `And` is vacuously true and an empty `Or` vacuously
                // false; both are legal models, so both need a spelling that
                // reads back as the same variant.
                if terms.is_empty() {
                    out.push_str(if me == 1 { "and()" } else { "or()" });
                    return;
                }
                let mut body = String::new();
                for (i, t) in terms.iter().enumerate() {
                    if i > 0 {
                        body.push_str(sep);
                    }
                    go(t, &mut body, me);
                }
                if me < parent {
                    let _ = write!(out, "({body})");
                } else {
                    out.push_str(&body);
                }
            }
        }
    }
    fn needs_parens(c: &SmCond) -> bool {
        match c {
            SmCond::And(t) | SmCond::Or(t) => !t.is_empty(),
            _ => false,
        }
    }
    let mut out = String::new();
    go(c, &mut out, 0);
    out
}

// ── reading ─────────────────────────────────────────────────────────────────

/// **Read a machine back out of its text form.**
///
/// Structurally total: every refusal names the key it happened at. This does
/// **not** run [`StateMachine::validate`] — the caller does, because a text
/// machine and a decoded one must be held to exactly the same door (P29.2's A1:
/// an editor that writes a file its own reader refuses).
pub fn from_toml(text: &str) -> Result<StateMachine, TextError> {
    let table: toml::Table = text
        .parse::<toml::Table>()
        .map_err(|e| TextError::Toml(e.to_string()))?;
    read_machine(&table, "")
}

fn read_machine(t: &toml::Table, path: &str) -> Result<StateMachine, TextError> {
    let at = |k: &str| {
        if path.is_empty() {
            k.to_string()
        } else {
            format!("{path}.{k}")
        }
    };

    let mut params = Vec::new();
    for (i, v) in array_of(t, "params", &at("params"))?.iter().enumerate() {
        let p = as_table(v, &format!("{}[{i}]", at("params")))?;
        let path = format!("{}[{i}]", at("params"));
        let kind = match req_str(p, "kind", &path)? {
            "bool" => SmParamKind::Bool,
            "int" => SmParamKind::Int,
            "float" => SmParamKind::Float,
            "trigger" => SmParamKind::Trigger,
            other => {
                return Err(TextError::BadEnum {
                    path: format!("{path}.kind"),
                    value: other.to_string(),
                    allowed: "bool, int, float, trigger".into(),
                })
            }
        };
        let default = match p.get("default") {
            Some(v) => read_value(v, &format!("{path}.default"))?,
            None => SmParam::new("", kind).default,
        };
        params.push(SmParam {
            name: req_str(p, "name", &path)?.to_string(),
            kind,
            default,
        });
    }

    let mut profiles = Vec::new();
    for (i, v) in array_of(t, "profiles", &at("profiles"))?.iter().enumerate() {
        let p = as_table(v, &format!("{}[{i}]", at("profiles")))?;
        let path = format!("{}[{i}]", at("profiles"));
        let mut weights = Vec::new();
        for (j, w) in array_of(p, "weights", &format!("{path}.weights"))?
            .iter()
            .enumerate()
        {
            let wp = format!("{path}.weights[{j}]");
            let w = as_table(w, &wp)?;
            weights.push(JointBlendWeight {
                joint: req_int(w, "joint", &wp)? as u16,
                weight: req_f64(w, "weight", &wp)? as f32,
            });
        }
        profiles.push(BlendProfile {
            name: req_str(p, "name", &path)?.to_string(),
            weights,
        });
    }

    let raw_states = array_of(t, "states", &at("states"))?;
    let mut states = Vec::with_capacity(raw_states.len());
    for (i, v) in raw_states.iter().enumerate() {
        let path = format!("{}[{i}]", at("states"));
        let s = as_table(v, &path)?;
        let motion_keys = ["clip", "blend1d", "blend2d", "machine"];
        let present: Vec<&str> = motion_keys
            .iter()
            .copied()
            .filter(|k| s.contains_key(*k))
            .collect();
        if present.len() != 1 {
            return Err(TextError::MotionAmbiguous { path: path.clone() });
        }
        let motion = match present[0] {
            "clip" => Motion::Clip(read_clip(s, "clip", &path)?),
            "blend1d" => {
                let b = as_table(&s["blend1d"], &format!("{path}.blend1d"))?;
                let bp = format!("{path}.blend1d");
                let mut entries = Vec::new();
                for (j, e) in array_of(b, "entries", &format!("{bp}.entries"))?
                    .iter()
                    .enumerate()
                {
                    let ep = format!("{bp}.entries[{j}]");
                    let e = as_table(e, &ep)?;
                    entries.push(BlendEntry1D {
                        pos: req_f64(e, "at", &ep)?,
                        clip: read_clip(e, "clip", &ep)?,
                    });
                }
                Motion::Blend1D(BlendSpace1D {
                    param: req_str(b, "param", &bp)?.to_string(),
                    entries,
                })
            }
            "blend2d" => {
                let b = as_table(&s["blend2d"], &format!("{path}.blend2d"))?;
                let bp = format!("{path}.blend2d");
                let axes = array_of(b, "params", &format!("{bp}.params"))?;
                if axes.len() != 2 {
                    return Err(TextError::Expected {
                        path: format!("{bp}.params"),
                        want: "two axis names".into(),
                    });
                }
                let axis = |k: usize| -> Result<String, TextError> {
                    axes[k]
                        .as_str()
                        .map(str::to_string)
                        .ok_or_else(|| TextError::Expected {
                            path: format!("{bp}.params[{k}]"),
                            want: "a string".into(),
                        })
                };
                let mut entries = Vec::new();
                for (j, e) in array_of(b, "entries", &format!("{bp}.entries"))?
                    .iter()
                    .enumerate()
                {
                    let ep = format!("{bp}.entries[{j}]");
                    let e = as_table(e, &ep)?;
                    let at = array_of(e, "at", &format!("{ep}.at"))?;
                    if at.len() != 2 {
                        return Err(TextError::Expected {
                            path: format!("{ep}.at"),
                            want: "two coordinates".into(),
                        });
                    }
                    entries.push(BlendEntry2D {
                        pos: [
                            as_f64(&at[0], &format!("{ep}.at[0]"))?,
                            as_f64(&at[1], &format!("{ep}.at[1]"))?,
                        ],
                        clip: read_clip(e, "clip", &ep)?,
                    });
                }
                Motion::Blend2D(BlendSpace2D {
                    params: (axis(0)?, axis(1)?),
                    entries,
                })
            }
            _ => {
                if !path.is_empty() && path.contains("machine") {
                    return Err(TextError::SubMachineTooDeep { path: path.clone() });
                }
                let sub = as_table(&s["machine"], &format!("{path}.machine"))?;
                Motion::SubMachine(Box::new(read_machine(sub, &format!("{path}.machine"))?))
            }
        };
        states.push(SmState {
            name: req_str(s, "name", &path)?.to_string(),
            motion,
            looping: opt_bool(s, "looping", &path)?.unwrap_or(true),
            speed: opt_f64(s, "speed", &path)?.unwrap_or(1.0),
            position: match s.get("position") {
                Some(v) => {
                    let a = as_array(v, &format!("{path}.position"))?;
                    if a.len() != 2 {
                        return Err(TextError::Expected {
                            path: format!("{path}.position"),
                            want: "two canvas coordinates".into(),
                        });
                    }
                    (
                        as_f64(&a[0], &format!("{path}.position[0]"))? as f32,
                        as_f64(&a[1], &format!("{path}.position[1]"))? as f32,
                    )
                }
                None => (0.0, 0.0),
            },
            on_enter: opt_strings(s, "on_enter", &path)?,
            on_exit: opt_strings(s, "on_exit", &path)?,
        });
    }

    // The name tables, built once — a reference is resolved against the states
    // this machine declares and against nothing else.
    let resolve_state =
        |v: &toml::Value, path: &str| -> Result<usize, TextError> {
            match v {
                toml::Value::Integer(i) => Ok(*i as usize),
                toml::Value::String(name) => states
                    .iter()
                    .position(|s| &s.name == name)
                    .ok_or_else(|| TextError::NoSuchState {
                        path: path.to_string(),
                        name: name.clone(),
                    }),
                _ => Err(TextError::Expected {
                    path: path.to_string(),
                    want: "a state name or index".into(),
                }),
            }
        };

    let entry = match t.get("entry") {
        Some(v) => resolve_state(v, &at("entry"))?,
        None => 0,
    };

    let mut transitions = Vec::new();
    for (i, v) in array_of(t, "transitions", &at("transitions"))?
        .iter()
        .enumerate()
    {
        let path = format!("{}[{i}]", at("transitions"));
        let tr = as_table(v, &path)?;
        let from_v = tr.get("from").ok_or_else(|| TextError::Expected {
            path: format!("{path}.from"),
            want: "a state name, an index, or \"any\"".into(),
        })?;
        let from = if from_v.as_str() == Some("any") {
            SmSource::Any {
                exclude_self: !opt_bool(tr, "any_may_re_enter", &path)?.unwrap_or(false),
            }
        } else {
            SmSource::State(resolve_state(from_v, &format!("{path}.from"))?)
        };
        let to_v = tr.get("to").ok_or_else(|| TextError::Expected {
            path: format!("{path}.to"),
            want: "a state name or index".into(),
        })?;
        let curve = match tr.get("curve").and_then(|v| v.as_str()).unwrap_or("linear") {
            "linear" => BlendCurve::Linear,
            "ease_in" => BlendCurve::EaseIn,
            "ease_out" => BlendCurve::EaseOut,
            "ease_in_out" => BlendCurve::EaseInOut,
            "step" => BlendCurve::Step,
            other => {
                return Err(TextError::BadEnum {
                    path: format!("{path}.curve"),
                    value: other.to_string(),
                    allowed: "linear, ease_in, ease_out, ease_in_out, step".into(),
                })
            }
        };
        let interrupt = read_interrupt(
            tr.get("interrupt").and_then(|v| v.as_str()),
            &format!("{path}.interrupt"),
        )?;
        let profile = match tr.get("profile") {
            None => None,
            Some(toml::Value::Integer(i)) => Some(*i as usize),
            Some(toml::Value::String(name)) => Some(
                profiles
                    .iter()
                    .position(|p| &p.name == name)
                    .ok_or_else(|| TextError::NoSuchProfile {
                        path: format!("{path}.profile"),
                        name: name.clone(),
                    })?,
            ),
            Some(_) => {
                return Err(TextError::Expected {
                    path: format!("{path}.profile"),
                    want: "a profile name or index".into(),
                })
            }
        };
        transitions.push(SmTransition {
            from,
            to: resolve_state(to_v, &format!("{path}.to"))?,
            duration: opt_f64(tr, "duration", &path)?.unwrap_or(0.0),
            condition: match tr.get("condition").and_then(|v| v.as_str()) {
                Some(text) => parse_cond(text)?,
                None => SmCond::Always,
            },
            exit_time: opt_f64(tr, "exit_time", &path)?,
            priority: opt_int(tr, "priority", &path)?.unwrap_or(0) as i32,
            interrupt,
            curve,
            profile,
        });
    }

    Ok(StateMachine {
        states,
        transitions,
        entry,
        params,
        profiles,
    })
}

fn read_interrupt(text: Option<&str>, path: &str) -> Result<SmInterrupt, TextError> {
    let Some(text) = text else {
        return Ok(SmInterrupt::default());
    };
    let (s, b) = text.split_once('/').ok_or_else(|| TextError::BadEnum {
        path: path.to_string(),
        value: text.to_string(),
        allowed: "<source>/<blend>, e.g. destination/carry".into(),
    })?;
    let source = match s {
        "none" => InterruptSource::None,
        "destination" => InterruptSource::Destination,
        "source_or_destination" => InterruptSource::SourceOrDestination,
        other => {
            return Err(TextError::BadEnum {
                path: path.to_string(),
                value: other.to_string(),
                allowed: "none, destination, source_or_destination".into(),
            })
        }
    };
    let blend = match b {
        "snap" => InterruptBlend::Snap,
        "carry" => InterruptBlend::Carry,
        other => {
            return Err(TextError::BadEnum {
                path: path.to_string(),
                value: other.to_string(),
                allowed: "snap, carry".into(),
            })
        }
    };
    Ok(SmInterrupt { source, blend })
}

// ── small TOML accessors, each naming its own path in the refusal ───────────

fn as_table<'a>(v: &'a toml::Value, path: &str) -> Result<&'a toml::Table, TextError> {
    v.as_table().ok_or_else(|| TextError::Expected {
        path: path.to_string(),
        want: "a table".into(),
    })
}

fn as_array<'a>(v: &'a toml::Value, path: &str) -> Result<&'a [toml::Value], TextError> {
    v.as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| TextError::Expected {
            path: path.to_string(),
            want: "an array".into(),
        })
}

fn array_of<'a>(t: &'a toml::Table, key: &str, path: &str) -> Result<&'a [toml::Value], TextError> {
    match t.get(key) {
        None => Ok(&[]),
        Some(v) => as_array(v, path),
    }
}

fn as_f64(v: &toml::Value, path: &str) -> Result<f64, TextError> {
    match v {
        toml::Value::Float(f) => Ok(*f),
        toml::Value::Integer(i) => Ok(*i as f64),
        _ => Err(TextError::Expected {
            path: path.to_string(),
            want: "a number".into(),
        }),
    }
}

fn req_str<'a>(t: &'a toml::Table, key: &str, path: &str) -> Result<&'a str, TextError> {
    t.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| TextError::Expected {
            path: format!("{path}.{key}"),
            want: "a string".into(),
        })
}

fn req_f64(t: &toml::Table, key: &str, path: &str) -> Result<f64, TextError> {
    match t.get(key) {
        Some(v) => as_f64(v, &format!("{path}.{key}")),
        None => Err(TextError::Expected {
            path: format!("{path}.{key}"),
            want: "a number".into(),
        }),
    }
}

fn req_int(t: &toml::Table, key: &str, path: &str) -> Result<i64, TextError> {
    t.get(key)
        .and_then(|v| v.as_integer())
        .ok_or_else(|| TextError::Expected {
            path: format!("{path}.{key}"),
            want: "an integer".into(),
        })
}

fn opt_f64(t: &toml::Table, key: &str, path: &str) -> Result<Option<f64>, TextError> {
    match t.get(key) {
        None => Ok(None),
        Some(v) => as_f64(v, &format!("{path}.{key}")).map(Some),
    }
}

fn opt_int(t: &toml::Table, key: &str, path: &str) -> Result<Option<i64>, TextError> {
    match t.get(key) {
        None => Ok(None),
        Some(toml::Value::Integer(i)) => Ok(Some(*i)),
        Some(_) => Err(TextError::Expected {
            path: format!("{path}.{key}"),
            want: "an integer".into(),
        }),
    }
}

fn opt_bool(t: &toml::Table, key: &str, path: &str) -> Result<Option<bool>, TextError> {
    match t.get(key) {
        None => Ok(None),
        Some(toml::Value::Boolean(b)) => Ok(Some(*b)),
        Some(_) => Err(TextError::Expected {
            path: format!("{path}.{key}"),
            want: "a boolean".into(),
        }),
    }
}

fn opt_strings(t: &toml::Table, key: &str, path: &str) -> Result<Vec<String>, TextError> {
    let mut out = Vec::new();
    for (i, v) in array_of(t, key, &format!("{path}.{key}"))?
        .iter()
        .enumerate()
    {
        out.push(
            v.as_str()
                .ok_or_else(|| TextError::Expected {
                    path: format!("{path}.{key}[{i}]"),
                    want: "a string".into(),
                })?
                .to_string(),
        );
    }
    Ok(out)
}

fn read_value(v: &toml::Value, path: &str) -> Result<SmValue, TextError> {
    match v {
        toml::Value::Boolean(b) => Ok(SmValue::Bool(*b)),
        toml::Value::Integer(i) => Ok(SmValue::Int(*i)),
        toml::Value::Float(f) => Ok(SmValue::Float(*f)),
        _ => Err(TextError::Expected {
            path: path.to_string(),
            want: "a boolean, an integer or a float".into(),
        }),
    }
}

fn read_clip(t: &toml::Table, key: &str, path: &str) -> Result<ClipRef, TextError> {
    let text = req_str(t, key, path)?;
    parse_clip(text).ok_or_else(|| TextError::BadClipRef {
        path: format!("{path}.{key}"),
        text: text.to_string(),
    })
}

/// 32 hex digits, hyphens optional and ignored.
fn parse_clip(text: &str) -> Option<ClipRef> {
    let hex: Vec<u8> = text.bytes().filter(|b| *b != b'-').collect();
    if hex.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, pair) in hex.chunks(2).enumerate() {
        let s = std::str::from_utf8(pair).ok()?;
        out[i] = u8::from_str_radix(s, 16).ok()?;
    }
    Some(out)
}

// ── the condition expression language ───────────────────────────────────────

/// Parse a condition expression. Public because the rule builder and the gate
/// both want to read one without a whole machine around it.
pub fn parse_cond(text: &str) -> Result<SmCond, TextError> {
    let mut p = Parser {
        src: text.as_bytes(),
        at: 0,
        depth: 0,
        text,
    };
    p.skip_ws();
    let c = p.parse_or()?;
    p.skip_ws();
    if p.at < p.src.len() {
        return Err(p.err("unexpected trailing input"));
    }
    Ok(c)
}

struct Parser<'a> {
    src: &'a [u8],
    at: usize,
    depth: usize,
    text: &'a str,
}

impl<'a> Parser<'a> {
    fn err(&self, message: &str) -> TextError {
        TextError::BadCondition {
            text: self.text.to_string(),
            message: message.to_string(),
            at: self.at,
        }
    }

    fn skip_ws(&mut self) {
        while self.at < self.src.len() && self.src[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }

    fn eat(&mut self, tok: &str) -> bool {
        self.skip_ws();
        if self.src[self.at..].starts_with(tok.as_bytes()) {
            self.at += tok.len();
            true
        } else {
            false
        }
    }

    /// The depth guard, in FRONT of the recursion (P29.1's A1 shape: bounding a
    /// tree after it is built is a bound applied one stack frame too late).
    fn deeper(&mut self) -> Result<(), TextError> {
        self.depth += 1;
        if self.depth > MAX_COND_PARSE_DEPTH {
            return Err(TextError::ConditionTooDeep);
        }
        Ok(())
    }

    fn parse_or(&mut self) -> Result<SmCond, TextError> {
        self.deeper()?;
        let first = self.parse_and()?;
        let mut terms = vec![first];
        while self.eat("||") {
            terms.push(self.parse_and()?);
        }
        self.depth -= 1;
        Ok(if terms.len() == 1 {
            terms.pop().expect("one term")
        } else {
            SmCond::Or(terms)
        })
    }

    fn parse_and(&mut self) -> Result<SmCond, TextError> {
        self.deeper()?;
        let first = self.parse_unary()?;
        let mut terms = vec![first];
        while self.eat("&&") {
            terms.push(self.parse_unary()?);
        }
        self.depth -= 1;
        Ok(if terms.len() == 1 {
            terms.pop().expect("one term")
        } else {
            SmCond::And(terms)
        })
    }

    fn parse_unary(&mut self) -> Result<SmCond, TextError> {
        self.skip_ws();
        if self.at >= self.src.len() {
            return Err(self.err("expected a condition"));
        }
        if self.src[self.at] == b'!' && self.src.get(self.at + 1) != Some(&b'=') {
            self.at += 1;
            self.deeper()?;
            let inner = self.parse_unary()?;
            self.depth -= 1;
            return Ok(SmCond::Not(Box::new(inner)));
        }
        if self.eat("(") {
            self.deeper()?;
            let inner = self.parse_or()?;
            self.depth -= 1;
            if !self.eat(")") {
                return Err(self.err("expected `)`"));
            }
            return Ok(inner);
        }
        self.parse_atom()
    }

    fn ident(&mut self) -> String {
        self.skip_ws();
        let start = self.at;
        while self.at < self.src.len() {
            let c = self.src[self.at];
            if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'-' {
                self.at += 1;
            } else {
                break;
            }
        }
        String::from_utf8_lossy(&self.src[start..self.at]).into_owned()
    }

    fn parse_atom(&mut self) -> Result<SmCond, TextError> {
        // The three keyword atoms come first, and each is checked as a WHOLE
        // word: a parameter called `alwaysOn` must not be read as `always`
        // followed by rubbish.
        for (kw, empty) in [("and()", true), ("or()", false)] {
            let save = self.at;
            if self.eat(kw) {
                return Ok(if empty {
                    SmCond::And(Vec::new())
                } else {
                    SmCond::Or(Vec::new())
                });
            }
            self.at = save;
        }
        let save = self.at;
        let word = self.ident();
        if word.is_empty() {
            self.at = save;
            return Err(self.err("expected a parameter name"));
        }
        if word == "always" {
            return Ok(SmCond::Always);
        }
        if word == "trigger" {
            let name = self.ident();
            if name.is_empty() {
                return Err(self.err("`trigger` needs a parameter name"));
            }
            return Ok(SmCond::Trigger(name));
        }
        self.skip_ws();
        // Two-character operators first, or `>=` reads as `>` and leaves `=`.
        let op = if self.eat(">=") {
            CmpOp::Ge
        } else if self.eat("<=") {
            CmpOp::Le
        } else if self.eat("==") {
            CmpOp::Eq
        } else if self.eat("!=") {
            CmpOp::Ne
        } else if self.eat(">") {
            CmpOp::Gt
        } else if self.eat("<") {
            CmpOp::Lt
        } else {
            return Err(self.err("expected a comparison operator"));
        };
        let value = self.parse_value()?;
        Ok(SmCond::Compare(SmCompare {
            param: word,
            op,
            value,
        }))
    }

    fn parse_value(&mut self) -> Result<SmValue, TextError> {
        self.skip_ws();
        let start = self.at;
        while self.at < self.src.len() {
            let c = self.src[self.at];
            if c.is_ascii_alphanumeric() || c == b'.' || c == b'+' || c == b'-' {
                self.at += 1;
            } else {
                break;
            }
        }
        let raw = String::from_utf8_lossy(&self.src[start..self.at]).into_owned();
        if raw.is_empty() {
            return Err(self.err("expected a value"));
        }
        match raw.as_str() {
            "true" => return Ok(SmValue::Bool(true)),
            "false" => return Ok(SmValue::Bool(false)),
            "nan" => return Ok(SmValue::Float(f64::NAN)),
            "inf" => return Ok(SmValue::Float(f64::INFINITY)),
            "-inf" => return Ok(SmValue::Float(f64::NEG_INFINITY)),
            _ => {}
        }
        // An **integer** is a value with no `.`, no exponent and no infinity —
        // the same rule `value_text` writes by, so `SmValue::Int(2)` and
        // `SmValue::Float(2.0)` survive a round trip as different values.
        let looks_float = raw.contains('.') || raw.contains('e') || raw.contains('E');
        if !looks_float {
            if let Ok(i) = raw.parse::<i64>() {
                return Ok(SmValue::Int(i));
            }
        }
        raw.parse::<f64>()
            .map(SmValue::Float)
            .map_err(|_| self.err("expected a number, `true` or `false`"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(n: u8) -> ClipRef {
        [n; 16]
    }

    /// A machine that uses **every** v2 shape the reader can decode — the
    /// round-trip fixture, and the reason the writer cannot be lossy by accident.
    fn every_shape() -> StateMachine {
        let mut m = StateMachine {
            states: vec![
                SmState::clip_at("idle", clip(1), (10.0, -20.5)),
                SmState {
                    name: "gait".into(),
                    motion: Motion::Blend1D(BlendSpace1D::new(
                        "speed",
                        vec![
                            BlendEntry1D {
                                pos: 0.0,
                                clip: clip(2),
                            },
                            BlendEntry1D {
                                pos: 3.75,
                                clip: clip(3),
                            },
                        ],
                    )),
                    looping: true,
                    speed: 1.25,
                    position: (200.0, 0.0),
                    on_enter: vec!["entered_gait".into()],
                    on_exit: vec!["left_gait".into(), "second".into()],
                },
                SmState {
                    name: "strafe".into(),
                    motion: Motion::Blend2D(BlendSpace2D::new(
                        "vx",
                        "vy",
                        vec![
                            BlendEntry2D {
                                pos: [-1.0, 0.0],
                                clip: clip(4),
                            },
                            BlendEntry2D {
                                pos: [1.0, 0.5],
                                clip: clip(5),
                            },
                        ],
                    )),
                    looping: false,
                    speed: 1.0,
                    position: (400.0, 60.0),
                    on_enter: Vec::new(),
                    on_exit: Vec::new(),
                },
                SmState {
                    name: "traversal".into(),
                    motion: Motion::SubMachine(Box::new(StateMachine {
                        states: vec![
                            SmState::clip("mantle_low", clip(6)),
                            SmState::clip("mantle_high", clip(7)),
                        ],
                        transitions: vec![SmTransition::on(
                            0,
                            1,
                            0.05,
                            "ledge_height",
                            CmpOp::Ge,
                            1.25,
                        )],
                        entry: 0,
                        params: vec![SmParam::float("ledge_height")],
                        profiles: Vec::new(),
                    })),
                    looping: false,
                    speed: 1.0,
                    position: (600.0, 0.0),
                    on_enter: Vec::new(),
                    on_exit: Vec::new(),
                },
            ],
            transitions: Vec::new(),
            entry: 0,
            params: vec![
                SmParam::float("speed"),
                SmParam::trigger("jump"),
                SmParam {
                    name: "tier".into(),
                    kind: SmParamKind::Int,
                    default: SmValue::Int(2),
                },
                SmParam {
                    name: "aiming".into(),
                    kind: SmParamKind::Bool,
                    default: SmValue::Bool(true),
                },
            ],
            profiles: vec![BlendProfile::new(
                "upper",
                vec![
                    JointBlendWeight {
                        joint: 3,
                        weight: 0.0,
                    },
                    JointBlendWeight {
                        joint: 4,
                        weight: 0.25,
                    },
                ],
            )],
        };
        m.transitions = vec![
            SmTransition {
                condition: SmCond::And(vec![
                    SmCond::float("speed", CmpOp::Gt, 0.6),
                    SmCond::Not(Box::new(SmCond::bool("aiming", true))),
                ]),
                exit_time: Some(0.25),
                priority: 3,
                curve: BlendCurve::EaseInOut,
                interrupt: SmInterrupt {
                    source: InterruptSource::SourceOrDestination,
                    blend: InterruptBlend::Snap,
                },
                profile: Some(0),
                ..SmTransition::new(0, 1, 0.15)
            },
            SmTransition {
                condition: SmCond::Or(vec![
                    SmCond::int("tier", CmpOp::Le, 1),
                    SmCond::And(vec![
                        SmCond::Trigger("jump".into()),
                        SmCond::float("speed", CmpOp::Lt, 9.5),
                    ]),
                ]),
                ..SmTransition::new(1, 0, 0.2)
            },
            SmTransition {
                condition: SmCond::Always,
                ..SmTransition::any(2, 0.1)
            },
            SmTransition {
                condition: SmCond::Trigger("jump".into()),
                ..SmTransition::new(2, 3, 0.0)
            },
        ];
        m
    }

    /// **The lossless claim**, over every shape the model has.
    #[test]
    fn every_v2_shape_round_trips_through_text() {
        let m = every_shape();
        let text = to_toml(&m);
        let back = from_toml(&text).expect("the text reads back");
        assert_eq!(back, m, "text round trip lost something:\n{text}");
        // …and the projection is a FUNCTION of the machine: re-writing what was
        // read gives the same bytes, so a save loop cannot churn a file.
        assert_eq!(to_toml(&back), text);
    }

    /// The writer is deterministic — the property every gate downstream rests on.
    #[test]
    fn the_projection_is_byte_stable() {
        let a = to_toml(&every_shape());
        let b = to_toml(&every_shape());
        assert_eq!(a, b);
    }

    /// **The one-value-one-line claim**, which is what makes a tweak a one-line
    /// diff. Asserted on the shape rather than described: every transition's
    /// blend duration is a line of its own, and changing one changes exactly one.
    #[test]
    fn a_blend_duration_is_a_line_of_its_own() {
        let m = every_shape();
        let before = to_toml(&m);
        // The sub-machine's own transition owns a line too — the count is every
        // transition in the file, at every level.
        let nested: usize = m
            .states
            .iter()
            .map(|s| match &s.motion {
                Motion::SubMachine(sub) => sub.transitions.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(
            before
                .lines()
                .filter(|l| l.starts_with("duration = "))
                .count(),
            m.transitions.len() + nested,
            "each transition owns one `duration` line"
        );
        let mut edited = m.clone();
        edited.transitions[0].duration = 0.35;
        let after = to_toml(&edited);
        let diff: Vec<(usize, &str, &str)> = before
            .lines()
            .zip(after.lines())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, a, b))
            .collect();
        assert_eq!(before.lines().count(), after.lines().count());
        assert_eq!(diff.len(), 1, "not a one-line diff: {diff:?}");
        assert_eq!(diff[0].1, "duration = 0.15");
        assert_eq!(diff[0].2, "duration = 0.35");
    }

    /// Every condition shape, through the expression language and back.
    #[test]
    fn every_condition_shape_round_trips() {
        let cases = [
            SmCond::Always,
            SmCond::Trigger("jump".into()),
            SmCond::float("speed", CmpOp::Gt, 2.6),
            SmCond::float("speed", CmpOp::Ge, -0.5),
            SmCond::int("tier", CmpOp::Eq, 3),
            SmCond::bool("aiming", false),
            SmCond::Not(Box::new(SmCond::Trigger("jump".into()))),
            SmCond::Not(Box::new(SmCond::And(vec![
                SmCond::float("a", CmpOp::Lt, 1.0),
                SmCond::float("b", CmpOp::Ne, 2.0),
            ]))),
            SmCond::And(vec![
                SmCond::Or(vec![
                    SmCond::float("a", CmpOp::Gt, 1.0),
                    SmCond::float("b", CmpOp::Gt, 2.0),
                ]),
                SmCond::float("c", CmpOp::Le, 3.0),
            ]),
            SmCond::Or(vec![
                SmCond::And(vec![
                    SmCond::float("a", CmpOp::Gt, 1.0),
                    SmCond::float("b", CmpOp::Gt, 2.0),
                ]),
                SmCond::float("c", CmpOp::Le, 3.0),
            ]),
            SmCond::And(Vec::new()),
            SmCond::Or(Vec::new()),
        ];
        for c in cases {
            let text = cond_text(&c);
            assert_eq!(parse_cond(&text).expect(&text), c, "`{text}`");
        }
    }

    /// **`&&` binds tighter than `||`**, and the writer's parentheses are what
    /// keep the two apart. Without them `a || b && c` would read back as
    /// `(a || b) && c` — a different machine that validates perfectly.
    #[test]
    fn precedence_survives_the_projection() {
        let a = SmCond::float("a", CmpOp::Gt, 1.0);
        let b = SmCond::float("b", CmpOp::Gt, 1.0);
        let c = SmCond::float("c", CmpOp::Gt, 1.0);
        let or_of_and = SmCond::Or(vec![SmCond::And(vec![a.clone(), b.clone()]), c.clone()]);
        let and_of_or = SmCond::And(vec![SmCond::Or(vec![a, b]), c]);
        assert_eq!(cond_text(&or_of_and), "a > 1.0 && b > 1.0 || c > 1.0");
        assert_eq!(cond_text(&and_of_or), "(a > 1.0 || b > 1.0) && c > 1.0");
        assert_ne!(or_of_and, and_of_or);
        assert_eq!(parse_cond(&cond_text(&or_of_and)).unwrap(), or_of_and);
        assert_eq!(parse_cond(&cond_text(&and_of_or)).unwrap(), and_of_or);
    }

    /// An `Int` and a `Float` that read the same number are different values,
    /// and the text says which is which.
    #[test]
    fn an_int_compare_does_not_become_a_float_one() {
        let i = SmCond::int("tier", CmpOp::Eq, 2);
        let f = SmCond::float("tier", CmpOp::Eq, 2.0);
        assert_eq!(cond_text(&i), "tier == 2");
        assert_eq!(cond_text(&f), "tier == 2.0");
        assert_eq!(parse_cond("tier == 2").unwrap(), i);
        assert_eq!(parse_cond("tier == 2.0").unwrap(), f);
    }

    /// The depth guard is in front of the recursion, and it refuses rather than
    /// overflowing — the P29.1 A1 lesson, applied to the text door.
    ///
    /// Both halves matter. A pathological file is a **named refusal**, not a
    /// `STATUS_STACK_OVERFLOW`; and a tree that is merely *deeper than an author
    /// may write* parses here and is refused by `validate`, exactly as it would
    /// be if it had arrived as bincode — the text door and the binary door hold
    /// the same line in the same place.
    #[test]
    fn a_pathological_condition_is_refused_rather_than_overflowing() {
        let deep = "!".repeat(200_000) + "a > 1.0";
        assert_eq!(parse_cond(&deep), Err(TextError::ConditionTooDeep));
        let nested = "(".repeat(200_000) + "a > 1.0" + &")".repeat(200_000);
        assert_eq!(parse_cond(&nested), Err(TextError::ConditionTooDeep));

        // Ten `!`s is well inside the parser's bound and well outside nothing:
        // it reads, and it is the machine it says it is.
        let ok = parse_cond(&("!".repeat(10) + "a > 1.0")).expect("ten is reachable");
        assert_eq!(ok.depth(), 11);

        // …and the MODEL's bound is still the model's: a 20-deep tree parses and
        // `validate` refuses it.
        let too_deep = parse_cond(&("!".repeat(20) + "a > 1.0")).expect("twenty parses");
        let m = StateMachine {
            states: vec![SmState::clip("a", clip(1)), SmState::clip("b", clip(2))],
            transitions: vec![SmTransition {
                condition: too_deep,
                ..SmTransition::new(0, 1, 0.1)
            }],
            entry: 0,
            params: Vec::new(),
            profiles: Vec::new(),
        };
        assert!(
            matches!(
                m.validate(),
                Err(crate::state_machine::SmError::ConditionTooDeep { .. })
            ),
            "the model's bound must still be the model's"
        );
    }

    /// Malformed text is a **value**: named, with the key it happened at.
    #[test]
    fn every_refusal_names_where_it_happened() {
        assert!(matches!(
            from_toml("this is not toml"),
            Err(TextError::Toml(_))
        ));
        assert!(matches!(
            from_toml("entry = \"nowhere\"\n[[states]]\nname = \"a\"\nclip = \"00000000-0000-0000-0000-000000000000\"\n"),
            Err(TextError::NoSuchState { .. })
        ));
        assert!(matches!(
            from_toml("[[states]]\nname = \"a\"\nclip = \"not-a-clip\"\n"),
            Err(TextError::BadClipRef { .. })
        ));
        assert!(matches!(
            from_toml("[[states]]\nname = \"a\"\n"),
            Err(TextError::MotionAmbiguous { .. })
        ));
        assert!(matches!(
            from_toml(
                "[[states]]\nname = \"a\"\nclip = \"00000000-0000-0000-0000-000000000000\"\nblend1d = { param = \"s\", entries = [] }\n"
            ),
            Err(TextError::MotionAmbiguous { .. })
        ));
        assert!(matches!(
            from_toml("[[params]]\nname = \"p\"\nkind = \"quaternion\"\n"),
            Err(TextError::BadEnum { .. })
        ));
    }

    /// A clip id round-trips through its hyphenated spelling, and a hyphen-free
    /// one reads too (the two ways a GUID is written down).
    #[test]
    fn a_clip_id_is_a_guid() {
        let c: ClipRef = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
            0x32, 0x10,
        ];
        assert_eq!(clip_text(c), "01234567-89ab-cdef-fedc-ba9876543210");
        assert_eq!(parse_clip(&clip_text(c)), Some(c));
        assert_eq!(parse_clip("0123456789abcdeffedcba9876543210"), Some(c));
        assert_eq!(parse_clip("0123"), None);
    }

    /// Two states with the same name fall back to indices rather than to a
    /// reference that resolves to the wrong one — the projection is total.
    #[test]
    fn ambiguous_names_fall_back_to_indices() {
        let m = StateMachine {
            states: vec![SmState::clip("dup", clip(1)), SmState::clip("dup", clip(2))],
            transitions: vec![SmTransition::new(1, 0, 0.1)],
            entry: 1,
            params: Vec::new(),
            profiles: Vec::new(),
        };
        let text = to_toml(&m);
        assert!(text.contains("entry = 1"), "{text}");
        assert!(text.contains("from = 1"), "{text}");
        assert_eq!(from_toml(&text).unwrap(), m);
    }
}
