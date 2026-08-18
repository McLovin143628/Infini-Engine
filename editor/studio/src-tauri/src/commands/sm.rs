//! Ring-2 command surface for the animation State Machine editor (P11.2).
//!
//! Unlike the Blueprint / material / PCG editors (which ride the `inf-graph`
//! dataflow substrate + a node registry), a state machine is a **plain typed
//! model** — states and transitions with layout positions. So there is no node
//! registry and no graph-compile step: the frontend edits a DTO mirror of
//! `inf_anim::StateMachine` and pushes the whole document back on
//! [`sm_save`], which converts it to a [`StateMachineAsset`] and writes the
//! `.inf_sm` (the engine's dual-format bincode payload).
//!
//! Commands: `sm_list` / `sm_create` / `sm_get` / `sm_save`, plus `sm_list_clips`
//! (the imported `.inf_anim` clips a state's Clip motion can point at).

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::State;
use uuid::Uuid;

use inf_anim::{
    BlendCurve, BlendEntry1D, BlendEntry2D, BlendProfile, BlendSpace1D, BlendSpace2D, CmpOp,
    InterruptBlend, InterruptSource, JointBlendWeight, Motion, SmCompare, SmCond, SmInterrupt,
    SmParam, SmParamKind, SmSource, SmState, SmTransition, SmValue, StateMachine,
    StateMachineAsset,
};
use inf_asset::AssetKind;

use super::assets::AssetState;

/// The nil clip GUID (`[0;16]`) — a Clip motion with no clip assigned yet.
const NIL_CLIP: [u8; 16] = [0u8; 16];

// ── DTOs (camelCase wire mirror of `inf_anim::StateMachine`) ─────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmDoc {
    pub id: String,
    pub name: String,
    pub machine: SmMachineDto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmMachineDto {
    pub states: Vec<SmStateDto>,
    pub transitions: Vec<SmTransitionDto>,
    pub entry: usize,
    /// Declared parameters (v2).
    #[serde(default)]
    pub params: Vec<SmParamDto>,
    /// Named per-joint blend masks a transition may point at (v2).
    #[serde(default)]
    pub profiles: Vec<SmProfileDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmStateDto {
    pub name: String,
    pub motion: SmMotionDto,
    pub looping: bool,
    pub speed: f64,
    pub x: f32,
    pub y: f32,
    /// Notify names emitted when this state is entered / exited (v2).
    #[serde(default)]
    pub on_enter: Vec<String>,
    #[serde(default)]
    pub on_exit: Vec<String>,
}

/// A state's motion, tagged by `kind`. v1 UI edits only `clip`; blend spaces
/// round-trip faithfully so a data-authored machine is never lossy on save.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SmMotionDto {
    /// A single clip; `clip` is a GUID string, or `null` when unassigned.
    Clip { clip: Option<String> },
    Blend1d {
        param: String,
        entries: Vec<Blend1dEntryDto>,
    },
    Blend2d {
        /// **Renamed explicitly, for the same reason `SmCondDto::Compare`'s
        /// `valueKind` is.** `rename_all = "camelCase"` on a tagged enum renames
        /// the *variants*, not their fields, so these two crossed the IPC
        /// boundary as `param_x` / `param_y` while the hand-written TypeScript
        /// mirror has read `paramX` / `paramY` since P11.2 — a 2D blend space's
        /// axis names have therefore been `undefined` in the editor for the
        /// whole life of the field. The P29.1 audit found it two fields away
        /// from the one the implementation battery caught, which is why the
        /// round-trip's wire-spelling arm now enumerates **every** renamed key
        /// in both tagged enums instead of sampling three.
        #[serde(rename = "paramX")]
        param_x: String,
        #[serde(rename = "paramY")]
        param_y: String,
        entries: Vec<Blend2dEntryDto>,
    },
    /// A nested machine (v2). Recursive, and lossless: the editor has no
    /// sub-machine UI yet, and a DTO that could not carry one would mean opening
    /// a machine with a nested state and saving it deleted the nesting.
    SubMachine { machine: Box<SmMachineDto> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Blend1dEntryDto {
    pub pos: f64,
    pub clip: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Blend2dEntryDto {
    pub x: f64,
    pub y: f64,
    pub clip: Option<String>,
}

/// A transition, camelCase.
///
/// # Why there are TWO condition fields, and which one wins
///
/// v1's inspector edits a **list** of `var op value` rows, and the overwhelming
/// majority of authored transitions still are exactly that. v2's model is a
/// tree, and a UI that can only draw the list would silently flatten — that is,
/// destroy — an `Or`, a `Not`, a trigger or a typed compare the moment somebody
/// opened a machine and pressed save.
///
/// So: `condition` is **always** the whole tree and is what a save falls back
/// to; `conditions` is `Some` only when the tree *is* a flat AND of float
/// compares, and is what the flat inspector binds to. A client that edited the
/// flat view sends it back and it wins; a client that could not draw the tree
/// receives `conditions: null`, has nothing to bind, and sends `condition`
/// through untouched. The rule is one line in [`dto_to_transition`] and the
/// property it buys is that no UI can lose data it cannot see.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmTransitionDto {
    /// Source state index, or `null` for an **any-state** transition.
    pub from: Option<usize>,
    /// Any-state only: whether the transition may re-enter the state it targets.
    #[serde(default)]
    pub exclude_self: bool,
    pub to: usize,
    pub duration: f64,
    /// The flat view — `null` when the tree is not a flat AND of float compares.
    pub conditions: Option<Vec<SmConditionDto>>,
    /// The full tree, always present.
    pub condition: SmCondDto,
    pub exit_time: Option<f64>,
    pub priority: i32,
    /// `"none"` / `"destination"` / `"sourceOrDestination"`.
    pub interrupt_source: String,
    /// `"snap"` / `"carry"`.
    pub interrupt_blend: String,
    /// `"linear"` / `"easeIn"` / `"easeOut"` / `"easeInOut"` / `"step"`.
    pub curve: String,
    /// Index into [`SmMachineDto::profiles`].
    pub profile: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmConditionDto {
    /// The parameter name (v1 called this `var`; the wire keeps that spelling so
    /// the flat inspector is unchanged).
    pub var: String,
    /// One of `">"`, `"<"`, `">="`, `"<="`, `"=="`, `"!="`.
    pub op: String,
    pub value: f64,
}

/// A condition tree node, tagged by `kind` — the shape a rule builder edits.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SmCondDto {
    Always,
    Compare {
        param: String,
        op: String,
        /// The typed operand, as a number; `valueKind` says how to read it.
        value: f64,
        /// `"bool"` / `"int"` / `"float"` / `"trigger"`.
        ///
        /// **Renamed explicitly.** `rename_all = "camelCase"` on a tagged enum
        /// renames the *variants*, not their fields, so this crossed the IPC
        /// boundary as `value_kind` while the hand-written TypeScript mirror
        /// read `valueKind` — every typed compare would have parsed as
        /// `undefined` on the far side. Caught by the round-trip's
        /// wire-spelling arm; the field-name check is why that arm asserts on
        /// the JSON rather than only on the decoded value.
        #[serde(rename = "valueKind")]
        value_kind: String,
    },
    Trigger {
        param: String,
    },
    And {
        terms: Vec<SmCondDto>,
    },
    Or {
        terms: Vec<SmCondDto>,
    },
    Not {
        term: Box<SmCondDto>,
    },
}

/// A declared parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmParamDto {
    pub name: String,
    /// `"bool"` / `"int"` / `"float"` / `"trigger"`.
    pub kind: String,
    /// The default, as a number (`bool` reads at `> 0.5`, matching the engine).
    pub default: f64,
}

/// A named per-joint blend mask.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmProfileDto {
    pub name: String,
    pub weights: Vec<SmJointWeightDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmJointWeightDto {
    pub joint: u16,
    pub weight: f32,
}

/// One imported `.inf_anim` clip, for the motion picker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmClipDto {
    pub id: String,
    pub name: String,
}

// ── conversion helpers ───────────────────────────────────────────────────────

fn guid_to_string(bytes: [u8; 16]) -> Option<String> {
    if bytes == NIL_CLIP {
        None
    } else {
        Some(Uuid::from_bytes(bytes).to_string())
    }
}

fn string_to_guid(s: &Option<String>) -> [u8; 16] {
    s.as_ref()
        .and_then(|s| Uuid::parse_str(s).ok())
        .map(|u| u.into_bytes())
        .unwrap_or(NIL_CLIP)
}

fn op_to_str(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Gt => ">",
        CmpOp::Lt => "<",
        CmpOp::Ge => ">=",
        CmpOp::Le => "<=",
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
    }
}

fn str_to_op(s: &str) -> CmpOp {
    match s {
        "<" => CmpOp::Lt,
        ">=" => CmpOp::Ge,
        "<=" => CmpOp::Le,
        "==" => CmpOp::Eq,
        "!=" => CmpOp::Ne,
        _ => CmpOp::Gt,
    }
}

fn motion_to_dto(m: &Motion) -> SmMotionDto {
    match m {
        Motion::Clip(id) => SmMotionDto::Clip {
            clip: guid_to_string(*id),
        },
        Motion::Blend1D(sp) => SmMotionDto::Blend1d {
            param: sp.param.clone(),
            entries: sp
                .entries
                .iter()
                .map(|e| Blend1dEntryDto {
                    pos: e.pos,
                    clip: guid_to_string(e.clip),
                })
                .collect(),
        },
        Motion::Blend2D(sp) => SmMotionDto::Blend2d {
            param_x: sp.params.0.clone(),
            param_y: sp.params.1.clone(),
            entries: sp
                .entries
                .iter()
                .map(|e| Blend2dEntryDto {
                    x: e.pos[0],
                    y: e.pos[1],
                    clip: guid_to_string(e.clip),
                })
                .collect(),
        },
        Motion::SubMachine(inner) => SmMotionDto::SubMachine {
            machine: Box::new(machine_to_dto(inner)),
        },
    }
}

fn dto_to_motion(m: &SmMotionDto) -> Motion {
    match m {
        SmMotionDto::Clip { clip } => Motion::Clip(string_to_guid(clip)),
        SmMotionDto::Blend1d { param, entries } => Motion::Blend1D(BlendSpace1D::new(
            param.clone(),
            entries
                .iter()
                .map(|e| BlendEntry1D {
                    pos: e.pos,
                    clip: string_to_guid(&e.clip),
                })
                .collect(),
        )),
        SmMotionDto::Blend2d {
            param_x,
            param_y,
            entries,
        } => Motion::Blend2D(BlendSpace2D::new(
            param_x.clone(),
            param_y.clone(),
            entries
                .iter()
                .map(|e| BlendEntry2D {
                    pos: [e.x, e.y],
                    clip: string_to_guid(&e.clip),
                })
                .collect(),
        )),
        SmMotionDto::SubMachine { machine } => {
            Motion::SubMachine(Box::new(dto_to_machine(machine)))
        }
    }
}

// -- v2 enum spellings (the wire strings) --------------------------------------
//
// Strings rather than indices on purpose: an index is what a re-ordered enum
// silently re-interprets, and this wire has a hand-written TypeScript mirror on
// the far side of it. An unrecognized string falls back to the v1-equivalent
// value and never panics -- a Tauri command that panics takes its handler down.

fn curve_to_str(c: BlendCurve) -> &'static str {
    match c {
        BlendCurve::Linear => "linear",
        BlendCurve::EaseIn => "easeIn",
        BlendCurve::EaseOut => "easeOut",
        BlendCurve::EaseInOut => "easeInOut",
        BlendCurve::Step => "step",
    }
}

fn str_to_curve(s: &str) -> BlendCurve {
    match s {
        "easeIn" => BlendCurve::EaseIn,
        "easeOut" => BlendCurve::EaseOut,
        "easeInOut" => BlendCurve::EaseInOut,
        "step" => BlendCurve::Step,
        _ => BlendCurve::Linear,
    }
}

fn source_to_str(s: InterruptSource) -> &'static str {
    match s {
        InterruptSource::None => "none",
        InterruptSource::Destination => "destination",
        InterruptSource::SourceOrDestination => "sourceOrDestination",
    }
}

fn str_to_source(s: &str) -> InterruptSource {
    match s {
        "none" => InterruptSource::None,
        "sourceOrDestination" => InterruptSource::SourceOrDestination,
        _ => InterruptSource::Destination,
    }
}

fn blend_to_str(b: InterruptBlend) -> &'static str {
    match b {
        InterruptBlend::Snap => "snap",
        InterruptBlend::Carry => "carry",
    }
}

fn str_to_blend(s: &str) -> InterruptBlend {
    match s {
        "snap" => InterruptBlend::Snap,
        _ => InterruptBlend::Carry,
    }
}

fn kind_to_str(k: SmParamKind) -> &'static str {
    match k {
        SmParamKind::Bool => "bool",
        SmParamKind::Int => "int",
        SmParamKind::Float => "float",
        SmParamKind::Trigger => "trigger",
    }
}

fn str_to_kind(s: &str) -> SmParamKind {
    match s {
        "bool" => SmParamKind::Bool,
        "int" => SmParamKind::Int,
        "trigger" => SmParamKind::Trigger,
        _ => SmParamKind::Float,
    }
}

fn value_to_dto(v: SmValue) -> (f64, &'static str) {
    match v {
        SmValue::Bool(b) => (b as i64 as f64, "bool"),
        SmValue::Int(i) => (i as f64, "int"),
        SmValue::Float(f) => (f, "float"),
    }
}

/// The typed operand, back from the numeric wire.
///
/// **`"trigger"` is here, and its absence was a real lossiness bug** that
/// `machine_dto_round_trips_every_v2_shape` caught on the first battery: a
/// `Trigger` parameter's default is `SmValue::Bool(false)`, and a `_ =>` arm
/// turned it into `Float(0.0)` on every round-trip. The two compare identically
/// through `as_f64`, so nothing downstream would have noticed — the machine
/// would simply have been re-saved with a different declared default than the one
/// it was authored with, for ever.
fn dto_to_value(value: f64, kind: &str) -> SmValue {
    match kind {
        // A trigger's operand is a flag, exactly like a bool's.
        "bool" | "trigger" => SmValue::Bool(value > 0.5),
        "int" => SmValue::Int(value as i64),
        _ => SmValue::Float(value),
    }
}

fn cond_to_dto(c: &SmCond) -> SmCondDto {
    match c {
        SmCond::Always => SmCondDto::Always,
        SmCond::Compare(cmp) => {
            let (value, value_kind) = value_to_dto(cmp.value);
            SmCondDto::Compare {
                param: cmp.param.clone(),
                op: op_to_str(cmp.op).to_string(),
                value,
                value_kind: value_kind.to_string(),
            }
        }
        SmCond::Trigger(name) => SmCondDto::Trigger {
            param: name.clone(),
        },
        SmCond::And(terms) => SmCondDto::And {
            terms: terms.iter().map(cond_to_dto).collect(),
        },
        SmCond::Or(terms) => SmCondDto::Or {
            terms: terms.iter().map(cond_to_dto).collect(),
        },
        SmCond::Not(inner) => SmCondDto::Not {
            term: Box::new(cond_to_dto(inner)),
        },
    }
}

fn dto_to_cond(c: &SmCondDto) -> SmCond {
    match c {
        SmCondDto::Always => SmCond::Always,
        SmCondDto::Compare {
            param,
            op,
            value,
            value_kind,
        } => SmCond::Compare(SmCompare {
            param: param.clone(),
            op: str_to_op(op),
            value: dto_to_value(*value, value_kind),
        }),
        SmCondDto::Trigger { param } => SmCond::Trigger(param.clone()),
        SmCondDto::And { terms } => SmCond::And(terms.iter().map(dto_to_cond).collect()),
        SmCondDto::Or { terms } => SmCond::Or(terms.iter().map(dto_to_cond).collect()),
        SmCondDto::Not { term } => SmCond::Not(Box::new(dto_to_cond(term))),
    }
}

fn transition_to_dto(t: &SmTransition) -> SmTransitionDto {
    let (from, exclude_self) = match t.from {
        SmSource::State(i) => (Some(i), false),
        SmSource::Any { exclude_self } => (None, exclude_self),
    };
    SmTransitionDto {
        from,
        exclude_self,
        to: t.to,
        duration: t.duration,
        // `Some` ONLY when the flat inspector can represent the whole tree.
        conditions: t.condition.as_flat_and().map(|flat| {
            flat.into_iter()
                .map(|c| SmConditionDto {
                    var: c.param,
                    op: op_to_str(c.op).to_string(),
                    value: c.value.as_f64(),
                })
                .collect()
        }),
        condition: cond_to_dto(&t.condition),
        exit_time: t.exit_time,
        priority: t.priority,
        interrupt_source: source_to_str(t.interrupt.source).to_string(),
        interrupt_blend: blend_to_str(t.interrupt.blend).to_string(),
        curve: curve_to_str(t.curve).to_string(),
        profile: t.profile,
    }
}

fn dto_to_transition(t: &SmTransitionDto) -> SmTransition {
    SmTransition {
        from: match t.from {
            Some(i) => SmSource::State(i),
            None => SmSource::Any {
                exclude_self: t.exclude_self,
            },
        },
        to: t.to,
        duration: t.duration,
        // The flat view wins when the client sent one -- it is the surface that
        // was edited. A client that received `null` (because the tree is not
        // flat) has nothing to send back, and the tree passes through untouched.
        condition: match &t.conditions {
            Some(flat) => SmCond::from_flat_and(
                flat.iter()
                    .map(|c| SmCompare::float(c.var.clone(), str_to_op(&c.op), c.value))
                    .collect(),
            ),
            None => dto_to_cond(&t.condition),
        },
        exit_time: t.exit_time,
        priority: t.priority,
        interrupt: SmInterrupt {
            source: str_to_source(&t.interrupt_source),
            blend: str_to_blend(&t.interrupt_blend),
        },
        curve: str_to_curve(&t.curve),
        profile: t.profile,
    }
}

fn machine_to_dto(m: &StateMachine) -> SmMachineDto {
    SmMachineDto {
        states: m
            .states
            .iter()
            .map(|s| SmStateDto {
                name: s.name.clone(),
                motion: motion_to_dto(&s.motion),
                looping: s.looping,
                speed: s.speed,
                x: s.position.0,
                y: s.position.1,
                on_enter: s.on_enter.clone(),
                on_exit: s.on_exit.clone(),
            })
            .collect(),
        transitions: m.transitions.iter().map(transition_to_dto).collect(),
        entry: m.entry,
        params: m
            .params
            .iter()
            .map(|p| SmParamDto {
                name: p.name.clone(),
                kind: kind_to_str(p.kind).to_string(),
                default: p.default.as_f64(),
            })
            .collect(),
        profiles: m
            .profiles
            .iter()
            .map(|p| SmProfileDto {
                name: p.name.clone(),
                weights: p
                    .weights
                    .iter()
                    .map(|w| SmJointWeightDto {
                        joint: w.joint,
                        weight: w.weight,
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn dto_to_machine(m: &SmMachineDto) -> StateMachine {
    StateMachine {
        states: m
            .states
            .iter()
            .map(|s| SmState {
                name: s.name.clone(),
                motion: dto_to_motion(&s.motion),
                looping: s.looping,
                speed: s.speed,
                position: (s.x, s.y),
                on_enter: s.on_enter.clone(),
                on_exit: s.on_exit.clone(),
            })
            .collect(),
        transitions: m.transitions.iter().map(dto_to_transition).collect(),
        entry: m.entry,
        params: m
            .params
            .iter()
            .map(|p| SmParam {
                name: p.name.clone(),
                kind: str_to_kind(&p.kind),
                default: dto_to_value(p.default, &p.kind),
            })
            .collect(),
        profiles: m
            .profiles
            .iter()
            .map(|p| BlendProfile {
                name: p.name.clone(),
                weights: p
                    .weights
                    .iter()
                    .map(|w| JointBlendWeight {
                        joint: w.joint,
                        weight: w.weight,
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// A fresh machine with a single idle state (unassigned clip) at the origin.
fn default_doc(id: String, name: String) -> SmDoc {
    let machine = StateMachine {
        states: vec![SmState::clip("Idle", NIL_CLIP)],
        transitions: vec![],
        entry: 0,
        ..Default::default()
    };
    SmDoc {
        id,
        name,
        machine: machine_to_dto(&machine),
    }
}

// ── state ────────────────────────────────────────────────────────────────────

struct SmStore {
    docs: BTreeMap<String, SmDoc>,
    counter: u32,
}

/// The Tauri-managed state for the state-machine editor (named `SmEditorState`,
/// not `SmState`, to avoid colliding with `inf_anim::SmState`).
pub struct SmEditorState {
    inner: Mutex<SmStore>,
}

impl Default for SmEditorState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(SmStore {
                docs: BTreeMap::new(),
                counter: 0,
            }),
        }
    }
}

impl SmEditorState {
    /// Drop a document from the workspace. Idempotent (closing an unknown id is a
    /// no-op).
    fn close(&self, id: &str) -> Result<(), String> {
        let mut s = self.inner.lock().map_err(|e| e.to_string())?;
        s.docs.remove(id);
        Ok(())
    }
}

// ── commands ─────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn sm_list(state: State<'_, SmEditorState>) -> Result<Vec<SmDoc>, String> {
    let s = state.inner.lock().map_err(|e| e.to_string())?;
    Ok(s.docs.values().cloned().collect())
}

#[tauri::command]
pub async fn sm_create(name: String, state: State<'_, SmEditorState>) -> Result<SmDoc, String> {
    let mut s = state.inner.lock().map_err(|e| e.to_string())?;
    s.counter += 1;
    let id = format!("sm{}", s.counter);
    let name = if name.is_empty() {
        format!("StateMachine{}", s.counter)
    } else {
        name
    };
    let doc = default_doc(id.clone(), name);
    s.docs.insert(id, doc.clone());
    Ok(doc)
}

#[tauri::command]
pub async fn sm_get(id: String, state: State<'_, SmEditorState>) -> Result<SmDoc, String> {
    let s = state.inner.lock().map_err(|e| e.to_string())?;
    s.docs
        .get(&id)
        .cloned()
        .ok_or_else(|| format!("no sm `{id}`"))
}

/// Persist the pushed document into a `.inf_sm` asset **and** update the
/// in-memory doc. Returns the written file name.
///
/// **Through the asset door** (C4-21) — `pcg_save`'s twin, and it had the same
/// defect: a bare `fs::write` into `<Content>/StateMachines/<slug>.inf_sm`,
/// non-atomic and with no sidecar, so the payload was promoted by the watcher
/// under a content-derived id that changed on every save.
///
/// # The rig this machine was authored against (round 3)
///
/// `skeleton_binding` records a rig's content hash in an animation asset's
/// sidecar so a re-imported rig can say *"every track index past the edit point
/// now names a different bone"*. It had **one** producer, the glTF importer, so
/// the two doors in this engine that author skeleton-bound animation — the
/// character wizard, and this — recorded nothing and were invisible to it.
///
/// A `.inf_sm` is the harder half, because it names **no skeleton of its own**:
/// `dto_to_machine` builds a machine whose states hold clip GUIDs and the
/// editor's DTO has no rig field. So the rig is resolved the only way it can be
/// — from the clips the machine plays (`StateMachine::clip_refs`, the one walk
/// that closes the machine→clip edge) — and the same resolution supplies the
/// **dependency edges**, which this door wrote as `Vec::new()`. That omission
/// was its own small defect: the delete-with-references guard and the drawer's
/// reference view could not see that a machine used a clip at all.
///
/// A machine with no assigned clips, or clips that name no rig, records
/// nothing. That is the honest answer and it is the silent one — the same rule
/// `advisories` states for an asset imported before the key existed.
#[tauri::command]
pub async fn sm_save(
    id: String,
    doc: SmDoc,
    name: String,
    state: State<'_, SmEditorState>,
    assets: State<'_, AssetState>,
) -> Result<String, String> {
    // Update in-memory + build the payload.
    let (machine, file_name) = {
        let mut s = state.inner.lock().map_err(|e| e.to_string())?;
        let machine = dto_to_machine(&doc.machine);
        // **A door must not manufacture a file its own reader rejects** (the
        // P23.6 ruling, met again at P29.1's audit). `StateMachine::validate`
        // had exactly ONE production caller — `StateMachineAsset::migrate`, on
        // the READ side — so this door wrote whatever the DTO said. v1 survived
        // that: its `migrate` asked no structural question, so a transition
        // index left dangling by a client bug was `.min()`-clamped at
        // evaluation and the file still opened. v2's `migrate` refuses it, which
        // turns the same client bug into a `.inf_sm` the editor can write and
        // then never load again. Asking here costs one walk of a document
        // somebody just edited, and the answer is a message rather than a dead
        // file.
        machine
            .validate()
            .map_err(|e| format!("this state machine cannot be saved: {e}"))?;
        s.docs.insert(id.clone(), doc.clone());
        let base = if name.is_empty() { &doc.name } else { &name };
        let slug: String = base
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        (machine, format!("{slug}.inf_sm"))
    };

    // **Off the async workers** (round-2, the sm/pcg MED). Wave A's atomic-write
    // conversion deleted the `spawn_blocking` wrapper AND the comment that said
    // why ("keep it off the async workers"), so a disk write ran in this
    // `async fn`'s body under two mutexes — inverting the same campaign's
    // Wave-E rule two waves later. `project_handle` exists because
    // `with_project` borrows through `&State`, which is not `Send`.
    let project = assets.project_handle()?;
    tauri::async_runtime::spawn_blocking(move || {
        let mut proj = project.lock().map_err(|e| e.to_string())?;
        let (skeleton, deps) = machine_binding(&proj, &machine);
        let import = inf_editor_core::assets::skeleton_binding::import_table(&proj, skeleton);
        let payload = StateMachineAsset::new(machine, skeleton.map(|s| *s.uuid().as_bytes()));
        let dir = proj
            .content_dir("StateMachines")
            .map_err(|e| e.to_string())?;
        proj.write_asset_at(&dir.join(&file_name), &payload, deps, import)
            .map_err(|e| e.to_string())?;
        Ok(file_name)
    })
    .await
    .map_err(|e| format!("asset write task failed to run: {e}"))?
}

/// The rig a machine is bound to, and the dependency edges that say so: every
/// clip it plays that the project has, plus the rig itself.
///
/// Deterministic — `clip_refs` is a `BTreeSet`, so the edge list is in ascending
/// GUID order and two saves of the same document write the same sidecar.
///
/// The rig is the first one any of the clips names. Clips that disagree are not
/// resolved here: a machine blending two rigs is a broken machine, and
/// `skeleton_binding::advisories` is the thing that says so — from the clips'
/// own recorded hashes, which is where that question belongs.
fn machine_binding(
    proj: &inf_editor_core::assets::AssetProject,
    machine: &StateMachine,
) -> (Option<inf_asset::AssetId>, Vec<inf_asset::AssetId>) {
    let mut deps: Vec<inf_asset::AssetId> = Vec::new();
    let mut skeleton = None;
    for c in machine.clip_refs() {
        let clip = inf_asset::AssetId(Uuid::from_bytes(c));
        if proj
            .db()
            .get(clip)
            .is_none_or(|e| e.kind() != AssetKind::AnimClip)
        {
            continue; // unassigned (the nil GUID), or a clip this project lost
        }
        deps.push(clip);
        if skeleton.is_none() {
            skeleton = inf_editor_core::assets::skeleton_binding::skeleton_of(proj, clip);
        }
    }
    if let Some(s) = skeleton {
        deps.push(s);
    }
    (skeleton, deps)
}

/// Close a document: free it from the workspace so open state machines don't
/// accumulate for the life of the session. Called when the editing surface is
/// discarded.
#[tauri::command]
pub async fn sm_close(id: String, state: State<'_, SmEditorState>) -> Result<(), String> {
    state.close(&id)
}

/// The imported `.inf_anim` clips (for a Clip motion's picker).
#[tauri::command]
pub async fn sm_list_clips(assets: State<'_, AssetState>) -> Result<Vec<SmClipDto>, String> {
    assets.with_project(|p| {
        Ok(p.db()
            .iter()
            .filter(|e| e.kind() == AssetKind::AnimClip)
            .map(|e| SmClipDto {
                id: e.id().to_string(),
                name: e.name.clone(),
            })
            .collect())
    })
}

/// A proposed machine, its reasoning, and the triggers gameplay has to arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmProposalDto {
    /// The machine, in exactly the shape [`sm_save`] takes — because it **is** a
    /// normal machine, and the panel loads it into the same canvas.
    pub machine: SmMachineDto,
    /// One line per decision, in the order they were taken.
    pub notes: Vec<String>,
    /// The trigger parameters the one-shots declared.
    pub triggers: Vec<String>,
    /// Why nothing was proposed. A **value**.
    pub refusal: Option<String>,
}

/// **Propose a state machine over a clip set** (P29.5, pillar S3).
///
/// The rule is `inf_anim::propose_machine`; this resolves the ids, reads each
/// clip's derived facts back off it (`inf_anim::facts_of`) and projects the
/// result into the same DTO the canvas already edits. Nothing is written — the
/// author gets a document, looks at it, changes it and saves it through
/// [`sm_save`] like any other, which is what "proposed" means.
///
/// A clip that cannot be loaded is **skipped with a note** rather than failing
/// the proposal: a set of forty clips with one broken asset in it should still
/// propose thirty-nine.
#[tauri::command]
pub async fn sm_propose(
    clips: Vec<String>,
    assets: State<'_, AssetState>,
) -> Result<SmProposalDto, String> {
    let mut ids: Vec<(inf_asset::AssetId, String)> = Vec::new();
    for c in &clips {
        let id = c
            .parse::<inf_asset::AssetId>()
            .map_err(|e| format!("bad clip id `{c}`: {e}"))?;
        ids.push((id, c.clone()));
    }
    let opts = inf_anim::ProposalOptions::default();
    let (facts, mut notes) = assets.with_project(|p| {
        let mut facts = Vec::new();
        let mut notes = Vec::new();
        for (id, _) in &ids {
            let Some(entry) = p.db().get(*id) else {
                notes.push(format!("skipped {id}: no such asset"));
                continue;
            };
            if entry.kind() != AssetKind::AnimClip {
                notes.push(format!("skipped `{}`: not an animation clip", entry.name));
                continue;
            }
            let name = entry.name.clone();
            match p.load_payload::<inf_anim::AnimClipAsset>(*id) {
                // The SAME ladder the proposal clusters against — `W_Gait` is a
                // tier and not a speed, so reading one back needs the three
                // numbers it was written against.
                Ok(a) => facts.push(inf_anim::facts_of(
                    name,
                    id.uuid().into_bytes(),
                    &a.clip,
                    opts.gait_speeds_mps,
                )),
                Err(e) => notes.push(format!("skipped `{name}`: {e}")),
            }
        }
        Ok((facts, notes))
    })?;

    match inf_anim::propose_machine(&facts, &opts) {
        Ok(p) => {
            notes.extend(p.notes);
            Ok(SmProposalDto {
                machine: machine_to_dto(&p.machine),
                notes,
                triggers: p.triggers,
                refusal: None,
            })
        }
        Err(e) => Ok(SmProposalDto {
            machine: machine_to_dto(&StateMachine::default()),
            notes,
            triggers: Vec::new(),
            refusal: Some(e.to_string()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A machine using **every** v2 shape, so the round-trip below is a claim
    /// about v2 and not about v1 wearing a new field list.
    fn v2_machine() -> StateMachine {
        let sub = StateMachine {
            states: vec![
                SmState::clip("stand", Uuid::from_u128(7).into_bytes()),
                SmState::clip("crouch", Uuid::from_u128(8).into_bytes()),
            ],
            transitions: vec![SmTransition::on(0, 1, 0.1, "crouching", CmpOp::Gt, 0.5)],
            entry: 0,
            ..Default::default()
        };
        let mut idle = SmState::clip("idle", NIL_CLIP);
        idle.on_enter = vec!["idle_begin".into()];
        idle.on_exit = vec!["idle_end".into()];
        StateMachine {
            states: vec![
                idle,
                SmState {
                    name: "loco".into(),
                    motion: Motion::Blend1D(BlendSpace1D::new(
                        "speed",
                        vec![BlendEntry1D {
                            pos: 1.0,
                            clip: Uuid::from_u128(5).into_bytes(),
                        }],
                    )),
                    looping: true,
                    speed: 1.5,
                    position: (10.0, 20.0),
                    on_enter: Vec::new(),
                    on_exit: Vec::new(),
                },
                SmState::sub_machine("ground", sub),
                // **A 2D blend space, which "every v2 shape" did not have.**
                // The fixture carried a 1D one and a sub-machine, so the two
                // fields only `SmMotionDto::Blend2d` has were never on the wire
                // this arm inspects — and that is precisely why their missing
                // `#[serde(rename)]` survived P11.2, P24 and P29.1's own
                // battery. A round-trip fixture that skips a variant is a
                // round-trip claim about the variants it kept.
                SmState {
                    name: "aim".into(),
                    motion: Motion::Blend2D(BlendSpace2D::new(
                        "aim_x",
                        "aim_y",
                        vec![BlendEntry2D {
                            pos: [1.0, -1.0],
                            clip: Uuid::from_u128(6).into_bytes(),
                        }],
                    )),
                    looping: true,
                    speed: 1.0,
                    position: (30.0, 40.0),
                    on_enter: Vec::new(),
                    on_exit: Vec::new(),
                },
            ],
            transitions: vec![
                SmTransition::on(0, 1, 0.2, "moving", CmpOp::Ge, 0.5)
                    .with_exit_time(0.8)
                    .with_priority(4)
                    .with_curve(BlendCurve::EaseInOut)
                    .with_profile(0)
                    .with_interrupt(SmInterrupt {
                        source: InterruptSource::SourceOrDestination,
                        blend: InterruptBlend::Snap,
                    }),
                // A tree the flat inspector cannot draw, plus an any-state source.
                SmTransition::any(2, 0.05).when(SmCond::Or(vec![
                    SmCond::Trigger("land".into()),
                    SmCond::Not(Box::new(SmCond::bool("airborne", true))),
                    SmCond::int("stance", CmpOp::Eq, 2),
                ])),
            ],
            entry: 0,
            params: vec![
                SmParam::float("moving"),
                SmParam::trigger("land"),
                SmParam::new("airborne", SmParamKind::Bool),
                SmParam::new("stance", SmParamKind::Int),
            ],
            profiles: vec![BlendProfile::new(
                "upper-body",
                vec![JointBlendWeight {
                    joint: 3,
                    weight: 0.25,
                }],
            )],
        }
    }

    /// **The DTO carries every v2 shape, losslessly.**
    ///
    /// This is the whole reason the Ring-2 mirror was rewritten in this batch
    /// rather than the next one: `sm_save` pushes the *document* back and writes
    /// whatever `dto_to_machine` produces. A DTO missing a field is not a
    /// cosmetic gap — it is the editor deleting a condition tree, a priority or a
    /// nested machine the moment somebody opens a file and saves it.
    #[test]
    fn machine_dto_round_trips_every_v2_shape() {
        let machine = v2_machine();
        let dto = machine_to_dto(&machine);
        let back = dto_to_machine(&dto);
        assert_eq!(back, machine);

        // …and through JSON, which is what actually crosses the IPC boundary.
        let json = serde_json::to_string(&dto).unwrap();
        let parsed: SmMachineDto = serde_json::from_str(&json).unwrap();
        assert_eq!(dto_to_machine(&parsed), machine);
        // **Every renamed key, by name.** Three samples is what this arm used to
        // check, and the P29.1 audit found a fourth defect it could not see:
        // `SmMotionDto::Blend2d`'s `param_x`/`param_y` crossed as snake_case
        // while `smTypes.ts` read `paramX`/`paramY`, for the same reason
        // `valueKind` did — `rename_all` on a **tagged enum** renames variants,
        // not their fields. The two tagged enums are therefore enumerated in
        // full here, and the plain structs' multi-word fields with them, because
        // a spelling this wire gets wrong is not a decode failure on the far
        // side: it is an `undefined` that renders.
        for key in [
            // plain structs (`rename_all` DOES reach their fields)
            "excludeSelf",
            "exitTime",
            "interruptSource",
            "interruptBlend",
            "onEnter",
            "onExit",
            // the tagged enums (`rename_all` does NOT reach their fields)
            "valueKind",
            "paramX",
            "paramY",
        ] {
            assert!(json.contains(key), "the wire is missing `{key}`: {json}");
        }
        // …and the snake_case spellings must not be there at all, which is the
        // half that actually falsifies: a `#[serde(rename)]` that was dropped
        // leaves the field present under the wrong name.
        for wrong in [
            "value_kind",
            "param_x",
            "param_y",
            "exclude_self",
            "exit_time",
        ] {
            assert!(
                !json.contains(wrong),
                "the wire carries `{wrong}` — a rename was dropped and the TS \
                 mirror reads `undefined`: {json}"
            );
        }
    }

    /// **The save door refuses a machine its own reader would refuse** (P29.1
    /// audit, finding A5 — the P23.6 ruling).
    ///
    /// `dto_to_machine` builds a `StateMachine` out of whatever JSON arrives:
    /// `to`, `entry` and `profile` are plain `usize`s off the wire with no bound
    /// on this side of the boundary. v1 survived that, because its `migrate`
    /// asked no structural question and evaluation clamped. v2's `migrate`
    /// refuses — so without this check a client bug writes a `.inf_sm` the
    /// editor can never open again.
    ///
    /// `sm_save` itself takes `State<'_, …>` and an async runtime and is not
    /// constructible in a unit test (the same reason every settle check in
    /// `commands::dcc` is a source gate), so what is driven here is the pair it
    /// delegates to — and the pair is the whole rule.
    #[test]
    fn the_save_door_refuses_what_the_load_door_would() {
        let mut dto = machine_to_dto(&v2_machine());
        // The shape a stale client sends after deleting a state without
        // reindexing: an edge pointing past the end.
        dto.transitions[0].to = 99;
        let machine = dto_to_machine(&dto);
        let err = machine
            .validate()
            .expect_err("a dangling transition must not be saveable");
        assert!(err.to_string().contains("does not exist"), "{err}");

        // …and the file it WOULD have written is one this engine cannot read,
        // which is the claim that makes the check worth its cost.
        let bytes = inf_asset::encode(&StateMachineAsset::new(machine, None)).unwrap();
        assert!(
            inf_asset::decode::<StateMachineAsset>(&bytes).is_err(),
            "the payload decoded, so the refusal above is guarding nothing"
        );

        // Control: the healthy document still saves.
        let healthy = dto_to_machine(&machine_to_dto(&v2_machine()));
        healthy
            .validate()
            .expect("a healthy machine must still save");
    }

    /// **The flat view is offered exactly when it is safe, and it wins when it
    /// is sent.**
    ///
    /// Two directions of the same rule:
    ///
    ///  * a flat AND of float compares gets `conditions: Some(..)`, so v1's
    ///    inspector keeps working unchanged;
    ///  * anything else gets `conditions: null`, so a UI that can only draw the
    ///    list has nothing to bind — and a save from that UI therefore restores
    ///    the tree rather than flattening it.
    ///
    /// The second arm is the one with teeth: it simulates the v1 inspector by
    /// round-tripping the DTO untouched and asserts the `Or` survived.
    #[test]
    fn the_flat_condition_view_never_destroys_a_tree_it_cannot_draw() {
        let dto = machine_to_dto(&v2_machine());
        let flat = &dto.transitions[0];
        let tree = &dto.transitions[1];
        assert!(
            flat.conditions.is_some(),
            "a flat float AND must still bind to the v1 inspector"
        );
        assert_eq!(flat.conditions.as_ref().unwrap()[0].var, "moving");
        assert!(
            tree.conditions.is_none(),
            "an Or/Not/typed tree must NOT be offered as a flat list"
        );

        // The v1 inspector's behaviour: send back exactly what it received.
        let back = dto_to_machine(&dto);
        assert!(
            matches!(back.transitions[1].condition, SmCond::Or(_)),
            "the tree was flattened by a save that never edited it: {:?}",
            back.transitions[1].condition
        );
        // …and the any-state source and its exclude_self survived too.
        assert_eq!(
            back.transitions[1].from,
            SmSource::Any { exclude_self: true }
        );

        // Editing the flat view still wins, which is what makes the inspector
        // functional rather than read-only.
        let mut edited = dto;
        edited.transitions[0].conditions.as_mut().unwrap()[0].value = 9.0;
        let back = dto_to_machine(&edited);
        let cmp = back.transitions[0]
            .condition
            .as_flat_and()
            .expect("still flat")[0]
            .clone();
        assert_eq!(cmp.value.as_f64(), 9.0);
    }

    /// Every v2 wire string round-trips, and an unrecognized one falls back to
    /// the v1-equivalent value rather than panicking — a Tauri command that
    /// panics takes its handler down.
    #[test]
    fn v2_wire_strings_round_trip_and_degrade_safely() {
        for c in [
            BlendCurve::Linear,
            BlendCurve::EaseIn,
            BlendCurve::EaseOut,
            BlendCurve::EaseInOut,
            BlendCurve::Step,
        ] {
            assert_eq!(str_to_curve(curve_to_str(c)), c);
        }
        for s in [
            InterruptSource::None,
            InterruptSource::Destination,
            InterruptSource::SourceOrDestination,
        ] {
            assert_eq!(str_to_source(source_to_str(s)), s);
        }
        for b in [InterruptBlend::Snap, InterruptBlend::Carry] {
            assert_eq!(str_to_blend(blend_to_str(b)), b);
        }
        for k in [
            SmParamKind::Bool,
            SmParamKind::Int,
            SmParamKind::Float,
            SmParamKind::Trigger,
        ] {
            assert_eq!(str_to_kind(kind_to_str(k)), k);
        }
        assert_eq!(str_to_curve("nonsense"), BlendCurve::Linear);
        assert_eq!(str_to_source("nonsense"), InterruptSource::Destination);
        assert_eq!(str_to_kind("nonsense"), SmParamKind::Float);
        // The typed operand keeps its kind across the numeric wire.
        assert_eq!(dto_to_value(1.0, "bool"), SmValue::Bool(true));
        assert_eq!(dto_to_value(0.4, "bool"), SmValue::Bool(false));
        assert_eq!(dto_to_value(2.9, "int"), SmValue::Int(2));
        assert_eq!(dto_to_value(2.5, "float"), SmValue::Float(2.5));
    }

    #[test]
    fn op_strings_round_trip() {
        for op in [
            CmpOp::Gt,
            CmpOp::Lt,
            CmpOp::Ge,
            CmpOp::Le,
            CmpOp::Eq,
            CmpOp::Ne,
        ] {
            assert_eq!(str_to_op(op_to_str(op)), op);
        }
    }

    #[test]
    fn default_doc_has_one_state() {
        let d = default_doc("sm1".into(), "Test".into());
        assert_eq!(d.machine.states.len(), 1);
        assert_eq!(d.machine.entry, 0);
        assert!(matches!(
            d.machine.states[0].motion,
            SmMotionDto::Clip { .. }
        ));
    }

    /// **Round 3: the rig a saved `.inf_sm` was authored against.**
    ///
    /// A state machine names no skeleton of its own, so the binding can only
    /// come from the clips it plays. `sm_save` itself takes `State<'_, …>` and
    /// an async runtime and is not constructible in a unit test — the same
    /// reason every settle check in `commands::dcc` is a source gate — so what
    /// is driven here is the resolution it delegates to, against a **real
    /// project on disk**, and the world it produces: the advisory fires.
    #[test]
    fn a_saved_machine_binds_to_the_rig_its_clips_name() {
        use inf_anim::{AnimClip, AnimClipAsset, Joint, JointTransform, SkeletonAsset};
        use inf_editor_core::assets::AssetProject;

        let joint = |name: &str, parent: Option<u16>| Joint {
            name: name.into(),
            parent,
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::IDENTITY,
        };
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let content = proj.content_dir("Anim").unwrap();

        let rig =
            inf_anim::Skeleton::new(vec![joint("root", None), joint("spine", Some(0))]).unwrap();
        let skel = proj
            .write_asset(
                &content,
                "Rig",
                &SkeletonAsset::new(rig),
                None,
                vec![],
                None,
            )
            .unwrap();
        let table = inf_editor_core::assets::skeleton_binding::import_table(&proj, Some(skel));
        let clip = proj
            .write_asset(
                &content,
                "Walk",
                &AnimClipAsset::new(
                    AnimClip::new("walk", Vec::new()),
                    Some(*skel.uuid().as_bytes()),
                ),
                None,
                vec![skel],
                table,
            )
            .unwrap();

        // The machine the editor would save: one assigned clip, one unassigned.
        let machine = StateMachine {
            states: vec![
                SmState::clip("walk", *clip.uuid().as_bytes()),
                SmState::clip("idle", NIL_CLIP),
            ],
            transitions: vec![],
            entry: 0,
            ..Default::default()
        };
        let (skeleton, deps) = machine_binding(&proj, &machine);
        assert_eq!(skeleton, Some(skel), "the rig came back from the clip");
        assert_eq!(
            deps,
            vec![clip, skel],
            "the edges name the clip and the rig; the nil clip is not an edge"
        );

        // The write door, with what `sm_save` resolves.
        let import = inf_editor_core::assets::skeleton_binding::import_table(&proj, skeleton);
        assert!(
            import.is_some(),
            "nothing was recorded — the rest is vacuous"
        );
        let payload = StateMachineAsset::new(machine, skeleton.map(|s| *s.uuid().as_bytes()));
        let path = proj
            .content_dir("StateMachines")
            .unwrap()
            .join("Loco.inf_sm");
        proj.write_asset_at(&path, &payload, deps, import).unwrap();

        assert!(
            inf_editor_core::assets::skeleton_binding::advisories(&proj).is_empty(),
            "a machine saved against the current rig raised an advisory"
        );

        // Re-import the rig with a joint INSERTED — the edit that renumbers
        // every track after it, in range, with nothing to refuse it.
        let three = inf_anim::Skeleton::new(vec![
            joint("root", None),
            joint("pelvis", Some(0)),
            joint("spine", Some(1)),
        ])
        .unwrap();
        proj.rewrite_payload(skel, &SkeletonAsset::new(three), vec![])
            .unwrap();

        let found = inf_editor_core::assets::skeleton_binding::advisories(&proj);
        assert_eq!(
            found.len(),
            2,
            "the clip AND the machine must each say so, got {found:?}"
        );
        assert!(
            found.iter().any(|a| a.contains("Loco")),
            "the machine's own advisory is the one this test exists for: {found:?}"
        );
    }

    /// A machine whose clips are all unassigned records nothing, and says so by
    /// being silent rather than by writing a key that means "no rig".
    #[test]
    fn a_machine_with_no_assigned_clips_records_no_binding() {
        use inf_editor_core::assets::AssetProject;
        let dir = tempfile::tempdir().unwrap();
        let proj = AssetProject::open(dir.path()).unwrap();
        let machine = StateMachine {
            states: vec![SmState::clip("idle", NIL_CLIP)],
            transitions: vec![],
            entry: 0,
            ..Default::default()
        };
        let (skeleton, deps) = machine_binding(&proj, &machine);
        assert_eq!(skeleton, None);
        assert!(deps.is_empty(), "the nil clip is not a dependency");
    }

    /// **The proposal the panel receives is one the save door accepts** (P29.5,
    /// pillar S3).
    ///
    /// `sm_propose` projects a Ring-0 [`inf_anim::Proposal`] through
    /// `machine_to_dto`, and the panel adopts it as its document — so the round
    /// trip back through `dto_to_machine` has to produce a machine
    /// `StateMachine::validate` accepts, or the author's first `Save` after a
    /// proposal is a refusal they did not cause.
    ///
    /// The anti-vacuity half matters here: a proposal of an EMPTY machine
    /// validates perfectly (`validate` returns early on no states), so the arm
    /// asserts the shapes as well — a blend space, an any-state edge, a declared
    /// trigger and a non-zero exit time all survive the projection.
    #[test]
    fn a_proposed_machine_survives_the_wire_and_validates() {
        let facts = |name: &str, speed: f32, plants: usize| inf_anim::ClipFacts {
            name: name.into(),
            clip: Uuid::from_u128(name.len() as u128 + 1).into_bytes(),
            duration_s: 1.0,
            speed_mps: speed,
            gait: inf_anim::gait_of(speed, [1.65, 3.75, 6.5]),
            plants,
        };
        let proposal = inf_anim::propose_machine(
            &[
                facts("Idle", 0.0, 0),
                facts("Walk", 1.6, 2),
                facts("Jog", 3.2, 2),
                facts("Runs", 4.0, 2),
                facts("Jump", 2.0, 1),
            ],
            &inf_anim::ProposalOptions::default(),
        )
        .expect("a proposal");
        proposal
            .machine
            .validate()
            .expect("Ring 0 validates its own");

        let dto = machine_to_dto(&proposal.machine);
        let json = serde_json::to_string(&dto).expect("the DTO serializes");
        let back: SmMachineDto = serde_json::from_str(&json).expect("…and comes back");
        let machine = dto_to_machine(&back);
        machine
            .validate()
            .expect("a proposal must validate after the round trip");

        // The shapes really made it across.
        assert!(
            machine
                .states
                .iter()
                .any(|s| matches!(s.motion, Motion::Blend1D(_))),
            "the blend space did not survive"
        );
        assert!(
            machine
                .transitions
                .iter()
                .any(|t| matches!(t.from, SmSource::Any { .. })),
            "the any-state edge did not survive"
        );
        assert!(
            machine
                .params
                .iter()
                .any(|p| p.kind == SmParamKind::Trigger && p.name == "play_jump"),
            "the declared trigger did not survive: {:?}",
            machine.params
        );
        assert!(
            machine.transitions.iter().any(|t| t.exit_time == Some(0.9)),
            "the exit time did not survive"
        );
        // …and the machine is not the empty one that would validate anyway.
        assert!(machine.states.len() >= 4, "{:?}", machine.states.len());
    }

    #[test]
    fn close_frees_the_doc() {
        let state = SmEditorState::default();
        {
            let mut s = state.inner.lock().unwrap();
            s.docs
                .insert("sm1".into(), default_doc("sm1".into(), "Test".into()));
        }
        state.close("sm1").unwrap();
        assert!(state.inner.lock().unwrap().docs.is_empty(), "doc freed");
    }
}
