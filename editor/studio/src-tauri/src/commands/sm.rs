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
        param_x: String,
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
        /// The typed operand: exactly one of these is set, chosen by `valueKind`.
        value: f64,
        /// `"bool"` / `"int"` / `"float"`.
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

fn dto_to_value(value: f64, kind: &str) -> SmValue {
    match kind {
        "bool" => SmValue::Bool(value > 0.5),
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
        // The wire really is camelCase, so the hand-written TS mirror binds.
        assert!(json.contains("interruptSource"), "{json}");
        assert!(json.contains("valueKind"), "{json}");
        assert!(json.contains("onEnter"), "{json}");
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
