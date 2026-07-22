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
    BlendEntry1D, BlendEntry2D, BlendSpace1D, BlendSpace2D, CmpOp, Motion, SmCondition, SmState,
    SmTransition, StateMachine, StateMachineAsset,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmTransitionDto {
    pub from: usize,
    pub to: usize,
    pub duration: f64,
    pub conditions: Vec<SmConditionDto>,
    pub exit_time: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmConditionDto {
    pub var: String,
    /// One of `">"`, `"<"`, `">="`, `"<="`, `"=="`, `"!="`.
    pub op: String,
    pub value: f64,
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
            })
            .collect(),
        transitions: m
            .transitions
            .iter()
            .map(|t| SmTransitionDto {
                from: t.from,
                to: t.to,
                duration: t.duration,
                conditions: t
                    .conditions
                    .iter()
                    .map(|c| SmConditionDto {
                        var: c.var.clone(),
                        op: op_to_str(c.op).to_string(),
                        value: c.value,
                    })
                    .collect(),
                exit_time: t.exit_time,
            })
            .collect(),
        entry: m.entry,
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
            })
            .collect(),
        transitions: m
            .transitions
            .iter()
            .map(|t| SmTransition {
                from: t.from,
                to: t.to,
                duration: t.duration,
                conditions: t
                    .conditions
                    .iter()
                    .map(|c| SmCondition {
                        var: c.var.clone(),
                        op: str_to_op(&c.op),
                        value: c.value,
                    })
                    .collect(),
                exit_time: t.exit_time,
            })
            .collect(),
        entry: m.entry,
    }
}

/// A fresh machine with a single idle state (unassigned clip) at the origin.
fn default_doc(id: String, name: String) -> SmDoc {
    let machine = StateMachine {
        states: vec![SmState::clip("Idle", NIL_CLIP)],
        transitions: vec![],
        entry: 0,
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
#[tauri::command]
pub async fn sm_save(
    id: String,
    doc: SmDoc,
    name: String,
    state: State<'_, SmEditorState>,
    assets: State<'_, AssetState>,
) -> Result<String, String> {
    // Update in-memory + encode the payload.
    let (bytes, file_name) = {
        let mut s = state.inner.lock().map_err(|e| e.to_string())?;
        let machine = dto_to_machine(&doc.machine);
        let payload = StateMachineAsset::new(machine, None);
        let bytes = inf_asset::encode(&payload).map_err(|e| e.to_string())?;
        s.docs.insert(id.clone(), doc.clone());
        let base = if name.is_empty() { &doc.name } else { &name };
        let slug: String = base
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        (bytes, format!("{slug}.inf_sm"))
    };

    let root = assets
        .content_root()
        .ok_or("open a project before saving a state machine")?;
    // Filesystem write is blocking — keep it off the async workers.
    tauri::async_runtime::spawn_blocking(move || {
        let dir = root.join("StateMachines");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(&file_name), &bytes).map_err(|e| e.to_string())?;
        Ok::<_, String>(file_name)
    })
    .await
    .map_err(|e| format!("sm_save task failed to run: {e}"))?
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

    #[test]
    fn machine_dto_round_trips() {
        let machine = StateMachine {
            states: vec![
                SmState::clip("idle", NIL_CLIP),
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
                },
            ],
            transitions: vec![SmTransition {
                from: 0,
                to: 1,
                duration: 0.2,
                conditions: vec![SmCondition {
                    var: "moving".into(),
                    op: CmpOp::Ge,
                    value: 0.5,
                }],
                exit_time: Some(0.8),
            }],
            entry: 0,
        };
        let dto = machine_to_dto(&machine);
        let back = dto_to_machine(&dto);
        assert_eq!(back, machine);
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
