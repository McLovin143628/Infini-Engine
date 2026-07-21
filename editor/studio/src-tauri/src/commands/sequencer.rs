//! Sequencer commands (P11.4): the seed timeline over the shared [`SceneDoc`].
//!
//! Sequences are project-local files (`<root>/.infinity/sequences/<slug>.toml`,
//! [`inf_editor_core::sequencer`]) — NOT a new asset kind (the `.inf_seq` asset is
//! the full-sequencer follow-up). The panel edits them through these commands.
//!
//! **Scrubbing** is a **non-undoable live preview** driven over the same
//! `SceneState::doc` the viewport renders: `seq_scrub` snapshots the affected
//! scene scalars once at gesture start, writes the sampled values via the
//! non-dirtying preview path (so the viewport re-syncs on the bumped version but
//! the document is never marked unsaved), and `seq_scrub_stop` restores the
//! snapshot so previewing leaves the scene byte-identical. Scrubbing is **refused
//! while Simulate is running** (both would write the shared world — the frontend
//! also disables the control).
//!
//! **Key edits** (`seq_key_set` / `seq_capture_key`) mutate the *sequence file*,
//! not the scene, so they are non-undoable in v1 (documented); each emits
//! `seq://changed` (payload = sequence name) so open panels re-sync.

use std::sync::Mutex;

use inf_editor_core::ipc::{SeqInterpDto, SeqKeyDto, SeqTrackDto, SequenceDto};
use inf_editor_core::sequencer::{self, Interp, Key, PropertyTrack, ScrubSnapshot, Sequence};
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;

use super::assets::AssetState;
use super::scene::SceneState;
use super::sim::SimState;

/// Ring-2 sequencer state: the in-flight scrub session (the restore-on-stop
/// snapshot + which sequence it belongs to), or `None` when not scrubbing.
#[derive(Default)]
pub struct SequencerState {
    session: Mutex<Option<ScrubSession>>,
}

struct ScrubSession {
    name: String,
    snapshot: ScrubSnapshot,
}

// ── DTO ↔ model conversion ───────────────────────────────────────────────────

fn interp_to_dto(i: Interp) -> SeqInterpDto {
    match i {
        Interp::Step => SeqInterpDto::Step,
        Interp::Linear => SeqInterpDto::Linear,
    }
}

fn interp_from_dto(i: SeqInterpDto) -> Interp {
    match i {
        SeqInterpDto::Step => Interp::Step,
        SeqInterpDto::Linear => Interp::Linear,
    }
}

fn to_dto(seq: &Sequence) -> SequenceDto {
    SequenceDto {
        name: seq.name.clone(),
        duration: seq.duration,
        fps_hint: seq.fps_hint,
        tracks: seq
            .tracks
            .iter()
            .map(|tr| SeqTrackDto {
                target: tr.target.to_string(),
                path: tr.path.clone(),
                keys: tr
                    .keys
                    .iter()
                    .map(|k| SeqKeyDto {
                        t: k.t,
                        value: k.value,
                        interp: interp_to_dto(k.interp),
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Convert an incoming DTO to the model, dropping tracks whose target guid is
/// unparseable (the caller supplies string guids).
fn from_dto(dto: SequenceDto) -> Sequence {
    let tracks = dto
        .tracks
        .into_iter()
        .filter_map(|tr| {
            let target = Uuid::parse_str(&tr.target).ok()?;
            Some(PropertyTrack {
                target,
                path: tr.path,
                keys: tr
                    .keys
                    .into_iter()
                    .map(|k| Key {
                        t: k.t,
                        value: k.value,
                        interp: interp_from_dto(k.interp),
                    })
                    .collect(),
            })
        })
        .collect();
    let mut seq = Sequence {
        name: dto.name,
        duration: dto.duration,
        fps_hint: dto.fps_hint.max(1),
        tracks,
    };
    seq.normalize();
    seq
}

fn content_root(assets: &AssetState) -> Result<std::path::PathBuf, String> {
    assets
        .content_root()
        .ok_or_else(|| "assets not initialized".to_string())
}

// ── reads ────────────────────────────────────────────────────────────────────

/// Every sequence in the current project, by display name (sorted).
#[tauri::command]
pub async fn seq_list(assets: State<'_, AssetState>) -> Result<Vec<String>, String> {
    let root = content_root(&assets)?;
    Ok(sequencer::list_sequences(&root))
}

/// Load one sequence by name (`None` when it does not exist).
#[tauri::command]
pub async fn seq_get(
    assets: State<'_, AssetState>,
    name: String,
) -> Result<Option<SequenceDto>, String> {
    let root = content_root(&assets)?;
    Ok(Sequence::load(&root, &name).as_ref().map(to_dto))
}

// ── writes (sequence file) ───────────────────────────────────────────────────

/// Persist a sequence (normalized + deterministic TOML). Emits `seq://changed`.
#[tauri::command]
pub async fn seq_save(
    app: AppHandle,
    assets: State<'_, AssetState>,
    sequence: SequenceDto,
) -> Result<SequenceDto, String> {
    let root = content_root(&assets)?;
    let seq = from_dto(sequence);
    seq.save(&root)?;
    let _ = app.emit("seq://changed", &seq.name);
    Ok(to_dto(&seq))
}

/// Delete a sequence file. Emits `seq://changed`.
#[tauri::command]
pub async fn seq_delete(
    app: AppHandle,
    assets: State<'_, AssetState>,
    name: String,
) -> Result<(), String> {
    let root = content_root(&assets)?;
    sequencer::delete_sequence(&root, &name)?;
    let _ = app.emit("seq://changed", &name);
    Ok(())
}

/// Set (or replace) one key on the `(target, path)` track, creating the track if
/// missing. Loads the on-disk sequence (or starts a fresh one), writes the key,
/// saves, and returns the updated sequence. The key edit is non-undoable in v1
/// (it mutates the sequence file, not the scene). Emits `seq://changed`.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn seq_key_set(
    app: AppHandle,
    assets: State<'_, AssetState>,
    name: String,
    target: String,
    path: String,
    t: f64,
    value: f64,
    interp: SeqInterpDto,
) -> Result<SequenceDto, String> {
    let root = content_root(&assets)?;
    let target = Uuid::parse_str(&target).map_err(|e| e.to_string())?;
    let mut seq = Sequence::load(&root, &name).unwrap_or_else(|| Sequence::new(&name));
    let snapped = seq.snap_time(t);
    seq.set_key(target, &path, snapped, value, interp_from_dto(interp));
    seq.save(&root)?;
    let _ = app.emit("seq://changed", &seq.name);
    Ok(to_dto(&seq))
}

/// Capture the CURRENT scene value of `(target, path)` into a key at (snapped)
/// time `t` — the "Add Key from the live pose" workflow primitive. Reads the
/// scalar out of the shared document, then behaves like [`seq_key_set`]. Errors
/// if the property path does not resolve on the target. Emits `seq://changed`.
#[tauri::command]
pub async fn seq_capture_key(
    app: AppHandle,
    scene: State<'_, SceneState>,
    assets: State<'_, AssetState>,
    name: String,
    target: String,
    path: String,
    t: f64,
) -> Result<SequenceDto, String> {
    let root = content_root(&assets)?;
    let target_id = Uuid::parse_str(&target).map_err(|e| e.to_string())?;
    let value = {
        let doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
        sequencer::read_scalar(&doc, target_id, &path)
            .ok_or_else(|| format!("property '{path}' not found on target"))?
    };
    let mut seq = Sequence::load(&root, &name).unwrap_or_else(|| Sequence::new(&name));
    let snapped = seq.snap_time(t);
    // A newly-captured key defaults to Linear (the common cutscene case).
    seq.set_key(target_id, &path, snapped, value, Interp::Linear);
    seq.save(&root)?;
    let _ = app.emit("seq://changed", &seq.name);
    Ok(to_dto(&seq))
}

// ── scrub (live preview) ─────────────────────────────────────────────────────

/// Scrub `name` to time `t`: preview the sampled values in the viewport without
/// dirtying the scene. The first call of a gesture (or a switch to a different
/// sequence) snapshots the affected scalars for restore-on-stop. Refused while
/// Simulate is running.
#[tauri::command]
pub async fn seq_scrub(
    scene: State<'_, SceneState>,
    sim: State<'_, SimState>,
    assets: State<'_, AssetState>,
    seq: State<'_, SequencerState>,
    name: String,
    t: f64,
) -> Result<(), String> {
    if sim.is_running() {
        return Err("cannot scrub while Simulate is running".to_string());
    }
    let root = content_root(&assets)?;
    let model = Sequence::load(&root, &name).ok_or_else(|| "sequence not found".to_string())?;

    let mut doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
    let mut session = seq.session.lock().map_err(|_| "sequencer lock poisoned")?;
    match session.as_ref() {
        // Same sequence still scrubbing — keep the existing snapshot.
        Some(s) if s.name == name => {}
        // Switching sequences: restore the previous one first, then re-snapshot.
        Some(s) => {
            sequencer::restore_snapshot(&mut doc, &s.snapshot);
            let snapshot = sequencer::capture_snapshot(&doc, &model);
            *session = Some(ScrubSession {
                name: name.clone(),
                snapshot,
            });
        }
        // Fresh gesture — snapshot the true scene.
        None => {
            let snapshot = sequencer::capture_snapshot(&doc, &model);
            *session = Some(ScrubSession {
                name: name.clone(),
                snapshot,
            });
        }
    }
    sequencer::apply_sequence_at(&mut doc, &model, t);
    Ok(())
}

/// Stop scrubbing: restore the captured pre-scrub scalars and end the session.
/// No-op when not scrubbing.
#[tauri::command]
pub async fn seq_scrub_stop(
    scene: State<'_, SceneState>,
    seq: State<'_, SequencerState>,
) -> Result<(), String> {
    let taken = seq
        .session
        .lock()
        .map_err(|_| "sequencer lock poisoned")?
        .take();
    if let Some(s) = taken {
        let mut doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
        sequencer::restore_snapshot(&mut doc, &s.snapshot);
    }
    Ok(())
}
