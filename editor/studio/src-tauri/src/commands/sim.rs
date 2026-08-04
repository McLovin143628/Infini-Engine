//! Simulate commands (P8.4): drive the in-editor [`SimSession`] over the shared
//! [`SceneDoc`] so a Blueprint + 2D-physics scene plays in the viewport.
//!
//! The session lives here behind a `Mutex`; it borrows the **same** `SceneDoc`
//! the viewport renders (`SceneState::doc`), so ticking mutates the live world
//! and the native viewport re-syncs on the bumped version. Enter snapshots the
//! world; Stop restores it exactly.
//!
//! Input path (P8.4.3): `sim_tick` receives the currently-held keys/actions from
//! the frontend each frame (a rAF loop) — the "frontend-focused" input route that
//! always works. When the **native** viewport holds OS focus it forwards
//! unconsumed keys over `viewport://key` (the Phase-2 focus-handoff channel); the
//! frontend merges those into the held set it passes here. Driving the tick from
//! a real frame timer is the human-verified remainder (CI scripts the session
//! directly — see `inf-editor-core`'s `simulate_platformer` test).

use std::sync::Mutex;
use std::time::Instant;

use glam::DVec2;
use inf_blueprint::{InterpDebug, LocalId};
use inf_editor_core::samples::bound_actors;
use inf_editor_core::simulate::{SimDebugHit, SimInput, SimSession, SIM_HZ};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use super::assets::AssetState;
use super::scene::SceneState;

/// Live Simulate state: the session (absent when stopped) + the last tick's
/// wall-clock instant (for the frame delta fed to the fixed-step accumulator).
#[derive(Default)]
pub struct SimState {
    inner: Mutex<SimInner>,
}

impl SimState {
    /// Whether a Simulate session is live — other command modules (e.g. the
    /// sequencer scrub, P11.4) consult this to avoid two writers fighting over the
    /// shared `SceneDoc`.
    pub fn is_running(&self) -> bool {
        self.inner
            .lock()
            .map(|inner| inner.session.is_some())
            .unwrap_or(false)
    }

    /// Live-apply a mixer to a running Simulate session (E-P9). A no-op when
    /// stopped. Mirrors the mixer load in [`sim_start`] so the Audio Mixer panel's
    /// save is heard immediately in an active Simulate. Poisoned lock is ignored
    /// (the next `sim_start` reloads the mixer from disk regardless).
    pub fn apply_mixer(&self, mixer: inf_audio::MixerConfig) {
        if let Ok(mut inner) = self.inner.lock() {
            if let Some(session) = inner.session.as_mut() {
                session.set_audio_mixer(mixer);
            }
        }
    }
}

#[derive(Default)]
struct SimInner {
    session: Option<SimSession>,
    last: Option<Instant>,
}

/// Enter Simulate over the current scene: snapshot the world, bind actors —
/// **preferring the scene's persisted `ActorClass` links** (P9.5), resolving each
/// via the project asset DB, and falling back to the `CharacterController2D`
/// heuristic for scenes with no bindings — and fire BeginPlay.
#[tauri::command]
pub async fn sim_start(
    app: AppHandle,
    scene: State<'_, SceneState>,
    sim: State<'_, SimState>,
    assets: State<'_, AssetState>,
) -> Result<(), String> {
    let mut doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
    let actors = bound_actors(&doc, |guid| {
        assets.load_blueprint_class(inf_asset::AssetId(guid))
    });
    // Resolve the scene's referenced anim assets (state machines + root-motion
    // clips) so an `AnimStateMachine` / `RootMotion` entity steps in Simulate
    // exactly as it will in the shipped player (P11.4).
    let (machines, root_clips) = inf_editor_core::simulate::resolve_anim_assets(&doc, |guid| {
        assets.load_anim_bytes(inf_asset::AssetId(guid))
    });
    // Resolve the scene's referenced `.inf_audio` clips (P12.3) so an `AudioSource`
    // plays the same clip in Simulate as in the shipped player.
    let audio_clips = inf_editor_core::simulate::resolve_audio_assets(&doc, |guid| {
        assets.load_audio_bytes(inf_asset::AssetId(guid))
    });
    // Resolve the scene's referenced `.inf_voxel` volumes (P21.2) so a Blueprint
    // calling `terrain.height_at` over a carved hole reads the cave floor beneath
    // it — the same map the shipped player seeds through `attach_voxel_volumes`.
    // This is the SIM's set, camera-free: the viewport host pages its own against
    // the editor camera, and a fixed step must never be able to see that.
    let mut voxel_volumes = inf_editor_core::simulate::resolve_voxel_volumes(&doc, |guid| {
        assets.load_voxel_bytes(inf_asset::AssetId(guid))
    });
    // …and then the carves that are NOT on disk yet (P21.2 audit). Resolving from
    // the asset alone hands Simulate the last *saved* cave, so pressing Play after
    // digging a tunnel dropped the player onto solid rock. A streamed terrain does
    // not behave that way — `SimSession::enter` snapshots with
    // `ScenePersist::Memory` precisely so unsaved sculpts survive into a session
    // (P16.4b) — and a carve is the same act on the other surface. It cannot ride
    // the snapshot the way a sculpt does (scene schema v19 is frozen, so chunks
    // are not in the document), so it is read out of the shared store the viewport
    // carves into, on the same lock order every other reader uses: **document
    // first, volumes second** (the doc guard is still held here, taken above).
    //
    // Only the store's DIRTY chunks are folded in, which is what keeps the
    // determinism seam on `set_voxel_volumes` intact: dirty is a function of what
    // was dug, never of where the editor camera has paged. Headless/CI has no
    // viewport and therefore no store, and no carve edits either — nothing to
    // fold, and the resolved map stands as it always did.
    if let Some(volumes) = app
        .try_state::<crate::commands::ViewportState>()
        .and_then(|v| v.voxel_volumes())
    {
        match volumes.lock() {
            Ok(store) => {
                let n = inf_editor_core::simulate::overlay_unsaved_carves(
                    &mut voxel_volumes,
                    |entity| store.slot(entity).map(|s| &s.data),
                );
                if n > 0 {
                    tracing::info!("simulate: {n} unsaved voxel chunk(s) carried into Simulate");
                }
            }
            // A poisoned store is a thread that already panicked mid-carve; its
            // chunks are not trustworthy input for a session. Play still starts,
            // on the saved volumes, rather than refusing outright.
            Err(_) => tracing::warn!(
                "simulate: voxel store lock poisoned — Simulate runs on the SAVED \
                 volumes, so unsaved carves will not be there"
            ),
        }
    }
    // Character applies its own gravity in the blueprint → world gravity is zero.
    let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
    session.set_state_machines(machines);
    for (clip_guid, skeleton, clip) in root_clips {
        session.register_root_motion_clip(clip_guid, skeleton, clip);
    }
    session.set_audio_clips(audio_clips);
    session.set_voxel_volumes(voxel_volumes);
    // Load the project mixer (`<project>/.infinity/mixer.toml`) if present; else the
    // default. The mixer lives at the project root (the parent of Content).
    if let Some(content) = assets.content_root() {
        let root = content.parent().map(|p| p.to_path_buf()).unwrap_or(content);
        if let Ok(mixer) = inf_audio::MixerConfig::load_or_default(&root) {
            session.set_audio_mixer(mixer);
        }
    }
    doc.bump_version_for_runtime();
    drop(doc);

    let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
    inner.session = Some(session);
    inner.last = Some(Instant::now());
    let _ = app.emit("sim://state", true);
    Ok(())
}

/// Advance Simulate by one frame with the given held keys/actions (e.g.
/// `["left","jump"]`). No-op when not running.
#[tauri::command]
pub async fn sim_tick(
    app: AppHandle,
    scene: State<'_, SceneState>,
    sim: State<'_, SimState>,
    keys: Vec<String>,
) -> Result<bool, String> {
    let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
    if inner.session.is_none() {
        return Ok(false);
    }
    let now = Instant::now();
    let dt = inner
        .last
        .map(|t| now.duration_since(t).as_secs_f64())
        .unwrap_or(1.0 / SIM_HZ)
        // Guard a huge first/hitched frame (the stepper also clamps).
        .clamp(0.0, 0.25);
    inner.last = Some(now);

    let mut doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
    let session = inner.session.as_mut().expect("session present");
    session.tick(&mut doc, dt, SimInput::with_down(keys));
    doc.bump_version_for_runtime();
    drop(doc);
    overlay_sim_carves(&app, session);
    emit_debug(&app, session);
    Ok(true)
}

/// **Mirror the Simulate session's runtime carves into the viewport's voxel
/// store** (P21.4) — the editor twin of `PlayerRenderHost::sync_voxels`' overlay.
///
/// A `voxel.*` node writes the *session's* volume map, which is deliberately not
/// the store the viewport draws from (the store is camera-paged; the sim's map
/// must never be). Without this the editor shows exactly what the shipped player
/// showed before the fold existed: a Blueprint digs, `terrain.height_at` moves,
/// the colliders rebuild — and the screen keeps drawing the rock.
///
/// `sim → render` only, and stamp-gated inside Ring 0
/// ([`inf_voxel::VoxelVolumes::overlay_sim`]): a frame in which nothing was dug
/// copies nothing. A poisoned store is skipped rather than fatal — Simulate keeps
/// running and the picture goes stale, which is strictly better than killing a
/// session over a render detail.
///
/// The terrain half needs no twin here: `SimSession` steps the **document's** own
/// world, so a runtime hole lands in the document and the viewport's existing
/// `overlay_document_edits` already mirrors it.
fn overlay_sim_carves(app: &AppHandle, session: &inf_editor_core::simulate::SimSession) {
    let volumes = session.voxel_volumes();
    if volumes.is_empty() {
        return;
    }
    let Some(store) = app
        .try_state::<crate::commands::ViewportState>()
        .and_then(|v| v.voxel_volumes())
    else {
        return;
    };
    let Ok(mut store) = store.lock() else {
        return;
    };
    for (entity, data) in volumes {
        store.overlay_sim(*entity, data);
    }
}

/// Advance Simulate by **exactly one fixed step** with the held keys (B-P4 tier
/// A′) — bypasses the wall-clock accumulator, so Step is a guaranteed single
/// step (fixes the documented `sim_step_fixed` gap). No-op when not running.
#[tauri::command]
pub async fn sim_step_fixed(
    app: AppHandle,
    scene: State<'_, SceneState>,
    sim: State<'_, SimState>,
    keys: Vec<String>,
) -> Result<bool, String> {
    let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
    if inner.session.is_none() {
        return Ok(false);
    }
    // A fixed step does not advance wall-clock time; reset `last` so the next
    // free-running tick's delta starts fresh from now.
    inner.last = Some(Instant::now());
    let mut doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
    let session = inner.session.as_mut().expect("session present");
    session.step_once(&mut doc, SimInput::with_down(keys));
    doc.bump_version_for_runtime();
    drop(doc);
    overlay_sim_carves(&app, session);
    emit_debug(&app, session);
    Ok(true)
}

/// Install the debugger config for an actor class (B-P4 tier A′): `breakpoints`
/// (IR `LocalId`s for hand-built classes) + wire `capture`. When a subsequent
/// step hits a breakpoint or captures wires, the backend emits `sim://debug`.
/// No-op (returns `false`) when no session is running.
#[tauri::command]
pub async fn sim_set_debug(
    sim: State<'_, SimState>,
    class_id: String,
    breakpoints: Vec<u32>,
    capture: bool,
) -> Result<bool, String> {
    let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
    let Some(session) = inner.session.as_mut() else {
        return Ok(false);
    };
    let debug = InterpDebug {
        breakpoints: breakpoints.into_iter().map(LocalId).collect(),
        capture_wires: capture,
    };
    session.set_debug(class_id, debug);
    Ok(true)
}

/// Exit Simulate: restore the pre-play world exactly.
#[tauri::command]
pub async fn sim_stop(
    app: AppHandle,
    scene: State<'_, SceneState>,
    sim: State<'_, SimState>,
) -> Result<(), String> {
    let session = {
        let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
        inner.last = None;
        inner.session.take()
    };
    if let Some(session) = session {
        let mut doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
        session.exit(&mut doc);
        doc.bump_version_for_runtime();
    }
    let _ = app.emit("sim://state", false);
    Ok(())
}

/// Whether a Simulate session is currently running.
#[tauri::command]
pub async fn sim_is_running(sim: State<'_, SimState>) -> Result<bool, String> {
    Ok(sim
        .inner
        .lock()
        .map_err(|_| "sim lock poisoned")?
        .session
        .is_some())
}

/// One handler's Blueprint debug observation emitted on `sim://debug` (B-P4 tier
/// A′). The wire-inspector JSON shape (mirrors the frontend `SimDebugEvent`).
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SimDebugEventDto {
    class_id: String,
    event: String,
    fn_name: String,
    hits: Vec<u32>,
    wires: Vec<(u32, String)>,
}

impl From<SimDebugHit> for SimDebugEventDto {
    fn from(h: SimDebugHit) -> Self {
        Self {
            class_id: h.class_id,
            event: h.event,
            fn_name: h.fn_name,
            hits: h.hits,
            wires: h.wires,
        }
    }
}

/// Drain the session's debug events and, when any were produced, broadcast them
/// on `sim://debug` (the frontend pauses on hits + shows wire values). No-op when
/// no class is being debugged.
fn emit_debug(app: &AppHandle, session: &mut SimSession) {
    let events = session.take_debug_events();
    if events.is_empty() {
        return;
    }
    let dto: Vec<SimDebugEventDto> = events.into_iter().map(Into::into).collect();
    let _ = app.emit("sim://debug", dto);
}
