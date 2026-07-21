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
use inf_editor_core::samples::bound_actors;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};
use tauri::{AppHandle, Emitter, State};

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
    // Character applies its own gravity in the blueprint → world gravity is zero.
    let session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
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
