//! Play-In-Editor (PIE) commands (P9.4): spawn `inf-player` as a crash-isolated
//! subprocess, hand it the **live** scene (unsaved edits included) as v3
//! `.inf_lvl` bytes + bound blueprint classes, and drive Play/Pause/Step/Stop/
//! Eject over the Spike D local channel. The player builds its world exactly like
//! the shipping pack path, so previewing never diverges from shipping.
//!
//! Two modes: `"embedded"` reparents the player's window into the viewport slot
//! (Windows, via the proven `SetParent` machinery — [`inf_viewport`]'s
//! `embed_foreign`), and `"window"` is the roadmap-sanctioned "Play in New
//! Window" fallback (always works, every OS). A deliberate PIE panic kills only
//! the player; a monitor thread observes the exit, surfaces the captured panic
//! text, and resets the toolbar — unsaved editor state is never at risk.
//!
//! The process-management + payload logic lives headlessly-testable in
//! `inf_editor_core::pie`; this Ring-2 layer stays thin (spawn + monitor +
//! window embedding).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use inf_editor_core::pie::{build_scene_payload, find_player_bin, PieSession, SessionHealth};
use inf_runtime::pie::PlayerToEditor;
use tauri::{AppHandle, Emitter, Manager, State};

use super::assets::AssetState;
use super::scene::SceneState;
use super::viewport::ViewportState;

/// The PIE tick rate handed to the player (fixed-step Hz).
const PIE_TICK_HZ: u32 = inf_runtime::TICK_HZ;

/// Live PIE state: the session (absent when stopped) + its mode + the monitor
/// thread's run flag.
#[derive(Default)]
pub struct PieState {
    inner: Mutex<PieInner>,
}

#[derive(Default)]
struct PieInner {
    session: Option<PieSession>,
    mode: String,
    paused: bool,
    /// `true` while a session's window has been reparented into the viewport.
    embedded: bool,
    /// Signals the monitor thread to stop (a fresh one is made per session).
    monitor_run: Option<Arc<AtomicBool>>,
    last_frame: u64,
}

/// The `pie://state` event payload (hand-mirrored in the frontend, like the
/// blueprint types — no ts-rs binding needed for this Ring-2-only shape).
#[derive(Clone, serde::Serialize)]
struct PieStateEvent {
    running: bool,
    paused: bool,
    /// `"embedded"`, `"window"`, or `""` when stopped.
    mode: String,
    frame: u64,
    /// A crash / load message when the session died unexpectedly.
    error: Option<String>,
}

fn emit_state(app: &AppHandle, ev: PieStateEvent) {
    let _ = app.emit("pie://state", ev);
}

/// Start PIE over the current scene. `mode` is `"embedded"` (reparent the player
/// window into the viewport, Windows) or `"window"` (new window, all OSes). The
/// live doc — unsaved edits included — is serialized and streamed to the player.
#[tauri::command]
pub async fn pie_start(
    app: AppHandle,
    mode: String,
    scene: State<'_, SceneState>,
    assets: State<'_, AssetState>,
    pie: State<'_, PieState>,
) -> Result<(), String> {
    // Already running? Resume if paused, else no-op.
    {
        let mut inner = pie.inner.lock().map_err(|_| "pie lock poisoned")?;
        if inner.session.is_some() {
            if inner.paused {
                inner
                    .session
                    .as_mut()
                    .expect("session present")
                    .resume()
                    .map_err(|e| e.to_string())?;
                inner.paused = false;
                let frame = inner.last_frame;
                let m = inner.mode.clone();
                drop(inner);
                emit_state(
                    &app,
                    PieStateEvent {
                        running: true,
                        paused: false,
                        mode: m,
                        frame,
                        error: None,
                    },
                );
            }
            return Ok(());
        }
    }

    // Build the payload from the live scene (embedded/window are both windowed).
    let payload = {
        let doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
        build_scene_payload(
            &doc,
            |guid| assets.load_blueprint_class(inf_asset::AssetId(guid)),
            |guid| assets.load_pcg_bytes(inf_asset::AssetId(guid)),
            |guid| assets.load_anim_bytes(inf_asset::AssetId(guid)),
            PIE_TICK_HZ,
            true,
        )
        .map_err(|e| e.to_string())?
    };

    let bin = find_player_bin();
    let session = PieSession::spawn_scene(&bin, &payload)
        .map_err(|e| format!("could not start the player ({}): {e}", bin.display()))?;

    let embedded = mode == "embedded" && cfg!(windows);
    let run = Arc::new(AtomicBool::new(true));
    {
        let mut inner = pie.inner.lock().map_err(|_| "pie lock poisoned")?;
        inner.session = Some(session);
        inner.mode = mode.clone();
        inner.paused = false;
        inner.embedded = false;
        inner.last_frame = 0;
        inner.monitor_run = Some(Arc::clone(&run));
    }

    spawn_monitor(app.clone(), run, embedded);
    emit_state(
        &app,
        PieStateEvent {
            running: true,
            paused: false,
            mode,
            frame: 0,
            error: None,
        },
    );
    Ok(())
}

/// The background monitor: drains player events (Window → reparent for embedded
/// PIE, Frame → frame counter, Error/exit → crash toast + reset), and detects a
/// crash by waiting on the child. Keeps the editor intact when the player dies.
fn spawn_monitor(app: AppHandle, run: Arc<AtomicBool>, embedded: bool) {
    std::thread::Builder::new()
        .name("inf-pie-monitor".into())
        .spawn(move || {
            while run.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(40));
                let pie = app.state::<PieState>();
                let mut inner = match pie.inner.lock() {
                    Ok(g) => g,
                    Err(_) => break,
                };
                if inner.session.is_none() {
                    break;
                }

                // Drain queued player events + read health in one scoped borrow,
                // then mutate `inner` fields afterwards (avoids aliasing).
                let (window_handle, latest_frame, last_error, health, stderr) = {
                    let session = inner.session.as_mut().expect("session present");
                    let mut window_handle: Option<i64> = None;
                    let mut latest_frame: Option<u64> = None;
                    let mut last_error: Option<String> = None;
                    while let Some(ev) = session.next_event(Duration::from_millis(0)) {
                        match ev {
                            PlayerToEditor::Window { handle } => window_handle = Some(handle),
                            PlayerToEditor::Frame { frame, .. } => latest_frame = Some(frame),
                            PlayerToEditor::State(s) => {
                                latest_frame = Some(s.frame);
                                if s.last_error.is_some() {
                                    last_error = s.last_error;
                                }
                            }
                            PlayerToEditor::Error { message } => last_error = Some(message),
                            _ => {}
                        }
                    }
                    let health = session.health();
                    let stderr = match health {
                        SessionHealth::Exited { .. } => session.stderr_lines(),
                        SessionHealth::Running => Vec::new(),
                    };
                    (window_handle, latest_frame, last_error, health, stderr)
                };
                if let Some(f) = latest_frame {
                    inner.last_frame = f;
                }

                // Embedded PIE: reparent the player's window into the viewport
                // slot once it reports its handle.
                if embedded && !inner.embedded {
                    if let Some(handle) = window_handle {
                        if handle != 0 {
                            app.state::<ViewportState>().embed_foreign(handle);
                            inner.embedded = true;
                        }
                    }
                }

                // Crash / clean-exit detection.
                match health {
                    SessionHealth::Running => {
                        if let Some(err) = last_error {
                            let mode = inner.mode.clone();
                            let frame = inner.last_frame;
                            drop(inner);
                            emit_state(
                                &app,
                                PieStateEvent {
                                    running: true,
                                    paused: false,
                                    mode,
                                    frame,
                                    error: Some(err),
                                },
                            );
                        }
                    }
                    SessionHealth::Exited { code } => {
                        // The player died. Restore the viewport, surface the
                        // captured panic text, and reset — the editor is intact.
                        let was_embedded = inner.embedded;
                        inner.session = None;
                        inner.embedded = false;
                        inner.paused = false;
                        inner.mode.clear();
                        drop(inner);
                        if was_embedded {
                            app.state::<ViewportState>().release_foreign();
                            app.state::<ViewportState>().set_visible(true);
                        }
                        let msg = crash_message(code, &stderr);
                        run.store(false, Ordering::SeqCst);
                        emit_state(
                            &app,
                            PieStateEvent {
                                running: false,
                                paused: false,
                                mode: String::new(),
                                frame: 0,
                                error: Some(msg),
                            },
                        );
                        break;
                    }
                }
            }
        })
        .expect("spawn PIE monitor thread");
}

/// A human crash message from the exit code + the tail of captured stderr (the
/// panic line the player wrote before dying).
fn crash_message(code: Option<i32>, stderr: &[String]) -> String {
    let panic_line = stderr
        .iter()
        .rev()
        .find(|l| l.contains("panic") || l.contains("PIE panic"))
        .cloned();
    match (code, panic_line) {
        (_, Some(line)) => format!("PIE player crashed: {line}"),
        (Some(c), None) => format!("PIE player exited with code {c}"),
        (None, None) => "PIE player exited unexpectedly".to_string(),
    }
}

/// Pause the running player.
#[tauri::command]
pub async fn pie_pause(app: AppHandle, pie: State<'_, PieState>) -> Result<(), String> {
    let mut inner = pie.inner.lock().map_err(|_| "pie lock poisoned")?;
    if let Some(session) = inner.session.as_mut() {
        session.pause().map_err(|e| e.to_string())?;
        inner.paused = true;
        let (mode, frame) = (inner.mode.clone(), inner.last_frame);
        drop(inner);
        emit_state(
            &app,
            PieStateEvent {
                running: true,
                paused: true,
                mode,
                frame,
                error: None,
            },
        );
    }
    Ok(())
}

/// Resume a paused player.
#[tauri::command]
pub async fn pie_resume(app: AppHandle, pie: State<'_, PieState>) -> Result<(), String> {
    let mut inner = pie.inner.lock().map_err(|_| "pie lock poisoned")?;
    if let Some(session) = inner.session.as_mut() {
        session.resume().map_err(|e| e.to_string())?;
        inner.paused = false;
        let (mode, frame) = (inner.mode.clone(), inner.last_frame);
        drop(inner);
        emit_state(
            &app,
            PieStateEvent {
                running: true,
                paused: false,
                mode,
                frame,
                error: None,
            },
        );
    }
    Ok(())
}

/// Advance one fixed step while paused.
#[tauri::command]
pub async fn pie_step(pie: State<'_, PieState>) -> Result<(), String> {
    let mut inner = pie.inner.lock().map_err(|_| "pie lock poisoned")?;
    if let Some(session) = inner.session.as_mut() {
        session.step(1).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Release input possession back to the editor (v1 Eject). The player keeps
/// running.
#[tauri::command]
pub async fn pie_eject(pie: State<'_, PieState>) -> Result<(), String> {
    let mut inner = pie.inner.lock().map_err(|_| "pie lock poisoned")?;
    if let Some(session) = inner.session.as_mut() {
        session.eject().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Stop PIE: kill the player, restore the viewport, reset the toolbar.
#[tauri::command]
pub async fn pie_stop(app: AppHandle, pie: State<'_, PieState>) -> Result<(), String> {
    let (session, was_embedded, run) = {
        let mut inner = pie.inner.lock().map_err(|_| "pie lock poisoned")?;
        let run = inner.monitor_run.take();
        let was_embedded = inner.embedded;
        inner.embedded = false;
        inner.paused = false;
        inner.mode.clear();
        inner.last_frame = 0;
        (inner.session.take(), was_embedded, run)
    };
    if let Some(run) = run {
        run.store(false, Ordering::SeqCst);
    }
    if was_embedded {
        let vp = app.state::<ViewportState>();
        vp.release_foreign();
        vp.set_visible(true);
    }
    if let Some(session) = session {
        // Graceful stop, then guaranteed teardown (Drop kills a stuck child).
        let _ = session.stop(Duration::from_secs(5));
    }
    emit_state(
        &app,
        PieStateEvent {
            running: false,
            paused: false,
            mode: String::new(),
            frame: 0,
            error: None,
        },
    );
    Ok(())
}

/// Whether a PIE session is currently running (mount-time sync).
#[tauri::command]
pub async fn pie_is_running(pie: State<'_, PieState>) -> Result<bool, String> {
    Ok(pie
        .inner
        .lock()
        .map_err(|_| "pie lock poisoned")?
        .session
        .is_some())
}
