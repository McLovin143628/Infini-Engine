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

use inf_editor_core::pie::{
    build_scene_payload, find_player_bin, PieSession, SessionHealth, TerrainRef,
};
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
    /// **`true` between `pie_start`'s decision to start and the session landing
    /// in `session`.**
    ///
    /// `pie_start` reads `session`, releases the lock to build the payload and
    /// spawn the process — hundreds of milliseconds, dominated by serializing
    /// the whole live scene — and only then takes the lock again to install what
    /// it made. Without this latch two overlapping calls both see `None`, both
    /// spawn a player, and the second overwrites the first: one orphan process
    /// nobody holds a handle to, plus its monitor thread polling a `PieState`
    /// that no longer describes it, for the life of the editor. Two calls are
    /// not exotic — the toolbar button and the `F5` chord are two paths to one
    /// command, and the `async` command runs on Tauri's pool.
    ///
    /// Cleared on **every** exit path by [`StartGuard`], including the `?`
    /// returns between the latch and the install.
    starting: bool,
}

impl PieInner {
    /// Install a fresh monitor stop-flag, **retiring the one it replaces**.
    ///
    /// Overwriting `monitor_run` drops the only handle that could ever have
    /// stopped the previous thread, and that thread's loop condition is the flag
    /// it still holds — so it goes on polling this same `PieState` every 40 ms
    /// for the life of the editor, reading a session it does not own and
    /// competing with the live monitor for the lock. A method rather than two
    /// lines at the call site so that the retirement is a property something can
    /// be pointed at: see `installing_a_monitor_retires_the_one_it_replaces`.
    fn install_monitor(&mut self, run: Arc<AtomicBool>) {
        if let Some(old) = self.monitor_run.take() {
            old.store(false, Ordering::SeqCst);
        }
        self.monitor_run = Some(run);
    }
}

/// Clears [`PieInner::starting`] however `pie_start` leaves.
///
/// A guard rather than three `inner.starting = false` lines, because two of the
/// three exits are `?` on a `Result` — the payload build and the spawn — and a
/// latch that survives a failed start is worse than no latch: it refuses every
/// subsequent Play until the editor is restarted.
struct StartGuard<'a>(&'a Mutex<PieInner>);

impl Drop for StartGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.0.lock() {
            inner.starting = false;
        }
    }
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
    // Already running, or already starting? Resume if paused, else no-op.
    //
    // The `starting` half is the whole point: everything between here and the
    // install below happens with the lock RELEASED, so `session.is_some()` alone
    // answers "no session yet" to a second caller that arrived a millisecond
    // later, and two players get spawned.
    let _start_guard = {
        let mut inner = pie.inner.lock().map_err(|_| "pie lock poisoned")?;
        if inner.starting {
            // A start is already in flight. Not an error — the second press of a
            // button is not a failure — and not a resume either, because there is
            // nothing yet to resume.
            return Ok(());
        }
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
        // Nothing running and nothing in flight: claim the start, and hand the
        // guard out of this scope so the claim is released however we leave.
        inner.starting = true;
        StartGuard(&pie.inner)
    };

    // Build the payload from the live scene (embedded/window are both windowed).
    let payload = {
        let doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
        build_scene_payload(
            &doc,
            |guid| assets.load_blueprint_class(inf_asset::AssetId(guid)),
            |guid| assets.load_pcg_bytes(inf_asset::AssetId(guid)),
            |guid| assets.load_anim_bytes(inf_asset::AssetId(guid)),
            |guid| assets.load_biome_set_bytes(inf_asset::AssetId(guid)),
            // P21.4: the `.inf_voxel` sources a level's `VoxelVolume`s name, so a
            // PIE session stands on the same cave floors the shipped build does.
            // The SAVED bytes — see `build_scene_payload`'s note on why shipping
            // the camera-paged editor store instead would make a preview depend on
            // where the author was looking.
            |guid| assets.load_voxel_bytes(inf_asset::AssetId(guid)),
            // P21.4 (the P16.3b2 deferral): the `.inf_terrain` a streamed
            // terrain's working set was stripped in favour of. Without it a PIE
            // session over an asset-backed terrain previews no ground at all.
            //
            // **By PATH** (wave GTA1): the editor always has a file for a terrain
            // in its database, and the island's is 549.9 MB — bigger than a PIE
            // frame may be, which is how Play died on the island with a one-line
            // status message. `TerrainRef::Bytes` is left for callers that have no
            // file; this one always does.
            |guid| {
                assets
                    .terrain_path(inf_asset::AssetId(guid))
                    .map(TerrainRef::Path)
            },
            // P22.3: the `.inf_mesh` bytes a `Destructible` actor's fracture is
            // DERIVED from — the one payload entry that is computed rather than
            // resolved. Without it a PIE session over a destructible level would
            // have nothing to break, and PIE == shipping would be comparing an
            // empty map against a full one.
            |guid| assets.load_mesh_bytes(inf_asset::AssetId(guid)),
            // P26.3b: the four kinds `ScenePayload` v8 added — `.inf_cloth`,
            // `.inf_hair`, `.inf_mat` and `.inf_tex`. One resolver because all
            // four are the same act (read this asset's committed bytes); the
            // KIND check lives in the loader so a mistyped binding resolves to
            // `None` here rather than shipping a garment's bytes as a texture.
            |guid| assets.load_binding_bytes(inf_asset::AssetId(guid)),
            PIE_TICK_HZ,
            true,
        )
        .map_err(|e| e.to_string())?
    };

    let bin = find_player_bin();
    // Wave GTA1: the OS's "cannot find the file specified" beside a path is not
    // something an author can act on. The overwhelmingly common cause in a dev
    // checkout has a one-line remedy, so say it.
    if let Some(advice) = inf_editor_core::pie::missing_player_advice(&bin) {
        return Err(advice);
    }
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
        // Retires the previous flag first. Unreachable while `starting` holds —
        // and this is what keeps a stray monitor from outliving its session if
        // the latch is ever wrong.
        inner.install_monitor(Arc::clone(&run));
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
                            // PRIMARY, explicitly (P23.2a): embedding reparents
                            // the player's window into the SCENE viewport's
                            // slot. Broadcasting it would try to adopt one
                            // window into several holes at once.
                            app.state::<ViewportState>()
                                .embed_foreign(super::Target::Primary, handle);
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
                            // The embed's twin, and Primary for the same
                            // reason: only the scene viewport was hidden, so
                            // only it is restored (P23.2a).
                            let vp = app.state::<ViewportState>();
                            vp.release_foreign(super::Target::Primary);
                            vp.set_visible(super::Target::Primary, true);
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
        // Primary, explicitly: the stop path restores exactly the slot the
        // embed took (P23.2a).
        vp.release_foreign(super::Target::Primary);
        vp.set_visible(super::Target::Primary, true);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// **A monitor thread must never outlive the flag that stops it**
    /// (Hardening Wave C, L6.F6).
    ///
    /// Built to falsify: replace `install_monitor`'s body with a bare
    /// `self.monitor_run = Some(run)` and the middle assertion fails. Nothing
    /// else in the tree can see the difference, because the symptom is a thread
    /// that keeps working correctly on a `PieState` that has moved on.
    #[test]
    fn installing_a_monitor_retires_the_one_it_replaces() {
        let mut inner = PieInner::default();

        let first = Arc::new(AtomicBool::new(true));
        inner.install_monitor(Arc::clone(&first));
        assert!(
            first.load(Ordering::SeqCst),
            "installing a monitor must not stop it"
        );

        let second = Arc::new(AtomicBool::new(true));
        inner.install_monitor(Arc::clone(&second));
        assert!(
            !first.load(Ordering::SeqCst),
            "the replaced monitor was never told to stop; its thread polls this \
             PieState every 40 ms for the life of the editor"
        );
        assert!(
            second.load(Ordering::SeqCst),
            "the new monitor was stopped along with the old one"
        );

        // Only ever one flag held, so `pie_stop` can always find the live one.
        assert!(Arc::ptr_eq(
            inner.monitor_run.as_ref().expect("a flag is installed"),
            &second
        ));
    }

    /// **The start latch is released however the start ends** — including the
    /// two `?` returns between claiming it and installing the session.
    ///
    /// A latch that survives a failed start is worse than no latch: it refuses
    /// every subsequent Play until the editor is restarted, and the failure that
    /// set it is exactly the one an author retries.
    #[test]
    fn the_start_latch_is_released_however_the_start_ends() {
        let state = Mutex::new(PieInner::default());

        // The claim, as `pie_start` makes it.
        let claimed = {
            let mut inner = state.lock().expect("fresh mutex");
            let free = !inner.starting && inner.session.is_none();
            if free {
                inner.starting = true;
            }
            free
        };
        assert!(claimed, "a fresh state must admit a starter");

        {
            let _guard = StartGuard(&state);
            assert!(
                state.lock().expect("not poisoned").starting,
                "the latch must hold for the whole window the lock is released \
                 over — that window is where the payload is built and the process \
                 spawned, and it is the only reason this latch exists"
            );
            // …and here the spawn fails, `?`-style.
        }

        assert!(
            !state.lock().expect("not poisoned").starting,
            "a failed start left the latch set — every later Play would be \
             silently refused"
        );

        // And a second starter is admitted again afterwards.
        let mut inner = state.lock().expect("not poisoned");
        assert!(!inner.starting && inner.session.is_none());
        inner.starting = true;
    }

    /// **The claim happens before the lock is released**, which is the whole of
    /// the fix — a source pin, because the property is an *ordering* between two
    /// statements and no value assertion in one thread can see it.
    ///
    /// The window is not theoretical: `build_scene_payload` serializes the
    /// entire live scene, and the toolbar button and the `F5` chord are two
    /// paths into one `async` command running on Tauri's pool.
    #[test]
    fn the_start_path_claims_the_latch_before_it_releases_the_lock() {
        // Normalized: `core.autocrlf = true` checks `.rs` out CRLF on Windows,
        // where a newline-delimited search finds nothing (the P22 law).
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/commands/pie.rs"),
        )
        .expect("this module is readable")
        .replace("\r\n", "\n");

        let refuses = src
            .find("if inner.starting {")
            .expect("`pie_start` no longer turns a second starter away");
        let claims = src
            .find("inner.starting = true;")
            .expect("`pie_start` no longer claims the latch");
        let guard = src
            .find("StartGuard(&pie.inner)")
            .expect("the claim is no longer released by a guard");
        let builds = src
            .find("build_scene_payload(")
            .expect("`pie_start` no longer builds a payload");
        let installs = src
            .find("inner.install_monitor(")
            .expect("`pie_start` no longer installs its monitor through the retiring door");

        assert!(
            refuses < claims && claims < guard && guard < builds && builds < installs,
            "the start sequence is out of order (refuse {refuses}, claim {claims}, \
             guard {guard}, build {builds}, install {installs}): the latch must be \
             claimed and guarded BEFORE the lock is dropped for the payload build, \
             or two overlapping calls both see no session and both spawn a player"
        );
    }
}
