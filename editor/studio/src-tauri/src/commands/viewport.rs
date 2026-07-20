//! Viewport commands: attach the native engine viewport to the editor window
//! and keep its rectangle in sync with the React layout hole.
//!
//! Events/commands here are the Spike A bridge (docs/ROADMAP.md §5): the
//! frontend reports CSS rects × devicePixelRatio; we forward physical pixels
//! to the `inf-viewport` thread.

use std::sync::Arc;
use std::sync::Mutex;

use inf_editor_core::ipc::{ViewportDrop, ViewportKey, ViewportRect};
use inf_viewport::ViewportEvent;
use tauri::{Emitter, Manager};

use super::scene::{emit_world_delta, SceneState};

#[derive(Default)]
pub struct ViewportState(Mutex<Option<inf_viewport::ViewportHandle>>);

/// Create (once) the native child viewport inside the calling window.
#[tauri::command]
pub async fn viewport_attach(
    window: tauri::Window,
    state: tauri::State<'_, ViewportState>,
    scene: tauri::State<'_, SceneState>,
) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if guard.is_some() {
        return Ok(()); // idempotent: React StrictMode double-mounts
    }

    *guard = Some(attach_native(&window, scene.doc.clone())?);
    tracing::info!("viewport attached");
    Ok(())
}

/// Build the event sink that turns [`ViewportEvent`]s into namespaced webview
/// events. Forwarded key chords arrive on `viewport://key` so the frontend's
/// keybinding dispatcher can replay them (focus handoff, P2.3.4).
fn event_sink(app: tauri::AppHandle) -> inf_viewport::ViewportEventSink {
    Arc::new(move |event: ViewportEvent| match event {
        ViewportEvent::Key(chord) => {
            if let Err(e) = app.emit("viewport://key", ViewportKey { chord: chord.chord }) {
                tracing::warn!("viewport://key emit failed: {e}");
            }
        }
        // A pick-select or gizmo drag on the viewport thread mutated the shared
        // document; re-emit the world delta so the Outliner/Details re-sync.
        ViewportEvent::WorldChanged => {
            if let Some(state) = app.try_state::<SceneState>() {
                emit_world_delta(&app, &state);
            }
        }
    })
}

#[cfg(windows)]
fn attach_native(
    window: &tauri::Window,
    scene: inf_viewport::SharedScene,
) -> Result<inf_viewport::ViewportHandle, String> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let hwnd = match window.window_handle().map_err(|e| e.to_string())?.as_raw() {
        RawWindowHandle::Win32(h) => h.hwnd.get(),
        _ => return Err("expected a Win32 window handle".into()),
    };
    Ok(inf_viewport::spawn(
        hwnd,
        event_sink(window.app_handle().clone()),
        scene,
    ))
}

#[cfg(target_os = "macos")]
fn attach_native(
    window: &tauri::Window,
    scene: inf_viewport::SharedScene,
) -> Result<inf_viewport::ViewportHandle, String> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let ns_view = match window.window_handle().map_err(|e| e.to_string())?.as_raw() {
        RawWindowHandle::AppKit(h) => h.ns_view.as_ptr() as isize,
        _ => return Err("expected an AppKit window handle".into()),
    };
    let sink = event_sink(window.app_handle().clone());
    // AppKit setup must happen on the main thread; commands run on a worker.
    let (tx, rx) = std::sync::mpsc::channel();
    window
        .run_on_main_thread(move || {
            let _ = tx.send(inf_viewport::spawn(ns_view, sink, scene));
        })
        .map_err(|e| e.to_string())?;
    rx.recv().map_err(|e| e.to_string())
}

#[cfg(not(any(windows, target_os = "macos")))]
fn attach_native(
    window: &tauri::Window,
    scene: inf_viewport::SharedScene,
) -> Result<inf_viewport::ViewportHandle, String> {
    // Linux embedding (X11 reparent / Wayland streaming fallback) is a later
    // Spike A batch — see ROADMAP §5.
    Ok(inf_viewport::spawn(
        0,
        event_sink(window.app_handle().clone()),
        scene,
    ))
}

/// Update the viewport rectangle ([`ViewportRect`]: physical pixels relative
/// to the window client area; the frontend multiplies by devicePixelRatio).
#[tauri::command]
pub async fn viewport_set_rect(
    rect: ViewportRect,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_rect(inf_viewport::ViewportRect {
            x: rect.x.round() as i32,
            y: rect.y.round() as i32,
            width: rect.width.round().max(0.0) as u32,
            height: rect.height.round().max(0.0) as u32,
        });
    }
    Ok(())
}

/// Show/hide the native viewport. The shell hides it while an HTML overlay
/// (menu, command palette, dialog, drag ghost) is open — the native child
/// otherwise draws OVER the webview (airspace rule) and occludes any
/// overlay crossing the hole. P2.1 explores flash-free alternatives.
#[tauri::command]
pub async fn viewport_set_visible(
    visible: bool,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_visible(visible);
    }
    Ok(())
}

/// Hand off a drag that ended over the viewport hole ([`ViewportDrop`]:
/// physical pixels relative to the hole's top-left corner). HTML drag ghosts
/// die over the native window (airspace rule), so the drop point crosses via
/// IPC and the engine side takes over — Phase 3 turns this into a pick-ray
/// spawn.
#[tauri::command]
pub async fn viewport_drop(
    drop: ViewportDrop,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.drop_payload(drop.x as f32, drop.y as f32, &drop.payload);
    }
    Ok(())
}
