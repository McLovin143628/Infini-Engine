//! Viewport commands: attach the native engine viewport to the editor window
//! and keep its rectangle in sync with the React layout hole.
//!
//! Events/commands here are the Spike A bridge (docs/ROADMAP.md §5): the
//! frontend reports CSS rects × devicePixelRatio; we forward physical pixels
//! to the `inf-viewport` thread.

use std::sync::Arc;
use std::sync::Mutex;

use inf_editor_core::ipc::{
    FoliageSettingsDto, GizmoModeDto, GizmoSpaceDto, SculptFalloffDto, SculptOpDto,
    SculptSettingsDto, Snap2DDto, Snap3DDto, ToolModeDto, ViewModeDto, ViewportDrop, ViewportKey,
    ViewportModeDto, ViewportRect,
};
use inf_viewport::{
    FoliageSettings, GizmoSpace, SculptFalloff, SculptOp, SculptSettings, Snap2DSettings,
    SnapSettings, ToolMode, ViewportEvent, ViewportMode,
};
use tauri::{Emitter, Manager};

use super::scene::{emit_world_delta, SceneState};

#[derive(Default)]
pub struct ViewportState(Mutex<Option<inf_viewport::ViewportHandle>>);

impl ViewportState {
    /// Show/hide the native viewport child (used by PIE embedding to hide the
    /// editor viewport while the player window occupies the slot).
    pub fn set_visible(&self, visible: bool) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.set_visible(visible);
            }
        }
    }

    /// Adopt a foreign (PIE player) window into the viewport slot (embedded PIE).
    pub fn embed_foreign(&self, hwnd: i64) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.embed_foreign(hwnd as isize);
            }
        }
    }

    /// Release an embedded foreign window and restore the native viewport child.
    pub fn release_foreign(&self) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.release_foreign();
            }
        }
    }
}

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
        // The gizmo mode changed on the viewport side (a W/E/R keypress or an IPC
        // set); forward it so the toolbar stays in sync (two-way, Wave 2).
        ViewportEvent::GizmoModeChanged(mode) => {
            if let Err(e) = app.emit("viewport://gizmo", mode) {
                tracing::warn!("viewport://gizmo emit failed: {e}");
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

/// Switch the active viewport projection (Perspective ↔ 2D ortho) from the
/// viewport toolbar (P8.2c).
#[tauri::command]
pub async fn viewport_set_mode(
    mode: ViewportModeDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_mode(match mode {
            ViewportModeDto::Perspective => ViewportMode::Perspective,
            ViewportModeDto::TwoD => ViewportMode::TwoD,
        });
    }
    Ok(())
}

/// Push the 2D-mode snapping configuration (grid + pixel snap) to the viewport
/// (P8.2c). The frontend folds the per-project pixels-per-unit into this.
#[tauri::command]
pub async fn viewport_set_snap2d(
    snap: Snap2DDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_snap_2d(Snap2DSettings {
            grid_enabled: snap.grid_enabled,
            grid_size: snap.grid_size,
            pixel_enabled: snap.pixel_enabled,
            pixels_per_unit: snap.pixels_per_unit,
        });
    }
    Ok(())
}

/// Switch the active viewport tool (Select ↔ Sculpt) from the toolbar (P10.2b).
#[tauri::command]
pub async fn viewport_set_tool_mode(
    mode: ToolModeDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_tool_mode(match mode {
            ToolModeDto::Select => ToolMode::Select,
            ToolModeDto::Sculpt => ToolMode::Sculpt,
            ToolModeDto::Foliage => ToolMode::Foliage,
        });
    }
    Ok(())
}

/// Push the sculpt brush configuration (op / radius / strength / falloff) to the
/// viewport (P10.2b).
#[tauri::command]
pub async fn viewport_set_sculpt(
    sculpt: SculptSettingsDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_sculpt(SculptSettings {
            op: match sculpt.op {
                SculptOpDto::Raise => SculptOp::Raise,
                SculptOpDto::Lower => SculptOp::Lower,
                SculptOpDto::Smooth => SculptOp::Smooth,
                SculptOpDto::Flatten => SculptOp::Flatten,
                SculptOpDto::Noise => SculptOp::Noise,
                SculptOpDto::Paint => SculptOp::Paint,
            },
            radius: sculpt.radius.max(0.0),
            strength: sculpt.strength,
            falloff: match sculpt.falloff {
                SculptFalloffDto::Smooth => SculptFalloff::Smooth,
                SculptFalloffDto::Linear => SculptFalloff::Linear,
                SculptFalloffDto::Sphere => SculptFalloff::Sphere,
                SculptFalloffDto::Sharp => SculptFalloff::Sharp,
            },
            paint_layer: sculpt.paint_layer.min(3),
        });
    }
    Ok(())
}

/// Push the foliage brush configuration (radius / density / kind / …) to the
/// viewport (E-P6).
#[tauri::command]
pub async fn viewport_set_foliage(
    foliage: FoliageSettingsDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_foliage(FoliageSettings {
            radius: foliage.radius.max(0.0),
            density: foliage.density.max(0.0),
            erase: foliage.erase,
            kind: foliage.kind,
            scale_jitter: foliage.scale_jitter.max(0.0),
            align_to_normal: foliage.align_to_normal,
            seed: foliage.seed,
        });
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

/// Set the transform-gizmo mode (translate/rotate/scale) from the toolbar or
/// command palette (Wave 2). The viewport echoes mode changes (including W/E/R
/// keypresses over the viewport) back on the `viewport://gizmo` event.
#[tauri::command]
pub async fn viewport_set_gizmo_mode(
    mode: GizmoModeDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_gizmo_mode(mode);
    }
    Ok(())
}

/// Switch the gizmo orientation frame (World ↔ Local) from the toolbar (Wave 2).
#[tauri::command]
pub async fn viewport_set_gizmo_space(
    space: GizmoSpaceDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_gizmo_space(match space {
            GizmoSpaceDto::World => GizmoSpace::World,
            GizmoSpaceDto::Local => GizmoSpace::Local,
        });
    }
    Ok(())
}

/// Set the shading view mode (Lit / Unlit / Wireframe) from the viewport toolbar
/// (R-P2). `Wireframe` degrades to `Unlit` in the renderer when the adapter lacks
/// `POLYGON_MODE_LINE`. Editor-transient (never persisted; the player never sets
/// it).
#[tauri::command]
pub async fn viewport_set_view_mode(
    mode: ViewModeDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_view_mode(mode);
    }
    Ok(())
}

/// Push the 3D transform-gizmo snap increments (translate/rotate/scale +
/// always-on) from the toolbar (Wave 2).
#[tauri::command]
pub async fn viewport_set_snap3d(
    snap: Snap3DDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_snap_3d(SnapSettings {
            translate: snap.translate.max(0.0),
            rotate_deg: snap.rotate_deg.max(0.0),
            scale: snap.scale.max(0.0),
            always_on: snap.always_on,
        });
    }
    Ok(())
}
