//! Viewport commands: attach the native engine viewport to the editor window
//! and keep its rectangle in sync with the React layout hole.
//!
//! Events/commands here are the Spike A bridge (docs/ROADMAP.md §5): the
//! frontend reports CSS rects × devicePixelRatio; we forward physical pixels
//! to the `inf-viewport` thread.

use std::sync::Mutex;

#[derive(Default)]
pub struct ViewportState(Mutex<Option<inf_viewport::ViewportHandle>>);

/// Create (once) the native child viewport inside the calling window.
#[tauri::command]
pub async fn viewport_attach(
    window: tauri::Window,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if guard.is_some() {
        return Ok(()); // idempotent: React StrictMode double-mounts
    }

    let raw = {
        use raw_window_handle::{HasWindowHandle, RawWindowHandle};
        match window.window_handle().map_err(|e| e.to_string())?.as_raw() {
            RawWindowHandle::Win32(h) => h.hwnd.get(),
            _ => {
                return Err(
                    "unsupported platform for native viewport (spike is Windows-only)".into(),
                )
            }
        }
    };

    *guard = Some(inf_viewport::spawn(raw));
    tracing::info!("viewport attached (parent hwnd {raw:#x})");
    Ok(())
}

/// Update the viewport rectangle. Arguments are physical pixels relative to
/// the window client area (the frontend multiplies by devicePixelRatio).
#[tauri::command]
pub async fn viewport_set_rect(
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_rect(inf_viewport::ViewportRect {
            x: x.round() as i32,
            y: y.round() as i32,
            width: width.round().max(0.0) as u32,
            height: height.round().max(0.0) as u32,
        });
    }
    Ok(())
}
