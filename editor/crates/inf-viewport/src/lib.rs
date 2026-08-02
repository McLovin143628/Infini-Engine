//! Native viewport host: per-OS child window embedding, input capture, surface lifecycle.
//!
//! Spike A (docs/ROADMAP.md §5): the engine renders into a real native child
//! window positioned *above* the webview inside a rectangular hole the React
//! layout reserves. This crate is Ring 1 — Tauri-free. The Studio app hands
//! us a raw parent window handle (HWND on Windows, contentView NSView on
//! macOS) and forwards rect updates from the frontend's `ResizeObserver`.
//!
//! Platform status: Windows (Win32 child HWND) implemented and verified;
//! macOS (CAMetalLayer sublayer) implemented, compile-verified only (needs
//! Mac hardware for a runtime pass); Linux (X11 reparent, Wayland streaming
//! fallback) follows per the roadmap.

pub mod camera;

pub use camera::{
    Camera2D, EditorCamera, FoliageSettings, GizmoSpace, SculptFalloff, SculptOp, SculptSettings,
    Snap2DSettings, SnapSettings, ToolMode, ViewportMode,
};

use std::sync::{Arc, Mutex};

use inf_editor_core::scene::SceneDoc;

/// The authoritative scene document, shared between the editor (Ring 2) and the
/// viewport thread. The viewport renders + picks against it and writes gizmo
/// edits back — one source of truth (P3.2).
pub type SharedScene = Arc<Mutex<SceneDoc>>;

#[cfg(any(windows, target_os = "macos"))]
mod host;

#[cfg(windows)]
mod win32;

#[cfg(target_os = "macos")]
mod macos;

/// A keyboard chord the native viewport didn't consume, forwarded to the
/// webview so global shortcuts (command palette, save, …) still fire while the
/// 3D view holds OS focus. Normalized to the frontend's `chordOf` format.
#[derive(Debug, Clone)]
pub struct KeyChord {
    /// e.g. "Ctrl+Shift+P", "F11". Modifier order: Ctrl, Alt, Shift.
    pub chord: String,
}

/// Events the viewport surfaces back to Ring 2 (Tauri), which turns them into
/// namespaced webview events. Ring 1 stays Tauri-free by going through this.
#[derive(Debug, Clone)]
pub enum ViewportEvent {
    /// A global-shortcut key chord to replay into the webview (focus handoff).
    Key(KeyChord),
    /// The viewport mutated the shared scene (a pick-select or a gizmo drag);
    /// Ring 2 responds by emitting a `world://delta` so the panels re-sync.
    WorldChanged,
    /// The transform-gizmo mode changed on the viewport side (a W/E/R keypress
    /// or an IPC set); Ring 2 forwards it on `viewport://gizmo` so the toolbar
    /// stays in sync (two-way, Wave 2). Carries the IPC DTO directly so the enum
    /// stays free of the platform-gated `inf_render::GizmoMode` (Linux builds
    /// this file too).
    GizmoModeChanged(inf_editor_core::ipc::GizmoModeDto),
    /// A tool raised a status message, and/or the projected terrain's *streamed*
    /// state changed (P16.4a). Ring 2 forwards it on `viewport://tool-status`:
    /// the message goes to the status bar, the flag greys out sculpt/paint —
    /// which is the smallest honest surface for "this terrain streams and the
    /// brushes cannot touch it yet".
    ToolStatus(inf_editor_core::ipc::ViewportToolStatusDto),
}

/// Sink the host installs to receive [`ViewportEvent`]s. `Arc` so the render
/// thread can hold it cheaply; `Send + Sync` so it crosses the thread boundary.
pub type ViewportEventSink = std::sync::Arc<dyn Fn(ViewportEvent) + Send + Sync>;

/// Physical-pixel rectangle of the viewport hole, relative to the parent
/// window's client area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewportRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Where the renderer should create its swapchain surface. Handles are raw
/// pointers smuggled as `isize` so the target can cross into the render
/// thread (the caller guarantees they outlive the viewport).
/// Gated like [`host`]: on platforms with no embedding backend the enum
/// would be empty and dead — Linux CI's clippy rejects that.
#[cfg(any(windows, target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
pub(crate) enum SurfaceTarget {
    #[cfg(windows)]
    Win32 { hwnd: isize, hinstance: isize },
    #[cfg(target_os = "macos")]
    MetalLayer { layer: isize },
}

#[cfg(any(windows, target_os = "macos"))]
impl SurfaceTarget {
    /// # Safety
    /// The underlying native handle must remain valid for the surface's
    /// lifetime.
    pub(crate) unsafe fn create_surface(
        self,
        instance: &wgpu::Instance,
    ) -> Result<wgpu::Surface<'static>, String> {
        match self {
            #[cfg(windows)]
            SurfaceTarget::Win32 { hwnd, hinstance } => {
                use raw_window_handle::{
                    RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
                };
                use std::num::NonZeroIsize;

                let mut win32 = Win32WindowHandle::new(
                    NonZeroIsize::new(hwnd).ok_or_else(|| "null HWND".to_string())?,
                );
                // Vulkan surface creation requires hinstance, not just hwnd.
                win32.hinstance = NonZeroIsize::new(hinstance);
                instance
                    .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                        raw_display_handle: Some(RawDisplayHandle::Windows(
                            WindowsDisplayHandle::new(),
                        )),
                        raw_window_handle: RawWindowHandle::Win32(win32),
                    })
                    .map_err(|e| format!("create_surface: {e}"))
            }
            #[cfg(target_os = "macos")]
            SurfaceTarget::MetalLayer { layer } => instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(
                    layer as *mut std::ffi::c_void,
                ))
                .map_err(|e| format!("create_surface: {e}")),
        }
    }
}

#[cfg(windows)]
pub use win32::{spawn, ViewportHandle};

#[cfg(target_os = "macos")]
pub use macos::{spawn, ViewportHandle};

/// Stub so Ring 2 code compiles on platforms without an embedding backend
/// yet (Linux: X11 reparent / Wayland streaming fallback per ROADMAP §5).
/// Attaching is a no-op that logs once.
#[cfg(not(any(windows, target_os = "macos")))]
pub struct ViewportHandle;

#[cfg(not(any(windows, target_os = "macos")))]
pub fn spawn(_parent: isize, _sink: ViewportEventSink, _scene: SharedScene) -> ViewportHandle {
    tracing::warn!(
        "inf-viewport: native embedding not yet implemented on this OS (see ROADMAP §5 Spike A)"
    );
    ViewportHandle
}

#[cfg(not(any(windows, target_os = "macos")))]
impl ViewportHandle {
    pub fn set_rect(&self, _rect: ViewportRect) {}
    pub fn set_visible(&self, _visible: bool) {}
    pub fn drop_payload(&self, _x: f32, _y: f32, _payload: &str) {}
    pub fn set_mode(&self, _mode: camera::ViewportMode) {}
    pub fn set_snap_2d(&self, _snap: camera::Snap2DSettings) {}
    pub fn set_tool_mode(&self, _mode: camera::ToolMode) {}
    pub fn set_sculpt(&self, _sculpt: camera::SculptSettings) {}
    pub fn set_foliage(&self, _foliage: camera::FoliageSettings) {}
    pub fn set_gizmo_mode(&self, _mode: inf_editor_core::ipc::GizmoModeDto) {}
    pub fn set_gizmo_space(&self, _space: camera::GizmoSpace) {}
    pub fn set_snap_3d(&self, _snap: camera::SnapSettings) {}
    pub fn set_view_mode(&self, _mode: inf_editor_core::ipc::ViewModeDto) {}
    pub fn set_content_root(&self, _root: Option<std::path::PathBuf>) {}
    pub fn refresh_asset_index(&self) {}
    pub fn reload_terrain_stores(&self) {}
    pub fn clear_streams(&self) {}
    pub fn embed_foreign(&self, _hwnd: isize) {}
    pub fn release_foreign(&self) {}
    pub fn destroy(&self) {}
}
