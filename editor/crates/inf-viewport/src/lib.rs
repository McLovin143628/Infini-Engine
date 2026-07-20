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

#[cfg(any(windows, target_os = "macos"))]
mod render;

#[cfg(windows)]
mod win32;

#[cfg(target_os = "macos")]
mod macos;

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
/// Gated like [`render`]: on platforms with no embedding backend the enum
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
pub fn spawn(_parent: isize) -> ViewportHandle {
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
    pub fn destroy(&self) {}
}
