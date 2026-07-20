//! Native viewport host: per-OS child window embedding, input capture, surface lifecycle.
//!
//! Spike A (docs/ROADMAP.md §5): the engine renders into a real native child
//! window positioned *above* the webview inside a rectangular hole the React
//! layout reserves. This crate is Ring 1 — Tauri-free. The Studio app hands
//! us a raw parent window handle (`isize` HWND on Windows) and forwards rect
//! updates from the frontend's `ResizeObserver`.
//!
//! Platform status: Windows (Win32 child HWND) implemented; macOS
//! (NSView/CAMetalLayer) and Linux (X11 reparent, Wayland streaming
//! fallback) follow per the roadmap.

#[cfg(windows)]
mod render;
#[cfg(windows)]
mod win32;

/// Physical-pixel rectangle of the viewport hole, relative to the parent
/// window's client area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewportRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

#[cfg(windows)]
pub use win32::{spawn, ViewportHandle};

/// Non-Windows stub so Ring 2 code compiles everywhere while the spike is
/// Windows-only. Attaching is a no-op that logs once.
#[cfg(not(windows))]
pub struct ViewportHandle;

#[cfg(not(windows))]
pub fn spawn(_parent: isize) -> ViewportHandle {
    tracing::warn!(
        "inf-viewport: native embedding not yet implemented on this OS (see ROADMAP §5 Spike A)"
    );
    ViewportHandle
}

#[cfg(not(windows))]
impl ViewportHandle {
    pub fn set_rect(&self, _rect: ViewportRect) {}
    pub fn destroy(&self) {}
}
