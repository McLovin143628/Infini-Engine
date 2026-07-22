//! Win32 child-window host. A dedicated thread owns the child HWND (Win32
//! ties a window to the thread that created it), pumps its messages, drives
//! the render loop, and runs the editor camera; commands arrive over a channel.
//!
//! Input model (UE parity):
//! - RMB captures for the flycam (WASD/QE, wheel = speed, Shift = boost).
//! - Alt+LMB orbit, MMB (or Alt+MMB) pan, Alt+RMB dolly, all around the pivot.
//! - Wheel with no button dollies toward the look point.
//! - F focuses; Ctrl+1..9 store camera bookmarks, 1..9 recall them.
//! - Global-shortcut chords (Ctrl+…, F11) are forwarded to the webview so the
//!   command palette etc. still work while the 3D view holds OS focus
//!   (focus handoff, P2.3.4).

use std::cell::RefCell;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::time::Instant;

use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, GetKeyState, ReleaseCapture, SetCapture, SetFocus, VK_CONTROL, VK_MENU,
    VK_SHIFT,
};
use windows::Win32::UI::Input::{
    GetRawInputData, RegisterRawInputDevices, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE,
    RAWINPUTDEVICE_FLAGS, RAWINPUTHEADER, RID_INPUT, RIM_TYPEMOUSE,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetCursorPos, GetWindowLongPtrW,
    PeekMessageW, RegisterClassW, SetCursorPos, SetParent, SetWindowLongPtrW, SetWindowPos,
    ShowCursor, ShowWindow, TranslateMessage, GWL_STYLE, HWND_TOP, MSG, PM_REMOVE, SWP_NOACTIVATE,
    SWP_NOZORDER, SW_HIDE, SW_SHOWNA, WINDOW_EX_STYLE, WM_CAPTURECHANGED, WM_ERASEBKGND, WM_INPUT,
    WM_KEYDOWN, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEMOVE,
    WM_MOUSEWHEEL, WM_RBUTTONDOWN, WM_RBUTTONUP, WM_SYSKEYDOWN, WNDCLASSW, WS_CHILD,
    WS_CLIPSIBLINGS, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

use glam::DVec2;

use crate::camera::{
    Bookmarks, Camera2D, EditorCamera, FlyInput, NavInput, NavMode, SculptSettings, Snap2DSettings,
    ToolMode, ViewportMode,
};
use crate::host::EngineHost;
use crate::{KeyChord, SharedScene, SurfaceTarget, ViewportEvent, ViewportEventSink, ViewportRect};
use inf_render::GizmoMode;

enum Cmd {
    SetRect(ViewportRect),
    SetVisible(bool),
    Drop {
        x: f32,
        y: f32,
        payload: String,
    },
    /// Switch the active projection (Perspective ↔ 2D ortho) from the toolbar.
    SetMode(ViewportMode),
    /// Replace the 2D-mode snapping configuration from the toolbar.
    SetSnap2D(Snap2DSettings),
    /// Switch the active tool (Select ↔ Sculpt) from the toolbar (P10.2b).
    SetToolMode(ToolMode),
    /// Replace the sculpt brush configuration from the toolbar (P10.2b).
    SetSculpt(SculptSettings),
    /// Adopt a foreign (PIE player) window into the viewport slot: reparent it
    /// to our parent, position it at the hole, and hide our own child (embedded
    /// PIE, P9.4). The `isize` is the foreign HWND.
    EmbedForeign(isize),
    /// Release the embedded foreign window and restore our own child.
    ReleaseForeign,
    Destroy,
}

/// Cheap-to-clone handle for controlling the viewport thread.
pub struct ViewportHandle {
    tx: Sender<Cmd>,
}

impl ViewportHandle {
    /// Move/resize the child window (physical pixels, parent-client-relative).
    pub fn set_rect(&self, rect: ViewportRect) {
        let _ = self.tx.send(Cmd::SetRect(rect));
    }

    /// Show/hide the child window. The shell hides the viewport while an
    /// HTML overlay (menu, palette, dialog, drag ghost) is open — the native
    /// window otherwise draws OVER the webview (airspace rule), occluding
    /// any overlay that crosses the hole. P2.1 explores flash-free
    /// alternatives (window-region cutouts / last-frame freeze).
    pub fn set_visible(&self, visible: bool) {
        let _ = self.tx.send(Cmd::SetVisible(visible));
    }

    /// Hand off a drag-drop that ended over the viewport hole (Spike A stub:
    /// the webview keeps mouse capture during HTML drags, so the drop point
    /// arrives via IPC in viewport-local physical pixels).
    pub fn drop_payload(&self, x: f32, y: f32, payload: &str) {
        let _ = self.tx.send(Cmd::Drop {
            x,
            y,
            payload: payload.to_owned(),
        });
    }

    /// Switch the active viewport projection (Perspective ↔ 2D ortho).
    pub fn set_mode(&self, mode: ViewportMode) {
        let _ = self.tx.send(Cmd::SetMode(mode));
    }

    /// Replace the 2D-mode snapping configuration (grid + pixel snap).
    pub fn set_snap_2d(&self, snap: Snap2DSettings) {
        let _ = self.tx.send(Cmd::SetSnap2D(snap));
    }

    /// Switch the active viewport tool (Select ↔ Sculpt) (P10.2b).
    pub fn set_tool_mode(&self, mode: ToolMode) {
        let _ = self.tx.send(Cmd::SetToolMode(mode));
    }

    /// Replace the sculpt brush configuration (op / radius / strength / falloff).
    pub fn set_sculpt(&self, sculpt: SculptSettings) {
        let _ = self.tx.send(Cmd::SetSculpt(sculpt));
    }

    /// Adopt a foreign (PIE player) window into the viewport slot (embedded PIE,
    /// P9.4). The foreign window is reparented under our parent, sized to the
    /// hole, and our own render child is hidden. `hwnd` is the player's HWND.
    pub fn embed_foreign(&self, hwnd: isize) {
        let _ = self.tx.send(Cmd::EmbedForeign(hwnd));
    }

    /// Release the embedded foreign window and restore our own render child.
    pub fn release_foreign(&self) {
        let _ = self.tx.send(Cmd::ReleaseForeign);
    }

    /// Tear down the viewport thread and its window.
    pub fn destroy(&self) {
        let _ = self.tx.send(Cmd::Destroy);
    }
}

/// Spawn the viewport thread: creates a `WS_CHILD` window parented to
/// `parent_hwnd`, brings up the engine on it, and renders until destroyed.
/// `sink` receives events the viewport surfaces back (forwarded key chords).
pub fn spawn(parent_hwnd: isize, sink: ViewportEventSink, scene: SharedScene) -> ViewportHandle {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("inf-viewport".into())
        .spawn(move || thread_main(parent_hwnd, rx, sink, scene))
        .expect("failed to spawn inf-viewport thread");
    ViewportHandle { tx }
}

/// Which mouse gesture currently owns capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Capture {
    #[default]
    None,
    Fly,
    Orbit,
    Pan,
    Dolly,
}

/// A discrete camera action queued by `wnd_proc`, drained by the frame loop.
#[derive(Debug, Clone, Copy)]
enum Action {
    Focus,
    StoreBookmark(usize),
    RecallBookmark(usize),
    SetGizmo(GizmoMode),
}

/// Input accumulated by `wnd_proc` between frames. Thread-local is safe: the
/// wnd_proc always runs on the thread that created the window.
#[derive(Default)]
struct InputState {
    capture: Capture,
    mouse_dx: f32,
    mouse_dy: f32,
    wheel_steps: i32,
    restore_cursor: POINT,
    actions: Vec<Action>,
    chords: Vec<String>,
    /// Latest cursor position in viewport-client (physical) pixels.
    cursor: (i32, i32),
    cursor_moved: bool,
    /// Plain-LMB (no Alt) is held: select/gizmo, not orbit.
    left_down: bool,
    /// A plain-LMB press this frame: (x, y, ctrl-held) for select/gizmo-begin.
    left_press: Option<(i32, i32, bool)>,
    left_release: bool,
}

thread_local! {
    static INPUT: RefCell<InputState> = RefCell::new(InputState::default());
}

fn modifier(vk: windows::Win32::UI::Input::KeyboardAndMouse::VIRTUAL_KEY) -> bool {
    (unsafe { GetKeyState(vk.0 as i32) } as u16 & 0x8000) != 0
}

/// Start a capture of `kind`: grab the mouse, hide the cursor, remember where
/// to restore it. Win32 calls stay OUTSIDE any RefCell borrow (SetCapture
/// re-enters this wnd_proc synchronously via WM_CAPTURECHANGED).
fn begin_capture(hwnd: HWND, kind: Capture) {
    let start = INPUT.with(|s| {
        let mut s = s.borrow_mut();
        if s.capture != Capture::None {
            return false;
        }
        s.capture = kind;
        true
    });
    if !start {
        return;
    }
    let mut pt = POINT::default();
    unsafe {
        let _ = GetCursorPos(&mut pt);
    }
    INPUT.with(|s| s.borrow_mut().restore_cursor = pt);
    unsafe {
        SetCapture(hwnd);
        let _ = SetFocus(Some(hwnd));
        ShowCursor(false);
    }
}

/// End the capture if `kind` owns it, restoring the cursor. Returns true if it
/// released (i.e. `kind` was the active capture).
fn end_capture(kind: Capture) -> bool {
    let restore = INPUT.with(|s| {
        let mut s = s.borrow_mut();
        if s.capture == kind {
            s.capture = Capture::None;
            Some(s.restore_cursor)
        } else {
            None
        }
    });
    if let Some(pt) = restore {
        unsafe {
            let _ = ReleaseCapture();
            ShowCursor(true);
            let _ = SetCursorPos(pt.x, pt.y);
        }
        true
    } else {
        false
    }
}

/// Chord name for a virtual key, or `None` if we don't forward it.
fn vk_name(vk: u32) -> Option<String> {
    match vk {
        0x41..=0x5A => Some(((b'A' + (vk - 0x41) as u8) as char).to_string()), // A–Z
        0x30..=0x39 => Some(((b'0' + (vk - 0x30) as u8) as char).to_string()), // 0–9
        0x20 => Some("Space".into()),
        0x70..=0x7B => Some(format!("F{}", vk - 0x6F)), // F1–F12
        _ => None,
    }
}

/// Handle a key-down: bookmarks, focus, or a forwarded global-shortcut chord.
fn on_key_down(vk: u32) {
    let ctrl = modifier(VK_CONTROL);
    let alt = modifier(VK_MENU);
    let shift = modifier(VK_SHIFT);

    // Ctrl+digit stores a bookmark; plain digit recalls it.
    if let 0x31..=0x39 = vk {
        let slot = (vk - 0x30) as usize;
        let action = if ctrl {
            Action::StoreBookmark(slot)
        } else if !alt {
            Action::RecallBookmark(slot)
        } else {
            return;
        };
        INPUT.with(|s| s.borrow_mut().actions.push(action));
        return;
    }

    // F focuses the selection (no modifiers).
    if vk == 0x46 && !ctrl && !alt && !shift {
        INPUT.with(|s| s.borrow_mut().actions.push(Action::Focus));
        return;
    }

    // W/E/R switch the transform-gizmo mode (UE parity), but only when not
    // flying — while RMB-captured those are WASD/QE flycam keys.
    if !ctrl && !alt && !shift {
        let mode = match vk {
            0x57 => Some(GizmoMode::Translate), // W
            0x45 => Some(GizmoMode::Rotate),    // E
            0x52 => Some(GizmoMode::Scale),     // R
            _ => None,
        };
        if let Some(mode) = mode {
            let idle = INPUT.with(|s| s.borrow().capture == Capture::None);
            if idle {
                INPUT.with(|s| s.borrow_mut().actions.push(Action::SetGizmo(mode)));
                return;
            }
        }
    }

    // Forward global-shortcut chords (Ctrl+… or F-keys) to the webview so
    // they keep working while the native viewport holds focus.
    let is_fkey = (0x70..=0x7B).contains(&vk);
    if !ctrl && !is_fkey {
        return;
    }
    let Some(name) = vk_name(vk) else { return };
    let mut parts: Vec<&str> = Vec::new();
    if ctrl {
        parts.push("Ctrl");
    }
    if alt {
        parts.push("Alt");
    }
    if shift {
        parts.push("Shift");
    }
    let mut chord = parts.join("+");
    if !chord.is_empty() {
        chord.push('+');
    }
    chord.push_str(&name);
    INPUT.with(|s| s.borrow_mut().chords.push(chord));
}

extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match msg {
        // The swapchain covers every pixel — skipping the GDI background
        // erase kills the white flash during splitter resizes.
        WM_ERASEBKGND => LRESULT(1),

        // RMB: flycam, or dolly when Alt is held.
        WM_RBUTTONDOWN => {
            let kind = if modifier(VK_MENU) {
                Capture::Dolly
            } else {
                Capture::Fly
            };
            begin_capture(hwnd, kind);
            LRESULT(0)
        }
        WM_RBUTTONUP => {
            // RMB owns whichever of Fly/Dolly it started.
            end_capture(Capture::Fly);
            end_capture(Capture::Dolly);
            LRESULT(0)
        }

        // Alt+LMB orbits. Plain LMB selects / drags a gizmo handle.
        WM_LBUTTONDOWN => {
            if modifier(VK_MENU) {
                begin_capture(hwnd, Capture::Orbit);
            } else {
                let x = (lparam.0 & 0xffff) as i16 as i32;
                let y = ((lparam.0 >> 16) & 0xffff) as i16 as i32;
                let ctrl = modifier(VK_CONTROL);
                unsafe {
                    // Capture so we keep getting moves/up even off-window while
                    // dragging a gizmo. This is the plain-LMB path (not orbit).
                    SetCapture(hwnd);
                    let _ = SetFocus(Some(hwnd));
                }
                INPUT.with(|s| {
                    let mut s = s.borrow_mut();
                    s.left_down = true;
                    s.left_press = Some((x, y, ctrl));
                    s.cursor = (x, y);
                });
            }
            LRESULT(0)
        }
        WM_LBUTTONUP => {
            if end_capture(Capture::Orbit) {
                // was an orbit gesture
            } else {
                let owns = INPUT.with(|s| {
                    let mut s = s.borrow_mut();
                    if s.left_down {
                        s.left_down = false;
                        s.left_release = true;
                        true
                    } else {
                        false
                    }
                });
                if owns {
                    unsafe {
                        let _ = ReleaseCapture();
                    }
                }
            }
            LRESULT(0)
        }

        WM_MOUSEMOVE => {
            let x = (lparam.0 & 0xffff) as i16 as i32;
            let y = ((lparam.0 >> 16) & 0xffff) as i16 as i32;
            INPUT.with(|s| {
                let mut s = s.borrow_mut();
                s.cursor = (x, y);
                s.cursor_moved = true;
            });
            LRESULT(0)
        }

        // MMB pans (with or without Alt).
        WM_MBUTTONDOWN => {
            begin_capture(hwnd, Capture::Pan);
            LRESULT(0)
        }
        WM_MBUTTONUP => {
            end_capture(Capture::Pan);
            LRESULT(0)
        }

        // Capture can be stolen (alt-tab, other SetCapture); un-hide cleanly
        // and drop any in-flight gizmo/selection drag.
        WM_CAPTURECHANGED => {
            let was = INPUT.with(|s| {
                let mut s = s.borrow_mut();
                if s.left_down {
                    s.left_down = false;
                    s.left_release = true;
                }
                std::mem::replace(&mut s.capture, Capture::None) != Capture::None
            });
            if was {
                unsafe {
                    ShowCursor(true);
                }
            }
            LRESULT(0)
        }

        WM_MOUSEWHEEL => {
            let delta = ((wparam.0 >> 16) & 0xffff) as u16 as i16 as i32 / 120;
            INPUT.with(|s| s.borrow_mut().wheel_steps += delta);
            LRESULT(0)
        }

        WM_KEYDOWN | WM_SYSKEYDOWN => {
            on_key_down(wparam.0 as u32);
            // SYSKEYDOWN must reach DefWindowProc or Alt-handling breaks.
            if msg == WM_SYSKEYDOWN {
                unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
            } else {
                LRESULT(0)
            }
        }

        WM_INPUT => {
            let mut raw = RAWINPUT::default();
            let mut size = std::mem::size_of::<RAWINPUT>() as u32;
            let read = unsafe {
                GetRawInputData(
                    HRAWINPUT(lparam.0 as *mut _),
                    RID_INPUT,
                    Some(&mut raw as *mut _ as *mut _),
                    &mut size,
                    std::mem::size_of::<RAWINPUTHEADER>() as u32,
                )
            };
            if read != u32::MAX && raw.header.dwType == RIM_TYPEMOUSE.0 {
                let mouse = unsafe { raw.data.mouse };
                // Bit 0 = MOUSE_MOVE_ABSOLUTE; we only want relative deltas.
                if mouse.usFlags.0 & 0x01 == 0 {
                    INPUT.with(|s| {
                        let mut s = s.borrow_mut();
                        if s.capture != Capture::None {
                            s.mouse_dx += mouse.lLastX as f32;
                            s.mouse_dy += mouse.lLastY as f32;
                        }
                    });
                }
            }
            unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
        }

        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

fn create_child_window(parent_hwnd: isize) -> windows::core::Result<HWND> {
    unsafe {
        let hinstance = GetModuleHandleW(None)?;
        let class_name = w!("InfinityViewportClass");

        let wc = WNDCLASSW {
            lpfnWndProc: Some(wnd_proc),
            hInstance: hinstance.into(),
            lpszClassName: class_name,
            ..Default::default()
        };
        // Returns 0 if the class already exists (second viewport); that's fine.
        RegisterClassW(&wc);

        CreateWindowExW(
            WINDOW_EX_STYLE::default(),
            class_name,
            w!("Infinity Viewport"),
            WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS,
            0,
            0,
            64,
            64,
            Some(HWND(parent_hwnd as *mut _)),
            None,
            Some(hinstance.into()),
            None,
        )
    }
}

/// Reparent a foreign (PIE player) window into our parent window tree — the
/// proven Spike D cross-process sequence: switch it to `WS_CHILD`, `SetParent`
/// onto the editor's parent window, then position it at the hole rect. The
/// parent window is owned by the Tauri main thread (which always pumps), so the
/// child's later `DestroyWindow` `WM_PARENTNOTIFY` never deadlocks us (the
/// Spike D pump-while-teardown finding).
fn embed_foreign_window(parent_hwnd: isize, foreign: isize, rect: Option<ViewportRect>) {
    let parent = HWND(parent_hwnd as *mut _);
    let child = HWND(foreign as *mut _);
    unsafe {
        // MSDN order: WS_CHILD before SetParent (a top-level window must lose its
        // overlapped styles first).
        let style = GetWindowLongPtrW(child, GWL_STYLE);
        let child_style = (style & !(WS_OVERLAPPEDWINDOW.0 as isize)) | (WS_CHILD.0 as isize);
        SetWindowLongPtrW(child, GWL_STYLE, child_style);
        if let Err(e) = SetParent(child, Some(parent)) {
            tracing::warn!("inf-viewport: SetParent for embedded PIE failed: {e}");
            return;
        }
        if let Some(r) = rect {
            let _ = SetWindowPos(
                child,
                Some(HWND_TOP),
                r.x,
                r.y,
                r.width.max(1) as i32,
                r.height.max(1) as i32,
                SWP_NOACTIVATE,
            );
        }
        let _ = ShowWindow(child, SW_SHOWNA);
    }
}

fn register_raw_mouse(hwnd: HWND) {
    // Usage page 0x01 (generic desktop), usage 0x02 (mouse); deltas are
    // delivered as WM_INPUT while our window has keyboard focus.
    let rid = RAWINPUTDEVICE {
        usUsagePage: 0x01,
        usUsage: 0x02,
        dwFlags: RAWINPUTDEVICE_FLAGS(0),
        hwndTarget: hwnd,
    };
    if let Err(e) =
        unsafe { RegisterRawInputDevices(&[rid], std::mem::size_of::<RAWINPUTDEVICE>() as u32) }
    {
        tracing::warn!("inf-viewport: raw input registration failed: {e}");
    }
}

fn key_down(vk: i32) -> bool {
    (unsafe { GetAsyncKeyState(vk) } as u16 & 0x8000) != 0
}

/// Drained input for one frame.
struct FrameInput {
    capture: Capture,
    dx: f32,
    dy: f32,
    wheel: i32,
    actions: Vec<Action>,
    chords: Vec<String>,
    cursor: (i32, i32),
    cursor_moved: bool,
    left_down: bool,
    left_press: Option<(i32, i32, bool)>,
    left_release: bool,
}

fn drain_input() -> FrameInput {
    INPUT.with(|s| {
        let mut s = s.borrow_mut();
        FrameInput {
            capture: s.capture,
            dx: std::mem::take(&mut s.mouse_dx),
            dy: std::mem::take(&mut s.mouse_dy),
            wheel: std::mem::take(&mut s.wheel_steps),
            actions: std::mem::take(&mut s.actions),
            chords: std::mem::take(&mut s.chords),
            cursor: s.cursor,
            cursor_moved: std::mem::take(&mut s.cursor_moved),
            left_down: s.left_down,
            left_press: std::mem::take(&mut s.left_press),
            left_release: std::mem::take(&mut s.left_release),
        }
    })
}

/// Write a viewport crash report for a caught panic and log where it landed.
/// Shared by the init and per-frame guards; the editor process survives either.
fn report_viewport_panic(location: &str, payload: &(dyn std::any::Any + Send)) {
    let msg = inf_editor_core::diagnostics::panic_message(payload);
    let report = inf_editor_core::diagnostics::CrashReport {
        app: "inf-viewport".into(),
        engine_version: env!("CARGO_PKG_VERSION").into(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        location: location.into(),
        message: msg.clone(),
        adapter: None,
        log_tail: Vec::new(),
    };
    let dir = std::env::temp_dir().join("InfinityEngine").join("crashes");
    match inf_editor_core::diagnostics::write_crash_report(&dir, &report) {
        Ok(path) => tracing::error!(
            "inf-viewport: {location} panicked ({msg}); crash report at {} — \
             viewport stopped, editor still running",
            path.display()
        ),
        Err(werr) => tracing::error!(
            "inf-viewport: {location} panicked ({msg}); failed to write crash report: {werr}"
        ),
    }
}

fn thread_main(parent_hwnd: isize, rx: Receiver<Cmd>, sink: ViewportEventSink, scene: SharedScene) {
    let hwnd = match create_child_window(parent_hwnd) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!("inf-viewport: failed to create child window: {e}");
            return;
        }
    };

    // Keep the viewport above the WebView2 sibling so it owns its rectangle.
    unsafe {
        let _ = SetWindowPos(hwnd, Some(HWND_TOP), 0, 0, 64, 64, SWP_NOACTIVATE);
    }
    register_raw_mouse(hwnd);

    let hinstance = unsafe { GetModuleHandleW(None) }
        .map(|h| h.0 as isize)
        .unwrap_or_default();
    let target = SurfaceTarget::Win32 {
        hwnd: hwnd.0 as isize,
        hinstance,
    };
    // Engine init compiles every pass shader up-front; guard it the same way as
    // the per-frame render so a validation panic degrades to "viewport
    // unavailable" instead of silently killing this thread (the pick-shader
    // composition bug did exactly that).
    let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        EngineHost::new(target, 64, 64)
    }));
    let mut host = match init {
        Ok(Ok(h)) => h,
        Ok(Err(e)) => {
            tracing::error!("inf-viewport: engine init failed: {e}");
            return;
        }
        Err(payload) => {
            report_viewport_panic("<viewport engine init>", &*payload);
            return;
        }
    };
    tracing::info!("inf-viewport: child window + engine renderer up");

    let mut camera = EditorCamera::default();
    // A separate 2D ortho camera; switching modes preserves both poses (P8.2c).
    let mut camera_2d = Camera2D::default();
    let mut bookmarks = Bookmarks::default();
    // Active smooth-focus goal, cleared once the camera settles.
    let mut focus_goal = None;
    let mut last_frame = Instant::now();

    // Embedded PIE (P9.4): the adopted foreign player window + the latest hole
    // rect, so rect changes follow the embedded window while our own child hides.
    let mut embedded: Option<HWND> = None;
    let mut last_rect: Option<ViewportRect> = None;

    'outer: loop {
        // 1. Apply pending commands (coalesce rect updates to the latest).
        let mut latest_rect: Option<ViewportRect> = None;
        loop {
            match rx.try_recv() {
                Ok(Cmd::SetRect(r)) => latest_rect = Some(r),
                Ok(Cmd::SetVisible(v)) => unsafe {
                    // While a PIE window is embedded, keep our own child hidden.
                    if embedded.is_none() {
                        // SW_SHOWNA: show without stealing focus from the webview.
                        let _ = ShowWindow(hwnd, if v { SW_SHOWNA } else { SW_HIDE });
                    }
                },
                Ok(Cmd::Drop { x, y, payload }) => {
                    // Spike A handoff stub: Phase 3 turns this into a pick
                    // ray + actor spawn. Logging proves the coordinate path.
                    tracing::info!(
                        "inf-viewport: drop '{payload}' at viewport-local ({x:.0}, {y:.0}) px"
                    );
                }
                Ok(Cmd::SetMode(m)) => host.set_mode(m),
                Ok(Cmd::SetSnap2D(s)) => host.set_snap_2d(s),
                Ok(Cmd::SetToolMode(m)) => host.set_tool_mode(m),
                Ok(Cmd::SetSculpt(s)) => host.set_sculpt(s),
                Ok(Cmd::EmbedForeign(foreign)) => {
                    embed_foreign_window(parent_hwnd, foreign, last_rect);
                    unsafe {
                        let _ = ShowWindow(hwnd, SW_HIDE);
                    }
                    embedded = Some(HWND(foreign as *mut _));
                }
                Ok(Cmd::ReleaseForeign) => {
                    embedded = None;
                    unsafe {
                        let _ = ShowWindow(hwnd, SW_SHOWNA);
                    }
                }
                Ok(Cmd::Destroy) | Err(TryRecvError::Disconnected) => break 'outer,
                Err(TryRecvError::Empty) => break,
            }
        }
        if let Some(r) = latest_rect {
            last_rect = Some(r);
            // Move the embedded PIE window (if any) instead of our hidden child;
            // always keep our own child sized so a release restores cleanly.
            let target = embedded.unwrap_or(hwnd);
            unsafe {
                let _ = SetWindowPos(
                    target,
                    None,
                    r.x,
                    r.y,
                    r.width.max(1) as i32,
                    r.height.max(1) as i32,
                    SWP_NOACTIVATE | SWP_NOZORDER,
                );
                if embedded.is_some() {
                    let _ = SetWindowPos(
                        hwnd,
                        None,
                        r.x,
                        r.y,
                        r.width.max(1) as i32,
                        r.height.max(1) as i32,
                        SWP_NOACTIVATE | SWP_NOZORDER,
                    );
                }
            }
            host.resize(r.width.max(1), r.height.max(1));
        }

        // 2. Pump this thread's window messages.
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, Some(hwnd), 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        // 3. Camera update from accumulated input.
        let dt = last_frame.elapsed().as_secs_f32().min(0.1);
        last_frame = Instant::now();
        let input = drain_input();

        // Forward global-shortcut chords to the webview (focus handoff).
        for chord in input.chords {
            sink(ViewportEvent::Key(KeyChord { chord }));
        }

        // Active projection + surface size for this frame's camera/gizmo math.
        let two_d = host.mode == ViewportMode::TwoD;
        let (vw, vh) = host.surface_size();
        let (vwf, vhf) = (vw as f64, vh.max(1) as f64);

        // Discrete actions: gizmo mode, focus, bookmarks. Focus/recall
        // interrupt an in-flight focus animation.
        for action in input.actions {
            match action {
                Action::SetGizmo(mode) => host.set_gizmo_mode(mode),
                Action::Focus => {
                    // Frame the selection, or the world origin if none.
                    let (center, radius) =
                        host.selection_focus().unwrap_or((glam::DVec3::ZERO, 4.0));
                    if two_d {
                        // Frame the selection's XY bounds instantly.
                        camera_2d.frame(
                            DVec2::new(center.x, center.y),
                            DVec2::splat(radius),
                            vwf / vhf,
                        );
                    } else {
                        focus_goal = Some(camera.focus_goal(center, radius));
                    }
                }
                Action::StoreBookmark(n) => {
                    bookmarks.store(n, camera.pose());
                    tracing::debug!("inf-viewport: stored camera bookmark {n}");
                }
                Action::RecallBookmark(n) => {
                    if let Some(pose) = bookmarks.recall(n) {
                        camera.set_pose(pose);
                        focus_goal = None;
                    }
                }
            }
        }

        // Project the shared world and run pointer interaction against it. The
        // document is the single source of truth: a pick updates its selection,
        // a gizmo drag writes transforms back as one undo entry, and both
        // signal Ring 2 (WorldChanged) so the Outliner/Details re-sync. The
        // lock is held only for this short section, never across the render.
        let mut world_changed = false;
        if let Ok(mut doc) = scene.lock() {
            host.sync_from_doc(&doc);

            // The render view for the ACTIVE mode drives picking + gizmo rays.
            let interact_view = if two_d {
                host.view_2d(&camera_2d)
            } else {
                host.view_for(&camera)
            };

            // Sculpt mode (perspective only): plain LMB paints a terrain height
            // stroke instead of selecting/gizmo-dragging. The stroke lives on the
            // viewport thread (like a gizmo drag) and commits one undo step at
            // mouse-up (P10.2b). 2D mode keeps Select regardless of the tool.
            let sculpting = host.tool_mode() == ToolMode::Sculpt && !two_d;
            if sculpting {
                if let Some((x, y, ctrl)) = input.left_press {
                    let (px, py) = (x.max(0) as u32, y.max(0) as u32);
                    host.begin_sculpt(&mut doc, &interact_view, px, py, ctrl);
                }
                if input.left_down && host.is_sculpting() {
                    let (x, y) = input.cursor;
                    host.update_sculpt(&mut doc, &interact_view, x.max(0) as u32, y.max(0) as u32);
                }
                if input.left_release && host.is_sculpting() && host.finish_sculpt(&mut doc) {
                    world_changed = true;
                }
                // Idle hover: the brush ring follows the cursor over terrain.
                if !input.left_down && input.capture == Capture::None && input.cursor_moved {
                    let (x, y) = input.cursor;
                    if x >= 0 && y >= 0 {
                        host.update_sculpt_hover(&doc, &interact_view, x as u32, y as u32);
                    }
                }
            } else {
                // Plain LMB: a handle under the cursor begins a gizmo drag (one
                // undo transaction), otherwise it selects the picked entity.
                if let Some((x, y, ctrl)) = input.left_press {
                    let (px, py) = (x.max(0) as u32, y.max(0) as u32);
                    if host.try_begin_gizmo(&interact_view, px, py) {
                        doc.begin_transaction("Move");
                    } else {
                        match host.pick_guid(&interact_view, px, py) {
                            Some(guid) => {
                                doc.select(&[guid], ctrl);
                                world_changed = true;
                            }
                            None if !ctrl => {
                                doc.clear_selection();
                                world_changed = true;
                            }
                            None => {}
                        }
                        host.sync_from_doc(&doc); // reflect the new selection now
                    }
                }

                if input.left_down && host.is_dragging_gizmo() {
                    let (x, y) = input.cursor;
                    // 2D: translate snaps to the toolbar's grid/pixel increment (or
                    // Shift for a 1 m fallback); rotate/scale keep the Shift snaps.
                    // Perspective keeps the Shift-drag snaps (1 m / 15° / 0.1 ratio).
                    let snap = if two_d {
                        match host.gizmo_mode {
                            GizmoMode::Translate => {
                                let s = host.snap_2d_translate();
                                if s > 0.0 {
                                    s
                                } else if key_down(0x10) {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            GizmoMode::Rotate if key_down(0x10) => 15f32.to_radians(),
                            GizmoMode::Scale if key_down(0x10) => 0.1,
                            _ => 0.0,
                        }
                    } else if key_down(0x10) {
                        match host.gizmo_mode {
                            GizmoMode::Translate => 1.0,
                            GizmoMode::Rotate => 15f32.to_radians(),
                            GizmoMode::Scale => 0.1,
                        }
                    } else {
                        0.0
                    };
                    host.update_gizmo(&interact_view, x.max(0) as u32, y.max(0) as u32, snap);
                }

                if input.left_release {
                    if host.is_dragging_gizmo() {
                        for (guid, transform) in host.selected_world_transforms() {
                            doc.edit_set_transform(guid, transform);
                        }
                        doc.commit_transaction();
                        world_changed = true;
                    }
                    host.end_gizmo();
                }

                // Hover highlight when idle (not dragging, not flying).
                if input.cursor_moved && !input.left_down && input.capture == Capture::None {
                    let (x, y) = input.cursor;
                    if x >= 0 && y >= 0 {
                        host.set_hover(&interact_view, x as u32, y as u32);
                    }
                }
            } // end Select-tool interaction (else of `if sculpting`)
        }
        if world_changed {
            sink(ViewportEvent::WorldChanged);
        }

        // A live navigation gesture cancels a focus animation.
        let navigating = input.capture != Capture::None || input.wheel != 0;
        if navigating {
            focus_goal = None;
        }

        if two_d {
            // 2D navigation: any drag capture pans in the plane; the wheel
            // zooms to the cursor (exponential half-height with clamps).
            match input.capture {
                Capture::None => {
                    if input.wheel != 0 {
                        let (cx, cy) = input.cursor;
                        camera_2d.zoom_at(input.wheel, cx as f64, cy as f64, vwf, vhf);
                    }
                }
                _ => camera_2d.pan(input.dx as f64, input.dy as f64, vwf, vhf),
            }
        } else {
            match input.capture {
                Capture::Fly => {
                    let fly = FlyInput {
                        mouse_dx: input.dx,
                        mouse_dy: input.dy,
                        wheel_steps: input.wheel,
                        forward: key_down(0x57), // W
                        back: key_down(0x53),    // S
                        right: key_down(0x44),   // D
                        left: key_down(0x41),    // A
                        up: key_down(0x45),      // E
                        down: key_down(0x51),    // Q
                        boost: key_down(0x10),   // Shift
                    };
                    camera.apply_fly(&fly, dt);
                }
                Capture::Orbit | Capture::Pan | Capture::Dolly => {
                    let mode = match input.capture {
                        Capture::Orbit => NavMode::Orbit,
                        Capture::Pan => NavMode::Pan,
                        _ => NavMode::Dolly,
                    };
                    let pivot = camera.pivot(None);
                    camera.apply_navigate(
                        &NavInput {
                            mode,
                            mouse_dx: input.dx,
                            mouse_dy: input.dy,
                            wheel_steps: input.wheel,
                        },
                        pivot,
                        dt,
                    );
                }
                Capture::None => {
                    // No button held: the wheel dollies toward the look point.
                    if input.wheel != 0 {
                        let pivot = camera.pivot(None);
                        camera.dolly(input.wheel as f32 * 0.12, pivot);
                    }
                }
            }
        }

        // Advance an in-flight focus animation (perspective only).
        if let Some(goal) = focus_goal {
            if camera.advance_focus(&goal, dt) {
                focus_goal = None;
            }
        }

        // 4. Rebase the floating origin on the active eye, build the view for
        //    the current mode, and render. FIFO present blocks at vsync and
        //    paces the loop. render_frame recovers from device loss internally —
        //    an error here means even the rebuild failed (driver truly gone).
        let eye = if two_d { camera_2d.eye() } else { camera.pos };
        host.origin.maybe_rebase(eye);
        let render_view = if two_d {
            host.view_2d(&camera_2d)
        } else {
            host.view_for(&camera)
        };
        // Guard the render against a panic in engine code (P15.2). Rather than let
        // it unwind across the OS thread boundary, we catch it, write a crash
        // report, log a graceful message, and exit the render loop — the editor
        // process (and its webview) survive instead of the whole app dying.
        let render_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            host.render_frame(&render_view)
        }));
        match render_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::error!("inf-viewport: unrecoverable render failure: {e}");
                break 'outer;
            }
            Err(payload) => {
                report_viewport_panic("<viewport render thread>", &*payload);
                break 'outer;
            }
        }
    }

    tracing::info!("inf-viewport: shutting down");
}
