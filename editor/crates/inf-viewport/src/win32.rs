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
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetCursorPos, PeekMessageW, RegisterClassW,
    SetCursorPos, SetWindowPos, ShowCursor, ShowWindow, TranslateMessage, HWND_TOP, MSG, PM_REMOVE,
    SWP_NOACTIVATE, SWP_NOZORDER, SW_HIDE, SW_SHOWNA, WINDOW_EX_STYLE, WM_CAPTURECHANGED,
    WM_ERASEBKGND, WM_INPUT, WM_KEYDOWN, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN,
    WM_MBUTTONUP, WM_MOUSEWHEEL, WM_RBUTTONDOWN, WM_RBUTTONUP, WM_SYSKEYDOWN, WNDCLASSW, WS_CHILD,
    WS_CLIPSIBLINGS, WS_VISIBLE,
};

use crate::camera::{Bookmarks, EditorCamera, FlyInput, NavInput, NavMode};
use crate::host::EngineHost;
use crate::{KeyChord, SurfaceTarget, ViewportEvent, ViewportEventSink, ViewportRect};

enum Cmd {
    SetRect(ViewportRect),
    SetVisible(bool),
    Drop { x: f32, y: f32, payload: String },
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

    /// Tear down the viewport thread and its window.
    pub fn destroy(&self) {
        let _ = self.tx.send(Cmd::Destroy);
    }
}

/// Spawn the viewport thread: creates a `WS_CHILD` window parented to
/// `parent_hwnd`, brings up the engine on it, and renders until destroyed.
/// `sink` receives events the viewport surfaces back (forwarded key chords).
pub fn spawn(parent_hwnd: isize, sink: ViewportEventSink) -> ViewportHandle {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("inf-viewport".into())
        .spawn(move || thread_main(parent_hwnd, rx, sink))
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

/// End the capture if `kind` owns it, restoring the cursor.
fn end_capture(kind: Capture) {
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

        // Alt+LMB orbits. Plain LMB is select/gizmo (P2.4) — no camera capture.
        WM_LBUTTONDOWN => {
            if modifier(VK_MENU) {
                begin_capture(hwnd, Capture::Orbit);
            }
            LRESULT(0)
        }
        WM_LBUTTONUP => {
            end_capture(Capture::Orbit);
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

        // Capture can be stolen (alt-tab, other SetCapture); un-hide cleanly.
        WM_CAPTURECHANGED => {
            let was = INPUT.with(|s| {
                let mut s = s.borrow_mut();
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
        }
    })
}

fn thread_main(parent_hwnd: isize, rx: Receiver<Cmd>, sink: ViewportEventSink) {
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
    let mut host = match EngineHost::new(target, 64, 64) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!("inf-viewport: engine init failed: {e}");
            return;
        }
    };
    tracing::info!("inf-viewport: child window + engine renderer up");

    let mut camera = EditorCamera::default();
    let mut bookmarks = Bookmarks::default();
    // Active smooth-focus goal, cleared once the camera settles.
    let mut focus_goal = None;
    let mut last_frame = Instant::now();

    'outer: loop {
        // 1. Apply pending commands (coalesce rect updates to the latest).
        let mut latest_rect: Option<ViewportRect> = None;
        loop {
            match rx.try_recv() {
                Ok(Cmd::SetRect(r)) => latest_rect = Some(r),
                Ok(Cmd::SetVisible(v)) => unsafe {
                    // SW_SHOWNA: show without stealing focus from the webview.
                    let _ = ShowWindow(hwnd, if v { SW_SHOWNA } else { SW_HIDE });
                },
                Ok(Cmd::Drop { x, y, payload }) => {
                    // Spike A handoff stub: Phase 3 turns this into a pick
                    // ray + actor spawn. Logging proves the coordinate path.
                    tracing::info!(
                        "inf-viewport: drop '{payload}' at viewport-local ({x:.0}, {y:.0}) px"
                    );
                }
                Ok(Cmd::Destroy) | Err(TryRecvError::Disconnected) => break 'outer,
                Err(TryRecvError::Empty) => break,
            }
        }
        if let Some(r) = latest_rect {
            unsafe {
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

        // Discrete actions: focus + bookmarks. Any of these interrupts an
        // in-flight focus animation.
        for action in input.actions {
            match action {
                Action::Focus => {
                    // Selection lands in P2.4; focus the world origin for now.
                    focus_goal = Some(camera.focus_goal(glam::DVec3::ZERO, 4.0));
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

        // A live navigation gesture cancels a focus animation.
        let navigating = input.capture != Capture::None || input.wheel != 0;
        if navigating {
            focus_goal = None;
        }

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

        // Advance an in-flight focus animation.
        if let Some(goal) = focus_goal {
            if camera.advance_focus(&goal, dt) {
                focus_goal = None;
            }
        }

        // 4. Render one frame; FIFO present blocks at vsync and paces the
        //    loop. render_frame recovers from device loss internally — an
        //    error here means even the rebuild failed (driver truly gone).
        if let Err(e) = host.render_frame(&camera) {
            tracing::error!("inf-viewport: unrecoverable render failure: {e}");
            break 'outer;
        }
    }

    tracing::info!("inf-viewport: shutting down");
}
