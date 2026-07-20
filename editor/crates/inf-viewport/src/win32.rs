//! Win32 child-window host. A dedicated thread owns the child HWND (Win32
//! ties a window to the thread that created it), pumps its messages, drives
//! the render loop, and runs the flycam; commands arrive over a channel.
//!
//! Input model (UE parity): RMB press captures the mouse (SetCapture + hide
//! cursor + raw WM_INPUT deltas), WASD/QE fly while captured, mouse wheel
//! scales fly speed, RMB release restores the cursor where it started.

use std::cell::RefCell;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::time::Instant;

use glam::Vec3;
use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, ReleaseCapture, SetCapture, SetFocus,
};
use windows::Win32::UI::Input::{
    GetRawInputData, RegisterRawInputDevices, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE,
    RAWINPUTDEVICE_FLAGS, RAWINPUTHEADER, RID_INPUT, RIM_TYPEMOUSE,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetCursorPos, PeekMessageW, RegisterClassW,
    SetCursorPos, SetWindowPos, ShowCursor, TranslateMessage, HWND_TOP, MSG, PM_REMOVE,
    SWP_NOACTIVATE, SWP_NOZORDER, WINDOW_EX_STYLE, WM_CAPTURECHANGED, WM_ERASEBKGND, WM_INPUT,
    WM_MOUSEWHEEL, WM_RBUTTONDOWN, WM_RBUTTONUP, WNDCLASSW, WS_CHILD, WS_CLIPSIBLINGS, WS_VISIBLE,
};

use crate::render::{Camera, Renderer};
use crate::{SurfaceTarget, ViewportRect};

enum Cmd {
    SetRect(ViewportRect),
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
/// `parent_hwnd`, brings up wgpu on it, and renders until destroyed.
pub fn spawn(parent_hwnd: isize) -> ViewportHandle {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("inf-viewport".into())
        .spawn(move || thread_main(parent_hwnd, rx))
        .expect("failed to spawn inf-viewport thread");
    ViewportHandle { tx }
}

/// Flycam input accumulated by `wnd_proc` between frames. Thread-local is
/// safe here: the wnd_proc always runs on the thread that created the window.
#[derive(Default)]
struct InputState {
    captured: bool,
    mouse_dx: f32,
    mouse_dy: f32,
    wheel_steps: i32,
    restore_cursor: POINT,
}

thread_local! {
    static INPUT: RefCell<InputState> = RefCell::new(InputState::default());
}

extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match msg {
        // The swapchain covers every pixel — skipping the GDI background
        // erase kills the white flash during splitter resizes.
        WM_ERASEBKGND => LRESULT(1),

        // IMPORTANT: Win32 calls below happen OUTSIDE the RefCell borrow —
        // SetCapture/ReleaseCapture dispatch WM_CAPTURECHANGED *synchronously*
        // back into this wnd_proc, and a nested borrow_mut is a panic (which
        // aborts the process: wnd_proc is a non-unwinding extern fn).
        WM_RBUTTONDOWN => {
            let begin = INPUT.with(|s| {
                let mut s = s.borrow_mut();
                !std::mem::replace(&mut s.captured, true)
            });
            if begin {
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
            LRESULT(0)
        }

        WM_RBUTTONUP => {
            let restore = INPUT.with(|s| {
                let mut s = s.borrow_mut();
                if std::mem::replace(&mut s.captured, false) {
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
            LRESULT(0)
        }

        // Capture can be stolen (alt-tab, other SetCapture); un-hide cleanly.
        WM_CAPTURECHANGED => {
            let was_captured =
                INPUT.with(|s| std::mem::replace(&mut s.borrow_mut().captured, false));
            if was_captured {
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
                        if s.captured {
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

fn thread_main(parent_hwnd: isize, rx: Receiver<Cmd>) {
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
    let mut renderer = match Renderer::new(target, 64, 64) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("inf-viewport: wgpu init failed: {e}");
            return;
        }
    };
    tracing::info!("inf-viewport: child window + wgpu surface up");

    let mut camera = Camera::default();
    let mut fly_speed = 4.0f32; // m/s, wheel-scaled while captured
    let mut last_frame = Instant::now();

    'outer: loop {
        // 1. Apply pending commands (coalesce rect updates to the latest).
        let mut latest_rect: Option<ViewportRect> = None;
        loop {
            match rx.try_recv() {
                Ok(Cmd::SetRect(r)) => latest_rect = Some(r),
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
            renderer.resize(r.width.max(1), r.height.max(1));
        }

        // 2. Pump this thread's window messages.
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, Some(hwnd), 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        // 3. Flycam update from accumulated input.
        let dt = last_frame.elapsed().as_secs_f32().min(0.1);
        last_frame = Instant::now();
        let (dx, dy, wheel, captured) = INPUT.with(|s| {
            let mut s = s.borrow_mut();
            let out = (s.mouse_dx, s.mouse_dy, s.wheel_steps, s.captured);
            s.mouse_dx = 0.0;
            s.mouse_dy = 0.0;
            s.wheel_steps = 0;
            out
        });
        if captured {
            if wheel != 0 {
                fly_speed = (fly_speed * 1.2f32.powi(wheel)).clamp(0.2, 250.0);
                tracing::debug!("inf-viewport: fly speed {fly_speed:.1} m/s");
            }
            const SENS: f32 = 0.0032; // rad per raw mouse count
            camera.yaw += dx * SENS;
            camera.pitch = (camera.pitch - dy * SENS).clamp(-1.55, 1.55);

            let mut mv = Vec3::ZERO;
            if key_down(0x57) {
                mv += camera.forward(); // W
            }
            if key_down(0x53) {
                mv -= camera.forward(); // S
            }
            if key_down(0x44) {
                mv += camera.right(); // D
            }
            if key_down(0x41) {
                mv -= camera.right(); // A
            }
            if key_down(0x45) {
                mv += Vec3::Y; // E
            }
            if key_down(0x51) {
                mv -= Vec3::Y; // Q
            }
            let boost = if key_down(0x10) { 4.0 } else { 1.0 }; // Shift
            if mv != Vec3::ZERO {
                camera.pos += mv.normalize() * fly_speed * boost * dt;
            }
        }

        // 4. Render one frame; FIFO present blocks at vsync and paces the loop.
        renderer.render(&camera);
    }

    tracing::info!("inf-viewport: shutting down");
}
