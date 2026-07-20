//! Win32 child-window host. A dedicated thread owns the child HWND (Win32
//! ties a window to the thread that created it), pumps its messages, and
//! drives the render loop; commands arrive over a channel.

use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, PeekMessageW, RegisterClassW, SetWindowPos,
    TranslateMessage, HWND_TOP, MSG, PM_REMOVE, SWP_NOACTIVATE, SWP_NOZORDER, WINDOW_EX_STYLE,
    WNDCLASSW, WS_CHILD, WS_CLIPSIBLINGS, WS_VISIBLE,
};

use crate::render::Renderer;
use crate::ViewportRect;

enum Cmd {
    SetRect(ViewportRect),
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

extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
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

    let hinstance = unsafe { GetModuleHandleW(None) }
        .map(|h| h.0 as isize)
        .unwrap_or_default();
    let mut renderer = match Renderer::new(hwnd.0 as isize, hinstance, 64, 64) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("inf-viewport: wgpu init failed: {e}");
            return;
        }
    };
    tracing::info!("inf-viewport: child window + wgpu surface up");

    'outer: loop {
        // 1. Apply pending commands (coalesce rect updates to the latest).
        let mut latest_rect: Option<ViewportRect> = None;
        loop {
            match rx.try_recv() {
                Ok(Cmd::SetRect(r)) => latest_rect = Some(r),
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

        // 3. Render one frame; FIFO present blocks at vsync and paces the loop.
        renderer.render();
    }

    tracing::info!("inf-viewport: shutting down");
}
