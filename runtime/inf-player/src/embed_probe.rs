//! `--embed-probe`: the Spike D "embed-window experiment".
//!
//! Creates a bare (hidden) top-level Win32 window, prints
//! `embed-probe-hwnd: <n>` on stdout, and pumps messages until stdin
//! closes. A test in another process then reparents this window into its
//! own HWND tree with `SetParent` — proving the P9 plan of embedding the
//! PIE player's window into the editor's viewport hole (Spike A machinery)
//! works across process boundaries on Windows.

use std::process::ExitCode;

use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GetMessageW, PostMessageW,
    PostQuitMessage, RegisterClassW, TranslateMessage, CW_USEDEFAULT, MSG, WINDOW_EX_STYLE,
    WM_CLOSE, WM_DESTROY, WNDCLASSW, WS_OVERLAPPEDWINDOW,
};

extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // Per the Spike A reentrancy rule: no shared mutable state in here.
    unsafe {
        match msg {
            WM_CLOSE => {
                let _ = DestroyWindow(hwnd);
                LRESULT(0)
            }
            WM_DESTROY => {
                PostQuitMessage(0);
                LRESULT(0)
            }
            _ => DefWindowProcW(hwnd, msg, wparam, lparam),
        }
    }
}

pub fn run() -> ExitCode {
    let hwnd = unsafe {
        let hinstance = match GetModuleHandleW(None) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("inf-player: GetModuleHandleW failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        let class = WNDCLASSW {
            lpfnWndProc: Some(wnd_proc),
            hInstance: hinstance.into(),
            lpszClassName: w!("InfinityEmbedProbe"),
            ..Default::default()
        };
        if RegisterClassW(&class) == 0 {
            eprintln!("inf-player: RegisterClassW failed");
            return ExitCode::FAILURE;
        }
        match CreateWindowExW(
            WINDOW_EX_STYLE(0),
            w!("InfinityEmbedProbe"),
            w!("Infini Engine embed probe"),
            WS_OVERLAPPEDWINDOW, // deliberately NOT WS_VISIBLE — stays hidden
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            320,
            240,
            None,
            None,
            Some(hinstance.into()),
            None,
        ) {
            Ok(hwnd) => hwnd,
            Err(e) => {
                eprintln!("inf-player: CreateWindowExW failed: {e}");
                return ExitCode::FAILURE;
            }
        }
    };

    // The handshake line the embedding test reads.
    println!("embed-probe-hwnd: {}", hwnd.0 as isize);

    // Close the window (ending the pump) when the parent drops our stdin.
    let hwnd_value = hwnd.0 as isize;
    std::thread::spawn(move || {
        use std::io::Read;
        let mut sink = Vec::new();
        let _ = std::io::stdin().lock().read_to_end(&mut sink);
        unsafe {
            let _ = PostMessageW(
                Some(HWND(hwnd_value as *mut _)),
                WM_CLOSE,
                WPARAM(0),
                LPARAM(0),
            );
        }
    });

    unsafe {
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    eprintln!("inf-player: embed probe window closed");
    ExitCode::SUCCESS
}
