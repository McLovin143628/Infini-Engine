//! Spike D exit criteria, run against the real `inf-player` binary (cargo
//! provides its path via `CARGO_BIN_EXE_inf-player`): headless determinism,
//! snapshot handoff over the local channel, pause/resume, crash isolation,
//! and (Windows) the cross-process window-embedding probe.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use inf_editor_core::pie::{PieSession, SessionHealth};
use inf_runtime::pie::{EditorToPlayer, PlayerToEditor};
use inf_runtime::{CookedSnapshot, World};

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

fn spawn_session() -> PieSession {
    // 240 Hz keeps the tests fast; determinism is per-step, not per-second.
    PieSession::spawn(&player_bin(), &CookedSnapshot::demo(), 240).expect("PIE session spawns")
}

#[test]
fn headless_run_is_deterministic_and_matches_in_process() {
    let run = || {
        let output = Command::new(player_bin())
            .args(["--headless", "--run-frames", "240", "--assert-exit"])
            .output()
            .expect("player runs");
        assert!(output.status.success(), "exit: {:?}", output.status);
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        stdout
            .lines()
            .find_map(|l| l.strip_prefix("final-state-hash: ").map(str::to_owned))
            .expect("hash line present")
    };
    let first = run();
    let second = run();
    assert_eq!(first, second, "two headless runs must agree");

    // And the subprocess agrees with stepping the same world in-process.
    let mut world = World::from_snapshot(&CookedSnapshot::demo());
    for _ in 0..240 {
        world.step();
    }
    assert_eq!(first, format!("{:016x}", world.state_hash()));
}

#[test]
fn pie_streams_advancing_frames_and_stops_cleanly() {
    let session = spawn_session();

    let first = session
        .wait_for(Duration::from_secs(5), |e| {
            matches!(e, PlayerToEditor::Frame { .. })
        })
        .expect("first frame");
    let PlayerToEditor::Frame {
        frame: first_frame,
        actors: first_actors,
        ..
    } = first
    else {
        unreachable!()
    };
    assert_eq!(first_actors.len(), 3);

    let later = session
        .wait_for(
            Duration::from_secs(5),
            |e| matches!(e, PlayerToEditor::Frame { frame, .. } if *frame >= first_frame + 10),
        )
        .expect("frames keep advancing");
    let PlayerToEditor::Frame {
        actors: later_actors,
        ..
    } = later
    else {
        unreachable!()
    };
    assert_ne!(
        first_actors[0].pos, later_actors[0].pos,
        "actors must move between frames"
    );

    let status = session.stop(Duration::from_secs(5)).expect("graceful stop");
    assert!(status.success(), "player must exit 0 on Stop: {status:?}");
}

#[test]
fn pause_stops_the_frame_stream_and_resume_restarts_it() {
    let mut session = spawn_session();

    session
        .wait_for(Duration::from_secs(5), |e| {
            matches!(e, PlayerToEditor::Frame { .. })
        })
        .expect("frames flowing");

    session.send(&EditorToPlayer::Pause).unwrap();
    session
        .wait_for(Duration::from_secs(5), |e| {
            matches!(e, PlayerToEditor::Paused)
        })
        .expect("pause acknowledged");

    // Drain frames already in the pipe, then confirm silence.
    while session.next_event(Duration::from_millis(200)).is_some() {}
    assert!(
        session.next_event(Duration::from_millis(300)).is_none(),
        "no frames while paused"
    );

    session.send(&EditorToPlayer::Resume).unwrap();
    session
        .wait_for(Duration::from_secs(5), |e| {
            matches!(e, PlayerToEditor::Frame { .. })
        })
        .expect("frames resume");

    session.stop(Duration::from_secs(5)).unwrap();
}

/// The reason the subprocess model exists: a script panic kills the player,
/// the editor keeps its state and can immediately start a fresh session.
#[test]
fn player_crash_is_isolated_from_the_editor() {
    let mut session = spawn_session();
    session
        .wait_for(Duration::from_secs(5), |e| {
            matches!(e, PlayerToEditor::Frame { .. })
        })
        .expect("session running before the crash");

    session.send(&EditorToPlayer::InjectPanic).unwrap();
    let status = session
        .wait_exit(Duration::from_secs(10))
        .expect("player process must die");
    assert!(!status.success(), "a panic must not exit 0: {status:?}");
    assert_eq!(
        session.health(),
        SessionHealth::Exited {
            code: status.code()
        }
    );

    // The panic message is captured for the Output Log.
    let stderr = session.stderr_lines().join("\n");
    assert!(
        stderr.contains("deliberate PIE panic"),
        "panic text must reach the editor; got:\n{stderr}"
    );

    // "Unsaved editor state survives" in test form: this process is fine and
    // a new session starts immediately.
    let second = spawn_session();
    second
        .wait_for(Duration::from_secs(5), |e| {
            matches!(e, PlayerToEditor::Frame { .. })
        })
        .expect("fresh session after the crash");
    second.stop(Duration::from_secs(5)).unwrap();
}

/// The Spike D "embed-window experiment": prove a window owned by another
/// process can be reparented into ours (the P9 plan for putting the PIE
/// player's swapchain inside the editor viewport hole).
#[cfg(windows)]
#[test]
fn foreign_process_window_can_be_reparented() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;

    use windows::core::w;
    use windows::Win32::Foundation::HWND;
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::DefWindowProcW;
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DestroyWindow, DispatchMessageW, GetParent, GetWindowLongPtrW,
        PeekMessageW, RegisterClassW, SetParent, SetWindowLongPtrW, TranslateMessage,
        CW_USEDEFAULT, GWL_STYLE, MSG, PM_REMOVE, WINDOW_EX_STYLE, WNDCLASSW, WS_CHILD,
        WS_OVERLAPPEDWINDOW,
    };

    extern "system" fn host_proc(
        hwnd: HWND,
        msg: u32,
        wparam: windows::Win32::Foundation::WPARAM,
        lparam: windows::Win32::Foundation::LPARAM,
    ) -> windows::Win32::Foundation::LRESULT {
        unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
    }

    fn pump() {
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    // Spawn the probe and read its window handle.
    let mut child = Command::new(player_bin())
        .arg("--embed-probe")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("probe spawns");
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    let hwnd_line = lines
        .next()
        .expect("probe prints a line")
        .expect("line reads");
    let foreign_hwnd: isize = hwnd_line
        .strip_prefix("embed-probe-hwnd: ")
        .expect("handshake line")
        .parse()
        .expect("hwnd parses");
    let foreign = HWND(foreign_hwnd as *mut _);

    unsafe {
        // A hidden host window in *this* process to adopt the probe window.
        let hinstance = GetModuleHandleW(None).unwrap();
        let class = WNDCLASSW {
            lpfnWndProc: Some(host_proc),
            hInstance: hinstance.into(),
            lpszClassName: w!("InfinityEmbedHostTest"),
            ..Default::default()
        };
        RegisterClassW(&class);
        let host = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            w!("InfinityEmbedHostTest"),
            w!("embed host"),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            320,
            240,
            None,
            None,
            Some(hinstance.into()),
            None,
        )
        .expect("host window");
        pump();

        // MSDN order: switch the foreign window to WS_CHILD, then reparent.
        // (SetParent on a top-level window returns the desktop HWND as the
        // previous parent — only the Result matters.)
        let style = GetWindowLongPtrW(foreign, GWL_STYLE);
        let child_style = (style & !(WS_OVERLAPPEDWINDOW.0 as isize)) | (WS_CHILD.0 as isize);
        SetWindowLongPtrW(foreign, GWL_STYLE, child_style);
        let reparent = SetParent(foreign, Some(host));
        pump();
        let parent_after = GetParent(foreign);
        let style_after = GetWindowLongPtrW(foreign, GWL_STYLE);

        // Cleanup BEFORE asserting so a failure can't orphan the probe:
        // closing stdin ends its message pump; it exits 0. Crucially we must
        // keep pumping while waiting — the embedded child's DestroyWindow
        // sends WM_PARENTNOTIFY synchronously to our host window, so a
        // non-pumping blocking wait here deadlocks both processes. (Spike
        // finding: the editor's embedding thread must never block without
        // pumping while it hosts a foreign window.)
        drop(child.stdin.take());
        let mut status = None;
        for _ in 0..500 {
            pump();
            if let Ok(Some(s)) = child.try_wait() {
                status = Some(s);
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let status = match status {
            Some(s) => s,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = DestroyWindow(host);
                panic!("probe did not exit after stdin closed");
            }
        };
        pump();
        let _ = DestroyWindow(host);

        reparent.expect("cross-process SetParent");
        let parent = parent_after.expect("foreign window has a parent now");
        assert_eq!(parent.0, host.0, "probe window reparented into our tree");
        assert_ne!(style_after & (WS_CHILD.0 as isize), 0, "WS_CHILD applied");
        assert!(status.success(), "probe exit: {status:?}");
    }
}
