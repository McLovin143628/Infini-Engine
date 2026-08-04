//! Spike D exit criteria, run against the real `inf-player` binary (cargo
//! provides its path via `CARGO_BIN_EXE_inf-player`): headless determinism,
//! snapshot handoff over the local channel, pause/resume, crash isolation,
//! and (Windows) the cross-process window-embedding probe.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use inf_editor_core::pie::{PieSession, SessionHealth};
use inf_runtime::pie::{EditorToPlayer, PlayerToEditor};
use inf_runtime::CookedSnapshot;

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

fn spawn_session() -> PieSession {
    // 240 Hz keeps the tests fast; determinism is per-step, not per-second.
    PieSession::spawn(&player_bin(), &CookedSnapshot::demo(), 240).expect("PIE session spawns")
}

/// The P9.3 headless smoke path now runs the real demo world (ECS + physics +
/// the Coyote blueprint) via `runtime_sim`, folding the same xxh3 trace the
/// `inf_runtime::replay` harness uses. Two subprocess runs agree, and the
/// subprocess agrees with the in-process demo trace.
#[test]
fn headless_demo_run_is_deterministic_and_matches_in_process() {
    let run = || {
        let output = Command::new(player_bin())
            .args([
                "--headless",
                "--demo",
                "--run-frames",
                "240",
                "--assert-exit",
            ])
            .output()
            .expect("player runs");
        assert!(
            output.status.success(),
            "exit: {:?}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        stdout
            .lines()
            .find_map(|l| l.strip_prefix("final-state-hash: ").map(str::to_owned))
            .expect("hash line present")
    };
    let first = run();
    let second = run();
    assert_eq!(first, second, "two headless runs must agree");

    // The subprocess agrees with the in-process demo trace (same 128-bit fold).
    assert_eq!(first, format!("{:032x}", inf_player::demo_trace(240)));
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
    //
    // `wait_exit` above drained stderr to EOF before returning, which is what
    // makes this deterministic: process exit and pipe EOF are different events,
    // and asserting on the capture after only the first is a race that loses
    // rarely and on a loaded machine — the worst failure rate a test can have.
    assert!(
        session.stderr_complete(),
        "the player's stderr never reached EOF, so this assertion would be \
         comparing against a truncated capture"
    );
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

// ── P9.4: real-content PIE (the PIE == shipping gate) ────────────────────────

use inf_runtime::pie::ScenePayload;

/// The live platformer scene + Coyote class as a headless (step-driven) PIE
/// payload — exactly what the editor streams over the channel.
fn platformer_payload() -> ScenePayload {
    let doc = inf_editor_core::samples::platformer_scene();
    inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| {
            (guid == inf_editor_core::samples::COYOTE_ASSET_GUID)
                .then(inf_editor_core::samples::coyote_class)
        },
        |_guid| None, // no PCG graphs in the platformer scene
        |_guid| None, // no animation assets in the platformer scene
        |_guid| None, // no biome sets in the platformer scene
        |_guid| None, // no voxel volumes in the platformer scene
        |_guid| None, // no streamed terrains in the platformer scene
        0,            // tick-hz 0: no per-frame sleep (step-driven determinism)
        false,
    )
    .expect("build scene payload")
}

/// **PIE == shipping.** A player fed the streamed live scene builds its world
/// through the *same* `InfSceneWorldBuilder` the cooked-pack boot uses, so its
/// per-step determinism trace is byte-identical to the in-process shipping
/// trace for the same content.
#[test]
fn pie_scene_trace_matches_shipping() {
    let payload = platformer_payload();
    const N: u32 = 120;

    let mut session = PieSession::spawn_scene(&player_bin(), &payload).expect("scene session");
    session.step(N).expect("step N");

    let mut got = Vec::with_capacity(N as usize);
    for _ in 0..N {
        let ev = session
            .wait_for(Duration::from_secs(10), |e| {
                matches!(e, PlayerToEditor::Frame { .. })
            })
            .expect("a frame per step");
        if let PlayerToEditor::Frame { state_hash, .. } = ev {
            got.push(state_hash);
        }
    }

    // The in-process reference is the shipping/pack-path build of the same
    // payload (same builder + RuntimeSim). Byte-identical == PIE preview can
    // never diverge from the shipped game.
    let want = inf_player::scene_trace(&payload, N as u64).expect("shipping trace");
    assert_eq!(
        got, want,
        "streamed PIE trace must equal the shipping trace"
    );
    // The trace is non-trivial: gravity + input-free physics advances state.
    assert!(got.windows(2).any(|w| w[0] != w[1]), "state must evolve");

    session.stop(Duration::from_secs(5)).expect("graceful stop");
}

/// Step control on real content: a step-driven session emits exactly one frame
/// per requested step and none unbidden, then stops cleanly.
#[test]
fn pie_scene_step_control_and_clean_stop() {
    let payload = platformer_payload();
    let mut session = PieSession::spawn_scene(&player_bin(), &payload).expect("scene session");

    // Nothing streams before a Step (real headless PIE is step-driven).
    assert!(
        session.next_event(Duration::from_millis(300)).is_none(),
        "no frames until stepped"
    );

    session.step(5).unwrap();
    for _ in 0..5 {
        session
            .wait_for(Duration::from_secs(5), |e| {
                matches!(e, PlayerToEditor::Frame { .. })
            })
            .expect("stepped frame");
    }
    // Exactly five frames — a trailing `State` ack is fine, but no sixth frame.
    while let Some(ev) = session.next_event(Duration::from_millis(300)) {
        assert!(
            !matches!(ev, PlayerToEditor::Frame { .. }),
            "no extra frames after the 5 stepped"
        );
    }

    let status = session.stop(Duration::from_secs(5)).expect("graceful stop");
    assert!(status.success(), "clean exit on Stop: {status:?}");
}

/// A script panic in a **real-content** session kills only the player; the
/// editor-side session observes the nonzero exit and captures the panic text
/// (the crash-isolation guarantee, exercised through the real world build).
#[test]
fn pie_scene_crash_is_isolated() {
    let payload = platformer_payload();
    let mut session = PieSession::spawn_scene(&player_bin(), &payload).expect("scene session");
    session.step(1).unwrap();
    session
        .wait_for(Duration::from_secs(5), |e| {
            matches!(e, PlayerToEditor::Frame { .. })
        })
        .expect("running before the crash");

    session.send(&EditorToPlayer::InjectPanic).unwrap();
    let status = session
        .wait_exit(Duration::from_secs(10))
        .expect("player must die");
    assert!(!status.success(), "a panic must not exit 0: {status:?}");
    assert_eq!(
        session.health(),
        SessionHealth::Exited {
            code: status.code()
        }
    );
    assert!(
        session
            .stderr_lines()
            .join("\n")
            .contains("deliberate PIE panic"),
        "panic text captured for the Output Log"
    );

    // The editor is unaffected: a fresh session starts immediately.
    let mut second = PieSession::spawn_scene(&player_bin(), &payload).expect("fresh session");
    second.step(1).unwrap();
    second
        .wait_for(Duration::from_secs(5), |e| {
            matches!(e, PlayerToEditor::Frame { .. })
        })
        .expect("fresh session runs");
    second.stop(Duration::from_secs(5)).unwrap();
}

/// A stopped session leaves no zombie: `stop` reaps the child, and even an
/// abrupt drop (no Stop) kills + reaps it via `Drop`.
#[test]
fn pie_scene_stop_leaves_no_zombie() {
    let payload = platformer_payload();

    // Graceful stop reaps the child (a real exit status, not a leak).
    let session = PieSession::spawn_scene(&player_bin(), &payload).expect("session");
    let status = session.stop(Duration::from_secs(5)).expect("stop reaps");
    assert!(status.success());

    // A dropped session (no Stop) must not leave the player running: Drop kills
    // + waits, so the handle teardown returns promptly.
    let mut dropped = PieSession::spawn_scene(&player_bin(), &payload).expect("session");
    dropped.step(1).unwrap();
    let start = std::time::Instant::now();
    drop(dropped);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "Drop must kill + reap the child promptly (no zombie / no hang)"
    );
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
