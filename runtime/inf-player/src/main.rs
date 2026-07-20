//! Infinity Engine standalone player.
//!
//! Three modes (Spike D):
//!
//! - `--headless --run-frames N [--assert-exit]` — run the demo snapshot N
//!   fixed steps with no window/pacing and print the determinism hash. The
//!   CI smoke path (`inf-player --headless --run-frames 300 --assert-exit`).
//! - `--pie [--tick-hz N]` — play-in-editor subprocess: speak the
//!   length-prefixed bincode protocol from `inf_runtime::pie` on
//!   stdin/stdout (stdout is protocol-only; logs go to stderr).
//! - `--embed-probe` *(Windows)* — create a bare Win32 window, print its
//!   HWND, and pump messages until stdin closes. Exists so tests can prove
//!   cross-process `SetParent` embedding (the P9 "PIE window in the
//!   viewport hole" plan) actually works.

use std::process::ExitCode;

use inf_runtime::pie::{read_msg, write_msg, EditorToPlayer, PlayerToEditor, PIE_PROTOCOL_VERSION};
use inf_runtime::{CookedSnapshot, World};

#[cfg(windows)]
mod embed_probe;

struct Args {
    headless: bool,
    run_frames: u64,
    pie: bool,
    tick_hz: u32,
    embed_probe: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        headless: false,
        run_frames: 0,
        pie: false,
        tick_hz: inf_runtime::TICK_HZ,
        embed_probe: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--headless" => args.headless = true,
            "--pie" => args.pie = true,
            "--embed-probe" => args.embed_probe = true,
            // Accepted for CI-invocation compatibility; exit codes already
            // reflect success/failure.
            "--assert-exit" => {}
            "--run-frames" => {
                let value = iter.next().ok_or("--run-frames needs a value")?;
                args.run_frames = value
                    .parse()
                    .map_err(|_| format!("bad --run-frames value '{value}'"))?;
            }
            "--tick-hz" => {
                let value = iter.next().ok_or("--tick-hz needs a value")?;
                args.tick_hz = value
                    .parse()
                    .map_err(|_| format!("bad --tick-hz value '{value}'"))?;
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(args)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("inf-player: {msg}");
            return ExitCode::FAILURE;
        }
    };
    if args.embed_probe {
        #[cfg(windows)]
        return embed_probe::run();
        #[cfg(not(windows))]
        {
            eprintln!("inf-player: --embed-probe is Windows-only");
            return ExitCode::FAILURE;
        }
    }
    if args.pie {
        return run_pie(args.tick_hz);
    }
    if args.headless {
        return run_headless(args.run_frames);
    }
    println!(
        "inf-player {} (windowed mode lands in Phase 9)",
        env!("CARGO_PKG_VERSION")
    );
    ExitCode::SUCCESS
}

fn run_headless(frames: u64) -> ExitCode {
    let mut world = World::from_snapshot(&CookedSnapshot::demo());
    for _ in 0..frames {
        world.step();
    }
    println!("ran {frames} frames");
    println!("final-state-hash: {:016x}", world.state_hash());
    ExitCode::SUCCESS
}

/// The PIE loop: a reader thread turns stdin frames into channel messages;
/// the main loop applies control, steps the world at `tick_hz`, and streams
/// `Frame` reports. Stdout carries protocol frames only.
fn run_pie(tick_hz: u32) -> ExitCode {
    use std::sync::mpsc;
    use std::time::Duration;

    let (tx, rx) = mpsc::channel::<EditorToPlayer>();
    std::thread::spawn(move || {
        let mut stdin = std::io::stdin().lock();
        while let Ok(msg) = read_msg::<EditorToPlayer>(&mut stdin) {
            if tx.send(msg).is_err() {
                break;
            }
        }
        // EOF or error: dropping the sender tells the main loop the editor
        // is gone.
    });

    let mut stdout = std::io::stdout().lock();
    if write_msg(
        &mut stdout,
        &PlayerToEditor::Ready {
            protocol: PIE_PROTOCOL_VERSION,
        },
    )
    .is_err()
    {
        return ExitCode::FAILURE;
    }
    eprintln!("inf-player: PIE session ready (tick-hz {tick_hz})");

    let mut world: Option<World> = None;
    let mut paused = false;
    let tick_duration = if tick_hz == 0 {
        Duration::ZERO
    } else {
        Duration::from_secs_f64(1.0 / tick_hz as f64)
    };

    loop {
        // Apply all pending control messages first.
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    if let Some(code) = handle_msg(msg, &mut world, &mut paused, &mut stdout) {
                        return code;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected {
            eprintln!("inf-player: editor closed the channel; exiting");
            return ExitCode::SUCCESS;
        }

        if let Some(world) = world.as_mut().filter(|_| !paused) {
            world.step();
            let frame = PlayerToEditor::Frame {
                frame: world.frame,
                state_hash: world.state_hash(),
                actors: world.actor_states(),
            };
            if write_msg(&mut stdout, &frame).is_err() {
                eprintln!("inf-player: editor closed stdout; exiting");
                return ExitCode::SUCCESS;
            }
            if !tick_duration.is_zero() {
                std::thread::sleep(tick_duration);
            }
        } else {
            // Idle (no level yet, or paused): block briefly on control.
            match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(msg) => {
                    if let Some(code) = handle_msg(msg, &mut world, &mut paused, &mut stdout) {
                        return code;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    eprintln!("inf-player: editor closed the channel; exiting");
                    return ExitCode::SUCCESS;
                }
            }
        }
    }
}

/// Apply one control message. `Some(code)` means "exit now with this code".
fn handle_msg(
    msg: EditorToPlayer,
    world: &mut Option<World>,
    paused: &mut bool,
    stdout: &mut impl std::io::Write,
) -> Option<ExitCode> {
    let reply = match msg {
        EditorToPlayer::Load(snapshot) => {
            eprintln!(
                "inf-player: loading level '{}' ({} actors)",
                snapshot.level,
                snapshot.actors.len()
            );
            let loaded = PlayerToEditor::Loaded {
                level: snapshot.level.clone(),
                actor_count: snapshot.actors.len(),
            };
            *world = Some(World::from_snapshot(&snapshot));
            loaded
        }
        EditorToPlayer::Pause => {
            *paused = true;
            PlayerToEditor::Paused
        }
        EditorToPlayer::Resume => {
            *paused = false;
            PlayerToEditor::Resumed
        }
        EditorToPlayer::Stop => {
            let _ = write_msg(stdout, &PlayerToEditor::Stopped);
            return Some(ExitCode::SUCCESS);
        }
        EditorToPlayer::InjectPanic => {
            // The crash-isolation drill: an uncontained "script" panic. The
            // process dies; the editor must not.
            panic!("deliberate PIE panic (injected by editor)");
        }
    };
    if write_msg(stdout, &reply).is_err() {
        return Some(ExitCode::SUCCESS);
    }
    None
}
