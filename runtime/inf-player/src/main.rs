//! Infinity Engine standalone player — binary entry.
//!
//! Modes (see [`inf_player::args`]):
//!
//! - **windowed** (default, or `--demo` / `--level`) — open a winit window and
//!   play; the fixed-step gameplay + interpolated rendering live in the library
//!   ([`inf_player::run`]).
//! - **headless** (`--headless --run-frames N [--assert-exit]`) — no window/GPU;
//!   run the sim N steps and print the determinism hash. The CI smoke path.
//! - **pie** (`--pie [--tick-hz N]`) — play-in-editor subprocess: speak the
//!   length-prefixed bincode protocol from `inf_runtime::pie` on stdin/stdout
//!   (stdout is protocol-only; logs go to stderr). Unchanged from Spike D; P9.4
//!   builds the editor side out on top of it.
//! - **embed-probe** (`--embed-probe`, Windows) — the Spike D cross-process
//!   window-embedding probe (proves the P9.4 "PIE window in the viewport hole"
//!   plan). Unchanged.
//!
//! `--pie` and `--embed-probe` are handled here (not via [`inf_player::run`])
//! because they own process stdio / native windows: installing the tracing
//! subscriber that tees logs to stdout would corrupt the PIE protocol stream.

use std::process::ExitCode;

use inf_player::args::{Args, Mode};
use inf_runtime::pie::{read_msg, write_msg, EditorToPlayer, PlayerToEditor, PIE_PROTOCOL_VERSION};
use inf_runtime::World;

#[cfg(windows)]
mod embed_probe;

fn main() -> ExitCode {
    let args = match Args::from_env() {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("inf-player: {msg}");
            return ExitCode::FAILURE;
        }
    };

    match args.mode {
        Mode::EmbedProbe => {
            #[cfg(windows)]
            {
                embed_probe::run()
            }
            #[cfg(not(windows))]
            {
                eprintln!("inf-player: --embed-probe is Windows-only");
                ExitCode::FAILURE
            }
        }
        Mode::Pie => run_pie(args.tick_hz),
        Mode::Windowed | Mode::Headless => inf_player::run(args),
    }
}

/// The PIE loop (Spike D): a reader thread turns stdin frames into channel
/// messages; the main loop applies control, steps the toy world at `tick_hz`, and
/// streams `Frame` reports. Stdout carries protocol frames only.
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
            panic!("deliberate PIE panic (injected by editor)");
        }
    };
    if write_msg(stdout, &reply).is_err() {
        return Some(ExitCode::SUCCESS);
    }
    None
}
