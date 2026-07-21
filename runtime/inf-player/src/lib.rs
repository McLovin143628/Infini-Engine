//! Infinity Engine standalone player (P9.3) — library entry.
//!
//! The player runs a cooked game: it opens a window and drives a fixed-step
//! [`RuntimeSim`](runtime_sim::RuntimeSim) (Blueprints + 2D physics) with
//! interpolated rendering, or — for CI — runs the same simulation **headless**
//! and prints a determinism hash. It also doubles as the play-in-editor
//! subprocess (the `--pie` / `--embed-probe` modes, owned by the binary entry in
//! `main.rs`).
//!
//! # Coordinate-free bring-up (the load-bearing scope decision)
//!
//! The runtime `.inf_lvl` reader is a **concurrent** task (P9.2, `inf-scene`), so
//! the player cannot yet decode a real level. Rather than block on it, everything
//! else is built for real — window, wgpu renderer + its own ECS→scene projection,
//! fixed-step loop, input mapping, and the actor-ticking sim — and the runtime is
//! proven against a **programmatic** platformer world ([`demo`], `--demo`) that
//! mirrors `samples/platformer-2d` and runs the sample's Coyote blueprint. The
//! level path is fully wired through the [`LevelSource`](level::LevelSource) /
//! [`WorldBuilder`](level::WorldBuilder) seams with a stub decoder; when P9.2
//! lands it is a one-call swap (see [`level`]). This kept P9.3 fully unblocked.

pub mod args;
pub mod demo;
pub mod input;
pub mod level;
pub mod log;
pub mod render;
pub mod runtime_sim;
pub mod window;

use std::panic::AssertUnwindSafe;
use std::process::ExitCode;

use xxhash_rust::xxh3::Xxh3;

use args::{Args, Mode, WorldChoice};
use level::{BuiltWorld, DevDirLevelSource, StubWorldBuilder};
use runtime_sim::{RuntimeInput, RuntimeSim};

/// Dispatch the player for `args` (windowed / headless). `--pie` and
/// `--embed-probe` are handled by the binary entry (`main.rs`) because they own
/// process stdio / native windows and must not install the log/crash subscriber
/// that would corrupt the PIE stdout protocol.
pub fn run(args: Args) -> ExitCode {
    log::init(args.log_file.clone(), args.crash_file.clone());
    match args.mode {
        Mode::Headless => run_headless(&args),
        Mode::Windowed => run_windowed(&args),
        Mode::Pie | Mode::EmbedProbe => {
            eprintln!("inf-player: {:?} is handled by the binary entry", args.mode);
            ExitCode::FAILURE
        }
    }
}

/// Build the world the player runs: the programmatic demo, or a level loaded
/// through the byte/decode seams. **The single P9.2 wiring point** is the
/// [`StubWorldBuilder`] below — swap it for the inf-scene-backed builder.
fn build_world(args: &Args) -> Result<BuiltWorld, String> {
    match &args.world {
        WorldChoice::Demo => Ok(demo::build()),
        WorldChoice::Level(path) => {
            let source = DevDirLevelSource::new(path);
            level::load(&source, &StubWorldBuilder)
        }
    }
}

/// Headless CI path: no window/GPU. Run `--run-frames N` fixed steps of the world
/// with no input, folding the same xxh3 trace `inf_runtime::replay` uses, and
/// print `final-state-hash`. Exit 0 on clean completion; a panic (a gameplay
/// script blowing up, or the `--panic-after` test hook) is caught → nonzero exit,
/// with the crash report already written to stderr + the crash file by the panic
/// hook.
pub fn run_headless(args: &Args) -> ExitCode {
    let built = match build_world(args) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("inf-player: {e}");
            return ExitCode::FAILURE;
        }
    };
    let label = built.label.clone();
    let frames = args.run_frames;
    let panic_after = args.panic_after;

    tracing::info!("inf-player: headless run of '{label}' for {frames} frame(s)");

    let outcome =
        std::panic::catch_unwind(AssertUnwindSafe(|| fold_trace(built, frames, panic_after)));

    match outcome {
        Ok(hash) => {
            println!("ran {frames} frames ({label})");
            println!("final-state-hash: {hash:032x}");
            ExitCode::SUCCESS
        }
        Err(_) => {
            // The panic hook (log.rs) already emitted the crash report.
            eprintln!("inf-player: run aborted by panic");
            ExitCode::FAILURE
        }
    }
}

/// Windowed path: open a window and play. Human-verified (needs a GPU + display).
pub fn run_windowed(args: &Args) -> ExitCode {
    let built = match build_world(args) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("inf-player: {e}");
            return ExitCode::FAILURE;
        }
    };
    let map = match &args.world {
        WorldChoice::Level(path) => input::load_map_beside(path),
        WorldChoice::Demo => input::default_map(),
    };
    let title = format!("Infinity Engine — {}", built.label);
    let sim = RuntimeSim::new(built.world, built.actors, built.gravity, built.hz);
    match window::run(title, args.width, args.height, sim, map) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("inf-player: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Run `frames` fixed steps of `built` (no input) and fold every step's
/// `Guid`-sorted state into a 128-bit xxh3 trace — the determinism fingerprint.
/// Panics deliberately at frame `panic_after` when set (the crash-path fixture).
pub fn fold_trace(built: BuiltWorld, frames: u64, panic_after: Option<u64>) -> u128 {
    let mut sim = RuntimeSim::new(built.world, built.actors, built.gravity, built.hz);
    let mut hasher = Xxh3::new();
    for step in 0..frames {
        if panic_after == Some(step) {
            panic!("deliberate headless panic at frame {step} (--panic-after)");
        }
        sim.step_once(RuntimeInput::default());
        hasher.update(&sim.state_bytes());
    }
    hasher.digest128()
}

/// Convenience for tests: the demo world's determinism trace over `frames`.
pub fn demo_trace(frames: u64) -> u128 {
    fold_trace(demo::build(), frames, None)
}
