//! Infinity Engine standalone player (P9.3) — library entry.
//!
//! The player runs a cooked game: it opens a window and drives a fixed-step
//! [`RuntimeSim`](runtime_sim::RuntimeSim) (Blueprints + 2D physics) with
//! interpolated rendering, or — for CI — runs the same simulation **headless**
//! and prints a determinism hash. It also doubles as the play-in-editor
//! subprocess (the `--pie` / `--embed-probe` modes, owned by the binary entry in
//! `main.rs`).
//!
//! # Worlds the player can run (P9.5)
//!
//! * `--demo` — the programmatic platformer ([`demo`]) that mirrors
//!   `samples/platformer-2d` and runs the sample's Coyote blueprint end to end
//!   (the runnable-gameplay proof, since physics/actor-binding aren't yet
//!   persisted in `.inf_lvl` — see [`level`]).
//! * `--level <path>` — a loose `.inf_lvl` decoded by the `inf-scene` reader.
//! * `--pack <dir-or-pack>` — a cooked `content.inf_pack` (+ `manifest.toml`).
//! * *no flag* — the pack named by a `player.toml` beside the executable, so a
//!   double-clicked **exported** game boots its own content ([`config`]).
//!
//! Levels/packs decode through [`level::InfSceneWorldBuilder`] into a real
//! [`EcsWorld`](inf_ecs::EcsWorld); the window, wgpu renderer, fixed-step loop,
//! input mapping, and actor-ticking sim are all real. The headless path folds the
//! same xxh3 determinism trace `inf_runtime::replay` uses — and a cooked pack runs
//! byte-identically to its dev-dir source (the cooked-==-uncooked gate).

pub mod args;
pub mod config;
pub mod demo;
pub mod input;
pub mod level;
pub mod log;
pub mod render;
pub mod runtime_sim;
pub mod window;

use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::process::ExitCode;

use xxhash_rust::xxh3::Xxh3;

use args::{Args, Mode, WorldChoice};
use level::{BuiltWorld, DevDirLevelSource, InfSceneWorldBuilder, PackLevelSource};
use runtime_sim::{RuntimeInput, RuntimeSim};

/// Dispatch the player for `args` (windowed / headless). `--pie` and
/// `--embed-probe` are handled by the binary entry (`main.rs`) because they own
/// process stdio / native windows and must not install the log/crash subscriber
/// that would corrupt the PIE stdout protocol.
pub fn run(mut args: Args) -> ExitCode {
    log::init(args.log_file.clone(), args.crash_file.clone());
    apply_boot_config(&mut args);
    match args.mode {
        Mode::Headless => run_headless(&args),
        Mode::Windowed => run_windowed(&args),
        Mode::Pie | Mode::EmbedProbe => {
            eprintln!("inf-player: {:?} is handled by the binary entry", args.mode);
            ExitCode::FAILURE
        }
    }
}

/// When no world was chosen on the command line, boot the pack named by a
/// `player.toml` beside the executable (the exported-game path). A no-op when a
/// world was given explicitly or no config is present.
fn apply_boot_config(args: &mut Args) {
    if args.world_explicit {
        return;
    }
    let Some(cfg) = config::load_beside_exe() else {
        return;
    };
    tracing::info!(
        "inf-player: booting from player.toml → {}",
        cfg.pack.display()
    );
    args.world = WorldChoice::Pack(cfg.pack);
    if let Some(t) = cfg.title {
        args.title_override = Some(t);
    }
    if let Some(w) = cfg.width {
        args.width = w;
    }
    if let Some(h) = cfg.height {
        args.height = h;
    }
}

/// Build the world the player runs: the programmatic demo, a loose `.inf_lvl`, or
/// a cooked pack. Levels/packs decode through the `inf-scene` reader and bind
/// actor classes via [`InfSceneWorldBuilder`].
fn build_world(args: &Args) -> Result<BuiltWorld, String> {
    match &args.world {
        WorldChoice::Demo => Ok(demo::build()),
        WorldChoice::Level(path) => {
            let source = DevDirLevelSource::new(path);
            let content_dir = args
                .content
                .clone()
                .or_else(|| path.parent().map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("."));
            let actors = level::load_actor_classes_from_dir(&content_dir);
            let by_guid = level::load_actor_classes_by_guid_from_dir(&content_dir);
            let pcgs = level::load_pcg_payloads_by_guid_from_dir(&content_dir);
            let builder = InfSceneWorldBuilder::with_defaults(actors)
                .with_bindings(by_guid)
                .with_pcgs(pcgs);
            level::load(&source, &builder)
        }
        WorldChoice::Pack(path) => {
            let source = PackLevelSource::open(path)?;
            let actors = source.actor_classes()?;
            let by_guid = source.blueprint_classes_by_guid()?;
            let pcgs = source.pcg_payloads_by_guid()?;
            let builder = InfSceneWorldBuilder::with_defaults(actors)
                .with_bindings(by_guid)
                .with_pcgs(pcgs);
            level::load(&source, &builder)
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
        WorldChoice::Demo | WorldChoice::Pack(_) => input::default_map(),
    };
    let title = match &args.title_override {
        Some(t) => t.clone(),
        None => format!("Infinity Engine — {}", built.label),
    };
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

// ── PIE real-content path (P9.4) ─────────────────────────────────────────────

use inf_blueprint::BlueprintClass;
use inf_runtime::pie::ScenePayload;

/// Build a [`BuiltWorld`] from a streamed [`ScenePayload`] **exactly like the
/// cooked-pack path**: decode the classes, key them by asset GUID, and run the
/// v3 `.inf_lvl` bytes through [`InfSceneWorldBuilder::with_bindings`]. This is
/// the seam that makes PIE == shipping — the same builder the `--pack` /
/// `--level` boot uses, fed the live editor scene instead of a cooked file.
pub fn build_world_from_payload(payload: &ScenePayload) -> Result<BuiltWorld, String> {
    use crate::level::WorldBuilder;
    use std::collections::HashMap;

    let mut fallback: Vec<BlueprintClass> = Vec::new();
    let mut by_guid: HashMap<uuid::Uuid, BlueprintClass> = HashMap::new();
    for (guid, bytes) in &payload.classes {
        let class: BlueprintClass = serde_json::from_slice(bytes)
            .map_err(|e| format!("decode blueprint class {guid}: {e}"))?;
        by_guid.insert(*guid, class.clone());
        fallback.push(class);
    }
    // Streamed PCG graph payloads: the same `.inf_pcg` bytes the cook ships, so
    // the PIE player evaluates scatter identically to the shipping player (the
    // PIE == shipping guarantee extends to terrain/PCG content).
    let mut pcgs: HashMap<uuid::Uuid, inf_pcg::PcgAssetPayload> = HashMap::new();
    for (guid, bytes) in &payload.pcgs {
        let p = inf_pcg::PcgAssetPayload::decode(bytes)
            .map_err(|e| format!("decode pcg graph {guid}: {e}"))?;
        pcgs.insert(*guid, p);
    }
    let builder = InfSceneWorldBuilder::with_defaults(fallback)
        .with_bindings(by_guid)
        .with_pcgs(pcgs);
    builder.build(&payload.level_bytes)
}

/// One fixed step's determinism fingerprint: xxh3-64 of the `Guid`-sorted sim
/// snapshot — the per-frame `state_hash` a PIE `Frame` reports. Shared by the
/// PIE loop and the in-process reference so a mismatch is a real divergence.
pub fn step_state_hash(sim: &mut RuntimeSim) -> u64 {
    xxhash_rust::xxh3::xxh3_64(&sim.state_bytes())
}

/// The in-process reference the PIE==shipping gate compares against: build the
/// world from `payload`, step it `frames` fixed steps with no input, and record
/// each step's [`step_state_hash`]. A PIE subprocess fed the same payload must
/// stream byte-identical per-step hashes.
pub fn scene_trace(payload: &ScenePayload, frames: u64) -> Result<Vec<u64>, String> {
    let built = build_world_from_payload(payload)?;
    let mut sim = RuntimeSim::new(built.world, built.actors, built.gravity, built.hz);
    let mut hashes = Vec::with_capacity(frames as usize);
    for _ in 0..frames {
        sim.step_once(RuntimeInput::default());
        hashes.push(step_state_hash(&mut sim));
    }
    Ok(hashes)
}
