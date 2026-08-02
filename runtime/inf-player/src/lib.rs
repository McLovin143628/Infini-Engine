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
/// Streamed-scene performance + residency budgets (P16.6) — the §8 ratchet.
pub mod budget;
/// World-partition cell streaming (P16.5) — sim-driven spawn/despawn.
pub mod cell_stream;
pub mod config;
pub mod demo;
pub mod input;
pub mod level;
pub mod log;
// Native-only sandboxed WASM mod loading (P14.5). Gated off wasm32 so the
// browser player never pulls `wasmtime`.
#[cfg(not(target_arch = "wasm32"))]
pub mod mods;
pub mod render;
pub mod runtime_sim;
/// Camera-driven terrain streaming (P16.3b2) — the sim/render want split.
pub mod terrain_stream;
pub mod vmesh;
// The browser (wasm32) entry point + fetch/run glue (P14.2). Gated to wasm so
// the desktop build never names wasm-bindgen/web-sys.
#[cfg(target_arch = "wasm32")]
pub mod web;
// The Android NativeActivity entry (`android_main`) (P14.1). Gated to Android;
// builds with the NDK (cargo-ndk), device-verified (see docs/android-player.md).
#[cfg(target_os = "android")]
pub mod android;
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

/// Where a world's `.inf_terrain` streaming assets come from (P16.3b2).
///
/// Held beside the built world so [`attach_terrain_streaming`] can resolve a
/// `Terrain.asset` GUID **without reopening the pack** — the mapping a
/// `PackTileStore` slices tiles out of must stay open for the life of the run.
pub enum TerrainContent {
    /// No streaming source (the `--demo` world, or content with no terrain asset).
    None,
    /// Loose `.inf_terrain` files beside a `--level`, indexed by asset GUID.
    Dir(std::collections::HashMap<uuid::Uuid, PathBuf>),
    /// The opened cooked pack (`--pack` / the exported game).
    Pack(PackLevelSource),
}

impl TerrainContent {
    /// Resolve one `Terrain.asset` GUID. `None` — a dangling ref, or no source —
    /// leaves the terrain on its inline data (the documented fallback).
    pub fn source(&self, guid: uuid::Uuid) -> Option<terrain_stream::TerrainSource> {
        match self {
            TerrainContent::None => None,
            TerrainContent::Dir(index) => {
                match level::terrain_source_from_file(index.get(&guid)?) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        tracing::error!("inf-player: terrain asset {guid}: {e}");
                        None
                    }
                }
            }
            TerrainContent::Pack(source) => match source.terrain_source(guid) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("inf-player: terrain asset {guid}: {e}");
                    None
                }
            },
        }
    }
}

/// The content directory a `--level` boot reads its sidecar assets from.
fn level_content_dir(args: &Args, level_path: &std::path::Path) -> PathBuf {
    args.content
        .clone()
        .or_else(|| level_path.parent().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Build the world the player runs: the programmatic demo, a loose `.inf_lvl`, or
/// a cooked pack. Levels/packs decode through the `inf-scene` reader and bind
/// actor classes via [`InfSceneWorldBuilder`].
///
/// Returns the world **plus** its streaming-terrain source, which the caller
/// hands to [`attach_terrain_streaming`] after building the sim.
fn build_world(args: &Args) -> Result<(BuiltWorld, TerrainContent), String> {
    match &args.world {
        WorldChoice::Demo => Ok((demo::build(), TerrainContent::None)),
        WorldChoice::Level(path) => {
            let source = DevDirLevelSource::new(path);
            let content_dir = level_content_dir(args, path);
            let actors = level::load_actor_classes_from_dir(&content_dir);
            let by_guid = level::load_actor_classes_by_guid_from_dir(&content_dir);
            let pcgs = level::load_pcg_payloads_by_guid_from_dir(&content_dir);
            let (skeletons, clips, machines) = level::load_anim_assets_from_dir(&content_dir);
            let audio = level::load_audio_assets_from_dir(&content_dir);
            let terrains = level::terrain_paths_by_guid_from_dir(&content_dir);
            let builder = InfSceneWorldBuilder::with_defaults(actors)
                .with_bindings(by_guid)
                .with_pcgs(pcgs)
                .with_anim_assets(skeletons, clips, machines)
                .with_audio(audio);
            Ok((
                level::load(&source, &builder)?,
                TerrainContent::Dir(terrains),
            ))
        }
        WorldChoice::Pack(path) => {
            let source = PackLevelSource::open(path)?;
            let built = build_world_from_pack(&source)?;
            Ok((built, TerrainContent::Pack(source)))
        }
    }
}

/// Attach world-partition **cell streaming** to `sim` (P16.5).
///
/// A no-op for [`PartitionContent::None`](level::PartitionContent::None) — every
/// unpartitioned level, i.e. every level the editor writes by default — so
/// unpartitioned behaviour is untouched.
///
/// The persistent cell is already in `sim`'s world: the level builder spawned it
/// before actor binding, because it *is* the level at step 0. This only attaches
/// the manager for what streams.
pub fn attach_cell_streaming(sim: &mut RuntimeSim, content: &level::PartitionContent) {
    let Some(store) = content.store() else {
        return;
    };
    let cells = cell_stream::CellStreaming::attach(
        store.clone(),
        content.settings(),
        cell_stream::CellStreamBudget::default(),
    );
    sim.set_cell_streaming(cells);
}

/// Attach camera-driven terrain streaming to `sim` for every asset-backed
/// `Terrain` in its world (P16.3b2). A no-op for a world with none — which is
/// every level the editor writes today, so inline terrain behaviour is untouched.
pub fn attach_terrain_streaming(sim: &mut RuntimeSim, content: &TerrainContent) {
    if matches!(content, TerrainContent::None) {
        return;
    }
    let streaming = terrain_stream::TerrainStreaming::attach(
        sim.world_mut(),
        inf_terrain::StreamBudget::default(),
        |guid| content.source(guid),
    );
    if !streaming.is_empty() {
        sim.set_terrain_streaming(streaming);
    }
}

/// Build a [`BuiltWorld`] from an opened cooked-pack source — shared by the
/// `--pack` desktop boot ([`build_world`]) and the web fetch path
/// ([`web`](crate::web)). Decodes the pack's actor classes, PCG graphs, anim
/// assets, and audio, keys them by GUID, and runs the root `.inf_lvl` bytes
/// through the same [`InfSceneWorldBuilder`] every boot path uses (so a pack runs
/// identically however it was loaded).
pub fn build_world_from_pack(source: &PackLevelSource) -> Result<BuiltWorld, String> {
    let actors = source.actor_classes()?;
    let by_guid = source.blueprint_classes_by_guid()?;
    let pcgs = source.pcg_payloads_by_guid()?;
    let (skeletons, clips, machines) = source.anim_assets()?;
    let audio = source.audio_assets()?;
    let builder = InfSceneWorldBuilder::with_defaults(actors)
        .with_bindings(by_guid)
        .with_pcgs(pcgs)
        .with_anim_assets(skeletons, clips, machines)
        .with_audio(audio)
        // P16.5: a partitioned cooked level resolves its derived `.inf_part` out
        // of this same (already-open) pack mapping.
        .with_partition_pack(source.reader().clone(), source.root_level());
    level::load(source, &builder)
}

/// Headless CI path: no window/GPU. Run `--run-frames N` fixed steps of the world
/// with no input, folding the same xxh3 trace `inf_runtime::replay` uses, and
/// print `final-state-hash`. Exit 0 on clean completion; a panic (a gameplay
/// script blowing up, or the `--panic-after` test hook) is caught → nonzero exit,
/// with the crash report already written to stderr + the crash file by the panic
/// hook.
pub fn run_headless(args: &Args) -> ExitCode {
    let (mut built, terrain_content) = match build_world(args) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("inf-player: {e}");
            return ExitCode::FAILURE;
        }
    };
    let partition = built.take_partition();
    let label = built.label.clone();
    let frames = args.run_frames;
    let panic_after = args.panic_after;

    tracing::info!("inf-player: headless run of '{label}' for {frames} frame(s)");

    let mut sim = sim_from_built(built);
    // Cells first: terrain residency is derived from the sim's entities, and a
    // freshly-activated cell brings some of them in.
    attach_cell_streaming(&mut sim, &partition);
    attach_terrain_streaming(&mut sim, &terrain_content);
    attach_mods(&mut sim, args);
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        fold_trace_sim(sim, frames, panic_after)
    }));

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
    let (mut built, terrain_content) = match build_world(args) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("inf-player: {e}");
            return ExitCode::FAILURE;
        }
    };
    let partition = built.take_partition();
    let map = match &args.world {
        WorldChoice::Level(path) => input::load_map_beside(path),
        WorldChoice::Demo | WorldChoice::Pack(_) => input::default_map(),
    };
    let title = match &args.title_override {
        Some(t) => t.clone(),
        None => format!("Infinity Engine — {}", built.label),
    };
    // P13.4: load the cook-derived vmesh DAGs so `MeshRef.asset` entities render
    // real geometry (meshlet path / classic fallback per the renderer's auto-tier).
    let vmeshes = std::sync::Arc::new(load_vmeshes(args));
    // R-P4: the level's scene-persisted render block (post/exposure/lighting),
    // captured before `built` is consumed, applied by the render host.
    let render = built.render;
    let mut sim = sim_from_built(built);
    attach_cell_streaming(&mut sim, &partition);
    attach_terrain_streaming(&mut sim, &terrain_content);
    attach_mods(&mut sim, args);
    match window::run(
        title,
        args.width,
        args.height,
        sim,
        map,
        vmeshes,
        render,
        args.debug_cells,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("inf-player: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Load the `.inf_vmesh` registry for the chosen world: from the cooked pack
/// (`--pack`), the level's dev-dir sidecars (`--level`), or empty (`--demo` /
/// no meshes). The renderer resolves a `MeshRef.asset` against it (P13.4).
fn load_vmeshes(args: &Args) -> vmesh::VmeshRegistry {
    match &args.world {
        WorldChoice::Demo => vmesh::VmeshRegistry::new(),
        WorldChoice::Level(path) => {
            let content_dir = args
                .content
                .clone()
                .or_else(|| path.parent().map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("."));
            vmesh::VmeshRegistry::from_dir(&content_dir)
        }
        WorldChoice::Pack(path) => {
            let pack_path = if path.is_dir() {
                path.join(level::PACK_FILE)
            } else {
                path.clone()
            };
            // One `Arc<PackReader>` shared by every indexed vmesh: the mapping is
            // opened once, and a meshlet page is a sub-slice of it (P18.2).
            match inf_asset::PackReader::open(&pack_path)
                .map_err(|e| e.to_string())
                .and_then(|r| vmesh::VmeshRegistry::from_pack(std::sync::Arc::new(r)))
            {
                Ok(reg) => reg,
                Err(e) => {
                    tracing::warn!("inf-player: no vmeshes loaded from pack: {e}");
                    vmesh::VmeshRegistry::new()
                }
            }
        }
    }
}

/// Construct a [`RuntimeSim`] from a [`BuiltWorld`], seeding it with the level's
/// resolved P11 animation assets (P11.4): the `.inf_sm` state machines and the
/// root-motion `(skeleton, clip)` pairs. Every sim call site goes through here so
/// windowed, headless, and PIE runs all step animation identically (preview ==
/// shipped).
pub fn sim_from_built(built: BuiltWorld) -> RuntimeSim {
    let BuiltWorld {
        world,
        actors,
        gravity,
        hz,
        state_machines,
        root_clips,
        audio_clips,
        ..
    } = built;
    let mut sim = RuntimeSim::new(world, actors, gravity, hz);
    sim.set_state_machines(state_machines);
    for (guid, skel, clip) in root_clips {
        sim.register_root_motion_clip(guid, skel, clip);
    }
    sim.set_audio_clips(audio_clips);
    sim
}

/// Attach the `--mods` directory's sandboxed WASM mods to `sim` (P14.5). Native
/// only; a no-op on the browser player (which loads no native mods) and when no
/// `--mods` dir was given.
#[cfg(not(target_arch = "wasm32"))]
pub fn attach_mods(sim: &mut RuntimeSim, args: &Args) {
    let Some(dir) = &args.mods_dir else {
        return;
    };
    // Spawned entities get ids above the actor range.
    let first_free = sim.entity_map().keys().max().copied().unwrap_or(0) + 1;
    match mods::PlayerMods::load(dir, first_free) {
        Ok(m) if !m.is_empty() => sim.set_mods(Box::new(m)),
        Ok(_) => tracing::warn!("inf-player: no mods loaded from {}", dir.display()),
        Err(e) => tracing::error!("inf-player: failed to load mods: {e}"),
    }
}

/// No-op on wasm32 (the browser player does not load native mods).
#[cfg(target_arch = "wasm32")]
pub fn attach_mods(_sim: &mut RuntimeSim, _args: &Args) {}

/// Run `frames` fixed steps of `built` (no input) and fold every step's
/// `Guid`-sorted state into a 128-bit xxh3 trace — the determinism fingerprint.
/// Panics deliberately at frame `panic_after` when set (the crash-path fixture).
pub fn fold_trace(built: BuiltWorld, frames: u64, panic_after: Option<u64>) -> u128 {
    fold_trace_sim(sim_from_built(built), frames, panic_after)
}

/// Fold `frames` fixed steps of an already-built [`RuntimeSim`] (so a caller can
/// attach `--mods` before folding). See [`fold_trace`].
pub fn fold_trace_sim(mut sim: RuntimeSim, frames: u64, panic_after: Option<u64>) -> u128 {
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
///
/// **Streamed terrain over the PIE wire is deferred (P16.3b2).** [`ScenePayload`]
/// carries level bytes plus blueprint/PCG/anim assets, not `.inf_terrain` ones, so
/// a PIE session over an asset-backed terrain would run with it unstreamed. That
/// is not yet reachable content — the editor cannot author a `Terrain.asset` until
/// the P16.4 import wizard — and extending the payload is an `inf-runtime` (Ring 0
/// protocol) change this batch deliberately did not make. The parity it would
/// prove is covered meanwhile by the streamed-terrain gate's cooked-vs-loose arm
/// (`runtime/inf-player/tests/streamed_terrain.rs`), which runs the identical
/// world off a pack and off loose files and compares the traces.
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
    // Streamed P11 animation assets: the same `.inf_skel` / `.inf_anim` / `.inf_sm`
    // bytes the cook ships, so the PIE player resolves state machines + root-motion
    // clips identically to the shipping player (PIE == shipping for animation).
    let mut skeletons: HashMap<uuid::Uuid, inf_anim::SkeletonAsset> = HashMap::new();
    for (guid, bytes) in &payload.skeletons {
        skeletons.insert(
            *guid,
            inf_asset::decode(bytes).map_err(|e| format!("decode skeleton {guid}: {e}"))?,
        );
    }
    let mut clips: HashMap<uuid::Uuid, inf_anim::AnimClipAsset> = HashMap::new();
    for (guid, bytes) in &payload.clips {
        clips.insert(
            *guid,
            inf_asset::decode(bytes).map_err(|e| format!("decode clip {guid}: {e}"))?,
        );
    }
    let mut machines: HashMap<uuid::Uuid, inf_anim::StateMachineAsset> = HashMap::new();
    for (guid, bytes) in &payload.machines {
        machines.insert(
            *guid,
            inf_asset::decode(bytes).map_err(|e| format!("decode state machine {guid}: {e}"))?,
        );
    }
    let builder = InfSceneWorldBuilder::with_defaults(fallback)
        .with_bindings(by_guid)
        .with_pcgs(pcgs)
        .with_anim_assets(skeletons, clips, machines);
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
    let mut built = build_world_from_payload(payload)?;
    // P16.5: a partitioned scene streams here too — the payload carries the
    // level's entities inline, so the in-memory binning path produces the very
    // same cells the cook would have. Without this the reference trace would run
    // an empty world and "PIE == shipping" would compare nothing to nothing.
    let partition = built.take_partition();
    let mut sim = sim_from_built(built);
    attach_cell_streaming(&mut sim, &partition);
    let mut hashes = Vec::with_capacity(frames as usize);
    for _ in 0..frames {
        sim.step_once(RuntimeInput::default());
        hashes.push(step_state_hash(&mut sim));
    }
    Ok(hashes)
}
