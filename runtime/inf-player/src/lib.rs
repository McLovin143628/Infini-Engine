//! Infini Engine standalone player (P9.3) — library entry.
//!
//! The player runs a cooked game: it opens a window and drives a fixed-step
//! [`RuntimeSim`] (Blueprints + 2D physics) with
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
//! * `--pack <dir-or-pack>` — a cooked `content.ipack` (+ `manifest.toml`).
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
pub mod fracture;
#[cfg(not(target_arch = "wasm32"))]
pub mod mods;
/// Driving and reading a headless PIE session over protocol 3's input and probe
/// frames (wave FIX1) — the door the `--pie` loop and every gate take.
pub mod pie_drive;
pub mod render;
pub mod runtime_sim;
pub mod scatter_mesh;
/// Skeletal render assets (the P18.3 follow-up) — bind-space skinned geometry,
/// skeletons and clips, resolved out of a cooked pack or a dev dir.
pub mod skinned;
/// The fixed step's own per-phase breakdown (island wave I4b) — off by default.
pub mod step_profile;
/// Camera-driven terrain streaming (P16.3b2) — the sim/render want split.
pub mod terrain_stream;
/// The in-game settings dialog, the toasts and the interaction prompt, wired to
/// the window (island wave I5). The pure half is `inf-ui`.
pub mod ui;
pub mod vmesh;
pub mod voxel;
/// The player's window as the OS sees it: its console (there must not be one)
/// and its keyboard focus (wave FIX1). Cross-platform surface, Windows body.
pub mod win_host;
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
    if let Err(e) = apply_boot_config(&mut args) {
        // C4-33: the file is there, so the customer asked for *their* game.
        // Booting the built-in demo instead — which is what falling through
        // did — is the worst possible answer, and it exited 0.
        tracing::error!("inf-player: {e}");
        eprintln!("inf-player: {e}");
        return ExitCode::FAILURE;
    }
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
///
/// Returns `Err` when the config **exists** and cannot be honoured; the caller
/// refuses to boot rather than falling back to the built-in demo (C4-33).
fn apply_boot_config(args: &mut Args) -> Result<(), config::ConfigError> {
    if args.world_explicit {
        return Ok(());
    }
    let Some(cfg) = config::load_beside_exe()? else {
        return Ok(());
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
    Ok(())
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
    /// What a **PIE payload** names (P21.4, both routes since `ScenePayload` v12):
    /// `bytes` for the terrains it carried inline, `files` for the ones it named
    /// by path.
    ///
    /// The path route is not a weaker version of the byte one — it is the same
    /// door the `--level` boot and the cooked pack take, and it exists because a
    /// 50 km² `.inf_terrain` is 549.9 MB and a PIE frame is capped at 256 MiB. The
    /// PIE player runs on the editor's machine, so the file it is pointed at is
    /// the file the editor would otherwise have read into the pipe. See
    /// [`ScenePayload::terrain_paths`](inf_runtime::pie::ScenePayload::terrain_paths).
    ///
    /// A guid appears in exactly one of the two maps; `files` is consulted first,
    /// which is also what the world builder's PCG resolver does.
    Payload {
        bytes: std::collections::HashMap<uuid::Uuid, Vec<u8>>,
        files: std::collections::HashMap<uuid::Uuid, PathBuf>,
    },
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
            TerrainContent::Payload { bytes, files } => {
                if let Some(path) = files.get(&guid) {
                    return match level::terrain_source_from_file(path) {
                        Ok(s) => Some(s),
                        Err(e) => {
                            tracing::error!("inf-player: terrain asset {guid} at {path:?}: {e}");
                            None
                        }
                    };
                }
                match level::terrain_source_from_bytes(bytes.get(&guid)?.clone()) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        tracing::error!("inf-player: terrain asset {guid}: {e}");
                        None
                    }
                }
            }
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
            let biome_sets = level::load_biome_sets_by_guid_from_dir(&content_dir);
            let (skeletons, clips, machines) = level::load_anim_assets_from_dir(&content_dir);
            let audio = level::load_audio_assets_from_dir(&content_dir);
            let cloths = level::load_cloth_assets_from_dir(&content_dir);
            let hairs = level::load_hair_assets_from_dir(&content_dir);
            let terrains = level::terrain_paths_by_guid_from_dir(&content_dir);
            // IB-1: PCG pages its ground from the same `.inf_terrain` files the
            // streamer will. Resolved lazily, per terrain the level actually
            // references, so a content directory full of terrains costs nothing.
            let pcg_terrains = terrains.clone();
            let builder = InfSceneWorldBuilder::with_defaults(actors)
                .with_bindings(by_guid)
                .with_pcgs(pcgs)
                .with_biome_sets(biome_sets)
                .with_anim_assets(skeletons, clips, machines)
                .with_cloth_assets(cloths)
                .with_hair_assets(hairs)
                .with_audio(audio)
                .with_terrain_resolver(std::sync::Arc::new(move |g| {
                    let path = pcg_terrains.get(&g)?;
                    match level::terrain_source_from_file(path) {
                        Ok(s) => Some(s),
                        Err(e) => {
                            tracing::error!("inf-player: terrain asset {g} (for PCG): {e}");
                            None
                        }
                    }
                }));
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

/// The level's **material content** for a world the player just built (P26.4).
///
/// Read off the already-open source in `content` rather than reopening anything:
/// a `--pack` boot has the mapping in hand and `PackLevelSource::material_content`
/// slices it, so this costs a level decode and no texture bytes at all.
///
/// **A `--level` dev boot has none, and that is structural rather than a gap in
/// the wiring.** A `.inf_matd` is a *derived* record: it is written by the cook
/// and by the PIE payload builder, both of which run in the authoring ring and
/// link `inf-material`. A shipped player must not (the P26.2 inversion), so it
/// cannot flatten a loose `.inf_mat` for itself — a dev-dir boot renders its
/// surfaces off their scalar attributes, exactly as every pre-v22 level does.
/// The two paths that *matter* for "PIE == shipping" — a cooked pack and a
/// streamed payload — both have the record.
pub fn material_content_for_world(content: &TerrainContent) -> MaterialContent {
    match content {
        TerrainContent::Pack(source) => source.material_content(),
        _ => MaterialContent::default(),
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
/// `pcg` is the [`BuiltWorld`]'s own PCG context (island
/// phase, IB-1) — the same graphs and the same terrain resolver the load-time
/// pass used, so a `PcgVolume` that arrives with a cell is evaluated the way one
/// in the persistent cell was, rather than not at all. Pass
/// [`PcgContext::default`](level::PcgContext::default) for a world with no PCG.
pub fn attach_cell_streaming(
    sim: &mut RuntimeSim,
    content: &level::PartitionContent,
    pcg: level::PcgContext,
) {
    let Some(store) = content.store() else {
        return;
    };
    let cells = cell_stream::CellStreaming::attach(
        store.clone(),
        content.settings(),
        cell_stream::CellStreamBudget::default(),
    )
    .with_pcg(pcg);
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
    let biome_sets = source.biome_sets_by_guid()?;
    let (skeletons, clips, machines) = source.anim_assets()?;
    let audio = source.audio_assets()?;
    let cloths = source.cloth_assets()?;
    let hairs = source.hair_assets()?;
    let builder = InfSceneWorldBuilder::with_defaults(actors)
        .with_bindings(by_guid)
        .with_pcgs(pcgs)
        .with_biome_sets(biome_sets)
        .with_anim_assets(skeletons, clips, machines)
        .with_cloth_assets(cloths)
        .with_hair_assets(hairs)
        .with_audio(audio)
        // P16.5: a partitioned cooked level resolves its derived `.inf_part` out
        // of this same (already-open) pack mapping.
        .with_partition_pack(source.reader().clone(), source.root_level())
        // IB-1: …and PCG pages its ground out of it too, so a scatter over a
        // streamed terrain lands on the terrain instead of on sea level. Sliced
        // from the mmap already open above, so this costs no bytes.
        .with_terrain_resolver({
            let s = source.clone();
            std::sync::Arc::new(move |g| match s.terrain_source(g) {
                Ok(src) => src,
                Err(e) => {
                    tracing::error!("inf-player: terrain asset {g} (for PCG): {e}");
                    None
                }
            })
        });
    let mut built = level::load(source, &builder)?;
    // **The cooked path's session blend mode** — the twin of the line
    // `build_world_from_payload` runs for the PIE path. Without it a cooked boot
    // applied nothing at all, so a project that set the mode previewed one blend
    // and shipped another (the I1 audit's carried bound, A5).
    let mode = match source.blend_mode() {
        1 => inf_anim::SmBlendMode::CrossFade,
        _ => inf_anim::SmBlendMode::Inertialize,
    };
    inf_ecs::pose::set_blend_mode(&mut built.world, mode);
    Ok(built)
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
    // IB-1: the graphs + terrain resolver a streamed cell's PcgVolumes need.
    let pcg_ctx = built.pcg_context();
    let label = built.label.clone();
    let frames = args.run_frames;
    let panic_after = args.panic_after;

    tracing::info!("inf-player: headless run of '{label}' for {frames} frame(s)");

    let mut sim = sim_from_built(built);
    // Cells first: terrain residency is derived from the sim's entities, and a
    // freshly-activated cell brings some of them in.
    attach_cell_streaming(&mut sim, &partition, pcg_ctx);
    attach_terrain_streaming(&mut sim, &terrain_content);
    attach_voxel_volumes(&mut sim, &load_voxel_assets(args));
    // P22.3: what a level's destructible actors break into. Only a cooked
    // pack has these — a fracture is derived, not authored.
    fracture::attach_fractures(&mut sim, &load_fracture_assets(args));
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
    // IB-1: the graphs + terrain resolver a streamed cell's PcgVolumes need.
    let pcg_ctx = built.pcg_context();
    let map = match &args.world {
        WorldChoice::Level(path) => input::load_map_beside(path),
        WorldChoice::Demo | WorldChoice::Pack(_) => input::default_map(),
    };
    let title = match &args.title_override {
        Some(t) => t.clone(),
        None => format!("Infini Engine — {}", built.label),
    };
    // P13.4: load the cook-derived vmesh DAGs so `MeshRef.asset` entities render
    // real geometry (meshlet path / classic fallback per the renderer's auto-tier),
    // and (the P18.3 follow-up) the skeletal store so a `SkeletalMesh` renders its
    // real posed geometry instead of a placeholder.
    let (vmeshes, skinned, voxel_assets) = load_render_assets(args);
    // Wave TER2b: the authored meshes a scatter kind names. Loaded here, once,
    // beside the other render stores and for the same reason -- a projection runs
    // every frame and must not open a file.
    let scatter_meshes = std::sync::Arc::new(load_scatter_meshes(args));
    // P26.4: the level's bound `.inf_mat` records + their `.inf_tex` containers,
    // sliced out of the pack that is already open. This is what makes a shipped
    // build's surfaces textured — see `material_content_for_world` for why a
    // `--level` dev boot has none.
    let materials = std::sync::Arc::new(material_content_for_world(&terrain_content));
    // R-P4: the level's scene-persisted render block (post/exposure/lighting),
    // captured before `built` is consumed, applied by the render host.
    let render = built.render;
    let mut sim = sim_from_built(built);
    // ── P29.6 the locomotion camera's table ── read from `camera.toml` beside
    //    the level, exactly as the input map above is. A camera is not sim state
    //    and has no home in the scene schema, so its tunables live as text an
    //    author owns and a reviewer can read; the character wizard writes one,
    //    and a level with none gets the ported ALS defaults.
    if let WorldChoice::Level(path) = &args.world {
        sim.camera_mut().tuning = input::load_camera_beside(path);
    }
    attach_cell_streaming(&mut sim, &partition, pcg_ctx);
    attach_terrain_streaming(&mut sim, &terrain_content);
    // The SAME registry the render host will page from — one source of bytes, two
    // working sets, and the sim's is the one that decides where the floor is.
    attach_voxel_volumes(&mut sim, &voxel_assets);
    fracture::attach_fractures(&mut sim, &load_fracture_assets(args));
    attach_mods(&mut sim, args);
    match window::run(
        title,
        args.width,
        args.height,
        sim,
        map,
        vmeshes,
        scatter_meshes,
        skinned,
        voxel_assets,
        materials,
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

/// Seed the **simulation's** voxel volumes from `assets` (P21.2), so a Blueprint
/// calling `terrain.height_at` over a carved hole reads the cave floor beneath it
/// instead of the `None` a holed bilinear cell produces.
///
/// A no-op for a world with no `.inf_voxel` content, in which case the step is
/// bit-identical to its pre-P21.2 self.
///
/// # Why this is not the render host's store
///
/// The render host pages its volumes against the camera
/// (`inf_voxel::VoxelVolumes::sync_camera`); a fixed step must never be able to
/// observe that, or the replay / PIE-==-shipping gates die. So the sim gets its
/// own map, resolved once from level state, and the two are structurally apart —
/// there is no reference from the world to the render store. Exactly the split
/// `attach_terrain_streaming` above makes between `sync_sim` and `sync_render`.
///
/// Called from **every** boot path that has a voxel source: `--pack` and
/// `--level` through [`load_voxel_assets`] / [`load_render_assets`], and `--pie`
/// through [`sim_from_payload`] over the payload's own bytes (P21.4 —
/// `ScenePayload` v5). Before that rung the PIE path had no source at all and both
/// sides of the phase gate saw an empty map and agreed, which is why the gate is
/// built on [`sim_from_payload`] and not on
/// [`build_world_from_payload`] + [`sim_from_built`].
pub fn attach_voxel_volumes(sim: &mut RuntimeSim, assets: &voxel::VoxelRegistry) {
    if assets.is_empty() {
        return;
    }
    let volumes = runtime_sim::resolve_voxel_volumes(sim.world(), |guid| assets.load(guid));
    sim.set_voxel_volumes(volumes);
}

/// The `.inf_voxel` source alone, for a boot path that needs the **gameplay** half
/// of P21.1 without a GPU.
///
/// Split out of [`load_render_assets`] rather than folded into it because the
/// headless CI path has no renderer and must not pay for opening the vmesh and
/// skeletal stores to reach one registry — while still seeding the sim, since a
/// determinism trace that skipped the cave floor would not be the same world the
/// windowed build runs.
/// The `.inf_fracture` source for this run (P22.3), or an empty registry.
///
/// Only a **cooked pack** has one: a fracture is derived at cook, so a `--level`
/// run over loose content has none and previews as indestructible. See
/// [`fracture::FractureRegistry`] for why re-deriving on load would be worse than
/// admitting that.
fn load_fracture_assets(args: &Args) -> fracture::FractureRegistry {
    match &args.world {
        WorldChoice::Demo | WorldChoice::Level(_) => fracture::FractureRegistry::new(),
        WorldChoice::Pack(path) => {
            let pack_path = if path.is_dir() {
                path.join(level::PACK_FILE)
            } else {
                path.clone()
            };
            match inf_asset::PackReader::open(&pack_path) {
                Ok(reader) => fracture::FractureRegistry::from_pack(std::sync::Arc::new(reader)),
                Err(e) => {
                    tracing::warn!("inf-player: no fracture assets loaded from pack: {e}");
                    fracture::FractureRegistry::new()
                }
            }
        }
    }
}

fn load_voxel_assets(args: &Args) -> voxel::VoxelRegistry {
    match &args.world {
        WorldChoice::Demo => voxel::VoxelRegistry::new(),
        WorldChoice::Level(path) => {
            let content_dir = args
                .content
                .clone()
                .or_else(|| path.parent().map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("."));
            voxel::VoxelRegistry::from_dir(&content_dir)
        }
        WorldChoice::Pack(path) => {
            let pack_path = if path.is_dir() {
                path.join(level::PACK_FILE)
            } else {
                path.clone()
            };
            match inf_asset::PackReader::open(&pack_path) {
                Ok(reader) => voxel::VoxelRegistry::from_pack(std::sync::Arc::new(reader)),
                Err(e) => {
                    tracing::warn!("inf-player: no voxel assets loaded from pack: {e}");
                    voxel::VoxelRegistry::new()
                }
            }
        }
    }
}

/// Load the render-asset stores for the chosen world: the `.inf_vmesh` registry a
/// `MeshRef.asset` resolves against (P13.4) and the skeletal store a
/// `SkeletalMesh` resolves against (the P18.3 follow-up). Both come from the
/// cooked pack (`--pack`), the level's dev-dir sidecars (`--level`), or are inert
/// (`--demo`).
///
/// Loaded together on purpose: on the pack path they share **one**
/// `Arc<PackReader>`, so the mapping is opened once however many kinds of asset
/// the renderer reaches into it for (the P18.2 rule).
/// The authored meshes scattered content draws (wave TER2b), for whichever world
/// the arguments name.
///
/// Empty for `--demo`: a demo world has no `.inf_pcg` and therefore no cover, and
/// an empty table is precisely "every scattered instance draws its placeholder",
/// which is what the engine did before this wave.
fn load_scatter_meshes(args: &Args) -> inf_render::ScatterMeshes {
    let mut table = scanned_scatter_meshes(args);
    // I8b: and the twelve building module families, which name no file.
    scatter_mesh::add_building_modules(&mut table);
    table
}

/// The half of [`load_scatter_meshes`] that comes off disk.
fn scanned_scatter_meshes(args: &Args) -> inf_render::ScatterMeshes {
    match &args.world {
        WorldChoice::Demo => inf_render::ScatterMeshes::new(),
        WorldChoice::Level(path) => {
            let content_dir = args
                .content
                .clone()
                .or_else(|| path.parent().map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("."));
            scatter_mesh::from_dir(&content_dir)
        }
        WorldChoice::Pack(path) => {
            let pack_path = if path.is_dir() {
                path.join(level::PACK_FILE)
            } else {
                path.clone()
            };
            match inf_asset::PackReader::open(&pack_path) {
                Ok(reader) => scatter_mesh::from_pack(&reader),
                Err(e) => {
                    tracing::warn!("inf-player: no scatter meshes loaded from pack: {e}");
                    inf_render::ScatterMeshes::new()
                }
            }
        }
    }
}

fn load_render_assets(
    args: &Args,
) -> (
    std::sync::Arc<vmesh::VmeshRegistry>,
    std::sync::Arc<skinned::SkinnedRegistry>,
    std::sync::Arc<voxel::VoxelRegistry>,
) {
    let (vmeshes, skinned, voxels) = match &args.world {
        WorldChoice::Demo => (
            vmesh::VmeshRegistry::new(),
            skinned::SkinnedRegistry::new(),
            voxel::VoxelRegistry::new(),
        ),
        WorldChoice::Level(path) => {
            let content_dir = args
                .content
                .clone()
                .or_else(|| path.parent().map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("."));
            (
                vmesh::VmeshRegistry::from_dir(&content_dir),
                skinned::SkinnedRegistry::from_dir(&content_dir),
                voxel::VoxelRegistry::from_dir(&content_dir),
            )
        }
        WorldChoice::Pack(path) => {
            let pack_path = if path.is_dir() {
                path.join(level::PACK_FILE)
            } else {
                path.clone()
            };
            // One `Arc<PackReader>` shared by every indexed vmesh AND by the
            // skeletal store: the mapping is opened once, and a meshlet page is a
            // sub-slice of it (P18.2).
            match inf_asset::PackReader::open(&pack_path).map_err(|e| e.to_string()) {
                Ok(reader) => {
                    let reader = std::sync::Arc::new(reader);
                    let skinned = skinned::SkinnedRegistry::from_pack(reader.clone());
                    // P21.1: the `.inf_voxel` source rides the SAME `Arc` — a voxel
                    // chunk is a sub-slice of the mapping exactly as a meshlet page
                    // is, so opening the pack a third time would buy nothing.
                    let voxels = voxel::VoxelRegistry::from_pack(reader.clone());
                    match vmesh::VmeshRegistry::from_pack(reader) {
                        Ok(reg) => (reg, skinned, voxels),
                        Err(e) => {
                            tracing::warn!("inf-player: no vmeshes loaded from pack: {e}");
                            (vmesh::VmeshRegistry::new(), skinned, voxels)
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("inf-player: no render assets loaded from pack: {e}");
                    (
                        vmesh::VmeshRegistry::new(),
                        skinned::SkinnedRegistry::new(),
                        voxel::VoxelRegistry::new(),
                    )
                }
            }
        }
    };
    (
        std::sync::Arc::new(vmeshes),
        std::sync::Arc::new(skinned),
        std::sync::Arc::new(voxels),
    )
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
        skeletons,
        pose_clips,
        audio_clips,
        cloths,
        hairs,
        pcg,
        ..
    } = built;
    let mut sim = RuntimeSim::with_gravity(world, actors, gravity, hz);
    // I7b: the palette + the graphs the **fixed step** refreshes the biome-bound
    // population from as the ground pages. Seeded HERE for the same reason every
    // line below it is — this is the one function every sim call site goes
    // through, and a boot path that forgets an attachment does not crash, it
    // agrees with itself (the P21.4 law).
    sim.set_pcg_context(pcg);
    sim.set_state_machines(state_machines);
    for (guid, skel, clip) in root_clips {
        sim.register_root_motion_clip(guid, skel, clip);
    }
    // P24.1: the skeletons + the clips a machine's states play. Without these the
    // machines still step and no pose is ever published, so every character draws
    // its rest pose — which is what the engine did before this batch, silently.
    // Seeded HERE, in the one function every sim call site goes through, for the
    // reason `sim_from_payload` records: a boot path that forgets an attachment
    // does not crash, it agrees with itself.
    sim.set_skeletons(skeletons);
    sim.set_pose_clips(pose_clips);
    sim.set_audio_clips(audio_clips);
    // P24.4: the garments a `ClothSim` wearer simulates. Seeded HERE for the same
    // reason as the two lines above — this is the one function every sim call site
    // goes through, and a boot path that forgets an attachment does not crash, it
    // agrees with itself (the P21.4 law).
    sim.set_cloths(cloths);
    sim.set_hairs(hairs);
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
///
/// **Voxel volumes DO cross the wire (P21.4, `ScenePayload` v5)** — but the map
/// they seed lives on the [`RuntimeSim`], not on the [`BuiltWorld`], so this
/// function alone is not enough. Use [`sim_from_payload`], which is this plus the
/// voxel attach; a caller that hand-rolls
/// `build_world_from_payload` + [`sim_from_built`] gets a world with an **empty**
/// voxel map and a `terrain.height_at` that answers `0.0` over every carved hole.
pub fn build_world_from_payload(payload: &ScenePayload) -> Result<BuiltWorld, String> {
    use crate::level::WorldBuilder;
    use std::collections::HashMap;

    // The one place every PIE consumer passes through, so the one place the
    // envelope's version is checked (P21.4). Nothing read `schema_version` before
    // v5 — which is how a mid-struct field insertion could ship as a *silent*
    // half-load rather than a refusal.
    payload.check_version()?;

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
    // Streamed `.inf_biomes` vocabularies (P19.3): the same bytes the cook ships,
    // so the PIE player dispatches each painted biome's graph exactly like the
    // shipping player — the binding half of PIE == shipping.
    let mut biome_sets: HashMap<uuid::Uuid, inf_terrain::BiomeSet> = HashMap::new();
    for (guid, bytes) in &payload.biome_sets {
        biome_sets.insert(
            *guid,
            inf_asset::decode(bytes).map_err(|e| format!("decode biome set {guid}: {e}"))?,
        );
    }
    // P26.3b (`ScenePayload` v8): the garments and hairstyles a v21 level's
    // `ClothSim.asset` / `HairGuides.asset` name. **This is the P24.4 gap.**
    // Until this batch the payload had no slot for them and this function called
    // `with_anim_assets` and neither cloth door, so a character previewed in PIE
    // wore nothing while the same character in the editor's Simulate and in a
    // cooked pack wore both. Resolved through the same `InfSceneWorldBuilder`
    // doors the `--pack` and `--level` boots use, so there is one seeding rule.
    let mut cloths: HashMap<uuid::Uuid, inf_anim::ClothAsset> = HashMap::new();
    for (guid, bytes) in &payload.cloths {
        cloths.insert(
            *guid,
            inf_asset::decode(bytes).map_err(|e| format!("decode cloth {guid}: {e}"))?,
        );
    }
    let mut hairs: HashMap<uuid::Uuid, inf_anim::HairAsset> = HashMap::new();
    for (guid, bytes) in &payload.hairs {
        hairs.insert(
            *guid,
            inf_asset::decode(bytes).map_err(|e| format!("decode hair {guid}: {e}"))?,
        );
    }
    // IB-1: the `.inf_terrain` bytes the payload already carries are also where
    // PIE's PCG pages its ground. Without this a PIE session over a streamed
    // terrain scattered at sea level while the cooked build scattered on the
    // hill — a preview that differs from the build, which is the whole point of
    // the envelope carrying these bytes at all.
    //
    // Both routes (`ScenePayload` v12): the bytes the payload carries, and the
    // **paths** it names for terrains too large to carry. Resolved through the
    // same two doors `TerrainContent` uses below, so PCG pages its ground off the
    // same source the streamer pages the world off — the IB-1 defect was exactly
    // those two disagreeing.
    let terrains: HashMap<uuid::Uuid, Vec<u8>> = payload.terrains.iter().cloned().collect();
    let terrain_files: HashMap<uuid::Uuid, PathBuf> = payload
        .terrain_paths
        .iter()
        .map(|(g, p)| (*g, PathBuf::from(p)))
        .collect();
    let builder = InfSceneWorldBuilder::with_defaults(fallback)
        .with_bindings(by_guid)
        .with_pcgs(pcgs)
        .with_biome_sets(biome_sets)
        .with_anim_assets(skeletons, clips, machines)
        .with_cloth_assets(cloths)
        .with_hair_assets(hairs)
        .with_terrain_resolver(std::sync::Arc::new(move |g| {
            // The path first: a terrain that rode as a path has no bytes here, and
            // a terrain that rode as bytes has no path. Both, for one guid, would
            // be an editor bug rather than a choice — the payload builder emits
            // exactly one — and the path is the one the streamer will also take.
            if let Some(path) = terrain_files.get(&g) {
                return match level::terrain_source_from_file(path) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        tracing::error!("inf-player: terrain path {g} (for PCG): {e}");
                        None
                    }
                };
            }
            let bytes = terrains.get(&g)?;
            match level::terrain_source_from_bytes(bytes.clone()) {
                Ok(s) => Some(s),
                Err(e) => {
                    tracing::error!("inf-player: terrain payload {g} (for PCG): {e}");
                    None
                }
            }
        }));
    let mut built = builder.build(&payload.level_bytes)?;
    // **The session's default blend mode** (`ScenePayload` v11). A transition
    // whose `.inf_sm` v3 `blend` says *inherit* inherits THIS, so a PIE session
    // that did not apply it would blend every inheriting edge differently from
    // the editor that spawned it — the P29.2 boundary, from the reading side.
    //
    // The discriminants are `SmBlendMode`'s frozen wire indices; an unknown one
    // reads as the default rather than refusing, because a preview that blends
    // conservatively is better than a preview that does not start.
    let mode = match payload.blend_mode {
        1 => inf_anim::SmBlendMode::CrossFade,
        _ => inf_anim::SmBlendMode::Inertialize,
    };
    inf_ecs::pose::set_blend_mode(&mut built.world, mode);
    Ok(built)
}

/// The **derived material records + their texture containers** a streamed
/// [`ScenePayload`] carries (P26.3b), decoded and keyed the way the render
/// projection wants them.
///
/// Keyed by the **`.inf_mat` GUID** rather than by the derived one the wire uses:
/// the wire key is `derived_material_id(mat)` so that a payload entry and a pack
/// entry are one lookup rule, and the *scene* names the material itself
/// (`Material.asset`), so exactly one of the two ends has to un-salt. Doing it
/// here — once, at the boundary — is what keeps the projector from carrying the
/// salt.
///
/// A record that does not decode is **reported and skipped**, not fatal: a
/// surface with no resolvable material renders off its scalar attributes, which
/// is the permanent no-texture path and precisely what a pre-v22 level did.
pub fn materials_from_payload(payload: &ScenePayload) -> MaterialContent {
    let mut materials = std::collections::HashMap::new();
    for (derived_guid, bytes) in &payload.materials {
        match inf_asset::decode::<inf_asset::DerivedMaterial>(bytes) {
            Ok(rec) => {
                // Un-salt: the wire carries the DERIVED id, the scene names the
                // material. The salt is involutive, so one call inverts it.
                let mat_id = inf_asset::derived_material_id(inf_asset::AssetId(*derived_guid));
                materials.insert(mat_id.uuid(), rec);
            }
            Err(e) => tracing::warn!(
                "inf-player: derived material {derived_guid} did not decode, so every \
                 surface bound to it renders off its scalar attributes: {e}"
            ),
        }
    }
    MaterialContent {
        materials,
        textures: payload
            .textures
            .iter()
            .map(|(g, b)| (*g, VtTextureBytes::owned(b.clone())))
            .collect(),
    }
}

/// One `.inf_tex` v2 container, **however this host holds it** (P26.4).
///
/// The P26.3b ledger's remainder, closed: *"`material_content` copies container
/// bytes out of the mmap … so a pack path's RAM scales with the texture bytes a
/// level binds — the cost SVT exists to avoid"*. A pack-backed world now holds a
/// GUID and a shared reader and slices the mapping on demand, which is what the
/// terrain (`inf_terrain::stream`) and meshlet (`inf_vgeom::asset`) streamers
/// already do; a streamed PIE payload holds a `Vec`, because the wire already
/// copied those bytes and there is no mapping to slice.
///
/// Both arms are one [`VtTileSource`](inf_render::VtTileSource), which is the
/// seam `inf-render` was built with so that neither host's storage reaches the
/// registry.
#[derive(Clone)]
pub enum VtTextureBytes {
    /// A payload the wire delivered — the PIE arm. `Arc` so cloning the content
    /// (a device-loss rebuild hands the same textures to a new host) does not
    /// re-copy megabytes.
    Owned(std::sync::Arc<Vec<u8>>),
    /// A **slice of the pack mapping** — the shipping arm.
    Pack(std::sync::Arc<PackTexture>),
}

impl VtTextureBytes {
    pub fn owned(bytes: Vec<u8>) -> Self {
        Self::Owned(std::sync::Arc::new(bytes))
    }

    /// The whole container.
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Pack(p) => p.payload(),
        }
    }
}

impl inf_render::VtTileSource for VtTextureBytes {
    fn payload(&self) -> &[u8] {
        self.bytes()
    }
}

/// Compared **by content**, so the phase gate's "the pack and the payload carry
/// the same texture bytes" is a comparison of bytes rather than of how each host
/// stores them — which is the only thing that claim could usefully mean.
impl PartialEq for VtTextureBytes {
    fn eq(&self, other: &Self) -> bool {
        self.bytes() == other.bytes()
    }
}

impl std::fmt::Debug for VtTextureBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            Self::Owned(_) => "owned",
            Self::Pack(_) => "pack",
        };
        write!(f, "VtTextureBytes::{kind}({} B)", self.bytes().len())
    }
}

/// A `.inf_tex` entry addressed **inside** an open pack mapping (P26.4).
///
/// `payload` re-resolves the entry on every call rather than caching a slice,
/// because a borrowed slice cannot outlive the `read_ref` guard that produced
/// it. The cost is a `BTreeMap` lookup and one relaxed atomic load per call, and
/// the call happens once per parsed reader plus once per tile fetch — not per
/// texel and not per frame. What it buys is the property the phase exists for:
/// a level binding 400 MB of textures costs 400 MB of *address space*, not of
/// resident RAM.
pub struct PackTexture {
    reader: std::sync::Arc<inf_asset::PackReader>,
    guid: inf_asset::AssetId,
    /// Only ever filled for a **compressed** pack entry, where `read_ref` must
    /// decode and therefore has nothing to borrow. `.inf_tex` is in the
    /// streaming class and cooks uncompressed (P26.1), so this stays empty in
    /// every pack this engine writes — it exists so an old pack, or a future
    /// policy change, degrades to a copy rather than to a blank texture.
    decoded: std::sync::OnceLock<Vec<u8>>,
}

impl PackTexture {
    pub fn new(reader: std::sync::Arc<inf_asset::PackReader>, guid: inf_asset::AssetId) -> Self {
        Self {
            reader,
            guid,
            decoded: std::sync::OnceLock::new(),
        }
    }

    /// The container's bytes, borrowed from the mapping where the entry is raw.
    pub fn payload(&self) -> &[u8] {
        if let Some(v) = self.decoded.get() {
            return v;
        }
        match self.reader.read_ref(self.guid) {
            Ok(std::borrow::Cow::Borrowed(s)) => s,
            Ok(std::borrow::Cow::Owned(v)) => self.decoded.get_or_init(|| v),
            Err(e) => {
                tracing::warn!(
                    "inf-player: pack texture {} is unreadable, so every surface \
                     naming it renders off its scalar attributes: {e}",
                    self.guid.uuid()
                );
                &[]
            }
        }
    }
}

/// Where a world's material bindings and their texture bytes come from
/// (P26.3b) — the [`TerrainContent`] shape, for materials.
///
/// Held beside the built world because the render host needs both halves at
/// once: a derived record names texture GUIDs, and the registry needs the
/// container bytes behind them to page a tile.
#[derive(Default)]
pub struct MaterialContent {
    /// Derived records keyed by the **`.inf_mat`** GUID a scene's
    /// `Material.asset` names.
    pub materials: std::collections::HashMap<uuid::Uuid, inf_asset::DerivedMaterial>,
    /// `.inf_tex` v2 containers keyed by texture asset GUID — a **slice of the
    /// pack mapping** on the shipping path, an owned `Vec` on the PIE one (see
    /// [`VtTextureBytes`]).
    pub textures: std::collections::HashMap<uuid::Uuid, VtTextureBytes>,
}

impl MaterialContent {
    /// Whether this world has any material binding at all. `false` is the
    /// scalars-only world — every pre-v22 level, and every v22 level whose
    /// surfaces were never bound.
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// One texture's bytes as the registration door takes them.
    ///
    /// The whole seam between "where this host keeps its containers" and "what
    /// `inf-render` registers": the pack arm hands over a mapping slicer and the
    /// payload arm an owned buffer, and the registry cannot tell them apart.
    pub fn source(&self, guid: u128) -> Option<std::sync::Arc<dyn inf_render::VtTileSource>> {
        let bytes = self.textures.get(&uuid::Uuid::from_u128(guid))?;
        Some(std::sync::Arc::new(bytes.clone()) as std::sync::Arc<dyn inf_render::VtTileSource>)
    }

    /// This content as the **registration door** takes it (P26.4): `.inf_mat`
    /// GUID → its three texture GUIDs, in a `BTreeMap` because the walk order is
    /// the residency.
    ///
    /// The conversion happens *here*, at the boundary, for the same reason the
    /// derived-id salt is inverted here: `inf-render` names no asset crate (the
    /// P26.2 inversion), so the registry speaks `u128`s and this is where an
    /// `inf_asset::DerivedMaterial` becomes one.
    pub fn vt_materials(&self) -> std::collections::BTreeMap<u128, inf_render::VtMaterialMaps> {
        self.materials
            .iter()
            .map(|(mat, rec)| {
                (
                    mat.as_u128(),
                    inf_render::VtMaterialMaps {
                        albedo: rec.albedo.map(|t| t.uuid().as_u128()),
                        normal: rec.normal.map(|t| t.uuid().as_u128()),
                        orm: rec.orm.map(|t| t.uuid().as_u128()),
                        // Wave G: the detail slot now has an authoring field
                        // behind it (`.inf_mat` v3 → `.inf_matd` v2), so both
                        // hosts bind it — and they bind it through the SAME
                        // `detail_scale_q8` on the record rather than each doing
                        // its own metres→8.8 conversion. Two conversions are two
                        // things that can disagree. The twin is
                        // `inf_editor_core::render_assets`; a mirror test pins
                        // them field for field.
                        detail: rec.detail.map(|t| t.uuid().as_u128()),
                        detail_scale_q8: rec.detail_scale_q8(),
                    },
                )
            })
            .collect()
    }

    /// Every `.inf_tex` GUID the records name, deduplicated and in the fixed
    /// registration order: materials by GUID, then each record's albedo → normal
    /// → ORM.
    ///
    /// **The order is the residency trace.** `VtTextures::want_floor` is a pure
    /// function of the registration *sequence*, so a host that walked these in
    /// hash-map order would produce a different page set from one that walked
    /// them sorted — on the same level, from the same pack.
    ///
    /// **P26.4: it delegates.** The rule itself is
    /// `inf_render::vt_library::registration_order`, in the crate BOTH hosts
    /// link, because the editor projector needs the identical walk and a second
    /// spelling of it here is exactly the drift the P26.3b audit measured one
    /// level down (a `HashMap` iteration that produced two answers for one
    /// level). This wrapper is the `Uuid` face of it and nothing more.
    pub fn registration_order(&self) -> Vec<uuid::Uuid> {
        inf_render::registration_order(&self.vt_materials())
            .into_iter()
            .map(uuid::Uuid::from_u128)
            .collect()
    }
}

/// The result of [`sim_from_payload`]: a ready-to-step sim plus the two facts a
/// PIE handshake reports back to the editor.
pub struct PayloadSim {
    /// The sim, with every attachment a PIE session gets already made.
    pub sim: RuntimeSim,
    /// The level label (from the payload's own `.inf_lvl` bytes) for the
    /// `Loaded` reply and the window title.
    pub label: String,
    /// How many blueprint actors were bound, for the `Loaded` reply.
    pub actor_count: usize,
    /// **The level's own render block** (wave CERT1), decoded from the same
    /// `level_bytes` the world was built from.
    ///
    /// # The hole this closes
    ///
    /// `build_world_from_payload` has decoded this into `BuiltWorld::render`
    /// since R-P4 and every PIE boot path then threw it away:
    /// `window::run_pie` passed `RenderSettingsRecord::default()`, with a
    /// comment saying the payload carried no settings. It carried them all
    /// along — `level_bytes` **is** the live document's `.inf_lvl` payload and
    /// the record sits inside its `RuntimeSettings`. The consequence was that
    /// the editor's Play button previewed a level UNLIT while the shipped build
    /// of the same level rendered it lit, which is the exact divergence
    /// PIE == shipping exists to forbid, on the one half of the frame no
    /// `state_bytes` fold can see.
    ///
    /// Carried here rather than appended to `ScenePayload` **because no schema
    /// has to move for it**: the bytes were already in the envelope and already
    /// decoded.
    pub render: inf_scene::RenderSettingsRecord,
}

/// Build the **runnable sim** for a streamed [`ScenePayload`] — the one entry
/// point every PIE boot path goes through (P21.4).
///
/// [`build_world_from_payload`] + [`sim_from_built`] builds the world; this adds
/// the attachments that live on the sim rather than on the world:
///
/// * **cell streaming** (P16.5) — a partitioned payload bins its entities in
///   memory with the same Ring-0 function the cook used;
/// * **voxel volumes** (P21.4) — the payload's `.inf_voxel` bytes resolved through
///   [`runtime_sim::resolve_voxel_volumes`], the *same* function the pack path
///   calls, so a carved cave answers `terrain.height_at` identically in preview
///   and in the shipped build.
///
/// It exists as one function because the failure it prevents is silent: a boot
/// path that forgets an attach does not crash and does not warn — it runs a world
/// whose caves are not there, and a PIE == shipping comparison between two such
/// worlds passes. (The `--pie` subprocess used to build its sim with a bare
/// `RuntimeSim::new` rather than [`sim_from_built`], so it also dropped the state
/// machines, root-motion clips and audio clips the in-process reference kept;
/// routing every path through here closes that drift too.)
///
/// * **streamed terrain** (P21.4, the P16.3b2 deferral) — the payload's
///   `.inf_terrain` bytes, attached through the same `TerrainStreaming` the pack
///   path uses, because a level whose caves have mouths keeps that mask in the
///   asset and nowhere else.
pub fn sim_from_payload(payload: &ScenePayload) -> Result<PayloadSim, String> {
    let mut built = build_world_from_payload(payload)?;
    let label = built.label.clone();
    let actor_count = built.actors.len();
    // Captured before the world is consumed, exactly where `run_windowed`
    // captures it on the pack path (`inf_player::run_windowed`'s `built.render`).
    let render = built.render;
    // P16.5: a partitioned scene streams in PIE too — the payload carries the
    // entities inline, so the in-memory binning path produces the same cells the
    // cook would. Skipping it would run an empty world and quietly break
    // PIE == shipping.
    let partition = built.take_partition();
    // IB-1: the graphs + terrain resolver a streamed cell's PcgVolumes need.
    let pcg_ctx = built.pcg_context();
    let mut sim = sim_from_built(built);
    // Cells first: a freshly-activated cell can bring in a `VoxelVolume` entity,
    // and the voxel resolution below walks the world as it stands.
    attach_cell_streaming(&mut sim, &partition, pcg_ctx);
    attach_voxel_volumes(
        &mut sim,
        &voxel::VoxelRegistry::from_payload(&payload.voxels),
    );
    // P22.3: the payload's derived `.inf_fracture` bytes (schema v6). Without
    // this arm a PIE session would have nothing to break and the phase-22 gate
    // would be comparing two empty maps — which is exactly what P21.4 found the
    // phase-21 gate doing, and it is the reason this seam is `sim_from_payload`
    // rather than `build_world_from_payload` + `sim_from_built`.
    fracture::attach_fractures(
        &mut sim,
        &fracture::FractureRegistry::from_payload(&payload.fractures),
    );
    // P21.4 (the P16.3b2 deferral): the payload's `.inf_terrain` bytes, streamed
    // through the same `TerrainStreaming` the pack path attaches. The level bytes
    // carry a blanked working set for these terrains, so without this the PIE
    // world has no ground — and, since P21.2, none of the hole mask either.
    //
    // **Both routes** (`ScenePayload` v12): a terrain rides as bytes or as a path,
    // never as both, and `TerrainContent::Payload` holds the pair so there is one
    // source rather than two attachments racing each other.
    if payload.has_terrain_sources() {
        let content = crate::TerrainContent::Payload {
            bytes: payload.terrains.iter().cloned().collect(),
            files: payload
                .terrain_paths
                .iter()
                .map(|(g, p)| (*g, PathBuf::from(p)))
                .collect(),
        };
        attach_terrain_streaming(&mut sim, &content);
    }
    Ok(PayloadSim {
        sim,
        label,
        actor_count,
        render,
    })
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
    // Through the same one seam the `--pie` subprocess takes, so the reference and
    // the thing it is a reference *for* cannot make different worlds out of one
    // payload. (Both would previously have run an empty voxel map and agreed.)
    let mut sim = sim_from_payload(payload)?.sim;
    let mut hashes = Vec::with_capacity(frames as usize);
    for _ in 0..frames {
        sim.step_once(RuntimeInput::default());
        hashes.push(step_state_hash(&mut sim));
    }
    Ok(hashes)
}
