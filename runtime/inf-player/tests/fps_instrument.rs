//! **THE FPS INSTRUMENT** (island wave I4) — what "≥ 60 fps" is allowed to mean
//! in this repository.
//!
//! Before this file the AAA-readiness certification's complaint was exact: *"the
//! only GPU frame harness renders 640 × 360… no test in this repo measures fps at
//! a shipping resolution, so '≥ 60 fps' for the island has no existing
//! instrument."* Every frame number the tree carried was a **mean CPU wall
//! clock** around 484 lit cubes at a resolution nobody ships.
//!
//! # What this measures
//!
//! One scene, composed of the heaviest things the engine already ships, at the
//! two resolutions a desktop title ships at:
//!
//! * the **phase-30 city** — 100 banded `PcgVolume` blocks, 1 000 grammar
//!   buildings, 370 468 solids, a real road mesh through the GIS import door;
//! * a **streamed terrain** beneath it, paging its render cut as the camera moves
//!   (flat at zero, so the composed level is *the same city* wave I3 measured —
//!   see `island_frame_source_png`);
//! * the **phase-29 wizard character**, skinned, with a live state machine;
//! * the sun, and the render settings a **shipped player** builds for this level
//!   on this adapter — through `inf_player::render::shipped_settings`, the one
//!   door, rather than a configuration a harness invented.
//!
//! and it drives them the way a frame is actually driven: a fixed step, a render
//! terrain sync, a full re-projection (`scene.mark_dirty()` every frame, which is
//! the *churning* regime Hardening Wave E proved is the only honest one), and a
//! camera that moves down the middle street so nothing is cached from last frame
//! because nothing moved.
//!
//! # How it is measured
//!
//! * **A whole discarded pass** runs first — pipelines compile, the terrain's
//!   render cut converges, the scatter payloads seat.
//! * **p50 / p95 / p99** of the per-frame CPU wall clock, plus per-pass GPU
//!   milliseconds from `inf_render::timing` — the query-set clock this wave
//!   built, because a whole-frame number cannot say where the frame went.
//! * **MIN of rounds**: several independent rounds, and the round with the lowest
//!   p50 is the one reported. A shared machine's slow round is a statement about
//!   the machine; the fastest round is the closest this can get to a statement
//!   about the engine. (`inf-anim`'s `inertialization` harness is the precedent.)
//!
//! # Where it asserts
//!
//! **Nowhere on CI, by name.** The two ceilings this file introduces are wall
//! clocks, and `inf_player::budget`'s header states the law: *"prefer a budget in
//! a unit the machine cannot inflate, and when only a clock will do, condition it
//! the way the rest of the tree already does."* A shared virtualized runner has
//! no GPU worth timing and preempts one leg and not another. So CI **reports**
//! every number in this file and asserts none of them; a real adapter on a real
//! machine — and the I9 certification — asserts.
//!
//! What IS asserted everywhere, unconditionally, because none of it is a clock:
//! the composed scene really carries the city, the terrain and the character; the
//! ground under the city changes not one building; and the per-pass report names
//! every pass the renderer built.

use std::path::{Path, PathBuf};

use glam::{DVec3, Vec3};
use inf_asset::PackReader;
use inf_ecs::components::{Guid, PcgVolume, SkeletalMesh, Terrain};
use inf_editor_core::samples;
use inf_math::FloatingOrigin;
use inf_packager::{cook, CookOptions};
use inf_player::budget::{
    RATCHET_NOTE, SHIPPING_FRAME_BUDGET_MS, SHIPPING_FRAME_CEILING_MS,
    SHIPPING_FRAME_P99_CEILING_MS,
};
use inf_player::level::PackLevelSource;
use inf_player::render::{project_scene_full, shipped_settings, sync_voxel_store};
use inf_player::runtime_sim::RuntimeSim;
use inf_project::ProjectManifest;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, RenderScene, RenderView, HEADLESS_FORMAT,
};

/// The two resolutions a desktop title ships at. 1080p is the floor the
/// certification asked for; 1440p is the second point, because a frame that is
/// fragment-bound and a frame that is draw-bound scale differently and one
/// resolution cannot tell them apart.
const RESOLUTIONS: [(u32, u32, &str); 2] = [(1920, 1080, "1080p"), (2560, 1440, "1440p")];

/// Frames measured per round — one pass of the scripted flythrough.
const FRAMES: usize = 120;
/// Independent rounds; the one with the lowest p50 is reported (MIN-of-rounds).
///
/// **Every round replays the identical camera sequence** (`step` restarts at 0),
/// and a whole discarded pass runs first. The first version of this harness let
/// `step` run on across rounds, so each round flew a *different* stretch of the
/// city and MIN-of-rounds picked the cheapest stretch rather than the least
/// disturbed round — it reported 28.3 ms where the same code repeating one
/// sequence reports what it really costs. A MIN over samples of different things
/// is not a minimum, it is a selection.
const ROUNDS: usize = 3;

/// The CPU stages one frame is split into, in the order the frame runs them.
///
/// A frame that is CPU-bound and cannot say WHERE is the same defect the
/// certification found on the GPU side, one processor over — so the instrument
/// splits the wall clock at the four seams a host actually has.
const CPU_STAGE_NAMES: [&str; CPU_STAGES] = [
    "sim fixed step",
    "stream sync",
    "projection",
    "render (record)",
    "poll (GPU wait)",
];
const CPU_STAGES: usize = 5;

// ── the fixture ─────────────────────────────────────────────────────────────

/// Scaffold a project holding the composed instrument level and cook it.
///
/// The city's **assets** are copied and its level is not: the instrument writes
/// its own `.inf_lvl` (city + ground + character), and two startup levels in one
/// project would make the cook choose.
fn cook_instrument(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Island Frame", "blank-3d")
        .save(&proj)
        .expect("the manifest saves");
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).expect("mkdir Content");

    let city = samples::city_dir();
    for f in [
        "CityBlock.inf_pcg",
        "CityBlock.inf_pcg.toml",
        "CityRoads.inf_mesh",
        "CityRoads.inf_mesh.toml",
    ] {
        std::fs::copy(city.join(f), content.join(f))
            .unwrap_or_else(|e| panic!("copy the city's {f}: {e}"));
    }
    let hero = samples::phase29_locomotion_dir();
    for f in samples::island_frame_character_files() {
        std::fs::copy(hero.join(f), content.join(f))
            .unwrap_or_else(|e| panic!("copy the character's {f}: {e}"));
    }
    samples::write_island_frame_terrain(&content).expect("the ground imports");
    samples::write_island_frame_level(&content).expect("the level saves");

    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("the instrument scene cooks");
    out
}

/// The pack's world, its sim, and the render stores a shipped player resolves
/// against — assembled exactly as `inf_player::load_render_assets` does, from
/// **one** `Arc<PackReader>` (the P18.2 rule).
struct Fixture {
    sim: RuntimeSim,
    vmeshes: inf_player::vmesh::VmeshRegistry,
    skinned: inf_player::skinned::SkinnedRegistry,
    voxel_assets: inf_player::voxel::VoxelRegistry,
    record: inf_scene::RenderSettingsRecord,
    materials: std::sync::Arc<inf_player::MaterialContent>,
}

fn open(pack: &Path) -> Fixture {
    let source = PackLevelSource::open(pack).expect("the pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("the pack world builds");
    // R-P4: the level's own render block, captured before the world is consumed
    // — exactly where `inf_player::run_windowed` captures it.
    let record = built.render;
    let materials = std::sync::Arc::new(source.material_content());
    let reader = std::sync::Arc::new(
        PackReader::open(&pack.join(inf_player::level::PACK_FILE)).expect("the pack maps"),
    );
    let skinned = inf_player::skinned::SkinnedRegistry::from_pack(reader.clone());
    let voxel_assets = inf_player::voxel::VoxelRegistry::from_pack(reader.clone());
    let vmeshes = inf_player::vmesh::VmeshRegistry::from_pack(reader)
        .expect("the pack's derived meshlet DAGs index");
    Fixture {
        sim: inf_player::sim_from_built(built),
        vmeshes,
        skinned,
        voxel_assets,
        record,
        materials,
    }
}

/// Register the level's virtual textures on `renderer` — `PlayerRenderHost::rebuild_vt`'s
/// body, through the same `inf_render::build_vt_level` door both hosts call, so
/// the instrument's frame samples textures the way a shipped frame does.
fn bind_virtual_textures(gpu: &GpuContext, renderer: &mut EngineRenderer, fx: &Fixture) -> usize {
    let mats = fx.materials.vt_materials();
    if mats.is_empty() {
        renderer.set_vt_level(None);
        return 0;
    }
    let budget = renderer.settings().vt.budget_bytes;
    let materials = fx.materials.clone();
    match inf_render::build_vt_level(
        &gpu.device,
        &gpu.queue,
        renderer.settings(),
        budget,
        &mats,
        |g| materials.source(g),
    ) {
        Some((textures, pools, report)) => {
            renderer.set_vt_level(Some((textures, pools)));
            report.textures
        }
        None => {
            renderer.set_vt_level(None);
            0
        }
    }
}

/// The flythrough: the camera rides the city's own scripted drive line, at eye
/// height, looking east down the middle street.
///
/// Scripted rather than free, for the phase-16 gate's reason — the frame
/// sequence has to be a function of the level alone, or two runs measure two
/// different worlds. Moving rather than parked, for
/// `frame_budget.rs::frame_stays_under_budget_under_version_churn`'s reason: a
/// still camera measures a frame this engine never draws.
fn fly(step: u64, width: u32, height: u32) -> RenderView {
    let p = samples::city_drive_point(step);
    let eye = DVec3::new(p.x, 2.2, p.z);
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: Vec3::new(1.0, -0.08, 0.0).normalize(),
        up: Vec3::Y,
        fov_y: 70f32.to_radians(),
        near: 0.05,
        width,
        height,
        ortho: None,
    }
}

// ── statistics ──────────────────────────────────────────────────────────────

/// One round's frame-time distribution, in milliseconds.
#[derive(Debug, Clone, Copy)]
struct Round {
    p50: f64,
    p95: f64,
    p99: f64,
    worst: f64,
}

/// Nearest-rank percentile over a sorted sample — the definition that always
/// names a frame that actually happened, rather than interpolating one that did
/// not.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn round_of(mut frames: Vec<f64>) -> Round {
    frames.sort_by(f64::total_cmp);
    Round {
        p50: percentile(&frames, 0.50),
        p95: percentile(&frames, 0.95),
        p99: percentile(&frames, 0.99),
        worst: *frames.last().unwrap_or(&0.0),
    }
}

/// Is this adapter one whose milliseconds mean anything?
fn representative(info: &wgpu::AdapterInfo) -> bool {
    let n = info.name.to_ascii_lowercase();
    let virtualized = n.contains("paravirtual") || n.contains("virtualbox") || n.contains("vmware");
    info.device_type != wgpu::DeviceType::Cpu && !virtualized
}

// ── the run ─────────────────────────────────────────────────────────────────

/// One measured configuration: the frame-time rounds and the best round's
/// per-pass GPU breakdown.
struct Measured {
    rounds: Vec<Round>,
    best: usize,
    passes: Vec<(&'static str, f64)>,
    gpu_frame_ms: f64,
    cpu_ms: [f64; CPU_STAGES],
    instances: usize,
    scatter_batches: usize,
    vgeom_instances: usize,
    skinned: usize,
    terrain_tiles: usize,
    vt_textures: usize,
}

impl Measured {
    fn round(&self) -> Round {
        self.rounds[self.best]
    }
}

/// Render `ROUNDS × FRAMES` frames of the composed scene at `(w, h)` and answer
/// the distribution.
///
/// The projection is the **production** one: `sync_voxel_store` then
/// `project_scene_full`, which is `PlayerRenderHost::project`'s whole body — the
/// two halves are public and named as such precisely so a gate can drive them
/// without a window.
fn measure(gpu: &GpuContext, fx: &mut Fixture, w: u32, h: u32) -> Measured {
    let (settings, _tier) = shipped_settings(gpu, fx.record);
    let target = HeadlessTarget::new(gpu, w, h);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);
    let vt_textures = bind_virtual_textures(gpu, &mut renderer, fx);
    let timed = renderer.set_gpu_timing(gpu, true);
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    let mut voxels = inf_voxel::VoxelVolumes::new();
    let mut debris = inf_render::DebrisCache::default();

    let mut step: u64 = 0;
    let mut frame = |scene: &mut RenderScene,
                     renderer: &mut EngineRenderer,
                     fx: &mut Fixture,
                     step: u64|
     -> (Option<inf_render::FrameTimings>, [f64; CPU_STAGES]) {
        let view = fly(step, w, h);
        // The CPU half, stage by stage. A frame that is CPU-bound and cannot say
        // WHERE is the same defect the certification found on the GPU side, one
        // processor over.
        let mut cpu = [0.0f64; CPU_STAGES];
        let t = std::time::Instant::now();
        fx.sim
            .step_once(inf_player::runtime_sim::RuntimeInput::default());
        cpu[0] = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        fx.sim.sync_render_terrain(view.eye_world);
        sync_voxel_store(&mut voxels, &fx.voxel_assets, &fx.sim, view.eye_world);
        cpu[1] = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        project_scene_full(
            scene,
            &fx.sim,
            1.0,
            &fx.vmeshes,
            &fx.skinned,
            &voxels,
            &mut debris,
            renderer.vt_textures(),
        );
        cpu[2] = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        renderer.render(gpu, scene, &view, &target.view, (w, h));
        cpu[3] = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        cpu[4] = t.elapsed().as_secs_f64() * 1000.0;
        (renderer.gpu_timings(gpu), cpu)
    };

    // The discarded pass: pipelines compile, the terrain's render cut converges
    // (`max_loads_per_sync` is 16, so a cut several rings deep takes tens of
    // frames to settle), and the scatter payloads seat in their content-keyed
    // buffers. Measuring any of that would be measuring a frame that happens
    // once.
    for _ in 0..FRAMES {
        frame(&mut scene, &mut renderer, fx, step);
        step += 1;
    }

    let mut rounds: Vec<Round> = Vec::with_capacity(ROUNDS);
    let mut best_passes: Vec<(&'static str, f64)> = Vec::new();
    let mut best_gpu = 0.0;
    let mut best_cpu = [0.0f64; CPU_STAGES];
    let mut best = 0usize;
    for r in 0..ROUNDS {
        let mut ms = Vec::with_capacity(FRAMES);
        let mut sums: Vec<(&'static str, f64)> = Vec::new();
        let mut gpu_total = 0.0;
        let mut cpu_total = [0.0f64; CPU_STAGES];
        step = 0;
        for _ in 0..FRAMES {
            let t0 = std::time::Instant::now();
            let (timings, cpu) = frame(&mut scene, &mut renderer, fx, step);
            ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            step += 1;
            for (slot, v) in cpu_total.iter_mut().zip(cpu) {
                *slot += v;
            }
            if let Some(t) = timings {
                gpu_total += t.total_ms;
                if sums.is_empty() {
                    sums = t.passes.iter().map(|p| (p.name, p.ms)).collect();
                } else {
                    for (slot, p) in sums.iter_mut().zip(&t.passes) {
                        slot.1 += p.ms;
                    }
                }
            }
        }
        let round = round_of(ms);
        if r == 0 || round.p50 < rounds[best].p50 {
            best = r;
            best_gpu = gpu_total / FRAMES as f64;
            best_cpu = cpu_total.map(|v| v / FRAMES as f64);
            best_passes = sums
                .into_iter()
                .map(|(n, total)| (n, total / FRAMES as f64))
                .collect();
        }
        rounds.push(round);
    }
    if !timed {
        best_passes.clear();
    }

    Measured {
        rounds,
        best,
        passes: best_passes,
        gpu_frame_ms: best_gpu,
        cpu_ms: best_cpu,
        instances: scene.instances.len(),
        scatter_batches: scene.scatter.len(),
        vgeom_instances: scene.vgeom_instances.len(),
        skinned: scene.skinned.len(),
        terrain_tiles: scene.terrains.iter().map(|t| t.tiles.len()).sum(),
        vt_textures,
    }
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **The composed scene really is the city, the ground and the character** —
/// asserted with no GPU, because a frame time over content that is not what the
/// ledger says it is is a number about nothing.
///
/// And the load-bearing half: **the ground under the city changes not one
/// building**. `island_frame_source_png` holds the terrain flat at exactly zero
/// so the composed level's PCG output is bit-identical to the flat city wave I3
/// measured; if that ever stops being true, the instrument stops being an
/// instrument over *that* city and every number in the I3 ledger stops being
/// comparable. The number is the one I3 printed: **370 468 solids**.
#[test]
fn the_instrument_scene_carries_the_city_the_ground_and_the_character() {
    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook_instrument(tmp.path());
    let source = PackLevelSource::open(&pack).expect("the pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("the pack world builds");
    let w = built.world.world();

    let mut volumes: Vec<(uuid::Uuid, PcgVolume)> = w
        .iter_entities()
        .filter_map(|e| Some((e.get::<Guid>()?.0, e.get::<PcgVolume>()?.clone())))
        .collect();
    volumes.sort_by_key(|(g, _)| *g);
    let solids: usize = volumes.iter().map(|(_, v)| v.structures.len()).sum();
    let buildings: usize = volumes.iter().map(|(_, v)| v.structure_groups.len()).sum();
    let terrains = w
        .iter_entities()
        .filter(|e| e.contains::<Terrain>())
        .count();
    let characters = w
        .iter_entities()
        .filter(|e| e.contains::<SkeletalMesh>())
        .count();

    println!(
        "instrument scene: {} volumes, {buildings} buildings, {solids} solids, \
         {terrains} terrain, {characters} skinned character",
        volumes.len()
    );

    assert_eq!(volumes.len(), 100, "the hundred city blocks must be here");
    assert_eq!(
        buildings, 1_000,
        "a thousand buildings, as wave I3 measured"
    );
    assert_eq!(
        solids, 370_468,
        "the ground under the city MOVED a building: wave I3's city is \
         370 468 solids and this one is {solids}. The instrument's terrain is \
         held flat at exactly zero so the composed level is the same city the \
         I3 ledger describes — if the terrain gains a slope, every number in \
         that ledger stops being comparable with every number in this one."
    );
    assert_eq!(terrains, 1, "the streamed ground must be in the world");
    assert_eq!(characters, 1, "the wizard character must be in the world");
}

/// **The instrument, at shipping resolution.** The wave's headline number.
#[test]
fn the_frame_at_shipping_resolution() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP fps_instrument: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let real = representative(&info);

    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook_instrument(tmp.path());
    let mut fx = open(&pack);
    let (settings, tier) = shipped_settings(&gpu, fx.record);

    println!(
        "\n=== THE FPS INSTRUMENT === {} ({:?}), tier {tier:?}\n\
         settings: vgeom {} / visbuffer {} / shadows {} / gi {} / vsm {} / taa {} / \
         ssao {} / bloom {} / vt budget {} MiB\n\
         method: {ROUNDS} rounds x {FRAMES} frames after one discarded pass of \
         {FRAMES}, MIN of rounds by p50, every round replaying the SAME camera \
         sequence; scripted drive-through, full re-projection every frame",
        info.name,
        info.device_type,
        settings.vgeom.enabled,
        settings.vgeom.visbuffer,
        settings.shadows.enabled,
        settings.gi.enabled,
        settings.vsm.enabled,
        settings.taa,
        settings.ssao.enabled,
        settings.bloom.enabled,
        settings.vt.budget_bytes / (1024 * 1024),
    );

    let mut worst_p95 = 0.0f64;
    let mut worst_p99 = 0.0f64;
    for (w, h, label) in RESOLUTIONS {
        let m = measure(&gpu, &mut fx, w, h);
        let r = m.round();
        println!(
            "\n{label} ({w}x{h}): p50 {:.3} ms ({:.1} fps) | p95 {:.3} ms | \
             p99 {:.3} ms | worst {:.3} ms   [round {} of {ROUNDS}]",
            r.p50,
            1000.0 / r.p50.max(1.0e-9),
            r.p95,
            r.p99,
            r.worst,
            m.best + 1,
        );
        println!(
            "{label} content: {} mesh instances, {} scatter batches, {} vgeom \
             instances, {} skinned, {} terrain tiles, {} virtual textures",
            m.instances,
            m.scatter_batches,
            m.vgeom_instances,
            m.skinned,
            m.terrain_tiles,
            m.vt_textures
        );
        for (i, rr) in m.rounds.iter().enumerate() {
            println!(
                "{label} round {}: p50 {:.3} p95 {:.3} p99 {:.3}",
                i + 1,
                rr.p50,
                rr.p95,
                rr.p99
            );
        }
        let cpu_sum: f64 = m.cpu_ms.iter().sum();
        println!("{label} CPU frame {cpu_sum:.3} ms, stage by stage:");
        for (n, ms) in CPU_STAGE_NAMES.iter().zip(m.cpu_ms) {
            println!(
                "{label}   {n:<16} {ms:7.3} ms  ({:5.1} % of the CPU frame)",
                ms / cpu_sum.max(1.0e-9) * 100.0
            );
        }
        if m.passes.is_empty() {
            println!("{label} per-pass: unavailable (no timestamp queries on this device)");
        } else {
            let mut by_cost = m.passes.clone();
            by_cost.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!(
                "{label} GPU frame {:.3} ms; dearest passes:",
                m.gpu_frame_ms
            );
            for (n, ms) in by_cost.iter().take(12) {
                println!(
                    "{label}   {n:<16} {ms:7.3} ms  ({:5.1} % of the GPU frame)",
                    ms / m.gpu_frame_ms.max(1.0e-9) * 100.0
                );
            }
            // Anti-vacuity: a report whose segments do not add up to the frame is
            // a report about a frame the renderer did not draw.
            let sum: f64 = m.passes.iter().map(|(_, ms)| ms).sum();
            assert!(
                (sum - m.gpu_frame_ms).abs() < 1.0e-6,
                "{label}: the per-pass segments ({sum:.6} ms) do not tile the \
                 GPU frame ({:.6} ms)",
                m.gpu_frame_ms
            );
        }
        // **The harness frame is SERIALIZED and a presenter's is not.** This loop
        // polls to completion every frame, because a frame time measured without
        // a sync point is a submission time; the price is that the GPU's work
        // lands *after* the CPU's instead of underneath it. A real presenter
        // overlaps them, so the frame it would show is bounded below by the
        // dearer of the two halves. Reported as an estimate and never asserted —
        // it is arithmetic over two measurements, not a third measurement.
        let submitted = cpu_sum - m.cpu_ms[4];
        let pipelined = submitted.max(m.gpu_frame_ms);
        println!(
            "{label} PIPELINED ESTIMATE {pipelined:.3} ms ({:.1} fps) = max(CPU without the wait {submitted:.3}, GPU frame {:.3}) — an estimate, not a measurement; the asserted number is the serialized p95 above",
            1000.0 / pipelined.max(1.0e-9),
            m.gpu_frame_ms
        );
        println!(
            "{label} DISTANCE FROM 60 fps: p50 {:+.3} ms, p95 {:+.3} ms against a \
             {SHIPPING_FRAME_BUDGET_MS} ms frame",
            r.p50 - SHIPPING_FRAME_BUDGET_MS,
            r.p95 - SHIPPING_FRAME_BUDGET_MS
        );
        worst_p95 = worst_p95.max(r.p95);
        worst_p99 = worst_p99.max(r.p99);
    }

    if !real {
        eprintln!(
            "\n{}: timing reported, not asserted (software/paravirtual adapter)",
            info.name
        );
        return;
    }
    if std::env::var_os("CI").is_some() {
        // **Every CI runner reports and does not assert, by name** — the law
        // `inf-anim`'s inertialization harness paid for and `inf_player::budget`'s
        // header states. A frame time on a shared virtualized runner is a
        // measurement of the runner. Locally, and in the I9 certification, the
        // ceilings below still bite.
        eprintln!("\nCI: frame times reported, not asserted (shared runner)");
        return;
    }
    if cfg!(debug_assertions) {
        // **The build is not the build, so this reports rather than asserts** —
        // the paravirtual-adapter law, one layer down. `[profile.dev]` is
        // `opt-level = 1` with debug assertions on for every workspace crate, so
        // the CPU half of the frame here is a measurement of a build nobody
        // ships; the GPU half is unaffected and is printed above either way.
        //
        // The full battery runs in this profile, which is exactly why the
        // ceilings are not asserted in it: a tripwire that only ever sees the
        // slow build would have to be set where the fast build cannot regress
        // past it, and would therefore never fire.
        //
        //   cargo test --release -p inf-player --test fps_instrument -- --nocapture
        //
        // is the run that asserts, and the one the I9 certification makes.
        eprintln!(
            "\ndev profile (opt-level 1, debug assertions ON): frame times \
             reported, not asserted — re-run with --release for the shipping-build \
             number the ceilings are set from"
        );
        return;
    }
    assert!(
        worst_p95 <= SHIPPING_FRAME_CEILING_MS,
        "the 95th-percentile frame cost {worst_p95:.3} ms at the worse of the two \
         shipping resolutions, over the {SHIPPING_FRAME_CEILING_MS} ms ceiling on \
         {} {RATCHET_NOTE}",
        info.name
    );
    assert!(
        worst_p99 <= SHIPPING_FRAME_P99_CEILING_MS,
        "the 99th-percentile frame cost {worst_p99:.3} ms, over the \
         {SHIPPING_FRAME_P99_CEILING_MS} ms hitch ceiling on {} {RATCHET_NOTE}",
        info.name
    );
}
