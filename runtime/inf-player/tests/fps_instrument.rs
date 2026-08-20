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
//! # What this frame does NOT draw (the I4 audit)
//!
//! The shipped settings for a level that authors no render block leave
//! **shadows, GI, VSM, TAA, SSAO, bloom and the visbuffer all off** —
//! `RenderSettingsRecord::default()` for five of them, `VsmSettings::default()`
//! for VSM, the tier for the visbuffer. So the headline below is honest about
//! what a shipped player draws today and is **not** a lit AAA frame, and the
//! audit's addition is that the harness says so in its own output and then
//! measures the difference: `THE STACK'S PRICE` runs the same content at 1080p
//! with the authorable half turned on through the same `shipped_settings` door,
//! and prints it — with, since island wave I4b, the same CPU-stage and per-pass
//! tables the shipped configuration gets, because a configuration whose price is
//! one number cannot be optimised.
//!
//! Measured on an RTX 4070 Ti, MIN of rounds, **after wave I4b**: lit p95
//! 38.1-41.8 ms against 15.8-19.5 as shipped, GPU frame 16.1-16.5 against
//! 2.9-6.0, and the pipelined estimate **16.4-16.5 ms lit (60.7-60.9 fps)**.
//! Wave I4 measured the same two configurations at **92.3-92.9 lit against
//! 43.7-44.0, GPU frame 35.8-36.1 against 17.3-19.4**.
//!
//! **Two of those movements are not the engine's**, and the file says so where
//! it prints them: the unlit GPU frame fell further than any change to the unlit
//! path can explain, because I4's frame left the GPU idle two thirds of every
//! frame and a card that is idle downclocks. A GPU millisecond is a measurement
//! of the device *in the state the frame put it in*, so GPU columns are only
//! comparable between runs whose CPU frames are comparable.
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
//!   It also means the headline is the *best* of `ROUNDS`, so the printed
//!   distance from 60 fps is a lower bound on the distance; every round's
//!   percentiles are printed beside it.
//! * **The per-stage tables are MEANS and the headline is a PERCENTILE**, which
//!   the output now says on the line above the table (the I4 audit). The CPU
//!   stages are asserted to tile the round's own **mean** frame, so a cost that
//!   sits in no stage — as the timestamp readback did — is a red arm rather than
//!   an unexplained few milliseconds between two tables.
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
    CITY_STEP_BUDGET_MS, RATCHET_NOTE, SHIPPING_FRAME_BUDGET_MS, SHIPPING_FRAME_CEILING_MS,
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
/// disturbed round. A MIN over samples of different things is not a minimum, it
/// is a selection, and that is why the reset is here.
///
/// *The 28.3 ms the wave attributed to its absence does not reproduce* (the I4
/// audit re-ran the mutation): `CITY_DRIVE_STEP_M` is 0.25 m, so a 120-frame
/// round is **thirty metres** and every round is inside the same district —
/// removing the reset moves the p50 by about 1 %, to 39.104 ms. The discipline is
/// right and is kept; the figure was retired.
const ROUNDS: usize = 3;

/// Steps discarded before the fixed step's own breakdown is measured (island
/// wave I4b) — the `FRAMES`-long discarded pass, one processor over: the band
/// seats, the terrain tiles mesh, and every `structure_stamps` miss there is
/// happens in here.
const STEP_WARMUP: usize = 120;
/// Steps per profiling round.
const STEP_SAMPLES: usize = 120;
/// Independent profiling rounds; the cheapest by the step's own total is
/// reported (MIN-of-rounds, the instrument's discipline).
const STEP_ROUNDS: usize = 3;

/// The CPU stages one frame is split into, in the order the frame runs them.
///
/// A frame that is CPU-bound and cannot say WHERE is the same defect the
/// certification found on the GPU side, one processor over — so the instrument
/// splits the wall clock at the four seams a host actually has.
/// The last stage is **the instrument's own overhead**, and it is here for the
/// reason the GPU segments tile the GPU frame: a breakdown whose parts do not add
/// up to the whole it sits beside is a breakdown of a frame nobody measured. The
/// timestamp readback (`gpu_timings`, a `map_async` + a poll) happens inside the
/// wall clock that produces p50/p95/p99, and a shipped frame does not pay it. It
/// is measured, printed, and subtracted by the reader rather than left as an
/// unnamed residue between a 37.8 ms stage table and a 39.9 ms p50 — which is
/// what the first version of this file left. (Added by the I4 audit.)
const CPU_STAGE_NAMES: [&str; CPU_STAGES] = [
    "sim fixed step",
    "stream sync",
    "projection",
    "render (record)",
    "poll (GPU wait)",
    "timing readback",
];
const CPU_STAGES: usize = 6;

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
    /// The **mean** of the same frames. Carried beside the percentiles because
    /// every per-stage number this file prints is a mean and every headline
    /// number is a percentile, and reading one against the other is only honest
    /// if both are on the page. It is also what makes the stage table's tiling
    /// assertion possible.
    mean: f64,
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
    let mean = match frames.is_empty() {
        true => 0.0,
        false => frames.iter().sum::<f64>() / frames.len() as f64,
    };
    Round {
        p50: percentile(&frames, 0.50),
        p95: percentile(&frames, 0.95),
        p99: percentile(&frames, 0.99),
        worst: *frames.last().unwrap_or(&0.0),
        mean,
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
    /// `(name, GPU ms, CPU record ms)` per pass — the CPU column is island
    /// wave I4b's addition, because a lit frame's dearest half turned out to be
    /// the recording rather than the drawing.
    passes: Vec<(&'static str, f64, f64)>,
    gpu_frame_ms: f64,
    cpu_ms: [f64; CPU_STAGES],
    instances: usize,
    scatter_batches: usize,
    vgeom_instances: usize,
    skinned: usize,
    terrain_tiles: usize,
    vt_textures: usize,
    /// The VSM caster pass's own counters over the measured rounds (island wave
    /// I4b) — `pages x groups` is what the pass RECORDS, and the record column
    /// is meaningless without them.
    vsm: Option<inf_render::VsmRasterStats>,
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
fn measure(
    gpu: &GpuContext,
    fx: &mut Fixture,
    w: u32,
    h: u32,
    settings: inf_render::RenderSettings,
) -> Measured {
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
        // **The instrument's own cost, inside the instrument's own clock.** The
        // readback below is a `map_async` plus a second poll, and it is inside
        // the `t0` span the percentiles are taken over — so it has to be a named
        // stage or the stage table stops tiling the frame it sits under.
        let t = std::time::Instant::now();
        let timings = renderer.gpu_timings(gpu);
        cpu[5] = t.elapsed().as_secs_f64() * 1000.0;
        (timings, cpu)
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
    let mut best_passes: Vec<(&'static str, f64, f64)> = Vec::new();
    let mut best_gpu = 0.0;
    let mut best_cpu = [0.0f64; CPU_STAGES];
    let mut best = 0usize;
    for r in 0..ROUNDS {
        let mut ms = Vec::with_capacity(FRAMES);
        let mut sums: Vec<(&'static str, f64, f64)> = Vec::new();
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
                    sums = t.passes.iter().map(|p| (p.name, p.ms, p.cpu_ms)).collect();
                } else {
                    for (slot, p) in sums.iter_mut().zip(&t.passes) {
                        slot.1 += p.ms;
                        slot.2 += p.cpu_ms;
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
                .map(|(n, gpu, cpu)| (n, gpu / FRAMES as f64, cpu / FRAMES as f64))
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
        vsm: renderer.vsm_raster_stats(),
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

/// **THE SHIPPED PLAYER PIPELINES, AND THAT IS WHAT THE ESTIMATE ASSUMES**
/// (island wave I4b).
///
/// Every run of this file prints a `PIPELINED ESTIMATE` — `max(CPU without the
/// wait, GPU frame)` — beside the serialized number it measures, on the stated
/// grounds that "a real presenter overlaps the halves". Wave I4 carried that as
/// arithmetic over two measurements and named a windowed present-to-present
/// harness as the honest closure. A harness needs a window and this battery has
/// none; what it can do, and what I4 did not, is **check the claim about the
/// player** rather than assert it in prose.
///
/// The player's frame path is four calls —
/// `SurfaceChain::acquire` → `EngineRenderer::render` (record + submit) →
/// `Queue::present` — and it contains **no blocking device poll**. The CPU
/// therefore runs ahead into the next frame while the GPU drains this one, and
/// `acquire` blocks only when the swap chain has no free image, which is the
/// definition of "the GPU is the bottleneck". That is exactly the model the
/// estimate is.
///
/// This arm is a **source scope**, not a substring ban: it extracts
/// `PlayerRenderHost::render`'s body and the windowed loop's own frame block, and
/// requires that neither waits. A ban over the whole file would have been
/// satisfied by a poll moved one function away, which is the shape the P23 byte
/// pin failed at.
#[test]
fn the_shipped_players_frame_path_does_not_wait_for_the_gpu() {
    let render_rs = include_str!("../src/render.rs");
    let start = render_rs
        .find("pub fn render(&mut self, view: &RenderView) {")
        .expect("PlayerRenderHost::render is the player's one frame call");
    let body = &render_rs[start..];
    let end = body
        .find("\n    }\n")
        .expect("the function body ends at a de-indented brace");
    let body = &body[..end];
    println!(
        "the player's frame path is {} lines: {}",
        body.lines().count(),
        body.lines()
            .filter(|l| {
                let l = l.trim();
                !l.is_empty() && !l.starts_with("//")
            })
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(" ")
    );
    assert!(
        body.contains("self.gpu.queue.present(frame)"),
        "the extracted body is not the present path — this arm is reading the \
         wrong function and would pass for anything"
    );
    assert!(
        !body.contains("poll("),
        "the shipped player's frame path now polls the device:\n{body}\nA poll \
         serializes the CPU and GPU halves, which is what the harness does on \
         purpose and what a presenter must not do — and it would make every \
         PIPELINED ESTIMATE this file prints a description of a frame the player \
         no longer draws."
    );

    // **AND THE LOOP THAT CALLS IT** (the I4b audit). The paragraph above said
    // this arm extracted "the windowed loop's own frame block" as well; it did
    // not, and a poll one caller up serializes the halves exactly as a poll one
    // caller down does. `PlayerApp::frame` is the block: it runs the fixed steps,
    // projects, and calls `host.render(&view)` — the whole of what a presented
    // frame costs the CPU.
    let window_rs = include_str!("../src/window.rs");
    let start = window_rs
        .find("fn frame(&mut self, event_loop: &ActiveEventLoop) {")
        .expect("PlayerApp::frame is the windowed loop's own frame block");
    let loop_body = &window_rs[start..];
    let end = loop_body
        .find("\n    }\n")
        .expect("the function body ends at a de-indented brace");
    let loop_body = &loop_body[..end];
    println!(
        "the windowed loop's frame block is {} lines",
        loop_body.lines().count()
    );
    assert!(
        loop_body.contains("live.host.render(&view)"),
        "the extracted block is not the windowed frame path — this arm is \
         reading the wrong function and would pass for anything"
    );
    assert!(
        !loop_body.contains("poll("),
        "the windowed loop now polls the device around its frame:\n{loop_body}"
    );
}

/// **THE FIXED STEP'S OWN BREAKDOWN** (island wave I4b) — the table wave I4
/// could not print.
///
/// I4 measured the frame, found it CPU-bound, and found the single dearest thing
/// in it to be the fixed step at **13.0–14.9 ms** over this city — of which
/// ~2.2 ms was the I3 collider band **and ~11.5 ms was unattributed**. "Attribute
/// it before prescribing" is what the I4 audit routed to this wave, and this arm
/// is the attribution: `RuntimeSim` marks every phase of its own body and the
/// phases tile the step by construction.
///
/// **No GPU.** The step is CPU work over a cooked pack, so this arm runs
/// everywhere the battery runs — and the number it prints in the `dev` profile
/// is a number about a build nobody ships (the I4 law), which is why the
/// **budget is asserted only in `--release`**, on a machine whose milliseconds
/// mean something, exactly like every other wall clock in this tree.
#[test]
fn the_fixed_steps_own_budget() {
    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook_instrument(tmp.path());
    let mut fx = open(&pack);

    // The discarded pass, for the frame instrument's reason one processor over:
    // the first steps seat the collider band, mesh the terrain tiles, and take
    // every `structure_stamps` miss there is. Measuring them would be measuring
    // a step that happens once.
    for _ in 0..STEP_WARMUP {
        fx.sim
            .step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    fx.sim.set_step_profiling(true);

    let mut rounds: Vec<(f64, inf_player::step_profile::StepProfile)> = Vec::new();
    for _ in 0..STEP_ROUNDS {
        let mut acc = inf_player::step_profile::StepProfile::default();
        let t0 = std::time::Instant::now();
        for _ in 0..STEP_SAMPLES {
            fx.sim
                .step_once(inf_player::runtime_sim::RuntimeInput::default());
            acc.accumulate(&fx.sim.step_profile());
        }
        let wall = t0.elapsed().as_secs_f64() * 1000.0 / STEP_SAMPLES as f64;
        acc.scale(1.0 / STEP_SAMPLES as f64);
        rounds.push((wall, acc));
    }
    // MIN of rounds, by the step's own total — the instrument's own discipline.
    let best = rounds
        .iter()
        .enumerate()
        .min_by(|a, b| a.1 .1.total_ms().total_cmp(&b.1 .1.total_ms()))
        .map(|(i, _)| i)
        .expect("at least one round");
    let (wall, prof) = rounds[best];

    println!(
        "\n=== THE FIXED STEP, PHASE BY PHASE === {STEP_ROUNDS} rounds x \
         {STEP_SAMPLES} steps after {STEP_WARMUP} discarded, MIN of rounds; \
         content: the phase-30 city (370 468 solids, 1 000 buildings), a streamed \
         terrain and a skinned character — the fps instrument's own scene"
    );
    for (i, (w, p)) in rounds.iter().enumerate() {
        println!(
            "round {}: step {:.3} ms (wall {:.3} ms)",
            i + 1,
            p.total_ms(),
            w
        );
    }
    println!(
        "STEP {:.3} ms  [round {} of {STEP_ROUNDS}]",
        prof.total_ms(),
        best + 1
    );
    for (n, ms) in prof.dearest_first() {
        if ms <= 0.0005 {
            continue;
        }
        println!(
            "  {n:<18} {ms:7.3} ms  ({:5.1} % of the step)",
            ms / prof.total_ms().max(1.0e-9) * 100.0
        );
    }
    let silent: Vec<&str> = prof
        .rows()
        .filter(|(_, ms)| *ms <= 0.0005)
        .map(|(n, _)| n)
        .collect();
    if !silent.is_empty() {
        println!("  under 0.0005 ms: {}", silent.join(", "));
    }
    println!(
        "  the step's own wall clock is {wall:.3} ms; the phases sum to {:.3} ms",
        prof.total_ms()
    );
    // **What the solver is actually paying for.** A step whose dearest phase is
    // `bridge3d.step` over a world with one moving thing in it is a step paying
    // for its own STATIC geometry, and the pair count is the evidence.
    let (tracked, touching) = fx.sim.bridge3d().world().contact_pair_counts();
    println!(
        "  physics world: {} bodies, {} admitted structure colliders, \
         {tracked} contact pairs tracked ({touching} touching)",
        fx.sim.bridge3d().body_count(),
        fx.sim.bridge3d().admitted_structures(),
    );

    // **THE PHASES TILE THE STEP.** The GPU segments' tiling assertion and the
    // CPU stages', one processor over: a breakdown whose parts do not add up to
    // the whole it sits beside is a breakdown of a step nobody measured. The
    // wall clock also carries `set_input` and the profile does not, which is
    // three `BTreeSet` differences on an empty input — hence a tolerance rather
    // than an equality, and it is in PROPORTION rather than in milliseconds so
    // it means the same thing in `dev` (where the step is slower) as in release.
    let drift = (wall - prof.total_ms()).abs() / wall.max(1.0e-9);
    assert!(
        drift < 0.10,
        "the phases sum to {:.3} ms beside a {wall:.3} ms step — {:.1} % of the \
         step is in no phase, so the breakdown describes a step this arm did not \
         time",
        prof.total_ms(),
        drift * 100.0
    );

    // The §8 budget itself — release only, real machine only, for
    // `inf_player::budget`'s stated reason.
    if cfg!(debug_assertions) {
        eprintln!(
            "\ndev profile (opt-level 1, debug assertions ON): the step is \
             reported, not asserted — re-run with --release for the number \
             CITY_STEP_BUDGET_MS is set from"
        );
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("\nCI: the step is reported, not asserted (shared runner)");
        return;
    }
    println!(
        "STEP BUDGET: {:.3} ms measured against a {CITY_STEP_BUDGET_MS} ms \
         ceiling {RATCHET_NOTE}",
        prof.total_ms()
    );
    assert!(
        prof.total_ms() <= CITY_STEP_BUDGET_MS,
        "the fixed step cost {:.3} ms over the city, past the \
         {CITY_STEP_BUDGET_MS} ms ceiling {RATCHET_NOTE}",
        prof.total_ms()
    );
}

/// **A stopwatch is not behaviour.** The phase clock reads no sim state, writes
/// none and changes no ordering, so a profiled step and an unprofiled one must
/// produce byte-identical sim state — and this is the arm that says so rather
/// than the comment that claims it.
///
/// Built to falsify: it compares `state_bytes()` (the same buffer the replay
/// fold, `step_state_hash` and every PIE == shipping arm consume) after the same
/// number of steps on two sims built from the same pack, one profiled.
#[test]
fn the_profile_does_not_move_the_simulation() {
    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook_instrument(tmp.path());
    let mut plain = open(&pack);
    let mut profiled = open(&pack);
    profiled.sim.set_step_profiling(true);
    for _ in 0..24 {
        plain
            .sim
            .step_once(inf_player::runtime_sim::RuntimeInput::default());
        profiled
            .sim
            .step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let a = plain.sim.state_bytes();
    let b = profiled.sim.state_bytes();
    println!(
        "24 steps: {} bytes of sim state, profiled and not, identical = {}",
        a.len(),
        a == b
    );
    assert!(
        profiled.sim.step_profile().total_ms() > 0.0,
        "the profiled sim reported a zero step — the clock is not armed, so the \
         comparison below is between two unprofiled runs"
    );
    assert_eq!(
        a, b,
        "a profiled step and an unprofiled one produced different sim state"
    );
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
    // **WHAT THIS FRAME DOES NOT DRAW, said out loud** (the I4 audit).
    //
    // Every one of those flags is `false`, and none of it is the instrument's
    // choice: `RenderSettingsRecord::default()` ships shadows / GI / TAA / SSAO
    // / bloom off, `VsmSettings::default().enabled` is `false` engine-wide
    // ("until P27.4 gives the pages a receiver"), and `VgeomSettings::visbuffer`
    // is off on every tier. So the headline below is an honest measurement of
    // **what a shipped player draws for a level that authors no render block**,
    // and it is emphatically not a measurement of a frame with a lighting stack
    // in it. Quoting the number without this line would make "≥ 60 fps" mean a
    // frame with no shadows in it. `the_price_of_the_lighting_stack` below runs
    // the same content with the authorable half turned on and prints what it
    // costs — reported, never asserted, because the ceilings are set from the
    // shipped configuration.
    println!(
        "NOTE — the measured frame draws NO shadows, NO GI, NO VSM, NO TAA, NO \
         SSAO, NO bloom and NO visbuffer. That is the shipped default for a level \
         with no authored render block, not a choice this harness made; the price \
         of turning the authorable half on is printed at the end of this run."
    );

    let mut worst_p95 = 0.0f64;
    let mut worst_p99 = 0.0f64;
    // The 1080p row's own p95 and GPU frame, kept so the lighting stack's price
    // is a difference between two runs of the SAME resolution.
    let mut shipped_1080 = (0.0f64, 0.0f64);
    for (w, h, label) in RESOLUTIONS {
        let m = measure(&gpu, &mut fx, w, h, settings);
        let r = m.round();
        if (w, h) == (RESOLUTIONS[0].0, RESOLUTIONS[0].1) {
            shipped_1080 = (r.p95, m.gpu_frame_ms);
        }
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
        println!(
            "{label} CPU frame {cpu_sum:.3} ms (a MEAN — every stage below is a \
             mean, and the p50/p95/p99 above are percentiles of the same {FRAMES} \
             frames; the round's own mean frame is {:.3} ms), stage by stage:",
            r.mean
        );
        for (n, ms) in CPU_STAGE_NAMES.iter().zip(m.cpu_ms) {
            println!(
                "{label}   {n:<16} {ms:7.3} ms  ({:5.1} % of the CPU frame)",
                ms / cpu_sum.max(1.0e-9) * 100.0
            );
        }
        // **THE STAGES TILE THE FRAME** — the CPU twin of the GPU segments'
        // tiling assertion below, and the arm that would have caught the residue
        // the I4 audit found: the timestamp readback sat inside the wall clock the
        // percentiles are taken over and inside no stage, so the table summed to
        // 37.839 ms beside a 39.792 ms headline with nothing naming the gap.
        // 0.5 ms is one `Instant::now()` pair per stage plus the loop's own
        // bookkeeping, which is what the difference is allowed to be.
        assert!(
            (cpu_sum - r.mean).abs() < 0.5,
            "{label}: the CPU stages sum to {cpu_sum:.3} ms beside a {:.3} ms mean \
             frame — {:.3} ms of the measured frame is in no stage, so the \
             breakdown describes a frame this harness did not time",
            r.mean,
            (r.mean - cpu_sum).abs()
        );
        if m.passes.is_empty() {
            println!("{label} per-pass: unavailable (no timestamp queries on this device)");
        } else {
            let mut by_cost = m.passes.clone();
            by_cost.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!(
                "{label} GPU frame {:.3} ms; dearest passes:",
                m.gpu_frame_ms
            );
            for (n, ms, cpu) in by_cost.iter().take(12) {
                println!(
                    "{label}   {n:<16} {ms:7.3} ms  ({:5.1} % of the GPU frame)   \
                     record {cpu:6.3} ms",
                    ms / m.gpu_frame_ms.max(1.0e-9) * 100.0
                );
            }
            // Anti-vacuity: a report whose segments do not add up to the frame is
            // a report about a frame the renderer did not draw.
            let sum: f64 = m.passes.iter().map(|(_, ms, _)| ms).sum();
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
        // The wait AND the instrument's own readback come off: neither is work a
        // presenter's frame does.
        let submitted = cpu_sum - m.cpu_ms[4] - m.cpu_ms[5];
        let pipelined = submitted.max(m.gpu_frame_ms);
        println!(
            "{label} PIPELINED ESTIMATE {pipelined:.3} ms ({:.1} fps) = max(CPU without the wait or the stopwatch {submitted:.3}, GPU frame {:.3}) — an estimate, not a measurement; the asserted number is the serialized p95 above",
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

    // ── THE PRICE OF THE LIGHTING STACK (the I4 audit) ──────────────────────
    //
    // Everything above is the shipped default, and the shipped default has no
    // shadows in it. A constitution that says what "≥ 60 fps" means over a frame
    // with the expensive half of the renderer switched off would be quoted for
    // years as though it covered a lit frame, so the same content is run once
    // more at 1080p with the stack on and the difference is printed.
    //
    // Through the **authoring door** (`RenderSettingsRecord` → `shipped_settings`),
    // not by poking `RenderSettings`: that is what an author enabling shadows in
    // Project Settings produces, so the number is a number about a level somebody
    // could ship. VSM is the one exception — it has no authorable field, because
    // `VsmSettings::default().enabled` is a *code* default the tier applies over —
    // so it is set beside the record and named as such.
    //
    // **Reported, never asserted, and never folded into `worst_p95`.** The
    // ratchets are set from the shipped configuration; a second configuration
    // asserted against them would be a ceiling for a frame nobody has ratcheted.
    //
    // **Behind the adapter and CI gates, on purpose.** It is 480 more frames of a
    // GI + VSM + TAA frame, which on a software rasterizer is minutes rather than
    // seconds, and the two runners this repository has are software rasterizers.
    // A diagnostic nobody asserts must not be a CI cost.
    {
        let lit_record = inf_scene::RenderSettingsRecord {
            bloom_enabled: true,
            ssao_enabled: true,
            taa: true,
            shadows_enabled: true,
            gi_enabled: true,
            ..fx.record
        };
        let (mut lit, lit_tier) = shipped_settings(&gpu, lit_record);
        lit.vsm.enabled = true;
        let (w, h, label) = RESOLUTIONS[0];
        // **A tier below High clamps the stack straight back off** (`RenderTier::Low`
        // sets `shadows.enabled` and `gi.enabled` to `false`), and a price printed
        // for a configuration the tier refused is the price of nothing. Reported
        // rather than asserted, because "this adapter is Medium" is a fact about a
        // machine and a red build on one is the one-platform hazard P25 paid for.
        let clamped = !(lit.shadows.enabled && lit.gi.enabled);
        if clamped {
            println!(
                "\n{label} the lighting stack is clamped off at tier {lit_tier:?} \
                 (shadows {} / gi {}) — no price to print on this adapter, and the \
                 shipped ceilings below still bite",
                lit.shadows.enabled, lit.gi.enabled
            );
        }
        if !clamped {
            let m = measure(&gpu, &mut fx, w, h, lit);
            let r = m.round();
            let (base_p95, base_gpu) = shipped_1080;
            println!(
                "\n{label} WITH THE LIGHTING STACK ON (tier {lit_tier:?}; shadows {} / \
             gi {} / vsm {} / taa {} / ssao {} / bloom {}): p50 {:.3} ms ({:.1} fps) \
             | p95 {:.3} ms | p99 {:.3} ms | GPU frame {:.3} ms",
                lit.shadows.enabled,
                lit.gi.enabled,
                lit.vsm.enabled,
                lit.taa,
                lit.ssao.enabled,
                lit.bloom.enabled,
                r.p50,
                1000.0 / r.p50.max(1.0e-9),
                r.p95,
                r.p99,
                m.gpu_frame_ms,
            );
            // **The lit frame gets the SAME two tables the shipped one gets**
            // (island wave I4b). A configuration whose price is quoted as one
            // number cannot be optimised: I4b's first act was to attribute the
            // CPU frame and the sim step, and a lit frame that says only "p95
            // 64.9" sends the next reader to the GPU when a third of it may be
            // on the other processor.
            if let Some(v) = m.vsm.as_ref() {
                println!(
                    "{label} lit VSM raster: {} frames opened the page pass, \
                     {} page rectangles, {} indirect draws, {} casters \
                     ({} from scatter, {} terrain), {} deferred pages, {} \
                     dropped casters — i.e. {:.1} pages and {:.0} draws per \
                     rastering frame",
                    v.frames,
                    v.pages,
                    v.draws,
                    v.casters,
                    v.scatter_casters,
                    v.terrain_casters,
                    v.deferred_pages,
                    v.dropped_casters,
                    v.pages as f64 / v.frames.max(1) as f64,
                    v.draws as f64 / v.frames.max(1) as f64,
                );
                println!(
                    "{label} lit VSM group mask: {} indirect draws skipped \
                     ({:.0} per rastering frame) against {} issued; \
                     {} invalidation touches ({:.0} per frame)",
                    v.skipped_draws,
                    v.skipped_draws as f64 / v.frames.max(1) as f64,
                    v.draws,
                    v.invalidation_touches,
                    v.invalidation_touches as f64 / v.frames.max(1) as f64,
                );
            }
            let lit_cpu: f64 = m.cpu_ms.iter().sum();
            println!("{label} lit CPU frame {lit_cpu:.3} ms (a MEAN), stage by stage:");
            for (n, ms) in CPU_STAGE_NAMES.iter().zip(m.cpu_ms) {
                println!(
                    "{label}   lit {n:<16} {ms:7.3} ms  ({:5.1} % of the lit CPU frame)",
                    ms / lit_cpu.max(1.0e-9) * 100.0
                );
            }
            if !m.passes.is_empty() {
                let mut by_cost = m.passes.clone();
                by_cost.sort_by(|a, b| (b.1 + b.2).total_cmp(&(a.1 + a.2)));
                println!(
                    "{label} lit GPU frame {:.3} ms; every pass, with what it cost \
                     to RECORD it:",
                    m.gpu_frame_ms
                );
                for (n, ms, cpu) in by_cost
                    .iter()
                    .filter(|(_, ms, cpu)| *ms >= 0.0005 || *cpu >= 0.0005)
                {
                    println!(
                        "{label}   lit {n:<16} {ms:7.3} ms  ({:5.1} % of the lit \
                         GPU frame)   record {cpu:6.3} ms",
                        ms / m.gpu_frame_ms.max(1.0e-9) * 100.0
                    );
                }
            }
            let lit_submitted = lit_cpu - m.cpu_ms[4] - m.cpu_ms[5];
            println!(
                "{label} lit PIPELINED ESTIMATE {:.3} ms ({:.1} fps) = max(CPU without the wait or the stopwatch {lit_submitted:.3}, GPU frame {:.3})",
                lit_submitted.max(m.gpu_frame_ms),
                1000.0 / lit_submitted.max(m.gpu_frame_ms).max(1.0e-9),
                m.gpu_frame_ms
            );
            println!(
                "{label} THE STACK'S PRICE, same resolution and same content: p95 \
             {:.3} ms lit against {base_p95:.3} ms as shipped ({:+.3} ms), GPU \
             frame {:.3} ms against {base_gpu:.3} ms ({:+.3} ms). Reported, never \
             asserted — every ceiling in this file is set from the shipped \
             configuration, and this is what the shipped configuration is NOT \
             paying for.",
                r.p95,
                r.p95 - base_p95,
                m.gpu_frame_ms,
                m.gpu_frame_ms - base_gpu,
            );
            // Anti-vacuity: the two configurations really are different renderers.
            // A GPU frame that did not move is a price printed for nothing — and
            // unlike the tier clamp above, that is a defect rather than a machine.
            assert!(
                m.gpu_frame_ms > base_gpu,
                "the lit frame's GPU time ({:.3} ms) is no dearer than the shipped \
             one's ({base_gpu:.3} ms) — shadows, GI and VSM came back enabled and \
             cost nothing, so the price printed above is the price of nothing",
                m.gpu_frame_ms
            );
        }
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
