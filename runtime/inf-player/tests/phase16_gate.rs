//! **THE PHASE 16 GATE** (P16.6): world scale & streaming, composed.
//!
//! The `phase16-world` sample puts every piece of the phase in one level — a
//! wizard-imported streamed terrain (the P16.4a chunked importer, 8 m/sample,
//! three LOD levels, 8.2 km of world), a world partition on top of it (a 4 × 4
//! grid of 2 km cells plus a persistent cell), a **second, inline** terrain on a
//! different grid, and a `Walker` that is simultaneously the cell-streaming
//! `StreamingSource` and the terrain observer. This file is the gate over it.
//!
//! The five claims, and why each is the one that matters:
//!
//! * **(a) full determinism across two runs** — the sim trace, the cell
//!   activation timeline, and the terrain render-cut trace all reproduce byte for
//!   byte. Two streamers running in one step is exactly where a "whatever finished
//!   in time" dependency would hide.
//! * **(b) cooked == uncooked** — the cooked-pack path (cells and terrain tiles
//!   sliced out of the mmap'd pack) and the editor-document path (a loose
//!   `.inf_lvl` binned in memory + a loose `.inf_terrain`) produce the same world,
//!   the same traces, and the same **projected render scene** — including **two**
//!   terrains, with the ids the GPU cache keys on. That is PIE == shipping for
//!   multi-terrain, end to end.
//! * **(c) residency stays under the ceilings** for the whole run — terrain page
//!   bytes, cell blob bytes, and active cells, against the committed
//!   [`inf_player::budget`] constants (the §8 ratchet).
//! * **(d) the step-ms budget holds** while both streamers are live.
//! * **(e) the sim trace is invariant under the camera path AND the prefetch
//!   margin** — the composed doctrine proof. P16.3b2 proved the camera cannot
//!   reach a fixed step; P16.5 proved look-ahead cannot. Composed, *both* are
//!   varied against one scripted walk and the sim trace does not move.
//!
//! GPU rendering stays human-verified as everywhere else; what CI asserts is the
//! CPU-side `RenderScene` DTO, which is the whole input to the passes.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use glam::DVec3;

use inf_editor_core::samples::{
    phase16_camera_a, phase16_camera_b, phase16_inline_origin, phase16_prop_guid,
    phase16_walk_point, write_phase16_terrain_asset, PHASE16_ACTIVATION_RADIUS_M,
    PHASE16_CELL_SIZE_M, PHASE16_GRID, PHASE16_INLINE_TERRAIN_GUID, PHASE16_LEVEL_GUID,
    PHASE16_MANAGER_GUID, PHASE16_SUN_GUID, PHASE16_TERRAIN_ASSET_GUID, PHASE16_TERRAIN_GUID,
    PHASE16_WALKER_GUID,
};
use inf_packager::{cook, CookOptions, DEFAULT_PACK_NAME};
use inf_player::budget::{
    CELL_RESIDENT_BYTES_CEILING, CELL_RESIDENT_CEILING, RATCHET_NOTE, STREAMED_STEP_BUDGET_MS,
    TERRAIN_RESIDENT_BYTES_CEILING,
};
use inf_player::cell_stream::CellEvent;
use inf_player::level::{self, InfSceneWorldBuilder, PackLevelSource};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_player::vmesh::VmeshRegistry;
use inf_player::TerrainContent;
use inf_project::ProjectManifest;
use inf_render::RenderScene;
use inf_terrain::TileKey;
use uuid::Uuid;

/// Steps the scripted walk covers. At a third of a cell per step the walker
/// crosses the 8.2 km world roughly three times, so cells activate, deactivate and
/// re-activate, and the terrain's level-0 neighbourhood slides the whole way.
const STEPS: usize = 36;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn sample_dir() -> PathBuf {
    workspace_root().join("samples/phase16-world")
}

/// Scaffold a project with the gate level copied in and its `.inf_terrain`
/// **imported** beside it (the fixture setup runs the real wizard import path),
/// then cook it. Returns `(pack dir, loose content dir)`.
fn cook_phase16(tmp: &Path) -> (PathBuf, PathBuf) {
    cook_phase16_with(tmp, "proj", "out", None)
}

/// [`cook_phase16`], optionally rewriting the level's **prefetch margin** first —
/// the one knob gate (e) varies.
fn cook_phase16_with(
    tmp: &Path,
    proj_name: &str,
    out_name: &str,
    prefetch_margin_m: Option<f64>,
) -> (PathBuf, PathBuf) {
    let proj = tmp.join(proj_name);
    ProjectManifest::new("Phase 16 World", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = sample_dir();

    let lvl_path = content.join("Phase16World.inf_lvl");
    match prefetch_margin_m {
        None => {
            for f in ["Phase16World.inf_lvl", "Phase16World.inf_lvl.toml"] {
                std::fs::copy(src.join(f), content.join(f)).unwrap();
            }
        }
        Some(margin) => {
            let bytes = std::fs::read(src.join("Phase16World.inf_lvl")).unwrap();
            let mut lvl = inf_scene::decode(&bytes).unwrap();
            lvl.settings.partition.prefetch_margin_m = margin;
            let bytes = lvl.encode().unwrap();
            std::fs::write(&lvl_path, &bytes).unwrap();
            inf_asset::AssetSidecar::new(
                inf_asset::AssetId(PHASE16_LEVEL_GUID),
                inf_asset::AssetKind::Level,
                inf_asset::ContentHash::of(&bytes),
            )
            .save(&lvl_path)
            .unwrap();
        }
    }

    // The asset is derived, not committed — import it exactly as the wizard does.
    write_phase16_terrain_asset(&content).expect("the terrain imports");

    let out = tmp.join(out_name);
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    (out, content)
}

/// Build the world off a cooked pack (the shipping `--pack` path) and attach
/// **both** streamers, exactly as `inf_player::run_headless` does.
fn pack_sim(pack_dir: &Path) -> RuntimeSim {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("pack level builds");
    let partition = built.take_partition();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition);
    inf_player::attach_terrain_streaming(&mut sim, &TerrainContent::Pack(source));
    sim
}

/// Build the world off the loose `.inf_lvl` + loose `.inf_terrain` (the dev-dir /
/// editor-document path) and attach both streamers the same way.
fn loose_sim(content: &Path) -> RuntimeSim {
    let source = level::DevDirLevelSource::new(content.join("Phase16World.inf_lvl"));
    let builder = InfSceneWorldBuilder::with_defaults(Vec::new());
    let mut built = level::load(&source, &builder).expect("loose level builds");
    let partition = built.take_partition();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition);
    let index = level::terrain_paths_by_guid_from_dir(content);
    assert!(
        index.contains_key(&PHASE16_TERRAIN_ASSET_GUID),
        "the imported .inf_terrain must be discoverable by GUID"
    );
    inf_player::attach_terrain_streaming(&mut sim, &TerrainContent::Dir(index));
    sim
}

/// Place the Walker at the scripted world position for `step`.
///
/// Goes through `inf_ecs::sim::set_translation` (which marks the hierarchy dirty)
/// rather than writing `Transform` directly: `propagate` is dirty-gated, so a raw
/// write would leave `GlobalTransform` — the thing both want sets read — stale.
fn place_walker(sim: &mut RuntimeSim, step: usize) {
    let p = phase16_walk_point(step);
    let world = sim.world_mut();
    let e = world.entity_of(PHASE16_WALKER_GUID).expect("walker");
    inf_ecs::sim::set_translation(world, e, inf_ecs::Vec3d::new(p.x, p.y, p.z));
    world.propagate();
}

/// One step's **sim-observable** state. Every field is something the fixed step
/// can act on, so byte-equality of this trace across two camera paths (or two
/// prefetch margins) *is* the composed doctrine.
#[derive(Debug, Clone, PartialEq)]
struct SimSample {
    walker: [f64; 3],
    /// The real `terrain.height_at` answer under the walker, through the host seam.
    height: f64,
    /// Level-0 residency of the STREAMED terrain's sim working set.
    terrain_resident: Vec<(i32, i32)>,
    /// Active partition cells.
    cells_resident: Vec<(i32, i32)>,
    /// Authored props (by cell) whose `Guid` resolves in the live world.
    props: Vec<(i32, i32)>,
    /// Cross-cell references currently disconnected (P16.6).
    unresolved_refs: usize,
}

fn sim_sample(sim: &RuntimeSim, step: usize) -> SimSample {
    use inf_ecs::components::Terrain;
    let p = phase16_walk_point(step);
    let world = sim.world();
    let terrain_resident = world
        .entity_of(PHASE16_TERRAIN_GUID)
        .and_then(|e| world.world().get::<Terrain>(e))
        .map(|t| t.data.tiles().map(|(&c, _)| c).collect())
        .unwrap_or_default();
    let mut props = Vec::new();
    for cz in 0..PHASE16_GRID {
        for cx in 0..PHASE16_GRID {
            if world.entity_of(phase16_prop_guid(cx, cz)).is_some() {
                props.push((cx, cz));
            }
        }
    }
    SimSample {
        walker: [p.x, p.y, p.z],
        height: sim.terrain_height_at(p.x, p.z),
        terrain_resident,
        cells_resident: sim.cell_streaming().resident().collect(),
        props,
        unresolved_refs: sim.cell_streaming().stats().unresolved_refs,
    }
}

/// The published render cut of both terrains (`None` for a terrain that does not
/// stream — the inline one, which is always fully resident).
fn render_cuts(sim: &RuntimeSim) -> (BTreeSet<TileKey>, Option<BTreeSet<TileKey>>) {
    let streaming = sim.terrain_streaming();
    (
        streaming
            .cut(PHASE16_TERRAIN_GUID)
            .cloned()
            .unwrap_or_default(),
        streaming.cut(PHASE16_INLINE_TERRAIN_GUID).cloned(),
    )
}

/// One projected terrain reduced to bit-exact fingerprints: its renderer id, tile
/// resolution, and per-tile (lod, coord, `f64` origin bits, xxh3 of heights).
///
/// `version` is deliberately absent — it is a process-global GPU-cache stamp,
/// documented as *not* reproducible across runs.
type TerrainFingerprint = (u64, u32, Vec<(u32, (i32, i32), [i64; 3], u64)>);

/// The whole projected scene's terrain half — the multi-terrain claim, as data.
fn projected_terrains(sim: &RuntimeSim) -> Vec<TerrainFingerprint> {
    let mut scene = RenderScene::default();
    inf_player::render::project_scene(&mut scene, sim, 0.0, &VmeshRegistry::new());
    scene
        .terrains
        .iter()
        .map(|t| {
            (
                t.id,
                t.tile_resolution,
                t.tiles
                    .iter()
                    .map(|tile| {
                        (
                            tile.key.lod,
                            tile.key.coord,
                            [
                                tile.origin.x.to_bits() as i64,
                                tile.origin.y.to_bits() as i64,
                                tile.origin.z.to_bits() as i64,
                            ],
                            xxhash_rust::xxh3::xxh3_64(&f32_le(&tile.heights)),
                        )
                    })
                    .collect(),
            )
        })
        .collect()
}

/// Little-endian bytes of an `f32` slice (a portable, bit-exact fingerprint
/// input — no `bytemuck` dependency needed for a test).
fn f32_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    out
}

/// Peak residency observed over a run — what gate (c) asserts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Peaks {
    terrain_bytes: u64,
    cell_bytes: u64,
    cells: usize,
}

/// Everything one scripted run records.
struct RunTrace {
    sim: Vec<SimSample>,
    cells: Vec<CellEvent>,
    cuts: Vec<BTreeSet<TileKey>>,
    frames: Vec<Vec<TerrainFingerprint>>,
    hash: u128,
    peaks: Peaks,
    /// Mean fixed-step wall time, milliseconds.
    mean_step_ms: f64,
}

/// Run the scripted sim + a scripted camera, collecting every trace the gates
/// compare. `camera` is what the render-sync point is fed each frame.
fn run_scripted(mut sim: RuntimeSim, camera: impl Fn(usize) -> DVec3) -> RunTrace {
    let mut trace = RunTrace {
        sim: Vec::with_capacity(STEPS),
        cells: Vec::new(),
        cuts: Vec::with_capacity(STEPS),
        frames: Vec::with_capacity(STEPS),
        hash: 0,
        peaks: Peaks::default(),
        mean_step_ms: 0.0,
    };
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    let mut step_time = std::time::Duration::ZERO;

    for step in 0..STEPS {
        // ── the fixed step (both streamers reconcile at its top) ──
        place_walker(&mut sim, step);
        let t0 = std::time::Instant::now();
        sim.step_once(RuntimeInput::default());
        step_time += t0.elapsed();

        trace.sim.push(sim_sample(&sim, step));
        hasher.update(&sim.state_bytes());

        // ── the render-sync point (camera wants, the OTHER working set) ──
        sim.sync_render_terrain(camera(step));
        let (streamed_cut, inline_cut) = render_cuts(&sim);
        assert!(
            inline_cut.is_none(),
            "the inline terrain must not stream — it has no asset"
        );
        trace.cuts.push(streamed_cut);
        trace.frames.push(projected_terrains(&sim));

        let tstats = sim.terrain_streaming().stats();
        let cstats = sim.cell_streaming().stats();
        trace.peaks.terrain_bytes = trace.peaks.terrain_bytes.max(tstats.bytes_resident);
        trace.peaks.cell_bytes = trace.peaks.cell_bytes.max(cstats.bytes_resident);
        trace.peaks.cells = trace.peaks.cells.max(cstats.cells_resident);
    }
    trace.cells = sim.cell_streaming().events().to_vec();
    trace.hash = hasher.digest128();
    trace.mean_step_ms = step_time.as_secs_f64() * 1000.0 / STEPS as f64;
    trace
}

// ── GATE (a) — full determinism across two runs ─────────────────────────────

/// Two independent headless runs of the identical scripted walk + camera path
/// reproduce a byte-identical **sim trace**, **cell activation timeline** and
/// **terrain cut trace** — and the projected render scene with them.
///
/// This is where a "whatever finished in time" dependency would show up: two
/// streamers reconcile at the top of every step, one of them bounded by a
/// per-sync load batch, and neither may let *how much work got done* decide
/// *what the world is*.
#[test]
fn the_composed_scene_is_fully_deterministic_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_phase16(dir.path());

    let a = run_scripted(pack_sim(&pack_dir), phase16_camera_a);
    let b = run_scripted(pack_sim(&pack_dir), phase16_camera_a);

    assert_eq!(a.sim, b.sim, "the sim trace is not reproducible");
    assert_eq!(a.cells, b.cells, "the cell timeline is not reproducible");
    assert_eq!(a.cuts, b.cuts, "the terrain cut trace is not reproducible");
    assert_eq!(
        a.frames, b.frames,
        "the projected scene is not reproducible"
    );
    assert_eq!(a.hash, b.hash, "the sim determinism hash diverged");

    // The run has to actually stream, on BOTH axes, or the assertions are vacuous.
    let cut_sizes: BTreeSet<usize> = a.cuts.iter().map(|c| c.len()).collect();
    assert!(cut_sizes.len() > 1, "the terrain cut never changed");
    let churn: usize = a
        .cuts
        .windows(2)
        .map(|w| w[0].symmetric_difference(&w[1]).count())
        .sum();
    assert!(churn > 20, "only {churn} page transitions — nothing paged");
    assert!(
        a.cells.iter().filter(|e| e.activated).count() >= PHASE16_GRID as usize,
        "the walk activated {} cell(s) — it never crossed the grid",
        a.cells.iter().filter(|e| e.activated).count()
    );
    assert!(
        a.cells.iter().any(|e| !e.activated),
        "no cell ever deactivated — residency is not bounded"
    );
    // Coarse pyramid pages really do serve the distance (the pyramid is live).
    assert!(a.cuts.iter().any(|c| c.iter().any(|k| k.lod > 0)));
    assert!(a.cuts.iter().any(|c| c.iter().any(|k| k.lod == 0)));
    // …and the sim genuinely read streamed ground.
    assert!(
        a.sim.iter().any(|s| s.height.abs() > 1.0),
        "every height read ~0 — the sim never saw a streamed tile"
    );
}

// ── GATE (b) — cooked == uncooked, with TWO terrains ────────────────────────

/// The cooked-pack path and the editor-document path stream the same world.
///
/// The shipping arm slices cells and terrain tiles **zero-copy out of the mmap'd
/// pack**; the editor arm bins the loose `.inf_lvl`'s own entities in memory and
/// reads the loose `.inf_terrain` off disk. Both must produce the identical sim
/// trace, cell timeline, terrain cut and **projected render scene** — the last of
/// which carries *two* terrains, each with the renderer id its GPU cache keys on.
#[test]
fn pie_equals_shipping_with_two_terrains() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, content) = cook_phase16(dir.path());

    let ship = run_scripted(pack_sim(&pack_dir), phase16_camera_a);
    let edit = run_scripted(loose_sim(&content), phase16_camera_a);

    assert_eq!(edit.sim, ship.sim, "editor sim trace != shipping");
    assert_eq!(edit.cells, ship.cells, "editor cell timeline != shipping");
    assert_eq!(edit.cuts, ship.cuts, "editor terrain cut != shipping");
    assert_eq!(edit.hash, ship.hash, "editor determinism hash != shipping");
    assert_eq!(
        edit.frames, ship.frames,
        "editor projected scene != shipping — multi-terrain is not PIE == shipping"
    );

    // ── the multi-terrain claim, as data ──
    let last = ship.frames.last().expect("frames recorded");
    assert_eq!(last.len(), 2, "the frame must carry BOTH terrains");
    let ids: BTreeSet<u64> = last.iter().map(|(id, _, _)| *id).collect();
    assert_eq!(ids.len(), 2, "the two terrains share a renderer id");
    assert!(!ids.contains(&0), "a projected terrain must be keyed");
    assert_eq!(
        ids,
        [
            inf_render::terrain_id_from_guid(PHASE16_TERRAIN_GUID.as_u128()),
            inf_render::terrain_id_from_guid(PHASE16_INLINE_TERRAIN_GUID.as_u128()),
        ]
        .into_iter()
        .collect::<BTreeSet<u64>>(),
        "the ids must be derived from the terrain entities' GUIDs"
    );
    // Different grids, and both non-empty — a shared cache key would collide.
    let resolutions: BTreeSet<u32> = last.iter().map(|(_, res, _)| *res).collect();
    assert_eq!(resolutions.len(), 2, "the two terrains share a resolution");
    for (id, _, tiles) in last {
        assert!(!tiles.is_empty(), "terrain {id} projected no tiles");
    }
    // The inline terrain sits where it was authored (its own world anchor).
    let inline_id = inf_render::terrain_id_from_guid(PHASE16_INLINE_TERRAIN_GUID.as_u128());
    let (_, _, inline_tiles) = last.iter().find(|(id, _, _)| *id == inline_id).unwrap();
    let origin_z = f64::from_bits(inline_tiles[0].2[2] as u64);
    assert!(
        (origin_z - phase16_inline_origin().z).abs() < 1.0,
        "the inline terrain is not at its authored origin ({origin_z})"
    );

    // The cook really did ship both a partition and an uncompressed terrain.
    let reader = inf_asset::PackReader::open(&pack_dir.join(DEFAULT_PACK_NAME)).unwrap();
    let terrain_id = inf_asset::AssetId(PHASE16_TERRAIN_ASSET_GUID);
    let entry = reader
        .entry(terrain_id)
        .expect("the .inf_terrain is packed");
    assert_eq!(entry.kind, inf_asset::AssetKind::Terrain);
    assert!(
        !entry.compressed,
        "terrain is streaming-class: a compressed entry can never be a zero-copy read"
    );
    // The level's asset id comes from the project database (the committed sample
    // ships a *scene* sidecar, not an `inf_asset` one, so the DB synthesizes an
    // id for it); the player derives the partition's GUID from that, with no side
    // index — which is exactly what is asserted here.
    let level_id = reader
        .index()
        .find(|e| e.kind == inf_asset::AssetKind::Level)
        .expect("the pack has a level")
        .guid;
    let part_id = inf_asset::AssetId(inf_player::cell_stream::derived_partition_id(
        level_id.uuid(),
    ));
    assert!(reader.contains(part_id), "the cook emitted no .inf_part");
}

// ── GATE (c) + (d) — residency ceilings and the step-ms budget ──────────────

/// Residency stays under the committed ceilings for the whole flythrough, and the
/// mean fixed step stays under its budget while both streamers are live
/// ([`inf_player::budget`], the §8 ratchet).
///
/// Measured values are printed on every run: that is how the ratchet gets tightened
/// — read the number, lower the constant, never the other way.
#[test]
fn streamed_scene_budgets_hold() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_phase16(dir.path());

    // Warm up one run so the timed one is not paying first-touch page faults on
    // the mapping, then measure.
    let _ = run_scripted(pack_sim(&pack_dir), phase16_camera_a);
    let t = run_scripted(pack_sim(&pack_dir), phase16_camera_a);

    eprintln!(
        "phase16 budgets: step mean {:.4} ms (budget {STREAMED_STEP_BUDGET_MS}), \
         terrain {:.2} MiB (ceiling {:.0} MiB), cells {:.2} KiB / {} active \
         (ceilings {:.0} KiB / {CELL_RESIDENT_CEILING})",
        t.mean_step_ms,
        t.peaks.terrain_bytes as f64 / (1024.0 * 1024.0),
        TERRAIN_RESIDENT_BYTES_CEILING as f64 / (1024.0 * 1024.0),
        t.peaks.cell_bytes as f64 / 1024.0,
        t.peaks.cells,
        CELL_RESIDENT_BYTES_CEILING as f64 / 1024.0,
    );

    // (c) residency ceilings, over the WHOLE run (peaks, not the final frame).
    assert!(
        t.peaks.terrain_bytes <= TERRAIN_RESIDENT_BYTES_CEILING,
        "terrain residency peaked at {} B, over the {TERRAIN_RESIDENT_BYTES_CEILING} B \
         ceiling {RATCHET_NOTE}",
        t.peaks.terrain_bytes
    );
    assert!(
        t.peaks.cell_bytes <= CELL_RESIDENT_BYTES_CEILING,
        "cell residency peaked at {} B, over the {CELL_RESIDENT_BYTES_CEILING} B \
         ceiling {RATCHET_NOTE}",
        t.peaks.cell_bytes
    );
    assert!(
        t.peaks.cells <= CELL_RESIDENT_CEILING,
        "{} cells were active at once, over the {CELL_RESIDENT_CEILING} ceiling {RATCHET_NOTE}",
        t.peaks.cells
    );

    // (d) the fixed-step budget, with both streamers reconciling at its top.
    assert!(
        t.mean_step_ms < STREAMED_STEP_BUDGET_MS,
        "streamed fixed step mean {:.4} ms exceeded the {STREAMED_STEP_BUDGET_MS} ms \
         budget {RATCHET_NOTE}",
        t.mean_step_ms
    );

    // The ceilings only mean something if residency was really moving.
    assert!(t.peaks.terrain_bytes > 0 && t.peaks.cell_bytes > 0);
    assert!(t.peaks.cells > 0);
    let stats = sim_stats(&pack_dir);
    assert!(stats.0 > 0 && stats.1 > 0, "terrain never loaded/evicted");
    assert_eq!(stats.2, 0, "a terrain page failed to decode");
}

/// `(terrain loads, terrain evictions, terrain failures)` after one scripted run —
/// the "residency really moved" sanity beside the ceilings.
fn sim_stats(pack_dir: &Path) -> (u64, u64, u64) {
    let mut sim = pack_sim(pack_dir);
    for step in 0..STEPS {
        place_walker(&mut sim, step);
        sim.step_once(RuntimeInput::default());
        sim.sync_render_terrain(phase16_camera_a(step));
    }
    let s = sim.terrain_streaming().stats();
    (s.loads, s.evictions, s.failed)
}

// ── GATE (e) — the composed doctrine proof ─────────────────────────────────

/// **The headline gate.** One scripted walk, run under two completely different
/// **camera paths** and two completely different **prefetch margins**, produces a
/// byte-identical sim trace and cell timeline.
///
/// P16.3b2 proved a camera cannot reach a fixed step (its pages land in a working
/// set the world holds no reference to). P16.5 proved look-ahead cannot (loading
/// early is allowed; activating late is not, and a cell that reaches its
/// activation step unloaded blocks). Composed in one scene, with both streamers
/// reconciling at the top of the same step, *neither* may move a result — and the
/// two knobs are varied together to prove it.
#[test]
fn the_sim_trace_ignores_both_the_camera_and_the_prefetch_margin() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_phase16(dir.path());

    // Camera A vs camera B over the same cooked pack.
    let a = run_scripted(pack_sim(&pack_dir), phase16_camera_a);
    let b = run_scripted(pack_sim(&pack_dir), phase16_camera_b);
    assert_eq!(
        a.sim, b.sim,
        "the simulation observed the camera — the determinism doctrine is broken"
    );
    assert_eq!(a.cells, b.cells, "the camera moved the cell timeline");
    assert_eq!(a.hash, b.hash, "the determinism hash depends on the camera");
    // …and the two camera paths really did drive different terrain residency.
    assert_ne!(a.cuts, b.cuts, "the two camera paths produced the same cut");

    // Prefetch 0 (every activation blocks) vs a wide margin, camera A both times.
    let none = cook_phase16_with(dir.path(), "proj-p0", "out-p0", Some(0.0)).0;
    let wide = cook_phase16_with(dir.path(), "proj-pw", "out-pw", Some(6144.0)).0;
    let c = run_scripted(pack_sim(&none), phase16_camera_a);
    let d = run_scripted(pack_sim(&wide), phase16_camera_a);

    assert_eq!(
        c.sim, d.sim,
        "the prefetch margin moved the simulation — look-ahead reached a fixed step"
    );
    assert_eq!(c.cells, d.cells, "the prefetch margin moved activations");
    assert_eq!(c.hash, d.hash, "the determinism hash depends on look-ahead");
    // …and it is the same simulation the unmodified level runs.
    assert_eq!(a.sim, c.sim, "changing only the margin changed the world");

    // The margin genuinely did different WORK (otherwise the test proves nothing).
    let blocking = |dir: &Path| {
        let mut sim = pack_sim(dir);
        for step in 0..STEPS {
            place_walker(&mut sim, step);
            sim.step_once(RuntimeInput::default());
        }
        sim.cell_streaming().stats().blocking_loads
    };
    let (cold, warm) = (blocking(&none), blocking(&wide));
    assert!(cold > 0, "the zero-margin run must actually block");
    assert!(
        warm < cold,
        "prefetch bought nothing: {warm} vs {cold} blocking loads"
    );
}

// ── the composed scene's own invariants ────────────────────────────────────

/// The persistent cell really is persistent, streamed props really are not, and
/// **no cross-cell reference is left dangling** by the authored scene.
///
/// `unresolved_refs` (P16.6) is the runtime face of the cook's `cross_cell_refs`
/// warning: it counts the `AttachedTo`/joint links that are disconnected right
/// now. A gate scene that trips it would be teaching the wrong pattern.
#[test]
fn the_persistent_cell_survives_and_no_reference_dangles() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_phase16(dir.path());
    let t = run_scripted(pack_sim(&pack_dir), phase16_camera_a);

    for s in &t.sim {
        assert_eq!(
            s.unresolved_refs, 0,
            "the gate scene left a cross-cell reference disconnected"
        );
    }

    // Both terrains, the sun, the manager and the walker exist at every step;
    // the props come and go.
    let sim = pack_sim(&pack_dir);
    let world = sim.world();
    for guid in [
        PHASE16_TERRAIN_GUID,
        PHASE16_INLINE_TERRAIN_GUID,
        PHASE16_SUN_GUID,
        PHASE16_MANAGER_GUID,
        PHASE16_WALKER_GUID,
    ] {
        assert!(
            world.entity_of(guid).is_some(),
            "{guid} must be in the persistent cell"
        );
    }
    let counts: BTreeSet<usize> = t.sim.iter().map(|s| s.props.len()).collect();
    assert!(
        counts.len() > 1,
        "the props never streamed — every one stayed resident"
    );
    assert!(
        t.sim.iter().all(|s| s.props.len() <= CELL_RESIDENT_CEILING),
        "more props were resident than the cell ceiling allows"
    );

    // Terrain sim residency stays a neighbourhood of the walker, not the world.
    let max_terrain = t
        .sim
        .iter()
        .map(|s| s.terrain_resident.len())
        .max()
        .unwrap();
    assert!(
        max_terrain <= 36,
        "the sim paged {max_terrain} level-0 tiles — that is not a neighbourhood"
    );

    // Every resident cell is within the activation radius of the walker.
    for s in &t.sim {
        for &c in &s.cells_resident {
            let d = inf_player::cell_stream::distance_to_cell(
                c,
                s.walker[0],
                s.walker[2],
                PHASE16_CELL_SIZE_M,
            );
            assert!(
                d <= PHASE16_ACTIVATION_RADIUS_M + 1e-9,
                "cell {c:?} is {d} m from the walker, beyond the {PHASE16_ACTIVATION_RADIUS_M} m \
                 activation radius"
            );
        }
    }
}

/// Sanity on the committed level: it stays small (the terrain is an asset and the
/// entities move into the `.inf_part`), and its two terrain entities are exactly
/// one streamed + one inline.
#[test]
fn the_committed_level_is_small_and_carries_two_terrains() {
    let bytes = std::fs::read(sample_dir().join("Phase16World.inf_lvl")).unwrap();
    assert!(
        bytes.len() < 64 * 1024,
        "the gate level must stay small, got {} bytes",
        bytes.len()
    );
    let level = inf_scene::decode(&bytes).expect("level decodes");
    let terrains: Vec<_> = level
        .entities
        .iter()
        .filter_map(|e| e.terrain.as_ref().map(|t| (e.guid, t)))
        .collect();
    assert_eq!(terrains.len(), 2);
    let streamed = terrains
        .iter()
        .find(|(g, _)| *g == PHASE16_TERRAIN_GUID)
        .unwrap()
        .1;
    assert_eq!(streamed.asset, Some(PHASE16_TERRAIN_ASSET_GUID));
    assert!(
        streamed.data.is_empty(),
        "the streamed terrain ships no tiles"
    );
    let inline = terrains
        .iter()
        .find(|(g, _)| *g == PHASE16_INLINE_TERRAIN_GUID)
        .unwrap()
        .1;
    assert!(inline.asset.is_none());
    assert!(
        !inline.data.is_empty(),
        "the inline terrain ships its tiles"
    );
    let _: Uuid = PHASE16_LEVEL_GUID;
}
