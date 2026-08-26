//! **P16.3b2 gate**: camera-driven terrain streaming, and the determinism
//! doctrine that governs it.
//!
//! The `streamed-terrain` sample is a 256 m × 256 m paged heightfield that lives
//! entirely in a `.inf_terrain` asset — the level ships an *empty* working set
//! plus the asset ref. These integration tests prove the four things the batch
//! promised:
//!
//! * **(a) cook** — the cook follows the `Terrain.asset` edge and ships the
//!   `.inf_terrain` **uncompressed** (streaming-class), so a runtime pages one
//!   tile straight out of the mmap'd pack.
//! * **(b) headless determinism** — a headless run over a scripted camera path
//!   reproduces a byte-identical *resident-set* trace **and** a byte-identical
//!   *rendered-frame* trace (the projected `RenderTerrain` DTO — the whole input
//!   to the terrain pass) across two independent runs.
//! * **(c) SIM DETERMINISM vs CAMERA** — the headline. The same scripted sim
//!   (a Walker crossing the terrain, probing the real `terrain.height_at` host
//!   seam every step) run under two *completely different* camera paths produces
//!   a **byte-identical sim trace**. If camera-driven residency could ever reach
//!   the fixed step, this is the test that fails.
//! * **(d) PIE == shipping** — the cooked-pack path (zero-copy tiles out of the
//!   pack mapping) and the editor-document path (a loose `.inf_terrain`) stream
//!   the same world to the same sim trace and the same resident set.
//!
//! GPU rendering stays human-verified as everywhere else; (b)'s "rendered frame"
//! is the CPU-side DTO the pass consumes, which is what determinism can actually
//! be asserted about in CI.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{MeshRef, Terrain};
use inf_editor_core::samples::{
    streamed_terrain_camera_a, streamed_terrain_camera_b, streamed_terrain_height,
    streamed_terrain_walk_point, write_streamed_terrain_asset, STREAMED_TERRAIN_ASSET_GUID,
    STREAMED_TERRAIN_TERRAIN_GUID, STREAMED_TERRAIN_WALKER_GUID,
};
use inf_packager::{cook, CookOptions, DEFAULT_PACK_NAME};
use inf_player::level::{self, InfSceneWorldBuilder, PackLevelSource};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_player::terrain_stream::TerrainStreaming;
use inf_player::TerrainContent;
use inf_project::ProjectManifest;
use inf_terrain::{StreamBudget, TileKey};

/// How many frames the scripted runs cover. Long enough for the walk to cross
/// several tiles and for the camera cut to page in and out repeatedly.
const FRAMES: usize = 90;

/// Tolerance when comparing a `height_at` answer to the analytic generator at an
/// **arbitrary** (non-grid) point.
///
/// `height_at` bilinearly interpolates the stored samples, which sit 2 m apart,
/// while the generator is a smooth trig surface — so the two differ by the
/// interpolation error of a chord across one cell (≈ `h²/8 · f″`, a few
/// centimetres here). This is a "did the sim read real streamed ground" sanity
/// bound, not an equality claim; the exact-at-a-sample-point claim is asserted in
/// [`cook_ships_the_terrain_asset_uncompressed`] against a grid-aligned probe.
const HEIGHT_TOLERANCE_M: f64 = 0.2;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn sample_dir() -> PathBuf {
    workspace_root().join("samples/streamed-terrain")
}

/// Scaffold a project with the streamed-terrain level copied in and its
/// `.inf_terrain` **generated** beside it (the fixture setup), then cook it.
fn cook_streamed_terrain(tmp: &Path) -> (PathBuf, PathBuf) {
    let proj = tmp.join("proj");
    ProjectManifest::new("Streamed Terrain", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = sample_dir();
    for f in ["StreamedTerrain.inf_lvl", "StreamedTerrain.inf_lvl.toml"] {
        std::fs::copy(src.join(f), content.join(f)).unwrap();
    }
    // The asset is derived, not committed — generate it exactly as the P16.4
    // import wizard will.
    write_streamed_terrain_asset(&content).expect("terrain asset writes");

    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    (out, content)
}

/// Build the world off a cooked pack (the shipping `--pack` path) and attach
/// streaming, exactly as `inf_player::run_headless` does.
fn pack_sim(pack_dir: &Path) -> RuntimeSim {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("pack level builds");
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_terrain_streaming(&mut sim, &TerrainContent::Pack(source));
    sim
}

/// Build the world off the loose level + loose `.inf_terrain` (the dev-dir /
/// editor-document path) and attach streaming the same way.
fn loose_sim(content: &Path) -> RuntimeSim {
    let source = level::DevDirLevelSource::new(content.join("StreamedTerrain.inf_lvl"));
    let builder = InfSceneWorldBuilder::with_defaults(Vec::new());
    let built = level::load(&source, &builder).expect("loose level builds");
    let mut sim = inf_player::sim_from_built(built);
    let index = level::terrain_paths_by_guid_from_dir(content);
    assert!(
        index.contains_key(&STREAMED_TERRAIN_ASSET_GUID),
        "the fixture's .inf_terrain must be discoverable by GUID"
    );
    inf_player::attach_terrain_streaming(&mut sim, &TerrainContent::Dir(index));
    sim
}

/// Place the Walker at the scripted world position for `step`.
///
/// Goes through `inf_ecs::set_translation` (which marks the hierarchy dirty)
/// rather than writing `Transform` directly: `propagate` is dirty-gated, so a raw
/// write would leave `GlobalTransform` — the thing `sim_wants` reads — stale.
fn place_walker(sim: &mut RuntimeSim, step: usize) {
    let p = streamed_terrain_walk_point(step);
    let world = sim.world_mut();
    let e = world
        .entity_of(STREAMED_TERRAIN_WALKER_GUID)
        .expect("walker");
    inf_ecs::sim::set_translation(world, e, inf_ecs::Vec3d::new(p.x, p.y, p.z));
    world.propagate();
    debug_assert_eq!(
        world
            .world()
            .get::<inf_ecs::components::GlobalTransform>(e)
            .map(|g| g.translation().x),
        Some(p.x)
    );
}

/// One step's **sim-observable** state: the walker's position, the real
/// `terrain.height_at` answer under it, and the sim's level-0 residency.
///
/// Every field is something the fixed step can act on, so byte-equality of this
/// trace across two camera paths *is* the doctrine.
#[derive(Debug, Clone, PartialEq)]
struct SimSample {
    walker: [f64; 3],
    height: f64,
    resident: Vec<(i32, i32)>,
}

/// Level-0 residency of the **component's** (sim's) working set.
fn sim_resident(sim: &RuntimeSim) -> Vec<(i32, i32)> {
    let w = sim.world();
    let e = w.entity_of(STREAMED_TERRAIN_TERRAIN_GUID).expect("terrain");
    let t = w.world().get::<Terrain>(e).expect("terrain component");
    t.data.tiles().map(|(&c, _)| c).collect()
}

/// The **render**-resident cut published for the streamed terrain.
fn render_cut(sim: &RuntimeSim) -> BTreeSet<TileKey> {
    sim.terrain_streaming()
        .cut(STREAMED_TERRAIN_TERRAIN_GUID)
        .cloned()
        .unwrap_or_default()
}

/// One projected terrain tile, reduced to bit-exact fingerprints: LOD, grid
/// coordinate, `f64` origin bits, and xxh3 of the heights + weights.
///
/// `version` is deliberately absent — it is a process-global GPU-cache stamp,
/// documented as *not* reproducible across runs.
type FrameTile = (u32, (i32, i32), [i64; 3], u64, u64);

/// One frame's projected terrain: the whole input the terrain pass consumes.
type FrameTrace = Vec<FrameTile>;

/// Everything one scripted run records.
type RunTrace = (
    Vec<SimSample>,
    Vec<BTreeSet<TileKey>>,
    Vec<FrameTrace>,
    u128,
);

/// The projected `RenderTerrain` reduced to its **reproducible** content.
fn rendered_frame(sim: &RuntimeSim) -> FrameTrace {
    let w = sim.world();
    let Some(e) = w.entity_of(STREAMED_TERRAIN_TERRAIN_GUID) else {
        return Vec::new();
    };
    let Some(terrain) = w.world().get::<Terrain>(e) else {
        return Vec::new();
    };
    let data = sim
        .terrain_streaming()
        .render_data(STREAMED_TERRAIN_TERRAIN_GUID)
        .unwrap_or(&terrain.data);
    // Unkeyed (`0`) and with nothing to carry: this fingerprints a COLD
    // projection on purpose — the whole point is that two runs build the same
    // bytes from the same world, which a memo would hide rather than prove.
    let dto =
        inf_player::render::project_terrain(terrain, data, DVec3::ZERO, 0, None, &mut Vec::new());
    dto.tiles
        .iter()
        .map(|t| {
            (
                t.key.lod,
                t.key.coord,
                [
                    t.origin.x.to_bits() as i64,
                    t.origin.y.to_bits() as i64,
                    t.origin.z.to_bits() as i64,
                ],
                // Content fingerprints (bit-exact, order-preserving).
                xxhash_rust::xxh3::xxh3_64(&bytemuck_le(&t.heights)),
                xxhash_rust::xxh3::xxh3_64(&t.weights.concat()),
            )
        })
        .collect()
}

/// Little-endian bytes of an `f32` slice (a portable, bit-exact fingerprint
/// input — no `bytemuck` dependency needed for a test).
fn bytemuck_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    out
}

/// Run the scripted sim + a scripted camera, collecting every trace the gates
/// compare. `camera` is what the render-sync point is fed each frame.
fn run_scripted(mut sim: RuntimeSim, camera: impl Fn(usize) -> DVec3) -> RunTrace {
    let mut sim_trace = Vec::with_capacity(FRAMES);
    let mut cut_trace = Vec::with_capacity(FRAMES);
    let mut frame_trace = Vec::with_capacity(FRAMES);
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    for step in 0..FRAMES {
        // ── the fixed step (sim wants load at its top, synchronously) ──
        place_walker(&mut sim, step);
        sim.step_once(RuntimeInput::default());
        let p = streamed_terrain_walk_point(step);
        sim_trace.push(SimSample {
            walker: [p.x, p.y, p.z],
            height: sim.terrain_height_at(p.x, p.z),
            resident: sim_resident(&sim),
        });
        hasher.update(&sim.state_bytes());

        // ── the render-sync point (camera wants, the OTHER working set) ──
        sim.sync_render_terrain(camera(step));
        cut_trace.push(render_cut(&sim));
        frame_trace.push(rendered_frame(&sim));
    }
    (sim_trace, cut_trace, frame_trace, hasher.digest128())
}

// ── GATE (a) ────────────────────────────────────────────────────────────────

/// The cook follows the `Terrain.asset` edge and ships the `.inf_terrain`
/// **uncompressed**, so a streaming runtime can slice one tile out of the
/// mapping with no decode and no copy.
#[test]
fn cook_ships_the_terrain_asset_uncompressed() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_streamed_terrain(dir.path());

    let reader = inf_asset::PackReader::open(&pack_dir.join(DEFAULT_PACK_NAME)).unwrap();
    let id = inf_asset::AssetId(STREAMED_TERRAIN_ASSET_GUID);
    assert!(
        reader.contains(id),
        "the cook must pull the .inf_terrain through the level→terrain edge"
    );
    let entry = reader.entry(id).expect("indexed");
    assert_eq!(entry.kind, inf_asset::AssetKind::Terrain);
    assert!(
        !entry.compressed,
        "terrain is streaming-class: a compressed entry can never be a zero-copy read"
    );
    assert_eq!(
        entry.stored_len, entry.uncompressed_len,
        "an uncompressed entry stores exactly its payload"
    );

    // …and the payload really is a valid `.inf_terrain` with a pyramid, readable
    // straight out of the mapping.
    let store = inf_terrain::PackTileStore::open(std::sync::Arc::new(reader), id)
        .expect("pack tile store opens");
    assert!(
        store.header().lod_levels >= 3,
        "level 0 + ≥ 2 coarse levels"
    );
    let tile = inf_terrain::TileStore::load_tile(&store, TileKey::lod0((3, 4)))
        .expect("decodes")
        .expect("present");
    // A sanity probe against the generator: sample (0,0) of tile (3,4).
    let span = (store.header().tile_resolution as f64 - 1.0) * store.header().meters_per_sample;
    let want = streamed_terrain_height(3.0 * span, 4.0 * span);
    assert!(
        (tile.world_height(store.header().tile_resolution, 0, 0) - want).abs() < 1e-3,
        "packed tile content diverged from the generator"
    );
}

// ── GATE (b) ────────────────────────────────────────────────────────────────

/// Two independent headless runs over the same scripted camera path publish the
/// identical resident-set trace and the identical rendered-frame trace.
///
/// This is the property that timing-driven asynchrony could not have: the
/// per-frame load batch is bounded but its *membership* is a pure function of the
/// want set, never of what finished in time.
#[test]
fn headless_streaming_is_reproducible_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_streamed_terrain(dir.path());

    let (sim_a, cut_a, frame_a, hash_a) =
        run_scripted(pack_sim(&pack_dir), streamed_terrain_camera_a);
    let (sim_b, cut_b, frame_b, hash_b) =
        run_scripted(pack_sim(&pack_dir), streamed_terrain_camera_a);

    assert_eq!(cut_a, cut_b, "the resident-set trace is not reproducible");
    assert_eq!(
        frame_a, frame_b,
        "the rendered-frame trace is not reproducible"
    );
    assert_eq!(sim_a, sim_b);
    assert_eq!(hash_a, hash_b, "the sim determinism hash diverged");

    // The run has to actually stream, or the assertions above are vacuous.
    let sizes: BTreeSet<usize> = cut_a.iter().map(|c| c.len()).collect();
    assert!(sizes.len() > 1, "the cut never changed — nothing streamed");
    let churn: usize = cut_a
        .windows(2)
        .map(|w| w[0].symmetric_difference(&w[1]).count())
        .sum();
    assert!(
        churn > 20,
        "only {churn} page transitions — camera isn't paging"
    );
    // The cut really is a quadtree cut at every frame: no page coexists with an
    // ancestor (the B2 contract the terrain pass documents).
    for cut in &cut_a {
        for &k in cut {
            let mut anc = k;
            for _ in 0..8 {
                anc = anc.parent();
                assert!(!cut.contains(&anc), "{k:?} coexists with ancestor {anc:?}");
            }
        }
        // …and never three-of-four children beside their parent.
        for &k in cut {
            if k.lod == 0 {
                continue;
            }
            let n = k.children().iter().filter(|c| cut.contains(c)).count();
            assert_eq!(n, 0, "{k:?} is resident beside {n} of its children");
        }
    }
    // Coarse pages really are being used for distance (the pyramid is live).
    assert!(cut_a.iter().any(|c| c.iter().any(|k| k.lod > 0)));
    assert!(cut_a.iter().any(|c| c.iter().any(|k| k.lod == 0)));
}

// ── GATE (c) — the doctrine ─────────────────────────────────────────────────

/// **The headline gate.** The identical scripted simulation, run under two
/// completely different camera paths, produces a byte-identical sim trace: the
/// same walker positions, the same `terrain.height_at` answers through the real
/// host seam, the same level-0 residency, and the same determinism hash.
///
/// Camera-driven residency lands in a working set the ECS world holds no
/// reference to, so there is no path by which it could reach a fixed step. This
/// test is what turns that structural claim into a checked one.
#[test]
fn sim_trace_is_identical_under_two_different_camera_paths() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_streamed_terrain(dir.path());

    let (sim_a, cut_a, _, hash_a) = run_scripted(pack_sim(&pack_dir), streamed_terrain_camera_a);
    let (sim_b, cut_b, _, hash_b) = run_scripted(pack_sim(&pack_dir), streamed_terrain_camera_b);

    assert_eq!(
        sim_a, sim_b,
        "the simulation observed the camera — the determinism doctrine is broken"
    );
    assert_eq!(
        hash_a, hash_b,
        "the sim determinism hash depends on the camera"
    );

    // The two camera paths really did drive different residency (otherwise the
    // assertion above proves nothing).
    assert_ne!(cut_a, cut_b, "the two camera paths produced the same cut");

    // The sim genuinely queried streamed ground: heights are non-trivial, match
    // the generator, and the sim's own residency slid as the walker moved.
    let heights: Vec<f64> = sim_a.iter().map(|s| s.height).collect();
    assert!(
        heights.iter().any(|h| h.abs() > 0.5),
        "every height read 0 — the sim never saw a streamed tile"
    );
    for s in &sim_a {
        let want = streamed_terrain_height(s.walker[0], s.walker[2]);
        assert!(
            (s.height - want).abs() < HEIGHT_TOLERANCE_M,
            "height_at{:?}={} but the generator says {want}",
            [s.walker[0], s.walker[2]],
            s.height
        );
    }
    let residencies: BTreeSet<Vec<(i32, i32)>> = sim_a.iter().map(|s| s.resident.clone()).collect();
    assert!(
        residencies.len() > 3,
        "the sim's residency never moved — the walk is not exercising paging"
    );
    // Sim residency stays small and bounded (a neighbourhood, not the world).
    let max_resident = sim_a.iter().map(|s| s.resident.len()).max().unwrap();
    assert!(
        max_resident <= 64,
        "the sim paged {max_resident} level-0 tiles — that is not a neighbourhood"
    );
}

// ── GATE (d) — PIE == shipping ──────────────────────────────────────────────

/// The cooked-pack path and the editor-document path stream the same terrain to
/// the same world.
///
/// The shipping arm pages tiles **zero-copy out of the mmap'd pack**; the editor
/// arm reads the loose `.inf_terrain` off disk. Both must produce the identical
/// sim trace, the identical render cut, and the identical rendered frame — the
/// same cooked-vs-editor parity the terrain-demo and character-demo gates assert,
/// applied to streamed terrain.
#[test]
fn pie_equals_shipping_for_streamed_terrain() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, content) = cook_streamed_terrain(dir.path());

    let (ship_sim, ship_cut, ship_frame, ship_hash) =
        run_scripted(pack_sim(&pack_dir), streamed_terrain_camera_a);
    let (edit_sim, edit_cut, edit_frame, edit_hash) =
        run_scripted(loose_sim(&content), streamed_terrain_camera_a);

    assert_eq!(edit_sim, ship_sim, "editor sim trace != shipping sim trace");
    assert_eq!(edit_hash, ship_hash, "editor determinism hash != shipping");
    assert_eq!(
        edit_cut, ship_cut,
        "editor render cut != shipping render cut"
    );
    assert_eq!(edit_frame, ship_frame, "editor frame != shipping frame");
    assert!(!ship_frame.last().unwrap().is_empty(), "nothing was drawn");
}

// ── memory bound ────────────────────────────────────────────────────────────

/// The residency budget really clamps the render cut — and **never** the sim's.
///
/// Correctness first: a render page short of budget costs detail, a sim page
/// short of budget changes the simulation, so the clamp is applied to one and not
/// the other.
#[test]
fn render_residency_respects_the_budget_and_sim_residency_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_streamed_terrain(dir.path());

    // A deliberately tiny render budget.
    const CEILING: usize = 12;
    let source = PackLevelSource::open(&pack_dir).expect("pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("level builds");
    let mut sim = inf_player::sim_from_built(built);
    let content = TerrainContent::Pack(source);
    let streaming = TerrainStreaming::attach(
        sim.world_mut(),
        StreamBudget {
            max_resident_tiles: CEILING,
            max_loads_per_sync: 4,
        },
        |guid| content.source(guid),
    );
    assert_eq!(streaming.len(), 1, "the sample has one streamed terrain");
    sim.set_terrain_streaming(streaming);

    let mut clamped_seen = false;
    for step in 0..FRAMES {
        place_walker(&mut sim, step);
        sim.step_once(RuntimeInput::default());
        sim.sync_render_terrain(streamed_terrain_camera_a(step));

        let cut = render_cut(&sim);
        assert!(
            cut.len() <= CEILING,
            "step {step}: render residency {} blew the {CEILING}-page budget",
            cut.len()
        );
        clamped_seen |= sim.terrain_streaming().stats().budget_clamped;

        // The sim, meanwhile, gets every page it asked for: the height under the
        // walker is always available, budget or no budget.
        let p = streamed_terrain_walk_point(step);
        let want = streamed_terrain_height(p.x, p.z);
        assert!(
            (sim.terrain_height_at(p.x, p.z) - want).abs() < HEIGHT_TOLERANCE_M,
            "step {step}: the render budget starved the SIM's terrain query"
        );
    }
    assert!(
        clamped_seen,
        "the budget was never actually binding — the test proves nothing"
    );

    // The diagnostics counters are live and describe the same run.
    let stats = sim.terrain_streaming().stats();
    assert!(stats.loads > 0 && stats.evictions > 0, "{stats:?}");
    assert_eq!(stats.failed, 0, "a page failed to decode: {stats:?}");
    assert!(stats.bytes_resident > 0);
    assert!(stats.sim_resident_level0 > 0);
    assert!(stats.summary().contains("terrain streaming"), "{stats:?}");
}

// ── the sim/render separation, structurally ─────────────────────────────────

/// The two working sets really are different containers.
///
/// The component's data holds the sim's level-0 neighbourhood; the streamer's
/// holds the camera's cut, coarse pages included. If a refactor ever merged
/// them, this test would notice before a gate did.
#[test]
fn the_sim_and_render_working_sets_are_separate_containers() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_streamed_terrain(dir.path());
    let mut sim = pack_sim(&pack_dir);

    // Walk the sim into one corner and point the camera at the far one.
    for step in 0..12 {
        place_walker(&mut sim, step);
        sim.step_once(RuntimeInput::default());
        sim.sync_render_terrain(DVec3::new(240.0, 40.0, 240.0));
    }

    let sim_keys: BTreeSet<TileKey> = sim_resident(&sim).into_iter().map(TileKey::lod0).collect();
    let render_keys = render_cut(&sim);
    assert!(!sim_keys.is_empty() && !render_keys.is_empty());

    // The render side carries coarse pyramid pages; the sim side never does.
    assert!(
        render_keys.iter().any(|k| k.lod > 0),
        "the render cut has no coarse pages: {render_keys:?}"
    );
    assert!(sim_keys.iter().all(|k| k.is_lod0()));

    // And they are not the same set: the camera is 240 m from the walker.
    assert_ne!(sim_keys, render_keys);

    // The sim's answer under the walker is still exact, even though the camera is
    // nowhere near it.
    let p = streamed_terrain_walk_point(11);
    assert!(
        (sim.terrain_height_at(p.x, p.z) - streamed_terrain_height(p.x, p.z)).abs()
            < HEIGHT_TOLERANCE_M
    );

    // A point far from BOTH the walker and the camera reads as unauthored — the
    // documented, deterministic answer outside the sim neighbourhood (never a
    // camera-dependent one).
    let far = DVec2::new(200.0, 20.0);
    let _ = far; // (kept explicit: the assertion below is about determinism)
    let a = sim.terrain_height_at(far.x, far.y);
    sim.sync_render_terrain(DVec3::new(far.x, 40.0, far.y));
    let b = sim.terrain_height_at(far.x, far.y);
    assert_eq!(
        a, b,
        "moving the camera onto a point changed what the SIM reads there"
    );
}

/// **Sim residency scales with terrain OBSERVERS, not with scenery** (P16.3b2
/// audit).
///
/// Sim wants are — correctly — never clamped to a budget, so the only thing
/// bounding them is which entities contribute a neighbourhood. If every entity
/// with a transform did, a level with a few thousand scattered static props would
/// ask for most of the level-0 grid and there would be no ceiling to catch it.
/// Here 2000 props are strewn across the whole 256 m world beside one walker: the
/// sim set must stay a small neighbourhood *of the walker*.
#[test]
fn scattered_static_props_do_not_inflate_sim_residency() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_streamed_terrain(dir.path());
    let mut sim = pack_sim(&pack_dir);

    // Strew static decorative meshes over the entire world — no controller, no
    // body, no actor class: nothing that can read the ground.
    const PROPS: usize = 2000;
    {
        let world = sim.world_mut();
        for i in 0..PROPS {
            let e = world.spawn_with_guid(
                Uuid::from_u128(0x9000_0000_0000_0000_0000_0000_0000_0000u128 + i as u128),
                "Prop",
                None,
            );
            // Deterministic scatter across the full 256 m extent.
            let x = ((i * 37) % 256) as f64;
            let z = ((i * 91) % 256) as f64;
            world.world_mut().entity_mut(e).insert(MeshRef::default());
            inf_ecs::sim::set_translation(world, e, inf_ecs::Vec3d::new(x, 0.0, z));
        }
        world.propagate();
    }

    let mut max_resident = 0usize;
    for step in 0..40 {
        place_walker(&mut sim, step);
        sim.step_once(RuntimeInput::default());
        sim.sync_render_terrain(streamed_terrain_camera_a(step));

        let resident = sim_resident(&sim);
        max_resident = max_resident.max(resident.len());

        // Every resident page is within the walker's neighbourhood, not the props'.
        let p = streamed_terrain_walk_point(step);
        let span = 16.0; // level-0 tile span for this sample (9 samples @ 2 m)
        let margin = inf_player::terrain_stream::SIM_MARGIN_TILES * span;
        for (tx, tz) in &resident {
            // `sim_wants` takes the tile range covering the AXIS-ALIGNED BOX
            // `p ± margin`, so the bound to check is the per-axis (Chebyshev)
            // distance from the walker to the tile's footprint, not the Euclidean
            // one (a corner tile is legitimately `margin·√2` away).
            let lo = DVec2::new(*tx as f64 * span, *tz as f64 * span);
            let here = DVec2::new(p.x, p.z);
            let gap = (lo - here)
                .max(here - (lo + DVec2::splat(span)))
                .max(DVec2::ZERO);
            let d = gap.x.max(gap.y);
            assert!(
                d <= margin + 1e-9,
                "step {step}: tile ({tx},{tz}) is {d} m (per-axis) from the walker — beyond \
                 the {margin} m sim margin, so something other than the walker pulled it in"
            );
        }
    }

    // A 2-tile margin around one walker is at most a 5x5 block.
    assert!(
        max_resident <= 25,
        "{PROPS} static props inflated sim residency to {max_resident} level-0 pages"
    );
    // Sanity: the props really are in the world and really do span it.
    let counted = sim
        .world()
        .world()
        .iter_entities()
        .filter(|e| e.contains::<MeshRef>())
        .count();
    assert_eq!(counted, PROPS);
    // And the walker's terrain query still works (the neighbourhood is present).
    let p = streamed_terrain_walk_point(39);
    assert!(
        (sim.terrain_height_at(p.x, p.z) - streamed_terrain_height(p.x, p.z)).abs()
            < HEIGHT_TOLERANCE_M
    );
}

/// The observer predicate itself: a plain transform is not enough, a character
/// controller (or an actor class / physics body) is.
#[test]
fn only_terrain_observers_contribute_to_sim_wants() {
    use inf_ecs::components::{CharacterController3D, RigidBody2D};

    let mut world = inf_ecs::EcsWorld::new();
    let decor = world.spawn_with_guid(Uuid::from_u128(1), "Decor", None);
    world
        .world_mut()
        .entity_mut(decor)
        .insert(MeshRef::default());
    let walker = world.spawn_with_guid(Uuid::from_u128(2), "Walker", None);
    world
        .world_mut()
        .entity_mut(walker)
        .insert(CharacterController3D::default());
    let body = world.spawn_with_guid(Uuid::from_u128(3), "Crate", None);
    world
        .world_mut()
        .entity_mut(body)
        .insert(RigidBody2D::default());
    let bare = world.spawn_with_guid(Uuid::from_u128(4), "Marker", None);

    use inf_player::terrain_stream::observes_terrain;
    assert!(
        !observes_terrain(&world, decor),
        "a static mesh is not an observer"
    );
    assert!(
        !observes_terrain(&world, bare),
        "a bare transform is not an observer"
    );
    assert!(
        observes_terrain(&world, walker),
        "a character mover observes"
    );
    assert!(observes_terrain(&world, body), "a physics body observes");
}

/// A `Terrain.asset` the content cannot serve must degrade to the inline data,
/// loudly but without breaking the world (the cook already warns about it).
#[test]
fn a_dangling_terrain_asset_falls_back_to_inline_data() {
    let dir = tempfile::tempdir().unwrap();
    let (_pack_dir, content) = cook_streamed_terrain(dir.path());
    // Remove the asset from the loose content: the level's ref now dangles.
    std::fs::remove_file(content.join("World.inf_terrain")).unwrap();
    std::fs::remove_file(content.join("World.inf_terrain.toml")).unwrap();

    let source = level::DevDirLevelSource::new(content.join("StreamedTerrain.inf_lvl"));
    let built = level::load(&source, &InfSceneWorldBuilder::with_defaults(Vec::new()))
        .expect("level still builds");
    let mut sim = inf_player::sim_from_built(built);
    let index = level::terrain_paths_by_guid_from_dir(&content);
    assert!(index.is_empty());
    inf_player::attach_terrain_streaming(&mut sim, &TerrainContent::Dir(index));

    assert!(sim.terrain_streaming().is_empty(), "nothing should stream");
    // The world still runs; the terrain is simply empty (unauthored ground).
    for step in 0..4 {
        place_walker(&mut sim, step);
        sim.step_once(RuntimeInput::default());
        sim.sync_render_terrain(streamed_terrain_camera_a(step));
    }
    assert_eq!(sim.terrain_height_at(10.0, 10.0), 0.0);
    assert!(sim_resident(&sim).is_empty());
}

/// Sanity: the streamed level itself is tiny. Terrain leaving the `.inf_lvl` blob
/// is the entire point of P16.3, so the sample pins it.
#[test]
fn the_streamed_level_carries_no_tiles() {
    let bytes = std::fs::read(sample_dir().join("StreamedTerrain.inf_lvl")).unwrap();
    assert!(
        bytes.len() < 8 * 1024,
        "a streamed level must stay tiny, got {} bytes",
        bytes.len()
    );
    let level = inf_scene::decode(&bytes).expect("level decodes");
    let terrain = level
        .entities
        .iter()
        .find_map(|e| e.terrain.as_ref())
        .expect("terrain entity persisted");
    assert_eq!(terrain.asset, Some(STREAMED_TERRAIN_ASSET_GUID));
    assert!(terrain.data.is_empty(), "the level must ship no tiles");
    let _: Uuid = STREAMED_TERRAIN_ASSET_GUID;
}
