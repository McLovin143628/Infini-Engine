//! **P16.5 gate**: world partition / level streaming, and the determinism
//! doctrine that governs it.
//!
//! The `partitioned-world` sample is a 4 × 4 grid of 128 m cells with one
//! authored prop per cell, a persistent manager + sun, and a `Walker` carrying a
//! [`StreamingSource`] that these tests script along the +X row. The level is
//! **one document** — that is what the editor authors; the cook is what splits it
//! and the player is what streams it.
//!
//! The six things this batch promised:
//!
//! * **(a) cook** — a partitioned level emits exactly one `.inf_part` entry,
//!   stored **uncompressed** (streaming-class, so a cell is sliced straight out
//!   of the mmap'd pack), and two cooks of the same project are byte-identical.
//!   The cooked `.inf_lvl` carries **no entities**: they moved.
//! * **(b) headless determinism** — two independent runs over the same scripted
//!   walk reproduce a byte-identical *activation trace* (which cell, activated or
//!   deactivated, on which step).
//! * **(c) SIM DETERMINISM vs PREFETCH** — the headline. The same scripted walk
//!   under two *completely different* prefetch margins produces a byte-identical
//!   sim trace and an identical activation trace. Prefetch may buy latency; it may
//!   never move a result. If look-ahead could ever reach the fixed step, this is
//!   the test that fails.
//! * **(d) PIE == shipping** — the cooked-pack path (cells sliced from the pack)
//!   and the editor-document path (the same level binned in memory) stream the
//!   same world to the same sim trace and the same activation trace.
//! * **(e) non-partitioned regression** — the platformer sample cooks to a pack
//!   that is byte-identical to one cooked with the partition code path never
//!   entered, and gains no partition entry.
//! * **(f) existence** — an entity in a far cell genuinely does not exist (the
//!   `Guid` does not resolve) until the walker approaches; then it exists with its
//!   authored transform.
//!
//! GPU rendering stays human-verified as everywhere else; the debug overlay is
//! covered by the CPU-side state it reads.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use uuid::Uuid;

use inf_editor_core::samples::{
    partitioned_prop_guid, partitioned_prop_position, partitioned_walk_point,
    PARTITIONED_ACTIVATION_RADIUS_M, PARTITIONED_CELL_SIZE_M, PARTITIONED_CHILD_GUID,
    PARTITIONED_GRID, PARTITIONED_LEVEL_GUID, PARTITIONED_MANAGER_GUID, PARTITIONED_SUN_GUID,
    PARTITIONED_WALKER_GUID,
};
use inf_packager::{cook, CookOptions, DEFAULT_PACK_NAME};
use inf_player::cell_stream::{derived_partition_id, CellEvent, CellState};
use inf_player::level::{self, InfSceneWorldBuilder, PackLevelSource};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_project::ProjectManifest;

/// How many steps the scripted walk covers — three per cell × 4 cells, twice
/// round, so cells activate, deactivate, and activate again.
const STEPS: usize = 24;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn sample_dir() -> PathBuf {
    workspace_root().join("samples/partitioned-world")
}

/// Scaffold a project with the partitioned-world level copied in, then cook it.
/// Returns `(pack dir, loose content dir)`.
fn cook_partitioned(tmp: &Path) -> (PathBuf, PathBuf) {
    cook_partitioned_named(tmp, "proj", "out")
}

fn cook_partitioned_named(tmp: &Path, proj_name: &str, out_name: &str) -> (PathBuf, PathBuf) {
    let proj = tmp.join(proj_name);
    ProjectManifest::new("Partitioned World", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = sample_dir();
    for f in ["PartitionedWorld.inf_lvl", "PartitionedWorld.inf_lvl.toml"] {
        std::fs::copy(src.join(f), content.join(f)).unwrap();
    }
    let out = tmp.join(out_name);
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    (out, content)
}

/// Build the world off a cooked pack (the shipping `--pack` path) and attach cell
/// streaming, exactly as `inf_player::run_headless` does.
fn pack_sim(pack_dir: &Path) -> RuntimeSim {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("pack level builds");
    let partition = built.take_partition();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition);
    sim
}

/// Build the world off the loose `.inf_lvl` (the dev-dir / editor-document path).
/// The level still carries all its entities inline, so the runtime bins them with
/// the same Ring-0 function the cook used.
fn loose_sim(content: &Path) -> RuntimeSim {
    let source = level::DevDirLevelSource::new(content.join("PartitionedWorld.inf_lvl"));
    let builder = InfSceneWorldBuilder::with_defaults(Vec::new());
    let mut built = level::load(&source, &builder).expect("loose level builds");
    let partition = built.take_partition();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition);
    sim
}

/// Place the Walker at the scripted world position for `step`.
///
/// Goes through `inf_ecs::sim::set_translation` (which marks the hierarchy dirty)
/// rather than writing `Transform` directly: `propagate` is dirty-gated, so a raw
/// write would leave `GlobalTransform` — the thing the want set reads — stale.
fn place_walker(sim: &mut RuntimeSim, step: usize) {
    let p = partitioned_walk_point(step);
    let world = sim.world_mut();
    let e = world.entity_of(PARTITIONED_WALKER_GUID).expect("walker");
    inf_ecs::sim::set_translation(world, e, inf_ecs::Vec3d::new(p.x, p.y, p.z));
    world.propagate();
}

/// One step's **sim-observable** residency state: which cells are active and
/// which authored props actually exist in the world.
///
/// Every field is something the fixed step can act on, so byte-equality of this
/// trace across two prefetch margins *is* the doctrine.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SimSample {
    resident: Vec<(i32, i32)>,
    /// The authored props (by cell) whose `Guid` resolves in the live world.
    existing: Vec<(i32, i32)>,
}

fn sample(sim: &RuntimeSim) -> SimSample {
    let world = sim.world();
    let mut existing = Vec::new();
    for cz in 0..PARTITIONED_GRID {
        for cx in 0..PARTITIONED_GRID {
            if world.entity_of(partitioned_prop_guid(cx, cz)).is_some() {
                existing.push((cx, cz));
            }
        }
    }
    SimSample {
        resident: sim.cell_streaming().resident().collect(),
        existing,
    }
}

/// Everything one scripted run records: the per-step residency samples, the full
/// activation trace, and the sim determinism hash.
type RunTrace = (Vec<SimSample>, Vec<CellEvent>, u128);

fn run_scripted(mut sim: RuntimeSim) -> RunTrace {
    let mut samples = Vec::with_capacity(STEPS);
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    for step in 0..STEPS {
        place_walker(&mut sim, step);
        sim.step_once(RuntimeInput::default());
        samples.push(sample(&sim));
        hasher.update(&sim.state_bytes());
    }
    (
        samples,
        sim.cell_streaming().events().to_vec(),
        hasher.digest128(),
    )
}

/// Re-cook the sample with a hand-edited prefetch margin, so the gate can vary
/// **only** that knob.
fn cook_with_prefetch(tmp: &Path, name: &str, margin_m: f64) -> PathBuf {
    let proj = tmp.join(name);
    ProjectManifest::new("Partitioned World", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let src = std::fs::read(sample_dir().join("PartitionedWorld.inf_lvl")).unwrap();
    let mut lvl = inf_scene::decode(&src).unwrap();
    lvl.settings.partition.prefetch_margin_m = margin_m;
    let bytes = lvl.encode().unwrap();
    let path = content.join("PartitionedWorld.inf_lvl");
    std::fs::write(&path, &bytes).unwrap();
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(PARTITIONED_LEVEL_GUID),
        inf_asset::AssetKind::Level,
        inf_asset::ContentHash::of(&bytes),
    )
    .save(&path)
    .unwrap();

    let out = tmp.join(format!("{name}-out"));
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

// ── GATE (a) ────────────────────────────────────────────────────────────────

/// The cook emits one `.inf_part`, **uncompressed**, with the persistent cell
/// plus one grid cell per authored cell — and the cooked `.inf_lvl` no longer
/// carries entities, because they moved.
#[test]
fn cook_emits_one_uncompressed_partition_entry() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_partitioned(dir.path());

    let reader = inf_asset::PackReader::open(&pack_dir.join(DEFAULT_PACK_NAME)).unwrap();
    // The level's asset id comes from the project database (the committed sample
    // ships a *scene* sidecar, not an `inf_asset` one, so the DB synthesizes an
    // id for it) — exactly as the runtime resolves it, through the pack index.
    // Deriving from that id is the contract: the player computes the partition's
    // GUID from the level's, with no side index.
    let level_id = reader
        .index()
        .find(|e| e.kind == inf_asset::AssetKind::Level)
        .expect("the pack has a level")
        .guid;
    let part_id = inf_asset::AssetId(derived_partition_id(level_id.uuid()));
    assert!(
        reader.contains(part_id),
        "the cook must derive a .inf_part for a partitioned level"
    );
    let entry = reader.entry(part_id).expect("indexed");
    assert_eq!(entry.kind, inf_asset::AssetKind::Partition);
    assert!(
        !entry.compressed,
        "a partition is streaming-class: a compressed entry can never be a zero-copy read"
    );
    assert_eq!(
        entry.stored_len, entry.uncompressed_len,
        "an uncompressed entry stores exactly its payload"
    );
    // Exactly one partition per level.
    assert_eq!(
        reader
            .index()
            .filter(|e| e.kind == inf_asset::AssetKind::Partition)
            .count(),
        1
    );

    // The payload is a real `.inf_part` with the persistent cell + the grid.
    let bytes = reader.read(part_id).unwrap();
    let asset = inf_scene::partition::PartitionAssetReader::new(bytes.as_slice()).unwrap();
    assert_eq!(asset.cell_size_m(), PARTITIONED_CELL_SIZE_M);
    let grid: BTreeSet<(i32, i32)> = asset.grid_coords().collect();
    assert_eq!(
        grid.len(),
        (PARTITIONED_GRID * PARTITIONED_GRID) as usize,
        "one cell per authored prop"
    );
    assert!(grid.len() >= 9, "the gate needs at least a 3×3 grid");
    assert_eq!(
        asset.cell_count(),
        grid.len() + 1,
        "the persistent cell is always in the directory"
    );
    // Persistent = the manager (Unplaced), the sun (AlwaysLoaded) and the walker
    // (StreamingSource). Nothing else.
    let persistent = asset
        .cell(inf_scene::partition::CellKey::PERSISTENT)
        .unwrap()
        .expect("persistent cell present");
    let names: BTreeSet<Uuid> = persistent.iter().map(|e| e.guid).collect();
    assert_eq!(
        names,
        BTreeSet::from([
            PARTITIONED_MANAGER_GUID,
            PARTITIONED_SUN_GUID,
            PARTITIONED_WALKER_GUID
        ])
    );
    // The far-corner cell carries BOTH the prop and its far-away child: a
    // hierarchy streams as one unit.
    let corner = (PARTITIONED_GRID - 1, PARTITIONED_GRID - 1);
    let cell = asset
        .cell(inf_scene::partition::CellKey::grid(corner))
        .unwrap()
        .expect("corner cell");
    let ids: BTreeSet<Uuid> = cell.iter().map(|e| e.guid).collect();
    assert_eq!(
        ids,
        BTreeSet::from([
            partitioned_prop_guid(corner.0, corner.1),
            PARTITIONED_CHILD_GUID
        ]),
        "a child authored a world away still streams with its root"
    );

    // The cooked LEVEL carries no entities — they moved.
    let level_bytes = reader.read(level_id).unwrap();
    let level = inf_scene::decode(&level_bytes).unwrap();
    assert!(
        level.entities.is_empty(),
        "a partitioned level must not ship its entities twice"
    );
    assert!(
        level.settings.partition.enabled,
        "…but it says where they went"
    );
}

/// Two cooks of the same project produce byte-identical packs — including the
/// derived partition (the cell directory is `BTreeMap`-ordered and the padding is
/// deterministic zeros, so nothing about the machine can leak in).
#[test]
fn partition_cook_is_deterministic_across_rebuilds() {
    let dir = tempfile::tempdir().unwrap();
    let (a, _) = cook_partitioned_named(dir.path(), "a", "a-out");
    let (b, _) = cook_partitioned_named(dir.path(), "b", "b-out");
    let pa = std::fs::read(a.join(DEFAULT_PACK_NAME)).unwrap();
    let pb = std::fs::read(b.join(DEFAULT_PACK_NAME)).unwrap();
    assert_eq!(pa, pb, "two cooks of one partitioned level must agree");
    assert!(!pa.is_empty());
}

// ── GATE (b) ────────────────────────────────────────────────────────────────

/// Two independent headless runs over the same scripted walk publish the
/// identical activation trace and the identical sim trace.
#[test]
fn headless_cell_streaming_is_reproducible_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_partitioned(dir.path());

    let (sim_a, events_a, hash_a) = run_scripted(pack_sim(&pack_dir));
    let (sim_b, events_b, hash_b) = run_scripted(pack_sim(&pack_dir));

    assert_eq!(
        events_a, events_b,
        "the activation trace is not reproducible"
    );
    assert_eq!(sim_a, sim_b, "the residency trace is not reproducible");
    assert_eq!(hash_a, hash_b, "the sim determinism hash diverged");

    // The run has to actually stream, or the assertions above are vacuous.
    let activations = events_a.iter().filter(|e| e.activated).count();
    let deactivations = events_a.iter().filter(|e| !e.activated).count();
    assert!(
        activations >= PARTITIONED_GRID as usize,
        "only {activations} activations — the walk isn't crossing cells"
    );
    assert!(
        deactivations > 0,
        "nothing was ever evicted — the world never streamed out"
    );
    let sizes: BTreeSet<usize> = sim_a.iter().map(|s| s.resident.len()).collect();
    assert!(!sizes.is_empty());
    // Events are in step order and each cell alternates activate/deactivate.
    assert!(events_a.windows(2).all(|w| w[0].step <= w[1].step));
    let mut live: BTreeSet<(i32, i32)> = BTreeSet::new();
    for e in &events_a {
        if e.activated {
            assert!(live.insert(e.coord), "{:?} activated twice", e.coord);
        } else {
            assert!(live.remove(&e.coord), "{:?} deactivated twice", e.coord);
        }
    }
}

// ── GATE (c) — the doctrine ─────────────────────────────────────────────────

/// **The headline gate.** The identical scripted walk, run under two completely
/// different prefetch margins, produces a byte-identical sim trace and activation
/// trace.
///
/// Prefetch decides *when* a cell's blob is decoded. Activation — the only thing
/// the simulation can observe — always gets its cell, blocking the step if the
/// look-ahead did not reach it. So a wider margin can only remove hitches. This
/// test is what turns that structural claim into a checked one.
#[test]
fn sim_trace_is_identical_under_two_different_prefetch_margins() {
    let dir = tempfile::tempdir().unwrap();
    // 0 m: every activation blocks. 1024 m: the whole world is prefetched.
    let cold = cook_with_prefetch(dir.path(), "cold", 0.0);
    let warm = cook_with_prefetch(dir.path(), "warm", 1024.0);

    let (sim_c, events_c, hash_c) = run_scripted(pack_sim(&cold));
    let (sim_w, events_w, hash_w) = run_scripted(pack_sim(&warm));

    assert_eq!(
        sim_c, sim_w,
        "the simulation observed the prefetch margin — the determinism doctrine is broken"
    );
    assert_eq!(
        events_c, events_w,
        "the activation trace depends on the prefetch margin"
    );
    assert_eq!(
        hash_c, hash_w,
        "the sim determinism hash depends on the prefetch margin"
    );

    // The two margins really did behave differently, or the assertions above
    // prove nothing: the cold run must block on activations the warm one does not.
    let blocks = |pack: &Path| {
        let mut sim = pack_sim(pack);
        for step in 0..STEPS {
            place_walker(&mut sim, step);
            sim.step_once(RuntimeInput::default());
        }
        sim.cell_streaming().stats().blocking_loads
    };
    let cold_blocks = blocks(&cold);
    let warm_blocks = blocks(&warm);
    assert!(cold_blocks > 0, "the 0 m margin never blocked");
    assert!(
        warm_blocks < cold_blocks,
        "prefetch bought nothing: {warm_blocks} vs {cold_blocks} blocking loads"
    );
}

// ── GATE (d) — PIE == shipping ──────────────────────────────────────────────

/// The cooked-pack path and the editor-document path stream the same world.
///
/// The shipping arm slices cells **out of the mmap'd pack**; the editor arm bins
/// the level's own inline entities in memory with the same Ring-0 function the
/// cook used. Both must produce the identical sim trace, activation trace and
/// determinism hash — one binning rule, two sources.
#[test]
fn pie_equals_shipping_for_a_partitioned_scene() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, content) = cook_partitioned(dir.path());

    let (ship_sim, ship_events, ship_hash) = run_scripted(pack_sim(&pack_dir));
    let (edit_sim, edit_events, edit_hash) = run_scripted(loose_sim(&content));

    assert_eq!(edit_sim, ship_sim, "editor residency trace != shipping");
    assert_eq!(
        edit_events, ship_events,
        "editor activation trace != shipping"
    );
    assert_eq!(edit_hash, ship_hash, "editor determinism hash != shipping");
    assert!(!ship_events.is_empty(), "nothing streamed");
}

// ── GATE (e) — non-partitioned regression ───────────────────────────────────

/// A level with partitioning **off** cooks exactly as it did before world
/// partition existed: no `.inf_part` entry, its entities still inline, and two
/// cooks byte-identical.
///
/// The platformer is the right subject: it is the oldest committed sample and the
/// one every other cook test measures against, so if the partition code path
/// could touch an unpartitioned level at all, it would show here.
#[test]
fn a_non_partitioned_level_cooks_exactly_as_before() {
    let dir = tempfile::tempdir().unwrap();
    let src = workspace_root().join("samples/platformer-2d");

    let cook_platformer = |name: &str| {
        let proj = dir.path().join(name);
        ProjectManifest::new("Platformer 2D", "platformer-2d")
            .save(&proj)
            .unwrap();
        let content = proj.join("Content");
        std::fs::create_dir_all(&content).unwrap();
        for f in [
            "Platformer.inf_lvl",
            "Coyote.inf_act",
            "Coyote.inf_act.toml",
        ] {
            std::fs::copy(src.join(f), content.join(f)).unwrap();
        }
        let out = dir.path().join(format!("{name}-out"));
        let report = cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
        (out, report)
    };

    let (out_a, report) = cook_platformer("p1");
    let (out_b, _) = cook_platformer("p2");

    assert_eq!(report.partitions_built, 0, "nothing to partition");
    assert_eq!(report.partition_cells, 0);

    let pack = inf_asset::PackReader::open(&out_a.join(DEFAULT_PACK_NAME)).unwrap();
    assert_eq!(
        pack.index()
            .filter(|e| e.kind == inf_asset::AssetKind::Partition)
            .count(),
        0,
        "an unpartitioned level must not gain a partition entry"
    );
    // Its entities are still where they always were.
    let level_id = pack
        .index()
        .find(|e| e.kind == inf_asset::AssetKind::Level)
        .unwrap()
        .guid;
    let level = inf_scene::decode(&pack.read(level_id).unwrap()).unwrap();
    assert_eq!(level.entities.len(), 5, "the platformer's five entities");
    assert!(!level.settings.partition.enabled);

    // Byte-identical across rebuilds (the P9.2 determinism gate, unchanged).
    assert_eq!(
        std::fs::read(out_a.join(DEFAULT_PACK_NAME)).unwrap(),
        std::fs::read(out_b.join(DEFAULT_PACK_NAME)).unwrap(),
    );

    // And it runs: an unpartitioned world attaches no streamer at all.
    let sim = pack_sim(&out_a);
    assert!(
        sim.cell_streaming().is_empty(),
        "an unpartitioned world must not stream"
    );
}

// ── GATE (f) — existence ────────────────────────────────────────────────────

/// An entity in a far cell **does not exist** until the streaming source
/// approaches — then it exists with its authored state.
///
/// This is the whole point of partitioning stated as a query: `entity_of` fails,
/// then succeeds, and the transform it comes back with is the one the level
/// author placed.
#[test]
fn a_far_entity_does_not_exist_until_the_source_approaches() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_partitioned(dir.path());
    let mut sim = pack_sim(&pack_dir);

    // Step 0 puts the walker in cell (0,0). The far prop is in cell (3,0).
    let far = (PARTITIONED_GRID - 1, 0);
    let far_guid = partitioned_prop_guid(far.0, far.1);
    place_walker(&mut sim, 0);
    sim.step_once(RuntimeInput::default());
    assert!(
        sim.world().entity_of(far_guid).is_none(),
        "a far cell's entity must not exist"
    );
    assert!(!sim.cell_streaming().is_resident(far));
    assert_eq!(sim.cell_streaming().cell_state(far), CellState::Cold);
    // The near one does exist.
    assert!(sim.world().entity_of(partitioned_prop_guid(0, 0)).is_some());
    // …and so do the persistent entities, wherever the walker is.
    assert!(sim.world().entity_of(PARTITIONED_MANAGER_GUID).is_some());
    assert!(sim.world().entity_of(PARTITIONED_SUN_GUID).is_some());

    // Walk until the far cell is reached.
    let mut arrived = None;
    for step in 1..STEPS {
        place_walker(&mut sim, step);
        sim.step_once(RuntimeInput::default());
        if sim.world().entity_of(far_guid).is_some() {
            arrived = Some(step);
            break;
        }
    }
    let step = arrived.expect("the walk never reached the far cell");

    // It exists, it is where the author put it, and the cell says so.
    let e = sim.world().entity_of(far_guid).expect("far entity spawned");
    let got = sim.world().world_translation(e).expect("world transform");
    let want = partitioned_prop_position(far.0, far.1);
    assert!(
        (got - want).length() < 1e-9,
        "step {step}: spawned at {got:?}, authored at {want:?}"
    );
    assert!(sim.cell_streaming().is_resident(far));
    assert_eq!(sim.cell_streaming().cell_state(far), CellState::Active);
    // It has its authored components too (a Cube's MeshRef).
    assert!(sim
        .world()
        .world()
        .get::<inf_ecs::components::MeshRef>(e)
        .is_some());

    // And the persistent entities never went anywhere.
    assert!(sim.world().entity_of(PARTITIONED_MANAGER_GUID).is_some());
    assert!(sim.world().entity_of(PARTITIONED_WALKER_GUID).is_some());
}

// ── residency shape + budget hooks ──────────────────────────────────────────

/// Residency stays a neighbourhood of the streaming sources, the counters
/// describe the run, and every resident cell really is within the activation
/// radius of a source.
#[test]
fn residency_tracks_the_source_and_the_counters_describe_it() {
    let dir = tempfile::tempdir().unwrap();
    let (pack_dir, _content) = cook_partitioned(dir.path());
    let mut sim = pack_sim(&pack_dir);

    let mut max_resident = 0usize;
    for step in 0..STEPS {
        place_walker(&mut sim, step);
        sim.step_once(RuntimeInput::default());

        let p = partitioned_walk_point(step);
        for coord in sim.cell_streaming().resident() {
            let d =
                inf_player::cell_stream::distance_to_cell(coord, p.x, p.z, PARTITIONED_CELL_SIZE_M);
            assert!(
                d <= PARTITIONED_ACTIVATION_RADIUS_M + 1e-9,
                "step {step}: cell {coord:?} is {d} m from the walker — beyond the \
                 {PARTITIONED_ACTIVATION_RADIUS_M} m activation radius, so something other \
                 than the source pulled it in"
            );
        }
        max_resident = max_resident.max(sim.cell_streaming().stats().cells_resident);
    }

    // A sub-cell radius around one walker touches at most a 2×2 block.
    assert!(
        max_resident <= 4,
        "residency grew to {max_resident} cells around one source"
    );

    let stats = sim.cell_streaming().stats();
    assert!(
        stats.activations > 0 && stats.deactivations > 0,
        "{stats:?}"
    );
    assert_eq!(stats.failed, 0, "a cell failed to decode: {stats:?}");
    assert!(stats.bytes_resident > 0, "{stats:?}");
    assert!(stats.entities_resident > 0, "{stats:?}");
    assert_eq!(
        stats.cells_pending, 0,
        "activation blocks, so nothing is ever left pending"
    );
    assert!(stats.summary().contains("cell streaming"), "{stats:?}");
}

/// The player's `derived_partition_id` must agree with the cook's, or a shipped
/// build looks for a `.inf_part` that is not there — and a partitioned level ships
/// no entities of its own, so it would boot empty.
#[test]
fn derived_partition_id_matches_the_cook() {
    for n in [0u128, 1, 0x8416_0500, u128::MAX >> 1] {
        let level = Uuid::from_u128(n);
        assert_eq!(
            derived_partition_id(level),
            inf_packager::derived_partition_id(inf_asset::AssetId(level)).uuid(),
            "the player's salt drifted from the cook's"
        );
    }
    // …and it is a bijection (distinct levels never collide).
    let a = derived_partition_id(Uuid::from_u128(1));
    let b = derived_partition_id(Uuid::from_u128(2));
    assert_ne!(a, b);
}

/// A blueprint on a **persistent** entity binds and ticks in a partitioned level,
/// exactly as it does in an unpartitioned one.
///
/// This is why the persistent cell is spawned by the level builder rather than by
/// the streaming manager: `RuntimeSim` assigns actor ids once, at construction,
/// so an entity that is not in the world by then never gets a blueprint. Before
/// that ordering was fixed, a partitioned level booted with **zero** actors
/// bound — silently, because an unbound actor is not an error, it is just a prop.
#[test]
fn a_persistent_blueprint_binds_in_a_partitioned_level() {
    use inf_ecs::components::{AlwaysLoaded, Collider2D, MeshRef, RigidBody2D, StreamingSource};

    let class = inf_editor_core::samples::coyote_class();
    let class_guid = inf_editor_core::samples::COYOTE_ASSET_GUID;
    let g = Uuid::from_u128;

    // A manager carrying the blueprint (unplaced → persistent), a source, and a
    // far prop that must stream.
    let base = |guid: Uuid, name: &str| inf_scene::RuntimeEntity {
        guid,
        name: name.into(),
        parent: None,
        transform: Default::default(),
        visible: true,
        mesh: None,
        material: None,
        light: None,
        camera: None,
        sprite: None,
        tilemap: None,
        nine_slice: None,
        text2d: None,
        light_2d: None,
        rigid_body_2d: None,
        collider_2d: None,
        character_controller_2d: None,
        rigid_body_3d: None,
        collider_3d: None,
        character_controller_3d: None,
        actor: None,
        terrain: None,
        pcg_volume: None,
        skeletal_mesh: None,
        anim_player: None,
        anim_state_machine: None,
        root_motion: None,
        attached_to: None,
        joint_2d: None,
        joint_3d: None,
        audio_source: None,
        audio_listener: None,
        decal: None,
        volume: None,
        spline: None,
        foliage: None,
        streaming_source: None,
        always_loaded: None,
        time_of_day: None,
        sky_atmosphere: None,
        water_body: None,
        buoyancy: None,
        voxel_volume: None,
    };

    let level = inf_scene::RuntimeLevel {
        title: "Bound".into(),
        entities: vec![
            // The persistent actor. It carries a body so Coyote's Tick can
            // complete its `physics2d::is_grounded` host call — without one the
            // handler aborts on the first statement and never reaches the
            // variable this test reads, which would make the probe vacuous.
            inf_scene::RuntimeEntity {
                actor: Some(class_guid),
                always_loaded: Some(AlwaysLoaded),
                rigid_body_2d: Some(RigidBody2D::default()),
                collider_2d: Some(Collider2D::default()),
                ..base(g(0xB001), "GameMode")
            },
            inf_scene::RuntimeEntity {
                streaming_source: Some(StreamingSource { radius_m: 0.0 }),
                ..base(g(0xB002), "Player")
            },
            // A NEAR prop carrying the SAME blueprint. A mesh occupies space, so
            // it bins into a streamed cell — the classic enemy/door/pickup shape.
            // It activates on the first sync (the source is at the origin), so
            // the negative assertion below is about binding, not about residency.
            inf_scene::RuntimeEntity {
                mesh: Some(MeshRef::default()),
                actor: Some(class_guid),
                rigid_body_2d: Some(RigidBody2D::default()),
                collider_2d: Some(Collider2D::default()),
                transform: inf_ecs::components::Transform::from_translation(glam::DVec3::new(
                    10.0, 0.0, 0.0,
                )),
                ..base(g(0xB003), "Streamed Enemy")
            },
        ],
        settings: inf_scene::RuntimeSettings {
            partition: inf_scene::PartitionSettings {
                enabled: true,
                cell_size_m: 128.0,
                activation_radius_m: 32.0,
                prefetch_margin_m: 0.0,
            },
            ..Default::default()
        },
    };
    let bytes = level.encode().unwrap();

    let builder = InfSceneWorldBuilder::with_defaults(Vec::new())
        .with_bindings(std::collections::HashMap::from([(class_guid, class)]));
    let mut built = {
        use inf_player::level::WorldBuilder;
        builder.build(&bytes).expect("partitioned level builds")
    };

    assert_eq!(
        built.actors.len(),
        1,
        "the persistent cell's blueprint must bind — a partitioned level that boots with \
         no actors is the silent failure this ordering exists to prevent"
    );
    assert_eq!(built.actors[0].0, g(0xB001));
    // …and the world it bound against is exactly the persistent cell: the
    // streamed enemy is not there yet.
    assert!(built.world.entity_of(g(0xB001)).is_some());
    assert!(built.world.entity_of(g(0xB002)).is_some());
    assert!(built.world.entity_of(g(0xB003)).is_none());

    // ── the v1 boundary, as a NEGATIVE assertion ──
    //
    // Run the thing. The persistent blueprint ticks; the streamed one exists and
    // does not. `RuntimeSim` assigns actor ids once, at construction, so an
    // entity that arrives later is fully live in every other respect (it is in
    // the world, it has its mesh, physics would see it) while its `ActorClass`
    // never runs. THIS TEST IS THE EXECUTABLE STATEMENT OF THAT LIMITATION —
    // when the actor map becomes dynamic it must fail first, before anything
    // else, which is the intended order. The cook warns about exactly this shape
    // (`inf_scene::partition::streamed_actors`).
    let partition = built.take_partition();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition);
    for _ in 0..8 {
        sim.step_once(RuntimeInput::default());
    }

    // The streamed enemy really did stream in — the negative assertion below is
    // about BINDING, not about the cell failing to activate.
    assert!(
        sim.world().entity_of(g(0xB003)).is_some(),
        "the enemy's cell should be resident (the source is right next to it)"
    );
    assert!(sim.cell_streaming().is_resident((0, 0)));

    // Coyote's Tick writes its `coyote` timer every step — reset to COYOTE_TIME
    // when grounded, counted down by `dt` when not — so a bound actor's timer has
    // moved off its declared default of 0.0 either way. That is the BEHAVIOURAL
    // proof the blueprint ran, not merely that a map entry exists. (`vy` is a
    // worse probe: the same Tick zeroes it again on landing.)
    let persistent_timer = sim
        .actor_var(g(0xB001), "coyote")
        .expect("the persistent actor is bound");
    match persistent_timer {
        inf_blueprint::Value::Float(v) => assert!(
            *v != 0.0,
            "the persistent blueprint's Tick did not run (coyote still at its 0.0 default)"
        ),
        other => panic!("coyote should be a float, got {other:?}"),
    }

    // …and the streamed one has no actor instance at all.
    assert!(
        sim.actor_var(g(0xB003), "coyote").is_none(),
        "a streamed entity gained a ticking blueprint — the v1 boundary moved, so update \
         the cook advisory (`streamed_actors`), the sample README and this test together"
    );
    assert_eq!(
        sim.entity_map().len(),
        1,
        "exactly one actor id was ever assigned"
    );
}

/// An **empty** partitioned level runs off-pack instead of erroring about a
/// `.inf_part` the author never made.
///
/// Reachable in about four clicks: make a scratch level, tick Enabled in World
/// Settings, hit Play. There is no pack and nothing to stream, and the honest
/// answer is an empty world — so `resolve_cell_store` discriminates on "was a
/// pack supplied", not on "is the level empty". Discriminating on emptiness sent
/// this case down the cooked arm and failed it with a message about an asset the
/// user has never heard of.
#[test]
fn an_empty_partitioned_level_plays_off_pack() {
    let level = inf_scene::RuntimeLevel {
        title: "Scratch".into(),
        entities: Vec::new(),
        settings: inf_scene::RuntimeSettings {
            partition: inf_scene::PartitionSettings {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        },
    };
    let bytes = level.encode().unwrap();

    let built = {
        use inf_player::level::WorldBuilder;
        InfSceneWorldBuilder::with_defaults(Vec::new())
            .build(&bytes)
            .expect("an empty partitioned level must build, not error about a .inf_part")
    };
    assert!(
        built.partition.store().is_some(),
        "it still streams (nothing)"
    );
    assert_eq!(built.actors.len(), 0);

    // And it runs: an empty world steps without incident.
    let mut built = built;
    let partition = built.take_partition();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition);
    for _ in 0..4 {
        sim.step_once(RuntimeInput::default());
    }
    assert_eq!(sim.cell_streaming().stats().cells_resident, 0);
    assert_eq!(sim.cell_streaming().available().count(), 0);

    // A level with entities but still no pack takes the same arm and streams them.
    let mut with_entities = level;
    with_entities.entities =
        inf_scene::decode(&std::fs::read(sample_dir().join("PartitionedWorld.inf_lvl")).unwrap())
            .unwrap()
            .entities;
    let built = {
        use inf_player::level::WorldBuilder;
        InfSceneWorldBuilder::with_defaults(Vec::new())
            .build(&with_entities.encode().unwrap())
            .expect("builds")
    };
    assert!(built.partition.store().unwrap().grid_coords().len() > 1);
}

/// The committed partitioned level is a plain single document — the editor never
/// writes cells, and that is the "the editor stays single-document" claim as an
/// executable assertion.
#[test]
fn the_committed_level_is_one_document() {
    let bytes = std::fs::read(sample_dir().join("PartitionedWorld.inf_lvl")).unwrap();
    let level = inf_scene::decode(&bytes).expect("level decodes");
    assert!(
        level.settings.partition.enabled,
        "it is a partitioned level"
    );
    assert_eq!(
        level.entities.len(),
        (PARTITIONED_GRID * PARTITIONED_GRID) as usize + 4,
        "…but it carries every entity inline, as the editor authored it"
    );
    // Exactly one streaming source and one always-loaded marker.
    assert_eq!(
        level
            .entities
            .iter()
            .filter(|e| e.streaming_source.is_some())
            .count(),
        1
    );
    assert_eq!(
        level
            .entities
            .iter()
            .filter(|e| e.always_loaded.is_some())
            .count(),
        1
    );
}
