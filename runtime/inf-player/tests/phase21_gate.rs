//! **THE PHASE 21 GATE** — volumetric terrain: caves, tunnels & excavation.
//!
//! The fixture is the committed `samples/phase21-cavern`: the plan's own
//! done-when sentence built as a level — a 128 m ridge on an **asset-backed**
//! terrain, with a **carved cave system** whose mouth is a real hole in the
//! heightfield, an **excavated foundation pit** with its **displaced spoil heap**,
//! an **underground room** under the pit joined to it by a shaft, and a Blueprint
//! **borer** that keeps digging at runtime.
//!
//! Six arms, in the order the phase's claim needs them:
//!
//! * **(a) determinism** — two fresh loads of one cooked pack produce
//!   bit-identical traces, **including the voxel field and the hole mask**.
//! * **(b) cooked == uncooked** — the same level off loose files and off a pack.
//! * **(c) PIE == shipping on the runtime-carve trace** — the M9 debt. The first
//!   voxel PIE-vs-shipping coverage this repository has ever had; before P21.4
//!   both sides ran an **empty voxel map** and agreed.
//! * **(d) the headline** — the workings survive save/reload byte-identical, the
//!   underground room is reachable, and a runtime carve is gated by
//!   `runtime_carve` **both ways**.
//! * **(e) the cook is silent** — no advisory fires on the flagship sample,
//!   including the two P21 ones.
//! * **(e2) pool-size invariance** — a runtime carve is the first thing in this
//!   engine that *writes* world state from inside a Blueprint handler, and the
//!   handler runs in the ECS schedule; a subprocess probe at 1/2/4/8 workers says
//!   whether that can be observed.
//! * **(f) budget + the cold-region count** — the composed scene loads inside the
//!   **load** budget and steps inside the **frame** budget, and a dig over a
//!   region **no camera has ever paged** counts and conserves everything.
//!
//! # Why the trace is bits and not floats
//!
//! A carve reports an exact integer sample count times a constant, so a
//! comparison that "passes within 1e-9" would accept precisely the drift this
//! gate exists to catch. Every arm that compares two runs compares `f64::to_bits`.
//!
//! GPU rendering is human-verified as elsewhere; this asserts the authoritative
//! deterministic state, headlessly.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use uuid::Uuid;

use inf_blueprint::Value;
use inf_editor_core::samples::{
    phase21_build, phase21_cavern_dir, phase21_height, phase21_pit_floor_y, phase21_room_probe_xz,
    PHASE21_BORER_ACTOR_GUID, PHASE21_CAVERN_GUID, PHASE21_LAMP_GUID, PHASE21_PIT_CENTER_XZ,
    PHASE21_ROOM_FLOOR_Y, PHASE21_TERRAIN_GUID, PHASE21_VOXEL_M,
};
use inf_packager::{cook, CookOptions};
use inf_player::level::{
    self, BuiltWorld, DevDirLevelSource, InfSceneWorldBuilder, PackLevelSource,
};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_project::ProjectManifest;

// Budgets are imported, never redeclared: a phase does not get its own budget for
// being new. Each arm takes the budget of its own *class* — the one-shot-load arm
// the load-class ceiling, the recurring-work arm the per-frame tripwire.
use inf_core::FRAME_BUDGET_MS;
use inf_player::budget::LOAD_BUDGET_MS;

/// Steps traced. Long enough for the borer to drive its drift 24 m through solid
/// rock (0.15 m per tick) without leaving the authored rock body — the bound is
/// the content's, not the gate's.
const STEPS: usize = 160;

/// Every committed file of the sample, so the fixture copy cannot silently miss
/// one as the sample grows.
fn sample_files() -> [&'static str; 8] {
    [
        "Phase21Cavern.inf_lvl",
        "Phase21Cavern.inf_lvl.toml",
        "Phase21Cavern.inf_terrain",
        "Phase21Cavern.inf_terrain.toml",
        "Cavern.inf_voxel",
        "Cavern.inf_voxel.toml",
        "Borer.inf_act",
        "Borer.inf_act.toml",
    ]
}

/// Scaffold a project holding the sample; returns its `Content` dir.
fn scaffold(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Phase 21 Cavern", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = phase21_cavern_dir();
    for f in sample_files() {
        std::fs::copy(src.join(f), content.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    content
}

/// Scaffold + cook; returns `(content, pack)`.
fn cook_cavern(tmp: &Path) -> (PathBuf, PathBuf) {
    let content = scaffold(tmp);
    let proj = tmp.join("proj");
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    (content, out)
}

/// The **shipping** side: a cooked pack, with terrain streaming and the voxel
/// volumes attached exactly as `run_headless` attaches them.
fn pack_sim(pack: &Path) -> RuntimeSim {
    let source = PackLevelSource::open(pack).expect("pack opens");
    let built: BuiltWorld = inf_player::build_world_from_pack(&source).expect("pack world builds");
    let mut sim = inf_player::sim_from_built(built);
    let pack_file = pack.join(inf_player::level::PACK_FILE);
    let reader =
        std::sync::Arc::new(inf_asset::PackReader::open(&pack_file).expect("pack reader opens"));
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Pack(PackLevelSource::open(pack).expect("pack opens")),
    );
    inf_player::attach_voxel_volumes(
        &mut sim,
        &inf_player::voxel::VoxelRegistry::from_pack(reader),
    );
    sim
}

/// The **loose-files** side, for the cooked-==-uncooked arm.
fn dir_sim(content: &Path) -> RuntimeSim {
    let source = DevDirLevelSource::new(content.join("Phase21Cavern.inf_lvl"));
    let builder = InfSceneWorldBuilder::with_defaults(Vec::new()).with_bindings(
        std::collections::HashMap::from([(
            PHASE21_BORER_ACTOR_GUID,
            inf_editor_core::samples::decode_actor(
                &std::fs::read(content.join("Borer.inf_act")).unwrap(),
            )
            .expect("the class decodes"),
        )]),
    );
    let built = level::load(&source, &builder).expect("dev-dir world builds");
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Dir(level::terrain_paths_by_guid_from_dir(content)),
    );
    inf_player::attach_voxel_volumes(
        &mut sim,
        &inf_player::voxel::VoxelRegistry::from_dir(content),
    );
    sim
}

/// The **editor preview** side: the payload the editor really builds, through the
/// one PIE boot seam the `--pie` subprocess takes.
fn pie_sim() -> RuntimeSim {
    let dir = phase21_cavern_dir();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("Phase21Cavern.inf_lvl"))
        .expect("the committed phase21 document loads");
    let act = inf_editor_core::samples::decode_actor(
        &std::fs::read(dir.join("Borer.inf_act")).expect("the borer class is committed"),
    )
    .expect("the class decodes");
    let voxel_bytes = std::fs::read(dir.join("Cavern.inf_voxel")).expect("the volume is committed");
    let terrain_bytes =
        std::fs::read(dir.join("Phase21Cavern.inf_terrain")).expect("the terrain is committed");
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| (guid == PHASE21_BORER_ACTOR_GUID).then(|| act.clone()),
        |_| None,
        |_| None,
        |_| None,
        |_| Some(voxel_bytes.clone()),
        |_| Some(terrain_bytes.clone()),
        60,
        false,
    )
    .expect("payload builds");
    // The payload really is carrying both — an empty one here is how this gate
    // silently became a comparison of two empty worlds before P21.4.
    assert_eq!(payload.voxels.len(), 1, "the .inf_voxel must ride the wire");
    assert_eq!(
        payload.terrains.len(),
        1,
        "the .inf_terrain must ride the wire"
    );
    inf_player::sim_from_payload(&payload)
        .expect("PIE world builds")
        .sim
}

// ── the trace ───────────────────────────────────────────────────────────────

/// One traced step, as raw bits: what the borer cut this tick, the running total,
/// and the two ground queries over the underground room — plus the *collider*
/// count under the volume, which is what makes this a trace of a world a body
/// could stand in rather than of four floats.
type Frame = [u64; 5];

fn var_bits(sim: &RuntimeSim, name: &str) -> u64 {
    match sim.actor_var(PHASE21_CAVERN_GUID, name) {
        Some(Value::Float(f)) => f.to_bits(),
        other => panic!("{name} is {other:?}"),
    }
}

fn frame(sim: &RuntimeSim) -> Frame {
    [
        var_bits(sim, "removed"),
        var_bits(sim, "total"),
        var_bits(sim, "room_ground"),
        var_bits(sim, "room_voxel"),
        // A carve that never reached the physics bridge would leave this constant
        // while the three floats above moved.
        sim.bridge3d().body_count() as u64,
    ]
}

fn run_trace(sim: &mut RuntimeSim) -> Vec<Frame> {
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            frame(sim)
        })
        .collect()
}

/// **ANTI-VACUITY, applied to every arm that compares two traces.**
///
/// Two identical traces of a world where nothing was dug would satisfy every
/// equality in this file.
fn assert_not_vacuous(trace: &[Frame]) {
    assert_eq!(trace.len(), STEPS);
    let total = f64::from_bits(trace.last().unwrap()[1]);
    assert!(
        total > 20.0,
        "the borer removed only {total} m³ over {STEPS} ticks — the trace is of a \
         script reporting zero, not of carving"
    );
    // The room's floor is where the level says it is, on every tick.
    for (i, f) in trace.iter().enumerate() {
        assert_eq!(
            f64::from_bits(f[2]),
            PHASE21_ROOM_FLOOR_Y,
            "tick {i}: the combined ground query left the room floor"
        );
    }
    // The per-tick cut really varies (sub-voxel steps overlap), so "equal traces"
    // is not "equal copies of one number".
    let cuts: std::collections::BTreeSet<u64> = trace.iter().map(|f| f[0]).collect();
    assert!(cuts.len() > 2, "every tick cut the same amount: {cuts:?}");
}

// ── (a) determinism, INCLUDING the field and the mask ───────────────────────

/// Two fresh loads of ONE cooked pack produce bit-identical traces — and the two
/// worlds hold the same voxel chunks and the same hole mask.
#[test]
fn the_cavern_traces_bit_identically_across_two_loads() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_cavern(tmp.path());

    let mut a = pack_sim(&pack);
    let mut b = pack_sim(&pack);
    let ta = run_trace(&mut a);
    let tb = run_trace(&mut b);
    assert_not_vacuous(&ta);
    assert_eq!(
        ta, tb,
        "the cavern moved between two loads of the same pack"
    );

    // The FIELD, not only the trace: the two sims' voxel volumes are equal chunk
    // for chunk (`VoxelData`'s `PartialEq` compares scale, anchor and chunks —
    // deliberately not the runtime stamps, which are process-global).
    assert_eq!(
        a.voxel_volumes(),
        b.voxel_volumes(),
        "two loads carved different rock"
    );
    assert!(!a.voxel_volumes().is_empty(), "no volume was seeded at all");

    // …and the HOLE MASK, which lives in the terrain and nowhere else.
    assert_eq!(hole_signature(&a), hole_signature(&b));
    assert!(
        hole_signature(&a).iter().any(|&h| h),
        "the reloaded terrain has no holes — the mask did not survive the cook"
    );
}

/// A deterministic sample of the terrain's hole mask over the whole world, on a
/// 4 m lattice. Cheap, order-free, and non-empty only if the mask really came
/// back.
fn hole_signature(sim: &RuntimeSim) -> Vec<bool> {
    let world = sim.world();
    let e = world.entity_of(PHASE21_TERRAIN_GUID).expect("the terrain");
    let data = &world
        .world()
        .get::<inf_ecs::components::Terrain>(e)
        .expect("terrain component")
        .data;
    let mut out = Vec::new();
    let mut x = 0.0;
    while x < 128.0 {
        let mut z = 0.0;
        while z < 128.0 {
            out.push(data.is_hole_at(glam::DVec2::new(x, z)));
            z += 4.0;
        }
        x += 4.0;
    }
    out
}

// ── (b) cooked == uncooked ──────────────────────────────────────────────────

/// The same level, off loose files and off a cooked pack, traces identically.
#[test]
fn cooked_equals_uncooked_on_the_cavern() {
    let tmp = tempfile::tempdir().unwrap();
    let (content, pack) = cook_cavern(tmp.path());
    let cooked = run_trace(&mut pack_sim(&pack));
    let loose = run_trace(&mut dir_sim(&content));
    assert_not_vacuous(&cooked);
    assert_eq!(
        cooked, loose,
        "the cook changed what the borer digs or what the ground answers"
    );
}

// ── (c) PIE == shipping (the M9 debt) ───────────────────────────────────────

/// **THE HOUSE GATE, on voxel ground for the first time.**
///
/// The editor's preview and the shipped build bore the same drift, remove the same
/// cubic metres, and stand on the same underground floor — bit for bit, over 160
/// steps. Until P21.4 the PIE payload carried no `.inf_voxel` and no
/// `.inf_terrain`, so both sides ran an empty voxel map over a blanked heightfield
/// and agreed about nothing.
#[test]
fn pie_equals_shipping_on_the_runtime_carve() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_cavern(tmp.path());
    let ship = run_trace(&mut pack_sim(&pack));
    let pie = run_trace(&mut pie_sim());
    assert_not_vacuous(&ship);
    assert_eq!(
        ship, pie,
        "a cooked build digs the cavern differently from the editor preview"
    );
}

// ── (d) the headline ────────────────────────────────────────────────────────

/// The cave system, the pit and the spoil heap **survive save and reload
/// byte-identical**, and the underground room is **reachable**.
#[test]
fn the_workings_survive_a_round_trip_and_the_room_is_reachable() {
    let tmp = tempfile::tempdir().unwrap();
    let (content, pack) = cook_cavern(tmp.path());

    // Byte-identical, asserted on the bytes: the cook re-emits the `.inf_voxel`
    // and the `.inf_terrain` (both streaming-class kinds, stored uncompressed),
    // and the generator's own build reproduces them.
    let want_voxel = std::fs::read(phase21_cavern_dir().join("Cavern.inf_voxel")).unwrap();
    assert_eq!(
        std::fs::read(content.join("Cavern.inf_voxel")).unwrap(),
        want_voxel
    );
    let reader = inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE)).unwrap();
    let cooked_voxel = reader
        .read(inf_asset::AssetId(
            inf_editor_core::samples::PHASE21_VOXEL_ASSET_GUID,
        ))
        .expect("the pack carries the volume");
    assert_eq!(
        cooked_voxel, want_voxel,
        "the cook changed the cave — a streaming-class kind must ride verbatim"
    );

    // The workings really are in there, read back out of the SHIPPED bytes rather
    // than out of the generator: the pit's floor, the room's floor, and a heap
    // that is taller than the ground it stands on.
    let volume = inf_voxel::sim_volume(&cooked_voxel, glam::DVec3::ZERO).expect("the volume loads");
    let (px, pz) = phase21_room_probe_xz();
    assert_eq!(volume.surface_y_at(px, pz), Some(PHASE21_ROOM_FLOOR_Y));
    // The pit floor is where the sky rule put it, sampled at a corner of the
    // footprint the room does not reach under.
    let (cx, cz) = PHASE21_PIT_CENTER_XZ;
    let floor = phase21_pit_floor_y();
    assert!(
        volume.is_solid_at(glam::DVec3::new(cx + 3.0, floor - 1.0, cz + 2.0)),
        "there is no rock under the pit floor"
    );
    assert!(
        !volume.is_solid_at(glam::DVec3::new(cx + 3.0, floor + 1.0, cz + 2.0)),
        "the pit was not excavated"
    );
    // The spoil heap stands proud of the ground it was dropped on.
    let w = phase21_build();
    let (sx, sz) = inf_editor_core::samples::PHASE21_SPOIL_SITE_XZ;
    let grade = phase21_height(sx, sz);
    let heap = volume.surface_y_at(sx, sz).expect("the heap has a surface");
    assert!(
        heap > grade + 1.0,
        "the spoil heap ({heap}) is not standing on the ground ({grade})"
    );
    // …and it holds exactly what the pit removed, per material.
    assert_eq!(w.spoil.placed, w.pit_removed);
    assert!(w.spoil.is_exact());

    // REACHABLE: a sim loaded off the pack answers the room floor at the room's
    // centre through the **combined** query — the seam a character controller
    // reads — and the level's own lamp is standing on that same number.
    let mut sim = pack_sim(&pack);
    sim.step_once(RuntimeInput::default());
    assert_eq!(sim.terrain_height_at(px, pz), PHASE21_ROOM_FLOOR_Y);
    let lamp = sim
        .world()
        .entity_of(PHASE21_LAMP_GUID)
        .and_then(|e| {
            sim.world()
                .world()
                .get::<inf_ecs::components::Transform>(e)
                .copied()
        })
        .expect("the room lamp survived the cook");
    assert_eq!(lamp.translation.y, PHASE21_ROOM_FLOOR_Y);
    // ANTI-VACUITY: on unholed ground away from the workings the query answers the
    // HEIGHTFIELD, so "the room floor" is not what this seam says everywhere.
    let far = sim.terrain_height_at(100.0, 96.0);
    assert!(
        (far - phase21_height(100.0, 96.0)).abs() < 0.5,
        "the far ridge answers {far}, not its own height"
    );
}

/// **THE `runtime_carve` GATE, both ways.** The committed sample carves because
/// its volume permits it; the identical world with permission withheld carves
/// nothing, deterministically, and says so.
#[test]
fn the_runtime_carve_flag_gates_the_borer_both_ways() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_cavern(tmp.path());

    let mut allowed = pack_sim(&pack);
    for _ in 0..8 {
        allowed.step_once(RuntimeInput::default());
    }
    let dug = match allowed.actor_var(PHASE21_CAVERN_GUID, "total") {
        Some(Value::Float(f)) => *f,
        other => panic!("total is {other:?}"),
    };
    assert!(dug > 0.0, "the permitted borer dug nothing");

    // The same world with the flag off. Flipping it on the loaded world rather
    // than authoring a second level is the point: everything else is identical,
    // so the difference cannot be anything but the flag.
    let mut locked = pack_sim(&pack);
    {
        let world = locked.world_mut();
        let e = world.entity_of(PHASE21_CAVERN_GUID).expect("the cavern");
        let mut v = world
            .world_mut()
            .get_mut::<inf_ecs::components::VoxelVolume>(e)
            .expect("the volume component");
        v.runtime_carve = false;
    }
    let before = locked.voxel_volumes().clone();
    for _ in 0..8 {
        locked.step_once(RuntimeInput::default());
    }
    let locked_total = match locked.actor_var(PHASE21_CAVERN_GUID, "total") {
        Some(Value::Float(f)) => *f,
        other => panic!("total is {other:?}"),
    };
    assert_eq!(locked_total, 0.0, "a locked volume was carved");
    assert_eq!(
        *locked.voxel_volumes(),
        before,
        "a refused carve still moved the field"
    );
    // The refusal is REPORTED — the component's own doc contract.
    assert!(
        locked
            .logs()
            .iter()
            .any(|l| l.contains("carve_sphere") && l.contains("runtime_carve")),
        "a refused carve said nothing"
    );
    // …and the ground the room stands on is untouched either way, because the
    // borer's drift is nowhere near it.
    let (px, pz) = phase21_room_probe_xz();
    assert_eq!(locked.terrain_height_at(px, pz), PHASE21_ROOM_FLOOR_Y);
}

// ── (e) the cook is silent on correct content ───────────────────────────────

/// **No advisory fires on the flagship sample** — including the two this phase
/// added. An advisory that fires on correct content is one nobody reads.
#[test]
fn the_cavern_draws_no_cook_advisory() {
    let tmp = tempfile::tempdir().unwrap();
    scaffold(tmp.path());
    let report = cook(
        &tmp.path().join("proj"),
        &tmp.path().join("out"),
        &CookOptions::default(),
    )
    .unwrap();
    assert!(
        report.warnings.is_empty(),
        "the flagship cavern sample trips cook advisories: {:?}",
        report.warnings
    );
}

// ── (e2) pool-size invariance ───────────────────────────────────────────────

/// Launch the `voxel_probe` binary at a fixed pool size and return the
/// `key=value` lines it printed.
///
/// **Subprocesses, not `init_ecs_task_pool` calls.** `bevy_ecs`'s
/// `ComputeTaskPool` is a process-global `OnceLock` — the first init wins and
/// later ones are no-ops that report the count already chosen — so an in-process
/// "matrix" runs every leg on one pool and is a duplicate of the two-load
/// determinism arm above wearing a stronger name. `crates/inf-runtime` reached the
/// same conclusion for the replay gate and P20.2 for water; this is the voxel
/// twin, and the `threads=` line it prints is what proves each process really got
/// the pool it was asked for.
fn probe(threads: usize, steps: usize) -> HashMap<String, String> {
    let exe = env!("CARGO_BIN_EXE_voxel_probe");
    let out = Command::new(exe)
        .arg(threads.to_string())
        .arg(steps.to_string())
        .output()
        .expect("spawn voxel_probe");
    assert!(
        out.status.success(),
        "voxel_probe (threads={threads}) failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("probe stdout utf8")
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, v)| (k.into(), v.into())))
        .collect()
}

/// **A runtime carve is invariant to the ECS worker-pool size** — in what the
/// Blueprints saw *and* in what the world became.
///
/// Four borers on four volumes, so the handlers really can interleave: one actor
/// could not observe an ordering dependency even if there were one. The `field=`
/// hash is the load-bearing half — a trace of what each script recorded could
/// agree while the chunks underneath diverged, and the chunks are what a save, a
/// replay and the next frame's colliders all read.
#[test]
fn the_runtime_carve_is_identical_across_pool_sizes() {
    const PROBE_STEPS: usize = 120;
    let sizes = [1usize, 2, 4, 8];
    let runs: Vec<HashMap<String, String>> = sizes.iter().map(|n| probe(*n, PROBE_STEPS)).collect();

    // The harness itself works: each subprocess really got the pool it asked for,
    // so the hashes below were produced under four genuinely different pools.
    for (n, run) in sizes.iter().zip(&runs) {
        assert_eq!(
            run.get("threads").map(String::as_str),
            Some(n.to_string().as_str()),
            "the probe did not get the pool size it was asked for: {run:?}"
        );
    }

    for key in ["trace", "field"] {
        let reference = runs[0]
            .get(key)
            .unwrap_or_else(|| panic!("probe printed {key}="));
        for (n, run) in sizes.iter().zip(&runs) {
            assert_eq!(
                run.get(key).unwrap_or_else(|| panic!("probe printed {key}=")),
                reference,
                "the runtime carve's {key} depends on the ECS pool size                  (threads={n}) — it is not replay-safe"
            );
        }
    }

    // Not vacuous: the borers really moved rock over the run, so the equal hashes
    // are equal traces rather than equal copies of nothing.
    assert_ne!(
        runs[0].get("first"),
        runs[0].get("last"),
        "nothing was dug across {PROBE_STEPS} steps — the comparison is vacuous"
    );
}

// ── (f) budgets + the cold-region count ─────────────────────────────────────

/// The composed cavern builds inside the **load** budget and steps inside the
/// **frame** budget — two different ceilings for two different classes of work,
/// both imported. A load measured against the frame budget is a category error.
#[test]
fn the_cavern_loads_and_steps_inside_its_budgets() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_cavern(tmp.path());

    let t0 = Instant::now();
    let mut sim = pack_sim(&pack);
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(
        load_ms < LOAD_BUDGET_MS,
        "the cavern took {load_ms:.1} ms to build (budget {LOAD_BUDGET_MS} ms)"
    );

    // Warm up, then measure the steady state: the first steps pay for rapier's
    // island construction and for meshing every resident chunk into a collider,
    // which are load costs wearing a step's clothes.
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default());
    }
    const MEASURED: usize = 60;
    let t1 = Instant::now();
    for _ in 0..MEASURED {
        sim.step_once(RuntimeInput::default());
    }
    let per_step_ms = t1.elapsed().as_secs_f64() * 1000.0 / MEASURED as f64;
    assert!(
        per_step_ms < FRAME_BUDGET_MS,
        "a cavern fixed step took {per_step_ms:.3} ms (budget {FRAME_BUDGET_MS} ms) — \
         a runtime carve plus its collider rebuild is per-step work"
    );

    // ANTI-VACUITY: the scene really is the one with the cave in it.
    let w = sim.world();
    assert!(w.entity_of(PHASE21_TERRAIN_GUID).is_some());
    assert!(w.entity_of(PHASE21_CAVERN_GUID).is_some());
    assert!(!sim.voxel_volumes().is_empty());
}

/// **THE COLD-REGION COUNT (the P21.3 carried debt), end to end.**
///
/// P21.3 fixed the paging — `carve_into` / `spoil_into` page their footprint
/// before they read it — and gated it at call-site level in Ring 1. The
/// *end-to-end* claim was never checked: that a dig over a region **no camera has
/// ever paged** removes and conserves exactly what the same dig over a warm region
/// does. It is checked here, on the shipped `.inf_voxel`, because that is the only
/// place a genuinely cold store exists.
///
/// The failure it exists to catch is invisible from inside: a non-resident chunk
/// reads as air, so the cut removes nothing there, counts nothing, spoils nothing
/// — **and conservation balances perfectly**, which is exactly what hides it.
#[test]
fn a_dig_over_a_never_paged_region_counts_and_conserves_everything() {
    use inf_voxel::{ChunkStore, MemoryChunkStore, VoxelData, VoxelOp, VoxelShape};

    let bytes = std::fs::read(phase21_cavern_dir().join("Cavern.inf_voxel")).unwrap();
    let asset = inf_voxel::VoxelAsset::from_bytes(bytes).expect("the committed volume loads");
    let reader = asset.reader();

    // The cut: a box in the rock body's south-east corner, a region no working of
    // the sample touches and no camera in this test has ever looked at.
    let cut = VoxelOp::carve(VoxelShape::Box {
        center: glam::DVec3::new(58.0, 22.0, 58.0),
        half_extents: glam::DVec3::new(3.0, 2.0, 3.0),
    });

    // WARM: every chunk resident before the cut — what `sim_volume` gives the sim.
    let mut warm = inf_voxel::sim_volume(asset.as_bytes(), glam::DVec3::ZERO).unwrap();
    let warm_before = warm.chunk_count();
    let (warm_report, _) = warm.apply_op(&cut);

    // COLD: the same volume with **nothing** resident, and its chunks only
    // reachable through a store — the shape a streaming volume really has.
    let mut store = MemoryChunkStore::new();
    for key in reader.keys() {
        store.insert_bytes(
            key,
            reader
                .chunk_bytes(key)
                .expect("the directory named it")
                .to_vec(),
        );
    }
    let mut cold = VoxelData::new(PHASE21_VOXEL_M).with_origin(reader.origin());
    assert_eq!(cold.chunk_count(), 0, "the cold volume starts empty");

    // Page the cut's own footprint first — the P21.3 rule, and the whole point:
    // WITHOUT this the assertions below fail with a *smaller* count and a
    // perfectly balanced conservation identity.
    let (lo, hi) = cut.shape.aabb_m(1.0);
    let want: std::collections::BTreeSet<inf_voxel::ChunkKey> = inf_voxel::chunk_range(
        inf_voxel::ChunkKey::of_sample(
            lo.x.floor() as i32,
            lo.y.floor() as i32,
            lo.z.floor() as i32,
        ),
        inf_voxel::ChunkKey::of_sample(hi.x.ceil() as i32, hi.y.ceil() as i32, hi.z.ceil() as i32),
    )
    .into_iter()
    .filter(|k| store.contains_chunk(*k))
    .collect();
    assert!(!want.is_empty(), "the cold cut names no chunks at all");
    let paged = cold.request_chunks(&want, &store);
    assert_eq!(paged.failed, Vec::new());
    assert!(cold.chunk_count() < warm_before, "the cold set is not cold");

    let (cold_report, _) = cold.apply_op(&cut);

    // THE IDENTITY: cold counts exactly what warm counts, per material.
    assert_eq!(
        cold_report.carved, warm_report.carved,
        "a dig over a never-paged region removed a different amount of rock"
    );
    assert!(
        warm_report.total_carved() > 0,
        "the fixture cut hit no rock at all — the comparison is vacuous"
    );

    // …and the spoil conserves it exactly, from cold, through the paged door.
    let plan = inf_voxel::SpoilPlan::new(
        cold_report.carved,
        glam::DVec3::new(40.0, phase21_height(40.0, 40.0), 40.0),
    );
    let mut builder = inf_voxel::VoxelDeltaBuilder::new();
    let spoil = cold.place_spoil_into_paged(&plan, &mut builder, &store);
    assert_eq!(
        spoil.placed, cold_report.carved,
        "removed != spoiled over a cold region"
    );
    assert_eq!(spoil.shortfall, [0; inf_voxel::MATERIAL_COUNT]);
    assert!(spoil.is_exact());
}

// ── the sim map is the sim's, not a camera's ────────────────────────────────

/// **The determinism seam, on shipped content.** Two sims loaded off the same
/// pack hold the same volumes whatever a *render* store has paged, because the
/// sim's map has no camera in it at all — `sim_volume` decodes fully resident on
/// purpose. Asserted here rather than reasoned about, because the field this
/// gate compares is now something gameplay *writes*.
#[test]
fn the_sims_voxel_map_is_camera_free() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_cavern(tmp.path());
    let a = pack_sim(&pack);
    let volumes: &BTreeMap<Uuid, inf_voxel::VoxelData> = a.voxel_volumes();
    assert_eq!(volumes.len(), 1);
    let seeded = volumes.values().next().unwrap();
    // Every chunk the asset holds is in the sim's map — no residency policy stands
    // between gameplay and the rock.
    let bytes = std::fs::read(phase21_cavern_dir().join("Cavern.inf_voxel")).unwrap();
    let asset = inf_voxel::VoxelAsset::from_bytes(bytes).unwrap();
    assert_eq!(seeded.chunk_count(), asset.reader().chunk_count());
}
