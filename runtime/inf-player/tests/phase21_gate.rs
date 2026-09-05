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
//! * **(a2) the carve is REAL** — the field the borer left differs from the
//!   seeded asset *where it dug* and is byte-identical *everywhere else*, and the
//!   **collider world** answers differently through the bore. Without these two,
//!   a `runtime_carve` applied to a throwaway clone — correct cubic metres
//!   reported, real world untouched — passes every other arm in this file.
//! * **(a3) the shipped player SEES it** — the render-side voxel store and the
//!   render-side terrain streamer both reflect the carve, so a game that digs is
//!   not looking at the rock it removed.
//! * **(a4) the REAL `--pie` subprocess** — one arm spawns the actual player
//!   binary in `--pie` mode and compares its per-step trace against the
//!   in-process reference, so a boot path that reverts to a divergent seam fails
//!   the battery instead of agreeing with itself.
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
    PHASE21_BORER_ACTOR_GUID, PHASE21_BORE_RADIUS_M, PHASE21_BORE_START, PHASE21_BORE_STEPS,
    PHASE21_BORE_STEP_M, PHASE21_CAVERN_GUID, PHASE21_LAMP_GUID, PHASE21_PIT_CENTER_XZ,
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
const STEPS: usize = PHASE21_BORE_STEPS;

/// Every committed file of the sample, so the fixture copy cannot silently miss
/// one as the sample grows.
fn sample_files() -> [&'static str; 9] {
    [
        "Phase21Cavern.inf_lvl",
        "Phase21Cavern.inf_lvl.toml",
        "Phase21Cavern.inf_terrain",
        "Phase21Cavern.inf_terrain.toml",
        "Cavern.inf_voxel",
        "Cavern.inf_voxel.toml",
        "Borer.inf_act",
        "Borer.inf_act.toml",
        "README.md",
    ]
}

/// …and the list really is **every** committed file. This function exists so a
/// sample that grows a file cannot be cooked without it, which only works if
/// something checks the list against the directory — it was eight of nine
/// (`README.md` was missing) from the day it was written.
#[test]
fn the_fixture_copies_every_committed_sample_file() {
    let listed: std::collections::BTreeSet<String> =
        sample_files().iter().map(|s| s.to_string()).collect();
    let on_disk: std::collections::BTreeSet<String> = std::fs::read_dir(phase21_cavern_dir())
        .expect("the sample directory exists")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        listed, on_disk,
        "`sample_files()` and samples/phase21-cavern disagree — a cooked fixture \
         would be missing content the committed sample has"
    );
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
    inf_player::sim_from_payload(&cavern_payload())
        .expect("PIE world builds")
        .sim
}

/// The payload the editor really builds for the committed sample — the input to
/// both the in-process PIE arm and the real-subprocess one, so the two cannot
/// disagree about what was sent.
fn cavern_payload() -> inf_runtime::pie::ScenePayload {
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
        // The BYTES route (`ScenePayload` v12 kept both): this cavern's terrain
        // is small, and a gate that only ever exercised the path route would
        // leave the inline one — which every in-memory caller still takes —
        // uncovered. `island_gate` drives the path route.
        |_| {
            Some(inf_editor_core::pie::TerrainRef::Bytes(
                terrain_bytes.clone(),
            ))
        },
        // P22.3: no destructible meshes in this fixture.
        |_| None,
        // P26.3b: the cloth / hair / material / texture byte resolver.
        |_| None,
        |_| None,
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
    payload
}

// ── the trace ───────────────────────────────────────────────────────────────

/// One traced step, as raw bits: what the borer cut this tick, the running total,
/// and the two ground queries over the underground room — plus the *collider*
/// count under the volume, which is what makes this a trace of a world a body
/// could stand in rather than of four floats.
type Frame = [u64; 6];

fn var_bits(sim: &RuntimeSim, name: &str) -> u64 {
    match sim.actor_var(PHASE21_CAVERN_GUID, name) {
        Some(Value::Float(f)) => f.to_bits(),
        other => panic!("{name} is {other:?}"),
    }
}

/// The boulder's world `y`.
fn boulder_y(sim: &RuntimeSim) -> f64 {
    sim.world()
        .entity_of(inf_editor_core::samples::PHASE21_BOULDER_GUID)
        .and_then(|e| {
            sim.world()
                .world()
                .get::<inf_ecs::components::Transform>(e)
                .map(|t| t.translation.y)
        })
        .unwrap_or(f64::NAN)
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
        // The boulder's height. The level's only dynamic body, resting on the
        // voxel rock over the drift (terrain has no collider in this engine), so
        // it falls when the borer takes that rock away — the one witness of the
        // carve that survives out to the PIE pipe, which streams poses and knows
        // nothing about chunks.
        boulder_y(sim).to_bits(),
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
    // The room's floor is where the level says it is, on every tick — and the
    // VOXEL-only query agrees with the combined one there, which the borer's own
    // docs call "the property the pair exists to show" and which nothing
    // asserted: `room_voxel` could have returned −12345.0 for ever.
    for (i, f) in trace.iter().enumerate() {
        assert_eq!(
            f64::from_bits(f[2]),
            PHASE21_ROOM_FLOOR_Y,
            "tick {i}: the combined ground query left the room floor"
        );
        assert_eq!(
            f64::from_bits(f[3]),
            PHASE21_ROOM_FLOOR_Y,
            "tick {i}: `voxel.ground_height` disagrees with `terrain.height_at` \
             over a holed sample, where the combined query IS the voxel one"
        );
    }
    // The per-tick cut really varies (sub-voxel steps overlap), so "equal traces"
    // is not "equal copies of one number".
    let cuts: std::collections::BTreeSet<u64> = trace.iter().map(|f| f[0]).collect();
    assert!(cuts.len() > 2, "every tick cut the same amount: {cuts:?}");

    // **THE BOULDER: three heights, and each one rules out a different failure.**
    //
    // It starts in the air, lands on the voxel rock, rests there while the borer
    // is still upstream of it, and then drops again when the drift takes that rock
    // away. Terrain has no rapier collider in this engine, so the only thing that
    // can ever hold it up is a voxel chunk trimesh.
    let (blx, blz) = inf_editor_core::samples::PHASE21_BOULDER_XZ;
    let half = inf_editor_core::samples::PHASE21_BOULDER_HALF_M;
    let first = f64::from_bits(trace[0][5]);
    let last = f64::from_bits(trace.last().unwrap()[5]);
    assert!(
        last < first,
        "the boulder never fell at all ({first} -> {last})"
    );

    // 1. It RESTED ON THE ROCK before the borer arrived. Deleting the voxel
    //    collider path entirely leaves it in free fall here, tens of metres down.
    let arrival = ((blx - PHASE21_BORE_START.0) / PHASE21_BORE_STEP_M) as usize;
    assert!(
        arrival > 20 && arrival < STEPS,
        "the fixture's timing broke"
    );
    // Ten ticks before the borer gets there: after the 2.1 m drop has settled
    // (~40 ticks) and before the rock under it goes.
    let resting = f64::from_bits(trace[arrival - 10][5]);
    let rock_top = phase21_height(blx, blz);
    assert!(
        (resting - (rock_top + half)).abs() < 0.6,
        "half-way through the run the boulder is at {resting}, not resting on the \
         rock at {} - nothing is holding it up, so the voxel colliders are absent \
         rather than merely stale",
        rock_top + half
    );

    // 2. It then FELL THROUGH the floor the borer opened...
    let crown = PHASE21_BORE_START.1 + PHASE21_BORE_RADIUS_M;
    assert!(
        last < crown,
        "the boulder rests at {last}, above the trench crown at {crown} - the borer \
         removed the rock under it in the SIM and the solver never heard"
    );

    // 3. ...and LANDED ON THE TRENCH FLOOR rather than falling out of the world,
    //    which is what a collider-free run looks like.
    let floor = PHASE21_BORE_START.1 - PHASE21_BORE_RADIUS_M;
    assert!(
        last > floor - 2.0,
        "the boulder ended at {last}, well below the trench floor at {floor} - it \
         fell through everything, so nothing caught it"
    );
}

// ── the field, and the collider world ───────────────────────────────────────

/// The sim's voxel field, as raw bytes: every chunk's signed distances and
/// materials, in `BTreeMap` order.
///
/// **This is what a carve is.** Everything else the gate reads — the cubic metres
/// a Blueprint recorded, the ground query, the collider count — is downstream of
/// it, and every one of them can be produced by a carve applied to a *clone*
/// while the world stays solid rock (mutation-proved: 10/10 green, and the run
/// got ten times faster because nothing was being dug).
fn field_bytes(sim: &RuntimeSim) -> Vec<(Uuid, inf_voxel::ChunkKey, Vec<u8>)> {
    let mut out = Vec::new();
    for (&guid, data) in sim.voxel_volumes() {
        for (&key, chunk) in data.chunks() {
            let mut bytes = Vec::with_capacity(inf_voxel::CHUNK_VOXELS * 5);
            for v in chunk.sdf() {
                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            bytes.extend_from_slice(chunk.materials());
            out.push((guid, key, bytes));
        }
    }
    out
}

/// Chunk keys the borer's drift passes through — the region that MUST change —
/// and, by exclusion, the region that must not.
fn bore_chunks() -> std::collections::BTreeSet<inf_voxel::ChunkKey> {
    let (bx, by, bz) = PHASE21_BORE_START;
    let r = PHASE21_BORE_RADIUS_M;
    let mut out = std::collections::BTreeSet::new();
    for i in 0..=STEPS {
        let x = bx + i as f64 * PHASE21_BORE_STEP_M;
        for (dx, dy, dz) in [(-r, -r, -r), (r, r, r)] {
            out.insert(inf_voxel::ChunkKey::of_sample(
                ((x + dx) / PHASE21_VOXEL_M).floor() as i32,
                ((by + dy) / PHASE21_VOXEL_M).floor() as i32,
                ((bz + dz) / PHASE21_VOXEL_M).floor() as i32,
            ));
        }
    }
    out
}

/// **THE FIELD ASSERTION.** Compare the world the borer left against the world it
/// started from: every chunk the drift touches must differ, and every chunk it
/// does not touch must be byte-identical.
///
/// The second half is what makes the first half a claim about *this* carve rather
/// than about any mutation at all.
fn assert_the_carve_landed(
    before: &[(Uuid, inf_voxel::ChunkKey, Vec<u8>)],
    after: &[(Uuid, inf_voxel::ChunkKey, Vec<u8>)],
) {
    assert_eq!(
        before.len(),
        after.len(),
        "the carve added or dropped chunks; the sample's volume is fixed"
    );
    let bored = bore_chunks();
    let mut changed = 0usize;
    let mut untouched_changed: Vec<inf_voxel::ChunkKey> = Vec::new();
    for ((gb, kb, b), (ga, ka, a)) in before.iter().zip(after) {
        assert_eq!((gb, kb), (ga, ka), "the chunk set moved");
        if b == a {
            continue;
        }
        if bored.contains(kb) {
            changed += 1;
        } else {
            untouched_changed.push(*kb);
        }
    }
    assert!(
        changed > 0,
        "NOTHING IN THE FIELD MOVED over {STEPS} ticks of carving. The Blueprint \
         reported cubic metres and the ground query answered — so the carve ran \
         against something that is not the world."
    );
    assert!(
        untouched_changed.is_empty(),
        "the borer changed chunks its drift never reaches: {untouched_changed:?}"
    );
}

/// Where a ray cast straight down the bore column stops, in the **collider**
/// world. `None` when it hits nothing.
///
/// The only probe in this file that asks the *solver* what it contains. A carve
/// that never reached `PhysicsBridge3D` moves every other number here and leaves
/// this one exactly where it was — which is how deleting `gather_voxels` outright
/// kept the gate green.
fn bore_ray_toi(sim: &mut RuntimeSim, x: f64) -> Option<f64> {
    let (_, by, bz) = PHASE21_BORE_START;
    sim.bridge3d_mut()
        .world_mut()
        .cast_ray(
            glam::DVec3::new(x, by + 12.0, bz),
            glam::DVec3::new(0.0, -1.0, 0.0),
            24.0,
        )
        .map(|h| h.toi)
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
    let seeded = field_bytes(&a);
    let ta = run_trace(&mut a);
    let tb = run_trace(&mut b);
    assert_not_vacuous(&ta);
    assert_eq!(
        ta, tb,
        "the cavern moved between two loads of the same pack"
    );
    // …and the carve was REAL: the field moved where the drift runs, and nowhere
    // else. Without this line every arm in the file passes with `runtime_carve`
    // applied to a throwaway clone.
    assert_the_carve_landed(&seeded, &field_bytes(&a));

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

// ── (a2) the carve is REAL, in the collider world ───────────────────────────

/// **THE COLLIDER ARM.** A ray down the bore column stops on rock before the
/// carve and falls through afterwards — asked of the *solver*, not of the sim.
///
/// This is the arm that fails when `gather_voxels` is deleted outright. Every
/// other number in this file — the cubic metres, the ground query, the entity
/// count — is unchanged by removing the voxel colliders entirely (mutation-proved:
/// an early `return` at the top of `gather_voxels` left the gate 10/10 green in
/// half a second), because none of them asks what a body would collide with.
///
/// The probe is a ray rather than a dropped body because a ray is a pure query:
/// no integration, no substeps, no tolerance, and the number it returns is the
/// distance to the first triangle — which is exactly the quantity a carve moves.
#[test]
fn the_carve_opens_the_collider_world_not_only_the_sim() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_cavern(tmp.path());
    let mut sim = pack_sim(&pack);

    // One step to build the colliders (the bridge syncs inside the step).
    sim.step_once(RuntimeInput::default());
    let (bx, _, _) = PHASE21_BORE_START;
    // A metre into the drift, which the first tick has not yet reached.
    let probe_x = bx + 8.0;
    let before = bore_ray_toi(&mut sim, probe_x).expect(
        "the ray hit nothing before the carve — there are no voxel colliders at all, \
         so this arm cannot see the carve either",
    );

    // …drive the borer through it.
    for _ in 0..STEPS {
        sim.step_once(RuntimeInput::default());
    }
    let after =
        bore_ray_toi(&mut sim, probe_x).expect("the ray fell out of the world after the carve");

    // The ray travels FURTHER before it stops: the crown the trench removed is
    // gone, so it falls through where the ground used to be and lands on the
    // trench floor. The bore is 2.5 m in radius, so a real opening moves the stop
    // by metres — a metre of tolerance would accept a rounding artefact.
    assert!(
        after > before + 2.0,
        "the ray stops at {after} after the bore and stopped at {before} before it — \
         the carve did not reach the colliders, so a body still stands on rock that \
         gameplay says is gone"
    );

    // ANTI-VACUITY: a column the drift never reaches is unmoved, so "the ray
    // changed" is a statement about the bore and not about the whole world.
    let untouched_x = bx - 6.0;
    let u_before = {
        let mut fresh = pack_sim(&pack);
        fresh.step_once(RuntimeInput::default());
        bore_ray_toi(&mut fresh, untouched_x)
    };
    assert!(
        u_before.is_some(),
        "the control column has no collider at all, so it cannot witness anything"
    );
    let u_after = bore_ray_toi(&mut sim, untouched_x);
    assert_eq!(
        u_before, u_after,
        "a column the bore never reaches moved — the comparison above is not about \
         the drift"
    );
}

// ── (a3) the shipped player SEES what it carves ─────────────────────────────

/// **THE RENDER ARM.** After the borer runs, the *render* stores reflect the
/// carve: the voxel store has mirrored the carved chunks, and the terrain
/// streamer has pinned the tiles whose hole mask moved.
///
/// Both were missing from the first cut of P21.4, and both fail the same way: the
/// player digs, the collider opens, gameplay walks in — and the screen keeps
/// drawing the rock and the unbroken ground. The render store never read
/// `sim.voxel_volumes()`, and `pin_tile`'s only production caller was the editor,
/// so on an **asset-backed** terrain (the only kind that can carry a hole mask,
/// and therefore this sample's necessary configuration) the mouth was never drawn.
///
/// Structural, not pixels: engagement counters on the two seams that carry the
/// state, which is what the house rule asks for when the claim is "the command
/// stream reached the renderer".
#[test]
fn the_render_side_reflects_the_runtime_carve() {
    let tmp = tempfile::tempdir().unwrap();
    let (content, pack) = cook_cavern(tmp.path());
    let mut sim = pack_sim(&pack);

    // `sync_voxel_store` is the CPU half of `PlayerRenderHost::sync_voxels` — the
    // same function the windowed player calls every frame, lifted out of the impl
    // precisely so this claim is checkable where there is no GPU.
    //
    // **In the PRODUCT's ordering**, which is the whole point of this arm. The
    // render host binds its volumes *inside* the sync, and the sync runs after the
    // frame has stepped — so the very first `sync_voxel_store` a session ever runs
    // already has a carve behind it. An earlier `overlay_sim` recorded that first
    // sight as a *baseline* and copied nothing, so a one-shot dig on the first Tick
    // was invisible for the rest of the session. Syncing before the loop (the
    // ordering this arm used to take) is the one ordering the product never runs,
    // and it is the ordering that hides it.
    let assets = inf_player::voxel::VoxelRegistry::from_dir(&content);
    let camera = glam::DVec3::new(48.0, 40.0, 48.0);

    // The control: a store synced against a sim that has never stepped, so it
    // holds exactly the authored surface.
    let mut control = inf_voxel::VoxelVolumes::new();
    let untouched = pack_sim(&pack);
    inf_player::render::sync_voxel_store(&mut control, &assets, &untouched, camera);
    let authored = control.triangle_count();
    assert!(authored > 0, "the render store drew nothing to begin with");

    // ONE step — one carve — and only then the first sync of the session.
    sim.step_once(RuntimeInput::default());
    let mut store = inf_voxel::VoxelVolumes::new();
    inf_player::render::sync_voxel_store(&mut store, &assets, &sim, camera);
    assert!(
        store.chunk_count() > 0,
        "the render store bound no chunks at all — the rest of this arm is vacuous"
    );
    assert!(
        store.overlaid_len(PHASE21_CAVERN_GUID.as_u128()) > 0,
        "the FIRST sync of the session copied nothing, so a one-shot dig on the \
         first Tick would never be drawn"
    );
    let after_one = store.triangle_count();
    assert_ne!(
        after_one, authored,
        "after one tick of carving the render store still draws the authored \
         surface — the first carve was baselined away"
    );

    // …and the rest of the run keeps reaching it.
    for _ in 1..STEPS {
        sim.step_once(RuntimeInput::default());
    }
    sim.sync_render_terrain(camera);
    inf_player::render::sync_voxel_store(&mut store, &assets, &sim, camera);
    assert_ne!(
        store.triangle_count(),
        after_one,
        "the render store stopped following the borer after its first tick"
    );

    // The terrain half: the runtime hole reached the render streamer, and its pin
    // set is bounded by the cut rather than growing for the life of the session.
    let pinned = sim.terrain_streaming().overlaid_len(PHASE21_TERRAIN_GUID);
    assert!(
        pinned > 0,
        "no terrain tile was pinned into the render streamer, so an asset-backed \
         terrain keeps drawing solid ground over the mouth the carve opened"
    );
    assert!(
        pinned <= inf_terrain::StreamBudget::default().max_resident_tiles,
        "the pin set ({pinned}) is not bounded by the residency budget — past it \
         `pin_ceiling` clamps the camera cut to 1 and the terrain silently stops \
         streaming"
    );
}

// ── (a4) the REAL `--pie` subprocess boundary ───────────────────────────────

/// **THE GUARD THE LAW DEMANDS.** Spawn the actual `inf-player` binary in `--pie`
/// mode over the cavern payload and compare its per-step state hashes against the
/// in-process reference.
///
/// Every other PIE arm in this file builds its "preview" side **in process**, by
/// calling `sim_from_payload` directly. That is a comparison of one function
/// against itself: revert `main.rs`'s `LoadScene` handler to the bare
/// `RuntimeSim::new` it used before P21.4 — dropping the voxel volumes, the
/// terrain, the state machines, the clips and the audio — and every one of them
/// stays green, because the thing that changed is the boot path neither side
/// runs. That is precisely this batch's own law:
///
/// > a boot path that forgets an attachment does not crash — it agrees with
/// > itself.
///
/// It is also the shape the P11 history had: the in-process reference always
/// carried the rich fixture, and the real subprocess had nothing to drop, so
/// nothing ever reported the divergence. This arm is the one that would have.
///
/// Compared as **state hashes**, which is what the PIE protocol streams — the
/// xxh3 of the `Guid`-sorted sim snapshot, so a voxel volume the subprocess
/// failed to attach shows up as a different ground query in a different pose in a
/// different hash.
#[test]
fn the_real_pie_subprocess_matches_the_in_process_reference() {
    use inf_editor_core::pie::PieSession;
    use inf_runtime::pie::PlayerToEditor;
    use std::time::Duration;

    let payload = cavern_payload();
    // The **whole** run, not a prefix. Every frame is a pipe round trip, so the
    // temptation is to trace forty and stop — and forty is inside the boulder's
    // free fall, where a subprocess with no voxel volumes and no colliders traces
    // *identically* because gravity is the only thing acting on anything. The
    // divergence starts when the boulder lands (about tick 40) and again when the
    // borer takes the rock out from under it (about tick 67), so the window has to
    // contain both. Measured before it was written down: at `N = 40` this arm
    // passed with the `--pie` boot seam reverted.
    const N: u32 = STEPS as u32;

    let mut session =
        PieSession::spawn_scene(&PathBuf::from(env!("CARGO_BIN_EXE_inf-player")), &payload)
            .expect("the player spawns in --pie mode");
    session.step(N).expect("step N");

    let mut got = Vec::with_capacity(N as usize);
    for _ in 0..N {
        let ev = session
            .wait_for(Duration::from_secs(20), |e| {
                matches!(e, PlayerToEditor::Frame { .. })
            })
            .expect("a frame per step");
        if let PlayerToEditor::Frame { state_hash, .. } = ev {
            got.push(state_hash);
        }
    }
    session
        .stop(Duration::from_secs(10))
        .expect("graceful stop");

    let want = inf_player::scene_trace(&payload, N as u64).expect("in-process reference");
    assert_eq!(
        got, want,
        "the REAL --pie subprocess ran a different world from the in-process \
         reference — one of the two boot paths is missing an attachment"
    );
    // ANTI-VACUITY: the trace evolves, so equal hashes are equal *worlds* and not
    // two copies of a scene that never moved.
    assert!(
        got.windows(2).any(|w| w[0] != w[1]),
        "the state hash never changed across {N} steps"
    );
}

// ── (b) cooked == uncooked ──────────────────────────────────────────────────

/// The same level, off loose files and off a cooked pack, traces identically.
#[test]
fn cooked_equals_uncooked_on_the_cavern() {
    let tmp = tempfile::tempdir().unwrap();
    let (content, pack) = cook_cavern(tmp.path());
    let mut cooked_sim = pack_sim(&pack);
    let mut loose_sim = dir_sim(&content);
    let seeded = field_bytes(&cooked_sim);
    let cooked = run_trace(&mut cooked_sim);
    let loose = run_trace(&mut loose_sim);
    assert_not_vacuous(&cooked);
    assert_eq!(
        cooked, loose,
        "the cook changed what the borer digs or what the ground answers"
    );
    // The trace is what a Blueprint *recorded*; the field is what the world
    // *became*, and the two are different claims.
    assert_the_carve_landed(&seeded, &field_bytes(&cooked_sim));
    assert_eq!(
        field_bytes(&cooked_sim),
        field_bytes(&loose_sim),
        "cooked and loose reported the same trace over different rock"
    );
}

// ── (c) PIE == shipping (the M9 debt) ───────────────────────────────────────

/// **THE HOUSE GATE, on voxel ground for the first time.**
///
/// The editor's preview and the shipped build bore the same drift, remove the same
/// cubic metres, and stand on the same underground floor — bit for bit, over
/// every step of the run. Until P21.4 the PIE payload carried no `.inf_voxel` and no
/// `.inf_terrain`, so both sides ran an empty voxel map over a blanked heightfield
/// and agreed about nothing.
#[test]
fn pie_equals_shipping_on_the_runtime_carve() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_cavern(tmp.path());
    let mut ship_sim = pack_sim(&pack);
    let mut pie_side = pie_sim();
    let seeded = field_bytes(&ship_sim);
    let ship = run_trace(&mut ship_sim);
    let pie = run_trace(&mut pie_side);
    assert_not_vacuous(&ship);
    assert_eq!(
        ship, pie,
        "a cooked build digs the cavern differently from the editor preview"
    );
    assert_the_carve_landed(&seeded, &field_bytes(&ship_sim));
    assert_eq!(
        field_bytes(&ship_sim),
        field_bytes(&pie_side),
        "preview and shipping report the same numbers over different rock"
    );
}

// ── (d) the headline ────────────────────────────────────────────────────────

/// The cave system, the pit and the spoil heap **survive save and reload
/// byte-identical**, and the underground room is **reachable**.
#[test]
fn the_workings_survive_a_round_trip_and_the_room_is_reachable() {
    let tmp = tempfile::tempdir().unwrap();
    let (content, pack) = cook_cavern(tmp.path());

    // Byte-identical **through the content directory**: the generator's own
    // build reproduces the committed `.inf_voxel`, so the sample on disk is the
    // sample the code makes.
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

    // **IASSET1 moved what "verbatim" means here, and not what it protects.**
    // This assertion used to compare the cooked bytes to the authored ones. The
    // cook now transcodes the container to per-chunk compression, so the bytes
    // differ; what must not differ is (a) the pack ENTRY's policy — a compressed
    // entry puts every chunk offset out of reach of the mapping, which is the
    // anti-clause — and (b) what each chunk decodes to, which is what a shipped
    // cave actually depends on. Both are checked, and the cooked volume is
    // smaller, because an SDF is the most compressible thing this engine ships.
    let entry = reader
        .entry(inf_asset::AssetId(
            inf_editor_core::samples::PHASE21_VOXEL_ASSET_GUID,
        ))
        .expect("the volume is indexed");
    assert!(
        !entry.compressed,
        "THE ANTI-CLAUSE: a streaming-class ENTRY must stay raw"
    );
    assert!(
        cooked_voxel.len() < want_voxel.len(),
        "the cooked volume ({} B) is not smaller than the authored ({} B)",
        cooked_voxel.len(),
        want_voxel.len()
    );
    {
        let cooked = inf_voxel::VoxelAssetReader::new(cooked_voxel.as_slice()).unwrap();
        let authored = inf_voxel::VoxelAssetReader::new(want_voxel.as_slice()).unwrap();
        assert_eq!(cooked.chunk_count(), authored.chunk_count());
        assert!(
            cooked
                .directory()
                .iter()
                .any(|e| e.codec != inf_asset::BlockCodec::Raw),
            "no chunk compressed; the size arm above would be the only honest one"
        );
        for e in authored.directory() {
            assert_eq!(
                cooked.chunk_bytes(e.key).unwrap(),
                authored.chunk_bytes(e.key).unwrap(),
                "the cook changed the cave at chunk {:?}",
                e.key
            );
        }
    }

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
/// **The premise this arm shipped with was wrong** and P22.4's audit corrected
/// it: the Blueprint tick does **not** run inside the ECS schedule
/// (`RuntimeSim::run_all_with_args` is a serial `for` loop and no `SimSchedule`
/// runs on the player's fixed-step path), so four pool sizes execute the same
/// serial program and this cannot *discover* an ordering race. What it is worth —
/// a regression tripwire for the day the tick pass parallelizes, plus a pin that
/// nothing on that path reached for the process-global pool — is stated in full
/// on `destruct_probe.rs` and applies here verbatim.
///
/// Four borers on four volumes, so that the day there IS interleaving there is
/// something to interleave. The `field=` hash is the load-bearing half — a trace
/// of what each script recorded could agree while the chunks underneath diverged,
/// and the chunks are what a save, a replay and the next frame's colliders all
/// read.
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
                run.get(key)
                    .unwrap_or_else(|| panic!("probe printed {key}=")),
                reference,
                "the runtime carve's {key} depends on the ECS pool size \
                 (threads={n}) — it is not replay-safe"
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
