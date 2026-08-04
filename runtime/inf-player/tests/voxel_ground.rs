//! **P21.2 — gameplay stands on a cave floor** (shipped-player half).
//!
//! The runtime twin of `inf-editor-core/tests/simulate_voxel_ground.rs`: the same
//! holed heightfield, the same cave, the same Blueprint, run through
//! `RuntimeSim`'s `RuntimeHost` instead of the editor's `SimHost`. Both must land
//! on the same number, because the failure this exists to catch is *the preview let
//! me walk into the cave and the shipped build dropped me through the floor* —
//! which no compiler and no screenshot finds.
//!
//! Pure Ring-0 + runtime, headless CI: no GPU, no window, no pack.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::{
    BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty, Value,
    Variable,
};
use inf_ecs::components::{Terrain, Transform, VoxelVolume};
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::{resolve_voxel_volumes, RuntimeInput, RuntimeSim};
use inf_terrain::TerrainData;
use inf_voxel::{ChunkKey, VoxelChunk, VoxelData};

// ── the fixture, shared character-for-character with the editor twin ─────────

/// 5 × 5 samples at 1 m ⇒ one tile spanning `[0, 4]²`.
const TILE_RES: u32 = 5;
const MPS: f64 = 1.0;
/// The flat heightfield's world height.
const GROUND_Y: f64 = 10.0;
/// The world XZ the Blueprint probes — the holed sample, so the heightfield
/// answers `None` there and the cave has to.
const PROBE: (f64, f64) = (2.0, 2.0);
/// A world XZ one cell away from the hole, where the heightfield still answers.
const SOLID_PROBE: (f64, f64) = (0.5, 0.5);
/// The cave floor's world height under the hole (1 m voxels, so global sample
/// `y = 3` is solid and `y = 4` is air ⇒ the crossing is exactly here).
const CAVE_FLOOR_Y: f64 = 3.5;

const TERRAIN_GUID: u128 = 0x2102_0001;
const CAVE_GUID: u128 = 0x2102_0002;
const WALKER_GUID: u128 = 0x2102_0003;

/// The heightfield: flat at [`GROUND_Y`], with sample `(2, 2)` carved through.
fn holed_terrain() -> TerrainData {
    let mut data = TerrainData::new(TILE_RES, MPS);
    data.author_tile((0, 0), |_, _| GROUND_Y);
    data.get_tile_mut((0, 0))
        .expect("the tile was just authored")
        .set_hole(TILE_RES, 2, 2, true);
    data
}

/// The cave: one chunk of rock whose top surface sits at [`CAVE_FLOOR_Y`],
/// anchored at the world origin (so the sim map's anchor needs no fold).
fn cave_volume() -> VoxelData {
    let mut v = VoxelData::new(1.0);
    v.insert_chunk(
        ChunkKey::new(0, 0, 0),
        // A signed distance of `y − CAVE_FLOOR_Y` in voxels: negative below the
        // floor, positive above, crossing exactly at 3.5.
        VoxelChunk::from_fn(|_, j, _| j as f64 - CAVE_FLOOR_Y),
    );
    v.clear_dirty();
    v
}

/// The scene: a holed terrain, a `VoxelVolume` entity for the cave, and a walker
/// carrying the probe Blueprint.
fn cave_world(terrain_y: f64, cave_y: f64, holed: bool) -> EcsWorld {
    let mut world = EcsWorld::new();

    let t = world.spawn_with_guid(Uuid::from_u128(TERRAIN_GUID), "Terrain", None);
    let mut data = holed_terrain();
    if !holed {
        data.get_tile_mut((0, 0))
            .expect("tile")
            .set_hole(TILE_RES, 2, 2, false);
    }
    world
        .world_mut()
        .entity_mut(t)
        .insert(Transform::from_translation(DVec3::new(0.0, terrain_y, 0.0)));
    world.world_mut().entity_mut(t).insert(Terrain {
        meters_per_sample: MPS,
        tile_resolution: TILE_RES,
        data,
        ..Terrain::default()
    });

    let c = world.spawn_with_guid(Uuid::from_u128(CAVE_GUID), "Cave", None);
    world
        .world_mut()
        .entity_mut(c)
        .insert(Transform::from_translation(DVec3::new(0.0, cave_y, 0.0)));
    world
        .world_mut()
        .entity_mut(c)
        .insert(VoxelVolume::default());

    let w = world.spawn_with_guid(Uuid::from_u128(WALKER_GUID), "Walker", None);
    world.world_mut().entity_mut(w).insert(Transform::IDENTITY);

    world.mark_dirty();
    world.propagate();
    world
}

/// A Blueprint whose Tick stores `terrain.height_at` at two probes. Shared
/// character-for-character with the editor twin, because the whole point is that
/// the two hosts answer the same.
fn ground_probe_class() -> BlueprintClass {
    let probe = |name: &str, x: f64, z: f64| {
        Stmt::ExprStmt(Expr::Call {
            path: vec!["vars".into(), "set".into()],
            args: vec![
                Expr::Lit(Lit::Str(name.into())),
                Expr::Call {
                    path: vec!["terrain".into(), "height_at".into()],
                    args: vec![Expr::Lit(Lit::Float(x)), Expr::Lit(Lit::Float(z))],
                },
            ],
        })
    };
    let slot = |name: &str| Variable {
        name: name.into(),
        ty: Ty::Float,
        default: Lit::Float(-1.0),
        exposed: false,
    };
    let mut class = BlueprintClass::new("act:ground-probe", "Ground Probe");
    class.variables = vec![slot("hole"), slot("solid")];
    class.events = vec![EventBinding {
        event: EventKind::Tick,
        body: BlueprintFn {
            id: "tick".into(),
            name: "tick".into(),
            params: vec![Param {
                name: "dt".into(),
                ty: Ty::Float,
            }],
            ret: Ty::Unit,
            body: vec![
                probe("hole", PROBE.0, PROBE.1),
                probe("solid", SOLID_PROBE.0, SOLID_PROBE.1),
            ],
        },
    }];
    class
}

fn sim_with(world: EcsWorld) -> RuntimeSim {
    RuntimeSim::new(
        world,
        vec![(Uuid::from_u128(WALKER_GUID), ground_probe_class())],
        DVec2::ZERO,
        60.0,
    )
}

fn probe(sim: &RuntimeSim, name: &str) -> f64 {
    match sim.actor_var(Uuid::from_u128(WALKER_GUID), name) {
        Some(Value::Float(f)) => *f,
        other => panic!("{name} is {other:?}"),
    }
}

// ── the deliverable ──────────────────────────────────────────────────────────

/// **THE GATE.** A Blueprint calling `terrain.height_at` over a holed sample gets
/// the cave floor's `y` — not the seam's `0.0` (which is what "no ground" reads
/// as), and not the pre-carve height (which would mean the carve never reached the
/// query).
#[test]
fn a_blueprint_over_a_holed_sample_stands_on_the_cave_floor() {
    let world = cave_world(0.0, 0.0, true);
    // The fixture has to be genuinely holed, or every assertion below is vacuous.
    let t = world.entity_of(Uuid::from_u128(TERRAIN_GUID)).unwrap();
    let data = &world.world().get::<Terrain>(t).expect("terrain").data;
    assert!(data.is_hole_at(DVec2::new(PROBE.0, PROBE.1)));
    assert!(data.height_at(DVec2::new(PROBE.0, PROBE.1)).is_none());

    let mut sim = sim_with(world);
    sim.set_voxel_volumes(BTreeMap::from([(
        Uuid::from_u128(CAVE_GUID),
        cave_volume(),
    )]));
    sim.step_once(RuntimeInput::default());

    let hole = probe(&sim, "hole");
    assert!(
        (hole - CAVE_FLOOR_Y).abs() < 1e-9,
        "over the hole the blueprint should read the cave floor {CAVE_FLOOR_Y}, got {hole}"
    );
    assert_ne!(hole, 0.0, "a hole must not read as `no ground`");
    assert_ne!(hole, GROUND_Y, "the pre-carve height must not survive");
    // …and the unholed ground one cell away is unchanged: the voxel half extends
    // the heightfield, it does not replace it.
    let solid = probe(&sim, "solid");
    assert!((solid - GROUND_Y).abs() < 1e-9, "{solid}");

    // The read-only host accessor answers the same thing the Blueprint saw — it is
    // the seam a gate traces, and a second implementation there would be exactly
    // the drift this file exists to stop.
    assert_eq!(sim.terrain_height_at(PROBE.0, PROBE.1), hole);
}

/// The other side of the same rule: with **no** volume seeded, a holed sample
/// falls back to the seam's documented `0.0`. Without this the test above could
/// pass on a fixture that was never holed at all.
#[test]
fn a_hole_with_no_volume_seeded_falls_back_to_the_seam_default() {
    let mut sim = sim_with(cave_world(0.0, 0.0, true));
    sim.step_once(RuntimeInput::default());
    assert_eq!(probe(&sim, "hole"), 0.0);
    assert!((probe(&sim, "solid") - GROUND_Y).abs() < 1e-9);
}

/// The **terrain entity's transform** reaches both halves: a terrain lifted 5 m
/// lifts its heightfield answer, and the cave — anchored in world space by the
/// resolver — does not move with it.
#[test]
fn the_terrain_transform_lifts_only_the_heightfield_half() {
    let mut sim = sim_with(cave_world(5.0, 0.0, true));
    sim.set_voxel_volumes(BTreeMap::from([(
        Uuid::from_u128(CAVE_GUID),
        cave_volume(),
    )]));
    sim.step_once(RuntimeInput::default());

    assert!((probe(&sim, "solid") - (GROUND_Y + 5.0)).abs() < 1e-9);
    assert!(
        (probe(&sim, "hole") - CAVE_FLOOR_Y).abs() < 1e-9,
        "the cave is anchored in world space, not to the terrain entity"
    );
}

/// The resolver is what production seeds through (`attach_voxel_volumes`), so its
/// two load-bearing properties are pinned here: it keys by **entity**, and it folds
/// the **entity's translation** into the volume's world anchor.
#[test]
fn the_resolver_keys_by_entity_and_folds_the_entity_translation() {
    let payload = inf_voxel::build_voxel_asset(&cave_volume())
        .expect("the fixture volume builds an asset")
        .into_bytes();
    let asset = Uuid::from_u128(0x2102_0A55);

    // Lift the cave entity 20 m: the resolver must anchor the volume there.
    let mut world = cave_world(0.0, 20.0, true);
    let c = world.entity_of(Uuid::from_u128(CAVE_GUID)).unwrap();
    world
        .world_mut()
        .entity_mut(c)
        .insert(VoxelVolume::from_asset(asset));
    world.mark_dirty();
    world.propagate();

    let volumes = resolve_voxel_volumes(&world, |guid| (guid == asset).then(|| payload.clone()));
    let cave = Uuid::from_u128(CAVE_GUID);
    assert_eq!(
        volumes.keys().copied().collect::<Vec<_>>(),
        vec![cave],
        "the map is keyed by the ENTITY, not by the asset"
    );
    let surface = volumes[&cave]
        .surface_y_at(PROBE.0, PROBE.1)
        .expect("the cave has a floor");
    assert!(
        (surface - (CAVE_FLOOR_Y + 20.0)).abs() < 1e-9,
        "the entity translation must reach the anchor: {surface}"
    );

    // …and end to end: seeded from the resolver, the Blueprint reads the lifted
    // floor.
    let mut sim = sim_with(world);
    sim.set_voxel_volumes(volumes);
    sim.step_once(RuntimeInput::default());
    assert!((probe(&sim, "hole") - (CAVE_FLOOR_Y + 20.0)).abs() < 1e-9);

    // An asset the loader cannot serve is skipped, never a floorless volume.
    assert!(resolve_voxel_volumes(sim.world(), |_| None).is_empty());
}

/// The hole is a **terrain** fact, not a voxel one: the same probe over the same
/// terrain answers the heightfield before the carve and the cave after it.
#[test]
fn the_carve_is_what_moves_the_answer() {
    let mut sim = sim_with(cave_world(0.0, 0.0, false));
    sim.set_voxel_volumes(BTreeMap::from([(
        Uuid::from_u128(CAVE_GUID),
        cave_volume(),
    )]));
    sim.step_once(RuntimeInput::default());
    assert!(
        (probe(&sim, "hole") - GROUND_Y).abs() < 1e-9,
        "un-holed ground answers the heightfield even with a cave under it"
    );
}
