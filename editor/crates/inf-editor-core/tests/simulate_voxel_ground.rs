//! **P21.2 — gameplay stands on a cave floor** (editor Simulate half).
//!
//! A carve holes the heightfield, and `TerrainData::height_at` answers `None`
//! through the whole bilinear cell a holed corner poisons. That is the right
//! *terrain* answer and a useless *gameplay* one: a Blueprint reading
//! `terrain.height_at` at a cave mouth would get the seam's `0.0` default and drop
//! the character to sea level. This pins the combined query end to end — through
//! the real interpreter, the real `SimHost`, the real fixed step.
//!
//! **The runtime twin is `inf-player/tests/voxel_ground.rs`**, built from the same
//! constants and the same Blueprint class, because the failure this exists to catch
//! is *the preview let me walk into the cave and the shipped build dropped me
//! through the floor* — which no compiler and no screenshot finds.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::{
    BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty, Value,
    Variable,
};
use inf_ecs::components::{ActorClass, Terrain, Transform, VoxelVolume};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};
use inf_terrain::TerrainData;
use inf_voxel::{ChunkKey, VoxelChunk, VoxelData};

// ── the fixture, shared character-for-character with the runtime twin ─────────

/// 5 × 5 samples at 1 m ⇒ one tile spanning `[0, 4]²`.
pub const TILE_RES: u32 = 5;
pub const MPS: f64 = 1.0;
/// The flat heightfield's world height.
pub const GROUND_Y: f64 = 10.0;
/// The world XZ the Blueprint probes — the holed sample, so the heightfield
/// answers `None` there and the cave has to.
pub const PROBE: (f64, f64) = (2.0, 2.0);
/// A world XZ one cell away from the hole, where the heightfield still answers.
pub const SOLID_PROBE: (f64, f64) = (0.5, 0.5);
/// The cave floor's world height under the hole (1 m voxels, so global sample
/// `y = 3` is solid and `y = 4` is air ⇒ the crossing is exactly here).
pub const CAVE_FLOOR_Y: f64 = 3.5;

pub const TERRAIN_GUID: u128 = 0x2102_0001;
pub const CAVE_GUID: u128 = 0x2102_0002;
pub const WALKER_GUID: u128 = 0x2102_0003;

/// The heightfield: flat at [`GROUND_Y`], with sample `(2, 2)` carved through.
pub fn holed_terrain() -> TerrainData {
    let mut data = TerrainData::new(TILE_RES, MPS);
    data.author_tile((0, 0), |_, _| GROUND_Y);
    data.get_tile_mut((0, 0))
        .expect("the tile was just authored")
        .set_hole(TILE_RES, 2, 2, true);
    data
}

/// The cave: one chunk of rock whose top surface sits at [`CAVE_FLOOR_Y`],
/// anchored at the world origin (so the sim map's anchor needs no fold).
pub fn cave_volume() -> VoxelData {
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

macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

/// The scene: a holed terrain, a `VoxelVolume` entity for the cave, and a walker
/// carrying the probe Blueprint.
fn cave_doc() -> SceneDoc {
    let mut doc = SceneDoc::new();
    let terrain = Uuid::from_u128(TERRAIN_GUID);
    let cave = Uuid::from_u128(CAVE_GUID);
    let walker = Uuid::from_u128(WALKER_GUID);

    doc.create_with_guid(terrain, SpawnKind::Empty, "Terrain", None);
    insert!(doc, terrain, Transform::IDENTITY);
    insert!(
        doc,
        terrain,
        Terrain {
            meters_per_sample: MPS,
            tile_resolution: TILE_RES,
            data: holed_terrain(),
            ..Terrain::default()
        }
    );

    doc.create_with_guid(cave, SpawnKind::Empty, "Cave", None);
    insert!(doc, cave, Transform::IDENTITY);
    insert!(doc, cave, VoxelVolume::default());

    doc.create_with_guid(walker, SpawnKind::Empty, "Walker", None);
    insert!(doc, walker, Transform::IDENTITY);
    insert!(doc, walker, ActorClass(Uuid::from_u128(0x2102_00AC)));
    doc.world_mut().propagate();
    doc
}

/// A Blueprint whose Tick stores `terrain.height_at` at two probes. Shared
/// character-for-character with the runtime twin, because the whole point is that
/// the two hosts answer the same.
pub fn ground_probe_class() -> BlueprintClass {
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

fn probe(session: &SimSession, name: &str) -> f64 {
    match session.actor_var(Uuid::from_u128(WALKER_GUID), name) {
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
    let mut doc = cave_doc();
    // The fixture has to be genuinely holed, or every assertion below is vacuous.
    let terrain_e = doc.entity_of(Uuid::from_u128(TERRAIN_GUID)).unwrap();
    let data = &doc
        .world()
        .world()
        .get::<Terrain>(terrain_e)
        .expect("terrain")
        .data;
    assert!(data.is_hole_at(DVec2::new(PROBE.0, PROBE.1)));
    assert!(data.height_at(DVec2::new(PROBE.0, PROBE.1)).is_none());

    let mut session = SimSession::enter(
        &mut doc,
        vec![(Uuid::from_u128(WALKER_GUID), ground_probe_class())],
        glam::DVec2::ZERO,
        SIM_HZ,
    );
    session.set_voxel_volumes(BTreeMap::from([(
        Uuid::from_u128(CAVE_GUID),
        cave_volume(),
    )]));
    session.step_once(&mut doc, SimInput::default());

    let hole = probe(&session, "hole");
    assert!(
        (hole - CAVE_FLOOR_Y).abs() < 1e-9,
        "over the hole the blueprint should read the cave floor {CAVE_FLOOR_Y}, got {hole}"
    );
    assert_ne!(hole, 0.0, "a hole must not read as `no ground`");
    assert_ne!(hole, GROUND_Y, "the pre-carve height must not survive");
    // …and the unholed ground one cell away is unchanged: the voxel half extends
    // the heightfield, it does not replace it.
    let solid = probe(&session, "solid");
    assert!((solid - GROUND_Y).abs() < 1e-9, "{solid}");
}

/// The other side of the same rule: with **no** volume seeded, a holed sample
/// falls back to the seam's documented `0.0`. Without this the test above could
/// pass on a fixture that was never holed at all.
#[test]
fn a_hole_with_no_volume_seeded_falls_back_to_the_seam_default() {
    let mut doc = cave_doc();
    let mut session = SimSession::enter(
        &mut doc,
        vec![(Uuid::from_u128(WALKER_GUID), ground_probe_class())],
        glam::DVec2::ZERO,
        SIM_HZ,
    );
    session.step_once(&mut doc, SimInput::default());
    assert_eq!(probe(&session, "hole"), 0.0);
    assert!((probe(&session, "solid") - GROUND_Y).abs() < 1e-9);
}

/// The **terrain entity's transform** reaches both halves: a terrain lifted 5 m
/// lifts its heightfield answer, and the cave — anchored in world space by the
/// resolver — does not move with it.
#[test]
fn the_terrain_transform_lifts_only_the_heightfield_half() {
    let mut doc = cave_doc();
    let terrain = Uuid::from_u128(TERRAIN_GUID);
    insert!(
        doc,
        terrain,
        Transform::from_translation(DVec3::new(0.0, 5.0, 0.0))
    );
    doc.world_mut().propagate();

    let mut session = SimSession::enter(
        &mut doc,
        vec![(Uuid::from_u128(WALKER_GUID), ground_probe_class())],
        glam::DVec2::ZERO,
        SIM_HZ,
    );
    session.set_voxel_volumes(BTreeMap::from([(
        Uuid::from_u128(CAVE_GUID),
        cave_volume(),
    )]));
    session.step_once(&mut doc, SimInput::default());

    assert!((probe(&session, "solid") - (GROUND_Y + 5.0)).abs() < 1e-9);
    assert!(
        (probe(&session, "hole") - CAVE_FLOOR_Y).abs() < 1e-9,
        "the cave is anchored in world space, not to the terrain entity"
    );
}

/// The resolver is what production seeds through (Ring 2's `sim_start`), so its
/// two load-bearing properties are pinned here: it keys by **entity**, and it
/// folds the **entity's translation** into the volume's world anchor.
#[test]
fn the_resolver_keys_by_entity_and_folds_the_entity_translation() {
    let payload = inf_voxel::build_voxel_asset(&cave_volume())
        .expect("the fixture volume builds an asset")
        .into_bytes();
    let asset = Uuid::from_u128(0x2102_0A55);

    let mut doc = cave_doc();
    let cave = Uuid::from_u128(CAVE_GUID);
    insert!(doc, cave, VoxelVolume::from_asset(asset));
    // Lift the cave entity 20 m: the resolver must anchor the volume there.
    insert!(
        doc,
        cave,
        Transform::from_translation(DVec3::new(0.0, 20.0, 0.0))
    );
    doc.world_mut().propagate();

    let volumes = inf_editor_core::simulate::resolve_voxel_volumes(&doc, |guid| {
        (guid == asset).then(|| payload.clone())
    });
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

    // An asset the loader cannot serve is skipped, never a floorless volume.
    let none = inf_editor_core::simulate::resolve_voxel_volumes(&doc, |_| None);
    assert!(none.is_empty());
}

/// The hole is a **terrain** fact, not a voxel one: the same probe over the same
/// terrain answers the heightfield before the carve and the cave after it.
#[test]
fn the_carve_is_what_moves_the_answer() {
    let mut doc = cave_doc();
    // Heal the hole: the fixture terrain is otherwise identical.
    let terrain_e = doc.entity_of(Uuid::from_u128(TERRAIN_GUID)).unwrap();
    doc.world_mut()
        .world_mut()
        .get_mut::<Terrain>(terrain_e)
        .expect("terrain")
        .data
        .get_tile_mut((0, 0))
        .expect("tile")
        .set_hole(TILE_RES, 2, 2, false);

    let mut session = SimSession::enter(
        &mut doc,
        vec![(Uuid::from_u128(WALKER_GUID), ground_probe_class())],
        glam::DVec2::ZERO,
        SIM_HZ,
    );
    session.set_voxel_volumes(BTreeMap::from([(
        Uuid::from_u128(CAVE_GUID),
        cave_volume(),
    )]));
    session.step_once(&mut doc, SimInput::default());
    assert!(
        (probe(&session, "hole") - GROUND_Y).abs() < 1e-9,
        "un-holed ground answers the heightfield even with a cave under it"
    );
}
