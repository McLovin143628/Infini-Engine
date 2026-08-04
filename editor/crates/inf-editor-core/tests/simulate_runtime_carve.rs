//! **P21.4 — a Blueprint digs, and the ground opens** (editor Simulate half).
//!
//! P21.2 let gameplay *stand on* a cave an author had carved. This is the other
//! direction: a Blueprint carving one itself, inside a fixed step, and the
//! heightfield opening a mouth over it through the same coupling rule the editor's
//! carve brush runs.
//!
//! **The runtime twin is `inf-player/tests/runtime_carve.rs`**, built from the same
//! constants, the same Blueprint class and the same **pinned integers**, because
//! the failure this exists to catch is *the preview dug a different hole from the
//! shipped build* — which no compiler and no screenshot finds. The two hosts each
//! own a copy of the `voxel.*` match arms and of `runtime_voxel_op`; "they carve
//! the same" is therefore a claim about two files, and this pair is what checks it
//! from the outside.

use std::collections::BTreeMap;

use glam::DVec2;
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

// ── the fixture, shared character-for-character with the runtime twin ────────

/// 5 × 5 samples at 1 m ⇒ one tile spanning `[0, 4]²`.
pub const TILE_RES: u32 = 5;
pub const MPS: f64 = 1.0;
/// The flat, **unholed** heightfield's world height — and the rock's top surface,
/// so the ground and the volume meet exactly.
pub const GROUND_Y: f64 = 8.0;

/// The dig: a ball centred on the surface, so it removes rock *and* crosses the
/// height samples.
pub const DIG_CENTER: (f64, f64, f64) = (2.0, GROUND_Y, 2.0);
pub const DIG_RADIUS_M: f64 = 2.0;

/// The XZ the Blueprint probes: the centre of the dig, holed after one tick.
pub const PROBE: (f64, f64) = (2.0, 2.0);
/// A corner of the tile, four metres away — outside the ball, so the heightfield
/// still answers there and "the terrain opened" is not "the terrain vanished".
pub const OUTSIDE_PROBE: (f64, f64) = (0.0, 0.0);

pub const TERRAIN_GUID: u128 = 0x2104_0001;
/// The digger **is** the volume entity: `vars::get("entity")` is how a node names
/// the volume it carves, and the IR has no way to name another entity (the same
/// limit the audio and physics kits live under).
pub const DIGGER_GUID: u128 = 0x2104_0002;
const DIGGER_CLASS_GUID: u128 = 0x2104_00AC;

/// **PINNED IDENTICALLY IN THE RUNTIME TWIN.** Rock samples the first tick's ball
/// removes, at 1 m voxels — so `removed_m3` is this number of cubic metres. An
/// integer, because the count is exact by construction (`OpReport` counts
/// samples), and pinned because "some rock moved" would pass on one voxel.
pub const EXPECTED_REMOVED_VOXELS: u64 = 10;

/// **PINNED IDENTICALLY IN THE RUNTIME TWIN.** Height **samples** the first
/// tick's ball opens: the nine of the 3 × 3 block around (2, 2) whose surface
/// point is *strictly* inside the ball. `(0, 2)` is exactly 2 m away and stays
/// closed — the documented boundary rule.
pub const EXPECTED_HOLED_SAMPLES: usize = 9;

/// **PINNED IDENTICALLY IN THE RUNTIME TWIN.** Integer world XZ points at which
/// `is_hole_at` answers true — **sixteen**, not nine, because a holed sample
/// poisons its whole bilinear cell (the P21.2 rule: `height_at` cannot
/// interpolate across a corner that has no height). Nine samples at (1..3)²
/// poison the sixteen cells of (0..3)². Both numbers are pinned because the
/// difference between them is exactly the thing an author trips over.
pub const EXPECTED_POISONED_POINTS: usize = 16;

/// **PINNED IDENTICALLY IN THE RUNTIME TWIN.** Where the combined ground query
/// lands over the hole after the dig: the bottom of the ball, two metres under
/// grade.
pub const EXPECTED_FLOOR_Y: f64 = GROUND_Y - DIG_RADIUS_M;

/// A flat, un-holed heightfield at [`GROUND_Y`].
pub fn flat_terrain() -> TerrainData {
    let mut data = TerrainData::new(TILE_RES, MPS);
    data.author_tile((0, 0), |_, _| GROUND_Y);
    data
}

/// The rock: one chunk whose top surface is exactly [`GROUND_Y`], anchored at the
/// world origin. Signed distance `j − GROUND_Y` in voxels — solid below, air
/// above, crossing on the terrain's own plane.
pub fn rock_volume() -> VoxelData {
    let mut v = VoxelData::new(MPS);
    v.insert_chunk(
        ChunkKey::new(0, 0, 0),
        VoxelChunk::from_fn(|_, j, _| j as f64 - GROUND_Y),
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

/// The scene: flat terrain over solid rock, and one digger entity carrying both
/// the volume and the Blueprint.
fn dig_doc(runtime_carve: bool) -> SceneDoc {
    let mut doc = SceneDoc::new();
    let terrain = Uuid::from_u128(TERRAIN_GUID);
    let digger = Uuid::from_u128(DIGGER_GUID);

    doc.create_with_guid(terrain, SpawnKind::Empty, "Terrain", None);
    insert!(doc, terrain, Transform::IDENTITY);
    insert!(
        doc,
        terrain,
        Terrain {
            meters_per_sample: MPS,
            tile_resolution: TILE_RES,
            data: flat_terrain(),
            ..Terrain::default()
        }
    );

    doc.create_with_guid(digger, SpawnKind::Empty, "Digger", None);
    insert!(doc, digger, Transform::IDENTITY);
    insert!(
        doc,
        digger,
        VoxelVolume {
            voxel_size_m: MPS,
            runtime_carve,
            ..VoxelVolume::default()
        }
    );
    insert!(doc, digger, ActorClass(Uuid::from_u128(DIGGER_CLASS_GUID)));
    doc.world_mut().propagate();
    doc
}

/// A Blueprint whose Tick digs once and records what it saw. Shared
/// character-for-character with the runtime twin.
///
/// `flow.do_once` would be the authored way to dig on the first tick only; here
/// the dig runs every tick on purpose, because a carve is **idempotent** and the
/// second tick removing nothing is itself a property worth asserting.
pub fn digger_class() -> BlueprintClass {
    let entity = || Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str("entity".into()))],
    };
    let set = |name: &str, value: Expr| {
        Stmt::ExprStmt(Expr::Call {
            path: vec!["vars".into(), "set".into()],
            args: vec![Expr::Lit(Lit::Str(name.into())), value],
        })
    };
    let slot = |name: &str| Variable {
        name: name.into(),
        ty: Ty::Float,
        default: Lit::Float(-1.0),
        exposed: false,
    };
    let mut class = BlueprintClass::new("act:runtime-digger", "Digger");
    class.variables = vec![
        slot("removed"),
        slot("ground"),
        slot("outside"),
        slot("vsurf"),
    ];
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
                set(
                    "removed",
                    Expr::Call {
                        path: vec!["voxel".into(), "carve_sphere".into()],
                        args: vec![
                            entity(),
                            Expr::Lit(Lit::Float(DIG_CENTER.0)),
                            Expr::Lit(Lit::Float(DIG_CENTER.1)),
                            Expr::Lit(Lit::Float(DIG_CENTER.2)),
                            Expr::Lit(Lit::Float(DIG_RADIUS_M)),
                        ],
                    },
                ),
                // Read AFTER the dig, in the same tick: the whole claim is that a
                // carve is visible to the queries it precedes.
                set(
                    "ground",
                    Expr::Call {
                        path: vec!["terrain".into(), "height_at".into()],
                        args: vec![
                            Expr::Lit(Lit::Float(PROBE.0)),
                            Expr::Lit(Lit::Float(PROBE.1)),
                        ],
                    },
                ),
                set(
                    "outside",
                    Expr::Call {
                        path: vec!["terrain".into(), "height_at".into()],
                        args: vec![
                            Expr::Lit(Lit::Float(OUTSIDE_PROBE.0)),
                            Expr::Lit(Lit::Float(OUTSIDE_PROBE.1)),
                        ],
                    },
                ),
                set(
                    "vsurf",
                    Expr::Call {
                        path: vec!["voxel".into(), "ground_height".into()],
                        args: vec![
                            Expr::Lit(Lit::Float(PROBE.0)),
                            Expr::Lit(Lit::Float(PROBE.1)),
                        ],
                    },
                ),
            ],
        },
    }];
    class
}

fn var(session: &SimSession, name: &str) -> f64 {
    match session.actor_var(Uuid::from_u128(DIGGER_GUID), name) {
        Some(Value::Float(f)) => *f,
        other => panic!("{name} is {other:?}"),
    }
}

/// `(holed height samples, poisoned integer query points)` — the two counts a
/// carve produces, which are deliberately not the same number.
fn hole_counts(doc: &SceneDoc) -> (usize, usize) {
    let e = doc
        .entity_of(Uuid::from_u128(TERRAIN_GUID))
        .expect("terrain");
    let data = &doc
        .world()
        .world()
        .get::<Terrain>(e)
        .expect("terrain component")
        .data;
    let mut samples = 0;
    let mut points = 0;
    if let Some(tile) = data.get_tile((0, 0)) {
        for i in 0..TILE_RES {
            for j in 0..TILE_RES {
                if tile.is_hole(TILE_RES, i, j) {
                    samples += 1;
                }
                if data.is_hole_at(DVec2::new(i as f64 * MPS, j as f64 * MPS)) {
                    points += 1;
                }
            }
        }
    }
    (samples, points)
}

fn session_over(doc: &mut SceneDoc) -> SimSession {
    let mut session = SimSession::enter(
        doc,
        vec![(Uuid::from_u128(DIGGER_GUID), digger_class())],
        glam::DVec2::ZERO,
        SIM_HZ,
    );
    session.set_voxel_volumes(BTreeMap::from([(
        Uuid::from_u128(DIGGER_GUID),
        rock_volume(),
    )]));
    session
}

// ── the deliverable ──────────────────────────────────────────────────────────

/// **THE GATE.** One tick of a Blueprint carve removes rock, opens the ground over
/// it, and the very next query in the same tick reads the new floor.
#[test]
fn a_blueprint_carve_removes_rock_and_opens_the_ground() {
    let mut doc = dig_doc(true);
    // ANTI-VACUITY: before the tick, nothing is holed and the ground is grade.
    assert_eq!(hole_counts(&doc), (0, 0), "the fixture starts unholed");

    let mut session = session_over(&mut doc);
    session.step_once(&mut doc, SimInput::default());

    // The exact volume, in cubic metres — 1 m voxels, so the number is the count.
    assert_eq!(
        var(&session, "removed"),
        EXPECTED_REMOVED_VOXELS as f64,
        "the dig removed a different volume than the twin does"
    );
    // The mouth: exactly the samples strictly inside the ball — and the wider
    // set of query points their poison covers.
    assert_eq!(
        hole_counts(&doc),
        (EXPECTED_HOLED_SAMPLES, EXPECTED_POISONED_POINTS)
    );
    // Over the mouth, the combined query has fallen to the new floor…
    let ground = var(&session, "ground");
    assert!(
        (ground - EXPECTED_FLOOR_Y).abs() < 1e-9,
        "over the fresh hole the blueprint reads {ground}, not the dug floor \
         {EXPECTED_FLOOR_Y}"
    );
    assert_ne!(ground, 0.0, "a hole must not read as `no ground`");
    // …while four metres away the heightfield is untouched.
    assert!(
        (var(&session, "outside") - GROUND_Y).abs() < 1e-9,
        "the dig moved ground it never reached"
    );
    // `voxel.ground_height` answers the voxel half alone, and here the two agree
    // because the hole hands the combined query straight to it.
    assert!((var(&session, "vsurf") - EXPECTED_FLOOR_Y).abs() < 1e-9);
}

/// **THE `runtime_carve` GATE — the reason the field was frozen into schema v19.**
/// With permission withheld the identical Blueprint on the identical world removes
/// nothing, opens nothing, and reads grade.
#[test]
fn a_locked_volume_refuses_the_same_blueprint() {
    let mut doc = dig_doc(false);
    let mut session = session_over(&mut doc);
    session.step_once(&mut doc, SimInput::default());

    assert_eq!(var(&session, "removed"), 0.0, "a locked volume was carved");
    assert_eq!(
        hole_counts(&doc),
        (0, 0),
        "a locked volume opened the ground"
    );
    assert!(
        (var(&session, "ground") - GROUND_Y).abs() < 1e-9,
        "the ground moved under a refused carve"
    );
    // The refusal is REPORTED, not swallowed — the component's own doc contract.
    assert!(
        session
            .logs()
            .iter()
            .any(|l| l.contains("carve_sphere") && l.contains("runtime_carve")),
        "a refused carve said nothing: {:?}",
        session.logs()
    );
}

/// The carve is **idempotent**: the second tick's identical ball removes nothing,
/// which is what makes a replayed step after a rollback safe.
#[test]
fn the_second_tick_removes_nothing() {
    let mut doc = dig_doc(true);
    let mut session = session_over(&mut doc);
    session.step_once(&mut doc, SimInput::default());
    assert_eq!(var(&session, "removed"), EXPECTED_REMOVED_VOXELS as f64);
    session.step_once(&mut doc, SimInput::default());
    assert_eq!(
        var(&session, "removed"),
        0.0,
        "the rock was already gone; carving it again reported volume"
    );
    // The world did not un-dig itself either.
    assert_eq!(
        hole_counts(&doc),
        (EXPECTED_HOLED_SAMPLES, EXPECTED_POISONED_POINTS)
    );
    assert!((var(&session, "ground") - EXPECTED_FLOOR_Y).abs() < 1e-9);
}

/// Two sessions over the same document produce the same trace, bit for bit — the
/// determinism seam, asserted on the *carve* rather than only on the query.
#[test]
fn two_runs_of_the_same_dig_agree_bit_for_bit() {
    let trace = |_: ()| -> Vec<[u64; 4]> {
        let mut doc = dig_doc(true);
        let mut session = session_over(&mut doc);
        (0..8)
            .map(|_| {
                session.step_once(&mut doc, SimInput::default());
                [
                    var(&session, "removed").to_bits(),
                    var(&session, "ground").to_bits(),
                    var(&session, "outside").to_bits(),
                    var(&session, "vsurf").to_bits(),
                ]
            })
            .collect()
    };
    let a = trace(());
    let b = trace(());
    assert_eq!(a, b);
    // ANTI-VACUITY: the trace really moves — the first tick differs from the rest.
    assert_ne!(a[0], a[1], "nothing happened on the first tick");
}
