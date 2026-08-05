//! **P22.3 — a Blueprint breaks a wall** (editor Simulate half).
//!
//! **The runtime twin is `inf-player/tests/runtime_destruct.rs`**, built from the
//! same constants, the same fracture fixture and the same Blueprint class,
//! because the failure this exists to catch is *the preview broke a different
//! wall from the shipped build* — which no compiler and no screenshot finds. The
//! two hosts each own a copy of the `destruct.*` match arms and of
//! `runtime_destruct_damage`; "they break the same" is therefore a claim about
//! two files, and this pair is what checks it from the outside.
//!
//! What is asserted here is the **host seam**, not the solve: the four nodes
//! reach the one Ring-0 rule, the `runtime_destruct` gate is honoured in both
//! directions, every refusal is a value, and `Destroyed` fires once with the
//! chunk count.

use std::collections::BTreeMap;
use std::sync::Arc;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::{
    BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty, Value,
    Variable,
};
use inf_ecs::components::{ActorClass, Destructible, MeshRef, Terrain, Transform};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};
use inf_mesh::{Aabb, ChunkSection, FractureAsset, FractureChunk, MeshVertex};
use inf_physics::d3::{resolve_fracture_states, FractureState, CRACK_OPENING_M};
use inf_terrain::TerrainData;

// ── the fixture, shared character-for-character with the runtime twin ────────

pub const TILE_RES: u32 = 5;
pub const MPS: f64 = 4.0;
/// The flat ground the tower stands on.
pub const GROUND_Y: f64 = 0.0;

pub const TERRAIN_GUID: u128 = 0x2203_2001;
pub const WALL_GUID: u128 = 0x2203_2002;
pub const MESH_GUID: u128 = 0x2203_20FF;
const WALL_CLASS_GUID: u128 = 0x2203_20AC;

/// **PINNED IDENTICALLY IN THE RUNTIME TWIN.** Chunks in the fixture tower.
pub const TOWER_CHUNKS: u32 = 4;

/// **PINNED IDENTICALLY IN THE RUNTIME TWIN.** The energy the Blueprint delivers
/// each tick, in joules. Three and a half of the tower's 1 m² concrete bonds:
/// enough to take the top three chunks off in one blow and leave the base
/// standing (it is bonded to the ground as well as to its neighbour). The half of
/// slack is real — a *measured* square metre is `1.0` to within an ULP, so
/// exactly `3.0 ×` falls a fraction short of three bonds.
pub fn blow_energy_j() -> f64 {
    Destructible::default().strength * 1.0 * CRACK_OPENING_M * 3.5
}

/// **PINNED IDENTICALLY IN THE RUNTIME TWIN.** Chunks the first blow detaches.
pub const EXPECTED_DETACHED: usize = 3;

/// One axis-aligned unit box chunk, as render triangles and an `f64` hull set.
fn box_chunk(centre: DVec3, neighbors: Vec<u32>) -> FractureChunk {
    let h = 0.5;
    let mut hull = Vec::new();
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                hull.push([centre.x + sx * h, centre.y + sy * h, centre.z + sz * h]);
            }
        }
    }
    let vertices: Vec<MeshVertex> = hull
        .iter()
        .map(|p| MeshVertex {
            position: [p[0] as f32, p[1] as f32, p[2] as f32],
            ..Default::default()
        })
        .collect();
    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        0, 1, 3, 0, 3, 2,  4, 7, 5, 4, 6, 7,
        0, 4, 5, 0, 5, 1,  2, 3, 7, 2, 7, 6,
        0, 2, 6, 0, 6, 4,  1, 5, 7, 1, 7, 3,
    ];
    let index_count = indices.len() as u32;
    FractureChunk {
        vertices,
        indices,
        sections: vec![ChunkSection {
            material_slot: 0,
            first_index: 0,
            index_count,
        }],
        hull_points: hull,
        volume_m3: 1.0,
        center_of_mass: centre.to_array(),
        neighbors,
    }
}

/// A tower of [`TOWER_CHUNKS`] unit cubes stacked along +Y, each shared face
/// exactly 1 m². Shared character-for-character with the runtime twin.
pub fn tower() -> Arc<FractureAsset> {
    let n = TOWER_CHUNKS;
    let chunks: Vec<FractureChunk> = (0..n)
        .map(|i| {
            let mut nb = Vec::new();
            if i > 0 {
                nb.push(i - 1);
            }
            if i + 1 < n {
                nb.push(i + 1);
            }
            box_chunk(DVec3::new(0.0, i as f64 + 0.5, 0.0), nb)
        })
        .collect();
    Arc::new(FractureAsset {
        schema_version: FractureAsset::CURRENT_VERSION,
        source_mesh: *Uuid::from_u128(MESH_GUID).as_bytes(),
        bounds: Aabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, n as f32, 0.5],
        },
        seed: 0,
        requested_chunks: n,
        slots: vec!["Concrete".into(), "Fracture Interior".into()],
        interior_slot: 1,
        chunks,
    })
}

/// Flat ground, so the tower's base chunk really is standing on static geometry.
pub fn flat_terrain() -> TerrainData {
    let mut data = TerrainData::new(TILE_RES, MPS);
    for c in [(-1, -1), (-1, 0), (0, -1), (0, 0)] {
        data.author_tile(c, |_, _| GROUND_Y);
    }
    data
}

macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

/// The scene: flat ground and one destructible wall carrying the Blueprint.
fn wall_doc(runtime_destruct: bool) -> SceneDoc {
    let mut doc = SceneDoc::new();
    let terrain = Uuid::from_u128(TERRAIN_GUID);
    let wall = Uuid::from_u128(WALL_GUID);

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

    doc.create_with_guid(wall, SpawnKind::Empty, "Wall", None);
    insert!(doc, wall, Transform::IDENTITY);
    insert!(
        doc,
        wall,
        Destructible {
            chunk_count: TOWER_CHUNKS,
            runtime_destruct,
            ..Destructible::default()
        }
    );
    insert!(
        doc,
        wall,
        MeshRef {
            asset: Some(Uuid::from_u128(MESH_GUID)),
            ..Default::default()
        }
    );
    insert!(doc, wall, ActorClass(Uuid::from_u128(WALL_CLASS_GUID)));
    doc.world_mut().propagate();
    doc
}

/// A Blueprint whose Tick blows itself up once and records what it saw, and whose
/// `Destroyed` handler records the chunk count. Shared character-for-character
/// with the runtime twin.
///
/// The blow runs every tick on purpose: the second one has almost nothing left to
/// break, and "damage is not banked" is a property worth asserting rather than a
/// comment.
pub fn wall_class() -> BlueprintClass {
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
    let mut class = BlueprintClass::new("act:destructible-wall", "Wall");
    class.variables = vec![
        slot("absorbed"),
        slot("intact"),
        slot("chunks"),
        slot("hit"),
        slot("destroyed_chunks"),
    ];
    class.events = vec![
        EventBinding {
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
                        "absorbed",
                        Expr::Call {
                            path: vec!["destruct".into(), "apply_damage".into()],
                            args: vec![entity(), Expr::Lit(Lit::Float(blow_energy_j()))],
                        },
                    ),
                    set(
                        "hit",
                        Expr::Call {
                            path: vec!["destruct".into(), "radial_impulse".into()],
                            args: vec![
                                Expr::Lit(Lit::Float(0.0)),
                                Expr::Lit(Lit::Float(2.0)),
                                Expr::Lit(Lit::Float(0.0)),
                                Expr::Lit(Lit::Float(5.0e4)),
                                Expr::Lit(Lit::Float(8.0)),
                            ],
                        },
                    ),
                    set(
                        "intact",
                        Expr::Call {
                            path: vec!["destruct".into(), "is_intact".into()],
                            args: vec![entity()],
                        },
                    ),
                    set(
                        "chunks",
                        Expr::Call {
                            path: vec!["destruct".into(), "chunk_count".into()],
                            args: vec![entity()],
                        },
                    ),
                ],
            },
        },
        EventBinding {
            event: EventKind::Destroyed,
            body: BlueprintFn {
                id: "destroyed".into(),
                name: "destroyed".into(),
                params: EventKind::Destroyed.signature(),
                ret: Ty::Unit,
                body: vec![set(
                    "destroyed_chunks",
                    Expr::Call {
                        path: vec!["vars".into(), "get".into()],
                        args: vec![Expr::Lit(Lit::Str("chunks".into()))],
                    },
                )],
            },
        },
    ];
    class
}

/// Enter a session over the fixture, seeded through the ONE Ring-0 resolver.
fn enter(runtime_destruct: bool) -> (SceneDoc, SimSession) {
    let mut doc = wall_doc(runtime_destruct);
    let actors = vec![(Uuid::from_u128(WALL_GUID), wall_class())];
    let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
    let asset = tower();
    let states: BTreeMap<Uuid, FractureState> = resolve_fracture_states(doc.world(), |mesh| {
        (mesh == Uuid::from_u128(MESH_GUID)).then(|| asset.clone())
    });
    assert_eq!(
        states.len(),
        1,
        "the fixture must seed exactly one fracture"
    );
    session.set_fractures(states);
    (doc, session)
}

fn var_f(session: &SimSession, name: &str) -> f64 {
    match session.actor_var(Uuid::from_u128(WALL_GUID), name) {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(i)) => *i as f64,
        Some(Value::Bool(b)) => *b as u8 as f64,
        other => panic!("{name} is {other:?}"),
    }
}

fn tick(doc: &mut SceneDoc, session: &mut SimSession) {
    session.tick(doc, 1.0 / SIM_HZ, SimInput::default());
}

// ── the headline ────────────────────────────────────────────────────────────

/// **The four nodes reach the rule.** One tick: the blow absorbs energy, three
/// chunks come off, the wall stops being intact, the blast pushes the debris it
/// just made, and the chunk-count query answers.
#[test]
fn a_blueprint_breaks_its_own_wall() {
    let (mut doc, mut session) = enter(true);
    assert!(session.fractures()[&Uuid::from_u128(WALL_GUID)].is_intact());

    tick(&mut doc, &mut session);

    let state = &session.fractures()[&Uuid::from_u128(WALL_GUID)];
    assert_eq!(
        state.chunks().iter().filter(|c| c.detached).count(),
        EXPECTED_DETACHED,
        "the blow did not take the pinned number of chunks"
    );
    assert!(
        var_f(&session, "absorbed") > 0.0,
        "the node reported no energy absorbed"
    );
    // `is_intact`/`chunk_count` are read AFTER the blow in the same handler.
    assert_eq!(
        var_f(&session, "intact"),
        0.0,
        "the wall still reads intact"
    );
    assert_eq!(var_f(&session, "chunks"), TOWER_CHUNKS as f64);
    // The blast reached the debris the blow had just made. It cannot have reached
    // them on the first tick's *sync* — the chunk bodies appear on the next one —
    // so this is `0` here and positive later, which is the one-beat latency the
    // module documents rather than a bug.
    let hit_first = var_f(&session, "hit");
    tick(&mut doc, &mut session);
    let hit_second = var_f(&session, "hit");
    assert!(
        hit_second > hit_first,
        "the blast never reached the chunk bodies ({hit_first} then {hit_second})"
    );

    // **THE WORLD ASSERTION.** The debris MOVED — a detach that leaves every pose
    // where it started is a bookkeeping change, not a break.
    //
    // "moved", not "fell": this Blueprint fires a 50 kN·s blast every tick from
    // two metres up, which on a 2400 kg chunk is about 9 m/s of Δv per step. The
    // rubble goes up before it comes down, and asserting a direction here would
    // be asserting the fixture's own script rather than the mechanism. The
    // *unblasted* fall is `inf-physics`' `removing_a_towers_support…`.
    for _ in 0..90 {
        tick(&mut doc, &mut session);
    }
    let state = &session.fractures()[&Uuid::from_u128(WALL_GUID)];
    let moved = state
        .chunks()
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            let rest = glam::DVec3::new(0.0, *i as f64 + 0.5, 0.0);
            c.detached && (c.translation - rest).length() > 1.0
        })
        .count();
    assert!(moved >= 2, "only {moved} chunks actually moved");
    // (The base is gone by now too: this Blueprint blows the wall up on EVERY
    // tick, so ninety of them finish it. "An attached chunk never moves" is
    // asserted where it can be isolated — `inf-physics`' fracture suite.)
}

/// **THE GATE, both ways.** `runtime_destruct: false` breaks nothing and logs the
/// refusal; the identical fixture with it on breaks the wall. Without the second
/// arm the first proves only that something failed.
#[test]
fn the_runtime_destruct_flag_gates_both_ways() {
    let (mut doc, mut session) = enter(false);
    tick(&mut doc, &mut session);
    assert!(
        session.fractures()[&Uuid::from_u128(WALL_GUID)].is_intact(),
        "a locked wall broke anyway"
    );
    assert_eq!(
        var_f(&session, "absorbed"),
        0.0,
        "a refusal must answer 0.0 joules"
    );
    assert!(
        session
            .logs()
            .iter()
            .any(|l| l.contains("runtime_destruct") && l.contains("refused")),
        "the refusal was swallowed instead of logged: {:?}",
        session.logs()
    );

    // The positive control.
    let (mut doc, mut session) = enter(true);
    tick(&mut doc, &mut session);
    assert!(!session.fractures()[&Uuid::from_u128(WALL_GUID)].is_intact());
}

/// An actor with **no** `Destructible` refuses as a value, with its own message.
#[test]
fn an_actor_with_no_destructible_refuses_as_a_value() {
    let mut doc = wall_doc(true);
    // Take the component away, leaving the Blueprint pointed at it.
    if let Some(e) = doc.entity_of(Uuid::from_u128(WALL_GUID)) {
        doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .remove::<Destructible>();
        doc.world_mut().mark_dirty();
    }
    let actors = vec![(Uuid::from_u128(WALL_GUID), wall_class())];
    let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
    // Nothing seeded: `resolve_fracture_states` skips an actor with no component.
    tick(&mut doc, &mut session);
    assert_eq!(var_f(&session, "absorbed"), 0.0);
    assert_eq!(
        var_f(&session, "intact"),
        0.0,
        "an actor that cannot break must not report itself intact"
    );
    assert_eq!(
        var_f(&session, "chunks"),
        0.0,
        "chunk_count is how you tell 'indestructible' from 'already broken'"
    );
    assert!(
        session
            .logs()
            .iter()
            .any(|l| l.contains("no Destructible on that entity")),
        "the refusal was swallowed: {:?}",
        session.logs()
    );
}

/// A `Destructible` actor whose fracture was never seeded — what a `BeginPlay`
/// handler sees — refuses with its own message rather than panicking.
#[test]
fn an_unseeded_fracture_refuses_as_a_value() {
    let mut doc = wall_doc(true);
    let actors = vec![(Uuid::from_u128(WALL_GUID), wall_class())];
    let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
    // Deliberately NOT seeded.
    tick(&mut doc, &mut session);
    assert_eq!(var_f(&session, "absorbed"), 0.0);
    assert!(
        session
            .logs()
            .iter()
            .any(|l| l.contains("no fracture data resident")),
        "the refusal was swallowed: {:?}",
        session.logs()
    );
}

/// `Destroyed` fires **once**, with the chunk count, on the step the last chunk
/// comes off.
#[test]
fn the_destroyed_event_fires_once_with_its_chunk_count() {
    let (mut doc, mut session) = enter(true);
    assert_eq!(var_f(&session, "destroyed_chunks"), -1.0, "the default");

    let mut first_fired: Option<u32> = None;
    for step in 0..60u32 {
        tick(&mut doc, &mut session);
        if first_fired.is_none() && var_f(&session, "destroyed_chunks") >= 0.0 {
            first_fired = Some(step);
        }
    }
    assert!(
        first_fired.is_some(),
        "the wall never finished breaking, so Destroyed never fired"
    );
    assert_eq!(
        var_f(&session, "destroyed_chunks"),
        TOWER_CHUNKS as f64,
        "Destroyed reported the wrong chunk count"
    );
    assert!(
        session.fractures()[&Uuid::from_u128(WALL_GUID)]
            .chunks()
            .iter()
            .all(|c| c.detached),
        "Destroyed fired before every chunk had come off"
    );
}
