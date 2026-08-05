//! **P22.3 — a Blueprint breaks a wall** (shipped-player half).
//!
//! The runtime twin of `inf-editor-core/tests/simulate_destruct.rs`: the same
//! flat ground under the same four-cube tower, the same Blueprint, the same
//! **pinned constants** — run through `RuntimeSim`'s `RuntimeHost` instead of the
//! editor's `SimHost`. Both must land on the same numbers, because the failure
//! this exists to catch is *the preview broke a different wall from the shipped
//! build*, which no compiler and no screenshot finds. The two hosts each own a
//! copy of the `destruct.*` match arms and of `runtime_destruct_damage`; "they
//! break the same" is therefore a claim about two files, and this pair is what
//! checks it from the outside.
//!
//! Pure Ring-0 + runtime, headless CI: no GPU, no window, no pack.

use std::collections::BTreeMap;
use std::sync::Arc;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::{
    BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty, Value,
    Variable,
};
use inf_ecs::components::{Destructible, MeshRef, Terrain, Transform};
use inf_ecs::EcsWorld;
use inf_mesh::{Aabb, ChunkSection, FractureAsset, FractureChunk, MeshVertex};
use inf_physics::d3::{resolve_fracture_states, FractureState, CRACK_OPENING_M};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_terrain::TerrainData;

// ── the fixture, shared character-for-character with the editor twin ────────

pub const TILE_RES: u32 = 5;
pub const MPS: f64 = 4.0;
/// The flat ground the tower stands on.
pub const GROUND_Y: f64 = 0.0;

pub const TERRAIN_GUID: u128 = 0x2203_2001;
pub const WALL_GUID: u128 = 0x2203_2002;
pub const MESH_GUID: u128 = 0x2203_20FF;

/// **PINNED IDENTICALLY IN THE EDITOR TWIN.** Chunks in the fixture tower.
pub const TOWER_CHUNKS: u32 = 4;

/// **PINNED IDENTICALLY IN THE EDITOR TWIN.** The energy the Blueprint delivers
/// each tick, in joules. Three and a half of the tower's 1 m² concrete bonds.
pub fn blow_energy_j() -> f64 {
    Destructible::default().strength * 1.0 * CRACK_OPENING_M * 3.5
}

/// **PINNED IDENTICALLY IN THE EDITOR TWIN.** Chunks the first blow detaches.
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
/// exactly 1 m². Shared character-for-character with the editor twin.
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

/// The world: flat ground and one destructible wall.
fn wall_world(runtime_destruct: bool, with_component: bool) -> EcsWorld {
    let mut world = EcsWorld::new();

    let t = world.spawn_with_guid(Uuid::from_u128(TERRAIN_GUID), "Terrain", None);
    world.world_mut().entity_mut(t).insert(Transform::IDENTITY);
    world.world_mut().entity_mut(t).insert(Terrain {
        meters_per_sample: MPS,
        tile_resolution: TILE_RES,
        data: flat_terrain(),
        ..Terrain::default()
    });

    let w = world.spawn_with_guid(Uuid::from_u128(WALL_GUID), "Wall", None);
    world.world_mut().entity_mut(w).insert(Transform::IDENTITY);
    if with_component {
        world.world_mut().entity_mut(w).insert(Destructible {
            chunk_count: TOWER_CHUNKS,
            runtime_destruct,
            ..Destructible::default()
        });
    }
    world.world_mut().entity_mut(w).insert(MeshRef {
        asset: Some(Uuid::from_u128(MESH_GUID)),
        ..Default::default()
    });
    world.mark_dirty();
    world.propagate();
    world
}

/// A Blueprint whose Tick blows itself up and records what it saw, and whose
/// `Destroyed` handler records the chunk count. Shared character-for-character
/// with the editor twin.
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
        Variable {
            name: "destroyed_fires".into(),
            ty: Ty::Float,
            default: Lit::Float(0.0),
            exposed: false,
        },
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
                body: vec![
                    set(
                        "destroyed_chunks",
                        Expr::Call {
                            path: vec!["vars".into(), "get".into()],
                            args: vec![Expr::Lit(Lit::Str("chunks".into()))],
                        },
                    ),
                    // **Counted, not latched.** The handler used to ASSIGN, so a
                    // second (or twentieth) firing produced the same variable
                    // value and a test named `fires_once` could not tell one from
                    // many. Incrementing is what makes "once" checkable.
                    set(
                        "destroyed_fires",
                        Expr::Binary(
                            inf_blueprint::BinOp::Add,
                            Box::new(Expr::Call {
                                path: vec!["vars".into(), "get".into()],
                                args: vec![Expr::Lit(Lit::Str("destroyed_fires".into()))],
                            }),
                            Box::new(Expr::Lit(Lit::Float(1.0))),
                        ),
                    ),
                ],
            },
        },
    ];
    class
}

fn sim(runtime_destruct: bool, seed: bool) -> RuntimeSim {
    let world = wall_world(runtime_destruct, true);
    let mut sim = RuntimeSim::new(
        world,
        vec![(Uuid::from_u128(WALL_GUID), wall_class())],
        DVec2::ZERO,
        60.0,
    );
    if seed {
        let asset = tower();
        let states: BTreeMap<Uuid, FractureState> = resolve_fracture_states(sim.world(), |mesh| {
            (mesh == Uuid::from_u128(MESH_GUID)).then(|| asset.clone())
        });
        assert_eq!(
            states.len(),
            1,
            "the fixture must seed exactly one fracture"
        );
        sim.set_fractures(states);
    }
    sim
}

fn var_f(sim: &RuntimeSim, name: &str) -> f64 {
    match sim.actor_var(Uuid::from_u128(WALL_GUID), name) {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(i)) => *i as f64,
        Some(Value::Bool(b)) => *b as u8 as f64,
        other => panic!("{name} is {other:?}"),
    }
}

// ── the headline ────────────────────────────────────────────────────────────

/// **The four nodes reach the rule** in the shipped host too.
#[test]
fn a_blueprint_breaks_its_own_wall() {
    let mut sim = sim(true, true);
    assert!(sim.fractures()[&Uuid::from_u128(WALL_GUID)].is_intact());

    sim.step_once(RuntimeInput::default());

    let state = &sim.fractures()[&Uuid::from_u128(WALL_GUID)];
    assert_eq!(
        state.chunks().iter().filter(|c| c.detached).count(),
        EXPECTED_DETACHED,
        "the blow did not take the pinned number of chunks"
    );
    assert!(var_f(&sim, "absorbed") > 0.0);
    assert_eq!(var_f(&sim, "intact"), 0.0);
    assert_eq!(var_f(&sim, "chunks"), TOWER_CHUNKS as f64);

    let hit_first = var_f(&sim, "hit");
    sim.step_once(RuntimeInput::default());
    assert!(
        var_f(&sim, "hit") > hit_first,
        "the blast never reached the chunk bodies"
    );

    for _ in 0..90 {
        sim.step_once(RuntimeInput::default());
    }
    let state = &sim.fractures()[&Uuid::from_u128(WALL_GUID)];
    let moved = state
        .chunks()
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            let rest = DVec3::new(0.0, *i as f64 + 0.5, 0.0);
            c.detached && (c.translation - rest).length() > 1.0
        })
        .count();
    assert!(moved >= 2, "only {moved} chunks actually moved");
}

/// **THE GATE, both ways** — the shipped half of the editor twin's arm.
#[test]
fn the_runtime_destruct_flag_gates_both_ways() {
    let mut off = sim(false, true);
    off.step_once(RuntimeInput::default());
    assert!(
        off.fractures()[&Uuid::from_u128(WALL_GUID)].is_intact(),
        "a locked wall broke anyway"
    );
    assert_eq!(var_f(&off, "absorbed"), 0.0);
    assert!(
        off.logs()
            .iter()
            .any(|l| l.contains("runtime_destruct") && l.contains("refused")),
        "the refusal was swallowed: {:?}",
        off.logs()
    );

    let mut on = sim(true, true);
    on.step_once(RuntimeInput::default());
    assert!(!on.fractures()[&Uuid::from_u128(WALL_GUID)].is_intact());
}

/// Refusals are values in the shipped host too: no component, and no seeded
/// fracture (what a `BeginPlay` handler sees).
#[test]
fn every_refusal_is_a_value() {
    // No component at all.
    let world = wall_world(true, false);
    let mut bare = RuntimeSim::new(
        world,
        vec![(Uuid::from_u128(WALL_GUID), wall_class())],
        DVec2::ZERO,
        60.0,
    );
    bare.step_once(RuntimeInput::default());
    assert_eq!(var_f(&bare, "absorbed"), 0.0);
    assert_eq!(var_f(&bare, "intact"), 0.0);
    assert_eq!(var_f(&bare, "chunks"), 0.0);
    assert!(
        bare.logs()
            .iter()
            .any(|l| l.contains("no Destructible on that entity")),
        "the refusal was swallowed: {:?}",
        bare.logs()
    );

    // A component, but nothing seeded.
    let mut unseeded = sim(true, false);
    unseeded.step_once(RuntimeInput::default());
    assert_eq!(var_f(&unseeded, "absorbed"), 0.0);
    assert!(
        unseeded
            .logs()
            .iter()
            .any(|l| l.contains("no fracture data resident")),
        "the refusal was swallowed: {:?}",
        unseeded.logs()
    );
}

/// `Destroyed` fires once, with the chunk count.
#[test]
fn the_destroyed_event_fires_once_with_its_chunk_count() {
    let mut sim = sim(true, true);
    assert_eq!(var_f(&sim, "destroyed_chunks"), -1.0);
    for _ in 0..60 {
        sim.step_once(RuntimeInput::default());
    }
    assert_eq!(
        var_f(&sim, "destroyed_chunks"),
        TOWER_CHUNKS as f64,
        "Destroyed reported the wrong chunk count"
    );
    // **ONCE.** The handler counts rather than latches, so this is the assertion
    // the test's name claims.
    assert_eq!(
        var_f(&sim, "destroyed_fires"),
        1.0,
        "Destroyed fired {} times",
        var_f(&sim, "destroyed_fires")
    );
}

// ── the point of the pair ───────────────────────────────────────────────────

/// **THE TWIN GATE: preview == shipped, bit for bit.**
///
/// The two hosts own two copies of the `destruct.*` arms and two copies of
/// `runtime_destruct_damage`, so "they break the same" is a claim about two
/// files. This runs the identical fixture through both and compares the
/// **fracture state as raw bits** — every chunk's detached flag, order, age and
/// pose — rather than only the floats the Blueprint happened to record, because
/// the two can disagree and the state is the authority (the P21.4 "assert the
/// WORLD, not the report" law).
#[test]
fn the_editor_and_the_shipped_host_break_the_same_wall() {
    const STEPS: usize = 45;

    // Shipped.
    let mut shipped = sim(true, true);
    for _ in 0..STEPS {
        shipped.step_once(RuntimeInput::default());
    }
    let shipped_bits = shipped.fractures()[&Uuid::from_u128(WALL_GUID)].bits();
    let shipped_absorbed = var_f(&shipped, "absorbed");

    // Editor preview, built from the same constants through the same Ring-0
    // resolver. (`inf-editor-core` is a dev-dependency here for exactly this.)
    let mut doc = inf_editor_core::scene::SceneDoc::new();
    let terrain = Uuid::from_u128(TERRAIN_GUID);
    let wall = Uuid::from_u128(WALL_GUID);
    doc.create_with_guid(
        terrain,
        inf_editor_core::ipc::SpawnKind::Empty,
        "Terrain",
        None,
    );
    doc.create_with_guid(wall, inf_editor_core::ipc::SpawnKind::Empty, "Wall", None);
    {
        let w = doc.world_mut();
        let t = w.entity_of(terrain).unwrap();
        w.world_mut().entity_mut(t).insert(Transform::IDENTITY);
        w.world_mut().entity_mut(t).insert(Terrain {
            meters_per_sample: MPS,
            tile_resolution: TILE_RES,
            data: flat_terrain(),
            ..Terrain::default()
        });
        let e = w.entity_of(wall).unwrap();
        w.world_mut().entity_mut(e).insert(Transform::IDENTITY);
        w.world_mut().entity_mut(e).insert(Destructible {
            chunk_count: TOWER_CHUNKS,
            runtime_destruct: true,
            ..Destructible::default()
        });
        w.world_mut().entity_mut(e).insert(MeshRef {
            asset: Some(Uuid::from_u128(MESH_GUID)),
            ..Default::default()
        });
        w.mark_dirty();
        w.propagate();
    }
    let mut session = inf_editor_core::simulate::SimSession::enter(
        &mut doc,
        vec![(wall, wall_class())],
        DVec2::ZERO,
        60.0,
    );
    let asset = tower();
    session.set_fractures(resolve_fracture_states(doc.world(), |mesh| {
        (mesh == Uuid::from_u128(MESH_GUID)).then(|| asset.clone())
    }));
    for _ in 0..STEPS {
        session.tick(
            &mut doc,
            1.0 / 60.0,
            inf_editor_core::simulate::SimInput::default(),
        );
    }
    let preview_bits = session.fractures()[&wall].bits();
    let preview_absorbed = match session.actor_var(wall, "absorbed") {
        Some(Value::Float(f)) => *f,
        other => panic!("absorbed is {other:?}"),
    };

    assert_eq!(
        preview_bits, shipped_bits,
        "the editor preview and the shipped player broke the wall differently"
    );
    assert_eq!(preview_absorbed, shipped_absorbed);

    // **ANTI-VACUITY.** The trace has to have done something, or two identical
    // empty maps would agree — the exact failure P21.4 found in the phase-21
    // gate, and the reason this assertion is here rather than assumed.
    let state = &shipped.fractures()[&Uuid::from_u128(WALL_GUID)];
    assert!(
        state.chunks().iter().any(|c| c.detached),
        "nothing broke, so the comparison above compared two intact walls"
    );
    assert!(
        state
            .chunks()
            .iter()
            .enumerate()
            .any(|(i, c)| (c.translation - DVec3::new(0.0, i as f64 + 0.5, 0.0)).length() > 1.0),
        "nothing moved, so the poses above were all the rest pose"
    );
}

// ── the render swap ─────────────────────────────────────────────────────────

/// **ATOMICITY, on the RENDER side: never both, never neither.**
///
/// The physics half of this is `inf-physics`' `the_intact_to_chunks_swap_is_atomic`
/// (the actor's collider goes as its chunk bodies arrive). This is the other
/// half, through the shipped host's real projector: an intact actor is drawn as
/// its own mesh and contributes no chunks; a broken one contributes chunks and no
/// mesh; and at no step is it drawn twice or not at all.
#[test]
fn the_render_swap_draws_a_mesh_or_its_chunks_and_never_both() {
    use inf_player::render::project_scene_with_skinned;

    let mut sim = sim(true, true);
    let vmeshes = inf_player::vmesh::VmeshRegistry::new();
    let skinned = inf_player::skinned::SkinnedRegistry::new();
    let voxels = inf_voxel::VoxelVolumes::default();
    let mut scene = inf_render::RenderScene::default();

    // A tally of how the wall was drawn on each step: (mesh instances that are
    // this actor's, chunks that are this actor's).
    let mut seen: Vec<(usize, usize)> = Vec::new();
    for step in 0..40u32 {
        project_scene_with_skinned(&mut scene, &sim, 0.0, &vmeshes, &skinned, &voxels);
        let chunks = scene
            .fracture_chunks
            .iter()
            .filter(|c| c.entity == inf_render::terrain_id_from_guid(WALL_GUID))
            .count();
        // The wall has no vgeom asset resolvable here, so an intact one falls back
        // to a primitive `MeshInstance` — which is exactly what must stop.
        let meshes = scene.instances.len();
        seen.push((meshes, chunks));
        assert!(
            (meshes > 0) != (chunks > 0),
            "step {step}: the wall drew {meshes} mesh instance(s) AND {chunks} \
             chunk(s) — the swap must be exactly one or the other"
        );
        if step == 0 {
            assert!(chunks == 0, "the wall broke before its first tick");
        }
        sim.step_once(RuntimeInput::default());
    }
    // **ANTI-VACUITY.** Both states really occurred, so the assertion above was
    // testing a swap rather than a constant.
    assert!(
        seen.iter().any(|(m, c)| *m > 0 && *c == 0),
        "the wall was never drawn as a mesh"
    );
    assert!(
        seen.iter().any(|(m, c)| *m == 0 && *c > 0),
        "the wall was never drawn as chunks"
    );

    // …and the projection is well-formed geometry the cache planner will accept —
    // otherwise "it drew chunks" would mean "it drew nothing, silently".
    assert!(
        scene
            .fracture_chunks
            .iter()
            .all(inf_render::passes::fracture::well_formed),
        "a projected chunk is malformed and would be dropped by the cache planner"
    );
}

/// **Order invariance in the shipped host.** Spawning the terrain before or after
/// the wall must not change what breaks or where it lands — the second of the two
/// determinism gates the P22.3 audit asked for. Passes today; pinned so it keeps
/// doing so.
#[test]
fn the_shipped_collapse_is_invariant_under_spawn_order() {
    let run = |terrain_first: bool| {
        let mut world = EcsWorld::new();
        let add_terrain = |w: &mut EcsWorld| {
            let t = w.spawn_with_guid(Uuid::from_u128(TERRAIN_GUID), "Terrain", None);
            w.world_mut().entity_mut(t).insert(Transform::IDENTITY);
            w.world_mut().entity_mut(t).insert(Terrain {
                meters_per_sample: MPS,
                tile_resolution: TILE_RES,
                data: flat_terrain(),
                ..Terrain::default()
            });
        };
        let add_wall = |w: &mut EcsWorld| {
            let e = w.spawn_with_guid(Uuid::from_u128(WALL_GUID), "Wall", None);
            w.world_mut().entity_mut(e).insert(Transform::IDENTITY);
            w.world_mut().entity_mut(e).insert(Destructible {
                chunk_count: TOWER_CHUNKS,
                runtime_destruct: true,
                ..Destructible::default()
            });
            w.world_mut().entity_mut(e).insert(MeshRef {
                asset: Some(Uuid::from_u128(MESH_GUID)),
                ..Default::default()
            });
        };
        if terrain_first {
            add_terrain(&mut world);
            add_wall(&mut world);
        } else {
            add_wall(&mut world);
            add_terrain(&mut world);
        }
        world.mark_dirty();
        world.propagate();

        let mut sim = RuntimeSim::new(
            world,
            vec![(Uuid::from_u128(WALL_GUID), wall_class())],
            DVec2::ZERO,
            60.0,
        );
        let asset = tower();
        sim.set_fractures(resolve_fracture_states(sim.world(), |mesh| {
            (mesh == Uuid::from_u128(MESH_GUID)).then(|| asset.clone())
        }));
        for _ in 0..45 {
            sim.step_once(RuntimeInput::default());
        }
        sim.fractures()[&Uuid::from_u128(WALL_GUID)].bits()
    };
    let a = run(true);
    let b = run(false);
    assert_eq!(
        a, b,
        "reversing spawn order changed the collapse — something is reading ECS \
         insertion order"
    );
    // Anti-vacuity: the trace really broke the wall, so the comparison was not of
    // two untouched towers.
    let untouched = FractureState::new(
        tower(),
        Destructible {
            chunk_count: TOWER_CHUNKS,
            ..Destructible::default()
        },
        glam::DAffine3::IDENTITY,
        false,
    );
    assert_ne!(a, untouched.bits());
}

/// **The host-level twin of the placement fix.** `inf-physics`'
/// `a_destructible_moved_before_breaking_shatters_where_it_is` drives the bridge
/// directly; this drives the **shipped host's fixed step**, which is what would
/// notice the host forgetting to call `follow_fractures` at all.
///
/// (The static half of that is `projector_mirror`'s
/// `both_fixed_steps_run_the_fracture_slots`, which reads both files. This is the
/// behavioural half: it moves a wall through the ECS and checks the rubble
/// followed.)
#[test]
fn the_shipped_host_shatters_a_moved_wall_where_it_stands() {
    let mut sim = sim(true, true);
    // Move it BEFORE the first step. This Blueprint blows the wall up on every
    // tick, so after one step it is already broken — and `follow` deliberately
    // stops following a broken actor (its chunks are solver-owned). Moving first
    // is the case the fix is about: an intact destructible relocated after load.
    {
        let e = sim.world().entity_of(Uuid::from_u128(WALL_GUID)).unwrap();
        let mut t = sim
            .world_mut()
            .world_mut()
            .get_mut::<Transform>(e)
            .expect("the wall has a transform");
        t.translation = inf_ecs::Vec3d::new(40.0, 0.0, 0.0);
    }
    sim.world_mut().mark_dirty();
    // `follow_fractures` reads `GlobalTransform`, which the fixed step does not
    // settle until after the sync — so propagate here, exactly as an editor drag
    // or a `world://delta` would have.
    sim.world_mut().propagate();
    sim.step_once(RuntimeInput::default());

    // The Blueprint blows it up every tick, so by now it has broken — AT x = 40.
    let state = &sim.fractures()[&Uuid::from_u128(WALL_GUID)];
    assert!(!state.is_intact(), "the wall never broke");
    let near = state
        .chunks()
        .iter()
        .filter(|c| (c.translation.x - 40.0).abs() < 2.0)
        .count();
    assert_eq!(
        near,
        state.chunks().len(),
        "{} of {} chunks are not near x = 40 — the shipped host is not following \
         its fractures, so the wall shattered where it was LOADED",
        state.chunks().len() - near,
        state.chunks().len()
    );
}

// ── P22.4: the per-tier debris budget ───────────────────────────────────────

/// **The High row IS the physics default.**
///
/// `inf_render::debris_budget_for` maps a GPU capability tier onto a debris
/// budget, and `inf-render` may not depend on `inf-physics` to say the numbers —
/// so the equality that makes the whole arrangement safe is asserted here, in the
/// crate that can see both, on the `destructible_defaults_match_the_mesh_crate`
/// precedent.
///
/// Why it matters more than a tidy constant: the budget is read by
/// `step_fractures`, which is **fixed-step simulation**. Every determinism gate in
/// this repository — the replay harness, `scene_trace`, the `--pie` protocol loop,
/// every headless `pack_sim` — builds no render host and therefore runs
/// `DebrisBudget::default()`. If High were anything else, a windowed player on a
/// capable machine would be simulating a *different* level from the one every gate
/// compares, and nothing would report it.
#[test]
fn the_high_tier_budget_is_the_physics_default() {
    let high = inf_render::debris_budget_for(inf_render::RenderTier::High);
    let default = inf_physics::d3::DebrisBudget::default();
    assert_eq!(
        high.max_live, default.max_live,
        "the High tier's chunk cap has drifted from `DEFAULT_DEBRIS_MAX_LIVE`"
    );
    assert_eq!(
        high.lifetime_s, default.lifetime_s,
        "the High tier's debris lifetime has drifted from \
         `DEFAULT_DEBRIS_LIFETIME_S`"
    );
    // …and the weaker tiers really are weaker, converted through the same seam
    // the window uses, so "clamps down, never up" is checked on the real types.
    for tier in [inf_render::RenderTier::Medium, inf_render::RenderTier::Low] {
        let spec = inf_render::debris_budget_for(tier);
        let budget = inf_physics::d3::DebrisBudget {
            max_live: spec.max_live,
            lifetime_s: spec.lifetime_s,
        };
        assert!(
            budget.max_live <= default.max_live,
            "{tier:?} raised the cap"
        );
        assert!(
            budget.lifetime_s <= default.lifetime_s,
            "{tier:?} lengthened the lifetime"
        );
        assert!(budget.max_live > 0 && budget.lifetime_s > 0.0);
    }
}

/// **A tier-clamped budget really does reclaim earlier** — the seam does
/// something, rather than being a struct nobody reads.
///
/// The wall's Blueprint blows it apart on every tick, so within a few steps every
/// chunk is live debris. Under a budget of one, the level keeps exactly one — and
/// the reclaimed chunks are `gone`, which is what removes them from the solver and
/// from both hosts' render projections on the same generation.
#[test]
fn a_tightened_debris_budget_reclaims_the_oldest_chunks() {
    let mut capped = sim(true, true);
    capped.set_debris_budget(inf_physics::d3::DebrisBudget {
        max_live: 1,
        lifetime_s: f64::INFINITY,
    });
    for _ in 0..8 {
        capped.step_once(RuntimeInput::default());
    }
    let state = &capped.fractures()[&Uuid::from_u128(WALL_GUID)];
    let live = state
        .chunks()
        .iter()
        .filter(|c| c.detached && !c.gone)
        .count();
    let gone = state.chunks().iter().filter(|c| c.gone).count();
    assert_eq!(live, 1, "the cap of 1 left {live} chunks live");
    assert!(gone > 0, "nothing was reclaimed, so the cap did nothing");
    assert_eq!(
        capped.fracture_audit().live_debris,
        1,
        "the audit disagrees with the state it is supposed to be counting"
    );

    // The control: the SAME fixture under the engine default keeps them all, so
    // the assertion above is about the budget rather than about the wall.
    let mut loose = sim(true, true);
    for _ in 0..8 {
        loose.step_once(RuntimeInput::default());
    }
    let loose_live = loose.fractures()[&Uuid::from_u128(WALL_GUID)]
        .chunks()
        .iter()
        .filter(|c| c.detached && !c.gone)
        .count();
    assert!(
        loose_live > 1,
        "the unclamped fixture only ever has {loose_live} live chunk(s), so the \
         cap of 1 proved nothing"
    );
}
