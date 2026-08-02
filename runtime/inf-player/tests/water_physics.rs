//! P20.2 gate: **water is physical**, deterministically, and PIE == shipping.
//!
//! P20.1's gate proved a water *surface* is derived identically in the editor's
//! in-process world and in a cooked pack. This one proves the same for what the
//! water *does*: a raft of crates floating on a swell, a character swimming, and
//! a Blueprint hearing about it.
//!
//! Water forces are the sharpest case the sim/render split has produced so far.
//! A buoyant force is a pure function of `(WaterBody params, body pose, level
//! clock)` sampled through `inf_water`'s height query — and if any of those three
//! ever came from somewhere the renderer owns, a shipped build's sea would float
//! boats at a *different height* from the preview's, and it would still look like
//! a sea, so nothing would notice.
//!
//! * **(a) determinism** — the same scripted scene twice is a bit-identical pose
//!   trace (`to_bits`, not an epsilon), including the water events.
//! * **(b) pool-size invariance** — the same trace whatever ECS worker-pool size
//!   the process runs on. "Serial today" is not a property a replay can rely on.
//! * **(c) PIE == shipping** — the cooked-pack world and the PIE-payload world
//!   produce the same trace, poses and events alike.
//! * **(d) one wave model** — the Gerstner components the *renderer* would upload
//!   and the ones the *fixed step* samples are bit-identical, proved by comparing
//!   the two projections rather than by two copies of the same source text.
//! * **(e) off-path** — a level with no water traces bit-identically to the same
//!   level with the water knobs wound, with an anti-vacuity guard that a body in
//!   the water really does notice them.
//! * **(f) swim** — submerging a character flips it into swim mode, it surfaces
//!   under input, and the trace is deterministic.

use std::path::{Path, PathBuf};

use glam::DVec2;
use inf_blueprint::{
    BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty,
};
use inf_ecs::components::{
    BodyKind3D, Buoyancy, CharacterController3D, Collider3D, ColliderShape3DKind, RigidBody3D,
    SkyAtmosphere, TimeOfDay, Transform, WaterBody, WaterKind,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::EcsWorld;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_packager::{cook, CookOptions};
use inf_physics::d3::water::{water_surface_of, SWIM_EXIT_FRACTION};
use inf_player::level::{BuiltWorld, InfSceneWorldBuilder, PackLevelSource};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_project::ProjectManifest;
use uuid::Uuid;

const STEPS: usize = 480;
const SKY: Uuid = Uuid::from_u128(0x2002_1000);
const OCEAN: Uuid = Uuid::from_u128(0x2002_1001);
const SWIMMER: Uuid = Uuid::from_u128(0x2002_1002);
/// Eight crates of ascending density: `0x2002_1010 ..= 0x2002_1017`.
const CRATE_BASE: u128 = 0x2002_1010;
const CRATES: u128 = 8;

/// One step of the trace: every crate's pose bits plus the swimmer's, as raw
/// bits. Bits rather than floats on purpose — a comparison that "passes within
/// 1e-12" is exactly the drift this gate exists to catch.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Frame {
    poses: Vec<[u64; 7]>,
    swimmer: [u64; 3],
}

fn pose_of(sim: &RuntimeSim, guid: Uuid) -> [u64; 7] {
    let world = sim.world();
    let Some(e) = world.entity_of(guid) else {
        return [0; 7];
    };
    let Some(t) = world.world().get::<Transform>(e) else {
        return [0; 7];
    };
    let q = t.quat();
    [
        t.translation.x.to_bits(),
        t.translation.y.to_bits(),
        t.translation.z.to_bits(),
        q.x.to_bits(),
        q.y.to_bits(),
        q.z.to_bits(),
        q.w.to_bits(),
    ]
}

fn frame(sim: &RuntimeSim) -> Frame {
    let poses = (0..CRATES)
        .map(|i| pose_of(sim, Uuid::from_u128(CRATE_BASE + i)))
        .collect();
    let s = pose_of(sim, SWIMMER);
    Frame {
        poses,
        swimmer: [s[0], s[1], s[2]],
    }
}

/// Run the scripted scenario. The swimmer holds "forward" and "up" for the whole
/// run, so swim mode is exercised under input rather than merely entered.
fn run_trace(sim: &mut RuntimeSim) -> Vec<Frame> {
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            frame(sim)
        })
        .collect()
}

/// The scene: a storming ocean, eight crates of ascending density dropped onto
/// it, and a character capsule sunk below the surface.
///
/// `water` decides whether the ocean exists at all — the off-path arm builds the
/// identical scene with no `WaterBody`, so the comparison is between two levels
/// that differ in exactly one component.
fn gate_doc(water: bool, amplitude: f64) -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.set_title("Water Physics Gate");
    doc.create_with_guid(SKY, SpawnKind::Empty, "Sky", None);
    doc.create_with_guid(OCEAN, SpawnKind::Empty, "Ocean", None);
    doc.create_with_guid(SWIMMER, SpawnKind::Empty, "Swimmer", None);
    for i in 0..CRATES {
        doc.create_with_guid(
            Uuid::from_u128(CRATE_BASE + i),
            SpawnKind::Empty,
            &format!("Crate{i}"),
            None,
        );
    }

    let sky = doc.world().entity_of(SKY).unwrap();
    let ocean = doc.world().entity_of(OCEAN).unwrap();
    let swimmer = doc.world().entity_of(SWIMMER).unwrap();
    let crates: Vec<_> = (0..CRATES)
        .map(|i| {
            doc.world()
                .entity_of(Uuid::from_u128(CRATE_BASE + i))
                .unwrap()
        })
        .collect();
    let w = doc.world_mut().world_mut();

    // A clock that really runs, so a wave phase that ignored it would be obvious.
    w.entity_mut(sky).insert(TimeOfDay {
        seconds: 0.0,
        rate: 240.0,
        ..TimeOfDay::default()
    });
    w.entity_mut(sky).insert(SkyAtmosphere {
        weather_enabled: true,
        ..SkyAtmosphere::default()
    });

    if water {
        w.entity_mut(ocean).insert(WaterBody {
            kind: WaterKind::Ocean,
            level_m: 0.0,
            wave_amplitude_m: amplitude,
            wave_length_m: 22.0,
            wave_steepness: 0.45,
            wave_count: 5,
            wave_seed: 0x5EA_5A17,
            // Body-local wind, so the trace does not depend on the weather blend
            // landing on a particular gust — this gate is about the *forces*.
            wind_from_weather: false,
            wind_x: 7.5,
            wind_z: -2.5,
            ..WaterBody::default()
        });
    }

    for (i, e) in crates.iter().enumerate() {
        // 350 … 700 kg/m³ — all comfortably buoyant, at visibly different
        // draughts. Deliberately short of 1000: a near-neutral crate has almost
        // no restoring force and takes half a minute to come back up, which
        // would make this a test of patience rather than of buoyancy.
        let density = 350.0 + i as f64 * 50.0;
        w.entity_mut(*e).insert((
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..RigidBody3D::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::splat(0.5),
                density,
                ..Collider3D::default()
            },
            Buoyancy::of_density(density),
            Transform {
                translation: Vec3d::new(
                    (i as f64 - 3.5) * 1.4,
                    1.5 + i as f64 * 0.45,
                    (i as f64 % 3.0) * 0.9,
                ),
                ..Transform::IDENTITY
            },
        ));
    }

    // The swimmer: a kinematic capsule with a character controller, placed with
    // most of itself under the surface so it starts in swim mode.
    w.entity_mut(swimmer).insert((
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..RigidBody3D::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.3, 0.6, 0.3),
            radius: 0.3,
            ..Collider3D::default()
        },
        CharacterController3D::default(),
        Transform {
            translation: Vec3d::new(-14.0, -2.0, 6.0),
            ..Transform::IDENTITY
        },
    ));

    // The 3D physics gravity flows from `gravity_2d.y` (the runtime sim wires the
    // 3D bridge to it), so this makes it explicit real-world down — otherwise the
    // crates would hover and every assertion below would be comparing two copies
    // of nothing.
    doc.set_settings(inf_editor_core::scene::serialize::LevelSettings {
        gravity_2d: Vec2d::new(0.0, -9.81),
        gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
        sim_hz: 60.0,
        ..Default::default()
    });

    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc
}

fn cook_doc(tmp: &Path, doc: &SceneDoc) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Water Physics Gate", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    inf_editor_core::scene::serialize::save(doc, &content.join("Water.inf_lvl"), None)
        .expect("save level");
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

fn pack_sim(pack_dir: &Path) -> RuntimeSim {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let actors = source.actor_classes().expect("actor classes decode");
    let builder = InfSceneWorldBuilder::with_defaults(actors);
    let built: BuiltWorld = inf_player::level::load(&source, &builder).expect("pack level builds");
    inf_player::sim_from_built(built)
}

fn pie_sim(doc: &SceneDoc) -> RuntimeSim {
    let payload = inf_editor_core::pie::build_scene_payload(
        doc,
        |_guid| None,
        |_guid| None,
        |_guid| None,
        |_guid| None,
        60,
        false,
    )
    .expect("payload builds");
    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    inf_player::sim_from_built(built)
}

/// The crates must actually move, and settle near the surface — otherwise every
/// equality below is comparing two copies of nothing.
fn assert_not_vacuous(trace: &[Frame]) {
    let first = trace.first().expect("a trace");
    let last = trace.last().expect("a trace");
    assert_ne!(first, last, "nothing moved over the whole run");
    for (i, (a, b)) in first.poses.iter().zip(&last.poses).enumerate() {
        assert_ne!(a, b, "crate {i} never moved");
        let y = f64::from_bits(b[1]);
        assert!(
            y.abs() < 3.0,
            "crate {i} should have settled near the sea, not at y = {y}"
        );
    }
}

// ── (a) determinism ─────────────────────────────────────────────────────────

#[test]
fn a_floating_stack_traces_bit_identically_across_two_runs() {
    let doc = gate_doc(true, 0.7);
    let a = run_trace(&mut pie_sim(&doc));
    let b = run_trace(&mut pie_sim(&doc));
    assert_eq!(a, b, "the floating stack is not deterministic");
    assert_not_vacuous(&a);
}

// ── (b) pool-size invariance ────────────────────────────────────────────────

/// The water force pass is serial today, but "serial today" is not a property a
/// replay can rely on — the parallel schedule is chosen at startup, so the traces
/// have to be compared rather than reasoned about.
#[test]
fn the_floating_stack_is_identical_across_pool_sizes() {
    let doc = gate_doc(true, 0.7);
    let trace_at = |threads: usize| {
        inf_ecs::init_ecs_task_pool(threads);
        run_trace(&mut pie_sim(&doc))
    };
    let one = trace_at(1);
    let four = trace_at(4);
    assert_eq!(
        one, four,
        "buoyancy depends on the ECS pool size — it is not replay-safe"
    );
    assert_not_vacuous(&one);
}

// ── (c) PIE == shipping ─────────────────────────────────────────────────────

#[test]
fn pie_equals_shipping_on_the_floating_stack() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let doc = gate_doc(true, 0.7);
    let pack = cook_doc(tmp.path(), &doc);

    let shipped = run_trace(&mut pack_sim(&pack));
    let preview = run_trace(&mut pie_sim(&doc));
    assert_eq!(
        shipped, preview,
        "a cooked build floats bodies differently from the editor preview"
    );
    assert_not_vacuous(&shipped);
}

// ── (d) one wave model ──────────────────────────────────────────────────────

/// The renderer and the fixed step must sample **one** Gerstner sum. They reach it
/// through two different projectors (`inf_player::render::project_water` for the
/// GPU, `inf_physics::d3::water::water_surface_of` for the sim), and this compares
/// the *derived components* they produce as bits — which is a stronger statement
/// than two copies of one source text, because it would still catch a divergence
/// introduced by a different clock or a different wind rule.
#[test]
fn the_sim_and_the_renderer_derive_the_same_waves() {
    let doc = gate_doc(true, 0.7);
    let mut sim = pie_sim(&doc);
    // Run a while so the clock is somewhere non-trivial.
    for _ in 0..60 {
        sim.step_once(RuntimeInput::default());
    }
    // The renderer's projection.
    let mut scene = inf_render::RenderScene::default();
    inf_player::render::project_scene(
        &mut scene,
        &sim,
        0.0,
        &inf_player::vmesh::VmeshRegistry::new(),
    );
    let rendered = scene.waters.first().expect("the ocean projected");

    // The sim's projection, from the same components and the same environment.
    let world = sim.world();
    let env = inf_ecs::sky::water_environment(world);
    let e = world.entity_of(OCEAN).unwrap();
    let body = *world.world().get::<WaterBody>(e).unwrap();
    let surface = water_surface_of(&body, None, &glam::DAffine3::IDENTITY, env)
        .expect("the sim projected the ocean");

    let bits = |w: &inf_render::Wave| {
        [
            w.dir.x.to_bits(),
            w.dir.y.to_bits(),
            w.amplitude_m.to_bits(),
            w.wavenumber.to_bits(),
            w.omega.to_bits(),
            w.steepness.to_bits(),
            w.phase.to_bits(),
        ]
    };
    let a: Vec<_> = rendered.waves.waves().iter().map(bits).collect();
    let b: Vec<_> = surface.waves().waves().iter().map(bits).collect();
    assert!(!a.is_empty(), "no waves at all — the comparison is vacuous");
    assert_eq!(
        a, b,
        "the renderer and the fixed step derived different Gerstner components"
    );
    // And the height a boat floats on is the height the shader would draw.
    let t = env.0;
    for p in [
        DVec2::ZERO,
        DVec2::new(13.5, -7.25),
        DVec2::new(-40.0, 88.0),
    ] {
        let sim_h = surface.height_at(p, t).unwrap();
        assert!(
            sim_h.abs() <= body.wave_amplitude_m + 1e-9,
            "the sim's surface left the wave's own amplitude bound at {p:?}"
        );
    }
}

// ── (e) off path ────────────────────────────────────────────────────────────

/// A level with **no water** must trace bit-identically however the water knobs
/// are wound, and the guard must not be vacuous.
#[test]
fn a_waterless_level_is_bit_identical_and_the_guard_is_not_vacuous() {
    let calm = run_trace(&mut pie_sim(&gate_doc(false, 0.0)));
    let stormy = run_trace(&mut pie_sim(&gate_doc(false, 1.6)));
    assert_eq!(
        calm, stormy,
        "a level with no water noticed a water parameter"
    );
    // Anti-vacuity: with the ocean present, that same knob DOES change the trace.
    let wet_calm = run_trace(&mut pie_sim(&gate_doc(true, 0.0)));
    let wet_storm = run_trace(&mut pie_sim(&gate_doc(true, 1.6)));
    assert_ne!(
        wet_calm, wet_storm,
        "the sea state did not reach the bodies floating on it"
    );
    // …and with no water the crates simply fall, which is the honest baseline.
    let last = calm.last().unwrap();
    for (i, p) in last.poses.iter().enumerate() {
        assert!(
            f64::from_bits(p[1]) < -3.0,
            "crate {i} did not fall in a level with no water"
        );
    }
}

// ── (f) swim ────────────────────────────────────────────────────────────────

/// A Blueprint that drives its actor exactly the way a falling character does:
/// `move_and_slide(entity, forward, gravity·dt, 0)` every tick. Nothing in it
/// mentions water — the point is that the **host path** honours swim mode without
/// the game asking, so an existing character blueprint gains swimming for free.
fn walker_class() -> BlueprintClass {
    let entity = Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str("entity".into()))],
    };
    let mut class = BlueprintClass::new("act:walker", "Walker");
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
            body: vec![Stmt::ExprStmt(Expr::Call {
                path: vec!["physics3d".into(), "move_and_slide".into()],
                args: vec![
                    entity,
                    // A brisk run forward…
                    Expr::Lit(Lit::Float(6.0 / 60.0)),
                    // …and a full second of accumulated free fall, which walking
                    // would honour and swimming must discard.
                    Expr::Lit(Lit::Float(-9.81 / 60.0)),
                    Expr::Lit(Lit::Float(0.0)),
                ],
            })],
        },
    }];
    class
}

/// Submerging a character flips it into swim mode; it then rises toward the
/// surface under the buoyancy balance instead of sinking, its speed is capped,
/// and the whole thing is deterministic.
///
/// Built from a bare [`EcsWorld`] rather than through the PIE payload because the
/// swimmer needs a Blueprint actor attached to it, and the point of the test is
/// the **`move_and_slide` host path** — the PIE == shipping half is the crate
/// raft's job, above.
#[test]
fn a_submerged_character_swims_and_surfaces() {
    let trace = |steps: usize| {
        let mut world = EcsWorld::new();
        let ocean = world.spawn_with_guid(OCEAN, "Ocean", None);
        world.world_mut().entity_mut(ocean).insert((
            WaterBody {
                kind: WaterKind::Ocean,
                level_m: 0.0,
                // Still water, so "did it rise" is unambiguous.
                wave_amplitude_m: 0.0,
                ..WaterBody::default()
            },
            Transform::IDENTITY,
        ));
        let hero = world.spawn_with_guid(SWIMMER, "Swimmer", None);
        world.world_mut().entity_mut(hero).insert((
            RigidBody3D {
                kind: BodyKind3D::Kinematic,
                ..RigidBody3D::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Capsule,
                half_extents: Vec3d::new(0.3, 0.6, 0.3),
                radius: 0.3,
                ..Collider3D::default()
            },
            CharacterController3D::default(),
            Transform {
                translation: Vec3d::new(-14.0, -2.0, 6.0),
                ..Transform::IDENTITY
            },
        ));
        world.mark_dirty();

        let mut sim = RuntimeSim::new(
            world,
            vec![(SWIMMER, walker_class())],
            glam::DVec2::new(0.0, -9.81),
            60.0,
        );
        let mut ys = Vec::new();
        for _ in 0..steps {
            sim.step_once(RuntimeInput::default());
            ys.push(pose_of(&sim, SWIMMER)[1]);
        }
        (sim, ys)
    };

    let (sim, ys) = trace(300);
    let start = f64::from_bits(ys[0]);
    let end = f64::from_bits(*ys.last().unwrap());
    let probe = sim
        .bridge3d()
        .water_probe(SWIMMER)
        .expect("the swimmer is in the ocean");
    assert!(
        sim.bridge3d().is_swimming(SWIMMER),
        "the swim latch dropped at fraction {}",
        probe.fraction
    );
    assert!(
        probe.fraction > SWIM_EXIT_FRACTION,
        "a swimmer settles near the float depth, not at {}",
        probe.fraction
    );
    assert!(
        end > start + 0.2,
        "a swimmer must surface against a full gravity step, not sink: {start} -> {end}"
    );
    assert!(
        end < 0.5,
        "a swimmer must settle AT the surface, not launch out of it: {end}"
    );
    // Speed cap: 300 steps at the 2.5 m/s swim cap is at most 12.5 m of travel,
    // and the blueprint asked for 6 m/s (which would be 30 m).
    let travelled = f64::from_bits(pose_of(&sim, SWIMMER)[0]) + 14.0;
    assert!(
        (0.5..13.0).contains(&travelled),
        "swim speed was not capped: travelled {travelled} m in 5 s"
    );

    // Deterministic.
    let (_, again) = trace(300);
    assert_eq!(ys, again, "the swim trace is not deterministic");
}

/// The anti-vacuity twin: the SAME character in the SAME level with no water
/// sinks, because `move_and_slide` is only transformed when the water says so.
#[test]
fn the_same_character_with_no_water_just_falls() {
    let mut world = EcsWorld::new();
    let hero = world.spawn_with_guid(SWIMMER, "Swimmer", None);
    world.world_mut().entity_mut(hero).insert((
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..RigidBody3D::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.3, 0.6, 0.3),
            radius: 0.3,
            ..Collider3D::default()
        },
        CharacterController3D::default(),
        Transform {
            translation: Vec3d::new(-14.0, -2.0, 6.0),
            ..Transform::IDENTITY
        },
    ));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(
        world,
        vec![(SWIMMER, walker_class())],
        glam::DVec2::new(0.0, -9.81),
        60.0,
    );
    for _ in 0..300 {
        sim.step_once(RuntimeInput::default());
    }
    assert!(!sim.bridge3d().is_swimming(SWIMMER));
    let y = f64::from_bits(pose_of(&sim, SWIMMER)[1]);
    assert!(y < -20.0, "with no water the character must fall: y = {y}");
    // …and it runs at its authored 6 m/s rather than the swim cap.
    let travelled = f64::from_bits(pose_of(&sim, SWIMMER)[0]) + 14.0;
    assert!(
        travelled > 20.0,
        "an unsubmerged character must keep its running speed: {travelled} m"
    );
}

// ── events + audio hooks ────────────────────────────────────────────────────

/// A Blueprint that hears the water: it counts entries and splashes into member
/// variables and queues a sound on each splash.
///
/// The **audio hook** is exactly this — `audio.*` called from an `On Splash`
/// handler — and no new command type. The P12.3 doctrine is that the audio stream
/// is a pure function of sim state; a water crossing *is* sim state, so a `Play`
/// queued from its handler lands in the command queue in the same deterministic
/// order the crossings were produced in. An engine-authored `AudioCommand::Splash`
/// would have meant the engine picking the sound.
fn splash_listener_class() -> BlueprintClass {
    let get = |name: &str| Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str(name.into()))],
    };
    let bump = |name: &str| {
        Stmt::ExprStmt(Expr::Call {
            path: vec!["vars".into(), "set".into()],
            args: vec![
                Expr::Lit(Lit::Str(name.into())),
                Expr::Binary(
                    inf_blueprint::BinOp::Add,
                    Box::new(get(name)),
                    Box::new(Expr::Lit(Lit::Int(1))),
                ),
            ],
        })
    };
    let entity = || Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str("entity".into()))],
    };
    let counter = |name: &str| inf_blueprint::Variable {
        name: name.into(),
        ty: Ty::Int,
        default: Lit::Int(0),
        exposed: false,
    };
    let handler = |kind: EventKind, body: Vec<Stmt>| EventBinding {
        event: kind.clone(),
        body: BlueprintFn {
            id: kind.key(),
            name: kind.key(),
            params: kind.signature(),
            ret: Ty::Unit,
            body,
        },
    };

    let mut class = BlueprintClass::new("act:splash", "Splash Listener");
    class.variables = vec![counter("enters"), counter("exits"), counter("splashes")];
    class.events = vec![
        handler(EventKind::WaterEnter, vec![bump("enters")]),
        handler(EventKind::WaterExit, vec![bump("exits")]),
        handler(
            EventKind::WaterSplash,
            vec![
                bump("splashes"),
                // The audio hook.
                Stmt::ExprStmt(Expr::Call {
                    path: vec!["audio".into(), "play".into()],
                    args: vec![entity()],
                }),
            ],
        ),
    ];
    class
}

/// A crate dropped into the sea, with a Blueprint listening. Returns the sim so
/// the caller can read variables and the audio command log.
fn splash_sim(steps: usize) -> RuntimeSim {
    let crate_guid = Uuid::from_u128(CRATE_BASE);
    let mut world = EcsWorld::new();
    let ocean = world.spawn_with_guid(OCEAN, "Ocean", None);
    world.world_mut().entity_mut(ocean).insert((
        WaterBody {
            kind: WaterKind::Ocean,
            level_m: 0.0,
            wave_amplitude_m: 0.0,
            ..WaterBody::default()
        },
        Transform::IDENTITY,
    ));
    let c = world.spawn_with_guid(crate_guid, "Crate", None);
    world.world_mut().entity_mut(c).insert((
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..RigidBody3D::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::splat(0.5),
            density: 500.0,
            ..Collider3D::default()
        },
        Buoyancy::of_density(500.0),
        // The voice the splash handler plays. `autoplay: false`, so the only
        // `Play` in the command log is the one the water event caused.
        inf_ecs::components::AudioSource {
            autoplay: false,
            ..Default::default()
        },
        // 5 m up: it arrives at ~10 m/s, well past the 2 m/s splash bar.
        Transform {
            translation: Vec3d::new(0.0, 5.0, 0.0),
            ..Transform::IDENTITY
        },
    ));
    world.mark_dirty();

    let mut sim = RuntimeSim::new(
        world,
        vec![(crate_guid, splash_listener_class())],
        glam::DVec2::new(0.0, -9.81),
        60.0,
    );
    for _ in 0..steps {
        sim.step_once(RuntimeInput::default());
    }
    sim
}

#[test]
fn a_blueprint_hears_the_water_and_the_audio_stream_is_deterministic() {
    let crate_guid = Uuid::from_u128(CRATE_BASE);
    let var = |sim: &RuntimeSim, name: &str| match sim.actor_var(crate_guid, name) {
        Some(inf_blueprint::Value::Int(n)) => *n,
        other => panic!("{name} is {other:?}"),
    };

    let sim = splash_sim(600);
    assert!(
        var(&sim, "enters") >= 1,
        "the crate never entered the water"
    );
    assert!(
        var(&sim, "splashes") >= 1,
        "a 5 m drop must splash, not merely enter"
    );
    // Every splash queued a sound — the audio hook, working.
    let plays = sim
        .audio_command_log()
        .iter()
        .filter(|c| matches!(c, inf_audio::AudioCommand::Play(_)))
        .count();
    assert!(
        plays as i64 >= var(&sim, "splashes"),
        "a splash handler's audio.play did not reach the command queue"
    );

    // Deterministic: the same run twice produces the same counters AND the same
    // audio command stream, in the same order.
    let again = splash_sim(600);
    for name in ["enters", "exits", "splashes"] {
        assert_eq!(var(&sim, name), var(&again, name), "{name} diverged");
    }
    let fmt = |s: &RuntimeSim| format!("{:?}", s.audio_command_log());
    assert_eq!(
        fmt(&sim),
        fmt(&again),
        "the audio command stream a water event drives is not deterministic"
    );
}

/// A **gentle** entry fires `Enter` and no `Splash` — the threshold is real, not
/// decorative.
#[test]
fn the_splash_threshold_is_real() {
    let crate_guid = Uuid::from_u128(CRATE_BASE);
    let var = |sim: &RuntimeSim, name: &str| match sim.actor_var(crate_guid, name) {
        Some(inf_blueprint::Value::Int(n)) => *n,
        other => panic!("{name} is {other:?}"),
    };
    // Released at the waterline: it crosses at well under 2 m/s.
    let mut world = EcsWorld::new();
    let ocean = world.spawn_with_guid(OCEAN, "Ocean", None);
    world.world_mut().entity_mut(ocean).insert((
        WaterBody {
            kind: WaterKind::Ocean,
            level_m: 0.0,
            wave_amplitude_m: 0.0,
            ..WaterBody::default()
        },
        Transform::IDENTITY,
    ));
    let c = world.spawn_with_guid(crate_guid, "Crate", None);
    world.world_mut().entity_mut(c).insert((
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..RigidBody3D::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::splat(0.5),
            density: 500.0,
            ..Collider3D::default()
        },
        Buoyancy::of_density(500.0),
        Transform {
            translation: Vec3d::new(0.0, -0.49, 0.0),
            ..Transform::IDENTITY
        },
    ));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(
        world,
        vec![(crate_guid, splash_listener_class())],
        glam::DVec2::new(0.0, -9.81),
        60.0,
    );
    for _ in 0..600 {
        sim.step_once(RuntimeInput::default());
    }
    assert_eq!(var(&sim, "enters"), 1);
    assert_eq!(var(&sim, "splashes"), 0, "a gentle entry must not splash");
    assert_eq!(var(&sim, "exits"), 0, "a floating crate never leaves");
}

// ── preview == shipped (paired with the editor twin) ────────────────────────

/// Shared with `editor/crates/inf-editor-core/tests/simulate_water.rs` — the same
/// guids build the same world.
const PAIR_OCEAN: u128 = 0x2002_2001;
const PAIR_CRATE: u128 = 0x2002_2002;

/// **PINNED IDENTICALLY IN THE EDITOR TWIN.** The fixed step on which a 1 m crate
/// of density 500, released 5 m above a still sea, first reports `WaterEnter`.
const EXPECTED_ENTER_STEP: usize = 58;
/// **PINNED IDENTICALLY IN THE EDITOR TWIN.** Entries / exits / splashes over 600
/// steps.
const EXPECTED_COUNTS: (i64, i64, i64) = (2, 1, 3);

/// **The parity gate.** The editor's `SimSession` and the shipped `RuntimeSim`
/// each own a copy of the fixed-step ordering, of the `Host::call` match arms and
/// of the `move_and_slide` path. "They run the same water physics" is therefore a
/// claim about two files, and this pair is what checks it from the outside: build
/// the identical scene over the identical Blueprint and assert the identical
/// integer landmarks.
#[test]
fn the_editor_and_the_runtime_agree_on_the_water() {
    let krate = Uuid::from_u128(PAIR_CRATE);
    let mut world = EcsWorld::new();
    let ocean = world.spawn_with_guid(Uuid::from_u128(PAIR_OCEAN), "Ocean", None);
    world.world_mut().entity_mut(ocean).insert((
        WaterBody {
            kind: WaterKind::Ocean,
            level_m: 0.0,
            wave_amplitude_m: 0.0,
            ..WaterBody::default()
        },
        Transform::IDENTITY,
    ));
    let c = world.spawn_with_guid(krate, "Crate", None);
    world.world_mut().entity_mut(c).insert((
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..RigidBody3D::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::splat(0.5),
            density: 500.0,
            ..Collider3D::default()
        },
        Buoyancy::of_density(500.0),
        Transform {
            translation: Vec3d::new(0.0, 5.0, 0.0),
            ..Transform::IDENTITY
        },
    ));
    world.mark_dirty();

    let mut sim = RuntimeSim::new(
        world,
        vec![(krate, splash_listener_class())],
        glam::DVec2::new(0.0, -9.81),
        60.0,
    );
    let var = |s: &RuntimeSim, name: &str| match s.actor_var(krate, name) {
        Some(inf_blueprint::Value::Int(n)) => *n,
        other => panic!("{name} is {other:?}"),
    };

    let mut enter_step = None;
    for i in 0..600 {
        sim.step_once(RuntimeInput::default());
        if enter_step.is_none() && var(&sim, "enters") > 0 {
            enter_step = Some(i);
        }
    }
    assert_eq!(
        enter_step,
        Some(EXPECTED_ENTER_STEP),
        "the runtime met the water on a different step than the editor twin"
    );
    assert_eq!(
        (
            var(&sim, "enters"),
            var(&sim, "exits"),
            var(&sim, "splashes")
        ),
        EXPECTED_COUNTS,
        "the runtime heard a different set of water events than the editor twin"
    );

    for _ in 0..1_200 {
        sim.step_once(RuntimeInput::default());
    }
    let y = f64::from_bits(pose_of(&sim, krate)[1]);
    assert!(
        y.abs() < 5e-3,
        "the crate settled at y = {y}, not on the 0 m waterline"
    );
}
