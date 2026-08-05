//! **P22.1 surface deformation: the editor preview and the shipped player press
//! the same footprints.**
//!
//! This is the sim half of the P22.1 mirror, and it is the half that decides
//! whether PIE will match shipping when P22.4 puts a deformation arm in a phase
//! gate. `projector_mirror.rs` asserts *structurally* that both fixed steps call
//! the one Ring-0 rule in the right slot; this asserts *behaviourally* that they
//! land the same bytes.
//!
//! The two hosts are driven directly rather than through a cooked pack: what is
//! being compared is the fixed step, and `RuntimeSim::new` / `SimSession::enter`
//! are the two fixed steps' own front doors. A cooked-pack arm belongs with the
//! P22.4 sample, where there is a level to cook.
//!
//! **The trace is bits, not floats** — `DeformField::state_bytes` throughout.

use glam::{DVec2, DVec3};
use inf_ecs::components::{
    CharacterController3D, Collider3D, ColliderShape3DKind, Terrain, Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_terrain::{TerrainData, DEFAULT_TILE_RESOLUTION};
use uuid::Uuid;

const HZ: f64 = 60.0;
const STEPS: u32 = 90;
const RES: u32 = DEFAULT_TILE_RESOLUTION;
/// The engine's snow layer — the deep archetype, so a walk leaves something to
/// compare rather than a millimetre of grass.
const SNOW: u8 = 3;

fn terrain_guid() -> Uuid {
    Uuid::from_u128(0x2201_0000_0000_0000_0000_0000_0000_0001)
}
fn walker_guid() -> Uuid {
    Uuid::from_u128(0x2201_0000_0000_0000_0000_0000_0000_0002)
}

/// A flat terrain painted entirely snow.
fn snow_terrain() -> Terrain {
    let mut data = TerrainData::new(RES, 1.0);
    data.author_tile((0, 0), |_, _| 0.0);
    let tile = data.get_or_create_tile((0, 0));
    tile.ensure_weights(RES);
    let mut w = [0u8; 4];
    w[SNOW as usize] = 255;
    for j in 0..RES {
        for i in 0..RES {
            tile.set_weight_sample(RES, i, j, w);
        }
    }
    Terrain {
        data,
        ..Default::default()
    }
}

/// The character: a capsule whose underside rests exactly on y = 0.
fn walker() -> (Transform, Collider3D, CharacterController3D) {
    (
        Transform {
            translation: Vec3d::new(4.0, 0.75, 6.0),
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.25, 0.5, 0.25),
            radius: 0.25,
            ..Default::default()
        },
        CharacterController3D::default(),
    )
}

/// Populate an `EcsWorld` with the fixture, using **explicit Guids** so the two
/// hosts' worlds are identical down to the sort key the stamp order uses.
fn build_world(world: &mut EcsWorld) {
    let t = world.spawn_with_guid(terrain_guid(), "terrain", None);
    world
        .world_mut()
        .entity_mut(t)
        .insert((snow_terrain(), Transform::default()));
    let a = world.spawn_with_guid(walker_guid(), "walker", None);
    let (transform, collider, cc) = walker();
    world
        .world_mut()
        .entity_mut(a)
        .insert((transform, collider, cc));
    world.propagate();
}

/// Move the walker to step `s` of a straight 4.5 m walk.
fn place(world: &mut EcsWorld, s: u32) {
    let Some(e) = world.entity_of(walker_guid()) else {
        return;
    };
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
        t.translation.x = 4.0 + s as f64 * 0.05;
    }
    world.mark_dirty();
    world.propagate();
}

fn player_trace() -> Vec<Vec<u8>> {
    let mut world = EcsWorld::new();
    build_world(&mut world);
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::new(0.0, -9.81), HZ);
    (0..STEPS)
        .map(|s| {
            place(sim.world_mut(), s);
            sim.step_once(RuntimeInput::default());
            inf_ecs::deform::deform_state_bytes(sim.world())
        })
        .collect()
}

fn editor_trace() -> Vec<Vec<u8>> {
    let mut doc = SceneDoc::new();
    build_world(doc.world_mut());
    let mut session = SimSession::enter(&mut doc, Vec::new(), DVec2::new(0.0, -9.81), HZ);
    let out = (0..STEPS)
        .map(|s| {
            place(doc.world_mut(), s);
            session.step_once(&mut doc, SimInput::default());
            inf_ecs::deform::deform_state_bytes(doc.world())
        })
        .collect();
    session.exit(&mut doc);
    out
}

/// **ANTI-VACUITY.** Two empty traces are equal too. The walk has to have
/// actually pressed snow, and it has to have gone on pressing as it moved.
fn assert_not_vacuous(trace: &[Vec<u8>]) {
    assert_eq!(trace.len() as u32, STEPS);
    assert!(
        !trace[0].is_empty(),
        "the first step pressed nothing — the fixture never touched the ground"
    );
    assert_ne!(
        trace.first(),
        trace.last(),
        "the field never changed over the whole walk"
    );
}

#[test]
fn the_editor_preview_and_the_shipped_player_press_the_same_field() {
    let shipped = player_trace();
    let preview = editor_trace();
    assert_not_vacuous(&shipped);
    assert_eq!(
        shipped, preview,
        "the shipped player and the editor Simulate pressed different footprints \
         into the same walk — PIE would stop matching shipping the first time a \
         character stood on snow"
    );
}

#[test]
fn the_shipped_trace_reproduces_from_cold() {
    // Determinism, per host: the field is a pure function of the step history, so
    // two fresh runs are byte-identical.
    let a = player_trace();
    let b = player_trace();
    assert_not_vacuous(&a);
    assert_eq!(a, b);
    let c = editor_trace();
    let d = editor_trace();
    assert_eq!(c, d);
}

/// The field rides the **existing** trace surface rather than a second one:
/// `RuntimeSim::state_bytes` — which `step_state_hash`, the replay fold and the
/// PIE `Frame::state_hash` all consume — now carries it.
///
/// Both halves are asserted, because either alone is a trap: that the deformation
/// really moves the hash (else every future PIE arm would be blind to it), and
/// that a level with **no** deformation produces exactly the bytes it always did
/// (else every pre-P22.1 trace would have silently changed).
#[test]
fn the_field_rides_the_state_hash_and_only_when_there_is_one() {
    let mut world = EcsWorld::new();
    build_world(&mut world);
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::new(0.0, -9.81), HZ);
    let before = sim.state_bytes();
    for s in 0..STEPS {
        place(sim.world_mut(), s);
        sim.step_once(RuntimeInput::default());
    }
    let after = sim.state_bytes();
    assert_ne!(before, after);
    assert!(
        after.len() > before.len(),
        "the deformation field is not reaching `state_bytes`"
    );
    assert!(sim.deform_field().is_some_and(|f| !f.is_empty()));

    // The off-path half: the identical fixture with the walker lifted clear of the
    // ground never grows a field, and its `state_bytes` carries no extra section.
    let mut world = EcsWorld::new();
    build_world(&mut world);
    let e = world.entity_of(walker_guid()).expect("walker");
    world
        .world_mut()
        .get_mut::<Transform>(e)
        .unwrap()
        .translation = Vec3d::new(4.0, 40.0, 6.0);
    world.mark_dirty();
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::new(0.0, -9.81), HZ);
    let n0 = sim.state_bytes().len();
    for _ in 0..STEPS {
        sim.step_once(RuntimeInput::default());
    }
    assert!(sim.deform_field().is_none());
    assert_eq!(
        sim.state_bytes().len(),
        n0,
        "a level that deforms nothing must trace exactly the bytes it always did"
    );
}

/// The stamp path is **outside** the scheduled world, so it cannot be reordered
/// by the parallel ECS executor and cannot depend on the task-pool size.
///
/// Stated as a test rather than only as a comment, because it is the reason this
/// batch needs no subprocess pool gate of its own: `step_deformation` is called
/// from the host's fixed step directly (like `advance_weather`), it holds the
/// world exclusively for its whole duration, and it iterates a `BTreeMap`. The
/// existing `replay_probe` / pool-invariance harness covers the scheduled half
/// and is unaffected.
#[test]
fn the_stamp_order_is_the_guid_order_and_nothing_elses() {
    let mut world = EcsWorld::new();
    build_world(&mut world);
    // A second walker with a HIGHER guid, overlapping the first's footprint. If
    // anything but Guid order decided the sequence, the two would race for the
    // shared samples and the bytes would wobble between runs.
    let second = Uuid::from_u128(0x2201_0000_0000_0000_0000_0000_0000_00ff);
    let b = world.spawn_with_guid(second, "walker-2", None);
    let (mut transform, collider, cc) = walker();
    transform.translation.x = 4.1;
    world
        .world_mut()
        .entity_mut(b)
        .insert((transform, collider, cc));
    world.mark_dirty();
    world.propagate();

    let contacts = inf_ecs::deform::ground_contacts(&world);
    assert_eq!(contacts.len(), 2);
    assert!(
        contacts[0].guid < contacts[1].guid,
        "contacts must be Guid-sorted before anything is stamped"
    );

    // (`EcsWorld` is not `Clone`, so the two runs rebuild the fixture rather than
    // sharing one.)
    let build_two = || {
        let mut world = EcsWorld::new();
        build_world(&mut world);
        let b = world.spawn_with_guid(second, "walker-2", None);
        let (mut transform, collider, cc) = walker();
        transform.translation.x = 4.1;
        world
            .world_mut()
            .entity_mut(b)
            .insert((transform, collider, cc));
        world.mark_dirty();
        world.propagate();
        let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::new(0.0, -9.81), HZ);
        for _ in 0..STEPS {
            sim.step_once(RuntimeInput::default());
        }
        inf_ecs::deform::deform_state_bytes(sim.world())
    };
    let a = build_two();
    let b2 = build_two();
    assert!(!a.is_empty());
    assert_eq!(a, b2);
}

/// The snowfall hook, through the hosts: the rate crosses from
/// `inf_ecs::sky::ResolvedSky::snow_accumulation_rate` into the field and nowhere
/// else. With no sky in the level the rate is zero, so the refill is inert —
/// which is what makes P22.1 free for every level that never opted into weather.
#[test]
fn a_level_with_no_sky_refills_nothing() {
    let mut world = EcsWorld::new();
    build_world(&mut world);
    assert_eq!(inf_ecs::deform::snow_accumulation_rate(&world), 0.0);
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::new(0.0, -9.81), HZ);
    for _ in 0..STEPS {
        sim.step_once(RuntimeInput::default());
    }
    let pressed = sim
        .deform_field()
        .expect("the walker pressed snow")
        .depth_at(DVec2::new(4.0, 6.0));
    assert!(pressed > 0.0);
    // Snow never recovers on its own (LAYER_RESPONSE), and with no sky nothing
    // refills it — so a thousand idle steps leave the print exactly as it was.
    for _ in 0..1_000 {
        sim.step_once(RuntimeInput::default());
    }
    assert_eq!(
        sim.deform_field().unwrap().depth_at(DVec2::new(4.0, 6.0)),
        pressed
    );
}

/// A sanity check on the fixture itself, so the arms above cannot pass because
/// the walk happened somewhere nobody looked.
#[test]
fn the_walk_lands_where_the_fixture_says() {
    let mut world = EcsWorld::new();
    build_world(&mut world);
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::new(0.0, -9.81), HZ);
    for s in 0..STEPS {
        place(sim.world_mut(), s);
        sim.step_once(RuntimeInput::default());
    }
    let field = sim.deform_field().expect("a field");
    assert!(field.depth_at(DVec2::new(4.0, 6.0)) > 0.0, "the start");
    assert!(field.depth_at(DVec2::new(8.4, 6.0)) > 0.0, "the end");
    assert_eq!(field.depth_at(DVec2::new(4.0, 20.0)), 0.0, "off the trail");
    assert_eq!(field.layer_at(DVec2::new(4.0, 6.0)), Some(SNOW));
    assert!(DVec3::ZERO.length() == 0.0);
}
