//! **The arm-the-wiring law, for wave SCRIPT2's seventeen new verbs.**
//!
//! A `NodeDef` is a claim, a host arm is a second claim, and
//! `both_hosts_dispatch_exactly_the_registered_verbs` proves only that the two
//! are *spelled* the same — it reads source text. What it cannot see is whether
//! dispatching a verb has the effect its description promises, and this house
//! has a law about exactly that: **a gate must aim at the thing it names**.
//!
//! So every verb the wave added is driven **from a `.infini` script**, through
//! `inf_script::compile` into a real `SimSession` over a real scene, and what is
//! asserted is the *world* and the script's own member variables — never a
//! report, and never the registry that was already checked.
//!
//! # What is covered
//!
//! | namespace | verbs |
//! |---|---|
//! | `terrain` | `height_at` — registered this wave for a verb both hosts had implemented since P21.2 |
//! | `sky` | `is_day`, `get_hour`, `get_cloud_coverage`, `get_fog_density` |
//! | `door` | `is_locked`, `use`, `lock` (and `is_open`, which now shares their one resolution) |
//! | `health` | `damage`, `fraction`, `is_downed` |
//! | `crowd` | `population`, `blocked`, `homes`, `workplaces` |
//! | `zone` | `contains`, `count` |
//!
//! # The anti-vacuity discipline
//!
//! Every count is asserted **non-zero somewhere**: a crowd is installed rather
//! than asserted empty, the zone box holds two bodies rather than none, the door
//! is opened and locked rather than only queried. A gate over an empty world
//! would pass with every arm returning the default, which is the shape the P22
//! and P23 laws were paid for.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::Value;
use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, SkyAtmosphere, Terrain, TimeOfDay,
    Transform,
};
use inf_ecs::crowd::{CrowdArchetype, CrowdRecord};
use inf_ecs::math::Vec3d;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};
use inf_terrain::TerrainData;

const PROBE_GUID: u128 = 0x0005_C819_2000;
const NEIGHBOUR_GUID: u128 = 0x0005_C819_2001;
const TERRAIN_GUID: u128 = 0x0005_C819_2002;
const SKY_GUID: u128 = 0x0005_C819_2003;

/// The flat ground's height, metres. Not zero, deliberately: a `terrain.height_at`
/// that answered its own "no ground here" default would pass against 0.
const GROUND_Y: f64 = 7.5;
/// Metres per terrain sample.
const MPS: f64 = 4.0;
/// Samples per terrain tile edge.
const TILE_RES: u32 = 5;
/// The level clock, in UTC seconds since midnight — noon, so `sky.is_day` is
/// true and `sky.get_hour` has a value a reader can check by eye.
const NOON_S: f64 = 12.0 * 3600.0;

/// How many crowd agents the fixture installs — a number the four `crowd.*`
/// verbs have to reproduce rather than default to.
const AGENTS: usize = 3;

/// **The script.** Every new verb, written the way a designer would write it.
///
/// The door TOML is spelled with `\n` escapes rather than literal newlines,
/// which is the `chr(92)` law met from the other side: this is a Rust string
/// literal holding a string literal, and a scripted edit that resolved one
/// escape would silently change the program.
const PROBE: &str = "\
actor \"ApiProbe\"

var entity: int = 0
var ground: float = 0.0
var daytime: bool = false
var hour: float = 0.0
var fog: float = 0.0
var clouds: float = 0.0
var doors: int = 0
var door_open: bool = false
var door_locked: bool = false
var door_moved: bool = false
var lock_changed: bool = false
var absorbed: float = 0.0
var left: float = 0.0
var downed: bool = false
var people: int = 0
var stuck: int = 0
var homes: int = 0
var jobs: int = 0
var in_zone: bool = false
var zone_bodies: int = 0

on begin_play()
    doors = door.spawn(door_toml())
    sky.set_weather(\"fog\", 0.0)
    health.set(entity, 1000.0)
end

on tick(dt)
    ground = terrain.height_at(0.0, 0.0)
    daytime = sky.is_day()
    hour = sky.get_hour()
    fog = sky.get_fog_density()
    clouds = sky.get_cloud_coverage()
    door_open = door.is_open(20.0, 1.05, 0.0)
    door_locked = door.is_locked(20.0, 1.05, 0.0)
    people = crowd.population()
    stuck = crowd.blocked()
    homes = crowd.homes()
    jobs = crowd.workplaces()
    in_zone = zone.contains(entity, 0.5, 1.0, 0.0, 1.5, 0.6, 1.0)
    zone_bodies = zone.count(0.5, 1.0, 0.0, 1.5, 0.6, 1.0)
end

on input \"use\"(pressed)
    if pressed then
        door_moved = door.use(20.0, 1.05, 0.0)
    end
end

on input \"lock\"(pressed)
    if pressed then
        lock_changed = door.lock(20.0, 1.05, 0.0)
    end
end

on input \"hit\"(pressed)
    if pressed then
        absorbed = health.damage(entity, 400.0)
        left = health.fraction(entity)
        downed = health.is_downed(entity)
    end
end

function door_toml() -> string
    return \"[probe]\\nlabel = \\\"probe door\\\"\\nhinge = [20.0, 1.05, 0.0]\\nclosed_yaw_deg = 0.0\\ninside_yaw_deg = 90.0\\nopen_limit_deg = 95.0\\nlocked = false\\n\"
end
";

macro_rules! insert {
    ($doc:expr, $guid:expr, $component:expr) => {{
        let e = $doc.entity_of($guid).expect("entity");
        $doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .insert($component);
    }};
}

fn flat_terrain() -> TerrainData {
    let mut data = TerrainData::new(TILE_RES, MPS);
    for c in [(-1, -1), (-1, 0), (0, -1), (0, 0)] {
        data.author_tile(c, |_, _| GROUND_Y);
    }
    data
}

/// A static 1 m box at `p`.
fn add_box(doc: &mut SceneDoc, guid: Uuid, name: &str, p: DVec3) {
    doc.create_with_guid(guid, SpawnKind::Empty, name, None);
    insert!(doc, guid, Transform::from_translation(p));
    insert!(
        doc,
        guid,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        guid,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(0.5, 0.5, 0.5),
            ..Default::default()
        }
    );
}

/// Ground, a sky authority at noon, the scripted probe and one neighbour, both
/// standing inside the zone box the script asks about.
fn probe_doc() -> SceneDoc {
    probe_doc_with(true)
}

/// `neighbour` places a second 1 m box beside the probe, inside the zone box —
/// the differential the zone count is measured against.
fn probe_doc_with(neighbour: bool) -> SceneDoc {
    let mut doc = SceneDoc::new();

    {
        let terrain = Uuid::from_u128(TERRAIN_GUID);
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
    }

    let sky = Uuid::from_u128(SKY_GUID);
    doc.create_with_guid(sky, SpawnKind::Empty, "Sky", None);
    insert!(doc, sky, Transform::IDENTITY);
    insert!(
        doc,
        sky,
        TimeOfDay {
            seconds: NOON_S,
            rate: 0.0,
            ..TimeOfDay::default()
        }
    );
    insert!(doc, sky, SkyAtmosphere::default());

    add_box(
        &mut doc,
        Uuid::from_u128(PROBE_GUID),
        "Probe",
        DVec3::new(0.0, 1.0, 0.0),
    );
    if neighbour {
        add_box(
            &mut doc,
            Uuid::from_u128(NEIGHBOUR_GUID),
            "Neighbour",
            DVec3::new(1.0, 1.0, 0.0),
        );
    }

    doc.world_mut().propagate();
    doc
}

/// A crowd for the four `crowd.*` verbs to find, standing **a hundred metres
/// away** so that the agents the tier system materializes cannot wander into the
/// zone box and make `zone.count` a number about the crowd.
///
/// Installed **after** `SimSession::enter`, and that is not a detail: entering a
/// session calls `inf_ecs::crowd::clear_crowd` on the document, deliberately, so
/// a Simulate never leaves agents standing in the author's scene. The door for a
/// tool or a test that wants a crowd of its own is `set_crowd_population`, and
/// its own doc comment says so.
fn install_crowd(doc: &mut SceneDoc, session: &mut SimSession) {
    let records: BTreeMap<Uuid, CrowdRecord> = (0..AGENTS)
        .map(|i| {
            (
                Uuid::from_u128(0x0005_C819_3000 + i as u128),
                CrowdRecord::standing(
                    CrowdArchetype::default(),
                    DVec3::new(100.0 + i as f64 * 4.0, 0.0, 0.0),
                ),
            )
        })
        .collect();
    session.set_crowd_population(doc, records);
    assert_eq!(
        inf_ecs::crowd::crowd_stats(doc.world()).total(),
        AGENTS,
        "the fixture must install its crowd"
    );
}

fn enter() -> (SceneDoc, SimSession) {
    let (class, warnings) = inf_script::compile(PROBE, "script:api-probe")
        .unwrap_or_else(|d| panic!("{}", inf_script::render(&d)));
    assert!(
        warnings.is_empty(),
        "the probe script should compile clean: {}",
        inf_script::render(&warnings)
    );
    let mut doc = probe_doc();
    let actors = vec![(Uuid::from_u128(PROBE_GUID), class)];
    let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
    install_crowd(&mut doc, &mut session);
    (doc, session)
}

fn var(session: &SimSession, name: &str) -> Value {
    session
        .actor_var(Uuid::from_u128(PROBE_GUID), name)
        .unwrap_or_else(|| panic!("the probe has no `{name}`"))
        .clone()
}

fn var_f(session: &SimSession, name: &str) -> f64 {
    match var(session, name) {
        Value::Float(f) => f,
        Value::Int(i) => i as f64,
        other => panic!("`{name}` is {other:?}"),
    }
}

fn var_i(session: &SimSession, name: &str) -> i64 {
    match var(session, name) {
        Value::Int(i) => i,
        other => panic!("`{name}` is {other:?}"),
    }
}

fn var_b(session: &SimSession, name: &str) -> bool {
    match var(session, name) {
        Value::Bool(b) => b,
        other => panic!("`{name}` is {other:?}"),
    }
}

fn tick(doc: &mut SceneDoc, session: &mut SimSession) {
    session.tick(doc, 1.0 / SIM_HZ, SimInput::default());
}

fn press(doc: &mut SceneDoc, session: &mut SimSession, key: &str) {
    session.tick(doc, 1.0 / SIM_HZ, SimInput::with_down([key]));
    // Release, so the next press is an edge again.
    session.tick(doc, 1.0 / SIM_HZ, SimInput::default());
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **`terrain.height_at`** — the verb that was implemented in both hosts and had
/// no `NodeDef` until this wave, so no author could reach it.
///
/// The ground is at 7.5 m rather than 0, because a verb that answered its own
/// "no ground here" default would pass against zero.
#[test]
fn a_script_reads_the_ground_height() {
    let (mut doc, mut session) = enter();
    tick(&mut doc, &mut session);
    assert_eq!(
        var_f(&session, "ground"),
        GROUND_Y,
        "logs: {:?}",
        session.logs()
    );
}

/// **The four `sky.*` reads**, against a clock the fixture set and weather the
/// script itself asked for.
#[test]
fn a_script_reads_the_sky() {
    let (mut doc, mut session) = enter();
    tick(&mut doc, &mut session);

    assert!(var_b(&session, "daytime"), "noon is daytime");
    let hour = var_f(&session, "hour");
    assert!(
        (0.0..24.0).contains(&hour),
        "the local hour must be in 0..24, got {hour}"
    );
    assert_eq!(
        hour,
        inf_ecs::sky::local_hour(doc.world()),
        "the verb must answer the same number the crowd's schedules run on"
    );

    // BeginPlay asked for fog with a zero blend, so the parameters are already
    // there on the first Tick — which is what `blend_seconds = 0` promises.
    let fog = var_f(&session, "fog");
    assert!(
        fog > 0.0,
        "a fog preset must have a non-zero extinction, got {fog}"
    );
    let clouds = var_f(&session, "clouds");
    assert!(
        (0.0..=1.0).contains(&clouds),
        "cloud coverage is a fraction, got {clouds}"
    );
}

/// **The door quartet, over one resolution.** The script hangs its own door,
/// asks whether it is open and locked, opens it, and throws the bolt — and every
/// one of those four verbs is about the same leaf because
/// `inf_physics::d3::door::nearest` decides which leaf a point is about, once.
#[test]
fn a_script_opens_and_locks_the_door_it_hung() {
    let (mut doc, mut session) = enter();
    tick(&mut doc, &mut session);

    assert_eq!(var_i(&session, "doors"), 1, "the script hangs one door");
    assert!(!var_b(&session, "door_open"), "a fresh door is shut");
    assert!(!var_b(&session, "door_locked"), "and unlocked");

    // The bolt first, because `door.lock` is refused on an OPEN leaf and the
    // order is the thing a designer has to know.
    press(&mut doc, &mut session, "lock");
    assert!(
        var_b(&session, "lock_changed"),
        "logs: {:?}",
        session.logs()
    );
    assert!(
        var_b(&session, "door_locked"),
        "the bolt is thrown, and `door.is_locked` sees it"
    );

    // A locked door does not move, and says so as a value.
    press(&mut doc, &mut session, "use");
    assert!(
        !var_b(&session, "door_moved"),
        "a locked door reports that nothing moved"
    );
    assert!(!var_b(&session, "door_open"));

    // Unlock, then open.
    press(&mut doc, &mut session, "lock");
    assert!(!var_b(&session, "door_locked"), "the bolt is released");
    press(&mut doc, &mut session, "use");
    assert!(var_b(&session, "door_moved"), "an unlocked door opens");

    // The leaf swings over a few steps; `door.is_open` reports when it is far
    // enough to walk through.
    for _ in 0..SIM_HZ as usize {
        tick(&mut doc, &mut session);
    }
    assert!(
        var_b(&session, "door_open"),
        "the leaf should have swung open within a second"
    );
}

/// **The three `health.*` adds**, over the same door a bullet uses
/// (`weapon::damage_entity`).
#[test]
fn a_script_hurts_a_body_and_reads_what_is_left() {
    let (mut doc, mut session) = enter();
    tick(&mut doc, &mut session);

    press(&mut doc, &mut session, "hit");
    assert_eq!(
        var_f(&session, "absorbed"),
        400.0,
        "a body with 1000 J absorbs all 400: {:?}",
        session.logs()
    );
    assert_eq!(var_f(&session, "left"), 0.6, "600 of 1000 J left");
    assert!(!var_b(&session, "downed"), "600 J left is not down");

    // Two more blows finish it: the third asks for 400 and only 200 is left, so
    // `absorbed` is the honest 200 rather than the amount asked for.
    press(&mut doc, &mut session, "hit");
    press(&mut doc, &mut session, "hit");
    assert_eq!(var_f(&session, "absorbed"), 200.0);
    assert_eq!(var_f(&session, "left"), 0.0);
}

/// **The four `crowd.*` counts.** The fixture installs three agents, so a verb
/// that answered its own empty-world default would fail.
#[test]
fn a_script_counts_the_crowd() {
    let (mut doc, mut session) = enter();
    tick(&mut doc, &mut session);

    assert_eq!(
        var_i(&session, "people"),
        AGENTS as i64,
        "logs: {:?}",
        session.logs()
    );
    assert_eq!(
        var_i(&session, "people") as usize,
        session.crowd_stats().total(),
        "the verb must answer the same number the session's own instrument does"
    );
    assert_eq!(var_i(&session, "stuck"), 0, "nothing is blocked here");
    // This fixture has no settlement, so the society offers nothing — asserted
    // rather than skipped, because "0" is the documented answer and the arm
    // above is what proves the pair is not simply always 0.
    assert_eq!(var_i(&session, "homes"), 0);
    assert_eq!(var_i(&session, "jobs"), 0);
    assert_eq!(
        var_i(&session, "homes") as usize,
        session.society_stats().homes
    );
    assert_eq!(
        var_i(&session, "jobs") as usize,
        session.society_stats().works
    );
}

/// **The two `zone.*` queries** — the mission-class primitive.
///
/// `contains` is asserted absolutely (the probe is in its own box) and `count`
/// is asserted **differentially**: adding one body to the box adds exactly one
/// to the count. A differential is the honest instrument here, because the
/// physics world holds colliders no `Collider3D` component describes — a
/// terrain heightfield, a door's leaf — and the verb's contract is "entities
/// with a collider overlapping this box", not "actors somebody authored".
#[test]
fn a_script_asks_who_is_in_the_zone() {
    let (mut doc, mut session) = enter();
    tick(&mut doc, &mut session);
    assert!(
        var_b(&session, "in_zone"),
        "the probe stands at the box's centre: {:?}",
        session.logs()
    );
    let with_neighbour = var_i(&session, "zone_bodies");

    let (class, _) = inf_script::compile(PROBE, "script:api-probe-alone")
        .unwrap_or_else(|d| panic!("{}", inf_script::render(&d)));
    let mut alone_doc = probe_doc_with(false);
    let actors = vec![(Uuid::from_u128(PROBE_GUID), class)];
    let mut alone = SimSession::enter(&mut alone_doc, actors, DVec2::ZERO, SIM_HZ);
    tick(&mut alone_doc, &mut alone);
    let without = var_i(&alone, "zone_bodies");

    assert!(
        without >= 1,
        "the probe itself must be in the box, so the count cannot be 0"
    );
    assert_eq!(
        with_neighbour,
        without + 1,
        "one more body in the box is one more in the count, and nothing else moved"
    );
}

/// **The anti-vacuity control for the zone pair**: a box five hundred metres
/// away holds nobody, and the probe is not in it. Without this, a
/// `zone.contains` that answered `true` unconditionally would pass every arm
/// above, and a `zone.count` that returned the whole world would too.
#[test]
fn a_box_with_nothing_in_it_answers_so() {
    let src = PROBE
        .replace(
            "zone.contains(entity, 0.5, 1.0, 0.0, 1.5, 0.6, 1.0)",
            "zone.contains(entity, 500.0, 1.0, 0.0, 1.5, 0.6, 1.0)",
        )
        .replace(
            "zone.count(0.5, 1.0, 0.0, 1.5, 0.6, 1.0)",
            "zone.count(500.0, 1.0, 0.0, 1.5, 0.6, 1.0)",
        );
    assert_ne!(src, PROBE, "the substitution must have taken");
    let (class, _) = inf_script::compile(&src, "script:api-probe-empty")
        .unwrap_or_else(|d| panic!("{}", inf_script::render(&d)));
    let mut doc = probe_doc();
    let actors = vec![(Uuid::from_u128(PROBE_GUID), class)];
    let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
    tick(&mut doc, &mut session);

    assert!(!var_b(&session, "in_zone"));
    assert_eq!(var_i(&session, "zone_bodies"), 0);
}

/// **Nothing the script called reached the unknown-call logger.**
///
/// Both hosts answer an unrecognised path by logging it and returning `Unit`, so
/// a verb registered but not dispatched *does nothing at all, silently* — the
/// exact failure `both_hosts_dispatch_exactly_the_registered_verbs` reads source
/// text to prevent, checked here from the other side, at run time, over a script
/// that calls all seventeen.
#[test]
fn no_verb_the_script_calls_falls_through_to_the_logger() {
    let (mut doc, mut session) = enter();
    tick(&mut doc, &mut session);
    press(&mut doc, &mut session, "use");
    press(&mut doc, &mut session, "lock");
    press(&mut doc, &mut session, "hit");

    let unknown: Vec<&String> = session
        .logs()
        .iter()
        .filter(|l| {
            [
                "terrain::",
                "sky::",
                "door::",
                "health::",
                "crowd::",
                "zone::",
            ]
            .iter()
            .any(|p| l.starts_with(p))
        })
        .collect();
    assert!(
        unknown.is_empty(),
        "these verbs fell through to the unknown-call logger: {unknown:?}"
    );
}
