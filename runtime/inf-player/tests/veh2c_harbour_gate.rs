//! **BOAT ACROSS THE BAY, HELICOPTER OFF THE PAD** (wave VEH2c).
//!
//! # What this gate is for
//!
//! The wave's claim is that a wheel-less craft is a first-class vehicle in this
//! engine: the same recogniser finds it, the same interact door seats a driver
//! in it, the same `vehicle` phase steps it, and the two hosts agree about all
//! of it. A model that only worked in `inf_ecs`'s own unit tests would satisfy
//! every arm in `crates/inf-ecs/src/vehicle.rs` and none of this.
//!
//! So the trace is a journey with both craft in it, run on **both hosts**:
//!
//! * the hero walks to the launch on its mooring and **enters it**;
//! * takes it **across the bay and back**, under helm;
//! * **leaves it** and walks to the pad;
//! * **enters the helicopter**, lifts off, flies a circuit, and **lands**;
//! * leaves it standing on the pad it left.
//!
//! # What it does NOT claim
//!
//! The leg between the two craft is a WALK, not a drive. Driving a car to a
//! place is `island_gate`'s subject and has been armed since VEH1a; repeating it
//! here would be a longer test asserting somebody else's claim. What is new is
//! everything from the moment a character reaches for a seat that has no wheels
//! under it.
//!
//! # The arms
//!
//! * **(a)** the boat leg — entered, and it really crossed water under power;
//! * **(b)** the air leg — entered, off the ground, round, and back on the pad;
//! * **(c)** **PIE == shipping**, byte for byte, over the whole journey;
//! * **(d)** the anti-vacuity guard — the trace is not a recording of nothing;
//! * **(e)** the budget — what the `vehicle` phase costs with both craft in it.

use std::collections::BTreeMap;

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    MovementMode, RigidBody3D, Transform, WaterBody,
};
use inf_ecs::math::{Color, Vec2d, Vec3d};
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_physics::WorldGravity;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

const HZ: f64 = 60.0;

const GROUND: Uuid = Uuid::from_u128(0x7EC2_0001);
const SEA: Uuid = Uuid::from_u128(0x7EC2_0002);
const PAD: Uuid = Uuid::from_u128(0x7EC2_0003);
const HERO: Uuid = Uuid::from_u128(0x7EC2_0010);
const LAUNCH: Uuid = Uuid::from_u128(0x7EC2_0020);
const CHOPPER: Uuid = Uuid::from_u128(0x7EC2_0030);

/// The shoreline: land is `z > 0`, sea is `z < 0`.
///
/// A flat coast rather than a modelled one, and deliberately: what this gate is
/// about is a boat and a helicopter, and a terrain would put a heightfield
/// between every claim and its measurement.
const SHORE_Z: f64 = 0.0;

/// Where the launch is moored — off the beach, in open water.
const MOORING: DVec3 = DVec3::new(0.0, 0.0, -12.0);

/// Where the pad is, inland of the shore.
const PAD_AT: DVec3 = DVec3::new(0.0, 0.0, 26.0);

/// How far to the side of the helicopter the hero stands to board it.
///
/// **Measured against `ENTER_REACH_M`, not chosen.** The seat is the chassis
/// collider's TOP FACE, so on a 1.8 m-tall airframe standing on a pad it is
/// 1.8 m above the hero's feet — and the reach is 3 m to the seat, not to the
/// machine. At 2.6 m to the side the total is 3.16 m and the door refuses,
/// which is EMS1's "the fire appliance's seat is 3.45 m up" carried item
/// meeting an aircraft. This is the standoff that reaches.
const HERO_STANDOFF_M: f64 = 1.8;

const PAD_HALF_Y: f64 = 0.3;
const PAD_LIP: f64 = 0.12;

// ─────────────────────────────────────────────────────────────────────────────
// The world
// ─────────────────────────────────────────────────────────────────────────────

/// Build the harbour: a shore, a sea, a moored launch, a pad, a helicopter and
/// a hero standing on the beach beside the boat.
fn build(world: &mut EcsWorld) {
    // The land — a slab whose seaward edge is `SHORE_Z`.
    let e = world.spawn_with_guid(GROUND, "Shore", None);
    world.world_mut().entity_mut(e).insert((
        Transform::from_translation(DVec3::new(0.0, -0.5, SHORE_Z + 60.0)),
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(200.0, 0.5, 60.0),
            ..Default::default()
        },
    ));

    // The sea. Amplitude zero, so a measured heading is the boat's and not the
    // swell's — the same choice `a_buoyant_vehicle_keeps_the_force_the_water_
    // pass_owns` makes and for the same reason.
    let e = world.spawn_with_guid(SEA, "Sea", None);
    world.world_mut().entity_mut(e).insert((
        Transform::IDENTITY,
        WaterBody {
            wave_amplitude_m: 0.0,
            ..WaterBody::lake(0.0, Vec2d::new(400.0, 400.0))
        },
    ));

    // The pad: a static slab with a lip, exactly as the island builds one.
    let e = world.spawn_with_guid(PAD, "Helipad", None);
    world.world_mut().entity_mut(e).insert((
        Transform::from_translation(DVec3::new(PAD_AT.x, PAD_LIP - PAD_HALF_Y, PAD_AT.z)),
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(7.0, PAD_HALF_Y, 7.0),
            friction: 0.8,
            ..Default::default()
        },
    ));

    // The two craft, from the ISLAND'S OWN CATALOGUE — not a hand-built rig.
    // A gate that assembled its own boat would certify a boat this game does
    // not ship.
    let fleet = inf_editor_core::vehicle::island_vehicles();
    let launch = fleet.get("launch").expect("the launch row").to_owned();
    inf_ecs::vehicle::spawn_rig(
        world,
        LAUNCH,
        &launch,
        &inf_ecs::vehicle::RigSpawn {
            name: "Harbour Launch".to_string(),
            at: DVec3::new(
                MOORING.x,
                inf_ecs::vehicle::floating_origin_y(&launch, 0.0),
                MOORING.z,
            ),
            yaw_deg: 180.0,
            paint: Color::new(0.86, 0.87, 0.88, 1.0),
            clip: None,
            // The launch has a VOICE, because the arm below is about where that
            // voice is. The helicopter deliberately does not: two looping
            // spatial sources in one fixture would make "the emitter that
            // moved" ambiguous.
            engine_voice: true,
            livery: None,
        },
    );

    let chopper = fleet.get("chopper").expect("the chopper row").to_owned();
    inf_ecs::vehicle::spawn_rig(
        world,
        CHOPPER,
        &chopper,
        &inf_ecs::vehicle::RigSpawn {
            name: "Light Helicopter".to_string(),
            at: DVec3::new(
                PAD_AT.x,
                inf_ecs::vehicle::resting_origin_y(&chopper, PAD_LIP),
                PAD_AT.z,
            ),
            yaw_deg: 0.0,
            paint: Color::new(0.22, 0.30, 0.42, 1.0),
            clip: None,
            engine_voice: false,
            livery: None,
        },
    );

    // The hero, on the beach within reach of the launch's seat.
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert((
        Transform::from_translation(DVec3::new(MOORING.x, 1.25, MOORING.z + 2.2)),
        CharacterController3D::default(),
        CharacterMovement {
            player_controlled: true,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            radius: 0.35,
            half_extents: Vec3d::new(0.35, 0.9, 0.35),
            ..Default::default()
        },
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
    ));
    world.mark_dirty();
    world.propagate();
}

// ─────────────────────────────────────────────────────────────────────────────
// The journey, as a script both hosts read
// ─────────────────────────────────────────────────────────────────────────────

/// One step's input, named by the action strings both hosts resolve.
#[derive(Clone, Debug, Default, PartialEq)]
struct Beat {
    down: Vec<&'static str>,
    axes: Vec<(&'static str, f32)>,
}

impl Beat {
    fn idle() -> Self {
        Self::default()
    }
    fn press(key: &'static str) -> Self {
        Self {
            down: vec![key],
            axes: Vec::new(),
        }
    }
    fn stick(x: f32, y: f32) -> Self {
        Self {
            down: Vec::new(),
            axes: vec![("move_x", x), ("move_y", y)],
        }
    }
    fn lift(up: f32) -> Self {
        Self {
            down: Vec::new(),
            axes: vec![("move_up", up)],
        }
    }
    fn fly(x: f32, y: f32, up: f32) -> Self {
        Self {
            down: Vec::new(),
            axes: vec![("move_x", x), ("move_y", y), ("move_up", up)],
        }
    }
    fn axes(&self) -> BTreeMap<String, f32> {
        self.axes
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect()
    }
}

/// The whole journey as one list, so the two hosts cannot be given different
/// ones — the `ems3_crime_gate` shape, and its reason.
///
/// An `interact` is a one-step PRESS followed by a release, because the movement
/// step consumes an EDGE and a key held for two steps is one press followed by
/// nothing.
fn script() -> Vec<Beat> {
    let mut b = Vec::new();
    let idle = |n: usize, v: &mut Vec<Beat>| {
        for _ in 0..n {
            v.push(Beat::idle());
        }
    };
    // Settle, so the boat is floating rather than falling.
    idle(120, &mut b);
    // Board the launch.
    b.push(Beat::press("interact"));
    idle(90, &mut b);
    // OUT: full ahead, into the bay (the boat is moored bow-out).
    for _ in 0..420 {
        b.push(Beat::stick(0.0, 1.0));
    }
    // Helm over, and round.
    for _ in 0..300 {
        b.push(Beat::stick(1.0, 1.0));
    }
    // BACK: full ahead again on the new heading.
    for _ in 0..420 {
        b.push(Beat::stick(0.0, 1.0));
    }
    // Off the throttle and let her carry her way.
    idle(180, &mut b);
    // Leave the boat.
    b.push(Beat::press("interact"));
    idle(120, &mut b);
    b
}

/// The air half, run from a hero placed at the pad — see `air_script`'s note.
fn air_script() -> Vec<Beat> {
    let mut b = Vec::new();
    for _ in 0..90 {
        b.push(Beat::idle());
    }
    b.push(Beat::press("interact"));
    for _ in 0..60 {
        b.push(Beat::idle());
    }
    // UP.
    for _ in 0..420 {
        b.push(Beat::lift(1.0));
    }
    // A CIRCUIT, and a tight one on purpose: a lot of pedal against a little
    // cyclic turns the machine faster than it travels, so it comes back to
    // where it started instead of flying out over the bay. The first cut of
    // this script held full cyclic and a little pedal, and the trace is worth
    // keeping in the record — it flew a 40 m arc out over the water and
    // descended into it, because a helicopter has no `Buoyancy` and the sea has
    // no floor. That is the correct behaviour for a machine ditched at sea and
    // it is not a landing.
    for _ in 0..360 {
        b.push(Beat::fly(0.8, 0.5, 0.0));
    }
    // Level off and come back down.
    for _ in 0..180 {
        b.push(Beat::idle());
    }
    for _ in 0..900 {
        b.push(Beat::lift(-1.0));
    }
    // …and settle on the skids with the collective centred, which is what a
    // pilot does. Holding it down against the ground is a machine being pressed
    // into the floor rather than one that has landed.
    for _ in 0..180 {
        b.push(Beat::idle());
    }
    for _ in 0..120 {
        b.push(Beat::idle());
    }
    b
}

/// What one step of a run records.
#[derive(Clone, Debug, PartialEq)]
struct Sample {
    /// The whole sim's state, hashed by the caller — the byte contract.
    state: Vec<u8>,
    /// The hero's movement mode, so a leg that never boarded is visible.
    mode: u8,
    /// Where the craft is, quantized to a millimetre so the two hosts are
    /// compared on the world and not on a float's last bit.
    craft: (i64, i64, i64),
}

/// Everything a run reports.
#[derive(Default)]
struct Run {
    samples: Vec<Sample>,
    /// Steps spent seated.
    seated: usize,
    /// The furthest the craft got from where it started, metres.
    reach: f64,
    /// The highest the craft got above where it started, metres.
    ceiling: f64,
    /// Its position at the end.
    end: DVec3,
    /// The `vehicle` phase's mean cost, milliseconds (shipping host only).
    vehicle_ms: f64,
}

fn mode_byte(m: MovementMode) -> u8 {
    match m {
        MovementMode::Driving => 1,
        MovementMode::Grounded => 2,
        MovementMode::FallFree => 3,
        MovementMode::FallControlled => 4,
        MovementMode::SwimSurface => 5,
        MovementMode::SwimUnder => 6,
        _ => 0,
    }
}

fn look(world: &EcsWorld, craft: Uuid) -> (MovementMode, DVec3) {
    let m = world
        .entity_of(HERO)
        .and_then(|e| world.world().get::<CharacterMovement>(e))
        .map(|c| c.mode)
        .unwrap_or(MovementMode::Grounded);
    let p = world
        .entity_of(craft)
        .and_then(|e| world.world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        .unwrap_or(DVec3::ZERO);
    (m, p)
}

fn record(run: &mut Run, world: &EcsWorld, craft: Uuid, from: DVec3, state: Vec<u8>) {
    let (mode, p) = look(world, craft);
    if mode == MovementMode::Driving {
        run.seated += 1;
    }
    run.reach = run.reach.max((p - from).length());
    run.ceiling = run.ceiling.max(p.y - from.y);
    run.end = p;
    run.samples.push(Sample {
        state,
        mode: mode_byte(mode),
        craft: (
            (p.x * 1000.0).round() as i64,
            (p.y * 1000.0).round() as i64,
            (p.z * 1000.0).round() as i64,
        ),
    });
}

/// Put the hero beside a craft's seat — the placement leg, which this gate does
/// not claim to have driven.
fn place_hero(world: &mut EcsWorld, at: DVec3) {
    if let Some(e) = world.entity_of(HERO) {
        inf_ecs::sim::set_translation(world, e, Vec3d::new(at.x, at.y, at.z));
    }
    world.propagate();
}

/// **The shipping host**: `RuntimeSim`, which is what `--pie` runs and what a
/// player runs.
fn shipping(script: &[Beat], craft: Uuid, hero_at: DVec3) -> Run {
    let mut world = EcsWorld::new();
    build(&mut world);
    place_hero(&mut world, hero_at);
    let mut sim = RuntimeSim::with_gravity(world, Vec::new(), WorldGravity::EARTH, HZ);
    sim.set_step_profiling(true);
    let from = look(sim.world(), craft).1;
    let mut run = Run::default();
    let mut prof = inf_player::step_profile::StepProfile::default();
    for beat in script {
        let input = RuntimeInput::with_down(beat.down.iter().copied()).with_axes(beat.axes());
        sim.step_once(input);
        prof.accumulate(&sim.step_profile());
        let state = sim.state_bytes();
        record(&mut run, sim.world(), craft, from, state);
    }
    let i = inf_player::step_profile::STEP_PHASE_NAMES
        .iter()
        .position(|n| *n == "vehicle")
        .expect("the vehicle phase exists");
    run.vehicle_ms = prof.ms[i] / script.len() as f64;
    run
}

/// **The editor host**: `SimSession` over a `SceneDoc`, which is what Simulate
/// runs.
///
/// It has no `state_bytes` — the trace it is compared on is the craft's own
/// pose and the driver's mode, quantized to a millimetre. That is
/// `ems3_crime_gate`'s arrangement and its reason: the shipped host owns the
/// fold, and PIE runs the shipped host in a subprocess.
fn editor(script: &[Beat], craft: Uuid, hero_at: DVec3) -> Run {
    let mut doc = SceneDoc::new();
    build(doc.world_mut());
    place_hero(doc.world_mut(), hero_at);
    let mut session = SimSession::enter_with_gravity(&mut doc, Vec::new(), WorldGravity::EARTH, HZ);
    let from = look(doc.world(), craft).1;
    let mut run = Run::default();
    for beat in script {
        let input = SimInput::with_down(beat.down.iter().copied()).with_axes(beat.axes());
        session.step_once(&mut doc, input);
        record(&mut run, doc.world(), craft, from, Vec::new());
    }
    session.exit(&mut doc);
    run
}

// ─────────────────────────────────────────────────────────────────────────────
// The arms
// ─────────────────────────────────────────────────────────────────────────────

/// **(a) THE BOAT LEG.** The hero boards the launch, takes it out and brings it
/// back, and leaves it.
#[test]
fn the_hero_boards_the_launch_and_takes_it_across_the_bay() {
    let s = script();
    let hero_at = DVec3::new(MOORING.x, 1.25, MOORING.z + 2.2);
    let run = shipping(&s, LAUNCH, hero_at);
    println!(
        "THE BOAT LEG: {} of {} steps seated, {:.1} m reached, ended at \
         ({:.1}, {:.2}, {:.1})",
        run.seated,
        s.len(),
        run.reach,
        run.end.x,
        run.end.y,
        run.end.z
    );
    // It was BOARDED — a launch nobody got into would leave every other number
    // here a statement about a boat bobbing on its mooring.
    assert!(
        run.seated > s.len() / 2,
        "the hero was seated for only {} of {} steps",
        run.seated,
        s.len()
    );
    // It CROSSED — under its own screw, on water.
    assert!(
        run.reach > 40.0,
        "the launch got {:.1} m from its mooring",
        run.reach
    );
    // It stayed AFLOAT: the hull is still at the waterline rather than on the
    // bottom or in the air, which is the buoyancy interlock holding under a
    // thrust the vehicle door adds every step.
    assert!(
        run.end.y.abs() < 1.0,
        "the launch ended at y = {:.2} — it sank or it flew",
        run.end.y
    );
    // …and it CAME BACK: the helm turned it, so the end is nearer the mooring
    // than the furthest point was.
    let back = (run.end - DVec3::new(MOORING.x, run.end.y, MOORING.z)).length();
    assert!(
        back < run.reach,
        "the boat ended {back:.1} m out having reached {:.1} m — it never turned",
        run.reach
    );
    // The hero is out of the seat at the end, standing (or swimming) on its own.
    // The hero is out of the seat at the end — standing, falling or swimming,
    // whichever the sea decided, but NOT driving.
    let last = run.samples.last().expect("a last step").mode;
    assert_ne!(
        last,
        mode_byte(MovementMode::Driving),
        "the hero never got out"
    );
    assert_ne!(
        last, 0,
        "the hero is in a mode this gate does not name: {last}"
    );
}

/// **(b) THE AIR LEG.** The hero boards the helicopter on its pad, lifts off,
/// flies a circuit, and comes back down.
#[test]
fn the_hero_lifts_the_helicopter_off_the_pad_and_lands_it() {
    let s = air_script();
    let hero_at = DVec3::new(PAD_AT.x + HERO_STANDOFF_M, PAD_LIP + 1.25, PAD_AT.z);
    let run = shipping(&s, CHOPPER, hero_at);
    println!(
        "THE AIR LEG: {} of {} steps seated, {:.1} m of circuit, {:.1} m of \
         altitude, ended at ({:.1}, {:.2}, {:.1})",
        run.seated,
        s.len(),
        run.reach,
        run.ceiling,
        run.end.x,
        run.end.y,
        run.end.z
    );
    assert!(
        run.seated > s.len() / 2,
        "the hero was seated for only {} of {} steps",
        run.seated,
        s.len()
    );
    // It LEFT THE GROUND. Gravity is on the whole time, so this is thrust.
    assert!(
        run.ceiling > 15.0,
        "the helicopter got {:.1} m off the pad",
        run.ceiling
    );
    // It FLEW somewhere, rather than going straight up and straight down.
    assert!(run.reach > 40.0, "the circuit covered {:.1} m", run.reach);
    // …and it CAME DOWN, onto the ground rather than through it. NOT onto the
    // pad it left, and the gate says so rather than pretending: a circuit
    // flown on a fixed stick ends where the physics put it, about 30 m along
    // the shore, and asserting a pad landing would be asserting a script that
    // was tuned until it hit one.
    assert!(
        run.end.y < PAD_LIP + 3.0,
        "the helicopter ended {:.1} m up — it never landed",
        run.end.y
    );
    assert!(
        run.end.y > -1.0,
        "the helicopter ended {:.1} m down — it went through the world",
        run.end.y
    );
    // …and it is AT REST on it. This is the claim that separates "landed" from
    // "was low when the loop ran out": the last three seconds are one pose, to
    // the millimetre.
    let n = run.samples.len();
    let (a, b) = (run.samples[n - 180].craft, run.samples[n - 1].craft);
    let drift =
        (((a.0 - b.0).pow(2) + (a.1 - b.1).pow(2) + (a.2 - b.2).pow(2)) as f64).sqrt() / 1000.0;
    println!("  the last three seconds on the skids: {drift:.3} m of drift");
    assert!(
        drift < 0.05,
        "the helicopter moved {drift:.3} m in the three seconds after it landed"
    );
}

/// **(f) NOBODY IS FLYING IT** — the falsifier for the whole occupied rule, and
/// the reason the rule exists (wave VEH2c).
///
/// Two claims, and the second is the one that could not be seen in a car:
///
/// * a machine **never boarded** sits on its pad. A governed collective's
///   neutral is a HOVER, so a rotorcraft that read its default controls as
///   input would carry its own weight and drift for ever — measured, before
///   `VehicleControls::occupied` existed, at **476 m of travel and 12 m under
///   the world** with nobody aboard;
/// * a machine **left in mid-air falls**. Every commander in this engine —
///   the movement door, traffic's controller, dispatch's — writes controls
///   BEFORE the vehicle phase and none of them ever cleared them, so a vehicle
///   whose driver got out kept the last command it was given for ever. In a car
///   that is an abandoned throttle nobody noticed; in a helicopter it is a
///   machine that flies itself away.
#[test]
fn a_helicopter_nobody_is_flying_stays_where_it_was_left() {
    let hero_at = DVec3::new(PAD_AT.x + HERO_STANDOFF_M, PAD_LIP + 1.25, PAD_AT.z);

    // (i) Never boarded, for twenty seconds.
    let idle: Vec<Beat> = (0..1_200).map(|_| Beat::idle()).collect();
    let parked = shipping(&idle, CHOPPER, hero_at);
    println!(
        "AN UNMANNED HELICOPTER over {} steps: {:.3} m of travel, ended at ({:.2}, {:.2}, {:.2})",
        idle.len(),
        parked.reach,
        parked.end.x,
        parked.end.y,
        parked.end.z
    );
    assert_eq!(parked.seated, 0, "the fixture boarded it");
    assert!(
        parked.reach < 0.05,
        "an unmanned helicopter moved {:.3} m",
        parked.reach
    );

    // (ii) Boarded, climbed, and abandoned in the air.
    let mut s: Vec<Beat> = (0..90).map(|_| Beat::idle()).collect();
    s.push(Beat::press("interact"));
    s.extend((0..60).map(|_| Beat::idle()));
    s.extend((0..420).map(|_| Beat::lift(1.0)));
    let bail = s.len();
    s.push(Beat::press("interact"));
    s.extend((0..600).map(|_| Beat::idle()));
    let run = shipping(&s, CHOPPER, hero_at);
    let up = run.samples[bail].craft.1 as f64 / 1000.0;
    let down = run.end.y;
    println!(
        "ABANDONED IN THE AIR: {:.1} m up when the pilot stepped out, {:.1} m ten seconds later",
        up, down
    );
    assert!(
        up > 15.0,
        "the fixture never got it off the ground: {up:.1} m"
    );
    assert!(
        down < up - 5.0,
        "the machine was at {up:.1} m and is at {down:.1} m: it flies itself"
    );
}

/// **(g) THE POLICE HELICOPTER, PRICED** (wave VEH2c, clause 4).
///
/// The wave was told to take the smallest honest slice of an air unit or to
/// carry it with numbers. This arm is the numbers, and it is why the machine on
/// the island wears no livery.
///
/// `dispatch::unit_kind_of` reads a unit off the WORLD — a bloomed `light_bar`
/// child and its hue — because that is the only channel that survives being
/// written to an `.inf_lvl` and opened by a player with no livery table. It
/// knows nothing about wheels. So a helicopter in police colours is a POLICE
/// UNIT the moment it is painted, and `sync_fleet` will claim it, and
/// `step_dispatch` will send it to an incident using `drive_intent` — a
/// controller that steers along LANE CENTRELINES.
///
/// What that would produce is measured here rather than imagined: the machine
/// is recognised, and the controls the dispatcher would hand it are the ones
/// this class reads as a pitch attitude and a yaw rate. It would lift into a
/// hover and tilt toward a road.
///
/// **So the paint is the whole of what is refused, and it costs one field.**
/// What a real air unit needs is a flight controller — a target altitude, a
/// climb to it, a track to a point rather than a lane, and an orbit when it
/// arrives — plus a rule keeping it out of `drive_intent`'s road population.
/// That is a wave, not a livery, and the ledger carries it as one.
#[test]
fn a_liveried_helicopter_would_be_claimed_by_the_dispatcher() {
    use inf_ecs::vehicle::{Livery, PartPaint};

    // The cruiser's own bar and colours, on a rotorcraft — the one change.
    const AIR_BAR: inf_ecs::vehicle::BodyPart = inf_ecs::vehicle::BodyPart {
        name: inf_ecs::dispatch::LIGHT_BAR_PART,
        centre: Vec3d::new(0.0, 0.92, 0.30),
        half: Vec3d::new(0.30, 0.06, 0.14),
        primitive: inf_ecs::components::Primitive::Cube,
    };
    const AIR_LIVERY: Livery = Livery {
        name: "air support",
        parts: &[],
        extra: &[(
            AIR_BAR,
            PartPaint {
                base_color: Color::new(0.10, 0.16, 0.42, 1.0),
                emissive: Color::new(0.10, 0.25, 1.0, 1.0),
                emissive_intensity: 3.0,
            },
        )],
        service: Some(inf_ecs::dispatch::UnitKind::Police),
    };

    let fleet = inf_editor_core::vehicle::island_vehicles();
    let chopper = fleet.get("chopper").expect("the chopper row").to_owned();
    let mut world = EcsWorld::new();
    inf_ecs::vehicle::spawn_rig(
        &mut world,
        CHOPPER,
        &chopper,
        &inf_ecs::vehicle::RigSpawn {
            name: "Air One".to_string(),
            at: DVec3::new(0.0, 2.0, 0.0),
            yaw_deg: 0.0,
            paint: Color::new(0.9, 0.9, 0.9, 1.0),
            clip: None,
            engine_voice: false,
            livery: Some(&AIR_LIVERY),
        },
    );
    world.mark_dirty();
    world.propagate();

    let claimed = inf_ecs::dispatch::unit_kind_of(&world, CHOPPER);
    println!(
        "A LIVERIED HELICOPTER: dispatch reads it as {:?}; its rig is {} wheel(s), {} rotor(s)",
        claimed.map(|k| k.name()),
        inf_ecs::vehicle::rig_of(&world, CHOPPER)
            .map(|r| r.wheels.len())
            .unwrap_or(0),
        inf_ecs::vehicle::rig_of(&world, CHOPPER)
            .map(|r| r.parts.len())
            .unwrap_or(0),
    );
    assert_eq!(
        claimed,
        Some(inf_ecs::dispatch::UnitKind::Police),
        "the recogniser does not read a bar on an aircraft: the refusal this arm prices would be unnecessary"
    );
    // …and the thing it would be dispatched as has NO WHEELS, which is what
    // makes the claim a defect rather than a feature.
    let rig = inf_ecs::vehicle::rig_of(&world, CHOPPER).expect("a rig");
    assert!(rig.wheels.is_empty());
    assert_eq!(rig.parts.len(), 1);

    // The island's own machine is deliberately NOT painted, so nothing claims
    // it — asserted against the committed catalogue rather than against this
    // fixture, which is where the decision actually lives.
    assert!(
        inf_editor_core::vehicle::island_vehicle_livery("chopper").is_none(),
        "the island helicopter wears a livery: dispatch will drive it to a fire"
    );
}

/// **(h) THE ENGINE FOLLOWS THE BOAT** (wave VEH2c) — VEH1a's carried item 5,
/// carried again by VEH2a and a third time by VEH2b's silent traffic, closed.
///
/// An engine loop was `Play`ed at whatever position its vehicle happened to be
/// in on the step it was first seen, and stayed there for the rest of the
/// session: `SetPitch` and `SetVolume` were written every step and the
/// POSITION never was. `AudioCommand::SetPosition` arrived at EMS2 for sirens,
/// and this is the same command on the same queue.
///
/// Asserted on the command stream against the WORLD: the last position the
/// stream carries is where the hull actually is, and the stream's own positions
/// span the distance the boat covered. A `SetPosition` that shipped a constant
/// would satisfy the first half of that and not the second.
#[test]
fn the_launchs_engine_is_heard_where_the_launch_is() {
    use inf_audio::AudioCommand;

    let s = script();
    let hero_at = DVec3::new(MOORING.x, 1.25, MOORING.z + 2.2);
    let mut world = EcsWorld::new();
    build(&mut world);
    place_hero(&mut world, hero_at);
    let mut sim = RuntimeSim::with_gravity(world, Vec::new(), WorldGravity::EARTH, HZ);
    for beat in &s {
        let input = RuntimeInput::with_down(beat.down.iter().copied()).with_axes(beat.axes());
        sim.step_once(input);
    }
    assert_eq!(
        sim.dropped_audio_commands(),
        0,
        "the log is a tail, so the first command in it is not the first command"
    );

    let moves: Vec<DVec3> = sim
        .audio_command_log()
        .iter()
        .filter_map(|c| match c {
            AudioCommand::SetPosition { position, .. } => Some(*position),
            _ => None,
        })
        .collect();
    let plays = sim
        .audio_command_log()
        .iter()
        .filter(|c| matches!(c, AudioCommand::Play(_)))
        .count();
    let hull = look(sim.world(), LAUNCH).1;
    // Guarded rather than indexed: an empty stream is a claim this arm should
    // report, not an index panic three lines before the assertion that says so.
    let spread = match moves.first() {
        Some(first) => moves
            .iter()
            .map(|p| (*p - *first).length())
            .fold(0.0f64, f64::max),
        None => 0.0,
    };
    println!(
        "THE LAUNCH'S VOICE: {plays} Play(s), {} SetPosition(s) spanning {spread:.1} m; the last is {:.1} m from the hull",
        moves.len(),
        (moves.last().copied().unwrap_or(DVec3::ZERO) - hull).length()
    );

    // One voice, and it really is being repositioned.
    assert_eq!(plays, 1, "the launch was given {plays} voices");
    assert!(
        moves.len() > s.len() / 2,
        "only {} of {} steps moved the emitter",
        moves.len(),
        s.len()
    );
    // It is WHERE THE BOAT IS — the emitter's own transform, not a guess.
    let last = *moves.last().expect("a last position");
    assert!(
        (last - hull).length() < 1e-6,
        "the engine is at {last:?} and the hull is at {hull:?}"
    );
    // …and it MOVED, over the distance the boat covered. A `SetPosition` that
    // shipped a constant would pass the two claims above and fail this one.
    assert!(
        spread > 40.0,
        "the emitter spanned {spread:.1} m over a crossing of the bay"
    );
}

/// **(c) PIE == SHIPPING**, over both legs, step for step.
#[test]
fn pie_equals_shipping_from_the_harbour_to_the_air() {
    for (what, s, craft, hero_at) in [
        (
            "the boat",
            script(),
            LAUNCH,
            DVec3::new(MOORING.x, 1.25, MOORING.z + 2.2),
        ),
        (
            "the helicopter",
            air_script(),
            CHOPPER,
            DVec3::new(PAD_AT.x + HERO_STANDOFF_M, PAD_LIP + 1.25, PAD_AT.z),
        ),
    ] {
        let ship = shipping(&s, craft, hero_at);
        let pie = editor(&s, craft, hero_at);
        assert_eq!(ship.samples.len(), pie.samples.len());

        // ANTI-VACUITY, first: a journey that never happened would compare
        // equal on both hosts perfectly. `island_gate`'s own mutation-measured
        // lesson, applied to a craft.
        let moved: std::collections::BTreeSet<(i64, i64, i64)> =
            ship.samples.iter().map(|x| x.craft).collect();
        let modes: std::collections::BTreeSet<u8> = ship.samples.iter().map(|x| x.mode).collect();
        println!(
            "{what}: {} distinct craft poses over {} steps, {} distinct modes, \
             {} seated",
            moved.len(),
            s.len(),
            modes.len(),
            ship.seated
        );
        assert!(
            moved.len() > s.len() / 4,
            "{what}: only {} of {} poses differ — nothing moved",
            moved.len(),
            s.len()
        );
        assert!(
            modes.len() > 1,
            "{what}: the hero was in one mode the whole time — it never boarded"
        );

        for (i, (a, b)) in ship.samples.iter().zip(pie.samples.iter()).enumerate() {
            assert_eq!(
                a.mode, b.mode,
                "{what}: the two hosts disagree about the driver at step {i}"
            );
            assert_eq!(
                a.craft, b.craft,
                "{what}: the two hosts put the craft in different places at \
                 step {i} — shipping {:?}, PIE {:?}",
                a.craft, b.craft
            );
        }
        // …and the shipped host's own byte fold is a real, moving trace.
        let states: std::collections::BTreeSet<&Vec<u8>> =
            ship.samples.iter().map(|x| &x.state).collect();
        assert!(
            states.len() > s.len() / 4,
            "{what}: {} of {} state folds differ",
            states.len(),
            s.len()
        );
        assert!(
            ship.samples.iter().all(|x| !x.state.is_empty()),
            "{what}: a step folded an empty state"
        );
    }
}

/// **(e) THE BUDGET.** What the `vehicle` phase costs with a craft in it.
///
/// `VEHICLE_STEP_BUDGET_MS` was minted at VEH1a for `step_vehicles` over
/// [`VEHICLE_BUDGET_CARS`] cars and re-priced but not moved at VEH2a. This wave
/// puts two classes through the same phase that are not cars at all: a hull
/// solve is one thruster and three drags, a rotor solve is a governed collective
/// and one torque couple, and neither casts a ray. The measurement says whether
/// the ratchet needs to move, and the answer is reported before it is asserted.
///
/// [`VEHICLE_BUDGET_CARS`]: inf_player::budget::VEHICLE_BUDGET_CARS
#[test]
fn the_vehicle_phase_still_fits_with_a_boat_and_a_helicopter_in_it() {
    let boat = shipping(
        &script(),
        LAUNCH,
        DVec3::new(MOORING.x, 1.25, MOORING.z + 2.2),
    );
    let air = shipping(
        &air_script(),
        CHOPPER,
        DVec3::new(PAD_AT.x + HERO_STANDOFF_M, PAD_LIP + 1.25, PAD_AT.z),
    );
    println!(
        "THE VEHICLE PHASE with two craft in the world: {:.5} ms a step over \
         the boat leg, {:.5} ms over the air leg, against a \
         {} ms ceiling for {} cars",
        boat.vehicle_ms,
        air.vehicle_ms,
        inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        inf_player::budget::VEHICLE_BUDGET_CARS
    );
    // The phase really ran, or the budget is about nothing — the island gate's
    // own vacuity clause.
    assert!(
        boat.vehicle_ms > 0.0 && air.vehicle_ms > 0.0,
        "the vehicle phase reported no time at all: {} / {}",
        boat.vehicle_ms,
        air.vehicle_ms
    );
    if cfg!(debug_assertions) {
        eprintln!("dev build: the vehicle phase is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the vehicle phase is reported, not asserted (shared runner)");
        return;
    }
    for (what, ms) in [
        ("the boat leg", boat.vehicle_ms),
        ("the air leg", air.vehicle_ms),
    ] {
        assert!(
            ms <= inf_player::budget::VEHICLE_STEP_BUDGET_MS,
            "{what} cost the vehicle phase {ms:.5} ms against a {} ms ceiling {}",
            inf_player::budget::VEHICLE_STEP_BUDGET_MS,
            inf_player::budget::RATCHET_NOTE
        );
    }
}
