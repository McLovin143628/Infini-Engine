//! **The dispatch step, against a world** (wave EMS2) — a town, a station with
//! an ambulance in it, somebody on the ground, and the drive between them.
//!
//! The unit arms for the recogniser, the draw, the choosing rule and the trace
//! are in `inf_ecs::dispatch`, where they need no rapier. This file exists for
//! the four claims that are only true of a *world*:
//!
//! * an emergency vehicle parked beside a block is **owned** by it — the edge
//!   EMS1 left out, recovered from what the livery left in the level rather than
//!   from a recipe a shipped player does not have;
//! * an incident opens, a unit is chosen, and the unit **actually drives** —
//!   covering metres, because a driver's stick was written by
//!   `inf_ecs::traffic::drive_intent` and turned into controls by the same
//!   `VehicleControls::from_intent` a player's stick goes through;
//! * it arrives, works the scene, and goes **home**, and the incident carries
//!   the response time;
//! * and a level with no emergency vehicle in it never gets a `DispatchRes` at
//!   all, so the fold stays empty and every trace committed before this wave is
//!   byte-identical.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, PcgVolume, ResidentSlot, RigidBody3D, SlotRole,
    StreamingSource, Transform,
};
use inf_ecs::dispatch::{self, IncidentState, UnitKind, UnitState};
use inf_ecs::math::{Color, Vec2d, Vec3d};
use inf_ecs::vehicle::{BodyPart, Livery, PartPaint, RigSpawn, VehicleDef};
use inf_ecs::weapon::Health;
use inf_ecs::EcsWorld;
use inf_physics::d3::PhysicsBridge3D;

const DT: f64 = 1.0 / 60.0;

/// A 3×3 grid of 80 m blocks on a 100 m pitch — two 20 m streets each way, the
/// shape `inf_editor_core::settlement` plans for a city and the fixture
/// `traffic_3d` already drives.
const PITCH: f64 = 100.0;
const STREET: f64 = 20.0;

const HERO: Uuid = Uuid::from_u128(0x0E52_1001);
const GROUND: Uuid = Uuid::from_u128(0x0E52_1002);
const AMBULANCE: Uuid = Uuid::from_u128(0x0E52_1003);
const CRUISER: Uuid = Uuid::from_u128(0x0E52_1004);
const PATIENT: Uuid = Uuid::from_u128(0x0E52_1005);
const APPLIANCE: Uuid = Uuid::from_u128(0x0E52_1006);

/// A red beacon, EMS1's own `BEACON_RED` numbers — the emissive is over 1 or
/// the HDR path never sees it, which is exactly what the recogniser keys on.
static BAR: BodyPart = BodyPart {
    name: "light_bar",
    centre: Vec3d::new(0.0, 1.02, 0.0),
    half: Vec3d::new(0.5, 0.06, 0.18),
    primitive: inf_ecs::components::Primitive::Cube,
};
static RED: PartPaint = PartPaint {
    base_color: Color::new(0.9, 0.1, 0.1, 1.0),
    emissive: Color::new(1.0, 0.16, 0.12, 1.0),
    emissive_intensity: 3.0,
};
static BLUE: PartPaint = PartPaint {
    base_color: Color::new(0.9, 0.9, 0.95, 1.0),
    emissive: Color::new(0.15, 0.35, 1.0, 1.0),
    emissive_intensity: 3.0,
};
static AMBULANCE_LIVERY: Livery = Livery {
    name: "ambulance",
    parts: &[],
    extra: &[(BAR, RED)],
    service: Some(UnitKind::Ambulance),
};
static CRUISER_LIVERY: Livery = Livery {
    name: "cruiser",
    parts: &[],
    extra: &[(BAR, BLUE)],
    service: Some(UnitKind::Police),
};
static ENGINE_LIVERY: Livery = Livery {
    name: "engine",
    parts: &[],
    extra: &[(BAR, RED)],
    service: Some(UnitKind::Fire),
};

fn blocks(world: &mut EcsWorld, cols: i32, rows: i32) {
    let half = (PITCH - STREET) * 0.5;
    for row in 0..rows {
        for col in 0..cols {
            let c = DVec2::new(f64::from(col) * PITCH, f64::from(row) * PITCH);
            let guid = Uuid::from_u64_pair(0x0E52, (row as u64) << 32 | col as u64);
            let e = world.spawn_with_guid(guid, "block", None);
            world.world_mut().entity_mut(e).insert(Transform {
                translation: Vec3d::new(c.x, 0.0, c.y),
                rotation: Vec3d::ZERO,
                scale: Vec3d::ONE,
            });
            let mut v = PcgVolume {
                extent: Vec2d::new(half, half),
                ..Default::default()
            };
            v.residents = vec![ResidentSlot {
                role: SlotRole::Home,
                at: DVec3::new(c.x, 0.0, c.y),
                room: 0,
                building: 0,
                floor: 0,
                index: 0,
                node: 0,
                posture: inf_ecs::components::SlotPosture::Stand,
                shift: inf_ecs::components::SlotShift::Day,
                face: DVec3::ZERO,
            }];
            world.world_mut().entity_mut(e).insert(v);
        }
    }
}

/// A big static floor, so a `Full` unit's four rays land on something.
fn ground(world: &mut EcsWorld) {
    let e = world.spawn_with_guid(GROUND, "Ground", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(100.0, -0.5, 100.0);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(400.0, 0.5, 400.0),
            ..Default::default()
        },
    ));
}

fn hero(world: &mut EcsWorld, at: DVec3) {
    let e = match world.entity_of(HERO) {
        Some(e) => e,
        None => {
            let e = world.spawn_with_guid(HERO, "Hero", None);
            world
                .world_mut()
                .entity_mut(e)
                .insert(StreamingSource { radius_m: 1024.0 });
            e
        }
    };
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
        t.translation = Vec3d::from_dvec3(at);
    } else {
        world
            .world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(at));
    }
    world.propagate();
}

/// An ambulance-shaped rig, parked at `at` — the same door EMS1's generator and
/// the traffic both build a car through.
/// What silhouette a fixture unit is built at.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// The default test saloon — a cruiser.
    Sedan,
    /// A box van — an ambulance.
    Van,
    /// A 7.8 m appliance, past `APPLIANCE_HALF_LENGTH_M`.
    Appliance,
}

fn park_unit(world: &mut EcsWorld, guid: Uuid, at: DVec3, livery: &'static Livery, shape: Shape) {
    let mut def = VehicleDef::default();
    if shape == Shape::Van {
        def.body = inf_ecs::vehicle::VehicleBody::Van;
        def.half_extents = Vec3d::new(1.0, 0.9, 2.4);
        def.half_track_m = 0.88;
        def.half_wheelbase_m = 1.85;
    }
    if shape == Shape::Appliance {
        def.body = inf_ecs::vehicle::VehicleBody::Truck;
        def.half_extents = Vec3d::new(1.05, 1.1, 3.9);
        def.half_track_m = 0.92;
        def.half_wheelbase_m = 2.6;
    }
    inf_ecs::traffic::size_the_suspension(&mut def);
    // **AT ITS OWN RESTING HEIGHT, and this cost the fixture an afternoon.**
    // `size_the_suspension` derives `wheel_drop_m` from the hull, so a body
    // dropped at an eyeballed `y` starts with its struts past their own travel:
    // the chassis rides on its bump stops, the tyres see a fraction of the load,
    // and full throttle produced **0.078 m/s** — which reads exactly like a
    // dispatcher that never wrote a stick. It is `size_the_suspension`'s own
    // documented failure ("the van drives on its BELLY"), met by a test fixture
    // instead of by a catalogue row.
    let sag = def.class.travel_m * inf_ecs::traffic::STATIC_SAG_FRAC;
    let rest_y = -def.wheel_drop_m + def.wheel_radius_m - sag;
    inf_ecs::vehicle::spawn_rig_at(
        world,
        guid,
        &def,
        &RigSpawn {
            name: "Unit".to_string(),
            at: DVec3::new(at.x, rest_y, at.z),
            yaw_deg: 0.0,
            paint: Color::new(0.35, 0.36, 0.38, 1.0),
            clip: None,
            engine_voice: false,
            livery: Some(livery),
        },
        true,
    );
}

struct Town {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Town {
    fn new() -> Self {
        let mut world = EcsWorld::new();
        blocks(&mut world, 3, 3);
        ground(&mut world);
        // Parked on the apron of block (0,0), well inside `STATION_CLAIM_M`.
        park_unit(
            &mut world,
            AMBULANCE,
            DVec3::new(-46.0, 0.0, 0.0),
            &AMBULANCE_LIVERY,
            Shape::Van,
        );
        park_unit(
            &mut world,
            CRUISER,
            DVec3::new(-46.0, 0.0, 12.0),
            &CRUISER_LIVERY,
            Shape::Sedan,
        );
        park_unit(
            &mut world,
            APPLIANCE,
            DVec3::new(-46.0, 0.0, 26.0),
            &ENGINE_LIVERY,
            Shape::Appliance,
        );
        hero(&mut world, DVec3::new(50.0, 0.0, 50.0));
        world.mark_dirty();
        world.propagate();
        let mut town = Self {
            world,
            bridge: PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0)),
        };
        town.bridge
            .sync_from_world_sim(&town.world, &Default::default(), &Default::default());
        town
    }

    /// One fixed step, in the phase order both hosts run: traffic, dispatch,
    /// sync, character move, vehicles, solve, write back.
    fn step(&mut self) -> inf_physics::d3::DispatchStats {
        inf_physics::d3::traffic::step_traffic(&mut self.world, &mut self.bridge, DT);
        let stats = inf_physics::d3::dispatch::step_dispatch(&mut self.world, &mut self.bridge, DT);
        self.bridge
            .sync_from_world_sim(&self.world, &Default::default(), &Default::default());
        inf_physics::d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
        inf_physics::d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
        self.bridge.step(DT);
        self.bridge.write_back_into(&mut self.world);
        self.world.propagate();
        stats
    }

    fn steps(&mut self, n: usize) -> Vec<inf_physics::d3::DispatchStats> {
        (0..n).map(|_| self.step()).collect()
    }

    fn at(&self, guid: Uuid) -> DVec3 {
        self.world
            .entity_of(guid)
            .and_then(|e| self.world.world().get::<Transform>(e))
            .map(|t| t.translation.to_dvec3())
            .expect("the entity is in the world")
    }
}

/// Put somebody on the ground at `at` — WPN1's `Downed` latch, which is the
/// medical feed.
fn collapse(world: &mut EcsWorld, guid: Uuid, at: DVec3) {
    let e = world.spawn_with_guid(guid, "Patient", None);
    world.world_mut().entity_mut(e).insert((
        Transform::from_translation(at),
        Health {
            dead: true,
            ..Default::default()
        },
    ));
    inf_ecs::weapon::mark_downed(world, guid);
    world.mark_dirty();
    world.propagate();
}

// ── (a) the ownership edge ──────────────────────────────────────────────────

/// **A PARKED VEHICLE IS OWNED BY THE BLOCK IT IS PARKED AT** — the edge EMS1
/// left out, and the recogniser that finds it in a *document* rather than in a
/// recipe.
///
/// Falsifies four ways: an ambulance is recognised as an ambulance and a cruiser
/// as police (so the colour channel is read), both are owned by the same block
/// (so the claim edge works), the ordinary parked traffic on the same streets is
/// **not** in the fleet (so a light bar is doing the discriminating and not
/// "is a vehicle"), and the derivation is a cache.
#[test]
fn a_parked_unit_is_owned_by_the_block_it_is_parked_at() {
    let mut town = Town::new();
    town.steps(20);

    let fleet = dispatch::fleet_of(&town.world).expect("a fleet");
    println!(
        "EMS2 fleet: {} unit(s), {} derivation(s)",
        fleet.units.len(),
        fleet.derivations
    );
    for (g, u) in &fleet.units {
        println!("  {g} -> {} at station {}", u.kind.name(), u.station);
    }
    assert_eq!(
        fleet.derivations, 1,
        "the fleet derivation is not a cache — it ran {} times over 20 settled \
         steps",
        fleet.derivations
    );
    assert_eq!(
        fleet.units.len(),
        3,
        "the fleet holds {} unit(s) against the three parked — a town whose kerbs \
         are full of traffic must contribute NONE of it, or the light bar is \
         not doing the discriminating",
        fleet.units.len()
    );
    assert_eq!(fleet.units[&AMBULANCE].kind, UnitKind::Ambulance);
    assert_eq!(fleet.units[&CRUISER].kind, UnitKind::Police);
    assert_eq!(
        fleet.units[&APPLIANCE].kind,
        UnitKind::Fire,
        "a 7.8 m red-barred vehicle is not an appliance — the length rule is \
         `APPLIANCE_HALF_LENGTH_M` and it is the one thing separating a fire \
         engine from an ambulance"
    );
    assert_eq!(
        fleet.units[&AMBULANCE].station, fleet.units[&CRUISER].station,
        "two vehicles on one apron belong to two different stations"
    );

    // …and the traffic on those same streets is not a fleet. Armed: a town with
    // no traffic in it would satisfy this by having nothing to exclude.
    let cars = inf_physics::d3::traffic::records(&town.world).len();
    println!("EMS2: {cars} traffic car(s), none of them a unit");
    assert!(
        cars > 10,
        "only {cars} traffic cars — this arm is then about an empty street"
    );
    for chassis in inf_physics::d3::traffic::records(&town.world).keys() {
        assert!(
            !fleet.units.contains_key(chassis),
            "a traffic car was recruited into the fleet"
        );
    }
}

// ── (b) the whole run ───────────────────────────────────────────────────────

/// **SOMEBODY COLLAPSES AND AN AMBULANCE COMES, WORKS AND GOES HOME.**
///
/// The headline. Every stage is asserted as a *world* fact rather than as a
/// counter: the incident opens, the ambulance (not the cruiser) is chosen, the
/// chassis covers metres, the crew ends up standing beside the patient, the
/// incident closes with a response time, and the vehicle is back on its own
/// apron with nobody in it.
#[test]
fn a_collapse_brings_the_ambulance_and_sends_it_home_again() {
    let mut town = Town::new();
    town.steps(10);
    let home = town.at(AMBULANCE);

    // Somebody goes down two blocks away, on the far side of the grid.
    collapse(&mut town.world, PATIENT, DVec3::new(150.0, 0.0, 50.0));

    let mut opened_at = None;
    let mut assigned_at = None;
    let mut arrived_at = None;
    let mut resolved_at = None;
    let mut home_at = None;
    let mut steered = 0usize;
    let mut hot = 0usize;
    // Stopped the step the unit is home, so the arms below read the world at the
    // end of ONE run rather than in the middle of a second: the patient is still
    // on the ground (this wave's carried item — see the ledger), so a town left
    // running long enough calls another ambulance for the same body.
    for i in 0..6000 {
        let s = town.step();
        steered += s.steered;
        hot += usize::from(s.running_hot > 0);
        if s.opened > 0 && opened_at.is_none() {
            opened_at = Some(i);
        }
        if s.assigned > 0 && assigned_at.is_none() {
            assigned_at = Some(i);
        }
        if s.arrived > 0 && arrived_at.is_none() {
            arrived_at = Some(i);
        }
        if s.resolved > 0 && resolved_at.is_none() {
            resolved_at = Some(i);
        }
        if s.returned > 0 {
            home_at = Some(i);
            break;
        }
    }
    println!(
        "EMS2 run: opened {opened_at:?}, assigned {assigned_at:?}, arrived \
         {arrived_at:?}, resolved {resolved_at:?}, home {home_at:?}; {steered} \
         stick(s), {hot} hot step(s)"
    );
    assert!(opened_at.is_some(), "the collapse opened no incident");
    assert!(assigned_at.is_some(), "nobody was sent");
    assert!(
        steered > 100,
        "{steered} stick(s) written over the whole run — a dispatcher that \
         assigned a unit and never steered it is a car that does not move"
    );
    assert!(arrived_at.is_some(), "the ambulance never got there");
    assert!(resolved_at.is_some(), "the scene was never worked");
    assert!(home_at.is_some(), "the ambulance never went home");

    let res = dispatch::dispatch_of(&town.world).expect("a dispatcher");
    // **The right unit went.** The cruiser is nearer to nothing and is the wrong
    // service; a dispatcher that ignored the service would have sent it.
    let incident = res
        .incidents
        .values()
        .find(|i| {
            matches!(i.kind, inf_ecs::dispatch::IncidentKind::Medical { .. })
                && i.state == IncidentState::Resolved
        })
        .expect("the medical incident is still in the ledger");
    assert_eq!(
        incident.unit,
        Some(AMBULANCE),
        "the wrong unit was sent to a medical call"
    );
    assert_eq!(incident.state, IncidentState::Resolved);
    let took = incident
        .response_steps()
        .expect("a resolved incident timed");
    println!(
        "EMS2 response: {took} step(s) = {:.2} s over {:.1} m",
        took as f64 * DT,
        (DVec3::new(150.0, 0.0, 50.0) - home).length()
    );
    assert!(took > 0, "an incident that resolved on the step it opened");
    assert_eq!(res.runs[&CRUISER].state, UnitState::InStation);
    assert_eq!(res.runs[&AMBULANCE].state, UnitState::InStation);

    // …and the world agrees: the ambulance is back on its apron, and its crew is
    // gone from the street.
    let back = town.at(AMBULANCE);
    println!("EMS2: home {home:?} -> back {back:?}");
    assert!(
        (back - home).length() < dispatch::HOME_M,
        "the ambulance parked {:.1} m from its own space",
        (back - home).length()
    );
    let crew = dispatch::crew_guid(AMBULANCE);
    assert!(
        town.world.entity_of(crew).is_none(),
        "the crew is still standing in the street after the unit went home"
    );
    assert!(
        !dispatch::is_responder(&town.world, crew),
        "a crew back at the station is still on duty, so it can never be \
         frightened again"
    );
}

// ── (c) it really drove ─────────────────────────────────────────────────────

/// **THE UNIT DRIVES; IT IS NOT TELEPORTED.**
///
/// The arm above could be satisfied by a dispatcher that snapped the chassis to
/// the scene, which is the shape a "responding" system most easily degenerates
/// into. This one measures the *middle*: at the halfway point of the drive the
/// ambulance is neither at its station nor at the patient, it is moving, and
/// there is a crew member in its seat.
#[test]
fn a_responding_unit_covers_the_ground_between() {
    let mut town = Town::new();
    town.steps(10);
    let home = town.at(AMBULANCE);
    let scene = DVec3::new(150.0, 0.0, 50.0);
    collapse(&mut town.world, PATIENT, scene);

    let mut best_middle = 0.0f64;
    let mut ever_seated = false;
    for _ in 0..1200 {
        let s = town.step();
        if s.arrived > 0 {
            break;
        }
        let at = town.at(AMBULANCE);
        let from_home = (at - home).length();
        let to_scene = (at - scene).length();
        best_middle = best_middle.max(from_home.min(to_scene));
        let crew = dispatch::crew_guid(AMBULANCE);
        if let Some(e) = town.world.entity_of(crew) {
            if let Some(cm) = town
                .world
                .world()
                .get::<inf_ecs::components::CharacterMovement>(e)
            {
                ever_seated |= cm.runtime.seat.vehicle == AMBULANCE;
            }
        }
    }
    println!("EMS2 drive: furthest-from-both {best_middle:.1} m");
    assert!(
        best_middle > 25.0,
        "the ambulance was never more than {best_middle:.1} m from BOTH ends — \
         it did not travel, it jumped"
    );
    assert!(ever_seated, "nobody was ever in the ambulance");
}

// ── (d) the siren and the bar ───────────────────────────────────────────────

/// **A RESPONDING UNIT SOUNDS ITS SIREN AND FLASHES ITS BAR — AND STOPS DOING
/// BOTH WHEN IT GETS HOME.**
///
/// Five claims, each killing a different degeneration:
///
/// 1. exactly **one** `Start` over the whole run — a siren re-started every step
///    is a `Play` that restarts the loop sixty times a second, which is silence
///    with a full command log;
/// 2. the `Move`s arrive on the **cadence**, not every step: the ring
///    arithmetic in `SIREN_POSITION_PERIOD` is the whole reason
///    `AudioCommand::SetPosition` was safe to add, and a per-step emit would
///    reintroduce the VEH2a eviction it exists to avoid;
/// 3. the emitter actually **moves** — a `SetPosition` stream that carried one
///    position would be a siren nailed to the station;
/// 4. exactly one `Stop`, and the unit is silent afterwards;
/// 5. the bar's authored intensity is **given back**. A pin with no release is a
///    leak with a deadline, and this one has a picture: an ambulance that comes
///    home with a black light on its roof.
#[test]
fn a_responding_unit_sounds_its_siren_and_gives_its_bar_back() {
    let mut town = Town::new();
    town.steps(10);
    let bar = dispatch::light_bar_of(&town.world, AMBULANCE).expect("the ambulance has a bar");
    let authored = bar_intensity(&town.world, bar);
    println!("EMS2 bar {bar} authored at {authored}");
    assert!(
        authored > 1.0,
        "the fixture's bar is not bloomed ({authored}) — nothing below is then \
         about a light"
    );

    collapse(&mut town.world, PATIENT, DVec3::new(150.0, 0.0, 50.0));
    let (mut starts, mut moves, mut stops) = (0usize, 0usize, 0usize);
    let mut positions: Vec<DVec3> = Vec::new();
    let mut move_steps: Vec<u64> = Vec::new();
    let mut lit: Vec<f32> = Vec::new();
    let mut hot_steps = 0usize;
    for _ in 0..6000 {
        let s = town.step();
        let res = dispatch::dispatch_of(&town.world).expect("a dispatcher");
        let step = res.steps.saturating_sub(1);
        for cue in &res.sirens {
            match *cue {
                dispatch::SirenCue::Start { source, at } => {
                    assert_eq!(source, dispatch::siren_guid(AMBULANCE));
                    starts += 1;
                    positions.push(at);
                }
                dispatch::SirenCue::Move { at, .. } => {
                    moves += 1;
                    move_steps.push(step);
                    positions.push(at);
                }
                dispatch::SirenCue::Stop { .. } => stops += 1,
            }
        }
        if !res.flashes.is_empty() {
            hot_steps += 1;
            // The host fence is what writes the material; this arm drives the
            // same rule through the same door so the cue's own claim is tested
            // without a renderer.
            let f = res.flashes[0];
            assert_eq!(f.bar, bar);
            assert_eq!(
                f.base_intensity, authored,
                "the flash is pinned to {} rather than to the authored {authored} \
                 — the pin is reading the LIVE value and will compound",
                f.base_intensity
            );
            lit.push(f.base_intensity);
        }
        if s.returned > 0 {
            break;
        }
    }
    println!(
        "EMS2 siren: {starts} start(s), {moves} move(s), {stops} stop(s) over \
         {hot_steps} hot step(s)"
    );
    assert_eq!(starts, 1, "the siren was started {starts} time(s)");
    assert_eq!(stops, 1, "the siren was stopped {stops} time(s)");
    assert!(hot_steps > 100, "only {hot_steps} hot step(s)");
    assert!(!lit.is_empty(), "the bar never flashed");

    // (2) THE CADENCE. Every `Move` lands on a multiple of the period, and there
    //     are about `hot / period` of them — a per-step emit would read `hot`.
    for at in &move_steps {
        assert_eq!(
            at % dispatch::SIREN_POSITION_PERIOD,
            0,
            "a siren moved on step {at}, which is not on the cadence"
        );
    }
    let want = hot_steps / dispatch::SIREN_POSITION_PERIOD as usize;
    println!("EMS2 cadence: {moves} move(s) against ~{want} expected");
    assert!(
        moves > want / 2 && moves <= want + 2,
        "{moves} `SetPosition`(s) over {hot_steps} hot steps at a period of {} \
         — a per-step emit would read about {hot_steps}",
        dispatch::SIREN_POSITION_PERIOD
    );

    // (3) …and the emitter travelled.
    let spread = positions
        .iter()
        .fold(0.0f64, |m, p| m.max((*p - positions[0]).length()));
    println!("EMS2 siren travelled {spread:.1} m");
    assert!(
        spread > 50.0,
        "the siren's emitter moved {spread:.1} m — a `SetPosition` stream that \
         carries one position is a siren nailed to the station"
    );

    // (5) THE RELEASE.
    let back = bar_intensity(&town.world, bar);
    println!("EMS2 bar back at {back} against an authored {authored}");
    assert_eq!(
        back, authored,
        "the bar came home at {back} against the {authored} the level authored \
         — the pin was never released"
    );
    let res = dispatch::dispatch_of(&town.world).expect("a dispatcher");
    assert!(
        res.bars.is_empty(),
        "a released pin is still held: {:?}",
        res.bars
    );
    assert!(
        res.siren_on.is_empty(),
        "a unit in station still has a siren on"
    );
}

/// A light bar's live emissive intensity.
fn bar_intensity(world: &EcsWorld, bar: Uuid) -> f32 {
    world
        .entity_of(bar)
        .and_then(|e| world.world().get::<inf_ecs::components::Material>(e))
        .map(|m| m.emissive_intensity)
        .expect("the bar is drawn")
}

// ── (e) the scene ───────────────────────────────────────────────────────────

/// **A BUILDING BURNS, THE APPLIANCE COMES, IT SMOKES, IT IS PUT OUT — AND THE
/// SMOKE LEAVES NOTHING BEHIND.**
///
/// Six claims, and the last two are the ones a screenshot cannot make:
///
/// 1. the right unit goes — the appliance and not the ambulance parked beside
///    it;
/// 2. the fire's **intensity** falls while a crew is on it, so a bigger fire
///    takes longer rather than every fire taking the same five seconds;
/// 3. smoke exists while it burns — real `Sprite` entities, spawned by the sim;
/// 4. the puffs are **bounded and reaped**: they never exceed `MAX_PUFFS` and
///    an old one is despawned rather than left rising for ever;
/// 5. a hose is drawn — `extinguish_beams` answers a segment while the crew
///    works and nothing when it does not;
/// 6. and after `clear_dispatch` **no smoke entity is left in the world**,
///    which is the P21 law applied to a sprite: a puff in the author's document
///    is a row in the Outliner that no Outliner row put there.
#[test]
fn a_fire_brings_the_appliance_smokes_and_leaves_nothing_behind() {
    let mut town = Town::new();
    town.steps(20);

    let at = DVec3::new(100.0, 0.0, 100.0);
    let fire = inf_physics::d3::dispatch::report_incident(
        &mut town.world,
        inf_ecs::dispatch::IncidentKind::Fire {
            building: Uuid::from_u128(0x0E52_2001),
            intensity: 1.0,
        },
        at,
    )
    .expect("the staging door opened a fire");

    let mut peak_puffs = 0usize;
    let mut ever_beamed = false;
    let mut min_intensity = 1.0f64;
    let mut resolved = false;
    let mut spawned_total = 0usize;
    let mut seen: std::collections::BTreeSet<Uuid> = Default::default();
    for _ in 0..6000 {
        let s = town.step();
        let res = dispatch::dispatch_of(&town.world).expect("a dispatcher");
        peak_puffs = peak_puffs.max(res.puffs.len());
        for g in res.puffs.keys() {
            if seen.insert(*g) {
                spawned_total += 1;
            }
        }
        if let Some(i) = res.incidents.get(&fire) {
            if let inf_ecs::dispatch::IncidentKind::Fire { intensity, .. } = i.kind {
                min_intensity = min_intensity.min(intensity);
            }
            resolved |= i.state == IncidentState::Resolved;
        }
        ever_beamed |= !inf_physics::d3::dispatch::extinguish_beams(&town.world).is_empty();
        if s.returned > 0 {
            break;
        }
    }
    let res = dispatch::dispatch_of(&town.world).expect("a dispatcher");
    let incident = res.incidents.get(&fire).expect("the fire is in the ledger");
    println!(
        "EMS2 fire: unit {:?}; intensity fell to {min_intensity:.3}; {peak_puffs} \
         puff(s) at peak, {spawned_total} over the whole fire; beam {ever_beamed}",
        incident.unit.map(|u| if u == APPLIANCE {
            "appliance"
        } else {
            "the WRONG one"
        })
    );
    assert_eq!(
        incident.unit,
        Some(APPLIANCE),
        "the wrong unit was sent to a fire"
    );
    assert!(resolved, "the fire was never put out");
    assert!(
        min_intensity <= 0.0,
        "the fire resolved at intensity {min_intensity:.3} — it was closed by a \
         clock rather than put out, so a bigger fire would take the same time"
    );
    assert!(peak_puffs > 0, "a burning building made no smoke");
    assert!(
        peak_puffs <= dispatch::MAX_PUFFS,
        "{peak_puffs} puffs against a ceiling of {}",
        dispatch::MAX_PUFFS
    );
    assert!(
        spawned_total > peak_puffs,
        "{spawned_total} puff(s) were ever spawned and {peak_puffs} were alive \
         at once — nothing was ever REAPED, so the column grows for the whole \
         session"
    );
    assert!(
        ever_beamed,
        "no extinguish line was ever drawn while a crew worked a fire"
    );
    assert!(
        inf_physics::d3::dispatch::extinguish_beams(&town.world).is_empty(),
        "a hose is still being played on a fire that is out"
    );

    // (6) …and the session leaves nothing behind.
    inf_ecs::dispatch::clear_dispatch(&mut town.world);
    let left = town
        .world
        .world()
        .iter_entities()
        .filter(|e| e.get::<inf_ecs::components::Sprite>().is_some())
        .count();
    assert_eq!(
        left, 0,
        "{left} smoke sprite(s) survived `clear_dispatch` — a puff in the \
         author's document is a row in the Outliner that no Outliner row put \
         there"
    );
}

/// **THE PARAMEDIC KNEELS, AND STANDS UP AGAIN.**
///
/// The posture is written onto the crew body's own `CrowdAgent`, which is where
/// `step_pose_evaluation` reads one from — and nothing else writes it for that
/// body, because a crew member is a `spawn_body` and not a population record.
/// The release is the half that is easy to leave out: a paramedic who drove home
/// on one knee is the posture write with no undo.
///
/// …and the same run proves the repeat-call loop is closed: the patient is still
/// on the ground at the end, and no second ambulance was called.
#[test]
fn a_paramedic_kneels_at_the_patient_and_stands_up_to_leave() {
    let mut town = Town::new();
    town.steps(10);
    collapse(&mut town.world, PATIENT, DVec3::new(150.0, 0.0, 50.0));
    let crew = dispatch::crew_guid(AMBULANCE);

    let mut knelt = 0usize;
    let mut opened = 0usize;
    for _ in 0..6000 {
        let s = town.step();
        opened += s.opened;
        if let Some(e) = town.world.entity_of(crew) {
            if let Some(a) = town.world.world().get::<inf_ecs::crowd::CrowdAgent>(e) {
                if a.posture == inf_ecs::components::SlotPosture::Kneel {
                    knelt += 1;
                }
            }
        }
        if s.returned > 0 {
            break;
        }
    }
    println!("EMS2 kneel: {knelt} step(s) on one knee, {opened} incident(s) opened");
    assert!(
        knelt > 100,
        "the paramedic knelt for {knelt} step(s) — `STABILIZE_S` is {} seconds, \
         so a working crew should be down for hundreds",
        dispatch::STABILIZE_S
    );
    assert!(
        town.world.entity_of(crew).is_none(),
        "the crew is still in the street"
    );

    // The repeat-call loop, closed. The body is still `Downed` — this engine has
    // no stretcher — and it does not call a second ambulance.
    assert!(
        inf_ecs::weapon::is_downed(&town.world, PATIENT),
        "the patient stopped being downed, so this half is about a body that is \
         no longer there rather than about the `treated` set"
    );
    let res = dispatch::dispatch_of(&town.world).expect("a dispatcher");
    assert!(
        res.treated.contains(&PATIENT),
        "the patient was never marked treated"
    );
    assert_eq!(
        opened, 1,
        "{opened} incident(s) were opened for one body — the `treated` set is \
         not stopping the second call"
    );
}

// ── (f) absent costs nothing ────────────────────────────────────────────────

/// **A TOWN WITH NO EMERGENCY VEHICLE IN IT HAS NO DISPATCHER.**
///
/// The property that keeps every trace committed before this wave
/// byte-identical: the fold is empty because the *resource does not exist*, not
/// because it happens to hold nothing.
#[test]
fn a_town_with_no_fleet_never_gets_a_dispatcher() {
    let mut world = EcsWorld::new();
    blocks(&mut world, 3, 3);
    ground(&mut world);
    hero(&mut world, DVec3::new(50.0, 0.0, 50.0));
    world.mark_dirty();
    world.propagate();
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    // Somebody goes down, so this is not "nothing happened" — it is "nothing
    // could be sent".
    collapse(&mut world, PATIENT, DVec3::new(150.0, 0.0, 50.0));
    for _ in 0..60 {
        inf_physics::d3::traffic::step_traffic(&mut world, &mut bridge, DT);
        let s = inf_physics::d3::dispatch::step_dispatch(&mut world, &mut bridge, DT);
        assert_eq!(s, inf_physics::d3::DispatchStats::default());
    }
    assert!(
        dispatch::dispatch_of(&world).is_none(),
        "a town with no fleet grew a dispatcher"
    );
    assert!(
        inf_ecs::dispatch::dispatch_state_bytes(&world).is_empty(),
        "the dispatch trace section is not empty on a level that has no \
         dispatcher — every pre-EMS2 committed hash moves"
    );
}
