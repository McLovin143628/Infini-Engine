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
/// A second cruiser, so the severity ladder has more than one car to ask for.
const CRUISER_B: Uuid = Uuid::from_u128(0x0E52_1007);

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
        Self::with_fire_hall(true)
    }

    /// The same town, optionally **without an appliance** — a settlement with a
    /// police station and a hospital and no fire hall, which is what
    /// `station_fleet` builds for every archetype that is not a `FireHall` and
    /// is therefore the ordinary case rather than a contrivance.
    fn with_fire_hall(fire: bool) -> Self {
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
        if fire {
            park_unit(
                &mut world,
                APPLIANCE,
                DVec3::new(-46.0, 0.0, 26.0),
                &ENGINE_LIVERY,
                Shape::Appliance,
            );
        }
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
    // end of ONE run rather than in the middle of a second. The patient is still
    // on the ground — this wave's carried item, there being no stretcher — but it
    // does **not** call another ambulance: `DispatchRes::treated` closes that
    // loop, and `a_paramedic_kneels_at_the_patient_and_stands_up_to_leave` is
    // where it is measured, past `INCIDENT_KEEP_STEPS` where the other guard has
    // stopped helping. (This comment claimed the loop was open; it was written
    // one commit before `treated` landed and outlived it — EMS2 audit.)
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

    // (6) …and the session leaves nothing behind — NEITHER of the two things
    //     this wave spawns.
    //
    //     The crew half is an EMS2 audit repair. A crew member is an
    //     `inf_ecs::crowd::spawn_body`, which is deliberately not a population
    //     record — so `clear_crowd` walks `CrowdPopulationRes` and never sees
    //     it, and `clear_dispatch`'s first cut despawned the sprites and left
    //     the people. Stopped mid-response, that is a person standing in the
    //     road of the author's document, which is the same sentence the puffs
    //     are refused by.
    inf_physics::d3::dispatch::report_incident(
        &mut town.world,
        inf_ecs::dispatch::IncidentKind::Fire {
            building: Uuid::from_u128(0x0E52_2003),
            intensity: 1.0,
        },
        DVec3::new(0.0, 0.0, 200.0),
    )
    .expect("a second fire, so a unit is OUT when the session is stopped");
    let crew = dispatch::crew_guid(APPLIANCE);
    let mut had_a_body = false;
    for _ in 0..2000 {
        town.step();
        had_a_body |= town.world.entity_of(crew).is_some();
        if had_a_body {
            break;
        }
    }
    assert!(
        had_a_body,
        "no crew body was ever built, so stopping the session below proves \
         nothing about one"
    );

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
    assert!(
        town.world.entity_of(crew).is_none(),
        "a crew member survived `clear_dispatch` — the session was stopped \
         while its unit was out, and the person it built is now a row in the \
         author's Outliner that no Outliner row put there"
    );
    assert!(
        !dispatch::is_responder(&town.world, crew),
        "the duty roster survived the clear"
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

    // ── AND PAST THE LEDGER'S OWN MEMORY (EMS2 audit) ──
    //
    // Everything above is satisfied by the *other* guard. `open_incidents`
    // refuses a body that already has an open medical incident, and this run
    // ends a few hundred steps after the resolve — deep inside
    // `INCIDENT_KEEP_STEPS`, where that guard is still doing all the work. Delete
    // `treated` entirely and every assertion above still passes.
    //
    // What `treated` is *for* is the window after the ledger forgets: the
    // `Downed` latch is permanent, the resolved incident is retired at 3 600
    // steps, and from then on the same body is indistinguishable from a fresh
    // collapse. So the town is run past that, and what is counted is medical
    // incidents **naming this patient** rather than everything the town opened
    // (the nine blocks' own ambient draw is running the whole time and is not
    // what this arm is about).
    let mut ever: std::collections::BTreeSet<Uuid> = Default::default();
    let mut forgotten = false;
    for _ in 0..(dispatch::INCIDENT_KEEP_STEPS + 600) {
        town.step();
        let res = dispatch::dispatch_of(&town.world).expect("a dispatcher");
        let mine: Vec<Uuid> = res
            .incidents
            .iter()
            .filter(|(_, i)| {
                matches!(i.kind, inf_ecs::dispatch::IncidentKind::Medical { npc, .. } if npc == PATIENT)
            })
            .map(|(g, _)| *g)
            .collect();
        forgotten |= mine.is_empty();
        ever.extend(mine);
    }
    println!(
        "EMS2 treated: {} medical incident(s) ever named the patient over {} \
         further steps; the ledger forgot it: {forgotten}",
        ever.len(),
        dispatch::INCIDENT_KEEP_STEPS + 600
    );
    assert!(
        forgotten,
        "the resolved incident was never retired from the ledger, so the run is \
         still inside `INCIDENT_KEEP_STEPS` and the guard being measured is the \
         incidents table rather than `treated`"
    );
    assert!(
        inf_ecs::weapon::is_downed(&town.world, PATIENT),
        "the patient is no longer downed, so the census has stopped offering it"
    );
    assert_eq!(
        ever.len(),
        1,
        "{} medical incident(s) named one patient once the ledger had forgotten \
         the first — the `Downed` latch is permanent, so without `treated` this \
         body calls an ambulance every {} steps for ever",
        ever.len(),
        dispatch::INCIDENT_KEEP_STEPS
    );
}

// ── (f) the two feeds nothing else drives ───────────────────────────────────

/// **THE AMBIENT DRAW REACHES THE DISPATCHER** (EMS2 audit).
///
/// `inf_ecs::dispatch::ambient_draw` has a unit arm that says the *function* is
/// sparse, reproducible and makes both kinds. Nothing said the **feed** works:
/// that `open_incidents` walks the level's blocks, takes the draw on the epoch
/// and mints an incident somebody is sent to. The whole branch could have been
/// dead — an inverted `is_multiple_of`, a `blocks_of` that answered nothing, a
/// walk that never reached `open` — and every other arm in this file and in
/// `ems2_dispatch_gate` would still be green, because all of them stage their
/// incidents through `report_incident`.
///
/// So this one stages **nothing**. The town is left alone and asserted to set
/// itself on fire: the epoch is found from the pure function first, so the arm
/// knows what it is waiting for and fails with a number rather than a timeout.
#[test]
fn the_ambient_draw_reaches_the_dispatcher_with_nothing_staged() {
    // What the pure function says this town will do, and when. The block guids
    // are `blocks`' own, and the dispatcher's epoch is `res.steps /
    // AMBIENT_PERIOD` — its own clock, which starts at 0 on the first step it
    // has a fleet.
    let mut want: Option<(u64, Uuid, inf_ecs::dispatch::IncidentKind)> = None;
    'search: for epoch in 0..16u64 {
        for row in 0..3u64 {
            for col in 0..3u64 {
                let block = Uuid::from_u64_pair(0x0E52, (row << 32) | col);
                if let Some(kind) = dispatch::ambient_draw(block, epoch) {
                    want = Some((epoch, block, kind));
                    break 'search;
                }
            }
        }
    }
    let (epoch, block, kind) = want.expect(
        "nine blocks over sixteen epochs drew nothing at all — `AMBIENT_CHANCE` \
         has been trimmed to nothing or the draw is broken",
    );
    println!(
        "EMS2 ambient feed: block {block} draws a {} at epoch {epoch} (step {})",
        kind.name(),
        epoch * dispatch::AMBIENT_PERIOD
    );

    let mut town = Town::new();
    let mut opened_kind: Option<inf_ecs::dispatch::IncidentKind> = None;
    let mut opened_at: Option<u64> = None;
    // One step past the epoch's own step, because the draw is taken *during*
    // that step and read out after it.
    for _ in 0..=(epoch * dispatch::AMBIENT_PERIOD + 1) {
        let s = town.step();
        if s.opened > 0 && opened_at.is_none() {
            let res = dispatch::dispatch_of(&town.world).expect("a dispatcher");
            opened_at = Some(res.steps);
            opened_kind = res
                .incidents
                .values()
                .find(|i| i.state != IncidentState::Resolved)
                .map(|i| i.kind);
        }
    }
    let res = dispatch::dispatch_of(&town.world).expect("a dispatcher");
    println!(
        "  the town opened {} incident(s) by step {}, {} assigned; first was {:?}",
        res.opened,
        res.steps,
        res.assigned,
        opened_kind.map(|k| k.name())
    );
    assert!(
        res.opened > 0,
        "nothing was staged and the town opened nothing — the ambient feed's \
         branch never reaches `open`, and every other arm in this wave stages \
         through `report_incident` so none of them can see it"
    );
    // …and it is the draw's own answer, not something else: the kind the pure
    // function named, at the block it named.
    let named = res.incidents.values().any(|i| match (i.kind, kind) {
        (
            inf_ecs::dispatch::IncidentKind::Fire { building: a, .. },
            inf_ecs::dispatch::IncidentKind::Fire { building: b, .. },
        ) => a == b,
        (
            inf_ecs::dispatch::IncidentKind::Medical { npc: a, .. },
            inf_ecs::dispatch::IncidentKind::Medical { npc: b, .. },
        ) => a == b,
        _ => false,
    });
    assert!(
        named,
        "the town opened something, but not the {} at block {block} the draw \
         says it should have — the feed is reaching `open` with the wrong \
         subject",
        kind.name()
    );
    // …and somebody was actually sent to it, which is what makes an ambient
    // incident a thing a player experiences rather than a row in a map.
    assert!(
        res.assigned > 0,
        "the town opened an incident it never answered"
    );
}

/// **A WITNESSED SHOT BECOMES A CRIME, A BURST BECOMES ONE CRIME, AND AN ACT IS
/// READ ONCE** (EMS2 audit).
///
/// The third feed, and the one nothing in this wave drove: every crime this tree
/// dispatches to in the gate and in the arms above is staged through
/// `report_incident`, so `open_incidents`' witness branch — the `> seen_act_step`
/// forward read, the `ActKind` → severity mapping, and the `CRIME_MERGE_M`
/// dedupe — was carried entirely by inspection.
///
/// Three claims, and the second and third are the ones that cost a station:
///
/// 1. a `Shot` opens a `Crime` with **severity 1** and a `Killed` opens one with
///    **2**;
/// 2. a second act **inside `CRIME_MERGE_M`** of an open crime does not open a
///    second one — a burst is one emergency, and a dispatcher that opened an
///    incident per round would empty a station into one street corner;
/// 3. the same act is never read twice, however many steps run past it — which
///    is what `seen_act_step` is, and a broken forward read would mint a new
///    crime scene *every step for ever*.
#[test]
fn a_witnessed_shot_is_a_crime_a_burst_is_one_and_an_act_is_read_once() {
    use inf_ecs::witness::{ActKind, WitnessedAct};

    let mut town = Town::new();
    // Past the epoch-0 ambient draw, so what is counted below is this feed's.
    town.steps(30);
    let crimes = |t: &Town| -> Vec<(u8, DVec3)> {
        dispatch::dispatch_of(&t.world)
            .map(|r| {
                r.incidents
                    .values()
                    .filter_map(|i| match i.kind {
                        inf_ecs::dispatch::IncidentKind::Crime { severity } => {
                            Some((severity, i.at))
                        }
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    assert!(
        crimes(&town).is_empty(),
        "the town started with a crime on it"
    );

    let act = |kind: ActKind, at: DVec3, step: u64| WitnessedAct {
        kind,
        actor: HERO,
        at,
        step,
        observers: Vec::new(),
        actor_look: 0,
        actor_vehicle: None,
    };
    let corner = DVec3::new(150.0, 0.0, 50.0);

    // (1) a shot.
    inf_ecs::witness::record_act(&mut town.world, act(ActKind::Shot, corner, 100));
    town.step();
    let after_one = crimes(&town);
    println!("EMS2 crime feed: after one shot {after_one:?}");
    assert_eq!(
        after_one.len(),
        1,
        "a witnessed shot opened {} crime(s)",
        after_one.len()
    );
    assert_eq!(after_one[0].0, 1, "a `Shot` is severity 1");

    // (2) the rest of the burst, eight metres along — one scene, not nine.
    for (n, step) in (101..109u64).enumerate() {
        inf_ecs::witness::record_act(
            &mut town.world,
            act(ActKind::Shot, corner + DVec3::X * (n as f64 + 1.0), step),
        );
    }
    town.step();
    let after_burst = crimes(&town);
    println!("  after eight more rounds {} crime(s)", after_burst.len());
    assert_eq!(
        after_burst.len(),
        1,
        "a burst of nine rounds inside {} m opened {} crime scene(s) — a \
         dispatcher that answers each round empties a station into one corner",
        inf_physics::d3::dispatch::CRIME_MERGE_M,
        after_burst.len()
    );

    // (3) a death, well outside the merge radius — a second scene, severity 2.
    let across_town = corner + DVec3::X * 200.0;
    inf_ecs::witness::record_act(&mut town.world, act(ActKind::Killed, across_town, 120));
    town.step();
    let after_kill = crimes(&town);
    println!("  after a death 200 m away {} crime(s)", after_kill.len());
    assert_eq!(
        after_kill.len(),
        2,
        "a death 200 m from an open crime scene did not open its own"
    );
    assert!(
        after_kill.iter().any(|(sev, _)| *sev == 2),
        "a `Killed` did not map to severity 2: {after_kill:?}"
    );

    // (4) THE FORWARD READ. Nothing new is recorded and the log still holds all
    //     ten acts; a `>=` where the code says `>`, or a `seen_act_step` that is
    //     never written, mints a new crime scene on every one of these steps.
    let before = crimes(&town).len();
    town.steps(120);
    let after = crimes(&town);
    println!("  120 quiet steps later {} crime(s)", after.len());
    assert!(
        after.len() <= before,
        "the log's ten acts were re-read: {before} crime(s) became {} over 120 \
         quiet steps — `seen_act_step` is not advancing",
        after.len()
    );
    assert!(
        !inf_ecs::witness::witnessed(&town.world).is_empty(),
        "the log emptied itself, so the forward read above is a claim about an \
         empty ring"
    );
}

/// **A SERIOUS FILE PULLS A SECOND CAR AND A PETTY ONE DOES NOT** (wave EMS3
/// audit) — the severity ladder, as a dispatcher and not as arithmetic.
///
/// # Nothing in the tree exercised this
///
/// EMS3 minted `Response::{MultiUnit, Swat}`, `wanted_units`, `units_on` and a
/// new disjunct in `assign`'s pending filter, and its ledger claims *"until this
/// line nothing in the engine could ever ask for three cars at once"*. At
/// `dda8d836` a grep for `Swat`, `MultiUnit`, `wanted_units` or `units_on` over
/// every test file in the repository returned **nothing**: the only coverage was
/// `Response::for_heat`'s own table, which is a pure function, and the gate's
/// single carjack, which is `Patrol`. The dispatcher half — the branch that
/// re-pends an incident somebody is already at — ran in no test at all.
///
/// So: one town, two cruisers, and the same crime at two heats.
///
/// * a `Killed` (heat 3 → `MultiUnit`, `units() == 2`) gets **two** cars;
/// * a `Carjack` (heat 1 → `Patrol`) gets **one**, which is the falsifier: if
///   the pending filter simply stopped excluding answered incidents, both rows
///   would read two.
#[test]
fn a_serious_file_pulls_a_second_car_and_a_petty_one_does_not() {
    use inf_ecs::witness::{ActKind, WitnessedAct};

    /// The most cars ever on one crime scene over `steps` fixed steps, and the
    /// rung the file reached — one script, run at two severities.
    fn most_cars_on_the_search(kind: ActKind, steps: usize) -> (usize, inf_ecs::crime::Response) {
        let mut town = Town::new();
        park_unit(
            &mut town.world,
            CRUISER_B,
            DVec3::new(-46.0, 0.0, 38.0),
            &CRUISER_LIVERY,
            Shape::Sedan,
        );
        town.world.mark_dirty();
        town.world.propagate();
        town.bridge
            .sync_from_world_sim(&town.world, &Default::default(), &Default::default());
        // Past the epoch-0 ambient draw, and long enough for both cruisers to be
        // derived into the fleet.
        town.steps(30);
        let police = dispatch::fleet_of(&town.world)
            .map(|f| {
                f.units
                    .values()
                    .filter(|u| u.kind == UnitKind::Police)
                    .count()
            })
            .unwrap_or(0);
        assert_eq!(
            police, 2,
            "the fixture derived {police} police unit(s) — a ladder that wants \
             two cars cannot be measured against one"
        );
        // A witnessed crime, so a profile opens: `report_act` refuses one nobody
        // saw, and the whole ladder hangs off `heat`.
        inf_ecs::witness::record_act(
            &mut town.world,
            WitnessedAct {
                kind,
                actor: HERO,
                at: DVec3::new(150.0, 0.0, 50.0),
                step: 100,
                observers: vec![PATIENT],
                actor_look: 0,
                actor_vehicle: None,
            },
        );
        let mut most = 0usize;
        let mut rung = inf_ecs::crime::Response::Cold;
        for _ in 0..steps {
            town.step();
            let Some(res) = dispatch::dispatch_of(&town.world) else {
                continue;
            };
            if let Some(p) = inf_ecs::crime::profile_of(&town.world, HERO) {
                rung = rung.max(p.response());
            }
            for incident in res.searches.keys() {
                let on = res
                    .runs
                    .values()
                    .filter(|r| r.incident == Some(*incident))
                    .count();
                most = most.max(on);
            }
        }
        (most, rung)
    }

    let (serious, serious_rung) = most_cars_on_the_search(ActKind::Killed, 600);
    let (petty, petty_rung) = most_cars_on_the_search(ActKind::Carjack, 600);
    println!(
        "a {} file ({}) drew {serious} car(s); a {} file ({}) drew {petty}",
        ActKind::Killed.name(),
        serious_rung.name(),
        ActKind::Carjack.name(),
        petty_rung.name()
    );
    // ARMED: the two runs really did reach two different rungs, so what follows
    // is about the ladder and not about two identical files.
    assert_eq!(serious_rung, inf_ecs::crime::Response::MultiUnit);
    assert_eq!(petty_rung, inf_ecs::crime::Response::Patrol);
    assert_eq!(
        serious, 2,
        "a three-heat file brought {serious} car(s) — `Response::MultiUnit` \
         means nothing and SWAT is still a parked van"
    );
    assert_eq!(
        petty, 1,
        "a one-heat file brought {petty} cars — the pending filter re-pends an \
         incident that is already answered, whatever the ladder says"
    );
}

/// **A STATION THAT STREAMS OUT MID-RESPONSE RETIRES ITS UNIT** (EMS2 audit).
///
/// The fleet is a derivation on the block stamp, so a cell that pages out takes
/// its blocks *and* its vehicles and `sync_fleet` rebuilds an empty one. Two
/// things were left behind by the first cut, and both outlive the session:
///
/// * the **crew** — a `spawn_body` that only `park` ever despawns, so the row
///   was dropped from `runs` and the person was not. It stands in the road,
///   `Driving` a chassis that no longer exists, and is on `RespondersRes` for
///   ever, which makes it permanently exempt from the panic;
/// * the **siren** — `step_dispatch` returned on an empty fleet *before* it
///   reached `sound_and_light`, so `DispatchRes::sirens` froze holding a `Move`
///   and both hosts' fenced audio blocks re-pushed it every step into a ring
///   that evicts. That is the VEH2a loss with a stuck tap instead of a busy town.
///
/// Falsified from the world rather than from the counters: the body has to have
/// existed and the roster has to have held it, or this arm is about a unit that
/// never left its bay.
#[test]
fn a_station_that_streams_out_mid_response_retires_its_crew_and_its_siren() {
    let mut town = Town::new();
    town.steps(20);
    collapse(&mut town.world, PATIENT, DVec3::new(150.0, 0.0, 50.0));
    let crew = dispatch::crew_guid(AMBULANCE);
    let mut out = false;
    for _ in 0..3000 {
        town.step();
        if town.world.entity_of(crew).is_some() && dispatch::is_responder(&town.world, crew) {
            out = true;
            break;
        }
    }
    assert!(
        out,
        "no crew was ever built and put on the roster, so streaming the station \
         out below proves nothing"
    );
    let hot_before = inf_physics::d3::dispatch::running_hot(&town.world).len();
    assert_eq!(hot_before, 1, "the ambulance is not running hot");

    // The cell pages out: its blocks and its vehicles go together, which is what
    // moves `block_stamp` and rebuilds the fleet.
    let gone: Vec<Uuid> = town
        .world
        .world()
        .iter_entities()
        .filter(|e| e.get::<PcgVolume>().is_some())
        .filter_map(|e| e.get::<inf_ecs::components::Guid>().map(|g| g.0))
        .chain([AMBULANCE, CRUISER, APPLIANCE])
        .collect();
    for g in gone {
        if let Some(e) = town.world.entity_of(g) {
            town.world.despawn(e);
        }
    }
    town.world.mark_dirty();
    town.world.propagate();
    town.steps(3);

    let res = dispatch::dispatch_of(&town.world).expect("the dispatcher survives its own fleet");
    println!(
        "EMS2 stream-out: {} run(s), {} siren cue(s), crew present {}, on duty {}",
        res.runs.len(),
        res.sirens.len(),
        town.world.entity_of(crew).is_some(),
        dispatch::is_responder(&town.world, crew),
    );
    assert!(res.runs.is_empty(), "a run survived its own fleet");
    assert!(
        town.world.entity_of(crew).is_none(),
        "the crew is still standing in a street whose blocks have been unloaded"
    );
    assert!(
        !dispatch::is_responder(&town.world, crew),
        "a crew whose unit no longer exists is still on the duty roster, and is \
         therefore exempt from every panic for the rest of the process"
    );
    assert!(
        res.sirens.is_empty(),
        "{} siren cue(s) are still being drained by both hosts every step — a \
         frozen `Move` re-pushed for ever is a ring that evicts",
        res.sirens.len()
    );
    assert!(
        inf_physics::d3::dispatch::running_hot(&town.world).is_empty(),
        "a unit that no longer exists is still running hot"
    );
}

/// **AN EMERGENCY NOBODY CAN ANSWER DOES NOT STOP THE ONES SOMEBODY CAN**
/// (EMS2 audit) — the starvation this audit found, as a regression arm.
///
/// # What it is about
///
/// `assign` considers at most `ASSIGNS_PER_STEP` = **one** incident a step,
/// because an assignment is a Dijkstra per candidate unit. The first cut took
/// that one in **guid order** — content-hash order — and took it whether or not
/// the attempt succeeded. An incident whose service has **no unit in the level
/// at all** therefore sat at the front of that order for ever and consumed the
/// slot every step, and nothing else was ever considered.
///
/// It is the ordinary case, not a corner: `station_fleet` gives an appliance
/// only to a `FireHall`, `ambient_draw` produces a fire half the time, and a
/// town with a police station and a hospital and no fire hall draws one within
/// a couple of minutes.
///
/// The fixture is that town. A fire is staged that nobody can go to, and then a
/// collapse that an ambulance is parked and idle for. **Both** halves are
/// asserted: the fire is never answered (or the arm is about a town that has an
/// appliance after all) and the collapse **is**.
#[test]
fn an_emergency_nobody_can_answer_does_not_stop_the_ones_somebody_can() {
    let mut town = Town::with_fire_hall(false);
    town.steps(20);
    let fleet = town
        .world
        .world()
        .get_resource::<inf_ecs::dispatch::FleetRes>()
        .expect("a fleet")
        .clone();
    assert!(
        !fleet
            .units
            .values()
            .any(|u| u.kind == inf_ecs::dispatch::UnitKind::Fire),
        "the fixture parked an appliance, so nothing here is unanswerable"
    );

    // The fire nobody can go to, first — so it is the incident an oldest-first
    // rule reaches before the collapse, which is the whole point.
    let fire = inf_physics::d3::dispatch::report_incident(
        &mut town.world,
        inf_ecs::dispatch::IncidentKind::Fire {
            building: Uuid::from_u128(0x0E52_3001),
            intensity: 1.0,
        },
        DVec3::new(200.0, 0.0, 0.0),
    )
    .expect("the staging door opened a fire");
    town.steps(2);
    collapse(&mut town.world, PATIENT, DVec3::new(150.0, 0.0, 50.0));

    let mut answered = None;
    for i in 0..4000 {
        let s = town.step();
        if s.resolved > 0 {
            answered = Some(i);
            break;
        }
    }
    let res = dispatch::dispatch_of(&town.world).expect("a dispatcher");
    let fire_state = res.incidents.get(&fire).map(|i| i.state);
    println!(
        "EMS2 starvation: the fire is {:?} after {} step(s); the collapse was \
         resolved at {answered:?}; {} unanswered",
        fire_state.map(|s| s.name()),
        res.steps,
        res.unanswered
    );
    assert_eq!(
        fire_state,
        Some(IncidentState::Reported),
        "the fire was answered by a town with no appliance in it, so this arm \
         is not about an unanswerable incident"
    );
    assert!(
        res.unanswered > 0,
        "a town holding an emergency it cannot answer reported zero unanswered \
         steps — the diagnostic that says *why* nobody came is dead"
    );
    assert!(
        answered.is_some(),
        "an ambulance parked in its bay never went to a collapse, because a \
         FIRE nobody could answer held the one assignment slot every step — \
         this is the starvation, and it takes the whole dispatcher down with it"
    );
}

// ── (g) absent costs nothing ────────────────────────────────────────────────

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
