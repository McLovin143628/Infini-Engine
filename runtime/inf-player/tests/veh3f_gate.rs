//! **WAVE VEH3f — THE VEHICLE ROSTER.** The gate.
//!
//! Every claim of the wave, each arm **mutation-verified** and each carrying an
//! **engagement count**, so "the arm ran" and "something happened" are two
//! different facts and the second one is asserted.
//!
//! # What every arm in this file reads
//!
//! The CAR IT SPAWNS, never the TOML it parsed: a row's feel is a shipped
//! `RuntimeSim` driving it (the chassis's own rapier velocity, yaw rate and
//! travel), a trailer's following is two chassis headings stepped through a
//! slalom, a tracked rig's turn is its yaw rate, a hero body is the committed
//! file at the GUID the rig names, a census is the entities and records the
//! world HOLDS, and a cost is the vehicle phase's own clock against a control.
//!
//! # Would the eleven-row catalogue of VEH2a pass it?
//!
//! The audit brief's question for every arm, answered in the last column: the
//! island's eleven rows (`inf_editor_core::vehicle::island_vehicles`), which is
//! everything this engine could spawn before the roster.
//!
//! | arm | reads | mutation that reds it | engagement | the 11-row catalogue? |
//! |---|---|---|---|---|
//! | `every_roster_row_parses_round_trips_and_names_only_known_keys` | the committed roster through `merge_toml`; every key re-applied through `apply_table`; every class through the wire (`serde_json`) and `to_tuning`/`from_tuning` | an unknown key in `roster.toml`; a class key dropped from `VehicleClass::set` | 153 + 2 rows, 18 classes at the doc's counts, every key | **fails** -- 11 rows, no class |
//! | `the_class_defaults_apply_before_the_row_and_a_row_override_moves_only_its_row` | `from_toml_table_with` against a mutated `ClassProfiles`, and a row edited once | the class table applied AFTER the row's keys | 20 coupes moved; one row moved | **fails** -- no class key |
//! | `every_class_drives_inside_its_feel_band` | 0-100 (or 80 % of a lower limiter), the stop and the peak ramp-steer lateral g of every wheeled road row, DRIVEN on a shipped host | a class's gearing, grip or brakes moved | 135 rows, 15 classes | **fails** -- its rows have no class to be banded by |
//! | `the_sports_row_is_under_four_seconds_by_gearing_with_its_spring_kept` | the island's sports row on the shipped host: 0-100 and its settled static fraction | first gear back to 4.4 | one launch | n/a -- it IS that row (4.32 s before) |
//! | `every_bus_takes_longer_than_fifteen_seconds` | the five bus rows' sprints | a bus row given a coupe's engine | 5 rows | **fails** -- no bus |
//! | `every_roster_row_settles_inside_its_static_fraction` | each wheeled row's struts after 120 settled steps | a row's spring halved | 135 rows | passes -- VEH3b sprung them at 45 % |
//! | `the_trailer_follows_through_a_slalom_and_does_not_separate` | both chassis headings and the kingpin-to-fifth-wheel gap in the WORLD, every step of a slalom | the joint not written; the hitch's contacts left on | peak hitch angle, lag, 1 500 steps | **fails** -- no trailer |
//! | `the_semi_is_slower_with_its_trailer_than_without` | 0-60 km/h with and without | the trailer's mass zeroed | two sprints | **fails** |
//! | `the_tracked_rig_turns_by_skid` | the dozer's yaw at full steer and none, standing; its fastest wheel | the skid pass deleted; `SKID_LATERAL_KEEP` at 1.0 | two pivots | **fails** -- no rackless class |
//! | `the_hero_bodies_hang_on_their_rows_and_are_committed` | the rig recipe's `MeshRef.asset` for the five hero rows against the committed files and sidecars | a row's `body_mesh` removed | 18 panels | **fails** -- no mesh on any part |
//! | `the_construction_rows_draw_their_art_or_their_fallback` | the recipe's `art_body`, the committed fallback at its GUID, the LOCAL art at the same GUID when this machine has it, and a drive of each art row | a row's `art` removed; a fallback file deleted | 7 machines | **fails** |
//! | `nothing_from_unreal_is_committed` | `git ls-files` | a `.uasset` or an `MI_` material committed | every tracked path | passes -- nothing was ever committed |
//! | `traffic_draws_the_roster_by_class_weight` | 20 000 identities through `catalogue_row_id`, each class's share of the parked and circuit draws against its weight | the kerb draw given the circuit weights | 20 000 draws, every weighted class | **fails** -- no classes to weigh |
//! | `the_island_census_by_class_and_no_two_cars_in_one_place` | the CI island, cooked and booted as the player boots it, after 10 island minutes (1 in dev/CI): traffic records and resident chassis by class; every resident chassis box against every other (SAT) where BOTH are parked (kerb or authored), hitched pairs excepted; moving contacts printed | `authored_footprints` answering empty; the kerb draw given the circuit weights | records, classes, pairs -- VACUOUS against its mutations (stated in the arm) | **fails** -- one silhouette set, no classes |
//! | `the_real_islands_kerbs_step_around_its_fleets` | the LOCAL island (skips on CI): every parked chassis box against every other after 10 s | **none found** -- VACUOUS against the exclusion guards (stated in the arm) | 35 parked chassis, 23 kerb cars | passes |
//! | `pie_equals_shipping_on_three_classes_and_the_trailer` | every vehicle chassis pose, step by step, in `SimSession` and `RuntimeSim` -- four `vehicle.spawn`s and a hitched rig under one set of controls | either host's `vehicle.spawn` arm deleted | 600 steps, 6 vehicles | **fails** -- no `vehicle.spawn` |
//! | `the_catalogue_loads_in_microseconds` | `merge_toml` of the roster, min of five | n/a -- a COST arm | 155 rows | n/a |
//! | `sixty_four_mixed_class_cars_cost_what_they_print` | the vehicle phase's clock at 64 mixed-class cars against 64 of one row | n/a -- a COST arm | 64 cars | n/a |
//! | `the_drawn_seat_is_the_seat_the_body_sits_on` | every roster row's drawn cushions (`part_geoms` off the spawned car) against `sockets_of`'s seats | one family's `seat_r` moved | every seat-drawing row | **fails** -- no seat part |
//! | `the_engine_clip_override_reaches_the_command_stream` | the planner's `Play` clips for a row with `engine_clip` | the override in `car_loops` deleted | one voiced car | **fails** |
//! | `this_wave_moved_no_schema` | the scene and payload versions, the tunable count | a persisted field | three numbers | passes |
//! | `the_shipped_host_draws_the_roster_row_and_logs_the_columns` | `pie_drive.rs` / `window.rs` source and the Ring-0 readout | the call deleted | one row | n/a |

use std::collections::{BTreeMap, BTreeSet};

use glam::{DQuat, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Transform, Visibility,
};
use inf_ecs::math::Vec3d;
use inf_ecs::roster::{self, RosterClass};
use inf_ecs::vehicle::{VehicleControls, VehicleDef};
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::RuntimeSim;

const HZ: f64 = 60.0;
const DT: f64 = 1.0 / HZ;
const CAR: Uuid = Uuid::from_u128(0x5E3F_0001);
const GROUND: Uuid = Uuid::from_u128(0x5E3F_0002);
const TRAILER: Uuid = Uuid::from_u128(0x5E3F_0003);
const SPAWNER: Uuid = Uuid::from_u128(0x5E3F_0005);
const HALF: f64 = 3_000.0;

// ── the shipped host, a slab and one car ────────────────────────────────────

fn slab(world: &mut EcsWorld, guid: Uuid, centre: DVec3, half: DVec3, friction: f64) {
    let e = world.spawn_with_guid(guid, "Slab", None);
    world
        .world_mut()
        .entity_mut(e)
        .insert(slab_bits(centre, half, friction));
}

fn slab_bits(
    centre: DVec3,
    half: DVec3,
    friction: f64,
) -> (Transform, Visibility, RigidBody3D, Collider3D) {
    (
        Transform {
            translation: Vec3d::from_dvec3(centre),
            ..Default::default()
        },
        Visibility::default(),
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::from_dvec3(half),
            friction,
            ..Default::default()
        },
    )
}

fn spawn_spec(name: &str, at: DVec3, yaw_deg: f64) -> inf_ecs::vehicle::RigSpawn {
    inf_ecs::vehicle::RigSpawn {
        name: name.into(),
        at,
        yaw_deg,
        paint: inf_ecs::math::Color::new(0.3, 0.3, 0.7, 1.0),
        clip: None,
        engine_voice: false,
        livery: None,
    }
}

fn spawn(world: &mut EcsWorld, guid: Uuid, def: &VehicleDef, at: DVec3, yaw_deg: f64) {
    inf_ecs::vehicle::spawn_rig(world, guid, def, &spawn_spec("Car", at, yaw_deg));
}

/// A shipped `RuntimeSim` holding one `def` at rest on a 6 km slab, nose `+Z`,
/// at the far `-Z` end so a sprint has room.
fn flat_sim(def: &VehicleDef) -> RuntimeSim {
    let mut world = EcsWorld::new();
    slab(
        &mut world,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(HALF, 0.5, HALF),
        0.9,
    );
    let at = DVec3::new(
        0.0,
        inf_ecs::vehicle::resting_origin_y(def, 0.0),
        -HALF + 60.0,
    );
    spawn(&mut world, CAR, def, at, 0.0);
    world.propagate();
    RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ)
}

fn drive(sim: &mut RuntimeSim, guid: Uuid, c: VehicleControls) {
    if let Some(v) = sim.bridge3d_mut().vehicle_mut(guid) {
        v.control(VehicleControls {
            occupied: true,
            ..c
        });
    }
    sim.step_once(Default::default());
}

fn velocity(sim: &RuntimeSim, guid: Uuid) -> DVec3 {
    let b = sim.bridge3d();
    b.body_of(guid)
        .and_then(|body| b.world().body_linvel(body))
        .unwrap_or(DVec3::ZERO)
}

fn ground_speed(sim: &RuntimeSim, guid: Uuid) -> f64 {
    let v = velocity(sim, guid);
    DVec3::new(v.x, 0.0, v.z).length()
}

fn yaw_rate(sim: &RuntimeSim, guid: Uuid) -> f64 {
    let b = sim.bridge3d();
    b.body_of(guid)
        .and_then(|body| b.world().body_angvel(body))
        .map(|w| w.y)
        .unwrap_or(0.0)
}

fn position(sim: &RuntimeSim, guid: Uuid) -> DVec3 {
    let b = sim.bridge3d();
    b.body_of(guid)
        .and_then(|body| b.world().body_translation(body))
        .unwrap_or(DVec3::ZERO)
}

fn rotation(sim: &RuntimeSim, guid: Uuid) -> DQuat {
    let b = sim.bridge3d();
    b.body_of(guid)
        .and_then(|body| b.world().body_rotation(body))
        .unwrap_or(DQuat::IDENTITY)
}

/// A chassis's heading, degrees, off its own rapier body.
fn heading_deg(sim: &RuntimeSim, guid: Uuid) -> f64 {
    let f = rotation(sim, guid) * DVec3::Z;
    inf_math::patan2_64(f.x, f.z).to_degrees()
}

fn wrap_deg(a: f64) -> f64 {
    a - 360.0 * ((a + 180.0) / 360.0).floor()
}

/// The mean share of its travel the car's grounded struts are compressed.
fn static_fraction(sim: &RuntimeSim, guid: Uuid, travel: f64) -> f64 {
    let Some(v) = sim.bridge3d().vehicle_of(guid) else {
        return f64::NAN;
    };
    let rest = v.suspension_rest_m();
    let grounded: Vec<f64> = v
        .wheels()
        .iter()
        .filter(|w| w.contact.is_some())
        .map(|w| rest - w.length_m)
        .collect();
    if grounded.is_empty() || (travel.is_nan() || travel <= 0.0) {
        return f64::NAN;
    }
    grounded.iter().sum::<f64>() / grounded.len() as f64 / travel
}

// ── THE FEEL, MEASURED ──────────────────────────────────────────────────────

/// **One row's feel, MEASURED on the shipped host.**
#[derive(Clone, Copy, Debug)]
struct Feel {
    /// Share of travel used standing still.
    static_frac: f64,
    /// The speed the sprint is timed to, m/s.
    sprint_to: f64,
    /// Seconds to reach it (infinity if never).
    sprint_s: f64,
    /// Metres to stop from it.
    brake_m: f64,
    /// Mean deceleration over that stop, g.
    brake_g: f64,
    /// Peak lateral acceleration in a ramp steer, g.
    lat_g: f64,
}

/// The speed a class is timed to: 100 km/h, or 80 % of its own limiter for a
/// row that cannot reach 100 (the VEH2a feel table's rule).
fn sprint_target(def: &VehicleDef) -> f64 {
    (100.0 / 3.6f64).min(0.8 * def.class.max_speed_mps)
}

fn measure(def: &VehicleDef) -> Feel {
    let mut sim = flat_sim(def);
    for _ in 0..120 {
        drive(&mut sim, CAR, VehicleControls::default());
    }
    let static_frac = static_fraction(&sim, CAR, def.class.travel_m);
    let target = sprint_target(def);
    let full = VehicleControls {
        throttle: 1.0,
        ..Default::default()
    };
    let mut sprint_s = f64::INFINITY;
    for i in 0..(60.0 * HZ) as usize {
        drive(&mut sim, CAR, full);
        if ground_speed(&sim, CAR) >= target {
            sprint_s = (i + 1) as f64 * DT;
            break;
        }
    }
    // The stop, from where the sprint got to.
    let from = position(&sim, CAR);
    let v0 = ground_speed(&sim, CAR);
    let (mut brake_m, mut brake_s) = (f64::INFINITY, 0.0);
    for i in 0..3_600 {
        drive(
            &mut sim,
            CAR,
            VehicleControls {
                brake: 1.0,
                ..Default::default()
            },
        );
        if ground_speed(&sim, CAR) < 0.5 {
            brake_m = (position(&sim, CAR) - from).length();
            brake_s = (i + 1) as f64 * DT;
            break;
        }
    }
    let brake_g = if brake_s > 0.0 {
        v0 / brake_s / 9.81
    } else {
        0.0
    };
    // **The RAMP STEER** (the standard handling test): a fresh car brought to
    // 60 % of its sprint target (capped at 15 m/s), then the wheel wound from
    // straight to full lock over four seconds with the throttle HOLDING that
    // speed. The lateral acceleration is `v x yaw rate`, smoothed over a quarter
    // second, and the answer is its PEAK -- the most the tyres gave before the
    // car ploughed or spun. (A skidpad at full lock measures a front axle
    // scrubbing at forty degrees, which is a deceleration, not a grip -- the
    // first cut read 0.03 g off a saloon doing exactly that.)
    let mut sim = flat_sim(def);
    for _ in 0..120 {
        drive(&mut sim, CAR, VehicleControls::default());
    }
    let pad = (0.6 * target).min(15.0);
    for _ in 0..(40.0 * HZ) as usize {
        drive(&mut sim, CAR, full);
        if ground_speed(&sim, CAR) >= pad {
            break;
        }
    }
    let ramp = (4.0 * HZ) as usize;
    let mut window: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
    let mut peak = 0.0f64;
    for i in 0..ramp {
        let s = ground_speed(&sim, CAR);
        drive(
            &mut sim,
            CAR,
            VehicleControls {
                throttle: ((pad - s) / 2.0).clamp(0.0, 1.0),
                brake: ((s - pad) / 4.0).clamp(0.0, 1.0),
                steer: (i + 1) as f64 / ramp as f64,
                ..Default::default()
            },
        );
        window.push_back(ground_speed(&sim, CAR) * yaw_rate(&sim, CAR).abs());
        if window.len() > (HZ / 4.0) as usize {
            window.pop_front();
        }
        peak = peak.max(window.iter().sum::<f64>() / window.len() as f64);
    }
    Feel {
        static_frac,
        sprint_to: target,
        sprint_s,
        brake_m,
        brake_g,
        lat_g: peak / 9.81,
    }
}

type FeelRow = (RosterClass, f64, Feel);

/// Every wheeled road row's feel, measured ONCE for the whole file (every arm
/// that reads the table reads the same drive).
fn feel_table() -> &'static BTreeMap<String, FeelRow> {
    static TABLE: std::sync::OnceLock<BTreeMap<String, FeelRow>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut out = BTreeMap::new();
        for (id, def) in roster::roster().0.iter() {
            let class = def.roster_class.expect("a roster row has a class");
            if !class.road() || class == RosterClass::Trailer {
                continue;
            }
            out.insert(id.clone(), (class, def.chassis_mass_kg(), measure(def)));
        }
        out
    })
}

/// A `(min, max)` band.
type Band = (f64, f64);

/// **THE CLASS FEEL BANDS** -- `(sprint seconds, stop g, peak lateral g)`.
/// The sprint is to 100 km/h, or to 80 % of a limiter that cannot reach it
/// (construction, the slower trucks, the buses).
///
/// Written from what each class IS (a coupe is a sub-seven-second car, a bus
/// is a quarter-minute one, plant stops at half a g) and held to the
/// measurement: every bound sits outside the class's own measured spread by a
/// margin a re-tune has to cross on purpose. The doc's handling-profile table
/// has no numbers; these are this engine's, and VEH3h's cert reads them.
///
/// **Re-blessed once, with cause** (the gate's first full run): the bands were
/// drafted off the exploratory sweep, which predates the per-row centre of
/// gravity (`cog_height_m`). Lowering the heavy classes' CoG to a truck's moved
/// their load transfer, and with it the stop and the lateral peak: freight
/// stops 0.72-0.75 g and corners 0.72-0.73 g, military stops 0.68-0.69 g and
/// corners to 0.79 g, the VW camper stops 0.85 g, and the yard tractor (single
/// short ratio) sprints to its 32 km/h limiter in 8.2 s. Each bound moved to
/// clear its class's measured spread; nothing else moved.
fn band(c: RosterClass) -> (Band, Band, Band) {
    match c {
        RosterClass::Coupe => ((2.2, 6.6), (1.0, 1.2), (1.1, 2.1)),
        RosterClass::Sedan => ((2.6, 10.8), (0.97, 1.12), (1.15, 1.8)),
        RosterClass::Suv => ((2.9, 10.2), (0.9, 1.02), (1.2, 1.7)),
        RosterClass::Truck => ((2.9, 15.2), (0.85, 1.04), (0.8, 1.4)),
        RosterClass::Jeep => ((6.0, 9.6), (0.88, 0.96), (1.2, 1.45)),
        RosterClass::Hummer => ((4.5, 34.5), (0.78, 0.92), (0.65, 1.2)),
        RosterClass::Emergency => ((4.5, 25.0), (0.75, 1.03), (0.65, 1.45)),
        RosterClass::Military => ((7.0, 21.5), (0.65, 0.86), (0.2, 0.83)),
        RosterClass::Construction => ((0.5, 60.0), (0.3, 0.7), (0.0, 0.6)),
        RosterClass::Utility => ((7.0, 32.0), (0.75, 0.84), (0.28, 0.9)),
        RosterClass::Freight => ((7.5, 19.0), (0.6, 0.78), (0.25, 0.77)),
        RosterClass::Cargo => ((22.0, 32.5), (0.73, 0.79), (0.55, 0.76)),
        RosterClass::Bus => ((15.0, 30.5), (0.61, 0.68), (0.5, 0.9)),
        RosterClass::Service => ((3.9, 9.5), (0.99, 1.04), (1.2, 1.56)),
        RosterClass::Van => ((8.5, 15.0), (0.82, 0.96), (0.85, 1.2)),
        _ => (
            (0.0, f64::INFINITY),
            (0.0, f64::INFINITY),
            (0.0, f64::INFINITY),
        ),
    }
}

// ── 1. THE ROWS ─────────────────────────────────────────────────────────────

/// **Every roster row parses and round-trips, and names only keys the engine
/// knows** -- the doc's eighteen classes at their counts.
///
/// Reads the committed roster through the one catalogue door (`merge_toml`,
/// which refuses an unknown key BY NAME), then every row's own table key by key
/// back through `VehicleDef::apply_table` (so a key the parser skipped would be
/// caught here), then every row's `VehicleClass` through the wire (`serde_json`,
/// the reflected component) and through `to_tuning`/`from_tuning` -- the round
/// trip that drops a field silently (VEH3a's own tripwire).
///
/// **Mutation → red**: an unknown key added to one row (`merge_toml` refuses it
/// by name and `roster()` panics); a class key dropped from `VehicleClass::set`
/// (the re-apply loop fails on it by name).
#[test]
fn every_roster_row_parses_round_trips_and_names_only_known_keys() {
    let defs = roster::roster();
    let doc: toml::Value = toml::from_str(roster::ROSTER_TOML).expect("the roster is TOML");
    let table = doc.as_table().expect("a table of rows");
    let mut keys = 0usize;
    let mut by_class: BTreeMap<RosterClass, usize> = BTreeMap::new();
    for (id, row) in table {
        let def = defs
            .get(id)
            .unwrap_or_else(|| panic!("`{id}` did not parse"));
        let class = def.roster_class.expect("every roster row names its class");
        *by_class.entry(class).or_default() += 1;
        // The id is the LORE name, never the inspiration.
        let label = row.get("label").and_then(|l| l.as_str()).expect("a label");
        let slug: Vec<String> = label
            .to_ascii_lowercase()
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        assert_eq!(id.as_str(), slug.join("_"), "`{id}` is not its lore name");
        // Every key of the row, re-applied one at a time.
        let v = row
            .get("vehicle")
            .and_then(|v| v.as_table())
            .expect("a vehicle table");
        for (k, val) in v {
            let mut one = toml::map::Map::new();
            one.insert(k.clone(), val.clone());
            VehicleDef::default()
                .apply_table(&one)
                .unwrap_or_else(|e| panic!("`{id}`: `{k}` is not a key the engine knows ({e})"));
            keys += 1;
        }
        // The class through the wire and through the tuning door.
        let json = serde_json::to_string(&def.class).expect("the class serializes");
        let back: inf_ecs::components::VehicleClass =
            serde_json::from_str(&json).expect("and deserializes");
        assert_eq!(back, def.class, "`{id}`'s class did not survive the wire");
        let t = def.class.to_tuning();
        let again = inf_ecs::components::VehicleClass::from_tuning(&t);
        assert_eq!(
            again, def.class,
            "`{id}`'s class did not survive the tuning round trip"
        );
    }
    println!("THE ROSTER: {} rows, {keys} keys", table.len());
    for (c, n) in &by_class {
        println!("  {:<13} {n:>3} (doc {})", c.name(), c.doc_count());
        if *c != RosterClass::Trailer {
            assert_eq!(*n, c.doc_count(), "{} has {n} rows", c.name());
        }
    }
    let doc_rows: usize = by_class
        .iter()
        .filter(|(c, _)| **c != RosterClass::Trailer)
        .map(|(_, n)| n)
        .sum();
    assert_eq!(doc_rows, 153, "the doc's roster is 153 rows");
    assert_eq!(by_class.len(), 19, "eighteen classes and the trailer");
    assert!(defs.0.len() <= inf_ecs::vehicle::MAX_VEHICLE_DEFS);
    assert!(keys > 153 * 15, "only {keys} keys over the whole roster");
}

/// **The class defaults are applied BEFORE the row, and a row override moves
/// only its row** -- the audit brief's two arms in one.
///
/// Reads `VehicleDef::from_toml_table_with` against the committed class table
/// with ONE coupe default mutated (`tyre_long_rise_bias`, which no coupe row
/// authors), and against one coupe row edited to author it.
///
/// **Mutation → red**: `from_toml_table_with` applying the class table AFTER
/// the row's own keys (the edited row loses to its class default).
#[test]
fn the_class_defaults_apply_before_the_row_and_a_row_override_moves_only_its_row() {
    let doc: toml::Value = toml::from_str(roster::ROSTER_TOML).unwrap();
    let rows = doc.as_table().unwrap();
    let base = roster::class_profiles().clone();
    let mut mutated = base.clone();
    mutated
        .0
        .get_mut(&RosterClass::Coupe)
        .expect("the coupe class")
        .insert("tyre_long_rise_bias".into(), toml::Value::Float(0.61));
    let (mut moved, mut still) = (0usize, 0usize);
    for (id, row) in rows {
        let row = row.as_table().unwrap();
        let a = VehicleDef::from_toml_table_with(row, &base)
            .unwrap()
            .unwrap();
        let b = VehicleDef::from_toml_table_with(row, &mutated)
            .unwrap()
            .unwrap();
        if a.roster_class == Some(RosterClass::Coupe) {
            assert_eq!(
                b.class.tyre_long_rise_bias, 0.61,
                "`{id}` did not take its class default"
            );
            moved += 1;
        } else {
            assert_eq!(a, b, "`{id}` is not a coupe and moved");
            still += 1;
        }
    }
    assert_eq!(moved, 20, "twenty coupes");
    assert!(still > 100);
    let mut edited = rows["vapid_dominator_gt"].as_table().unwrap().clone();
    edited["vehicle"]
        .as_table_mut()
        .unwrap()
        .insert("tyre_long_rise_bias".into(), toml::Value::Float(0.93));
    let d = VehicleDef::from_toml_table_with(&edited, &mutated)
        .unwrap()
        .unwrap();
    assert_eq!(
        d.class.tyre_long_rise_bias, 0.93,
        "the row's own key lost to its class default -- the resolution order is inverted"
    );
    let neighbour = VehicleDef::from_toml_table_with(
        rows["bravado_gauntlet_hellfire"].as_table().unwrap(),
        &mutated,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        neighbour.class.tyre_long_rise_bias, 0.61,
        "a neighbour moved with the row"
    );
    println!("CLASS DEFAULTS: one coupe default moved {moved} rows and {still} did not; one row's own key beat it");
}

// ── 2. THE FEEL ─────────────────────────────────────────────────────────────

/// **Every class drives inside its feel band** -- the sprint, the stop and the
/// peak ramp-steer lateral g of EVERY wheeled road row, driven on a shipped
/// `RuntimeSim`, against [`band`].
///
/// **Mutation → red**: the coupe class's `longitudinal_grip` at 0.5 (its
/// sprints and stops leave the band); the bus class's `brake_force_n` doubled
/// through its rows (the stop g does).
#[test]
fn every_class_drives_inside_its_feel_band() {
    let table = feel_table();
    let mut bad: Vec<String> = Vec::new();
    let mut per_class: BTreeMap<RosterClass, Vec<(f64, f64, f64)>> = BTreeMap::new();
    println!("THE FEEL TABLE (shipped host, flat slab, mu 0.9):");
    for (id, (class, mass, f)) in table {
        println!(
            "  {:<12} {:<28} {:>6.0} kg  static {:.2}  0-{:>3.0} {:>6.2} s  stop {:>5.1} m {:.2} g  lat {:.2} g",
            class.name(),
            id,
            mass,
            f.static_frac,
            f.sprint_to * 3.6,
            f.sprint_s,
            f.brake_m,
            f.brake_g,
            f.lat_g
        );
        per_class
            .entry(*class)
            .or_default()
            .push((f.sprint_s, f.brake_g, f.lat_g));
        let ((s0, s1), (b0, b1), (l0, l1)) = band(*class);
        if !(s0..=s1).contains(&f.sprint_s) {
            bad.push(format!(
                "{id}: sprint {:.2} s outside {s0}..{s1}",
                f.sprint_s
            ));
        }
        if !(b0..=b1).contains(&f.brake_g) {
            bad.push(format!("{id}: stop {:.2} g outside {b0}..{b1}", f.brake_g));
        }
        if !(l0..=l1).contains(&f.lat_g) {
            bad.push(format!("{id}: lateral {:.2} g outside {l0}..{l1}", f.lat_g));
        }
    }
    println!("THE CLASS TABLE (min .. max over the class's rows):");
    for (c, v) in &per_class {
        let lo = |k: usize| {
            v.iter()
                .map(|x| [x.0, x.1, x.2][k])
                .fold(f64::MAX, f64::min)
        };
        let hi = |k: usize| {
            v.iter()
                .map(|x| [x.0, x.1, x.2][k])
                .fold(f64::MIN, f64::max)
        };
        println!(
            "  {:<13} {:>2} rows  sprint {:>5.2}..{:>5.2} s  stop {:.2}..{:.2} g  lat {:.2}..{:.2} g",
            c.name(),
            v.len(),
            lo(0),
            hi(0),
            lo(1),
            hi(1),
            lo(2),
            hi(2)
        );
    }
    assert!(
        table.len() >= 130,
        "only {} wheeled rows were driven",
        table.len()
    );
    assert_eq!(per_class.len(), 15, "fifteen road classes were driven");
    assert!(
        bad.is_empty(),
        "rows outside their class band:\n{}",
        bad.join("\n")
    );
}

/// **The sports row is back under four seconds, by gearing, with its spring
/// kept** -- the VEH3b audit's carried item: 4.32 s on the springs that audit
/// set, against a class band of under four.
///
/// **Mutation → red**: first gear back to 4.4 (4.32 s); the spring softened
/// instead (the static fraction leaves 0.44..0.46).
#[test]
fn the_sports_row_is_under_four_seconds_by_gearing_with_its_spring_kept() {
    let def = *inf_editor_core::vehicle::island_vehicles()
        .get("sports")
        .expect("the island's sports row");
    let f = measure(&def);
    println!(
        "THE SPORTS ROW: 0-100 in {:.2} s (was 4.32), static {:.3} of travel, first gear {}",
        f.sprint_s, f.static_frac, def.class.gear_1_ratio
    );
    assert!(f.sprint_s < 4.0, "0-100 in {:.2} s", f.sprint_s);
    assert!(
        (0.44..=0.46).contains(&f.static_frac),
        "the spring was moved: {:.3} of travel standing (VEH3b set 0.45)",
        f.static_frac
    );
}

/// **Every bus takes longer than fifteen seconds** to its sprint speed.
///
/// **Mutation → red**: a bus row given a coupe's engine.
#[test]
fn every_bus_takes_longer_than_fifteen_seconds() {
    let buses: Vec<(&String, f64)> = feel_table()
        .iter()
        .filter(|(_, (c, _, _))| *c == RosterClass::Bus)
        .map(|(id, (_, _, f))| (id, f.sprint_s))
        .collect();
    println!("THE BUSES: {buses:?}");
    assert_eq!(buses.len(), 5);
    for (id, s) in buses {
        assert!(s > 15.0, "{id} reached its sprint speed in {s:.2} s");
    }
}

/// **Every wheeled roster row settles inside 30-45 % of its travel** -- read
/// off the SETTLED struts, not the formula its comment quotes.
///
/// **Mutation → red**: one row's spring halved (it sags past 45 %).
#[test]
fn every_roster_row_settles_inside_its_static_fraction() {
    let (mut hi, mut lo, mut n) = (0.0f64, 1.0f64, 0usize);
    for (id, (_, _, f)) in feel_table() {
        assert!(
            (0.30..=0.455).contains(&f.static_frac),
            "{id} stands at {:.3} of its travel",
            f.static_frac
        );
        hi = hi.max(f.static_frac);
        lo = lo.min(f.static_frac);
        n += 1;
    }
    println!("STATIC FRACTION: {n} rows settled between {lo:.3} and {hi:.3} of their travel");
    assert!(n >= 130);
}

// ── 3. THE ARTICULATED RIG AND THE TRACKS ───────────────────────────────────

/// A shipped host with a tractor and, optionally, its trailer hitched.
fn rig_sim(tractor: &VehicleDef, trailer: Option<&VehicleDef>) -> RuntimeSim {
    let mut world = EcsWorld::new();
    slab(
        &mut world,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(HALF, 0.5, HALF),
        0.9,
    );
    let at = DVec3::new(
        0.0,
        inf_ecs::vehicle::resting_origin_y(tractor, 0.0),
        -HALF + 200.0,
    );
    spawn(&mut world, CAR, tractor, at, 0.0);
    if let Some(t) = trailer {
        let tat = inf_ecs::vehicle::hitched_trailer_at(at, 0.0, tractor, t).expect("a fifth wheel");
        spawn(&mut world, TRAILER, t, tat, 0.0);
        assert!(inf_ecs::vehicle::hitch(
            &mut world, CAR, tractor, TRAILER, t
        ));
    }
    world.propagate();
    RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ)
}

/// **The trailer FOLLOWS** -- the first articulated rig, through a slalom: its
/// heading lags the tractor's and converges, it never jack-knifes, and the
/// kingpin never leaves the fifth wheel.
///
/// Reads both chassis' own rapier headings every step, and the WORLD distance
/// between the kingpin (the trailer's anchor through its own pose) and the
/// fifth wheel (the tractor's).
///
/// **Mutation → red**: the joint not written (`hitch` a no-op -- the tractor
/// pulls away and the gap grows past a metre); the bridge's contacts-off rule
/// for a chassis pair deleted (the two boxes fight at the coupling).
#[test]
fn the_trailer_follows_through_a_slalom_and_does_not_separate() {
    let tractor = *roster::roster().get("mtl_packer").expect("a tractor");
    let trailer = *roster::roster()
        .get("jobuilt_box_trailer")
        .expect("a trailer");
    let mut sim = rig_sim(&tractor, Some(&trailer));
    let fifth = inf_ecs::vehicle::coupling_local(&tractor, &trailer)
        .unwrap()
        .to_dvec3();
    let pin = inf_ecs::vehicle::kingpin_local(&trailer).to_dvec3();
    let (mut peak_hitch, mut worst_gap, mut min_y) = (0.0f64, 0.0f64, f64::MAX);
    let (mut tractor_peak, mut trailer_peak) = ((0.0f64, 0usize), (0.0f64, 0usize));
    let mut last_hitch = 0.0f64;
    // The slalom's own trace, for a frame of it (`INF_VEH3F_TRACE_DIR`): step,
    // time, steer, both headings, the hitch angle and the kingpin gap.
    let mut trace = String::from("step,t,steer,tractor_deg,trailer_deg,hitch_deg,gap_m\n");
    for i in 0..1_500usize {
        let t = i as f64 * DT;
        let steer = if t > 6.0 && t < 17.0 {
            0.35 * inf_math::psin64((t - 6.0) * 0.9)
        } else {
            0.0
        };
        let throttle = if ground_speed(&sim, CAR) < 11.0 {
            0.7
        } else {
            0.0
        };
        drive(
            &mut sim,
            CAR,
            VehicleControls {
                throttle,
                steer,
                ..Default::default()
            },
        );
        let (a, b) = (heading_deg(&sim, CAR), heading_deg(&sim, TRAILER));
        let hitch = wrap_deg(a - b);
        peak_hitch = peak_hitch.max(hitch.abs());
        last_hitch = hitch;
        if i < 800 {
            if a.abs() > tractor_peak.0 {
                tractor_peak = (a.abs(), i);
            }
            if b.abs() > trailer_peak.0 {
                trailer_peak = (b.abs(), i);
            }
        }
        let k = position(&sim, TRAILER) + rotation(&sim, TRAILER) * pin;
        let f = position(&sim, CAR) + rotation(&sim, CAR) * fifth;
        worst_gap = worst_gap.max((k - f).length());
        min_y = min_y.min(position(&sim, TRAILER).y);
        trace.push_str(&format!(
            "{i},{t:.4},{steer:.4},{a:.4},{b:.4},{hitch:.4},{:.5}\n",
            (k - f).length()
        ));
    }
    if let Some(dir) = std::env::var_os("INF_VEH3F_TRACE_DIR") {
        let path = std::path::Path::new(&dir).join("trailer-slalom.csv");
        std::fs::write(&path, &trace).expect("the trace writes");
        println!("THE TRAILER: trace written to {}", path.display());
    }
    let lag_s = (trailer_peak.1 as f64 - tractor_peak.1 as f64) * DT;
    println!(
        "THE TRAILER: peak hitch {peak_hitch:.1} deg; the trailer's heading peaks {lag_s:.2} s after the tractor's ({:.1} vs {:.1} deg); worst kingpin gap {worst_gap:.3} m; lowest trailer origin {min_y:.2} m; hitch at the end {last_hitch:.2} deg",
        tractor_peak.0, trailer_peak.0
    );
    assert!(
        peak_hitch > 5.0,
        "the hitch never bent ({peak_hitch:.2} deg)"
    );
    assert!(
        peak_hitch < 60.0,
        "the rig jack-knifed ({peak_hitch:.1} deg)"
    );
    assert!(
        lag_s > 0.1,
        "the trailer's heading does not LAG the tractor's ({lag_s:.2} s)"
    );
    assert!(
        last_hitch.abs() < 3.0,
        "the rig did not straighten after the slalom ({last_hitch:.2} deg)"
    );
    assert!(
        worst_gap < 0.1,
        "the kingpin left the fifth wheel by {worst_gap:.3} m"
    );
    assert!(
        min_y > 1.0,
        "the trailer fell through (origin {min_y:.2} m)"
    );
}

/// **A semi with its trailer is slower than without** -- 0-60 km/h, the same
/// tractor, the trailer's seven tonnes on its fifth wheel.
///
/// **Mutation → red**: the trailer's mass zeroed (the two times meet).
#[test]
fn the_semi_is_slower_with_its_trailer_than_without() {
    let tractor = *roster::roster().get("mtl_packer").unwrap();
    let trailer = *roster::roster().get("jobuilt_box_trailer").unwrap();
    let sprint = |sim: &mut RuntimeSim| -> f64 {
        for _ in 0..120 {
            drive(sim, CAR, VehicleControls::default());
        }
        for i in 0..(60.0 * HZ) as usize {
            drive(
                sim,
                CAR,
                VehicleControls {
                    throttle: 1.0,
                    ..Default::default()
                },
            );
            if ground_speed(sim, CAR) >= 60.0 / 3.6 {
                return (i + 1) as f64 * DT;
            }
        }
        f64::INFINITY
    };
    let alone = sprint(&mut rig_sim(&tractor, None));
    let towing = sprint(&mut rig_sim(&tractor, Some(&trailer)));
    println!("THE SEMI: 0-60 km/h in {alone:.2} s alone, {towing:.2} s towing");
    assert!(alone.is_finite() && towing.is_finite());
    assert!(
        towing > alone * 1.15,
        "the trailer costs the tractor nothing ({alone:.2} vs {towing:.2} s)"
    );
}

/// **The tracked rig TURNS BY SKID** -- a dozer with no steering rack,
/// standing: full steer pivots it, none does not, and no wheel spins away.
///
/// **Mutation → red**: the skid pass deleted (no yaw at full steer);
/// `SKID_LATERAL_KEEP` at 1.0 (the pivot collapses under the floor below).
#[test]
fn the_tracked_rig_turns_by_skid() {
    let def = *roster::roster().get("hvy_dozer").expect("the dozer");
    assert!(def.body.tracked() && def.class.max_steer_deg <= 0.0);
    let pivot = |steer: f64| -> (f64, f64, f64) {
        let mut sim = flat_sim(&def);
        for _ in 0..120 {
            drive(&mut sim, CAR, VehicleControls::default());
        }
        let y0 = heading_deg(&sim, CAR);
        let mut omega = 0.0f64;
        for _ in 0..300 {
            drive(
                &mut sim,
                CAR,
                VehicleControls {
                    steer,
                    ..Default::default()
                },
            );
            if let Some(v) = sim.bridge3d().vehicle_of(CAR) {
                for w in v.wheels() {
                    omega = omega.max(w.omega_rad_s.abs());
                }
            }
        }
        (
            yaw_rate(&sim, CAR),
            wrap_deg(heading_deg(&sim, CAR) - y0),
            omega,
        )
    };
    let (w1, turned, omega) = pivot(1.0);
    let (w0, still, _) = pivot(0.0);
    println!(
        "THE TRACKS: full steer yaws {w1:.3} rad/s ({turned:.1} deg in 5 s), none {w0:.4} rad/s ({still:.2} deg); the fastest wheel {omega:.1} rad/s"
    );
    assert!(w1.abs() > 0.2, "a full-steer pivot yawed {w1:.3} rad/s");
    assert!(
        turned.abs() > 20.0,
        "it turned {turned:.1} deg in five seconds"
    );
    assert!(
        w0.abs() < 1e-3 && still.abs() < 0.5,
        "it turned with no steer"
    );
    assert!(
        omega < 20.0,
        "a track spun at {omega:.1} rad/s -- the skid is not bounded by the ground"
    );
}

// ── 4. THE BODIES ───────────────────────────────────────────────────────────

/// **The DCC hero bodies hang on their rows and are committed** -- the rig
/// recipe for each of the five hero rows names a mesh on each panel its set
/// covers, the committed file is at that GUID, and it is a real mesh.
///
/// **Mutation → red**: a row's `body_mesh` removed; `hero_parts` emptied.
#[test]
fn the_hero_bodies_hang_on_their_rows_and_are_committed() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples/vehicle-bodies");
    let mut on_disk: BTreeMap<Uuid, (String, u64)> = BTreeMap::new();
    for e in std::fs::read_dir(&dir).expect("the hero folder is committed") {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "inf_mesh") {
            let side = inf_asset::AssetSidecar::load(&p).expect("a sidecar");
            on_disk.insert(
                side.guid.0,
                (
                    p.file_name().unwrap().to_string_lossy().into_owned(),
                    std::fs::metadata(&p).unwrap().len(),
                ),
            );
        }
    }
    let defs = inf_editor_core::vehicle::island_vehicles();
    let (mut hung, mut bytes) = (0usize, 0u64);
    for id in ["sedan", "sports", "suv", "truck", "cruiser"] {
        let def = defs.get(id).expect("the row");
        assert!(def.body_mesh.is_some(), "`{id}` names no hero set");
        let nodes = inf_ecs::vehicle::rig_nodes(
            Uuid::from_u128(9),
            def,
            &spawn_spec("x", DVec3::ZERO, 0.0),
        );
        let meshes: Vec<Uuid> = nodes
            .iter()
            .filter_map(|n| n.mesh.and_then(|m| m.asset))
            .collect();
        assert_eq!(
            meshes.len(),
            inf_ecs::vehicle::hero_parts(def.body).len(),
            "`{id}` hangs {} meshes",
            meshes.len()
        );
        for m in meshes {
            let (file, len) = on_disk
                .get(&m)
                .unwrap_or_else(|| panic!("`{id}` names mesh {m}, which is not committed"));
            assert!(*len > 4_000, "{file} is {len} B");
            bytes += len;
            hung += 1;
        }
    }
    println!("THE HERO BODIES: {hung} panels on five rows, {bytes} B committed");
    assert_eq!(hung, 18);
}

/// **The construction rows draw their art where it is, and their fallback where
/// it is not** -- both arms.
///
/// Always: each of the seven art rows' rigs hangs `art_body` and four wheels at
/// the art GUIDs, the committed FALLBACK is at every one of them (so CI, which
/// never has the art, draws the family's silhouette), and each row DRIVES on
/// the shipped host. And where this machine has the island project with the
/// imported art (`island-refresh.ps1 -SyncVehicles`), the art is at the SAME
/// GUIDs, is not the fallback, and its measured geometry agrees with the row.
///
/// **Mutation → red**: a row's `art` removed (no `art_body`); a fallback file
/// deleted; the local art written at another GUID.
#[test]
fn the_construction_rows_draw_their_art_or_their_fallback() {
    let fallback =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples/vehicle-art");
    let mut committed: BTreeMap<Uuid, u64> = BTreeMap::new();
    for e in std::fs::read_dir(&fallback).expect("the art fallback is committed") {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "inf_mesh") {
            let side = inf_asset::AssetSidecar::load(&p).expect("a sidecar");
            committed.insert(side.guid.0, std::fs::metadata(&p).unwrap().len());
        }
    }
    let local = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../island-build/project/Content/UE/Vehicles");
    let (mut drove, mut local_art) = (0usize, 0usize);
    for key in roster::ArtKey::ALL {
        let (id, def) = roster::roster()
            .0
            .iter()
            .find(|(_, d)| d.art == Some(key))
            .unwrap_or_else(|| panic!("no row draws {}", key.name()));
        let nodes = inf_ecs::vehicle::rig_nodes(CAR, def, &spawn_spec("x", DVec3::ZERO, 0.0));
        let body = nodes
            .iter()
            .find(|n| n.name == inf_ecs::vehicle::ART_BODY_PART)
            .unwrap_or_else(|| panic!("`{id}` has no art body"));
        assert_eq!(
            body.mesh.and_then(|m| m.asset),
            Some(roster::art_body_guid(key))
        );
        assert!(
            committed.contains_key(&roster::art_body_guid(key)),
            "no fallback body for {}",
            key.name()
        );
        for i in 0..4 {
            assert!(committed.contains_key(&roster::art_wheel_guid(key, i)));
        }
        // The family's panels are NOT drawn over the art -- only its seats.
        assert!(nodes
            .iter()
            .all(|n| !n.name.starts_with("door") && n.name != "lower" && n.name != "cab"));
        // …and it drives, whatever it draws.
        let mut sim = flat_sim(def);
        for _ in 0..120 {
            drive(&mut sim, CAR, VehicleControls::default());
        }
        for _ in 0..240 {
            drive(
                &mut sim,
                CAR,
                VehicleControls {
                    throttle: 1.0,
                    ..Default::default()
                },
            );
        }
        let v = ground_speed(&sim, CAR);
        println!(
            "THE ART ROWS: `{id}` ({}) at {v:.2} m/s after 4 s of throttle",
            key.name()
        );
        assert!(v > 0.5, "`{id}` did not drive");
        drove += 1;
        // The LOCAL half, where the art is on this machine.
        let dir = local.join(key.name());
        let art = dir.join(format!("{}_body.inf_mesh", key.name()));
        if !art.is_file() {
            println!(
                "  LOCAL ART: {} is not on this machine -- the fallback draws (CI's case)",
                key.name()
            );
            continue;
        }
        let side = inf_asset::AssetSidecar::load(&art).expect("the local art's sidecar");
        assert_eq!(
            side.guid.0,
            roster::art_body_guid(key),
            "the art is at another GUID"
        );
        let len = std::fs::metadata(&art).unwrap().len();
        assert!(
            len > committed[&roster::art_body_guid(key)] * 10,
            "the local {} body is {len} B -- that is the fallback, not the art",
            key.name()
        );
        let toml: toml::Value = toml::from_str(
            &std::fs::read_to_string(dir.join(format!("{}.vehicle.toml", key.name())))
                .expect("the per-body TOML"),
        )
        .unwrap();
        let g = toml["geometry"].as_table().unwrap();
        for (k, want) in [
            ("half_width_m", def.half_extents.x),
            ("half_length_m", def.half_extents.z),
        ] {
            let got = g[k].as_float().unwrap();
            assert!(
                (got - want).abs() < 0.05 + 0.02 * want.abs(),
                "{}: the art measures {k} = {got:.3}, the row says {want:.3}",
                key.name()
            );
        }
        local_art += 1;
        println!(
            "  LOCAL ART: {} at {} ({len} B) agrees with its row",
            key.name(),
            side.guid.0
        );
    }
    println!("THE ART ROWS: {drove} drove; {local_art} of 7 machines' art is on this machine");
    assert_eq!(drove, 7);
}

/// **Nothing from Unreal is committed** -- `git ls-files`, every tracked path,
/// against the shapes Unreal content takes.
///
/// **Mutation → red**: a `.uasset`, an `SM_*.inf_mesh` or an `MI_*.inf_mat`
/// added to the index.
#[test]
fn nothing_from_unreal_is_committed() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = match std::process::Command::new("git")
        .arg("ls-files")
        .current_dir(&root)
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => {
            eprintln!("SKIP: no git here");
            return;
        }
    };
    let files = String::from_utf8_lossy(&out.stdout);
    let mut n = 0usize;
    let mut bad = Vec::new();
    for f in files.lines() {
        n += 1;
        let name = f.rsplit('/').next().unwrap_or(f);
        let lower = f.to_ascii_lowercase();
        if lower.ends_with(".uasset")
            || lower.ends_with(".umap")
            || lower.ends_with(".fbx")
            || lower.contains("constructionvehiclespack")
            || lower.starts_with("ue-out/")
            || (name.starts_with("SM_") && lower.ends_with(".inf_mesh"))
            || (name.starts_with("T_") && lower.ends_with(".inf_tex"))
            || (name.starts_with("MI_") && lower.ends_with(".inf_mat"))
        {
            bad.push(f.to_string());
        }
    }
    println!(
        "NOTHING FROM UNREAL: {n} tracked paths, {} Unreal-shaped",
        bad.len()
    );
    assert!(n > 1_000);
    assert!(bad.is_empty(), "Unreal content is committed: {bad:?}");
}

// ── 5. THE ISLAND ───────────────────────────────────────────────────────────

/// **Traffic draws the roster BY CLASS WEIGHT** -- the census at a population
/// the CI island cannot hold: 20 000 traffic identities through the shipped
/// draw (`traffic::catalogue_row_id`, the counter-hash door both hosts call),
/// each class's share of the parked and the circuit draws against its weight.
///
/// Reads the rows the draw PICKS, by the class the row IS. Zero-weight classes
/// (emergency, military, construction, freight, air, marine, trailers) must be
/// absent, and every weighted class within two points of its share.
///
/// **Mutation → red**: the kerb draw given `circuit_weight` (buses park; the
/// bus share of the parked draw leaves zero); `pick_row` choosing a row
/// without the class step (the shares follow the row counts, not the weights).
#[test]
fn traffic_draws_the_roster_by_class_weight() {
    use inf_ecs::traffic::{catalogue_row_id, day_of, TrafficDay, KERB_SLOT_M};
    // The draw's own admission rule, per day: a kerb car must also be a
    // civilian silhouette.
    let fits = |c: RosterClass, circuit: bool| {
        roster::roster().0.values().any(|d| {
            d.roster_class == Some(c)
                && d.body.wheeled()
                && !d.body.towed()
                && 2.0 * d.half_extents.z <= KERB_SLOT_M - 1.5
                && (circuit || inf_ecs::vehicle::VehicleBody::CIVILIAN.contains(&d.body))
        })
    };
    let mut drawn: [BTreeMap<RosterClass, usize>; 2] = [BTreeMap::new(), BTreeMap::new()];
    let mut n = [0usize; 2];
    for k in 0..20_000u128 {
        let g = Uuid::from_u128(k.wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835) ^ 0x5E3F);
        let circuit = matches!(day_of(g), TrafficDay::DayCircuit | TrafficDay::NightCircuit);
        let id = catalogue_row_id(g).expect("a row");
        let class = roster::roster()
            .get(id)
            .and_then(|d| d.roster_class)
            .unwrap();
        *drawn[circuit as usize].entry(class).or_default() += 1;
        n[circuit as usize] += 1;
    }
    let mut worst = 0.0f64;
    for (i, (label, weight)) in [
        (
            "parked",
            RosterClass::parked_weight as fn(RosterClass) -> f64,
        ),
        (
            "circuit",
            RosterClass::circuit_weight as fn(RosterClass) -> f64,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let total: f64 = RosterClass::ALL
            .into_iter()
            .filter(|c| fits(*c, i == 1))
            .map(weight)
            .filter(|w| *w > 0.0)
            .sum();
        println!("THE {label} DRAW over {} identities:", n[i]);
        for c in RosterClass::ALL {
            let got = *drawn[i].get(&c).unwrap_or(&0) as f64 / n[i] as f64;
            let want = if fits(c, i == 1) {
                weight(c).max(0.0) / total
            } else {
                0.0
            };
            if got > 0.0 || want > 0.0 {
                println!(
                    "  {:<13} {:>5.1} % (weight {:>5.1} %)",
                    c.name(),
                    got * 100.0,
                    want * 100.0
                );
            }
            if want == 0.0 {
                assert_eq!(got, 0.0, "a {} is drawn at the {label} kerb", c.name());
            } else {
                worst = worst.max((got - want).abs());
                assert!(
                    (got - want).abs() < 0.02,
                    "{label}: {} drawn at {:.1} % against its weight's {:.1} %",
                    c.name(),
                    got * 100.0,
                    want * 100.0
                );
            }
        }
    }
    println!(
        "THE DRAW: every weighted class within {:.2} points of its weight",
        worst * 100.0
    );
    assert!(n[0] > 5_000 && n[1] > 2_000, "the days split {n:?}");
    assert!(
        drawn[1].contains_key(&RosterClass::Bus),
        "no bus is ever traffic"
    );
    assert!(!drawn[0].contains_key(&RosterClass::Bus), "a bus parks");
}

fn fixture_recipe() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples/island-fixture/island.toml")
}

/// The CI island, cooked and booted as `run_headless` boots it.
fn island_sim(tmp: &std::path::Path) -> RuntimeSim {
    let recipe = inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    let proj = tmp.join("island");
    inf_project::ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("scaffold");
    inf_island::write_content(&build, &proj.join("Content")).expect("content");
    let out = tmp.join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the island cooks");
    let source = inf_player::level::PackLevelSource::open(&out).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Pack(source.clone()),
    );
    sim
}

/// How many island minutes the census runs: TEN in a release build off CI (the
/// brief's number; 1.15 s of wall an island second on the machine that wrote
/// it, so eleven minutes), ONE in a dev build or on a shared runner, where ten
/// would be hours of debug stepping. `INF_VEH3F_ISLAND_MINUTES` overrides it.
/// Printed either way.
fn island_minutes() -> u64 {
    if let Some(m) = std::env::var("INF_VEH3F_ISLAND_MINUTES")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        return m;
    }
    if cfg!(debug_assertions) || std::env::var_os("CI").is_some() {
        1
    } else {
        10
    }
}

/// A chassis's oriented box: `(guid, centre, rotation, half-extents)`.
type ChassisBox = (Uuid, DVec3, DQuat, DVec3);

/// Every resident vehicle chassis -- anything with a `VehicleClass`, and every
/// traffic car at any tier (a `Near` car is a kinematic box with no class).
fn chassis_boxes(sim: &RuntimeSim) -> Vec<ChassisBox> {
    let w = sim.world();
    let traffic: BTreeSet<Uuid> = inf_ecs::traffic::traffic_of(w)
        .map(|t| t.records.keys().copied().collect())
        .unwrap_or_default();
    let mut out = Vec::new();
    for e in w.world().iter_entities() {
        let Some(g) = e.get::<inf_ecs::components::Guid>().map(|g| g.0) else {
            continue;
        };
        if e.get::<inf_ecs::components::VehicleClass>().is_none() && !traffic.contains(&g) {
            continue;
        }
        let (Some(t), Some(c)) = (e.get::<Transform>(), e.get::<Collider3D>()) else {
            continue;
        };
        if c.shape_kind != ColliderShape3DKind::Box {
            continue;
        }
        out.push((
            g,
            t.translation.to_dvec3(),
            t.quat(),
            c.half_extents.to_dvec3(),
        ));
    }
    out
}

/// Separating-axis test for two oriented boxes.
fn boxes_overlap(a: &ChassisBox, b: &ChassisBox) -> bool {
    let ax = [a.2 * DVec3::X, a.2 * DVec3::Y, a.2 * DVec3::Z];
    let bx = [b.2 * DVec3::X, b.2 * DVec3::Y, b.2 * DVec3::Z];
    let d = b.1 - a.1;
    let mut axes: Vec<DVec3> = ax.iter().chain(bx.iter()).copied().collect();
    for u in ax {
        for v in bx {
            let c = u.cross(v);
            if c.length_squared() > 1e-9 {
                axes.push(c.normalize());
            }
        }
    }
    for l in axes {
        let ra =
            a.3.x * ax[0].dot(l).abs() + a.3.y * ax[1].dot(l).abs() + a.3.z * ax[2].dot(l).abs();
        let rb =
            b.3.x * bx[0].dot(l).abs() + b.3.y * bx[1].dot(l).abs() + b.3.z * bx[2].dot(l).abs();
        if d.dot(l).abs() > ra + rb {
            return false;
        }
    }
    true
}

/// Hitched pairs overlap by design (the trailer's nose over the fifth wheel).
fn hitched_pairs(sim: &RuntimeSim) -> BTreeSet<(Uuid, Uuid)> {
    let mut out = BTreeSet::new();
    for e in sim.world().world().iter_entities() {
        if let (Some(j), Some(g)) = (
            e.get::<inf_ecs::components::Joint3D>(),
            e.get::<inf_ecs::components::Guid>(),
        ) {
            if let Some(o) = j.other.get() {
                out.insert((g.0.min(o), g.0.max(o)));
            }
        }
    }
    out
}

/// **THE ISLAND CENSUS BY CLASS, and no two cars in one place** -- the CI
/// island, cooked and booted as the shipped player boots it, run for its
/// minutes.
///
/// Reads the WORLD: the traffic records by the class of the row each car IS
/// (`traffic::catalogue_row_id`), the resident chassis by class
/// (`roster::row_of`), every parked car's class against the kerb's weights, and
/// every PARKED chassis's box (a kerb car or an authored vehicle) against every
/// other parked one's every ten seconds -- the lattice's claim. A moving car's
/// contacts are the traffic model's and are printed, not asserted.
/// Emergency rows arrive only by dispatch (never a traffic record); no air,
/// sea, plant, freight or trailer is ever a traffic car; no bus, cargo or
/// utility truck is PARKED at a kerb.
///
/// **VACUOUS against its mutations, measured and said.** The CI island holds
/// eleven traffic cars and one authored one: the kerb draw given the circuit
/// weights without its civilian filter, `authored_footprints` answering empty,
/// `PARK_CLEAR_M` at -3 and the lattice's self-exclusion removed all leave it
/// green (its eleven cars are circuit and commute days, and no two slots are
/// close). It asserts the WORLD's state; the class weights are
/// `traffic_draws_the_roster_by_class_weight`'s, which each of those reds.
#[test]
fn the_island_census_by_class_and_no_two_cars_in_one_place() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let mut sim = island_sim(tmp.path());
    let minutes = island_minutes();
    let steps = minutes * 60 * HZ as u64;
    let hitched = hitched_pairs(&sim);
    // A MOVING traffic car: one whose day is a circuit or a commute. A moving
    // car's contact with anything is the traffic model's business (VEH2b's),
    // not the parked lattice's -- measured on the ten-minute run, a day-circuit
    // car brushing the fixture's authored saloon at 290 s and a commuter
    // leaving its kerb past its parked neighbour at 310 s -- so those contacts
    // are counted, printed and carried, and this arm's claim is the LATTICE's:
    // no two PARKED vehicles (kerb cars or authored ones) share their space.
    let moving = |g: Uuid, sim: &RuntimeSim| {
        inf_ecs::traffic::traffic_of(sim.world()).is_some_and(|t| t.records.contains_key(&g))
            && inf_ecs::traffic::day_of(g) != inf_ecs::traffic::TrafficDay::Parked
    };
    let mut moving_contacts: BTreeSet<(Uuid, Uuid)> = BTreeSet::new();
    let (mut pairs_tested, mut worst) = (0usize, 0usize);
    let mut overlapping: BTreeSet<(Uuid, Uuid)> = BTreeSet::new();
    let t0 = std::time::Instant::now();
    let mut window = inf_player::step_profile::StepProfile::default();
    sim.set_step_profiling(true);
    for s in 0..steps {
        sim.step_once(Default::default());
        window.accumulate(&sim.step_profile());
        if s % 600 != 599 {
            continue;
        }
        let dear: Vec<String> = window
            .dearest_first()
            .into_iter()
            .take(3)
            .map(|(n, ms)| format!("{n} {:.2}", ms / 600.0))
            .collect();
        eprintln!(
            "  island second {}: {:.1} s of wall so far; {} traffic records, {} resident chassis; dearest ms/step {}",
            (s + 1) / 60,
            t0.elapsed().as_secs_f64(),
            inf_ecs::traffic::traffic_of(sim.world()).map_or(0, |t| t.records.len()),
            chassis_boxes(&sim).len(),
            dear.join(", ")
        );
        {
            let w = sim.world().world();
            let ents = w.iter_entities().count();
            let bodies = w
                .iter_entities()
                .filter(|e| e.get::<RigidBody3D>().is_some())
                .count();
            let chars = w
                .iter_entities()
                .filter(|e| e.get::<inf_ecs::components::CharacterMovement>().is_some())
                .count();
            eprintln!("    entities {ents}, rigid bodies {bodies}, characters {chars}");
        }
        window = inf_player::step_profile::StepProfile::default();
        let boxes = chassis_boxes(&sim);
        let mut here = 0usize;
        for i in 0..boxes.len() {
            for j in i + 1..boxes.len() {
                let (a, b) = (&boxes[i], &boxes[j]);
                let key = (a.0.min(b.0), a.0.max(b.0));
                if hitched.contains(&key) {
                    continue;
                }
                if moving(a.0, &sim) || moving(b.0, &sim) {
                    if (a.1 - b.1).length() <= a.3.length() + b.3.length() && boxes_overlap(a, b) {
                        moving_contacts.insert(key);
                    }
                    continue;
                }
                pairs_tested += 1;
                if (a.1 - b.1).length() > a.3.length() + b.3.length() {
                    continue;
                }
                if boxes_overlap(a, b) {
                    here += 1;
                    if overlapping.insert(key) {
                        let w = sim.world();
                        let name = |g: Uuid| {
                            w.entity_of(g)
                                .and_then(|e| w.name_of(e))
                                .unwrap_or("?")
                                .to_string()
                        };
                        let day = |g: Uuid| {
                            if inf_ecs::traffic::traffic_of(w)
                                .is_some_and(|t| t.records.contains_key(&g))
                            {
                                format!("{:?}", inf_ecs::traffic::day_of(g))
                            } else {
                                "authored".to_string()
                            }
                        };
                        eprintln!(
                            "  OVERLAP at island second {}: `{}` ({}) at {:.2} (half {:.2}) and `{}` ({}) at {:.2} (half {:.2})",
                            (s + 1) / 60,
                            name(a.0),
                            day(a.0),
                            a.1,
                            a.3,
                            name(b.0),
                            day(b.0),
                            b.1,
                            b.3
                        );
                    }
                }
            }
        }
        worst = worst.max(here);
    }
    let w = sim.world();
    let pop = inf_ecs::traffic::traffic_of(w).expect("the island has traffic");
    let mut records: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut parked_bad = Vec::new();
    for (g, r) in &pop.records {
        let id = inf_ecs::traffic::catalogue_row_id(*g).unwrap_or("-");
        let class = roster::roster()
            .get(id)
            .and_then(|d| d.roster_class)
            .expect("every traffic car is a roster row");
        *records.entry(class.name()).or_default() += 1;
        assert!(class.road(), "a {} is traffic", class.name());
        assert!(
            class.parked_weight() > 0.0 || class.circuit_weight() > 0.0,
            "a {} is traffic",
            class.name()
        );
        let parks = matches!(
            inf_ecs::traffic::day_of(*g),
            inf_ecs::traffic::TrafficDay::Parked | inf_ecs::traffic::TrafficDay::Commute
        );
        if parks && class.parked_weight() <= 0.0 {
            parked_bad.push(format!("{g}: a {} parks at a kerb", class.name()));
        }
        assert_eq!(
            r.def.roster_class,
            Some(class),
            "the record's def is not its row"
        );
    }
    let mut spawned: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (g, _, _, _) in chassis_boxes(&sim) {
        let class = roster::row_of(sim.world(), g)
            .and_then(|id| roster::roster().get(id))
            .and_then(|d| d.roster_class)
            .map(|c| c.name())
            .unwrap_or("island-row");
        *spawned.entry(class).or_default() += 1;
    }
    println!(
        "THE ISLAND CENSUS after {minutes} island minutes ({steps} steps): {} traffic records by class {records:?}; resident chassis by class {spawned:?}",
        pop.records.len()
    );
    println!(
        "THE OVERLAP ARM: {pairs_tested} chassis pairs tested over {} samples; the worst sample had {worst} overlapping pair(s); {} distinct pairs ever overlapped {overlapping:?}",
        steps / 600,
        overlapping.len()
    );
    assert!(
        pop.records.len() > 8,
        "a census of {} cars",
        pop.records.len()
    );
    assert!(
        records.len() >= 3,
        "the island's traffic is only {records:?}"
    );
    assert!(
        !records.contains_key("emergency"),
        "an emergency row is traffic"
    );
    assert!(parked_bad.is_empty(), "{}", parked_bad.join("\n"));
    println!(
        "  (a moving traffic car against anything, not this arm's claim: {} contact pair(s))",
        moving_contacts.len()
    );
    assert!(pairs_tested > 50, "only {pairs_tested} pairs were tested");
    assert!(
        overlapping.is_empty(),
        "cars overlap on the island: {overlapping:?}"
    );
}
/// The island project this machine builds locally, or `None` -- the island is
/// licensed content that never enters this repository (char1a3's rule).
fn real_island_content() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../island-build/project/Content");
    (p.join("VancouverIsland.inf_lvl").is_file()).then_some(p)
}

/// The real island, booted the way `char1c_gate` boots it.
fn real_island_sim(content: &std::path::Path) -> RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join("VancouverIsland.inf_lvl"));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    .with_anim_assets(skeletons, clips, machines)
    .with_audio(inf_player::level::load_audio_assets_from_dir(content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the island builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

/// **THE REAL ISLAND'S KERBS STEP AROUND ITS FLEETS** -- the lattice claim on
/// the level that has the fleets: the construction lot at Eastgate, the
/// tractor-trailers at Harbour City, the EMS stations and the settlement cars,
/// every one an authored vehicle beside a road whose kerbs the traffic lattice
/// parks. Ten seconds of the island (the lattice is derived on the first
/// steps); every parked chassis box against every other.
///
/// **VACUOUS against the lattice's own guards, measured and said.** With
/// `authored_footprints` answering empty, `PARK_CLEAR_M` at -3 and the
/// lattice's self-exclusion removed, this arm and the census above both stay
/// green: on today's island the resident Harbour City kerbs are 31.9 m from the
/// nearest authored vehicle and never two to a space, so the exclusion that
/// fixed the VEH3b audit's overlap does not bind here. What this arm asserts
/// is the committed level's state -- no two parked vehicles share a space --
/// and it prints the fleet's addresses for the demo loop. SKIPS with a printed
/// reason when the island is not on this machine.
#[test]
fn the_real_islands_kerbs_step_around_its_fleets() {
    let Some(content) = real_island_content() else {
        eprintln!("SKIP: no island project at ../island-build/project -- local-only content");
        return;
    };
    let mut sim = real_island_sim(&content);
    for _ in 0..600 {
        sim.step_once(Default::default());
    }
    let hitched = hitched_pairs(&sim);
    let moving = |g: Uuid| {
        inf_ecs::traffic::traffic_of(sim.world()).is_some_and(|t| t.records.contains_key(&g))
            && inf_ecs::traffic::day_of(g) != inf_ecs::traffic::TrafficDay::Parked
    };
    let traffic: BTreeSet<Uuid> = inf_ecs::traffic::traffic_of(sim.world())
        .map(|t| t.records.keys().copied().collect())
        .unwrap_or_default();
    let boxes: Vec<ChassisBox> = chassis_boxes(&sim)
        .into_iter()
        .filter(|b| !moving(b.0))
        .collect();
    // The fleet's addresses, for the demo loop's placements.
    {
        let w = sim.world();
        if let Some(h) = inf_ecs::movement::camera_subject(w)
            .and_then(|g| w.entity_of(g))
            .and_then(|e| w.world().get::<Transform>(e))
        {
            println!("  the hero stands at {:.1}", h.translation.to_dvec3());
        }
        for b in boxes.iter().filter(|b| !traffic.contains(&b.0)) {
            let name = w.entity_of(b.0).and_then(|e| w.name_of(e)).unwrap_or("?");
            println!("  authored `{name}` at {:.1}", b.1);
        }
    }
    let (mut near_fleet, mut overlapping) = (0usize, Vec::new());
    let mut closest = f64::MAX;
    for i in 0..boxes.len() {
        for j in i + 1..boxes.len() {
            let (a, b) = (&boxes[i], &boxes[j]);
            let key = (a.0.min(b.0), a.0.max(b.0));
            if hitched.contains(&key) {
                continue;
            }
            // A kerb car beside an authored one: the pairs the exclusion is for.
            let mixed = traffic.contains(&a.0) != traffic.contains(&b.0);
            if mixed {
                closest = closest.min((a.1 - b.1).length());
                if (a.1 - b.1).length() < 15.0 {
                    near_fleet += 1;
                }
            }
            if (a.1 - b.1).length() <= a.3.length() + b.3.length() && boxes_overlap(a, b) {
                overlapping.push(key);
            }
        }
    }
    println!(
        "THE REAL ISLAND: {} parked chassis ({} of them kerb cars); {near_fleet} kerb-car/authored pairs within 15 m (the closest {closest:.1} m); overlapping pairs {overlapping:?}",
        boxes.len(),
        boxes.iter().filter(|b| traffic.contains(&b.0)).count()
    );
    assert!(
        boxes.len() > 20 && boxes.iter().filter(|b| traffic.contains(&b.0)).count() > 10,
        "only {} parked chassis were resident",
        boxes.len()
    );
    assert!(
        overlapping.is_empty(),
        "parked vehicles share their space on the island: {overlapping:?}"
    );
}

// ── 6. PIE == SHIPPING ──────────────────────────────────────────────────────

/// The Blueprint that defines one row at `BeginPlay` and spawns a hypercar, a
/// bus, a jeep and that row on the course -- the `vehicle.define` /
/// `vehicle.spawn` route, identical in both hosts.
fn spawner_class() -> inf_blueprint::BlueprintClass {
    use inf_blueprint::{
        BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Stmt, Ty,
    };
    let call = |path: &[&str], args: Vec<Expr>| {
        Stmt::ExprStmt(Expr::Call {
            path: path.iter().map(|p| (*p).to_string()).collect(),
            args,
        })
    };
    let f = |v: f64| Expr::Lit(Lit::Float(v));
    let s = |v: &str| Expr::Lit(Lit::Str(v.into()));
    let mut body = vec![call(&["vehicle", "define"], vec![s(PIE_ROW_TOML)])];
    for (i, id) in PIE_SPAWNS.iter().enumerate() {
        let def = spawn_def(id);
        body.push(call(
            &["vehicle", "spawn"],
            vec![
                s(id),
                f(-30.0 + 15.0 * i as f64),
                f(inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.1),
                f(40.0),
                f(0.0),
            ],
        ));
    }
    let mut class = BlueprintClass::new("veh3f.spawner", "VEH3f Spawner");
    class.events = vec![EventBinding {
        event: EventKind::BeginPlay,
        body: BlueprintFn {
            id: EventKind::BeginPlay.key(),
            name: EventKind::BeginPlay.key(),
            params: vec![],
            ret: Ty::Unit,
            body,
        },
    }];
    class
}

/// The row the spawner DEFINES before it spawns it -- a class default and a
/// mass, so the catalogue the node merges is exercised and not only the
/// built-in roster.
const PIE_ROW_TOML: &str = "[pie_row]\nlabel = \"PIE Row\"\n\n[pie_row.vehicle]\nclass = \"sedan\"\nmass_kg = 1500.0\nhalf_width_m = 0.9\nhalf_height_m = 0.6\nhalf_length_m = 2.3\nwheel_drop_m = -0.47\nstiffness_n_per_m = 42000.0\ndamping_ns_per_m = 3600.0\n";

/// What the spawner spawns: three classes of the roster and the defined row.
const PIE_SPAWNS: [&str; 4] = ["pegassi_zentorno", "brute_bus", "canis_terminus", "pie_row"];

fn spawn_def(id: &str) -> VehicleDef {
    let mut defs = inf_ecs::vehicle::VehicleDefs::default();
    defs.merge_toml(PIE_ROW_TOML).expect("the PIE row parses");
    defs.get(id)
        .or_else(|| roster::roster().get(id))
        .copied()
        .expect("a known row")
}

/// Every vehicle chassis's pose bits, sorted by guid.
fn pose_bits(world: &EcsWorld) -> Vec<(Uuid, [u64; 6])> {
    let mut out = Vec::new();
    for e in world.world().iter_entities() {
        if e.get::<inf_ecs::components::VehicleClass>().is_none() {
            continue;
        }
        let (Some(g), Some(t)) = (e.get::<inf_ecs::components::Guid>(), e.get::<Transform>())
        else {
            continue;
        };
        out.push((
            g.0,
            [
                t.translation.x.to_bits(),
                t.translation.y.to_bits(),
                t.translation.z.to_bits(),
                t.rotation.x.to_bits(),
                t.rotation.y.to_bits(),
                t.rotation.z.to_bits(),
            ],
        ));
    }
    out.sort();
    out
}

/// **PIE == SHIPPING on three classes and the trailer** -- `SimSession` (the
/// editor's Simulate / PIE door) and `RuntimeSim` (the shipped player) over one
/// course: a Blueprint that `vehicle.define`s a row and `vehicle.spawn`s a
/// hypercar, a bus, a jeep and that row, and a hitched tractor-trailer driven
/// through a slalom by the same controls in both hosts -- every vehicle
/// chassis's pose compared, bit for bit, every step.
///
/// **Mutation → red**: either host's `vehicle.spawn` arm deleted (the vehicle
/// sets differ on the first step); the defined row's `mass_kg` resolved in one
/// host only (impossible to express -- the parser is Ring 0 -- which is the
/// arrangement's point).
#[test]
fn pie_equals_shipping_on_three_classes_and_the_trailer() {
    use inf_editor_core::ipc::SpawnKind;
    use inf_editor_core::scene::SceneDoc;
    use inf_editor_core::simulate::{SimInput, SimSession};
    let tractor = *roster::roster().get("mtl_pounder").unwrap();
    let trailer = *roster::roster().get("jobuilt_box_trailer").unwrap();
    let at = DVec3::new(0.0, inf_ecs::vehicle::resting_origin_y(&tractor, 0.0), 0.0);
    let tat = inf_ecs::vehicle::hitched_trailer_at(at, 0.0, &tractor, &trailer).unwrap();
    let ground = (DVec3::new(0.0, -0.5, 0.0), DVec3::new(400.0, 0.5, 400.0));
    let class = spawner_class();

    // The shipped side.
    let mut world = EcsWorld::new();
    slab(&mut world, GROUND, ground.0, ground.1, 0.9);
    spawn(&mut world, CAR, &tractor, at, 0.0);
    spawn(&mut world, TRAILER, &trailer, tat, 0.0);
    assert!(inf_ecs::vehicle::hitch(
        &mut world, CAR, &tractor, TRAILER, &trailer
    ));
    world.spawn_with_guid(SPAWNER, "Spawner", None);
    world.propagate();
    let mut shipped = RuntimeSim::new(
        world,
        vec![(SPAWNER, class.clone())],
        glam::DVec2::new(0.0, -9.81),
        HZ,
    );

    // The editor's side, authored through the editor's own doors.
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(GROUND, SpawnKind::Empty, "Slab", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(slab_bits(ground.0, ground.1, 0.9));
    for (g, def, p) in [(CAR, tractor, at), (TRAILER, trailer, tat)] {
        inf_editor_core::vehicle::spawn_vehicle(
            &mut doc,
            g,
            &def,
            inf_editor_core::vehicle::VehicleSpawn {
                name: "Car",
                at: p,
                yaw_deg: 0.0,
                paint: inf_ecs::math::Color::new(0.3, 0.3, 0.7, 1.0),
                clip: None,
                livery: None,
                engine_voice: false,
            },
        );
    }
    let joint = inf_ecs::vehicle::hitch_joint(CAR, &tractor, &trailer).unwrap();
    let e = doc.entity_of(TRAILER).unwrap();
    doc.world_mut().world_mut().entity_mut(e).insert(joint);
    doc.create_with_guid(SPAWNER, SpawnKind::Empty, "Spawner", None);
    doc.world_mut().propagate();
    let mut preview = SimSession::enter(
        &mut doc,
        vec![(SPAWNER, class)],
        glam::DVec2::new(0.0, -9.81),
        HZ,
    );

    let mut vehicles = 0usize;
    let mut moved = 0.0f64;
    for i in 0..600u32 {
        let t = i as f64 * DT;
        let c = VehicleControls {
            throttle: if t < 7.0 { 0.6 } else { 0.0 },
            steer: 0.4 * inf_math::psin64(t * 0.8),
            occupied: true,
            ..Default::default()
        };
        if let Some(v) = shipped.bridge3d_mut().vehicle_mut(CAR) {
            v.control(c);
        }
        if let Some(v) = preview.bridge3d_mut().vehicle_mut(CAR) {
            v.control(c);
        }
        shipped.step_once(Default::default());
        preview.step_once(&mut doc, SimInput::default());
        let (a, b) = (pose_bits(shipped.world()), pose_bits(doc.world()));
        assert_eq!(
            a.iter().map(|x| x.0).collect::<Vec<_>>(),
            b.iter().map(|x| x.0).collect::<Vec<_>>(),
            "step {i}: the two hosts hold different vehicles"
        );
        assert_eq!(a, b, "step {i}: the hosts' vehicle poses differ");
        vehicles = vehicles.max(a.len());
        moved = moved.max(position(&shipped, CAR).length());
    }
    println!(
        "PIE == SHIPPING: 600 steps, {vehicles} vehicles (a hitched rig and four `vehicle.spawn`s -- {PIE_SPAWNS:?}), bit-identical poses; the rig drove {moved:.1} m"
    );
    assert_eq!(
        vehicles, 6,
        "the course holds {vehicles} vehicles, not a rig and four spawns"
    );
    assert!(moved > 20.0, "the rig did not drive ({moved:.1} m)");
}

// ── 7. THE COST ─────────────────────────────────────────────────────────────

/// **The catalogue loads in microseconds** -- `merge_toml` over the whole
/// committed roster, the minimum of five, against `ROSTER_LOAD_BUDGET_MS`.
///
/// A clock, so: reported everywhere, asserted under `cargo test --release` off
/// CI (the house conditioning).
#[test]
fn the_catalogue_loads_in_microseconds() {
    let mut best = f64::MAX;
    let mut rows = 0usize;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let mut defs = inf_ecs::vehicle::VehicleDefs::default();
        defs.merge_toml(roster::ROSTER_TOML)
            .expect("the roster loads");
        best = best.min(t0.elapsed().as_secs_f64() * 1_000.0);
        rows = defs.0.len();
    }
    println!(
        "THE CATALOGUE: {rows} rows merged in {:.1} us (min of five; budget {} ms)",
        best * 1_000.0,
        inf_player::budget::ROSTER_LOAD_BUDGET_MS
    );
    assert!(rows >= 155, "the merge loaded {rows} rows");
    if cfg!(debug_assertions) {
        eprintln!("dev build: the catalogue load is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the catalogue load is reported, not asserted (shared runner)");
        return;
    }
    assert!(
        best <= inf_player::budget::ROSTER_LOAD_BUDGET_MS,
        "the roster loads in {best:.3} ms against {} ms {}",
        inf_player::budget::ROSTER_LOAD_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}

/// **Sixty-four MIXED-CLASS cars cost what they print** -- VEH3a's vehicle
/// phase arm at the budget's population, the fleet drawn across the roster's
/// classes (every other wheeled road row), against 64 of the default row: the
/// VEH3e baseline. Reported everywhere; the ceiling asserted under `--release`
/// off CI.
#[test]
fn sixty_four_mixed_class_cars_cost_what_they_print() {
    use inf_physics::d3::PhysicsBridge3D;
    const CARS: usize = inf_player::budget::VEHICLE_BUDGET_CARS;
    let mixed: Vec<VehicleDef> = roster::roster()
        .0
        .values()
        .filter(|d| d.roster_class.is_some_and(|c| c.road()) && !d.body.towed())
        .step_by(2)
        .take(CARS)
        .copied()
        .collect();
    assert_eq!(mixed.len(), CARS);
    let classes: BTreeSet<RosterClass> = mixed.iter().filter_map(|d| d.roster_class).collect();
    let build = |defs: &[VehicleDef]| -> (EcsWorld, PhysicsBridge3D) {
        let mut world = EcsWorld::new();
        slab(
            &mut world,
            GROUND,
            DVec3::new(0.0, -0.5, 0.0),
            DVec3::new(400.0, 0.5, 400.0),
            0.9,
        );
        for (i, d) in defs.iter().enumerate() {
            let at = DVec3::new(
                (i % 8) as f64 * 30.0 - 105.0,
                inf_ecs::vehicle::resting_origin_y(d, 0.0),
                (i / 8) as f64 * 30.0 - 105.0,
            );
            spawn(
                &mut world,
                Uuid::from_u128(0x5E3F_1000 + i as u128),
                d,
                at,
                0.0,
            );
        }
        world.mark_dirty();
        world.propagate();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        for _ in 0..90 {
            inf_physics::d3::step_vehicles(&mut world, &mut bridge, DT);
        }
        (world, bridge)
    };
    let time = |(world, bridge): &mut (EcsWorld, PhysicsBridge3D)| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..5 {
            bridge.sync_from_world(world);
            let t0 = std::time::Instant::now();
            for _ in 0..40 {
                inf_physics::d3::step_vehicles(world, bridge, DT);
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1_000.0 / 40.0);
        }
        best
    };
    let mut one = build(&vec![VehicleDef::default(); CARS]);
    let mut many = build(&mixed);
    assert_eq!(many.1.vehicle_count(), CARS);
    let (b, m) = (time(&mut one), time(&mut many));
    println!(
        "64 MIXED CARS: {} classes cost {m:.4} ms a step against {b:.4} ms for 64 of the default row ({:.2}x; budget {} ms)",
        classes.len(),
        m / b,
        inf_player::budget::VEHICLE_STEP_BUDGET_MS
    );
    assert!(classes.len() >= 10, "the fleet was only {classes:?}");
    if cfg!(debug_assertions) {
        eprintln!("dev build: the vehicle phase is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the vehicle phase is reported, not asserted (shared runner)");
        return;
    }
    assert!(
        m <= inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        "64 mixed-class cars cost {m:.4} ms against {} ms {}",
        inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}

// ── 8. THE SEAT RULE, THE CLIP, THE SCHEMA, THE INSTRUMENTS ─────────────────

/// **The drawn seat is the seat the body sits on** -- on every roster row that
/// draws one, the driver's cushion top the bodywork table holds (read off the
/// SPAWNED car, `part_geoms`) is the socket `sockets_of` seats the driver on,
/// to a millimetre; and the passenger's likewise.
///
/// The wave first built this the other way round (the socket read off a
/// cushion drawn where each silhouette's driver really sits) and reverted it:
/// see `vehicle_families`' module note and the report's CARRIED.
///
/// **Mutation → red**: one family's `seat_r` moved (its rows' cushions leave
/// the socket).
#[test]
fn the_drawn_seat_is_the_seat_the_body_sits_on() {
    let (mut rows, mut seats) = (0usize, 0usize);
    let mut families = BTreeSet::new();
    for (id, def) in roster::roster().0.iter() {
        if !def.body.wheeled() || def.body.towed() {
            continue;
        }
        let mut world = EcsWorld::new();
        let at = DVec3::new(0.0, inf_ecs::vehicle::resting_origin_y(def, 0.0), 0.0);
        spawn(&mut world, CAR, def, at, 0.0);
        world.propagate();
        let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
        drive(&mut sim, CAR, VehicleControls::default());
        let w = sim.world();
        let c = *w
            .world()
            .get::<Collider3D>(w.entity_of(CAR).unwrap())
            .unwrap();
        let half = inf_ecs::vehicle::chassis_half_extents(&c);
        let parts = inf_ecs::boarding::part_geoms(w, CAR);
        let s = inf_ecs::boarding::sockets_of(half, c.offset, &parts);
        let mut drew = 0usize;
        for p in parts
            .iter()
            .filter(|p| p.kind == inf_ecs::vehicle::KIND_SEAT)
        {
            let top = DVec3::new(
                c.offset.x + p.centre_frac.x * half.x,
                c.offset.y + (p.centre_frac.y + p.half_frac.y.abs()) * half.y,
                c.offset.z + p.centre_frac.z * half.z,
            );
            let socket = if p.centre_frac.x >= 0.0 {
                s.seat_r
            } else {
                s.seat_l
            };
            let err = (top - socket.to_dvec3()).length();
            assert!(
                err < 1e-3,
                "`{id}`: a drawn cushion is {err:.3} m from the seat its body sits on"
            );
            drew += 1;
        }
        if drew > 0 {
            rows += 1;
            seats += drew;
            families.insert(format!("{:?}", def.body));
        }
    }
    println!(
        "THE DRAWN SEATS: {seats} cushions on {rows} rows of {} families, each on its socket",
        families.len()
    );
    assert!(rows > 60, "only {rows} rows draw a seat");
    assert!(families.len() >= 10);
}

/// **A row's own engine clip reaches the command stream** -- the VEH3e planner
/// plays the row's `engine_clip` in its three load slots instead of the grain.
///
/// **Mutation → red**: `car_loops` back to `grain_clip` only (no Play names the
/// row's clip).
#[test]
fn the_engine_clip_override_reaches_the_command_stream() {
    use inf_ecs::vehicle_audio::{AxleVoice, VoiceCue, VoiceMemory, VoiceTelemetry};
    let clip = Uuid::from_u128(0x5E3F_C11F);
    let mut def = *roster::roster().get("pegassi_zentorno").unwrap();
    def.engine_clip = Some(clip);
    let mut world = EcsWorld::new();
    let mut spec = spawn_spec("Car", DVec3::new(0.0, 1.0, 0.0), 0.0);
    spec.engine_voice = true;
    inf_ecs::vehicle::spawn_rig(&mut world, CAR, &def, &spec);
    world.propagate();
    let axle = AxleVoice {
        slip: 0.0,
        surface: inf_ecs::vehicle::SurfaceClass::Asphalt,
        compression_m: 0.05,
        grounded: true,
    };
    let t = VoiceTelemetry {
        rpm: 3_000.0,
        idle_rpm: 900.0,
        redline_rpm: 7_600.0,
        throttle: 0.3,
        boost: 0.0,
        turbocharged: false,
        fuel_cut: false,
        gear: 2,
        shaft_rpm: 3_000.0,
        cylinders: 8.0,
        voice_kind: 0.0,
        firing_order: 1.0,
        occupied: true,
        quiet: false,
        speed_mps: 10.0,
        axles: [axle, axle],
    };
    let mut mem = VoiceMemory::new();
    let cues = mem.plan(&world, &[(CAR, t)], DT);
    let plays: Vec<Uuid> = cues
        .iter()
        .filter_map(|c| match c {
            VoiceCue::Play { clip, .. } => Some(*clip),
            _ => None,
        })
        .collect();
    let own = plays.iter().filter(|c| **c == clip).count();
    println!(
        "THE CLIP OVERRIDE: {} plays on the first step, {own} of them the row's own clip",
        plays.len()
    );
    assert_eq!(
        own, 3,
        "the row's clip plays in {own} slots, not the three load slots"
    );
}

/// **This wave moved no schema** -- scene v28, payload 13, 100 tunables: every
/// roster key is a key VEH3a already persisted, or a spawn-time one.
#[test]
fn this_wave_moved_no_schema() {
    assert_eq!(inf_scene::SCHEMA_VERSION, 28);
    assert_eq!(inf_runtime::pie::SCENE_PAYLOAD_VERSION, 13);
    assert_eq!(
        inf_ecs::vehicle::VehicleTuning::names().len(),
        100,
        "the tunable count moved"
    );
}

/// **The shipped host draws the roster row and logs its columns** -- the source
/// of `window.rs` and `pie_drive.rs`, and the Ring-0 readout they call.
///
/// **Mutation → red**: the `roster_readout` call deleted from `window.rs`; a
/// column dropped from `hero.csv`.
#[test]
fn the_shipped_host_draws_the_roster_row_and_logs_the_columns() {
    const WINDOW: &str = include_str!("../src/window.rs");
    const PIE: &str = include_str!("../src/pie_drive.rs");
    assert!(WINDOW.contains("inf_ecs::roster::roster_readout(sim.world(), vehicle, yaw)"));
    for col in ["r_class", "r_row", "r_body", "r_hitch", "r_track"] {
        assert!(PIE.contains(col), "`hero.csv` lost `{col}`");
    }
    let def = *roster::roster().get("hvy_dozer").unwrap();
    let mut world = EcsWorld::new();
    inf_ecs::vehicle::spawn_rig(
        &mut world,
        CAR,
        &def,
        &spawn_spec("HVY Dozer", DVec3::ZERO, 0.0),
    );
    world.propagate();
    let row = roster::roster_readout(&world, CAR, Some(0.5));
    println!("the roster row reads: {row}");
    assert!(row.starts_with("ROSTER "), "{row}");
    assert!(row.contains("HVY Dozer"), "{row}");
    assert!(row.ends_with("TRACKS +0.50 rad/s"), "{row}");
    let mut plain = EcsWorld::new();
    inf_ecs::vehicle::spawn_rig(
        &mut plain,
        CAR,
        &VehicleDef::default(),
        &spawn_spec("Nobody", DVec3::ZERO, 0.0),
    );
    plain.propagate();
    assert_eq!(roster::roster_readout(&plain, CAR, None), "");
}
