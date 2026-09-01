//! **THE GRADE PROMISE, DRIVEN** (island wave VEH1a).
//!
//! Wave I7 planned the island's roads under a grade ceiling and audited them
//! against it, and its ledger says the audit *"says a car could climb it"*. That
//! was an inference from a number, not a measurement of a car: nothing in this
//! repository had ever put a vehicle on a slope at that grade and asked.
//!
//! This file asks. The grade is read off the **committed route layers** — the
//! same vertices `player_start` and the fleet placement read, so it is the
//! island's own worst planned stretch and not a number chosen to pass — and the
//! car is the **catalogue's own sedan**, tuned by the `VehicleClass` the level
//! authors, driven up a ramp of exactly that grade through the same
//! `step_character_movement` / `step_vehicles` pair both hosts run.
//!
//! # Why a ramp and not the island
//!
//! The island's terrain is a build artifact of one machine (`inf-island`'s own
//! portability note) and CI never builds the full one. A ramp at the measured
//! grade is the same *question* with none of that: the force balance a car
//! climbs a slope by is a function of the slope, the mass and the tuning, and
//! all three are here. What a ramp cannot answer is whether the island's road
//! *surface* is drivable where it bends, and `island_gate`'s drive arm is what
//! answers that.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{Terrain, Transform};
use inf_ecs::math::Color;
use inf_ecs::vehicle::VehicleControls;
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_physics::d3::PhysicsBridge3D;
use inf_terrain::TerrainData;

const DT: f64 = 1.0 / 60.0;
const TERRAIN: Uuid = Uuid::from_u128(0x7614_2001);
const CAR: Uuid = Uuid::from_u128(0x7614_2002);

/// **The worst grade wave I7's own AUDIT found on the shipped island**, as
/// against the worst the planner *plans*.
///
/// The two are different numbers about different things and the wave's brief
/// names the second, so both are driven below:
///
/// * the **planned** worst is rise-over-run between adjacent route vertices, and
///   the planner routes under `RoadSpec::max_grade × PLAN_GRADE_MARGIN` — so it
///   comes out at or under the ceiling by construction (measured: 0.0810 on the
///   island's 0.080 ceiling, 0.0993 on the fixture's 0.100);
/// * the **audited** worst is `inf_island::grade_audit` re-sampling the draped
///   road against the *built terrain* at the recipe's own step, which is finer
///   than the route's vertices and includes the two places where routes cross at
///   different elevations. I7 measured **0.108 over 2 442 stretches, five of
///   them past the 0.080 ceiling**, and `samples/island/README.md` records it.
///
/// It is a literal here rather than a re-derivation because re-deriving it needs
/// the 342 MB `.inf_terrain` CI never builds — the same reason the island's
/// level is authored from committed design alone. What keeps it honest is that
/// it is the *pessimistic* half: driving the audited number is driving a slope
/// steeper than anything the planner will admit.
const I7_AUDITED_WORST_GRADE: f64 = 0.108;

/// A ramp 512 m long at `grade` (rise over run), climbing along `+Z`.
const MPS: f64 = 1.0;
const TILE_RES: u32 = 65;
const TILES: i32 = 8;

fn ramp(grade: f64) -> TerrainData {
    let mut data = TerrainData::new(TILE_RES, MPS);
    for tz in 0..TILES {
        for tx in 0..TILES {
            data.author_tile((tx, tz), |_, z| 10.0 + grade * z);
        }
    }
    data
}

/// The **worst grade between adjacent vertices** of a committed island's own
/// planned routes, and where it is.
///
/// The vertices carry the ground each was planned at, which is the one committed
/// elevation on this island — the same fact `player_start` has read since I7.
fn worst_planned_grade(rel: &str) -> Option<(String, f64, f64)> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(rel);
    let recipe = inf_island::IslandRecipe::load(&path).ok()?;
    let design = inf_island::read_design(&recipe).ok()?;
    let mut worst = 0.0f64;
    let mut n = 0usize;
    for r in &design.routes {
        for w in r.points.windows(2) {
            let run = ((w[1].x - w[0].x).powi(2) + (w[1].z - w[0].z).powi(2)).sqrt();
            if run <= 1e-9 {
                continue;
            }
            n += 1;
            worst = worst.max((w[1].y - w[0].y).abs() / run);
        }
    }
    (n > 0).then(|| (recipe.name.clone(), worst, recipe.roads.max_grade))
}

/// Build a world holding a ramp at `grade` with one catalogue car on it, facing
/// **up** the slope.
fn ramp_world(grade: f64, id: &str) -> (SceneDoc, PhysicsBridge3D, f64) {
    let def = *inf_editor_core::vehicle::island_vehicles()
        .get(id)
        .unwrap_or_else(|| panic!("the catalogue has no `{id}` row"));
    let mut doc = SceneDoc::new();
    let e = doc.world_mut().spawn_with_guid(TERRAIN, "Ramp", None);
    doc.world_mut().world_mut().entity_mut(e).insert(Terrain {
        meters_per_sample: MPS,
        tile_resolution: TILE_RES,
        data: ramp(grade),
        ..Terrain::default()
    });
    // At the foot, nose up the hill (`+Z` is forward and the ramp climbs +Z).
    let z = 16.0;
    let ground = 10.0 + grade * z;
    inf_editor_core::vehicle::spawn_vehicle(
        &mut doc,
        CAR,
        &def,
        inf_editor_core::vehicle::VehicleSpawn {
            name: "Climber",
            at: DVec3::new(
                200.0,
                inf_editor_core::vehicle::resting_origin_y(&def, ground),
                z,
            ),
            yaw_deg: 0.0,
            paint: Color::new(0.6, 0.1, 0.1, 1.0),
            clip: None,
        },
    );
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(doc.world());
    let mass =
        def.half_extents.x * def.half_extents.y * def.half_extents.z * 8.0 * def.density_kg_m3;
    (doc, bridge, mass)
}

/// Build a world holding a **big flat static floor** with one catalogue car on
/// it, facing `+Z`.
///
/// A collider and not a terrain, and the reason is a measurement: the feel table
/// runs a coupe at full throttle for a minute, which is three kilometres, and the
/// ramp fixture's terrain is 512 m square. The first cut used it and the coupe
/// simply drove off the end of the world — it reported a stop that never happened
/// because it was in free fall, `inf` metres from where the brakes went on.
fn flat_world(id: &str) -> (SceneDoc, PhysicsBridge3D, f64) {
    let def = *inf_editor_core::vehicle::island_vehicles()
        .get(id)
        .unwrap_or_else(|| panic!("the catalogue has no `{id}` row"));
    let mut doc = SceneDoc::new();
    let e = doc.world_mut().spawn_with_guid(TERRAIN, "Floor", None);
    doc.world_mut().world_mut().entity_mut(e).insert(
        inf_ecs::components::Transform::from_translation(DVec3::new(0.0, 9.5, 0.0)),
    );
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(inf_ecs::components::RigidBody3D {
            kind: inf_ecs::components::BodyKind3D::Static,
            ..Default::default()
        });
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(inf_ecs::components::Collider3D {
            shape_kind: inf_ecs::components::ColliderShape3DKind::Box,
            half_extents: inf_ecs::math::Vec3d::new(4_000.0, 0.5, 4_000.0),
            friction: 0.9,
            ..Default::default()
        });
    inf_editor_core::vehicle::spawn_vehicle(
        &mut doc,
        CAR,
        &def,
        inf_editor_core::vehicle::VehicleSpawn {
            name: "Runner",
            at: DVec3::new(
                0.0,
                inf_editor_core::vehicle::resting_origin_y(&def, 10.0),
                -3_500.0,
            ),
            yaw_deg: 0.0,
            paint: Color::new(0.1, 0.1, 0.6, 1.0),
            clip: None,
        },
    );
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(doc.world());
    let mass =
        def.half_extents.x * def.half_extents.y * def.half_extents.z * 8.0 * def.density_kg_m3;
    (doc, bridge, mass)
}

/// One fixed step, in the slot both hosts run it in.
fn step(world: &mut EcsWorld, bridge: &mut PhysicsBridge3D, controls: VehicleControls) {
    bridge.sync_from_world(world);
    if let Some(v) = bridge.vehicle_mut(CAR) {
        v.control(controls);
    }
    inf_physics::d3::step_character_movement(world, bridge, DT);
    inf_physics::d3::step_vehicles(world, bridge, DT);
    bridge.step(DT);
    bridge.write_back_into(world);
    world.propagate();
}

fn car_at(doc: &SceneDoc) -> DVec3 {
    let e = doc.world().entity_of(CAR).expect("the car");
    doc.world()
        .world()
        .get::<Transform>(e)
        .map(|t| t.translation.to_dvec3())
        .expect("with a transform")
}

/// **THE ANSWER.** The catalogue's sedan climbs the steepest stretch either
/// committed island plans, and here is how fast.
///
/// Three claims, and the third is what makes the first two mean something:
///
/// 1. the grade is read off committed content and printed with the ceiling the
///    recipe set, so a reader can see whether the planner met its own promise;
/// 2. the sedan gains real height on a ramp at that grade under full throttle;
/// 3. and it does **not** climb an absurd one — a control at 0.60 (31°), which
///    an arm without it would pass for a car sliding uphill on a bug.
#[test]
fn the_catalogue_sedan_climbs_the_grade_the_island_was_audited_at() {
    let mut steepest = 0.0f64;
    for rel in [
        "samples/island-fixture/island.toml",
        "samples/island/island.toml",
    ] {
        let Some((name, worst, ceiling)) = worst_planned_grade(rel) else {
            println!("SKIP: no {rel} in this tree");
            continue;
        };
        println!(
            "THE GRADE: {name} plans a worst stretch of {worst:.4} against its \
             own {ceiling:.3} ceiling"
        );
        steepest = steepest.max(worst);
    }
    assert!(
        steepest > 0.0,
        "no committed island in this tree, so there is no grade to drive"
    );
    // **Drive the AUDITED worst, not the planned one.** The planner routes under
    // its own ceiling by construction; the audit re-samples the draped road
    // against the built terrain and found 0.108 where two routes cross. The
    // steeper of the two is the honest slope to ask a car about.
    println!(
        "THE GRADE: wave I7's audit of the draped road found \
         {I7_AUDITED_WORST_GRADE:.4} — steeper than anything the planner emits, \
         and the number this arm drives"
    );
    let steepest = steepest.max(I7_AUDITED_WORST_GRADE);

    // ── THE CLIMB ──
    let (mut doc, mut bridge, mass) = ramp_world(steepest, "sedan");
    // Settle on its springs first, so the height gained below is a climb and not
    // a suspension unloading.
    for _ in 0..90 {
        step(doc.world_mut(), &mut bridge, VehicleControls::default());
    }
    let start = car_at(&doc);
    let mut top_speed = 0.0f64;
    for _ in 0..900 {
        step(
            doc.world_mut(),
            &mut bridge,
            VehicleControls {
                throttle: 1.0,
                ..Default::default()
            },
        );
        if let Some(v) = bridge
            .body_of(CAR)
            .and_then(|b| bridge.world().body_linvel(b))
        {
            top_speed = top_speed.max(v.length());
        }
    }
    let end = car_at(&doc);
    let climbed = end.y - start.y;
    let along = end.z - start.z;
    println!(
        "THE CLIMB: a {mass:.0} kg sedan on a {steepest:.4} grade covered \
         {along:.1} m in fifteen seconds and gained {climbed:.2} m, topping out \
         at {top_speed:.2} m/s ({:.0} km/h)",
        top_speed * 3.6
    );
    assert!(
        along > 50.0,
        "the sedan covered {along} m up a {steepest} grade in fifteen seconds of \
         full throttle — the grade the island's roads are planned to is not \
         drivable, and I7's `a car could climb it` is wrong"
    );
    assert!(
        climbed > along * steepest * 0.8,
        "it gained {climbed} m over {along} m of a {steepest} grade, which is \
         less height than the slope it drove along has"
    );

    // ── THE CONTROL ── a grade no road generator would admit. Without it, an
    //    arm that passed because gravity was pointing the wrong way would read
    //    exactly the same.
    let (mut doc, mut bridge, _) = ramp_world(0.60, "sedan");
    for _ in 0..90 {
        step(doc.world_mut(), &mut bridge, VehicleControls::default());
    }
    let start = car_at(&doc);
    for _ in 0..900 {
        step(
            doc.world_mut(),
            &mut bridge,
            VehicleControls {
                throttle: 1.0,
                ..Default::default()
            },
        );
    }
    let wall = car_at(&doc).z - start.z;
    println!("THE CONTROL: on a 0.60 grade the same car covered {wall:.1} m");
    assert!(
        wall < along * 0.5,
        "the sedan covered {wall} m up a 0.60 grade against {along} m up a \
         {steepest} one — a car that climbs a thirty-one-degree slope as easily \
         as a six-degree one is not climbing, it is being pushed"
    );
}

/// The **truck** climbs it too, and it is the heavier, shorter-geared row — so
/// the catalogue's second silhouette is not a sedan with a different name.
#[test]
fn the_catalogue_truck_climbs_it_too_and_it_is_a_different_car() {
    if worst_planned_grade("samples/island/island.toml").is_none()
        && worst_planned_grade("samples/island-fixture/island.toml").is_none()
    {
        println!("SKIP: no committed island in this tree");
        return;
    }
    let worst = I7_AUDITED_WORST_GRADE;
    let mut ran: Vec<(String, f64, f64, f64)> = Vec::new();
    for id in ["sedan", "truck"] {
        let (mut doc, mut bridge, mass) = ramp_world(worst, id);
        for _ in 0..90 {
            step(doc.world_mut(), &mut bridge, VehicleControls::default());
        }
        let start = car_at(&doc);
        for _ in 0..600 {
            step(
                doc.world_mut(),
                &mut bridge,
                VehicleControls {
                    throttle: 1.0,
                    ..Default::default()
                },
            );
        }
        let end = car_at(&doc);
        ran.push((id.to_string(), mass, end.z - start.z, end.y - start.y));
    }
    for (id, mass, along, up) in &ran {
        println!(
            "THE FLEET ON A {worst:.4} GRADE: {id} ({mass:.0} kg) ran {along:.1} \
             m and climbed {up:.2} m in ten seconds"
        );
        assert!(
            *along > 30.0,
            "{id} covered {along} m up the island's worst planned grade"
        );
    }
    // The two rows really are two cars.
    assert!(
        ran[1].1 > ran[0].1,
        "the truck ({} kg) is not heavier than the sedan ({} kg)",
        ran[1].1,
        ran[0].1
    );
    // **AND THE TWO ROWS CLIMB IT DIFFERENTLY**, which is all this arm asserts
    // about the ORDER — deliberately, and the history is the reason.
    //
    // Under P29.7's flat drive curve the saloon was the faster climber because
    // the pickup was simply heavier with no more force. When VEH2a gave the
    // pickup a diesel and made the saloon front-wheel drive, the answer inverted
    // (124.4 m against 105.1) and this arm was inverted with it. Then the driver
    // aids became feed-forward torque caps instead of feedback controllers, and
    // it inverted BACK (130.8 against 144.8), because a front-drive car climbing
    // a hill is exactly the case a good traction controller rescues.
    //
    // Two flips inside one wave is a knife edge, and an arm that pins which side
    // of it the model is on is an arm that reds on every tuning pass without
    // telling anyone anything. What is stable, and what a fleet needs to be true,
    // is that the two rows are not one car: they climb the same hill by a margin
    // no rename could produce.
    assert!(
        (ran[1].2 - ran[0].2).abs() > 8.0,
        "the pickup ran {} m and the saloon {} m up the same grade — a difference \
         of {:.1} m is two names on one car",
        ran[1].2,
        ran[0].2,
        (ran[1].2 - ran[0].2).abs()
    );
}

// ── THE FEEL TABLE (island wave VEH2a) ──────────────────────────────────────

/// What one catalogue row is **specified** to do, and what this file measures it
/// doing.
///
/// A spec row rather than a blessed measurement: each bound is a claim about the
/// *kind of vehicle* the row is meant to be, written before the run and wide
/// enough that a tuning nudge does not red the gate — and narrow enough that a
/// coupe that accelerates like a van, or a van that stops like a coupe, does.
///
/// `sprint_to_mps` is 100 km/h where the class's own limiter can reach it. Two
/// rows cannot: the pickup is limited to 27 m/s (97 km/h) and would be measured
/// against a speed it is not allowed to have, so they are specified against 80 %
/// of their own limiter instead and the row says so. Measuring every class
/// against a number two of them cannot reach is how a gate ends up asserting the
/// governor rather than the engine.
struct Spec {
    id: &'static str,
    /// The speed the sprint is timed to, m/s.
    sprint_to_mps: f64,
    /// The most seconds that sprint may take.
    sprint_max_s: f64,
    /// …and the least, so a row cannot pass by becoming a rocket.
    sprint_min_s: f64,
    /// The most metres it may take to stop from `sprint_to_mps`.
    brake_max_m: f64,
    /// …and the least, so a class cannot pass by stopping like a wall.
    brake_min_m: f64,
    /// The band the achieved top speed must sit in, as a fraction of the class's
    /// own `max_speed_mps`. A governor that never engaged, or one that engaged
    /// early, both fail here.
    top_frac: (f64, f64),
}

/// The five rows' specs. **Chosen from what each vehicle IS**, not from a run:
/// a 1 374 kg rear-drive coupe with 420 N·m is a five-second car, a 3 296 kg
/// diesel van is not, and a van's brakes work on a van's mass.
const SPECS: [Spec; 5] = [
    Spec {
        id: "sports",
        sprint_to_mps: 27.78,
        sprint_max_s: 8.0,
        sprint_min_s: 2.5,
        brake_max_m: 46.0,
        brake_min_m: 20.0,
        top_frac: (0.88, 1.02),
    },
    Spec {
        id: "sedan",
        sprint_to_mps: 27.78,
        sprint_max_s: 16.0,
        sprint_min_s: 4.0,
        brake_max_m: 60.0,
        brake_min_m: 22.0,
        top_frac: (0.88, 1.02),
    },
    Spec {
        id: "suv",
        sprint_to_mps: 27.78,
        sprint_max_s: 16.0,
        sprint_min_s: 4.0,
        brake_max_m: 70.0,
        brake_min_m: 24.0,
        top_frac: (0.88, 1.02),
    },
    // Limited to 32 m/s, so 100 km/h is reachable but only just; timed to it all
    // the same, because a van that cannot reach a motorway speed is a van
    // nobody would drive on the island's circuit.
    Spec {
        id: "van",
        sprint_to_mps: 25.6,
        sprint_max_s: 26.0,
        sprint_min_s: 6.0,
        brake_max_m: 85.0,
        brake_min_m: 26.0,
        top_frac: (0.88, 1.02),
    },
    // Limited to 27 m/s (97 km/h): timed to 80 % of its own limiter, which is
    // the only honest thing to time a governed vehicle to.
    Spec {
        id: "truck",
        sprint_to_mps: 21.6,
        sprint_max_s: 18.0,
        sprint_min_s: 4.0,
        brake_max_m: 55.0,
        brake_min_m: 18.0,
        top_frac: (0.88, 1.02),
    },
];

/// **THE FEEL TABLE**: every catalogue row's sprint, stop and top speed,
/// measured on flat ground and bounded by its own spec row.
///
/// The numbers this wave exists to be judged on. A model can be described in any
/// amount of prose and still produce a car that reaches sixty in a minute; this
/// is the arm that would notice.
///
/// Flat ground and the same `ramp_world` fixture at grade zero, so the rig, the
/// step order and the tuning are exactly the ones the grade arms use.
#[test]
fn every_catalogue_row_sprints_stops_and_tops_out_inside_its_own_spec() {
    for spec in &SPECS {
        let (mut doc, mut bridge, mass) = flat_world(spec.id);
        let class = *inf_editor_core::vehicle::island_vehicles()
            .get(spec.id)
            .expect("the row");
        let limiter = class.class.max_speed_mps;
        for _ in 0..90 {
            step(doc.world_mut(), &mut bridge, VehicleControls::default());
        }
        let speed = |b: &PhysicsBridge3D| -> f64 {
            b.body_of(CAR)
                .and_then(|body| b.world().body_linvel(body))
                .map(|v| DVec3::new(v.x, 0.0, v.z).length())
                .unwrap_or(0.0)
        };

        // ── the sprint ──
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        let (mut sprint_s, mut top) = (f64::INFINITY, 0.0f64);
        for i in 0..3_600 {
            step(doc.world_mut(), &mut bridge, full);
            let v = speed(&bridge);
            top = top.max(v);
            if v >= spec.sprint_to_mps && sprint_s.is_infinite() {
                sprint_s = (i + 1) as f64 * DT;
            }
        }

        // ── the stop, from the speed the sprint was timed to ──
        while speed(&bridge) > spec.sprint_to_mps {
            step(
                doc.world_mut(),
                &mut bridge,
                VehicleControls {
                    brake: 1.0,
                    ..Default::default()
                },
            );
        }
        let from = car_at(&doc);
        let mut brake_m = f64::INFINITY;
        for _ in 0..1_800 {
            step(
                doc.world_mut(),
                &mut bridge,
                VehicleControls {
                    brake: 1.0,
                    ..Default::default()
                },
            );
            if speed(&bridge) < 0.5 {
                brake_m = (car_at(&doc) - from).length();
                break;
            }
        }

        println!(
            "THE FEEL TABLE: {:>6} ({:>4.0} kg) 0-{:.0} km/h in {:>5.2} s, stops \
             in {:>5.1} m, tops out at {:>5.1} m/s ({:.0} km/h) of a {:.0} m/s \
             limiter",
            spec.id,
            mass,
            spec.sprint_to_mps * 3.6,
            sprint_s,
            brake_m,
            top,
            top * 3.6,
            limiter
        );
        assert!(
            sprint_s >= spec.sprint_min_s && sprint_s <= spec.sprint_max_s,
            "{}: 0-{:.0} km/h took {sprint_s:.2} s, outside its spec of \
             {:.1}..{:.1}",
            spec.id,
            spec.sprint_to_mps * 3.6,
            spec.sprint_min_s,
            spec.sprint_max_s
        );
        assert!(
            brake_m >= spec.brake_min_m && brake_m <= spec.brake_max_m,
            "{}: stopping from {:.0} km/h took {brake_m:.1} m, outside its spec \
             of {:.0}..{:.0}",
            spec.id,
            spec.sprint_to_mps * 3.6,
            spec.brake_min_m,
            spec.brake_max_m
        );
        assert!(
            top >= limiter * spec.top_frac.0 && top <= limiter * spec.top_frac.1,
            "{}: topped out at {top:.1} m/s against a {limiter:.0} m/s limiter — \
             outside {:.2}..{:.2} of it, so either the governor never engaged or \
             it engaged early",
            spec.id,
            spec.top_frac.0,
            spec.top_frac.1
        );
    }
}
