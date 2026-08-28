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
        "Climber",
        DVec3::new(
            200.0,
            inf_editor_core::vehicle::resting_origin_y(&def, ground),
            z,
        ),
        0.0,
        &def,
        Color::new(0.6, 0.1, 0.1, 1.0),
        None,
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
    // The two rows really are two cars: the truck is heavier and geared to a
    // lower top speed, so it must not out-run the saloon on a hill.
    assert!(
        ran[1].1 > ran[0].1,
        "the truck ({} kg) is not heavier than the sedan ({} kg)",
        ran[1].1,
        ran[0].1
    );
    assert!(
        ran[1].2 < ran[0].2,
        "the truck ran {} m and the sedan {} m — the catalogue's two rows drive \
         identically, so its second silhouette is a rename",
        ran[1].2,
        ran[0].2
    );
}
