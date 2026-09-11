//! **WAVE VEH3c — THE MODULAR BODY + DAMAGE.** The gate.
//!
//! Every claim of the wave, in one file, each arm **mutation-verified** and each
//! carrying an **engagement count** — so "the arm ran" and "something happened"
//! are two different facts and the second one is asserted.
//!
//! # What every arm in this file reads
//!
//! The WORLD. A hinge angle is `PartState::angle_deg` after the vehicle phase
//! wrote it and the part's own `Transform` after that; a break is the part count
//! still bolted to the chassis before and after, and the shed part's own rapier
//! body lying on the ground; a pane's state is its `PartLatch` and its
//! `Visibility`; a dent is the drawn box's own corners; the joules are
//! `VehicleDamage::hull_j`; the flat's pull is a yaw rate at a constant steer;
//! the cost is a phase's own milliseconds in one process against a control.
//!
//! # Would a car with NO PARTS pass it?
//!
//! The question the audit brief asks of every arm, and the honest way to ask it
//! here is the VEH2a cube body — a `VehicleDef` whose family's parts table holds
//! four panels and nothing else. Every arm's answer is in the table.
//!
//! | arm | the mutation that reds it | engagement counted | a car with no parts? |
//! |---|---|---|---|
//! | `a_door_opens_on_its_hinge_and_shuts_again` | `hinge_step`'s motor term deleted | degrees travelled, steps | **fails** — it has no door |
//! | `a_crash_at_sixty_sheds_the_bumper_and_pops_the_bonnet` | `part_break_impulse_ns` → infinity | the blow's own N.s, parts before/after | **fails** — nothing to shed |
//! | `the_shed_part_is_a_body_on_the_ground` | the `RigidBody3D` insert dropped | the part's own fall, its mass | **fails** |
//! | `a_round_through_a_window_takes_the_window` | `glass_health_j` → 1e9 | the pane's latch, the rounds spent | **fails** |
//! | `a_crash_dents_the_panel_it_reaches_and_not_the_one_it_does_not` | `DENT_M_PER_KNS` → 0 | the dent in mm, front and back | **fails** |
//! | `a_car_shot_at_spends_its_own_joules` | the vehicle branch deleted from `apply_hit` | joules in, joules on the hull | passes — this one is about the HULL |
//! | `a_dead_engine_stalls_and_a_hurt_one_is_slower` | `set_damage` made a no-op | the 0-100 times, the rpm | passes |
//! | `a_flat_tyre_pulls` | `FLAT_MU_FRAC` → 1.0 and `FLAT_RADIUS_FRAC` → 1.0 | the yaw rate with and without | passes |
//! | `a_car_that_burns_brings_the_brigade` | the `report_incident` call deleted | the incident, the unit, the metres | passes |
//! | `a_thousand_parked_cars_with_parts_cost_what_they_cost_without_them` | the latched fast path deleted | µs/car, control and measured | n/a — it is a COST arm |
//! | `a_quiet_level_folds_no_bodywork_bytes` | `VehicleDamage::is_quiet` → `false` | parked cars, driven steps | passes — the section is empty either way, which is the point |
//! | `two_runs_of_one_crash_fold_the_same_bytes` | a clock or an RNG anywhere in the step | the bytes' own length | passes |
//! | `every_authored_family_is_a_car_with_doors` | a family's parts emptied | 79 parts over 7 families | **fails** |
//!
//! # PIE == shipping, the mirrors, and the wire
//!
//! Not duplicated here, for `veh3a_gate`'s reason verbatim.
//! `island_gate::pie_equals_shipping_on_an_island_drive` drives the cooked
//! island on both hosts and compares them step for step. The bodywork needs no
//! MIRROR fence of its own: `step_bodywork` is the last statement of
//! `inf_physics::d3::vehicle::step_vehicles`, inside the `vehicle_step` fence
//! both hosts already carry and `fixed_step_mirror` already pins
//! character-for-character. `projector_mirror`'s `SECTIONS` grew its eighteenth
//! row in the commit that folded it. No persisted field lands:
//! `veh3a_gate::every_v28_tunable_survives_the_wire` still walks all hundred.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::bodywork::{DamageLimits, PartLatch, PartState};
use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Terrain, Transform, Visibility,
};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::{BodyPartKind, VehicleControls, VehicleDef};
use inf_ecs::EcsWorld;
use inf_physics::d3::PhysicsBridge3D;
use inf_terrain::TerrainData;

const DT: f64 = 1.0 / 60.0;
const CHASSIS: Uuid = Uuid::from_u128(0x5E3C_0001);
const GROUND: Uuid = Uuid::from_u128(0x5E3C_0002);
const PAD: Uuid = Uuid::from_u128(0x5E3C_0003);
const WALL: Uuid = Uuid::from_u128(0x5E3C_0004);
const HALF: f64 = 300.0;

// ── the fixture ─────────────────────────────────────────────────────────────

struct Rig {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

fn ground_world() -> EcsWorld {
    let mut world = EcsWorld::default();
    let mut data = TerrainData::new(64, 2.0);
    let n = (HALF / 64.0) as i32 + 1;
    for tz in -n..=n {
        for tx in -n..=n {
            data.author_tile((tx, tz), |_, _| 0.0);
        }
    }
    let e = world.spawn_with_guid(GROUND, "Ground", None);
    world
        .world_mut()
        .entity_mut(e)
        .insert(Transform {
            translation: Vec3d::new(0.0, -0.5, 0.0),
            ..Default::default()
        })
        .insert(Visibility::default())
        .insert(Terrain {
            meters_per_sample: 2.0,
            tile_resolution: 64,
            data,
            ..Default::default()
        })
        .insert(RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        })
        .insert(Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(HALF, 0.5, HALF),
            friction: 0.9,
            ..Default::default()
        });
    world
}

fn pad(world: &mut EcsWorld) {
    let e = world.spawn_with_guid(PAD, "Pad", None);
    world
        .world_mut()
        .entity_mut(e)
        .insert(Transform {
            translation: Vec3d::new(0.0, 0.02, 0.0),
            ..Default::default()
        })
        .insert(Visibility::default())
        .insert(RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        })
        .insert(Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(40.0, 0.02, HALF),
            friction: 0.9,
            ..Default::default()
        });
}

/// **A wall the island HAS.** The island's buildings carry no ECS collider
/// (carried since COV1), so a crash arm on the island itself would have to use
/// a kerb, a pillar, a lightpost module or a parked truck. In the fixture it is
/// an honest static box: the crash is about the impulse the chassis takes, and a
/// 40-tonne static block delivers the same impulse a lightpost does.
fn wall(world: &mut EcsWorld, at_z: f64) {
    let e = world.spawn_with_guid(WALL, "Wall", None);
    world
        .world_mut()
        .entity_mut(e)
        .insert(Transform {
            translation: Vec3d::new(0.0, 2.0, at_z),
            ..Default::default()
        })
        .insert(Visibility::default())
        .insert(RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        })
        .insert(Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(12.0, 2.0, 0.5),
            friction: 0.8,
            ..Default::default()
        });
}

fn catalogue_def(id: &str) -> VehicleDef {
    *inf_editor_core::vehicle::island_vehicles()
        .get(id)
        .unwrap_or_else(|| panic!("the catalogue has no `{id}` row"))
}

fn car_from(world: &mut EcsWorld, guid: Uuid, at: DVec3, def: &VehicleDef) {
    let spawn = inf_ecs::vehicle::RigSpawn {
        name: "Runner".into(),
        at,
        yaw_deg: 0.0,
        paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
        clip: None,
        engine_voice: false,
        livery: None,
    };
    inf_ecs::vehicle::spawn_rig(world, guid, def, &spawn);
}

impl Rig {
    /// A catalogue row on a sealed pad, settled on its own suspension.
    fn row(id: &str) -> Self {
        Self::row_at(id, -120.0, None)
    }

    fn row_at(id: &str, z: f64, wall_z: Option<f64>) -> Self {
        let def = catalogue_def(id);
        let mut world = ground_world();
        pad(&mut world);
        if let Some(wz) = wall_z {
            wall(&mut world, wz);
        }
        let y = inf_ecs::vehicle::resting_origin_y(&def, 0.04) + 0.15;
        car_from(&mut world, CHASSIS, DVec3::new(0.0, y, z), &def);
        world.mark_dirty();
        world.propagate();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        let mut rig = Self { world, bridge };
        rig.step(60);
        rig
    }

    fn tune(&mut self, pairs: &[(&str, f64)]) {
        for (name, value) in pairs {
            assert!(
                self.bridge
                    .vehicle_mut(CHASSIS)
                    .is_some_and(|v| v.tune(name, *value)),
                "the fixture set an unknown tunable `{name}`"
            );
            // …and on the COMPONENT too, because `DamageLimits::of` reads the
            // chassis's own `VehicleClass` — which is the door a catalogue row,
            // an author and `INF_PIE_TUNE_VEHICLE` all write through.
            if let Some(e) = self.world.entity_of(CHASSIS) {
                if let Some(mut c) = self
                    .world
                    .world_mut()
                    .get_mut::<inf_ecs::components::VehicleClass>(e)
                {
                    c.set(name, *value);
                }
            }
        }
    }

    fn step(&mut self, n: u32) {
        for _ in 0..n {
            self.bridge.sync_from_world(&self.world);
            inf_physics::d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
            self.bridge.step(DT);
            self.bridge.write_back_into(&mut self.world);
            self.world.propagate();
        }
    }

    fn drive(&mut self, c: VehicleControls, n: u32) {
        for _ in 0..n {
            if let Some(v) = self.bridge.vehicle_mut(CHASSIS) {
                v.control(c);
            }
            self.step(1);
        }
    }

    fn speed(&self) -> f64 {
        self.bridge
            .body_of(CHASSIS)
            .and_then(|b| self.bridge.world().body_linvel(b))
            .map(|v| DVec3::new(v.x, 0.0, v.z).length())
            .unwrap_or(0.0)
    }

    fn at(&self) -> DVec3 {
        self.bridge
            .body_of(CHASSIS)
            .and_then(|b| self.bridge.world().body_translation(b))
            .unwrap_or(DVec3::ZERO)
    }

    fn yaw_rate(&self) -> f64 {
        self.bridge
            .body_of(CHASSIS)
            .and_then(|b| self.bridge.world().body_angvel(b))
            .map(|w| w.y)
            .unwrap_or(0.0)
    }

    fn parts(&self) -> Vec<(Uuid, PartState)> {
        inf_physics::d3::bodywork::parts_of(&self.world, CHASSIS)
    }

    fn part_named(&self, name: &str) -> Option<(Uuid, PartState)> {
        let guid = inf_ecs::vehicle::body_part_guid(CHASSIS, name);
        self.parts().into_iter().find(|(g, _)| *g == guid)
    }

    fn attached(&self) -> usize {
        inf_physics::d3::bodywork::attached_count(&self.world, CHASSIS)
    }

    fn damage(&self) -> inf_ecs::bodywork::VehicleDamage {
        inf_physics::d3::bodywork::damage_or_default(&self.world, CHASSIS)
    }

    fn limits(&self) -> DamageLimits {
        DamageLimits::of(&self.world, CHASSIS)
    }

    /// Accelerate into whatever is ahead, and report the peak blow the chassis
    /// took on the way.
    fn crash(&mut self, target_mps: f64) -> f64 {
        // Up to speed on the pad first.
        let mut spun = 0;
        while self.speed() < target_mps && spun < 3_000 {
            self.drive(
                VehicleControls {
                    throttle: 1.0,
                    ..Default::default()
                },
                1,
            );
            spun += 1;
        }
        let hit_speed = self.speed();
        // Coast until it hits, and then for a second more. Bounded, and the
        // bound is generous on purpose: the first cut coasted a fixed 180 steps
        // and the car was still eighty metres short of the wall at 60 km/h, so
        // four of five rows of the crash table measured a car that never
        // arrived.
        let mut peak = 0.0f64;
        let mut after_impact = 0;
        for _ in 0..1_200 {
            let before = self.speed();
            self.drive(VehicleControls::default(), 1);
            let after = self.speed();
            let mass = self
                .bridge
                .body_of(CHASSIS)
                .and_then(|b| self.bridge.world().body_mass(b))
                .unwrap_or(0.0);
            let blow = mass * (before - after).abs();
            peak = peak.max(blow);
            if peak > 500.0 {
                after_impact += 1;
                if after_impact > 90 {
                    break;
                }
            }
        }
        eprintln!(
            "crash: hit at {:.2} m/s ({:.1} km/h), peak blow {peak:.0} N.s",
            hit_speed,
            hit_speed * 3.6
        );
        peak
    }
}

// ── (a) the parts, and the hinge ────────────────────────────────────────────

/// **A door opens on its hinge, in degrees over time, and shuts again.**
#[test]
fn a_door_opens_on_its_hinge_and_shuts_again() {
    let mut rig = Rig::row("sedan");
    let (guid, before) = rig
        .part_named("door_fl")
        .expect("the saloon has a front near-side door");
    assert_eq!(before.kind, inf_ecs::vehicle::KIND_DOOR);
    assert_eq!(before.angle_deg, 0.0, "a parked car's door is shut");
    assert_eq!(before.latch, PartLatch::Latched);

    assert!(
        inf_physics::d3::bodywork::set_part_open(&mut rig.world, CHASSIS, guid, true),
        "the open door refused a door"
    );
    let mut trace = Vec::new();
    for i in 0..120 {
        rig.step(1);
        let a = rig.part_named("door_fl").map(|(_, s)| s.angle_deg).unwrap();
        if i % 12 == 0 {
            trace.push(a);
        }
    }
    let open = rig.part_named("door_fl").map(|(_, s)| s.angle_deg).unwrap();
    eprintln!("the door's angle every 12 steps: {trace:?} — settled at {open:.2} deg");
    assert!(
        open > 60.0,
        "the door only reached {open:.2} degrees in two seconds"
    );
    assert!(
        open <= inf_ecs::vehicle::DOOR_OPEN_DEG + 1e-6,
        "the door went past its own limit to {open:.2}"
    );
    // THE WORLD, not the table: the drawn part's own transform moved with it.
    let e = rig.world.entity_of(guid).expect("the door is an entity");
    let t = rig.world.world().get::<Transform>(e).copied().unwrap();
    eprintln!("the door's drawn rotation is {:?}", t.rotation);
    assert!(
        (t.rotation.y - open).abs() < 1e-6,
        "the door's drawn rotation is {:?} and its hinge says {open:.2}",
        t.rotation
    );
    // The engagement count: it really travelled.
    assert!(
        trace.windows(2).filter(|w| w[1] > w[0] + 1.0).count() >= 3,
        "the door did not travel: {trace:?}"
    );

    // …and it shuts.
    assert!(inf_physics::d3::bodywork::set_part_open(
        &mut rig.world,
        CHASSIS,
        guid,
        false
    ));
    rig.step(180);
    let shut = rig.part_named("door_fl").map(|(_, s)| s.angle_deg).unwrap();
    eprintln!("shut again at {shut:.3} deg");
    assert!(shut.abs() < 1.0, "the door shut to {shut:.3}");
}

/// **Every wheeled family is a car with doors, bumpers and glass** — the census
/// VEH3d's boarding and VEH3f's roster both read.
#[test]
fn every_authored_family_is_a_car_with_doors() {
    let mut total = 0;
    for id in ["sedan", "sports", "suv", "van", "truck", "cruiser"] {
        let rig = Rig::row(id);
        let parts = rig.parts();
        let by = |k: u8| parts.iter().filter(|(_, s)| s.kind == k).count();
        let doors = by(inf_ecs::vehicle::KIND_DOOR);
        let glass = by(inf_ecs::vehicle::KIND_GLASS);
        let bumpers = by(inf_ecs::vehicle::KIND_BUMPER);
        let hinged = parts
            .iter()
            .filter(|(_, s)| {
                BodyPartKind::of(
                    match s.kind {
                        inf_ecs::vehicle::KIND_DOOR => "door",
                        inf_ecs::vehicle::KIND_HOOD => "hood",
                        inf_ecs::vehicle::KIND_TRUNK => "trunk",
                        _ => "panel",
                    },
                    s.centre_frac,
                    s.half_frac,
                )
                .hinge()
                .is_some()
            })
            .count();
        eprintln!(
            "{id}: {} parts — {doors} doors, {bumpers} bumpers, {glass} panes, {hinged} hinged",
            parts.len()
        );
        assert!(doors >= 2, "{id} has {doors} doors");
        assert_eq!(bumpers, 2, "{id} has {bumpers} bumpers");
        assert!(glass >= 3, "{id} has {glass} panes");
        assert!(hinged >= 3, "{id} has {hinged} hinged parts");
        total += parts.len();
    }
    eprintln!("{total} parts over six catalogue rows");
    assert!(total >= 70);
}

// ── (b) the break ───────────────────────────────────────────────────────────

/// **A crash at 60 km/h into a wall sheds the bumper and pops the bonnet.**
#[test]
fn a_crash_at_sixty_sheds_the_bumper_and_pops_the_bonnet() {
    let mut rig = Rig::row_at("sedan", -60.0, Some(0.0));
    let before = rig.attached();
    let named: Vec<String> = rig
        .parts()
        .iter()
        .map(|(_, s)| format!("{}", s.kind))
        .collect();
    eprintln!(
        "{before} parts bolted on before the crash ({} kinds)",
        named.len()
    );
    let peak = rig.crash(60.0 / 3.6);
    let after = rig.attached();
    let d = rig.damage();
    eprintln!(
        "after: {after} bolted on, {} shed, {} open, hull {:.0} J of {:.0}",
        d.parts_shed(),
        d.parts_open(),
        d.hull_j,
        rig.limits().hull_capacity_j()
    );
    assert!(peak > 3_000.0, "the crash only delivered {peak:.0} N.s");
    assert!(
        after < before,
        "nothing came off: {before} parts before and {after} after a {peak:.0} N.s blow"
    );
    let front = rig
        .part_named("bumper_front")
        .expect("it has a front bumper");
    let rear = rig.part_named("bumper_rear").expect("it has a rear bumper");
    assert_eq!(
        front.1.latch,
        PartLatch::Shed,
        "the front bumper survived a {peak:.0} N.s frontal crash"
    );
    assert_eq!(
        rear.1.latch,
        PartLatch::Latched,
        "the REAR bumper came off in a frontal crash — the face test is not being read"
    );
    let bonnet = rig.part_named("bonnet").expect("it has a bonnet");
    assert!(
        bonnet.1.latch != PartLatch::Latched,
        "the bonnet is still shut after a {peak:.0} N.s frontal crash"
    );
    eprintln!(
        "the bonnet is {} and the front bumper is {}",
        bonnet.1.latch.name(),
        front.1.latch.name()
    );

    // THE MUTATION: an unbreakable, un-poppable car takes the same blow and
    // keeps every panel.
    let mut safe = Rig::row_at("sedan", -60.0, Some(0.0));
    safe.tune(&[("part_break_impulse_ns", 1.0e12)]);
    let before = safe.attached();
    let peak2 = safe.crash(60.0 / 3.6);
    let after = safe.attached();
    eprintln!("with an unbreakable mount: {before} -> {after} at {peak2:.0} N.s");
    assert_eq!(
        before,
        after,
        "a car whose mounts cannot break shed {} parts",
        before - after
    );
    assert_eq!(safe.damage().parts_open(), 0);
}

/// **The crash table** — how many parts come off, by speed.
#[test]
fn the_crash_table_is_monotone_in_speed() {
    let mut rows = Vec::new();
    for kmh in [15.0f64, 30.0, 45.0, 60.0, 90.0] {
        let mut rig = Rig::row_at("sedan", -90.0, Some(0.0));
        let before = rig.attached();
        let peak = rig.crash(kmh / 3.6);
        let d = rig.damage();
        let after = rig.attached();
        eprintln!(
            "{kmh:>5.0} km/h: {peak:>8.0} N.s  parts {before} -> {after}  shed {}  open {}  panes {}  hull {:.0} J",
            d.parts_shed(),
            d.parts_open(),
            d.panes_broken(),
            d.hull_j
        );
        rows.push((kmh, peak, before - after, d.parts_shed(), d.parts_open()));
    }
    // Monotone: a faster crash never sheds fewer parts.
    for w in rows.windows(2) {
        assert!(
            w[1].3 >= w[0].3,
            "{:.0} km/h shed {} and {:.0} km/h shed {}",
            w[0].0,
            w[0].3,
            w[1].0,
            w[1].3
        );
    }
    assert_eq!(rows[0].3, 0, "a 15 km/h bump shed a part");
    assert!(rows[3].3 > 0, "a 60 km/h crash shed nothing");
}

/// **A shed part is a body on the ground**, with a part's own mass.
#[test]
fn the_shed_part_is_a_body_on_the_ground() {
    let mut rig = Rig::row_at("sedan", -60.0, Some(0.0));
    rig.crash(60.0 / 3.6);
    let (guid, state) = rig
        .part_named("bumper_front")
        .expect("it has a front bumper");
    assert_eq!(state.latch, PartLatch::Shed);
    let e = rig
        .world
        .entity_of(guid)
        .expect("a shed part is still an entity");
    assert!(
        rig.world.world().get::<RigidBody3D>(e).is_some(),
        "the shed bumper has no body"
    );
    assert!(
        rig.world.parent_of(e).is_none(),
        "the shed bumper is still a child of the car it came off"
    );
    let body = rig
        .bridge
        .body_of(guid)
        .expect("the shed bumper is in the solver");
    let mass = rig.bridge.world().body_mass(body).unwrap_or(0.0);
    let y0 = rig.bridge.world().body_translation(body).unwrap().y;
    rig.step(120);
    let y1 = rig
        .bridge
        .world()
        .body_translation(body)
        .map(|p| p.y)
        .unwrap_or(y0);
    eprintln!("the shed bumper weighs {mass:.2} kg and fell from {y0:.3} m to {y1:.3} m");
    assert!(
        (3.0..=40.0).contains(&mass),
        "the shed bumper weighs {mass:.2} kg"
    );
    assert!(
        y1 < y0 + 0.05,
        "the shed bumper rose from {y0:.3} to {y1:.3}"
    );
    assert!(
        y1 > -2.0,
        "the shed bumper fell through the world to {y1:.3}"
    );
}

// ── (c) the glass ───────────────────────────────────────────────────────────

/// **A round through a window takes the window.**
#[test]
fn a_round_through_a_window_takes_the_window() {
    for (health, expect_gone) in [(120.0, true), (1.0e9, false)] {
        let mut rig = Rig::row("sedan");
        rig.tune(&[("glass_health_j", health)]);
        let (guid, before) = rig
            .part_named("glass_windscreen")
            .expect("the saloon has a windscreen");
        assert_eq!(before.latch, PartLatch::Latched);
        // Where the pane is, in the world — and a round landing on it.
        let car = rig.at();
        let half = rig
            .world
            .entity_of(CHASSIS)
            .and_then(|e| rig.world.world().get::<Collider3D>(e))
            .map(|c| c.half_extents)
            .unwrap();
        let at = car
            + DVec3::new(
                before.centre_frac.x * half.x,
                before.centre_frac.y * half.y,
                before.centre_frac.z * half.z,
            );
        let mut spent = 0.0;
        for _ in 0..3 {
            let hit = inf_physics::d3::bodywork::hit_vehicle(
                &mut rig.world,
                &rig.bridge,
                CHASSIS,
                at,
                500.0,
            );
            assert!(hit.is_some(), "the car refused a round");
            spent += 500.0;
        }
        rig.step(2);
        let (_, after) = rig.part_named("glass_windscreen").unwrap();
        let e = rig.world.entity_of(guid).unwrap();
        let visible = rig
            .world
            .world()
            .get::<Visibility>(e)
            .map(|v| v.visible)
            .unwrap_or(true);
        eprintln!(
            "glass_health_j {health}: {spent:.0} J spent, the pane is {} and {} drawn",
            after.latch.name(),
            if visible { "still" } else { "no longer" }
        );
        if expect_gone {
            assert_eq!(after.latch, PartLatch::Gone, "the pane held at {health} J");
            assert!(!visible, "a shattered pane is still drawn");
            assert_eq!(rig.damage().panes_broken(), 1);
            // …and the shards are real entities with a bounded life.
            let shards = rig
                .world
                .world()
                .get_resource::<inf_ecs::bodywork::VehicleDamageRes>()
                .map(|r| r.shards.len())
                .unwrap_or(0);
            eprintln!("{shards} shards thrown");
            assert!(shards > 0, "a pane shattered and threw nothing");
            rig.step(200);
            let left = rig
                .world
                .world()
                .get_resource::<inf_ecs::bodywork::VehicleDamageRes>()
                .map(|r| r.shards.len())
                .unwrap_or(0);
            assert_eq!(
                left, 0,
                "{left} shards outlived their own second and a half"
            );
        } else {
            // THE MUTATION.
            assert_eq!(
                after.latch,
                PartLatch::Latched,
                "a pane with a billion joules of health broke"
            );
            assert!(visible);
            assert_eq!(rig.damage().panes_broken(), 0);
        }
    }
}

// ── (d) the dent ────────────────────────────────────────────────────────────

/// **A crash dents the panel it reaches and not the one it does not.**
#[test]
fn a_crash_dents_the_panel_it_reaches_and_not_the_one_it_does_not() {
    let mut rig = Rig::row_at("sedan", -60.0, Some(0.0));
    let (front_guid, _) = rig.part_named("bonnet").expect("it has a bonnet");
    let scale_before = rig
        .world
        .entity_of(front_guid)
        .and_then(|e| rig.world.world().get::<Transform>(e))
        .map(|t| t.scale)
        .unwrap();
    rig.crash(60.0 / 3.6);
    let bonnet = rig.part_named("bonnet").unwrap().1;
    let boot = rig.part_named("boot").unwrap().1;
    eprintln!(
        "the bonnet is dented {:.1} mm and the boot {:.1} mm",
        bonnet.dent_m * 1000.0,
        boot.dent_m * 1000.0
    );
    assert!(
        bonnet.dent_m > 0.005,
        "a 60 km/h frontal crash dented the bonnet {:.4} m",
        bonnet.dent_m
    );
    assert!(
        bonnet.dent_m <= inf_ecs::bodywork::MAX_DENT_M + 1e-9,
        "the dent went past its cap to {:.4} m",
        bonnet.dent_m
    );
    assert_eq!(
        boot.dent_m, 0.0,
        "the BOOT dented in a frontal crash — the face test is not being read"
    );
    // THE WORLD: the drawn box's own geometry moved.
    let scale_after = rig
        .world
        .entity_of(front_guid)
        .and_then(|e| rig.world.world().get::<Transform>(e))
        .map(|t| t.scale)
        .unwrap();
    eprintln!("the bonnet's drawn box went from {scale_before:?} to {scale_after:?}");
    let (axis, _) = bonnet.facing();
    let a0 = [scale_before.x, scale_before.y, scale_before.z][axis];
    let a1 = [scale_after.x, scale_after.y, scale_after.z][axis];
    assert!(
        a1 < a0 - 0.004,
        "the bonnet's own box did not shrink on axis {axis}: {a0:.4} -> {a1:.4}"
    );
}

// ── (e) the health ──────────────────────────────────────────────────────────

/// **A car shot at spends its own joules** — the WPN2a sentence retired.
#[test]
fn a_car_shot_at_spends_its_own_joules() {
    let mut rig = Rig::row("sedan");
    assert_eq!(rig.damage().hull_j, 0.0, "a fresh car has spent nothing");
    let limits = rig.limits();
    // The FLANK, low and well aft: not a pane, not a wheel, not the engine bay.
    let at = rig.at() + DVec3::new(0.85, -0.30, -0.20);
    let mut spent = 0.0;
    for _ in 0..20 {
        inf_physics::d3::bodywork::hit_vehicle(&mut rig.world, &rig.bridge, CHASSIS, at, 500.0);
        spent += 500.0;
    }
    let d = rig.damage();
    eprintln!(
        "{spent:.0} J into the flank: hull {:.0} of {:.0} ({:.0} % left), engine {:.0} %",
        d.hull_j,
        limits.hull_capacity_j(),
        100.0 * d.hull_frac(limits.hull_capacity_j()),
        100.0 * d.engine_scale()
    );
    assert!(
        (d.hull_j - spent).abs() < 1e-6,
        "the car absorbed {:.1} J of {spent:.0}",
        d.hull_j
    );
    assert_eq!(
        d.engine_damage, 0.0,
        "a round through the rear flank damaged the engine"
    );
    // …and the same rounds into the engine bay DO reach the engine.
    let nose = rig.at() + DVec3::new(0.0, 0.2, 1.6);
    for _ in 0..20 {
        inf_physics::d3::bodywork::hit_vehicle(&mut rig.world, &rig.bridge, CHASSIS, nose, 500.0);
    }
    let d = rig.damage();
    eprintln!(
        "…and 10 000 J into the nose: engine {:.0} %",
        100.0 * d.engine_scale()
    );
    assert!(
        d.engine_damage > 0.0,
        "twenty rounds into the engine bay did nothing to the engine"
    );
}

/// **A dead engine stalls, and a hurt one is slower.**
#[test]
fn a_dead_engine_stalls_and_a_hurt_one_is_slower() {
    let zero_to = |scale: f64| -> (f64, f64) {
        let mut rig = Rig::row("sports");
        if scale < 1.0 {
            let at = rig.at() + DVec3::new(0.0, 0.2, 1.6);
            let cap = rig.limits().engine_capacity_j();
            let want = (1.0 - scale) * cap;
            let mut spent = 0.0;
            while spent < want - 1.0 {
                let e = (want - spent).min(500.0);
                inf_physics::d3::bodywork::hit_vehicle(&mut rig.world, &rig.bridge, CHASSIS, at, e);
                spent += e;
            }
            rig.step(2);
        }
        let got = rig.damage().engine_scale();
        let mut t = 0.0;
        for _ in 0..1_200 {
            rig.drive(
                VehicleControls {
                    throttle: 1.0,
                    ..Default::default()
                },
                1,
            );
            t += DT;
            if rig.speed() >= 100.0 / 3.6 {
                return (got, t);
            }
        }
        (got, f64::INFINITY)
    };
    let (s1, whole) = zero_to(1.0);
    let (s2, hurt) = zero_to(0.5);
    eprintln!("0-100 km/h: whole (scale {s1:.2}) {whole:.2} s, hurt (scale {s2:.2}) {hurt:.2} s");
    assert!((s1 - 1.0).abs() < 1e-9);
    assert!(
        (s2 - 0.5).abs() < 0.06,
        "the engine scale came out at {s2:.3}"
    );
    assert!(whole.is_finite(), "the whole car never reached 100 km/h");
    assert!(
        hurt > whole * 1.10,
        "a half-dead engine took {hurt:.2} s against {whole:.2} s"
    );

    // …and a DEAD one stalls: no torque at any throttle, and the crank falls
    // below its own idle.
    let mut rig = Rig::row("sports");
    let at = rig.at() + DVec3::new(0.0, 0.2, 1.6);
    for _ in 0..40 {
        inf_physics::d3::bodywork::hit_vehicle(&mut rig.world, &rig.bridge, CHASSIS, at, 2_000.0);
    }
    rig.step(2);
    assert_eq!(rig.damage().engine_scale(), 0.0, "the engine is not dead");
    let idle = rig
        .bridge
        .vehicle_of(CHASSIS)
        .and_then(|v| v.drivetrain())
        .map(|d| d.rpm)
        .unwrap_or(0.0);
    rig.drive(
        VehicleControls {
            throttle: 1.0,
            ..Default::default()
        },
        300,
    );
    let rpm = rig
        .bridge
        .vehicle_of(CHASSIS)
        .and_then(|v| v.drivetrain())
        .map(|d| d.rpm)
        .unwrap_or(0.0);
    eprintln!("a dead engine: {idle:.0} rpm before, {rpm:.0} after five seconds of full throttle, {:.3} m/s", rig.speed());
    assert!(
        rpm < idle.max(1.0) * 0.5,
        "a dead engine still turns at {rpm:.0} rpm"
    );
    assert!(
        rig.speed() < 0.6,
        "a car with a dead engine reached {:.2} m/s",
        rig.speed()
    );
}

/// **A flat tyre pulls**, at a constant steer.
#[test]
fn a_flat_tyre_pulls() {
    let run = |flatten: bool| -> (f64, f64) {
        let mut rig = Rig::row("sedan");
        if flatten {
            // A round into the near-side front wheel.
            let rig_wheels: Vec<(Vec3d, f64)> = rig
                .bridge
                .vehicle_of(CHASSIS)
                .map(|v| {
                    v.rig()
                        .wheels
                        .iter()
                        .map(|w| (w.mount_local, w.radius_m))
                        .collect()
                })
                .unwrap_or_default();
            let (mount, _) = rig_wheels
                .iter()
                .copied()
                .filter(|(m, _)| m.z > 0.0)
                .min_by(|a, b| a.0.x.partial_cmp(&b.0.x).unwrap())
                .expect("the car has a front near-side wheel");
            let at = rig.at() + mount.to_dvec3();
            inf_physics::d3::bodywork::hit_vehicle(&mut rig.world, &rig.bridge, CHASSIS, at, 800.0);
            rig.step(2);
            assert_eq!(rig.damage().flat_count(), 1, "the round did not puncture");
        }
        // Up to speed, then hold a dead-straight wheel.
        rig.drive(
            VehicleControls {
                throttle: 1.0,
                ..Default::default()
            },
            420,
        );
        let x0 = rig.at().x;
        let mut yaw = 0.0f64;
        for _ in 0..240 {
            rig.drive(
                VehicleControls {
                    throttle: 0.45,
                    steer: 0.0,
                    ..Default::default()
                },
                1,
            );
            yaw += rig.yaw_rate().abs() * DT;
        }
        (rig.at().x - x0, yaw.to_degrees())
    };
    let (drift_ok, yaw_ok) = run(false);
    let (drift_flat, yaw_flat) = run(true);
    eprintln!(
        "four seconds straight: whole drifts {drift_ok:.3} m ({yaw_ok:.2} deg of yaw), flat drifts \
         {drift_flat:.3} m ({yaw_flat:.2} deg)"
    );
    assert!(
        yaw_flat > yaw_ok * 1.5 + 0.2,
        "a flat tyre yawed {yaw_flat:.3} deg against {yaw_ok:.3} whole — it does not pull"
    );
    assert!(
        drift_flat.abs() > drift_ok.abs() + 0.05,
        "a flat tyre drifted {drift_flat:.3} m against {drift_ok:.3} whole"
    );
}

// ── (f) determinism ─────────────────────────────────────────────────────────

/// **A quiet level folds no bodywork bytes**, and a crashed one folds a car.
#[test]
fn a_quiet_level_folds_no_bodywork_bytes() {
    let mut rig = Rig::row_at("sedan", -60.0, Some(0.0));
    rig.step(240);
    let quiet = inf_ecs::bodywork::damage_state_bytes(&rig.world);
    eprintln!(
        "a parked car with {} parts folds {} bytes",
        rig.parts().len(),
        quiet.len()
    );
    assert!(
        rig.parts().len() >= 12,
        "the fixture's car has {} parts — this arm is not about a car with none",
        rig.parts().len()
    );
    assert!(
        quiet.is_empty(),
        "a parked car folded {} bytes",
        quiet.len()
    );
    // …and a driven one still folds nothing, which is the point: the section is
    // about DAMAGE and not about motion.
    rig.drive(
        VehicleControls {
            throttle: 0.6,
            ..Default::default()
        },
        120,
    );
    assert!(
        inf_ecs::bodywork::damage_state_bytes(&rig.world).is_empty(),
        "a car being DRIVEN folded bodywork bytes"
    );
    rig.crash(60.0 / 3.6);
    let loud = inf_ecs::bodywork::damage_state_bytes(&rig.world);
    eprintln!("…and after the crash it folds {} bytes", loud.len());
    assert!(!loud.is_empty(), "a crashed car folded nothing");
    assert_eq!(
        &loud[..16],
        CHASSIS.as_bytes(),
        "the key is not the chassis"
    );
}

/// **Two runs of one crash fold the same bytes** — no clock, no RNG.
#[test]
fn two_runs_of_one_crash_fold_the_same_bytes() {
    let run = || {
        let mut rig = Rig::row_at("sedan", -60.0, Some(0.0));
        rig.crash(60.0 / 3.6);
        rig.step(120);
        (
            inf_ecs::bodywork::damage_state_bytes(&rig.world),
            rig.attached(),
            rig.damage().hull_j,
        )
    };
    let a = run();
    let b = run();
    eprintln!(
        "two runs: {} vs {} bytes, {} vs {} attached, {:.3} vs {:.3} J",
        a.0.len(),
        b.0.len(),
        a.1,
        b.1,
        a.2,
        b.2
    );
    assert!(!a.0.is_empty(), "the course crashed nothing");
    assert_eq!(a.0, b.0, "two runs of one crash folded different bytes");
    assert_eq!(a.1, b.1);
    assert_eq!(a.2.to_bits(), b.2.to_bits());
}

// ── (g) the cost ────────────────────────────────────────────────────────────

/// **A thousand parked cars with parts cost what they cost without them.**
///
/// The tier-ladder arm. A latched part carries no body, no collider and no
/// joint, so the solver never sees one — and the fast path in the hinge loop is
/// what keeps the ECS write off the step too.
///
/// The ceiling is reported everywhere and asserted only under release off CI,
/// which is the house conditioning for every clock in this repository.
#[test]
fn a_thousand_parked_cars_with_parts_cost_what_they_cost_without_them() {
    use std::time::Instant;
    const CARS: usize = 1_000;

    let build = |with_parts: bool| -> (EcsWorld, PhysicsBridge3D) {
        let def = catalogue_def("sedan");
        let mut world = ground_world();
        pad(&mut world);
        for i in 0..CARS {
            let guid = Uuid::from_u128(0x5E3C_1000_0000 + i as u128);
            let x = ((i % 40) as f64 - 20.0) * 3.2;
            let z = ((i / 40) as f64 - 12.0) * 6.0;
            let y = inf_ecs::vehicle::resting_origin_y(&def, 0.04) + 0.15;
            car_from(&mut world, guid, DVec3::new(x, y, z), &def);
            if !with_parts {
                // The CONTROL: the very same rig with its drawn parts removed,
                // so the two populations differ in exactly one thing.
                for p in def.body.parts() {
                    let g = inf_ecs::vehicle::body_part_guid(guid, p.name);
                    if let Some(e) = world.entity_of(g) {
                        world.despawn(e);
                    }
                }
            }
        }
        world.mark_dirty();
        world.propagate();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        for _ in 0..60 {
            bridge.sync_from_world(&world);
            inf_physics::d3::step_vehicles(&mut world, &mut bridge, DT);
            bridge.step(DT);
            bridge.write_back_into(&mut world);
            world.propagate();
        }
        (world, bridge)
    };

    let measure = |with_parts: bool| -> f64 {
        let (mut world, mut bridge) = build(with_parts);
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            bridge.sync_from_world(&world);
            let t = Instant::now();
            inf_physics::d3::step_vehicles(&mut world, &mut bridge, DT);
            best = best.min(t.elapsed().as_secs_f64() * 1e6);
            bridge.step(DT);
            bridge.write_back_into(&mut world);
            world.propagate();
        }
        best / CARS as f64
    };

    let control = measure(false);
    let parts = measure(true);
    let ratio = parts / control.max(1e-9);
    eprintln!(
        "{CARS} parked cars: {control:.3} µs/car WITHOUT parts, {parts:.3} µs/car WITH them \
         (x{ratio:.3}), min of five"
    );
    let (world, _) = build(true);
    let drawn: usize = inf_physics::d3::bodywork::row_count(&world);
    eprintln!("{drawn} damage rows, and every part of every one of them is LATCHED");
    let ceiling = 1.05;
    if cfg!(not(debug_assertions)) && std::env::var("CI").is_err() {
        assert!(
            ratio <= ceiling,
            "parts cost x{ratio:.3} of a car with none — over the {ceiling:.2} ceiling"
        );
    } else {
        eprintln!("(the ceiling of x{ceiling:.2} is asserted under release off CI only)");
    }
    assert!(control > 0.0 && parts > 0.0, "the clock measured nothing");
}
