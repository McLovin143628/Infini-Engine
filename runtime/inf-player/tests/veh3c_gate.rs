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
//! | `the_shed_part_falls_to_the_ground_and_lies_there` | `step_debris` stopped stepping | its mass, its fall, its rest, its reap | **fails** |
//! | `a_round_through_a_window_takes_the_window` | `glass_health_j` → 1e9 | the pane's latch, the rounds spent | **fails** |
//! | `a_crash_dents_the_panel_it_reaches_and_not_the_one_it_does_not` | `DENT_M_PER_KNS` → 0 | the dent in mm, front and back | **fails** |
//! | `a_car_shot_at_spends_its_own_joules` | the vehicle branch deleted from `apply_hit` | joules in, joules on the hull | passes — this one is about the HULL |
//! | `a_dead_engine_stalls_and_a_hurt_one_is_slower` | `set_damage` made a no-op | the 0-100 times, the rpm | passes |
//! | `a_flat_tyre_pulls` | `FLAT_MU_FRAC` → 1.0 **or** `FLAT_RADIUS_FRAC` → 1.0, each on its own | the yaw, the drift and the RIDE HEIGHT with and without | passes |
//! | `a_car_with_no_hull_left_burns` | the ignite branch deleted from `hit_vehicle` | the joules at which it lit, the trace | passes |
//! | `the_crash_table_is_monotone_in_speed` | `impact_share` flattened to one number | five speeds, the N.s and the parts of each | **fails** |
//! | `a_thousand_parked_cars_with_parts_cost_what_they_cost_without_them` | the latched fast path deleted | µs/car, control and measured | n/a — it is a COST arm |
//! | `a_quiet_level_folds_no_bodywork_bytes` | `VehicleDamage::is_quiet` → `false` | parked cars, driven steps | passes — the section is empty either way, which is the point |
//! | `two_runs_of_one_crash_fold_the_same_bytes` | a clock or an RNG anywhere in the step | the bytes' own length | passes |
//! | `pie_equals_shipping_on_a_crash_course` | either host's `step_bodywork` call deleted | 108 quiet steps then 439 bytes | passes — two empty traces agree too, which the anti-vacuity half refuses |
//! | `every_authored_family_is_a_car_with_doors` | a family's parts emptied | 84 parts over 6 catalogue rows | **fails** |
//! | `a_blast_reaches_the_car_and_not_the_panel_that_came_off_it` | the chassis walk removed from `apply_blast` | `blasts.len()`, `vehicle_hits`, the closed form on the hull, the debris' own height | **fails** |
//! | `the_shipped_host_draws_the_damage_row` | the call deleted from `inf_player::window` | three source fragments and one row | n/a |
//! | `this_wave_moved_no_schema` | a `Serialize` on `BodyPart` | 100 tunables, three names | n/a |
//!
//! The BRIGADE's arrival is measured where a town is:
//! `inf_physics::tests::dispatch_3d::a_burning_car_brings_the_appliance` walks
//! the whole EMS2 chain on a burning CAR over the `Town` fixture that already
//! has blocks, a carriageway, a station and an appliance — assigned on step 0,
//! on scene at 57.2 s and 11.64 m, resolved at 62.3 s, 22 puffs of smoke, the
//! intensity hosed to zero. Re-building that fixture in this file would be a
//! second spelling of it, which is the defect this repository has paid for at
//! five separate seams.
//!
//! # PIE == shipping, the mirrors, and the wire
//!
//! `pie_equals_shipping_on_a_crash_course` is this wave's own, on
//! `deform_parity`'s shape: the editor's `SimSession` and the shipped player's
//! `RuntimeSim` are driven directly through their own front doors over a car
//! dropped onto a slab, and `damage_state_bytes` is compared step by step. The
//! WIDER claim rides `island_gate::pie_equals_shipping_on_an_island_drive`,
//! which drives the cooked island against the loose one — and every car on that
//! island has doors now, so the bodywork step runs on both paths there too. The bodywork needs no
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
/// The blast's own shooter — a guid the world does not hold, so the sweep's
/// shooter-exclusion has something to exclude and the car is the only candidate.
const SHOOTER: Uuid = Uuid::from_u128(0x5E3C_0005);
/// A lamp post — the one thing on a street that can take a door off.
const POST: Uuid = Uuid::from_u128(0x5E3C_0006);
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
        inf_physics::d3::bodywork::set_part_open(
            &mut rig.world,
            &mut rig.bridge,
            CHASSIS,
            guid,
            true
        ),
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
        open <= inf_ecs::vehicle::DOOR_OPEN_DEG + 1.0,
        "the door went past its own limit to {open:.2}"
    );
    // **IT IS ON A REAL JOINT**, and that is what this arm is about since the
    // audit's closure. The door is a rapier body, the revolute is live, and the
    // angle above is READ OFF that joint rather than integrated beside it.
    let p = rig
        .bridge
        .part_body(guid)
        .expect("an opened door is a body on a hinge, not a drawn child");
    assert!(
        p.joint.is_some(),
        "the door has a body and no joint — nothing is holding it on"
    );
    assert_eq!(p.chassis, CHASSIS);
    assert!(
        rig.world
            .entity_of(guid)
            .and_then(|e| rig.world.parent_of(e))
            .is_none(),
        "a live door is still a CHILD — the bridge mirrors a body at its local          transform, so a body-carrying entity has to be a root"
    );
    // THE WORLD, not the table: the drawn part's own transform is where the
    // SOLVER put it, and the solver put it away from the chassis' centre.
    let e = rig.world.entity_of(guid).expect("the door is an entity");
    let t = rig.world.world().get::<Transform>(e).copied().unwrap();
    let swing = (t.translation.to_dvec3() - rig.at()).length();
    eprintln!(
        "the door's drawn pose is {:?} — {swing:.3} m from the chassis origin,          and the hinge carries {:.1} N.s",
        t.translation,
        rig.bridge.part_joint_impulse(guid).unwrap_or(0.0)
    );
    assert!(
        swing > 0.5,
        "the drawn door is {swing:.3} m from the car's own origin — the solver          is not moving it"
    );
    // The engagement count: it really travelled.
    assert!(
        trace.windows(2).filter(|w| w[1] > w[0] + 1.0).count() >= 3,
        "the door did not travel: {trace:?}"
    );

    // …and it shuts.
    assert!(inf_physics::d3::bodywork::set_part_open(
        &mut rig.world,
        &mut rig.bridge,
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

/// **A shed part falls to the ground and lies there**, with a part's own mass.
///
/// It is DRAWN debris and not a rapier body — see
/// `inf_physics::d3::bodywork`'s own note for the three measurements that
/// decided it — so what this reads is the `Debris` row the bodywork owns and the
/// entity's own `Transform` following it down.
#[test]
fn the_shed_part_falls_to_the_ground_and_lies_there() {
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
        rig.world.parent_of(e).is_none(),
        "the shed bumper is still a child of the car it came off"
    );
    // **IT IS A RAPIER BODY**, which is the audit's closure of the joint clause:
    // a part that leaves a car is a thing the world can hit, and the wave's own
    // "a bumper in the road cannot be run over" is retired by
    // `a_shed_bumper_in_the_road_is_run_over`.
    let p = rig
        .bridge
        .part_body(guid)
        .expect("a shed bumper is a body of its own");
    assert!(
        p.joint.is_none(),
        "the shed bumper is still on its joint -- it did not let go, it stretched"
    );
    let body = p.body;
    let mass = rig.bridge.world().body_mass(body).unwrap_or(0.0);
    let y0 = rig.bridge.world().body_translation(body).unwrap().y;
    rig.step(240);
    let after = rig.bridge.world().body_translation(body).unwrap();
    let vel = rig
        .bridge
        .world()
        .body_linvel(body)
        .map(|v| v.length())
        .unwrap_or(0.0);
    eprintln!(
        "the shed bumper weighs {mass:.2} kg, fell from {y0:.3} m to {:.3} m and is          doing {vel:.4} m/s four seconds later",
        after.y
    );
    // The MASS is the parts table's shell mass, not a box at a material density.
    assert!(
        (3.0..=40.0).contains(&mass),
        "the shed bumper weighs {mass:.2} kg"
    );
    // It came to REST, on something, above the bottom of the world.
    assert!(
        vel < 0.5,
        "the shed bumper is still doing {vel:.3} m/s four seconds after it let go"
    );
    assert!(
        after.y > -2.0,
        "the shed bumper fell through the world to {:.3}",
        after.y
    );
    assert!(
        after.y <= y0 + 0.05,
        "the shed bumper ROSE from {y0:.3} to {:.3} -- two overlapping dynamic          bodies are a depenetration force with nowhere to go, which is the          P29.6 shape wearing a bumper",
        after.y
    );
    // …and the drawn entity is where the solver put it.
    let drawn = rig
        .world
        .entity_of(guid)
        .and_then(|e| rig.world.world().get::<Transform>(e))
        .map(|t| t.translation)
        .unwrap();
    assert!(
        (drawn.y - after.y).abs() < 1e-9,
        "the drawn bumper is at {:.3} and its body is at {:.3}",
        drawn.y,
        after.y
    );
    // …and it is reaped, body and all.
    rig.step((inf_ecs::bodywork::PART_DEBRIS_LIFETIME_S * 60.0) as u32 + 10);
    assert!(
        rig.world.entity_of(guid).is_none(),
        "the shed bumper outlived its own debris lifetime"
    );
    assert!(
        rig.bridge.part_body(guid).is_none(),
        "the entity went and its rapier body stayed -- a collider nothing draws          and nothing owns"
    );
}

/// **A SHED BUMPER IN THE ROAD IS RUN OVER** — the sentence the wave carried,
/// retired (the audit's joint closure).
///
/// The wave shipped a shed part as DRAWN debris the solver could not see, and
/// named the cost: *"a bumper in the road cannot be run over."* It can now, and
/// this reads both halves out of the world — a downward cast that FINDS it, and
/// a car that drives at it and shoves it.
#[test]
fn a_shed_bumper_in_the_road_is_run_over() {
    let mut rig = Rig::row_at("sedan", -60.0, Some(0.0));
    rig.crash(60.0 / 3.6);
    let (guid, _) = rig
        .part_named("bumper_front")
        .expect("it has a front bumper");
    let p = rig
        .bridge
        .part_body(guid)
        .expect("a shed bumper is a body of its own");

    // **Back the car off and put the bumper in the OPEN ROAD.** A crash leaves
    // the panel pinned between a nose and a wall, which is a picture of nothing:
    // what this arm is about is a car meeting a shed part on a clear street. The
    // placement is the fixture's, and it is the only thing about the bumper that
    // is not the solver's.
    // The car goes back down the road through `place_vehicle` — the audit's own
    // teleport door, which moves the rig as a UNIT — so there is clear asphalt
    // in front of it and the wall it crashed into is a hundred metres away.
    let back = DVec3::new(0.0, rig.at().y, -140.0);
    assert!(inf_physics::d3::bodywork::place_vehicle(
        &mut rig.world,
        &mut rig.bridge,
        CHASSIS,
        back,
        glam::DQuat::IDENTITY,
    ));
    rig.step(60);
    let car = rig.at();
    // **In a WHEEL TRACK**, not under the middle of the car. A bumper lying flat
    // is a 19 mm panel and a saloon's floor clears it; what runs it over is a
    // tyre, so the fixture puts it where a tyre is going to be. The offset is
    // the rig's own front wheel mount, read out of the model rather than
    // guessed.
    let track = rig
        .bridge
        .vehicle_of(CHASSIS)
        .and_then(|v| {
            v.rig()
                .wheels
                .iter()
                .filter(|w| w.mount_local.z > 0.0)
                .map(|w| w.mount_local.x)
                .next()
        })
        .unwrap_or(0.0);
    let ahead = DVec3::new(car.x + track, car.y + 0.4, car.z + 14.0);
    rig.bridge.world_mut().set_body_translation(p.body, ahead);
    rig.bridge.world_mut().set_body_linvel(p.body, DVec3::ZERO);
    rig.step(90);
    let at = rig.bridge.world().body_translation(p.body).unwrap();

    // **A CAST FINDS IT.** Straight down from a metre over it, everything solid.
    // Six metres to the side the same cast reaches the road; over the bumper it
    // stops short — and that difference is the whole of what separates debris a
    // wheel ray can see from debris it cannot.
    let none = std::collections::BTreeSet::new();
    let over = rig
        .bridge
        .world_mut()
        .cast_ray_where(
            at + DVec3::Y,
            -DVec3::Y,
            4.0,
            &none,
            inf_physics::d3::CastTargets::AllSolid,
        )
        .map(|h| h.toi)
        .expect("nothing at all under the cast");
    let beside = rig
        .bridge
        .world_mut()
        .cast_ray_where(
            at + DVec3::Y + DVec3::new(6.0, 0.0, 0.0),
            -DVec3::Y,
            4.0,
            &none,
            inf_physics::d3::CastTargets::AllSolid,
        )
        .map(|h| h.toi)
        .expect("nothing at all under the control cast");
    eprintln!(
        "a cast over the shed bumper stops at {over:.4} m and the same cast six metres to the side reaches {beside:.4} m — it stands {:.1} mm proud of the road",
        (beside - over) * 1000.0
    );
    // Ten millimetres, and it is a measurement of the thing rather than a round
    // number: a bumper is a SHELL — the parts table gives it a half-extent of
    // 0.015 of the chassis — so lying flat on a road it stands about 19 mm
    // proud, which is exactly what a wheel ray has to be able to find.
    assert!(
        over < beside - 0.01,
        "the cast over the bumper went as deep as the one beside it: {over:.4} against {beside:.4} — a wheel ray cannot see it"
    );

    // **AND THE CAR DRIVES OVER IT.** The bumper is SHOVED and the car is still
    // moving on the other side. What this refuses is the wave's own second
    // measurement — a responding ambulance that never got home because a 7 kg
    // panel stopped it.
    let before = rig.bridge.world().body_translation(p.body).unwrap();
    //
    // **WHAT "RUN OVER" MEANS ON A RAYCAST VEHICLE.** A wheel here is a ray and
    // a suspension force, not a colliding cylinder: a wheel that meets a panel
    // does not shove it, it FINDS it — the ray stops 19 mm early, the strut
    // compresses and the corner rises. That is exactly the mechanism the wave
    // described when it said a bumper *"goes under a wheel ray"*, and it is the
    // thing to measure. So this reads the chassis' own ride height over the
    // panel against its ride height on clear road either side of it.
    let mut on_clear: f64 = 0.0;
    let mut over_panel: f64 = 0.0;
    let mut min_speed_past = f64::INFINITY;
    let mut passed = false;
    // **AT WALKING PACE, and that is a measurement about the SAMPLE RATE.** At
    // full throttle the car crosses a 70 mm panel in a fifth of a step and the
    // suspension never samples it -- measured: 25.84 m/s gave -1.3 mm of lift,
    // which is a car that flew over it between two fixed steps. Held at about
    // three metres a second the panel is under a wheel for a step and a half.
    for _ in 0..1_100 {
        let throttle = if rig.speed() > 3.0 { 0.0 } else { 0.35 };
        rig.drive(
            VehicleControls {
                throttle,
                ..Default::default()
            },
            1,
        );
        let here = rig.at();
        let dz = (here.z - before.z).abs();
        if dz < 1.6 {
            over_panel = over_panel.max(here.y);
        } else if (3.0..7.0).contains(&dz) {
            on_clear = on_clear.max(here.y);
        }
        if here.z > before.z {
            passed = true;
            min_speed_past = min_speed_past.min(rig.speed());
        }
    }
    let rise = over_panel - on_clear;
    eprintln!(
        "the car drove at it from {:.1} m away: it rode at {on_clear:.4} m on clear road and {over_panel:.4} m over the panel — {:.1} mm of lift — and it is {} doing {:.2} m/s",
        (before - car).length(),
        rise * 1000.0,
        if passed { "past it," } else { "STILL SHORT OF IT," },
        rig.speed()
    );
    assert!(
        passed,
        "the car never reached the bumper — this arm measured nothing"
    );
    // **Where the bound comes from.** ONE wheel of four rides up an 18.5 mm
    // panel, so the chassis ORIGIN can rise by at most a quarter of it — 4.6 mm
    // — and the other three springs take some of that back. 2.7 mm is the right
    // order; 1.5 mm is comfortably clear of it and a mile from the -1.3 mm a car
    // that flew over the panel between two steps reads.
    eprintln!(
        "one wheel of four on an {:.1} mm panel is at most {:.1} mm at the origin; measured {:.1}",
        (beside - over) * 1000.0,
        (beside - over) * 250.0,
        rise * 1000.0
    );
    assert!(
        rise > 0.0015,
        "the car rode {:.1} mm higher over the panel than on clear road — the wheel ray went straight through it, which is the wave's own 'a bumper in the road cannot be run over'",
        rise * 1000.0
    );
    assert!(
        min_speed_past > 0.5,
        "the car was down to {min_speed_past:.2} m/s as it went over — a 7 kg panel stopped a 1.5 t car, which is the defect the wave refused the body to avoid"
    );
}

/// **AN OPEN DOOR IS TORN OFF BY A LAMP POST** — the other sentence the wave
/// carried, and the break watch's first real caller.
///
/// `PhysicsWorld3D::joint_impulse` and `BreakWatch3D` were built by the wave,
/// proven by their own arms in `joints3d.rs`, and called by nothing. This is
/// what calls them: the door's own hinge is read every step, and over
/// `part_break_impulse_ns` `remove_joint` lets it go.
#[test]
fn an_open_door_is_torn_off_by_a_lamp_post() {
    let run = |post: bool| -> (usize, f64) {
        let mut rig = Rig::row_at("sedan", -60.0, None);
        // A lamp post: a static pillar just outside the car's own flank, far
        // enough ahead that the car is up to speed when the door reaches it.
        if post {
            let e = rig.world.spawn_with_guid(POST, "Lamp post", None);
            rig.world
                .world_mut()
                .entity_mut(e)
                .insert(Transform {
                    translation: Vec3d::new(-1.30, 1.5, -40.0),
                    ..Default::default()
                })
                .insert(Visibility::default())
                .insert(RigidBody3D {
                    kind: BodyKind3D::Static,
                    ..Default::default()
                })
                .insert(Collider3D {
                    shape_kind: ColliderShape3DKind::Box,
                    half_extents: Vec3d::new(0.12, 1.5, 0.12),
                    friction: 0.8,
                    ..Default::default()
                });
            rig.world.mark_dirty();
            rig.world.propagate();
        }
        let (guid, _) = rig.part_named("door_fl").expect("it has a near-side door");
        assert!(inf_physics::d3::bodywork::set_part_open(
            &mut rig.world,
            &mut rig.bridge,
            CHASSIS,
            guid,
            true
        ));
        // Let it swing all the way out before the car moves.
        rig.step(120);
        let before = rig.attached();
        let mut peak = 0.0f64;
        for _ in 0..900 {
            rig.drive(
                VehicleControls {
                    throttle: 1.0,
                    ..Default::default()
                },
                1,
            );
            if let Some(ns) = rig.bridge.part_joint_impulse(guid) {
                peak = peak.max(ns);
            }
            if rig.part_named("door_fl").map(|(_, s)| s.latch) == Some(PartLatch::Shed) {
                break;
            }
            if rig.at().z > -10.0 {
                break;
            }
        }
        let latch = rig.part_named("door_fl").map(|(_, s)| s.latch).unwrap();
        let after = rig.attached();
        eprintln!(
            "post {post:<5}: the hinge peaked at {peak:>9.0} N.s against a {:.0} N.s mount; {before} parts on before, {after} after; the door is {} and the car reached {:.1} m/s",
            rig.limits().part_break_impulse_ns,
            latch.name(),
            rig.speed()
        );
        (before - after, peak)
    };

    // THE CONTROL first: the same car, the same open door, nothing to hit.
    let (lost_clear, peak_clear) = run(false);
    let (lost_post, peak_post) = run(true);
    assert_eq!(
        lost_clear, 0,
        "a car that hit nothing lost {lost_clear} part(s) — the hinge is tearing itself off under its own weight"
    );
    assert!(
        lost_post >= 1,
        "the door survived a lamp post: the hinge peaked at {peak_post:.0} N.s against a mount of its own class"
    );
    assert!(
        peak_post > peak_clear * 2.0,
        "the post cost the hinge {peak_post:.0} N.s and open air cost it {peak_clear:.0} — the door did not hit anything"
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

/// **A flat tyre pulls**, at a constant steer — **and the corner sits down.**
///
/// # Two numbers, and the first draft could only see one of them
///
/// A flat costs the tyre the research doc's own two things: a rolling radius of
/// [`FLAT_RADIUS_FRAC`](inf_ecs::bodywork::FLAT_RADIUS_FRAC) (the corner sits
/// down) and a grip of [`FLAT_MU_FRAC`](inf_ecs::bodywork::FLAT_MU_FRAC) (the
/// car pulls). The wave's arm measured the PULL alone — and the audit's
/// mutation battery found that **neither constant alone reds it**: forcing the
/// grip to 1.0 leaves the radius still pulling, forcing the radius to 1.0
/// leaves the grip still pulling, and only the pair together turns it green.
/// An arm that cannot tell which of two numbers is doing the work is an arm
/// that will not notice one of them being deleted.
///
/// So the RIDE HEIGHT is read too, which is what `FLAT_RADIUS_FRAC` and nothing
/// else moves: the suspension length is computed from the deflated radius, so a
/// punctured corner drops and takes the chassis origin with it. The ray keeps
/// the INFLATED radius on purpose — it has to reach past where the tyre would be
/// if it were whole, or a flat tyre would find no ground at all.
#[test]
fn a_flat_tyre_pulls() {
    // The chassis origin at rest, measured on the same settle on both runs.
    let mut ride = [0.0f64; 2];
    let run = |flatten: bool, ride: &mut f64| -> (f64, f64) {
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
        // **THE CORNER SITS DOWN.** The same settle on both runs, before either
        // touches the throttle, so what separates the two numbers is the flat
        // and nothing else.
        rig.step(180);
        *ride = rig.at().y;
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
    let (drift_ok, yaw_ok) = run(false, &mut ride[0]);
    let (drift_flat, yaw_flat) = run(true, &mut ride[1]);
    eprintln!(
        "four seconds straight: whole drifts {drift_ok:.3} m ({yaw_ok:.2} deg of yaw), flat drifts \
         {drift_flat:.3} m ({yaw_flat:.2} deg)"
    );
    // **THE RADIUS, ON ITS OWN.** `FLAT_MU_FRAC` cannot move this number and
    // `FLAT_RADIUS_FRAC` is the only thing that can, so each of the two halves
    // of a flat now has an assertion that only it can fail.
    let sank = ride[0] - ride[1];
    eprintln!(
        "at rest the chassis sits at {:.4} m whole and {:.4} m on a flat — {:.1} mm lower",
        ride[0],
        ride[1],
        sank * 1000.0
    );
    assert!(
        sank > 0.005,
        "a punctured corner dropped the chassis {:.1} mm — the rolling radius is not being read",
        sank * 1000.0
    );
    // **AND THE GRIP, ON ITS OWN**, as a CEILING on the pull. A flat that pulls
    // HARDER than this is a flat carrying more grip than the model gives it:
    // measured, `FLAT_MU_FRAC` forced to 1.0 takes the drift from **6.846 m** to
    // **9.843** and the yaw from 1.36° to 1.76°, because the deflated corner
    // then has a whole tyre's lateral authority to steer the car with. Without a
    // ceiling that mutation leaves this arm green — it pulls, just wrongly — and
    // half of what a flat costs would be deletable in silence.
    assert!(
        drift_flat.abs() < 8.5,
        "a flat drifted {drift_flat:.3} m in four seconds — more than a deflated tyre's own \
         grip can steer with, so `FLAT_MU_FRAC` is not being applied"
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
/// which is the house conditioning for every clock in this repository. **It is
/// the RELEASE number that matters**: the wave reported x1.083 in debug and the
/// release run, which is the one the ceiling governs, measured **x1.097 and
/// FAILED**. See `VehicleDamage::loud_parts` for the three corrections that took
/// it to x0.978–x1.030, and the audit report for the eleven-run distribution.
///
/// # The CONTROL is not any shipped car, and that matters to the ratio
///
/// It is a rig with **every** drawn part despawned, which no catalogue row is:
/// even the VEH2a cube body has four panels. A car whose damage row never fills
/// takes the `walk` branch every step — the chassis' children, and the `wheels`
/// gather with it — because the fast path keys on the row having parts in it. So
/// the control pays a walk the measured population does not, and the ratio this
/// arm prints is therefore an **under**-statement of what parts cost. That is
/// the safe direction for a ceiling and it is why the number can read below
/// 1.000; it is stated rather than left for the next wave to discover.
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

// ── (e) the fire ────────────────────────────────────────────────────────────

/// **A car with no hull left BURNS**, and files the incident the brigade
/// answers.
///
/// The EMS2 door, unchanged: a burning car is an `IncidentKind::Fire` whose
/// `building` slot carries the CHASSIS' guid. That slot is read for
/// de-duplication and never dereferenced, so a car drops into it with no schema
/// move — and the whole chain (open, assign, route, arrive, hose, resolve) works
/// on it without knowing it is a car.
///
/// **The ARRIVAL is measured where a town is**, not here:
/// `inf_physics::tests::dispatch_3d::a_burning_car_brings_the_appliance` runs it
/// on the `Town` fixture, which already has blocks, a carriageway, a station and
/// an appliance. Re-building one of those in this file would be a second
/// spelling of a fixture, which is the defect this repository has paid for at
/// five separate seams. What this arm owns is the half that is about the CAR:
/// the joules, the latch, and that a level with no dispatcher REFUSES the
/// incident as a value rather than failing.
#[test]
fn a_car_with_no_hull_left_burns() {
    let mut rig = Rig::row("sedan");
    let cap = rig.limits().hull_capacity_j();
    // The FLANK, low and well aft: not a pane, not a wheel, not the engine bay.
    let at = rig.at() + DVec3::new(0.85, -0.30, -0.20);
    assert!(!rig.damage().burning(), "a fresh car is on fire");
    let mut spent = 0.0;
    let mut lit_at = None;
    while spent < cap + 2_000.0 {
        inf_physics::d3::bodywork::hit_vehicle(&mut rig.world, &rig.bridge, CHASSIS, at, 1_000.0);
        spent += 1_000.0;
        if lit_at.is_none() && rig.damage().burning() {
            lit_at = Some(spent);
        }
    }
    let d = rig.damage();
    eprintln!(
        "the saloon caught fire at {:?} J of a {cap:.0} J hull ({spent:.0} spent); its own step counter says {}",
        lit_at, d.fire_step
    );
    assert_eq!(
        lit_at,
        Some(cap),
        "the car lit at {lit_at:?} J and its hull is worth {cap:.0}"
    );
    assert!(d.burning());
    assert!(
        d.fire_step > 0,
        "the fire step is {} — the bodywork's own clock is standing still",
        d.fire_step
    );
    // …and the fire is in the TRACE, which is what makes two hosts agree about
    // a burning car.
    let bytes = inf_ecs::bodywork::damage_state_bytes(&rig.world);
    assert!(!bytes.is_empty(), "a burning car folds nothing");
    // A level with no dispatcher refuses the incident and does not fail: the
    // car still burns, which is a REFUSAL AS A VALUE rather than a panic.
    assert!(
        inf_ecs::dispatch::dispatch_of(&rig.world).is_none(),
        "the fixture grew a dispatcher — this half of the arm is vacuous"
    );
}

// ── the blast, and the lightest body in its radius ──────────────────────────

/// **A blast reaches the CAR and never the panel that came off it.**
///
/// The WPN2d levitation law, met head on: *size a blast against the LIGHTEST
/// body in its radius*. The lightest body a crashed car leaves on the road is a
/// 2.6 kg pane of glass, and a blast sized for a tonne and a half of car would
/// put it over a rooftop — so a shed part is deliberately NOT in the blast's
/// candidate set. The chassis is, which is what closes the sentence the WPN2d
/// audit left this wave by name: *"a vehicle that carries `Destructible` takes
/// blast damage the day it is authored"*, and no vehicle carries one.
#[test]
fn a_blast_reaches_the_car_and_not_the_panel_that_came_off_it() {
    let mut rig = Rig::row_at("sedan", -60.0, Some(0.0));
    rig.crash(60.0 / 3.6);
    let (shed_guid, shed) = rig
        .part_named("bumper_front")
        .expect("it has a front bumper");
    assert_eq!(shed.latch, PartLatch::Shed, "nothing came off to test with");
    let before_hull = rig.damage().hull_j;
    let shed_body = rig
        .bridge
        .part_body(shed_guid)
        .expect("a shed bumper is a body of its own")
        .body;
    let before_at = rig
        .bridge
        .world()
        .body_translation(shed_body)
        .map(|p| p.y)
        .unwrap_or(f64::NAN);

    // The candidate set the blast walks, measured rather than asserted about.
    let candidates: Vec<Uuid> = rig.bridge.vehicle_guids();
    eprintln!(
        "the blast's vehicle candidates are {:?} and the shed bumper is {shed_guid}",
        candidates
    );
    assert!(
        candidates.contains(&CHASSIS),
        "the chassis is not a blast candidate"
    );
    assert!(
        !candidates.contains(&shed_guid),
        "a 7 kg shed bumper is a blast candidate — this is the WPN2d levitation \
         defect with a bumper in it"
    );
    // **A shed part IS a rapier body now**, and it is still not a blast
    // candidate — which is the WPN2d levitation law met where it actually
    // lives. `apply_blast` walks `vehicle_guids`, which is the bridge's map of
    // CARS; a part is in `part_bodies`, a different map that the sweep does not
    // read. Sizing a blast against a tonne and a half of car and letting it
    // reach a 2.6 kg pane is how a windscreen ends up over a rooftop.
    assert!(
        rig.bridge.part_body(shed_guid).is_some(),
        "the shed bumper has no body, so nothing can run it over"
    );
    assert!(
        rig.bridge.body_of(shed_guid).is_none(),
        "the shed bumper is in the bridge's ENTITY map — a blast walks that map,          and a 7 kg panel with a car's impulse on it is the levitation defect"
    );

    // …and A REAL BLAST really does reach the chassis.
    //
    // **Through `apply_blast` itself**, by its own public gate door, and never
    // through `hit_vehicle` — which is what the first cut of this arm did, and
    // which made it VACUOUS. The mutation this arm's own table names is *the
    // chassis walk removed from `apply_blast`*, and an arm that never calls
    // `apply_blast` stays green through it: what it read as "the blast's
    // candidate set" was `PhysicsBridge3D::vehicle_guids`, which is the bridge's
    // list of cars and is the same list with or without the walk.
    // `blast_for_test` is a `pub` alias for the one blast sweep in this engine.
    let mut def = inf_ecs::weapon::WeaponDef {
        blast_radius_m: 6.0,
        blast_damage_j: 4_000.0,
        ..Default::default()
    };
    def.class = Some(inf_ecs::weapon::WeaponClass::Launcher);
    let blast_at = rig.at() + DVec3::new(0.85, -0.30, -0.20);
    let chassis_at = rig
        .world
        .entity_of(CHASSIS)
        .and_then(|e| {
            rig.world
                .world()
                .get::<inf_ecs::components::GlobalTransform>(e)
        })
        .map(|g| g.0.translation)
        .expect("the chassis has a global transform");
    let mut report = inf_physics::d3::GameplayReport::default();
    inf_physics::d3::gameplay::blast_for_test(
        &mut rig.world,
        &mut rig.bridge,
        SHOOTER,
        blast_at,
        &def,
        DT,
        &mut report,
    );
    let after_hull = rig.damage().hull_j;
    rig.step(60);
    let rose = rig
        .bridge
        .world()
        .body_translation(shed_body)
        .map(|p| p.y)
        .unwrap_or(f64::NAN)
        - before_at;
    eprintln!(
        "a real blast: {} bodies hurt, {} of them VEHICLES; the hull went {before_hull:.0} -> \
         {after_hull:.0} J and the shed bumper moved {rose:+.4} m vertically in the second after",
        report.blasts.first().map(|b| b.hurt).unwrap_or(0),
        report.vehicle_hits
    );
    // **THE ENGAGEMENT COUNT.** `GameplayReport::vehicle_hits` and its three
    // siblings were added by this wave and read by NOTHING in the tree — four
    // counters whose whole purpose is to make "a round reached a car" a
    // measurement rather than a claim, and no arm asked them. This is their
    // reader, and it is this arm's anti-vacuity half.
    assert_eq!(
        report.blasts.len(),
        1,
        "the blast did not go off at all — this arm measures nothing"
    );
    assert_eq!(
        report.vehicle_hits, 1,
        "a blast six metres from a car reached {} vehicles — the chassis walk is \
         not in `apply_blast`",
        report.vehicle_hits
    );
    assert_eq!(
        report.vehicle_panes, 0,
        "a blast beside the flank took a pane"
    );
    // **THE CLOSED FORM, not "roughly all of it".** The blast's candidate point
    // is the chassis's own `GlobalTransform`, which is the car's centre and not
    // the point the charge went off at, so the joules that arrive are the WPN2d
    // falloff at that distance — measured against `ballistics::blast_damage_j`
    // rather than against the charge's face value.
    let want = inf_ecs::ballistics::blast_damage_j(&def, (chassis_at - blast_at).length());
    eprintln!(
        "the chassis is {:.3} m from the charge, so the closed form owes it {want:.3} J",
        (chassis_at - blast_at).length()
    );
    assert!(want > 2_000.0, "the fixture put the charge too far off");
    assert!(
        ((after_hull - before_hull) - want).abs() < 1e-6,
        "the blast spent {:.3} J on the hull and the closed form owes {want:.3}",
        after_hull - before_hull
    );
    assert!(
        rose < 0.05,
        "the shed bumper rose {rose:.4} m — something is pushing a 7 kg panel with a car's impulse"
    );
}

// ── the shipped host, and the source pins ───────────────────────────────────

/// **The shipped host draws the damage row**, and the readout says what the
/// bodywork knows.
///
/// A SOURCE pin, because `inf_player::window` cannot be constructed in a test —
/// it owns a window. The formatting is Ring 0's and is measured on real state
/// below; this is what says the host calls it, which is `veh3b_gate`'s own split
/// verbatim.
#[test]
fn the_shipped_host_draws_the_damage_row() {
    const SRC: &str = include_str!("../src/window.rs");
    for needle in [
        "inf_ecs::bodywork::damage_readout",
        "inf_ecs::bodywork::damage_of",
        "inf_ecs::bodywork::DamageLimits::of",
    ] {
        assert!(
            SRC.contains(needle),
            "`runtime/inf-player/src/window.rs` no longer calls `{needle}` — the \
             hero's car damage is computed and nothing draws it"
        );
    }
    // …and the row is EMPTY on a whole car, which is what keeps every pre-VEH3c
    // HUD identical.
    let limits = DamageLimits::default();
    let whole = inf_ecs::bodywork::VehicleDamage::default();
    assert!(whole.is_quiet(), "a fresh car is not quiet");
    let mut hurt = whole.clone();
    hurt.hull_j = limits.hull_capacity_j() * 0.25;
    hurt.engine_damage = 0.4;
    hurt.flatten(2);
    hurt.parts.insert(
        Uuid::from_u128(1),
        PartState::new(inf_ecs::vehicle::BodyPartKind::Glass),
    );
    hurt.parts.get_mut(&Uuid::from_u128(1)).unwrap().latch = PartLatch::Gone;
    hurt.fire_step = 7;
    let row = inf_ecs::bodywork::damage_readout(&hurt, limits);
    eprintln!("the damage row reads: {row}");
    assert_eq!(row, "HULL 75%  ENG 60%  FLATS 1  GLASS 1/1  ON FIRE");
}

/// **Every new number this wave added is a RUNTIME one** — no persisted field
/// landed, and VEH3a's spent schema window stays spent.
#[test]
fn this_wave_moved_no_schema() {
    // The three tunables the bodywork reads all predate this wave: VEH3a landed
    // them in the v28 window and `veh3a_gate::every_v28_tunable_survives_the_wire`
    // walks all hundred through the editor's own codec.
    let names = inf_ecs::vehicle::VehicleTuning::names();
    for n in ["glass_health_j", "panel_health_j", "part_break_impulse_ns"] {
        assert!(names.contains(&n), "`{n}` is not a v28 tunable");
    }
    assert_eq!(names.len(), 100, "the tunable count moved");
    // …and `BodyPart` itself is not serializable, which is what makes
    // `BodyPartKind` free. A source pin, because the type has no `Serialize` to
    // assert the absence of.
    const SRC: &str = include_str!("../../../crates/inf-ecs/src/vehicle.rs");
    let at = SRC
        .find("pub struct BodyPart {")
        .expect("`BodyPart` is in `inf-ecs::vehicle`");
    let head = &SRC[at.saturating_sub(400)..at];
    assert!(
        !head.contains("Serialize"),
        "`BodyPart` grew a `Serialize` — a parts table on the wire is a schema \
         window, and VEH3a spent the only one this arc gets"
    );
    assert!(
        head.contains("#[derive(Clone, Copy, Debug, PartialEq)]"),
        "`BodyPart`'s derive list changed: {head:?}"
    );
}

// ── PIE == shipping, on a crash ─────────────────────────────────────────────

/// **THE EDITOR'S PREVIEW AND THE SHIPPED PLAYER CRASH THE SAME CAR.**
///
/// `deform_parity`'s shape, one system over: the two fixed steps are driven
/// directly through their own front doors (`RuntimeSim::new` and
/// `SimSession::enter`), and what is compared is
/// `bodywork::damage_state_bytes` step by step.
///
/// **The course is a DROP and not a drive**, and that is deliberate rather than
/// convenient: `SimSession` exposes no bridge, so a driven car would need an
/// input path that differs between the two hosts — and what this arm is about is
/// the FIXED STEP, not the controller. A car dropped onto a slab takes a real
/// blow from the world, dents the panels that face it, pops what its latches
/// cannot hold and spends its hull, with **no input at all** on either side.
///
/// The wider PIE == shipping claim rides
/// `island_gate::pie_equals_shipping_on_an_island_drive`, which drives the
/// cooked island against the loose one — and every car on that island has doors
/// now, so the bodywork step runs on both paths there too.
#[test]
fn pie_equals_shipping_on_a_crash_course() {
    use inf_editor_core::scene::SceneDoc;
    use inf_editor_core::simulate::{SimInput, SimSession};
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

    const HZ: f64 = 60.0;
    const STEPS: u32 = 150;
    /// High enough that the struts bottom out and the CHASSIS takes the blow.
    /// A car that lands on its suspension is not crashing, and the model says so
    /// by subtracting what the suspension asked for.
    const DROP_M: f64 = 16.0;

    fn slab_bits() -> (Transform, RigidBody3D, Collider3D) {
        (
            Transform {
                translation: Vec3d::new(0.0, -0.5, 0.0),
                ..Default::default()
            },
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(60.0, 0.5, 60.0),
                friction: 0.9,
                ..Default::default()
            },
        )
    }

    let def = catalogue_def("sedan");
    let spawn = inf_ecs::vehicle::RigSpawn {
        name: "Dropped".into(),
        at: DVec3::new(0.0, DROP_M, 0.0),
        yaw_deg: 0.0,
        paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
        clip: None,
        engine_voice: false,
        livery: None,
    };

    let shipped: Vec<Vec<u8>> = {
        let mut world = EcsWorld::new();
        let g = world.spawn_with_guid(GROUND, "slab", None);
        world.world_mut().entity_mut(g).insert(slab_bits());
        inf_ecs::vehicle::spawn_rig(&mut world, CHASSIS, &def, &spawn);
        world.propagate();
        let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
        (0..STEPS)
            .map(|_| {
                sim.step_once(RuntimeInput::default());
                inf_ecs::bodywork::damage_state_bytes(sim.world())
            })
            .collect()
    };

    let preview: Vec<Vec<u8>> = {
        use inf_editor_core::ipc::SpawnKind;
        let mut doc = SceneDoc::new();
        let g = doc.create_with_guid(GROUND, SpawnKind::Empty, "slab", None);
        doc.world_mut()
            .world_mut()
            .entity_mut(g)
            .insert(slab_bits());
        inf_editor_core::vehicle::spawn_vehicle(
            &mut doc,
            CHASSIS,
            &def,
            inf_editor_core::vehicle::VehicleSpawn {
                name: &spawn.name,
                at: spawn.at,
                yaw_deg: spawn.yaw_deg,
                paint: spawn.paint,
                clip: None,
                engine_voice: false,
                livery: None,
            },
        );
        doc.world_mut().propagate();
        let mut session = SimSession::enter(&mut doc, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
        let out = (0..STEPS)
            .map(|_| {
                session.step_once(&mut doc, SimInput::default());
                inf_ecs::bodywork::damage_state_bytes(doc.world())
            })
            .collect();
        session.exit(&mut doc);
        out
    };

    // **ANTI-VACUITY.** Two empty traces are equal too: the drop has to have
    // hurt the car, and the section has to have been empty before it did.
    let first_loud = shipped.iter().position(|b| !b.is_empty());
    eprintln!(
        "the drop from {DROP_M} m: the bodywork section is empty for the first {:?} steps and \
         {} bytes at the end",
        first_loud,
        shipped.last().map(|b| b.len()).unwrap_or(0)
    );
    assert!(
        shipped[0].is_empty(),
        "the car folded {} bytes before it had touched anything",
        shipped[0].len()
    );
    assert!(
        first_loud.is_some(),
        "a {DROP_M} m drop did nothing to the car — this arm compares two \
         recordings of nothing happening"
    );
    assert!(
        !shipped.last().unwrap().is_empty(),
        "the damage went away again"
    );

    assert_eq!(
        shipped, preview,
        "the shipped player and the editor's Simulate crashed the same car \
         differently — PIE would stop matching shipping the first time somebody \
         hit something"
    );
}
