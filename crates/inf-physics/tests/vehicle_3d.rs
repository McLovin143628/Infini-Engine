//! **P29.7: the raycast vehicle, against a real world.**
//!
//! `inf_ecs::vehicle`'s own tests pin the arithmetic — the spring, the engine
//! curve, the friction circle — as functions of numbers. These pin the *world*:
//! a rig built out of scene components, discovered by the bridge, stepped by the
//! one movement door, and asked afterwards where it actually is.
//!
//! Every claim here is about metres, radians or body counts. The one thing this
//! file deliberately does not do is assert on `VehicleOutcome`, which is a
//! report: a door that computed nothing and reported four grounded wheels would
//! satisfy that and nothing else.

use std::collections::BTreeMap;

use glam::{DVec3, EulerRot};
use uuid::Uuid;

use inf_ecs::components::{BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Transform};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::{ChassisState, VehicleControls, VehicleRig, WheelForce, WheelState};
use inf_ecs::EcsWorld;
use inf_physics::d3::PhysicsBridge3D;

const HZ: f64 = 60.0;
const DT: f64 = 1.0 / HZ;

const GROUND: Uuid = Uuid::from_u128(0x2907_1000);
const CHASSIS: Uuid = Uuid::from_u128(0x2907_1001);
const WHEEL_BASE: u128 = 0x2907_1010;
const LOOSE_SENSOR: Uuid = Uuid::from_u128(0x2907_1099);
/// The lake the buoyancy-ownership arm floats the car on.
const LAKE: Uuid = Uuid::from_u128(0x2907_1098);
/// The wheel-less hull and its screw (wave VEH2c).
const HULL: Uuid = Uuid::from_u128(0x2907_1200);
const THRUSTER: Uuid = Uuid::from_u128(0x2907_1201);
/// A hull is far denser than a car body: it is a shell full of engine, tanks and
/// people, and 400 kg/m3 in 1 000 kg/m3 water floats it with 40 % of its depth
/// under — which is a launch with freeboard and, crucially, a screw that is IN
/// the water at rest. (At the car's 150 it would float with its propeller in the
/// air, which the immersion rule correctly refuses to drive.)
const HULL_DENSITY: f64 = 400.0;
/// The rotorcraft and its disc (wave VEH2c).
const HELI: Uuid = Uuid::from_u128(0x2907_1300);
const ROTOR: Uuid = Uuid::from_u128(0x2907_1301);
/// An airframe is mostly air: 190 kg/m3 over the 8 m3 hull is 1 520 kg, which
/// is a light twin-seat helicopter.
const HELI_DENSITY: f64 = 190.0;
/// The equilibrium origin height for that density: half the hull, less the 40 %
/// of its full depth that is under. Spawned here so the arms measure a boat and
/// not a splash.
const HULL_FLOAT_Y: f64 = HALF.y - 2.0 * HALF.y * 0.4;

/// The chassis is 4 × 1 × 2 m at 150 kg/m³ — a hollow shell, per
/// `Collider3D::density`'s own note, which is 1 200 kg.
const HALF: Vec3d = Vec3d::new(2.0, 0.5, 1.0);
const DENSITY: f64 = 150.0;
const MASS_KG: f64 = 8.0 * DENSITY;
const WHEEL_RADIUS: f64 = 0.35;
/// The wheel centre in the chassis frame, at full extension.
const WHEEL_Y: f64 = -0.5;
/// So a chassis at this height has its wheels exactly touching a floor at y = 0
/// with the suspension fully extended — the placement an author would make.
const SPAWN_Y: f64 = -WHEEL_Y + WHEEL_RADIUS;

fn ground(world: &mut EcsWorld) {
    let e = world.spawn_with_guid(GROUND, "Ground", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, -0.5, 0.0);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(80.0, 0.5, 80.0),
            ..Default::default()
        },
    ));
}

/// The committed rig's shape, built by hand: a dynamic box with four sphere
/// **sensors** hanging off it.
fn car(world: &mut EcsWorld, y: f64) {
    let e = world.spawn_with_guid(CHASSIS, "Car", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, y, 0.0);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            // A car does not tumble on its own axis for want of damping; these
            // are the numbers the sample carries too.
            angular_damping: 0.5,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: HALF,
            density: DENSITY,
            friction: 0.5,
            ..Default::default()
        },
    ));
    for (i, (x, z)) in [(-0.9, 1.4), (0.9, 1.4), (-0.9, -1.4), (0.9, -1.4)]
        .into_iter()
        .enumerate()
    {
        let w = world.spawn_with_guid(Uuid::from_u128(WHEEL_BASE + i as u128), "Wheel", Some(e));
        let mut wt = Transform::IDENTITY;
        wt.translation = Vec3d::new(x, WHEEL_Y, z);
        world.world_mut().entity_mut(w).insert((
            wt,
            Collider3D {
                shape_kind: ColliderShape3DKind::Sphere,
                radius: WHEEL_RADIUS,
                sensor: true,
                ..Default::default()
            },
        ));
    }
}

/// A wheel-less hull: the same dynamic box, with a **box sensor** at the stern
/// where a screw goes (wave VEH2c).
fn hull(world: &mut EcsWorld, y: f64) {
    let e = world.spawn_with_guid(HULL, "Boat", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, y, 0.0);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            angular_damping: 0.5,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: HALF,
            density: HULL_DENSITY,
            friction: 0.5,
            ..Default::default()
        },
    ));
    thruster(world, THRUSTER, HULL);
}

/// One box sensor, at the stern and below the waterline.
fn thruster(world: &mut EcsWorld, guid: Uuid, parent: Uuid) {
    let parent = world.entity_of(parent).expect("a chassis to hang it on");
    let p = world.spawn_with_guid(guid, "Screw", Some(parent));
    let mut pt = Transform::IDENTITY;
    // Just above the keel, so it is under water whenever the hull is floating.
    pt.translation = Vec3d::new(0.0, -0.45, -1.9);
    world.world_mut().entity_mut(p).insert((
        pt,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(0.25, 0.15, 0.25),
            sensor: true,
            ..Default::default()
        },
    ));
}

struct Rig {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Rig {
    fn new(y: f64) -> Self {
        let mut world = EcsWorld::new();
        ground(&mut world);
        car(&mut world, y);
        world.mark_dirty();
        world.propagate();
        let bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        let mut rig = Self { world, bridge };
        rig.bridge.sync_from_world(&rig.world);
        rig
    }

    fn step(&mut self, n: u32) {
        for _ in 0..n {
            self.bridge.sync_from_world(&self.world);
            inf_physics::d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
            // The `vehicle` phase (island wave VEH1a): the door left
            // `step_character_movement`'s last statement for a `STEP_PHASES`
            // row of its own, so a fixture that is standing in for a host has
            // to call it too — in the slot both hosts call it in.
            inf_physics::d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
            self.bridge.step(DT);
            self.bridge.write_back_into(&mut self.world);
            self.world.propagate();
        }
    }

    fn drive(&mut self, controls: VehicleControls, n: u32) {
        for _ in 0..n {
            if let Some(v) = self.bridge.vehicle_mut(CHASSIS) {
                v.control(controls);
            }
            self.step(1);
        }
    }

    fn chassis(&self) -> Transform {
        let e = self.world.entity_of(CHASSIS).expect("the car exists");
        *self
            .world
            .world()
            .get::<Transform>(e)
            .expect("…with a transform")
    }

    fn y(&self) -> f64 {
        self.chassis().translation.y
    }

    fn z(&self) -> f64 {
        self.chassis().translation.z
    }

    fn yaw_deg(&self) -> f64 {
        self.chassis().rotation.y
    }
}

/// **The headline.** A rig authored with its wheels on the floor settles onto
/// its springs, stops, and stays there.
///
/// Three claims in one, and the third is the one a fixture would miss: it must
/// not sink (a suspension that answers zero load lets the box rest on its own
/// collider, half a metre lower), it must not bounce for ever (the damper), and
/// it must not creep (the friction circle at zero slip).
#[test]
fn a_parked_rig_settles_on_its_springs_and_stays_there() {
    let mut rig = Rig::new(SPAWN_Y);
    rig.step(300);
    let settled = rig.y();
    // 1 200 kg on four 20 000 N/m springs is 0.147 m of compression.
    let want = SPAWN_Y - MASS_KG * 9.81 / 4.0 / 20_000.0;
    assert!(
        (settled - want).abs() < 0.02,
        "the rig settled at {settled} m; its springs say {want} m"
    );
    // …and the box's own underside is at `settled - HALF.y`; if the springs had
    // failed it would be resting on the floor at HALF.y.
    assert!(
        settled - HALF.y > 0.1,
        "the chassis is resting on its own collider, not on its wheels: {settled}"
    );
    let before = rig.y();
    rig.step(120);
    assert!(
        (rig.y() - before).abs() < 1e-3,
        "a parked car crept {} m in two seconds",
        rig.y() - before
    );
    assert!(
        rig.z().abs() < 0.01,
        "…and wandered {} m sideways with no input",
        rig.z()
    );
}

/// **A floating vehicle keeps the water pass's force** — the door's OTHER
/// force-ownership rule, given the falsifier it did not have (P29.7 audit, A3).
///
/// `step_vehicles` clears a chassis's persistent force before applying its own,
/// because a rapier force persists until `reset_forces` (P20.2's law). It must
/// **not** clear one it does not own: `apply_water_forces` resets and re-applies
/// every buoyant body at fixed-step stage 8 and this door runs at stage 12, so
/// an unconditional clear would delete this step's buoyancy before the solver
/// ever saw it. `PhysicsBridge3D::is_buoyant` is the one line that says so.
///
/// Measured: with that guard removed (`if true`) the car sinks past −100 m in
/// ten seconds; with it, it floats. Before this arm the whole rule was a
/// comment — nothing in the workspace reddened when the guard was deleted.
///
/// There is deliberately **no ground**: the only thing holding this car up is
/// the force the vehicle door must not touch.
#[test]
fn a_buoyant_vehicle_keeps_the_force_the_water_pass_owns() {
    use inf_ecs::components::{Buoyancy, WaterBody};

    let mut world = EcsWorld::new();
    let lake = world.spawn_with_guid(LAKE, "Lake", None);
    world.world_mut().entity_mut(lake).insert((
        WaterBody {
            // Amplitude 0 makes the height query exact, so the tolerance below
            // is the solver's settling and not the Gerstner inversion's.
            wave_amplitude_m: 0.0,
            ..WaterBody::lake(0.0, inf_ecs::math::Vec2d::splat(100.0))
        },
        Transform::IDENTITY,
    ));
    car(&mut world, 0.0);
    let chassis = world.entity_of(CHASSIS).expect("the car exists");
    world.world_mut().entity_mut(chassis).insert(Buoyancy {
        // The chassis's own 150 kg/m³, so the hull floats with its deck clear
        // rather than awash — and so this arm's claim is Archimedes' rather
        // than a tuning coincidence.
        density_kg_m3: DENSITY,
        ..Default::default()
    });
    world.mark_dirty();
    world.propagate();

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&world);
    assert!(
        bridge.is_buoyant(CHASSIS),
        "the fixture must opt the chassis in, or the guard is never reached"
    );
    // The runtime's own order: sync → water → movement → **vehicle** → solve →
    // write-back. (The vehicle step was the last statement of the movement door
    // until island wave VEH1a gave it a `STEP_PHASES` row; both hosts now call
    // it here, and so does every fixture that wants to be the runtime.)
    let float_y = |world: &mut EcsWorld, bridge: &mut PhysicsBridge3D, n: u32| -> f64 {
        for _ in 0..n {
            bridge.sync_from_world(world);
            bridge.apply_water_forces(DT);
            inf_physics::d3::step_character_movement(world, bridge, DT);
            inf_physics::d3::step_vehicles(world, bridge, DT);
            bridge.step(DT);
            bridge.write_back_into(world);
            world.propagate();
        }
        world
            .world()
            .get::<Transform>(world.entity_of(CHASSIS).expect("the car survives"))
            .expect("…with a transform")
            .translation
            .y
    };
    let half = float_y(&mut world, &mut bridge, 300);
    let full = float_y(&mut world, &mut bridge, 300);
    assert!(
        full > -0.5,
        "the car sank to y = {full} in ten seconds — the vehicle door cleared \
         the force the water pass owns"
    );
    // …and it is *floating* rather than merely not-yet-fallen: it stopped. A
    // body whose buoyancy is being deleted every step is still accelerating
    // downward at five seconds and at ten.
    assert!(
        (full - half).abs() < 0.05,
        "the car was at {half} m after five seconds and {full} m after ten — it \
         is still sinking, not floating"
    );
}

/// A wheel is **consumed** by its vehicle, not mirrored into rapier — so it has
/// no collider, and the world it drives on is unchanged by having wheels in it.
#[test]
fn a_wheel_never_reaches_the_physics_world() {
    let mut rig = Rig::new(SPAWN_Y);
    rig.step(1);
    assert_eq!(rig.bridge.vehicle_count(), 1, "the car is recognised");
    for i in 0..4u128 {
        let w = Uuid::from_u128(WHEEL_BASE + i);
        assert!(
            rig.bridge.collider_of(w).is_none(),
            "wheel {i} was mirrored into rapier"
        );
        assert!(rig.bridge.body_of(w).is_none(), "wheel {i} has a body");
    }
    // Two bodies: the ground and the chassis. A wheel that slipped through would
    // make it six.
    assert_eq!(rig.bridge.body_count(), 2, "{:?}", rig.bridge.body_count());
}

/// **THE SEAM OPENS** (wave VEH2c): a chassis with NO wheels is a vehicle, its
/// thruster is consumed exactly as a wheel is, and the same box on a wheeled car
/// stays a trigger.
///
/// Three claims in one arm because they are one rule read from three sides, and
/// the third is the one that protects every level that already exists. Before
/// this wave `rig_of` answered `None` for a wheel-less chassis and
/// `reconcile_vehicles` walked the wheels map, so a boat was **structurally
/// invisible**: `vehicle_count()` was 0 and nothing could be sat in.
#[test]
fn a_wheel_less_hull_is_a_vehicle_and_its_thruster_is_consumed() {
    let mut world = EcsWorld::new();
    ground(&mut world);
    hull(&mut world, 1.0);
    world.mark_dirty();
    world.propagate();
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&world);

    // (a) It is a vehicle, with the rig the scene describes.
    assert_eq!(bridge.vehicle_count(), 1, "the hull is not recognised");
    let rig = bridge
        .vehicle_of(HULL)
        .expect("the hull is a vehicle")
        .rig();
    assert_eq!(rig.wheels.len(), 0);
    assert_eq!(rig.parts.len(), 1);
    assert_eq!(rig.parts[0].kind, inf_ecs::vehicle::PartKind::Thruster);
    assert_eq!(rig.parts[0].guid, THRUSTER);
    assert_eq!(rig.parts[0].mount_local, Vec3d::new(0.0, -0.45, -1.9));
    // …and the seat is still the collider's top face, so the interact door
    // needs nothing new to find it.
    assert_eq!(rig.seat_local, Vec3d::new(0.0, HALF.y, 0.0));

    // (b) The thruster is CONSUMED, exactly as a wheel is: two bodies (the
    //     ground and the hull) and no body or collider of its own.
    assert!(
        bridge.collider_of(THRUSTER).is_none(),
        "the screw is in rapier"
    );
    assert!(bridge.body_of(THRUSTER).is_none());
    assert_eq!(bridge.body_count(), 2);

    // (c) THE COMPATIBILITY CLAIM. The identical box under a WHEELED chassis is
    //     an ordinary trigger — mirrored, and not a part — so no car in any
    //     committed level changes by one byte.
    let mut world = EcsWorld::new();
    ground(&mut world);
    car(&mut world, SPAWN_Y);
    thruster(&mut world, THRUSTER, CHASSIS);
    world.mark_dirty();
    world.propagate();
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&world);
    let rig = bridge.vehicle_of(CHASSIS).expect("the car").rig();
    assert_eq!((rig.wheels.len(), rig.parts.len()), (4, 0));
    assert!(
        bridge.collider_of(THRUSTER).is_some(),
        "a trigger on a car was eaten by the part recogniser"
    );
    // Ground + chassis + the trigger's own body: THREE, where the hull above
    // had two. That difference is the consume rule, measured rather than
    // described — a mirrored sensor costs a body and a consumed part does not.
    assert_eq!(bridge.body_count(), 3);
}

/// …and the falsifier: a sphere sensor whose parent is **not** a chassis is an
/// ordinary sensor and is mirrored exactly as it always was.
///
/// Without this the consume rule would be "every sphere sensor disappears",
/// which is a silent removal of a trigger volume from somebody's level.
#[test]
fn a_sphere_sensor_that_is_not_a_wheel_is_still_mirrored() {
    let mut world = EcsWorld::new();
    ground(&mut world);
    // Parented to the GROUND, which is static — so it is not a wheel.
    let parent = world.entity_of(GROUND).unwrap();
    let e = world.spawn_with_guid(LOOSE_SENSOR, "Trigger", Some(parent));
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(3.0, 1.0, 0.0);
    world.world_mut().entity_mut(e).insert((
        t,
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 1.0,
            sensor: true,
            ..Default::default()
        },
    ));
    world.mark_dirty();
    world.propagate();
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&world);
    assert_eq!(
        bridge.vehicle_count(),
        0,
        "a static parent is not a vehicle"
    );
    assert!(
        bridge.collider_of(LOOSE_SENSOR).is_some(),
        "an ordinary sphere sensor must survive the wheel rule"
    );
}

/// Throttle moves it, and the brake stops it — in metres, over the floor.
#[test]
fn throttle_drives_it_and_the_brake_stops_it() {
    let mut rig = Rig::new(SPAWN_Y);
    rig.step(60);
    let start = rig.z();
    rig.drive(
        VehicleControls {
            throttle: 1.0,
            ..Default::default()
        },
        180,
    );
    let travelled = rig.z() - start;
    assert!(
        travelled > 5.0,
        "three seconds of full throttle moved it {travelled} m"
    );
    // A brake takes it to a stop and holds it there.
    let before_brake = rig.z();
    rig.drive(
        VehicleControls {
            brake: 1.0,
            ..Default::default()
        },
        180,
    );
    let coasted = rig.z() - before_brake;
    // 0.7 of the accelerating distance, and the number moved with the model
    // rather than the model being bent to the number. This ratio IS the brake's
    // strength against the engine's: the same speed takes `v^2/2a` to reach and
    // `v^2/2b` to shed.
    //
    // It was 0.5 under P29.7, whose brake was a force with no limit and no such
    // thing as a locked wheel. VEH2a's brake locks — 1 050 N.m against a tyre
    // that can take about 900 — so it stops at the SLIDING coefficient, and the
    // arm went to 0.8 for one commit at a measured 16.09 m of 21.25. **ABS took
    // it back**: holding the slip near the peak instead of past it is worth 17 %
    // of the stop, measured at 13.29 m of 21.69, so the arm is tighter than the
    // model it now guards and looser than the fiction it replaced.
    println!("THE BRAKE: {travelled:.2} m accelerating, {coasted:.2} m stopping");
    assert!(
        coasted < travelled * 0.7,
        "braking for as long as it accelerated covered {coasted} m against {travelled} m"
    );
    let stopped = rig.z();
    rig.drive(
        VehicleControls {
            brake: 1.0,
            ..Default::default()
        },
        60,
    );
    assert!(
        (rig.z() - stopped).abs() < 0.05,
        "a braked car rolled another {} m",
        rig.z() - stopped
    );
    // **And a resistive force may not REVERSE the motion** (P29.7 audit, A4).
    //
    // The ledger's first measured defect — rolling resistance applied outside
    // the stop clamp — is asserted here and NOT by the position window above,
    // which is twenty times too wide to see it once the pitch pump is fixed.
    // The two defects were measured together (5.8 cm/s), and separating them
    // was this audit's job: with `contact_velocity` correct, restoring the
    // unclamped rolling term leaves the *position* pinned to four decimal
    // places and puts a permanent **−2.45 mm/s** on the body — a car that is
    // driving backwards against its own brakes and getting nowhere, which is
    // the same defect one integration short of being visible.
    //
    // So the claim is the body's own velocity, read out of the solver, and it
    // is a claim about the world rather than about a report: a braked car is
    // stopped, not slowly reversing. Correct: 6.4e−10 m/s. Defective: 2.45e−3.
    rig.drive(
        VehicleControls {
            brake: 1.0,
            ..Default::default()
        },
        240,
    );
    let body = rig
        .bridge
        .body_of(CHASSIS)
        .expect("the chassis is mirrored");
    let residual = rig
        .bridge
        .world()
        .body_linvel(body)
        .expect("…with a velocity")
        .z;
    assert!(
        residual.abs() < 1e-6,
        "a braked car is still moving at {residual} m/s after four seconds on \
         the brake — a resistive force that overshoots reverses the motion"
    );
}

/// Steering turns it, and turns it the way the sign says.
///
/// `+steer` is right, and a right turn is a **positive** yaw here: euler-Y
/// rotates `+Z` (forward) toward `+X` (right), which is the convention
/// `Transform::quat` and the movement step's `rotate_from_frame` both use. The
/// arm exists because a sign error is invisible to every other claim in the
/// file, and it measures at **1.5 seconds** rather than at four — full lock at
/// low speed turns about 43°/s, so a longer sample wraps past ±180° and the sign
/// of the answer stops meaning what it looks like.
#[test]
fn steering_turns_it_and_the_sign_is_the_one_the_control_says() {
    let turn = |steer: f64| {
        let mut rig = Rig::new(SPAWN_Y);
        rig.step(60);
        rig.drive(
            VehicleControls {
                throttle: 1.0,
                steer,
                ..Default::default()
            },
            90,
        );
        rig.yaw_deg()
    };
    let right = turn(1.0);
    assert!(
        right > 5.0 && right < 170.0,
        "a second and a half of full right lock turned it {right} degrees"
    );
    let left = turn(-1.0);
    assert!(left < -5.0, "…and a left turn the other way, not {left}");
    assert!(
        (left + right).abs() < right.abs() * 0.35,
        "the two turns should mirror: {right} against {left}"
    );
}

/// The same world, stepped twice, is the same world — the determinism the whole
/// replay discipline rests on, over a dynamic body under a force this wave
/// added.
#[test]
fn two_identical_rigs_drive_byte_for_byte() {
    let trace = |seed: u32| {
        let mut rig = Rig::new(SPAWN_Y);
        let mut out: Vec<u8> = Vec::new();
        for i in 0..240u32 {
            let controls = VehicleControls {
                throttle: if (i + seed) % 90 < 60 { 1.0 } else { -0.5 },
                steer: if i % 70 < 35 { 0.8 } else { -0.6 },
                brake: 0.0,
                handbrake: i % 121 == 0,
                vertical: 0.0,
                occupied: true,
            };
            rig.drive(controls, 1);
            let t = rig.chassis();
            for v in [
                t.translation.x,
                t.translation.y,
                t.translation.z,
                t.rotation.x,
                t.rotation.y,
                t.rotation.z,
            ] {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        out
    };
    let a = trace(0);
    let b = trace(0);
    assert_eq!(a, b, "two identical vehicle runs must agree bit for bit");
    assert_ne!(
        a,
        trace(7),
        "…and a different control script must produce a different world, or the \
         comparison above is a comparison of two cars that never moved"
    );
}

/// **The handbrake slows the car**, and the world says so in metres.
///
/// The other half of the claim — that it is the REAR wheels, which is what makes
/// a handbrake turn a turn rather than a stop — is asserted at the model, where
/// a wheel's identity is visible (`inf_ecs::vehicle`'s
/// `the_handbrake_is_the_rear_wheels_and_only_those`).
#[test]
fn the_handbrake_slows_the_car() {
    let roll = |handbrake: bool| -> f64 {
        let mut rig = Rig::new(SPAWN_Y);
        rig.step(60);
        rig.drive(
            VehicleControls {
                throttle: 1.0,
                ..Default::default()
            },
            150,
        );
        let start = rig.z();
        rig.drive(
            VehicleControls {
                handbrake,
                ..Default::default()
            },
            240,
        );
        rig.z() - start
    };
    let free = roll(false);
    let held = roll(true);
    assert!(free > 3.0, "the fixture must be rolling: {free} m");
    // Four seconds rather than two, and the arm got STRONGER for it (0.6 where
    // it used to ask 0.7). A handbrake now locks two wheels instead of applying
    // 9 000 N at the rear contacts, so it decelerates at what two sliding tyres
    // can actually take — about 4 m/s^2 against the old model's 7.5. That is
    // less brake and more car, and a longer window is the honest way to measure
    // it rather than a looser threshold.
    assert!(
        held < free * 0.6,
        "four seconds on the handbrake covered {held} m against {free} m coasting"
    );
}

// ── the trait seam ──────────────────────────────────────────────────────────

/// A second implementation of [`inf_ecs::vehicle::Vehicle`] — a hovercraft with
/// no wheels that pushes straight up — which is what makes the trait a seam
/// rather than a description of the one class this phase ships.
///
/// It counts what the door did to it, so the arm below can assert that the door
/// routed input, cast nothing it did not ask for, and applied what it answered.
struct Hover {
    rig: VehicleRig,
    wheels: Vec<WheelState>,
    controls: VehicleControls,
    solves: u32,
    lift_n: f64,
}

impl inf_ecs::vehicle::Vehicle for Hover {
    /// A hovercraft has no gearbox; neutral is the honest answer.
    fn gear(&self) -> i32 {
        0
    }

    fn rig(&self) -> &VehicleRig {
        &self.rig
    }
    fn set_rig(&mut self, rig: VehicleRig) {
        self.rig = rig;
    }
    fn wheels(&self) -> &[WheelState] {
        &self.wheels
    }
    fn wheels_mut(&mut self) -> &mut [WheelState] {
        &mut self.wheels
    }
    fn control(&mut self, controls: VehicleControls) {
        self.controls = controls;
    }
    fn tune(&mut self, name: &str, value: f64) -> bool {
        if name == "lift_n" {
            self.lift_n = value;
            return true;
        }
        false
    }
    fn suspension_rest_m(&self) -> f64 {
        0.0
    }
    fn seat_warp(&self) -> (f64, inf_anim::WarpWindow) {
        (0.2, inf_anim::WarpWindow::new(0.0, 0.2))
    }
    fn solve(&mut self, chassis: ChassisState, _dt: f64, out: &mut Vec<WheelForce>) {
        self.solves = self.solves.saturating_add(1);
        out.push(WheelForce {
            point: chassis.position,
            force: DVec3::Y * self.lift_n * self.controls.throttle.max(0.0),
        });
    }
}

/// **The seam is real**: an island class installed over the derived one is the
/// class the door drives.
#[test]
fn an_installed_class_is_the_one_the_door_drives() {
    let mut rig = Rig::new(SPAWN_Y);
    rig.step(1);
    let derived = rig
        .bridge
        .vehicle_of(CHASSIS)
        .expect("the car was derived")
        .rig()
        .clone();
    assert_eq!(derived.wheels.len(), 4);
    let hover = Hover {
        rig: derived,
        wheels: Vec::new(),
        controls: VehicleControls::default(),
        solves: 0,
        // Twice its weight, so "it went up" is unambiguous.
        lift_n: MASS_KG * 9.81 * 2.0,
    };
    rig.bridge.install_vehicle(CHASSIS, Box::new(hover));
    assert!(
        rig.bridge
            .vehicle_mut(CHASSIS)
            .expect("installed")
            .tune("lift_n", MASS_KG * 9.81 * 2.0),
        "the tuning door reaches the installed class by its own names"
    );
    let before = rig.y();
    rig.drive(
        VehicleControls {
            throttle: 1.0,
            ..Default::default()
        },
        120,
    );
    assert!(
        rig.y() - before > 1.0,
        "the hovercraft rose {} m; the door is still driving the raycast class",
        rig.y() - before
    );
    // …and the derived rig survived the swap, which is what `set_rig` is for:
    // the bridge re-derives every sync and must not overwrite the installed
    // class with a fresh `RaycastVehicle`.
    assert_eq!(
        rig.bridge.vehicle_of(CHASSIS).unwrap().rig().wheels.len(),
        4,
        "the re-derive replaced the installed class"
    );
}

/// A rig in the air produces no force at all, so a car thrown off a cliff falls
/// like a box — the off path, asserted rather than assumed.
#[test]
fn a_rig_in_the_air_falls_like_a_box() {
    let mut rig = Rig::new(SPAWN_Y + 20.0);
    let start = rig.y();
    rig.drive(
        VehicleControls {
            throttle: 1.0,
            ..Default::default()
        },
        30,
    );
    let fell = start - rig.y();
    // Free fall for half a second is about 1.2 m; a suspension that pushed
    // against nothing would hold it up.
    assert!(
        fell > 1.0 && fell < 1.5,
        "an airborne rig fell {fell} m in half a second"
    );
    assert!(
        rig.z().abs() < 0.05,
        "…and full throttle in the air moved it {} m forward",
        rig.z()
    );
}

/// The rig is derived from the scene through the same recogniser a sample
/// generator reads — one spelling of "what is a wheel".
#[test]
fn the_public_deriver_agrees_with_the_bridge() {
    let mut rig = Rig::new(SPAWN_Y);
    rig.step(1);
    let by_hand = inf_ecs::vehicle::rig_of(&rig.world, CHASSIS).expect("derived by hand");
    let in_bridge = rig.bridge.vehicle_of(CHASSIS).unwrap().rig();
    assert_eq!(&by_hand.wheels, &in_bridge.wheels);
    assert_eq!(by_hand.seat_local, in_bridge.seat_local);
    assert_eq!(
        by_hand.seat_local,
        Vec3d::new(0.0, HALF.y, 0.0),
        "the seat is the top face of the chassis collider"
    );
    // A guid that is not a chassis answers with a refusal rather than a rig.
    assert!(inf_ecs::vehicle::rig_of(&rig.world, GROUND).is_none());
}

/// The wheels are drawn where the suspension put them: compressed under load,
/// extended in the air, and steered by the control.
#[test]
fn the_wheels_are_drawn_where_the_suspension_put_them() {
    let mut rig = Rig::new(SPAWN_Y);
    rig.step(180);
    let local = |guid: Uuid, world: &EcsWorld| -> Transform {
        let e = world.entity_of(guid).unwrap();
        *world.world().get::<Transform>(e).unwrap()
    };
    let front_left = Uuid::from_u128(WHEEL_BASE);
    let t = local(front_left, &rig.world);
    assert!(
        t.translation.y > WHEEL_Y + 0.05,
        "a loaded wheel must ride up into its arch: {}",
        t.translation.y
    );
    // Steering shows on the front wheels and not on the rear.
    rig.drive(
        VehicleControls {
            steer: 1.0,
            ..Default::default()
        },
        10,
    );
    let front = local(front_left, &rig.world).rotation.y;
    let rear = local(Uuid::from_u128(WHEEL_BASE + 2), &rig.world)
        .rotation
        .y;
    assert!(front.abs() > 5.0, "the front wheel steered {front} degrees");
    assert_eq!(rear, 0.0, "the rear wheel steered {rear} degrees");
    // And the rotation the door writes is the one a renderer would read.
    let q = local(front_left, &rig.world).quat();
    let (y, _, _) = q.to_euler(EulerRot::YXZ);
    assert!((y.to_degrees() - front).abs() < 1e-9);
}

/// A parked rig's wheel state is stable across a re-derive — the authored mount
/// is taken once, so a sync in the middle of a run does not walk the car into
/// the floor.
#[test]
fn a_resync_does_not_move_a_parked_car() {
    let mut rig = Rig::new(SPAWN_Y);
    rig.step(180);
    let settled = rig.y();
    let mounts: BTreeMap<Uuid, Vec3d> = rig
        .bridge
        .vehicle_of(CHASSIS)
        .unwrap()
        .rig()
        .wheels
        .iter()
        .map(|w| (w.guid, w.mount_local))
        .collect();
    // Twenty extra syncs, which is what an editor doing anything at all causes.
    for _ in 0..20 {
        rig.bridge.sync_from_world(&rig.world);
    }
    let after: BTreeMap<Uuid, Vec3d> = rig
        .bridge
        .vehicle_of(CHASSIS)
        .unwrap()
        .rig()
        .wheels
        .iter()
        .map(|w| (w.guid, w.mount_local))
        .collect();
    assert_eq!(
        mounts, after,
        "the authored mount moved on a re-derive — the visual write fed back in"
    );
    rig.step(60);
    assert!(
        (rig.y() - settled).abs() < 1e-3,
        "the car moved {} m after twenty resyncs",
        rig.y() - settled
    );
}

/// A tuner's probe, and the one that found the pitch pump
/// (`ChassisState::contact_velocity`'s doc records the measurement).
///
/// `#[ignore]`d because it is a diagnostic rather than a claim — the same
/// standing this phase's own `probe_the_course` has. Run it with
/// `--ignored --nocapture` when a spring rate changes.
#[test]
#[ignore]
fn probe_brake() {
    let mut rig = Rig::new(SPAWN_Y);
    rig.step(60);
    rig.drive(
        VehicleControls {
            throttle: 1.0,
            ..Default::default()
        },
        180,
    );
    for i in 0..12 {
        rig.drive(
            VehicleControls {
                brake: 1.0,
                ..Default::default()
            },
            30,
        );
        let body = rig.bridge.body_of(CHASSIS).unwrap();
        let lin = rig.bridge.world().body_linvel(body).unwrap();
        let ang = rig.bridge.world().body_angvel(body).unwrap();
        println!(
            "t={:.1}s z={:.4} lin={:?} ang={:?}",
            (i as f64 + 1.0) * 0.5,
            rig.z(),
            lin,
            ang
        );
    }
}

// ── the seat: enter, drive, exit (P29.7) ────────────────────────────────────

const HERO: Uuid = Uuid::from_u128(0x2907_1002);
const HERO_2: Uuid = Uuid::from_u128(0x2907_1003);
const HERO_RADIUS: f64 = 0.3;

/// A character beside the car, on the floor, facing it.
fn hero(world: &mut EcsWorld, guid: Uuid, x: f64, z: f64) {
    let cm = inf_ecs::components::CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let e = world.spawn_with_guid(guid, "Hero", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(x, cm.stand_half_height_m + HERO_RADIUS, z);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(HERO_RADIUS, cm.stand_half_height_m, HERO_RADIUS),
            radius: HERO_RADIUS,
            ..Default::default()
        },
        inf_ecs::components::CharacterController3D::default(),
        cm,
    ));
}

/// A rig with a driver beside the car.
struct Crew {
    rig: Rig,
}

impl Crew {
    fn new() -> Self {
        let mut world = EcsWorld::new();
        ground(&mut world);
        car(&mut world, SPAWN_Y);
        hero(&mut world, HERO, 2.5, 0.0);
        world.mark_dirty();
        world.propagate();
        let bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        let mut rig = Rig { world, bridge };
        rig.bridge.sync_from_world(&rig.world);
        Self { rig }
    }

    fn step(&mut self, intent: &inf_ecs::movement::MovementIntent, n: u32) {
        for _ in 0..n {
            inf_ecs::movement::apply_intent(&mut self.rig.world, intent);
            self.rig.step(1);
        }
    }

    fn driver(&self) -> inf_ecs::components::CharacterMovement {
        let e = self.rig.world.entity_of(HERO).expect("the hero exists");
        self.rig
            .world
            .world()
            .get::<inf_ecs::components::CharacterMovement>(e)
            .expect("…with a movement component")
            .clone()
    }

    fn driver_pos(&self) -> DVec3 {
        let e = self.rig.world.entity_of(HERO).unwrap();
        self.rig
            .world
            .world()
            .get::<Transform>(e)
            .unwrap()
            .translation
            .to_dvec3()
    }
}

fn interact() -> inf_ecs::movement::MovementIntent {
    inf_ecs::movement::MovementIntent {
        interact: true,
        ..Default::default()
    }
}

fn forward() -> inf_ecs::movement::MovementIntent {
    inf_ecs::movement::MovementIntent {
        move_input: inf_ecs::math::Vec2d::new(0.0, 1.0),
        ..Default::default()
    }
}

/// **The enter choreography is a WINDOW, not a slide.**
///
/// `WarpWindow` has been named as a zero-caller by three ledgers, and this is
/// what having one means: before the window opens the character has not moved,
/// while it is open the character warps to the seat, and after it closes the
/// character is exactly on the seat. An implementation that lerped over the
/// whole duration would fail the first clause, which is the one that makes it a
/// choreography.
#[test]
fn the_enter_warp_is_a_window_and_lands_the_character_on_the_seat() {
    let mut crew = Crew::new();
    crew.step(&Default::default(), 60);
    let standing = crew.driver_pos();
    crew.step(&interact(), 1);
    let cm = crew.driver();
    assert_eq!(
        cm.mode,
        inf_ecs::components::MovementMode::Driving,
        "the enter control takes"
    );
    assert!(cm.runtime.seat.entering, "…and the warp is running");
    assert_eq!(cm.runtime.seat.vehicle, CHASSIS);
    // The character's own collider is parked for the whole choreography.
    let collider = crew
        .rig
        .bridge
        .collider_of(HERO)
        .expect("a mirrored capsule");
    assert!(
        crew.rig.bridge.world().collider_enabled(collider) == Some(false),
        "a capsule sliding into a seat with its collider live pushes the car away"
    );

    // Before the window opens (0.10 s), nothing has moved.
    crew.step(&Default::default(), 4);
    let early = crew.driver_pos();
    assert!(
        (early - standing).length() < 0.02,
        "the character moved {} m before the warp window opened",
        (early - standing).length()
    );

    // After it closes (0.45 s), the character is on the seat.
    crew.step(&Default::default(), 30);
    assert!(!crew.driver().runtime.seat.entering, "the warp finishes");
    let seat = crew.rig.chassis().translation.to_dvec3()
        + DVec3::Y * (crew.driver().stand_half_height_m + HERO_RADIUS + HALF.y);
    let seated = crew.driver_pos();
    assert!(
        (seated - seat).length() < 0.05,
        "the character sat at {seated:?}, the seat is at {seat:?}"
    );
    assert!(
        (seated - standing).length() > 1.0,
        "…and it actually travelled to get there"
    );
    // The collider comes back on the way out, and only then.
    assert!(crew.rig.bridge.world().collider_enabled(collider) == Some(false));
}

/// A vehicle out of reach is a refusal, as a value: the control does nothing and
/// the character carries on standing.
#[test]
fn a_vehicle_out_of_reach_is_a_refusal_and_not_a_teleport() {
    let mut world = EcsWorld::new();
    ground(&mut world);
    car(&mut world, SPAWN_Y);
    hero(&mut world, HERO, 40.0, 0.0);
    world.mark_dirty();
    world.propagate();
    let bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    let mut crew = Crew {
        rig: Rig { world, bridge },
    };
    crew.rig.bridge.sync_from_world(&crew.rig.world);
    crew.step(&Default::default(), 30);
    let before = crew.driver_pos();
    crew.step(&interact(), 30);
    assert_eq!(
        crew.driver().mode,
        inf_ecs::components::MovementMode::Grounded
    );
    assert!(!crew.driver().runtime.seat.is_seated());
    assert!((crew.driver_pos() - before).length() < 0.05);
}

/// **An enter press the step could not honour is CONSUMED, not banked**
/// (P29.7 audit, A1).
///
/// The movement door's own law, written above its edge-consumption block: *the
/// edges are consumed whether or not they were honoured; an unconsumed edge
/// fires again next step off the same press.* `press_interact` is only read on a
/// grounded step, and it was the one edge nothing cleared on the other paths —
/// so a press made in mid-air survived the whole fall and climbed into whatever
/// car happened to be in reach when the character landed, which is the input
/// buffer nobody asked for.
///
/// The fixture drops the character in beside the car so the press lands on an
/// airborne step, and then lets it land **inside** `ENTER_REACH_M` — which is
/// what makes this an assertion about the edge rather than about the reach.
#[test]
fn an_enter_press_made_in_the_air_does_not_fire_on_landing() {
    let mut world = EcsWorld::new();
    ground(&mut world);
    car(&mut world, SPAWN_Y);
    hero(&mut world, HERO, 2.5, 0.0);
    {
        let e = world.entity_of(HERO).expect("the hero exists");
        let mut t = world
            .world_mut()
            .get_mut::<Transform>(e)
            .expect("…with a transform");
        t.translation.y += 6.0;
    }
    world.mark_dirty();
    world.propagate();
    let bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    let mut crew = Crew {
        rig: Rig { world, bridge },
    };
    crew.rig.bridge.sync_from_world(&crew.rig.world);
    crew.step(&Default::default(), 5);
    assert!(
        !crew.driver().runtime.grounded,
        "the fixture must press the control in the AIR, or it proves nothing"
    );
    crew.step(&interact(), 1);
    assert!(
        !crew.driver().runtime.seat.is_seated(),
        "an airborne character climbed into a car"
    );
    crew.step(&Default::default(), 180);
    assert!(
        crew.driver().runtime.grounded,
        "the character must land for this arm to mean anything"
    );
    // …and it landed in reach, so the refusal below is the EDGE and not the
    // distance.
    let seat = crew.rig.chassis().translation.to_dvec3() + DVec3::Y * HALF.y;
    let feet = crew.driver_pos() - DVec3::Y * (crew.driver().stand_half_height_m + HERO_RADIUS);
    assert!(
        (seat - feet).length() < inf_physics::d3::vehicle::ENTER_REACH_M,
        "the character landed {} m from the seat, outside the reach",
        (seat - feet).length()
    );
    assert!(
        !crew.driver().runtime.seat.is_seated(),
        "a press the airborne step could not honour was banked and fired on \
         landing"
    );
}

/// **The car carries its driver**, and the driver's stick drives the car.
#[test]
fn the_driver_drives_and_the_car_carries_the_driver() {
    let mut crew = Crew::new();
    crew.step(&Default::default(), 60);
    crew.step(&interact(), 1);
    crew.step(&Default::default(), 40);
    let car_before = crew.rig.z();
    let driver_before = crew.driver_pos();
    crew.step(&forward(), 180);
    let car_moved = crew.rig.z() - car_before;
    let driver_moved = crew.driver_pos().z - driver_before.z;
    assert!(
        car_moved > 5.0,
        "three seconds of driving moved the car {car_moved} m"
    );
    assert!(
        (driver_moved - car_moved).abs() < 0.05,
        "the driver moved {driver_moved} m and the car {car_moved} m"
    );
    // The driver's own velocity is the car's, which is what makes the exit
    // handoff a handoff rather than a guess.
    //
    // To within one step: the movement door runs BEFORE the solver, so what the
    // runtime holds is the chassis velocity as it was at the top of this step —
    // the same staleness every derived output on `MovementRuntime` carries, and
    // the reason the *position* half is corrected in the write-back instead.
    let body = crew.rig.bridge.body_of(CHASSIS).unwrap();
    let linvel = crew.rig.bridge.world().body_linvel(body).unwrap();
    let seen = crew.driver().runtime.velocity.to_dvec3();
    assert!(
        (seen - linvel).length() < 0.2,
        "the driver's velocity {seen:?} is not the car's {linvel:?}"
    );
    assert!(
        seen.length() > 5.0,
        "…and the comparison is not two zeroes: {seen:?}"
    );
}

/// **Leaving a moving vehicle inherits its velocity** — the ragdoll's precedent,
/// and the reason `Driving` has an airborne destination in the mode table.
#[test]
fn leaving_a_moving_vehicle_inherits_its_velocity() {
    let mut crew = Crew::new();
    crew.step(&Default::default(), 60);
    crew.step(&interact(), 1);
    crew.step(&Default::default(), 40);
    crew.step(&forward(), 180);
    let body = crew.rig.bridge.body_of(CHASSIS).unwrap();
    let linvel = crew.rig.bridge.world().body_linvel(body).unwrap();
    assert!(linvel.z > 3.0, "the car must be moving: {linvel:?}");
    crew.step(&interact(), 1);
    let cm = crew.driver();
    assert_eq!(
        cm.mode,
        inf_ecs::components::MovementMode::FallControlled,
        "stepping out at {} m/s is not standing up",
        linvel.length()
    );
    assert!(
        (cm.runtime.velocity.to_dvec3() - linvel).length() < 0.5,
        "the exit velocity {:?} is not the car's {linvel:?}",
        cm.runtime.velocity
    );
    assert!(!cm.runtime.seat.is_seated());
    // The collider comes back, or the character walks through the world for ever.
    let collider = crew.rig.bridge.collider_of(HERO).unwrap();
    assert_eq!(
        crew.rig.bridge.world().collider_enabled(collider),
        Some(true)
    );
    // …and it lands, rather than sliding for ever.
    crew.step(&Default::default(), 600);
    assert!(
        crew.driver().runtime.grounded,
        "the character never landed after the exit"
    );
}

/// Leaving a **stopped** vehicle is a stand, not a fall.
#[test]
fn leaving_a_parked_vehicle_is_a_stand() {
    let mut crew = Crew::new();
    crew.step(&Default::default(), 60);
    crew.step(&interact(), 1);
    crew.step(&Default::default(), 60);
    crew.step(&interact(), 1);
    assert_eq!(
        crew.driver().mode,
        inf_ecs::components::MovementMode::Grounded
    );
    crew.step(&Default::default(), 120);
    assert!(crew.driver().runtime.grounded);
    // Beside the car, not inside it.
    let gap = (crew.driver_pos() - crew.rig.chassis().translation.to_dvec3()).length();
    assert!(
        gap > HALF.x.min(HALF.z),
        "the exit put the character {gap} m from the car's centre"
    );
}

/// **THE SEAT AND AN AUTHORED INTERACTABLE GO THROUGH ONE DOOR** (island wave
/// I5).
///
/// P29.7's `try_enter` was the only interaction in the engine. It is now a call
/// into `inf_physics::d3::interact`, which builds candidates out of *both* seats
/// and `Interactable` components and ranks them with the one Ring-0 rule.
///
/// Three claims, and the third is the one that says the migration is real rather
/// than a rename:
///
/// 1. the seat is still found, with the same reach and the same tie-break —
///    which the twenty-one arms above already say, and this one says again
///    through the *new* door;
/// 2. an authored interactable is a candidate through the same door;
/// 3. **a nearer interactable that is not an `Enter` does not put the character
///    in the car** — the door answers with the nearest thing, and only the
///    `Enter` verb has a consumer in the movement step. Without this, "one door"
///    would mean "the vehicle path now fires on a lamp post".
#[test]
fn the_seat_and_an_authored_interactable_share_one_door() {
    use inf_ecs::interact::{InteractVerb, Interactable, NO_VIEW_TEST_DEG};
    use std::collections::BTreeSet;

    // ── (1) the door finds the seat, from where P29.7's own fixture stands ──
    let crew = Crew::new();
    let feet = crew.driver_pos() - DVec3::Y * (crew.driver().stand_half_height_m + HERO_RADIUS);
    let seat = inf_physics::d3::interact::nearest_seat(&crew.rig.bridge, feet, &BTreeSet::new());
    assert_eq!(seat, Some(CHASSIS), "the one door lost the seat");
    let hit = inf_physics::d3::interact::resolve(
        &crew.rig.world,
        &crew.rig.bridge,
        feet,
        0.0,
        &BTreeSet::new(),
    )
    .expect("the full door finds it too");
    assert_eq!(hit.guid, CHASSIS);
    assert_eq!(hit.verb, InteractVerb::Enter);
    assert_eq!(hit.label, inf_physics::d3::interact::VEHICLE_LABEL);

    // ── (2) and (3): a NEARER interactable that is not a seat ──
    let mut crew = Crew::new();
    let lamp = Uuid::from_u128(0x2907_10AA);
    {
        let hero_pos = crew.driver_pos();
        let e = crew.rig.world.spawn_with_guid(lamp, "Lamp", None);
        let mut t = Transform::IDENTITY;
        // Half a metre from the character — nearer than the seat by a long way.
        t.translation = Vec3d::new(hero_pos.x + 0.5, 0.0, hero_pos.z);
        crew.rig.world.world_mut().entity_mut(e).insert((
            t,
            Interactable {
                verb: InteractVerb::Use,
                label: "lamp".into(),
                range_m: 3.0,
                enabled: true,
                view_cone_deg: NO_VIEW_TEST_DEG,
                grip: None,
            },
        ));
        crew.rig.world.mark_dirty();
        crew.rig.world.propagate();
    }
    let feet = crew.driver_pos() - DVec3::Y * (crew.driver().stand_half_height_m + HERO_RADIUS);
    let hit = inf_physics::d3::interact::resolve(
        &crew.rig.world,
        &crew.rig.bridge,
        feet,
        0.0,
        &BTreeSet::new(),
    )
    .expect("something is in reach");
    println!(
        "the nearest candidate is {} ({:?}) at {:.3} m",
        hit.label, hit.verb, hit.distance_m
    );
    assert_eq!(
        hit.guid, lamp,
        "the nearer authored interactable did not win"
    );
    assert_eq!(hit.verb, InteractVerb::Use);

    // …and pressing E with the lamp nearest does **not** enter the car.
    crew.step(&interact(), 1);
    crew.step(&inf_ecs::movement::MovementIntent::default(), 4);
    assert_ne!(
        crew.driver().mode,
        inf_ecs::components::MovementMode::Driving,
        "a `Use` interactable put the character in the driving seat"
    );

    // The control: with the lamp DISABLED, the same press enters the car — so
    // the assertion above is about the verb and not about a broken press.
    let e = crew.rig.world.entity_of(lamp).unwrap();
    crew.rig
        .world
        .world_mut()
        .get_mut::<Interactable>(e)
        .unwrap()
        .enabled = false;
    crew.rig.world.mark_dirty();
    crew.rig.world.propagate();
    crew.step(&interact(), 1);
    crew.step(&inf_ecs::movement::MovementIntent::default(), 4);
    assert_eq!(
        crew.driver().mode,
        inf_ecs::components::MovementMode::Driving,
        "with nothing nearer, the same press must still enter the car"
    );
}

/// **THE MERGED CANDIDATE WALK IS ONE SORTED WALK, AND A TIE PROVES IT** (I5
/// audit, A3).
///
/// `d3::interact::candidates` concatenates two lists that are each already in
/// `Guid` order — the seats out of a `BTreeMap`, the interactables out of their
/// own sorted walk — and then sorts the whole thing. The sort is the load-bearing
/// line and its own comment says so ("a seat and an item at exactly the same
/// distance must resolve the same way in both hosts"), and **deleting it killed
/// nothing in this tree**: every other arm puts its candidates at different
/// distances, where the tie-break never runs.
///
/// So this one manufactures the tie exactly, by putting the item **on the seat**.
/// The item's `Guid` is lower than the chassis's, so the `Guid`-ordered walk with
/// a strict `<` must answer with the item; without the merged sort the seats come
/// first in the concatenation and the *seat* wins, which is the wrong answer and
/// — worse — an answer that depends on which list was extended onto which.
#[test]
fn a_seat_and_an_item_at_the_same_distance_break_by_guid() {
    use inf_ecs::interact::{InteractVerb, Interactable, NO_VIEW_TEST_DEG};
    use std::collections::BTreeSet;

    let mut crew = Crew::new();
    let (seat, _, _) = inf_physics::d3::vehicle::seat_pose(&crew.rig.bridge, CHASSIS)
        .expect("the fixture's car has a seat");
    // **Lower than `CHASSIS`**, so "lowest guid wins the tie" and "whichever list
    // came first wins the tie" give different answers.
    let item = Uuid::from_u128(0x2907_0001);
    assert!(
        item < CHASSIS,
        "the fixture cannot tell the two rules apart"
    );
    {
        let e = crew.rig.world.spawn_with_guid(item, "Ticket", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(seat.x, seat.y, seat.z);
        crew.rig.world.world_mut().entity_mut(e).insert((
            t,
            Interactable {
                verb: InteractVerb::PickUp,
                label: "ticket".into(),
                // The seat's own reach, so the two are admitted on equal terms.
                range_m: inf_physics::d3::vehicle::ENTER_REACH_M,
                enabled: true,
                view_cone_deg: NO_VIEW_TEST_DEG,
                grip: None,
            },
        ));
        crew.rig.world.mark_dirty();
        crew.rig.world.propagate();
    }
    let feet = crew.driver_pos() - DVec3::Y * (crew.driver().stand_half_height_m + HERO_RADIUS);

    // The walk itself is sorted — the property the rule's tie-break rests on.
    let cands = inf_physics::d3::interact::candidates(
        &crew.rig.world,
        &crew.rig.bridge,
        feet,
        &BTreeSet::new(),
    );
    let mut sorted = cands.clone();
    sorted.sort_by_key(|c| c.guid);
    assert_eq!(
        cands.iter().map(|c| c.guid).collect::<Vec<_>>(),
        sorted.iter().map(|c| c.guid).collect::<Vec<_>>(),
        "the merged candidate walk is not in `Guid` order, so the tie-break is \
         a function of which list was extended onto which"
    );

    let hit = inf_physics::d3::interact::resolve(
        &crew.rig.world,
        &crew.rig.bridge,
        feet,
        0.0,
        &BTreeSet::new(),
    )
    .expect("both are in reach");
    let d_seat = (seat - feet).length();
    println!(
        "the seat and the item are both at {d_seat:.6} m; the door answered {} ({:?})",
        hit.label, hit.verb
    );
    // The tie is EXACT rather than approximate — if it were not, this arm would
    // be measuring "nearer wins" again and could not see the sort at all.
    assert_eq!(
        hit.distance_m, d_seat,
        "the item is not at the seat, so there is no tie here to break"
    );
    assert_eq!(
        hit.guid, item,
        "the tie went to the seat, so the merged walk is not `Guid`-ordered"
    );
    assert_eq!(hit.verb, InteractVerb::PickUp);
}

/// Two characters cannot climb into one seat.
#[test]
fn two_characters_cannot_share_one_seat() {
    let mut world = EcsWorld::new();
    ground(&mut world);
    car(&mut world, SPAWN_Y);
    hero(&mut world, HERO, 2.5, 0.0);
    hero(&mut world, HERO_2, -2.5, 0.0);
    world.mark_dirty();
    world.propagate();
    let bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    let mut crew = Crew {
        rig: Rig { world, bridge },
    };
    crew.rig.bridge.sync_from_world(&crew.rig.world);
    crew.step(&Default::default(), 60);
    // `apply_intent` writes onto EVERY player-controlled character, so one press
    // is two attempts — which is exactly the case the occupancy set exists for.
    crew.step(&interact(), 1);
    crew.step(&Default::default(), 60);
    let seated: Vec<Uuid> = [HERO, HERO_2]
        .into_iter()
        .filter(|g| {
            let e = crew.rig.world.entity_of(*g).unwrap();
            crew.rig
                .world
                .world()
                .get::<inf_ecs::components::CharacterMovement>(e)
                .unwrap()
                .runtime
                .seat
                .is_seated()
        })
        .collect();
    assert_eq!(seated.len(), 1, "two characters took one seat: {seated:?}");
    assert_eq!(
        seated[0], HERO,
        "…and the winner is the nearer one, not the archetype order"
    );
}

// ── The authored vehicle class (scene v25, island phase IB-10) ──────────────

/// A class the Ring-0 defaults do not hold: a much stronger engine, a much
/// higher top speed and a stiffer spring.
///
/// `peak_torque_nm` is what makes it stronger since wave VEH2a, and its absence
/// here was the first thing the wave's own re-run caught: `max_engine_force_n`
/// is now the driveline CEILING rather than the engine, so raising it three-fold
/// over an engine that never reached the old ceiling either moved the car
/// **10 cm** further in two seconds. A ceiling is not a curve.
fn island_class() -> inf_ecs::components::VehicleClass {
    inf_ecs::components::VehicleClass {
        peak_torque_nm: 620.0,
        max_engine_force_n: 24_000.0,
        max_speed_mps: 60.0,
        stiffness_n_per_m: 40_000.0,
        ..inf_ecs::components::VehicleClass::default()
    }
}

/// **An authored `VehicleClass` reaches the running vehicle** — the reader the
/// v25 slot lands with.
///
/// P29.7's remainder was that "a committed rig uses the Ring-0 defaults in both
/// hosts, because a tune is an editor-only door by law and a scene field is a
/// schema move". This is the scene field, and this is the arm that says it is
/// not a reserved slot: the numbers on the component are the numbers the sim
/// steps with, applied at creation through `Vehicle::tune`.
#[test]
fn an_authored_vehicle_class_reaches_the_running_vehicle() {
    // The control: no class ⇒ the Ring-0 defaults, which is exactly what every
    // pre-v25 level meant.
    let bare = Rig::new(SPAWN_Y);
    let d = bare.bridge.vehicle_of(CHASSIS).expect("a vehicle");
    let defaults = inf_ecs::vehicle::VehicleTuning::default();
    assert_eq!(
        d.suspension_rest_m(),
        defaults.rest_length_m,
        "a classless rig must be on the Ring-0 defaults"
    );

    // The subject: the same rig with a class on its chassis.
    let mut world = EcsWorld::new();
    ground(&mut world);
    car(&mut world, SPAWN_Y);
    let e = world.entity_of(CHASSIS).unwrap();
    world.world_mut().entity_mut(e).insert(island_class());
    world.mark_dirty();
    world.propagate();
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&world);

    let v = bridge.vehicle_mut(CHASSIS).expect("a vehicle");
    // Read back through the door the class was written through, so this is a
    // statement about the RUNNING vehicle rather than about the component.
    let class = island_class();
    for (name, want) in class.settings() {
        // `set` answers true and leaves the value; setting it to itself is the
        // only read the trait exposes, and a value that had not been installed
        // would have been overwritten here — so the check below is the real one.
        assert!(v.tune(name, want), "the vehicle refuses `{name}`");
    }
    // The suspension rest length is on the trait directly, so it can be read
    // without the tuning door at all — the independent confirmation.
    assert_eq!(
        v.suspension_rest_m(),
        class.rest_length_m,
        "the authored suspension rest length did not reach the vehicle"
    );

    // …and the class really changes the world. Same throttle, same time, but a
    // 24 kN engine against 8 kN moves the car measurably further.
    let mut tuned = Rig { world, bridge };
    let mut plain = Rig::new(SPAWN_Y);
    let controls = VehicleControls {
        throttle: 1.0,
        ..VehicleControls::default()
    };
    tuned.drive(controls, 120);
    plain.drive(controls, 120);
    assert!(
        tuned.z().abs() > plain.z().abs() * 1.2,
        "the authored class drove {:.3} m against the default's {:.3} m — a \
         three-times engine that moves the car the same distance is a class \
         that never reached the sim",
        tuned.z(),
        plain.z()
    );
    eprintln!(
        "v25 vehicle class: authored {:.3} m vs default {:.3} m over 120 steps at \
         full throttle",
        tuned.z(),
        plain.z()
    );
}

// ── the boat, against a real world (wave VEH2c) ─────────────────────────────

/// The sea the boat arms run on: flat, so a measured heading is the boat's and
/// not the swell's.
fn sea(world: &mut EcsWorld) {
    use inf_ecs::components::WaterBody;
    let e = world.spawn_with_guid(LAKE, "Sea", None);
    world.world_mut().entity_mut(e).insert((
        WaterBody {
            wave_amplitude_m: 0.0,
            ..WaterBody::lake(0.0, inf_ecs::math::Vec2d::splat(4_000.0))
        },
        Transform::IDENTITY,
    ));
}

/// A boat: the wheel-less hull, buoyant, on the sea, with a catalogue-shaped
/// tuning rather than a car's defaults.
struct Boat {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Boat {
    fn new() -> Self {
        use inf_ecs::components::Buoyancy;
        let mut world = EcsWorld::new();
        sea(&mut world);
        // Floating with the origin near the waterline: a 150 kg/m³ hull in
        // 1 000 kg/m³ water sits 15 % down, which is a launch with freeboard.
        hull(&mut world, HULL_FLOAT_Y);
        let chassis = world.entity_of(HULL).expect("the hull exists");
        world.world_mut().entity_mut(chassis).insert(Buoyancy {
            density_kg_m3: HULL_DENSITY,
            // A hull is SHAPED to go through water: P20.2's isotropic linear
            // drag defaults to a blunt body and it is what sets a boat top
            // speed, not the class. See the boat-feel note in the wave ledger.
            linear_drag: 0.25,
            ..Default::default()
        });
        world.mark_dirty();
        world.propagate();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        assert!(bridge.is_buoyant(HULL), "the boat must float");
        let mut boat = Self { world, bridge };
        boat.tune();
        boat
    }

    /// The launch's numbers — the row `samples`' catalogue carries, applied
    /// through the same `tune` door an authored `VehicleClass` uses.
    fn tune(&mut self) {
        let v = self
            .bridge
            .vehicle_mut(HULL)
            .expect("the boat is a vehicle");
        for (name, value) in [
            ("max_engine_force_n", 9_000.0),
            ("max_speed_mps", 16.0),
            ("drag_n_per_mps2", 35.0),
            ("drag_lateral_n_per_mps2", 6_000.0),
            ("max_steer_deg", 35.0),
            ("min_steer_deg", 20.0),
            ("steer_rate_deg_per_s", 90.0),
            ("steer_return_deg_per_s", 120.0),
        ] {
            assert!(v.tune(name, value), "the hull refused `{name}`");
        }
    }

    /// The runtime's own order, water pass included — the shape
    /// `a_buoyant_vehicle_keeps_the_force_the_water_pass_owns` established.
    fn drive(&mut self, controls: VehicleControls, n: u32) {
        for _ in 0..n {
            self.bridge.sync_from_world(&self.world);
            self.bridge.apply_water_forces(DT);
            if let Some(v) = self.bridge.vehicle_mut(HULL) {
                v.control(controls);
            }
            inf_physics::d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
            inf_physics::d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
            self.bridge.step(DT);
            self.bridge.write_back_into(&mut self.world);
            self.world.propagate();
        }
    }

    fn at(&self) -> DVec3 {
        let e = self.world.entity_of(HULL).expect("the boat exists");
        self.world
            .world()
            .get::<Transform>(e)
            .expect("…with a transform")
            .translation
            .to_dvec3()
    }

    fn speed(&self) -> f64 {
        self.bridge
            .body_of(HULL)
            .and_then(|b| self.bridge.world().body_linvel(b))
            .map(|v| v.length())
            .unwrap_or(0.0)
    }

    /// The hull's heading, degrees — the engine's own euler yaw, read off the
    /// WORLD, which is the number `Heli::yaw_deg` reads for an aircraft.
    fn yaw_deg(&self) -> f64 {
        let e = self.world.entity_of(HULL).expect("the boat exists");
        self.world.world().get::<Transform>(e).unwrap().rotation.y
    }
}

const AHEAD: VehicleControls = VehicleControls {
    throttle: 1.0,
    steer: 0.0,
    brake: 0.0,
    handbrake: false,
    vertical: 0.0,
    occupied: true,
};

/// **THE BOAT DRIVES** — and the table it prints is this wave's boat-feel row.
///
/// Three claims: it accelerates, it reaches a top speed and settles there
/// (drag balances thrust, which is what makes a top speed a property rather
/// than a clamp), and it stops when the throttle is closed instead of coasting
/// for ever.
#[test]
fn a_boat_makes_way_reaches_a_top_speed_and_carries_its_way() {
    let mut b = Boat::new();
    // Let the buoyancy settle before anything is measured, or the first
    // seconds are a splash.
    b.drive(VehicleControls::default(), 180);
    let start = b.at();
    assert!(
        b.speed() < 0.2,
        "the boat had not settled: {} m/s",
        b.speed()
    );

    let mut samples = Vec::new();
    for s in 1..=16u32 {
        b.drive(AHEAD, 60);
        samples.push((s, b.speed(), (b.at() - start).length()));
    }
    let (_, top, run) = *samples.last().expect("eight seconds");
    println!("BOAT, full ahead from rest (16 s):");
    for (s, v, d) in &samples {
        println!("  t={s}s  {v:.2} m/s ({:.1} kn)  {d:.1} m", v * 1.94384);
    }

    // (a) it accelerates.
    assert!(
        top > 8.0,
        "the boat reached only {top:.2} m/s in sixteen seconds"
    );
    // (b) it SETTLES: the last second adds little, because the hull's drag has
    //     caught the screw's thrust. A boat still accelerating at eight seconds
    //     has no top speed, it has a clamp somewhere else.
    let prev = samples[samples.len() - 2].1;
    assert!(
        (top - prev).abs() < 0.10,
        "the boat was at {prev:.2} m/s and {top:.2} m/s — it has not settled"
    );
    // …and the settled speed is the balance point, below the tuning's own
    // `max_speed_mps` because the falloff reaches zero there.
    assert!(top < 16.0, "the boat passed its own top speed: {top:.2}");
    assert!(
        run > 100.0,
        "sixteen seconds of full ahead covered {run:.1} m"
    );

    // (c) it carries its way and stops — a boat has no brakes and the hull is
    //     what slows it.
    let coast_from = b.at();
    b.drive(VehicleControls::default(), 600);
    let carried = (b.at() - coast_from).length();
    println!(
        "  ten seconds with the throttle closed: {carried:.1} m carried, {:.2} m/s left",
        b.speed()
    );
    assert!(
        carried > 5.0,
        "the boat stopped dead in the water: {carried:.1} m"
    );
    assert!(
        b.speed() < top * 0.35,
        "the boat did not slow at all: {:.2} m/s of {top:.2}",
        b.speed()
    );
}

/// **THE BOAT TURNS**, to the side the helm is put over, in a measurable
/// circle — the second half of this wave's boat-feel row.
#[test]
fn a_boat_turns_toward_its_helm_and_the_circle_is_measurable() {
    /// One steady turn, sampled until the boat has been **all the way round**.
    ///
    /// **The sweep is the whole correction the VEH2c audit made here.** The
    /// first cut sampled a fixed 240 steps and took the mean semi-axis of the
    /// axis-aligned box the track swept. That identity — box semi-axis equals
    /// radius — is only true for a **complete** circle, and 240 steps is
    /// **49.9 degrees** of one: the box was the arc's chord and sagitta, and the
    /// 10.6 m it reported was not a radius at all. Sampling to 360 degrees makes
    /// the same arithmetic exact, and the answer is **37.25 m**.
    ///
    /// Returns the radius, the same radius derived independently as `v / omega`,
    /// and the boat's displacement in X so the helm's own sense is checked.
    let radius = |steer: f64| -> (f64, f64, f64) {
        let mut b = Boat::new();
        b.drive(VehicleControls::default(), 180);
        b.drive(AHEAD, 300);
        let entry = b.at();
        let turning = VehicleControls { steer, ..AHEAD };
        // Long enough to be well into a steady turn before the first sample.
        b.drive(turning, 240);
        let mut minx = f64::MAX;
        let mut maxx = f64::MIN;
        let mut minz = f64::MAX;
        let mut maxz = f64::MIN;
        let mut swept = 0.0f64;
        let mut prev_yaw = b.yaw_deg();
        let mut steps = 0u32;
        // A whole revolution, and a cap so a boat that stopped turning fails the
        // assertion below rather than running for ever.
        while swept < 360.0 && steps < 20_000 {
            steps += 1;
            b.drive(turning, 1);
            let y = b.yaw_deg();
            swept += inf_ecs::movement::angle_delta_deg(y, prev_yaw).abs();
            prev_yaw = y;
            let p = b.at();
            minx = minx.min(p.x);
            maxx = maxx.max(p.x);
            minz = minz.min(p.z);
            maxz = maxz.max(p.z);
        }
        // The mean semi-axis of the circle the boat swept — measured off the
        // world rather than inferred from a yaw rate — and then the SAME radius
        // inferred from the yaw rate, which is the cross-check that says the
        // box is a circle's box.
        let r = ((maxx - minx) + (maxz - minz)) / 4.0;
        let secs = f64::from(steps) / HZ;
        let omega = swept.to_radians() / secs;
        let r_kinematic = b.speed() / omega;
        println!(
            "  helm {steer:+.0}: r = {r:.2} m off the track, {r_kinematic:.2} m from \
             {:.2} m/s over {:.1} deg/s ({swept:.1} deg in {steps} steps)",
            b.speed(),
            omega.to_degrees()
        );
        (r, r_kinematic, b.at().x - entry.x)
    };
    println!("BOAT turning circle (the launch fixture, a 4 m hull):");
    let (r_stbd, k_stbd, dx_stbd) = radius(1.0);
    let (r_port, k_port, dx_port) = radius(-1.0);
    println!(
        "BOAT turning circle: starboard r = {r_stbd:.2} m, port r = {r_port:.2} m \
         ({:.1} hull lengths)",
        r_stbd / (2.0 * HALF.x)
    );

    // The sweep really closed, or the box is an arc's box and the number above
    // is not a radius — which is exactly the defect this arm was rewritten for.
    assert!(
        r_stbd > 2.0 && r_stbd < 200.0,
        "the turning circle is {r_stbd:.1} m — that is not a boat"
    );
    // Starboard helm takes the boat to starboard (+X), port to port.
    assert!(dx_stbd > 5.0, "starboard helm went {dx_stbd:.1} m in X");
    assert!(dx_port < -5.0, "port helm went {dx_port:.1} m in X");
    // TWO INDEPENDENT MEASUREMENTS OF ONE RADIUS. The swept box and `v / omega`
    // share no arithmetic, so agreeing to a percent is what says the track is a
    // circle rather than a shape whose box happens to be square.
    for (what, r, k) in [("starboard", r_stbd, k_stbd), ("port", r_port, k_port)] {
        assert!(
            (r - k).abs() < r * 0.02,
            "the {what} track's box says {r:.2} m and its yaw rate says {k:.2} m — \
             the boat is not going round in a circle"
        );
    }
    // …and the two hands mirror each other. Measured at 37.2501 both ways at the
    // audit head, so a per-cent bound is a real claim and the first cut's 25 %
    // (2.65 m of slack on a 10.6 m figure) was not.
    assert!(
        (r_stbd - r_port).abs() < r_stbd * 0.01,
        "the boat turns better one way than the other: {r_stbd:.2} against {r_port:.2}"
    );
}
/// **A boat on dry land is inert** — the falsifier for the whole immersion
/// rule, asserted on the WORLD rather than on a force list.
///
/// The identical hull, the identical throttle, and no water: it must not move.
/// Without the bite term this hull drives across a car park.
#[test]
fn a_boat_out_of_the_water_goes_nowhere() {
    let mut world = EcsWorld::new();
    ground(&mut world);
    hull(&mut world, SPAWN_Y);
    world.mark_dirty();
    world.propagate();
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&world);
    assert_eq!(bridge.vehicle_count(), 1);
    // HORIZONTALLY, not down one axis: a hull that drove off sideways would
    // satisfy a `z`-only reading, which is what the first cut took.
    let start = {
        let e = world.entity_of(HULL).unwrap();
        let t = world.world().get::<Transform>(e).unwrap().translation;
        glam::DVec2::new(t.x, t.z)
    };
    for _ in 0..600 {
        bridge.sync_from_world(&world);
        if let Some(v) = bridge.vehicle_mut(HULL) {
            v.control(AHEAD);
        }
        inf_physics::d3::step_character_movement(&mut world, &mut bridge, DT);
        inf_physics::d3::step_vehicles(&mut world, &mut bridge, DT);
        bridge.step(DT);
        bridge.write_back_into(&mut world);
        world.propagate();
    }
    let end = {
        let e = world.entity_of(HULL).unwrap();
        let t = world.world().get::<Transform>(e).unwrap().translation;
        glam::DVec2::new(t.x, t.z)
    };
    // PRINTED, because the ledger quotes this number and nothing printed it:
    // the row said "0.00 m" and the only bound was half a metre.
    println!(
        "BOAT on dry land, full throttle for ten seconds: {:.4} m",
        (end - start).length()
    );
    assert!(
        (end - start).length() < 0.01,
        "a boat on a car park drove {:.4} m at full throttle",
        (end - start).length()
    );
}

/// **The rudder is DRAWN**: the part's own transform carries the helm angle,
/// and its translation is never written (wave VEH2c).
///
/// The second half matters more than the first: a part whose translation the
/// door wrote would be re-read as a moved mount by the next reconcile, which is
/// the feedback loop `RaycastVehicle`'s "the authored mount is taken once" note
/// exists to prevent.
#[test]
fn the_rudder_is_drawn_and_its_mount_never_moves() {
    let mut b = Boat::new();
    let mount = {
        let e = b.world.entity_of(THRUSTER).expect("the screw exists");
        b.world.world().get::<Transform>(e).unwrap().translation
    };
    b.drive(
        VehicleControls {
            steer: 1.0,
            ..AHEAD
        },
        180,
    );
    let e = b.world.entity_of(THRUSTER).expect("the screw survives");
    let t = *b.world.world().get::<Transform>(e).unwrap();
    assert!(
        t.rotation.y > 1.0,
        "the drawn rudder never turned: {:?}",
        t.rotation
    );
    assert_eq!(t.rotation.x, 0.0);
    assert_eq!(t.rotation.z, 0.0);
    assert_eq!(t.translation, mount, "the door moved a part's mount");
    // …and the rig still reports the authored mount after a reconcile, which is
    // the property the translation write would have broken.
    b.bridge.sync_from_world(&b.world);
    let rig = b.bridge.vehicle_of(HULL).unwrap().rig();
    assert_eq!(rig.parts[0].mount_local, mount);
}

// ── the helicopter, against a real world (wave VEH2c) ───────────────────────

/// A rotorcraft: the same dynamic box at an airframe's density, with a
/// **capsule sensor** above the cabin where the disc goes.
fn airframe(world: &mut EcsWorld, y: f64) {
    let e = world.spawn_with_guid(HELI, "Helicopter", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, y, 0.0);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            angular_damping: 0.5,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: HALF,
            density: HELI_DENSITY,
            friction: 0.6,
            ..Default::default()
        },
    ));
    let r = world.spawn_with_guid(ROTOR, "Rotor", Some(e));
    let mut rt = Transform::IDENTITY;
    rt.translation = Vec3d::new(0.0, 1.3, 0.0);
    world.world_mut().entity_mut(r).insert((
        rt,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            radius: 5.0,
            half_extents: Vec3d::new(5.0, 0.05, 5.0),
            sensor: true,
            ..Default::default()
        },
    ));
}

struct Heli {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Heli {
    fn new() -> Self {
        let mut world = EcsWorld::new();
        ground(&mut world);
        airframe(&mut world, HALF.y);
        world.mark_dirty();
        world.propagate();
        let bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        let mut h = Self { world, bridge };
        h.bridge.sync_from_world(&h.world);
        assert_eq!(h.bridge.vehicle_count(), 1, "the airframe is not a vehicle");
        let v = h.bridge.vehicle_mut(HELI).expect("the rotorcraft");
        for (name, value) in [
            ("max_engine_force_n", 26_000.0),
            ("max_speed_mps", 70.0),
            ("max_steer_deg", 26.0),
            ("min_steer_deg", 14.0),
            ("steer_rate_deg_per_s", 60.0),
            ("steer_return_deg_per_s", 90.0),
            ("drag_n_per_mps2", 3.0),
            ("drag_lateral_n_per_mps2", 220.0),
        ] {
            assert!(v.tune(name, value), "the rotorcraft refused `{name}`");
        }
        h
    }

    fn fly(&mut self, controls: VehicleControls, n: u32) {
        for _ in 0..n {
            self.bridge.sync_from_world(&self.world);
            if let Some(v) = self.bridge.vehicle_mut(HELI) {
                v.control(controls);
            }
            inf_physics::d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
            inf_physics::d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
            self.bridge.step(DT);
            self.bridge.write_back_into(&mut self.world);
            self.world.propagate();
        }
    }

    fn at(&self) -> DVec3 {
        let e = self.world.entity_of(HELI).expect("the aircraft exists");
        self.world
            .world()
            .get::<Transform>(e)
            .expect("…with a transform")
            .translation
            .to_dvec3()
    }

    fn vel(&self) -> DVec3 {
        self.bridge
            .body_of(HELI)
            .and_then(|b| self.bridge.world().body_linvel(b))
            .unwrap_or(DVec3::ZERO)
    }

    /// The airframe's pitch, degrees, nose-up positive — read off the WORLD, so
    /// the attitude arms measure the aircraft and not the command.
    fn pitch_deg(&self) -> f64 {
        let e = self.world.entity_of(HELI).expect("the aircraft exists");
        let t = self.world.world().get::<Transform>(e).unwrap();
        let fwd = t.quat() * DVec3::Z;
        fwd.y.clamp(-1.0, 1.0).asin().to_degrees()
    }

    /// The airframe's BANK, degrees, right-wing-down positive — read off the
    /// WORLD for the same reason `pitch_deg` is.
    fn roll_deg(&self) -> f64 {
        let e = self.world.entity_of(HELI).expect("the aircraft exists");
        let t = self.world.world().get::<Transform>(e).unwrap();
        let right = t.quat() * DVec3::X;
        (-right.y).clamp(-1.0, 1.0).asin().to_degrees()
    }

    /// The airframe's heading, degrees — the engine's OWN euler yaw, which is
    /// the number `steering_turns_it_and_the_sign_is_the_one_the_control_says`
    /// measures a car's turn with. One convention, read one way.
    fn yaw_deg(&self) -> f64 {
        let e = self.world.entity_of(HELI).expect("the aircraft exists");
        self.world.world().get::<Transform>(e).unwrap().rotation.y
    }
}

const HOVER: VehicleControls = VehicleControls {
    throttle: 0.0,
    steer: 0.0,
    brake: 0.0,
    handbrake: false,
    vertical: 0.0,
    occupied: true,
};

/// **THE HELICOPTER FLIES** — lifts off, holds a height, and comes back down.
///
/// The climb-rate row of this wave's air-feel table, and the claim that makes
/// it a helicopter rather than a flycam: gravity is ON. The hover is an
/// equilibrium the rotor is paying for every step, not a mode with the physics
/// switched off — which is exactly what `MovementMode::Flying`'s 6-DOF flycam
/// is and is why it was only ever the FEEL reference.
#[test]
fn a_helicopter_lifts_off_holds_a_hover_and_comes_back_down() {
    let mut h = Heli::new();
    // On the ground, rotor turning at the hover setting: it must not sink into
    // the floor and must not drift up.
    h.fly(HOVER, 60);
    let ground_y = h.at().y;

    let climb = VehicleControls {
        vertical: 1.0,
        ..HOVER
    };
    let mut rows = Vec::new();
    for s in 1..=6u32 {
        h.fly(climb, 60);
        rows.push((s, h.at().y - ground_y, h.vel().y));
    }
    println!("HELICOPTER, full collective from the ground:");
    for (s, alt, rate) in &rows {
        println!("  t={s}s  {alt:.1} m up  {rate:.2} m/s");
    }
    let (_, alt, rate) = *rows.last().expect("six seconds");
    assert!(
        alt > 20.0,
        "six seconds of full collective climbed {alt:.1} m"
    );
    assert!(rate > 3.0, "the climb settled at {rate:.2} m/s");

    // Neutral collective is a HOVER — once the climb has bled off. The
    // governor holds the WEIGHT, not the height, so an aircraft that was
    // climbing at ten metres a second keeps going up until the fuselage's own
    // drag has taken it, which is what a real one does and is measured here
    // rather than assumed away.
    h.fly(HOVER, 900);
    let held = h.at().y;
    h.fly(HOVER, 180);
    let first = h.at().y - held;
    let held = h.at().y;
    h.fly(HOVER, 180);
    let second = h.at().y - held;
    println!("  neutral collective, fifteen seconds after the climb: {first:+.2} m then {second:+.2} m per three seconds");
    // The coast DECAYS: the fuselage's drag is quadratic, so it approaches zero
    // without ever arriving, and asserting a hard stop would be asserting a
    // model this one deliberately does not have.
    assert!(
        second.abs() < first.abs(),
        "the coast is not decaying: {first:.2} m then {second:.2} m"
    );
    assert!(
        first.abs() < 1.5 && second.abs() < 1.0,
        "a settled hover is still moving: {first:.2} m then {second:.2} m"
    );

    // Full down descends, and the GROUND stops it rather than the model —
    // fifteen seconds, which is comfortably past the time the descent needs, so
    // what arrests it is a contact and not the end of the loop.
    let descend = VehicleControls {
        vertical: -1.0,
        ..HOVER
    };
    h.fly(descend, 900);
    // …and the collective back to neutral, which is what a pilot does on the
    // skids. Holding it down against the ground is a machine being pressed into
    // the floor, not a machine that has landed.
    h.fly(HOVER, 120);
    let landed = h.at().y;
    println!(
        "  fifteen seconds of down collective, then neutral: {:.2} m above where it started",
        landed - ground_y
    );
    println!("  attitude on the skids: pitch {:+.1} deg", h.pitch_deg());
    assert!(
        (landed - ground_y).abs() < 0.5,
        "the aircraft came to rest at {landed:.2} m against a ground at {ground_y:.2}"
    );
    // …and it is RESTING, not still falling through the floor.
    assert!(
        h.vel().y.abs() < 0.5,
        "it is still moving at {:.2} m/s",
        h.vel().y
    );
}

/// **The stick tilts the machine and the machine goes where it is pointed** —
/// the top-speed and tilt-authority rows of the air-feel table.
///
/// The translation is not commanded anywhere in this model: it emerges from a
/// thrust vector that is no longer vertical, which is how a helicopter works.
/// A tilt that produced no motion, or motion with no tilt, would both be
/// caught here.
#[test]
fn a_helicopter_pitches_over_and_the_speed_that_follows_has_a_ceiling() {
    let mut h = Heli::new();
    h.fly(
        VehicleControls {
            vertical: 1.0,
            ..HOVER
        },
        240,
    );
    h.fly(HOVER, 120);
    let base = h.at();

    let ahead = VehicleControls {
        throttle: 1.0,
        ..HOVER
    };
    let mut rows = Vec::new();
    for s in 1..=30u32 {
        h.fly(ahead, 60);
        let v = h.vel();
        rows.push((s, DVec3::new(v.x, 0.0, v.z).length(), h.pitch_deg()));
    }
    println!("HELICOPTER, full forward cyclic from the hover:");
    for (s, speed, pitch) in &rows {
        println!("  t={s}s  {speed:.1} m/s  pitch {pitch:+.1} deg");
    }
    let (_, top, pitch) = *rows.last().expect("thirty seconds");
    // (a) it is nose DOWN, and inside the authority its own tuning allows.
    assert!(
        pitch < -5.0 && pitch > -30.0,
        "the aircraft is at {pitch:.1} degrees"
    );
    // (b) it MOVED, forward, a long way.
    let run = h.at() - base;
    assert!(
        run.z > 700.0,
        "thirty seconds of forward flight covered {:.1} m",
        run.z
    );
    assert!(
        run.x.abs() < run.z.abs() * 0.25,
        "it flew sideways: {run:?}"
    );
    // (c) the speed has a CEILING, because the fuselage's drag catches the
    //     tilted thrust. A machine still accelerating at twelve seconds has no
    //     top speed.
    let prev = rows[rows.len() - 2].1;
    assert!(
        (top - prev).abs() < 0.15,
        "the aircraft was at {prev:.1} m/s and {top:.1} m/s — no ceiling"
    );
    assert!(top > 35.0, "it only reached {top:.1} m/s");

    // (d) and it comes back: the stick to neutral levels the machine.
    h.fly(HOVER, 300);
    println!(
        "  five seconds of neutral stick: pitch {:+.1} deg",
        h.pitch_deg()
    );
    assert!(
        h.pitch_deg().abs() < 4.0,
        "the machine stayed at {:.1} degrees with the stick centred",
        h.pitch_deg()
    );
}

/// **The pedals point the nose**, and they do it on the spot — the yaw row of
/// the air-feel table.
#[test]
fn the_pedals_turn_the_nose_and_the_bank_only_arrives_with_speed() {
    let mut h = Heli::new();
    h.fly(
        VehicleControls {
            vertical: 1.0,
            ..HOVER
        },
        240,
    );
    h.fly(HOVER, 120);
    let before = h.yaw_deg();
    let start = h.at();

    let right = VehicleControls {
        steer: 1.0,
        ..HOVER
    };
    h.fly(right, 180);
    let turned = inf_ecs::movement::angle_delta_deg(h.yaw_deg(), before);
    let moved = (h.at() - start).length();
    println!(
        "HELICOPTER, three seconds of right pedal in the hover: {turned:+.0} deg, moved {moved:.1} m"
    );
    assert!(
        turned > 60.0,
        "three seconds of right pedal turned {turned:.0} degrees"
    );
    // A pedal turn is a turn ON THE SPOT: the aircraft does not go anywhere.
    assert!(moved < 12.0, "the pedal turn wandered {moved:.1} m");

    // Left pedal is the other way.
    let before = h.yaw_deg();
    h.fly(
        VehicleControls {
            steer: -1.0,
            ..HOVER
        },
        180,
    );
    let back = inf_ecs::movement::angle_delta_deg(h.yaw_deg(), before);
    assert!(back < -60.0, "the left pedal turned {back:.0} degrees");
}

/// **The rotor is DRAWN turning**, and its mount never moves — the part-pose
/// door, asserted through the world exactly as the rudder's is.
#[test]
fn the_rotor_is_drawn_turning_and_its_mount_never_moves() {
    let mut h = Heli::new();
    let mount = {
        let e = h.world.entity_of(ROTOR).expect("the rotor exists");
        h.world.world().get::<Transform>(e).unwrap().translation
    };
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..120 {
        h.fly(HOVER, 1);
        let e = h.world.entity_of(ROTOR).unwrap();
        let t = *h.world.world().get::<Transform>(e).unwrap();
        assert_eq!(t.translation, mount, "the door moved the rotor's mount");
        assert_eq!(t.rotation.x, 0.0);
        assert_eq!(t.rotation.z, 0.0);
        seen.insert(t.rotation.y.to_bits());
    }
    assert!(
        seen.len() > 50,
        "the blade barely turned: {} angles",
        seen.len()
    );
    h.bridge.sync_from_world(&h.world);
    let rig = h.bridge.vehicle_of(HELI).unwrap().rig();
    assert_eq!(rig.parts[0].mount_local, mount);
    assert_eq!(rig.parts[0].kind, inf_ecs::vehicle::PartKind::Rotor);
}

/// **A TURN AT SPEED IS A SKID, because the bank saturates on a CAR'S STEERING
/// RACK** (written by wave VEH2c's audit).
///
/// `RotorVehicle`'s own doc refuses uncoordinated flight in as many words — the
/// bank is *derived* from the yaw rate and the speed, "so this machine cannot
/// slip or skid". **The derivation is right and the refusal is not**, because
/// the derived bank is then clamped by `steer_limit_deg`, which is the PITCH
/// stick's speed-tapered authority — a road car's steering rack:
///
/// ```text
/// bank_cmd = (turn * forward_mps * HELI_BANK_PER_TURN_DEG).clamp(-limit, limit)
/// ```
///
/// The coordinated bank GROWS with speed and with the pedal; `limit` SHRINKS
/// with speed and knows nothing about the pedal. So they cross, and on this
/// fixture's numbers — 26 deg of authority at rest tapering to 14 deg at
/// `max_speed_mps` 70 — they cross at a **twentieth of the pedal's travel**.
/// Measured, from the cruise, one second of a held pedal after the attitude has
/// settled:
///
/// | pedal | speed | yaw rate | bank held | bank a coordinated turn needs | the rack |
/// |---|---|---|---|---|---|
/// | 0.05 | 37.6 m/s | 4.9 deg/s | **18.4 deg** | 18.1 deg | 19.6 deg |
/// | 0.10 | 35.7 m/s | 9.9 deg/s | **20.9 deg** | 32.1 deg | 19.9 deg |
/// | 0.25 | 27.5 m/s | 25.2 deg/s | **24.2 deg** | 51.0 deg | 21.3 deg |
/// | 1.00 | 4.4 m/s | 94.8 deg/s | **29.2 deg** | 36.5 deg | 25.2 deg |
///
/// The first row is the model working: the machine holds the coordinated bank
/// to a third of a degree, which is the relation doing exactly what its own doc
/// says. Every row below it is a skid, and at a quarter pedal the machine is
/// holding 24 degrees of a turn that wants 51.
///
/// **This arm measures it rather than fixing it**, so that the class's refusal
/// and the ledger's *"the rotor's own ceiling arrives FIRST"* stop being prose.
/// Both of the collective's documented edges — the rotor's ceiling at about
/// 56 deg of bank on 26 kN, and the governor's own at `HELI_MIN_LIFT_COS` —
/// are **unreachable from the stick**, because the rack binds first at 26 deg
/// and below; the two arms that measure those edges get there by rotating the
/// chassis directly rather than by flying there. Giving the bank a limit of its
/// own, derived from the rotor's ceiling rather than borrowed from the pitch
/// stick, is a flight-model change and is carried with that size.
#[test]
fn a_turn_at_speed_saturates_the_bank_on_the_pitch_sticks_limit() {
    /// One turn, flown from cold: climb, accelerate to the ceiling, then hold
    /// `steer` until the attitude settles and measure over one second.
    ///
    /// A fresh airframe per pedal position, because a machine that has already
    /// been turning is not at its cruise speed and what would be measured is
    /// the recovery.
    fn turn_at(steer: f64) -> (f64, f64, f64, f64) {
        let mut h = Heli::new();
        h.fly(
            VehicleControls {
                vertical: 1.0,
                ..HOVER
            },
            420,
        );
        let ahead = VehicleControls {
            throttle: 1.0,
            ..HOVER
        };
        h.fly(ahead, 1800);
        let turning = VehicleControls {
            throttle: 1.0,
            steer,
            ..HOVER
        };
        h.fly(turning, 180);
        let yaw0 = h.yaw_deg();
        h.fly(turning, 60);
        let rate = inf_ecs::movement::angle_delta_deg(h.yaw_deg(), yaw0).abs();
        let bank = h.roll_deg().abs();
        let v = h.vel();
        let speed = DVec3::new(v.x, 0.0, v.z).length();
        // What a COORDINATED turn at this rate and this speed would need, from
        // the relation `HELI_BANK_PER_TURN_DEG` is the linearisation of.
        let coordinated = ((rate.to_radians() * speed) / 9.81).atan().to_degrees();
        (speed, rate, bank, coordinated)
    }

    /// The bank the PITCH stick's rack allows at `speed`, through the same door
    /// the class clamps with rather than a second spelling of the taper. The
    /// three numbers are `Heli::new`'s own.
    fn rack_at(speed: f64) -> f64 {
        let t = inf_ecs::vehicle::VehicleTuning {
            max_steer_deg: 26.0,
            min_steer_deg: 14.0,
            max_speed_mps: 70.0,
            ..inf_ecs::vehicle::VehicleTuning::default()
        };
        inf_ecs::vehicle::steer_limit_deg(&t, speed)
    }

    println!("HELICOPTER, a held pedal from the cruise (26 kN, 1 520 kg fixture):");
    println!("  pedal   speed   yaw rate      bank   coordinated   the rack");
    let mut rows = Vec::new();
    for steer in [0.05, 0.10, 0.25, 1.00] {
        let (speed, rate, bank, coordinated) = turn_at(steer);
        let rack = rack_at(speed);
        println!(
            "  {steer:5.2}  {speed:5.1} m/s  {rate:6.1} d/s  {bank:5.1} deg  {coordinated:8.1} deg \
             {rack:8.1} deg"
        );
        rows.push((steer, speed, rate, bank, coordinated, rack));
    }

    // (a) THE CONTROL, and it is the first thing asserted: below the crossing
    //     the model does exactly what it claims. A twentieth of a pedal at the
    //     cruise holds the coordinated bank, so everything below is a statement
    //     about the CLAMP and not about a relation that never worked.
    let (_, speed, rate, bank, coordinated, rack) = rows[0];
    assert!(
        speed > 20.0,
        "the gentlest turn was flown at {speed:.1} m/s, which is not a cruise"
    );
    assert!(
        (bank - coordinated).abs() < 2.0,
        "at a twentieth of a pedal ({rate:.1} deg/s at {speed:.1} m/s) the machine \
         held {bank:.1} deg where the coordinated turn is {coordinated:.1} — the \
         relation itself is wrong, which is a bigger finding than the clamp"
    );
    assert!(
        bank < rack,
        "even the gentlest turn is already on the rack ({bank:.1} of {rack:.1}), so \
         this arm has no unsaturated control left"
    );

    // (b) AND ABOVE IT, EVERY TURN IS A SKID: the bank is the rack's, and it
    //     falls short of what the turn the machine is actually flying needs.
    for (steer, speed, rate, bank, coordinated, rack) in rows.into_iter().skip(1) {
        assert!(
            bank < rack * 1.3,
            "at pedal {steer:.2} the bank reached {bank:.1} deg against a rack of \
             {rack:.1} — this arm's whole subject is that clamp, and it is not \
             clamping"
        );
        assert!(
            coordinated - bank > 5.0,
            "at pedal {steer:.2}, {rate:.1} deg/s at {speed:.1} m/s wants \
             {coordinated:.1} deg of bank and the machine is holding {bank:.1} — if \
             these have converged the saturation is gone and this arm should go \
             with the carried item it exists for"
        );
        // …and the two edges the class documents stay out of reach of the stick.
        assert!(
            rack < 56.0,
            "the rack allows {rack:.1} deg, which is past the rotor's own ceiling \
             — the ledger's `the rotor's ceiling arrives first` would be true again"
        );
    }
}
