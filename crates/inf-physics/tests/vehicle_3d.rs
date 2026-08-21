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
    // The runtime's own order: sync → water → movement (which is where the
    // vehicle door lives) → solve → write-back.
    let float_y = |world: &mut EcsWorld, bridge: &mut PhysicsBridge3D, n: u32| -> f64 {
        for _ in 0..n {
            bridge.sync_from_world(world);
            bridge.apply_water_forces(DT);
            inf_physics::d3::step_character_movement(world, bridge, DT);
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
    assert!(
        coasted < travelled * 0.5,
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
            120,
        );
        rig.z() - start
    };
    let free = roll(false);
    let held = roll(true);
    assert!(free > 3.0, "the fixture must be rolling: {free} m");
    assert!(
        held < free * 0.7,
        "two seconds on the handbrake covered {held} m against {free} m coasting"
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
            },
        ));
        crew.rig.world.mark_dirty();
        crew.rig.world.propagate();
    }
    let feet = crew.driver_pos() - DVec3::Y * (crew.driver().stand_half_height_m + HERO_RADIUS);

    // The walk itself is sorted — the property the rule's tie-break rests on.
    let cands =
        inf_physics::d3::interact::candidates(&crew.rig.world, &crew.rig.bridge, &BTreeSet::new());
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
fn island_class() -> inf_ecs::components::VehicleClass {
    inf_ecs::components::VehicleClass {
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
