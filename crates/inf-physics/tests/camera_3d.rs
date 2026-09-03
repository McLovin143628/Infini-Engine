//! **The locomotion camera, in a world** (P29.6) — the arms for
//! `inf_physics::d3::camera`.
//!
//! Out here rather than inside the module for the reason its three neighbours
//! (`d3/movement.rs`, `d3/traversal.rs`, `d3/ragdoll_bridge.rs`) are: they are on
//! `portable_character.rs`'s ban list, which scans the WHOLE file, and a fixture
//! that builds a rotation from an angle would fail that gate for the wrong
//! reason. The gate says so itself, in the assertion that refuses a file with a
//! `#[cfg(test)]` region in it.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::camera::{CameraPose, LocomotionCamera};
use inf_ecs::components::{
    BodyKind3D, CharacterMovement, Collider3D, ColliderShape3DKind, MovementMode, RigidBody3D,
    RotationMode, Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::movement::MovementIntent;
use inf_ecs::EcsWorld;
use inf_physics::d3::{step_character_movement, step_locomotion_camera};
use inf_physics::PhysicsBridge3D;

const DT: f64 = 1.0 / 60.0;
const HERO: Uuid = Uuid::from_u128(0x2906_0001);
const GROUND: Uuid = Uuid::from_u128(0x2906_0002);
const WALL: Uuid = Uuid::from_u128(0x2906_0003);
const RADIUS: f64 = 0.3;

struct Sim {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    cam: LocomotionCamera,
}

impl Sim {
    /// A character on a floor, facing `+Z`, with an optional wall standing
    /// **behind** it — which is where a third-person camera goes.
    fn new(wall_at: Option<f64>) -> Self {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(GROUND, "Ground", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, -0.5, 0.0);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(60.0, 0.5, 60.0),
                ..Default::default()
            },
            t,
        ));
        if let Some(z) = wall_at {
            let e = world.spawn_with_guid(WALL, "Wall", None);
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(0.0, 2.0, z);
            world.world_mut().entity_mut(e).insert((
                RigidBody3D {
                    kind: BodyKind3D::Static,
                    ..Default::default()
                },
                Collider3D {
                    shape_kind: ColliderShape3DKind::Box,
                    half_extents: Vec3d::new(10.0, 4.0, 0.25),
                    ..Default::default()
                },
                t,
            ));
        }
        let cm = CharacterMovement {
            player_controlled: true,
            rotation_mode: RotationMode::LookingDirection,
            ..Default::default()
        };
        let e = world.spawn_with_guid(HERO, "Hero", None);
        let mut t = Transform::IDENTITY;
        // Authored EXACTLY on the floor — the placement a level author
        // makes, and the one `settle_on_spawn` exists for.
        t.translation = Vec3d::new(0.0, cm.stand_half_height_m + RADIUS, 0.0);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Kinematic,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Capsule,
                half_extents: Vec3d::new(RADIUS, cm.stand_half_height_m, RADIUS),
                radius: RADIUS,
                ..Default::default()
            },
            inf_ecs::components::CharacterController3D::default(),
            cm,
            t,
        ));
        world.mark_dirty();
        world.propagate();
        Self {
            world,
            bridge: PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0)),
            cam: LocomotionCamera::default(),
        }
    }

    /// Everything a fixed step does **except** the camera.
    ///
    /// Split out for one reason (P29.6 audit, A5): the purity arm's control has
    /// to be this program minus its last line, and the first cut wrote the
    /// control out by hand and dropped the solver and the write-back with it.
    /// It therefore compared `sim + solver + write-back + camera` against `sim`
    /// alone and read the result as proof the camera was inert -- true only
    /// because the two omissions happened to be no-ops for the bytes it
    /// sampled, which is a coincidence a test must not rest on.
    fn step_sim(&mut self, intent: &MovementIntent) {
        self.bridge.sync_from_world(&self.world);
        inf_ecs::movement::apply_intent(&mut self.world, intent);
        step_character_movement(&mut self.world, &mut self.bridge, DT);
        // The solver runs, as it does in both hosts: `step_character_movement`
        // sits BEFORE it in each fixed step, and a fixture that skipped it
        // would be measuring a different program.
        self.bridge.step(DT);
        self.bridge.write_back_into(&mut self.world);
        self.world.propagate();
    }

    fn step(&mut self, intent: &MovementIntent) -> Option<CameraPose> {
        self.step_sim(intent);
        step_locomotion_camera(&self.world, &mut self.bridge, &mut self.cam, HERO, DT)
    }
}

/// The camera sits **behind and above** the character it follows, at the
/// arm length its state asks for.
#[test]
fn the_camera_trails_the_character_it_follows() {
    let mut sim = Sim::new(None);
    let idle = MovementIntent::default();
    let pose = sim.step(&idle).expect("a camera for a real character");
    // Facing +Z at yaw 0, so the camera is at −Z…
    assert!(
        pose.position.z < -1.0,
        "the camera is not behind the character: {:?}",
        pose.position
    );
    // …and above the feet, near the pivot's height.
    assert!(
        pose.position.y > 1.0,
        "the camera is on the floor: {:?}",
        pose.position
    );
    assert_eq!(sim.cam.collision_pull_m, 0.0, "nothing was in the way");
}

/// **The sweep** — `cast_shape`'s third consumer. A wall behind the
/// character pulls the camera in; the same character with no wall does not.
#[test]
fn a_wall_behind_the_character_pulls_the_camera_in() {
    let idle = MovementIntent::default();
    let mut clear = Sim::new(None);
    let mut walled = Sim::new(Some(-1.5));
    for _ in 0..8 {
        clear.step(&idle);
        walled.step(&idle);
    }
    let free = clear.cam.pose.position.z;
    let blocked = walled.cam.pose.position.z;
    assert!(
        free < -2.5,
        "the control camera did not get an arm's length out: {free}"
    );
    assert!(
        blocked > free + 0.5,
        "the wall did not move the camera: {blocked} against {free}"
    );
    assert!(
        blocked > -1.5,
        "the camera ended up INSIDE the wall at z = -1.5: {blocked}"
    );
    assert!(
        walled.cam.collision_pull_m > 0.5,
        "the pull was not recorded: {}",
        walled.cam.collision_pull_m
    );
    assert_eq!(clear.cam.collision_pull_m, 0.0, "the control must not pull");
}

/// **The camera reads the sim and never writes it.**
///
/// Two identical worlds run the identical program, one with the final
/// `step_locomotion_camera` line and one without, and the simulation is
/// byte-identical -- the ViewMode ruling's proof at the unit level
/// (`phase29_gate` repeats it over the whole course).
///
/// # What "the simulation" means here (P29.6 audit, A5)
///
/// The world **and the physics bridge**. The door holds a `&mut
/// PhysicsBridge3D` -- it must, because a sphere sweep needs the query pipeline
/// -- so the bridge is the one place a leak would actually land, and the first
/// cut of this arm sampled nine floats off one entity and nothing else. A
/// mutation that nudged the subject's body by a picometre through that `&mut`
/// survived it. The bridge's own body positions are in the record now, so it
/// does not.
#[test]
fn stepping_a_camera_changes_nothing_about_the_simulation() {
    let script = |i: u32| MovementIntent {
        move_input: inf_ecs::Vec2d::new(0.0, 1.0),
        look_yaw_dps: if i > 30 { 90.0 } else { 0.0 },
        sprint: i > 60,
        ..Default::default()
    };
    let mut with = Sim::new(Some(-1.5));
    let mut without = Sim::new(Some(-1.5));
    for i in 0..120 {
        with.step(&script(i));
        // The control is `step` MINUS its last line. Nothing else differs.
        without.step_sim(&script(i));
    }
    let bytes = |s: &Sim| {
        let e = s.world.entity_of(HERO).unwrap();
        let w = s.world.world();
        let t = w.get::<Transform>(e).unwrap();
        let cm = w.get::<CharacterMovement>(e).unwrap();
        let mut out: Vec<u8> = Vec::new();
        for v in [
            t.translation.x,
            t.translation.y,
            t.translation.z,
            t.rotation.x,
            t.rotation.y,
            t.rotation.z,
            cm.runtime.velocity.x,
            cm.runtime.velocity.y,
            cm.runtime.velocity.z,
            cm.runtime.aim_yaw_deg,
            cm.runtime.aim_pitch_deg,
            cm.runtime.body_yaw_deg,
            cm.runtime.mapped_speed,
        ] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out.push(cm.mode as u8);
        out.push(cm.runtime.actual_gait as u8);
        // **The bridge**, which is what the door can actually reach: every
        // body's translation, in the bridge's own deterministic id order.
        for b in s.bridge.world().body_ids() {
            let p = s.bridge.world().body_translation(b).unwrap_or(DVec3::ZERO);
            for v in [p.x, p.y, p.z] {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        out
    };
    let with_bytes = bytes(&with);
    assert!(
        with_bytes.len() > 15 * 8,
        "the record covers no bodies, so the bridge half is about nothing"
    );
    assert_eq!(
        with_bytes,
        bytes(&without),
        "stepping a camera moved the simulation"
    );
    // ...and the camera really ran, or the equality above is about nothing.
    assert_ne!(with.cam.pose.position, Vec3d::ZERO);
    assert!(with.cam.yaw_deg.abs() > 1.0, "the camera never turned");
}

/// A subject that is not a character is a **value**: the camera holds its
/// pose rather than the call failing.
#[test]
fn a_camera_with_nothing_to_follow_holds_its_pose() {
    let mut sim = Sim::new(None);
    sim.step(&MovementIntent::default());
    let held = sim.cam.pose;
    let ghost = Uuid::from_u128(0xDEAD);
    assert_eq!(
        step_locomotion_camera(&sim.world, &mut sim.bridge, &mut sim.cam, ghost, DT),
        None
    );
    assert_eq!(sim.cam.pose, held, "the camera moved for a ghost");
    // The other half: an entity that exists and is not a character.
    assert_eq!(
        step_locomotion_camera(&sim.world, &mut sim.bridge, &mut sim.cam, GROUND, DT),
        None
    );
    assert_eq!(sim.cam.pose, held);
}

/// **First person is a blend weight**, not a mode: switching moves the
/// camera continuously to the seat and the arm goes to nothing.
#[test]
fn first_person_blends_to_the_seat() {
    let mut sim = Sim::new(None);
    let idle = MovementIntent::default();
    for _ in 0..8 {
        sim.step(&idle);
    }
    let third = sim.cam.pose.position;
    sim.cam.view_mode = inf_ecs::camera::ViewMode::FirstPerson;
    sim.step(&idle);
    let one = sim.cam.pose.position;
    assert!(
        one.to_dvec3().distance(third.to_dvec3()) > 1e-6
            && one.to_dvec3().distance(third.to_dvec3()) < 1.0,
        "one step jumped the whole way: {third:?} -> {one:?}"
    );
    for _ in 0..300 {
        sim.step(&idle);
    }
    let seat = sim.cam.pose.position;
    let pivot = sim.cam.pivot;
    assert!(
        seat.to_dvec3().distance(pivot.to_dvec3()) < 0.5,
        "first person did not reach the seat: {seat:?} against a pivot at {pivot:?}"
    );
    assert!((sim.cam.fp_weight - 1.0).abs() < 1e-6);
}

/// The camera follows the **mode**, not only the position: a crouch pulls it
/// in through the low-stance block.
#[test]
fn the_camera_follows_the_movement_mode() {
    let mut sim = Sim::new(None);
    let idle = MovementIntent::default();
    for _ in 0..60 {
        sim.step(&idle);
    }
    let standing_arm = sim.cam.settings.arm_length_m;
    let crouch = MovementIntent {
        crouch: true,
        ..idle
    };
    sim.step(&crouch);
    for _ in 0..120 {
        sim.step(&idle);
    }
    let e = sim.world.entity_of(HERO).unwrap();
    assert_eq!(
        sim.world.world().get::<CharacterMovement>(e).unwrap().mode,
        MovementMode::Crouch,
        "the fixture never crouched, so this arm is about nothing"
    );
    assert!(
        sim.cam.settings.arm_length_m < standing_arm - 0.3,
        "the camera did not follow the crouch: {} against {standing_arm}",
        sim.cam.settings.arm_length_m
    );
}

// ── the seat (P29.7) ────────────────────────────────────────────────────────

const CAR: Uuid = Uuid::from_u128(0x2906_0010);
const WHEEL: u128 = 0x2906_0020;

/// Put a **long** vehicle where the character can climb into it.
///
/// Six metres long on purpose. The seat is the chassis collider's top face, so a
/// third-person boom leaves the pivot above the roof and comes down behind it —
/// and on a two-metre car it is past the tail before it has dropped far enough
/// to matter. On the showcase course's own car the boom clears the bodywork
/// entirely and the exclusion never fires, which is exactly why this fixture is
/// not that car: a rule with no falsifier is a rule nobody can break.
fn park_a_car(world: &mut EcsWorld, half_z: f64) {
    let e = world.spawn_with_guid(CAR, "Car", None);
    let mut t = Transform::IDENTITY;
    // Beside the character rather than in front of it: a six-metre car whose
    // seat is its own centre cannot be reached from its nose, and the reach is
    // measured to the seat.
    t.translation = Vec3d::new(2.5, 0.75 + 0.35, 0.0);
    world.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            angular_damping: 0.5,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(2.0, 0.5, half_z),
            density: 150.0,
            ..Default::default()
        },
        t,
    ));
    for (i, (x, z)) in [(-0.9, 1.4), (0.9, 1.4), (-0.9, -1.4), (0.9, -1.4)]
        .into_iter()
        .enumerate()
    {
        let w = world.spawn_with_guid(Uuid::from_u128(WHEEL + i as u128), "Wheel", Some(e));
        let mut wt = Transform::IDENTITY;
        wt.translation = Vec3d::new(x, -0.75, z);
        world.world_mut().entity_mut(w).insert((
            wt,
            Collider3D {
                shape_kind: ColliderShape3DKind::Sphere,
                radius: 0.35,
                sensor: true,
                ..Default::default()
            },
        ));
    }
    world.mark_dirty();
    world.propagate();
}

/// **The camera excludes the vehicle its subject is driving** — the P29.6
/// audit's A4, a third time.
///
/// The first two were the character's own collider and the ragdoll's limbs (a
/// ragdoll spawns a dozen and *disables* the one the camera knew about). The
/// third is a vehicle: a seated character's capsule is parked and the thing
/// filling the space around it is a chassis, so without the exclusion the sweep
/// finds bodywork at nearly zero distance and the camera sits at
/// `min_arm_fraction` — inside the driver — for the whole drive.
///
/// The fixture is a **six-metre-long** car and the camera is pitched **up**,
/// which drops the boom into the bodywork behind the seat. Both halves are
/// needed: on a two-metre car the boom is past the tail before it has fallen far
/// enough to touch anything, so an arm built on the showcase's own rig passed
/// with the exclusion deleted — measured, before this fixture was this shape.
#[test]
fn the_camera_excludes_the_vehicle_its_subject_is_driving() {
    let mut sim = Sim::new(None);
    park_a_car(&mut sim.world, 3.0);
    let idle = MovementIntent::default();
    for _ in 0..60 {
        sim.step(&idle);
    }
    // Climb in, and let the warp finish.
    sim.step(&MovementIntent {
        interact: true,
        ..Default::default()
    });
    for _ in 0..90 {
        sim.step(&idle);
    }
    let cm = {
        let e = sim.world.entity_of(HERO).unwrap();
        sim.world
            .world()
            .get::<CharacterMovement>(e)
            .unwrap()
            .clone()
    };
    assert_eq!(
        cm.mode,
        inf_ecs::components::MovementMode::Driving,
        "the fixture must be driving before it can measure a drive camera"
    );
    assert_eq!(cm.runtime.seat.vehicle, CAR);

    // Pitch UP: a third-person camera looking up drops its boom, which is what
    // sends it into the roof of the thing the character is sitting on.
    let look_up = MovementIntent {
        look_pitch_dps: 30.0,
        ..Default::default()
    };
    let mut worst = f64::MAX;
    for _ in 0..150 {
        if let Some(pose) = sim.step(&look_up) {
            let e = sim.world.entity_of(HERO).unwrap();
            let at = sim
                .world
                .world()
                .get::<Transform>(e)
                .unwrap()
                .translation
                .to_dvec3();
            worst = worst.min((pose.position.to_dvec3() - at).length());
        }
    }
    assert!(
        worst > 1.0,
        "the camera came within {worst:.3} m of the driver — it is sweeping into \
         the car it is riding"
    );
}

/// **THE DRIVE CAMERA REALLY ENGAGES IN A DRIVE** (`audit:` VEH2a) — and the
/// arm it sits on is the CHASSIS'S OWN LENGTH.
///
/// VEH2a's drive-camera arms are all on the Ring-0 door: they hand
/// `LocomotionCamera::advance` a `DrivingView` by literal and assert what it
/// does with one. None of them runs the path a player takes, and that path has
/// a piece of code with no arm at all —
/// `inf_physics::d3::camera::step_locomotion_camera` looking the half-length up
/// off the seated character's vehicle:
///
/// ```ignore
/// .map(|c| c.half_extents.z.abs())
/// ```
///
/// `half_extents` is `(half_width, half_height, half_length)`, so `.x` is a
/// plausible typo that produces a plausible number and would have made every
/// car in the fleet sit at the same distance. Two cars of different LENGTHS and
/// the same width is what tells the two apart, and it is what this fixture is.
///
/// It also closes the reachability question the block's own doc raises: the
/// branch is answered off `CharacterMovement::mode`, which `step_driving`
/// writes, and this is the arm that says a hero who pressed the interact key
/// arrives in it.
#[test]
fn a_driven_car_gets_the_drive_camera_and_sits_back_by_its_own_length() {
    let settled = |half_z: f64| -> (inf_ecs::camera::CameraSettings, f64) {
        let mut sim = Sim::new(None);
        park_a_car(&mut sim.world, half_z);
        let idle = MovementIntent::default();
        for _ in 0..60 {
            sim.step(&idle);
        }
        sim.step(&MovementIntent {
            interact: true,
            ..Default::default()
        });
        for _ in 0..120 {
            sim.step(&idle);
        }
        let mode = {
            let e = sim.world.entity_of(HERO).unwrap();
            sim.world.world().get::<CharacterMovement>(e).unwrap().mode
        };
        assert_eq!(
            mode,
            MovementMode::Driving,
            "the fixture must be driving before it can measure a drive camera"
        );
        (sim.cam.settings, sim.cam.tuning.driving.arm_per_length_m)
    };

    // The on-foot camera the same hero would have had — the block a drive used
    // to inherit, latched at the gait it happened to be in beside the door.
    let walking = {
        let mut sim = Sim::new(None);
        park_a_car(&mut sim.world, 1.5);
        for _ in 0..90 {
            sim.step(&MovementIntent::default());
        }
        sim.cam.settings
    };

    let (short, per_m) = settled(1.5);
    let (long, _) = settled(6.0);
    println!(
        "THE DRIVE CAMERA IN A DRIVE: a 3 m car got a {:.3} m arm / {:.1}° FOV, a \
         12 m one {:.3} m / {:.1}°, against the walking camera's {:.3} m / {:.1}° \
         and a {per_m} m-per-half-metre rule",
        short.arm_length_m,
        short.fov_deg,
        long.arm_length_m,
        long.fov_deg,
        walking.arm_length_m,
        walking.fov_deg
    );

    // It is the DRIVE block and not the gait one.
    assert!(
        short.fov_deg > walking.fov_deg && short.arm_length_m > walking.arm_length_m,
        "a driving hero got a {:.1}° / {:.2} m camera against a walking {:.1}° / \
         {:.2} m — the Driving branch was never reached",
        short.fov_deg,
        short.arm_length_m,
        walking.fov_deg,
        walking.arm_length_m
    );
    // The shoulder offset is BLENDED away rather than switched, so this is a
    // tolerance and not an equality — `state_blend_speed` is exponential and
    // never reaches its target exactly.
    assert!(
        short.camera_offset.x.abs() < 1e-3,
        "the drive camera kept a {:.4} m shoulder offset — a car is not looked over",
        short.camera_offset.x
    );
    // …and the arm grew by the vehicle's own half-length, through the collider
    // lookup. Read as a DIFFERENCE so the base block cancels out and only the
    // per-length term is under test.
    let grew = long.arm_length_m - short.arm_length_m;
    let want = per_m * (6.0 - 1.5);
    assert!(
        (grew - want).abs() < 0.05,
        "a 12 m car sat {grew:.3} m further back than a 3 m one where its own \
         half-length asks for {want:.3} — the camera is reading the wrong axis \
         of the chassis collider, or not reading it at all"
    );
}

/// Park a **wheel-less craft** beside the hero: the same dynamic box, with one
/// mount instead of four wheels (wave VEH2c).
fn park_a_craft(world: &mut EcsWorld, half_z: f64, rotor: bool) {
    let e = world.spawn_with_guid(CAR, "Craft", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(2.5, 0.75 + 0.35, 0.0);
    world.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            angular_damping: 0.5,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(2.0, 0.5, half_z),
            density: 150.0,
            ..Default::default()
        },
        t,
    ));
    let m = world.spawn_with_guid(Uuid::from_u128(WHEEL + 9), "Mount", Some(e));
    let mut mt = Transform::IDENTITY;
    mt.translation = Vec3d::new(0.0, if rotor { 1.2 } else { -0.4 }, -half_z * 0.8);
    let collider = if rotor {
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            radius: 4.0,
            half_extents: Vec3d::new(4.0, 0.05, 4.0),
            sensor: true,
            ..Default::default()
        }
    } else {
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(0.25, 0.15, 0.25),
            sensor: true,
            ..Default::default()
        }
    };
    world.world_mut().entity_mut(m).insert((mt, collider));
    world.mark_dirty();
    world.propagate();
}

/// **A BOAT AND A HELICOPTER GET THE DRIVE CAMERA, and it is the right size for
/// each of them** (wave VEH2c) — the wave's camera ruling, measured.
///
/// The ruling is REUSE, and this is what makes it a ruling rather than an
/// omission. `CameraTuning::settings_for` answers `self.driving.base` for
/// `MovementMode::Driving` before it looks at a gait, and a character in a boat
/// or a helicopter is in `MovementMode::Driving` — the seat is the seat. So both
/// craft already get the block VEH2a minted, including the per-half-length arm
/// that makes a 6.4 m launch sit further back than a 3 m saloon.
///
/// What a `Flying` branch would buy, priced: a second `CameraSettings` table, a
/// `FlyingView` on `CameraInput` (a breaking change at ~8 literals), a second
/// early return in `settings_for`, and — the only genuinely new part — a camera
/// pitch that followed the AIRCRAFT'S pitch rather than the player's look. That
/// last one is the reason it is refused: this helicopter's attitude is a
/// commanded thing that moves 25 degrees in a second and returns to level on a
/// centred stick, and a camera that followed it would pitch the horizon around
/// the player every time they touched the stick. The player's own look already
/// has full pitch freedom and it is the freedom that was asked for.
#[test]
fn a_boat_and_a_helicopter_get_the_drive_camera_at_their_own_size() {
    let settled = |half_z: f64, rotor: Option<bool>| -> inf_ecs::camera::CameraSettings {
        let mut sim = Sim::new(None);
        match rotor {
            Some(r) => park_a_craft(&mut sim.world, half_z, r),
            None => park_a_car(&mut sim.world, half_z),
        }
        let idle = MovementIntent::default();
        for _ in 0..60 {
            sim.step(&idle);
        }
        sim.step(&MovementIntent {
            interact: true,
            ..Default::default()
        });
        for _ in 0..120 {
            sim.step(&idle);
        }
        let mode = {
            let e = sim.world.entity_of(HERO).unwrap();
            sim.world.world().get::<CharacterMovement>(e).unwrap().mode
        };
        assert_eq!(
            mode,
            MovementMode::Driving,
            "a seat is a seat: a craft's occupant is Driving"
        );
        sim.cam.settings
    };

    let walking = {
        let mut sim = Sim::new(None);
        park_a_craft(&mut sim.world, 3.2, false);
        for _ in 0..90 {
            sim.step(&MovementIntent::default());
        }
        sim.cam.settings
    };
    let boat = settled(3.2, Some(false));
    let heli = settled(2.4, Some(true));
    let car = settled(1.5, None);
    println!(
        "THE DRIVE CAMERA OVER THREE HULLS: a 6.4 m launch {:.3} m / {:.1}°, a \
         4.8 m helicopter {:.3} m / {:.1}°, a 3 m car {:.3} m / {:.1}°, against \
         a walking {:.3} m / {:.1}°",
        boat.arm_length_m,
        boat.fov_deg,
        heli.arm_length_m,
        heli.fov_deg,
        car.arm_length_m,
        car.fov_deg,
        walking.arm_length_m,
        walking.fov_deg
    );

    // (a) Both craft reach the DRIVE block, not the gait one.
    for (what, s) in [("the launch", boat), ("the helicopter", heli)] {
        assert!(
            s.fov_deg > walking.fov_deg && s.arm_length_m > walking.arm_length_m,
            "{what} got a {:.1}° / {:.2} m camera against a walking {:.1}° / \
             {:.2} m — the Driving branch was never reached",
            s.fov_deg,
            s.arm_length_m,
            walking.fov_deg,
            walking.arm_length_m
        );
    }
    // (b) …and the arm is sized to the CRAFT: the launch sits further back than
    //     the helicopter, which sits further back than the car, by the same
    //     per-half-length rule a car got at VEH2a.
    assert!(
        boat.arm_length_m > heli.arm_length_m && heli.arm_length_m > car.arm_length_m,
        "the arms do not scale: launch {:.3}, helicopter {:.3}, car {:.3}",
        boat.arm_length_m,
        heli.arm_length_m,
        car.arm_length_m
    );
    let per_m = {
        let mut sim = Sim::new(None);
        park_a_craft(&mut sim.world, 3.2, false);
        sim.step(&MovementIntent::default());
        sim.cam.tuning.driving.arm_per_length_m
    };
    assert!(
        ((boat.arm_length_m - heli.arm_length_m) - (3.2 - 2.4) * per_m).abs() < 0.05,
        "the difference is {:.3} m and the rule says {:.3}",
        boat.arm_length_m - heli.arm_length_m,
        (3.2 - 2.4) * per_m
    );
}
