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
