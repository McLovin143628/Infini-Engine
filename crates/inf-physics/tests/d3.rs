//! Behavioural + determinism tests for the 3D physics facade — the `d3` mirror of
//! `tests/d2.rs`.

use glam::{DQuat, DVec3};
use inf_physics::d3::ColliderDesc3D;
use inf_physics::{
    BodyKind3D, CharacterMover3D, ColliderShape3D, ContactPhase, FixedStepper, PhysicsWorld3D,
};

const DT: f64 = 1.0 / 60.0;

/// Serialize every body's pose (translation + rotation quaternion) to a flat
/// little-endian byte buffer, in deterministic handle order, so two worlds can be
/// compared byte-for-byte.
fn pose_bytes(world: &PhysicsWorld3D) -> Vec<u8> {
    let mut bytes = Vec::new();
    for id in world.body_ids() {
        let t = world.body_translation(id).unwrap();
        let r = world.body_rotation(id).unwrap();
        for c in [t.x, t.y, t.z, r.x, r.y, r.z, r.w] {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    bytes
}

/// A world with a static floor at y=0 and a dynamic ball dropped above it.
fn build_drop_world() -> PhysicsWorld3D {
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));

    let floor = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    world.add_collider(
        floor,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(50.0, 0.5, 50.0),
        }),
    );

    let ball = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(0.0, 10.0, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        ball,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.5 })
            .restitution(0.0)
            .friction(0.8),
    );

    world
}

#[test]
fn determinism_two_identical_worlds_match_byte_for_byte() {
    let mut a = build_drop_world();
    let mut b = build_drop_world();

    for _ in 0..300 {
        a.step(DT);
        b.step(DT);
    }

    assert_eq!(
        pose_bytes(&a),
        pose_bytes(&b),
        "two identically-built worlds diverged after 300 steps"
    );
}

#[test]
fn determinism_repeatable_across_runs() {
    let run = || {
        let mut w = build_drop_world();
        for _ in 0..300 {
            w.step(DT);
        }
        pose_bytes(&w)
    };
    assert_eq!(run(), run(), "the same simulation was not repeatable");
}

#[test]
fn ball_drops_and_comes_to_rest_on_floor() {
    let mut world = build_drop_world();
    let ball = *world.body_ids().last().unwrap();

    // Simulate 3 seconds; the ball should fall and settle on the floor.
    for _ in 0..180 {
        world.step(DT);
    }

    let y = world.body_translation(ball).unwrap().y;
    // Floor top is at y = 0.5, ball radius 0.5 → rest centre near y = 1.0.
    assert!(
        (y - 1.0).abs() < 0.05,
        "ball should rest near y=1.0, got y={y}"
    );
    // No restitution → it should be essentially at rest, not bouncing.
    let vy = world.body_linvel(ball).unwrap().y;
    assert!(vy.abs() < 0.05, "ball should be at rest, got vy={vy}");
}

#[test]
fn character_walks_along_floor_blocked_by_wall_and_reports_grounded() {
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));

    // Floor spanning x,z in [-10, 10] at y=0 (top surface y=0.5).
    let floor = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    world.add_collider(
        floor,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(10.0, 0.5, 10.0),
        }),
    );
    // A wall at x=3 (spanning x in [2.8, 3.2]).
    let wall = world.add_body(
        BodyKind3D::Static,
        DVec3::new(3.0, 1.5, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        wall,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(0.2, 2.0, 10.0),
        }),
    );

    // Character: a 0.5-radius ball starting on the floor at x=0, y=1.0.
    let mover = CharacterMover3D::new(ColliderShape3D::Sphere { radius: 0.5 })
        .offset(0.02)
        .snap_to_ground(Some(0.2));

    let mut pos = DVec3::new(0.0, 1.0, 0.0);
    let mut grounded_any = false;
    // Walk right into the wall; gravity pulls down each step so it stays grounded.
    for _ in 0..240 {
        let desired = DVec3::new(0.08, -0.05, 0.0);
        let mv = world.move_character(&mover, pos, desired, None);
        pos += mv.translation;
        grounded_any |= mv.grounded;
    }

    assert!(grounded_any, "character never reported grounded");
    // The wall's near face is at x = 2.8; a 0.5-radius ball (plus offset) is
    // stopped short of it and cannot pass.
    assert!(
        pos.x < 2.4,
        "character should be blocked by the wall, but reached x={}",
        pos.x
    );
    assert!(
        pos.x > 1.5,
        "character should have advanced toward the wall"
    );
    // Stayed on the floor (didn't sink or fly).
    assert!(
        (pos.y - 1.0).abs() < 0.2,
        "character drifted off the floor: y={}",
        pos.y
    );
}

#[test]
fn sensor_fires_started_and_stopped_exactly_once() {
    let mut world = PhysicsWorld3D::new(DVec3::ZERO);

    // A static sensor box at the origin.
    let sensor_body = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    let sensor = world
        .add_collider(
            sensor_body,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::splat(1.0),
            })
            .sensor(true),
        )
        .unwrap();

    // A kinematic body that we drive left-to-right through the sensor.
    let mover_body = world.add_body(
        BodyKind3D::Kinematic,
        DVec3::new(-5.0, 0.0, 0.0),
        DQuat::IDENTITY,
    );
    let mover_col = world
        .add_collider(
            mover_body,
            ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.5 }),
        )
        .unwrap();

    let mut started = 0;
    let mut stopped = 0;
    let mut started_pair_ok = true;

    let mut x = -5.0;
    for _ in 0..200 {
        x += 0.1;
        world.set_body_translation(mover_body, DVec3::new(x, 0.0, 0.0));
        world.step(DT);
        for ev in world.drain_contact_events() {
            assert!(ev.sensor, "expected a sensor overlap event");
            let pair = [ev.collider_a, ev.collider_b];
            started_pair_ok &= pair.contains(&sensor) && pair.contains(&mover_col);
            match ev.phase {
                ContactPhase::Started => started += 1,
                ContactPhase::Stopped => stopped += 1,
            }
        }
    }

    assert!(started_pair_ok, "event referenced unexpected colliders");
    assert_eq!(started, 1, "sensor Started should fire exactly once");
    assert_eq!(stopped, 1, "sensor Stopped should fire exactly once");
}

#[test]
fn raycast_hits_expected_point_and_normal() {
    let mut world = PhysicsWorld3D::new(DVec3::ZERO);

    // A box occupying [-1,1]^3 centred at origin (top face at y=1).
    let body = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    world.add_collider(
        body,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::splat(1.0),
        }),
    );

    // Cast straight down from above the box.
    let hit = world
        .cast_ray(DVec3::new(0.0, 5.0, 0.0), DVec3::new(0.0, -1.0, 0.0), 100.0)
        .expect("ray should hit the box");

    // Impact at the top face: y = 1, toi = 5 - 1 = 4.
    assert!((hit.toi - 4.0).abs() < 1e-9, "toi = {}", hit.toi);
    assert!(
        (hit.point.y - 1.0).abs() < 1e-9,
        "point.y = {}",
        hit.point.y
    );
    assert!(hit.point.x.abs() < 1e-9, "point.x = {}", hit.point.x);
    // Normal points up (out of the top face), toward the ray origin.
    assert!(hit.normal.y > 0.9, "normal = {:?}", hit.normal);

    // A ray that misses returns None.
    assert!(
        world
            .cast_ray(
                DVec3::new(50.0, 5.0, 0.0),
                DVec3::new(0.0, -1.0, 0.0),
                100.0
            )
            .is_none(),
        "ray far from the box should miss"
    );
}

#[test]
fn trimesh_collider_from_buffers_is_hit_by_raycast() {
    let mut world = PhysicsWorld3D::new(DVec3::ZERO);

    // A flat quad at y=0 spanning x,z in [-5, 5], as two triangles — the
    // mesh-collider seam (P12).
    let vertices = vec![
        DVec3::new(-5.0, 0.0, -5.0),
        DVec3::new(5.0, 0.0, -5.0),
        DVec3::new(5.0, 0.0, 5.0),
        DVec3::new(-5.0, 0.0, 5.0),
    ];
    let indices = vec![[0u32, 1, 2], [0, 2, 3]];

    let body = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    let collider = world
        .add_collider(
            body,
            ColliderDesc3D::new(ColliderShape3D::Trimesh { vertices, indices }),
        )
        .expect("trimesh should build from valid buffers");

    let hit = world
        .cast_ray(DVec3::new(0.0, 5.0, 0.0), DVec3::new(0.0, -1.0, 0.0), 100.0)
        .expect("ray should hit the trimesh");
    assert_eq!(hit.collider, collider);
    assert!((hit.toi - 5.0).abs() < 1e-9, "toi = {}", hit.toi);

    // A degenerate (empty) trimesh is refused rather than panicking.
    let empty = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    assert!(world
        .add_collider(
            empty,
            ColliderDesc3D::new(ColliderShape3D::Trimesh {
                vertices: vec![],
                indices: vec![],
            }),
        )
        .is_none());
}

#[test]
fn point_and_aabb_queries_are_deterministically_ordered() {
    let mut world = PhysicsWorld3D::new(DVec3::ZERO);

    // Two overlapping boxes at the origin so a point query returns both.
    for _ in 0..2 {
        let b = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
        world.add_collider(
            b,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::splat(1.0),
            }),
        );
    }

    let at_origin = world.intersect_point(DVec3::ZERO);
    assert_eq!(at_origin.len(), 2, "both boxes contain the origin");
    // Sorted ascending — deterministic ordering guarantee.
    assert!(at_origin[0] < at_origin[1]);

    let overlap = world.intersect_aabb(DVec3::splat(-0.5), DVec3::splat(0.5));
    assert_eq!(overlap.len(), 2);
    assert!(overlap[0] < overlap[1]);

    // A point far away hits nothing.
    assert!(world.intersect_point(DVec3::splat(100.0)).is_empty());
}

#[test]
fn fixed_stepper_drives_step_count() {
    // The stepper and world compose: two frames of 1.5 ticks each → 3 steps.
    let mut stepper = FixedStepper::new(DT);
    let mut world = build_drop_world();

    let mut total = 0;
    for _ in 0..2 {
        let n = stepper.accumulate(DT * 1.5);
        for _ in 0..n {
            world.step(stepper.fixed_dt());
            total += 1;
        }
    }
    assert_eq!(
        total, 3,
        "expected 3 fixed steps across two 1.5-tick frames"
    );
}
