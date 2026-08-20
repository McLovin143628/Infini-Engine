//! Behavioural + determinism tests for the 2D physics facade.

use glam::DVec2;
use inf_physics::d2::ColliderDesc2D;
use inf_physics::{
    BodyKind, CharacterMover2D, ColliderShape2D, ContactPhase, FixedStepper, PhysicsWorld2D,
};

const DT: f64 = 1.0 / 60.0;

/// Serialize every body's pose (translation + rotation) to a flat little-endian
/// byte buffer, in deterministic handle order, so two worlds can be compared
/// byte-for-byte.
fn pose_bytes(world: &PhysicsWorld2D) -> Vec<u8> {
    let mut bytes = Vec::new();
    for id in world.body_ids() {
        let t = world.body_translation(id).unwrap();
        let r = world.body_rotation(id).unwrap();
        bytes.extend_from_slice(&t.x.to_le_bytes());
        bytes.extend_from_slice(&t.y.to_le_bytes());
        bytes.extend_from_slice(&r.to_le_bytes());
    }
    bytes
}

/// A world with a static floor at y=0 and a dynamic ball dropped above it.
fn build_drop_world() -> PhysicsWorld2D {
    let mut world = PhysicsWorld2D::new(DVec2::new(0.0, -9.81));

    let floor = world.add_body(BodyKind::Static, DVec2::new(0.0, 0.0), 0.0);
    world.add_collider(
        floor,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 50.0,
            half_height: 0.5,
        }),
    );

    let ball = world.add_body(BodyKind::Dynamic, DVec2::new(0.0, 10.0), 0.0);
    world.add_collider(
        ball,
        ColliderDesc2D::new(ColliderShape2D::Circle { radius: 0.5 })
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
    // A world built + stepped now must match one built + stepped again — the
    // same call sequence yields the same bytes every time.
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
    let mut world = PhysicsWorld2D::new(DVec2::new(0.0, -9.81));

    // Floor spanning x in [-10, 10] at y=0 (top surface y=0.5).
    let floor = world.add_body(BodyKind::Static, DVec2::new(0.0, 0.0), 0.0);
    world.add_collider(
        floor,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 10.0,
            half_height: 0.5,
        }),
    );
    // A wall at x=3 (spanning x in [2.8, 3.2]).
    let wall = world.add_body(BodyKind::Static, DVec2::new(3.0, 1.5), 0.0);
    world.add_collider(
        wall,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 0.2,
            half_height: 2.0,
        }),
    );

    // Character: a 0.5-radius ball starting on the floor at x=0, y=1.0.
    let mover = CharacterMover2D::new(ColliderShape2D::Circle { radius: 0.5 })
        .offset(0.02)
        .snap_to_ground(Some(0.2));

    let mut pos = DVec2::new(0.0, 1.0);
    let mut grounded_any = false;
    // Walk right into the wall; gravity pulls down each step so it stays grounded.
    for _ in 0..240 {
        let desired = DVec2::new(0.08, -0.05);
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
    let mut world = PhysicsWorld2D::new(DVec2::ZERO);

    // A static sensor box at the origin.
    let sensor_body = world.add_body(BodyKind::Static, DVec2::ZERO, 0.0);
    let sensor = world
        .add_collider(
            sensor_body,
            ColliderDesc2D::new(ColliderShape2D::Box {
                half_width: 1.0,
                half_height: 1.0,
            })
            .sensor(true),
        )
        .unwrap();

    // A kinematic body that we drive left-to-right through the sensor.
    let mover_body = world.add_body(BodyKind::Kinematic, DVec2::new(-5.0, 0.0), 0.0);
    let mover_col = world
        .add_collider(
            mover_body,
            ColliderDesc2D::new(ColliderShape2D::Circle { radius: 0.5 }),
        )
        .unwrap();

    let mut started = 0;
    let mut stopped = 0;
    let mut started_pair_ok = true;

    let mut x = -5.0;
    for _ in 0..200 {
        x += 0.1;
        world.set_body_translation(mover_body, DVec2::new(x, 0.0));
        world.step(1.0 / 60.0);
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
    let mut world = PhysicsWorld2D::new(DVec2::ZERO);

    // A box occupying x,y in [-1,1] centred at origin (top face at y=1).
    let body = world.add_body(BodyKind::Static, DVec2::ZERO, 0.0);
    world.add_collider(
        body,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 1.0,
            half_height: 1.0,
        }),
    );

    // Cast straight down from above the box.
    let hit = world
        .cast_ray(DVec2::new(0.0, 5.0), DVec2::new(0.0, -1.0), 100.0)
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
            .cast_ray(DVec2::new(50.0, 5.0), DVec2::new(0.0, -1.0), 100.0)
            .is_none(),
        "ray far from the box should miss"
    );
}

#[test]
fn point_and_aabb_queries_are_deterministically_ordered() {
    let mut world = PhysicsWorld2D::new(DVec2::ZERO);

    // Two overlapping boxes at the origin so a point query returns both.
    for _ in 0..2 {
        let b = world.add_body(BodyKind::Static, DVec2::ZERO, 0.0);
        world.add_collider(
            b,
            ColliderDesc2D::new(ColliderShape2D::Box {
                half_width: 1.0,
                half_height: 1.0,
            }),
        );
    }

    let at_origin = world.intersect_point(DVec2::ZERO);
    assert_eq!(at_origin.len(), 2, "both boxes contain the origin");
    // Sorted ascending — deterministic ordering guarantee.
    assert!(at_origin[0] < at_origin[1]);

    let overlap = world.intersect_aabb(DVec2::new(-0.5, -0.5), DVec2::new(0.5, 0.5));
    assert_eq!(overlap.len(), 2);
    assert!(overlap[0] < overlap[1]);

    // A point far away hits nothing.
    assert!(world.intersect_point(DVec2::new(100.0, 100.0)).is_empty());
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

/// **THE 2D MIRROR OF THE `FIXED_FIXED` NARROWING** (island wave I4b; armed by
/// its audit).
///
/// `PhysicsWorld2D::active_collision_types` is written as "the MIRROR of
/// `d3::PhysicsWorld3D::active_collision_types`, which carries the full argument
/// and the measurement" — and a mirror with no arm is two declarations agreeing
/// with each other rather than with the world. The 3D half has two arms in
/// `step_cost_3d.rs` and this is their twin, in the units this facade exposes:
/// **contact events**, not the narrow phase's pair table (`contact_pair_counts`
/// is a 3D diagnostic and 2D has no equivalent).
///
/// Both halves, because only the pair says anything: a static **solid** pair that
/// overlaps must now report nothing, and a static **sensor** over the same static
/// scenery must still report — which is the one case the flags were widened for
/// and is what `sensor => all()` keeps.
#[test]
fn a_static_solid_pair_reports_nothing_and_a_static_sensor_pair_still_does() {
    let overlap = |sensor: bool| -> usize {
        let mut world = PhysicsWorld2D::new(DVec2::new(0.0, -9.81));
        let a = world.add_body(BodyKind::Static, DVec2::ZERO, 0.0);
        world
            .add_collider(
                a,
                ColliderDesc2D::new(ColliderShape2D::Box {
                    half_width: 1.0,
                    half_height: 1.0,
                }),
            )
            .expect("the scenery's collider attaches");
        let b = world.add_body(BodyKind::Static, DVec2::new(0.5, 0.0), 0.0);
        let mut desc = ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 1.0,
            half_height: 1.0,
        });
        desc.sensor = sensor;
        world
            .add_collider(b, desc)
            .expect("the second collider attaches");
        let mut events = 0usize;
        for _ in 0..4 {
            world.step(DT);
            events += world.drain_contact_events().len();
        }
        events
    };

    let solid = overlap(false);
    let sensor = overlap(true);
    println!("2D static pair: {solid} events solid, {sensor} events with a sensor");
    assert_eq!(
        solid, 0,
        "two overlapping static SOLID colliders reported {solid} contact events \
         — `FIXED_FIXED` is back on for 2D solids, and the 2D world is paying \
         the manifold the 3D one stopped paying"
    );
    assert!(
        sensor > 0,
        "a static 2D sensor over static scenery reported nothing — the \
         `FIXED_FIXED` narrowing took the sensor case with it, which is the one \
         case the engine widened the flags for"
    );
}
