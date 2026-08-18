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

// ── convex hulls (P22.2) ────────────────────────────────────────────────────

/// The eight corners of an axis-aligned box of the given half-extents, plus a
/// few interior points a hull must simply absorb.
fn box_point_cloud(h: f64) -> Vec<DVec3> {
    let mut pts = Vec::new();
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                pts.push(DVec3::new(sx * h, sy * h, sz * h));
            }
        }
    }
    // Interior points: a hull ignores them, so their presence must change nothing.
    pts.push(DVec3::ZERO);
    pts.push(DVec3::splat(h * 0.25));
    pts
}

/// A hull built from a box's corner cloud collides exactly like the equivalent
/// `Box` — same raycast hit, same volume — which is what "the hull of a box is
/// the box" has to mean if the variant is to be trusted with chunk geometry.
#[test]
fn convex_hull_of_a_box_cloud_collides_like_the_box() {
    let h = 0.5;
    let hull = ColliderShape3D::ConvexHull {
        points: box_point_cloud(h),
    };
    let cube = ColliderShape3D::Box {
        half_extents: DVec3::splat(h),
    };

    // Same solid: the volume rapier will integrate is the cube's, to f64 slop.
    let vh = hull
        .volume_m3()
        .expect("a hull is a solid and has a volume");
    let vc = cube.volume_m3().expect("a cuboid has a volume");
    assert!(
        (vh - vc).abs() < 1e-9 && (vc - (2.0 * h).powi(3)).abs() < 1e-9,
        "hull {vh} vs box {vc}"
    );

    // Same surface: a downward ray from above hits both at the same distance.
    let hit_at = |shape: ColliderShape3D| {
        let mut world = PhysicsWorld3D::new(DVec3::ZERO);
        let b = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
        world
            .add_collider(b, ColliderDesc3D::new(shape))
            .expect("shape builds");
        world
            .cast_ray(DVec3::new(0.0, 5.0, 0.0), DVec3::new(0.0, -1.0, 0.0), 100.0)
            .expect("ray hits")
            .toi
    };
    let (th, tc) = (hit_at(hull), hit_at(cube));
    assert!((th - tc).abs() < 1e-9, "hull toi {th} vs box toi {tc}");
    assert!((tc - (5.0 - h)).abs() < 1e-9, "toi {tc}");
}

/// A point set that bounds no volume is **refused**, cleanly, at every door:
/// the pre-check says so, the shape build says so, and `add_collider` returns
/// `None` instead of panicking or inserting a collider with no shape.
///
/// The `coplanar` arm is the one that earned its keep: `parry`'s own hull builder
/// *accepts* a flat cloud and hands back a zero-thickness polyhedron, so a
/// refusal delegated entirely to it would have shipped fracture chunks with zero
/// mass. `MIN_HULL_VOLUME_M3` is the answer, and this is what found the need.
#[test]
fn degenerate_point_sets_refuse_cleanly() {
    let cases: [(&str, Vec<DVec3>); 5] = [
        ("empty", vec![]),
        ("single point", vec![DVec3::ZERO]),
        (
            "collinear",
            (0..8).map(|i| DVec3::new(i as f64, 0.0, 0.0)).collect(),
        ),
        (
            // A filled square in the y = 0 plane: plenty of points, zero thickness.
            "coplanar",
            (0..4)
                .flat_map(|i| (0..4).map(move |j| DVec3::new(i as f64, 0.0, j as f64)))
                .collect(),
        ),
        (
            "three points (a triangle is a surface, not a solid)",
            vec![
                DVec3::ZERO,
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
        ),
    ];

    let mut world = PhysicsWorld3D::new(DVec3::ZERO);
    for (what, points) in cases {
        assert!(
            !inf_physics::d3::convex_hull_is_buildable(&points),
            "{what}: the pre-check must refuse a point set that bounds no volume"
        );
        let shape = ColliderShape3D::ConvexHull { points };
        assert_eq!(shape.volume_m3(), None, "{what}: no volume to report");
        let body = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
        assert!(
            world
                .add_collider(body, ColliderDesc3D::new(shape))
                .is_none(),
            "{what}: add_collider must refuse, not panic"
        );
        // The refusal leaves the world usable: the body is still there, with no
        // collider attached to it.
        assert!(world.contains_body(body));
    }

    // …and the pre-check agrees with the build on a *valid* set, which is the
    // half that stops it being a rubber stamp.
    let good = box_point_cloud(0.5);
    assert!(inf_physics::d3::convex_hull_is_buildable(&good));
    let body = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    assert!(world
        .add_collider(
            body,
            ColliderDesc3D::new(ColliderShape3D::ConvexHull { points: good })
        )
        .is_some());
}

/// **The reason the variant exists**: a hull body has real mass, so it falls,
/// lands and rests. The same body built as a `Trimesh` of the same solid has no
/// mass at all — asserted here so the claim in `ColliderShape3D::ConvexHull`'s
/// docs is a measurement rather than a belief.
#[test]
fn a_dynamic_convex_hull_body_has_real_mass_and_comes_to_rest() {
    let h = 0.5;
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));

    // Floor.
    let floor = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, -1.0, 0.0),
        DQuat::IDENTITY,
    );
    world
        .add_collider(
            floor,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::new(20.0, 1.0, 20.0),
            }),
        )
        .unwrap();

    // A hull chunk dropped from 4 m, at a real material density (concrete).
    let start_y = 4.0;
    let chunk = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(0.0, start_y, 0.0),
        DQuat::IDENTITY,
    );
    world
        .add_collider(
            chunk,
            ColliderDesc3D::new(ColliderShape3D::ConvexHull {
                points: box_point_cloud(h),
            })
            .density(2400.0),
        )
        .unwrap();

    for _ in 0..(3.0 / DT) as usize {
        world.step(DT);
    }

    let y = world.body_translation(chunk).unwrap().y;
    // It fell (gravity acted on a real mass) and it rests ON the floor — the
    // hull's half-extent above the floor's top face at y = 0.
    assert!(y < start_y - 1.0, "the hull body never fell: y = {y}");
    assert!(
        (y - h).abs() < 0.05,
        "the hull body did not come to rest on the floor: y = {y}"
    );

    // The mass really is density × volume — and the trimesh of the same solid
    // has no volume to multiply, which is the whole point of the variant.
    let vol = ColliderShape3D::ConvexHull {
        points: box_point_cloud(h),
    }
    .volume_m3()
    .unwrap();
    assert!((vol - (2.0 * h).powi(3)).abs() < 1e-9);
    assert_eq!(
        ColliderShape3D::Trimesh {
            vertices: box_point_cloud(h),
            indices: vec![[0, 1, 2]],
        }
        .volume_m3(),
        None,
        "a triangle soup is a surface: it has no volume, hence no mass"
    );
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

// ── shape casts (P29.3) ─────────────────────────────────────────────────────
//
// The engine had no swept-volume query at all before this wave — the only one
// anywhere ran inside rapier's `move_shape` and was unreachable from outside the
// mover. These arms pin the properties every consumer of `cast_shape` depends
// on: the distance is metres along a UNIT direction, the hit is the nearest one,
// "already inside" is a distinguishable answer rather than a distance of zero
// that looks like a touch, and the exclusion set really keeps a collider out of
// the answer (a sweep almost always starts on top of its own character).

/// A world with a ceiling slab whose underside sits at `y = under`.
fn world_with_ceiling(under: f64) -> PhysicsWorld3D {
    let mut world = PhysicsWorld3D::new(DVec3::ZERO);
    let slab = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, under + 0.5, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        slab,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(5.0, 0.5, 5.0),
        }),
    );
    world
}

#[test]
fn a_swept_capsule_reports_the_distance_it_travelled_in_metres() {
    // Ceiling underside at y = 2. A 0.5 m-radius sphere centred at y = 0 has its
    // top at 0.5, so it may rise 1.5 m before touching.
    let mut world = world_with_ceiling(2.0);
    let none = std::collections::BTreeSet::new();

    let hit = world
        .cast_shape(
            &ColliderShape3D::Sphere { radius: 0.5 },
            DVec3::ZERO,
            DQuat::IDENTITY,
            DVec3::Y,
            10.0,
            &none,
        )
        .expect("the sphere must find the ceiling");
    assert!(
        (hit.toi - 1.5).abs() < 1e-9,
        "a unit direction makes the time of impact a distance in metres: {}",
        hit.toi
    );
    assert!(!hit.started_penetrating, "the sweep started in free space");
    assert!(
        hit.normal.y < -0.9,
        "the ceiling's outward normal points DOWN at the thing below it: {:?}",
        hit.normal
    );
    assert!(
        (hit.point.y - 2.0).abs() < 1e-6,
        "the witness point is on the ceiling's underside: {:?}",
        hit.point
    );

    // A budget shorter than the gap is a miss, not a near-miss: the clearance
    // caller asks "is anything within 1 m", and 1.5 m away is not.
    assert!(
        world
            .cast_shape(
                &ColliderShape3D::Sphere { radius: 0.5 },
                DVec3::ZERO,
                DQuat::IDENTITY,
                DVec3::Y,
                1.0,
                &none,
            )
            .is_none(),
        "a hit past `max_toi` is not a hit"
    );
}

#[test]
fn a_sweep_that_starts_inside_says_so_instead_of_reporting_a_touch() {
    // The clearance probe's most important answer. A capsule tall enough to
    // already intersect the ceiling must not come back as "hit at 0 m" — which
    // is what a grazing contact also looks like — but as "you are inside it".
    let mut world = world_with_ceiling(1.0);
    let none = std::collections::BTreeSet::new();

    let hit = world
        .cast_shape(
            &ColliderShape3D::Capsule {
                half_height: 0.9,
                radius: 0.3,
            },
            DVec3::new(0.0, 0.5, 0.0),
            DQuat::IDENTITY,
            DVec3::Y,
            2.0,
            &none,
        )
        .expect("an overlapping sweep is a hit, not a miss");
    assert!(
        hit.started_penetrating,
        "the capsule reaches 1.7 and the ceiling starts at 1.0 — that is an overlap"
    );
    assert_eq!(hit.toi, 0.0, "an overlap is time zero");

    // The control: a capsule that clears the ceiling sweeps cleanly. Without it
    // the arm above would pass on a probe that reported `started_penetrating`
    // unconditionally.
    let clear = world
        .cast_shape(
            &ColliderShape3D::Capsule {
                half_height: 0.2,
                radius: 0.1,
            },
            DVec3::ZERO,
            DQuat::IDENTITY,
            DVec3::Y,
            2.0,
            &none,
        )
        .expect("it still reaches the ceiling within 2 m");
    assert!(
        !clear.started_penetrating,
        "a capsule reaching 0.3 under a ceiling at 1.0 starts clear"
    );
    assert!((clear.toi - 0.7).abs() < 1e-9, "toi = {}", clear.toi);
}

#[test]
fn the_nearest_collider_wins_and_an_excluded_one_is_invisible() {
    let mut world = PhysicsWorld3D::new(DVec3::ZERO);
    // Two walls on the +X axis, at x = 2 and x = 6 (each 0.5 m half-thick).
    let mut ids = Vec::new();
    for x in [2.0_f64, 6.0] {
        let b = world.add_body(BodyKind3D::Static, DVec3::new(x, 0.0, 0.0), DQuat::IDENTITY);
        ids.push(
            world
                .add_collider(
                    b,
                    ColliderDesc3D::new(ColliderShape3D::Box {
                        half_extents: DVec3::new(0.5, 5.0, 5.0),
                    }),
                )
                .expect("a box collider builds"),
        );
    }
    let none = std::collections::BTreeSet::new();
    let probe = ColliderShape3D::Sphere { radius: 0.25 };

    let near = world
        .cast_shape(&probe, DVec3::ZERO, DQuat::IDENTITY, DVec3::X, 20.0, &none)
        .expect("something is in the way");
    assert_eq!(near.collider, ids[0], "the NEAR wall is the answer");
    assert!((near.toi - 1.25).abs() < 1e-9, "toi = {}", near.toi);

    // Exclude the near wall and the far one becomes the answer — the property
    // every character sweep rests on, since a character's own collider is always
    // the nearest thing to its own capsule.
    let mut skip = std::collections::BTreeSet::new();
    skip.insert(ids[0]);
    let far = world
        .cast_shape(&probe, DVec3::ZERO, DQuat::IDENTITY, DVec3::X, 20.0, &skip)
        .expect("the far wall is still there");
    assert_eq!(
        far.collider, ids[1],
        "the excluded collider must not answer"
    );
    assert!((far.toi - 5.25).abs() < 1e-9, "toi = {}", far.toi);

    // Exclude both and the sweep is clear.
    skip.insert(ids[1]);
    assert!(
        world
            .cast_shape(&probe, DVec3::ZERO, DQuat::IDENTITY, DVec3::X, 20.0, &skip)
            .is_none(),
        "with every collider excluded the sweep hits nothing"
    );
}

#[test]
fn a_sweep_that_describes_no_sweep_answers_no_rather_than_panicking() {
    let mut world = world_with_ceiling(2.0);
    let none = std::collections::BTreeSet::new();
    let probe = ColliderShape3D::Sphere { radius: 0.5 };

    for (label, dir, max) in [
        ("a zero direction", DVec3::ZERO, 10.0),
        ("a zero budget", DVec3::Y, 0.0),
        ("a negative budget", DVec3::Y, -1.0),
        ("a NaN budget", DVec3::Y, f64::NAN),
    ] {
        assert!(
            world
                .cast_shape(&probe, DVec3::ZERO, DQuat::IDENTITY, dir, max, &none)
                .is_none(),
            "{label} describes no sweep and must answer None"
        );
    }

    // A shape rapier refuses to build is refused here too, rather than panicking
    // inside parry (the `to_shared` door the collider builder already uses).
    assert!(
        world
            .cast_shape(
                &ColliderShape3D::ConvexHull {
                    points: vec![DVec3::ZERO, DVec3::X],
                },
                DVec3::ZERO,
                DQuat::IDENTITY,
                DVec3::Y,
                10.0,
                &none,
            )
            .is_none(),
        "a degenerate hull bounds no volume, so it sweeps nothing"
    );

    // Anti-vacuity: the same call with a real shape and a real budget DOES hit,
    // so the four Nones above are the refusals and not a broken probe.
    assert!(
        world
            .cast_shape(&probe, DVec3::ZERO, DQuat::IDENTITY, DVec3::Y, 10.0, &none)
            .is_some(),
        "the control sweep must hit, or this arm proves nothing"
    );
}

#[test]
fn an_unnormalized_direction_still_measures_in_metres() {
    // The doc promises `toi` is a distance. `cast_shape` normalizes, so a caller
    // passing a velocity-shaped vector (as land prediction will) gets metres and
    // not seconds-of-that-velocity.
    let mut world = world_with_ceiling(2.0);
    let none = std::collections::BTreeSet::new();
    let probe = ColliderShape3D::Sphere { radius: 0.5 };

    let unit = world
        .cast_shape(&probe, DVec3::ZERO, DQuat::IDENTITY, DVec3::Y, 10.0, &none)
        .expect("hit");
    let fast = world
        .cast_shape(
            &probe,
            DVec3::ZERO,
            DQuat::IDENTITY,
            DVec3::new(0.0, 37.0, 0.0),
            10.0,
            &none,
        )
        .expect("hit");
    assert!(
        (unit.toi - fast.toi).abs() < 1e-9,
        "a 37 m/s direction must not rescale the answer: {} vs {}",
        unit.toi,
        fast.toi
    );
}

#[test]
fn the_swept_shapes_rotation_is_honoured() {
    // A long bar swept upward: axis-aligned it is 0.2 m tall and clears; rolled
    // 90 degrees about Z its 2 m length becomes its height and it does not. The
    // mantle and camera probes both sweep rotated shapes, so this is not
    // decoration.
    let mut world = world_with_ceiling(1.0);
    let none = std::collections::BTreeSet::new();
    let bar = ColliderShape3D::Box {
        half_extents: DVec3::new(1.0, 0.1, 0.1),
    };

    let flat = world
        .cast_shape(&bar, DVec3::ZERO, DQuat::IDENTITY, DVec3::Y, 5.0, &none)
        .expect("hit");
    let rolled = world
        .cast_shape(
            &bar,
            DVec3::ZERO,
            DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            DVec3::Y,
            5.0,
            &none,
        )
        .expect("hit");
    assert!((flat.toi - 0.9).abs() < 1e-9, "flat toi = {}", flat.toi);
    assert!(
        rolled.started_penetrating,
        "rolled on its end the bar already touches the ceiling: toi = {}, penetrating = {}",
        rolled.toi, rolled.started_penetrating
    );
}
