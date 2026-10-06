//! **STATIC SCENERY SLEEPS WHERE NOTHING CAN TOUCH IT** (wave PERF1b) --
//! the arms for `inf_physics::d3::dormancy`.
//!
//! The mechanism takes static colliders out of rapier's broad phase while no
//! mover is within reach of them, which is where the island's 56 000-100 000
//! fixed-fixed pairs went. What it must NOT change is everything a mover can
//! see: the contacts a body makes with scenery, the events they raise, and
//! every query. The arms run the same scripted world with the dormancy on and
//! off and compare.

use std::collections::BTreeSet;

use glam::{DQuat, DVec3};
use inf_physics::d3::{BodyKind3D, CastTargets, ColliderDesc3D, ColliderShape3D, PhysicsWorld3D};

const DT: f64 = 1.0 / 60.0;

fn static_box(w: &mut PhysicsWorld3D, at: DVec3, half: DVec3) {
    let b = w.add_body(BodyKind3D::Static, at, DQuat::IDENTITY);
    w.add_collider(
        b,
        ColliderDesc3D::new(ColliderShape3D::Box { half_extents: half }),
    )
    .expect("a static box attaches");
}

/// A street of touching floor tiles (every tile overlaps its neighbours --
/// the fixed-fixed pairs) with a wall across its far end, a dynamic box
/// dropped and pushed along it, and a fast ball thrown at the wall.
struct Street {
    w: PhysicsWorld3D,
    crate_body: inf_physics::d3::BodyId3D,
    ball: inf_physics::d3::BodyId3D,
}

fn street(dormant: bool) -> Street {
    let mut w = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    w.set_static_dormancy(dormant);
    for i in 0..80 {
        for j in -2..=2 {
            static_box(
                &mut w,
                DVec3::new(f64::from(i) * 2.0, -0.5, f64::from(j) * 2.0),
                DVec3::new(1.05, 0.5, 1.05),
            );
        }
    }
    // The wall across the street's far end, 120 m out.
    static_box(
        &mut w,
        DVec3::new(120.0, 2.0, 0.0),
        DVec3::new(0.5, 2.0, 5.0),
    );
    let crate_body = w.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(2.0, 1.0, 0.0),
        DQuat::IDENTITY,
    );
    w.add_collider(
        crate_body,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::splat(0.4),
        }),
    )
    .expect("the crate attaches");
    let ball = w.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(60.0, 1.5, 0.0),
        DQuat::IDENTITY,
    );
    w.add_collider(
        ball,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.3 }),
    )
    .expect("the ball attaches");
    Street {
        w,
        crate_body,
        ball,
    }
}

/// **A MOVER SEES THE SAME SCENERY** (wave PERF1b). Six hundred steps of a
/// crate dropped onto a street of overlapping static tiles and shoved along it
/// at 8 m/s, and a ball thrown at 30 m/s into a wall 60 m away: with the
/// dormancy on, both end where they end with it off (to a tenth of a
/// millimetre -- the broad phase holds fewer pairs, so rapier's pair order
/// may differ in the last bits), the ball is stopped by the wall rather than
/// tunnelling through a static woken too late, the count of contact-start
/// events is the same, and the dormant world tracked a small fraction of the
/// awake world's pairs while statics actually slept and woke.
#[test]
fn a_mover_meets_the_same_scenery_with_the_statics_asleep() {
    let mut runs = Vec::new();
    for dormant in [false, true] {
        let mut s = street(dormant);
        let mut started = 0usize;
        let mut pairs_max = 0usize;
        for step in 0..600 {
            if step == 60 {
                s.w.set_body_linvel(s.crate_body, DVec3::new(8.0, 0.0, 0.0));
                s.w.set_body_linvel(s.ball, DVec3::new(30.0, 0.0, 0.0));
            }
            s.w.step(DT);
            started +=
                s.w.drain_contact_events()
                    .iter()
                    .filter(|e| e.phase == inf_physics::ContactPhase::Started)
                    .count();
            pairs_max = pairs_max.max(s.w.contact_pair_counts().0);
        }
        let crate_at = s.w.body_translation(s.crate_body).expect("the crate");
        let ball_at = s.w.body_translation(s.ball).expect("the ball");
        let stats = s.w.dormancy_stats();
        println!(
            "dormancy {dormant}: crate {crate_at:.4}, ball {ball_at:.4}, {started} contact starts, \
             at most {pairs_max} pairs tracked, {stats:?}"
        );
        runs.push((crate_at, ball_at, started, pairs_max, stats));
    }
    let (awake, asleep) = (&runs[0], &runs[1]);
    assert!(
        (awake.0 - asleep.0).length() < 1e-4,
        "the crate ended {:?} awake and {:?} with the statics asleep",
        awake.0,
        asleep.0
    );
    assert!(
        (awake.1 - asleep.1).length() < 1e-4,
        "the ball ended {:?} awake and {:?} with the statics asleep",
        awake.1,
        asleep.1
    );
    assert!(
        asleep.1.x < 119.5,
        "the ball went through the wall at x = 120: {:?}",
        asleep.1
    );
    assert_eq!(awake.2, asleep.2, "the contact-start counts differ");
    assert!(
        asleep.3 * 3 < awake.3,
        "the dormant world tracked {} pairs against {} awake -- the dormancy bought nothing",
        asleep.3,
        awake.3
    );
    assert!(
        asleep.4.asleep > 300,
        "too few statics asleep: {:?}",
        asleep.4
    );
    assert!(
        asleep.4.toggles > asleep.4.managed as u64,
        "statics never woke: {:?}",
        asleep.4
    );
    assert!(
        asleep.4.requeries > 10,
        "the movers never re-queried: {:?}",
        asleep.4
    );
}

/// **A SLEEPING WALL STILL STOPS A BULLET** (wave PERF1b). The filtered query
/// doors refuse a DISABLED collider (a parked capsule is a hole in the air);
/// a static the dormancy put to sleep is disabled in rapier and must not be
/// refused. Nothing moves near the wall, so it is asleep; a ray fired at it
/// through `cast_ray_where` hits it, and `collider_enabled` calls it enabled.
#[test]
fn a_sleeping_wall_still_stops_a_ray() {
    let mut w = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    let b = w.add_body(
        BodyKind3D::Static,
        DVec3::new(10.0, 0.0, 0.0),
        DQuat::IDENTITY,
    );
    let wall = w
        .add_collider(
            b,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::new(0.5, 3.0, 3.0),
            }),
        )
        .expect("the wall attaches");
    for _ in 0..3 {
        w.step(DT);
    }
    assert_eq!(w.dormancy_stats().asleep, 1, "the wall should be asleep");
    assert_eq!(
        w.collider_enabled(wall),
        Some(true),
        "a sleeping wall reports enabled"
    );
    let hit = w.cast_ray_where(
        DVec3::ZERO,
        DVec3::X,
        100.0,
        &BTreeSet::new(),
        CastTargets::All,
    );
    let hit = hit.expect("the ray passed through a sleeping wall");
    println!("ray hit the sleeping wall at toi {:.3}", hit.toi);
    assert!((hit.toi - 9.5).abs() < 1e-6, "hit at toi {}", hit.toi);
}
