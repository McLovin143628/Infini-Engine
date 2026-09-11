//! Joint, collision-layer, and material-combine behaviour for the 3D facade (P12.1).

use glam::{DQuat, DVec3};
use inf_physics::d3::ColliderDesc3D;
use inf_physics::{
    BodyKind3D, ColliderShape3D, CollisionLayers, CombineRule, JointDesc3D, JointKind3D,
    JointMotor3D, PhysicsWorld3D,
};

const DT: f64 = 1.0 / 60.0;

fn dynamic_box(world: &mut PhysicsWorld3D, at: DVec3) -> inf_physics::BodyId3D {
    let b = world.add_body(BodyKind3D::Dynamic, at, DQuat::IDENTITY);
    world.add_collider(
        b,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::splat(0.5),
        }),
    );
    b
}

#[test]
fn fixed_joint_welds_two_bodies() {
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    // A static anchor and a dynamic body 2 units to its right, welded.
    let anchor = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    let body = dynamic_box(&mut world, DVec3::new(2.0, 0.0, 0.0));
    let jid = world
        .add_joint(
            anchor,
            body,
            JointDesc3D::new(JointKind3D::Fixed).local_anchor1(DVec3::new(2.0, 0.0, 0.0)),
        )
        .expect("joint should build");
    assert!(world.contains_joint(jid));

    for _ in 0..180 {
        world.step(DT);
    }
    // Welded to a static anchor: the body barely moves despite gravity.
    let p = world.body_translation(body).unwrap();
    assert!(
        (p - DVec3::new(2.0, 0.0, 0.0)).length() < 0.2,
        "fixed joint let the body fall to {p:?}"
    );

    assert!(world.remove_joint(jid));
    assert!(!world.contains_joint(jid));
}

#[test]
fn revolute_joint_swings_like_a_pendulum() {
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    let anchor = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, 5.0, 0.0),
        DQuat::IDENTITY,
    );
    // Bar hangs to the right of the anchor; hinge about Z at the anchor point.
    let bar = dynamic_box(&mut world, DVec3::new(1.0, 5.0, 0.0));
    world
        .add_joint(
            anchor,
            bar,
            JointDesc3D::new(JointKind3D::Revolute {
                axis: DVec3::Z,
                limits: None,
                motor: None,
            })
            .local_anchor1(DVec3::ZERO)
            .local_anchor2(DVec3::new(-1.0, 0.0, 0.0)),
        )
        .unwrap();

    let start = world.body_translation(bar).unwrap();
    for _ in 0..30 {
        world.step(DT);
    }
    let now = world.body_translation(bar).unwrap();
    // Gravity swings the bar down about the hinge: it drops and stays roughly a
    // unit from the anchor (rigid link length preserved).
    assert!(
        now.y < start.y - 0.05,
        "pendulum did not swing down: {now:?}"
    );
    let radius = (now - DVec3::new(0.0, 5.0, 0.0)).length();
    assert!(
        (radius - 1.0).abs() < 0.2,
        "hinge link length drifted: r={radius}"
    );
}

#[test]
fn revolute_motor_drives_rotation() {
    let mut world = PhysicsWorld3D::new(DVec3::ZERO); // no gravity: isolate the motor
    let anchor = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    let wheel = dynamic_box(&mut world, DVec3::new(0.0, 0.0, 0.0));
    world
        .add_joint(
            anchor,
            wheel,
            JointDesc3D::new(JointKind3D::Revolute {
                axis: DVec3::Z,
                limits: None,
                motor: Some(JointMotor3D {
                    target_vel: 6.0,
                    stiffness: 0.0,
                    damping: 1.0,
                    ..Default::default()
                }),
            }),
        )
        .unwrap();

    for _ in 0..120 {
        world.step(DT);
    }
    // The motor should spin the wheel up toward its target angular velocity.
    let w = world.body_angvel(wheel).unwrap();
    assert!(
        w.z > 3.0,
        "motor failed to drive rotation (angvel.z={})",
        w.z
    );
}

#[test]
fn distance_joint_limits_separation() {
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    let anchor = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, 5.0, 0.0),
        DQuat::IDENTITY,
    );
    let bob = dynamic_box(&mut world, DVec3::new(0.0, 4.5, 0.0));
    world
        .add_joint(
            anchor,
            bob,
            JointDesc3D::new(JointKind3D::Distance { max_distance: 2.0 }),
        )
        .unwrap();

    for _ in 0..240 {
        world.step(DT);
    }
    // The bob falls but the rope caps it at 2 units below the anchor.
    let d = (world.body_translation(bob).unwrap() - DVec3::new(0.0, 5.0, 0.0)).length();
    assert!(d <= 2.05, "rope stretched past max_distance: {d}");
    assert!(d > 1.5, "rope did not extend under gravity: {d}");
}

#[test]
fn spherical_joint_holds_anchor() {
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    let anchor = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, 5.0, 0.0),
        DQuat::IDENTITY,
    );
    let ball = dynamic_box(&mut world, DVec3::new(0.0, 4.0, 0.0));
    world
        .add_joint(
            anchor,
            ball,
            JointDesc3D::new(JointKind3D::Spherical)
                .local_anchor1(DVec3::ZERO)
                .local_anchor2(DVec3::new(0.0, 1.0, 0.0)),
        )
        .unwrap();

    for _ in 0..240 {
        world.step(DT);
    }
    // The ball-socket keeps the anchor point fixed: the ball hangs one unit below.
    let p = world.body_translation(ball).unwrap();
    let anchor_gap = (p + DVec3::new(0.0, 1.0, 0.0) - DVec3::new(0.0, 5.0, 0.0)).length();
    assert!(
        anchor_gap < 0.2,
        "spherical anchor drifted: gap={anchor_gap}"
    );
}

#[test]
fn collision_layers_disable_interaction() {
    // A ball whose filter excludes the floor's layer falls straight through it.
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));

    let floor = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    // Floor is a member of layer bit 0 only.
    world.add_collider(
        floor,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(50.0, 0.5, 50.0),
        })
        .layers(CollisionLayers::new(0b1, 0xFFFF_FFFF)),
    );

    let ball = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(0.0, 5.0, 0.0),
        DQuat::IDENTITY,
    );
    // Ball is on layer bit 1 and only filters for layer bit 1 — NOT the floor's
    // bit 0 — so the pair never interacts.
    world.add_collider(
        ball,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.5 })
            .layers(CollisionLayers::new(0b10, 0b10)),
    );

    for _ in 0..120 {
        world.step(DT);
    }
    // Fell through: well below the floor surface.
    let y = world.body_translation(ball).unwrap().y;
    assert!(y < -1.0, "ball should have passed through the floor, y={y}");
}

#[test]
fn restitution_max_combine_makes_a_bouncy_pair() {
    // A very bouncy ball onto a dead floor. With Max combine the pair's effective
    // restitution is the ball's high value, so it rebounds; with Min it would not.
    fn peak_rebound(rule: CombineRule) -> f64 {
        let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
        let floor = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
        world.add_collider(
            floor,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::new(50.0, 0.5, 50.0),
            })
            .restitution(0.0)
            .restitution_combine(rule),
        );
        let ball = world.add_body(
            BodyKind3D::Dynamic,
            DVec3::new(0.0, 3.0, 0.0),
            DQuat::IDENTITY,
        );
        world.add_collider(
            ball,
            ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.5 })
                .restitution(0.95)
                .restitution_combine(rule),
        );

        // Let it hit the floor, then track the highest point it rebounds to.
        let mut settled_min = f64::MAX;
        let mut peak_after: f64 = 0.0;
        for i in 0..240 {
            world.step(DT);
            let y = world.body_translation(ball).unwrap().y;
            if i > 40 {
                settled_min = settled_min.min(y);
                if y > settled_min + 0.01 {
                    peak_after = peak_after.max(y);
                }
            }
        }
        peak_after
    }

    let bouncy = peak_rebound(CombineRule::Max);
    let dead = peak_rebound(CombineRule::Min);
    assert!(
        bouncy > dead + 0.3,
        "Max-combine should bounce higher than Min-combine (max={bouncy}, min={dead})"
    );
}

// ── breakable joints (wave VEH3c) ───────────────────────────────────────────

/// A heavy body hung off a hinge, and the hinge watched.
///
/// Returns `(broke, peak_ns, steps_to_break)`. The load is a real one: a 2 000 kg
/// box on a 0.5 m arm under gravity pulls about 9.8 kN, i.e. roughly 163 N.s
/// every 60 Hz step, so a 60 N.s threshold lets go and a 10 000 N.s one does not.
fn hinge_under_load(threshold_ns: f64) -> (bool, f64, usize) {
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    let anchor = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, 5.0, 0.0),
        DQuat::IDENTITY,
    );
    let arm = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(0.5, 5.0, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        arm,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(0.5, 0.1, 0.5),
        })
        .density(4_000.0),
    );
    let jid = world
        .add_joint(
            anchor,
            arm,
            JointDesc3D::new(JointKind3D::Revolute {
                axis: DVec3::Z,
                limits: None,
                motor: None,
            })
            .local_anchor1(DVec3::ZERO)
            .local_anchor2(DVec3::new(-0.5, 0.0, 0.0)),
        )
        .expect("the hinge builds");
    let watch = [inf_physics::d3::BreakWatch3D {
        joint: jid,
        threshold_ns,
    }];
    let mut peak = 0.0f64;
    let mut broke_at = None;
    for i in 0..240 {
        world.step(DT);
        if let Some(imp) = world.joint_impulse(jid) {
            peak = peak.max(imp.magnitude_ns());
        }
        let broken = world.break_over_threshold(&watch);
        if let Some(b) = broken.first() {
            assert_eq!(b.joint, jid);
            assert!(b.impulse_ns > b.threshold_ns);
            broke_at = Some(i);
            break;
        }
    }
    (broke_at.is_some(), peak, broke_at.unwrap_or(240))
}

#[test]
fn a_watched_hinge_lets_go_over_its_threshold_and_holds_under_it() {
    // The measurement first: what does this hinge actually carry?
    let (_, peak, _) = hinge_under_load(f64::INFINITY);
    eprintln!("the loaded hinge carries a peak of {peak:.1} N.s a step");
    assert!(
        peak > 1.0,
        "the fixture's hinge carries {peak:.3} N.s -- it is not loaded, so \
         neither half of this arm measures anything"
    );

    // Over the threshold: it lets go, and the handle is dead afterwards.
    let (broke, at_peak, step) = hinge_under_load(peak * 0.5);
    assert!(
        broke,
        "a hinge carrying {peak:.1} N.s did not break at {:.1}",
        peak * 0.5
    );
    eprintln!("broke on step {step} carrying {at_peak:.1} N.s");

    // THE MUTATION: an unbreakable threshold, and nothing lets go in four
    // seconds of the same load.
    let (broke, _, step) = hinge_under_load(peak * 1000.0);
    assert!(!broke, "an unbreakable hinge broke on step {step}");

    // …and a refusal is a value: zero and NaN are UNBREAKABLE, not instant.
    for t in [0.0, -1.0, f64::NAN] {
        let (broke, _, _) = hinge_under_load(t);
        assert!(!broke, "a threshold of {t} broke the hinge");
    }
}

#[test]
fn a_broken_joint_leaves_a_free_body_behind() {
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    let anchor = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, 20.0, 0.0),
        DQuat::IDENTITY,
    );
    let arm = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(0.5, 20.0, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        arm,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(0.5, 0.1, 0.5),
        })
        .density(4_000.0),
    );
    let jid = world
        .add_joint(
            anchor,
            arm,
            JointDesc3D::new(JointKind3D::Revolute {
                axis: DVec3::Z,
                limits: None,
                motor: None,
            })
            .local_anchor1(DVec3::ZERO)
            .local_anchor2(DVec3::new(-0.5, 0.0, 0.0)),
        )
        .expect("the hinge builds");
    for _ in 0..30 {
        world.step(DT);
    }
    let held = world.body_translation(arm).unwrap().y;
    assert!(
        held > 19.0,
        "the hinge did not hold the arm up: it is at {held:.3}"
    );
    let broken = world.break_over_threshold(&[inf_physics::d3::BreakWatch3D {
        joint: jid,
        threshold_ns: 1.0,
    }]);
    assert_eq!(broken.len(), 1);
    assert!(!world.contains_joint(jid), "the handle is still live");
    for _ in 0..60 {
        world.step(DT);
    }
    let fell = world.body_translation(arm).unwrap().y;
    eprintln!("held at {held:.3} m, fell to {fell:.3} m in one second");
    assert!(
        fell < held - 4.0,
        "the freed body only fell from {held:.3} to {fell:.3}"
    );
}
