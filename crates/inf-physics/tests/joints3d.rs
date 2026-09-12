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

/// **A JOINTED RIG SURVIVES BEING TELEPORTED** — wave VEH3c's audit.
///
/// # Why this arm exists
///
/// Wave VEH3c built the breakable-joint facade above and then deliberately did
/// **not** call it from the bodywork, on the strength of one measurement: *"a
/// door on a real revolute, held to a chassis the dispatcher teleports along a
/// nav path and zeroes the velocities of, was yanked by its own joint until the
/// ambulance was at 1 705 metres."* That refusal is the first thing the wave
/// carries and the first thing VEH3d or VEH3f is told to price.
///
/// It does not reproduce, and this arm is the measurement that says so. The
/// shape here is `dispatch::escort_nudge`'s own: every step, write the chassis'
/// position AND rotation, and zero both of its velocities — a body that is
/// being dragged along a route rather than driven. A 22 kg door hangs off it on
/// a real revolute with limits and a motor, drawn where the bodywork draws one,
/// which is HALF INSIDE the chassis hull.
///
/// Three regimes, and the numbers are in the output:
///
/// * **naive** — the chassis alone is written. The chassis stays on its own
///   schedule to **4 mm** over 240 drags and the door stays **1.6 m** from it.
///   The peak the joint carries is 202 N·s, which is a fifth of a saloon
///   bumper's 4 500 N·s mount.
/// * **as a unit** — the door is re-placed with the chassis and its velocities
///   zeroed too. 102 N·s.
/// * **as a unit, joint remade** — `remove_joint` then `add_joint` at the
///   teleport door, which is what resets rapier's accumulated impulses.
///   **0.0 N·s**: there is nothing left for a joint to do.
///
/// Nothing is flung in any of the three. What the wave measured was the
/// SPURIOUS crash its `applied_n` correction manufactured — a bellied van
/// reading 1 370 N·s of "crash" every step for four thousand steps, popping
/// doors on units that were never hit — and that correction is gone (see
/// `d3::bodywork`'s four-facts note). The refusal was never re-measured after
/// the cause was removed.
///
/// **The other half of the same ruling is `JointDesc3D::contacts`.** A part is
/// drawn on the chassis' outer face, so half its box is inside the chassis
/// collider, and two overlapping dynamic bodies that are also constrained
/// together are the P29.6 depenetration shape wearing a door. Measured below:
/// with contacts ON the pair reaches **6.07 m/s**, with them OFF **1.82** — the
/// ragdoll ruling, met at a car door.
#[test]
fn a_jointed_rig_survives_being_teleported_as_a_unit() {
    /// mode 0: the chassis alone. 1: the door moves with it. 2: and the joint is remade.
    fn drag(mode: u8, contacts: bool) -> (f64, f64, f64) {
        let mut w = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
        let g = w.add_body(
            BodyKind3D::Static,
            DVec3::new(0.0, -0.5, 0.0),
            DQuat::IDENTITY,
        );
        w.add_collider(
            g,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::new(400.0, 0.5, 400.0),
            }),
        );
        let chassis = w.add_body(
            BodyKind3D::Dynamic,
            DVec3::new(0.0, 0.8, 0.0),
            DQuat::IDENTITY,
        );
        w.add_collider(
            chassis,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::new(0.9, 0.7, 2.3),
            })
            .density(180.0),
        );
        // Half inside the hull, which is where a drawn part is.
        let door_local = DVec3::new(-0.85, 0.05, 0.6);
        let door = w.add_body(
            BodyKind3D::Dynamic,
            DVec3::new(0.0, 0.8, 0.0) + door_local,
            DQuat::IDENTITY,
        );
        w.add_collider(
            door,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::new(0.10, 0.45, 0.55),
            })
            .density(400.0),
        );
        let hinge = || {
            let mut d = JointDesc3D::new(JointKind3D::Revolute {
                axis: DVec3::Y,
                limits: Some([0.0, 66f64.to_radians()]),
                motor: Some(JointMotor3D {
                    target_pos: 66f64.to_radians(),
                    stiffness: 60.0,
                    damping: 12.0,
                    ..Default::default()
                }),
            })
            .local_anchor1(door_local + DVec3::new(0.0, 0.0, 0.55))
            .local_anchor2(DVec3::new(0.0, 0.0, 0.55));
            // **The P29.6 ruling at a car door**: a drawn part straddles the
            // hull face it is drawn on, and two overlapping dynamic bodies that
            // are also constrained together are a depenetration force with
            // nowhere to go. Measured by the arm below.
            d.contacts = contacts;
            d
        };
        let mut j = w
            .add_joint(chassis, door, hinge())
            .expect("the hinge builds");
        for _ in 0..60 {
            w.step(DT);
        }
        let mut peak = 0.0f64;
        for i in 0..240 {
            // `escort_nudge`'s own four writes.
            let at = DVec3::new(0.0, 0.8, i as f64 * 0.20);
            w.set_body_translation(chassis, at);
            w.set_body_rotation(chassis, DQuat::IDENTITY);
            w.set_body_linvel(chassis, DVec3::ZERO);
            w.set_body_angvel(chassis, DVec3::ZERO);
            if mode >= 1 {
                let rot = w.body_rotation(chassis).unwrap();
                w.set_body_translation(door, at + rot * door_local);
                w.set_body_rotation(door, rot);
                w.set_body_linvel(door, DVec3::ZERO);
                w.set_body_angvel(door, DVec3::ZERO);
            }
            if mode >= 2 {
                w.remove_joint(j);
                j = w.add_joint(chassis, door, hinge()).expect("it rebuilds");
            }
            w.step(DT);
            if let Some(imp) = w.joint_impulse(j) {
                peak = peak.max(imp.magnitude_ns());
            }
        }
        let c = w.body_translation(chassis).unwrap();
        let d = w.body_translation(door).unwrap();
        (
            (c - DVec3::new(0.0, 0.8, 239.0 * 0.20)).length(),
            (d - c).length(),
            peak,
        )
    }

    let mut peaks = [[0.0f64; 3]; 2];
    for (ci, contacts) in [true, false].into_iter().enumerate() {
        for mode in 0..3u8 {
            let (off_schedule, door_away, peak) = drag(mode, contacts);
            peaks[ci][mode as usize] = peak;
            eprintln!(
                "contacts {contacts:<5} mode {mode}: the chassis finished {off_schedule:.4} m off \
                 its own schedule, the door {door_away:.3} m from it, peak joint impulse \
                 {peak:.1} N.s"
            );
            // **NOTHING IS FLUNG**, in any of the six. This is the assertion the
            // wave's carried refusal says should be impossible.
            assert!(
                off_schedule < 0.25,
                "contacts {contacts} mode {mode}: a teleported chassis with a door on it finished \
                 {off_schedule:.3} m off the route it was dragged along — the joint is moving the car"
            );
            assert!(
                door_away < 3.0,
                "contacts {contacts} mode {mode}: the door ended {door_away:.3} m from the chassis \
                 it is hinged to"
            );
        }
    }
    // …and each of the two corrections takes a bite out of what the joint has to
    // carry, which is the recipe VEH3d inherits: turn the pair's contacts off,
    // and move the rig as a UNIT.
    eprintln!(
        "the peak the hinge carries: {:.1} N.s naive with contacts on, {:.1} with them off, \
         {:.1} moved as a unit, {:.1} with the joint remade",
        peaks[0][0], peaks[1][0], peaks[1][1], peaks[1][2]
    );
    assert!(
        peaks[1][0] < peaks[0][0] * 0.5,
        "turning the pair's contacts off left the hinge carrying {:.1} N.s against {:.1}",
        peaks[1][0],
        peaks[0][0]
    );
    assert!(
        peaks[1][1] < peaks[1][0] * 0.5,
        "moving the rig as a unit left the hinge carrying {:.1} N.s against {:.1}",
        peaks[1][1],
        peaks[1][0]
    );
    // Remaking the joint on top of that buys nothing measurable, and saying so
    // is the point: rapier's accumulated impulse is not what was wrong, so what
    // a teleport door owes a jointed rig is a PLACEMENT, not a solver reset.
    assert!(
        peaks[1][2] <= peaks[1][1] + 1e-9,
        "remaking the joint made it worse: {:.3} against {:.3}",
        peaks[1][2],
        peaks[1][1]
    );
}

/// **A PART DRAWN ON ITS OWN HULL FACE NEEDS `contacts = false`** — wave VEH3c's
/// audit, and the P29.6 ragdoll ruling met one system over.
///
/// A bodywork part is drawn on the chassis' OUTER FACE, so half its box is
/// inside the chassis collider by construction — exactly the way a thigh and a
/// shin overlap by two radii. Two overlapping dynamic bodies that are also
/// constrained together are a depenetration force with nowhere to go, and wave
/// VEH3c measured what it costs with a bumper in it: *"a 7 kg bumper that came
/// off at 60 km/h rose from 0.501 m to 1.328 m in the second after it let go."*
///
/// [`JointDesc3D::contacts`] is the door that closes it, and this is the
/// measurement. It is a SECOND cause behind wave VEH3c's refusal, distinct from
/// the teleport above, and it is the one the wave's first cut did not use.
#[test]
fn a_hinged_part_that_straddles_its_own_hull_needs_its_contacts_off() {
    fn overlap(contacts: bool) -> f64 {
        let mut w = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
        let g = w.add_body(
            BodyKind3D::Static,
            DVec3::new(0.0, -0.5, 0.0),
            DQuat::IDENTITY,
        );
        w.add_collider(
            g,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::new(400.0, 0.5, 400.0),
            }),
        );
        let chassis = w.add_body(
            BodyKind3D::Dynamic,
            DVec3::new(0.0, 0.8, 0.0),
            DQuat::IDENTITY,
        );
        w.add_collider(
            chassis,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::new(0.9, 0.7, 2.3),
            })
            .density(180.0),
        );
        let door_local = DVec3::new(-0.85, 0.05, 0.6);
        let door = w.add_body(
            BodyKind3D::Dynamic,
            DVec3::new(0.0, 0.8, 0.0) + door_local,
            DQuat::IDENTITY,
        );
        w.add_collider(
            door,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::new(0.10, 0.45, 0.55),
            })
            .density(400.0),
        );
        let mut d = JointDesc3D::new(JointKind3D::Revolute {
            axis: DVec3::Y,
            limits: Some([0.0, 66f64.to_radians()]),
            motor: Some(JointMotor3D {
                target_pos: 66f64.to_radians(),
                stiffness: 60.0,
                damping: 12.0,
                ..Default::default()
            }),
        })
        .local_anchor1(door_local + DVec3::new(0.0, 0.0, 0.55))
        .local_anchor2(DVec3::new(0.0, 0.0, 0.55));
        d.contacts = contacts;
        w.add_joint(chassis, door, d).expect("the hinge builds");
        let mut peak = 0.0f64;
        for _ in 0..240 {
            w.step(DT);
            for b in [chassis, door] {
                if let Some(v) = w.body_linvel(b) {
                    peak = peak.max(v.length());
                }
            }
        }
        peak
    }
    let on = overlap(true);
    let off = overlap(false);
    eprintln!(
        "a door half inside its own hull, on a hinge: peak {on:.2} m/s with contacts ON, \
         {off:.2} m/s with them OFF"
    );
    assert!(
        off < on * 0.75,
        "turning the pair's contacts off did not calm it: {off:.2} m/s against {on:.2}"
    );
    assert!(
        off < 3.0,
        "even with contacts off the pair reached {off:.2} m/s — something else is pushing it"
    );
}
