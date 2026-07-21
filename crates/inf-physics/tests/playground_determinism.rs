//! P12.2 replay-determinism gate for the deepened physics.
//!
//! A "physics playground" scene composes **every** P12.1 feature at once — box
//! stacks, a fixed/revolute/prismatic/spherical/distance joint zoo (including a
//! motorized hinge), a CCD bullet fired at a thin wall, collision-layer filtering,
//! and a kinematic body sweeping a sensor — then runs 300 fixed steps twice and
//! asserts the two runs are **byte-identical** across every body pose. This is the
//! per-facade determinism guarantee (crate docs) extended to the new solver
//! features; the existing `d2`/`d3` drop-world determinism tests are the template.

use glam::{DQuat, DVec2, DVec3};
use inf_physics::d2::ColliderDesc2D;
use inf_physics::d3::ColliderDesc3D;
use inf_physics::{
    BodyKind, BodyKind3D, ColliderShape2D, ColliderShape3D, CollisionLayers, JointDesc2D,
    JointDesc3D, JointKind2D, JointKind3D, JointMotor2D, JointMotor3D, PhysicsWorld2D,
    PhysicsWorld3D,
};

const DT: f64 = 1.0 / 60.0;
const STEPS: usize = 300;

// ── 3D ──────────────────────────────────────────────────────────────────────

fn pose_bytes_3d(world: &PhysicsWorld3D) -> Vec<u8> {
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

/// Build + run the 3D playground for `STEPS` fixed steps; return the pose trace.
fn run_playground_3d() -> Vec<u8> {
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));

    // Ground.
    let floor = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    world.add_collider(
        floor,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(50.0, 0.5, 50.0),
        }),
    );

    // A stack of 6 dynamic boxes (contact solver + resting stability).
    for i in 0..6 {
        let b = world.add_body(
            BodyKind3D::Dynamic,
            DVec3::new(0.0, 1.0 + i as f64 * 1.02, 0.0),
            DQuat::IDENTITY,
        );
        world.add_collider(
            b,
            ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: DVec3::splat(0.5),
            })
            .friction(0.7),
        );
    }

    // A revolute pendulum.
    let anchor = world.add_body(
        BodyKind3D::Static,
        DVec3::new(-6.0, 6.0, 0.0),
        DQuat::IDENTITY,
    );
    let bar = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(-5.0, 6.0, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        bar,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(0.5, 0.1, 0.1),
        }),
    );
    world
        .add_joint(
            anchor,
            bar,
            JointDesc3D::new(JointKind3D::Revolute {
                axis: DVec3::Z,
                limits: Some([-1.5, 1.5]),
                motor: None,
            })
            .local_anchor2(DVec3::new(-1.0, 0.0, 0.0)),
        )
        .unwrap();

    // A motorized revolute wheel (no gravity influence isolatable, but part of
    // the deterministic trace).
    let hub = world.add_body(
        BodyKind3D::Static,
        DVec3::new(6.0, 6.0, 0.0),
        DQuat::IDENTITY,
    );
    let wheel = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(6.0, 6.0, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        wheel,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::splat(0.4),
        }),
    );
    world
        .add_joint(
            hub,
            wheel,
            JointDesc3D::new(JointKind3D::Revolute {
                axis: DVec3::Z,
                limits: None,
                motor: Some(JointMotor3D {
                    target_vel: 8.0,
                    damping: 1.0,
                    ..Default::default()
                }),
            }),
        )
        .unwrap();

    // A prismatic slider under gravity with limits.
    let rail = world.add_body(
        BodyKind3D::Static,
        DVec3::new(10.0, 6.0, 0.0),
        DQuat::IDENTITY,
    );
    let slider = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(10.0, 6.0, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        slider,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.3 }),
    );
    world
        .add_joint(
            rail,
            slider,
            JointDesc3D::new(JointKind3D::Prismatic {
                axis: DVec3::Y,
                limits: Some([-2.0, 0.0]),
                motor: None,
            }),
        )
        .unwrap();

    // A spherical ball joint + a distance rope.
    let sp_anchor = world.add_body(
        BodyKind3D::Static,
        DVec3::new(-10.0, 6.0, 0.0),
        DQuat::IDENTITY,
    );
    let sp_ball = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(-10.0, 5.0, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        sp_ball,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.3 }),
    );
    world
        .add_joint(
            sp_anchor,
            sp_ball,
            JointDesc3D::new(JointKind3D::Spherical).local_anchor2(DVec3::new(0.0, 1.0, 0.0)),
        )
        .unwrap();
    let rope_bob = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(-10.0, 3.0, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        rope_bob,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.3 }),
    );
    world
        .add_joint(
            sp_ball,
            rope_bob,
            JointDesc3D::new(JointKind3D::Distance { max_distance: 1.5 }),
        )
        .unwrap();

    // A CCD bullet fired at a thin wall (fast body + CCD in the trace).
    let wall = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, 3.0, 15.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        wall,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(5.0, 5.0, 0.05),
        }),
    );
    let bullet = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(0.0, 3.0, 5.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        bullet,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.1 }),
    );
    world.set_body_ccd(bullet, true);
    world.set_body_gravity_scale(bullet, 0.0);
    world.set_body_linvel(bullet, DVec3::new(0.0, 0.0, 200.0));

    // A collision-layer ghost that ignores the floor and free-falls.
    let ghost = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(20.0, 3.0, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        ghost,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.3 })
            .layers(CollisionLayers::new(0b10, 0b10)),
    );

    // A kinematic body sweeping a sensor volume.
    let sensor_body = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, 1.0, -10.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        sensor_body,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::splat(1.0),
        })
        .sensor(true),
    );
    let sweeper = world.add_body(
        BodyKind3D::Kinematic,
        DVec3::new(-5.0, 1.0, -10.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        sweeper,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.4 }),
    );

    for i in 0..STEPS {
        // Drive the kinematic sweeper deterministically by step index.
        let x = -5.0 + (i as f64) * 0.03;
        world.set_body_translation(sweeper, DVec3::new(x, 1.0, -10.0));
        world.step(DT);
        // Draining events must not perturb the deterministic pose state.
        let _ = world.drain_contact_events();
    }
    pose_bytes_3d(&world)
}

#[test]
fn playground_3d_is_byte_identical_across_two_runs() {
    let a = run_playground_3d();
    let b = run_playground_3d();
    assert_eq!(a, b, "3D physics playground diverged across two runs");
    assert!(!a.is_empty());
}

// ── 2D ──────────────────────────────────────────────────────────────────────

fn pose_bytes_2d(world: &PhysicsWorld2D) -> Vec<u8> {
    let mut bytes = Vec::new();
    for id in world.body_ids() {
        let t = world.body_translation(id).unwrap();
        let r = world.body_rotation(id).unwrap();
        for c in [t.x, t.y, r] {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    bytes
}

fn run_playground_2d() -> Vec<u8> {
    let mut world = PhysicsWorld2D::new(DVec2::new(0.0, -9.81));

    // Ground.
    let floor = world.add_body(BodyKind::Static, DVec2::new(0.0, 0.0), 0.0);
    world.add_collider(
        floor,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 50.0,
            half_height: 0.5,
        }),
    );

    // A stack of boxes.
    for i in 0..5 {
        let b = world.add_body(
            BodyKind::Dynamic,
            DVec2::new(0.0, 1.0 + i as f64 * 1.02),
            0.0,
        );
        world.add_collider(
            b,
            ColliderDesc2D::new(ColliderShape2D::Box {
                half_width: 0.5,
                half_height: 0.5,
            })
            .friction(0.7),
        );
    }

    // Revolute pendulum (2D hinge about Z).
    let anchor = world.add_body(BodyKind::Static, DVec2::new(-6.0, 6.0), 0.0);
    let bar = world.add_body(BodyKind::Dynamic, DVec2::new(-5.0, 6.0), 0.0);
    world.add_collider(
        bar,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 0.5,
            half_height: 0.1,
        }),
    );
    world
        .add_joint(
            anchor,
            bar,
            JointDesc2D::new(JointKind2D::Revolute {
                limits: None,
                motor: None,
            })
            .local_anchor2(DVec2::new(-1.0, 0.0)),
        )
        .unwrap();

    // Motorized revolute wheel.
    let hub = world.add_body(BodyKind::Static, DVec2::new(6.0, 6.0), 0.0);
    let wheel = world.add_body(BodyKind::Dynamic, DVec2::new(6.0, 6.0), 0.0);
    world.add_collider(
        wheel,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 0.4,
            half_height: 0.4,
        }),
    );
    world
        .add_joint(
            hub,
            wheel,
            JointDesc2D::new(JointKind2D::Revolute {
                limits: None,
                motor: Some(JointMotor2D {
                    target_vel: 8.0,
                    damping: 1.0,
                    ..Default::default()
                }),
            }),
        )
        .unwrap();

    // Prismatic slider with limits.
    let rail = world.add_body(BodyKind::Static, DVec2::new(10.0, 6.0), 0.0);
    let slider = world.add_body(BodyKind::Dynamic, DVec2::new(10.0, 6.0), 0.0);
    world.add_collider(
        slider,
        ColliderDesc2D::new(ColliderShape2D::Circle { radius: 0.3 }),
    );
    world
        .add_joint(
            rail,
            slider,
            JointDesc2D::new(JointKind2D::Prismatic {
                axis: DVec2::new(0.0, 1.0),
                limits: Some([-2.0, 0.0]),
                motor: None,
            }),
        )
        .unwrap();

    // Distance rope.
    let ranchor = world.add_body(BodyKind::Static, DVec2::new(-10.0, 6.0), 0.0);
    let bob = world.add_body(BodyKind::Dynamic, DVec2::new(-10.0, 4.5), 0.0);
    world.add_collider(
        bob,
        ColliderDesc2D::new(ColliderShape2D::Circle { radius: 0.3 }),
    );
    world
        .add_joint(
            ranchor,
            bob,
            JointDesc2D::new(JointKind2D::Distance { max_distance: 1.5 }),
        )
        .unwrap();

    // CCD bullet vs thin wall.
    let wall = world.add_body(BodyKind::Static, DVec2::new(15.0, 3.0), 0.0);
    world.add_collider(
        wall,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 0.05,
            half_height: 5.0,
        }),
    );
    let bullet = world.add_body(BodyKind::Dynamic, DVec2::new(5.0, 3.0), 0.0);
    world.add_collider(
        bullet,
        ColliderDesc2D::new(ColliderShape2D::Circle { radius: 0.1 }),
    );
    world.set_body_ccd(bullet, true);
    world.set_body_gravity_scale(bullet, 0.0);
    world.set_body_linvel(bullet, DVec2::new(200.0, 0.0));

    for _ in 0..STEPS {
        world.step(DT);
        let _ = world.drain_contact_events();
    }
    pose_bytes_2d(&world)
}

#[test]
fn playground_2d_is_byte_identical_across_two_runs() {
    let a = run_playground_2d();
    let b = run_playground_2d();
    assert_eq!(a, b, "2D physics playground diverged across two runs");
    assert!(!a.is_empty());
}
