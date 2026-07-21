//! Joint + collision-layer behaviour for the 2D facade (P12.1), mirroring the 3D
//! tests at two dimensions.

use glam::DVec2;
use inf_physics::d2::ColliderDesc2D;
use inf_physics::{
    BodyKind, ColliderShape2D, CollisionLayers, JointDesc2D, JointKind2D, JointMotor2D,
    PhysicsWorld2D,
};

const DT: f64 = 1.0 / 60.0;

#[test]
fn revolute_pendulum_swings_and_keeps_link_length() {
    let mut world = PhysicsWorld2D::new(DVec2::new(0.0, -9.81));
    let anchor = world.add_body(BodyKind::Static, DVec2::new(0.0, 5.0), 0.0);
    let bar = world.add_body(BodyKind::Dynamic, DVec2::new(1.0, 5.0), 0.0);
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

    let start = world.body_translation(bar).unwrap();
    for _ in 0..30 {
        world.step(DT);
    }
    let now = world.body_translation(bar).unwrap();
    assert!(
        now.y < start.y - 0.05,
        "pendulum did not swing down: {now:?}"
    );
    let r = (now - DVec2::new(0.0, 5.0)).length();
    assert!((r - 1.0).abs() < 0.2, "hinge link length drifted: r={r}");
}

#[test]
fn revolute_motor_spins_the_wheel() {
    let mut world = PhysicsWorld2D::new(DVec2::ZERO);
    let hub = world.add_body(BodyKind::Static, DVec2::ZERO, 0.0);
    let wheel = world.add_body(BodyKind::Dynamic, DVec2::ZERO, 0.0);
    world.add_collider(
        wheel,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 0.5,
            half_height: 0.5,
        }),
    );
    world
        .add_joint(
            hub,
            wheel,
            JointDesc2D::new(JointKind2D::Revolute {
                limits: None,
                motor: Some(JointMotor2D {
                    target_vel: 6.0,
                    damping: 1.0,
                    ..Default::default()
                }),
            }),
        )
        .unwrap();

    for _ in 0..120 {
        world.step(DT);
    }
    assert!(
        world.body_angvel(wheel).unwrap() > 3.0,
        "motor failed to spin the 2D wheel"
    );
}

#[test]
fn collision_layers_disable_interaction_2d() {
    let mut world = PhysicsWorld2D::new(DVec2::new(0.0, -9.81));
    let floor = world.add_body(BodyKind::Static, DVec2::ZERO, 0.0);
    world.add_collider(
        floor,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 50.0,
            half_height: 0.5,
        })
        .layers(CollisionLayers::new(0b1, 0xFFFF_FFFF)),
    );
    let ball = world.add_body(BodyKind::Dynamic, DVec2::new(0.0, 5.0), 0.0);
    world.add_collider(
        ball,
        ColliderDesc2D::new(ColliderShape2D::Circle { radius: 0.5 })
            .layers(CollisionLayers::new(0b10, 0b10)),
    );
    for _ in 0..120 {
        world.step(DT);
    }
    assert!(
        world.body_translation(ball).unwrap().y < -1.0,
        "ball should have passed through the floor"
    );
}
