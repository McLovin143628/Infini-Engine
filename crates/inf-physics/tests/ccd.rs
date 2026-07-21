//! Continuous Collision Detection (P12.1): a fast small body tunnels through a
//! thin static wall WITHOUT ccd and is stopped by it WITH ccd — both dimensions.

use glam::{DQuat, DVec2, DVec3};
use inf_physics::d2::ColliderDesc2D;
use inf_physics::d3::ColliderDesc3D;
use inf_physics::{
    BodyKind, BodyKind3D, ColliderShape2D, ColliderShape3D, PhysicsWorld2D, PhysicsWorld3D,
};

const DT: f64 = 1.0 / 60.0;

/// Fire a small fast bullet at a thin wall at x≈0; return its final x. With `ccd`
/// the sweep catches the wall; without it, the bullet leaps past in one step.
fn bullet_final_x_3d(ccd: bool) -> f64 {
    let mut world = PhysicsWorld3D::new(DVec3::ZERO); // no gravity: pure translation

    // A thin static wall in the y-z plane at x=0 (10 cm thick).
    let wall = world.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    world.add_collider(
        wall,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(0.05, 5.0, 5.0),
        }),
    );

    // A small fast bullet approaching from the left.
    let bullet = world.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(-5.0, 0.0, 0.0),
        DQuat::IDENTITY,
    );
    world.add_collider(
        bullet,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.1 }).restitution(0.0),
    );
    world.set_body_ccd(bullet, ccd);
    // ~5 world units per step — far more than the wall thickness.
    world.set_body_linvel(bullet, DVec3::new(300.0, 0.0, 0.0));

    for _ in 0..30 {
        world.step(DT);
    }
    world.body_translation(bullet).unwrap().x
}

fn bullet_final_x_2d(ccd: bool) -> f64 {
    let mut world = PhysicsWorld2D::new(DVec2::ZERO);

    let wall = world.add_body(BodyKind::Static, DVec2::ZERO, 0.0);
    world.add_collider(
        wall,
        ColliderDesc2D::new(ColliderShape2D::Box {
            half_width: 0.05,
            half_height: 5.0,
        }),
    );

    let bullet = world.add_body(BodyKind::Dynamic, DVec2::new(-5.0, 0.0), 0.0);
    world.add_collider(
        bullet,
        ColliderDesc2D::new(ColliderShape2D::Circle { radius: 0.1 }).restitution(0.0),
    );
    world.set_body_ccd(bullet, ccd);
    world.set_body_linvel(bullet, DVec2::new(300.0, 0.0));

    for _ in 0..30 {
        world.step(DT);
    }
    world.body_translation(bullet).unwrap().x
}

#[test]
fn ccd_stops_tunnelling_3d() {
    let without = bullet_final_x_3d(false);
    let with = bullet_final_x_3d(true);
    assert!(
        without > 1.0,
        "without CCD the bullet should tunnel past the wall (x={without})"
    );
    assert!(
        with < 0.0,
        "with CCD the bullet should be stopped before the wall (x={with})"
    );
}

#[test]
fn ccd_stops_tunnelling_2d() {
    let without = bullet_final_x_2d(false);
    let with = bullet_final_x_2d(true);
    assert!(
        without > 1.0,
        "without CCD the bullet should tunnel past the wall (x={without})"
    );
    assert!(
        with < 0.0,
        "with CCD the bullet should be stopped before the wall (x={with})"
    );
}
