//! **THE VISIBILITY LAW** (P20.4), pinned from the simulation's side.
//!
//! > `Visibility` filters what is **drawn**. It never filters what is
//! > **simulated**.
//!
//! P20.3 shipped a "KNOWN DIVERGENCE": the render projectors skip a `WaterBody`
//! on a hidden entity, while `PhysicsBridge3D`'s gather walks every one of them,
//! so hiding a lake in the outliner leaves a swimmer swimming while the camera
//! stays dry. P20.4 decided it, and decided it in favour of the **simulation**,
//! because nothing in the engine's simulation has ever read visibility:
//!
//! * the 2D and 3D bridges gather rigid bodies, colliders and joints on component
//!   presence alone — a hidden wall still blocks;
//! * P19.5's `ScatteredSolid` colliders likewise;
//! * `terrain.height_at` picks the lowest-`Guid` non-empty terrain with no
//!   visibility test, `AudioSource`s keep playing, sensors keep triggering, and
//!   `partition::occupies_space` bins a hidden entity like any other.
//!
//! `ComputedVisibility` has exactly three readers in the repository: the two
//! render projectors and the Outliner DTO. Water is not going to be the fourth,
//! because that would make an **editor authoring toggle** — one that ships inside
//! the cooked pack — change physics.
//!
//! So this file asserts the law twice over, and the second half is what makes the
//! first half mean something: a hidden **lake** floats a boat bit-identically to a
//! visible one, *and* a hidden **collider** still blocks a falling body. If some
//! future change teaches the sim to read visibility, both fail together and the
//! decision gets revisited on purpose.
//!
//! The render half is pinned in `runtime/inf-player/tests/water_projection.rs`
//! (`a_hidden_water_body_is_not_drawn_but_is_still_simulated`), and the
//! *rationale* lives on `RenderWater::surface()`.

use glam::DVec3;
use inf_ecs::components::{
    BodyKind3D, Buoyancy, Collider3D, ColliderShape3DKind, RigidBody3D, Transform, Visibility,
    WaterBody,
};
use inf_ecs::math::Vec2d;
use inf_ecs::{ComputedVisibility, EcsWorld, Vec3d};
use inf_physics::PhysicsBridge3D;
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);

const LAKE: Uuid = Uuid::from_u128(0x2004_0001);
const BOX: Uuid = Uuid::from_u128(0x2004_0002);
const GROUND: Uuid = Uuid::from_u128(0x2004_0003);
const FALLER: Uuid = Uuid::from_u128(0x2004_0004);

/// A still lake (amplitude 0 ⇒ the height query is exact) at `level`.
fn still_lake(level: f64) -> WaterBody {
    WaterBody {
        wave_amplitude_m: 0.0,
        ..WaterBody::lake(level, Vec2d::splat(100.0))
    }
}

fn unit_box_collider(density: f64) -> Collider3D {
    Collider3D {
        shape_kind: ColliderShape3DKind::Box,
        half_extents: Vec3d::splat(0.5),
        density,
        ..Default::default()
    }
}

fn step(w: &mut EcsWorld, bridge: &mut PhysicsBridge3D) {
    bridge.sync_from_world(w);
    bridge.apply_water_forces(DT);
    bridge.step(DT);
    bridge.write_back_into(w);
    w.propagate();
}

fn body_y(w: &EcsWorld, guid: Uuid) -> f64 {
    let e = w.entity_of(guid).expect("entity");
    w.world().get::<Transform>(e).unwrap().translation.y
}

/// Build the float-on-a-lake world. `lake_visible` is the only variable.
fn floating_world(lake_visible: bool) -> EcsWorld {
    let mut w = EcsWorld::new();
    let lake = w.spawn_with_guid(LAKE, "Lake", None);
    w.world_mut()
        .entity_mut(lake)
        .insert((still_lake(4.0), Transform::IDENTITY));
    w.set_visible(lake, lake_visible);

    let b = w.spawn_with_guid(BOX, "Crate", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, 7.0, 0.0);
    w.world_mut().entity_mut(b).insert((
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        },
        unit_box_collider(500.0),
        Buoyancy {
            density_kg_m3: 500.0,
            ..Default::default()
        },
        t,
    ));
    w.mark_dirty();
    w.propagate();
    w
}

/// The whole trace as raw bits: an epsilon comparison here would hide exactly the
/// drift this gate is for.
fn trace(w: &mut EcsWorld, steps: usize) -> Vec<u64> {
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    (0..steps)
        .map(|_| {
            step(w, &mut bridge);
            body_y(w, BOX).to_bits()
        })
        .collect()
}

/// **The law, half one.** Hiding a lake changes the physics trace by exactly
/// nothing.
#[test]
fn a_hidden_lake_floats_a_boat_bit_identically() {
    let mut visible = floating_world(true);
    let mut hidden = floating_world(false);

    // ANTI-VACUITY (a): the entity really is hidden, all the way through
    // propagation — a test whose "hidden" flag never landed would pass trivially.
    let e = hidden.entity_of(LAKE).unwrap();
    assert!(!hidden.world().get::<Visibility>(e).unwrap().visible);
    assert!(!hidden.world().get::<ComputedVisibility>(e).unwrap().0);
    let ve = visible.entity_of(LAKE).unwrap();
    assert!(visible.world().get::<ComputedVisibility>(ve).unwrap().0);

    // Long enough for the crate to fall in, bob and settle: the claim is that it
    // FLOATS, not merely that two traces of a still-moving body agree.
    let a = trace(&mut visible, 900);
    let b = trace(&mut hidden, 900);
    assert_eq!(a, b, "hiding a lake changed the simulation");

    // ANTI-VACUITY (b): the lake is doing something. The box must have settled at
    // the waterline rather than fallen through — otherwise two identical traces
    // would prove only that neither had water.
    let rest = f64::from_bits(*b.last().unwrap());
    assert!(
        (rest - 4.0).abs() < 0.05,
        "a box of half the water's density should float with its centre on the \
         4 m waterline, not rest at {rest}"
    );

    // ANTI-VACUITY (c): the trace is not a constant. A world with NO water gives
    // a different answer, so the comparison above had something to catch.
    let mut dry = floating_world(true);
    {
        let e = dry.entity_of(LAKE).unwrap();
        dry.world_mut().entity_mut(e).remove::<WaterBody>();
        dry.mark_dirty();
        dry.propagate();
    }
    let c = trace(&mut dry, 900);
    assert_ne!(
        a, c,
        "removing the water changed nothing — the gate is blind"
    );
    assert!(
        f64::from_bits(*c.last().unwrap()) < 0.0,
        "it should have fallen"
    );
}

/// **The law, half two — the consistency evidence.** A hidden *collider* still
/// blocks, which is why water not honouring visibility is the engine being
/// consistent rather than water being special.
///
/// If this ever starts failing, the water assertion above is no longer the
/// engine's rule and the P20.4 decision has to be re-argued, not patched.
#[test]
fn a_hidden_collider_still_blocks_and_that_is_why_water_does_too() {
    let build = |visible: bool| {
        let mut w = EcsWorld::new();
        let g = w.spawn_with_guid(GROUND, "Ground", None);
        let mut gt = Transform::IDENTITY;
        gt.translation = Vec3d::new(0.0, 0.0, 0.0);
        w.world_mut().entity_mut(g).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(20.0, 0.5, 20.0),
                ..Default::default()
            },
            gt,
        ));
        w.set_visible(g, visible);

        let f = w.spawn_with_guid(FALLER, "Faller", None);
        let mut ft = Transform::IDENTITY;
        ft.translation = Vec3d::new(0.0, 5.0, 0.0);
        w.world_mut().entity_mut(f).insert((
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            },
            unit_box_collider(1000.0),
            ft,
        ));
        w.mark_dirty();
        w.propagate();
        w
    };

    let mut shown = build(true);
    let mut hidden = build(false);
    let e = hidden.entity_of(GROUND).unwrap();
    assert!(!hidden.world().get::<ComputedVisibility>(e).unwrap().0);

    let mut b1 = PhysicsBridge3D::new(GRAVITY);
    let mut b2 = PhysicsBridge3D::new(GRAVITY);
    for _ in 0..240 {
        step(&mut shown, &mut b1);
        step(&mut hidden, &mut b2);
    }
    let (a, b) = (body_y(&shown, FALLER), body_y(&hidden, FALLER));
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "hiding the ground changed where the box landed"
    );
    // …and it really did land ON the ground (top at y = 0.5, box half-height 0.5)
    // rather than falling forever, which is what makes the equality meaningful.
    assert!((a - 1.0).abs() < 0.05, "the faller did not land: {a}");
}

/// The hidden lake's **events** fire too — the same claim as the trace, made
/// about the discrete side of the water where a silent drop would not move a
/// single position bit until much later.
#[test]
fn a_hidden_lake_still_fires_its_water_events() {
    let mut w = floating_world(false);
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    let mut kinds: Vec<inf_physics::d3::WaterEventKind3D> = Vec::new();
    for _ in 0..240 {
        bridge.sync_from_world(&w);
        bridge.apply_water_forces(DT);
        bridge.step(DT);
        kinds.extend(bridge.drain_water_events().into_iter().map(|e| e.kind));
        bridge.write_back_into(&mut w);
        w.propagate();
    }
    assert!(
        kinds.contains(&inf_physics::d3::WaterEventKind3D::Enter),
        "a hidden lake swallowed its Enter event: {kinds:?}"
    );
    // A crate dropped from 3 m above the surface is doing more than 2 m/s when it
    // arrives, so the splash fires as well.
    assert!(
        kinds.contains(&inf_physics::d3::WaterEventKind3D::Splash),
        "{kinds:?}"
    );
}
