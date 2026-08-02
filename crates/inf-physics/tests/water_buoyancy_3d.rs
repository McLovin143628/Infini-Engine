//! P20.2: buoyancy, drag, water events and the swim latch, driven end-to-end
//! through the real [`PhysicsBridge3D`] over an [`EcsWorld`].
//!
//! `d3::water`'s unit tests pin the *model* (a solved force against an analytic
//! weight). These pin what actually happens when the solver integrates it: a box
//! of density 500 really does come to rest half submerged, a box of density 2000
//! really does sink, and a body outside the water is really untouched.
//!
//! Tolerances are sized consciously against the P20.1 height-query bound. Every
//! statics case here authors `wave_amplitude_m = 0`, which makes the Gerstner
//! displacement **identically zero** and the height query exact — so the residual
//! being absorbed is the solver's own settling, not the 6-iteration displacement
//! inversion's ≤3 mm typical / ~17 cm honest worst case. The wave case
//! (`a_floating_box_rides_a_swell`) is the one that has to budget for it, and it
//! budgets for the honest bound rather than the typical one.

use glam::{DVec2, DVec3};
use inf_ecs::components::{
    BodyKind3D, Buoyancy, CharacterController3D, Collider3D, ColliderShape3DKind, RigidBody3D,
    Transform, WaterBody,
};
use inf_ecs::math::Vec2d;
use inf_ecs::{EcsWorld, Vec3d};
use inf_physics::d3::WaterEventKind3D;
use inf_physics::PhysicsBridge3D;
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);

const LAKE: Uuid = Uuid::from_u128(0x2002_0001);
const BOX: Uuid = Uuid::from_u128(0x2002_0002);

/// A still lake (amplitude 0 ⇒ the height query is exact) at `level`, 200 m
/// square, centred on the origin.
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
        // The collider's density is what rapier turns into MASS. Keeping it equal
        // to the flotation density is what makes the model exactly Archimedes
        // rather than "as if the body had that density" — see `Buoyancy`'s docs.
        density,
        ..Default::default()
    }
}

fn spawn_water(w: &mut EcsWorld, guid: Uuid, body: WaterBody) {
    let e = w.spawn_with_guid(guid, "Water", None);
    w.world_mut()
        .entity_mut(e)
        .insert((body, Transform::IDENTITY));
    w.mark_dirty();
    w.propagate();
}

fn spawn_float(
    w: &mut EcsWorld,
    guid: Uuid,
    y: f64,
    density: f64,
    buoyancy: Buoyancy,
) -> inf_ecs::Entity {
    let e = w.spawn_with_guid(guid, "Float", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, y, 0.0);
    w.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        },
        unit_box_collider(density),
        buoyancy,
        t,
    ));
    w.mark_dirty();
    w.propagate();
    e
}

/// One fixed step, in the ORDER the runtime runs it: sync → water → step →
/// events → write-back. A test that stepped in a different order would be
/// testing a physics engine nobody ships.
fn step(w: &mut EcsWorld, bridge: &mut PhysicsBridge3D) -> Vec<inf_physics::d3::WaterEvent3D> {
    bridge.sync_from_world(w);
    bridge.apply_water_forces(DT);
    bridge.step(DT);
    let events = bridge.drain_water_events();
    bridge.write_back_into(w);
    w.propagate();
    events
}

fn settle(w: &mut EcsWorld, bridge: &mut PhysicsBridge3D, steps: usize) {
    for _ in 0..steps {
        step(w, bridge);
    }
}

fn body_y(w: &EcsWorld, guid: Uuid) -> f64 {
    let e = w.entity_of(guid).expect("entity");
    w.world().get::<Transform>(e).unwrap().translation.y
}

// ── statics ─────────────────────────────────────────────────────────────────

/// **The analytic assert.** A box whose flotation density is half the water's
/// settles with exactly half of itself under the surface — `level_m - 0`, because
/// a unit box's centre sits on the waterline when it is half submerged.
#[test]
fn a_box_of_half_density_floats_half_submerged() {
    let mut w = EcsWorld::new();
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_water(&mut w, LAKE, still_lake(10.0));
    spawn_float(&mut w, BOX, 12.0, 500.0, Buoyancy::of_density(500.0));

    settle(&mut w, &mut bridge, 900);

    let probe = bridge.water_probe(BOX).expect("in water");
    assert!(
        (probe.fraction - 0.5).abs() < 5e-3,
        "settled at {:.6} submerged, expected 0.5",
        probe.fraction
    );
    // And the pose says the same thing: the centre of a half-submerged unit box
    // is on the waterline.
    assert!(
        (body_y(&w, BOX) - 10.0).abs() < 5e-3,
        "settled at y = {:.6}, expected the 10 m waterline",
        body_y(&w, BOX)
    );
}

/// **Statics, in the strict sense.** Place each box at the pose the contract says
/// is its equilibrium and assert the solver leaves it there — a body that is
/// already in equilibrium must not drift, which is a sharper statement than
/// "it eventually converges" and cannot be satisfied by a long settle.
#[test]
fn a_body_placed_at_its_equilibrium_stays_there() {
    for density in [250.0, 500.0, 750.0] {
        let mut w = EcsWorld::new();
        let mut bridge = PhysicsBridge3D::new(GRAVITY);
        spawn_water(&mut w, LAKE, still_lake(0.0));
        let b = Buoyancy::of_density(density);
        // A unit box submerged to fraction `f` has its centre `(0.5 − f)` metres
        // above the waterline.
        let y = 0.5 - b.equilibrium_fraction();
        spawn_float(&mut w, BOX, y, density, b);
        settle(&mut w, &mut bridge, 600);
        let probe = bridge.water_probe(BOX).expect("in water");
        assert!(
            (probe.fraction - b.equilibrium_fraction()).abs() < 1e-3,
            "density {density}: drifted to {:.6}, contract says {:.6}",
            probe.fraction,
            b.equilibrium_fraction()
        );
        assert!(
            (body_y(&w, BOX) - y).abs() < 1e-3,
            "density {density}: drifted from y = {y} to {}",
            body_y(&w, BOX)
        );
    }
}

#[test]
fn a_box_denser_than_water_sinks() {
    let mut w = EcsWorld::new();
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_water(&mut w, LAKE, still_lake(0.0));
    spawn_float(&mut w, BOX, 0.0, 2000.0, Buoyancy::of_density(2000.0));

    let start = body_y(&w, BOX);
    settle(&mut w, &mut bridge, 300);
    let end = body_y(&w, BOX);
    assert!(
        end < start - 5.0,
        "a density-2000 box must sink: {start} → {end}"
    );
    // …and it is still *falling*, not hovering: buoyancy at full submersion is
    // half its weight, so it accelerates downward forever (there is no bed here).
    let probe = bridge.water_probe(BOX).unwrap();
    assert_eq!(probe.fraction, 1.0);
}

#[test]
fn a_neutrally_buoyant_box_hovers_where_it_is_put() {
    let mut w = EcsWorld::new();
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_water(&mut w, LAKE, still_lake(0.0));
    // Placed well under the surface so it is fully submerged from step one.
    spawn_float(&mut w, BOX, -20.0, 1000.0, Buoyancy::of_density(1000.0));

    let start = body_y(&w, BOX);
    settle(&mut w, &mut bridge, 600);
    let end = body_y(&w, BOX);
    assert!(
        (end - start).abs() < 1e-6,
        "neutral buoyancy must hover: {start} → {end}"
    );
    assert_eq!(bridge.water_probe(BOX).unwrap().fraction, 1.0);
}

/// A body nowhere near the water is not merely unaffected — it is never even
/// enumerated. The visible half of that is asserted here; the structural half is
/// `a_body_outside_every_bounded_body_is_untouched` in the index's own tests.
#[test]
fn a_body_outside_the_lake_falls_exactly_as_it_would_with_no_water() {
    let far = Uuid::from_u128(0x2002_0009);

    let mut wet = EcsWorld::new();
    let mut wet_bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_water(&mut wet, LAKE, still_lake(0.0));
    spawn_float(&mut wet, far, 5.0, 500.0, Buoyancy::of_density(500.0));
    // Move it far outside the lake's 100 m half-extent.
    {
        let e = wet.entity_of(far).unwrap();
        wet.world_mut()
            .get_mut::<Transform>(e)
            .unwrap()
            .translation
            .x = 5_000.0;
        wet.mark_dirty();
        wet.propagate();
    }

    let mut dry = EcsWorld::new();
    let mut dry_bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_float(&mut dry, far, 5.0, 500.0, Buoyancy::of_density(500.0));
    {
        let e = dry.entity_of(far).unwrap();
        dry.world_mut()
            .get_mut::<Transform>(e)
            .unwrap()
            .translation
            .x = 5_000.0;
        dry.mark_dirty();
        dry.propagate();
    }

    for _ in 0..240 {
        step(&mut wet, &mut wet_bridge);
        step(&mut dry, &mut dry_bridge);
    }
    assert_eq!(
        body_y(&wet, far).to_bits(),
        body_y(&dry, far).to_bits(),
        "a body outside the water must fall bit-identically to one in a level \
         with no water at all"
    );
}

/// The off-path guard, from the inside: winding every water knob on a level whose
/// only buoyant body is out of reach must not move a single bit — and the
/// anti-vacuity half, that a body IN the water really is moved by those knobs.
#[test]
fn water_off_path_is_bit_identical_and_the_guard_is_not_vacuous() {
    fn trace(place_in_water: bool, amplitude: f64) -> Vec<u64> {
        let mut w = EcsWorld::new();
        let mut bridge = PhysicsBridge3D::new(GRAVITY);
        spawn_water(
            &mut w,
            LAKE,
            WaterBody {
                wave_amplitude_m: amplitude,
                wave_length_m: 9.0,
                wave_steepness: 0.7,
                wave_count: 5,
                ..WaterBody::lake(0.0, Vec2d::splat(100.0))
            },
        );
        spawn_float(&mut w, BOX, 3.0, 500.0, Buoyancy::of_density(500.0));
        if !place_in_water {
            let e = w.entity_of(BOX).unwrap();
            w.world_mut().get_mut::<Transform>(e).unwrap().translation.x = 5_000.0;
            w.mark_dirty();
            w.propagate();
        }
        let mut out = Vec::new();
        for _ in 0..180 {
            step(&mut w, &mut bridge);
            out.push(body_y(&w, BOX).to_bits());
        }
        out
    }
    assert_eq!(
        trace(false, 0.0),
        trace(false, 1.4),
        "a body out of the water must not notice the sea state"
    );
    assert_ne!(
        trace(true, 0.0),
        trace(true, 1.4),
        "anti-vacuity: a body IN the water must notice it"
    );
}

/// The height query is the seam, and its documented worst case is ~17 cm. A box
/// riding a real swell has to stay within the wave it is riding plus that bound —
/// which is the tolerance written down rather than a number tuned until green.
#[test]
fn a_floating_box_rides_a_swell() {
    const AMPLITUDE: f64 = 0.8;
    /// The P20.1 honest worst-case residual of the 6-iteration Gerstner
    /// displacement inversion, metres. The measured residual on the steepest sea
    /// the P20.1 tests author is ≤ 3 mm; quoting *that* here would be tuning a
    /// tolerance to a measurement instead of to a guarantee.
    const QUERY_BOUND_M: f64 = 0.17;

    let mut w = EcsWorld::new();
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_water(
        &mut w,
        LAKE,
        WaterBody {
            wave_amplitude_m: AMPLITUDE,
            wave_length_m: 24.0,
            wave_steepness: 0.4,
            wave_count: 4,
            wind_from_weather: false,
            wind_x: 8.0,
            ..WaterBody::lake(0.0, Vec2d::splat(200.0))
        },
    );
    spawn_float(&mut w, BOX, 2.0, 500.0, Buoyancy::of_density(500.0));

    settle(&mut w, &mut bridge, 600);
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut moved = false;
    let mut last = body_y(&w, BOX);
    for _ in 0..600 {
        step(&mut w, &mut bridge);
        let y = body_y(&w, BOX);
        lo = lo.min(y);
        hi = hi.max(y);
        if (y - last).abs() > 1e-6 {
            moved = true;
        }
        last = y;
    }
    assert!(moved, "a box on a swell must actually move");
    // It rides the wave: never further from the still level than the wave's own
    // amplitude plus the height query's honest bound.
    let bound = AMPLITUDE + QUERY_BOUND_M;
    assert!(
        lo > -bound && hi < bound,
        "rode outside the wave it is on: [{lo:.4}, {hi:.4}] vs ±{bound:.4}"
    );
}

// ── events ──────────────────────────────────────────────────────────────────

#[test]
fn entering_and_leaving_the_water_fires_events_with_a_splash_threshold() {
    let mut w = EcsWorld::new();
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_water(&mut w, LAKE, still_lake(0.0));
    // Dropped from 6 m: it arrives at ~10 m/s, well past the 2 m/s splash bar.
    spawn_float(&mut w, BOX, 6.0, 500.0, Buoyancy::of_density(500.0));

    let mut events = Vec::new();
    for i in 0..1_200 {
        for ev in step(&mut w, &mut bridge) {
            assert_eq!(ev.body, BOX);
            assert_eq!(ev.water, LAKE, "the event names the water it crossed");
            events.push((i, ev.kind, ev.speed_m_s));
        }
    }
    assert!(events.len() >= 2, "{events:?}");
    assert_eq!(events[0].1, WaterEventKind3D::Enter);
    assert_eq!(
        events[1].1,
        WaterEventKind3D::Splash,
        "a 6 m drop lands at ~10 m/s, well past the 2 m/s splash bar"
    );
    assert!(events[1].2 >= 2.0, "splash speed {:?}", events[1].2);
    assert_eq!(
        events[0].2, events[1].2,
        "the splash reports the entry speed"
    );
    // A box that plunges deep enough really does bounce back out — that is
    // physics, not chatter — but the ringing must DIE. Nothing fires once it has
    // settled.
    assert!(
        events.iter().all(|(i, _, _)| *i < 900),
        "still crossing the surface after 15 s: {events:?}"
    );
}

#[test]
fn a_gentle_entry_does_not_splash() {
    let mut w = EcsWorld::new();
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_water(&mut w, LAKE, still_lake(0.0));
    // Released from 1 cm above the waterline: it crosses at well under 2 m/s.
    spawn_float(&mut w, BOX, 0.01, 500.0, Buoyancy::of_density(500.0));

    let mut kinds = Vec::new();
    for _ in 0..600 {
        for ev in step(&mut w, &mut bridge) {
            kinds.push(ev.kind);
        }
    }
    assert_eq!(kinds, vec![WaterEventKind3D::Enter], "{kinds:?}");
}

/// A body bobbing at the waterline must not chatter. The hysteresis band is 5 %
/// of the body's own height, so a settled float — which sits with its underside
/// permanently wet — fires exactly one `Enter` and never an `Exit`.
#[test]
fn a_settled_float_does_not_chatter() {
    let mut w = EcsWorld::new();
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_water(
        &mut w,
        LAKE,
        WaterBody {
            wave_amplitude_m: 0.25,
            wave_length_m: 6.0,
            wind_from_weather: false,
            wind_x: 4.0,
            ..WaterBody::lake(0.0, Vec2d::splat(100.0))
        },
    );
    // Placed at its equilibrium so the trace is about the waves, not about a drop.
    spawn_float(&mut w, BOX, 0.0, 500.0, Buoyancy::of_density(500.0));

    // One entry as it meets the water, then silence — for twenty seconds of chop.
    let mut early = Vec::new();
    for _ in 0..300 {
        early.extend(step(&mut w, &mut bridge).into_iter().map(|e| e.kind));
    }
    assert_eq!(early, vec![WaterEventKind3D::Enter], "{early:?}");
    let mut late = Vec::new();
    for _ in 0..1_200 {
        late.extend(step(&mut w, &mut bridge).into_iter().map(|e| e.kind));
    }
    assert!(
        late.is_empty(),
        "a settled float must not chatter as waves pass: {late:?}"
    );
}

#[test]
fn deleting_the_water_makes_every_body_in_it_exit() {
    let mut w = EcsWorld::new();
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_water(&mut w, LAKE, still_lake(0.0));
    spawn_float(&mut w, BOX, 0.0, 500.0, Buoyancy::of_density(500.0));
    settle(&mut w, &mut bridge, 120);
    assert!(bridge.water_probe(BOX).is_some());

    let e = w.entity_of(LAKE).unwrap();
    w.despawn(e);
    let events = step(&mut w, &mut bridge);
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0].kind, WaterEventKind3D::Exit);
    assert_eq!(
        events[0].water, LAKE,
        "an exit names the water it left, even after that water is gone"
    );
    assert!(bridge.water().is_empty());
}

// ── swim ────────────────────────────────────────────────────────────────────

#[test]
fn a_character_flips_into_swim_mode_when_it_is_deep_enough() {
    let hero = Uuid::from_u128(0x2002_0010);
    let mut w = EcsWorld::new();
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    spawn_water(&mut w, LAKE, still_lake(0.0));

    let e = w.spawn_with_guid(hero, "Hero", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, 5.0, 0.0);
    w.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.3, 0.6, 0.3),
            radius: 0.3,
            ..Default::default()
        },
        CharacterController3D::default(),
        t,
    ));
    w.mark_dirty();
    w.propagate();

    // Standing well clear of the water: not swimming, and the motion is untouched.
    bridge.sync_from_world(&w);
    assert!(!bridge.update_swim(hero));
    let dry = DVec3::new(0.1, -0.5, 0.0);
    assert_eq!(bridge.apply_swim_motion(hero, dry, DT), dry);

    // Sink it: a 0.9 m half-height capsule at y = -0.5 is ~78 % under.
    let e = w.entity_of(hero).unwrap();
    w.world_mut().get_mut::<Transform>(e).unwrap().translation.y = -0.5;
    w.mark_dirty();
    w.propagate();
    bridge.sync_from_world(&w);
    let probe = bridge.water_probe(hero).expect("in water");
    assert!(probe.fraction > 0.6, "fraction {}", probe.fraction);
    assert!(bridge.update_swim(hero), "deep enough must swim");

    // Swimming: accumulated free-fall is discarded, horizontal speed is capped.
    let sprint = DVec3::new(6.0 * DT, -20.0 * DT, 0.0);
    let swum = bridge.apply_swim_motion(hero, sprint, DT);
    assert!(swum.y > sprint.y, "gravity is replaced, not honoured");
    assert!(
        DVec2::new(swum.x, swum.z).length() / DT <= 2.5 + 1e-9,
        "swim speed cap"
    );

    // Wading back to knee depth drops it out of swim mode.
    let e = w.entity_of(hero).unwrap();
    w.world_mut().get_mut::<Transform>(e).unwrap().translation.y = 0.6;
    w.mark_dirty();
    w.propagate();
    bridge.sync_from_world(&w);
    assert!(!bridge.update_swim(hero), "wading is not swimming");
}

// ── determinism ─────────────────────────────────────────────────────────────

/// The replay gate at the facade level: a floating stack over a wavy sea traces
/// bit-identically across two runs. `runtime/inf-player/tests/water_physics.rs`
/// extends it to pool sizes and to PIE == shipping.
#[test]
fn a_floating_stack_is_bit_identical_across_two_runs() {
    fn run() -> Vec<u64> {
        let mut w = EcsWorld::new();
        let mut bridge = PhysicsBridge3D::new(GRAVITY);
        spawn_water(
            &mut w,
            LAKE,
            WaterBody {
                wave_amplitude_m: 0.6,
                wave_length_m: 18.0,
                wave_steepness: 0.5,
                wave_count: 5,
                wind_from_weather: false,
                wind_x: 7.0,
                wind_z: -2.0,
                ..WaterBody::lake(0.0, Vec2d::splat(120.0))
            },
        );
        for i in 0..8u128 {
            let guid = Uuid::from_u128(0x2002_0100 + i);
            let e = w.spawn_with_guid(guid, "Crate", None);
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(
                (i as f64 - 3.5) * 0.7,
                4.0 + i as f64 * 1.3,
                (i as f64 % 3.0) * 0.6,
            );
            w.world_mut().entity_mut(e).insert((
                RigidBody3D {
                    kind: BodyKind3D::Dynamic,
                    ..Default::default()
                },
                unit_box_collider(400.0 + i as f64 * 60.0),
                Buoyancy::of_density(400.0 + i as f64 * 60.0),
                t,
            ));
        }
        w.mark_dirty();
        w.propagate();

        let mut trace = Vec::new();
        for _ in 0..400 {
            for ev in step(&mut w, &mut bridge) {
                trace.push(ev.body.as_u128() as u64);
                trace.push(ev.kind as u64);
                trace.push(ev.speed_m_s.to_bits());
            }
            for i in 0..8u128 {
                let guid = Uuid::from_u128(0x2002_0100 + i);
                let e = w.entity_of(guid).unwrap();
                let t = w.world().get::<Transform>(e).unwrap();
                trace.push(t.translation.x.to_bits());
                trace.push(t.translation.y.to_bits());
                trace.push(t.translation.z.to_bits());
            }
        }
        trace
    }
    let a = run();
    let b = run();
    assert_eq!(a, b, "the floating stack diverged across two runs");
    assert!(!a.is_empty());
}

// ── scale / the off-path discipline ─────────────────────────────────────────

/// **The town statement.** A furnished town is thousands of static colliders and
/// zero `Buoyancy` components; adding a lake to it must cost nothing.
///
/// Asserted the only way a claim like that can be honestly asserted from the
/// outside: the same town traced with and without the lake is **bit-identical**,
/// so the water pass demonstrably touched nothing. The structural reason is that
/// [`PhysicsBridge3D::apply_water_forces`] iterates the *buoyant* set — not the
/// rigid bodies — and that set is empty here, so the whole pass is one
/// `is_empty()` branch per step regardless of how many colliders the town has.
#[test]
fn a_town_of_colliders_with_a_lake_in_it_costs_nothing() {
    const COLLIDERS: u128 = 3_000;

    fn town(with_lake: bool) -> (EcsWorld, PhysicsBridge3D) {
        let mut w = EcsWorld::new();
        if with_lake {
            spawn_water(&mut w, LAKE, still_lake(0.0));
        }
        // The static furniture: a grid of boxes, none of them buoyant.
        for i in 0..COLLIDERS {
            let e = w.spawn_with_guid(Uuid::from_u128(0x2002_5000 + i), "Wall", None);
            let x = (i % 60) as f64 * 3.0 - 90.0;
            let z = (i / 60) as f64 * 3.0 - 90.0;
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(x, 0.5, z);
            w.world_mut().entity_mut(e).insert((
                RigidBody3D {
                    kind: BodyKind3D::Static,
                    ..Default::default()
                },
                Collider3D {
                    shape_kind: ColliderShape3DKind::Box,
                    half_extents: Vec3d::splat(0.5),
                    ..Default::default()
                },
                t,
            ));
        }
        // One dynamic body, so the trace is not trivially constant.
        let e = w.spawn_with_guid(BOX, "Ball", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, 6.0, 0.0);
        w.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            },
            unit_box_collider(500.0),
            t,
        ));
        w.mark_dirty();
        w.propagate();
        let bridge = PhysicsBridge3D::new(GRAVITY);
        (w, bridge)
    }

    let trace = |with_lake: bool| {
        let (mut w, mut bridge) = town(with_lake);
        let mut out = Vec::new();
        for _ in 0..120 {
            step(&mut w, &mut bridge);
            out.push(body_y(&w, BOX).to_bits());
        }
        (out, bridge)
    };

    let (dry, dry_bridge) = trace(false);
    let (wet, wet_bridge) = trace(true);
    assert_eq!(
        dry, wet,
        "a lake changed a town whose bodies have no Buoyancy component"
    );
    // Anti-vacuity: the lake really is in the index, and the town really is big.
    assert!(dry_bridge.water().is_empty());
    assert_eq!(wet_bridge.water().len(), 1, "the lake was never indexed");
    assert!(dry.iter().any(|y| *y != dry[0]), "nothing moved at all");
    // …and it really would have floated, had it been asked to.
    let mut w2 = EcsWorld::new();
    let mut b2 = PhysicsBridge3D::new(GRAVITY);
    spawn_water(&mut w2, LAKE, still_lake(0.0));
    spawn_float(&mut w2, BOX, 6.0, 500.0, Buoyancy::of_density(500.0));
    settle(&mut w2, &mut b2, 900);
    assert!(
        body_y(&w2, BOX) > -1.0,
        "the same body WITH a Buoyancy component must float"
    );
}
