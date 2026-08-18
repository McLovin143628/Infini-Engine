//! **P29.3: the character movement fixed step**, driven end-to-end through the
//! real [`PhysicsBridge3D`] over an [`EcsWorld`].
//!
//! `inf_ecs::movement`'s own tests pin the *rules* — the gait ladder, the
//! quadrant buffer, the landing thresholds — as functions of numbers. These pin
//! what happens when those rules meet a world: whether the capsule actually
//! shrinks, whether the character actually climbs the step, whether standing up
//! under a table is actually refused.
//!
//! **Every arm here asserts the WORLD.** A refusal is checked by reading the
//! capsule's half-height and the transform back out of the ECS, not by reading
//! the function's own report — the P21 law, met here because a movement step
//! that returned "refused" while resizing anyway would pass the easy version of
//! all of these.

use glam::DVec3;
use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind, Gait,
    LandingKind, MovementMode, MovementRefusal, RigidBody3D, Transform, WaterBody,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::movement::MovementIntent;
use inf_ecs::{EcsWorld, Vec2d as EcsVec2d};
use inf_physics::d3::step_character_movement;
use inf_physics::PhysicsBridge3D;
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);

const HERO: Uuid = Uuid::from_u128(0x2903_0001);
const GROUND: Uuid = Uuid::from_u128(0x2903_0002);
const LAKE: Uuid = Uuid::from_u128(0x2903_0003);

/// The hero's capsule radius, and the half-heights the defaults give it.
const RADIUS: f64 = 0.3;

fn stand_half() -> f64 {
    CharacterMovement::default().stand_half_height_m
}
fn crouch_half() -> f64 {
    CharacterMovement::default().crouch_half_height_m
}

/// Spawn a static box collider at `centre` with `half_extents` — a floor, a
/// step, a ceiling; whatever the arm needs the world to contain.
fn spawn_block(w: &mut EcsWorld, guid: Uuid, centre: DVec3, half: DVec3) {
    let e = w.spawn_with_guid(guid, "Block", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::from_dvec3(centre);
    w.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::from_dvec3(half),
            ..Default::default()
        },
        t,
    ));
    w.mark_dirty();
    w.propagate();
}

/// Spawn the character: a kinematic capsule with a movement component.
///
/// `feet_y` is where the soles go; the transform is the capsule's CENTRE, which
/// is `half_height + radius` above them.
fn spawn_hero(w: &mut EcsWorld, feet_y: f64, x: f64, z: f64) {
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let e = w.spawn_with_guid(HERO, "Hero", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(x, feet_y + cm.stand_half_height_m + RADIUS, z);
    w.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(RADIUS, cm.stand_half_height_m, RADIUS),
            radius: RADIUS,
            ..Default::default()
        },
        CharacterController3D::default(),
        cm,
        t,
    ));
    w.mark_dirty();
    w.propagate();
}

/// One fixed step, in the order both hosts run it: bridge sync, then the
/// movement door. A test that stepped in a different order would be testing a
/// sequence nobody ships.
fn step(w: &mut EcsWorld, b: &mut PhysicsBridge3D, intent: &MovementIntent) {
    b.sync_from_world(w);
    inf_ecs::movement::apply_intent(w, intent);
    step_character_movement(w, b, DT);
}

fn hero(w: &EcsWorld) -> CharacterMovement {
    let e = w.entity_of(HERO).expect("the hero exists");
    w.world()
        .get::<CharacterMovement>(e)
        .expect("with a movement component")
        .clone()
}

fn hero_capsule_half(w: &EcsWorld) -> f64 {
    let e = w.entity_of(HERO).unwrap();
    w.world().get::<Collider3D>(e).unwrap().half_extents.y
}

fn hero_pos(w: &EcsWorld) -> DVec3 {
    let e = w.entity_of(HERO).unwrap();
    w.world()
        .get::<Transform>(e)
        .unwrap()
        .translation
        .to_dvec3()
}

/// The soles: the capsule's centre minus its half-height and radius.
fn hero_feet(w: &EcsWorld) -> f64 {
    hero_pos(w).y - hero_capsule_half(w) - RADIUS
}

fn walk_forward() -> MovementIntent {
    MovementIntent {
        move_input: EcsVec2d::new(0.0, 1.0),
        ..Default::default()
    }
}

fn idle() -> MovementIntent {
    MovementIntent::default()
}

fn press_crouch() -> MovementIntent {
    MovementIntent {
        crouch: true,
        ..Default::default()
    }
}

fn press_jump() -> MovementIntent {
    MovementIntent {
        jump: true,
        ..Default::default()
    }
}

// ── stairs ──────────────────────────────────────────────────────────────────

/// **THE stairs arm.** `.autostep(` appeared exactly twice in this repository
/// before P29.3 — at its own definition in `d2` and in `d3` — and was called
/// from no production code at all, so rapier's default (`None`) applied and a
/// character walked into a step instead of up it. §13 names this as the one
/// thing the roadmap knew was visibly broken.
///
/// It measures the world: after walking at a four-step flight, is the character
/// standing on top of it? The control is the same flight with the step height
/// tuned to zero, which is what the engine did before this wave — and it must
/// NOT climb, or this arm is measuring the ability to walk rather than the
/// ability to climb.
#[test]
fn a_character_climbs_a_flight_of_stairs_and_could_not_before() {
    fn run(step_height_m: f64) -> (f64, f64) {
        let mut w = EcsWorld::new();
        let mut b = PhysicsBridge3D::new(GRAVITY);
        // A floor up to z = 1.3, four 0.2 m risers, then a landing at 0.8 m.
        spawn_block(
            &mut w,
            GROUND,
            DVec3::new(0.0, -0.5, -9.0),
            DVec3::new(20.0, 0.5, 10.3),
        );
        for i in 0..4 {
            let h = 0.2 * (i + 1) as f64;
            spawn_block(
                &mut w,
                Uuid::from_u128(0x2903_1000 + i as u128),
                DVec3::new(0.0, h / 2.0, 1.5 + 0.4 * i as f64),
                DVec3::new(4.0, h / 2.0, 0.2),
            );
        }
        spawn_block(
            &mut w,
            Uuid::from_u128(0x2903_1100),
            DVec3::new(0.0, 0.4, 12.9),
            DVec3::new(4.0, 0.4, 10.0),
        );
        spawn_hero(&mut w, 0.0, 0.0, 0.0);
        {
            let e = w.entity_of(HERO).unwrap();
            let mut cm = w.world_mut().get_mut::<CharacterMovement>(e).unwrap();
            cm.step_height_m = step_height_m;
        }
        // A WALK, so the measurement is "did it climb" and not "how fast".
        let walk = MovementIntent {
            move_input: EcsVec2d::new(0.0, 1.0),
            walk: true,
            ..Default::default()
        };
        for _ in 0..300 {
            step(&mut w, &mut b, &walk);
        }
        (hero_feet(&w), hero_pos(&w).z)
    }

    let (climbed, climbed_z) = run(0.45);
    let (flat, flat_z) = run(0.0);
    assert!(
        climbed > 0.7,
        "with autostep the character should be standing on the 0.8 m landing; \
         its feet are at {climbed} (z = {climbed_z})"
    );
    assert!(
        climbed_z > 3.0,
        "and it should be PAST the flight: z = {climbed_z}"
    );
    assert!(
        flat < 0.05 && flat_z < 1.4,
        "with autostep OFF -- the engine's state before P29.3 -- it must be stopped dead at the \
         first 20 cm riser, and it reached feet {flat} at z = {flat_z}. If this control climbs, \
         the arm above is measuring walking."
    );
}

// ── clearance ───────────────────────────────────────────────────────────────

/// **Standing up under a table refuses** — the catalogue's own example, and the
/// day-one consumer of the shape cast.
///
/// Asserted on the WORLD: the capsule's half-height and the transform, read back
/// out of the ECS. A step that reported a refusal while resizing anyway would
/// pass any assertion made on the return value.
#[test]
fn standing_up_under_a_ceiling_is_refused_and_the_capsule_does_not_grow() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(20.0, 0.5, 20.0),
    );
    spawn_hero(&mut w, 0.0, 0.0, 0.0);

    // Crouch first, on open ground.
    step(&mut w, &mut b, &press_crouch());
    assert_eq!(hero(&w).mode, MovementMode::Crouch);
    assert!(
        (hero_capsule_half(&w) - crouch_half()).abs() < 1e-9,
        "the capsule shrank: {}",
        hero_capsule_half(&w)
    );
    assert!(
        hero_feet(&w).abs() < 0.05,
        "and the FEET stayed planted: {}",
        hero_feet(&w)
    );

    // Now put a ceiling just above the crouched capsule: standing would need
    // 2 * (0.6 - 0.3) = 0.6 m of headroom and there is 0.2 m.
    let crouched_top = hero_pos(&w).y + crouch_half() + RADIUS;
    spawn_block(
        &mut w,
        Uuid::from_u128(0x2903_2000),
        DVec3::new(0.0, crouched_top + 0.2 + 0.25, 0.0),
        DVec3::new(3.0, 0.25, 3.0),
    );
    b.sync_from_world(&w);

    let before_half = hero_capsule_half(&w);
    let before_pos = hero_pos(&w);
    step(&mut w, &mut b, &press_crouch());

    let after = hero(&w);
    assert_eq!(after.mode, MovementMode::Crouch, "still crouched");
    assert_eq!(
        after.runtime.refusal,
        MovementRefusal::NoOverheadClearance,
        "and it knows why"
    );
    assert!(after.runtime.refusals >= 1, "and it counted the refusal");
    assert!(
        (hero_capsule_half(&w) - before_half).abs() < 1e-12,
        "THE WORLD: the capsule must not have grown -- {} -> {}",
        before_half,
        hero_capsule_half(&w)
    );
    assert!(
        (hero_pos(&w).y - before_pos.y).abs() < 0.02,
        "THE WORLD: and the body must not have risen -- {} -> {}",
        before_pos.y,
        hero_pos(&w).y
    );

    // The control, and it is the load-bearing half: step OUT from under the
    // ceiling and the same press stands up. Without this the arm above is
    // satisfied by a crouch that can never be left.
    for _ in 0..300 {
        step(
            &mut w,
            &mut b,
            &MovementIntent {
                move_input: EcsVec2d::new(0.0, 1.0),
                ..Default::default()
            },
        );
    }
    assert!(
        hero_pos(&w).z > 3.5,
        "the character walked clear of the 3 m ceiling: z = {}",
        hero_pos(&w).z
    );
    step(&mut w, &mut b, &press_crouch());
    assert_eq!(hero(&w).mode, MovementMode::Grounded, "and now it stands");
    assert!(
        (hero_capsule_half(&w) - stand_half()).abs() < 1e-9,
        "with the tall capsule back: {}",
        hero_capsule_half(&w)
    );
    assert!(
        hero_feet(&w).abs() < 0.05,
        "and the feet still on the floor: {}",
        hero_feet(&w)
    );
}

// ── the landing classifier ──────────────────────────────────────────────────

/// The classifier is keyed to **impact speed**, so the arm drops the character
/// from three heights either side of the two thresholds and reads back what it
/// decided — and the decision is a *component field*, not a return value.
#[test]
fn the_landing_classifier_reads_the_speed_it_actually_landed_at() {
    fn drop_from(height: f64, with_input: bool) -> (LandingKind, f64) {
        let mut w = EcsWorld::new();
        let mut b = PhysicsBridge3D::new(GRAVITY);
        spawn_block(
            &mut w,
            GROUND,
            DVec3::new(0.0, -0.5, 0.0),
            DVec3::new(40.0, 0.5, 40.0),
        );
        spawn_hero(&mut w, height, 0.0, 0.0);
        // Nothing under it but air: the first step walks off into a fall.
        let intent = if with_input { walk_forward() } else { idle() };
        for _ in 0..600 {
            step(&mut w, &mut b, &intent);
            let h = hero(&w);
            if h.runtime.landing != LandingKind::None && h.runtime.grounded {
                return (h.runtime.landing, h.runtime.land_impact_mps);
            }
        }
        (LandingKind::None, 0.0)
    }

    // v = sqrt(2 g h): 1 m -> 4.4 m/s (soft), 4 m -> 8.9 (hard/roll band),
    // 8 m -> 12.5 (past the ragdoll threshold).
    let (soft, v_soft) = drop_from(1.0, false);
    assert_eq!(soft, LandingKind::Soft, "1 m is a soft landing ({v_soft})");
    assert!((3.5..5.5).contains(&v_soft), "impact {v_soft}");

    let (hard, v_hard) = drop_from(4.0, false);
    assert_eq!(
        hard,
        LandingKind::Hard,
        "4 m with no input plants ({v_hard})"
    );
    assert!((7.0..10.0).contains(&v_hard), "impact {v_hard}");

    let (roll, _) = drop_from(4.0, true);
    assert_eq!(
        roll,
        LandingKind::Roll,
        "the same fall WITH movement input break-falls"
    );

    let (ragdoll, v_rag) = drop_from(9.0, false);
    assert_eq!(ragdoll, LandingKind::Ragdoll, "9 m is past 10 m/s");
    assert!(v_rag >= 10.0, "impact {v_rag}");
}

/// Falling is bounded by the terminal velocity, and air control is really
/// reduced in a controlled fall — the two halves of the catalogue's fall row
/// that are numbers rather than states.
#[test]
fn a_fall_is_bounded_and_a_controlled_fall_steers_less_than_a_free_one() {
    fn fall(free: bool, terminal: f64) -> (f64, f64) {
        let mut w = EcsWorld::new();
        let mut b = PhysicsBridge3D::new(GRAVITY);
        spawn_hero(&mut w, 400.0, 0.0, 0.0);
        {
            let e = w.entity_of(HERO).unwrap();
            let mut cm = w.world_mut().get_mut::<CharacterMovement>(e).unwrap();
            cm.terminal_velocity_mps = terminal;
            cm.mode = if free {
                MovementMode::FallFree
            } else {
                MovementMode::FallControlled
            };
        }
        let mut after_one_second = 0.0;
        for i in 0..600 {
            step(&mut w, &mut b, &walk_forward());
            if i == 59 {
                after_one_second = hero(&w).runtime.velocity.z;
            }
        }
        (hero(&w).runtime.velocity.y, after_one_second)
    }

    let (vy, free_vz) = fall(true, 20.0);
    assert!(
        (vy + 20.0).abs() < 1e-6,
        "ten seconds of free fall is clamped to the terminal velocity: {vy}"
    );
    let (_, controlled_vz) = fall(false, 20.0);
    assert!(
        free_vz > controlled_vz * 1.5,
        "after one second of steering, a free fall must have gained appreciably more \
         horizontal speed than a controlled one: {free_vz} vs {controlled_vz}"
    );
    assert!(controlled_vz > 0.0, "but a controlled fall still steers");
}

// ── modes ───────────────────────────────────────────────────────────────────

/// Entering a mode whose mechanics belong to a later sub-phase is a **typed
/// refusal**, not a stub and not a panic. The world is unchanged and the
/// character knows why.
#[test]
fn a_mode_owned_by_a_later_wave_refuses_and_changes_nothing() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(20.0, 0.5, 20.0),
    );
    spawn_hero(&mut w, 0.0, 0.0, 0.0);
    step(&mut w, &mut b, &idle());

    for deferred in [
        MovementMode::Mantle,
        MovementMode::Ragdoll,
        MovementMode::Driving,
        MovementMode::Flying,
    ] {
        let before_half = hero_capsule_half(&w);
        let before_pos = hero_pos(&w);
        // Ask through the model door the step itself uses.
        let verdict = inf_ecs::movement::request_mode(MovementMode::Grounded, deferred, true, true);
        assert_eq!(verdict.mode, MovementMode::Grounded, "{deferred:?}");
        assert_eq!(verdict.refusal, MovementRefusal::ModeNotYetImplemented);
        step(&mut w, &mut b, &idle());
        assert_eq!(hero(&w).mode, MovementMode::Grounded);
        assert!((hero_capsule_half(&w) - before_half).abs() < 1e-12);
        assert!((hero_pos(&w).y - before_pos.y).abs() < 0.02);
    }
}

/// Crouch, prone and slide each resize the capsule to their own height and each
/// leave the feet where they were — the three-way version of the resize claim,
/// asserted on the collider rather than on the mode.
#[test]
fn each_stance_resizes_the_capsule_and_plants_the_feet() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(20.0, 0.5, 20.0),
    );
    spawn_hero(&mut w, 0.0, 0.0, 0.0);
    let cm = CharacterMovement::default();

    step(&mut w, &mut b, &idle());
    assert!((hero_capsule_half(&w) - cm.stand_half_height_m).abs() < 1e-9);

    step(&mut w, &mut b, &press_crouch());
    assert_eq!(hero(&w).mode, MovementMode::Crouch);
    assert!((hero_capsule_half(&w) - cm.crouch_half_height_m).abs() < 1e-9);
    assert!(hero_feet(&w).abs() < 0.05, "feet at {}", hero_feet(&w));

    step(
        &mut w,
        &mut b,
        &MovementIntent {
            prone: true,
            ..Default::default()
        },
    );
    assert_eq!(hero(&w).mode, MovementMode::Prone);
    assert!((hero_capsule_half(&w) - cm.prone_half_height_m).abs() < 1e-9);
    assert!(hero_feet(&w).abs() < 0.05, "feet at {}", hero_feet(&w));

    // Prone -> crouch is the same press again.
    step(
        &mut w,
        &mut b,
        &MovementIntent {
            prone: true,
            ..Default::default()
        },
    );
    assert_eq!(hero(&w).mode, MovementMode::Crouch);
    assert!((hero_capsule_half(&w) - cm.crouch_half_height_m).abs() < 1e-9);
}

/// A slide is entered from **sprint + crouch** and only from there, runs on the
/// crouch capsule, and exits into a crouch when it has run out of speed.
#[test]
fn a_slide_needs_sprint_speed_and_ends_in_a_crouch() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(40.0, 0.5, 40.0),
    );
    spawn_hero(&mut w, 0.0, 0.0, 0.0);

    // Crouch pressed from a standstill with sprint held is NOT a slide: the
    // entry condition is speed, which is what makes it a slide rather than a
    // second crouch button.
    step(
        &mut w,
        &mut b,
        &MovementIntent {
            sprint: true,
            crouch: true,
            ..Default::default()
        },
    );
    assert_eq!(
        hero(&w).mode,
        MovementMode::Crouch,
        "standing still, crouch is a crouch"
    );

    // Stand back up, get to sprint speed, then press crouch.
    step(&mut w, &mut b, &press_crouch());
    assert_eq!(hero(&w).mode, MovementMode::Grounded);
    let sprint = MovementIntent {
        move_input: EcsVec2d::new(0.0, 1.0),
        sprint: true,
        ..Default::default()
    };
    for _ in 0..180 {
        step(&mut w, &mut b, &sprint);
    }
    let speed = hero(&w).runtime.velocity.z;
    assert!(speed > 4.0, "the hero reached sprint speed: {speed}");

    step(
        &mut w,
        &mut b,
        &MovementIntent {
            move_input: EcsVec2d::new(0.0, 1.0),
            sprint: true,
            crouch: true,
            ..Default::default()
        },
    );
    assert_eq!(hero(&w).mode, MovementMode::Slide, "sprint + crouch slides");
    assert!(
        (hero_capsule_half(&w) - crouch_half()).abs() < 1e-9,
        "on the crouch capsule"
    );

    // It runs out and lands in a crouch, not in a stand.
    let mut ended = MovementMode::Slide;
    for _ in 0..600 {
        step(&mut w, &mut b, &idle());
        ended = hero(&w).mode;
        if ended != MovementMode::Slide {
            break;
        }
    }
    assert_eq!(ended, MovementMode::Crouch, "a slide exits into a crouch");
}

// ── swim ────────────────────────────────────────────────────────────────────

/// **The swim latch is absorbed, not re-implemented.**
///
/// The mode reads P20's latch — the same `update_swim` / `apply_swim_motion` pair
/// `physics3d.move_and_slide` has called since P20.2 — so a character deep enough
/// to swim is in `MovementMode::SwimSurface` rather than in `Grounded` with a
/// special case bolted on. The arm asserts the two agree in BOTH directions,
/// which is what "one door, two readers" has to mean.
#[test]
fn the_swim_mode_is_the_p20_latch_and_agrees_with_it_both_ways() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    // A still lake (amplitude 0 makes the height query exact) with its surface
    // at y = 6, and a floor far below.
    let lake = w.spawn_with_guid(LAKE, "Lake", None);
    w.world_mut().entity_mut(lake).insert((
        WaterBody {
            wave_amplitude_m: 0.0,
            ..WaterBody::lake(6.0, Vec2d::splat(100.0))
        },
        Transform::IDENTITY,
    ));
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(40.0, 0.5, 40.0),
    );
    // Deep under the surface.
    spawn_hero(&mut w, 1.0, 0.0, 0.0);
    w.mark_dirty();
    w.propagate();

    step(&mut w, &mut b, &idle());
    assert!(
        b.is_swimming(HERO),
        "P20's latch says the hero is swimming (the control for everything below)"
    );
    assert!(
        hero(&w).mode.is_swimming(),
        "and the MODE agrees: {:?}",
        hero(&w).mode
    );

    // Swim up and out. The surface is at 6 and the floor at 0, so rising past
    // the exit band must drop BOTH the latch and the mode, together.
    let up = MovementIntent {
        vertical: 1.0,
        ..Default::default()
    };
    let mut left_the_water = false;
    for _ in 0..3000 {
        step(&mut w, &mut b, &up);
        assert_eq!(
            b.is_swimming(HERO),
            hero(&w).mode.is_swimming(),
            "the latch and the mode must never disagree: latch = {}, mode = {:?}, y = {}",
            b.is_swimming(HERO),
            hero(&w).mode,
            hero_pos(&w).y
        );
        if !b.is_swimming(HERO) {
            left_the_water = true;
            break;
        }
    }
    assert!(
        left_the_water,
        "the swimmer must eventually leave the water, or the agreement above is vacuous"
    );
    assert!(
        !hero(&w).mode.is_swimming(),
        "and the mode left with it: {:?}",
        hero(&w).mode
    );
}

// ── determinism ─────────────────────────────────────────────────────────────

/// The new fixed-step outputs are sim state, so two worlds given the same input
/// must produce the same bytes — the §8 replay guarantee, extended to movement.
#[test]
fn two_identical_worlds_move_byte_for_byte() {
    fn run() -> Vec<u8> {
        let mut w = EcsWorld::new();
        let mut b = PhysicsBridge3D::new(GRAVITY);
        spawn_block(
            &mut w,
            GROUND,
            DVec3::new(0.0, -0.5, 0.0),
            DVec3::new(40.0, 0.5, 40.0),
        );
        spawn_block(
            &mut w,
            Uuid::from_u128(0x2903_3000),
            DVec3::new(0.0, 0.1, 3.0),
            DVec3::new(4.0, 0.1, 0.3),
        );
        spawn_hero(&mut w, 0.0, 0.0, 0.0);
        let mut bytes = Vec::new();
        for i in 0..300 {
            let intent = MovementIntent {
                move_input: EcsVec2d::new(
                    if i % 97 < 20 { 0.6 } else { 0.0 },
                    if i % 53 < 40 { 1.0 } else { -0.4 },
                ),
                look_yaw_dps: if i % 31 < 10 { 90.0 } else { 0.0 },
                sprint: i % 7 == 0,
                crouch: i == 120 || i == 180,
                jump: i == 60,
                ..Default::default()
            };
            step(&mut w, &mut b, &intent);
            let h = hero(&w);
            let p = hero_pos(&w);
            for v in [
                p.x,
                p.y,
                p.z,
                h.runtime.velocity.x,
                h.runtime.velocity.y,
                h.runtime.velocity.z,
                h.runtime.aim_yaw_deg,
                h.runtime.aim_yaw_rate_dps,
                h.runtime.mapped_speed,
                h.runtime.gait_scalar,
                h.runtime.stride_blend,
                h.runtime.walk_run_blend,
                h.runtime.relative_accel.x,
                h.runtime.relative_accel.y,
            ] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes.push(h.mode as u8);
            bytes.push(h.runtime.actual_gait as u8);
            bytes.push(h.runtime.direction as u8);
            bytes.push(h.runtime.landing as u8);
        }
        bytes
    }
    let a = run();
    let b = run();
    assert_eq!(a.len(), 300 * (14 * 8 + 4));
    assert_eq!(a, b, "two runs of the same movement diverged");
    // Anti-vacuity: the trace must contain more than one mode and more than one
    // gait, or "identical" is a claim about a character standing still.
    let modes: std::collections::BTreeSet<u8> = a.chunks(14 * 8 + 4).map(|c| c[14 * 8]).collect();
    assert!(
        modes.len() >= 3,
        "the trace visited only {} modes -- it is not exercising the step",
        modes.len()
    );
}

// ── the mover ───────────────────────────────────────────────────────────────

/// The mover is built from ONE function now (`mover_for`), and the arm that
/// matters is the one that says an entity WITHOUT a movement component is
/// untouched by this wave: every committed sample is such an entity, and turning
/// autostep on for all of them would have quietly changed the platformer, the
/// coastal swimmer and the physics playground.
#[test]
fn an_entity_without_a_movement_component_moves_exactly_as_it_did() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(20.0, 0.5, 20.0),
    );
    // A 0.2 m step at z = 1.
    spawn_block(
        &mut w,
        Uuid::from_u128(0x2903_4000),
        DVec3::new(0.0, 0.1, 1.0),
        DVec3::new(4.0, 0.1, 0.2),
    );
    // A plain P9.1-era character: controller + collider, no movement component.
    let e = w.spawn_with_guid(HERO, "Legacy", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, 0.8, 0.0);
    w.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(RADIUS, 0.5, RADIUS),
            radius: RADIUS,
            ..Default::default()
        },
        CharacterController3D::default(),
        t,
    ));
    w.mark_dirty();
    w.propagate();
    b.sync_from_world(&w);

    let mover = inf_physics::d3::mover_for(&w, HERO);
    let mut pos = DVec3::new(0.0, 0.8, 0.0);
    for _ in 0..120 {
        let r = b
            .world_mut()
            .move_character(&mover, pos, DVec3::new(0.0, -0.05, 0.03), None);
        pos += r.translation;
    }
    assert!(
        pos.z < 0.75,
        "with no movement component there is no autostep, so the legacy character \
         stops at the step exactly as it did before P29.3: z = {}",
        pos.z
    );

    // And the movement step itself does not touch it: it has no
    // `CharacterMovement`, so it is not in the query at all.
    let outcomes = step_character_movement(&mut w, &mut b, DT);
    assert!(
        outcomes.is_empty(),
        "the movement step must ignore entities that never asked for it"
    );
}

/// A character whose collider is not a capsule keeps its shape and still gets a
/// mode — a value, not a refusal, because the speeds and the mode are still
/// meaningful even when nothing can resize.
#[test]
fn a_non_capsule_character_still_moves_and_simply_does_not_resize() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(20.0, 0.5, 20.0),
    );
    let e = w.spawn_with_guid(HERO, "Boxy", None);
    let mut t = Transform::IDENTITY;
    // Exactly on the floor: a box of half-height 0.9 has its underside at y = 0.
    t.translation = Vec3d::new(0.0, 0.9, 0.0);
    w.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(0.4, 0.9, 0.4),
            ..Default::default()
        },
        CharacterController3D::default(),
        CharacterMovement {
            player_controlled: true,
            ..Default::default()
        },
        t,
    ));
    w.mark_dirty();
    w.propagate();

    step(&mut w, &mut b, &press_crouch());
    assert_eq!(
        hero(&w).mode,
        MovementMode::Crouch,
        "the mode still changes"
    );
    assert!(
        (hero_capsule_half(&w) - 0.9).abs() < 1e-12,
        "but the box keeps its half-extent: {}",
        hero_capsule_half(&w)
    );
    for _ in 0..60 {
        step(&mut w, &mut b, &walk_forward());
    }
    assert!(hero_pos(&w).z > 0.1, "and it still walks");
}

/// A jump takes off, the gait ladder runs, and the derived anim inputs move —
/// the one arm that reads the P29.4 hand-off surface end to end.
#[test]
fn a_jump_leaves_the_ground_and_the_derived_outputs_track_the_gait() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    // A floor long enough that fifteen seconds of running cannot reach its edge:
    // the first version was 40 m, the character covered about 60, and the gait
    // ladder correctly reported `Run` because it had walked off the world.
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(40.0, 0.5, 400.0),
    );
    spawn_hero(&mut w, 0.0, 0.0, 0.0);
    step(&mut w, &mut b, &idle());
    assert!(hero(&w).runtime.grounded, "it starts on the floor");

    step(&mut w, &mut b, &press_jump());
    assert_eq!(hero(&w).mode, MovementMode::FallFree, "jump is a free fall");
    assert!(hero(&w).runtime.velocity.y > 0.0, "and it is going up");
    let peak_start = hero_pos(&w).y;
    for _ in 0..300 {
        step(&mut w, &mut b, &idle());
        if hero(&w).runtime.grounded && !hero(&w).mode.is_falling() {
            break;
        }
    }
    assert_eq!(hero(&w).mode, MovementMode::Grounded, "and it comes back");
    assert!(
        (hero_pos(&w).y - peak_start).abs() < 0.1,
        "landing back where it started: {} vs {peak_start}",
        hero_pos(&w).y
    );

    // Walk, then run, then sprint: the actual gait ladder climbs and the derived
    // blends climb with it.
    let mut seen: Vec<(Gait, f64)> = Vec::new();
    for (intent, _label) in [
        (
            MovementIntent {
                move_input: EcsVec2d::new(0.0, 1.0),
                walk: true,
                ..Default::default()
            },
            "walk",
        ),
        (walk_forward(), "run"),
        (
            MovementIntent {
                move_input: EcsVec2d::new(0.0, 1.0),
                sprint: true,
                ..Default::default()
            },
            "sprint",
        ),
    ] {
        for _ in 0..300 {
            step(&mut w, &mut b, &intent);
        }
        let h = hero(&w);
        seen.push((h.runtime.actual_gait, h.runtime.mapped_speed));
    }
    assert_eq!(seen[0].0, Gait::Walk, "walk: {:?}", seen[0]);
    assert_eq!(seen[1].0, Gait::Run, "run: {:?}", seen[1]);
    assert_eq!(seen[2].0, Gait::Sprint, "sprint: {:?}", seen[2]);
    assert!(
        seen[0].1 < seen[1].1 && seen[1].1 < seen[2].1,
        "the normalized speed climbs with the gait: {seen:?}"
    );
    let h = hero(&w);
    assert!(
        h.runtime.stride_blend > 0.9 && h.runtime.walk_run_blend > 0.9,
        "and the derived blends are at the top of their range at a sprint: {:?}",
        (h.runtime.stride_blend, h.runtime.walk_run_blend)
    );
    assert!(
        h.runtime.gait_scalar > 1.5,
        "as is the W_Gait-style scalar: {}",
        h.runtime.gait_scalar
    );
}

// ── body rotation ───────────────────────────────────────────────────────────

/// The body turns to face where it is going, and the turn is **rate-bounded**:
/// a goal that flips by 180 degrees does not snap the character round, because
/// ALS's first smoothing stage chases the goal at a constant rate.
///
/// The measurement is a `Transform`, not a runtime field: a step that turned its
/// own bookkeeping and never wrote the entity would leave the character facing
/// the way it started on screen.
///
/// The `AimYawRate` multiplier is NOT measured here, deliberately. It is a
/// property of one number (`grounded_rotation_rate`, unit-tested three ways in
/// `inf_ecs::movement`), and a world-level version of it cannot isolate the
/// variable: the movement input is expressed in the AIM frame, so spinning the
/// camera to raise the yaw rate also spins the direction the character is asked
/// to run in, and the body ends up chasing a different goal rather than the same
/// goal faster. Measuring it here would have produced a number that looked like
/// a regression and was a control error.
#[test]
fn the_body_faces_where_it_is_going_and_the_turn_is_rate_bounded() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(80.0, 0.5, 80.0),
    );
    spawn_hero(&mut w, 0.0, 0.0, 0.0);
    let body_yaw = |w: &EcsWorld| {
        w.world()
            .get::<Transform>(w.entity_of(HERO).unwrap())
            .unwrap()
            .rotation
            .y
    };

    for _ in 0..180 {
        step(&mut w, &mut b, &walk_forward());
    }
    assert!(
        body_yaw(&w).abs() < 5.0,
        "running +Z, the body faces +Z: {}",
        body_yaw(&w)
    );

    // Flip the input to dead astern. The goal moves 180 degrees in one step; the
    // BODY must not.
    let back = MovementIntent {
        move_input: EcsVec2d::new(0.0, -1.0),
        ..Default::default()
    };
    let mut worst_step = 0.0_f64;
    let mut prev = body_yaw(&w);
    for _ in 0..300 {
        step(&mut w, &mut b, &back);
        let now = body_yaw(&w);
        worst_step = worst_step.max(inf_ecs::movement::angle_delta_deg(now, prev).abs());
        prev = now;
    }
    // 500 deg/s at 1/60 s is 8.3 degrees; the exponential second stage only ever
    // lags that, so a per-step jump much past it is a snap.
    assert!(
        worst_step < 12.0,
        "the body snapped {worst_step} degrees in one step — the constant-rate \
         stage is not bounding it"
    );
    assert!(
        inf_ecs::movement::angle_delta_deg(body_yaw(&w), 180.0).abs() < 10.0,
        "…and it did eventually get there: {}",
        body_yaw(&w)
    );
}

/// Standing still and aiming, the body is **clamped** to within 100 degrees of
/// the aim rather than turned to face it — ALS's `LimitRotation`, and what stops
/// an idle character spinning under its own camera.
#[test]
fn an_idle_aiming_character_is_clamped_rather_than_turned() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(20.0, 0.5, 20.0),
    );
    spawn_hero(&mut w, 0.0, 0.0, 0.0);

    // Aim held, no movement input, camera swinging round at 120 deg/s.
    let aim_and_look = MovementIntent {
        aim: true,
        look_yaw_dps: 120.0,
        ..Default::default()
    };
    for _ in 0..30 {
        step(&mut w, &mut b, &aim_and_look);
    }
    let h = hero(&w);
    let e = w.entity_of(HERO).unwrap();
    let body = w.world().get::<Transform>(e).unwrap().rotation.y;
    // Half a second at 120 deg/s is 60 degrees of aim, which is inside the
    // hundred-degree band: the body has not been dragged round at all.
    assert!(
        (h.runtime.aim_yaw_deg - 60.0).abs() < 2.0,
        "the aim moved: {}",
        h.runtime.aim_yaw_deg
    );
    assert!(
        body.abs() < 1.0,
        "and the body did NOT follow it inside the band: {body}"
    );

    // Keep going past the band and the clamp starts pulling the body along.
    for _ in 0..90 {
        step(&mut w, &mut b, &aim_and_look);
    }
    let h = hero(&w);
    let body = w
        .world()
        .get::<Transform>(w.entity_of(HERO).unwrap())
        .unwrap()
        .rotation
        .y;
    let delta = inf_ecs::movement::angle_delta_deg(h.runtime.aim_yaw_deg, body);
    // The clamp is a FIRST-ORDER pull toward the bound, not a hard stop at it,
    // so a camera that keeps moving is always a little ahead: at 120 deg/s and
    // an interp of 20 the steady-state lag is 2 / (20/60) = 6 degrees, and the
    // measured 104 is that. Asserting 101 here would have been asserting a
    // behaviour ALS does not have.
    assert!(
        delta <= 108.0,
        "past the band the body is pulled to the bound, not left behind: aim {} body {body} delta {delta}",
        h.runtime.aim_yaw_deg
    );
    assert!(
        body > 1.0,
        "and it really was pulled — a body that never moved would pass the bound above"
    );
}
