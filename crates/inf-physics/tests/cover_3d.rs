//! **WAVE COV1: taking cover**, driven end to end through the real
//! [`PhysicsBridge3D`] over an [`EcsWorld`].
//!
//! `inf_ecs::cover`'s own tests pin the *rules* as functions of numbers — what
//! counts as cover, where a slide stops, which way a peek leans. These pin what
//! happens when those rules meet a world: whether the character actually ends up
//! against the wall, whether the capsule actually crouches behind a car, whether
//! a ray from the far side actually hits the cover instead of the head.
//!
//! **Every arm asserts the WORLD** — the transform, the collider, the ray — and
//! never the report. A cover system that set a mode and moved nothing would pass
//! the easy version of all of these; the CHAR1b.1 law, met here.
//!
//! The fixture is deliberately three boxes: a floor, a KERB (0.15 m, which must
//! never be cover), a CAR (0.80 m, which must be `Low`) and a WALL (3 m, which
//! must be `High`). Those three heights are the classification table, and a
//! mutation of either threshold moves at least one of them into the wrong class.

use glam::DVec3;
use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    MovementMode, RigidBody3D, Transform,
};
use inf_ecs::cover::{CoverClass, CoverSide};
use inf_ecs::math::Vec3d;
use inf_ecs::movement::MovementIntent;
use inf_ecs::{EcsWorld, Vec2d as EcsVec2d};
use inf_physics::d3::step_character_movement;
use inf_physics::PhysicsBridge3D;
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);

const HERO: Uuid = Uuid::from_u128(0xC0_0001);
const GROUND: Uuid = Uuid::from_u128(0xC0_0002);
const SURFACE: Uuid = Uuid::from_u128(0xC0_0003);

const RADIUS: f64 = 0.3;

/// The three heights the classification table is measured on, metres.
///
/// A **kerb** is `inf_ecs::traffic::KERB_HEIGHT_M`; a **car**'s flank is the
/// VEH2b chassis lattice's own half-height doubled; a **wall** is a storey.
const KERB_TOP_M: f64 = 0.15;
const CAR_TOP_M: f64 = 0.80;
const WALL_TOP_M: f64 = 3.00;

/// **A surface between the split and the probe's look**, metres — a parapet, a
/// skip, a shipping container on its side.
///
/// It is here because a MUTATION found the hole: moving the LOW/HIGH split from
/// 1.25 m to 3.00 m reddened NOTHING in this file. A 3 m wall has no top within
/// `CoverSettings::max_top_m` (2.50 m), so `classify` answers `High` off the
/// infinity and never compares the split at all; the car and the counter are
/// below 1.25 either way. 1.80 m is the only kind of surface whose class the
/// split actually decides, and without one of them the split was a number no
/// arm in this repository could falsify.
const PARAPET_TOP_M: f64 = 1.80;

/// **A LOW cover tall enough to hide a crouched head**, metres — a car's window
/// line, or a bar counter.
///
/// The shipped `CharacterMovement` crouches to a 0.3 m half-height on a 0.3 m
/// radius, so a crouched capsule is **1.20 m tall**, and the LOW/HIGH split is
/// `inf_anim::MANTLE_HIGH_SPLIT_M` = 1.25 m. The window in which a LOW cover
/// hides a crouched character's head is therefore 1.20-1.25 m and nothing
/// below it does: a 0.80 m car door leaves 40 cm of person above the metal.
/// That is a real property of this engine's capsule and it is stated rather
/// than tuned around — `CAR_TOP_M` above is what the CLASS arms use, because it
/// is far from both thresholds, and this is what the EXPOSURE arm uses, because
/// it is the only kind of low cover a body can actually get behind.
const COUNTER_TOP_M: f64 = 1.24;

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

fn spawn_hero(w: &mut EcsWorld, x: f64, z: f64) {
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let e = w.spawn_with_guid(HERO, "Hero", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(x, cm.stand_half_height_m + RADIUS, z);
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

/// A world with a floor, a surface of height `top_m` running along X at `z = 1`,
/// and the hero standing a little south of it facing north (`+Z`).
///
/// `half_x` is how far the surface runs each way, which is what the extent
/// search measures and what the corner stop is asserted against.
fn world_with(top_m: f64, half_x: f64) -> (EcsWorld, PhysicsBridge3D) {
    let mut w = EcsWorld::new();
    let b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(40.0, 0.5, 40.0),
    );
    spawn_block(
        &mut w,
        SURFACE,
        // Centred at z = 1.3 and 0.3 m thick, so its near face is at z = 1.0.
        DVec3::new(0.0, top_m * 0.5, 1.3),
        DVec3::new(half_x, top_m * 0.5, 0.3),
    );
    spawn_hero(&mut w, 0.0, 0.4);
    (w, b)
}

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

fn hero_pos(w: &EcsWorld) -> DVec3 {
    let e = w.entity_of(HERO).unwrap();
    w.world()
        .get::<Transform>(e)
        .unwrap()
        .translation
        .to_dvec3()
}

fn hero_half(w: &EcsWorld) -> f64 {
    let e = w.entity_of(HERO).unwrap();
    w.world().get::<Collider3D>(e).unwrap().half_extents.y
}

/// The top of the hero's head, world metres — the capsule's own extent, which is
/// what a ray at head height is asserted against.
fn hero_head_y(w: &EcsWorld) -> f64 {
    hero_pos(w).y + hero_half(w) + RADIUS
}

fn idle() -> MovementIntent {
    MovementIntent::default()
}

fn press_cover() -> MovementIntent {
    MovementIntent {
        cover: true,
        ..Default::default()
    }
}

/// Settle the hero on the ground, then take cover, then let the snap finish.
///
/// Answers the movement component afterwards. The settle matters: a character
/// that has never been stepped has no `seeded` facing, and the cover press
/// reaches in the direction it is facing.
fn take_cover(w: &mut EcsWorld, b: &mut PhysicsBridge3D) -> CharacterMovement {
    for _ in 0..30 {
        step(w, b, &idle());
    }
    step(w, b, &press_cover());
    for _ in 0..40 {
        step(w, b, &idle());
    }
    hero(w)
}

// ─────────────────────────────────────────────────────────────────────────────
// (1) THE CLASSIFICATION TABLE, ON THE WORLD
// ─────────────────────────────────────────────────────────────────────────────

/// **A kerb is not cover, a car is crouch cover, a wall is stand cover** — the
/// clause-1 table, measured against three real colliders and asserted on the
/// CAPSULE rather than on the class.
///
/// The three heights bracket both thresholds, so a mutation of either one moves
/// a row: raise the floor past 0.80 and the car stops being cover; lower the
/// split below 0.80 and the character stands behind a car; raise the split past
/// 3.00 and it crouches behind a wall.
#[test]
fn a_kerb_is_not_cover_a_car_is_crouch_cover_and_a_wall_is_stand_cover() {
    let stand = CharacterMovement::default().stand_half_height_m;
    let crouch = CharacterMovement::default().crouch_half_height_m;

    // ── the kerb: the press does nothing at all.
    let (mut w, mut b) = world_with(KERB_TOP_M, 4.0);
    for _ in 0..30 {
        step(&mut w, &mut b, &idle());
    }
    // The refusal is read on the step the press was MADE: `runtime.refusal` is
    // this step's answer and the idle steps after it clear it, which is the
    // shape every other refusal in this engine has.
    step(&mut w, &mut b, &press_cover());
    assert_eq!(
        hero(&w).runtime.refusal,
        inf_ecs::components::MovementRefusal::ConditionNotMet,
        "the refusal is a VALUE and says why"
    );
    for _ in 0..40 {
        step(&mut w, &mut b, &idle());
    }
    let cm = hero(&w);
    assert_eq!(
        cm.mode,
        MovementMode::Grounded,
        "a {KERB_TOP_M} m kerb is cover, which it must never be"
    );
    assert!(!cm.runtime.cover.active);
    assert!(
        (hero_half(&w) - stand).abs() < 1e-9,
        "the kerb press resized the capsule"
    );

    // ── the car: cover, crouched.
    let (mut w, mut b) = world_with(CAR_TOP_M, 4.0);
    let cm = take_cover(&mut w, &mut b);
    assert_eq!(cm.mode, MovementMode::Cover, "a car's flank is not cover");
    assert_eq!(cm.runtime.cover.class, CoverClass::Low);
    assert!(cm.runtime.cover.crouched);
    assert!(
        (hero_half(&w) - crouch).abs() < 1e-9,
        "behind a {CAR_TOP_M} m car the capsule is {} and crouch is {crouch}",
        hero_half(&w)
    );
    // The measured top is the car's own height, off the WORLD.
    assert!(
        (cm.runtime.cover.top_m - CAR_TOP_M).abs() < 0.05,
        "measured top {:.4} against {CAR_TOP_M}",
        cm.runtime.cover.top_m
    );

    // ── the parapet: HIGH, and it is the row the SPLIT decides.
    let (mut w, mut b) = world_with(PARAPET_TOP_M, 4.0);
    let cm = take_cover(&mut w, &mut b);
    assert_eq!(cm.mode, MovementMode::Cover, "a parapet is not cover");
    assert_eq!(
        cm.runtime.cover.class,
        CoverClass::High,
        "a {PARAPET_TOP_M} m parapet classified {:?}; its measured top is {:.4} m \
         and the split is {}",
        cm.runtime.cover.class,
        cm.runtime.cover.top_m,
        inf_anim::MANTLE_HIGH_SPLIT_M
    );
    assert!(
        (cm.runtime.cover.top_m - PARAPET_TOP_M).abs() < 0.05,
        "the parapet measured {:.4} m",
        cm.runtime.cover.top_m
    );
    assert!(
        (hero_half(&w) - stand).abs() < 1e-9,
        "behind a {PARAPET_TOP_M} m parapet the capsule is {}",
        hero_half(&w)
    );

    // ── the wall: cover, standing.
    let (mut w, mut b) = world_with(WALL_TOP_M, 4.0);
    let cm = take_cover(&mut w, &mut b);
    assert_eq!(cm.mode, MovementMode::Cover, "a wall is not cover");
    assert_eq!(cm.runtime.cover.class, CoverClass::High);
    assert!(!cm.runtime.cover.crouched);
    assert!(
        (hero_half(&w) - stand).abs() < 1e-9,
        "against a wall the capsule is {} and standing is {stand}",
        hero_half(&w)
    );
    assert!(
        cm.runtime.cover.top_m > inf_anim::MANTLE_HIGH_SPLIT_M,
        "measured top {:.4}",
        cm.runtime.cover.top_m
    );
}

/// **The back is against the wall** — the standoff, measured off the transform
/// and the collider rather than off the anchor the state carries.
///
/// The surface's near face is at `z = 1.0`; the capsule's own surface must sit
/// `STANDOFF_M` from it, within the brief's own 5 cm.
#[test]
fn the_character_ends_up_against_the_surface_at_its_standoff() {
    let (mut w, mut b) = world_with(WALL_TOP_M, 4.0);
    let before = hero_pos(&w);
    let cm = take_cover(&mut w, &mut b);
    assert_eq!(cm.mode, MovementMode::Cover);
    let after = hero_pos(&w);
    let gap = 1.0 - (after.z + RADIUS);
    assert!(
        (gap - inf_ecs::cover::STANDOFF_M).abs() <= inf_ecs::cover::STANDOFF_TOLERANCE_M,
        "the capsule's surface is {gap:.4} m off the wall, wanted {:.4} +- {:.4}",
        inf_ecs::cover::STANDOFF_M,
        inf_ecs::cover::STANDOFF_TOLERANCE_M
    );
    // It MOVED to get there — an arm that passed on a character already at the
    // wall would prove nothing.
    assert!(
        (after.z - before.z).abs() > 0.2,
        "the hero travelled {:.4} m to reach the wall",
        after.z - before.z
    );
    // And it faces INTO the surface: the wall's outward normal is `-Z`, so the
    // character looks at `+Z`, which is yaw zero on this engine's compass.
    let yaw = inf_ecs::movement::angle_delta_deg(cm.runtime.body_yaw_deg, 0.0);
    assert!(yaw.abs() < 2.0, "the body faces {yaw:.3} deg off the wall");
}

/// **The snap is a blend, not a teleport.** The largest single-step displacement
/// over the press is bounded; a teleport would cover the whole distance in one
/// step and be twelve times that.
#[test]
fn the_snap_is_bounded_and_a_teleport_would_not_be() {
    let (mut w, mut b) = world_with(WALL_TOP_M, 4.0);
    for _ in 0..30 {
        step(&mut w, &mut b, &idle());
    }
    let mut prev = hero_pos(&w);
    let mut worst = 0.0f64;
    let mut total = 0.0f64;
    step(&mut w, &mut b, &press_cover());
    for _ in 0..40 {
        step(&mut w, &mut b, &idle());
        let now = hero_pos(&w);
        let d = ((now.x - prev.x).powi(2) + (now.z - prev.z).powi(2)).sqrt();
        worst = worst.max(d / DT);
        total += d;
        prev = now;
    }
    assert_eq!(hero(&w).mode, MovementMode::Cover);
    let bound = inf_ecs::cover::max_snap_speed_mps() * 1.5;
    assert!(
        worst <= bound,
        "the snap peaked at {worst:.4} m/s against the {bound:.4} m/s bound"
    );
    assert!(
        total > 0.2,
        "the snap covered {total:.4} m, which is nothing"
    );
    // What a teleport would have measured: the SAME distance in one step, which
    // is what the arm is distinguishing the blend from. The comparison is
    // against the blend's own peak rather than against the bound, so a fixture
    // whose wall is close still falsifies a cut.
    assert!(
        total / DT > worst * 2.0,
        "the snap peaked at {worst:.4} m/s, which a step covering its whole \
         {total:.4} m ({:.4} m/s) is not distinguishable from",
        total / DT
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) THE SLIDE AND THE CORNER
// ─────────────────────────────────────────────────────────────────────────────

/// **The slide stops at the corner the probe measured**, with the capsule's own
/// edge on it — it never slides off the end.
///
/// The wall runs 1.5 m each way from the origin, so a character sliding left
/// must stop with its centre at `1.5 - RADIUS = 1.2` and go no further however
/// long the stick is held. The control is the same wall twice as long, where the
/// character passes 1.2 comfortably — without it the arm would pass on a
/// character that simply could not walk.
#[test]
fn a_cover_slide_stops_at_the_corner_and_does_not_walk_off_it() {
    fn slide_to_the_end(half_x: f64) -> f64 {
        let (mut w, mut b) = world_with(WALL_TOP_M, half_x);
        let cm = take_cover(&mut w, &mut b);
        assert_eq!(cm.mode, MovementMode::Cover, "the wall is not cover");
        // Left along the wall: the character faces `+Z`, so its left is `-X`…
        // and the stick is in the AIM frame, where `-x` is left.
        let left = MovementIntent {
            move_input: EcsVec2d::new(-1.0, 0.0),
            ..Default::default()
        };
        let mut furthest = hero_pos(&w).x;
        for _ in 0..400 {
            step(&mut w, &mut b, &left);
            furthest = furthest.min(hero_pos(&w).x);
            assert_eq!(
                hero(&w).mode,
                MovementMode::Cover,
                "the slide left cover on its own"
            );
        }
        -furthest
    }
    let short = slide_to_the_end(1.5);
    let long = slide_to_the_end(4.0);
    assert!(
        (short - (1.5 - RADIUS)).abs() < 0.10,
        "the slide stopped at {short:.4} m and the corner is at {:.4} m",
        1.5 - RADIUS
    );
    assert!(
        long > short + 1.0,
        "the control wall is {long:.4} m and the short one {short:.4} — the arm \
         is measuring how far a character can walk, not where it stops"
    );
}

/// **LOW becomes HIGH along the surface** — the re-classification, and the
/// capsule stands up for it.
///
/// The fixture is a car for the first two metres and a wall after it, which is
/// the brief's own "a car's hood into its roof line".
#[test]
fn sliding_from_a_low_surface_onto_a_high_one_stands_the_character_up() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(40.0, 0.5, 40.0),
    );
    // The car: from x = -0.2 leftward (toward -X) for 2 m.
    spawn_block(
        &mut w,
        SURFACE,
        DVec3::new(-1.2, CAR_TOP_M * 0.5, 1.3),
        DVec3::new(1.0, CAR_TOP_M * 0.5, 0.3),
    );
    // The wall, continuing where the car ends.
    spawn_block(
        &mut w,
        Uuid::from_u128(0xC0_0004),
        DVec3::new(-3.7, WALL_TOP_M * 0.5, 1.3),
        DVec3::new(1.5, WALL_TOP_M * 0.5, 0.3),
    );
    spawn_hero(&mut w, -1.2, 0.4);
    let cm = take_cover(&mut w, &mut b);
    assert_eq!(cm.mode, MovementMode::Cover);
    assert_eq!(cm.runtime.cover.class, CoverClass::Low, "the car");
    let crouched_half = hero_half(&w);
    let left = MovementIntent {
        move_input: EcsVec2d::new(-1.0, 0.0),
        ..Default::default()
    };
    let mut became_high = false;
    for _ in 0..300 {
        step(&mut w, &mut b, &left);
        if hero(&w).runtime.cover.class == CoverClass::High {
            became_high = true;
            break;
        }
    }
    assert!(
        became_high,
        "the slide never re-classified: it is still {:?} at x = {:.3}",
        hero(&w).runtime.cover.class,
        hero_pos(&w).x
    );
    // A few more steps for the capsule to follow the class.
    for _ in 0..10 {
        step(&mut w, &mut b, &left);
    }
    assert!(
        hero_half(&w) > crouched_half + 0.1,
        "the capsule is {} and it was {crouched_half} — the class changed and \
         the body did not",
        hero_half(&w)
    );
    assert_eq!(
        hero(&w).mode,
        MovementMode::Cover,
        "and it is still in cover"
    );
}

/// **The stick held AWAY leaves cover, and a flick does not.**
#[test]
fn the_stick_held_away_leaves_cover_and_one_frame_of_it_does_not() {
    let away = MovementIntent {
        move_input: EcsVec2d::new(0.0, -1.0),
        ..Default::default()
    };
    // One step of it: still in cover.
    let (mut w, mut b) = world_with(WALL_TOP_M, 4.0);
    assert_eq!(take_cover(&mut w, &mut b).mode, MovementMode::Cover);
    step(&mut w, &mut b, &away);
    assert_eq!(
        hero(&w).mode,
        MovementMode::Cover,
        "one frame of the stick popped the character out of cover"
    );
    // Held past the window: out.
    let held = (inf_ecs::cover::AWAY_LEAVE_S / DT).ceil() as usize + 4;
    for _ in 0..held {
        step(&mut w, &mut b, &away);
    }
    assert_ne!(
        hero(&w).mode,
        MovementMode::Cover,
        "the stick was held away for {:.2} s and the character is still stuck",
        held as f64 * DT
    );
    assert!(!hero(&w).runtime.cover.active, "the state went with it");
}

/// **The press leaves cover too**, and it goes back to the stance the surface
/// implied rather than always standing.
#[test]
fn a_second_press_leaves_cover_into_the_stance_the_surface_implied() {
    let (mut w, mut b) = world_with(CAR_TOP_M, 4.0);
    assert_eq!(take_cover(&mut w, &mut b).mode, MovementMode::Cover);
    step(&mut w, &mut b, &press_cover());
    for _ in 0..5 {
        step(&mut w, &mut b, &idle());
    }
    assert_eq!(
        hero(&w).mode,
        MovementMode::Crouch,
        "leaving a LOW cover stood the character up into whatever was above it"
    );
    assert!(!hero(&w).runtime.cover.active);
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) THE PEEK, AND WHAT A BULLET CAN REACH
// ─────────────────────────────────────────────────────────────────────────────

/// **The head is behind the cover while tucked in and above it while peeking** —
/// clause 4's exposure claim, asserted with a RAY through the physics world
/// rather than with a state name.
///
/// The shooter is on the far side of a low wall, firing at head height. Tucked
/// in, the ray hits the COVER; peeking over the top, it reaches the character.
#[test]
fn a_shot_at_a_crouched_character_hits_the_cover_and_a_peeking_one_is_exposed() {
    let (mut w, mut b) = world_with(COUNTER_TOP_M, 4.0);
    let cm = take_cover(&mut w, &mut b);
    assert_eq!(cm.mode, MovementMode::Cover);
    assert_eq!(cm.runtime.cover.class, CoverClass::Low);

    // The shot: from the far side of the surface, level, aimed at the
    // character's own upper chest — a hand's width below the crown, which is
    // where a shooter aims and which MOVES with the stance. That is the whole
    // point: the ray is not at a fixed height, it is at the height the body it
    // is aimed at happens to be.
    let cast = |w: &EcsWorld, b: &mut PhysicsBridge3D| -> (Option<Uuid>, f64) {
        let y = hero_head_y(w) - 0.15;
        let from = DVec3::new(0.0, y, 6.0);
        let hit = b.world_mut().cast_ray_excluding(
            from,
            DVec3::new(0.0, 0.0, -1.0),
            20.0,
            &Default::default(),
        );
        (hit.and_then(|h| b.guid_of_collider(h.collider)), y)
    };
    b.sync_from_world(&w);
    let (tucked, y) = cast(&w, &mut b);
    assert_eq!(
        tucked,
        Some(SURFACE),
        "a shot at {y:.4} m at a crouched character in cover hit {tucked:?}, not \
         the cover; the head is at {:.4} and the cover's top is {COUNTER_TOP_M}",
        hero_head_y(&w)
    );
    assert!(
        hero_head_y(&w) < COUNTER_TOP_M,
        "the head is at {:.4} m and the cover's top is {COUNTER_TOP_M} — it is \
         not behind anything",
        hero_head_y(&w)
    );

    // Now peek: hold aim. A LOW cover's peek is `Over` — the character stands.
    let aim = MovementIntent {
        aim: true,
        ..Default::default()
    };
    for _ in 0..40 {
        step(&mut w, &mut b, &aim);
    }
    let cm = hero(&w);
    assert_eq!(cm.runtime.cover.side, CoverSide::Over, "the peek's side");
    assert!(
        cm.runtime.cover.peek > 0.9,
        "peek {:.3}",
        cm.runtime.cover.peek
    );
    assert!(
        hero_head_y(&w) > COUNTER_TOP_M + 0.4,
        "peeking, the head is at {:.4} m and the cover's top is {COUNTER_TOP_M} \
         — it never came up",
        hero_head_y(&w)
    );
    b.sync_from_world(&w);
    let (exposed, y) = cast(&w, &mut b);
    assert_eq!(
        exposed,
        Some(HERO),
        "the peeking character was not exposed: the ray at {y:.4} m hit \
         {exposed:?}"
    );

    // …and releasing aim puts it back.
    for _ in 0..40 {
        step(&mut w, &mut b, &idle());
    }
    assert!(hero(&w).runtime.cover.peek < 0.05);
    b.sync_from_world(&w);
    assert_eq!(
        cast(&w, &mut b).0,
        Some(SURFACE),
        "un-peeking left the head out"
    );
}

/// **A corner peek steps the capsule PAST the corner**, which is what makes a
/// round-the-corner shot possible at all — and coming back puts it behind again.
#[test]
fn a_corner_peek_puts_the_capsule_past_the_corner_and_back() {
    // A wall that ends 1.5 m to the character's left.
    let (mut w, mut b) = world_with(WALL_TOP_M, 1.5);
    let cm = take_cover(&mut w, &mut b);
    assert_eq!(cm.runtime.cover.class, CoverClass::High);
    // Slide to the left corner.
    let left = MovementIntent {
        move_input: EcsVec2d::new(-1.0, 0.0),
        ..Default::default()
    };
    for _ in 0..300 {
        step(&mut w, &mut b, &left);
    }
    let at_corner = hero_pos(&w).x;
    assert!(
        at_corner < -1.0,
        "the slide reached {at_corner:.3}, which is not a corner"
    );
    // Aim: the nearer corner is the left one.
    let aim = MovementIntent {
        aim: true,
        ..Default::default()
    };
    for _ in 0..40 {
        step(&mut w, &mut b, &aim);
    }
    let cm = hero(&w);
    assert_eq!(cm.runtime.cover.side, CoverSide::Left, "the nearer corner");
    let leaned = hero_pos(&w).x;
    assert!(
        leaned < at_corner - inf_ecs::cover::PEEK_LATERAL_M * 0.8,
        "the peek moved the capsule from {at_corner:.4} to {leaned:.4}, which is \
         less than the {:.2} m lean",
        inf_ecs::cover::PEEK_LATERAL_M
    );
    // The capsule's outer edge is past the wall's end.
    assert!(
        leaned - RADIUS < -1.5,
        "the capsule's edge is at {:.4} and the corner at -1.5 — nothing is out",
        leaned - RADIUS
    );
    for _ in 0..40 {
        step(&mut w, &mut b, &idle());
    }
    assert!(
        (hero_pos(&w).x - at_corner).abs() < 0.05,
        "coming back left the character at {:.4} instead of {at_corner:.4}",
        hero_pos(&w).x
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) THE VAULT, AND THE COST
// ─────────────────────────────────────────────────────────────────────────────

/// **A vault out of LOW cover crosses the object** — the mantle door, taken from
/// cover with one press and no stick, and measured on the character's own
/// position rather than on the mode it announced.
#[test]
fn a_vault_out_of_low_cover_puts_the_character_on_the_far_side() {
    let (mut w, mut b) = world_with(CAR_TOP_M, 4.0);
    let cm = take_cover(&mut w, &mut b);
    assert_eq!(cm.mode, MovementMode::Cover);
    let before = hero_pos(&w);
    // The jump press, with the stick CENTRED — the cover state is the facing.
    let jump = MovementIntent {
        jump: true,
        ..Default::default()
    };
    step(&mut w, &mut b, &jump);
    let mantled = hero(&w).mode == MovementMode::Mantle;
    assert!(
        mantled,
        "one press out of cover did not vault: the mode is {:?} and the refusal \
         {:?}",
        hero(&w).mode,
        hero(&w).runtime.refusal
    );
    assert!(
        !hero(&w).runtime.cover.active,
        "the vault kept the cover state, which would pin the capsule back"
    );
    for _ in 0..200 {
        step(&mut w, &mut b, &idle());
        if hero(&w).mode != MovementMode::Mantle {
            break;
        }
    }
    let after = hero_pos(&w);
    assert!(
        after.z > 1.6,
        "the vault ended at z = {:.4}; the object's far face is at 1.6 and the \
         character started at {:.4}",
        after.z,
        before.z
    );
    // It came back DOWN on the far side rather than standing on the bonnet: a
    // vault goes over, a mantle goes onto. The comparison is on the FEET,
    // because the character entered cover crouched (a low surface) and leaves
    // the vault standing -- two capsule centres 28 cm apart with the soles in
    // the same place.
    let feet_after = after.y - hero_half(&w) - RADIUS;
    assert!(
        feet_after.abs() < 0.15,
        "the vault's feet ended at y = {feet_after:.4} and the ground is at 0 -- it is standing on the object rather than over it"
    );
    assert_eq!(
        hero(&w).mode,
        MovementMode::Grounded,
        "the vault did not hand the character back"
    );
}

/// **A JUMP THAT CANNOT VAULT LEAVES NO COVER STATE BEHIND** — the invariant
/// the audit's own frames found broken.
///
/// A press and an away-stick both clear `CoverState`, and so does a successful
/// vault. A jump press whose mantle REFUSES does not go through any of those:
/// it falls through the traversal chain with the mode untouched, and the next
/// step that finds no ground under the character puts it in a fall — out of
/// `MovementMode::Cover` with `cover.active` still true.
///
/// Measured in the demo loop before the fix: **133 samples** of the hero
/// walking the rest of a session as `Grounded` while the hero log read `class
/// High, side Left, peek 1.000`, the pose step still rolling its torso and
/// `anim_bridge` still publishing `cover = 2` into the machine.
///
/// So: a HIGH wall nothing can climb, a press, a jump, and then the invariant
/// asserted on **every step** of the drive that follows.
#[test]
fn a_jump_that_cannot_vault_leaves_no_cover_state_behind() {
    let (mut w, mut b) = world_with(WALL_TOP_M, 4.0);
    let cm = take_cover(&mut w, &mut b);
    assert_eq!(cm.mode, MovementMode::Cover);
    assert!(cm.runtime.cover.active);
    let jump = MovementIntent {
        jump: true,
        ..Default::default()
    };
    step(&mut w, &mut b, &jump);
    println!(
        "  after the jump press: mode {:?}, cover active {}",
        hero(&w).mode,
        hero(&w).runtime.cover.active
    );
    // ── **AND ANY OTHER WAY OUT**, which is what actually happened.
    //
    //    The island's hero left `Cover` for a FALL. A fixture cannot easily
    //    make the ground disappear, so the mode is moved directly -- which is
    //    precisely the invariant under test: whatever put the character in
    //    another mode, the cover state does not survive it. Without the one
    //    line in `step_character_movement` this assertion fails.
    {
        let e = w.entity_of(HERO).expect("the hero is there");
        let mut cmc = w
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("the hero moves");
        cmc.mode = MovementMode::Grounded;
        assert!(cmc.runtime.cover.active, "the fixture set up nothing");
    }
    step(&mut w, &mut b, &idle());
    let after = hero(&w);
    println!(
        "  one step after the mode was taken away: mode {:?}, cover active {}",
        after.mode, after.runtime.cover.active
    );
    assert!(
        !after.runtime.cover.active,
        "the character is in {:?} and still carries a live cover state (class {:?}, peek          {:.3})",
        after.mode,
        after.runtime.cover.class,
        after.runtime.cover.peek
    );

    // Walk about afterwards, which is what the demo loop did.
    let mut worst: Option<(usize, MovementMode)> = None;
    for i in 0..300 {
        let intent = MovementIntent {
            move_input: EcsVec2d::new(if i % 3 == 0 { 1.0 } else { 0.0 }, 1.0),
            ..Default::default()
        };
        step(&mut w, &mut b, &intent);
        let c = hero(&w);
        if c.mode != MovementMode::Cover && c.runtime.cover.active && worst.is_none() {
            worst = Some((i, c.mode));
        }
    }
    assert!(
        worst.is_none(),
        "at step {:?} the character was in {:?} with a live cover state — the state \
         outlived the mode",
        worst.map(|(i, _)| i),
        worst.map(|(_, m)| m)
    );
    let c = hero(&w);
    println!(
        "  300 steps later: mode {:?}, cover active {}, class {:?}, peek {:.3}",
        c.mode, c.runtime.cover.active, c.runtime.cover.class, c.runtime.cover.peek
    );
}

/// **A low surface with a wall behind it is MANTLED, not vaulted** — the
/// refusal that makes the vault a measurement rather than an assumption.
///
/// The same press, the same class, the same cover; the only difference is that
/// there is nowhere to land on the far side, and the character ends up on TOP.
#[test]
fn a_vault_with_nowhere_to_land_becomes_an_ordinary_mantle() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(40.0, 0.5, 40.0),
    );
    // The low cover…
    spawn_block(
        &mut w,
        SURFACE,
        DVec3::new(0.0, CAR_TOP_M * 0.5, 1.3),
        DVec3::new(4.0, CAR_TOP_M * 0.5, 0.3),
    );
    // …with a tall wall immediately behind it, so every far-side sample is
    // either inside the wall or on top of something.
    spawn_block(
        &mut w,
        Uuid::from_u128(0xC0_0005),
        DVec3::new(0.0, 2.0, 2.4),
        DVec3::new(4.0, 2.0, 0.5),
    );
    spawn_hero(&mut w, 0.0, 0.4);
    let cm = take_cover(&mut w, &mut b);
    assert_eq!(cm.mode, MovementMode::Cover);
    assert_eq!(cm.runtime.cover.class, CoverClass::Low);
    let before = hero_pos(&w);
    let jump = MovementIntent {
        jump: true,
        ..Default::default()
    };
    step(&mut w, &mut b, &jump);
    assert_eq!(hero(&w).mode, MovementMode::Mantle, "the press did nothing");
    for _ in 0..200 {
        step(&mut w, &mut b, &idle());
        if hero(&w).mode != MovementMode::Mantle {
            break;
        }
    }
    let after = hero_pos(&w);
    assert!(
        after.y > before.y + 0.4,
        "with nowhere to land the character should be ON the object: y went \
         {:.4} -> {:.4}",
        before.y,
        after.y
    );
    assert!(
        after.z < 1.9,
        "and it should not have crossed the wall behind it: z = {:.4}",
        after.z
    );
}

/// **SPRINTING INTO COVER GETS THE LONGER WINDOW** — clause 3's slide-in.
///
/// A body arriving at sprint speed does not stop in a quarter of a second, so
/// the entry is given [`inf_ecs::cover::SLIDE_IN_S`] instead of
/// [`inf_ecs::cover::SNAP_S`] and CHAR1b.2's authored slide plays over it. The
/// arm reads the STATE's own window and the number of steps the snap actually
/// took, not the flag beside them: a `slide_in` that set a bool and used the
/// short window would pass on the flag alone.
#[test]
fn sprinting_into_cover_takes_the_longer_slide_in_window() {
    fn enter(sprint: bool) -> (bool, f64, usize) {
        // Far enough back that a sprint has room to reach its speed.
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
            SURFACE,
            DVec3::new(0.0, WALL_TOP_M * 0.5, 12.3),
            DVec3::new(6.0, WALL_TOP_M * 0.5, 0.3),
        );
        spawn_hero(&mut w, 0.0, 0.0);
        for _ in 0..30 {
            step(&mut w, &mut b, &idle());
        }
        // Run at the wall until the press can take it.
        let drive = MovementIntent {
            move_input: EcsVec2d::new(0.0, 1.0),
            sprint,
            ..Default::default()
        };
        let mut pressed = false;
        let mut snapping = 0usize;
        let mut window = 0.0f64;
        let mut slide_in = false;
        for _ in 0..900 {
            // **Inside the PROBE's reach**, not inside the snap's bound: the
            // face is at z = 12.0 and `CoverSettings::reach_m` is 0.90 m from
            // the feet, so a press from a metre and a half back probes empty
            // air. The same lesson the NPC search learned the hard way.
            if !pressed && hero_pos(&w).z > 11.15 {
                let mut p = drive;
                p.cover = true;
                step(&mut w, &mut b, &p);
                pressed = true;
                let c = hero(&w).runtime.cover;
                slide_in = c.slide_in;
                window = c.snap_s;
                continue;
            }
            let still = idle();
            step(&mut w, &mut b, if pressed { &still } else { &drive });
            if hero(&w).runtime.cover.snapping() {
                snapping += 1;
            }
            if pressed && !hero(&w).runtime.cover.snapping() && hero(&w).runtime.cover.active {
                break;
            }
        }
        assert_eq!(hero(&w).mode, MovementMode::Cover, "the press did not take");
        (slide_in, window, snapping)
    }
    let (walked_flag, walked_window, walked_steps) = enter(false);
    let (sprint_flag, sprint_window, sprint_steps) = enter(true);
    println!(
        "\n=== the slide-in ===\n  walked in: slide_in {walked_flag}, window {walked_window:.3} s, \
         {walked_steps} snapping steps\n  sprinted in: slide_in {sprint_flag}, window \
         {sprint_window:.3} s, {sprint_steps} snapping steps"
    );
    assert!(sprint_flag, "a sprint arrival was not a slide-in");
    assert!(
        !walked_flag,
        "a walk arrival was one too, which makes it meaningless"
    );
    assert!(
        (sprint_window - inf_ecs::cover::SLIDE_IN_S).abs() < 1e-9,
        "the sprint window is {sprint_window:.4} s"
    );
    assert!(
        (walked_window - inf_ecs::cover::SNAP_S).abs() < 1e-9,
        "the walked window is {walked_window:.4} s"
    );
    // …and the SNAP really took longer, measured in steps rather than read off
    // the field the branch set.
    assert!(
        sprint_steps > walked_steps,
        "the sprint's snap took {sprint_steps} steps and the walk's {walked_steps} — the \
         longer window is a number nothing spent"
    );
}

/// **A character that is not in cover and did not press the key spends ZERO
/// shape casts on cover** — the budget arm, and it reads the step's own counter
/// rather than a timer.
#[test]
fn an_idle_character_probes_for_cover_exactly_never() {
    let (mut w, mut b) = world_with(WALL_TOP_M, 4.0);
    // Standing right next to a wall, which is the worst case for a probe that
    // ran opportunistically.
    let walk = MovementIntent {
        move_input: EcsVec2d::new(0.0, 1.0),
        ..Default::default()
    };
    let still = idle();
    let mut spent = 0u32;
    for i in 0..240 {
        b.sync_from_world(&w);
        inf_ecs::movement::apply_intent(&mut w, if i % 2 == 0 { &walk } else { &still });
        for o in step_character_movement(&mut w, &mut b, DT) {
            spent += o.cover_sweeps;
        }
    }
    assert_eq!(
        spent, 0,
        "a character that never pressed cover spent {spent} shape casts looking \
         for it"
    );
    assert_eq!(hero(&w).runtime.cover.sweeps, 0);

    // …and the press is what costs something, so the zero above is a measurement
    // and not a probe that is switched off.
    b.sync_from_world(&w);
    inf_ecs::movement::apply_intent(&mut w, &press_cover());
    let pressed: u32 = step_character_movement(&mut w, &mut b, DT)
        .iter()
        .map(|o| o.cover_sweeps)
        .sum();
    assert!(
        pressed > 0,
        "the press spent nothing either — the probe is not running at all"
    );
    // And in cover it keeps costing, because the surface changes under a slide.
    let held: u32 = {
        b.sync_from_world(&w);
        inf_ecs::movement::apply_intent(&mut w, &idle());
        step_character_movement(&mut w, &mut b, DT)
            .iter()
            .map(|o| o.cover_sweeps)
            .sum()
    };
    assert!(held > 0, "a character IN cover re-probes nothing");
}

/// **A DIAGNOSTIC, and it asserts nothing** — run it by name while tuning.
///
/// It prints the slide step by step: the position, the projected stick, the
/// velocity, the anchor, the extents and the corner latch. It is the tool that
/// found the two defects this wave's own arms were written blind to — an anchor
/// taken at a capsule sweep's WITNESS POINT, which walked sideways under a
/// standing character, and a peek whose lateral sign was the animation
/// parameter's rather than the tangent's. Kept because the next person tuning a
/// cover surface will want it, `#[ignore]`d because a test that asserts nothing
/// is not a gate arm.
#[test]
#[ignore = "a diagnostic that prints; it asserts nothing"]
fn probe_the_slide() {
    let (mut w, mut b) = world_with(WALL_TOP_M, 1.5);
    let cm = take_cover(&mut w, &mut b);
    println!(
        "entered: class {:?} anchor {:?} normal {:?} L {:.4} R {:.4} yaw {:.2}",
        cm.runtime.cover.class,
        cm.runtime.cover.anchor,
        cm.runtime.cover.normal,
        cm.runtime.cover.left_m,
        cm.runtime.cover.right_m,
        cm.runtime.cover.yaw_deg
    );
    let left = MovementIntent {
        move_input: EcsVec2d::new(-1.0, 0.0),
        ..Default::default()
    };
    for i in 0..120 {
        step(&mut w, &mut b, &left);
        if i < 14 || i % 10 == 0 {
            let c = hero(&w);
            println!(
                "{i:3} pos {:.4} {:.4} intent {:.3},{:.3} v {:.3},{:.3} along {:.4} L {:.4} R {:.4} anchor {:.4},{:.4} corner {} mode {:?}",
                hero_pos(&w).x,
                hero_pos(&w).z,
                c.runtime.intent_move.x,
                c.runtime.intent_move.y,
                c.runtime.velocity.x,
                c.runtime.velocity.z,
                c.runtime.cover.along_m,
                c.runtime.cover.left_m,
                c.runtime.cover.right_m,
                c.runtime.cover.anchor.x,
                c.runtime.cover.anchor.z,
                c.runtime.cover.at_corner,
                c.mode
            );
        }
    }
}
