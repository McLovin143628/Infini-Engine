//! **P29.4: the traversal detectors and the mantle**, driven end-to-end through
//! the real [`PhysicsBridge3D`] over an [`EcsWorld`].
//!
//! `movement_3d.rs`'s discipline, unchanged and for the same reason: **every arm
//! asserts the WORLD.** "Did the mantle happen" is answered by reading the
//! character's transform back out of the ECS and checking that it is standing on
//! top of the ledge — not by reading a mode, which a step that set the mode and
//! moved nothing would satisfy perfectly.
//!
//! Every ledge arm has a **control**: the same character, the same wall, one
//! thing changed so the mantle must NOT happen. A probe that answered `Some` for
//! everything would pass the positive half of all of these.

use glam::DVec3;
use inf_ecs::components::{
    BodyKind3D, CharacterMovement, Collider3D, ColliderShape3DKind, LandingKind, MovementMode,
    RigidBody3D, RotationMode, Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::movement::MovementIntent;
use inf_ecs::{EcsWorld, Vec2d};
use inf_physics::d3::{step_character_movement, traversal, CastTargets, ColliderShape3D};
use inf_physics::PhysicsBridge3D;
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const HERO: Uuid = Uuid::from_u128(0x2904_0001);
const GROUND: Uuid = Uuid::from_u128(0x2904_0002);
const LEDGE: Uuid = Uuid::from_u128(0x2904_0003);
const RADIUS: f64 = 0.3;

fn spawn_block(w: &mut EcsWorld, guid: Uuid, centre: DVec3, half: DVec3, kind: BodyKind3D) {
    let e = w.spawn_with_guid(guid, "Block", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::from_dvec3(centre);
    w.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind,
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

fn spawn_hero(w: &mut EcsWorld, feet_y: f64, z: f64) {
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let e = w.spawn_with_guid(HERO, "Hero", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, feet_y + cm.stand_half_height_m + RADIUS, z);
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
        inf_ecs::components::CharacterController3D::default(),
        cm,
        t,
    ));
    w.mark_dirty();
    w.propagate();
}

fn step(w: &mut EcsWorld, b: &mut PhysicsBridge3D, intent: &MovementIntent) {
    b.sync_from_world(w);
    inf_ecs::movement::apply_intent(w, intent);
    step_character_movement(w, b, DT);
}

fn hero(w: &EcsWorld) -> CharacterMovement {
    let e = w.entity_of(HERO).unwrap();
    w.world().get::<CharacterMovement>(e).unwrap().clone()
}

fn hero_pos(w: &EcsWorld) -> DVec3 {
    let e = w.entity_of(HERO).unwrap();
    w.world()
        .get::<Transform>(e)
        .unwrap()
        .translation
        .to_dvec3()
}

fn hero_feet(w: &EcsWorld) -> f64 {
    let e = w.entity_of(HERO).unwrap();
    let half = w.world().get::<Collider3D>(e).unwrap().half_extents.y;
    hero_pos(w).y - half - RADIUS
}

fn forward() -> MovementIntent {
    MovementIntent {
        move_input: Vec2d::new(0.0, 1.0),
        ..Default::default()
    }
}

fn forward_jump() -> MovementIntent {
    MovementIntent {
        move_input: Vec2d::new(0.0, 1.0),
        jump: true,
        ..Default::default()
    }
}

/// A world with a floor and one ledge of height `h` whose face is at `z = 2.0`.
///
/// The hero stands at `z = 1.2`, which is 0.8 m back from the face — inside the
/// default 0.75 m reach once the probe's own 0.30 m of back-off is counted.
fn world_with_ledge(h: f64, kind: BodyKind3D) -> (EcsWorld, PhysicsBridge3D) {
    let mut w = EcsWorld::new();
    let b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(10.0, 0.5, 10.0),
        BodyKind3D::Static,
    );
    // The ledge: a block whose top is at `h` and whose face is at z = 2.0.
    spawn_block(
        &mut w,
        LEDGE,
        DVec3::new(0.0, h / 2.0, 2.0 + 2.0),
        DVec3::new(4.0, h / 2.0, 2.0),
        kind,
    );
    spawn_hero(&mut w, 0.0, 1.2);
    (w, b)
}

/// Run `steps` fixed steps with `intent` and answer where the feet ended.
fn run(w: &mut EcsWorld, b: &mut PhysicsBridge3D, steps: usize, intent: &MovementIntent) {
    for _ in 0..steps {
        step(w, b, intent);
    }
}

// ── the probe itself ────────────────────────────────────────────────────────

/// The character's own collider, which every probe must be blind to.
///
/// The forward sweeper starts 30 cm BEHIND the feet and is 2.6 m tall, so it
/// begins inside the very character it is probing for; a sweep that started
/// penetrating is refused (there is no wall face to read a normal off), so
/// without this the probe answers `None` for every ledge in the world. The
/// movement step passes the same exclusion — this is that call's contract, made
/// visible.
fn only_self(b: &PhysicsBridge3D) -> std::collections::BTreeSet<inf_physics::d3::ColliderId3D> {
    let mut set = std::collections::BTreeSet::new();
    if let Some(c) = b.collider_of(HERO) {
        set.insert(c);
    }
    set
}

/// **The five steps, each rejection reachable.** The probe is asked directly,
/// because a `None` from inside the movement step is indistinguishable from a
/// mantle that was never attempted.
#[test]
fn the_ledge_probe_finds_a_wall_and_refuses_five_things_that_are_not_one() {
    let settings = traversal::LedgeSettings::default();
    let feet = DVec3::new(0.0, 0.0, 1.2);
    let fwd = DVec3::Z;

    // A real 1 m ledge is found, classified low, and faces back at the character.
    let (mut w, mut b) = world_with_ledge(1.0, BodyKind3D::Static);
    b.sync_from_world(&w);
    let none = only_self(&b);
    let ledge = traversal::probe_ledge(
        b.world_mut(),
        feet,
        fwd,
        RADIUS,
        0.9,
        45.0,
        &settings,
        &none,
    )
    .expect("a 1 m ledge in reach must be found");
    assert!(
        (ledge.height_m - 1.0).abs() < 0.05,
        "height {}",
        ledge.height_m
    );
    assert!(!ledge.high, "1.0 m is under the 1.25 m split");
    assert!(ledge.feet.y > 0.9, "the target is on top: {:?}", ledge.feet);
    // Facing INTO the wall: the wall's outward normal is -Z, so the character
    // faces +Z, which is yaw 0 in this engine's convention.
    assert!(ledge.yaw_deg.abs() < 1.0, "yaw {}", ledge.yaw_deg);

    // A 2 m ledge is a HIGH mantle — the 1.25 m split, measured.
    let (mut w2, mut b2) = world_with_ledge(2.0, BodyKind3D::Static);
    b2.sync_from_world(&w2);
    let none = only_self(&b2);
    let high = traversal::probe_ledge(
        b2.world_mut(),
        feet,
        fwd,
        RADIUS,
        0.9,
        45.0,
        &settings,
        &none,
    )
    .expect("a 2 m ledge is still inside the 2.5 m ceiling");
    assert!(high.high, "2 m must classify high: {}", high.height_m);
    let _ = (&mut w, &mut w2);

    // 1. **Nothing there.** A floor and no wall.
    let mut empty = EcsWorld::new();
    let mut eb = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut empty,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(10.0, 0.5, 10.0),
        BodyKind3D::Static,
    );
    eb.sync_from_world(&empty);
    let nobody = std::collections::BTreeSet::new();
    assert!(
        traversal::probe_ledge(
            eb.world_mut(),
            feet,
            fwd,
            RADIUS,
            0.9,
            45.0,
            &settings,
            &nobody
        )
        .is_none(),
        "an open plain is not a ledge"
    );

    // 2. **Too low.** A 0.2 m kerb is the mover's autostep, not a mantle.
    let (mut low, mut lb) = world_with_ledge(0.2, BodyKind3D::Static);
    lb.sync_from_world(&low);
    let none = only_self(&lb);
    assert!(
        traversal::probe_ledge(
            lb.world_mut(),
            feet,
            fwd,
            RADIUS,
            0.9,
            45.0,
            &settings,
            &none
        )
        .is_none(),
        "a 20 cm kerb is below the minimum ledge height"
    );
    let _ = &mut low;

    // 3. **Too high.** A 4 m wall is past the ceiling.
    let (mut tall, mut tb) = world_with_ledge(4.0, BodyKind3D::Static);
    tb.sync_from_world(&tall);
    let none = only_self(&tb);
    assert!(
        traversal::probe_ledge(
            tb.world_mut(),
            feet,
            fwd,
            RADIUS,
            0.9,
            45.0,
            &settings,
            &none
        )
        .is_none(),
        "a 4 m wall is not climbable"
    );
    let _ = &mut tall;

    // 4. **A moving crate is not a ledge** — and the refusal is in the BROAD
    //    PHASE, which is the P22.3 M4 law: with the check downstream, the crate
    //    would hide the wall behind it instead of being ignored.
    let (mut dyn_w, mut db) = world_with_ledge(1.0, BodyKind3D::Dynamic);
    db.sync_from_world(&dyn_w);
    let none = only_self(&db);
    assert!(
        traversal::probe_ledge(
            db.world_mut(),
            feet,
            fwd,
            RADIUS,
            0.9,
            45.0,
            &settings,
            &none
        )
        .is_none(),
        "a dynamic body must not be mantled"
    );
    let _ = &mut dyn_w;

    // 5. **A degenerate facing** answers rather than dividing.
    let (mut ok, mut ob) = world_with_ledge(1.0, BodyKind3D::Static);
    ob.sync_from_world(&ok);
    let none = only_self(&ob);
    assert!(traversal::probe_ledge(
        ob.world_mut(),
        feet,
        DVec3::ZERO,
        RADIUS,
        0.9,
        45.0,
        &settings,
        &none
    )
    .is_none());
    // …and excluding the ledge's own collider is what `IgnoreOnlyPawn` does: with
    // it excluded there is nothing to find, which proves the exclusion is live.
    let ledge_collider = ob.collider_of(LEDGE).expect("the ledge has a collider");
    let mut skip = none.clone();
    skip.insert(ledge_collider);
    assert!(traversal::probe_ledge(
        ob.world_mut(),
        feet,
        fwd,
        RADIUS,
        0.9,
        45.0,
        &settings,
        &skip
    )
    .is_none());
    let _ = &mut ok;
}

/// The `targets` filter is **not** the exclusion set: a dynamic crate in front of
/// a static wall is *skipped* by `Fixed` and the wall behind it is returned, which
/// is the whole reason the filter is in the broad phase.
#[test]
fn the_fixed_filter_sees_past_a_dynamic_body_rather_than_being_blocked_by_it() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    // A static wall at z = 4, and a dynamic crate at z = 2 in front of it.
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, 1.0, 4.0),
        DVec3::new(4.0, 2.0, 0.5),
        BodyKind3D::Static,
    );
    spawn_block(
        &mut w,
        LEDGE,
        DVec3::new(0.0, 1.0, 2.0),
        DVec3::new(0.5, 0.5, 0.5),
        BodyKind3D::Dynamic,
    );
    b.sync_from_world(&w);
    let none = std::collections::BTreeSet::new();
    let probe = ColliderShape3D::Sphere { radius: 0.1 };
    let all = b
        .world_mut()
        .cast_shape(
            &probe,
            DVec3::new(0.0, 1.0, 0.0),
            glam::DQuat::IDENTITY,
            DVec3::Z,
            10.0,
            &none,
        )
        .expect("the crate is in the way");
    let fixed = b
        .world_mut()
        .cast_shape_where(
            &probe,
            DVec3::new(0.0, 1.0, 0.0),
            glam::DQuat::IDENTITY,
            DVec3::Z,
            10.0,
            &none,
            CastTargets::Fixed,
        )
        .expect("the wall is behind it");
    assert!(
        all.toi < 1.6,
        "the unfiltered cast hits the crate: {}",
        all.toi
    );
    assert!(
        fixed.toi > 3.0,
        "the filtered cast must see the WALL, not the crate: {}",
        fixed.toi
    );
    assert_ne!(all.collider, fixed.collider);
}

// ── the mantle, in the world ────────────────────────────────────────────────

/// **THE mantle arm.** A character jumps at a 1 m ledge holding forward, and
/// ends up **standing on top of it** — the world, not the mode.
///
/// The control is the same character at the same wall with **no movement input**,
/// which must jump instead: without it this arm would be satisfied by a probe
/// that fired on every jump.
#[test]
fn a_character_mantles_onto_a_ledge_and_a_jump_without_input_does_not() {
    for (height, label) in [(1.0_f64, "low"), (2.0_f64, "high")] {
        let (mut w, mut b) = world_with_ledge(height, BodyKind3D::Static);
        run(&mut w, &mut b, 5, &forward());
        let before = hero_feet(&w);
        assert!(
            before.abs() < 0.05,
            "{label}: starts on the floor: {before}"
        );

        step(&mut w, &mut b, &forward_jump());
        assert_eq!(
            hero(&w).mode,
            MovementMode::Mantle,
            "{label}: the jump at the ledge must start a mantle"
        );
        let m = hero(&w).runtime.mantle;
        assert!(m.active && m.duration_s > 0.0, "{label}: {m:?}");
        assert!(
            (m.height_m - height).abs() < 0.06,
            "{label}: probe height {} vs {height}",
            m.height_m
        );
        assert_eq!(
            m.high,
            height > 1.25,
            "{label}: the 1.25 m split classified it wrongly"
        );
        // A high mantle takes longer than a low one, and is entered at a
        // different point of its clip — ALS's height remap, kept.
        assert!(m.clip_start_s >= 0.0 && m.play_rate > 0.0, "{label}: {m:?}");

        // Run it out, then STOP. A character that kept walking would cross the
        // ledge and fall off its far side, and the arm would be measuring that.
        let mut ran = 0;
        while hero(&w).mode == MovementMode::Mantle && ran < 240 {
            step(&mut w, &mut b, &forward());
            ran += 1;
        }
        assert!(ran < 240, "{label}: the mantle never ended");
        step(&mut w, &mut b, &MovementIntent::default());
        // **THE ASSERTION**: the character is standing ON the ledge.
        let feet = hero_feet(&w);
        assert!(
            (feet - height).abs() < 0.08,
            "{label}: the character must end ON the ledge (feet {feet}, ledge {height})"
        );
        assert!(
            hero_pos(&w).z > 2.0,
            "{label}: …and past its face: z = {}",
            hero_pos(&w).z
        );
        assert_eq!(hero(&w).mode, MovementMode::Grounded, "{label}");
        assert!(!hero(&w).runtime.mantle.active, "{label}");

        // **THE CONTROL**: the same jump with no movement input is a jump.
        let (mut c, mut cb) = world_with_ledge(height, BodyKind3D::Static);
        run(&mut c, &mut cb, 5, &MovementIntent::default());
        step(
            &mut c,
            &mut cb,
            &MovementIntent {
                jump: true,
                ..Default::default()
            },
        );
        assert_ne!(
            hero(&c).mode,
            MovementMode::Mantle,
            "{label}: a jump with the stick centred is a jump, not a climb"
        );
    }
}

/// The warp **lands exactly**, and it does so monotonically: the feet rise every
/// step and never overshoot the ledge on the way.
///
/// A lerp between two transforms would pass "ends on the ledge" too. What it
/// would not pass is the endpoint being reached with the residual under a
/// millimetre while the path stays inside the two heights.
#[test]
fn the_warp_is_monotone_and_its_endpoint_is_exact() {
    let (mut w, mut b) = world_with_ledge(1.0, BodyKind3D::Static);
    run(&mut w, &mut b, 5, &forward());
    step(&mut w, &mut b, &forward_jump());
    assert_eq!(hero(&w).mode, MovementMode::Mantle);

    let mut last = hero_feet(&w);
    let mut steps = 0;
    while hero(&w).mode == MovementMode::Mantle && steps < 240 {
        step(&mut w, &mut b, &forward());
        let feet = hero_feet(&w);
        assert!(
            feet >= last - 1e-9,
            "step {steps}: the feet went DOWN during a mantle ({last} -> {feet})"
        );
        assert!(
            feet <= 1.0 + 1e-6,
            "step {steps}: the warp overshot the ledge: {feet}"
        );
        last = feet;
        steps += 1;
    }
    assert!(
        steps > 5,
        "a mantle that finishes in {steps} steps is a snap"
    );
    assert!(steps < 240, "the mantle never ended");
    // The endpoint is exact by construction — sub-millimetre, not "close".
    assert!(
        (hero_feet(&w) - 1.0).abs() < 1e-3,
        "the warp must land ON the target: {}",
        hero_feet(&w)
    );
    // …and the mantle left no velocity behind to launch the character with.
    let v = hero(&w).runtime.velocity;
    assert!(v.x.abs() < 1e-9 && v.z.abs() < 1e-9, "{v:?}");
}

/// **The falling catch** (ALS's automatic path): a character falling past a ledge
/// while holding forward catches it, with **no jump pressed**.
///
/// The fixture is a wall beside a pit, so the only way onto the ledge is to catch
/// it: a character that simply landed on top would never have been in the air
/// beside its face. The control is the same fall with the stick centred, which
/// must go all the way down.
#[test]
fn a_falling_character_holding_forward_catches_a_ledge() {
    fn fixture() -> (EcsWorld, PhysicsBridge3D) {
        let mut w = EcsWorld::new();
        let b = PhysicsBridge3D::new(GRAVITY);
        // A wall whose top is at y = 1, going down into a pit.
        spawn_block(
            &mut w,
            LEDGE,
            DVec3::new(0.0, -2.0, 4.0),
            DVec3::new(4.0, 3.0, 2.0),
            BodyKind3D::Static,
        );
        // The pit's floor, far below.
        spawn_block(
            &mut w,
            GROUND,
            DVec3::new(0.0, -8.0, 0.0),
            DVec3::new(20.0, 0.5, 20.0),
            BodyKind3D::Static,
        );
        // Falling, beside the wall's face.
        spawn_hero(&mut w, 1.6, 1.2);
        (w, b)
    }

    let (mut w, mut b) = fixture();
    let mut caught = false;
    for _ in 0..240 {
        step(&mut w, &mut b, &forward());
        assert!(
            !hero(&w).runtime.press_jump,
            "no jump is pressed anywhere in this arm"
        );
        if hero(&w).mode == MovementMode::Mantle {
            caught = true;
            break;
        }
    }
    assert!(
        caught,
        "a falling character holding forward must catch a ledge (ended at {:?})",
        hero_pos(&w)
    );
    let mut ran = 0;
    while hero(&w).mode == MovementMode::Mantle && ran < 240 {
        step(&mut w, &mut b, &forward());
        ran += 1;
    }
    step(&mut w, &mut b, &MovementIntent::default());
    assert_eq!(hero(&w).mode, MovementMode::Grounded);
    assert!(
        (hero_feet(&w) - 1.0).abs() < 0.1,
        "…and end standing on the ledge: {}",
        hero_feet(&w)
    );

    // **THE CONTROL**: the same fall with the stick centred goes down the pit.
    let (mut c, mut cb) = fixture();
    for _ in 0..240 {
        step(&mut c, &mut cb, &MovementIntent::default());
        assert_ne!(
            hero(&c).mode,
            MovementMode::Mantle,
            "a fall with no input must not catch anything"
        );
    }
    assert!(
        hero_feet(&c) < -6.0,
        "the control must reach the pit floor: {}",
        hero_feet(&c)
    );
}

// ── land prediction ─────────────────────────────────────────────────────────

/// **The prediction agrees with the classifier.** A character dropped from a
/// height is asked, every step, what it thinks the landing will be — and the
/// answer it gives before the touch is the answer the classifier gives at it.
///
/// This is the arm that makes the sweep worth having: an in-air animation can
/// prepare for a hard landing only if "hard" is knowable early.
#[test]
fn the_land_prediction_agrees_with_the_classifier_before_the_touch() {
    for drop_m in [1.0_f64, 4.0, 9.0] {
        let mut w = EcsWorld::new();
        let mut b = PhysicsBridge3D::new(GRAVITY);
        spawn_block(
            &mut w,
            GROUND,
            DVec3::new(0.0, -0.5, 0.0),
            DVec3::new(20.0, 0.5, 20.0),
            BodyKind3D::Static,
        );
        spawn_hero(&mut w, drop_m, 0.0);

        let mut last_prediction = LandingKind::None;
        let mut first_prediction = LandingKind::None;
        let mut first_alpha = 1.0f64;
        let mut landed = LandingKind::None;
        for _ in 0..300 {
            let before = hero(&w).runtime.grounded;
            step(&mut w, &mut b, &MovementIntent::default());
            let h = hero(&w);
            if !h.runtime.grounded && h.runtime.predicted_landing != LandingKind::None {
                if first_prediction == LandingKind::None {
                    first_prediction = h.runtime.predicted_landing;
                    first_alpha = h.runtime.land_alpha;
                }
                last_prediction = h.runtime.predicted_landing;
                assert!(
                    h.runtime.land_alpha > 0.0 && h.runtime.land_alpha <= 1.0,
                    "alpha out of range: {}",
                    h.runtime.land_alpha
                );
            }
            if h.runtime.grounded && !before {
                landed = h.runtime.landing;
                break;
            }
        }
        assert_ne!(
            first_prediction,
            LandingKind::None,
            "{drop_m} m: nothing was ever predicted"
        );
        assert_ne!(landed, LandingKind::None, "{drop_m} m: never landed");
        assert_eq!(
            last_prediction, landed,
            "{drop_m} m: the prediction and the classifier must agree"
        );
        // **The FIRST prediction is already right**, which is the whole reason
        // the sweep exists: an animation can only prepare for a hard landing if
        // "hard" is knowable while the character is still a long way up. The
        // sweep reports the speed the character WILL arrive at, not the one it
        // has — and this is the half that tells the two apart, because near the
        // ground they are the same number.
        assert_eq!(
            first_prediction, landed,
            "{drop_m} m: the first prediction (alpha {first_alpha:.3}) already has to be the verdict — reporting the CURRENT speed rather than the arrival speed passes at the last instant and fails here"
        );
        // …and a grounded character predicts nothing at all: a prediction is a
        // statement about THIS step.
        for _ in 0..5 {
            step(&mut w, &mut b, &MovementIntent::default());
        }
        assert_eq!(hero(&w).runtime.predicted_landing, LandingKind::None);
        assert_eq!(hero(&w).runtime.land_alpha, 0.0);
    }
}

// ── turn in place ───────────────────────────────────────────────────────────

/// **Turn in place**, in the world: a standing character whose camera has moved
/// 90° away and settled turns its body to follow, after a delay that scales with
/// the angle — and does **not** turn while the camera is still moving.
#[test]
fn a_standing_character_turns_in_place_only_once_the_camera_settles() {
    fn setup() -> (EcsWorld, PhysicsBridge3D) {
        let mut w = EcsWorld::new();
        let b = PhysicsBridge3D::new(GRAVITY);
        spawn_block(
            &mut w,
            GROUND,
            DVec3::new(0.0, -0.5, 0.0),
            DVec3::new(20.0, 0.5, 20.0),
            BodyKind3D::Static,
        );
        spawn_hero(&mut w, 0.0, 0.0);
        let e = w.entity_of(HERO).unwrap();
        w.world_mut()
            .get_mut::<CharacterMovement>(e)
            .unwrap()
            .rotation_mode = RotationMode::LookingDirection;
        (w, b)
    }

    // Swing the camera 90 degrees over a second, then hold it still.
    let (mut w, mut b) = setup();
    let look = MovementIntent {
        look_yaw_dps: 90.0,
        ..Default::default()
    };
    for _ in 0..60 {
        step(&mut w, &mut b, &look);
    }
    let aim = hero(&w).runtime.aim_yaw_deg;
    assert!(
        (aim - 90.0).abs() < 5.0,
        "the camera should have swung 90 degrees: {aim}"
    );
    // **The control**: while the camera is still moving, the body has not
    // started turning — the rate gate.
    assert!(
        !hero(&w).runtime.turning_in_place,
        "a turn must not begin while the camera is still moving"
    );
    let body_while_looking = hero(&w).runtime.body_yaw_deg;
    assert!(
        body_while_looking.abs() < 2.0,
        "the body should not have followed yet: {body_while_looking}"
    );

    // Let go of the stick: the camera settles, the delay accumulates, the turn
    // fires and the body arrives.
    let mut started = false;
    for _ in 0..300 {
        step(&mut w, &mut b, &MovementIntent::default());
        started |= hero(&w).runtime.turning_in_place;
        if !hero(&w).runtime.turning_in_place && started {
            break;
        }
    }
    assert!(started, "the turn never began once the camera settled");
    let body = hero(&w).runtime.body_yaw_deg;
    assert!(
        inf_ecs::movement::angle_delta_deg(body, aim).abs() < 2.0,
        "the body must arrive at the aim: body {body}, aim {aim}"
    );
    // The world agrees with the runtime — the transform is what is drawn.
    let e = w.entity_of(HERO).unwrap();
    let drawn = w.world().get::<Transform>(e).unwrap().rotation.y;
    assert!((drawn - body).abs() < 1e-9, "{drawn} vs {body}");

    // **A small offset never fires at all**: 20 degrees is under the 45 gate.
    let (mut small, mut sb) = setup();
    let nudge = MovementIntent {
        look_yaw_dps: 20.0,
        ..Default::default()
    };
    for _ in 0..60 {
        step(&mut small, &mut sb, &nudge);
    }
    for _ in 0..300 {
        step(&mut small, &mut sb, &MovementIntent::default());
        assert!(
            !hero(&small).runtime.turning_in_place,
            "a 20-degree offset must not turn the body"
        );
    }
}

/// **Rotate in place** is the aiming half, and it is a *play rate* rather than a
/// turn: the gates open past 50° and the rate tracks the camera.
#[test]
fn an_aiming_character_rotates_in_place_instead_of_turning() {
    let mut w = EcsWorld::new();
    let mut b = PhysicsBridge3D::new(GRAVITY);
    spawn_block(
        &mut w,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(20.0, 0.5, 20.0),
        BodyKind3D::Static,
    );
    spawn_hero(&mut w, 0.0, 0.0);
    let aim_and_look = MovementIntent {
        aim: true,
        look_yaw_dps: 120.0,
        ..Default::default()
    };
    let mut saw_right = false;
    let mut fastest = 0.0f64;
    for _ in 0..60 {
        step(&mut w, &mut b, &aim_and_look);
        let h = hero(&w);
        saw_right |= h.runtime.rotate_right;
        fastest = fastest.max(h.runtime.rotate_rate);
        assert!(
            !h.runtime.turning_in_place,
            "an aiming character rotates, it does not turn in place"
        );
    }
    assert!(saw_right, "the right-rotate gate never opened");
    assert!(
        fastest > 1.15,
        "the play rate must track the camera: {fastest}"
    );
    assert_eq!(hero(&w).rotation_mode, RotationMode::Aiming);
}

// ── A8: the slope authority, closed (P29.3 audit, declined and carried) ─────

/// **The two numbers that meant one thing.**
///
/// `mover_for` rules that a character's `CharacterMovement::slope_limit_deg`
/// overrides its `CharacterController3D::max_slope_deg`. The P29.3 audit declined
/// to close it and said why: the ruling had **no arm**, and a mutation that
/// ignores it kills nothing because both defaults are 45 degrees, so the mutation
/// is a no-op on every fixture in the tree. Closing it needed a ramp fixture and a
/// public reader for the mover's slope, and neither existed.
///
/// Both exist now. This arm has three halves:
///
/// 1. the **reader** says the component's number reached the mover;
/// 2. the **world** agrees — a 40-degree ramp is climbed by a character whose
///    component allows 50 and refused by one whose component allows 30, with the
///    *controller's* number held at 45 in both, so only the override can explain
///    the difference;
/// 3. the **control**: with no movement component at all the controller's number
///    is what applies, which is the clause `mover_for` documents and the half a
///    test of the override alone would not see.
#[test]
fn the_movement_components_slope_limit_is_the_one_that_reaches_the_mover() {
    /// A ramp of `deg` degrees rising along +Z from the origin, as a rotated box.
    fn ramp(w: &mut EcsWorld, guid: Uuid, deg: f64) {
        let e = w.spawn_with_guid(guid, "Ramp", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, 0.0, 6.0);
        // `Transform::rotation` is euler DEGREES (YXZ) — a positive X rotation
        // tips the far end of the box up.
        t.rotation.x = -deg;
        w.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(6.0, 0.25, 8.0),
                ..Default::default()
            },
            t,
        ));
        w.mark_dirty();
        w.propagate();
    }

    fn run(component_slope_deg: Option<f64>) -> (f64, f64) {
        let mut w = EcsWorld::new();
        let mut b = PhysicsBridge3D::new(GRAVITY);
        spawn_block(
            &mut w,
            GROUND,
            DVec3::new(0.0, -0.5, -2.0),
            DVec3::new(6.0, 0.5, 4.0),
            BodyKind3D::Static,
        );
        ramp(&mut w, LEDGE, 40.0);
        spawn_hero(&mut w, 0.0, 0.0);
        {
            let e = w.entity_of(HERO).unwrap();
            // The CONTROLLER's number stays at its 45-degree default in every
            // case, so nothing below can be explained by changing it.
            assert_eq!(
                w.world()
                    .get::<inf_ecs::components::CharacterController3D>(e)
                    .unwrap()
                    .max_slope_deg,
                45.0
            );
            match component_slope_deg {
                Some(deg) => {
                    w.world_mut()
                        .get_mut::<CharacterMovement>(e)
                        .unwrap()
                        .slope_limit_deg = deg;
                }
                None => {
                    // No movement component at all: the controller's number is
                    // what applies, which is `mover_for`'s other clause.
                    w.world_mut().entity_mut(e).remove::<CharacterMovement>();
                }
            }
        }
        b.sync_from_world(&w);
        let mover_deg = inf_physics::d3::mover_for(&w, HERO)
            .slope_limit_rad()
            .to_degrees();
        if component_slope_deg.is_none() {
            return (mover_deg, 0.0);
        }
        for _ in 0..240 {
            step(&mut w, &mut b, &forward());
        }
        (mover_deg, hero_feet(&w))
    }

    // 1. THE READER. The component's number is the one on the mover.
    let (steep, climbed) = run(Some(50.0));
    assert!(
        (steep - 50.0).abs() < 1e-9,
        "the component's 50 degrees did not reach the mover: {steep}"
    );
    let (shallow, refused) = run(Some(30.0));
    assert!(
        (shallow - 30.0).abs() < 1e-9,
        "the component's 30 degrees did not reach the mover: {shallow}"
    );

    // 2. THE WORLD. A 40-degree ramp is climbed at a 50-degree limit and not at
    //    a 30-degree one — the same ramp, the same character, the same
    //    controller, one number apart.
    assert!(
        climbed > 0.5,
        "a 40-degree ramp must be climbable at a 50-degree limit: feet {climbed}"
    );
    assert!(
        refused < 0.25,
        "a 40-degree ramp must NOT be climbable at a 30-degree limit: feet {refused}"
    );
    assert!(
        climbed > refused + 0.4,
        "climbed {climbed} vs refused {refused} — the limit changed nothing"
    );

    // 3. THE CONTROL. With no movement component the CONTROLLER's number applies,
    //    which is the clause an override-only arm would not see.
    let (fallback, _) = run(None);
    assert!(
        (fallback - 45.0).abs() < 1e-9,
        "without a movement component the controller's 45 degrees must apply: {fallback}"
    );
}
