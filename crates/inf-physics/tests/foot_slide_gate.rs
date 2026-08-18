//! **THE FOOT-SLIDE GATE** (P29.4, clause 5) — a number, in **metres**, with a
//! control that must fail it.
//!
//! §13's "done when" clause names foot slide measured in metres as one of the
//! phase's acceptance criteria, and this is the arm that holds it. The claim is
//! narrow and checkable: *while a clip says a foot is planted, that foot does not
//! move across the world.*
//!
//! # Why the metric is a POSITION and not the lock's own report
//!
//! `FootLock::slide_m` answers `0.0` for a foot that is not locked — an unplanted
//! foot makes no claim, which is the honest answer for that function and useless
//! for a gate: a mechanism that never engaged would score a perfect zero. So the
//! measurement is the **drawn foot's world displacement** over the planted
//! window, which is a number both halves of the comparison produce.
//!
//! # The control
//!
//! The same character, the same walk, the same clip — with `FootLock_L` authored
//! below the engage threshold, which is exactly what a rig without the channel
//! (or an animator who has not authored it) gets. The foot then travels with the
//! body, and the arm asserts it does: without that half, a gate that measured
//! zero because nothing was moving at all would look identical.

use std::collections::BTreeMap;

use glam::DVec3;
use inf_anim::{
    channels::als, AnimClip, CurveChannel, Interpolation, Joint, JointTransform, Skeleton,
    SkeletonAsset, SmState, StateMachine,
};
use inf_ecs::components::{
    AnimStateMachine, BodyKind3D, CharacterMovement, Collider3D, ColliderShape3DKind, RigidBody3D,
    SkeletalMesh, Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::movement::MovementIntent;
use inf_ecs::EcsWorld;
use inf_physics::d3::step_character_movement;
use inf_physics::PhysicsBridge3D;
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const HERO: Uuid = Uuid::from_u128(0x2904_2001);
const GROUND: Uuid = Uuid::from_u128(0x2904_2002);
const SKEL: Uuid = Uuid::from_u128(0x2904_2003);
const SM: Uuid = Uuid::from_u128(0x2904_2004);
const CLIP: inf_anim::ClipRef = [11; 16];
const RADIUS: f64 = 0.3;

/// A two-leg rig: hips, then thigh → shin → foot on each side, with the foot
/// joints named so [`inf_ecs::pose`]'s matcher finds them.
fn legs() -> SkeletonAsset {
    fn joint(name: &str, parent: Option<u16>, local: glam::Vec3) -> Joint {
        Joint {
            name: name.into(),
            parent,
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::from_trs(local, glam::Quat::IDENTITY, glam::Vec3::ONE),
        }
    }
    SkeletonAsset::new(
        Skeleton::new(vec![
            joint("Hips", None, glam::Vec3::new(0.0, 1.0, 0.0)),
            joint("Thigh.L", Some(0), glam::Vec3::new(0.1, -0.05, 0.0)),
            joint("Shin.L", Some(1), glam::Vec3::new(0.0, -0.45, 0.0)),
            joint("Foot.L", Some(2), glam::Vec3::new(0.0, -0.45, 0.0)),
            joint("Thigh.R", Some(0), glam::Vec3::new(-0.1, -0.05, 0.0)),
            joint("Shin.R", Some(4), glam::Vec3::new(0.0, -0.45, 0.0)),
            joint("Foot.R", Some(5), glam::Vec3::new(0.0, -0.45, 0.0)),
        ])
        .expect("a valid pair of legs"),
    )
}

/// A one-second looping walk whose LEFT foot is authored planted for the first
/// half — with `plant` deciding whether the lock channel ever reaches the engage
/// threshold, which is the whole of the control.
fn walk_clip(plant: bool) -> AnimClip {
    walk_clip_at(plant, HIPS_FEET_AT_ORIGIN)
}

/// The hip height that puts this rig's **feet at model y = 0** — the leg chain's
/// own length (0.05 + 0.45 + 0.45).
///
/// P29.6 ruled that pose space **is** feet-at-origin character space, so this is
/// the convention every fixture in this file authors against and the publisher
/// (`inf_ecs::pose::model_to_world`) is what subtracts the worn capsule. Before
/// the ruling these fixtures had to put the hips at the capsule's *centre* to
/// reach the floor at all, which is the seam the P29.4 audit pinned.
const HIPS_FEET_AT_ORIGIN: f32 = 0.95;

/// [`walk_clip`] with the hips placed at `hips_y` in **model** space.
///
/// The height is the clip's, not the skeleton's: joint 0 is animated, so its
/// bind pose is irrelevant and the clip is the only thing that decides where the
/// rig's feet end up. [`HIPS_FEET_AT_ORIGIN`] is the convention; a larger value
/// stands the ankle that much above the ground the character is on.
fn walk_clip_at(plant: bool, hips_y: f32) -> AnimClip {
    walk_clip_ik(plant, hips_y, 1.0)
}

/// [`walk_clip_at`] with `Enable_FootIK` authored at `ik` on both feet.
///
/// **Why the slide gate turns it off** (P29.6). Foot IK and the foot lock are two
/// different mechanisms, and until this wave they could not be confused because
/// the fixture's feet stood 0.95 m in the air — outside ALS's probe envelope — so
/// no goal was ever published (that was the A12 seam). Character space puts the
/// feet on the floor, which turns foot IK **on** for these fixtures, and the goal
/// it solves to is one fixed step **stale** (the movement step reads the feet the
/// pose step published last time), so it is itself a partial anti-slide: measured
/// at **0.229 m** of skate against the lock-free control's 0.892 m. The gate below
/// measures the *lock*, so it authors the IK off and its numbers are the P29.4
/// ledger's unchanged; `foot_ik_alone_is_a_partial_brake_and_not_the_lock` records
/// the other number rather than leaving it as a mystery.
fn walk_clip_ik(plant: bool, hips_y: f32, ik: f32) -> AnimClip {
    let lock_value = if plant { 1.0 } else { 0.6 };
    // **The clip needs a DURATION**, and `AnimClip::new` derives it from the
    // joint keys — so a clip with curve channels and no tracks is a clip whose
    // play-head never leaves zero, and whose curves therefore never change. Two
    // identical keys a second apart give it a timeline without giving it a pose.
    // (Found by this arm: the lock engaged and never released, because the clip
    // was one frame long and that frame said "planted".)
    let track = inf_anim::JointTrack {
        joint: 0,
        translation: Some(inf_anim::Vec3Track::new(
            vec![0.0, 1.0],
            vec![[0.0, hips_y, 0.0], [0.0, hips_y, 0.0]],
            Interpolation::Linear,
        )),
        rotation: None,
        scale: None,
    };
    AnimClip::new("walk", vec![track]).with_curves(vec![
        CurveChannel::constant(als::ENABLE_FOOT_IK_L, ik),
        CurveChannel::constant(als::ENABLE_FOOT_IK_R, ik),
        // Planted for the first half of the cycle, swinging for the second.
        CurveChannel::new(
            als::FOOT_LOCK_L,
            vec![0.0, 0.5, 0.5001, 1.0],
            vec![lock_value, lock_value, 0.0, 0.0],
            Interpolation::Step,
        ),
        CurveChannel::constant(als::FOOT_LOCK_R, 0.0),
    ])
}

struct Sim {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    skeleton: SkeletonAsset,
    machine: StateMachine,
    clip: AnimClip,
}

impl Sim {
    fn new(plant: bool) -> Self {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(GROUND, "Ground", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, -0.5, 0.0);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(60.0, 0.5, 60.0),
                ..Default::default()
            },
            t,
        ));
        let cm = CharacterMovement {
            player_controlled: true,
            ..Default::default()
        };
        let e = world.spawn_with_guid(HERO, "Hero", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, cm.stand_half_height_m + RADIUS, 0.0);
        world.world_mut().entity_mut(e).insert((
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
            AnimStateMachine {
                sm: Some(SM),
                ..Default::default()
            },
            SkeletalMesh {
                mesh: Some(Uuid::from_u128(1)),
                skeleton: Some(SKEL),
            },
            t,
        ));
        world.mark_dirty();
        world.propagate();
        Self {
            world,
            bridge: PhysicsBridge3D::new(GRAVITY),
            skeleton: legs(),
            machine: StateMachine {
                states: vec![SmState::clip("walk", CLIP)],
                entry: 0,
                ..Default::default()
            },
            clip: walk_clip(plant),
        }
    }

    fn step(&mut self, intent: &MovementIntent) {
        self.bridge.sync_from_world(&self.world);
        inf_ecs::movement::apply_intent(&mut self.world, intent);
        step_character_movement(&mut self.world, &mut self.bridge, DT);
        self.world.propagate();
        let (machine, skeleton, clip) = (&self.machine, &self.skeleton, &self.clip);
        let machines = |g: Uuid| (g == SM).then_some(machine);
        let skels = |g: Uuid| (g == SKEL).then_some(skeleton);
        let clips = |c: inf_anim::ClipRef| (c == CLIP).then_some(clip);
        let vars = |_: Uuid| BTreeMap::new();
        inf_ecs::pose::step_pose_evaluation(&mut self.world, DT, &machines, &skels, &clips, &vars);
    }

    fn hero(&self) -> CharacterMovement {
        let e = self.world.entity_of(HERO).unwrap();
        self.world
            .world()
            .get::<CharacterMovement>(e)
            .unwrap()
            .clone()
    }
}

/// Walk forward for `steps` and answer the **worst ground-plane displacement of
/// the drawn left foot across a single planted window**, in metres, along with
/// how far the body itself travelled.
fn worst_slide(plant: bool, steps: usize) -> (f64, f64) {
    worst_slide_ik(plant, steps, 0.0)
}

/// [`worst_slide`] with `Enable_FootIK` authored at `ik`.
///
/// The gate calls it with **0**: it measures the *lock*, and character space
/// (P29.6) put these feet inside the IK envelope for the first time, so the two
/// mechanisms would otherwise be measured together. See [`walk_clip_ik`].
fn worst_slide_ik(plant: bool, steps: usize, ik: f32) -> (f64, f64) {
    let mut sim = Sim::new(plant);
    sim.clip = walk_clip_ik(plant, HIPS_FEET_AT_ORIGIN, ik);
    let forward = MovementIntent {
        move_input: inf_ecs::Vec2d::new(0.0, 1.0),
        ..Default::default()
    };
    // Let the machine start and the bridge fill: the movement step reads the feet
    // the pose step published, so nothing is measurable on step one.
    for _ in 0..3 {
        sim.step(&forward);
    }
    let start_body = sim.world.entity_of(HERO).unwrap();
    let body0 = sim
        .world
        .world()
        .get::<Transform>(start_body)
        .unwrap()
        .translation
        .to_dvec3();

    let mut worst = 0.0f64;
    let mut anchor: Option<DVec3> = None;
    let mut was_locked = false;
    for _ in 0..steps {
        sim.step(&forward);
        let h = sim.hero();
        let locked = h.runtime.foot_lock_l.locked && h.runtime.foot_lock_l.alpha > 0.0;
        let foot = h.runtime.foot_world_l.to_dvec3();
        if plant {
            if locked && !was_locked {
                // A new plant: the window opens here.
                anchor = Some(foot);
            }
            if !locked {
                anchor = None;
            }
        } else {
            // The control's lock never engages, so the window is the AUTHORED
            // one: the same question, asked of a mechanism that is not running.
            let planted =
                inf_ecs::anim_bridge::anim_curve(&sim.world, HERO, als::FOOT_LOCK_L, 0.0) > 0.5;
            if planted {
                anchor.get_or_insert(foot);
            } else {
                anchor = None;
            }
        }
        if let Some(a) = anchor {
            let d = foot - a;
            worst = worst.max((d.x * d.x + d.z * d.z).sqrt());
        }
        was_locked = locked;
    }
    let body1 = sim
        .world
        .world()
        .get::<Transform>(start_body)
        .unwrap()
        .translation
        .to_dvec3();
    (worst, (body1 - body0).length())
}

/// **THE GATE.** A planted foot holds its place in the world while the body walks
/// away from it — and the same walk without the lock does not.
#[test]
fn a_planted_foot_does_not_slide_and_the_unlocked_control_does() {
    // 30 steps is half a second: the first half of the authored cycle, which is
    // exactly the planted window.
    let (locked_slide, locked_travel) = worst_slide(true, 28);
    let (loose_slide, loose_travel) = worst_slide(false, 28);

    // The control must actually be a control: the body really walked.
    assert!(
        locked_travel > 0.5 && loose_travel > 0.5,
        "the body must move, or neither number means anything: {locked_travel} / {loose_travel}"
    );

    // **THE NUMBER**, in metres. Measured on this machine over a 1.017 m walk:
    // **0.0 m exactly** locked against **0.892 m** unlocked. P29.4 measured
    // 7e-9 m here and this is the same mechanism unperturbed: character space
    // (P29.6) brought the feet inside the foot-IK envelope, so the gate now
    // authors the IK off to isolate the lock, and with nothing else touching the
    // pose the drawn foot IS the lock's own world position — the residue that
    // produced 7e-9 was the inert-but-present solve. The bound stays 1e-5 m,
    // which is what a real regression fails and floating-point weather does not.
    const BOUND_M: f64 = 1.0e-5;
    assert!(
        locked_slide <= BOUND_M,
        "a planted foot slid {locked_slide:.9} m (the bound is {BOUND_M} m)"
    );

    // **THE FALSIFICATION.** Without the lock the same foot travels with the
    // body — orders of magnitude more, not a little more.
    assert!(
        loose_slide > 0.5,
        "the control must skate, or the gate above proves nothing: {loose_slide:.6} m"
    );
    assert!(
        loose_slide > locked_slide * 1000.0,
        "locked {locked_slide:.9} m vs unlocked {loose_slide:.6} m — not a separation"
    );
    // …and the skate really is the body's own travel, not some other motion: an
    // unlocked foot goes where the character goes.
    assert!(
        (loose_slide - loose_travel).abs() < 0.25,
        "the unlocked foot travelled {loose_slide:.3} m while the body went {loose_travel:.3} m"
    );
}

/// **Foot IK is a partial brake, and it is not the lock** (P29.6).
///
/// A finding the character-space ruling exposed, written down rather than left
/// as a discrepancy between two runs of the same fixture. The goal the movement
/// step publishes is read off the feet the *pose* step published last step — one
/// fixed step stale, by the ordering P29.4's ledger records — so solving to it
/// drags a swinging foot back toward where it was, and the skate falls by roughly
/// a factor of four. That is a long way from a lock (eight orders of magnitude
/// below) and it is not nothing, which is exactly why the gate above authors the
/// IK off: a control that quietly carried a second anti-slide mechanism would be
/// measuring the pair and reporting the lock.
#[test]
fn foot_ik_alone_is_a_partial_brake_and_not_the_lock() {
    let (bare, travel) = worst_slide_ik(false, 28, 0.0);
    let (with_ik, _) = worst_slide_ik(false, 28, 1.0);
    assert!(travel > 0.5, "the body must move: {travel}");
    assert!(
        bare > with_ik * 2.0,
        "foot IK did not brake at all: {bare:.3} m bare against {with_ik:.3} m \
         with it — if these are equal the goal is no longer being published, \
         which means character space stopped reaching the probe"
    );
    assert!(
        with_ik > 0.05,
        "foot IK held the foot still ({with_ik:.6} m) — it is not a lock and must \
         not read like one"
    );
}

/// The lock is **released** when the clip stops planting, and the foot then
/// returns to the animation rather than staying pinned for ever.
#[test]
fn the_lock_releases_when_the_clip_stops_planting() {
    let mut sim = Sim::new(true);
    let forward = MovementIntent {
        move_input: inf_ecs::Vec2d::new(0.0, 1.0),
        ..Default::default()
    };
    let mut ever_locked = false;
    let mut ever_released = false;
    for _ in 0..90 {
        sim.step(&forward);
        let locked = sim.hero().runtime.foot_lock_l.locked;
        ever_locked |= locked;
        if ever_locked && !locked {
            ever_released = true;
        }
    }
    assert!(ever_locked, "the lock never engaged");
    assert!(
        ever_released,
        "the lock never released — a foot pinned for ever is a foot dragged behind the character"
    );
    // …and it re-locks on the next cycle, which is the "or lock to a new
    // position" half of the rule.
    let mut relocked = false;
    for _ in 0..90 {
        sim.step(&forward);
        if sim.hero().runtime.foot_lock_l.locked {
            relocked = true;
            break;
        }
    }
    assert!(relocked, "the lock never engaged again");
}

/// **A turn breaks the lock** (P29.4 audit, A2) — ALS's `|RotationAmount| <=
/// 0.001` gate, ported as the body's own measured turn.
///
/// A pinned foot under a rotating hip points the wrong way, so the lock is
/// released for as long as the body is turning. The rule is one `||` in
/// `step_feet`, and the audit's mutation that replaced it with `false` killed
/// nothing in the tree: every fixture walked in a straight line.
///
/// Both halves, on one fixture: walking straight the same clip locks, and
/// walking while the camera swings it never does.
#[test]
fn a_turning_body_never_locks_a_foot() {
    let straight = MovementIntent {
        move_input: inf_ecs::Vec2d::new(0.0, 1.0),
        ..Default::default()
    };
    let turning = MovementIntent {
        look_yaw_dps: 120.0,
        ..straight
    };

    // The control: the same clip, the same walk, no turn — it locks.
    let mut plain = Sim::new(true);
    let mut locked_straight = false;
    for _ in 0..60 {
        plain.step(&straight);
        locked_straight |= plain.hero().runtime.foot_lock_l.locked;
    }
    assert!(
        locked_straight,
        "the control must lock, or the claim below is about a mechanism that \
         never runs"
    );

    // The claim: turning, it never does — on any step, for either foot.
    let mut spun = Sim::new(true);
    for i in 0..60 {
        spun.step(&turning);
        let h = spun.hero();
        assert!(
            !h.runtime.foot_lock_l.locked && !h.runtime.foot_lock_r.locked,
            "step {i}: a foot locked while the body was turning under it"
        );
        assert_eq!(h.runtime.foot_slide_l_m, 0.0);
    }
    // …and the turn really was one: the camera moved far enough for the gate to
    // be the thing that stopped it, not an idle body.
    assert!(
        spun.hero().runtime.aim_yaw_rate_dps.abs() > 100.0,
        "the fixture never turned: {}",
        spun.hero().runtime.aim_yaw_rate_dps
    );
}

/// **A character that loses its machine loses its published feet** (P29.4 audit,
/// A3).
///
/// `step_pose_evaluation`'s no-targets early return dropped the state, the root
/// motion and the curves and kept the **feet** — the one published map whose
/// reader is in the *other* fixed step. A character that stopped carrying a
/// machine therefore went on being locked to the last place a pose it no longer
/// has had put its foot, which is `PoseStoreRes`'s rule 4 in the one shape it
/// was not applied in.
#[test]
fn unbinding_the_machine_drops_the_published_feet() {
    let mut sim = Sim::new(true);
    let forward = MovementIntent {
        move_input: inf_ecs::Vec2d::new(0.0, 1.0),
        ..Default::default()
    };
    for _ in 0..10 {
        sim.step(&forward);
    }
    assert!(
        inf_ecs::anim_bridge::feet_of(&sim.world, HERO).is_some(),
        "the fixture must publish feet, or this arm asserts nothing"
    );
    assert!(sim.hero().runtime.foot_lock_l.locked, "…and lock one");

    // Take the machine away: the early return is the *other* way a level reaches
    // "nothing poses", and the pruning the main path does never runs there.
    let e = sim.world.entity_of(HERO).unwrap();
    sim.world
        .world_mut()
        .entity_mut(e)
        .remove::<AnimStateMachine>();
    sim.step(&forward);
    assert!(
        inf_ecs::anim_bridge::feet_of(&sim.world, HERO).is_none(),
        "a foot published by a machine that is gone is a stale answer"
    );
    // …and the movement step notices, releasing rather than pinning.
    sim.step(&forward);
    assert!(!sim.hero().runtime.foot_lock_l.locked);
    assert_eq!(sim.hero().runtime.foot_slide_l_m, 0.0);
}

/// **The foot-IK half of clause 5, in the world** (P29.4 audit, A12).
///
/// Everything above measures the **lock**. The other mechanism — a downward
/// probe under the foot, a vertical offset so the sole rests on the surface, and
/// the P24.2 chain solve that puts it there — had no world arm at all: the only
/// assertion about `foot_ik` in the tree was that it is **empty**. This is the
/// positive half.
///
/// Three claims: a goal is published for a foot that is over ground inside
/// ALS's envelope; its target sits one `FOOT_HEIGHT_M` above the surface, which
/// is what "the sole rests on it" means; and the goal really is the *ground's*
/// answer, because raising the floor raises the target by the same amount.
#[test]
fn a_foot_over_ground_inside_the_envelope_is_given_a_goal_on_the_surface() {
    /// `(goal target y, published foot y, floor top)` after twelve steps.
    fn probe(floor_top: f64) -> (f64, f64) {
        let mut sim = Sim::new(true);
        // 8 cm above the feet-at-origin height puts the ANKLE 8 cm above the
        // floor, which is what a real foot joint does — and it is the only way
        // the two readings of `ground_offset`'s third argument can be told apart
        // (A13): with the ankle passed in, the solve drives it down onto the
        // floor; with the floor point, a foot on the body's own plane is left
        // where it is.
        sim.clip = walk_clip_at(true, HIPS_FEET_AT_ORIGIN + 0.08);
        // Move the floor: the ground block's top is at `translation.y + 0.5`.
        let e = sim.world.entity_of(GROUND).unwrap();
        sim.world
            .world_mut()
            .get_mut::<Transform>(e)
            .unwrap()
            .translation
            .y = floor_top - 0.5;
        sim.world.mark_dirty();
        sim.world.propagate();
        let forward = MovementIntent {
            move_input: inf_ecs::Vec2d::new(0.0, 1.0),
            ..Default::default()
        };
        for _ in 0..12 {
            sim.step(&forward);
        }
        let goals = inf_ecs::anim_bridge::bridge(&sim.world)
            .and_then(|b| b.foot_ik.get(&HERO).copied())
            .expect("a foot over ground inside the envelope must be given a goal");
        let left = goals[0].expect("the left foot's goal");
        assert!(
            (left.weight - 1.0).abs() < 1e-6,
            "the Enable_FootIK curve is the weight: {}",
            left.weight
        );
        let foot = inf_ecs::anim_bridge::feet_of(&sim.world, HERO).unwrap()[0]
            .expect("a left foot")
            .world
            .y;
        assert!(
            foot - floor_top > 0.05,
            "the fixture must stand its ANKLE clear of the floor, or the two \
             readings of `ground_offset`'s floor argument are the same number: \
             {foot} over {floor_top}"
        );
        (left.target.y, foot)
    }

    // **On a flat floor the correction is nothing.** The ground under the foot
    // and the ground under the body are the same plane, so the pose is left
    // alone — which is what "IK against terrain" means and what the ankle-versus-
    // floor-point defect (A13) broke: with the ankle passed in, the solve drove
    // the ankle down onto the floor by however high it was standing.
    let (flat, foot) = probe(0.0);
    let correction = flat - foot;
    // Not exactly zero: the mover leaves the capsule a skin width clear of the
    // floor, and the offset is the ground under the FOOT against the ground
    // under the BODY, so it carries that centimetre or two. What it must not
    // carry is the ankle's own height.
    assert!(
        correction.abs() < 0.03,
        "a foot on the same floor the body stands on must not be moved: goal \
         {flat} against a foot at {foot}"
    );
    assert!(
        correction.abs() < foot * 0.5,
        "the correction is the ankle's height ({foot} m), which is the defect \
         A13 names: the solve is driving the ANKLE onto the floor instead of \
         comparing two ground planes"
    );
    // **And it really is the GROUND's answer.** Raise the floor 12 cm under the
    // whole character and the target rises with it, because the probe reports a
    // surface rather than a constant.
    let (raised, _) = probe(0.12);
    assert!(
        (raised - flat - 0.12).abs() < 0.02,
        "the goal did not follow the floor: {flat} -> {raised}"
    );
}

/// **THE FOOT-PUBLISH SEAM, DECIDED** (P29.6; the P29.4 audit's A12 pinned it and
/// this arm replaces the pin).
///
/// The seam was a disagreement nothing in the tree had ruled on: `inf_ecs::pose`
/// lifted a model-space pose with the entity's own `GlobalTransform`, and the
/// movement step keeps a character's transform at its capsule **centre**
/// (`feet = translation - (half_height + radius)`) — while `inf_anim::template`
/// (and every glTF character anybody imports) authors a rig with its **feet at
/// model y = 0**. So a rig published its feet one half-height plus one radius
/// above the floor it stood on: **0.9 m** for the default 1.8 m capsule, outside
/// ALS's ±50/45 cm envelope, so the foot IK could never reach the ground and the
/// lock pinned a point in the air. The old arm asserted that number, on purpose,
/// so that deciding would break it.
///
/// **The ruling: pose space is feet-at-origin character space.** The publisher —
/// `inf_ecs::pose::model_to_world`, the one door — subtracts the capsule the
/// character is **wearing**. Nothing is authored, nothing is stored, no schema
/// moves, and every entity that is not a character is composed with the identity,
/// so no committed level moves a byte.
///
/// Three claims: the feet-at-origin rig now publishes its feet **on the floor**;
/// the number is the *worn* capsule and not the mode's requested one; and the
/// consequence the seam blocked is unblocked — the probe reaches the ground and a
/// goal is published.
#[test]
fn a_feet_at_origin_rig_publishes_its_feet_on_the_floor() {
    let mut sim = Sim::new(true); // `walk_clip` — hips at 0.95, feet at model 0
    let forward = MovementIntent {
        move_input: inf_ecs::Vec2d::new(0.0, 1.0),
        ..Default::default()
    };
    for _ in 0..6 {
        sim.step(&forward);
    }
    let e = sim.world.entity_of(HERO).unwrap();
    let centre = sim
        .world
        .world()
        .get::<Transform>(e)
        .unwrap()
        .translation
        .to_dvec3();
    let half = sim.hero().stand_half_height_m;
    let feet_y = centre.y - half - RADIUS;
    assert!(
        feet_y.abs() < 0.05,
        "the capsule stands on the floor: {feet_y}"
    );

    // The lift really is the entity transform minus the worn capsule, asserted
    // against the door rather than against a restated constant.
    let lift = inf_ecs::pose::model_to_world(&sim.world, e).translation;
    assert!(
        (lift.y - feet_y).abs() < 1e-9,
        "character space starts at the feet: {lift:?} against {feet_y}"
    );
    let worn = sim
        .world
        .world()
        .get::<Collider3D>(e)
        .expect("the fixture wears a capsule");
    assert!(
        (centre.y - lift.y - (worn.half_extents.y + worn.radius)).abs() < 1e-9,
        "the drop is the WORN capsule (half {} + radius {}), not the mode's \
         requested one",
        worn.half_extents.y,
        worn.radius
    );

    let left = inf_ecs::anim_bridge::feet_of(&sim.world, HERO).unwrap()[0].expect("a left foot");
    let published = left.world.y;
    // A centimetre, not an epsilon: the published foot is the drawn one, so it
    // carries the mover's skin width and — now that the seam is closed — the
    // foot-IK solve's own correction. The *exact* claim is the lift above.
    assert!(
        (published - feet_y).abs() < 0.01,
        "a feet-at-origin rig publishes its feet on the floor: {published} vs \
         {feet_y}"
    );
    // …and it is inside the probe envelope now, which is the whole consequence.
    assert!(
        (published - feet_y).abs() < inf_anim::TRACE_BELOW_M,
        "…within the {} m probe reach",
        inf_anim::TRACE_BELOW_M
    );
    // The positive half: the probe finds the floor and a goal is published, which
    // is what the seam made impossible.
    assert!(
        inf_ecs::anim_bridge::bridge(&sim.world)
            .and_then(|b| b.foot_ik.get(&HERO).copied())
            .map(|g| g[0].is_some())
            .unwrap_or(false),
        "the probe found no ground under a foot standing on it — character space \
         is not reaching the publisher"
    );
}

/// A character whose clips carry **no curve channels at all** pays nothing: no
/// lock engages, no goal is published, and the pose is exactly what the machine
/// produced. That is what keeps every committed sample byte-identical.
#[test]
fn a_clip_with_no_channels_engages_nothing() {
    let mut sim = Sim::new(true);
    sim.clip = AnimClip::new("bare", Vec::new());
    let forward = MovementIntent {
        move_input: inf_ecs::Vec2d::new(0.0, 1.0),
        ..Default::default()
    };
    for _ in 0..60 {
        sim.step(&forward);
        let h = sim.hero();
        assert!(!h.runtime.foot_lock_l.locked, "a bare clip locked a foot");
        assert!(!h.runtime.foot_lock_r.locked);
        assert_eq!(h.runtime.foot_slide_l_m, 0.0);
        // …and the pelvis is not asked to drop for feet nobody is solving.
        assert_eq!(h.runtime.pelvis_offset, Vec3d::ZERO);
    }
    assert!(
        inf_ecs::anim_bridge::bridge(&sim.world)
            .map(|b| b.foot_ik.is_empty())
            .unwrap_or(true),
        "a bare clip published a foot-IK goal"
    );
}
