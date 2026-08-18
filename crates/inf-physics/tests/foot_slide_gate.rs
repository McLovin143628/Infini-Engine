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
    walk_clip_at(plant, 1.0)
}

/// [`walk_clip`] with the hips placed at `hips_y` in **model** space.
///
/// The height is the clip's, not the skeleton's: joint 0 is animated, so its
/// bind pose is irrelevant and the clip is the only thing that decides where the
/// rig's feet end up. `1.0` is a feet-at-origin rig (the `inf_anim::template`
/// convention); `0.05` puts the same legs under a hip that sits at the capsule's
/// centre, which is the only way the feet reach the floor at all — see
/// `a_feet_at_origin_rig_publishes_its_feet_half_a_capsule_up`.
fn walk_clip_at(plant: bool, hips_y: f32) -> AnimClip {
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
        CurveChannel::constant(als::ENABLE_FOOT_IK_L, 1.0),
        CurveChannel::constant(als::ENABLE_FOOT_IK_R, 1.0),
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
    let mut sim = Sim::new(plant);
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
    // **7e-9 m** locked against **0.892 m** unlocked — eight orders of
    // magnitude, which is the separation a mechanism that works produces and a
    // tolerance never does. The bound is 1e-5 m: a thousand times the
    // measurement, so a real regression fails it and floating-point weather does
    // not, and ninety thousand times under the control.
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
        // The hips at the capsule's centre rather than a metre above it, so the
        // feet land on the floor instead of a metre over it (the seam the arm
        // below records).
        // 0.13 puts the ANKLE 8 cm above the floor, which is what a real foot
        // joint does — and it is the only way the two readings of
        // `ground_offset`'s third argument can be told apart (A13): with the
        // ankle passed in, the solve drives it down onto the floor; with the
        // floor point, a foot on the body's own plane is left where it is.
        sim.clip = walk_clip_at(true, 0.13);
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

/// **The rig rides the entity transform, and a character's entity transform is
/// its capsule CENTRE** — measured, because nothing in the tree decides it
/// (P29.4 audit, A12).
///
/// `inf_ecs::pose` lifts model space into world space with the entity's own
/// `GlobalTransform`, and the movement step keeps that transform at the capsule's
/// centre (`feet = translation - (half_height + radius)`). A rig authored the way
/// `inf_anim::template` authors one — hips at `h × hip_height_ratio`, **feet at
/// model y = 0** — therefore publishes its feet one half-height plus one radius
/// above the floor it is standing on. For the default 1.8 m capsule that is
/// **0.9 m**, and ALS's probe envelope is ±50/45 cm, so the foot IK can never
/// reach the ground for such a character and the foot lock pins a point in the
/// air.
///
/// No committed content carries a `CharacterMovement` and a `SkeletalMesh` at
/// once (the P29.4 ledger says so), which is why nothing has noticed. The
/// decision — a mesh offset on the character, a feet-anchored transform, or the
/// capsule-centred rig convention `walk_clip_at(_, 0.05)` assumes — belongs with
/// P29.6's sample character. **This arm asserts the current answer**, so the day
/// that decision lands it fails and the ledger has to be rewritten rather than
/// quietly disagreeing with the engine.
#[test]
fn a_feet_at_origin_rig_publishes_its_feet_half_a_capsule_up() {
    let mut sim = Sim::new(true); // `legs()` — hips at model y = 1.0, feet at 0
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

    let left = inf_ecs::anim_bridge::feet_of(&sim.world, HERO).unwrap()[0].expect("a left foot");
    // The rig's own foot is at model y = 0.05 (1.0 − 0.05 − 0.45 − 0.45).
    let published = left.world.y;
    assert!(
        (published - (centre.y + 0.05)).abs() < 1e-6,
        "the published foot is the model position lifted by the ENTITY transform: \
         {published} vs {}",
        centre.y + 0.05
    );
    // …which is most of a capsule above the floor, and outside the envelope.
    let above_floor = published - feet_y;
    assert!(
        above_floor > half + RADIUS - 0.06,
        "a feet-at-origin rig should publish its feet ~{} m up: {above_floor}",
        half + RADIUS
    );
    assert!(
        above_floor > inf_anim::TRACE_BELOW_M,
        "…which is what puts the floor out of the {} m probe reach",
        inf_anim::TRACE_BELOW_M
    );
    // And the consequence, asserted rather than described: no goal is published,
    // because the probe finds nothing under a foot that is standing in the air.
    assert!(
        inf_ecs::anim_bridge::bridge(&sim.world)
            .map(|b| b.foot_ik.is_empty())
            .unwrap_or(true),
        "the probe reached the floor after all — the seam this arm records is \
         closed, and the ledger must be rewritten"
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
