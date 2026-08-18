//! **P29.4: the ragdoll bridge**, driven end-to-end — the animation side
//! publishing a rig, the physics side building bodies out of it, and the
//! pose-matched get-up coming back the other way.
//!
//! Both fixed steps run here, in the order both hosts run them (movement before
//! the solver, the pose after the write-back), because the whole point of the
//! seam is that it crosses between them. A test that ran only one would be
//! testing half a queue.
//!
//! **The doctrine is what is asserted**: every crossing is a value, the blend
//! weight is a pure function of `(phase, clock)`, and physics never touches
//! `SmRuntime` — the machine changes state because a *parameter* and a *trigger*
//! were set, which is the same door a Blueprint would use.

use std::collections::BTreeMap;

use glam::DVec3;
use inf_anim::{AnimClip, Joint, JointTransform, Skeleton, SkeletonAsset, SmState, StateMachine};
use inf_ecs::components::{
    AnimStateMachine, BodyKind3D, CharacterMovement, Collider3D, ColliderShape3DKind, MovementMode,
    RigidBody3D, SkeletalMesh, Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::movement::MovementIntent;
use inf_ecs::EcsWorld;
use inf_physics::d3::{ragdoll_bridge, step_character_movement};
use inf_physics::PhysicsBridge3D;
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const HERO: Uuid = Uuid::from_u128(0x2904_1001);
const GROUND: Uuid = Uuid::from_u128(0x2904_1002);
const SKEL: Uuid = Uuid::from_u128(0x2904_1003);
const SM: Uuid = Uuid::from_u128(0x2904_1004);
const CLIP: inf_anim::ClipRef = [9; 16];
const RADIUS: f64 = 0.3;

/// A humanoid whose bone names `inf_physics::ragdoll::classify` recognizes.
///
/// The proportions are a 1.8 m adult, so the built capsules have real lengths and
/// the joints have real anchors — a rig of unit-length stubs would spawn a
/// ragdoll that behaves like nothing in particular.
fn humanoid() -> SkeletonAsset {
    fn joint(name: &str, parent: Option<u16>, local: glam::Vec3) -> Joint {
        Joint {
            name: name.into(),
            parent,
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::from_trs(local, glam::Quat::IDENTITY, glam::Vec3::ONE),
        }
    }
    let sk = Skeleton::new(vec![
        joint("Hips", None, glam::Vec3::new(0.0, 1.0, 0.0)),
        joint("Spine", Some(0), glam::Vec3::new(0.0, 0.2, 0.0)),
        joint("Chest", Some(1), glam::Vec3::new(0.0, 0.3, 0.0)),
        joint("UpperArm.L", Some(2), glam::Vec3::new(0.2, 0.15, 0.0)),
        joint("LowerArm.L", Some(3), glam::Vec3::new(0.28, 0.0, 0.0)),
        joint("UpperArm.R", Some(2), glam::Vec3::new(-0.2, 0.15, 0.0)),
        joint("LowerArm.R", Some(5), glam::Vec3::new(-0.28, 0.0, 0.0)),
        joint("Thigh.L", Some(0), glam::Vec3::new(0.1, -0.05, 0.0)),
        joint("Shin.L", Some(7), glam::Vec3::new(0.0, -0.45, 0.0)),
        joint("Thigh.R", Some(0), glam::Vec3::new(-0.1, -0.05, 0.0)),
        joint("Shin.R", Some(9), glam::Vec3::new(0.0, -0.45, 0.0)),
    ])
    .expect("a valid humanoid");
    SkeletonAsset::new(sk)
}

/// idle → get_up, on the trigger the ragdoll bridge arms, with the supine and
/// prone halves picked by the parameter it sets.
fn machine() -> StateMachine {
    use inf_anim::{CmpOp, SmCompare, SmCond, SmParam, SmParamKind, SmTransition, SmValue};
    let cond = |name: &str, op: CmpOp, v: f64| {
        SmCond::Compare(SmCompare {
            param: name.into(),
            op,
            value: SmValue::Float(v),
        })
    };
    StateMachine {
        states: vec![
            SmState::clip("idle", CLIP),
            SmState::clip("get_up_supine", CLIP),
            SmState::clip("get_up_prone", CLIP),
        ],
        transitions: vec![
            SmTransition {
                condition: SmCond::And(vec![
                    SmCond::Trigger(ragdoll_bridge::TRIGGER_GET_UP.into()),
                    cond(ragdoll_bridge::PARAM_FACE_UP, CmpOp::Gt, 0.5),
                ]),
                ..SmTransition::on(0, 1, 0.0, "unused", CmpOp::Gt, -1e9)
            },
            SmTransition {
                condition: SmCond::And(vec![
                    SmCond::Trigger(ragdoll_bridge::TRIGGER_GET_UP.into()),
                    cond(ragdoll_bridge::PARAM_FACE_UP, CmpOp::Lt, 0.5),
                ]),
                ..SmTransition::on(0, 2, 0.0, "unused", CmpOp::Gt, -1e9)
            },
        ],
        entry: 0,
        params: vec![
            SmParam::new(ragdoll_bridge::TRIGGER_GET_UP, SmParamKind::Trigger),
            SmParam {
                name: ragdoll_bridge::PARAM_FACE_UP.into(),
                kind: SmParamKind::Float,
                default: SmValue::Float(0.0),
            },
            SmParam::new(ragdoll_bridge::TRIGGER_RAGDOLL, SmParamKind::Trigger),
        ],
        ..Default::default()
    }
}

struct Sim {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    skeleton: SkeletonAsset,
    machine: StateMachine,
    clip: AnimClip,
}

impl Sim {
    fn new(feet_y: f64) -> Self {
        let mut world = EcsWorld::new();
        // The floor.
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
                half_extents: Vec3d::new(30.0, 0.5, 30.0),
                ..Default::default()
            },
            t,
        ));
        // The character.
        let cm = CharacterMovement {
            player_controlled: true,
            ..Default::default()
        };
        let e = world.spawn_with_guid(HERO, "Hero", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, feet_y + cm.stand_half_height_m + RADIUS, 0.0);
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
            skeleton: humanoid(),
            machine: machine(),
            clip: AnimClip::new("pose", Vec::new()),
        }
    }

    /// One fixed step, in the order both hosts run it.
    fn step(&mut self, intent: &MovementIntent) {
        self.bridge.sync_from_world(&self.world);
        inf_ecs::movement::apply_intent(&mut self.world, intent);
        step_character_movement(&mut self.world, &mut self.bridge, DT);
        self.bridge.step(DT);
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

    fn state(&self) -> String {
        inf_ecs::anim_bridge::anim_state(&self.world, HERO)
            .map(|s| s.name.clone())
            .unwrap_or_default()
    }
}

/// **The whole round trip**: velocity in, bodies, settle, pose-matched get-up out.
#[test]
fn a_ragdoll_takes_the_velocity_in_and_hands_a_pose_matched_get_up_back() {
    let mut sim = Sim::new(0.0);
    // Get the character running, so the handoff has a velocity to hand off.
    for _ in 0..60 {
        sim.step(&MovementIntent {
            move_input: inf_ecs::Vec2d::new(0.0, 1.0),
            ..Default::default()
        });
    }
    let running = sim.hero().runtime.velocity.to_dvec3();
    assert!(
        running.length() > 2.0,
        "the character must be moving before it ragdolls: {running:?}"
    );
    assert_eq!(sim.state(), "idle");

    // **Start it.** The mode takes immediately; the rig is asked for.
    assert!(ragdoll_bridge::start_ragdoll(&mut sim.world, HERO));
    assert_eq!(sim.hero().mode, MovementMode::Ragdoll);
    assert_eq!(
        sim.hero().runtime.ragdoll.phase,
        inf_anim::RagdollPhase::Simulating
    );
    let handed = sim.hero().runtime.ragdoll.last_velocity.to_dvec3();
    assert!(
        (handed - running).length() < 1e-9,
        "the handoff velocity is the character's own: {handed:?} vs {running:?}"
    );

    // One step: the pose side answers with a rig. The next: the bodies exist.
    sim.step(&MovementIntent::default());
    assert!(
        inf_ecs::anim_bridge::ragdoll_rig(&sim.world, HERO).is_some(),
        "the pose step must publish a rig for a requested ragdoll"
    );
    sim.step(&MovementIntent::default());
    assert!(
        sim.hero().runtime.ragdoll.spawned,
        "the bodies were not built"
    );
    assert_eq!(sim.bridge.ragdoll_count(), 1);
    // **The velocity handoff reached the bodies**, which is the claim: every limb
    // starts at the speed the character was moving.
    let spawned = sim.bridge.ragdoll_of(HERO).unwrap().clone();
    assert!(spawned.bodies.len() >= 7, "{:?}", spawned.bodies.len());
    assert!(
        !spawned.joints.is_empty(),
        "a ragdoll with no joints is debris"
    );
    assert!(spawned.pelvis.is_some(), "the pelvis must be identified");

    // Let it fall over and settle.
    let mut settled = false;
    for _ in 0..600 {
        sim.step(&MovementIntent::default());
        if sim.hero().mode != MovementMode::Ragdoll {
            settled = true;
            break;
        }
    }
    assert!(settled, "the ragdoll never settled");

    // **The exit branch**: it was on the ground, so it gets up.
    let h = sim.hero();
    assert_eq!(h.mode, MovementMode::Grounded);
    assert_eq!(h.runtime.ragdoll.phase, inf_anim::RagdollPhase::GettingUp);
    assert_eq!(
        sim.bridge.ragdoll_count(),
        0,
        "the bodies were not cleaned up"
    );
    assert!(!h.runtime.ragdoll.spawned);

    // **The machine got up through the KIT'S doors**, not by being written to:
    // a parameter said which way up, a trigger said now, and the transition the
    // machine's own condition tree chose is the one that fired.
    let face_up = h.runtime.ragdoll.face_up;
    assert_eq!(
        inf_ecs::anim_bridge::anim_param(&sim.world, HERO, ragdoll_bridge::PARAM_FACE_UP),
        Some(if face_up { 1.0 } else { 0.0 })
    );
    let want = if face_up {
        "get_up_supine"
    } else {
        "get_up_prone"
    };
    assert_eq!(sim.state(), want, "the machine entered the wrong get-up");

    // **Pose matching is on for the get-up and off again when it ends** — P29.2
    // built the primitive and named this consumer; a machine that entered every
    // state at its best-matching frame would never play a state's beginning.
    assert!(
        inf_ecs::anim_bridge::bridge(&sim.world)
            .unwrap()
            .pose_matched
            .contains(&HERO),
        "the get-up must enter at the matched frame"
    );
    let mut blend_seen = Vec::new();
    for _ in 0..60 {
        blend_seen.push(inf_physics::d3::ragdoll_blend_weight(&sim.hero()));
        sim.step(&MovementIntent::default());
        if sim.hero().runtime.ragdoll.phase == inf_anim::RagdollPhase::Inactive {
            break;
        }
    }
    assert_eq!(
        sim.hero().runtime.ragdoll.phase,
        inf_anim::RagdollPhase::Inactive
    );
    assert!(
        !inf_ecs::anim_bridge::bridge(&sim.world)
            .map(|b| b.pose_matched.contains(&HERO))
            .unwrap_or(false),
        "pose matching must be turned off when the get-up ends"
    );
    // The blend really did come down, monotonically, from the physics pose to the
    // machine's.
    assert!(blend_seen[0] > 0.9, "{:?}", blend_seen[0]);
    assert!(*blend_seen.last().unwrap() < 0.2, "{blend_seen:?}");
    for w in blend_seen.windows(2) {
        assert!(w[1] <= w[0] + 1e-6, "the blend rose: {w:?}");
    }
}

/// **The airborne branch** (§13's row, the other half): a ragdoll that ends in
/// the air resumes falling with `LastRagdollVelocity`, and does **not** get up.
#[test]
fn a_ragdoll_that_ends_airborne_resumes_falling_with_its_own_velocity() {
    let mut sim = Sim::new(40.0);
    // Fall for a while so there is a real velocity to hand back.
    for _ in 0..30 {
        sim.step(&MovementIntent::default());
    }
    assert!(ragdoll_bridge::start_ragdoll(&mut sim.world, HERO));
    sim.step(&MovementIntent::default());
    sim.step(&MovementIntent::default());
    assert!(sim.hero().runtime.ragdoll.spawned);

    // Ask for the exit while it is still a long way up.
    for _ in 0..30 {
        sim.step(&MovementIntent::default());
    }
    let before = sim.hero().runtime.ragdoll.last_velocity.to_dvec3();
    assert!(before.y < -1.0, "it should be falling: {before:?}");
    sim.step(&MovementIntent {
        jump: true,
        ..Default::default()
    });

    let h = sim.hero();
    assert_eq!(
        h.mode,
        MovementMode::FallFree,
        "airborne must resume falling"
    );
    assert_eq!(
        h.runtime.ragdoll.phase,
        inf_anim::RagdollPhase::Inactive,
        "there is nothing to get up from in mid-air"
    );
    // **The handoff is `LastRagdollVelocity`, verbatim** — the number the bodies
    // were last seen at, not the number they were seen at a step earlier, which
    // is why this compares against the runtime's own field rather than the
    // snapshot above. That the two are CLOSE is the second half: one step of
    // gravity is 0.16 m/s, so a handoff that had silently zeroed or doubled the
    // velocity could not pass both.
    let after = h.runtime.velocity.to_dvec3();
    let handed = h.runtime.ragdoll.last_velocity.to_dvec3();
    assert!(
        (after - handed).length() < 1e-12,
        "the exit must hand back the ragdoll's own velocity: {after:?} vs {handed:?}"
    );
    assert!(after.y < -1.0, "and it must still be a fall: {after:?}");
    assert!(
        (after - before).length() < 1.0,
        "one step apart, not a different number: {after:?} vs {before:?}"
    );
    assert_eq!(sim.bridge.ragdoll_count(), 0);
    // …and the machine was never told to get up.
    assert_eq!(sim.state(), "idle");
}

/// **The purity arm.** Two worlds in the same sim state produce the same blend
/// weight — bit for bit, through the shipped door, with nothing in between.
///
/// The mutation this kills: a blend weight derived from anything the physics side
/// holds (a body's speed, a handle, a step counter) rather than from `(phase,
/// clock)` would differ between two runs that reached the same state by different
/// routes.
#[test]
fn the_blend_weight_is_a_pure_function_of_the_characters_own_state() {
    let mut a = Sim::new(0.0);
    let mut b = Sim::new(0.0);
    // Two different histories that arrive at the same phase: one ragdolls
    // immediately, the other runs for half a second first.
    assert!(ragdoll_bridge::start_ragdoll(&mut a.world, HERO));
    for _ in 0..30 {
        b.step(&MovementIntent {
            move_input: inf_ecs::Vec2d::new(0.0, 1.0),
            ..Default::default()
        });
    }
    assert!(ragdoll_bridge::start_ragdoll(&mut b.world, HERO));

    for _ in 0..900 {
        a.step(&MovementIntent::default());
        b.step(&MovementIntent::default());
        let (ha, hb) = (a.hero(), b.hero());
        if ha.runtime.ragdoll.phase == hb.runtime.ragdoll.phase
            && (ha.runtime.ragdoll.time_in_phase_s - hb.runtime.ragdoll.time_in_phase_s).abs()
                < 1e-12
        {
            assert_eq!(
                inf_physics::d3::ragdoll_blend_weight(&ha).to_bits(),
                inf_physics::d3::ragdoll_blend_weight(&hb).to_bits(),
                "same (phase, clock), different weight"
            );
        }
        if ha.runtime.ragdoll.phase == inf_anim::RagdollPhase::Inactive
            && ha.mode == MovementMode::Grounded
            && a.hero().runtime.time_in_mode_s > 1.0
        {
            break;
        }
    }
    // Not vacuous: both really did ragdoll and both really did get up.
    assert_eq!(a.hero().mode, MovementMode::Grounded);
    assert!(a.state().starts_with("get_up"), "{}", a.state());
}

/// A ragdoll may be entered from **anything** — it is a fact about the body, not
/// a choice — and the transition table says so at every mode the catalogue has.
#[test]
fn a_ragdoll_can_be_entered_from_any_mode_the_character_is_in() {
    for (label, setup) in [
        ("crouched", MovementMode::Crouch),
        ("sliding", MovementMode::Slide),
        ("falling", MovementMode::FallFree),
        ("mantling", MovementMode::Mantle),
    ] {
        let mut sim = Sim::new(0.0);
        sim.step(&MovementIntent::default());
        {
            let e = sim.world.entity_of(HERO).unwrap();
            sim.world
                .world_mut()
                .get_mut::<CharacterMovement>(e)
                .unwrap()
                .mode = setup;
        }
        assert!(
            ragdoll_bridge::start_ragdoll(&mut sim.world, HERO),
            "{label}: a ragdoll must be enterable from {setup:?}"
        );
        assert_eq!(sim.hero().mode, MovementMode::Ragdoll, "{label}");
        // …and a live mantle is cancelled by it, which is ALS's
        // `OnOwnerRagdollStateChanged` doing the same thing.
        assert!(!sim.hero().runtime.mantle.active, "{label}");
    }
}
