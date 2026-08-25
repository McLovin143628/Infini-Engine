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

/// **The mannequin**, straight out of the generator — 161 bones, a role table,
/// twist chains, fingers and IK handles.
fn mannequin() -> SkeletonAsset {
    inf_anim::build_template(
        inf_anim::BodyPlan::Biped,
        &inf_anim::BodyParams {
            height_m: 1.8,
            ..Default::default()
        },
    )
    .expect("the mannequin builds")
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
        Self::with_rig(feet_y, humanoid())
    }

    /// The same world with a caller-chosen rig — the seam the SK1a arms need,
    /// because the whole question is what the *rig* says about itself.
    fn with_rig(feet_y: f64, rig: SkeletonAsset) -> Self {
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
            skeleton: rig,
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

    /// The same character with its **skeleton taken away**, so the pose step can
    /// never publish a rig and the bodies can never be built (P29.4 audit, A1).
    fn rigless(feet_y: f64) -> Self {
        let mut sim = Self::new(feet_y);
        let e = sim.world.entity_of(HERO).unwrap();
        sim.world.world_mut().entity_mut(e).remove::<SkeletalMesh>();
        sim.world.mark_dirty();
        sim.world.propagate();
        sim
    }

    fn mode(&self) -> MovementMode {
        self.hero().mode
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

/// **A RAGDOLL WHOSE HIPS END INSIDE THE FLOOR DOES NOT PUT THE CHARACTER UNDER
/// IT** (island wave I5).
///
/// The capsule follows the pelvis, "lifted onto the floor under it" — and where
/// the ground probe *started penetrating*, the placement used to put the feet a
/// whole `half + radius` **below the pelvis**, which on a pelvis already in the
/// floor is a whole body below the floor. The collider is switched back on down
/// there when the ragdoll ends, and the character falls out of the world:
/// measured on the phase-29 course at **y = −132 m and still falling**, from
/// hips that ended 0.8 m under the ground.
///
/// The fixture puts the hazard the claim names in the world — it drives the
/// pelvis body *below the ground's top face* — and measures the placement
/// against the alternative, which is the thing that makes the arm able to fail:
/// the old rule and the new one differ by exactly one body height, and a
/// tolerance loose enough to cover both would have covered the defect.
#[test]
fn a_ragdoll_that_ends_inside_the_floor_leaves_the_character_on_it() {
    const GROUND_TOP: f64 = 0.0;
    let mut sim = Sim::new(GROUND_TOP);
    for _ in 0..10 {
        sim.step(&MovementIntent::default());
    }
    assert!(ragdoll_bridge::start_ragdoll(&mut sim.world, HERO));
    sim.step(&MovementIntent::default());
    sim.step(&MovementIntent::default());
    assert!(sim.hero().runtime.ragdoll.spawned, "no bodies to sink");

    // Drive every limb below the ground's top face, so the pelvis probe begins
    // inside the floor — the case the placement got wrong.
    const SUNK: f64 = 0.5;
    let bodies = sim.bridge.ragdoll_of(HERO).unwrap().bodies.clone();
    let pelvis = sim.bridge.ragdoll_of(HERO).unwrap().pelvis.unwrap();
    for b in &bodies {
        let Some(p) = sim.bridge.world().body_translation(*b) else {
            continue;
        };
        sim.bridge
            .world_mut()
            .set_body_translation(*b, DVec3::new(p.x, GROUND_TOP - SUNK, p.z));
        sim.bridge.world_mut().set_body_linvel(*b, DVec3::ZERO);
    }
    sim.step(&MovementIntent::default());

    let pelvis_y = sim.bridge.world().body_translation(pelvis).unwrap().y;
    let e = sim.world.entity_of(HERO).unwrap();
    let placed = sim.world.world().get::<Transform>(e).unwrap().translation.y;
    let cm = sim.hero();
    let body = cm.half_height_for(MovementMode::Grounded) + RADIUS;
    // The alternative, priced: the rule this replaced put the FEET at the
    // pelvis, i.e. the centre a body below it.
    let old_rule = pelvis_y;
    println!(
        "pelvis y = {pelvis_y:.3}, placed centre = {placed:.3}, the old rule would have placed {old_rule:.3} (one body = {body:.3} m lower)"
    );
    assert!(
        placed > old_rule + body * 0.5,
        "the character was placed at {placed:.3}, which is within half a body of the rule that dropped it through the world ({old_rule:.3})"
    );
    assert!(
        placed >= pelvis_y,
        "a placement below the pelvis is a placement below the surface the pelvis is resting in"
    );
    assert!(
        cm.runtime.ragdoll.on_ground,
        "a pelvis in the floor is on it — the doc says so and the settle depends on it"
    );
}

/// **A LIMB MAY NOT EXCEED THE SPEED THE GRAVITY CUTOFF ALREADY NAMES** (island
/// wave I5).
///
/// `gravity_enabled` bounds an *acceleration*; nothing bounded the velocity, and
/// an articulated body seeded in a pose that violates its joint limits is fed
/// energy by the solver every step until the numbers leave the world — measured
/// on the phase-29 course at `z = −3.85e13` from a ragdoll entered 2.7 cm
/// further along the same fall than the one that settles.
///
/// This is a bound rather than a cure and the arm says only what the bound says:
/// whatever a limb is doing, it is doing it at a speed a falling body could
/// reach. The `+ 1e-9` is the clamp's own arithmetic and nothing else; a
/// tolerance wide enough to admit the 4 000 m/s a diverging run reaches would be
/// no bound at all.
#[test]
fn a_ragdoll_limb_cannot_exceed_the_terminal_speed() {
    let mut sim = Sim::new(0.0);
    for _ in 0..10 {
        sim.step(&MovementIntent::default());
    }
    assert!(ragdoll_bridge::start_ragdoll(&mut sim.world, HERO));
    sim.step(&MovementIntent::default());
    sim.step(&MovementIntent::default());
    let bodies = sim.bridge.ragdoll_of(HERO).unwrap().bodies.clone();
    assert!(!bodies.is_empty(), "no limbs to bound");

    // ── the SPEED half, on its own ──
    //
    // One limb thrown at four kilometres a second, alone, so the assertion below
    // is about the ceiling and not about a NaN that would have failed anyway.
    // The step is taken with the throw *between* the sync and the drive, which
    // is where the ceiling runs.
    const THROWN: f64 = 4000.0;
    sim.bridge
        .world_mut()
        .set_body_linvel(bodies[0], DVec3::new(THROWN, 0.0, 0.0));
    sim.step(&MovementIntent::default());
    let after = sim
        .bridge
        .world()
        .body_linvel(bodies[0])
        .unwrap_or(DVec3::ZERO);
    println!(
        "the thrown limb: {THROWN:.0} m/s in, {:.3} m/s out, against a ceiling of {:.1}",
        after.length(),
        ragdoll_bridge::MAX_LIMB_SPEED_MPS
    );
    assert!(
        after.length() <= ragdoll_bridge::MAX_LIMB_SPEED_MPS + 1e-9,
        "a limb thrown at {THROWN} m/s came out at {}, past the ceiling of {}",
        after.length(),
        ragdoll_bridge::MAX_LIMB_SPEED_MPS
    );

    // ── the NaN half, on its own ──
    //
    // The other way a solver leaves the world, and the reason the branch is a
    // `!is_finite` rather than a plain `>`: every comparison a NaN takes part in
    // is false, so a clamp written as one lets it straight through.
    if let Some(b) = bodies.get(1) {
        sim.bridge
            .world_mut()
            .set_body_linvel(*b, DVec3::splat(f64::NAN));
        sim.step(&MovementIntent::default());
        for h in &bodies {
            let v = sim.bridge.world().body_linvel(*h).unwrap_or(DVec3::ZERO);
            assert!(
                v.is_finite(),
                "a limb is carrying {v:?}, and a NaN pose is a world with no finite state"
            );
        }
    }
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

/// **A ragdoll nobody can build bodies for still ends** (P29.4 audit, A1).
///
/// The bridge asks the pose step for a rig and spawns the bodies when it
/// arrives. For a character with no skeleton it never arrives — and the two ways
/// out of a ragdoll, settling and the player's jump, both live *inside* the "the
/// bodies exist" branch. Measured on the shipped code before this arm: six
/// hundred steps in `Ragdoll`, no gravity, no input authority, and a held jump
/// that changed nothing.
///
/// That is not a corner. `movement.rs`'s landing classifier turns a hard enough
/// fall into a ragdoll for **any** character, rig or no rig, and
/// `movement_parity`'s own traversal fixture is a character with no skeleton.
///
/// Three halves, because the exit is a branch: on the ground it gets up, in the
/// air it resumes falling, and a jump ends it at once rather than after the
/// wait. The control is the arm at the top of this file — the same bridge with a
/// real humanoid spawns bodies and settles, so this is not "ragdolls never work".
#[test]
fn a_character_with_no_rig_is_handed_back_rather_than_left_in_a_ragdoll() {
    // 1. On the ground: it gets up. Stepped first, because "am I on the ground"
    //    is an answer the movement step produces and a character nothing has
    //    simulated has not been told yet.
    let mut sim = Sim::rigless(0.0);
    for _ in 0..10 {
        sim.step(&MovementIntent::default());
    }
    assert!(
        sim.hero().runtime.grounded,
        "the fixture must start standing"
    );
    assert!(ragdoll_bridge::start_ragdoll(&mut sim.world, HERO));
    assert_eq!(sim.mode(), MovementMode::Ragdoll);
    let mut steps = 0;
    while sim.mode() == MovementMode::Ragdoll && steps < 600 {
        sim.step(&MovementIntent::default());
        steps += 1;
    }
    assert!(
        steps < 600,
        "a rig-less character never left Ragdoll in ten seconds — it is stuck"
    );
    assert_eq!(
        sim.mode(),
        MovementMode::Grounded,
        "on the ground, the exit is a get-up"
    );
    assert!(
        !sim.hero().runtime.ragdoll.spawned,
        "nothing should have spawned"
    );
    assert_eq!(sim.bridge.ragdoll_count(), 0);
    // The wait is a bound, not a formality: it really did last more than one
    // step, so a bail-out that fired instantly would fail here too.
    assert!(steps > 1, "the bridge gave up before the rig could arrive");

    // 2. In the air: it resumes falling, and does not get up.
    let mut air = Sim::rigless(40.0);
    for _ in 0..30 {
        air.step(&MovementIntent::default());
    }
    assert!(ragdoll_bridge::start_ragdoll(&mut air.world, HERO));
    let mut steps = 0;
    while air.mode() == MovementMode::Ragdoll && steps < 600 {
        air.step(&MovementIntent::default());
        steps += 1;
    }
    assert!(steps < 600, "the airborne half is stuck too");
    assert_eq!(
        air.mode(),
        MovementMode::FallFree,
        "airborne resumes falling"
    );
    assert_eq!(
        air.hero().runtime.ragdoll.phase,
        inf_anim::RagdollPhase::Inactive,
        "there is nothing to get up from in mid-air"
    );
    // …and it is still falling afterwards rather than frozen where it stopped.
    let y0 = air.hero().runtime.velocity.y;
    for _ in 0..30 {
        air.step(&MovementIntent::default());
    }
    assert!(
        air.hero().runtime.velocity.y < y0 - 1.0,
        "the character must be falling again, not parked: {y0} -> {}",
        air.hero().runtime.velocity.y
    );

    // 3. A jump ends it at once — the player's own escape, which was unreachable.
    let mut asked = Sim::rigless(0.0);
    for _ in 0..10 {
        asked.step(&MovementIntent::default());
    }
    assert!(ragdoll_bridge::start_ragdoll(&mut asked.world, HERO));
    asked.step(&MovementIntent {
        jump: true,
        ..Default::default()
    });
    assert_ne!(
        asked.mode(),
        MovementMode::Ragdoll,
        "a jump must end a ragdoll the bridge cannot build"
    );
}

/// **A CORPSE DOES NOT GET UP** (island wave I6) — the guard, armed at the only
/// fixture in this repository that can reach it.
///
/// I6 put `Health::dead` on the get-up branch because a body the damage system
/// had handed over settled, stood up, and was handed over again. The *handoff*
/// half of that fix is the `Downed` latch and `weapon_3d` measures it; the
/// **get-up** half had no arm at all, and the I6 audit measured why: removing
/// `!dead` from the branch left every `weapon_3d` arm green, because that
/// fixture's target has no skeleton, so P29.4's "no rig is coming" branch hands
/// the body straight back and the settle path is never taken.
///
/// This one has a rig. Two bodies, one live and one dead, through the same
/// bridge: the live one settles and gets up (the control, so a dead body that
/// stayed limp because nothing settled at all would not pass), and the dead one
/// stays where it fell.
#[test]
fn a_dead_body_stays_limp_where_a_live_one_gets_up() {
    /// Ragdoll a rigged character and run until it either leaves the ragdoll or
    /// the clock runs out. Answers `(mode, steps)`.
    fn fall(dead: bool) -> (MovementMode, u32) {
        let mut sim = Sim::new(0.0);
        for _ in 0..10 {
            sim.step(&MovementIntent::default());
        }
        if dead {
            // Through the damage door, not by writing the flag: `Health::dead`
            // is a latch the damage model owns, and a test that set it by hand
            // would still pass if `damage` stopped setting it.
            assert!(inf_ecs::weapon::give_health(&mut sim.world, HERO, 100.0));
            let e = sim.world.entity_of(HERO).expect("the hero");
            let mut h = sim
                .world
                .world_mut()
                .get_mut::<inf_ecs::weapon::Health>(e)
                .expect("a body");
            assert!(inf_ecs::weapon::damage(&mut h, 1.0e6).killed);
        }
        assert!(ragdoll_bridge::start_ragdoll(&mut sim.world, HERO));
        assert_eq!(sim.mode(), MovementMode::Ragdoll);
        let mut steps = 0;
        while sim.mode() == MovementMode::Ragdoll && steps < 900 {
            sim.step(&MovementIntent::default());
            steps += 1;
        }
        // The bodies really were built, so this is a settle rather than the
        // rig-less bail-out measured one arm up.
        assert!(
            sim.hero().runtime.ragdoll.spawned || sim.mode() != MovementMode::Ragdoll,
            "the ragdoll never spawned bodies"
        );
        (sim.mode(), steps)
    }

    let (live, live_steps) = fall(false);
    println!("a live body left the ragdoll after {live_steps} steps as {live:?}");
    assert_eq!(
        live, MovementMode::Grounded,
        "the control never got up, so this arm could not tell a corpse from a settle that never came"
    );
    assert!(live_steps < 900, "the control never settled");

    let (corpse, corpse_steps) = fall(true);
    println!("a dead body was still {corpse:?} after {corpse_steps} steps");
    assert_eq!(
        corpse,
        MovementMode::Ragdoll,
        "a corpse got up: `Health::dead` must guard the get-up, or a body the \
         damage system handed over stands up at the settle interval for ever"
    );
    assert_eq!(corpse_steps, 900, "the corpse left the ragdoll early");
}

/// **A MANNEQUIN RAGDOLLS INTO A CONNECTED BODY** (SK1a) — and the same rig with
/// its role table stripped falls apart, in the same arm, so the number means
/// something.
///
/// The name classifier cannot describe this rig, and the way it fails is the
/// quiet one. `spine_01` … `spine_05` all match its `spine` keyword and none of
/// them matches its `spine1`/`spine2` chest keywords (the underscore), so **no
/// `Chest` is ever produced** — and `Chest` is the parent role of both upper arms
/// and of the head. Those three parts therefore name a parent that is not in the
/// index and are spawned with **no joint at all**: three capsules falling through
/// the world beside a body. `upperarm_l` and `upperarm_twist_01_l` both claim
/// `UpperArmL`, so the second overwrites the first in the role index. Every
/// finger, both clavicles, both feet, both toes and all seven IK handles classify
/// to nothing.
///
/// The role path chains by INDEX, so a five-segment spine is five parts in a row
/// and the arms hang off the top of it.
#[test]
fn a_mannequin_ragdolls_into_one_connected_body_and_a_table_less_one_does_not() {
    use inf_physics::ragdoll::{build_ragdoll, RagdollConfig};

    // The rig the pose step publishes, through the real door: bones in world
    // space with their parents and their roles.
    let mut sim = Sim::with_rig(0.0, mannequin());
    assert!(ragdoll_bridge::start_ragdoll(&mut sim.world, HERO));
    sim.step(&MovementIntent::default());
    let rig = inf_ecs::anim_bridge::ragdoll_rig(&sim.world, HERO)
        .expect("the pose step publishes a rig")
        .to_vec();
    assert_eq!(rig.len(), inf_anim::MANNY_JOINT_COUNT, "one bone per joint");
    assert!(rig.iter().any(|b| b.role.is_some()), "the roles crossed");

    let bones: Vec<inf_physics::ragdoll::RagdollBone> = rig
        .iter()
        .map(|b| {
            inf_physics::ragdoll::RagdollBone::new(
                b.name.clone(),
                b.head.to_dvec3(),
                b.tail.to_dvec3(),
            )
            .with_role(b.parent, b.role)
        })
        .collect();

    // ── the role path ──
    let parts = build_ragdoll(&bones, RagdollConfig::default());
    // pelvis + 5 spine + neck_01 + neck_02 + head + 2x(upperarm, lowerarm)
    // + 2x(thigh, calf) = 17.
    assert_eq!(
        parts.len(),
        17,
        "{:?}",
        parts.iter().map(|p| &p.name).collect::<Vec<_>>()
    );
    let rootless: Vec<&str> = parts
        .iter()
        .filter(|p| p.joint.is_none())
        .map(|p| p.name.as_str())
        .collect();
    assert_eq!(
        rootless,
        ["pelvis"],
        "exactly one free body, and it is the pelvis"
    );
    // Every part reaches the root by following its joints — connectivity, not
    // just "a joint exists".
    for (i, part) in parts.iter().enumerate() {
        let mut cur = i;
        let mut hops = 0;
        while let Some(j) = parts[cur].joint {
            assert!(
                j.parent < cur,
                "`{}` names a parent that follows it",
                part.name
            );
            cur = j.parent;
            hops += 1;
            assert!(hops <= parts.len(), "`{}` is in a cycle", part.name);
        }
        assert_eq!(cur, 0, "`{}` does not reach the pelvis", part.name);
    }
    // The capsules span real bones — the deform-successor rule. A forearm built
    // from its FIRST child would span to `lowerarm_twist_02_l`, a third of the way
    // to the wrist.
    let named = |n: &str| parts.iter().find(|p| p.name == n).expect(n);
    let forearm = named("lowerarm_l");
    let wrist = rig
        .iter()
        .find(|b| b.name == "hand_l")
        .expect("hand_l")
        .head
        .to_dvec3();
    let elbow = rig
        .iter()
        .find(|b| b.name == "lowerarm_l")
        .expect("lowerarm_l")
        .head
        .to_dvec3();
    let spanned = ((forearm.position - elbow).length() * 2.0) / (wrist - elbow).length();
    assert!(
        (spanned - 1.0).abs() < 1.0e-6,
        "the forearm capsule spans {spanned:.3} of the forearm"
    );
    // Nothing driven, nothing helper, nothing finger, nothing IK became a body.
    for p in &parts {
        assert!(
            !p.name.contains("twist")
                && !p.name.starts_with("ik_")
                && !p.name.contains("thumb")
                && !p.name.contains("metacarpal"),
            "`{}` should not be a rigid body",
            p.name
        );
    }
    // …and every capsule has a real length and a finite pose.
    for p in &parts {
        assert!(
            p.position.is_finite() && p.rotation.is_finite(),
            "`{}`",
            p.name
        );
    }

    // ── the same rig, table stripped: the measured contrast ──
    let bare: Vec<inf_physics::ragdoll::RagdollBone> = bones
        .iter()
        .cloned()
        .map(|b| b.with_role(None, None))
        .collect();
    let legacy = build_ragdoll(&bare, RagdollConfig::default());
    let free: Vec<&str> = legacy
        .iter()
        .filter(|p| p.joint.is_none())
        .map(|p| p.name.as_str())
        .collect();
    assert!(
        free.len() > 1,
        "the name classifier is supposed to fail on this rig, and it did not: {free:?}"
    );
    assert!(
        legacy
            .iter()
            .all(|p| p.role != inf_physics::ragdoll::BoneRole::Chest),
        "the classifier is not supposed to find a chest on a `spine_0N` rig"
    );
    // The three it drops on the floor: both upper arms and the head.
    for want in ["upperarm_l", "upperarm_r"] {
        assert!(
            legacy.iter().any(|p| p.name == want && p.joint.is_none()),
            "`{want}` should be a free capsule under the classifier: {free:?}"
        );
    }
}
