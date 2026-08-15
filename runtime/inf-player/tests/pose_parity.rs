//! **P24.1: the state machine drives the DRAWN pose, identically in both hosts.**
//!
//! Three claims, in the order they matter:
//!
//! 1. **The machine reaches the renderer at all.** Before this batch
//!    `inf_anim::eval_pose` had zero production callers: both hosts stepped an
//!    `AnimStateMachine` correctly and both render stores read only
//!    `AnimPlayer.clip + t`, so a character in a non-entry state drew its *rest
//!    pose* for ever. The palette assertions below are what make "the machine
//!    wins" — which `AnimStateMachine`'s own doc has claimed since P11.2 — true of
//!    the renderer.
//! 2. **The pose is sim state.** It is evaluated once, at fixed step, and folded
//!    into `RuntimeSim::state_bytes`, so it is covered by every determinism gate
//!    the engine already has rather than by a new one.
//! 3. **Both hosts agree, byte for byte.** The `deform_parity` precedent, on the
//!    pose. Note what makes this one *structurally* cheaper to keep true: the
//!    evaluation itself is one Ring-0 function
//!    (`inf_ecs::pose::step_pose_evaluation`) that both fixed steps call, so the
//!    two hosts cannot evaluate differently — what this file proves is that they
//!    *feed* it the same thing and *slot* it in the same place.
//!
//! The trace is bits, not floats (`pose_state_bytes` throughout), and every
//! comparison is guarded against vacuity: two characters both drawn at rest agree
//! perfectly, which is exactly the state this batch exists to leave behind.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3, Mat4, Quat, Vec3};
use uuid::Uuid;

use inf_anim::{
    AnimClip, Interpolation, Joint, JointTrack, JointTransform, QuatTrack, Skeleton, SkeletonAsset,
    SmState, SmTransition, Socket, StateMachine, Vec3Track,
};
use inf_ecs::components::{AnimPlayer, AnimStateMachine, AttachedTo, SkeletalMesh, Transform};
use inf_ecs::math::Vec3d;
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_render::{SkinnedMeshData, SkinnedVertex};

const HZ: f64 = 60.0;
const STEPS: u32 = 4;

const HERO: Uuid = Uuid::from_u128(0x2401_0000_0000_0000_0000_0000_0000_0001);
const SM: Uuid = Uuid::from_u128(0x2401_0000_0000_0000_0000_0000_0000_0002);
const SKEL: Uuid = Uuid::from_u128(0x2401_0000_0000_0000_0000_0000_0000_0003);
const MESH: Uuid = Uuid::from_u128(0x2401_0000_0000_0000_0000_0000_0000_0004);
const IDLE: Uuid = Uuid::from_u128(0x2401_0000_0000_0000_0000_0000_0000_0010);
const WALK: Uuid = Uuid::from_u128(0x2401_0000_0000_0000_0000_0000_0000_0011);
const RUN: Uuid = Uuid::from_u128(0x2401_0000_0000_0000_0000_0000_0000_0012);

// ── the fixture ─────────────────────────────────────────────────────────────

/// A 3-joint chain along +Y with a `hand_r` socket on the tip.
fn rig() -> SkeletonAsset {
    let mut joints = Vec::new();
    let mut global = Mat4::IDENTITY;
    for i in 0..3 {
        let local = JointTransform::from_trs(
            if i == 0 { Vec3::ZERO } else { Vec3::Y },
            Quat::IDENTITY,
            Vec3::ONE,
        );
        global *= local.to_mat4();
        joints.push(Joint {
            name: format!("j{i}"),
            parent: if i == 0 { None } else { Some(i as u16 - 1) },
            inverse_bind: global.inverse().to_cols_array(),
            local_bind: local,
        });
    }
    SkeletonAsset::with_sockets(
        Skeleton::new(joints).unwrap(),
        vec![Socket::new("hand_r", 2)],
    )
}

/// A clip holding joint 1 at `deg` about +Z for its whole (1 s) duration, so the
/// pose a state produces is a *constant* the assertions can name.
///
/// **One key, stepped** — deliberately. Two identical keys under `Linear` are not
/// a constant: `slerp(q, q, frac).normalize()` drifts by a ULP as `frac` moves,
/// which is invisible to a human and fatal to a byte comparison. A fixture whose
/// "nothing changed" arm is only approximately true cannot support the arms that
/// say something DID change.
fn hold(deg: f32) -> AnimClip {
    let q = Quat::from_rotation_z(deg.to_radians()).to_array();
    AnimClip {
        name: format!("hold{deg}"),
        duration: 1.0,
        tracks: vec![JointTrack {
            joint: 1,
            translation: None,
            rotation: Some(QuatTrack::new(vec![0.0], vec![q], Interpolation::Step)),
            scale: None,
        }],
    }
}

/// A clip that **actually interpolates**: two distinct keys under
/// `Interpolation::Linear`, which is the `#[default]` and what the glTF importer
/// emits.
///
/// # Why the fixture had to grow this (P24.2 audit M-SLERP)
///
/// Every arm above used `hold` — one key, `Step` — and the machine's transitions
/// had `duration: 0.0`. That is the **one configuration in which no quaternion
/// blend runs at all**: `Step` never slerps, and a zero-duration cross-fade never
/// calls `blend_poses`. So the parity gate was certifying portability on exactly
/// the path that avoids the non-portable operation, and `Quat::slerp` — three
/// `std` `sin` calls — sat on the committed trace unexamined.
///
/// Its keys are 40° apart so the interpolated value at any `frac` is far from
/// both, and the joint it drives is the same one `hold` drives, so the two
/// fixtures are directly comparable.
fn sweep(from_deg: f32, to_deg: f32) -> AnimClip {
    AnimClip {
        name: format!("sweep{from_deg}_{to_deg}"),
        duration: 1.0,
        tracks: vec![JointTrack {
            joint: 1,
            translation: None,
            rotation: Some(QuatTrack::new(
                vec![0.0, 1.0],
                vec![
                    Quat::from_rotation_z(from_deg.to_radians()).to_array(),
                    Quat::from_rotation_z(to_deg.to_radians()).to_array(),
                ],
                Interpolation::Linear,
            )),
            scale: None,
        }],
    }
}

/// idle → walk → run, both transitions **unconditional**, so the machine walks
/// its states on consecutive fixed steps with no actor and no Blueprint variable
/// in the picture. Each state holds a different angle, so "which state is the
/// machine in" is directly readable off the pose.
fn machine() -> StateMachine {
    StateMachine {
        states: vec![
            SmState::clip("idle", *IDLE.as_bytes()),
            SmState::clip("walk", *WALK.as_bytes()),
            SmState::clip("run", *RUN.as_bytes()),
        ],
        transitions: vec![
            SmTransition {
                from: 0,
                to: 1,
                duration: 0.0,
                conditions: vec![],
                exit_time: None,
            },
            SmTransition {
                from: 1,
                to: 2,
                duration: 0.0,
                conditions: vec![],
                exit_time: None,
            },
        ],
        entry: 0,
    }
}

fn pose_clips() -> BTreeMap<Uuid, AnimClip> {
    BTreeMap::from([(IDLE, hold(0.0)), (WALK, hold(35.0)), (RUN, hold(80.0))])
}

fn skeletons() -> BTreeMap<Uuid, SkeletonAsset> {
    BTreeMap::from([(SKEL, rig())])
}

fn machines() -> BTreeMap<Uuid, StateMachine> {
    BTreeMap::from([(SM, machine())])
}

/// The character's components — machine-driven **and** carrying an `AnimPlayer`
/// on a clip that resolves to nothing.
///
/// The player is there on purpose: rule 2 of the pose rule says the machine
/// overrules it, and a fixture with no `AnimPlayer` could not tell "the machine
/// won" from "there was nothing to lose to".
fn character() -> (AnimStateMachine, SkeletalMesh, AnimPlayer) {
    (
        AnimStateMachine {
            sm: Some(SM),
            params_from_vars: true,
            ..Default::default()
        },
        SkeletalMesh {
            mesh: Some(MESH),
            skeleton: Some(SKEL),
        },
        AnimPlayer {
            clip: Some(Uuid::from_u128(0xC0FFEE)),
            t: 0.25,
            ..AnimPlayer::default()
        },
    )
}

// ── the two hosts ───────────────────────────────────────────────────────────

fn player_trace() -> Vec<Vec<u8>> {
    player_trace_with_ik(None)
}

/// The player's trace, optionally with an IK goal set through the Ring-0 write
/// door before the first step (P24.2).
fn player_trace_with_ik(goal: Option<inf_ecs::IkGoal>) -> Vec<Vec<u8>> {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character());
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    if let Some(g) = goal {
        inf_ecs::set_ik_goals(sim.world_mut(), HERO, vec![g]);
    }
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            inf_ecs::pose::pose_state_bytes(sim.world())
        })
        .collect()
}

fn editor_trace() -> Vec<Vec<u8>> {
    editor_trace_with_ik(None)
}

/// The editor's trace, through the same Ring-0 door.
fn editor_trace_with_ik(goal: Option<inf_ecs::IkGoal>) -> Vec<Vec<u8>> {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(character());
    doc.world_mut().mark_dirty();
    let mut session = SimSession::enter(&mut doc, Vec::new(), DVec2::ZERO, HZ);
    session.set_state_machines(machines());
    session.set_skeletons(skeletons());
    session.set_pose_clips(pose_clips());
    if let Some(g) = goal {
        inf_ecs::set_ik_goals(doc.world_mut(), HERO, vec![g]);
    }
    let out = (0..STEPS)
        .map(|_| {
            session.step_once(&mut doc, SimInput::default());
            inf_ecs::pose::pose_state_bytes(doc.world())
        })
        .collect();
    session.exit(&mut doc);
    out
}

/// **ANTI-VACUITY.** Two empty traces are equal, and so are two traces of a
/// character standing still: the fixture has to have posed something, and the
/// pose has to have MOVED when the machine changed state.
fn assert_not_vacuous(trace: &[Vec<u8>]) {
    assert_eq!(trace.len() as u32, STEPS);
    assert!(
        !trace[0].is_empty(),
        "step 0 published no pose at all — the machine never reached a skeleton, \
         which is precisely the pre-P24.1 state this gate exists to leave behind"
    );
    assert_ne!(
        trace[0], trace[1],
        "the pose did not change when the machine transitioned idle → walk"
    );
    assert_ne!(
        trace[1], trace[2],
        "the pose did not change when the machine transitioned walk → run"
    );
    // …and it settles once the machine has nowhere left to go, so the difference
    // above is the STATE changing and not a play-head drifting.
    assert_eq!(
        trace[2], trace[3],
        "a machine parked in its last state must publish a settled pose (both \
         clips hold a constant angle), or the assertions above prove nothing \
         about transitions"
    );
}

#[test]
fn both_hosts_evaluate_the_same_pose_byte_for_byte() {
    let player = player_trace();
    let editor = editor_trace();
    assert_not_vacuous(&player);
    assert_not_vacuous(&editor);
    assert_eq!(
        player, editor,
        "the shipped player and the editor Simulate evaluated different poses for \
         the same machine — PIE would stop matching shipping, and a character \
         would animate differently in the preview than in the shipped build"
    );
}

/// The evaluated pose is folded into the sim's **trace**, so every determinism
/// gate the engine already has covers it: the replay fold, `step_state_hash`, and
/// the PIE `Frame::state_hash` the PIE == shipping arms compare.
#[test]
fn the_pose_rides_the_sim_trace() {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character());
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());

    sim.step_once(RuntimeInput::default());
    let idle_to_walk = sim.state_bytes();
    sim.step_once(RuntimeInput::default());
    let walk_to_run = sim.state_bytes();
    assert_ne!(
        idle_to_walk, walk_to_run,
        "the sim trace must move when the machine transitions — if it does not, \
         the pose is not sim state and no replay/PIE gate can see it"
    );
    // A world with no machine at all leaves the trace byte-identical to its
    // pre-P24.1 self, which is what keeps every existing golden trace valid.
    let mut bare = RuntimeSim::new(EcsWorld::new(), Vec::new(), DVec2::ZERO, HZ);
    assert!(inf_ecs::pose::pose_state_bytes(bare.world()).is_empty());
    let before = bare.state_bytes();
    bare.step_once(RuntimeInput::default());
    assert_eq!(before, bare.state_bytes());
}

/// **The headline claim, end to end**: sim → store → *palette*.
///
/// A character in a non-entry machine state draws a DIFFERENT palette from one at
/// rest — asserted on the world (the palette bytes the skinned pass uploads), not
/// on pixels. This is what "the machine drives the drawn pose" means, and it is
/// the assertion that would have failed on every commit before this batch.
#[test]
fn the_drawn_palette_follows_the_machine() {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character());
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());

    // The store the shipped player's projector resolves through.
    let mut store = inf_player::skinned::SkinnedRegistry::new();
    store.insert_mesh(MESH, triangle());
    store.insert_skeleton(SKEL, rig().skeleton);
    let sm = SkeletalMesh {
        mesh: Some(MESH),
        skeleton: Some(SKEL),
    };
    let rest = store
        .resolve_skinned(&sm, None, None)
        .expect("a bound character draws")
        .palette;

    sim.step_once(RuntimeInput::default()); // → walk
    let walk = palette(&store, &sm, &sim);
    sim.step_once(RuntimeInput::default()); // → run
    let run = palette(&store, &sm, &sim);

    assert_ne!(
        rest[1].to_cols_array(),
        walk[1].to_cols_array(),
        "a machine-driven character still drew its REST pose — this is the P24.1 \
         defect itself"
    );
    assert_ne!(
        walk[1].to_cols_array(),
        run[1].to_cols_array(),
        "the drawn palette did not follow the machine's transition"
    );
}

/// The Ring-1 resolver seeds all four maps a machine-driven character needs —
/// including the **transitively** referenced clips, which no component names.
///
/// Without that hop the sim resolves the machine, steps it correctly, and poses
/// every state as the rest pose: the P24.1 defect one level up, and invisible to
/// a comparison of two hosts that both lack the clips.
#[test]
fn the_editor_resolver_seeds_the_clips_the_machine_plays() {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(character());

    let bytes: BTreeMap<Uuid, Vec<u8>> = BTreeMap::from([
        (
            SM,
            inf_asset::encode(&inf_anim::StateMachineAsset::new(
                machine(),
                Some(*SKEL.as_bytes()),
            ))
            .unwrap(),
        ),
        (SKEL, inf_asset::encode(&rig()).unwrap()),
        (
            IDLE,
            inf_asset::encode(&inf_anim::AnimClipAsset::new(hold(0.0), None)).unwrap(),
        ),
        (
            WALK,
            inf_asset::encode(&inf_anim::AnimClipAsset::new(hold(35.0), None)).unwrap(),
        ),
        (
            RUN,
            inf_asset::encode(&inf_anim::AnimClipAsset::new(hold(80.0), None)).unwrap(),
        ),
    ]);

    let (machines, _root, skels, clips) =
        inf_editor_core::simulate::resolve_anim_assets(&doc, |g| bytes.get(&g).cloned());
    assert_eq!(machines.len(), 1, "the machine resolves");
    assert_eq!(skels.len(), 1, "the SkeletalMesh's skeleton resolves");
    assert_eq!(
        clips.len(),
        3,
        "every clip the machine's three states play must resolve — none of them \
         is named by any component"
    );
    // The clip GUIDs are the machine's own, not the AnimPlayer's (which does not
    // resolve at all here) — a resolver that walked only components would have
    // produced an empty map and this count would be 0.
    for c in [IDLE, WALK, RUN] {
        assert!(clips.contains_key(&c), "{c} missing");
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn palette(
    store: &inf_player::skinned::SkinnedRegistry,
    sm: &SkeletalMesh,
    sim: &RuntimeSim,
) -> Vec<Mat4> {
    let posed = inf_ecs::pose::evaluated_pose(sim.world(), HERO);
    assert!(posed.is_some(), "the sim published no pose to draw");
    store
        .resolve_skinned(sm, None, posed)
        .expect("a bound character draws")
        .palette
}

/// A 3-vertex skinned triangle — the smallest thing the skinned pass can draw.
fn triangle() -> SkinnedMeshData {
    let v = |x: f32, y: f32, j: u32| SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [x, y],
        joints: [j, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    SkinnedMeshData {
        vertices: vec![v(0.0, 0.0, 0), v(1.0, 0.0, 1), v(0.0, 2.0, 2)],
        indices: vec![0, 1, 2],
    }
}
// ── P24.1 repair 3: the socket attachment, in both hosts ────────────────────

const SWORD: Uuid = Uuid::from_u128(0x2401_0000_0000_0000_0000_0000_0000_0020);

/// The sword that rides the character's `hand_r` socket.
fn sword() -> (Transform, AttachedTo) {
    (
        Transform::IDENTITY,
        AttachedTo::new(HERO, "hand_r", Vec3d::ZERO),
    )
}

/// Where the sword ended up, in world space.
fn follower_at(world: &EcsWorld) -> DVec3 {
    let e = world.entity_of(SWORD).expect("the sword exists");
    world
        .world()
        .get::<inf_ecs::components::GlobalTransform>(e)
        .expect("a global transform")
        .translation()
}

fn player_attachment_trace() -> Vec<DVec3> {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character());
    let s = world.spawn_with_guid(SWORD, "Sword", None);
    world.world_mut().entity_mut(s).insert(sword());
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            follower_at(sim.world())
        })
        .collect()
}

fn editor_attachment_trace() -> Vec<DVec3> {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(character());
    let s = doc.create_with_guid(SWORD, inf_editor_core::ipc::SpawnKind::Empty, "Sword", None);
    doc.world_mut().world_mut().entity_mut(s).insert(sword());
    doc.world_mut().mark_dirty();
    let mut session = SimSession::enter(&mut doc, Vec::new(), DVec2::ZERO, HZ);
    session.set_state_machines(machines());
    session.set_skeletons(skeletons());
    session.set_pose_clips(pose_clips());
    let out = (0..STEPS)
        .map(|_| {
            session.step_once(&mut doc, SimInput::default());
            follower_at(doc.world())
        })
        .collect();
    session.exit(&mut doc);
    out
}

/// **The socket lands on the animated joint, in both hosts, and it MOVES.**
///
/// The independent expectation is computed here out of `global_transforms` — the
/// test does not read back the expression the attachment system wrote — and the
/// anti-vacuity arm is the one that matters: a sword pinned to the character's
/// ORIGIN, which is where it rode for three phases, agrees with a correct one on
/// every assertion that only looks at one frame.
#[test]
fn both_hosts_land_the_attachment_on_the_same_animated_joint() {
    let player = player_attachment_trace();
    let editor = editor_attachment_trace();
    assert_eq!(player.len() as u32, STEPS);
    assert_eq!(player, editor, "the two hosts placed the sword differently");

    // It moved with the machine's state, so it is riding the JOINT and not the
    // entity origin (which never moves in this fixture).
    assert_ne!(
        player[0], player[1],
        "the attachment did not follow the machine out of `idle` — a sword on the \
         character's origin would produce exactly this trace"
    );
    assert_ne!(player[1], player[2], "walk → run did not move the socket");

    // …and it is where the skeleton says it is. The machine is in `run` from step
    // 2 on, so rebuild that pose and place joint 2 (the socket's) in world space.
    let rig = rig();
    let mut want_pose = inf_anim::Pose::rest(&rig.skeleton);
    want_pose.locals[1].rotation = Quat::from_rotation_z(80f32.to_radians()).to_array();
    let want = inf_anim::global_transforms(&rig.skeleton, &want_pose)[2]
        .transform_point3(Vec3::ZERO)
        .as_dvec3();
    assert!(
        (player[2] - want).length() < 1e-5,
        "sword at {:?}, the animated joint is at {want:?}",
        player[2]
    );
    // The origin control: the character never leaves the origin, so a
    // pelvis-riding sword would sit at zero for the whole trace.
    assert!(
        player.iter().all(|p| p.length() > 1e-6),
        "the sword sat on the character's origin"
    );
}

// ── P24.2: IK is a post-pass in the SAME door, so both hosts inherit it ──────

/// **The IK arm of the parity gate.**
///
/// Three claims, and the middle one is what makes the other two mean anything:
///
/// 1. **A goal moves the pose.** The same fixture with and without an IK target
///    produces different trace bytes — so the post-pass ran, and this is not two
///    identical traces agreeing about nothing.
/// 2. **Moving the target moves the trace.** Two *different* targets disagree, so
///    the pose is a function of the goal rather than of "an IK pass happened".
/// 3. **Both hosts agree, byte for byte**, with IK on.
///
/// It costs no host-side change at all, which is the point of putting the solve
/// inside `step_pose_evaluation`: neither `SimSession::fixed_step` nor
/// `RuntimeSim::fixed_step` mentions IK, so they cannot mention it differently.
///
/// # "Both hosts" means IN-PROCESS, and that is a real limit (audit M-PIE)
///
/// This compares `SimSession` and `RuntimeSim` inside **one** process. It is not
/// a `--pie` subprocess arm, and the P24.2 report claimed one — wrongly. Every
/// arm in this file is in-process.
///
/// The gap is not closed by adding a test-only hook that injects a goal into the
/// player, and that ruling is the boot-paths law: a boot path that exists only
/// for a test is a path the shipped binary does not take, and P21.4 already paid
/// for exactly that (`sim_from_payload` had to become the one PIE seam because
/// the real `--pie` binary was building its sim differently from every test).
///
/// It closes naturally at **P24.3**, where the authored `IkTarget` component
/// lands behind the scene bump: the payload then carries the component, the
/// subprocess builds its sim from that payload through the door it already uses,
/// and IK engages with no hook at all. Ledgered in ROADMAP §12's P24 block.
#[test]
fn both_hosts_apply_ik_the_same_way_and_a_moved_target_moves_the_trace() {
    // The fixture's rig is a straight 3-joint chain up +Y with 1 m bones, so
    // `[0, 1, 2]` is a two-bone chain with 2 m of reach.
    let goal = |x: f32| inf_ecs::IkGoal {
        chain: vec![0, 1, 2],
        target: [x, 1.2, 0.0],
        pole: Some([2.0, 1.0, 0.0]),
        weight: 1.0,
    };

    let plain = player_trace();
    let with_ik = player_trace_with_ik(Some(goal(1.2)));
    assert_not_vacuous(&plain);

    // (1) The post-pass ran.
    assert_ne!(
        plain, with_ik,
        "an IK goal did not change the pose at all — the post-pass is not \
         running, and every assertion below would pass on a no-op"
    );
    // …and it did not simply erase the machine: the trace still MOVES as the
    // machine walks its states, so IK bends the animated pose rather than
    // replacing it.
    assert_ne!(
        with_ik[0], with_ik[1],
        "with IK on, the pose stopped following the machine"
    );

    // (2) ANTI-VACUITY: a different target is a different pose.
    let moved = player_trace_with_ik(Some(goal(-1.2)));
    assert_ne!(
        with_ik, moved,
        "moving the IK target did not move the trace, so the pose is not a \
         function of the goal"
    );

    // (3) The two hosts agree, with IK on.
    let editor = editor_trace_with_ik(Some(goal(1.2)));
    assert_eq!(
        with_ik, editor,
        "the editor's Simulate and the player's runtime evaluated different \
         IK-solved poses"
    );
    // …and the no-IK traces still agree, so arm (3) is not passing because both
    // hosts happen to ignore the goal.
    assert_eq!(plain, editor_trace());
}

/// A goal the chain cannot satisfy is a **value, never a NaN or a panic** — and
/// a goal naming a chain that is not a chain costs its own chain and nothing
/// else.
///
/// The refusal is swallowed inside the fixed step on purpose (one bad goal must
/// not stop a level animating), so what this pins is the *outcome*: the trace is
/// still produced, still finite, and still identical between two runs.
#[test]
fn a_refused_or_unreachable_goal_leaves_a_usable_trace() {
    for goal in [
        // Unreachable: 100 m away from a 2 m chain.
        inf_ecs::IkGoal {
            chain: vec![0, 1, 2],
            target: [100.0, 0.0, 0.0],
            pole: None,
            weight: 1.0,
        },
        // Not a chain (0 is not the parent of 2 here — it skips a joint).
        inf_ecs::IkGoal {
            chain: vec![0, 2],
            target: [0.5, 1.0, 0.0],
            pole: None,
            weight: 1.0,
        },
        // A joint the skeleton does not have.
        inf_ecs::IkGoal {
            chain: vec![0, 1, 9],
            target: [0.5, 1.0, 0.0],
            pole: None,
            weight: 1.0,
        },
    ] {
        let a = player_trace_with_ik(Some(goal.clone()));
        let b = player_trace_with_ik(Some(goal.clone()));
        assert_eq!(a.len() as u32, STEPS, "{goal:?} lost the trace");
        assert!(!a[0].is_empty(), "{goal:?} stopped the character posing");
        assert_eq!(a, b, "{goal:?} is not deterministic");
        assert_eq!(
            a,
            editor_trace_with_ik(Some(goal.clone())),
            "{goal:?}: the two hosts disagree about a refused goal"
        );
    }
}

/// **B1 at the host boundary**: a goal whose numbers are not numbers cannot
/// panic the fixed step, and cannot put a NaN in the trace.
///
/// The audit's finding was reachable exactly here — `step_pose_evaluation` calls
/// `solve_chain` on every goal, and before the fix a NaN target reached
/// `f32::clamp`'s `assert!(min <= max)` and aborted the session, while `+inf`
/// produced `[NaN; 4]` in `pose.locals` and rode `state_bytes` into the palette.
///
/// The assertion is against the **no-IK baseline**, not merely "it did not
/// crash": a refused goal must leave the pose exactly as the machine evaluated
/// it, so the trace is byte-identical to a character with no goal at all.
#[test]
fn a_non_finite_goal_cannot_panic_the_step_or_reach_the_trace() {
    let baseline = player_trace();
    for target in [
        [f32::NAN, 1.0, 0.0],
        [f32::INFINITY, 1.0, 0.0],
        [f32::NEG_INFINITY, 1.0, 0.0],
        [0.5, f32::NAN, 0.0],
    ] {
        let goal = inf_ecs::IkGoal {
            chain: vec![0, 1, 2],
            target,
            pole: None,
            weight: 1.0,
        };
        let trace = player_trace_with_ik(Some(goal.clone()));
        assert_eq!(
            trace, baseline,
            "{target:?}: a non-finite goal changed the trace; it must be refused \
             and leave the machine's pose alone"
        );
        // Not a single NaN reached the bytes. `pose_state_bytes` is little-endian
        // f32s, so this reads them back and checks every one.
        for step in &trace {
            for chunk in step[32..].chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                assert!(!v.is_nan(), "{target:?}: a NaN reached the trace");
            }
        }
        // Both hosts agree about the refusal, as they do about everything else.
        assert_eq!(trace, editor_trace_with_ik(Some(goal)));
    }
    // ANTI-VACUITY: a FINITE goal on the same chain does move the trace, so the
    // equalities above are about the refusal and not about IK being inert.
    let live = player_trace_with_ik(Some(inf_ecs::IkGoal {
        chain: vec![0, 1, 2],
        target: [1.2, 1.2, 0.0],
        pole: Some([2.0, 1.0, 0.0]),
        weight: 1.0,
    }));
    assert_ne!(live, baseline);
}

/// **B1's ACTUAL vector at the host**: a non-finite *pose*, not a non-finite
/// target.
///
/// Written after the first attempt at this arm failed to falsify. That one fed a
/// NaN **target**, which `solve_chain` has refused since the day it was written
/// (`finite("an IK target", …)`) — so removing every guard the B1 fix added left
/// it green. The finding was about the pose: a clip whose key is not a number
/// puts a NaN in `pose.locals`, `two_bone_positions` computed
/// `l1 = (mid − root).length() = NaN`, `NaN < MIN` is **false**, and the guard
/// waved it through to `f32::clamp`, whose `assert!(min <= max)` aborts the
/// fixed step.
///
/// So the fixture manufactures one: a state machine playing a clip with a
/// `[NaN; 3]` **translation** key. Two claims, and the first is the blocker:
///
/// 1. **the step completes** — pre-fix this panicked, taking the session down;
/// 2. the trace is byte-identical to the same world with no goal at all, so the
///    refused solve contributed nothing rather than half-solving.
///
/// # Why translation and not rotation (hardening wave B, C4-31)
///
/// The fixture used to put its NaN in a `[NaN; 4]` **rotation** key, and that
/// route no longer produces a non-finite pose: `QuatTrack::sample` now goes
/// through `normalized_or_identity`, because `glam` is pinned with no
/// `glam-assert` feature and `Quat::normalize()` on a degenerate quaternion is
/// four NaNs *silently* — which rode `pose_state_bytes`, this very trace, into
/// the PIE-==-shipping parity comparison. Closing that at the sampler is right,
/// and it un-broke this fixture: the clip sampled to identity, the pose was
/// finite, the solve legitimately ran, and the trace moved.
///
/// The property this arm names is unchanged and still real — `solve_chain`
/// refuses a non-finite **pose**, and the docstring above describes the
/// mechanism in terms of `(mid − root).length()`, which is a *translation*
/// computation. So the fixture moves to the channel that computation actually
/// reads, and that nothing between here and there sanitizes: `Vec3Track::sample`
/// hands its stored array to `Vec3::from_array` unguarded, by design — a
/// translation has no normalization to fall back to, and inventing one would be
/// making up a position.
///
/// The vacuity assertion at the end of this test is what makes the swap safe to
/// trust: it fails if the fixture's pose is finite, whichever channel carries
/// the poison.
#[test]
fn a_goal_on_a_non_finite_pose_refuses_instead_of_panicking() {
    fn broken_clips() -> BTreeMap<Uuid, AnimClip> {
        BTreeMap::from([(
            IDLE,
            AnimClip {
                name: "broken".into(),
                duration: 1.0,
                tracks: vec![JointTrack {
                    joint: 1,
                    // Not a number, in the one place a corrupt clip, a divide by
                    // zero in a blend, or a bad import would put one — and the
                    // channel `two_bone_positions` measures. See the doc above
                    // for why this is the translation and no longer the rotation.
                    translation: Some(Vec3Track::new(
                        vec![0.0],
                        vec![[f32::NAN; 3]],
                        Interpolation::Step,
                    )),
                    rotation: None,
                    scale: None,
                }],
            },
        )])
    }
    fn run(goal: Option<inf_ecs::IkGoal>) -> (Vec<Vec<u8>>, Vec<inf_ecs::IkOutcome>) {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(HERO, "Hero", None);
        world.world_mut().entity_mut(e).insert(character());
        world.mark_dirty();
        let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
        // One state, no transitions: the machine sits on the broken clip.
        sim.set_state_machines(BTreeMap::from([(
            SM,
            StateMachine {
                states: vec![SmState::clip("idle", *IDLE.as_bytes())],
                transitions: vec![],
                entry: 0,
            },
        )]));
        sim.set_skeletons(skeletons());
        sim.set_pose_clips(broken_clips());
        if let Some(g) = goal {
            inf_ecs::set_ik_goals(sim.world_mut(), HERO, vec![g]);
        }
        let bytes = (0..STEPS)
            .map(|_| {
                // The blocker: pre-fix, THIS call aborted the process.
                sim.step_once(RuntimeInput::default());
                inf_ecs::pose::pose_state_bytes(sim.world())
            })
            .collect();
        let verdicts = inf_ecs::ik_outcomes(sim.world(), HERO)
            .map(|v| v.to_vec())
            .unwrap_or_default();
        (bytes, verdicts)
    }

    let (without, no_verdicts) = run(None);
    assert!(no_verdicts.is_empty(), "no goal, no verdict");
    let (with, verdicts) = run(Some(inf_ecs::IkGoal {
        chain: vec![0, 1, 2],
        target: [0.5, 1.0, 0.0],
        pole: None,
        weight: 1.0,
    }));

    // **The load-bearing claim** (hardening wave B). Byte-equality alone cannot
    // carry it: the poisoned pose puts NaNs in BOTH traces, and two identical
    // NaN bit patterns compare EQUAL as bytes — so a solve that ran on
    // not-numbers and wrote not-numbers back is invisible to it. Measured:
    // stripping every finiteness guard out of `inf_anim::ik` leaves the byte
    // assertion below green. The recorded verdict is the half that falsifies.
    match verdicts.as_slice() {
        [inf_ecs::IkOutcome::Refused(e)] => {
            let named = format!("{e:?}");
            assert!(
                named.contains("NonFinite"),
                "refused, but not for the reason this arm is about: {named}"
            );
        }
        other => panic!(
            "a goal over a non-finite pose must be REFUSED by name; the solve \
             returned {other:?}, so it ran on numbers that are not numbers"
        ),
    }
    assert_eq!(
        with, without,
        "the refused solve still moved the trace, so it half-solved rather than \
         contributing nothing"
    );
    // The fixture really is broken — otherwise this test proves nothing about
    // non-finite poses, only that two clean traces agree.
    assert!(
        without[0][32..]
            .chunks_exact(4)
            .any(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_nan()),
        "the fixture's pose is finite, so it is not exercising B1 at all"
    );
}

// ── P24.2 audit M-SLERP: the gate must certify the REAL configuration ───────

/// **The parity gate on the configuration that actually blends.**
///
/// Every other arm in this file uses `Step` keys and zero-duration transitions —
/// the one shape in which no quaternion blend runs at all. This one uses
/// `Interpolation::Linear` clips (the `#[default]`, and what the glTF importer
/// emits) **and** a cross-fade with `duration > 0`, so both blend paths execute:
/// `QuatTrack::sample`'s slerp between two keys, and `blend_poses`' slerp between
/// two states.
///
/// Both are now `inf_math::pslerp`. Before this batch they were `Quat::slerp`,
/// which is three `std` `sin` calls, and the P14 law says those are not
/// bit-identical across targets — on a trace that is folded into `state_bytes`
/// and compared between the editor and the shipped player.
///
/// Nothing in a single-machine test can *prove* cross-target identity; what it
/// can do is prove the fixture reaches the code, which is the half that was
/// missing. `the_animation_blend_uses_no_platform_dependent_trigonometry` is the
/// other half.
#[test]
fn both_hosts_agree_on_a_fixture_that_really_interpolates() {
    fn blending_machine() -> StateMachine {
        StateMachine {
            states: vec![
                SmState::clip("idle", *IDLE.as_bytes()),
                SmState::clip("walk", *WALK.as_bytes()),
            ],
            transitions: vec![SmTransition {
                from: 0,
                to: 1,
                // NON-ZERO: this is what makes `blend_poses` run at all.
                duration: 0.25,
                conditions: vec![],
                exit_time: None,
            }],
            entry: 0,
        }
    }
    fn clips() -> BTreeMap<Uuid, AnimClip> {
        BTreeMap::from([
            (IDLE, sweep(0.0, 40.0)),
            (WALK, sweep(40.0, 80.0)),
            (RUN, sweep(80.0, 120.0)),
        ])
    }
    fn player() -> Vec<Vec<u8>> {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(HERO, "Hero", None);
        world.world_mut().entity_mut(e).insert(character());
        world.mark_dirty();
        let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
        sim.set_state_machines(BTreeMap::from([(SM, blending_machine())]));
        sim.set_skeletons(skeletons());
        sim.set_pose_clips(clips());
        (0..STEPS)
            .map(|_| {
                sim.step_once(RuntimeInput::default());
                inf_ecs::pose::pose_state_bytes(sim.world())
            })
            .collect()
    }
    fn editor() -> Vec<Vec<u8>> {
        let mut doc = SceneDoc::new();
        let e = doc.create_with_guid(HERO, inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
        doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .insert(character());
        doc.world_mut().mark_dirty();
        let mut session = SimSession::enter(&mut doc, Vec::new(), DVec2::ZERO, HZ);
        session.set_state_machines(BTreeMap::from([(SM, blending_machine())]));
        session.set_skeletons(skeletons());
        session.set_pose_clips(clips());
        let out = (0..STEPS)
            .map(|_| {
                session.step_once(&mut doc, SimInput::default());
                inf_ecs::pose::pose_state_bytes(doc.world())
            })
            .collect();
        session.exit(&mut doc);
        out
    }

    let p = player();
    // NOT `assert_not_vacuous`: that helper asserts the last two steps are
    // EQUAL, because the `hold` fixture parks on a constant angle. A sweeping
    // clip legitimately keeps moving, which is the entire point of this arm — so
    // it carries its own anti-vacuity instead, and a stronger one.
    assert_eq!(p.len() as u32, STEPS);
    assert!(p.iter().all(|b| !b.is_empty()), "nothing was posed");
    // THE FIXTURE REALLY BLENDS: consecutive steps differ *within* a state, which
    // a `Step` clip cannot produce, and which is the whole point of this arm.
    assert_ne!(
        p[0], p[1],
        "the pose is constant between steps — the fixture is not interpolating, \
         so this arm is certifying the same thing the Step arms already do"
    );
    // The two hosts agree over the interpolating path, byte for byte.
    assert_eq!(
        p,
        editor(),
        "the editor's Simulate and the player's runtime disagree once a real \
         quaternion blend is in the picture"
    );
}

/// **The IK verdict is OBSERVABLE** (P24.2 audit M-CALLER).
///
/// The fixed step spelled the solve `let _ = solve_chain(..)`, which made four
/// typed error variants and every field of `IkReport` unreachable by any layer of
/// the engine: an unreachable target, a chain that is not a chain, and a clean hit
/// were the same silence. This asserts all three are now distinguishable from
/// outside the step, through `inf_ecs::ik_outcomes`.
#[test]
fn the_step_records_what_each_ik_goal_did() {
    fn run(goal: inf_ecs::IkGoal) -> Vec<inf_ecs::IkOutcome> {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(HERO, "Hero", None);
        world.world_mut().entity_mut(e).insert(character());
        world.mark_dirty();
        let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
        sim.set_state_machines(machines());
        sim.set_skeletons(skeletons());
        sim.set_pose_clips(pose_clips());
        inf_ecs::set_ik_goals(sim.world_mut(), HERO, vec![goal]);
        sim.step_once(RuntimeInput::default());
        inf_ecs::ik_outcomes(sim.world(), HERO)
            .expect("a goal was set, so a verdict must exist")
            .to_vec()
    }

    // (a) REACHED — the rig is a 3-joint chain of 1 m bones, so a target 1.2 m
    //     out is comfortably inside its reach.
    let reached = run(inf_ecs::IkGoal {
        chain: vec![0, 1, 2],
        target: [1.2, 1.2, 0.0],
        pole: Some([2.0, 1.0, 0.0]),
        weight: 1.0,
    });
    match reached.as_slice() {
        [inf_ecs::IkOutcome::Solved(r)] => {
            assert!(r.reached, "{r:?}");
            assert!(r.reach_error < 1.0e-3, "{r:?}");
            assert!((r.chain_length - 2.0).abs() < 1e-4, "{r:?}");
        }
        other => panic!("expected one Solved, got {other:?}"),
    }

    // (b) OUT OF REACH — not a refusal, and the number says how far by.
    let missed = run(inf_ecs::IkGoal {
        chain: vec![0, 1, 2],
        target: [100.0, 0.0, 0.0],
        pole: None,
        weight: 1.0,
    });
    match missed.as_slice() {
        [inf_ecs::IkOutcome::Solved(r)] => {
            assert!(!r.reached, "{r:?}");
            assert!(
                (r.reach_error - 98.0).abs() < 0.1,
                "the miss distance must be the honest one: {r:?}"
            );
        }
        other => panic!("expected one Solved-but-unreached, got {other:?}"),
    }

    // (c) REFUSED, by name — 0 is not the parent of 2 in this rig.
    let refused = run(inf_ecs::IkGoal {
        chain: vec![0, 2],
        target: [0.5, 1.0, 0.0],
        pole: None,
        weight: 1.0,
    });
    match refused.as_slice() {
        [inf_ecs::IkOutcome::Refused(inf_anim::IkError::NotAChain { parent, child })] => {
            assert_eq!((*parent, *child), (0, 2));
        }
        other => panic!("expected a named refusal, got {other:?}"),
    }

    // ANTI-VACUITY: the three outcomes are actually different from each other.
    // Three identical `Solved` values would satisfy every `match` above only by
    // accident, and this says so directly.
    assert_ne!(reached, missed);
    assert_ne!(reached, refused);

    // …and the slot is REBUILT each step, not accumulated: a world whose goals
    // are cleared reports nothing rather than yesterday's verdict.
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character());
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    inf_ecs::set_ik_goals(
        sim.world_mut(),
        HERO,
        vec![inf_ecs::IkGoal {
            chain: vec![0, 1, 2],
            target: [1.2, 1.2, 0.0],
            pole: None,
            weight: 1.0,
        }],
    );
    sim.step_once(RuntimeInput::default());
    assert!(inf_ecs::ik_outcomes(sim.world(), HERO).is_some());
    inf_ecs::set_ik_goals(sim.world_mut(), HERO, Vec::new());
    sim.step_once(RuntimeInput::default());
    assert!(
        inf_ecs::ik_outcomes(sim.world(), HERO).is_none(),
        "a cleared goal left its verdict behind"
    );
}
