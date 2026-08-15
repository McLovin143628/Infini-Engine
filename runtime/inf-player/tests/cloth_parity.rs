//! **P24.4: a garment folds identically in both hosts.**
//!
//! The `pose_parity` precedent, on cloth. Four claims, in the order they matter:
//!
//! 1. **A `ClothSim` component is READ at all.** Scene v21 landed the component
//!    at P24.3 as reference-plus-knobs with the note *"as of P24.3 nothing
//!    simulates cloth"* written on the type. These arms are what take that
//!    sentence off it.
//! 2. **The garment is sim state.** It is folded at fixed step and appended to
//!    `RuntimeSim::state_bytes`, so it is covered by every determinism gate the
//!    engine already has — the replay fold, `step_state_hash`, and the PIE
//!    `Frame::state_hash` the PIE == shipping arms compare — rather than by a new
//!    one.
//! 3. **Both hosts agree, byte for byte.** Structurally cheaper to keep true than
//!    it looks: the solve is one Ring-0 function
//!    (`inf_ecs::cloth::step_cloth_simulation`) that both fixed steps call, so
//!    what these arms prove is that the two hosts *feed* it the same thing and
//!    *slot* it in the same place.
//! 4. **A level with no garment is byte-identical to its pre-P24.4 self**, which
//!    is what keeps every committed trace in the tree valid.
//!
//! Every comparison is guarded against vacuity: two characters both wearing
//! nothing agree perfectly, and so do two garments that never moved.

use std::collections::BTreeMap;

use glam::{DVec2, Mat4, Quat, Vec3};
use uuid::Uuid;

use inf_anim::{
    AnimClip, ClothAsset, ClothCapsule, ClothMaterial, Interpolation, Joint, JointTrack,
    JointTransform, QuatTrack, Skeleton, SkeletonAsset, SmState, SmTransition, StateMachine,
};
use inf_ecs::components::{AnimStateMachine, ClothSim, SkeletalMesh, Transform};
use inf_ecs::math::Vec3d;
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

const HZ: f64 = 60.0;
const STEPS: u32 = 12;

const HERO: Uuid = Uuid::from_u128(0x2404_0000_0000_0000_0000_0000_0000_0001);
const SM: Uuid = Uuid::from_u128(0x2404_0000_0000_0000_0000_0000_0000_0002);
const SKEL: Uuid = Uuid::from_u128(0x2404_0000_0000_0000_0000_0000_0000_0003);
const MESH: Uuid = Uuid::from_u128(0x2404_0000_0000_0000_0000_0000_0000_0004);
const CLOTH: Uuid = Uuid::from_u128(0x2404_0000_0000_0000_0000_0000_0000_0005);
const IDLE: Uuid = Uuid::from_u128(0x2404_0000_0000_0000_0000_0000_0000_0010);
const WAVE: Uuid = Uuid::from_u128(0x2404_0000_0000_0000_0000_0000_0000_0011);

// ── the fixture ─────────────────────────────────────────────────────────────

/// A 3-joint chain along +Y with 1 m bones — the same rig `pose_parity` uses, so
/// the two files' fixtures are directly comparable.
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
    SkeletonAsset::new(Skeleton::new(joints).unwrap())
}

/// A clip holding joint 1 at `deg` about +Z for its whole duration, so the pose a
/// state produces — and therefore where the collision capsule *is* — is a
/// constant the assertions can name. One key, `Step`: two identical `Linear` keys
/// drift by a ULP as the play-head moves, which is fatal to a byte comparison.
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

/// idle → wave, unconditional, so the machine leaves its entry state on the first
/// step and the capsule under the garment MOVES.
fn machine() -> StateMachine {
    StateMachine {
        states: vec![
            SmState::clip("idle", *IDLE.as_bytes()),
            SmState::clip("wave", *WAVE.as_bytes()),
        ],
        transitions: vec![SmTransition::new(0, 1, 0.0)],
        entry: 0,
        ..Default::default()
    }
}

/// The cape: a 5×5 sheet pinned along its `j == 0` row, hung across the rig's
/// chain and fitted with one capsule over joints **1→2** — the segment the `wave`
/// clip actually moves.
///
/// The joint pair is load-bearing. The first cut fitted the capsule over 0→1 and
/// the arm that says "the machine's pose moves the cloth" failed: bending joint 1
/// rotates joint 1's *frame*, which moves joint **2** and leaves the 0→1 segment
/// exactly where it was. A capsule over a bone the animation cannot move is a
/// capsule that proves nothing.
///
/// Placement: the chain runs up +Y from the origin with 1 m bones, so segment 1→2
/// spans `y ∈ [1, 2]` on the axis. The sheet hangs from `y = 2.3` over
/// `x ∈ [-0.2, 0.2]`, `z ∈ [-0.2, 0.2]`, which puts it inside the 0.25 m capsule
/// as it falls — and clear of it the moment the clip swings the bone aside.
fn garment() -> ClothAsset {
    let mut pos = Vec::new();
    for j in 0..5u32 {
        for i in 0..5u32 {
            pos.push([i as f32 * 0.1 - 0.2, 2.3, j as f32 * 0.1 - 0.2]);
        }
    }
    let mut idx = Vec::new();
    for j in 0..4u32 {
        for i in 0..4u32 {
            let a = j * 5 + i;
            idx.extend_from_slice(&[a, a + 1, a + 5, a + 1, a + 6, a + 5]);
        }
    }
    ClothAsset::from_garment(
        *MESH.as_bytes(),
        &pos,
        &idx,
        &[0, 1, 2, 3, 4],
        ClothMaterial::default(),
    )
    .expect("the fixture garment prepares")
    .with_capsules(vec![ClothCapsule {
        joint_a: 1,
        joint_b: 2,
        radius_m: 0.25,
    }])
}

fn skeletons() -> BTreeMap<Uuid, SkeletonAsset> {
    BTreeMap::from([(SKEL, rig())])
}

fn machines() -> BTreeMap<Uuid, StateMachine> {
    BTreeMap::from([(SM, machine())])
}

fn pose_clips() -> BTreeMap<Uuid, AnimClip> {
    BTreeMap::from([(IDLE, hold(0.0)), (WAVE, hold(70.0))])
}

fn cloths() -> BTreeMap<Uuid, ClothAsset> {
    BTreeMap::from([(CLOTH, garment())])
}

/// The character's components. `wearing == false` drops the `ClothSim`, which is
/// how the "no garment ⇒ pre-P24.4 trace" arm builds its control.
fn character(wearing: bool) -> (AnimStateMachine, SkeletalMesh, Transform, ClothSim) {
    (
        AnimStateMachine {
            sm: Some(SM),
            ..Default::default()
        },
        SkeletalMesh {
            mesh: Some(MESH),
            skeleton: Some(SKEL),
        },
        Transform {
            translation: Vec3d::ZERO,
            ..Default::default()
        },
        ClothSim {
            asset: wearing.then_some(CLOTH),
            enabled: true,
            quality: 0,
        },
    )
}

// ── the two hosts ───────────────────────────────────────────────────────────

/// The shipped player's per-step cloth trace.
fn player_trace(wearing: bool) -> Vec<Vec<u8>> {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character(wearing));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    sim.set_cloths(cloths());
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            inf_ecs::cloth::cloth_state_bytes(sim.world())
        })
        .collect()
}

/// The editor Simulate's per-step cloth trace, through the same Ring-0 door.
fn editor_trace(wearing: bool) -> Vec<Vec<u8>> {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(character(wearing));
    doc.world_mut().mark_dirty();
    let mut session = SimSession::enter(&mut doc, Vec::new(), DVec2::ZERO, HZ);
    session.set_state_machines(machines());
    session.set_skeletons(skeletons());
    session.set_pose_clips(pose_clips());
    session.set_cloths(cloths());
    let out = (0..STEPS)
        .map(|_| {
            session.step_once(&mut doc, SimInput::default());
            inf_ecs::cloth::cloth_state_bytes(doc.world())
        })
        .collect();
    session.exit(&mut doc);
    out
}

/// **ANTI-VACUITY.** Two empty traces are equal, and so are two traces of a
/// garment that never moved: the fixture has to have simulated something, and the
/// garment has to have MOVED.
fn assert_not_vacuous(trace: &[Vec<u8>]) {
    assert_eq!(trace.len() as u32, STEPS);
    assert!(
        !trace[0].is_empty(),
        "step 0 simulated no garment at all — the `ClothSim` component was not \
         read, which is precisely the pre-P24.4 state this gate exists to leave \
         behind"
    );
    assert_ne!(
        trace[0], trace[1],
        "the garment did not move between the first two steps"
    );
    assert_ne!(
        trace.first(),
        trace.last(),
        "the garment is frozen over the whole run"
    );
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **The headline gate.** The editor's Simulate and the shipped player fold the
/// same garment, byte for byte, over every step.
#[test]
fn both_hosts_fold_the_same_garment_byte_for_byte() {
    let player = player_trace(true);
    let editor = editor_trace(true);
    assert_not_vacuous(&player);
    assert_not_vacuous(&editor);
    assert_eq!(
        player, editor,
        "the shipped player and the editor Simulate folded different garments — \
         PIE would stop matching shipping, and a coat would hang differently in \
         the preview than in the shipped build"
    );
}

/// The garment rides the sim's **trace**, so every determinism gate the engine
/// already has covers it — and a level with no garment is byte-identical to its
/// pre-P24.4 self.
#[test]
fn the_garment_rides_the_sim_trace() {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character(true));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    sim.set_cloths(cloths());

    sim.step_once(RuntimeInput::default());
    let early = sim.state_bytes();
    for _ in 0..6 {
        sim.step_once(RuntimeInput::default());
    }
    let late = sim.state_bytes();
    assert_ne!(
        early, late,
        "the sim trace must move as the garment folds — if it does not, cloth is \
         not sim state and no replay/PIE gate can see it"
    );

    // **THE MUTATION ARM for the state-bytes append.** The same character with no
    // `ClothSim` bound produces a trace with no cloth section at all, and that
    // trace is what every pre-P24.4 hash in the tree was folded over. Sever the
    // append in `RuntimeSim::state_bytes` and the inequality above becomes an
    // equality; sever the *seeding* and this equality becomes an inequality.
    let mut bare_world = EcsWorld::new();
    let be = bare_world.spawn_with_guid(HERO, "Hero", None);
    bare_world
        .world_mut()
        .entity_mut(be)
        .insert(character(false));
    bare_world.mark_dirty();
    let mut bare = RuntimeSim::new(bare_world, Vec::new(), DVec2::ZERO, HZ);
    bare.set_state_machines(machines());
    bare.set_skeletons(skeletons());
    bare.set_pose_clips(pose_clips());
    bare.set_cloths(cloths());
    assert!(inf_ecs::cloth::cloth_state_bytes(bare.world()).is_empty());
    let before = bare.state_bytes();
    bare.step_once(RuntimeInput::default());
    let after = bare.state_bytes();
    assert!(
        inf_ecs::cloth::cloth_state_bytes(bare.world()).is_empty(),
        "a character with no bound garment grew a cloth store"
    );
    // The pose still moves (the machine transitions), so this is not the trace
    // standing still — only the CLOTH section is absent.
    assert_ne!(before, after);
    // …and the trace's TAIL is the pose section, i.e. nothing at all was appended
    // after it. A cloth section that emitted a header (a count, a zero) for a
    // level with no cloth in it would fail here, and would have silently changed
    // every committed trace hash in the tree.
    let pose = inf_ecs::pose::pose_state_bytes(bare.world());
    assert!(!pose.is_empty(), "the control character is not even posed");
    assert!(
        after.ends_with(&pose),
        "a level with no garment does not end its trace at the pose section — \
         something is being appended after it"
    );
}

/// **A worn garment changes the trace; an unworn one does not.**
///
/// The arm that would fail if `step_cloth_simulation` were a no-op: the same
/// world, the same machine, the same clips, differing only in whether the
/// `ClothSim` names an asset.
#[test]
fn wearing_a_garment_is_what_changes_the_trace() {
    let worn = player_trace(true);
    let bare = player_trace(false);
    assert_not_vacuous(&worn);
    assert!(
        bare.iter().all(Vec::is_empty),
        "a character with no bound garment produced cloth trace bytes"
    );
    assert_ne!(worn, bare);
    // …and the editor agrees about BOTH, so the headline arm is not passing
    // because both hosts happen to ignore the component.
    assert_eq!(bare, editor_trace(false));
}

/// **The capsule is read off the POSE**, so a machine that moves the body moves
/// the cloth — the property that makes "cloth collides against the character"
/// mean something rather than "cloth collides against the bind pose".
///
/// Two runs of the identical garment differing only in what the state machine
/// does: one whose clip bends the bone under the cape, one whose clip does not.
#[test]
fn the_machines_pose_moves_the_cloth() {
    let trace = |bend_deg: f32| -> Vec<Vec<u8>> {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(HERO, "Hero", None);
        world.world_mut().entity_mut(e).insert(character(true));
        world.mark_dirty();
        let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
        sim.set_state_machines(machines());
        sim.set_skeletons(skeletons());
        sim.set_pose_clips(BTreeMap::from([(IDLE, hold(0.0)), (WAVE, hold(bend_deg))]));
        sim.set_cloths(cloths());
        (0..STEPS)
            .map(|_| {
                sim.step_once(RuntimeInput::default());
                inf_ecs::cloth::cloth_state_bytes(sim.world())
            })
            .collect()
    };
    let upright = trace(0.0);
    let bent = trace(70.0);
    assert_not_vacuous(&upright);
    assert_ne!(
        upright, bent,
        "bending the bone under the cape did not change how the cape folds — the \
         collision capsules are not following the evaluated pose, so the garment \
         is colliding against a body that never moves"
    );
}

/// **Determinism**, twice: two runs of the same world are bit-identical, and the
/// bytes move (a constant compares equal too).
#[test]
fn the_fold_is_bit_identical_between_two_runs() {
    let a = player_trace(true);
    let b = player_trace(true);
    assert_not_vacuous(&a);
    assert_eq!(a, b);
    assert_eq!(editor_trace(true), editor_trace(true));
}

/// A garment whose `.inf_cloth` does not resolve simulates nothing and **leaves
/// the component alone** — the rule-2 refusal, at the host boundary.
///
/// It is asserted against the *no-garment baseline* rather than merely "it did
/// not crash": an unresolvable reference must leave a trace byte-identical to a
/// character wearing nothing.
#[test]
fn an_unresolvable_garment_leaves_a_usable_trace() {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character(true));
    world
        .world_mut()
        .get_mut::<ClothSim>(e)
        .unwrap()
        .asset
        .replace(Uuid::from_u128(0xDEAD));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    sim.set_cloths(cloths());
    let trace: Vec<Vec<u8>> = (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            inf_ecs::cloth::cloth_state_bytes(sim.world())
        })
        .collect();
    assert!(
        trace.iter().all(Vec::is_empty),
        "a dangling garment reference simulated something"
    );
    // The component survived: skipping is not unbinding.
    let e = sim.world().entity_of(HERO).unwrap();
    assert_eq!(
        sim.world().world().get::<ClothSim>(e).unwrap().asset,
        Some(Uuid::from_u128(0xDEAD))
    );
    // ANTI-VACUITY: the SAME sim with a resolvable reference does simulate, so
    // the emptiness above is the refusal and not the registry being empty.
    assert!(!player_trace(true)[0].is_empty());
}

/// **Simulate leaves no fold behind.** Two consecutive sessions over the same
/// document produce identical traces, which is what stops run 2 starting from run
/// 1's settled coat — the divergence `clear_cloth` at both ends exists to prevent.
#[test]
fn two_simulate_sessions_over_one_document_agree() {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(character(true));
    doc.world_mut().mark_dirty();

    let run = |doc: &mut SceneDoc| -> Vec<Vec<u8>> {
        let mut session = SimSession::enter(doc, Vec::new(), DVec2::ZERO, HZ);
        session.set_state_machines(machines());
        session.set_skeletons(skeletons());
        session.set_pose_clips(pose_clips());
        session.set_cloths(cloths());
        let out = (0..STEPS)
            .map(|_| {
                session.step_once(doc, SimInput::default());
                inf_ecs::cloth::cloth_state_bytes(doc.world())
            })
            .collect();
        session.exit(doc);
        out
    };
    let first = run(&mut doc);
    assert_not_vacuous(&first);
    // …and the document is left with no fold on it at all.
    assert_eq!(inf_ecs::cloth::cloth_count(doc.world()), 0);
    assert!(inf_ecs::cloth::cloth_state_bytes(doc.world()).is_empty());
    let second = run(&mut doc);
    assert_eq!(
        first, second,
        "run 2 folded a different garment from run 1 — the session started on the \
         previous run's settled coat, which the shipped player never does"
    );
}

// ── the garment RENDERS (P24.4 commit 2) ────────────────────────────────────

/// **The headline render gate, asserted on the WORLD**: the sim's fold reaches
/// the render scene as real vertex bytes, and those bytes MOVE as the garment
/// falls.
///
/// Buffer bytes, not pixels — the house rule. The editor half of this pair is
/// covered by `inf-editor-core`'s `tests/projector_mirror.rs`
/// (`project_cloth_is_identical_in_both_projectors`), which is what lets a Linux
/// CI leg with no GPU see the editor's copy at all.
#[test]
fn the_simulated_garment_reaches_the_render_scene() {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character(true));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    sim.set_cloths(cloths());

    let vmeshes = inf_player::vmesh::VmeshRegistry::new();
    let project = |sim: &RuntimeSim| -> inf_render::RenderScene {
        let mut scene = inf_render::RenderScene::default();
        inf_player::render::project_scene(&mut scene, sim, 1.0, &vmeshes);
        scene
    };

    // Before any step the sim has folded nothing, so there is no garment to draw.
    let before = project(&sim);
    assert!(
        before.skinned.is_empty(),
        "a garment was drawn before the sim folded one"
    );

    sim.step_once(RuntimeInput::default());
    let first = project(&sim);
    let garment = first.skinned.iter().find(|i| i.palette.len() == 1).expect(
        "the simulated garment did not reach the render scene — the projector \
             is not reading the cloth store",
    );
    assert_eq!(
        garment.palette[0],
        Mat4::IDENTITY,
        "a garment must carry the one-entry IDENTITY palette; anything else would \
         deform model-space particle positions a second time"
    );
    assert_eq!(garment.color, inf_render::CLOTH_TINT);
    assert_eq!(
        garment.id,
        inf_render::ID_NONE,
        "a garment is not pickable in v1"
    );
    let data = &first.skinned_meshes[garment.mesh];
    assert_eq!(
        data.vertices.len(),
        garment_particle_count(),
        "the drawn garment has a different vertex count from the simulated one"
    );
    assert!(!data.indices.is_empty(), "the garment drew no triangles");
    // Every vertex is pinned to joint 0 at weight 1 — what makes the identity
    // palette a no-op rather than a coincidence.
    for v in &data.vertices {
        assert_eq!(v.joints, [0; 4]);
        assert_eq!(v.weights, [1.0, 0.0, 0.0, 0.0]);
        assert!(v.pos.iter().all(|c| c.is_finite()));
        let n = glam::Vec3::from_array(v.normal);
        assert!(
            (n.length() - 1.0).abs() < 1e-4,
            "a recomputed normal is not unit: {n:?}"
        );
    }

    // …and the bytes MOVE, which is what makes this a statement about the
    // simulation rather than about a mesh being copied.
    for _ in 0..8 {
        sim.step_once(RuntimeInput::default());
    }
    let later = project(&sim);
    let moved = later
        .skinned
        .iter()
        .find(|i| i.palette.len() == 1)
        .expect("the garment stopped being drawn");
    assert_ne!(
        first.skinned_meshes[garment.mesh].vertices, later.skinned_meshes[moved.mesh].vertices,
        "the drawn garment's vertices are frozen while the sim's are not — the \
         projector is drawing a cached fold"
    );

    // The wearer is drawn TOO: a garment is worn beside its body, not instead of
    // it. (The character has no resolvable skinned mesh here, so it falls through
    // to the slate placeholder — which is exactly the branch that must survive.)
    assert!(
        !later.instances.is_empty(),
        "drawing a garment swallowed the wearer's own draw"
    );
}

/// A wearer with no garment adds nothing to the scene at all — the projector's
/// half of "a level with no cloth is byte-identical to its pre-P24.4 self".
#[test]
fn a_wearer_with_no_garment_draws_no_cloth() {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character(false));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    sim.set_cloths(cloths());
    for _ in 0..4 {
        sim.step_once(RuntimeInput::default());
    }
    let mut scene = inf_render::RenderScene::default();
    let vmeshes = inf_player::vmesh::VmeshRegistry::new();
    inf_player::render::project_scene(&mut scene, &sim, 1.0, &vmeshes);
    assert!(
        scene.skinned.is_empty() && scene.skinned_meshes.is_empty(),
        "an unworn garment still produced a skinned draw"
    );
}

/// How many particles the fixture garment has — computed from the asset rather
/// than written down, so the assertion above cannot drift from the fixture.
fn garment_particle_count() -> usize {
    garment().particle_count()
}
