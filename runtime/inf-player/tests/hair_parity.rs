//! **P24.4: strand hair simulates identically in both hosts, and it is drawn.**
//!
//! The `cloth_parity` file's four claims, on hair: the `HairGuides` component is
//! read at all (scene v21 landed it at P24.3 with "nothing renders or simulates
//! hair" written on the type), the strands are sim state folded into
//! `state_bytes`, the two hosts agree byte for byte, and a level with no hair is
//! byte-identical to its pre-P24.4 self.

use std::collections::BTreeMap;

use glam::{DVec2, Mat4, Quat, Vec3};
use uuid::Uuid;

use inf_anim::{
    AnimClip, ClothCapsule, HairAsset, HairMaterial, HairRoot, Interpolation, Joint, JointTrack,
    JointTransform, QuatTrack, Skeleton, SkeletonAsset, SmState, SmTransition, StateMachine,
};
use inf_ecs::components::{AnimStateMachine, HairGuides, SkeletalMesh, Transform};
use inf_ecs::math::Vec3d;
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

const HZ: f64 = 60.0;
const STEPS: u32 = 12;

const HERO: Uuid = Uuid::from_u128(0x2404_1000_0000_0000_0000_0000_0000_0001);
const SM: Uuid = Uuid::from_u128(0x2404_1000_0000_0000_0000_0000_0000_0002);
const SKEL: Uuid = Uuid::from_u128(0x2404_1000_0000_0000_0000_0000_0000_0003);
const MESH: Uuid = Uuid::from_u128(0x2404_1000_0000_0000_0000_0000_0000_0004);
const STYLE: Uuid = Uuid::from_u128(0x2404_1000_0000_0000_0000_0000_0000_0005);
const IDLE: Uuid = Uuid::from_u128(0x2404_1000_0000_0000_0000_0000_0000_0010);
const WAVE: Uuid = Uuid::from_u128(0x2404_1000_0000_0000_0000_0000_0000_0011);

/// A 3-joint chain along +Y with 1 m bones.
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

/// A clip holding joint 1 at `deg` about +Z — one `Step` key, so a state's pose
/// is a constant and two runs cannot drift by a ULP.
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

/// idle then wave, unconditional, so the head the strands are rooted on moves on
/// the first step.
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

/// Six strands rooted on **joint 2** — the tip of the chain, which bending joint
/// 1 swings. Rooting them on joint 0 would put them on a bone the animation
/// cannot move, and every "the pose moves the hair" arm would be vacuous.
fn hairstyle() -> HairAsset {
    let roots: Vec<HairRoot> = (0..6)
        .map(|i| HairRoot {
            joint: 2,
            offset: [i as f32 * 0.02 - 0.05, 0.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            clump: i as u16 / 3,
        })
        .collect();
    // A real groom, not the identity: two clumps of three with a loose wave, so
    // the parity arms compare hosts on a hairstyle that is actually SHAPED. A
    // straight, unclumped fixture would agree between the two hosts for reasons
    // that have nothing to do with the groom being applied the same way twice.
    HairAsset::grow(
        *SKEL.as_bytes(),
        &roots,
        0.25,
        5,
        HairMaterial::default(),
        inf_anim::HairGroom {
            clump_strength: 0.5,
            curl_radius_m: 0.01,
            curl_turns: 1.5,
        },
    )
    .expect("the fixture hairstyle grows")
    .with_capsules(vec![ClothCapsule {
        joint_a: 1,
        joint_b: 2,
        radius_m: 0.15,
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
fn hairs() -> BTreeMap<Uuid, HairAsset> {
    BTreeMap::from([(STYLE, hairstyle())])
}

fn character(wearing: bool) -> (AnimStateMachine, SkeletalMesh, Transform, HairGuides) {
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
        HairGuides {
            asset: wearing.then_some(STYLE),
            enabled: true,
            quality: 0,
        },
    )
}

fn player_trace(wearing: bool, clips: BTreeMap<Uuid, AnimClip>) -> Vec<Vec<u8>> {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character(wearing));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(clips);
    sim.set_hairs(hairs());
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            inf_ecs::hair::hair_state_bytes(sim.world())
        })
        .collect()
}

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
    session.set_hairs(hairs());
    let out = (0..STEPS)
        .map(|_| {
            session.step_once(&mut doc, SimInput::default());
            inf_ecs::hair::hair_state_bytes(doc.world())
        })
        .collect();
    session.exit(&mut doc);
    out
}

/// **The headline hair gate**: both hosts simulate the same strands, byte for
/// byte, and the component is read at all.
#[test]
fn both_hosts_simulate_the_same_hair_byte_for_byte() {
    let player = player_trace(true, pose_clips());
    let editor = editor_trace(true);
    assert_eq!(player.len() as u32, STEPS);
    assert!(
        !player[0].is_empty(),
        "step 0 simulated no hair at all — the HairGuides component was not read, \
         which is the pre-P24.4 state this gate exists to leave behind"
    );
    assert_ne!(player[0], player[1], "the strands never moved");
    assert_eq!(
        player, editor,
        "the shipped player and the editor Simulate simulated different hair"
    );
    // ANTI-VACUITY: an unworn hairstyle produces nothing, so the equality above is
    // not two empty traces agreeing.
    let bare = player_trace(false, pose_clips());
    assert!(bare.iter().all(Vec::is_empty));
    assert_ne!(player, bare);
    assert_eq!(bare, editor_trace(false));
}

/// **A hairstyle starts ON the scalp it was grown from** (P24.4 audit F3).
///
/// The absolute half of the arm below, which is a *relative* one: "the pose moves
/// the hair" is satisfied by any transform that varies with the pose, including
/// one that puts the whole head of hair a bone-length away from the head. So this
/// holds the rig at its **bind pose** (a machine with one state and no
/// transitions, playing a clip whose only key is the identity rotation the rig is
/// already in) and asserts that after a step every strand's particle 0 is where
/// the generator grew it.
///
/// It was not: `roots_for` was fed `global_transforms` while
/// `HairStrand::root_offset` is a model-space bind position, so every root was
/// displaced by its joint's bind transform on the first step — two metres, on
/// this fixture, whose strands ride joint 2 of a chain of 1 m bones. Both hosts
/// did it identically, which is why `both_hosts_simulate_the_same_hair_byte_for_byte`
/// could not see it, and the anti-vacuity arms could not either: an unworn
/// hairstyle is still empty and a swung joint still moves.
#[test]
fn the_authored_roots_start_on_the_scalp() {
    // One state, no transitions, holding the bind rotation: the pose IS the bind
    // pose, so a correctly-carried root cannot have moved at all.
    let still = StateMachine {
        states: vec![SmState::clip("idle", *IDLE.as_bytes())],
        transitions: vec![],
        entry: 0,
        ..Default::default()
    };
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character(true));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(BTreeMap::from([(SM, still)]));
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(BTreeMap::from([(IDLE, hold(0.0))]));
    sim.set_hairs(hairs());
    sim.step_once(RuntimeInput::default());

    let asset = hairstyle();
    let live =
        inf_ecs::hair::live_hair(sim.world(), HERO).expect("the wearer is simulating a hairstyle");
    assert_eq!(live.state.strand_count(), asset.strand_count());
    for (i, strand) in asset.strands.iter().enumerate() {
        let grown = Vec3::from_array(strand.points[0]);
        let pinned = Vec3::from_array(live.state.x[live.state.starts[i] as usize]);
        assert!(
            (pinned - grown).length() < 1e-4,
            "strand {i} was grown at {grown:?} and the first step pinned it at \
             {pinned:?} — {:.3} m off the scalp, at the BIND pose",
            (pinned - grown).length()
        );
    }
}

/// **The roots ride the POSE**: a machine that swings the head swings the hair.
/// The arm that fails if `roots_for` were fed the bind pose.
#[test]
fn the_machines_pose_moves_the_hair() {
    let upright = player_trace(true, BTreeMap::from([(IDLE, hold(0.0)), (WAVE, hold(0.0))]));
    let bent = player_trace(true, pose_clips());
    assert!(!upright[0].is_empty());
    assert_ne!(
        upright, bent,
        "swinging the joint the strands are rooted on did not move the hair — the \
         roots are not following the evaluated pose"
    );
}

/// Determinism, and the trace really moves.
#[test]
fn the_strands_are_bit_identical_between_two_runs() {
    let a = player_trace(true, pose_clips());
    assert_eq!(a, player_trace(true, pose_clips()));
    assert_eq!(editor_trace(true), editor_trace(true));
    assert_ne!(a[0], a[STEPS as usize - 1]);
}

/// The hair rides `state_bytes` **after** the cloth section, and a level with
/// neither leaves the trace ending at the pose section.
#[test]
fn the_hair_rides_the_sim_trace() {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character(true));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    sim.set_hairs(hairs());
    sim.step_once(RuntimeInput::default());
    let early = sim.state_bytes();
    for _ in 0..6 {
        sim.step_once(RuntimeInput::default());
    }
    assert_ne!(early, sim.state_bytes(), "the hair is not in the trace");

    let mut bw = EcsWorld::new();
    let be = bw.spawn_with_guid(HERO, "Hero", None);
    bw.world_mut().entity_mut(be).insert(character(false));
    bw.mark_dirty();
    let mut bare = RuntimeSim::new(bw, Vec::new(), DVec2::ZERO, HZ);
    bare.set_state_machines(machines());
    bare.set_skeletons(skeletons());
    bare.set_pose_clips(pose_clips());
    bare.set_hairs(hairs());
    bare.step_once(RuntimeInput::default());
    let pose = inf_ecs::pose::pose_state_bytes(bare.world());
    assert!(!pose.is_empty(), "the control character is not even posed");
    assert!(
        bare.state_bytes().ends_with(&pose),
        "a level with no hair and no cloth does not end its trace at the pose \
         section — something is being appended after it"
    );
}

/// **The ribbons reach the render scene**, and they move — buffer bytes, not
/// pixels. The editor half is covered by `projector_mirror`'s
/// `project_hair_is_identical_in_both_projectors`.
#[test]
fn the_simulated_hair_reaches_the_render_scene() {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character(true));
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    sim.set_hairs(hairs());

    let vmeshes = inf_player::vmesh::VmeshRegistry::new();
    let project = |sim: &RuntimeSim| {
        let mut scene = inf_render::RenderScene::default();
        inf_player::render::project_scene(&mut scene, sim, 1.0, &vmeshes);
        scene
    };
    assert!(
        project(&sim).skinned.is_empty(),
        "hair was drawn before the sim simulated any"
    );

    sim.step_once(RuntimeInput::default());
    let first = project(&sim);
    let ribbon = first
        .skinned
        .iter()
        .find(|i| i.color == inf_render::HAIR_TINT)
        .expect("the simulated hair did not reach the render scene");
    assert_eq!(ribbon.palette, vec![Mat4::IDENTITY]);
    assert_eq!(ribbon.id, inf_render::ID_NONE);
    let data = &first.skinned_meshes[ribbon.mesh];
    assert_eq!(
        data.vertices.len(),
        hairstyle().particle_count() * 2,
        "two ribbon vertices per strand particle"
    );
    assert!(data
        .vertices
        .iter()
        .all(|v| v.pos.iter().all(|c| c.is_finite())));

    for _ in 0..8 {
        sim.step_once(RuntimeInput::default());
    }
    let later = project(&sim);
    let moved = later
        .skinned
        .iter()
        .find(|i| i.color == inf_render::HAIR_TINT)
        .expect("the hair stopped being drawn");
    assert_ne!(
        first.skinned_meshes[ribbon.mesh].vertices, later.skinned_meshes[moved.mesh].vertices,
        "the drawn ribbons are frozen while the sim's strands are not"
    );
}
