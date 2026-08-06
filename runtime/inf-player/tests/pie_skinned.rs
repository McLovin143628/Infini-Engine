//! **P24.1: a skinned character in PIE.**
//!
//! `run_pie` took no `SkinnedRegistry` and `PlayerApp::new` seeded the inert
//! `Content::None` one, so every character in a **windowed** PIE session — the
//! embedded viewport, the new-window preview — drew as a placeholder cube, while
//! the headless PIE the gates drive and the shipped build both drew real posed
//! geometry. Skeletons, clips and machines had ridden the `ScenePayload` since
//! v3; the `.inf_mesh` bytes the skinned pass needs were the one part that never
//! crossed. `ScenePayload` **v7** carries them.
//!
//! P21.4 fixed the identical class for voxel volumes by threading `voxel_assets`
//! into `run_pie`; this is the same fix, and it carries the same lesson forward:
//! **two empty maps agree perfectly**, so every comparison below asserts the
//! payload's `meshes` vector is non-empty, with an exact count taken from the
//! fixture, before it compares anything.
//!
//! The rendering claim is asserted **structurally** — the registry resolves the
//! mesh to real bind-space geometry and a palette — not in pixels: `run_pie`
//! itself needs a GPU and a display, so like every GPU path it is compile-checked
//! and human-verified. What is machine-checked here is the whole chain that feeds
//! it: document → payload → registry → `SkinnedDraw`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use glam::{Mat4, Quat, Vec3};
use uuid::Uuid;

use inf_anim::{
    AnimClip, AnimClipAsset, Interpolation, Joint, JointTrack, JointTransform, QuatTrack, Skeleton,
    SkeletonAsset, SmState, SmTransition, StateMachine, StateMachineAsset,
};
use inf_ecs::components::{AnimStateMachine, SkeletalMesh};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::pie::{build_scene_payload, PieSession};
use inf_editor_core::scene::SceneDoc;
use inf_mesh::{MeshAsset, MeshVertex, SubMesh, VertexSkin};
use inf_player::skinned::SkinnedRegistry;
use inf_runtime::pie::{PlayerToEditor, ScenePayload};

const HERO: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0001);
const MESH: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0002);
const SKEL: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0003);
const SM: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0004);
const IDLE: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0005);
const WAVE: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0006);
const SALUTE: Uuid = Uuid::from_u128(0x2401_1000_0000_0000_0000_0000_0000_0007);

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

// ── the fixture ─────────────────────────────────────────────────────────────

/// A 2-joint chain: root, then a tip 1 m up.
fn rig() -> SkeletonAsset {
    SkeletonAsset::new(
        Skeleton::new(vec![
            Joint {
                name: "root".into(),
                parent: None,
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            },
            Joint {
                name: "tip".into(),
                parent: Some(0),
                inverse_bind: Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)).to_cols_array(),
                local_bind: JointTransform::from_trs(Vec3::Y, Quat::IDENTITY, Vec3::ONE),
            },
        ])
        .unwrap(),
    )
}

/// A skinned triangle — three vertices, real influences, so
/// `skinned_mesh_data` accepts it and the skinned pass has something to draw.
fn body() -> MeshAsset {
    let v = |x: f32, y: f32| MeshVertex {
        position: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        ..Default::default()
    };
    let w = |j: u16| VertexSkin {
        joints: [j, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    MeshAsset::new(
        vec![SubMesh {
            name: "tri".into(),
            vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 2.0)],
            indices: vec![0, 1, 2],
            material_slot: None,
            skin: vec![w(0), w(0), w(1)],
        }],
        vec![],
    )
}

/// A clip holding the tip joint at `deg` about +Z (one stepped key, so the pose
/// a state produces is an exact constant).
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

/// idle → wave → salute, both transitions unconditional, so the machine walks
/// its states on consecutive fixed steps with no actor and no Blueprint variable
/// involved.
///
/// **Three states, not two.** Each state holds a *constant* pose, so a two-state
/// chain settles after one transition and every hash from step 2 on is identical
/// — which would make the "the trace moved" arm below fail for the right reason
/// and the equality arm pass for the wrong one.
fn machine() -> StateMachine {
    StateMachine {
        states: vec![
            SmState::clip("idle", *IDLE.as_bytes()),
            SmState::clip("wave", *WAVE.as_bytes()),
            SmState::clip("salute", *SALUTE.as_bytes()),
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

/// The asset bytes the editor would resolve out of a project's content root.
fn anim_bytes() -> BTreeMap<Uuid, Vec<u8>> {
    BTreeMap::from([
        (SKEL, inf_asset::encode(&rig()).unwrap()),
        (
            SM,
            inf_asset::encode(&StateMachineAsset::new(machine(), Some(*SKEL.as_bytes()))).unwrap(),
        ),
        (
            IDLE,
            inf_asset::encode(&AnimClipAsset::new(hold(0.0), Some(*SKEL.as_bytes()))).unwrap(),
        ),
        (
            WAVE,
            inf_asset::encode(&AnimClipAsset::new(hold(60.0), Some(*SKEL.as_bytes()))).unwrap(),
        ),
        (
            SALUTE,
            inf_asset::encode(&AnimClipAsset::new(hold(-25.0), Some(*SKEL.as_bytes()))).unwrap(),
        ),
    ])
}

/// A one-character document, streamed exactly as the editor streams it.
fn character_payload(windowed: bool) -> ScenePayload {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, SpawnKind::Empty, "Hero", None);
    doc.world_mut().world_mut().entity_mut(e).insert((
        SkeletalMesh {
            mesh: Some(MESH),
            skeleton: Some(SKEL),
        },
        AnimStateMachine {
            sm: Some(SM),
            ..Default::default()
        },
    ));
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();

    let anim = anim_bytes();
    let mesh_bytes = inf_asset::encode(&body()).unwrap();
    build_scene_payload(
        &doc,
        |_| None,
        |_| None,
        |g| anim.get(&g).cloned(),
        |_| None,
        |_| None,
        |_| None,
        |g| (g == MESH).then(|| mesh_bytes.clone()),
        0, // tick-hz 0: no per-frame sleep (step-driven determinism)
        windowed,
    )
    .expect("build scene payload")
}

// ── the gates ───────────────────────────────────────────────────────────────

/// **The payload carries a character.** The exact-count guard the P21.4 lesson
/// asks for, asserted at the seam that produces the payload, so nothing below can
/// be two empty maps agreeing.
#[test]
fn the_payload_carries_the_characters_mesh_skeleton_machine_and_clips() {
    let payload = character_payload(false);
    assert_eq!(
        payload.meshes.len(),
        1,
        "the payload carries no `.inf_mesh` — every character in a windowed PIE \
         session would draw as a placeholder cube (this is the P24.1 defect)"
    );
    assert_eq!(payload.meshes[0].0, MESH);
    assert_eq!(payload.skeletons.len(), 1);
    assert_eq!(payload.machines.len(), 1);
    // Both clips, and NEITHER is named by a component: they live inside the
    // machine's own payload and reach the wire only through the transitive walk.
    let clip_ids: Vec<Uuid> = payload.clips.iter().map(|(g, _)| *g).collect();
    assert!(clip_ids.contains(&IDLE), "{clip_ids:?}");
    assert!(clip_ids.contains(&WAVE), "{clip_ids:?}");
    assert!(clip_ids.contains(&SALUTE), "{clip_ids:?}");
    assert_eq!(clip_ids.len(), 3, "{clip_ids:?}");
    assert_eq!(
        payload.schema_version,
        inf_runtime::pie::SCENE_PAYLOAD_VERSION
    );
}

/// **A PIE session draws non-cube geometry.** The registry `run_pie` is now
/// handed resolves the payload's bytes to real bind-space geometry plus a
/// palette — the structural form of "the character is not a placeholder".
///
/// The negative control is the store PIE used to get: `SkinnedRegistry::new()`
/// resolves the same binding to `None`, which is precisely how the projector fell
/// through to its slate cube.
#[test]
fn the_pie_registry_resolves_real_skinned_geometry() {
    let payload = character_payload(true);
    assert_eq!(payload.meshes.len(), 1, "nothing to resolve");

    let store = SkinnedRegistry::from_payload(&payload.meshes, &payload.skeletons, &payload.clips);
    let sm = SkeletalMesh {
        mesh: Some(MESH),
        skeleton: Some(SKEL),
    };
    let draw = store
        .resolve_skinned(&sm, None, None)
        .expect("the PIE payload's character resolves to real skinned geometry");
    assert_eq!(draw.mesh.vertices.len(), 3, "the bind-space stream is real");
    assert_eq!(draw.mesh.indices, vec![0, 1, 2]);
    assert_eq!(draw.palette.len(), 2, "one matrix per joint");
    assert_eq!(draw.key, (MESH, SKEL));
    assert_eq!(store.loaded_skinned(), 1);
    assert!(store.has_content());

    // The negative control — the store a windowed PIE session used to be handed.
    assert!(
        SkinnedRegistry::new()
            .resolve_skinned(&sm, None, None)
            .is_none(),
        "the inert store must still miss, or this test proves nothing about the \
         one that was threaded in"
    );
}

/// **The machine's pose reaches the drawn palette, through the payload.** Build
/// the sim the PIE subprocess builds, step it, and resolve the same character
/// through the same store: the palette must follow the machine out of its entry
/// state.
#[test]
fn the_payload_built_sim_poses_the_character_it_ships() {
    let payload = character_payload(false);
    let mut sim = inf_player::sim_from_payload(&payload)
        .expect("the payload builds a sim")
        .sim;
    let store = SkinnedRegistry::from_payload(&payload.meshes, &payload.skeletons, &payload.clips);
    let sm = SkeletalMesh {
        mesh: Some(MESH),
        skeleton: Some(SKEL),
    };
    let rest = store.resolve_skinned(&sm, None, None).unwrap().palette;

    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    let posed = inf_ecs::pose::evaluated_pose(sim.world(), HERO).expect(
        "the payload-built sim published a pose — if it did not, the \
                 skeleton or the machine's clips never crossed the wire",
    );
    let drawn = store
        .resolve_skinned(&sm, None, Some(posed))
        .unwrap()
        .palette;
    assert_ne!(
        rest[1].to_cols_array(),
        drawn[1].to_cols_array(),
        "the character shipped by the payload still draws its rest pose"
    );
}

/// **PIE == shipping, with a character in it.** The real `--pie` subprocess,
/// driven step by step, must fold the same per-step `state_hash` the in-process
/// shipping build of the same payload does — and since P24.1 that hash carries the
/// evaluated pose, so this arm is now a comparison of *what is drawn* as well as
/// of what is simulated.
#[test]
fn pie_subprocess_trace_matches_shipping_with_a_skinned_character() {
    let payload = character_payload(false);
    // Anti-vacuity FIRST: a payload with no character would make both sides agree
    // about a world containing nothing to pose.
    assert_eq!(payload.meshes.len(), 1);
    assert_eq!(payload.skeletons.len(), 1);
    assert_eq!(payload.machines.len(), 1);

    const N: u32 = 8;
    let mut session = PieSession::spawn_scene(&player_bin(), &payload).expect("scene session");
    session.step(N).expect("step N");
    let mut got = Vec::with_capacity(N as usize);
    for _ in 0..N {
        let ev = session
            .wait_for(Duration::from_secs(10), |e| {
                matches!(e, PlayerToEditor::Frame { .. })
            })
            .expect("a frame per step");
        if let PlayerToEditor::Frame { state_hash, .. } = ev {
            got.push(state_hash);
        }
    }
    let want = inf_player::scene_trace(&payload, N as u64).expect("shipping trace");
    assert_eq!(
        got, want,
        "the streamed PIE trace must equal the shipping trace for a level with a \
         machine-driven character"
    );
    // …and the trace MOVES on the step the machine transitions, which is what
    // makes the equality above a statement about the pose rather than about two
    // static worlds.
    assert!(
        got.windows(2).any(|w| w[0] != w[1]),
        "the trace never changed — the machine never left its entry state, so this \
         comparison says nothing about the pose"
    );
    session.stop(Duration::from_secs(5)).expect("graceful stop");
}
