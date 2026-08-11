//! **Phase 24 gate — cloth & hair, authored to shipped** (P24.4).
//!
//! The phase's "done when" ends with *"cloth and hair attach and both run under
//! existing state machines"*. This file is that clause, measured from the door an
//! author presses to the binary a player runs, in five arms:
//!
//! * **(a) AUTHORING** — the Model Editor's Ring-1 doors turn a mesh and a
//!   selection into a `.inf_cloth` and an `.inf_hair` that validate, seed and
//!   encode deterministically. Before P24.4's authoring batch,
//!   `ClothAsset::from_garment` and `HairAsset::grow` had no caller outside a
//!   test.
//! * **(b) BOTH HOSTS** — a character wearing both, driven by an
//!   `AnimStateMachine`, folds the same trace in the editor's `SimSession` and in
//!   the shipped `RuntimeSim`, with a non-empty cloth section and a non-empty
//!   hair section, and with the anti-vacuity arm that **taking the garment off
//!   changes the trace**.
//! * **(c) THE COOK** — a project holding that character cooks to a pack that
//!   carries both payloads, at an exact expected count taken from the fixture
//!   (the P21.4 rule: `!is_empty()` still passes on a level whose second asset
//!   silently stopped resolving).
//! * **(d) THE SHIPPED BINARY** — the real `inf-player` executable, booted on
//!   that pack with `--headless --run-frames`, folds the same hash as the
//!   in-process reference over the same pack. A subprocess, not a library call,
//!   for the P21.4 reason: a boot path that forgets an attachment does not crash,
//!   it agrees with itself.
//! * **(e) THE BOUND** — and the one place this does **not** hold, asserted
//!   rather than hoped: `ScenePayload` (v7, frozen for this phase) has no slot
//!   for a `.inf_cloth` or an `.inf_hair`, so the editor's `--pie` preview
//!   resolves neither. The arm measures the gap and **fails the day it closes**,
//!   which is what turns a ledger entry into something that cannot be forgotten.
//!
//! # The fixture is generated, not committed
//!
//! Every byte here is produced by the shipped generators — `inf_dcc::plane`,
//! `inf_editor_core::groom`, `inf_anim::build_template`-shaped rigs — so there is
//! no committed binary that can drift from the code that reads it, and the
//! authoring arm and the shipping arm are demonstrably looking at the *same*
//! garment.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use glam::{DVec2, Mat4, Quat, Vec3};
use uuid::Uuid;

use inf_anim::{
    AnimClip, ClothAsset, HairAsset, HairGroom, Interpolation, Joint, JointTransform, QuatTrack,
    Skeleton, SkeletonAsset, SmState, SmTransition, StateMachine, StateMachineAsset,
};
use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_ecs::components::{AnimStateMachine, ClothSim, HairGuides, SkeletalMesh, Transform};
use inf_ecs::math::Vec3d;
use inf_ecs::EcsWorld;
use inf_editor_core::groom::{garment_from_session, groom_from_session, GarmentSpec, GroomSpec};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::{serialize, SceneDoc};
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_packager::{cook, CookOptions};
use inf_player::runtime_sim::RuntimeSim;
use inf_project::ProjectManifest;

const HZ: f64 = 60.0;
/// Steps every trace in this file is folded over.
const STEPS: u64 = 12;

const HERO: Uuid = Uuid::from_u128(0x2404_2000_0000_0000_0000_0000_0000_0001);
const LEVEL: Uuid = Uuid::from_u128(0x2404_2000_0000_0000_0000_0000_0000_0002);
const SKEL: Uuid = Uuid::from_u128(0x2404_2000_0000_0000_0000_0000_0000_0003);
const MESH: Uuid = Uuid::from_u128(0x2404_2000_0000_0000_0000_0000_0000_0004);
const SM: Uuid = Uuid::from_u128(0x2404_2000_0000_0000_0000_0000_0000_0005);
const CLOTH: Uuid = Uuid::from_u128(0x2404_2000_0000_0000_0000_0000_0000_0006);
const HAIR: Uuid = Uuid::from_u128(0x2404_2000_0000_0000_0000_0000_0000_0007);
const IDLE: Uuid = Uuid::from_u128(0x2404_2000_0000_0000_0000_0000_0000_0010);
const WAVE: Uuid = Uuid::from_u128(0x2404_2000_0000_0000_0000_0000_0000_0011);

/// The exact number of `.inf_cloth` + `.inf_hair` entries the cooked pack must
/// carry — **one each**, taken from the fixture. An exact count and not
/// `!is_empty()`, which is the P21.4/P22.4 rule: a floor still passes on a level
/// whose second reference silently stopped resolving.
const EXPECTED_CLOTH_IN_PACK: usize = 1;
const EXPECTED_HAIR_IN_PACK: usize = 1;

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

// ── the character ───────────────────────────────────────────────────────────

/// A 3-joint chain up +Y with half-metre bones: root, mid (which the clips
/// swing), tip.
///
/// The garment's capsules and the hair's roots both ride joint **1 and 2** — the
/// bones the animation actually moves. The P24.4 cloth batch recorded the
/// alternative as a fixture that was wrong first: *a capsule over a bone the
/// animation cannot move proves nothing.*
fn rig() -> SkeletonAsset {
    let mut joints = Vec::new();
    let mut global = Mat4::IDENTITY;
    for i in 0..3 {
        let local = JointTransform::from_trs(
            if i == 0 { Vec3::ZERO } else { Vec3::Y * 0.5 },
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
        tracks: vec![inf_anim::JointTrack {
            joint: 1,
            translation: None,
            rotation: Some(QuatTrack::new(vec![0.0], vec![q], Interpolation::Step)),
            scale: None,
        }],
    }
}

/// idle → wave, unconditional, so the body the garment hangs on moves on the
/// first step. This is the "under existing state machines" half of the done-when
/// clause: nothing here drives the cloth directly.
fn machine() -> StateMachine {
    StateMachine {
        states: vec![
            SmState::clip("idle", *IDLE.as_bytes()),
            SmState::clip("wave", *WAVE.as_bytes()),
        ],
        transitions: vec![SmTransition {
            from: 0,
            to: 1,
            duration: 0.0,
            conditions: vec![],
            exit_time: None,
        }],
        entry: 0,
    }
}

/// **The garment, through the Model Editor's own door.** A 1 m plane, its
/// first four vertices pinned, prepared against the rig above.
fn authored_garment() -> ClothAsset {
    let mesh = inf_dcc::plane(1.0);
    let mut sel = inf_dcc::SelectionSet::new(0);
    for v in mesh.vert_ids().take(2) {
        sel.set_vert(v, true);
    }
    let (asset, report) = garment_from_session(
        &mesh,
        &sel,
        *MESH.as_bytes(),
        GarmentSpec {
            body_radius_m: 0.2,
            ..GarmentSpec::default()
        },
        Some(&rig().skeleton),
    )
    .expect("the plane is a garment");
    assert!(report.pinned > 0, "the fixture must pin something");
    assert!(report.capsules > 0, "the fixture must have a body to hit");
    asset
}

/// **The hairstyle, through the Model Editor's own door.** Guides grown out of
/// the plane's faces, clumped and curled — a *shaped* groom, because a straight
/// unclumped fixture agrees between hosts for reasons that have nothing to do
/// with the groom being applied the same way twice.
fn authored_hair() -> HairAsset {
    let mesh = inf_dcc::plane(1.0);
    let mut sel = inf_dcc::SelectionSet::new(0);
    for f in mesh.face_ids() {
        sel.set_face(f, true);
    }
    let (asset, report) = groom_from_session(
        &mesh,
        &sel,
        *MESH.as_bytes(),
        GroomSpec {
            length_m: 0.3,
            segments: 5,
            groom: HairGroom {
                clump_strength: 0.5,
                curl_radius_m: 0.01,
                curl_turns: 1.5,
            },
            clump_spacing_m: 0.5,
            // Joint 2 is the tip of the chain, which bending joint 1 swings.
            fallback_joint: 2,
            body_radius_m: 0.2,
            ..GroomSpec::default()
        },
        Some(&rig().skeleton),
    )
    .expect("the plane grows guides");
    assert!(report.strands > 0, "the fixture must grow strands");
    assert!(report.capsules > 0, "the fixture must have a head to miss");
    asset
}

/// The character's components. `wearing` is the anti-vacuity switch: `false`
/// leaves both components authored and **unbound**, which is a character with the
/// same everything else and no secondary animation at all.
fn character(
    wearing: bool,
) -> (
    SkeletalMesh,
    AnimStateMachine,
    Transform,
    ClothSim,
    HairGuides,
) {
    (
        SkeletalMesh {
            mesh: Some(MESH),
            skeleton: Some(SKEL),
        },
        AnimStateMachine {
            sm: Some(SM),
            ..Default::default()
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
        HairGuides {
            asset: wearing.then_some(HAIR),
            enabled: true,
            quality: 0,
        },
    )
}

fn doc_with_character(wearing: bool) -> SceneDoc {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, SpawnKind::Empty, "Hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(character(wearing));
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc
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

/// The world both host traces are folded over.
fn world_with_character(wearing: bool) -> EcsWorld {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character(wearing));
    world.mark_dirty();
    world
}

/// **The three sections `state_bytes` appends in order** — pose, then cloth, then
/// hair — read straight off a world.
///
/// The comparison unit for both hosts, rather than each host's whole
/// `state_bytes`: the editor's `SimSession` has no `state_bytes` of its own (the
/// snapshot half is the document's), so comparing the sections is comparing
/// exactly the thing both hosts are claimed to agree about, with nothing in it
/// that only one of them has. The ORDER is the one pinned by `inf-ecs`.
fn sections(world: &EcsWorld) -> Vec<u8> {
    let mut out = inf_ecs::pose::pose_state_bytes(world);
    out.extend_from_slice(&inf_ecs::cloth::cloth_state_bytes(world));
    out.extend_from_slice(&inf_ecs::hair::hair_state_bytes(world));
    out
}

/// The full sim trace: pose ++ cloth ++ hair, per step, as the shipped
/// `RuntimeSim` folds it.
fn runtime_trace(wearing: bool) -> Vec<Vec<u8>> {
    let mut sim = RuntimeSim::new(world_with_character(wearing), Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    sim.set_cloths(BTreeMap::from([(CLOTH, authored_garment())]));
    sim.set_hairs(BTreeMap::from([(HAIR, authored_hair())]));
    (0..STEPS)
        .map(|_| {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            sections(sim.world())
        })
        .collect()
}

/// The same, through the editor's `SimSession`.
fn editor_trace(wearing: bool) -> Vec<Vec<u8>> {
    let mut doc = doc_with_character(wearing);
    let mut session = SimSession::enter(&mut doc, Vec::new(), DVec2::ZERO, HZ);
    session.set_state_machines(machines());
    session.set_skeletons(skeletons());
    session.set_pose_clips(pose_clips());
    session.set_cloths(BTreeMap::from([(CLOTH, authored_garment())]));
    session.set_hairs(BTreeMap::from([(HAIR, authored_hair())]));
    let out = (0..STEPS)
        .map(|_| {
            session.step_once(&mut doc, SimInput::default());
            sections(doc.world())
        })
        .collect();
    session.exit(&mut doc);
    out
}

// ── (a) AUTHORING ───────────────────────────────────────────────────────────

/// **The Model Editor's doors mint both payloads.**
///
/// P24.4's first two batches built the payloads, the solvers and both hosts'
/// fixed steps and left the two generators with no caller outside a test — so
/// "authored in the Model Editor" was a claim about a button nobody had built.
/// This is that claim, measured through the Ring-1 doors the commands call.
#[test]
fn the_model_editor_doors_mint_a_garment_and_a_hairstyle() {
    let garment = authored_garment();
    assert_eq!(garment.validate(), Ok(()));
    assert!(inf_anim::ClothState::seed(&garment).is_ok());
    assert!(
        garment.distance.len() >= 4 && !garment.bending.is_empty(),
        "a garment with no constraints is a bag of dust: {} stretch, {} bend",
        garment.distance.len(),
        garment.bending.len()
    );

    let hair = authored_hair();
    assert_eq!(hair.validate(), Ok(()));
    let seeded = inf_anim::HairState::seed(&hair).expect("the hairstyle seeds");
    assert!(seeded.strand_count() > 0);

    // Deterministic to the byte, on both doors — the property every committed
    // asset in this engine rests on.
    assert_eq!(
        inf_asset::encode(&garment).unwrap(),
        inf_asset::encode(&authored_garment()).unwrap()
    );
    assert_eq!(
        inf_asset::encode(&hair).unwrap(),
        inf_asset::encode(&authored_hair()).unwrap()
    );
}

// ── (b) BOTH HOSTS ──────────────────────────────────────────────────────────

/// **The done-when clause, in one arm**: a character wearing an authored garment
/// AND an authored hairstyle, driven by a state machine, folds the same trace in
/// the editor and in the shipped runtime — with both sections non-empty and with
/// the anti-vacuity arm that taking them off changes the trace.
#[test]
fn both_hosts_fold_the_authored_character_byte_for_byte() {
    let editor = editor_trace(true);
    let runtime = runtime_trace(true);
    assert_eq!(editor.len(), STEPS as usize);
    assert_eq!(
        editor, runtime,
        "the editor's Simulate and the shipped RuntimeSim folded different \
         bytes for one authored character"
    );

    // The world is alive: the machine is animating under the two solvers.
    assert!(
        editor.windows(2).any(|w| w[0] != w[1]),
        "the trace never changed — this comparison says nothing"
    );

    // Both sections are actually present. `state_bytes` appends
    // snapshot ++ deform ++ pose ++ cloth ++ hair, so an equality between two
    // characters that wear nothing is an equality about the pose alone.
    let mut sim = RuntimeSim::new(world_with_character(true), Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(pose_clips());
    sim.set_cloths(BTreeMap::from([(CLOTH, authored_garment())]));
    sim.set_hairs(BTreeMap::from([(HAIR, authored_hair())]));
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    assert!(
        !inf_ecs::cloth::cloth_state_bytes(sim.world()).is_empty(),
        "no garment simulated — the ClothSim component was not read"
    );
    assert!(
        !inf_ecs::hair::hair_state_bytes(sim.world()).is_empty(),
        "no hair simulated — the HairGuides component was not read"
    );

    // ANTI-VACUITY: unbind both and the trace must move. A host that carried the
    // components and never folded them passes everything above and fails here.
    assert_ne!(
        runtime,
        runtime_trace(false),
        "taking the garment and the hairstyle off changed nothing — the two \
         components are being carried and not simulated"
    );
}

// ── (c) THE COOK ────────────────────────────────────────────────────────────

/// Write a payload + sidecar into `Content`, registered under `id`.
fn put<T: inf_asset::AssetPayload>(content: &Path, name: &str, id: Uuid, payload: &T) {
    let bytes = inf_asset::encode(payload).expect("payload encodes");
    let path = content.join(name);
    std::fs::write(&path, &bytes).unwrap();
    AssetSidecar::new(AssetId(id), T::KIND, ContentHash::of(&bytes))
        .save(&path)
        .unwrap();
}

/// Scaffold a project holding the character level and every asset it references.
/// Returns the project root.
fn scaffold(tmp: &Path, wearing: bool) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Phase 24 Character", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let doc = doc_with_character(wearing);
    let level = serialize::encode(&serialize::to_scene_file(&doc)).expect("encode level");
    let path = content.join("Character.inf_lvl");
    std::fs::write(&path, &level).unwrap();
    AssetSidecar::new(AssetId(LEVEL), AssetKind::Level, ContentHash::of(&level))
        .save(&path)
        .unwrap();

    put(&content, "Body.inf_mesh", MESH, &body_mesh());
    put(&content, "Rig.inf_skel", SKEL, &rig());
    put(
        &content,
        "Machine.inf_sm",
        SM,
        &StateMachineAsset {
            schema_version: StateMachineAsset::CURRENT_VERSION,
            machine: machine(),
            skeleton: Some(*SKEL.as_bytes()),
        },
    );
    put(
        &content,
        "Idle.inf_anim",
        IDLE,
        &inf_anim::AnimClipAsset::new(hold(0.0), Some(*SKEL.as_bytes())),
    );
    put(
        &content,
        "Wave.inf_anim",
        WAVE,
        &inf_anim::AnimClipAsset::new(hold(70.0), Some(*SKEL.as_bytes())),
    );
    put(&content, "Coat.inf_cloth", CLOTH, &authored_garment());
    put(&content, "Mane.inf_hair", HAIR, &authored_hair());
    proj
}

/// A minimal skinned body mesh — the `SkeletalMesh.mesh` reference. Its geometry
/// is irrelevant to every claim here (nothing in this file draws); what matters
/// is that the reference resolves, because a cook whose dependency walk cannot
/// find it drops the level's edges around it.
fn body_mesh() -> inf_mesh::MeshAsset {
    let (asset, _) =
        inf_dcc::to_mesh_asset(&inf_dcc::cube(0.5), &inf_dcc::ExportOptions::default());
    asset
}

/// The cook options every arm here uses: vgeom off, because deriving meshlet
/// DAGs dominates the runtime of a test that only cares about whether two
/// payloads shipped (`fracture_equivalence`'s rationale, verbatim).
fn cook_opts() -> CookOptions {
    CookOptions {
        vgeom: inf_packager::VgeomCookOptions {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// **The cooked pack carries the garment and the hairstyle**, at an exact count.
///
/// The edge that makes this true is the cook's dependency walk over
/// `ClothSim.asset` / `HairGuides.asset`. Without it a shipped character wears a
/// component nothing can resolve and simulates no cloth at all — while the
/// editor's Simulate, reading the project's content root directly, folds the coat
/// perfectly. That asymmetry is exactly what arm (d) below would otherwise fail
/// to notice, because two hosts that both simulate nothing agree.
#[test]
fn the_cooked_pack_carries_the_authored_garment_and_hairstyle() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let proj = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");
    let reader = inf_asset::PackReader::open(&report.pack_path).expect("open pack");
    let kinds: Vec<AssetKind> = reader.index().map(|e| e.kind).collect();
    assert_eq!(
        kinds.iter().filter(|k| **k == AssetKind::Cloth).count(),
        EXPECTED_CLOTH_IN_PACK,
        "the pack's `.inf_cloth` count is wrong — kinds: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == AssetKind::Hair).count(),
        EXPECTED_HAIR_IN_PACK,
        "the pack's `.inf_hair` count is wrong — kinds: {kinds:?}"
    );
    // …and the bytes are the ones the author's doors produced, not a re-derivation.
    let shipped: ClothAsset = inf_asset::decode(&reader.read(AssetId(CLOTH)).expect("cloth bytes"))
        .expect("the shipped garment decodes");
    assert_eq!(shipped, authored_garment());
    let shipped_hair: HairAsset =
        inf_asset::decode(&reader.read(AssetId(HAIR)).expect("hair bytes"))
            .expect("the shipped hairstyle decodes");
    assert_eq!(shipped_hair, authored_hair());
}

// ── (d) THE SHIPPED BINARY ──────────────────────────────────────────────────

/// Fold the pack's trace **in process**, through the same builder the player's
/// `--pack` boot uses.
fn pack_trace(pack: &Path) -> u128 {
    let source = inf_player::level::PackLevelSource::open(pack).expect("open pack");
    let built = inf_player::build_world_from_pack(&source).expect("build world from pack");
    inf_player::fold_trace_sim(inf_player::sim_from_built(built), STEPS, None)
}

/// Run the real `inf-player` binary on `pack`, headless, and return its printed
/// `final-state-hash`.
fn shipped_hash(pack: &Path) -> u128 {
    let out = Command::new(player_bin())
        .args([
            "--pack",
            &pack.display().to_string(),
            "--headless",
            "--run-frames",
            &STEPS.to_string(),
        ])
        .output()
        .expect("the player binary runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "the shipped player exited {:?}\nstdout:\n{stdout}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let line = stdout
        .lines()
        .find_map(|l| l.strip_prefix("final-state-hash: "))
        .unwrap_or_else(|| panic!("no final-state-hash in:\n{stdout}"));
    u128::from_str_radix(line.trim(), 16).expect("the hash is hex")
}

/// **THE SHIPPING ARM.** The real player executable, booted on a cooked pack,
/// folds the same trace as the in-process reference over the same pack — and a
/// pack whose character wears nothing folds a *different* one.
///
/// A subprocess rather than a library call, for the P21.4 reason this repository
/// paid for once already: a boot path that forgets an attachment does not crash,
/// it agrees with itself. `--pack` is the path a shipped game takes, and it is
/// the only path today that carries a `.inf_cloth` at all (see arm (e)).
#[test]
fn the_shipped_player_runs_the_authored_character_from_its_pack() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let proj = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    let want = pack_trace(&report.pack_path);
    let got = shipped_hash(&report.pack_path);
    assert_eq!(
        got, want,
        "the shipped player binary folded a different trace from the in-process \
         reference over the SAME cooked pack"
    );

    // ANTI-VACUITY: the same level with both components unbound must fold a
    // different hash — otherwise the equality above is a statement about a
    // character that is not wearing anything.
    let bare = tempfile::tempdir().expect("tempdir");
    let bare_proj = scaffold(bare.path(), false);
    let bare_report =
        cook(&bare_proj, &bare.path().join("out"), &cook_opts()).expect("the bare project cooks");
    assert_ne!(
        shipped_hash(&bare_report.pack_path),
        got,
        "the shipped player folded the same trace with and without the garment \
         and the hairstyle — the components crossed into the pack and nothing \
         simulated them"
    );
}

// ── (e) THE BOUND ───────────────────────────────────────────────────────────

/// **PIE cannot preview a garment yet, and this is the measurement** (P24.4).
///
/// `ScenePayload` is frozen at v7 for this phase and has no slot for a
/// `.inf_cloth` or an `.inf_hair`: `skeletons`, `clips`, `machines`, `voxels`,
/// `terrains`, `fractures` and `meshes` are the whole tail. So the editor's PIE
/// preview — the windowed and the `--pie` subprocess paths alike — builds its
/// world through `build_world_from_payload`, which calls `with_anim_assets` and
/// **not** `with_cloth_assets` / `with_hair_assets`, and a character previewed in
/// PIE wears nothing while the same character in Simulate and in the shipped
/// pack wears both.
///
/// The P24.1 note on `SCENE_PAYLOAD_VERSION` predicted otherwise — *"P24.4's
/// cloth is sim state over a garment that is itself an `.inf_mesh`, so it needs a
/// GUID added to the collector and nothing added to the wire"*. That prediction
/// is wrong now: P24.4 **did** choose a distinct `.inf_cloth` kind (and an
/// `.inf_hair`), which by that same note's own words is "the honest reason for" a
/// v8.
///
/// So this arm asserts the gap rather than the fix, and it is written to **fail
/// the day the payload learns to carry them** — a ledger entry that cannot be
/// forgotten, the discipline `a_multi_axis_limit_is_not_applied` established.
#[test]
fn the_pie_payload_carries_no_garment_and_that_is_measured() {
    // The wire has no slot. Read as source, because the claim is about the type
    // and a value cannot show that a field is absent.
    let wire = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/inf-runtime/src/pie.rs"),
    )
    .expect("pie.rs is readable")
    .replace("\r\n", "\n");
    let decl = wire
        .split_once("pub struct ScenePayload {")
        .and_then(|(_, rest)| rest.split_once("\n}\n"))
        .map(|(body, _)| body.to_string())
        .expect("ScenePayload's declaration is findable");
    assert!(
        decl.contains("pub meshes:"),
        "the ScenePayload scope this arm read is not the struct — it found no \
         `meshes` field, so every assertion below would pass vacuously"
    );
    for field in ["pub cloths:", "pub hairs:"] {
        assert!(
            !decl.contains(field),
            "`ScenePayload` grew `{field}` — PIE can carry a garment now. Delete \
             this arm, wire `build_world_from_payload` through \
             `with_cloth_assets`/`with_hair_assets`, and retire the ledger entry \
             that says a PIE preview shows a character wearing nothing."
        );
    }
    // …and the payload builder really does not resolve them: the one seam every
    // PIE boot takes calls the anim door and neither cloth door.
    let player = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .expect("lib.rs is readable")
        .replace("\r\n", "\n");
    let payload_fn = player
        .split_once("pub fn build_world_from_payload(")
        .and_then(|(_, rest)| rest.split_once("\n}\n"))
        .map(|(body, _)| body.to_string())
        .expect("build_world_from_payload is findable");
    assert!(
        payload_fn.contains("with_anim_assets("),
        "the scope this arm read is not `build_world_from_payload` — it does not \
         even wire the anim assets, so the absence below proves nothing"
    );
    assert!(
        !payload_fn.contains("with_cloth_assets(") && !payload_fn.contains("with_hair_assets("),
        "the PIE payload path now seeds cloth/hair — good. Replace this arm with \
         a real `--pie` subprocess comparison over a garment-wearing character, \
         the way `pie_skinned.rs` does for the authored IK target."
    );
}
