//! Phase-12-closing gate: the committed physics playground runs deterministically
//! under the shipping pipeline, and PIE == shipping.
//!
//! The editor authors the `physics-playground` sample (current-schema `.inf_lvl` + the
//! two `.inf_audio` clips). These integration tests prove:
//!
//! * **(a) byte-identical save/reload** — the committed doc save→load→save is
//!   byte-identical (a genuine current-schema payload carrying joints + audio).
//! * **(b) cooked headless replay** — cooking the demo ships the referenced
//!   `.inf_audio` clips through the new level→audio dependency edge; the player
//!   builds the world off the pack and runs 300 fixed steps **twice**, yielding a
//!   byte-identical **pose trace** (Guid-sorted transforms) AND an identical
//!   **audio command stream**. The ragdoll settles bounded, the CCD bullet is
//!   stopped by the thin wall, and the collision-layer ghost pair interpenetrates
//!   unimpeded (the layer proof).
//! * **(c) PIE payload round-trip** — `build_scene_payload` over the loaded doc →
//!   `build_world_from_payload` produces an identical pose trace + audio command
//!   stream to the shipping pack path (PIE == shipping for physics + audio).
//!
//! GPU rendering + device audio are human-verified as elsewhere; these assert the
//! authoritative deterministic sim state (entity transforms + the drained audio
//! command queue), which the render/audio host paths then consume.

use std::path::{Path, PathBuf};

use inf_audio::AudioCommand;
use inf_editor_core::samples::{
    PLAYGROUND_BULLET_GUID, PLAYGROUND_GHOST_A_GUID, PLAYGROUND_GHOST_B_GUID,
    PLAYGROUND_RAGDOLL_BASE_GUID, PLAYGROUND_SENSOR_CLIP_GUID, PLAYGROUND_SPINNER_CLIP_GUID,
    PLAYGROUND_SPINNER_WHEEL_GUID,
};
use inf_packager::{cook, CookOptions, DEFAULT_PACK_NAME};
use inf_player::level::{BuiltWorld, InfSceneWorldBuilder, PackLevelSource};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_project::ProjectManifest;

const STEPS: usize = 300;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn playground_dir() -> PathBuf {
    workspace_root().join("samples/physics-playground")
}

/// The committed physics-playground files (payload + inf_asset sidecars).
const DEMO_FILES: &[&str] = &[
    "Playground.inf_lvl",
    "Playground.inf_lvl.toml",
    "Spinner.inf_audio",
    "Spinner.inf_audio.toml",
    "Sensor.inf_audio",
    "Sensor.inf_audio.toml",
];

/// Scaffold a project with the playground copied into its Content dir, cook it.
fn cook_playground(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Physics Playground", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = playground_dir();
    for f in DEMO_FILES {
        std::fs::copy(src.join(f), content.join(f)).unwrap();
    }
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

/// Build the playground world off a cooked pack (the shipping `--pack` path),
/// seeding the audio clips so the AudioSources resolve like the editor Simulate.
fn pack_built(pack_dir: &Path) -> BuiltWorld {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let actors = source.actor_classes().expect("actor classes decode");
    let audio = source.audio_assets().expect("audio index");
    let builder = InfSceneWorldBuilder::with_defaults(actors).with_audio(audio);
    inf_player::level::load(&source, &builder).expect("pack level builds")
}

/// The Guid-sorted pose bytes of the sim world (translation + euler rotation of
/// every entity carrying a transform) — the deterministic pose sample.
fn pose_bytes(sim: &RuntimeSim) -> Vec<u8> {
    use inf_ecs::components::Transform;
    use inf_ecs::Guid;
    let w = sim.world().world();
    let mut poses: Vec<(uuid::Uuid, Transform)> = w
        .iter_entities()
        .filter_map(|e| Some((e.get::<Guid>()?.0, *e.get::<Transform>()?)))
        .collect();
    poses.sort_by_key(|(g, _)| *g);
    let mut bytes = Vec::new();
    for (g, t) in poses {
        bytes.extend_from_slice(g.as_bytes());
        for c in [
            t.translation.x,
            t.translation.y,
            t.translation.z,
            t.rotation.x,
            t.rotation.y,
            t.rotation.z,
        ] {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    bytes
}

/// Run `STEPS` fixed steps, accumulating a per-step pose trace + returning the
/// final drained audio command stream (as a deterministic debug digest).
fn run_trace(sim: &mut RuntimeSim) -> (Vec<u8>, String) {
    let mut trace = Vec::new();
    for _ in 0..STEPS {
        sim.step_once(RuntimeInput::default());
        trace.extend_from_slice(&pose_bytes(sim));
    }
    let audio = format!("{:?}", sim.audio_command_log());
    (trace, audio)
}

/// The number of `Play` commands in a sim's drained audio stream.
fn play_count(sim: &RuntimeSim) -> usize {
    sim.audio_command_log()
        .iter()
        .filter(|c| matches!(c, AudioCommand::Play(_)))
        .count()
}

/// The world translation of a Guid-identified entity in the sim.
fn pos_of(sim: &RuntimeSim, guid: uuid::Uuid) -> glam::DVec3 {
    let e = sim.world().entity_of(guid).expect("entity present");
    sim.world().world_translation(e).expect("world transform")
}

/// GATE (a): the committed doc save→load→save is byte-identical at the current schema.
#[test]
fn committed_playground_saves_and_reloads_byte_identical() {
    let doc = inf_editor_core::scene::serialize::load(&playground_dir().join("Playground.inf_lvl"))
        .expect("playground doc loads");
    let bytes1 = inf_editor_core::scene::serialize::encode(
        &inf_editor_core::scene::serialize::to_scene_file(&doc),
    )
    .expect("encode");
    assert_eq!(
        bytes1[0] as u32,
        inf_scene::SCHEMA_VERSION,
        "the committed playground is a current-schema payload"
    );

    let mut doc2 = inf_editor_core::scene::SceneDoc::new();
    inf_editor_core::scene::serialize::apply_to_doc(
        &mut doc2,
        &inf_editor_core::scene::serialize::decode(&bytes1).unwrap(),
    );
    let bytes2 = inf_editor_core::scene::serialize::encode(
        &inf_editor_core::scene::serialize::to_scene_file(&doc2),
    )
    .expect("encode");
    assert_eq!(bytes1, bytes2, "save→load→save must be byte-identical");
}

/// The cook's level→`AudioSource.clip` dependency edge ships the referenced
/// `.inf_audio` clips (they are not default cook roots — they enter only through
/// the level's dependency closure).
#[test]
fn cook_ships_the_referenced_audio_clips_via_the_audio_edge() {
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_playground(dir.path());
    let reader = inf_asset::PackReader::open(&pack.join(DEFAULT_PACK_NAME)).unwrap();
    for (label, guid) in [
        ("spinner clip", PLAYGROUND_SPINNER_CLIP_GUID),
        ("sensor clip", PLAYGROUND_SENSOR_CLIP_GUID),
    ] {
        assert!(
            reader.contains(inf_asset::AssetId(guid)),
            "the cook ships the {label} ({guid}) via the AudioSource.clip edge"
        );
    }
}

/// GATE (b): cook → pack → 300-step replay twice is byte-identical in both the
/// pose trace and the audio command stream, and the physics behaviours hold.
#[test]
fn cooked_playground_replays_deterministically() {
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_playground(dir.path());

    let built = pack_built(&pack);
    assert_eq!(built.label, "Physics Playground");
    // The two committed clips resolved into the runtime audio map.
    assert!(built
        .audio_clips
        .contains_key(&PLAYGROUND_SPINNER_CLIP_GUID));
    assert!(built.audio_clips.contains_key(&PLAYGROUND_SENSOR_CLIP_GUID));

    let mut sim = inf_player::sim_from_built(built);
    let (trace, audio) = run_trace(&mut sim);
    assert!(!trace.is_empty());

    // ── Audio: exactly the two autoplay sources each enqueued one Play. ──
    assert_eq!(
        play_count(&sim),
        2,
        "the two autoplay AudioSources each enqueue exactly one Play"
    );

    // ── The CCD bullet is stopped by the thin wall (top ≈ 5.02), not tunnelled. ──
    let bullet_y = pos_of(&sim, PLAYGROUND_BULLET_GUID).y;
    assert!(
        bullet_y > 4.5,
        "CCD stops the fast bullet on the thin wall (y={bullet_y}, not tunnelled below)"
    );

    // ── The collision-layer ghost pair interpenetrates unimpeded: co-located
    //    (the solver never shoved them apart) and free-fell through the floor. ──
    let ga = pos_of(&sim, PLAYGROUND_GHOST_A_GUID);
    let gb = pos_of(&sim, PLAYGROUND_GHOST_B_GUID);
    assert!(
        (ga - gb).length() < 1e-6,
        "the ghost pair stays perfectly co-located (interpenetrating): {ga:?} vs {gb:?}"
    );
    assert!(
        ga.y < 0.0,
        "the ghost pair free-fell through the floor (layer filter ignores it): y={}",
        ga.y
    );

    // ── The ragdoll settles bounded (every part finite + within a sane box). ──
    for i in 0..8u128 {
        let g = uuid::Uuid::from_u128(PLAYGROUND_RAGDOLL_BASE_GUID + i);
        if let Some(e) = sim.world().entity_of(g) {
            let p = sim.world().world_translation(e).unwrap();
            assert!(
                p.is_finite() && p.length() < 100.0,
                "ragdoll part {i} settled bounded (p={p:?})"
            );
        }
    }

    // ── The spinner is actually rotating (its motorized joint drives it): the
    //    wheel's euler rotation is non-zero after the run. ──
    let we = sim
        .world()
        .entity_of(PLAYGROUND_SPINNER_WHEEL_GUID)
        .unwrap();
    let wheel_rot = sim
        .world()
        .world()
        .get::<inf_ecs::components::Transform>(we)
        .unwrap()
        .rotation;
    assert!(
        wheel_rot.to_dvec3().length() > 0.05,
        "the motorized spinner joint rotated the wheel (rot={wheel_rot:?})"
    );

    // ── Determinism: a second independent cook+run yields the identical trace +
    //    audio command stream. ──
    let dir2 = tempfile::tempdir().unwrap();
    let pack2 = cook_playground(dir2.path());
    let mut sim2 = inf_player::sim_from_built(pack_built(&pack2));
    let (trace2, audio2) = run_trace(&mut sim2);
    assert_eq!(
        trace, trace2,
        "the pose trace is byte-identical across two runs"
    );
    assert_eq!(
        audio, audio2,
        "the audio command stream is identical across two runs"
    );
}

/// GATE (c): PIE payload round-trip == shipping for the physics playground (pose
/// trace + audio command stream).
#[test]
fn pie_payload_matches_shipping_for_playground() {
    // The shipping reference (cook → pack → run).
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_playground(dir.path());
    let mut ship_sim = inf_player::sim_from_built(pack_built(&pack));
    let (ship_trace, ship_audio) = run_trace(&mut ship_sim);

    // The PIE path: load the committed doc, build the payload, build the world
    // exactly like the player. The AudioSource/AudioListener components ride along
    // in the level bytes (schema v6), so the audio command stream matches even
    // though the PIE payload carries no audio-clip bytes (the stream is a pure
    // function of components + transforms + the occlusion raycast).
    let doc = inf_editor_core::scene::serialize::load(&playground_dir().join("Playground.inf_lvl"))
        .expect("playground doc loads");
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |_guid| None, // no blueprint actors
        |_guid| None, // no PCG graphs
        |_guid| None, // no animation assets
        |_guid| None, // no biome sets
        |_guid| None, // no voxel volumes
        |_guid| None, // no streamed terrains
        // P22.3: no destructible meshes in this fixture.
        |_| None,
        // P26.3b: the cloth / hair / material / texture byte resolver. This
        // fixture authors none of the four.
        |_| None,
        60,
        false,
    )
    .expect("payload builds");

    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    let mut pie_sim = inf_player::sim_from_built(built);
    let (pie_trace, pie_audio) = run_trace(&mut pie_sim);

    assert_eq!(
        pie_trace, ship_trace,
        "PIE pose trace == shipping (PIE == shipping for physics)"
    );
    assert_eq!(
        pie_audio, ship_audio,
        "PIE audio command stream == shipping (PIE == shipping for audio)"
    );
}
