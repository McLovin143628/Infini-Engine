//! Phase-11-closing gate: an idle/run/jump state-machine character, driven by a
//! Blueprint across P10-style terrain, survives the shipping pipeline.
//!
//! The editor authors the `character-demo` sample (schema-v5 `.inf_lvl` + the
//! `.inf_skel` / `.inf_anim` / `.inf_sm` anim assets + the `.inf_act` actor). These
//! integration tests prove:
//!
//! * **(b) cook → pack → player headless run** — cooking the demo ships every
//!   referenced anim asset through the level→component + state-machine→clip
//!   dependency edges; the player builds the world off the pack, and a scripted
//!   input run shows the character **cross the terrain** (x advances while Y tracks
//!   the terrain height at its X), **jump** (Y rises above the terrain then
//!   returns), and its **state machine transition idle→run→jump**. Deterministic
//!   across two runs.
//! * **(c) PIE payload round-trip** — `build_scene_payload` over the loaded doc →
//!   `build_world_from_payload` produces an identical position + state-machine
//!   trace to the shipping pack path (PIE == shipping for animation).
//!
//! Pose rendering (skinning) is human-verified as elsewhere; these assert the
//! authoritative sim state (entity transforms + state-machine indices) the render
//! path then draws.

use std::path::{Path, PathBuf};

use uuid::Uuid;

use inf_ecs::components::AnimStateMachine;
use inf_editor_core::samples::{
    character_demo_height, character_demo_state_machine, CHARACTER_DEMO_ACTOR_GUID,
    CHARACTER_DEMO_CHARACTER_GUID, CHARACTER_DEMO_IDLE_CLIP_GUID, CHARACTER_DEMO_JUMP_CLIP_GUID,
    CHARACTER_DEMO_RUN_CLIP_GUID, CHARACTER_DEMO_SKELETON_GUID, CHARACTER_DEMO_SM_GUID, CHAR_STAND,
};
use inf_packager::{cook, CookOptions, DEFAULT_PACK_NAME};
use inf_player::level::{BuiltWorld, PackLevelSource};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_project::ProjectManifest;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn character_demo_dir() -> PathBuf {
    workspace_root().join("samples/character-demo")
}

/// The committed character-demo files (payload + inf_asset sidecars).
const DEMO_FILES: &[&str] = &[
    "Character.inf_lvl",
    "Character.inf_lvl.toml",
    "Character.inf_skel",
    "Character.inf_skel.toml",
    "Idle.inf_anim",
    "Idle.inf_anim.toml",
    "Run.inf_anim",
    "Run.inf_anim.toml",
    "Jump.inf_anim",
    "Jump.inf_anim.toml",
    "Locomotion.inf_sm",
    "Locomotion.inf_sm.toml",
    "Character.inf_act",
    "Character.inf_act.toml",
];

/// Scaffold a project with the character-demo copied into its Content dir, cook it.
fn cook_character_demo(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Character Demo", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = character_demo_dir();
    for f in DEMO_FILES {
        std::fs::copy(src.join(f), content.join(f)).unwrap();
    }
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

/// One recorded step of a scripted run: the character's world X + Y and its
/// state-machine state index.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Sample {
    x: f64,
    y: f64,
    state: usize,
}

/// Read the character's `(x, y, sm-state)` sample from a sim.
fn sample(sim: &RuntimeSim) -> Sample {
    let e = sim
        .world()
        .entity_of(CHARACTER_DEMO_CHARACTER_GUID)
        .expect("character entity");
    let p = sim.world().world_translation(e).expect("world transform");
    let state = sim
        .world()
        .world()
        .get::<AnimStateMachine>(e)
        .expect("AnimStateMachine")
        .runtime
        .current;
    Sample {
        x: p.x,
        y: p.y,
        state,
    }
}

/// The scripted input program (held keys per step), the same on every path:
/// warm-up idle, then hold "right" across the terrain, a one-step "jump" rising
/// edge (still holding right), then keep running.
fn input_program() -> Vec<Vec<&'static str>> {
    let mut prog: Vec<Vec<&'static str>> = Vec::new();
    prog.extend((0..8).map(|_| vec![])); // idle warm-up
    prog.extend((0..120).map(|_| vec!["right"])); // run across the terrain
    prog.push(vec!["right", "jump"]); // jump rising edge (frame J)
    prog.extend((0..90).map(|_| vec!["right"])); // keep running (jump arc + land)
    prog
}

/// Run the scripted program on `sim`, recording a sample after every step.
fn run_program(sim: &mut RuntimeSim) -> Vec<Sample> {
    input_program()
        .into_iter()
        .map(|keys| {
            sim.step_once(RuntimeInput::with_down(keys));
            sample(sim)
        })
        .collect()
}

/// Build the character-demo world off a cooked pack (the shipping `--pack` path),
/// seeding the anim assets so the state machine steps.
fn pack_built(pack_dir: &Path) -> BuiltWorld {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let actors = source.actor_classes().expect("actor classes decode");
    let by_guid = source.blueprint_classes_by_guid().expect("blueprint index");
    let (skeletons, clips, machines) = source.anim_assets().expect("anim index");
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(actors)
        .with_bindings(by_guid)
        .with_anim_assets(skeletons, clips, machines);
    inf_player::level::load(&source, &builder).expect("pack level builds")
}

/// GATE (b): cook → pack → player headless run of the scripted character.
#[test]
fn cooked_character_demo_crosses_terrain_jumps_and_transitions() {
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_character_demo(dir.path());

    // The cook shipped every referenced anim asset through the dep edges: the
    // skeleton (SkeletalMesh + clip/sm refs), the state machine (AnimStateMachine),
    // its three clips (state-machine → clip closure), and the actor blueprint.
    let reader = inf_asset::PackReader::open(&pack.join(DEFAULT_PACK_NAME)).unwrap();
    for (label, guid) in [
        ("skeleton", CHARACTER_DEMO_SKELETON_GUID),
        ("state machine", CHARACTER_DEMO_SM_GUID),
        ("idle clip", CHARACTER_DEMO_IDLE_CLIP_GUID),
        ("run clip", CHARACTER_DEMO_RUN_CLIP_GUID),
        ("jump clip", CHARACTER_DEMO_JUMP_CLIP_GUID),
        ("actor", CHARACTER_DEMO_ACTOR_GUID),
    ] {
        assert!(
            reader.contains(inf_asset::AssetId(guid)),
            "the cook ships the {label} ({guid})"
        );
    }

    let built = pack_built(&pack);
    assert_eq!(built.label, "Character Demo");
    // The state machine resolved into the builder's seed map.
    assert!(
        built.state_machines.contains_key(&CHARACTER_DEMO_SM_GUID),
        "the .inf_sm resolves into the runtime state-machine map"
    );

    let mut sim = inf_player::sim_from_built(built);
    let trace = run_program(&mut sim);

    // ── The crossing-terrain proof: x advances AND y tracks the terrain height. ──
    let start = trace[7]; // end of the idle warm-up
    let after_run = trace[7 + 120]; // after holding right across the terrain
    assert!(
        after_run.x - start.x > 6.0,
        "holding right advances x across the terrain ({} → {})",
        start.x,
        after_run.x
    );
    // At the end of the run (grounded, pre-jump) Y == terrain height at X + stand.
    let want_y = character_demo_height(after_run.x, 0.0) + CHAR_STAND;
    assert!(
        (after_run.y - want_y).abs() < 0.1,
        "the character's Y tracks the terrain height at its X (y={}, want≈{want_y})",
        after_run.y
    );
    // It genuinely climbed: the terrain (and the character) rose from the origin.
    assert!(
        after_run.y - start.y > 0.5,
        "the character crossed onto higher terrain (y {} → {})",
        start.y,
        after_run.y
    );

    // ── The jump proof: Y rises above the terrain after the rising edge, then
    //    returns to the ground. ──
    let jump_idx = 8 + 120; // the "right"+"jump" step
    let ground_at = |i: usize| character_demo_height(trace[i].x, 0.0) + CHAR_STAND;
    let peak_above = (jump_idx..jump_idx + 40)
        .map(|i| trace[i].y - ground_at(i))
        .fold(f64::MIN, f64::max);
    assert!(
        peak_above > 0.4,
        "the jump lifts Y clearly above the terrain (peak +{peak_above})"
    );
    let landed = *trace.last().unwrap();
    assert!(
        (landed.y - (character_demo_height(landed.x, 0.0) + CHAR_STAND)).abs() < 0.1,
        "the character returns to the ground after the jump (y={})",
        landed.y
    );

    // ── The state-machine proof: idle(0) → run(1) → jump(2) all observed, in order. ──
    let sm = character_demo_state_machine();
    let name = |i: usize| sm.states[i].name.as_str();
    assert_eq!(name(0), "idle");
    assert_eq!(name(1), "run");
    assert_eq!(name(2), "jump");
    let first = |target: usize| trace.iter().position(|s| s.state == target);
    let idle_i = first(0);
    let run_i = first(1).expect("reaches the run state");
    let jump_i = first(2).expect("reaches the jump state");
    assert!(
        idle_i.map(|i| i < run_i).unwrap_or(true) && run_i < jump_i,
        "state machine transitions idle→run→jump in order (idle={idle_i:?}, run={run_i}, jump={jump_i})"
    );

    // ── Determinism: a second independent cook+run yields the identical trace. ──
    let dir2 = tempfile::tempdir().unwrap();
    let pack2 = cook_character_demo(dir2.path());
    let mut sim2 = inf_player::sim_from_built(pack_built(&pack2));
    let trace2 = run_program(&mut sim2);
    assert_eq!(
        trace, trace2,
        "the scripted run is deterministic across two runs"
    );
}

/// GATE (c): PIE payload round-trip == shipping for the animated character.
#[test]
fn pie_payload_matches_shipping_for_character_demo() {
    // The shipping reference (cook → pack → run).
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_character_demo(dir.path());
    let mut ship_sim = inf_player::sim_from_built(pack_built(&pack));
    let ship_trace = run_program(&mut ship_sim);

    // The PIE path: load the committed doc, build the payload (level bytes + the
    // actor class + the referenced skeleton/state-machine bytes), build the world
    // exactly like the player.
    let cdir = character_demo_dir();
    let doc = inf_editor_core::scene::serialize::load(&cdir.join("Character.inf_lvl"))
        .expect("character-demo doc loads");
    let act_bytes = std::fs::read(cdir.join("Character.inf_act")).unwrap();
    let class: inf_blueprint::BlueprintClass = serde_json::from_slice(&act_bytes).unwrap();
    // Resolve an anim asset GUID to its committed bytes.
    let anim_bytes = move |guid: Uuid| -> Option<Vec<u8>> {
        let file = if guid == CHARACTER_DEMO_SKELETON_GUID {
            "Character.inf_skel"
        } else if guid == CHARACTER_DEMO_SM_GUID {
            "Locomotion.inf_sm"
        } else if guid == CHARACTER_DEMO_IDLE_CLIP_GUID {
            "Idle.inf_anim"
        } else if guid == CHARACTER_DEMO_RUN_CLIP_GUID {
            "Run.inf_anim"
        } else if guid == CHARACTER_DEMO_JUMP_CLIP_GUID {
            "Jump.inf_anim"
        } else {
            return None;
        };
        std::fs::read(cdir.join(file)).ok()
    };
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| (guid == CHARACTER_DEMO_ACTOR_GUID).then(|| class.clone()),
        |_guid| None, // no PCG graphs
        anim_bytes,
        60,
        false,
    )
    .expect("payload builds");
    // The payload carries the directly-referenced skeleton + state machine.
    assert_eq!(
        payload.machines.len(),
        1,
        "the referenced .inf_sm rides along"
    );
    assert_eq!(
        payload.skeletons.len(),
        1,
        "the referenced .inf_skel rides along"
    );

    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    let mut pie_sim = inf_player::sim_from_built(built);
    let pie_trace = run_program(&mut pie_sim);

    assert_eq!(
        pie_trace, ship_trace,
        "PIE trace (position + state machine) == shipping (PIE == shipping for animation)"
    );
}
