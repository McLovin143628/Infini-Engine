//! The sample moddable game, end to end (ROADMAP P14.5, deliverable 4).
//!
//! Build the committed `samples/mods/spinner` mod to wasm (via the cook target's
//! build helper), drop it into a mods dir with its `mod.toml`, load it into a
//! real [`RuntimeSim`] over a tiny scene, step N fixed steps, and assert the mod
//! moved its entity — the full author → wasm → sandbox → sim story.
//!
//! Skips (does not fail) when the `wasm32-unknown-unknown` target is missing.

use std::path::{Path, PathBuf};

use glam::DVec2;
use uuid::Uuid;

use inf_blueprint::BlueprintClass;
use inf_ecs::EcsWorld;
use inf_packager::{build_mod_wasm, ModBuildOutcome};
use inf_player::mods::PlayerMods;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

fn spinner_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("samples")
        .join("mods")
        .join("spinner")
}

#[test]
fn scene_plus_spinner_mod_moves_its_entity() {
    // 1. Cook the sample mod to wasm.
    //
    // **Tool-absent skips; build-failed panics** — the distinction C4-40 built,
    // used here for the reason it exists. Only the dedicated wasm CI job installs
    // `wasm32-unknown-unknown`; the three ordinary Rust legs do not, and a test
    // that cannot build its subject there has proven nothing rather than found
    // something. A mod that fails to *compile* is a real defect and still fails.
    //
    // The skip prints the outcome's own instructions rather than a fixed line, so
    // a reader sees the named remedy (`rustup target add …`) and not just a
    // verdict — the house GPU-less / rust-analyzer-absent skip pattern.
    let wasm = match build_mod_wasm(&spinner_dir()) {
        Ok(ModBuildOutcome::Built(p)) => p,
        Ok(ModBuildOutcome::ToolchainMissing(why)) => {
            eprintln!(
                "SKIP {}: the wasm32-unknown-unknown target is not installed.\n{why}",
                module_path!()
            );
            return;
        }
        Err(e) => panic!("building spinner mod: {e}"),
    };

    // 2. Stage a mods dir: the built wasm + the sample's mod.toml capability grant.
    let mods_dir = tempfile::tempdir().unwrap();
    std::fs::copy(&wasm, mods_dir.path().join("spinner.wasm")).unwrap();
    std::fs::copy(
        spinner_dir().join("mod.toml"),
        mods_dir.path().join("mod.toml"),
    )
    .unwrap();

    // 3. A tiny scene: one actor entity (the spinner drives entity id 1, which the
    //    sim assigns to the first actor in Guid order).
    let mut world = EcsWorld::new();
    let guid = Uuid::from_u128(0x1111_1111_1111_1111_1111_1111_1111_1111);
    world.spawn_with_guid(guid, "Orbiter", None);
    let actors = vec![(guid, BlueprintClass::new("act:orbiter", "Orbiter"))];
    let mut sim = RuntimeSim::new(world, actors, DVec2::ZERO, 60.0);

    // 4. Load + attach the mod.
    let pm = PlayerMods::load(mods_dir.path(), 2).expect("load mods");
    assert!(!pm.is_empty(), "spinner mod should load");
    assert_eq!(pm.enabled_count(), 1);
    sim.set_mods(Box::new(pm));

    // 5. Step the sim; the mod orbits its entity.
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default());
    }

    let entity = sim.world().entity_of(guid).unwrap();
    let p = sim.world().world_translation(entity).unwrap();
    assert!((p.y - 1.0).abs() < 1e-6, "y pinned by the mod, got {p:?}");
    let r = (p.x * p.x + p.z * p.z).sqrt();
    assert!((r - 2.0).abs() < 1e-6, "on the orbit radius, got r={r}");
    assert!(p.x.abs() > 1e-6 || p.z.abs() > 1e-6, "moved off the origin");
}
