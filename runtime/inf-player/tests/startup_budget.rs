//! Player startup budget: pack-load-to-first-world time (P15.1 / §8 ratchet).
//!
//! Cooks the platformer sample, then times the **load** phase — opening the
//! cooked pack and building the first world from it (the `--pack` boot path) —
//! and asserts it is under a generous bound. A regression tripwire (the number is
//! small; the budget only ratchets **down** per §8), not a perf claim. No GPU:
//! this is the CPU pack-decode + ECS world-build path.
//!
//! The bound is the shared [`LOAD_BUDGET_MS`], not a private copy: this test and
//! the Phase 19 town build measure the same *class* of thing (a one-shot world
//! build), and a class deserves one number. Its value is unchanged — this test is
//! the P15.1 precedent the constant was lifted from.

use std::path::{Path, PathBuf};

use inf_packager::{cook, CookOptions};
use inf_player::budget::LOAD_BUDGET_MS;
use inf_player::level::{self, InfSceneWorldBuilder, PackLevelSource};
use inf_project::ProjectManifest;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn cook_sample(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Platformer Sample", "2d-platformer")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let sample = workspace_root().join("samples/platformer-2d");
    for f in [
        "Platformer.inf_lvl",
        "Coyote.inf_act",
        "Coyote.inf_act.toml",
    ] {
        std::fs::copy(sample.join(f), content.join(f)).unwrap();
    }
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

#[test]
fn pack_load_to_first_world_under_budget() {
    let tmp = tempfile::tempdir().unwrap();
    let pack_dir = cook_sample(tmp.path());

    // Time only the load path: open pack → resolve actors/blueprints → build world.
    let start = std::time::Instant::now();
    let source = PackLevelSource::open(&pack_dir).expect("pack opens");
    let actors = source.actor_classes().expect("actor classes decode");
    let by_guid = source
        .blueprint_classes_by_guid()
        .expect("pack blueprint index");
    let builder = InfSceneWorldBuilder::with_defaults(actors).with_bindings(by_guid);
    let built = level::load(&source, &builder).expect("pack level builds");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Sanity: the world actually built.
    assert!(
        built.world.world().iter_entities().count() > 0,
        "the loaded world has entities"
    );

    eprintln!("startup: pack-load-to-first-world {elapsed_ms:.1} ms (budget {LOAD_BUDGET_MS} ms)");
    assert!(
        elapsed_ms < LOAD_BUDGET_MS,
        "pack load {elapsed_ms:.1} ms exceeded the {LOAD_BUDGET_MS} ms load budget \
         (the §8 budget only ratchets DOWN — investigate the regression, do not raise it)"
    );
}
