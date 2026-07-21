//! P9.5 deliverable 1: the player wired to real cooked content.
//!
//! Proves the `inf-scene`-backed world builder against the committed platformer
//! level, and — the strong gate — that a level run **off a cooked pack** produces
//! the byte-identical determinism trace as the same level read from the dev
//! directory (cooked == uncooked).

use std::path::{Path, PathBuf};

use inf_ecs::components::{Light2D, Sprite, Tilemap};
use inf_ecs::Guid;

use inf_player::level::{self, InfSceneWorldBuilder, PackLevelSource, WorldBuilder};
use inf_player::{fold_trace, level::BuiltWorld};

use inf_packager::{cook, CookOptions, DEFAULT_PACK_NAME};
use inf_project::ProjectManifest;

/// The workspace root, from this crate at `runtime/inf-player`.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn sample_dir() -> PathBuf {
    workspace_root().join("samples/platformer-2d")
}

/// Build the platformer world by reading the committed `.inf_lvl` + `.inf_act`
/// straight from the sample directory (the `--level` dev-dir path).
fn dev_built() -> BuiltWorld {
    let sample = sample_dir();
    let bytes = std::fs::read(sample.join("Platformer.inf_lvl")).unwrap();
    let actors = level::load_actor_classes_from_dir(&sample);
    InfSceneWorldBuilder::with_defaults(actors)
        .build(&bytes)
        .expect("dev-dir level builds")
}

/// Scaffold a project with the platformer sample and cook it into `out`.
fn cook_sample(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Platformer Sample", "2d-platformer")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let sample = sample_dir();
    for f in ["Platformer.inf_lvl", "Coyote.inf_act"] {
        std::fs::copy(sample.join(f), content.join(f)).unwrap();
    }
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

/// Build the platformer world off a cooked pack (the `--pack` path).
fn pack_built(pack_dir: &Path) -> BuiltWorld {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let actors = source.actor_classes().expect("actor classes decode");
    let builder = InfSceneWorldBuilder::with_defaults(actors);
    level::load(&source, &builder).expect("pack level builds")
}

#[test]
fn dev_level_builds_expected_entities_and_components() {
    let built = dev_built();
    assert_eq!(built.label, "Platformer 2D");

    let w = built.world.world();
    let entities = w.iter_entities().filter(|e| e.contains::<Guid>()).count();
    assert_eq!(entities, 5, "platformer has 5 entities");

    // The persisted visual components are instantiated.
    assert!(
        w.iter_entities().filter(|e| e.contains::<Sprite>()).count() >= 1,
        "player sprite present"
    );
    assert_eq!(
        w.iter_entities()
            .filter(|e| e.contains::<Tilemap>())
            .count(),
        1,
        "one tilemap ground strip"
    );
    assert_eq!(
        w.iter_entities()
            .filter(|e| e.contains::<Light2D>())
            .count(),
        1,
        "one 2D light"
    );

    // Physics/binding aren't persisted in `.inf_lvl` (documented follow-up), so
    // the level's actor list is empty today.
    assert!(
        built.actors.is_empty(),
        "no CharacterController2D persists in the level → no bound actors yet"
    );
}

#[test]
fn dev_level_trace_is_nonzero_and_reproducible() {
    let hash = fold_trace(dev_built(), 60, None);
    assert_ne!(hash, 0, "a real world hashes to a nonzero fingerprint");
    assert_eq!(
        hash,
        fold_trace(dev_built(), 60, None),
        "the 60-step trace is deterministic"
    );
}

#[test]
fn cooked_and_uncooked_produce_the_same_trace() {
    let tmp = tempfile::tempdir().unwrap();
    let out = cook_sample(tmp.path());
    assert!(out.join(DEFAULT_PACK_NAME).exists(), "pack cooked");

    let dev = fold_trace(dev_built(), 60, None);
    let cooked = fold_trace(pack_built(&out), 60, None);
    assert_eq!(
        dev, cooked,
        "the level runs identically whether read from the dev dir or the cooked pack"
    );
}

#[test]
fn pack_source_resolves_root_level_from_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let out = cook_sample(tmp.path());
    // Opening the directory (with manifest.toml) resolves the boot level + label.
    let source = PackLevelSource::open(&out).unwrap();
    let bytes = {
        use inf_player::level::LevelSource;
        source.level_bytes().unwrap()
    };
    let level = inf_scene::decode(&bytes).unwrap();
    assert_eq!(level.title, "Platformer 2D");
}
