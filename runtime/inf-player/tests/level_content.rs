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
    let by_guid = level::load_actor_classes_by_guid_from_dir(&sample);
    InfSceneWorldBuilder::with_defaults(actors)
        .with_bindings(by_guid)
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

/// Build the platformer world off a cooked pack (the `--pack` path).
fn pack_built(pack_dir: &Path) -> BuiltWorld {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let actors = source.actor_classes().expect("actor classes decode");
    let by_guid = source
        .blueprint_classes_by_guid()
        .expect("pack blueprint index");
    let builder = InfSceneWorldBuilder::with_defaults(actors).with_bindings(by_guid);
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

    // Physics is now persisted (schema v3): the player carries a body + collider
    // + character controller.
    use inf_ecs::components::{Collider2D, RigidBody2D};
    assert!(
        w.iter_entities()
            .filter(|e| e.contains::<RigidBody2D>())
            .count()
            >= 1,
        "rigid bodies persist in v3"
    );
    assert!(
        w.iter_entities()
            .filter(|e| e.contains::<Collider2D>())
            .count()
            >= 1,
        "colliders persist in v3"
    );

    // The persisted `actor` binding resolves the Coyote class onto the player.
    assert_eq!(
        built.actors.len(),
        1,
        "the player's actor binding is resolved"
    );
    assert_eq!(built.actors[0].0, uuid::Uuid::from_u128(0x8401_0004));
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

/// THE PAYOFF (P9.5): cook the v3 platformer → run it headless off the pack →
/// the persisted Coyote blueprint actually runs, so the trace differs from a bake
/// that binds no actor (gameplay is present, not just a static scene).
#[test]
fn cooked_gameplay_trace_differs_from_a_no_actor_bake() {
    let tmp = tempfile::tempdir().unwrap();
    let out = cook_sample(tmp.path());

    // With the persisted actor binding → gameplay runs.
    let with_actor = fold_trace(pack_built(&out), 90, None);

    // A no-actor bake: same cooked pack + level, but zero bindings and zero
    // fallback classes → the character is never ticked.
    let no_actor = {
        let source = PackLevelSource::open(&out).unwrap();
        use inf_player::level::LevelSource;
        let bytes = source.level_bytes().unwrap();
        let built = InfSceneWorldBuilder::new(vec![], glam::DVec2::ZERO, 60.0)
            .build(&bytes)
            .unwrap();
        assert!(built.actors.is_empty(), "the no-actor bake binds nothing");
        fold_trace(built, 90, None)
    };

    assert_ne!(
        with_actor, no_actor,
        "the coyote blueprint running off the cooked level changes the trace"
    );
}

/// The other half of the payoff: a **scripted-input** run off the cooked pack
/// moves the character (the blueprint's `input.is_down("right")` → move_and_slide
/// drives the player right), proving real, controllable gameplay off real content.
#[test]
fn scripted_input_moves_the_character_off_the_cooked_pack() {
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

    let tmp = tempfile::tempdir().unwrap();
    let out = cook_sample(tmp.path());
    let built = pack_built(&out);
    assert_eq!(built.actors.len(), 1, "the player's coyote actor is bound");

    let player = uuid::Uuid::from_u128(0x8401_0004);
    let mut sim = RuntimeSim::with_gravity(built.world, built.actors, built.gravity, built.hz);

    let x_of = |sim: &RuntimeSim| {
        let e = sim.world().entity_of(player).unwrap();
        sim.world().world_translation(e).unwrap().x
    };
    let before = x_of(&sim);
    for _ in 0..40 {
        sim.step_once(RuntimeInput::with_down(["right"]));
    }
    let after = x_of(&sim);
    assert!(
        after > before + 0.5,
        "holding 'right' should move the character right (before {before}, after {after})"
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
