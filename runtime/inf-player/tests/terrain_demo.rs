//! Phase-10-closing gate: terrain + PCG scatter survive the shipping pipeline.
//!
//! The editor authors the `terrain-demo` sample (schema-v4 `.inf_lvl` +
//! `Scatter.inf_pcg`). These integration tests prove:
//!
//! * **(b) cook → pack → player headless populate** — cooking the demo ships the
//!   referenced `.inf_pcg` through the level→graph dependency edge; the player
//!   builds the world off the pack with the terrain queryable (`height_at` matches
//!   the generator's function) and the PCG volume's scatter re-evaluated on load
//!   (nonzero, deterministic across two loads).
//! * **(c) PIE payload round-trip** — `build_scene_payload` over the loaded doc →
//!   `build_world_from_payload` yields the same terrain height probe + the same
//!   instance count as the shipping pack path (PIE == shipping for terrain/PCG).
//!
//! GPU rendering is human-verified as elsewhere; these assert **world
//! population** (entity counts, terrain data present, evaluated instances), which
//! is what the player render path (`render::project_scene`) then draws.

use std::path::{Path, PathBuf};

use inf_ecs::components::{PcgVolume, Terrain};
use inf_ecs::Guid;

use inf_editor_core::samples::{
    terrain_demo_height, TERRAIN_DEMO_PCG_ASSET_GUID, TERRAIN_DEMO_TERRAIN_GUID,
};
use inf_packager::{cook, CookOptions, DEFAULT_PACK_NAME};
use inf_player::level::{self, BuiltWorld, InfSceneWorldBuilder, PackLevelSource};
use inf_project::ProjectManifest;

/// The workspace root, from this crate at `runtime/inf-player`.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn terrain_demo_dir() -> PathBuf {
    workspace_root().join("samples/terrain-demo")
}

/// A grid-aligned probe point (a terrain sample, so `height_at` is exact to `f32`).
const PROBE: glam::DVec2 = glam::DVec2::new(16.0, 16.0);

/// Scaffold a project with the terrain-demo copied into its Content dir, cook it.
fn cook_terrain_demo(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Terrain Demo", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = terrain_demo_dir();
    for f in [
        "TerrainDemo.inf_lvl",
        "Scatter.inf_pcg",
        "Scatter.inf_pcg.toml",
    ] {
        std::fs::copy(src.join(f), content.join(f)).unwrap();
    }
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    out
}

/// Build the terrain-demo world off a cooked pack (the shipping `--pack` path),
/// including the PCG graph index so scatter re-evaluates on load.
fn pack_built(pack_dir: &Path) -> BuiltWorld {
    let source = PackLevelSource::open(pack_dir).expect("pack opens");
    let actors = source.actor_classes().expect("actor classes decode");
    let by_guid = source.blueprint_classes_by_guid().expect("blueprint index");
    let pcgs = source.pcg_payloads_by_guid().expect("pcg index");
    let builder = InfSceneWorldBuilder::with_defaults(actors)
        .with_bindings(by_guid)
        .with_pcgs(pcgs);
    level::load(&source, &builder).expect("pack level builds")
}

/// The terrain's `height_at(PROBE)` (must equal the generator function) + the
/// total count of PCG-evaluated instances in a built world.
fn probe_and_instance_count(built: &BuiltWorld) -> (f64, usize) {
    let te = built
        .world
        .entity_of(TERRAIN_DEMO_TERRAIN_GUID)
        .expect("terrain entity present");
    let w = built.world.world();
    let terrain = w
        .get::<Terrain>(te)
        .expect("terrain component instantiated");
    assert!(terrain.data.tile_count() >= 4, "multi-tile terrain loaded");
    let probe = terrain.data.height_at(PROBE).expect("probe on terrain");

    let instances: usize = w
        .iter_entities()
        .filter_map(|e| e.get::<PcgVolume>())
        .map(|v| v.evaluated.len())
        .sum();
    (probe, instances)
}

/// GATE (b): cook → pack → player headless populate.
#[test]
fn cooked_terrain_demo_populates_terrain_and_pcg_scatter() {
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_terrain_demo(dir.path());

    // The cook shipped the referenced `.inf_pcg` through the level→graph edge.
    let reader = inf_asset::PackReader::open(&pack.join(DEFAULT_PACK_NAME)).unwrap();
    assert!(
        reader.contains(inf_asset::AssetId(TERRAIN_DEMO_PCG_ASSET_GUID)),
        "the cook ships the PcgVolume.graph-referenced .inf_pcg"
    );

    let built = pack_built(&pack);
    assert_eq!(built.label, "Terrain Demo");

    // Terrain is queryable and matches the generator's height function.
    let (probe, instances) = probe_and_instance_count(&built);
    assert!(
        (probe - terrain_demo_height(PROBE.x, PROBE.y)).abs() < 1e-3,
        "height_at{PROBE:?}={probe} matches the generator function"
    );

    // The PCG volume re-evaluated on load → nonzero scatter.
    assert!(
        instances > 100,
        "PCG scatter evaluated on load (a few hundred instances), got {instances}"
    );

    // Deterministic across two independent loads off the same pack.
    let (_, instances2) = probe_and_instance_count(&pack_built(&pack));
    assert_eq!(
        instances, instances2,
        "PCG scatter is deterministic across two loads"
    );

    // Sanity: the demo has 4 entities (terrain, volume, sun, camera).
    let entities = built
        .world
        .world()
        .iter_entities()
        .filter(|e| e.contains::<Guid>())
        .count();
    assert_eq!(entities, 4);
}

/// The cook's level→`PcgVolume.graph` dependency edge: with ONLY the level as an
/// explicit root, the referenced `.inf_pcg` is still shipped (pulled through the
/// real level→graph edge, not because it is a default root).
#[test]
fn explicit_level_root_pulls_the_referenced_pcg_via_graph_edge() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Terrain Demo", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = terrain_demo_dir();
    for f in [
        "TerrainDemo.inf_lvl",
        "Scatter.inf_pcg",
        "Scatter.inf_pcg.toml",
    ] {
        std::fs::copy(src.join(f), content.join(f)).unwrap();
    }

    // Resolve the level's asset GUID from a scan of the same content root.
    let mut db = inf_asset::AssetDb::new(content.clone());
    db.scan().unwrap();
    let level_id = db
        .iter()
        .find(|e| e.kind() == inf_asset::AssetKind::Level)
        .expect("level present")
        .id();

    let out = proj.join("out");
    let report = cook(
        &proj,
        &out,
        &CookOptions {
            roots: Some(vec![level_id]),
            pack_name: None,
            ..Default::default()
        },
    )
    .expect("explicit-roots cook succeeds");

    assert_eq!(
        report.kinds.get("pcg"),
        Some(&1),
        "the referenced .inf_pcg is packed via the graph edge, not as a default root"
    );
    let reader = inf_asset::PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    assert!(
        reader.contains(inf_asset::AssetId(TERRAIN_DEMO_PCG_ASSET_GUID)),
        "the pack contains the referenced scatter graph"
    );
}

/// GATE (c): PIE payload round-trip == shipping for terrain/PCG content.
#[test]
fn pie_payload_matches_shipping_for_terrain_and_pcg() {
    // The shipping reference (cook → pack → populate).
    let dir = tempfile::tempdir().unwrap();
    let pack = cook_terrain_demo(dir.path());
    let (ship_probe, ship_instances) = probe_and_instance_count(&pack_built(&pack));

    // The PIE path: load the committed doc, build the payload (level bytes + the
    // referenced `.inf_pcg` bytes), and build the world exactly like the player.
    let tdir = terrain_demo_dir();
    let doc = inf_editor_core::scene::serialize::load(&tdir.join("TerrainDemo.inf_lvl"))
        .expect("terrain-demo doc loads");
    let pcg_bytes = std::fs::read(tdir.join("Scatter.inf_pcg")).unwrap();
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        // No blueprint classes in this scene.
        |_guid| None,
        // Resolve the PcgVolume.graph ref to its `.inf_pcg` bytes.
        |guid| (guid == TERRAIN_DEMO_PCG_ASSET_GUID).then(|| pcg_bytes.clone()),
        // No animation assets in this scene.
        |_guid| None,
        // No biome sets in this scene.
        |_guid| None,
        // No voxel volumes in this scene.
        |_guid| None,
        // The terrain is INLINE in this scene, so there is no `.inf_terrain`.
        |_guid| None,
        60,
        false,
    )
    .expect("payload builds");
    // The payload carries the referenced PCG graph.
    assert_eq!(payload.pcgs.len(), 1, "the referenced .inf_pcg rides along");

    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    let (pie_probe, pie_instances) = probe_and_instance_count(&built);

    assert_eq!(
        pie_probe, ship_probe,
        "PIE terrain height probe == shipping"
    );
    assert_eq!(
        pie_instances, ship_instances,
        "PIE PCG instance count == shipping (PIE == shipping for terrain/PCG)"
    );
    assert!(pie_instances > 100, "nonzero scatter through the PIE path");
}
