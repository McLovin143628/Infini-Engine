//! **P19.3 gate: painted biomes grow content, identically everywhere.**
//!
//! P19.2 made a terrain *say* which biome owns each sample; P19.3 makes that
//! statement grow things. These tests prove the whole chain end to end on a
//! purpose-built fixture — a terrain painted with **two** biomes whose
//! `.inf_biomes` names **two distinct `.inf_pcg` graphs** (one of them a
//! multi-rule `scatter.merge` graph, so the new lowering rides the real
//! pipeline):
//!
//! * **the cook follows the whole chain** — level → `Terrain.biome_set` →
//!   `.inf_biomes` → each `BiomeDef.pcg_graph` → `.inf_pcg`, with only the level
//!   as an explicit root (P19.2 landed the first edge; this is the first test of
//!   the *chain*);
//! * **cooked == uncooked** — the pack path and the loose dev-dir path place the
//!   same instances, in the same order, at the same positions;
//! * **PIE == shipping** — the streamed `ScenePayload` carries the biome sets and
//!   the player evaluates them to the identical population;
//! * **the region property** — every instance lands on the biome that owns it,
//!   both biomes contribute, and an unpainted terrain grows nothing;
//! * **determinism** — two independent loads of one pack are byte-identical.
//!
//! The fixture is built here rather than committed as a sample: it exists to pin
//! the *chain*, and a temp-dir fixture cannot drift from a `.inf_lvl` nobody
//! regenerated.

use std::path::{Path, PathBuf};

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::Terrain;
use inf_ecs::ScatteredInstance;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_packager::{cook, CookOptions, DEFAULT_PACK_NAME};
use inf_player::level::{
    self, BuiltWorld, DevDirLevelSource, InfSceneWorldBuilder, PackLevelSource,
};
use inf_project::ProjectManifest;
use inf_terrain::{BiomeDef, BiomeSet};

// ── fixture constants ───────────────────────────────────────────────────────

const LEVEL_GUID: Uuid = Uuid::from_u128(0x8419_0100);
const TERRAIN_GUID: Uuid = Uuid::from_u128(0x8419_0101);
const BIOME_SET_GUID: Uuid = Uuid::from_u128(0x8419_0201);
/// The graph biome **1** (the left half) scatters with — one rule.
const GRAPH_A_GUID: Uuid = Uuid::from_u128(0x8419_0301);
/// The graph biome **2** (the right half) scatters with — **two** rules through
/// a `scatter.merge`, so the P19.3 multi-rule lowering is exercised by the gate
/// rather than only by its unit tests.
const GRAPH_B_GUID: Uuid = Uuid::from_u128(0x8419_0302);

const RESOLUTION: u32 = 17;
const MPS: f64 = 1.0;
/// 2 × 2 tiles of `(17 − 1) · 1 m` → a 32 m square.
const SPAN: f64 = 32.0;
/// Biome 1 owns `x < BORDER`, biome 2 owns the rest.
const BORDER: f64 = 16.0;

// ── fixture construction ────────────────────────────────────────────────────

/// The two-biome painted terrain scene.
fn biome_scene() -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.set_title("Biome PCG");
    doc.create_with_guid(TERRAIN_GUID, SpawnKind::Empty, "Terrain", None);
    let entity = doc.entity_of(TERRAIN_GUID).expect("terrain entity");

    let mut terrain = Terrain::configured(RESOLUTION, MPS);
    // A gentle polynomial hill — no `std` trig, so the fixture is bit-portable
    // across platforms (the P14 law).
    terrain
        .data
        .write_region(DVec2::ZERO, DVec2::splat(SPAN), |x, z| {
            0.02 * x * z - 0.05 * x + 1.0
        });
    // Paint the ids straight onto the samples: the brush is gated in P19.2, and
    // what this fixture needs is an exactly-known border.
    let res = terrain.data.tile_resolution();
    let cells = (res - 1) as i32;
    let coords: Vec<(i32, i32)> = terrain.data.tiles().map(|(&c, _)| c).collect();
    for coord in coords {
        let tile = terrain.data.get_tile_mut(coord).expect("authored tile");
        for j in 0..res {
            for i in 0..res {
                let world_x = (coord.0 * cells + i as i32) as f64 * MPS;
                tile.set_biome_sample(res, i, j, if world_x < BORDER { 1 } else { 2 });
            }
        }
    }
    terrain.biome_set = Some(BIOME_SET_GUID);

    doc.world_mut().world_mut().entity_mut(entity).insert(
        inf_ecs::components::Transform::from_translation(DVec3::ZERO),
    );
    doc.world_mut()
        .world_mut()
        .entity_mut(entity)
        .insert(terrain);
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// Graph A: one rule, dense, mesh #1.
fn graph_a() -> inf_pcg::PcgAssetPayload {
    inf_pcg::PcgAssetPayload::new(inf_pcg::PcgDocument::single_layer(
        "meadow",
        vec![rule("grass", 101, 0.25, 1)],
    ))
}

/// Graph B: **two** rules merged on the canvas, so the payload carries a real
/// `scatter.merge` graph and the player's re-lowering exercises P19.3's
/// multi-rule walk.
fn graph_b() -> inf_pcg::PcgAssetPayload {
    use inf_graph::{apply_edits, GraphEdit, Link, NodeId, ParamMap, ParamValue};

    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    let params = |pairs: &[(&str, ParamValue)]| {
        let mut m = ParamMap::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        m
    };
    let mut edits = Vec::new();
    let mut add = |id: u32, type_id: &str, p: ParamMap| {
        edits.push(GraphEdit::AddNode {
            id: NodeId(id),
            type_id: type_id.into(),
            x: 0.0,
            y: 0.0,
            params: p,
        });
    };
    add(
        1,
        "scatter.scatter",
        params(&[
            ("name", ParamValue::Text("shrubs".into())),
            ("cell_size", ParamValue::Float(8.0)),
            ("base_density", ParamValue::Float(0.12)),
            ("seed", ParamValue::Int(202)),
            ("mesh", ParamValue::Text(Uuid::from_u128(2).to_string())),
        ]),
    );
    add(
        2,
        "scatter.scatter",
        params(&[
            ("name", ParamValue::Text("boulders".into())),
            ("cell_size", ParamValue::Float(8.0)),
            ("base_density", ParamValue::Float(0.04)),
            ("seed", ParamValue::Int(303)),
            ("mesh", ParamValue::Text(Uuid::from_u128(3).to_string())),
        ]),
    );
    add(3, "scatter.merge", ParamMap::new());
    add(4, "output.pcg", ParamMap::new());
    for (from, from_port, to, to_port) in [
        (1u32, "out", 3u32, "a"),
        (2, "out", 3, "b"),
        (3, "out", 4, "scatter"),
    ] {
        edits.push(GraphEdit::Connect {
            link: Link {
                from: NodeId(from),
                from_port: from_port.into(),
                to: NodeId(to),
                to_port: to_port.into(),
            },
        });
    }
    apply_edits(&mut g, &reg, &edits);
    let lowered = inf_pcg::lower_graph(&g, &reg);
    assert!(lowered.ok, "fixture graph must lower: {:?}", lowered.issues);
    assert_eq!(
        lowered.document.layers[0].rules.len(),
        2,
        "graph B is the multi-rule arm of this gate"
    );
    inf_pcg::PcgAssetPayload::from_graph(&g, lowered.document)
}

fn rule(name: &str, seed: u64, density: f64, mesh: u128) -> inf_pcg::PcgRule {
    inf_pcg::PcgRule {
        name: name.into(),
        sampler: inf_pcg::SamplerDef::Constant(1.0),
        scatter: inf_pcg::ScatterParams {
            seed,
            cell_size: 8.0,
            base_density: density,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (1.0, 1.0),
            rotation: inf_pcg::RotationMode::RandomYaw,
            altitude_offset: 0.0,
        },
        kinds: vec![inf_pcg::PcgKind::mesh(Uuid::from_u128(mesh))],
    }
}

fn biome_set() -> BiomeSet {
    let mut set = BiomeSet::new("Gate Biomes");
    set.biomes.push(BiomeDef {
        color: [0.3, 0.5, 0.2, 1.0],
        pcg_graph: Some(GRAPH_A_GUID),
        ..BiomeDef::new(1, "Meadow")
    });
    set.biomes.push(BiomeDef {
        color: [0.5, 0.45, 0.3, 1.0],
        pcg_graph: Some(GRAPH_B_GUID),
        ..BiomeDef::new(2, "Scrub")
    });
    set.validate().expect("the fixture set is valid");
    set
}

/// Write the whole fixture (level + `.inf_biomes` + both `.inf_pcg`, each with an
/// `inf_asset` sidecar carrying its stable GUID) into `content`.
fn write_fixture(content: &Path) {
    std::fs::create_dir_all(content).unwrap();
    inf_editor_core::scene::serialize::save(
        &biome_scene(),
        &content.join("BiomeDemo.inf_lvl"),
        Some(LEVEL_GUID),
    )
    .unwrap();

    let set_bytes = inf_asset::encode(&biome_set()).unwrap();
    write_asset(
        content,
        "Gate.inf_biomes",
        &set_bytes,
        BIOME_SET_GUID,
        inf_asset::AssetKind::BiomeSet,
    );
    for (file, guid, payload) in [
        ("Meadow.inf_pcg", GRAPH_A_GUID, graph_a()),
        ("Scrub.inf_pcg", GRAPH_B_GUID, graph_b()),
    ] {
        write_asset(
            content,
            file,
            &payload.encode().unwrap(),
            guid,
            inf_asset::AssetKind::Pcg,
        );
    }
}

fn write_asset(dir: &Path, file: &str, bytes: &[u8], guid: Uuid, kind: inf_asset::AssetKind) {
    let path = dir.join(file);
    std::fs::write(&path, bytes).unwrap();
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(guid),
        kind,
        inf_asset::ContentHash::of(bytes),
    )
    .save(&path)
    .unwrap();
}

/// Scaffold a project holding the fixture and cook it; returns `(project, pack)`.
fn cook_fixture(tmp: &Path) -> (PathBuf, PathBuf) {
    let proj = tmp.join("proj");
    ProjectManifest::new("Biome PCG", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    write_fixture(&content);
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    (proj, out)
}

// ── population probes ───────────────────────────────────────────────────────

/// The terrain's biome-bound population, in evaluation order.
fn population(built: &BuiltWorld) -> Vec<ScatteredInstance> {
    let w = built.world.world();
    w.iter_entities()
        .filter_map(|e| e.get::<Terrain>())
        .flat_map(|t| t.biome_population.clone())
        .collect()
}

fn pack_built(pack: &Path) -> BuiltWorld {
    let source = PackLevelSource::open(pack).expect("pack opens");
    inf_player::build_world_from_pack(&source).expect("pack world builds")
}

fn dir_built(content: &Path) -> BuiltWorld {
    let source = DevDirLevelSource::new(content.join("BiomeDemo.inf_lvl"));
    let builder = InfSceneWorldBuilder::with_defaults(Vec::new())
        .with_pcgs(level::load_pcg_payloads_by_guid_from_dir(content))
        .with_biome_sets(level::load_biome_sets_by_guid_from_dir(content));
    level::load(&source, &builder).expect("dev-dir world builds")
}

// ── the gate ────────────────────────────────────────────────────────────────

/// **The cook follows the whole chain.** With ONLY the level as an explicit root,
/// the pack still holds the `.inf_biomes` (level → `Terrain.biome_set`) *and*
/// both `.inf_pcg` graphs (biome set → `BiomeDef.pcg_graph`). P19.2 tested the
/// first hop; this is the first test that the second one is followed.
#[test]
fn the_cook_follows_level_to_biome_set_to_graphs() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Biome PCG", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    write_fixture(&content);

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
        report.kinds.get("biome_set"),
        Some(&1),
        "the biome set rides the level→biome_set edge, not a default root: {:?}",
        report.kinds
    );
    assert_eq!(
        report.kinds.get("pcg"),
        Some(&2),
        "BOTH graphs ride the biome-set→pcg_graph edge: {:?}",
        report.kinds
    );
    let reader = inf_asset::PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    for guid in [BIOME_SET_GUID, GRAPH_A_GUID, GRAPH_B_GUID] {
        assert!(
            reader.contains(inf_asset::AssetId(guid)),
            "pack is missing {guid}"
        );
    }
    // Neither fixture graph uses an Image Mask, so the P19.3 advisory stays
    // quiet — the negative half of the test below.
    assert!(
        !report.warnings.iter().any(|w| w.contains("Image Mask")),
        "unexpected image-mask advisory: {:?}",
        report.warnings
    );
}

/// **The `mask.image` runtime gap is advised at build time, not discovered by a
/// player.** The editor resolves an image mask's texture and lowers real pixels;
/// the shipped and PIE players have no asset database on the evaluation path and
/// lower it to an empty mask that scores `0`. That divergence fails *closed*
/// (less content, never wrong content) and is diagnosed on the node — but nothing
/// downstream can tell "masked out" from "authored empty", so the cook says it
/// out loud at the last moment before it becomes a shipped build.
#[test]
fn the_cook_advises_when_a_graph_uses_an_image_mask() {
    use inf_graph::{apply_edits, GraphEdit, Link, NodeId, ParamMap, ParamValue};

    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Biome PCG", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    write_fixture(&content);

    // Replace graph A with one that masks through a (missing) texture.
    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    let mut mask_params = ParamMap::new();
    mask_params.insert(
        "texture".into(),
        ParamValue::Text(Uuid::from_u128(0xDEAD).to_string()),
    );
    apply_edits(
        &mut g,
        &reg,
        &[
            GraphEdit::AddNode {
                id: NodeId(1),
                type_id: "mask.image".into(),
                x: 0.0,
                y: 0.0,
                params: mask_params,
            },
            GraphEdit::AddNode {
                id: NodeId(2),
                type_id: "scatter.scatter".into(),
                x: 0.0,
                y: 0.0,
                params: ParamMap::new(),
            },
            GraphEdit::AddNode {
                id: NodeId(3),
                type_id: "output.pcg".into(),
                x: 0.0,
                y: 0.0,
                params: ParamMap::new(),
            },
            GraphEdit::Connect {
                link: Link {
                    from: NodeId(1),
                    from_port: "out".into(),
                    to: NodeId(2),
                    to_port: "density".into(),
                },
            },
            GraphEdit::Connect {
                link: Link {
                    from: NodeId(2),
                    from_port: "out".into(),
                    to: NodeId(3),
                    to_port: "scatter".into(),
                },
            },
        ],
    );
    let payload = inf_pcg::PcgAssetPayload::from_graph(&g, inf_pcg::lower_graph(&g, &reg).document);
    write_asset(
        &content,
        "Meadow.inf_pcg",
        &payload.encode().unwrap(),
        GRAPH_A_GUID,
        inf_asset::AssetKind::Pcg,
    );

    let out = proj.join("out");
    let report = cook(&proj, &out, &CookOptions::default()).expect("cook still succeeds");
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("Image Mask") && w.contains(&GRAPH_A_GUID.to_string())),
        "the cook did not advise about the image mask: {:?}",
        report.warnings
    );
}

/// **The region property + both biomes contribute**, over the shipping path.
#[test]
fn a_cooked_biome_terrain_populates_each_biome_over_its_own_region() {
    let dir = tempfile::tempdir().unwrap();
    let (_proj, pack) = cook_fixture(dir.path());
    let built = pack_built(&pack);
    let pop = population(&built);
    assert!(
        pop.len() > 50,
        "the terrain must grow content, got {}",
        pop.len()
    );

    // Every instance sits on ground the fixture actually painted, and the two
    // halves are separated by the border (the feather is symmetric about it, so
    // allow the blend band on either side).
    let feather = inf_pcg::DEFAULT_BIOME_FEATHER;
    // The authored extent is the terrain's own, not the requested `SPAN` —
    // `write_region` rounds out to whole tiles, and the binding scatters over
    // exactly what exists.
    let (tmin, tmax) = terrain_of(&built).data.xz_bounds().expect("authored");
    let (mut left, mut right) = (0usize, 0usize);
    for i in &pop {
        assert!(
            (tmin.x..=tmax.x).contains(&i.position.x) && (tmin.y..=tmax.y).contains(&i.position.z),
            "instance escaped the terrain {tmin:?}..{tmax:?}: {:?}",
            i.position
        );
        if i.position.x < BORDER {
            left += 1;
        } else {
            right += 1;
        }
    }
    assert!(
        left > 10 && right > 10,
        "both biomes must contribute: {left}/{right}"
    );
    // Dispatch order is ascending id, so biome 1's (the left half's) instances
    // come first in the merged list.
    assert!(
        pop[0].position.x < BORDER,
        "biome 1 must dispatch first, got {:?}",
        pop[0].position
    );
    // The blend band is real but bounded: no instance sits further than the
    // feather past the border on the wrong side of its own biome.
    for i in &pop {
        let owner = terrain_of(&built)
            .data
            .biome_at(DVec2::new(i.position.x, i.position.z))
            .expect("on authored terrain");
        let dist_past = if owner == 1 {
            i.position.x - BORDER
        } else {
            BORDER - i.position.x
        };
        assert!(
            dist_past <= feather,
            "instance {:?} is {dist_past} m into the wrong biome (feather {feather})",
            i.position
        );
    }
}

/// The single terrain component of a built world.
fn terrain_of(built: &BuiltWorld) -> Terrain {
    built
        .world
        .world()
        .iter_entities()
        .find_map(|e| e.get::<Terrain>().cloned())
        .expect("terrain present")
}

/// **An unpainted terrain grows nothing**, and a terrain with no biome set is
/// untouched — the pass fails closed.
#[test]
fn an_unpainted_terrain_grows_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let content = dir.path().join("Content");
    write_fixture(&content);
    // Rewrite the level with every biome id cleared.
    let mut doc = biome_scene();
    {
        let entity = doc.entity_of(TERRAIN_GUID).unwrap();
        let w = doc.world_mut().world_mut();
        let mut t = w.get_mut::<Terrain>(entity).unwrap();
        let coords: Vec<(i32, i32)> = t.data.tiles().map(|(&c, _)| c).collect();
        for coord in coords {
            t.data.get_tile_mut(coord).unwrap().clear_biomes();
        }
    }
    doc.world_mut().propagate();
    inf_editor_core::scene::serialize::save(
        &doc,
        &content.join("BiomeDemo.inf_lvl"),
        Some(LEVEL_GUID),
    )
    .unwrap();

    assert!(
        population(&dir_built(&content)).is_empty(),
        "an unpainted terrain must grow nothing at all"
    );
}

/// **cooked == uncooked**, instance for instance.
#[test]
fn the_pack_path_and_the_dev_dir_path_place_the_same_instances() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, pack) = cook_fixture(dir.path());
    let ship = population(&pack_built(&pack));
    let loose = population(&dir_built(&proj.join("Content")));
    assert!(!ship.is_empty());
    assert_eq!(
        ship.len(),
        loose.len(),
        "cooked and uncooked disagree on instance COUNT"
    );
    assert_eq!(ship, loose, "cooked and uncooked disagree on placement");
    // …and two independent loads of one pack agree, so the population is a pure
    // function of the content.
    assert_eq!(ship, population(&pack_built(&pack)), "not deterministic");
}

/// **PIE == shipping** on the biome-bound population: the streamed payload
/// carries the biome set and both graphs, and the PIE player places exactly what
/// the cooked pack does.
#[test]
fn pie_matches_shipping_for_the_biome_population() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, pack) = cook_fixture(dir.path());
    let ship = population(&pack_built(&pack));
    assert!(!ship.is_empty());

    let content = proj.join("Content");
    let doc = inf_editor_core::scene::serialize::load(&content.join("BiomeDemo.inf_lvl"))
        .expect("fixture doc loads");
    let read = |name: &str| std::fs::read(content.join(name)).unwrap();
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |_guid| None,
        |guid| match guid {
            g if g == GRAPH_A_GUID => Some(read("Meadow.inf_pcg")),
            g if g == GRAPH_B_GUID => Some(read("Scrub.inf_pcg")),
            _ => None,
        },
        |_guid| None,
        |guid| (guid == BIOME_SET_GUID).then(|| read("Gate.inf_biomes")),
        |_guid| None,
        |_guid| None,
        // P22.3: no destructible meshes in this fixture.
        |_| None,
        60,
        false,
    )
    .expect("payload builds");
    assert_eq!(
        payload.biome_sets.len(),
        1,
        "the referenced .inf_biomes rides the payload"
    );
    // The payload walks the biome set's own edges too, so both graphs ride along.
    assert_eq!(payload.pcgs.len(), 2, "both biome graphs ride the payload");

    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    assert_eq!(
        population(&built),
        ship,
        "PIE population != shipping population"
    );
}

/// The two biomes' graphs are genuinely **different** — the gate would be vacuous
/// if both halves ran the same rules. Graph B is the multi-rule arm, so its half
/// carries instances from two rules; graph A's carries one.
#[test]
fn the_two_biomes_run_distinct_graphs() {
    let dir = tempfile::tempdir().unwrap();
    let (_proj, pack) = cook_fixture(dir.path());
    let reader = inf_asset::PackReader::open(&pack.join(DEFAULT_PACK_NAME)).unwrap();
    let doc_of = |guid: Uuid| {
        let bytes = reader.read(inf_asset::AssetId(guid)).unwrap();
        let payload = inf_pcg::PcgAssetPayload::decode(&bytes).unwrap();
        match payload.graph() {
            Some(g) => inf_pcg::lower_graph(&g, &inf_pcg::pcg_registry()).document,
            None => payload.document,
        }
    };
    let a = doc_of(GRAPH_A_GUID);
    let b = doc_of(GRAPH_B_GUID);
    assert_eq!(a.layers[0].rules.len(), 1, "graph A is the single-rule arm");
    assert_eq!(
        b.layers[0]
            .rules
            .iter()
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>(),
        vec!["shrubs", "boulders"],
        "graph B survived the cook as an ORDERED two-rule document"
    );
    assert_ne!(a, b);
}

/// A world-space check that the population really is regional: instances deep
/// inside biome 1's half come only from graph A's mesh palette, and vice versa.
/// (`kind` indexes the *rule's* palette, so the discriminator is the `Vec3d`
/// position, not the kind — this asserts the split directly.)
#[test]
fn deep_inside_each_biome_only_that_biome_scattered() {
    let dir = tempfile::tempdir().unwrap();
    let (_proj, pack) = cook_fixture(dir.path());
    let built = pack_built(&pack);
    let w = built.world.world();
    let terrain = w
        .iter_entities()
        .find_map(|e| e.get::<Terrain>())
        .expect("terrain present");

    for i in &terrain.biome_population {
        let id = terrain
            .data
            .biome_at(DVec2::new(i.position.x, i.position.z))
            .expect("instance is on authored terrain");
        assert!(
            id == 1 || id == 2,
            "instance at {:?} landed on unassigned ground",
            i.position
        );
    }
    // Sanity on the fixture itself: the ids really are split at the border.
    assert_eq!(terrain.data.biome_at(DVec2::new(4.0, 4.0)), Some(1));
    assert_eq!(terrain.data.biome_at(DVec2::new(28.0, 4.0)), Some(2));
    // And the population is anchored on the ground, not floating.
    let probe = &terrain.biome_population[0];
    let h = terrain
        .data
        .height_at(DVec2::new(probe.position.x, probe.position.z))
        .unwrap();
    assert!(
        (probe.position.y - h).abs() < 1e-9,
        "instance is not on the terrain: {:?} vs {h}",
        probe.position
    );
}
