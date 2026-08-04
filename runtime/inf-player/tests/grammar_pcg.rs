//! **P19.4 gate: a grammar builds the same structure everywhere.**
//!
//! P19.3 made a painted terrain *grow* things. P19.4 makes an authored line
//! *build* one — a fence along a spline, a wall around a footprint — and the
//! failure mode is worse than scatter's: a wall assembled from a different
//! derivation is not "slightly different foliage", it is a building whose pieces
//! are somewhere else.
//!
//! The fixture is a `PcgVolume` whose `.inf_pcg` carries **three passes on one
//! canvas**: a scatter rule, a spline grammar following a real `Spline` entity,
//! and a footprint grammar around the volume's own box — merged through
//! `scatter.merge`, so P19.3's multi-pass lowering carries a P19.4 generator and
//! the mix rides the shipping pipeline rather than only its unit tests.
//!
//! What this proves:
//!
//! * **the cook follows the new chain** — level → `PcgVolume.graph` → `.inf_pcg`
//!   → each grammar module's `.inf_mesh`, with only the level as an explicit
//!   root (the `.inf_pcg` hop existed since P10.6; the module hop is new);
//! * **a dangling module is advised at build time**, not discovered by a player
//!   walking into a wall with a hole in it;
//! * **cooked == uncooked** — pack and dev-dir place the same instances, in the
//!   same order, at the same positions;
//! * **PIE == shipping** — the streamed payload carries the graph and the player
//!   builds the identical structure;
//! * **determinism** — two loads of one pack agree, and the shipped content's
//!   own passes are invariant under worker-pool size through the real
//!   `evaluate_grammars_in` seam;
//! * **the structure is real** — instances follow the spline, ring the
//!   footprint, sit on the terrain, and carry palette-indexed kinds.
//!
//! The fixture is built here rather than committed, for the same reason
//! `biome_pcg.rs` builds its own: it exists to pin the *chain*, and a temp-dir
//! fixture cannot drift from a `.inf_lvl` nobody regenerated.

use std::path::{Path, PathBuf};

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{PcgVolume, Spline, SplineInterp, Terrain, Transform};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::ScatteredInstance;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_graph::{apply_edits, GraphEdit, Link, NodeId, ParamMap, ParamValue};
use inf_packager::{cook, CookOptions, DEFAULT_PACK_NAME};
use inf_player::level::{
    self, BuiltWorld, DevDirLevelSource, InfSceneWorldBuilder, PackLevelSource,
};
use inf_project::ProjectManifest;

// ── fixture constants ───────────────────────────────────────────────────────

const LEVEL_GUID: Uuid = Uuid::from_u128(0x9419_0400);
const TERRAIN_GUID: Uuid = Uuid::from_u128(0x9419_0401);
const SPLINE_GUID: Uuid = Uuid::from_u128(0x9419_0402);
const VOLUME_GUID: Uuid = Uuid::from_u128(0x9419_0403);
const GRAPH_GUID: Uuid = Uuid::from_u128(0x9419_0501);
/// Two modules name meshes that really exist in the project …
const POST_MESH: Uuid = Uuid::from_u128(0x9419_0601);
const PANEL_MESH: Uuid = Uuid::from_u128(0x9419_0602);
/// … and one names a mesh nobody ever imported — the advisory's subject.
const GHOST_MESH: Uuid = Uuid::from_u128(0x9419_06FF);

const RESOLUTION: u32 = 17;
const MPS: f64 = 1.0;
/// 2 × 2 tiles of `(17 − 1) · 1 m` → a 32 m square.
const SPAN: f64 = 32.0;

/// The volume sits at the terrain's centre and covers all of it.
const CENTER: DVec3 = DVec3::new(16.0, 0.0, 16.0);
const EXTENT: f64 = 15.0;

/// The spline runs a straight 24 m along +X, two metres in from the near edge.
const SPLINE_START: DVec3 = DVec3::new(4.0, 0.0, 2.0);
const SPLINE_END: DVec3 = DVec3::new(28.0, 0.0, 2.0);

/// The authored rule text. Deliberately exercises every DSL feature the gate
/// cares about: a palette with real and dangling meshes, a fill repetition, a
/// weighted alternation, a flexible-size gap, and a corner module.
fn rule_text() -> String {
    format!(
        "# P19.4 gate fixture\n\
         module Post   = mesh {POST_MESH} size 0.5\n\
         module Panel  = mesh {PANEL_MESH} size 2\n\
         module Ghost  = mesh {GHOST_MESH} size 2\n\
         \n\
         Fence -> Post Bay* Post\n\
         Bay   -> Panel | Ghost@0.5\n"
    )
}

// ── fixture construction ────────────────────────────────────────────────────

/// The scene: a terrain, a spline, and a PCG volume bound to the grammar graph.
fn grammar_scene() -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.set_title("Grammar PCG");

    // Terrain — a gentle polynomial hill. No `std` trig, so the fixture is
    // bit-portable across platforms (the P14 law).
    doc.create_with_guid(TERRAIN_GUID, SpawnKind::Empty, "Terrain", None);
    let te = doc.entity_of(TERRAIN_GUID).expect("terrain entity");
    let mut terrain = Terrain::configured(RESOLUTION, MPS);
    terrain
        .data
        .write_region(DVec2::ZERO, DVec2::splat(SPAN), |x, z| {
            0.01 * x * z - 0.02 * x + 1.0
        });
    {
        let w = doc.world_mut().world_mut();
        w.entity_mut(te)
            .insert(Transform::from_translation(DVec3::ZERO));
        w.entity_mut(te).insert(terrain);
    }

    // The spline the fence follows.
    doc.create_with_guid(SPLINE_GUID, SpawnKind::Empty, "Road", None);
    let se = doc.entity_of(SPLINE_GUID).expect("spline entity");
    {
        let w = doc.world_mut().world_mut();
        w.entity_mut(se)
            .insert(Transform::from_translation(DVec3::ZERO));
        w.entity_mut(se).insert(Spline {
            points: vec![
                Vec3d::new(SPLINE_START.x, SPLINE_START.y, SPLINE_START.z),
                Vec3d::new(SPLINE_END.x, SPLINE_END.y, SPLINE_END.z),
            ],
            closed: false,
            interp: SplineInterp::Linear,
        });
    }

    // The volume that evaluates the graph.
    doc.create_with_guid(VOLUME_GUID, SpawnKind::Empty, "Village", None);
    let ve = doc.entity_of(VOLUME_GUID).expect("volume entity");
    {
        let w = doc.world_mut().world_mut();
        w.entity_mut(ve).insert(Transform::from_translation(CENTER));
        w.entity_mut(ve).insert(PcgVolume {
            graph: Some(GRAPH_GUID),
            extent: Vec2d::new(EXTENT, EXTENT),
            seed: 17,
            ..PcgVolume::default()
        });
    }

    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The authored graph: one scatter rule and **two** grammar passes, merged.
fn grammar_graph() -> inf_graph::Graph {
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

    // 1: the shared rule text. 2: the spline span. 3: the fence expander.
    add(
        1,
        "grammar.rules",
        params(&[("rules", ParamValue::Text(rule_text()))]),
    );
    add(
        2,
        "grammar.spline",
        params(&[
            ("spline", ParamValue::Text(SPLINE_GUID.to_string())),
            ("samples_per_segment", ParamValue::Int(8)),
        ]),
    );
    add(
        3,
        "grammar.expand",
        params(&[
            ("name", ParamValue::Text("fence".into())),
            ("axiom", ParamValue::Text("Fence".into())),
            ("seed", ParamValue::Int(1001)),
        ]),
    );
    // 4: the footprint span (perimeter, corners). 5: the wall expander.
    add(
        4,
        "grammar.footprint",
        params(&[
            ("mode", ParamValue::Enum("Perimeter".into())),
            ("corner", ParamValue::Text("Post".into())),
            ("corner_size", ParamValue::Float(0.5)),
        ]),
    );
    add(
        5,
        "grammar.expand",
        params(&[
            ("name", ParamValue::Text("walls".into())),
            ("axiom", ParamValue::Text("Fence".into())),
            ("seed", ParamValue::Int(2002)),
        ]),
    );
    // 6/7: a plain scatter rule, so a document and a grammar share one volume.
    add(
        6,
        "const.density",
        params(&[("value", ParamValue::Float(0.4))]),
    );
    add(
        7,
        "scatter.scatter",
        params(&[
            ("name", ParamValue::Text("grass".into())),
            ("cell_size", ParamValue::Float(8.0)),
            ("base_density", ParamValue::Float(0.05)),
            ("seed", ParamValue::Int(303)),
        ]),
    );
    // 8/9: merges. 10: the sink.
    add(8, "scatter.merge", ParamMap::new());
    add(9, "scatter.merge", ParamMap::new());
    add(10, "output.pcg", ParamMap::new());

    for (from, from_port, to, to_port) in [
        (1u32, "out", 3u32, "rules"),
        (2, "out", 3, "span"),
        (1, "out", 5, "rules"),
        (4, "out", 5, "span"),
        (6, "out", 7, "density"),
        (7, "out", 8, "a"),
        (3, "out", 8, "b"),
        (8, "out", 9, "a"),
        (5, "out", 9, "b"),
        (9, "out", 10, "scatter"),
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
    g
}

fn grammar_payload() -> inf_pcg::PcgAssetPayload {
    let g = grammar_graph();
    let lowered = inf_pcg::lower_graph(&g, &inf_pcg::pcg_registry());
    assert!(
        lowered.ok,
        "the fixture graph must lower cleanly: {:?}",
        lowered.issues
    );
    assert_eq!(lowered.grammars.len(), 2, "two grammar passes");
    assert_eq!(
        lowered.document.layers[0].rules.len(),
        1,
        "one scatter rule beside them"
    );
    inf_pcg::PcgAssetPayload::from_graph(&g, lowered.document)
}

/// A trivial two-triangle mesh — the gate cares that the asset *exists* and is
/// reachable through the new cook edge, not what it looks like.
fn tiny_mesh(name: &str) -> inf_mesh::MeshAsset {
    let v = |x: f32, z: f32| inf_mesh::MeshVertex {
        position: [x, 0.0, z],
        normal: [0.0, 1.0, 0.0],
        ..Default::default()
    };
    inf_mesh::MeshAsset::new(
        vec![inf_mesh::SubMesh {
            name: name.into(),
            vertices: vec![v(-0.5, -0.5), v(0.5, -0.5), v(-0.5, 0.5), v(0.5, 0.5)],
            indices: vec![0, 1, 2, 2, 1, 3],
            material_slot: None,
            skin: Vec::new(),
        }],
        Vec::new(),
    )
}

/// Write the whole fixture (level + `.inf_pcg` + the two real `.inf_mesh`
/// modules, each with an `inf_asset` sidecar carrying its stable GUID).
fn write_fixture(content: &Path) {
    std::fs::create_dir_all(content).unwrap();
    inf_editor_core::scene::serialize::save(
        &grammar_scene(),
        &content.join("GrammarDemo.inf_lvl"),
        Some(LEVEL_GUID),
    )
    .unwrap();
    write_asset(
        content,
        "Village.inf_pcg",
        &grammar_payload().encode().unwrap(),
        GRAPH_GUID,
        inf_asset::AssetKind::Pcg,
    );
    for (file, guid, name) in [
        ("Post.inf_mesh", POST_MESH, "post"),
        ("Panel.inf_mesh", PANEL_MESH, "panel"),
    ] {
        write_asset(
            content,
            file,
            &inf_asset::encode(&tiny_mesh(name)).unwrap(),
            guid,
            inf_asset::AssetKind::Mesh,
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
    ProjectManifest::new("Grammar PCG", "blank-3d")
        .save(&proj)
        .unwrap();
    write_fixture(&proj.join("Content"));
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    (proj, out)
}

// ── population probes ───────────────────────────────────────────────────────

/// The volume's evaluated cache, in evaluation order.
fn population(built: &BuiltWorld) -> Vec<ScatteredInstance> {
    let w = built.world.world();
    w.iter_entities()
        .filter_map(|e| e.get::<PcgVolume>())
        .flat_map(|v| v.evaluated.clone())
        .collect()
}

/// Every instance's placement as raw bits — position, rotation and scale.
///
/// The parity arms compare THIS, not the derived `PartialEq`: a low-bit
/// divergence between two hosts is exactly the failure a `f64 == f64` comparison
/// would still catch today and a future `impl PartialEq` with a tolerance would
/// not, and "the shipped wall is 3 nm from the preview's" is the shape of bug
/// that gets found by a player.
fn bits(v: &[ScatteredInstance]) -> Vec<[u64; 8]> {
    v.iter()
        .map(|i| {
            let r = i.rotation.to_array();
            [
                i.position.x.to_bits(),
                i.position.y.to_bits(),
                i.position.z.to_bits(),
                r[0].to_bits(),
                r[1].to_bits(),
                r[2].to_bits(),
                r[3].to_bits(),
                i.scale.to_bits(),
            ]
        })
        .collect()
}

fn pack_built(pack: &Path) -> BuiltWorld {
    let source = PackLevelSource::open(pack).expect("pack opens");
    inf_player::build_world_from_pack(&source).expect("pack world builds")
}

fn dir_built(content: &Path) -> BuiltWorld {
    let source = DevDirLevelSource::new(content.join("GrammarDemo.inf_lvl"));
    let builder = InfSceneWorldBuilder::with_defaults(Vec::new())
        .with_pcgs(level::load_pcg_payloads_by_guid_from_dir(content));
    level::load(&source, &builder).expect("dev-dir world builds")
}

// ── the gate ────────────────────────────────────────────────────────────────

/// **The cook follows the whole chain.** With ONLY the level as an explicit
/// root, the pack holds the `.inf_pcg` (level → `PcgVolume.graph`, since P10.6)
/// *and* both grammar-module `.inf_mesh` assets (`.inf_pcg` → module mesh, the
/// P19.4 edge). Without the new edge the pack would ship a graph that expands to
/// a derivation naming meshes it does not contain.
#[test]
fn the_cook_follows_level_to_graph_to_module_meshes() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Grammar PCG", "blank-3d")
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
        report.kinds.get("pcg"),
        Some(&1),
        "the graph rides the level→PcgVolume.graph edge: {:?}",
        report.kinds
    );
    assert_eq!(
        report.kinds.get("mesh"),
        Some(&2),
        "BOTH module meshes ride the new .inf_pcg→module edge: {:?}",
        report.kinds
    );
    let reader = inf_asset::PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    for guid in [GRAPH_GUID, POST_MESH, PANEL_MESH] {
        assert!(
            reader.contains(inf_asset::AssetId(guid)),
            "pack is missing {guid}"
        );
    }
}

/// **A dangling module is advised at build time.** A grammar fails *quietly* —
/// the derivation runs, the slot consumes its span, and the wall simply has a
/// piece missing — so the cook says it out loud at the last moment before it
/// becomes a shipped build, naming both the graph and the mesh.
#[test]
fn the_cook_advises_about_a_dangling_grammar_module() {
    let dir = tempfile::tempdir().unwrap();
    let (_proj, out) = cook_fixture(dir.path());
    let report = {
        // Re-cook to read the report (cook_fixture discards it).
        let proj = dir.path().join("proj");
        cook(&proj, &out, &CookOptions::default()).expect("cook succeeds")
    };
    let advisory = report
        .warnings
        .iter()
        .find(|w| w.contains("grammar module"))
        .unwrap_or_else(|| panic!("no grammar-module advisory: {:?}", report.warnings));
    assert!(advisory.contains(&GHOST_MESH.to_string()), "{advisory}");
    assert!(advisory.contains(&GRAPH_GUID.to_string()), "{advisory}");
    // …and it names ONLY the ghost: the two real modules resolve.
    for real in [POST_MESH, PANEL_MESH] {
        assert!(
            !report
                .warnings
                .iter()
                .any(|w| w.contains("grammar module") && w.contains(&real.to_string())),
            "{real} is in the project and must not be advised"
        );
    }
    // Deduplicated: the ghost is named once, however many passes reference the
    // palette that declares it (this fixture has two).
    assert_eq!(
        report
            .warnings
            .iter()
            .filter(|w| w.contains("grammar module"))
            .count(),
        1,
        "the advisory is not deduplicated: {:?}",
        report.warnings
    );
}

/// **The negative control.** A project whose `.inf_pcg` carries no grammar at
/// all draws no grammar-module advisory — so the warning above is evidence of
/// something, rather than noise every cook emits. (The biome gate's own
/// image-mask advisory is pinned the same way.)
#[test]
fn a_graph_with_no_grammar_draws_no_module_advisory() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Grammar PCG", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    write_fixture(&content);
    // Replace the graph with a scatter-only one.
    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    apply_edits(
        &mut g,
        &reg,
        &[
            GraphEdit::AddNode {
                id: NodeId(1),
                type_id: "scatter.scatter".into(),
                x: 0.0,
                y: 0.0,
                params: ParamMap::new(),
            },
            GraphEdit::AddNode {
                id: NodeId(2),
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
                    to_port: "scatter".into(),
                },
            },
        ],
    );
    let payload = inf_pcg::PcgAssetPayload::from_graph(&g, inf_pcg::lower_graph(&g, &reg).document);
    write_asset(
        &content,
        "Village.inf_pcg",
        &payload.encode().unwrap(),
        GRAPH_GUID,
        inf_asset::AssetKind::Pcg,
    );

    let report =
        cook(&proj, &dir.path().join("out"), &CookOptions::default()).expect("cook still succeeds");
    assert!(
        !report.warnings.iter().any(|w| w.contains("grammar module")),
        "a grammar-free graph must draw no module advisory: {:?}",
        report.warnings
    );
}

/// **The cook's module edge does not depend on the graph LOWERING.**
///
/// A Span pin nobody has dragged yet is an ordinary mid-authoring state: the
/// graph does not lower, no pass exists — and the palette still plainly names
/// its meshes. If the edge were taken from the lowered passes, cooking that
/// project would ship a wall with pieces missing *and* stay silent about the
/// dangling one.
#[test]
fn an_unlowerable_graph_still_contributes_its_module_edges() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Grammar PCG", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    write_fixture(&content);

    // The same palette, but the expander's Span pin is left unconnected.
    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    let mut rules_params = ParamMap::new();
    rules_params.insert("rules".into(), ParamValue::Text(rule_text()));
    apply_edits(
        &mut g,
        &reg,
        &[
            GraphEdit::AddNode {
                id: NodeId(1),
                type_id: "grammar.rules".into(),
                x: 0.0,
                y: 0.0,
                params: rules_params,
            },
            GraphEdit::AddNode {
                id: NodeId(2),
                type_id: "grammar.expand".into(),
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
                    to_port: "rules".into(),
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
    let lowered = inf_pcg::lower_graph(&g, &reg);
    assert!(
        !lowered.ok,
        "the fixture must be a graph that does NOT lower"
    );
    assert!(lowered.grammars.is_empty());
    let payload = inf_pcg::PcgAssetPayload::from_graph(&g, lowered.document);
    write_asset(
        &content,
        "Village.inf_pcg",
        &payload.encode().unwrap(),
        GRAPH_GUID,
        inf_asset::AssetKind::Pcg,
    );

    let out = dir.path().join("out");
    let report = cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    let reader = inf_asset::PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    for guid in [POST_MESH, PANEL_MESH] {
        assert!(
            reader.contains(inf_asset::AssetId(guid)),
            "the module edge vanished with the wiring: {guid} is not in the pack"
        );
    }
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("grammar module") && w.contains(&GHOST_MESH.to_string())),
        "the dangling-module advisory vanished with the wiring: {:?}",
        report.warnings
    );
}

/// **The structure is real**, over the shipping path: the fence follows the
/// spline, the walls ring the footprint, everything sits on the terrain, and the
/// kinds index the authored palette.
#[test]
fn a_cooked_grammar_volume_builds_the_structure() {
    let dir = tempfile::tempdir().unwrap();
    let (_proj, pack) = cook_fixture(dir.path());
    let built = pack_built(&pack);
    let pop = population(&built);
    assert!(
        pop.len() > 30,
        "the volume must build something: {}",
        pop.len()
    );

    // Every kind is a palette index: 0 = Post, 1 = Panel, 2 = Ghost.
    assert!(pop.iter().all(|i| i.kind < 3), "kind out of palette range");
    assert!(pop.iter().any(|i| i.kind == 0), "no posts placed");
    assert!(pop.iter().any(|i| i.kind == 1), "no panels placed");

    // The fence pass ran along the spline: instances on the spline's own line
    // (z ≈ 2, x within its span) exist and are numerous enough to be the fence
    // rather than stray scatter.
    let on_spline = pop
        .iter()
        .filter(|i| {
            (i.position.z - SPLINE_START.z).abs() < 1e-6
                && i.position.x >= SPLINE_START.x - 1e-6
                && i.position.x <= SPLINE_END.x + 1e-6
        })
        .count();
    assert!(on_spline >= 10, "only {on_spline} instances on the spline");

    // The footprint pass ringed the volume: the perimeter of a 30 × 30 box
    // centred on CENTER, so instances sit on one of the four edges.
    let hx = EXTENT;
    let on_perimeter = pop
        .iter()
        .filter(|i| {
            let dx = (i.position.x - CENTER.x).abs();
            let dz = (i.position.z - CENTER.z).abs();
            ((dx - hx).abs() < 1e-6 && dz <= hx + 1e-6)
                || ((dz - hx).abs() < 1e-6 && dx <= hx + 1e-6)
        })
        .count();
    assert!(on_perimeter >= 20, "only {on_perimeter} on the perimeter");

    // Everything is ON the terrain, not floating: the fixture's height function
    // is exact and the grammar's default ground mode samples it.
    let terrain = built
        .world
        .world()
        .iter_entities()
        .find_map(|e| e.get::<Terrain>().cloned())
        .expect("terrain present");
    for i in &pop {
        if let Some(h) = terrain
            .data
            .height_at(DVec2::new(i.position.x, i.position.z))
        {
            assert!(
                (i.position.y - h).abs() < 1e-9,
                "instance is not on the terrain: {:?} vs {h}",
                i.position
            );
        }
    }
    // The scatter half is in the same cache and comes FIRST — a fixed order, so
    // the cache is a pure function of the content.
    let scatter_first = pop
        .iter()
        .position(|i| (i.position.z - SPLINE_START.z).abs() < 1e-6)
        .expect("some fence instance");
    assert!(
        scatter_first > 0,
        "the scatter rule contributed nothing, so the mix is untested"
    );
}

/// **cooked == uncooked**, instance for instance — and two loads of one pack
/// agree, so the population is a pure function of the content.
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
    assert_eq!(ship, population(&pack_built(&pack)), "not deterministic");
    // Bit identity, not just `PartialEq` — a future sloppy comparison cannot
    // hide a low-bit divergence in a position or a rotation.
    assert_eq!(bits(&ship), bits(&loose));
    assert_eq!(bits(&ship), bits(&population(&pack_built(&pack))));
}

/// **PIE == shipping** on the grammar population: the streamed payload carries
/// the graph and the PIE player builds exactly what the cooked pack does —
/// compared **bit for bit**, the same standard as the cooked-vs-uncooked arm
/// beside it. A structure whose pieces are a low bit apart on two hosts is a
/// wall that moves when you ship it.
#[test]
fn pie_matches_shipping_for_the_grammar_population() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, pack) = cook_fixture(dir.path());
    let ship = population(&pack_built(&pack));
    assert!(
        ship.len() > 30,
        "the parity arm must compare a real structure, got {}",
        ship.len()
    );

    let content = proj.join("Content");
    let doc = inf_editor_core::scene::serialize::load(&content.join("GrammarDemo.inf_lvl"))
        .expect("fixture doc loads");
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |_guid| None,
        |guid| {
            (guid == GRAPH_GUID).then(|| std::fs::read(content.join("Village.inf_pcg")).unwrap())
        },
        |_guid| None,
        |_guid| None,
        |_guid| None,
        60,
        false,
    )
    .expect("payload builds");
    assert_eq!(payload.pcgs.len(), 1, "the graph rides the payload");

    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    let pie = population(&built);
    assert_eq!(pie, ship, "PIE population != shipping population");
    assert_eq!(
        bits(&pie),
        bits(&ship),
        "PIE and shipping agree on placement but not on its BITS"
    );
}

/// **Pool-size invariance, through the SHIPPED content's own passes.** The
/// passes are read back out of the cooked pack and driven through
/// `evaluate_grammars_in` at 1/2/4/8 workers — so the span build, the
/// derivation, the layout, the corner stamping and the concatenation order all
/// participate, on the very bytes a player would load.
#[test]
fn the_shipped_passes_are_invariant_under_pool_size() {
    use inf_core::JobPool;

    let dir = tempfile::tempdir().unwrap();
    let (_proj, pack) = cook_fixture(dir.path());
    let reader = inf_asset::PackReader::open(&pack.join(DEFAULT_PACK_NAME)).unwrap();
    let bytes = reader.read(inf_asset::AssetId(GRAPH_GUID)).unwrap();
    let payload = inf_pcg::PcgAssetPayload::decode(&bytes).unwrap();
    let graph = payload.graph().expect("the cooked payload keeps its graph");
    let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
    assert_eq!(lowered.grammars.len(), 2);

    // The same spline the level carries, resolved the way both hosts resolve it.
    let built = pack_built(&pack);
    let splines = level::collect_spline_paths(&built.world);
    assert!(
        splines.contains_key(&SPLINE_GUID),
        "the spline survived the cook"
    );
    let height = inf_pcg::FnHeight::new(|_, _| Some(0.0));
    let cx = inf_pcg::GrammarContext {
        entity: Some(VOLUME_GUID),
        center: CENTER,
        extent: DVec2::splat(EXTENT),
        seed_offset: 17,
    };

    let runs: Vec<Vec<_>> = [1usize, 2, 4, 8]
        .into_iter()
        .map(|n| {
            inf_pcg::evaluate_grammars_in(
                &JobPool::new(n),
                &lowered.grammars,
                &splines,
                &height,
                &cx,
            )
            .instances
        })
        .collect();
    assert!(runs[0].len() > 30, "only {} instances", runs[0].len());
    for (n, run) in [1usize, 2, 4, 8].into_iter().zip(&runs).skip(1) {
        assert_eq!(run, &runs[0], "the structure differs on a {n}-worker pool");
    }
}

/// A volume whose graph carries **no** grammar is untouched by any of this — the
/// pass is additive, and a pre-P19.4 `.inf_pcg` still evaluates to exactly its
/// scatter.
#[test]
fn a_graph_without_a_grammar_still_evaluates_to_its_scatter_alone() {
    let dir = tempfile::tempdir().unwrap();
    let content = dir.path().join("Content");
    write_fixture(&content);
    // Replace the graph with a bare document-only (v1-shaped) payload.
    let doc = inf_pcg::PcgDocument::single_layer(
        "layer",
        vec![inf_pcg::PcgRule {
            name: "grass".into(),
            sampler: inf_pcg::SamplerDef::Constant(1.0),
            scatter: inf_pcg::ScatterParams {
                seed: 5,
                cell_size: 8.0,
                base_density: 0.05,
                jitter: 1.0,
                align_to_normal: false,
                scale_range: (1.0, 1.0),
                rotation: inf_pcg::RotationMode::RandomYaw,
                altitude_offset: 0.0,
            },
            kinds: vec![inf_pcg::PcgKind::mesh(POST_MESH)],
        }],
    );
    write_asset(
        &content,
        "Village.inf_pcg",
        &inf_pcg::PcgAssetPayload::new(doc.clone()).encode().unwrap(),
        GRAPH_GUID,
        inf_asset::AssetKind::Pcg,
    );
    let pop = population(&dir_built(&content));
    assert!(!pop.is_empty(), "the scatter still runs");
    // Every instance is the scatter's single kind — no grammar contributed.
    assert!(pop.iter().all(|i| i.kind == 0));
    // …and the same document evaluated directly agrees, so nothing was added.
    let region = inf_pcg::Region::from_xz(
        CENTER.x - EXTENT,
        CENTER.z - EXTENT,
        CENTER.x + EXTENT,
        CENTER.z + EXTENT,
    );
    let mut seeded = doc;
    for layer in &mut seeded.layers {
        for rule in &mut layer.rules {
            rule.scatter.seed = rule.scatter.seed.wrapping_add(17);
        }
    }
    let terrain = grammar_scene()
        .world()
        .world()
        .iter_entities()
        .find_map(|e| e.get::<Terrain>().cloned())
        .expect("terrain");
    let provider = inf_pcg::FnHeight::new(move |x, z| terrain.data.height_at(DVec2::new(x, z)));
    assert_eq!(
        inf_pcg::evaluate(&seeded, &provider, region).len(),
        pop.len()
    );
}
