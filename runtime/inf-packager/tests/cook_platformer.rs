//! End-to-end cook tests (P9.2): cook the committed platformer sample into a
//! temp dir, assert determinism, blueprint validation, and closure filtering.

use std::path::{Path, PathBuf};

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash, PackReader};
use inf_packager::{
    cook, derived_vmesh_id, CookError, CookManifest, CookOptions, DEFAULT_PACK_NAME, MANIFEST_FILE,
};
use inf_project::ProjectManifest;

/// The workspace root, from this crate at `runtime/inf-packager`.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Scaffold a minimal project in `root` with the platformer sample copied into
/// its Content directory. Returns the project root.
fn make_platformer_project(root: &Path) {
    ProjectManifest::new("Platformer Sample", "2d-platformer")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let sample = workspace_root().join("samples/platformer-2d");
    // Copy the blueprint's sidecar too, so its **stable** asset GUID (the one the
    // level's persisted `actor` binding points at) survives the cook.
    for f in [
        "Platformer.inf_lvl",
        "Coyote.inf_act",
        "Coyote.inf_act.toml",
    ] {
        std::fs::copy(sample.join(f), content.join(f)).unwrap();
    }
}

#[test]
fn cooks_the_platformer_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_platformer_project(&proj);
    let out = dir.path().join("out");

    let report = cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");

    // The pack + manifest exist.
    let pack_path = out.join(DEFAULT_PACK_NAME);
    assert!(pack_path.exists(), "pack written");
    assert!(out.join(MANIFEST_FILE).exists(), "manifest written");

    // The level + the blueprint were packed.
    assert_eq!(report.asset_count, 2, "level + blueprint");
    assert_eq!(report.kinds.get("level"), Some(&1));
    assert_eq!(report.kinds.get("blueprint"), Some(&1));
    assert_eq!(report.blueprints_validated, 1);
    assert_eq!(report.levels_rewritten, 1);
    assert_eq!(report.levels.len(), 1);
    assert!(report.root_level.is_some());
    assert!(report.pack_bytes > 0);

    // The pack reads back, and the root level decodes to the platformer.
    let reader = PackReader::open(&pack_path).unwrap();
    assert_eq!(reader.len(), 2);
    let root = report.root_level.unwrap();
    let level_bytes = reader.read(root).expect("root level in pack");
    let level = inf_scene::decode(&level_bytes).expect("cooked level decodes");
    assert_eq!(level.title, "Platformer 2D");
    assert_eq!(level.len(), 5);

    // The manifest names the pack + root level.
    let manifest =
        CookManifest::from_toml(&std::fs::read_to_string(out.join(MANIFEST_FILE)).unwrap())
            .unwrap();
    assert_eq!(manifest.packs, vec![DEFAULT_PACK_NAME.to_string()]);
    assert_eq!(manifest.root_level, Some(root.uuid()));
    assert_eq!(manifest.asset_count, 2);
    assert_eq!(manifest.project_name, "Platformer Sample");
}

#[test]
fn cook_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_platformer_project(&proj);

    let out_a = dir.path().join("a");
    let out_b = dir.path().join("b");
    cook(&proj, &out_a, &CookOptions::default()).unwrap();
    cook(&proj, &out_b, &CookOptions::default()).unwrap();

    let pack_a = std::fs::read(out_a.join(DEFAULT_PACK_NAME)).unwrap();
    let pack_b = std::fs::read(out_b.join(DEFAULT_PACK_NAME)).unwrap();
    assert_eq!(pack_a, pack_b, "two cooks → byte-identical pack");

    let man_a = std::fs::read(out_a.join(MANIFEST_FILE)).unwrap();
    let man_b = std::fs::read(out_b.join(MANIFEST_FILE)).unwrap();
    assert_eq!(man_a, man_b, "two cooks → byte-identical manifest");
}

#[test]
fn a_broken_blueprint_fails_with_a_handler_anchored_error() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Broken", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    // A `.inf_act` whose Tick body references an undefined local (n99).
    let broken = r#"{
        "schema_version": 1,
        "id": "act:broken",
        "name": "Broken Actor",
        "events": [
            { "event": "Tick",
              "body": { "id": "tick", "name": "tick",
                        "params": [{ "name": "dt", "ty": "Float" }],
                        "ret": "Unit",
                        "body": [ { "ExprStmt": { "Local": 99 } } ] } }
        ]
    }"#;
    std::fs::write(content.join("Broken.inf_act"), broken).unwrap();

    let out = proj.join("out");
    let err = cook(&proj, &out, &CookOptions::default()).unwrap_err();
    match err {
        CookError::Blueprint {
            class,
            handler,
            message,
            ..
        } => {
            assert_eq!(class, "Broken Actor");
            assert_eq!(handler, "tick");
            assert!(message.contains("n99"), "message: {message}");
        }
        other => panic!("expected a Blueprint error, got {other:?}"),
    }
}

#[test]
fn explicit_level_root_pulls_the_bound_blueprint_via_actor_edge() {
    // Deliverable 5: with ONLY the level as an explicit root, the blueprint bound
    // by the level's persisted `actor` slot is still shipped — pulled through the
    // real level→blueprint dependency edge (not because scripts are default roots).
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_platformer_project(&proj);

    // Resolve the level's asset GUID from a scan of the same content root.
    let mut db = inf_asset::AssetDb::new(proj.join("Content"));
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
        report.kinds.get("blueprint"),
        Some(&1),
        "the bound blueprint is packed via the actor edge, not as a default root"
    );
    let reader = PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    assert!(
        reader
            .index()
            .any(|e| e.kind == inf_asset::AssetKind::Blueprint),
        "the pack contains the bound Coyote blueprint"
    );
}

// ─────────────── virtualized-geometry cook derivation (P13.1) ───────────────

/// A fixed asset id so both cooks (and the runtime) agree on the mesh GUID.
fn fixed_mesh_id() -> AssetId {
    "11111111-1111-4111-8111-111111111111".parse().unwrap()
}

/// A dense grid mesh (> the 2048-triangle default vmesh threshold).
///
/// The displacement uses `psin64`/`pcos64`, never `std` trig (the P14 LAW): f32
/// std trig is not bit-portable, so a fixture built with it feeds `meshopt`
/// different vertices per platform and the cook derives a different meshlet DAG on
/// each — the root cause of two macOS-only vgeom CI failures. Nothing here asserts
/// a count, but a cook fixture has no business being platform-dependent.
fn dense_mesh() -> inf_mesh::MeshAsset {
    let n = 40usize; // 39·39·2 = 3042 triangles
    let mut vertices = Vec::new();
    for z in 0..n {
        for x in 0..n {
            let (fx, fz) = (x as f32, z as f32);
            let y = (0.5 * inf_math::psin64(fx as f64 * 0.4) * inf_math::pcos64(fz as f64 * 0.4))
                as f32;
            vertices.push(inf_mesh::MeshVertex {
                position: [fx, y, fz],
                ..Default::default()
            });
        }
    }
    let mut indices = Vec::new();
    let idx = |x: usize, z: usize| (z * n + x) as u32;
    for z in 0..n - 1 {
        for x in 0..n - 1 {
            indices.extend_from_slice(&[idx(x, z), idx(x + 1, z), idx(x + 1, z + 1)]);
            indices.extend_from_slice(&[idx(x, z), idx(x + 1, z + 1), idx(x, z + 1)]);
        }
    }
    let sm = inf_mesh::SubMesh {
        name: "grid".into(),
        vertices,
        indices,
        material_slot: None,
        skin: Vec::new(),
    };
    inf_mesh::MeshAsset::new(vec![sm], Vec::new())
}

/// Scaffold a project whose Content holds one dense `.inf_mesh` (payload +
/// fixed-GUID sidecar). Returns the mesh id.
fn make_mesh_project(root: &Path) -> AssetId {
    ProjectManifest::new("Vmesh Sample", "blank-3d")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let mesh = dense_mesh();
    let bytes = inf_asset::encode(&mesh).unwrap();
    let path = content.join("Dense.inf_mesh");
    std::fs::write(&path, &bytes).unwrap();
    let id = fixed_mesh_id();
    AssetSidecar::new(id, AssetKind::Mesh, ContentHash::of(&bytes))
        .save(&path)
        .unwrap();
    id
}

#[test]
fn cook_derives_a_vmesh_for_a_dense_mesh() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let mesh_id = make_mesh_project(&proj);
    let out = dir.path().join("out");

    // Cook the mesh directly (explicit root — a static mesh is not a default root).
    let report = cook(
        &proj,
        &out,
        &CookOptions {
            roots: Some(vec![mesh_id]),
            ..Default::default()
        },
    )
    .expect("cook succeeds");

    assert_eq!(report.meshlet_meshes_derived, 1, "one vmesh derived");
    assert_eq!(report.kinds.get("mesh"), Some(&1));
    assert_eq!(report.kinds.get("meshlet_mesh"), Some(&1));
    assert_eq!(report.asset_count, 2, "the mesh + its derived vmesh");

    // The pack carries both, and the vmesh id is the deterministic derivation.
    let reader = PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    let vmesh_id = derived_vmesh_id(mesh_id);
    assert_ne!(vmesh_id, mesh_id, "derived id differs from the mesh id");
    assert!(reader.contains(mesh_id), "mesh in pack");
    assert!(reader.contains(vmesh_id), "derived vmesh in pack");
    assert_eq!(reader.entry(vmesh_id).unwrap().kind, AssetKind::MeshletMesh);

    // The derived payload is the **v2 paged image** (P18.2), not a bincode blob:
    // the runtime indexes its header + page directory without decoding, and the
    // entry is uncompressed so a page is a borrowed slice of the mapping.
    let vbytes = reader.read(vmesh_id).unwrap();
    assert!(
        inf_vgeom::asset::is_v2(&vbytes),
        "the cook must emit the paged .inf_vmesh image"
    );
    assert!(
        matches!(
            reader.read_ref(vmesh_id).unwrap(),
            std::borrow::Cow::Borrowed(_)
        ),
        "a streaming-class entry must be readable straight out of the mapping"
    );
    let source = inf_vgeom::VgeomSource::from_image(vbytes).expect("index the image");
    assert!(source.meshlet_count() > 0);
    assert!(
        source.pages().len() >= 2,
        "a dense mesh builds a DAG with pages beyond the roots"
    );
    assert!(source.pages()[0].is_root_page());
    assert!(source.total_resident_bytes() > 0);

    // And it still materializes to the same DAG the pre-streaming path carried.
    let vgeom = source.to_mesh().expect("materialize");
    assert!(vgeom.meshlet_count() > 0);
    assert!(vgeom.level_count() >= 2, "dense mesh built a DAG");
    assert!(vgeom.total_triangles() > 0);
}

#[test]
fn cook_with_vmesh_derivation_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let mesh_id = make_mesh_project(&proj);

    let out_a = dir.path().join("a");
    let out_b = dir.path().join("b");
    let opts = CookOptions {
        roots: Some(vec![mesh_id]),
        ..Default::default()
    };
    cook(&proj, &out_a, &opts).unwrap();
    cook(&proj, &out_b, &opts).unwrap();

    let pack_a = std::fs::read(out_a.join(DEFAULT_PACK_NAME)).unwrap();
    let pack_b = std::fs::read(out_b.join(DEFAULT_PACK_NAME)).unwrap();
    assert_eq!(
        pack_a, pack_b,
        "two cooks with vmesh derivation → byte-identical pack"
    );
}

#[test]
fn a_small_mesh_stays_below_the_vmesh_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Small", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    // A tiny 2-triangle quad — well under the 2048-triangle threshold.
    let v = |x: f32, z: f32| inf_mesh::MeshVertex {
        position: [x, 0.0, z],
        ..Default::default()
    };
    let sm = inf_mesh::SubMesh {
        name: "quad".into(),
        vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)],
        indices: vec![0, 1, 2, 0, 2, 3],
        material_slot: None,
        skin: Vec::new(),
    };
    let mesh = inf_mesh::MeshAsset::new(vec![sm], Vec::new());
    let bytes = inf_asset::encode(&mesh).unwrap();
    let path = content.join("Quad.inf_mesh");
    std::fs::write(&path, &bytes).unwrap();
    let id = fixed_mesh_id();
    AssetSidecar::new(id, AssetKind::Mesh, ContentHash::of(&bytes))
        .save(&path)
        .unwrap();

    let out = proj.join("out");
    let report = cook(
        &proj,
        &out,
        &CookOptions {
            roots: Some(vec![id]),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(report.meshlet_meshes_derived, 0, "small mesh: no vmesh");
    assert!(!report.kinds.contains_key("meshlet_mesh"));
    assert_eq!(report.asset_count, 1, "just the mesh");
}

#[test]
fn dependency_closure_excludes_an_unreferenced_stray() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_platformer_project(&proj);

    // A stray texture that no level/blueprint references (synthesized sidecar,
    // no dependency edges, not a root kind) → must not be packed.
    std::fs::write(
        proj.join("Content").join("Stray.inf_tex"),
        b"a stray unreferenced texture payload nobody depends on",
    )
    .unwrap();

    let out = proj.join("out");
    let report = cook(&proj, &out, &CookOptions::default()).unwrap();

    // Still just the level + blueprint; the texture was dropped.
    assert_eq!(report.asset_count, 2, "stray excluded from the closure");
    assert!(
        !report.kinds.contains_key("texture"),
        "no texture in the pack"
    );
}

// ── P16.3: the Terrain.asset → .inf_terrain cook edge ────────────────────────

/// A deterministic 4 × 4-tile terrain from a **polynomial** height field (never
/// `std` trig — the P14 bit-portability law).
fn streaming_terrain() -> inf_terrain::TerrainData {
    let mut t = inf_terrain::TerrainData::new(9, 2.0);
    for tz in 0..4 {
        for tx in 0..4 {
            t.author_tile((tx, tz), |x, z| x * 0.25 - z * 0.125 + 5.0);
        }
    }
    t
}

/// Scaffold a project holding one level whose `Terrain` streams from a
/// `.inf_terrain` asset, plus that asset. Returns the terrain asset's GUID.
fn make_streaming_terrain_project(root: &Path) -> AssetId {
    make_streaming_terrain_project_with(root, BiomeSetFixture::None)
}

/// What (if anything) the scaffolded level's `Terrain.biome_set` points at
/// (P19.2). The three cases are the three cook behaviours worth pinning: no
/// edge at all, a resolvable edge that must be *followed*, and a dangling edge
/// that must be *advised about*.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BiomeSetFixture {
    /// No `biome_set` — the pre-P19.2 shape.
    None,
    /// A real `.inf_biomes` on disk, with NO sidecar dependency declared — so the
    /// cook can only reach it by walking `Terrain.biome_set`.
    Present,
    /// A `biome_set` GUID that names nothing. Non-fatal: the level still cooks,
    /// but the painted ids resolve to nothing, and the cook must say so.
    Dangling,
    /// A real `.inf_biomes` whose first biome names a `.inf_pcg` — the P19.3
    /// hook. Neither sidecar declares a dependency, so the `.inf_pcg` is only
    /// reachable by walking `Terrain.biome_set` and then the SET's own payload.
    PresentWithGraph,
}

/// The GUID the [`BiomeSetFixture::PresentWithGraph`] scatter graph is written
/// under.
const BIOME_GRAPH_ID: AssetId = AssetId(uuid::Uuid::from_u128(0x1902_0200));

/// The GUID a [`BiomeSetFixture::Present`] set is written under.
const BIOME_SET_ID: AssetId = AssetId(uuid::Uuid::from_u128(0x1902_0100));
/// The GUID a [`BiomeSetFixture::Dangling`] level points at — deliberately absent.
const MISSING_BIOME_SET_ID: AssetId = AssetId(uuid::Uuid::from_u128(0x1902_0DEA));

fn make_streaming_terrain_project_with(root: &Path, biomes: BiomeSetFixture) -> AssetId {
    ProjectManifest::new("Terrain Streaming", "blank-3d")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    // The `.inf_terrain` asset: tiles + LOD pyramid, written as the RAW payload
    // image (never `inf_asset::encode` — a length prefix would misalign tiles).
    let src = streaming_terrain();
    let pyramid = inf_terrain::build_pyramid(&src, inf_terrain::PyramidOptions::default());
    let asset =
        inf_terrain::build_terrain_asset(&src, &pyramid, inf_terrain::PyramidOptions::default())
            .unwrap();
    let terrain_id = AssetId(uuid::Uuid::from_u128(0x1603_0100));
    let terrain_path = content.join("World.inf_terrain");
    std::fs::write(&terrain_path, asset.as_bytes()).unwrap();
    AssetSidecar::new(
        terrain_id,
        AssetKind::Terrain,
        ContentHash::of(asset.as_bytes()),
    )
    .save(&terrain_path)
    .unwrap();

    // A level whose single entity carries a Terrain pointing at it. The sidecar
    // deliberately declares NO dependencies, so the edge can only be found by
    // walking the level's persisted `Terrain.asset`.
    let level = inf_scene::RuntimeLevel {
        title: "Streaming Terrain".into(),
        entities: vec![inf_scene::RuntimeEntity {
            guid: uuid::Uuid::from_u128(0x1603_0101),
            name: "Terrain".into(),
            parent: None,
            transform: Default::default(),
            visible: true,
            mesh: None,
            material: None,
            light: None,
            camera: None,
            sprite: None,
            tilemap: None,
            nine_slice: None,
            text2d: None,
            light_2d: None,
            rigid_body_2d: None,
            collider_2d: None,
            character_controller_2d: None,
            rigid_body_3d: None,
            collider_3d: None,
            character_controller_3d: None,
            actor: None,
            terrain: Some(inf_ecs::components::Terrain {
                asset: Some(terrain_id.uuid()),
                biome_set: match biomes {
                    BiomeSetFixture::None => None,
                    BiomeSetFixture::Present | BiomeSetFixture::PresentWithGraph => {
                        Some(BIOME_SET_ID.uuid())
                    }
                    BiomeSetFixture::Dangling => Some(MISSING_BIOME_SET_ID.uuid()),
                },
                ..inf_ecs::components::Terrain::configured(9, 2.0)
            }),
            pcg_volume: None,
            skeletal_mesh: None,
            anim_player: None,
            anim_state_machine: None,
            root_motion: None,
            attached_to: None,
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
        }],
        settings: Default::default(),
    };
    // The `.inf_biomes` (P19.2), when the fixture wants one. Its sidecar declares
    // no dependencies either, so — exactly like the terrain — the only way the
    // cook can find it is by walking the level's persisted `Terrain.biome_set`.
    if matches!(
        biomes,
        BiomeSetFixture::Present | BiomeSetFixture::PresentWithGraph
    ) {
        let mut set = inf_terrain::BiomeSet::starter();
        if biomes == BiomeSetFixture::PresentWithGraph {
            set.biomes[0].pcg_graph = Some(BIOME_GRAPH_ID.uuid());
            // The cook treats a `.inf_pcg` payload as opaque (it rides through
            // verbatim), so a stub body is enough to prove the EDGE is walked —
            // which is the whole claim here — without pulling `inf-pcg` in for a
            // dependency test.
            let graph = b"stub .inf_pcg payload".to_vec();
            let gpath = content.join("BiomeScatter.inf_pcg");
            std::fs::write(&gpath, &graph).unwrap();
            AssetSidecar::new(BIOME_GRAPH_ID, AssetKind::Pcg, ContentHash::of(&graph))
                .save(&gpath)
                .unwrap();
        }
        let bytes = inf_asset::encode(&set).unwrap();
        let path = content.join("World.inf_biomes");
        std::fs::write(&path, &bytes).unwrap();
        AssetSidecar::new(BIOME_SET_ID, AssetKind::BiomeSet, ContentHash::of(&bytes))
            .save(&path)
            .unwrap();
    }

    let level_bytes = level.encode().unwrap();
    let level_path = content.join("World.inf_lvl");
    std::fs::write(&level_path, &level_bytes).unwrap();
    AssetSidecar::new(
        AssetId(uuid::Uuid::from_u128(0x1603_0102)),
        AssetKind::Level,
        ContentHash::of(&level_bytes),
    )
    .save(&level_path)
    .unwrap();

    terrain_id
}

#[test]
fn cook_follows_the_terrain_asset_edge_and_stores_it_uncompressed() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let terrain_id = make_streaming_terrain_project(&proj);
    let out = dir.path().join("out");

    let report = cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    assert_eq!(report.asset_count, 2, "level + its .inf_terrain");
    assert_eq!(report.kinds.get("terrain"), Some(&1));

    let reader = PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    let entry = reader
        .entry(terrain_id)
        .expect("terrain packed via the edge");
    assert_eq!(entry.kind, AssetKind::Terrain);
    assert!(
        !entry.compressed,
        "a streaming-class kind must cook uncompressed"
    );

    // Page tiles straight out of the pack mapping.
    let payload = reader.read_ref(terrain_id).unwrap();
    assert!(matches!(payload, std::borrow::Cow::Borrowed(_)));
    let view = inf_terrain::TerrainAssetReader::new(&*payload).unwrap();
    let src = streaming_terrain();
    assert_eq!(view.tile_resolution(), 9);
    assert!(view.lod_levels() > 1, "the LOD pyramid shipped too");
    for (&coord, tile) in src.tiles() {
        let key = inf_terrain::TileKey::lod0(coord);
        assert_eq!(&view.tile(key).unwrap().unwrap(), tile);
        assert_eq!(
            view.tile_bytes(key).unwrap().as_ptr() as usize % 16,
            0,
            "tile {coord:?} is not 16-byte aligned in the mapping"
        );
    }
}

#[test]
fn cook_with_a_terrain_asset_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_streaming_terrain_project(&proj);
    let a = dir.path().join("a");
    let b = dir.path().join("b");
    cook(&proj, &a, &CookOptions::default()).unwrap();
    cook(&proj, &b, &CookOptions::default()).unwrap();
    assert_eq!(
        std::fs::read(a.join(DEFAULT_PACK_NAME)).unwrap(),
        std::fs::read(b.join(DEFAULT_PACK_NAME)).unwrap(),
        "two cooks of one terrain project are byte-identical"
    );
}

// ── P19.2: the Terrain.biome_set → .inf_biomes cook edge ─────────────────────

/// **The edge is followed.** A level whose terrain names a `.inf_biomes` ships
/// that set in the pack, found only by walking `Terrain.biome_set` — the sidecar
/// declares no dependency at all.
///
/// Without this edge a cooked level would carry per-sample biome ids with no
/// vocabulary to resolve them: the overlay is blank and P19.3's per-biome
/// dispatch finds no graphs, with nothing on disk saying anything is wrong.
#[test]
fn cook_follows_the_biome_set_edge_and_compresses_it() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_streaming_terrain_project_with(&proj, BiomeSetFixture::Present);
    let out = dir.path().join("out");

    let report = cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    assert_eq!(report.asset_count, 3, "level + .inf_terrain + .inf_biomes");
    assert_eq!(report.kinds.get("biome_set"), Some(&1));
    assert!(
        report.warnings.iter().all(|w| !w.contains("biome set")),
        "a resolvable edge must not advise: {:?}",
        report.warnings
    );

    let reader = PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    let entry = reader
        .entry(BIOME_SET_ID)
        .expect("biome set packed via the edge");
    assert_eq!(entry.kind, AssetKind::BiomeSet);

    // …and it decodes back to the set that was authored. (A biome set is authored
    // data, not a streaming page, so — unlike the terrain — it takes the ordinary
    // compression policy; whether zstd actually shrinks this particular tiny
    // payload is not the claim, so the round trip is.)
    let payload = reader.read(BIOME_SET_ID).unwrap();
    let set: inf_terrain::BiomeSet = inf_asset::decode(&payload).unwrap();
    assert_eq!(set, inf_terrain::BiomeSet::starter());
}

/// **The chain is two links long.** `level → biome set → .inf_pcg`: a graph a
/// biome scatters with ships because the cook re-derives the set's edges from its
/// **payload**, exactly like every other referencing kind. Neither sidecar
/// declares the dependency, so a `Vec::new()` fall-through in `asset_deps` would
/// leave the graph out of the pack.
///
/// Latent until P19.3 populates `pcg_graph` — which is precisely why it is pinned
/// now: nothing else would fail if the arm went missing.
#[test]
fn cook_follows_a_biome_sets_pcg_graph_edge() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_streaming_terrain_project_with(&proj, BiomeSetFixture::PresentWithGraph);
    let out = dir.path().join("out");

    let report = cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    assert_eq!(
        report.asset_count, 4,
        "level + .inf_terrain + .inf_biomes + the biome's .inf_pcg"
    );
    assert_eq!(report.kinds.get("pcg"), Some(&1));

    let reader = PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    assert!(
        reader.entry(BIOME_GRAPH_ID).is_some(),
        "the biome's scatter graph was not pulled in through the set's payload"
    );

    // The edge really is payload-derived: nothing declared it.
    let sidecar = AssetSidecar::load(&proj.join("Content").join("World.inf_biomes")).unwrap();
    assert!(
        sidecar.dependencies.is_empty(),
        "the fixture must not declare the dependency — that is the whole point"
    );
}

/// **A dangling edge is advised about, not fatal.** The level is still valid —
/// its ids are stored on the tiles and cook fine — but nothing can resolve them,
/// which is precisely the silent hole an advisory exists for.
#[test]
fn a_dangling_biome_set_reference_is_a_cook_advisory() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_streaming_terrain_project_with(&proj, BiomeSetFixture::Dangling);
    let out = dir.path().join("out");

    let report = cook(&proj, &out, &CookOptions::default()).expect("a dangling ref is not fatal");
    assert_eq!(report.asset_count, 2, "level + .inf_terrain, and no set");
    let advisory = report
        .warnings
        .iter()
        .find(|w| w.contains("missing biome set"))
        .unwrap_or_else(|| panic!("no advisory raised: {:?}", report.warnings));
    assert!(
        advisory.contains(&MISSING_BIOME_SET_ID.to_string()),
        "the advisory must name the missing GUID: {advisory}"
    );
}

/// **An invalid biome set fails the BUILD.** Ambiguous ids cannot be recovered
/// from at runtime — the per-sample values are already baked into the tiles — so
/// the cook refuses, where it is one edit to fix.
#[test]
fn an_invalid_biome_set_fails_the_cook() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_streaming_terrain_project_with(&proj, BiomeSetFixture::Present);

    // Overwrite the set with one that claims the reserved id 0. The encode side
    // has no validation, so this is exactly the hand-edited / corrupt-file case.
    let mut bad = inf_terrain::BiomeSet::starter();
    bad.biomes.push(inf_terrain::BiomeDef::new(
        inf_terrain::UNASSIGNED_BIOME,
        "Void",
    ));
    let bytes = inf_asset::encode(&bad).unwrap();
    let path = proj.join("Content").join("World.inf_biomes");
    std::fs::write(&path, &bytes).unwrap();
    AssetSidecar::new(BIOME_SET_ID, AssetKind::BiomeSet, ContentHash::of(&bytes))
        .save(&path)
        .unwrap();

    let out = dir.path().join("out");
    match cook(&proj, &out, &CookOptions::default()) {
        Err(CookError::BiomeSet { guid, message }) => {
            assert_eq!(guid, BIOME_SET_ID);
            assert!(message.contains("reserved"), "unhelpful message: {message}");
        }
        other => panic!("expected CookError::BiomeSet, got {other:?}"),
    }
}

#[test]
fn a_corrupt_terrain_asset_fails_the_cook() {
    // The runtime pages tiles by trusting a header + directory it validated once,
    // so a malformed `.inf_terrain` must break the BUILD, not the shipped player.
    // Three shapes of wrong, each caught at cook:
    //   1. bincode-framed (what a generic `inf_asset::encode` would have written —
    //      the exact mistake the closed write door exists to prevent);
    //   2. truncated mid-directory;
    //   3. a directory entry pointing past the end of the payload.
    let src = streaming_terrain();
    let pyramid = inf_terrain::build_pyramid(&src, inf_terrain::PyramidOptions::default());
    let good =
        inf_terrain::build_terrain_asset(&src, &pyramid, inf_terrain::PyramidOptions::default())
            .unwrap();
    let image = good.as_bytes();

    let framed = bincode::serde::encode_to_vec(image, bincode::config::standard()).unwrap();
    // Mid-directory: past the fixed header (v2: 128 B), inside the first entry.
    let hlen = inf_terrain::HEADER_LEN_V2 as usize;
    let truncated = image[..hlen + 8].to_vec();
    let oob = {
        let mut b = image.to_vec();
        // First directory entry's offset field → past the end.
        b[hlen + 16..hlen + 24].copy_from_slice(&u64::MAX.to_le_bytes());
        b
    };

    for (label, payload) in [
        ("bincode-framed", framed),
        ("truncated", truncated),
        ("out-of-bounds blob", oob),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().join("proj");
        let terrain_id = make_streaming_terrain_project(&proj);
        // Overwrite the good asset with the corrupt one (sidecar hash is not the
        // guard here — a structural check is).
        std::fs::write(proj.join("Content/World.inf_terrain"), &payload).unwrap();

        let err = cook(&proj, &dir.path().join("out"), &CookOptions::default())
            .expect_err(&format!("{label} must fail the cook"));
        match err {
            CookError::Terrain { guid, message } => {
                assert_eq!(guid, terrain_id, "{label} error names the asset");
                assert!(!message.is_empty(), "{label} error explains itself");
            }
            other => panic!("{label}: expected CookError::Terrain, got {other:?}"),
        }
    }

    // …and the unmodified asset still cooks, so the check is not just "always fail".
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_streaming_terrain_project(&proj);
    cook(&proj, &dir.path().join("out"), &CookOptions::default())
        .expect("a valid terrain asset still cooks");
}

#[test]
fn a_dangling_terrain_reference_warns_without_failing() {
    // A level naming a terrain asset the project no longer has: the closure cannot
    // follow the edge, so the level would ship with ground that never streams.
    // Non-fatal (the inline data stays authoritative) but never silent.
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let terrain_id = make_streaming_terrain_project(&proj);
    // Delete the terrain asset + its sidecar, leaving the level's reference dangling.
    std::fs::remove_file(proj.join("Content/World.inf_terrain")).unwrap();
    std::fs::remove_file(proj.join("Content/World.inf_terrain.toml")).unwrap();

    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default())
        .expect("a dangling ref is an advisory, not a cook failure");
    assert_eq!(report.asset_count, 1, "just the level; the terrain is gone");
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains(&terrain_id.to_string()) && w.contains("terrain")),
        "the dangling terrain ref must be reported: {:?}",
        report.warnings
    );
}

/// **The sub-threshold advisory reaches the report** (P18.3 audit).
///
/// A mesh below `min_triangles` gets no `.inf_vmesh`, and since `RenderScene` has
/// exactly one door for real geometry, that means the shipped build draws a
/// placeholder cube — while the editor, which derives from one triangle, shows the
/// real thing. The threshold is a defensible cost decision; being silent about it
/// is how "it looked right in the editor" becomes a shipped bug.
#[test]
fn cook_advises_when_a_mesh_is_below_the_vgeom_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let mesh_id = make_mesh_project(&proj);
    let out = dir.path().join("out");

    // The fixture is ~3k triangles; raise the bar above it so the cook declines.
    let report = cook(
        &proj,
        &out,
        &CookOptions {
            roots: Some(vec![mesh_id]),
            vgeom: inf_packager::VgeomCookOptions {
                enabled: true,
                min_triangles: 1_000_000,
            },
            ..Default::default()
        },
    )
    .expect("cook succeeds");

    assert_eq!(report.meshlet_meshes_derived, 0, "nothing derived");
    let advisory = report
        .warnings
        .iter()
        .find(|w| w.contains(&mesh_id.to_string()))
        .unwrap_or_else(|| panic!("no sub-threshold advisory in {:?}", report.warnings));
    assert!(advisory.contains("PLACEHOLDER CUBE"), "{advisory}");
    assert!(
        advisory.contains("1000000"),
        "states the threshold: {advisory}"
    );
    assert!(advisory.contains("min_triangles"), "{advisory}");

    // …and a mesh the cook DOES virtualize raises nothing, so the advisory cannot
    // become background noise.
    let out2 = dir.path().join("out2");
    let ok = cook(
        &proj,
        &out2,
        &CookOptions {
            roots: Some(vec![mesh_id]),
            ..Default::default()
        },
    )
    .expect("cook succeeds");
    assert_eq!(ok.meshlet_meshes_derived, 1);
    assert!(
        !ok.warnings.iter().any(|w| w.contains("PLACEHOLDER CUBE")),
        "a derived mesh must not be advised about: {:?}",
        ok.warnings
    );
}

// ───────────────── P28.2: the cluster → tile pairing at cook ─────────────────

/// An entity with every component slot empty — the same field list the terrain
/// fixture above spells out, hoisted so a second fixture does not spell it again.
fn bare_entity(guid: u128, name: &str) -> inf_scene::RuntimeEntity {
    inf_scene::RuntimeEntity {
        guid: uuid::Uuid::from_u128(guid),
        name: name.into(),
        parent: None,
        transform: Default::default(),
        visible: true,
        mesh: None,
        material: None,
        light: None,
        camera: None,
        sprite: None,
        tilemap: None,
        nine_slice: None,
        text2d: None,
        light_2d: None,
        rigid_body_2d: None,
        collider_2d: None,
        character_controller_2d: None,
        rigid_body_3d: None,
        collider_3d: None,
        character_controller_3d: None,
        actor: None,
        terrain: None,
        pcg_volume: None,
        skeletal_mesh: None,
        anim_player: None,
        anim_state_machine: None,
        root_motion: None,
        attached_to: None,
        joint_2d: None,
        joint_3d: None,
        audio_source: None,
        audio_listener: None,
        decal: None,
        volume: None,
        spline: None,
        foliage: None,
        streaming_source: None,
        always_loaded: None,
        time_of_day: None,
        sky_atmosphere: None,
        water_body: None,
        buoyancy: None,
        voxel_volume: None,
        destructible: None,
        ik_target: None,
        cloth_sim: None,
        hair_guides: None,
    }
}

/// A 512² checker as a v2 tiled `.inf_tex` — a pyramid of 128-texel tiles, which
/// is enough levels for the mip rule to have something to say at more than one
/// page.
fn tiled_texture_bytes() -> Vec<u8> {
    let (w, h) = (512u32, 512u32);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / 32) + (y / 32)) % 2 == 0;
            let v = if on { 220u8 } else { 40u8 };
            rgba.extend_from_slice(&[v, v / 2, 255 - v, 255]);
        }
    }
    inf_material::tiles::build_tiled_texture(
        rgba,
        w,
        h,
        inf_material::texture::TextureImportSettings::default(),
    )
    .expect("tile the texture")
    .as_bytes()
    .to_vec()
}

/// [`make_mesh_project`] plus a `.inf_tex`, a `.inf_mat` that samples it, and a
/// level whose one entity binds BOTH the mesh and the material — which is the
/// only place in the tree where those two facts meet, and therefore the only
/// place a cluster→tile pairing can be derived from.
fn make_textured_mesh_project(root: &Path) -> (AssetId, AssetId, AssetId) {
    let mesh_id = make_mesh_project(root);
    let content = root.join("Content");

    let tex_id = AssetId(uuid::Uuid::from_u128(0x2802_0001));
    let tex_bytes = tiled_texture_bytes();
    let tex_path = content.join("Checker.inf_tex");
    std::fs::write(&tex_path, &tex_bytes).unwrap();
    AssetSidecar::new(tex_id, AssetKind::Texture, ContentHash::of(&tex_bytes))
        .save(&tex_path)
        .unwrap();

    let mat_id = AssetId(uuid::Uuid::from_u128(0x2802_0002));
    let mat = inf_material::MaterialAsset {
        base_color_texture: Some(tex_id),
        ..Default::default()
    };
    let mat_bytes = inf_asset::encode(&mat).unwrap();
    let mat_path = content.join("Checker.inf_mat");
    std::fs::write(&mat_path, &mat_bytes).unwrap();
    AssetSidecar::new(mat_id, AssetKind::Material, ContentHash::of(&mat_bytes))
        .save(&mat_path)
        .unwrap();

    let mut e = bare_entity(0x2802_0010, "Prop");
    e.mesh = Some(inf_ecs::components::MeshRef {
        asset: Some(mesh_id.uuid()),
        ..Default::default()
    });
    e.material = Some(inf_ecs::components::Material {
        asset: Some(mat_id.uuid()),
        ..Default::default()
    });
    let level = inf_scene::RuntimeLevel {
        title: "Textured Cluster Pages".into(),
        entities: vec![e],
        settings: Default::default(),
    };
    let level_bytes = level.encode().unwrap();
    let level_path = content.join("Prop.inf_lvl");
    std::fs::write(&level_path, &level_bytes).unwrap();
    AssetSidecar::new(
        AssetId(uuid::Uuid::from_u128(0x2802_0011)),
        AssetKind::Level,
        ContentHash::of(&level_bytes),
    )
    .save(&level_path)
    .unwrap();

    (mesh_id, mat_id, tex_id)
}

/// **The cook pairs a cluster page with the tiles its materials sample** — the
/// P28.2 clause-1 gate, asserted on the packed bytes rather than on the planner.
///
/// Falsifiable in three directions at once: the pairing has to be non-empty, it
/// has to name the texture the LEVEL bound (not some other asset in the project),
/// and the mip it names has to follow the stated rule against the page's own LOD
/// level. A cook that emitted a plausible but empty tiles section passes none.
#[test]
fn cook_pairs_cluster_pages_with_the_tiles_their_materials_sample() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let (mesh_id, _mat_id, tex_id) = make_textured_mesh_project(&proj);
    let out = dir.path().join("out");
    let report = cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    assert!(
        report.warnings.is_empty(),
        "a fully-bound project raised advisories: {:?}",
        report.warnings
    );

    let reader = PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    let vbytes = reader.read(derived_vmesh_id(mesh_id)).unwrap();
    assert_eq!(
        inf_vgeom::asset::container_version(&vbytes),
        Some(inf_vgeom::VMESH_ASSET_SCHEMA_VERSION),
        "the cook emits the current container"
    );
    let r = inf_vgeom::VgeomAssetReader::new(vbytes.as_slice()).expect("index");

    // The texture's own mip grid, read from the packed `.inf_tex` — the pairing's
    // second input, fetched independently of the planner that consumed it.
    let tex = reader.read(tex_id).unwrap();
    let desc = inf_material::tiles::TiledTextureReader::new(tex.as_slice())
        .expect("v2 container")
        .vt_desc();

    let mut paired = 0usize;
    for (p, e) in r.pages().iter().enumerate() {
        let s = r.page_sections(p).expect("sections");
        let refs = s.tile_refs();
        assert_eq!(refs.len(), e.tile_count as usize);
        assert!(!refs.is_empty(), "page {p} carries no pairing");
        paired += refs.len();
        let want_mip = inf_vgeom::tile_mip_for_lod(e.lod, desc.mip_count());
        for t in refs {
            assert_eq!(t.texture(), tex_id, "page {p} names a foreign texture");
            assert_eq!(t.mip, want_mip, "page {p} paired the wrong mip");
            let m = &desc.mips[t.mip as usize];
            assert!(t.x < m.tiles_x && t.y < m.tiles_y, "tile outside the grid");
        }
    }
    assert!(paired >= r.pages().len(), "every page paired something");

    // The finest page reaches mip 0. This is the whole point of the rule — a
    // pairing that capped one level short would leave "detailed mesh, blurry
    // texture" reachable at exactly the range it exists to close.
    let finest = r.pages().last().expect("pages");
    assert_eq!(
        inf_vgeom::tile_mip_for_lod(finest.lod, desc.mip_count()),
        0,
        "the finest cluster page must pair with the finest texture level"
    );

    // And the tiles the pages name are ADDRESSES into the packed `.inf_tex`, not
    // copies of it — the measurement that chose references over embedded texels.
    assert!(
        (vbytes.len() as f64) < (tex.len() as f64) * 4.0,
        "the pairing looks like embedded texels: vmesh {} B vs texture {} B",
        vbytes.len(),
        tex.len()
    );
}

/// Two cooks of one paired project are byte-identical — the pairing is inside the
/// cook's determinism guarantee, not beside it.
#[test]
fn cook_with_a_cluster_pairing_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let (mesh_id, _, _) = make_textured_mesh_project(&proj);
    let a = dir.path().join("a");
    let b = dir.path().join("b");
    cook(&proj, &a, &CookOptions::default()).expect("cook a");
    cook(&proj, &b, &CookOptions::default()).expect("cook b");
    let ra = PackReader::open(&a.join(DEFAULT_PACK_NAME)).unwrap();
    let rb = PackReader::open(&b.join(DEFAULT_PACK_NAME)).unwrap();
    let id = derived_vmesh_id(mesh_id);
    assert_eq!(ra.read(id).unwrap(), rb.read(id).unwrap());
    assert_eq!(
        std::fs::read(a.join(DEFAULT_PACK_NAME)).unwrap(),
        std::fs::read(b.join(DEFAULT_PACK_NAME)).unwrap(),
        "two cooks of a paired project must be byte-identical"
    );
}

/// **The control**: the same mesh, cooked with no level binding it to a material,
/// gets a current container with EMPTY tile sections. Without this the arm above
/// is satisfied by a cook that pairs every mesh with every texture in a project.
#[test]
fn an_unbound_mesh_cooks_a_container_with_no_pairing() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let mesh_id = make_mesh_project(&proj);
    let out = dir.path().join("out");
    let report = cook(
        &proj,
        &out,
        &CookOptions {
            roots: Some(vec![mesh_id]),
            ..Default::default()
        },
    )
    .expect("cook succeeds");
    assert_eq!(report.meshlet_meshes_derived, 1);

    let reader = PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    let vbytes = reader.read(derived_vmesh_id(mesh_id)).unwrap();
    let r = inf_vgeom::VgeomAssetReader::new(vbytes.as_slice()).expect("index");
    assert!(r.pages().len() >= 2, "a real DAG, not an empty shell");
    for e in r.pages() {
        assert_eq!(e.tile_count, 0, "an unbound mesh paired something");
    }
}

/// A bound material whose texture is a **v1** `.inf_tex` cannot be paired, and the
/// cook says so against the MESH — the pages that silently lose their coupling —
/// rather than only against the material.
#[test]
fn a_mesh_bound_to_an_unpageable_texture_raises_a_cluster_advisory() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let (mesh_id, _mat, tex_id) = make_textured_mesh_project(&proj);
    // Overwrite the tiled container with a v1 bincode payload, keeping the GUID:
    // this is exactly the shape of every project that predates P26.1.
    let v1 = inf_asset::encode(&inf_material::TextureAsset {
        schema_version: inf_material::TextureAsset::CURRENT_VERSION,
        width: 4,
        height: 4,
        format: inf_material::TextureFormat::Rgba8,
        srgb: true,
        mips: vec![inf_material::TextureMip {
            width: 4,
            height: 4,
            data: vec![255; 4 * 4 * 4],
        }],
    })
    .unwrap();
    let path = proj.join("Content").join("Checker.inf_tex");
    std::fs::write(&path, &v1).unwrap();
    AssetSidecar::new(tex_id, AssetKind::Texture, ContentHash::of(&v1))
        .save(&path)
        .unwrap();

    let out = dir.path().join("out");
    let report = cook(&proj, &out, &CookOptions::default()).expect("cook still succeeds");
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains(&mesh_id.to_string()) && w.contains("cluster pages")),
        "no cluster-pairing advisory for an unpageable texture: {:?}",
        report.warnings
    );
    let reader = PackReader::open(&out.join(DEFAULT_PACK_NAME)).unwrap();
    let vbytes = reader.read(derived_vmesh_id(mesh_id)).unwrap();
    let r = inf_vgeom::VgeomAssetReader::new(vbytes.as_slice()).expect("index");
    for e in r.pages() {
        assert_eq!(e.tile_count, 0, "an unpairable texture paired anyway");
    }
}
