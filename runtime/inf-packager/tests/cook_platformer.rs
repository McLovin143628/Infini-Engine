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
fn dense_mesh() -> inf_mesh::MeshAsset {
    let n = 40usize; // 39·39·2 = 3042 triangles
    let mut vertices = Vec::new();
    for z in 0..n {
        for x in 0..n {
            let (fx, fz) = (x as f32, z as f32);
            let y = 0.5 * (fx * 0.4).sin() * (fz * 0.4).cos();
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

    // The derived payload decodes to a real multi-level meshlet DAG.
    let vbytes = reader.read(vmesh_id).unwrap();
    let vgeom: inf_vgeom::VgeomMesh = inf_asset::decode(&vbytes).unwrap();
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
    ProjectManifest::new("Terrain Streaming", "blank-3d")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    // The `.inf_terrain` asset: tiles + LOD pyramid, written as the RAW payload
    // image (never `inf_asset::encode` — a length prefix would misalign tiles).
    let src = streaming_terrain();
    let pyramid = inf_terrain::build_pyramid(&src, inf_terrain::PyramidOptions::default());
    let asset = inf_terrain::build_terrain_asset(&src, &pyramid).unwrap();
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
        }],
        settings: Default::default(),
    };
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
    let good = inf_terrain::build_terrain_asset(&src, &pyramid).unwrap();
    let image = good.as_bytes();

    let framed = bincode::serde::encode_to_vec(image, bincode::config::standard()).unwrap();
    let truncated = image[..80].to_vec();
    let oob = {
        let mut b = image.to_vec();
        // First directory entry's offset field → past the end.
        b[64 + 16..64 + 24].copy_from_slice(&u64::MAX.to_le_bytes());
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
