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
