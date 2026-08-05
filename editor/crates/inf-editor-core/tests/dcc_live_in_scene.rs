//! **P23.4's live-in-scene proof**: a mesh edited in the Model Editor is drawn
//! by a scene that references it, with no invalidation call anywhere.
//!
//! This is the wire the phase gate (P23.6) will pull harder, proven now at the
//! unit level because every link in it is owned by a different module and the
//! interesting failures live in the joins:
//!
//! ```text
//!   open  → inf_dcc::from_mesh_asset            (the kernel's reader)
//!   edit  → Op::ExtrudeFaces                    (P23.4)
//!   save  → AssetProject::rewrite_payload
//!         → assets::vmesh::ensure_vmesh          SYNCHRONOUSLY (the P23.1 rule)
//!   draw  → EditorRenderAssets::resolve_vgeom    re-keyed by CONTENT HASH
//! ```
//!
//! # The two claims, and why each has to be measured rather than reasoned about
//!
//! **1. The render key moves.** `EditorRenderAssets` keys a loaded `.inf_vmesh`
//! by its payload's content hash, not its GUID (`render_assets`'s module docs:
//! a cooked pack is immutable, a project's content root is not). So new bytes are
//! a new key, the old entry is unreachable, and a stale draw is *unrepresentable*
//! rather than merely unlikely. Asserting the id CHANGED is asserting that
//! machinery is actually in the path.
//!
//! **2. The bytes behind the new key are the edited shape.** An id that changed
//! proves a rewrite happened; it does not prove the rewrite went through the
//! derivation. Without the synchronous `ensure_vmesh` the P23.1 memo insists on,
//! the `.inf_mesh` would be new and the `.inf_vmesh` beside it would still
//! describe the *old* geometry — no error, no warning, and the viewport redrawing
//! the previous surface with complete confidence. So the test decodes the derived
//! payload and counts what is in it.
//!
//! The one thing this file does not do is render: `inf_viewport::host` is
//! `#[cfg(any(windows, target_os = "macos"))]` and needs a GPU, so — exactly as
//! `editor_real_meshes.rs` does for P18.3 — it drives the Ring-1 resolution that
//! the host's branch calls, and `tests/projector_mirror.rs` separately pins that
//! branch against the shipped player's.

use inf_asset::{AssetId, AssetKind};
use inf_dcc::{FaceId, MeshSession, Op};
use inf_editor_core::assets::{vmesh, AssetProject};
use inf_editor_core::render_assets::EditorRenderAssets;

/// Write `mesh` as a `.inf_mesh` and derive its DAG — the same two steps the
/// save path takes, so a fixture and a save differ only in which one is first.
fn write_mesh(proj: &mut AssetProject, name: &str, mesh: &inf_dcc::Mesh) -> AssetId {
    let (asset, _) = inf_dcc::to_mesh_asset(mesh, &inf_dcc::ExportOptions::default());
    let dir = proj.content_dir("Meshes").unwrap();
    let id = proj
        .write_asset(&dir, name, &asset, None, vec![], None)
        .unwrap();
    assert_eq!(proj.db().get(id).unwrap().kind(), AssetKind::Mesh);
    vmesh::ensure_vmesh(proj, id).unwrap();
    id
}

/// The **derived** `.inf_vmesh`, decoded off disk: `(vertex count, highest Y)`.
///
/// Read rather than inferred, because this is exactly the pair that would NOT
/// have moved if `ensure_vmesh` had been left to a background sweep — the
/// `.inf_mesh` beside it would be new and this would still describe the cube.
fn derived_probe(root: &std::path::Path, mesh: AssetId) -> (usize, f32) {
    let proj = AssetProject::open(root).unwrap();
    let derived = inf_vgeom::derived_vmesh_id(mesh);
    let entry = proj
        .db()
        .get(derived)
        .expect("the mesh has a derived .inf_vmesh");
    let bytes = std::fs::read(&entry.path).unwrap();
    let source = inf_vgeom::VgeomSource::from_payload(bytes).expect("a readable DAG");
    let mesh = source.to_mesh().expect("the DAG decodes");
    let top = mesh
        .vertices
        .iter()
        .map(|v| v.position[1])
        .fold(f32::MIN, f32::max);
    (mesh.vertices.len(), top)
}

/// Every face lying on the box's +Y plane.
///
/// Plural, and that is the point: a `MeshAsset` is **triangles**, so what was a
/// quad in the kernel comes back as two of them. Extruding one alone would raise
/// half the lid; extruding both as a REGION is the thing an author means, and it
/// is the region-border rule that makes the shared diagonal get no wall.
fn top_faces(mesh: &inf_dcc::Mesh, y: f64) -> Vec<FaceId> {
    mesh.face_ids()
        .filter(|&f| {
            mesh.face_verts(f)
                .unwrap()
                .iter()
                .all(|&v| (mesh.position(v).unwrap().y - y).abs() < 1e-9)
        })
        .collect()
}

#[test]
fn a_mesh_edited_in_the_model_editor_re_keys_and_redraws_in_the_scene() {
    let root = tempfile::tempdir().unwrap();
    let mut proj = AssetProject::open(root.path()).unwrap();
    let mesh_id = write_mesh(&mut proj, "Prop", &inf_dcc::cube(2.0));
    drop(proj);

    // ── what the scene draws BEFORE the edit ──────────────────────────────
    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(root.path().to_path_buf()));
    let before = store
        .resolve_vgeom(mesh_id.uuid())
        .expect("a scene referencing this mesh resolves real geometry");
    let key_before = before.id;
    let (verts_before, top_before) = derived_probe(root.path(), mesh_id);
    assert!(
        (top_before - 1.0).abs() < 1e-5,
        "a 2 m cube tops out at 1 m"
    );

    // ── dcc_open ──────────────────────────────────────────────────────────
    let mut proj = AssetProject::open(root.path()).unwrap();
    let payload: inf_mesh::MeshAsset = proj.load_payload(mesh_id).unwrap();
    let import = inf_dcc::from_mesh_asset(&payload).expect("the kernel reads its own writer");
    assert_eq!(
        import.report.boundary_edges, 0,
        "a closed solid arrives closed — the fragmentation counter the panel surfaces"
    );
    let mut session = MeshSession::new(import.mesh);

    // ── dcc_apply: extrude ────────────────────────────────────────────────
    assert_eq!(
        session.mesh().face_count(),
        12,
        "an asset is triangles, so the kernel opens the cube as 12 of them"
    );
    let top = top_faces(session.mesh(), 1.0);
    assert_eq!(top.len(), 2, "the lid is two triangles");
    session
        .apply(Op::ExtrudeFaces {
            faces: top,
            distance: 1.5,
        })
        .expect("extrude");
    assert_eq!(
        session.mesh().face_count(),
        16,
        "10 untouched + 2 moved caps + FOUR walls — the shared diagonal is \
         interior to the region and gets none"
    );

    // ── dcc_save ──────────────────────────────────────────────────────────
    let (out, report) = inf_dcc::to_mesh_asset(session.mesh(), &inf_dcc::ExportOptions::default());
    assert_eq!(
        report.coincident_vertices, 0,
        "the save's advisory is clean on a 1.5 m extrude"
    );
    proj.rewrite_payload(mesh_id, &out, vec![]).unwrap();
    // **Synchronously**, in the same unit of work as the rewrite (P23.1 §2). A
    // background derivation is a window in which the level draws the old mesh.
    vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap();
    drop(proj);

    // ── what the scene draws AFTER ────────────────────────────────────────
    store.refresh_index();
    let after = store
        .resolve_vgeom(mesh_id.uuid())
        .expect("still resolves after the rewrite");
    assert_ne!(
        after.id, key_before,
        "the render key is the payload's CONTENT HASH — new bytes must be a new \
         key, or the renderer's cache keeps drawing the old surface"
    );

    // **The load-bearing assertion.** The id moving proves a rewrite happened;
    // only this proves the rewrite went through the derivation. Without the
    // synchronous `ensure_vmesh` the DAG here would still top out at 1 m and the
    // viewport would redraw the cube with complete confidence.
    let (verts_after, top_after) = derived_probe(root.path(), mesh_id);
    assert!(
        (top_after - 2.5).abs() < 1e-5,
        "the DERIVED geometry must be the extruded shape: it tops out at \
         {top_after} m, and the cube it was built from topped out at {top_before}"
    );
    assert!(
        verts_after > verts_before,
        "and it grew ({verts_before} -> {verts_after} vertices)"
    );

    // And the new bytes really decode to the taller solid: the extrude added
    // 1.5 m of height on +Y and nothing anywhere else.
    let proj = AssetProject::open(root.path()).unwrap();
    let reread: inf_mesh::MeshAsset = proj.load_payload(mesh_id).unwrap();
    assert!(
        (reread.bounds.max[1] - 2.5).abs() < 1e-5,
        "the saved mesh tops out at {} m, not 2.5",
        reread.bounds.max[1]
    );
    assert!((reread.bounds.min[1] + 1.0).abs() < 1e-5);
    assert!(
        (reread.bounds.max[0] - 1.0).abs() < 1e-5,
        "and did not widen"
    );
    // The round trip is honest: what was saved reads back as what was modelled.
    let back = inf_dcc::from_mesh_asset(&reread).expect("reads back");
    assert_eq!(
        back.mesh.face_count(),
        20,
        "16 kernel faces, of which the 4 walls are quads that the asset carries          as triangle pairs"
    );
    assert_eq!(back.report.boundary_edges, 0, "still a closed solid");
    assert_eq!(inf_dcc::validate(&back.mesh), Ok(()));
}

#[test]
fn a_save_that_changes_nothing_leaves_the_render_key_alone() {
    // The other half, and the reason the key can be trusted: it moves when the
    // bytes move and NOT otherwise. A key that changed on every save would make
    // the assertion above pass for a save path that had done nothing at all —
    // the vacuity the P19 law is about.
    let root = tempfile::tempdir().unwrap();
    let mut proj = AssetProject::open(root.path()).unwrap();
    let mesh_id = write_mesh(&mut proj, "Prop", &inf_dcc::cube(2.0));
    drop(proj);

    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(root.path().to_path_buf()));
    let before = store.resolve_vgeom(mesh_id.uuid()).unwrap().id;

    let mut proj = AssetProject::open(root.path()).unwrap();
    let payload: inf_mesh::MeshAsset = proj.load_payload(mesh_id).unwrap();
    let import = inf_dcc::from_mesh_asset(&payload).unwrap();
    let session = MeshSession::new(import.mesh);
    // Open and save with no edit in between.
    let (out, _) = inf_dcc::to_mesh_asset(session.mesh(), &inf_dcc::ExportOptions::default());
    proj.rewrite_payload(mesh_id, &out, vec![]).unwrap();
    vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap();
    drop(proj);

    store.refresh_index();
    assert_eq!(
        store.resolve_vgeom(mesh_id.uuid()).unwrap().id,
        before,
        "open-then-save must be a no-op — the kernel's round trip is a fixed point"
    );
}
