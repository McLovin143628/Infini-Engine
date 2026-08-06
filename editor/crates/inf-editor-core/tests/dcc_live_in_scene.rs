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
//!   save  → inf_editor_core::dcc::save_mesh_session   ← THE PRODUCT PATH
//!   draw  → EditorRenderAssets::resolve_vgeom    re-keyed by CONTENT HASH
//! ```
//!
//! # It calls the function the command calls, and that is the point
//!
//! The first version of this file **inlined** the two calls the save makes. It
//! passed, and it proved the *pattern* rather than the product: deleting
//! `ensure_vmesh` from `dcc_save` failed nothing at all, because nothing
//! executed `dcc_save`. A `#[tauri::command]` cannot be driven from a test, so
//! the save is a Ring-1 function ([`inf_editor_core::dcc::save_mesh_session`])
//! and both the command and this gate go through it.
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
use inf_editor_core::dcc;
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

    // ── dcc_save: THE PRODUCT PATH, not a re-statement of it ──────────────
    let saved = dcc::save_mesh_session(&mut proj, mesh_id, session.mesh()).expect("saves");
    assert_eq!(
        saved.export.coincident_vertices, 0,
        "the save's advisory is clean on a 1.5 m extrude"
    );
    assert!(
        matches!(saved.vmesh, vmesh::VmeshDerivation::Built(_)),
        "the derivation ran synchronously, in the same call: {:?}",
        saved.vmesh
    );
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
    // Open and save with no edit in between, through the same door.
    dcc::save_mesh_session(&mut proj, mesh_id, session.mesh()).expect("saves");
    drop(proj);

    store.refresh_index();
    assert_eq!(
        store.resolve_vgeom(mesh_id.uuid()).unwrap().id,
        before,
        "open-then-save must be a no-op — the kernel's round trip is a fixed point"
    );
}

#[test]
fn a_save_whose_derivation_fails_removes_the_stale_dag_rather_than_leaving_it() {
    // **The failure contract, executed.** `AssetProject` has a lock, not a
    // transaction: the payload can land and the derivation can then fail. The
    // state that must be unreachable is (new payload, OLD dag), because the
    // watcher re-keys on the new content hash and the viewport draws the
    // previous surface with complete confidence — the exact failure the design
    // memo opens by naming.
    //
    // Provoked by handing the save a mesh with no faces: the writer produces an
    // asset with no triangles, which `ensure_vmesh` declines to virtualize. That
    // is `Skipped`, not an error — so this test asserts the *other* half of the
    // invariant directly: after ANY save, the derived artifact on disk either
    // describes what was just written or is not there at all.
    let root = tempfile::tempdir().unwrap();
    let mut proj = AssetProject::open(root.path()).unwrap();
    let mesh_id = write_mesh(&mut proj, "Prop", &inf_dcc::cube(2.0));
    let derived = inf_vgeom::derived_vmesh_id(mesh_id);
    assert!(proj.db().contains(derived), "the fixture has a DAG");
    let (before_verts, _) = derived_probe(root.path(), mesh_id);

    // A save that empties the mesh: the DAG that described the cube may not
    // survive it in its old form.
    let mut empty = inf_dcc::cube(2.0);
    let faces: Vec<FaceId> = empty.face_ids().collect();
    for f in faces {
        inf_dcc::ops::apply(&mut empty, &Op::RemoveFace { face: f }).unwrap();
    }
    assert_eq!(empty.face_count(), 0);
    let outcome = dcc::save_mesh_session(&mut proj, mesh_id, &empty);
    drop(proj);

    let proj = AssetProject::open(root.path()).unwrap();
    match outcome {
        Ok(_) => {
            // The derivation ran (or skipped). Either way nothing may still
            // describe the cube.
            if proj.db().contains(derived) {
                let (verts_now, _) = derived_probe(root.path(), mesh_id);
                assert_ne!(
                    verts_now, before_verts,
                    "the DAG still describes the mesh that was replaced"
                );
            }
        }
        Err(e) => {
            // A failure must have taken the stale artifact with it, and must say
            // what disk holds.
            assert!(
                !proj.db().contains(derived) || matches!(e, dcc::SaveError::Torn { .. }),
                "a failed derivation left a stale DAG without saying so: {e}"
            );
            assert!(
                e.to_string().contains("saved"),
                "the error names the state: {e}"
            );
        }
    }
}

/// **`SaveError::Torn`, provoked.**
///
/// The one state in which something can still draw the previous geometry, and
/// until the verdict was checked against the *filesystem* it was unreachable by
/// any failure at all: `AssetProject::delete` drops both `remove_file` results
/// and returns `Ok` unconditionally, so the old `.is_ok()` was a database
/// question. An author whose `.inf_vmesh` could not be unlinked was told it had
/// been, and the viewport went on drawing from it.
///
/// Windows only, and that is the honest scope rather than a convenience: an open
/// handle blocks `DeleteFile` there (Rust's `File::open` does not pass
/// `FILE_SHARE_DELETE`), which is exactly the renderer-holds-the-mmap case P16
/// lives with. POSIX unlinks a mapped file happily, so on Linux and macOS this
/// state is not reachable through the filesystem at all — which is worth knowing
/// rather than papering over with a `#[ignore]`.
#[cfg(windows)]
#[test]
fn a_save_that_cannot_remove_the_stale_dag_reports_torn_and_says_so() {
    use std::os::windows::fs::OpenOptionsExt;

    let root = tempfile::tempdir().unwrap();
    let mut proj = AssetProject::open(root.path()).unwrap();
    let mesh_id = write_mesh(&mut proj, "Prop", &inf_dcc::cube(2.0));
    let derived = inf_vgeom::derived_vmesh_id(mesh_id);
    let path = proj
        .db()
        .get(derived)
        .expect("the fixture has a DAG")
        .path
        .clone();

    // Hold the DAG open the way a renderer holding it mapped does — **without
    // `FILE_SHARE_DELETE`**. Rust's `File::open` passes all three share flags by
    // default, so a plain open does not block `DeleteFile` at all and the first
    // version of this test proved nothing (it measured a successful removal and
    // called it a locked file). The share mode is the whole fixture.
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    let _held = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ)
        .open(&path)
        .expect("the DAG is readable");

    // A mesh with no faces derives nothing, so the save must remove the DAG that
    // describes the cube — and cannot.
    let mut empty = inf_dcc::cube(2.0);
    let faces: Vec<FaceId> = empty.face_ids().collect();
    for f in faces {
        inf_dcc::ops::apply(&mut empty, &Op::RemoveFace { face: f }).unwrap();
    }
    let err = dcc::save_mesh_session(&mut proj, mesh_id, &empty)
        .expect_err("the stale DAG cannot be removed while it is held open");

    assert!(
        matches!(err, dcc::SaveError::Torn { .. }),
        "expected Torn, got {err}"
    );
    // The message is the contract: it must NOT claim the DAG is gone.
    let text = err.to_string();
    assert!(text.contains("could not be removed"), "{text}");
    assert!(text.contains("PREVIOUS geometry"), "{text}");
    assert!(
        !text.contains("has been removed"),
        "the error claims a removal that did not happen: {text}"
    );
    // And the claim is true: the file really is still there.
    assert!(path.exists(), "the held file must still be on disk");
}

/// The other side of the same coin: with nothing holding it, the removal really
/// happens and the verdict is the honest one.
///
/// Without this the disk check could be inverted (`path.exists()`) and every
/// save would report `Torn` — which the test above would happily accept.
#[test]
fn a_save_that_can_remove_the_stale_dag_does_and_says_nothing_about_tearing() {
    let root = tempfile::tempdir().unwrap();
    let mut proj = AssetProject::open(root.path()).unwrap();
    let mesh_id = write_mesh(&mut proj, "Prop", &inf_dcc::cube(2.0));
    let derived = inf_vgeom::derived_vmesh_id(mesh_id);
    let path = proj
        .db()
        .get(derived)
        .expect("the fixture has a DAG")
        .path
        .clone();

    let mut empty = inf_dcc::cube(2.0);
    let faces: Vec<FaceId> = empty.face_ids().collect();
    for f in faces {
        inf_dcc::ops::apply(&mut empty, &Op::RemoveFace { face: f }).unwrap();
    }
    let out = dcc::save_mesh_session(&mut proj, mesh_id, &empty);
    assert!(out.is_ok(), "nothing is holding the file: {out:?}");
    assert!(!path.exists(), "the obsolete DAG is gone from disk");
    assert!(!proj.db().contains(derived));
}
