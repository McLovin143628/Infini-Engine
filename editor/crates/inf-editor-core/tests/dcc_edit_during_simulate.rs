//! **THE dcc-vision HEADLINE, executed** (P23.6): a mesh is opened, edited and
//! **saved while Simulate is running**, and all four halves of that claim are
//! measured rather than reasoned about.
//!
//! `dcc_live_in_scene.rs` proves the save→re-key wire with the world standing
//! still. This file runs the same wire *through* a live [`SimSession`], which is
//! the case the P23.1 memo (§3) built the whole asset-scoped ruling on:
//!
//! > `SimSession::exit` reverts *the document*. Asset edits are not in the
//! > snapshot, because they are not in the document. So a mesh edited while
//! > Simulate runs survives Stop.
//!
//! Four claims, one per assertion block, in the order they have to hold:
//!
//! * **(a) the simulation is undisturbed.** The step trace of a run with a save
//!   spliced into the middle of it is compared **bit for bit** against a control
//!   run that never saved anything. Not "close" — `f64::to_bits`, because a
//!   comparison that passes within 1e-9 accepts exactly the drift this exists to
//!   catch.
//! * **(b) the editor re-keys mid-Simulate.** The resolve boundary
//!   ([`EditorRenderAssets`]) is driven *while the session is alive*, and the
//!   content-hash render key must move. There is nothing sim-shaped in that path
//!   and this is what makes that a measurement rather than a claim.
//! * **(c) Stop restores the DOCUMENT byte-for-byte and the ASSET keeps the
//!   edit.** Both halves, because either alone is satisfiable by a save that did
//!   nothing (the document is trivially restored) or by a Stop that did nothing.
//! * **(d) the journal outlives the session.** Undo in the Model Editor after
//!   Stop still walks back to the import baseline, because a `MeshSession` has no
//!   relationship with `SimSession` at all — which is a *structural* fact and is
//!   asserted as one.
//!
//! # What this file cannot execute, and what covers it instead
//!
//! The link between "the save finished" and "the viewport's store was told" is a
//! `#[tauri::command]` — `dcc_save`'s `viewport.refresh_asset_index(Target::All)`
//! — and a Tauri command cannot be driven from a test on any CI leg. So the
//! chain is split the way this codebase already splits `projector_mirror` and
//! `viewport_pump_mirror`: the two *ends* are executed here, and the middle is
//! held by a **source-scope gate** in `commands/dcc.rs`'s own test module
//! (`the_save_pushes_its_invalidation_unconditionally`), which fails if that push
//! is ever put behind a condition — a Simulate gate included. Between them there
//! is no unread link.

use glam::{DVec2, DVec3};
use inf_asset::{AssetId, AssetKind};
use inf_dcc::{FaceId, MeshSession, Op};
use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, MeshRef, RigidBody3D, Transform,
};
use inf_editor_core::assets::{vmesh, AssetProject};
use inf_editor_core::dcc;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::render_assets::EditorRenderAssets;
use inf_editor_core::scene::{serialize, SceneDoc};
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};
use uuid::Uuid;

const PROP_GUID: u128 = 0x2306_0000_0000_4000_8000_0000_0000_0002;
const GROUND_GUID: u128 = 0x2306_0000_0000_4000_8000_0000_0000_0003;

/// Steps traced. 90 at 60 Hz is 1.5 s: the prop falls 3 m, lands and settles, so
/// the trace contains motion, contact and rest. The save is spliced at
/// [`SAVE_AT`], which is *during the fall* — the part of the run where a
/// perturbation would be loudest.
const STEPS: usize = 90;
const SAVE_AT: usize = 30;

macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        let e = $doc.entity_of($guid).expect("entity");
        $doc.world_mut().world_mut().entity_mut(e).insert($comp);
        $doc.world_mut().mark_dirty();
    }};
}

/// Write `mesh` as a `.inf_mesh` and derive its DAG — the fixture door, identical
/// to `dcc_live_in_scene.rs`'s.
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

/// The scene: a static floor and one dynamic prop that **references the mesh
/// under edit**.
///
/// The reference is load-bearing. A prop with no `MeshRef.asset` would make claim
/// (a) vacuous — of course an unrelated simulation is undisturbed. This one draws
/// the very asset the save rewrites, and its collider is *authored* rather than
/// derived from the mesh, which is precisely why the edit cannot reach it.
fn workshop_doc(mesh: Uuid) -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.set_title("DCC Edit During Simulate");
    let ground = Uuid::from_u128(GROUND_GUID);
    let prop = Uuid::from_u128(PROP_GUID);

    doc.create_with_guid(ground, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        ground,
        Transform::from_translation(DVec3::new(0.0, -0.5, 0.0))
    );
    insert!(
        doc,
        ground,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        ground,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: inf_ecs::math::Vec3d::new(20.0, 0.5, 20.0),
            ..Default::default()
        }
    );

    doc.create_with_guid(prop, SpawnKind::Empty, "Prop", None);
    insert!(
        doc,
        prop,
        Transform::from_translation(DVec3::new(0.0, 4.0, 0.0))
    );
    insert!(
        doc,
        prop,
        MeshRef {
            asset: Some(mesh),
            ..Default::default()
        }
    );
    insert!(
        doc,
        prop,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        prop,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: inf_ecs::math::Vec3d::splat(1.0),
            density: 500.0,
            ..Default::default()
        }
    );
    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// One step of the trace: the prop's world translation, as raw bits.
fn sample(doc: &SceneDoc) -> [u64; 3] {
    let e = doc
        .entity_of(Uuid::from_u128(PROP_GUID))
        .expect("the prop is in the document");
    let t = doc
        .world()
        .world()
        .get::<Transform>(e)
        .expect("the prop has a transform")
        .translation;
    [t.x.to_bits(), t.y.to_bits(), t.z.to_bits()]
}

/// The document as the bytes a save would write. `ScenePersist::Memory` is what
/// `SimSession::enter` snapshots with, so this is the same projection the restore
/// is defined against.
fn doc_bytes(doc: &SceneDoc) -> Vec<u8> {
    serialize::encode(&serialize::to_scene_file_for(
        doc,
        serialize::ScenePersist::Memory,
    ))
    .expect("the document encodes")
}

/// Enter a session over a fresh copy of the scene and run it, calling `mid` once,
/// after step [`SAVE_AT`]. Returns the trace and the session's own exit verdict.
fn run<F: FnMut()>(mesh: Uuid, mut mid: F) -> (Vec<[u64; 3]>, Vec<u8>, Vec<u8>) {
    let mut doc = workshop_doc(mesh);
    let before = doc_bytes(&doc);
    let mut session = SimSession::enter(&mut doc, vec![], DVec2::new(0.0, -9.81), SIM_HZ);
    let mut trace = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        session.step_once(&mut doc, SimInput::default());
        trace.push(sample(&doc));
        if step == SAVE_AT {
            mid();
        }
    }
    // The document really moved while it ran — without this, "restored
    // byte-for-byte" is satisfied by a simulation that did nothing at all.
    let during = doc_bytes(&doc);
    assert_ne!(
        during, before,
        "the document did not change during Simulate, so restoring it proves nothing"
    );
    session.exit(&mut doc);
    let after = doc_bytes(&doc);
    (trace, before, after)
}

/// The top of the box, as two triangles (a `MeshAsset` carries no quads).
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

/// Open the asset, extrude its lid 1.5 m and save through the product path.
/// Returns the session, so the caller can go on undoing after Stop.
fn edit_and_save(root: &std::path::Path, mesh_id: AssetId) -> MeshSession {
    let mut proj = AssetProject::open(root).unwrap();
    let payload: inf_mesh::MeshAsset = proj.load_payload(mesh_id).unwrap();
    let import = inf_dcc::from_mesh_asset(&payload).expect("the kernel reads its own writer");
    let mut session = MeshSession::new(import.mesh);
    let top = top_faces(session.mesh(), 1.0);
    assert_eq!(top.len(), 2, "the lid is two triangles");
    session
        .apply(Op::ExtrudeFaces {
            faces: top,
            distance: 1.5,
        })
        .expect("extrude");
    let saved = dcc::save_mesh_session(&mut proj, mesh_id, session.mesh()).expect("saves");
    assert!(
        matches!(saved.vmesh, vmesh::VmeshDerivation::Built(_)),
        "the derivation ran synchronously in the same call: {:?}",
        saved.vmesh
    );
    session
}

// ── (a) + (b) + (c) ─────────────────────────────────────────────────────────

#[test]
fn a_mesh_saved_mid_simulate_leaves_the_run_untouched_and_survives_stop() {
    let root = tempfile::tempdir().unwrap();
    let mut proj = AssetProject::open(root.path()).unwrap();
    let mesh_id = write_mesh(&mut proj, "Prop", &inf_dcc::cube(2.0));
    drop(proj);

    // What the scene draws before anything happens.
    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(root.path().to_path_buf()));
    let key_before = store
        .resolve_vgeom(mesh_id.uuid())
        .expect("a scene referencing this mesh resolves real geometry")
        .id;

    // ── the CONTROL run: identical scene, identical steps, no save ──────────
    let (control, control_before, control_after) = run(mesh_id.uuid(), || {});
    assert_eq!(
        control_before, control_after,
        "Stop must restore the document byte-for-byte even with nothing else \
         going on — if this fails, nothing below means anything"
    );

    // ── the EXPERIMENT run: the same, with a real save spliced into it ──────
    let mut key_after = key_before;
    let mut re_keyed_mid_run = false;
    let root_path = root.path().to_path_buf();
    let (traced, before, after) = run(mesh_id.uuid(), || {
        edit_and_save(&root_path, mesh_id);
        // **(b) the resolve boundary, driven WHILE the session is alive.** This
        // is the editor's store, and it re-keys because the render key is the
        // payload's content hash — new bytes are a new key and there is nothing
        // between them that knows a simulation exists.
        store.refresh_index();
        key_after = store
            .resolve_vgeom(mesh_id.uuid())
            .expect("still resolves after the rewrite")
            .id;
        re_keyed_mid_run = true;
    });
    assert!(re_keyed_mid_run, "the mid-run hook never fired");
    assert_ne!(
        key_after, key_before,
        "the editor kept the OLD render key across a save made during Simulate — \
         the content-hash re-key is what makes a stale draw unrepresentable, and \
         it did not happen"
    );

    // ── (a) the simulation is undisturbed, bit for bit ──────────────────────
    assert_eq!(
        traced, control,
        "the step trace diverged from the control run: saving an asset reached \
         the simulation"
    );
    // ANTI-VACUITY: the trace has to be a trace. A world that never moved would
    // compare equal to any other world that never moved.
    assert!(
        traced.windows(2).any(|w| w[0] != w[1]),
        "the prop never moved, so 'the trace is identical' says nothing"
    );
    assert!(
        traced.first() != traced.last(),
        "the prop ended where it started"
    );

    // ── (c) the DOCUMENT is restored, the ASSET keeps the edit ──────────────
    assert_eq!(
        before, after,
        "Stop did not restore the document byte-for-byte after a mid-run save"
    );
    let proj = AssetProject::open(root.path()).unwrap();
    let reread: inf_mesh::MeshAsset = proj.load_payload(mesh_id).unwrap();
    assert!(
        (reread.bounds.max[1] - 2.5).abs() < 1e-5,
        "the asset lost the edit at Stop: it tops out at {} m, not 2.5 — \
         SimSession::exit reverted something that was never in the document",
        reread.bounds.max[1]
    );
    // And the DERIVED artifact moved with it, so the viewport after Stop draws
    // the edited surface rather than the geometry the session started on.
    let derived = inf_vgeom::derived_vmesh_id(mesh_id);
    let bytes = std::fs::read(&proj.db().get(derived).expect("a derived DAG").path).unwrap();
    let dag = inf_vgeom::VgeomSource::from_payload(bytes)
        .expect("a readable DAG")
        .to_mesh()
        .expect("the DAG decodes");
    let top = dag
        .vertices
        .iter()
        .map(|v| v.position[1])
        .fold(f32::MIN, f32::max);
    assert!(
        (top - 2.5).abs() < 1e-5,
        "the derived geometry tops out at {top} m, not 2.5"
    );
}

// ── (d) ─────────────────────────────────────────────────────────────────────

#[test]
fn the_op_journal_outlives_the_simulate_session() {
    // **The claim is structural**: a `MeshSession` is an asset document's journal
    // and a `SimSession` is a level's play state, and nothing connects them. So
    // this drives the whole lifecycle in the order an author would — Play, edit,
    // Save, Stop, Ctrl+Z — and asserts the undo lands on the *import baseline*,
    // byte for byte, rather than merely "somewhere earlier".
    let root = tempfile::tempdir().unwrap();
    let mut proj = AssetProject::open(root.path()).unwrap();
    let mesh_id = write_mesh(&mut proj, "Prop", &inf_dcc::cube(2.0));
    drop(proj);

    let mut doc = workshop_doc(mesh_id.uuid());
    let mut sim = SimSession::enter(&mut doc, vec![], DVec2::new(0.0, -9.81), SIM_HZ);
    for _ in 0..SAVE_AT {
        sim.step_once(&mut doc, SimInput::default());
    }

    let mut session = edit_and_save(root.path(), mesh_id);
    let baseline = session.base().encoded();
    let edited = session.mesh().encoded();
    assert_ne!(baseline, edited, "the extrude changed the mesh");

    // Stop. The journal is not consulted, not cleared and not aware.
    sim.exit(&mut doc);

    assert!(
        session.can_undo(),
        "the journal survived Stop with its cursor"
    );
    assert!(session.undo(), "undo after Stop");
    assert_eq!(
        session.mesh().encoded(),
        baseline,
        "undo after Stop landed somewhere other than the import baseline"
    );
    assert!(session.redo(), "redo after Stop");
    assert_eq!(
        session.mesh().encoded(),
        edited,
        "redo after Stop did not return to the edited state"
    );

    // …and undoing then saving really puts the baseline back on disk, which is
    // the half an author would notice.
    assert!(session.undo());
    let mut proj = AssetProject::open(root.path()).unwrap();
    dcc::save_mesh_session(&mut proj, mesh_id, session.mesh()).expect("saves");
    let reread: inf_mesh::MeshAsset = proj.load_payload(mesh_id).unwrap();
    assert!(
        (reread.bounds.max[1] - 1.0).abs() < 1e-5,
        "the undone save left {} m on disk, not the 1 m cube",
        reread.bounds.max[1]
    );
}

/// **The other half of (b)**, and the reason the re-key above can be trusted: the
/// store's key moves when the bytes move and *not otherwise*, even mid-Simulate.
///
/// Without this, a store that minted a fresh key on every `refresh_index` would
/// satisfy the assertion above for a save that had written nothing at all.
#[test]
fn a_no_op_save_during_simulate_leaves_the_render_key_alone() {
    let root = tempfile::tempdir().unwrap();
    let mut proj = AssetProject::open(root.path()).unwrap();
    let mesh_id = write_mesh(&mut proj, "Prop", &inf_dcc::cube(2.0));
    drop(proj);

    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(root.path().to_path_buf()));
    let before = store.resolve_vgeom(mesh_id.uuid()).unwrap().id;

    let mut doc = workshop_doc(mesh_id.uuid());
    let mut sim = SimSession::enter(&mut doc, vec![], DVec2::new(0.0, -9.81), SIM_HZ);
    for _ in 0..SAVE_AT {
        sim.step_once(&mut doc, SimInput::default());
    }

    let mut proj = AssetProject::open(root.path()).unwrap();
    let payload: inf_mesh::MeshAsset = proj.load_payload(mesh_id).unwrap();
    let import = inf_dcc::from_mesh_asset(&payload).unwrap();
    let session = MeshSession::new(import.mesh);
    dcc::save_mesh_session(&mut proj, mesh_id, session.mesh()).expect("saves");
    drop(proj);

    store.refresh_index();
    assert_eq!(
        store.resolve_vgeom(mesh_id.uuid()).unwrap().id,
        before,
        "open-then-save must be a no-op — a key that moved anyway would make the \
         live-update claim vacuous"
    );
    sim.exit(&mut doc);
}
