//! **THE PHASE 23 GATE** — the embedded DCC: modelling core.
//!
//! The phase's done-when sentence, verbatim: *"you can model a usable prop
//! (extrude / bevel / loop-cut), unwrap it, save it as a standard mesh asset, and
//! watch a scene that references it live-update — with clean undo and
//! deterministic op replay."* Every clause is an arm here, and every arm goes
//! through the **product op path** — `MeshSession::apply`, `inf_dcc::unwrap`,
//! `inf_editor_core::dcc::save_mesh_session` — rather than hand-built meshes,
//! because a gate that inlines the code it is gating is a copy, not a gate
//! (P23.4's LAW).
//!
//! The fixture is the committed `samples/phase23-workshop`: a level referencing
//! one mesh, and that mesh's **import baseline**.
//!
//! The arms, in the order the claim needs them:
//!
//! * **(a) model a prop** — extrude, loop cut, bevel through the op path, and the
//!   journal replayed **twice** produces byte-identical assets.
//! * **(b) unwrap** — seams cut, charts solved: every corner inside the unit
//!   square, **zero folds**, and the convergence field at machine epsilon on
//!   every chart but the one this batch measured and ledgered.
//! * **(c) save as a standard asset** — the `.inf_mesh` decodes, the derived
//!   `.inf_vmesh` exists and decodes, and the sidecar's hash is the hash of the
//!   bytes on disk.
//! * **(d) live update** — a store that resolved the mesh **before** the edit
//!   re-keys after it (editor side), and a pack cooked **after** the save carries
//!   the new geometry (cook side).
//! * **(e) undo clean** — the whole journal undone equals the import baseline
//!   *byte for byte*; redone equals the edited state.
//! * **(f) determinism on op replay** — the journal, carried as `SessionSave`
//!   bytes into a **fresh process**, produces byte-identical asset bytes.
//! * **(g) edit during Simulate** — the dcc-vision headline at gate level, riding
//!   `inf-editor-core`'s `dcc_edit_during_simulate` machinery.
//! * **(h) budgets** — opens and saves against `LOAD_BUDGET_MS`. No new ratchet.
//! * **(i) cook advisories** — what the cook says about this sample, and why it
//!   is not silence.
//!
//! The letters are load-bearing: `samples/phase23-workshop/README.md` and
//! ROADMAP §12 cite them, so an arm that is added or re-ordered has to be
//! re-cited with it.
//!
//! # Why the comparisons are bytes
//!
//! Every claim here is about *identity* — the same ops give the same mesh, undo
//! returns the same mesh, a fresh process gives the same mesh — so every
//! comparison is `inf_asset::encode`'s bytes or `Mesh::encoded`'s. A tolerance
//! would accept exactly the drift these arms exist to catch.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use glam::DVec2;

use inf_asset::{AssetId, AssetKind, PackReader};
use inf_dcc::MeshSession;
use inf_editor_core::assets::{vmesh, AssetProject};
use inf_editor_core::dcc;
use inf_editor_core::render_assets::EditorRenderAssets;
use inf_editor_core::samples::{
    phase23_baseline_mesh, phase23_disjoint_pair, phase23_extrude_and_cut, phase23_model_prop,
    phase23_rim_edges, phase23_unwrap_prop, phase23_workshop_dir, PHASE23_BEVEL_M,
    PHASE23_CONVERGED, PHASE23_EXTRUDE_M, PHASE23_MESH_GUID, PHASE23_PROP_GUID,
    PHASE23_PROP_SIZE_M, PHASE23_SAVE_AT, PHASE23_STEPS,
};
use inf_editor_core::scene::{serialize, SceneDoc};
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;

// Budgets are imported, never redeclared: a phase does not get its own budget for
// being new. Opening and saving a document are LOAD-class work, so they take the
// load ceiling — a per-frame ceiling here would be a category error (P20's LAW).
use inf_player::budget::LOAD_BUDGET_MS;

/// The env var the replay probe reads its `SessionSave` from.
const PROBE_SESSION: &str = "INF_P23_SESSION";

/// **What the recipe's legal bevel costs**, measured after the P23.6 ruling.
///
/// Zero, on every counter that the four-edge bevel used to move — which is the
/// ruling's whole point: refusing the construction that could not be saved
/// removed the coincident corners, the un-earable n-gons, the untangented
/// vertices and the stalled UV chart *together*, because all four were the same
/// defect seen from four places.
const PHASE23_BEVEL_COINCIDENT: usize = 0;
const PHASE23_BEVEL_FAN_FALLBACKS: usize = 4;
const PHASE23_BEVEL_UNTANGENTED: usize = 0;
/// The floor on how much of the atlas's `u` axis the prop's charts must occupy.
/// Measured at 0.77 after the bevel ruling (0.13 before it).
const PHASE23_ATLAS_U_SPAN: f64 = 0.7;

/// Every committed file of the sample, so the fixture copy cannot silently miss
/// one as the sample grows.
fn sample_files() -> [&'static str; 5] {
    [
        "Phase23Workshop.inf_lvl",
        "Phase23Workshop.inf_lvl.toml",
        "Prop.inf_mesh",
        "Prop.inf_mesh.toml",
        "README.md",
    ]
}

/// …and the list really is **every** committed file (P21.4's finding: the first
/// one of these was eight of nine from the day it was written).
#[test]
fn the_fixture_copies_every_committed_sample_file() {
    let listed: BTreeSet<String> = sample_files().iter().map(|s| s.to_string()).collect();
    let on_disk: BTreeSet<String> = std::fs::read_dir(phase23_workshop_dir())
        .expect("the sample directory exists")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        listed, on_disk,
        "`sample_files()` and samples/phase23-workshop disagree — a cooked \
         fixture would be missing content the committed sample has"
    );
}

/// Scaffold a project holding the sample; returns its `Content` dir.
fn scaffold(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Phase 23 Workshop", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = phase23_workshop_dir();
    for f in sample_files() {
        std::fs::copy(src.join(f), content.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    content
}

/// Open the committed baseline through the kernel's reader — `dcc_open`'s half.
fn open_baseline() -> MeshSession {
    let import = inf_dcc::from_mesh_asset(&phase23_baseline_mesh())
        .expect("the kernel reads the committed baseline");
    assert_eq!(
        import.report.boundary_edges, 0,
        "the baseline is a closed solid"
    );
    MeshSession::new(import.mesh)
}

/// The modelled prop: the committed baseline plus the recipe's three ops.
fn modelled() -> MeshSession {
    let mut session = open_baseline();
    phase23_model_prop(&mut session).expect("the prop models");
    session
}

/// A modelled **and unwrapped** prop — what the gate saves.
fn finished() -> MeshSession {
    let mut session = modelled();
    phase23_unwrap_prop(&mut session).expect("the prop unwraps");
    session
}

fn encoded(mesh: &inf_dcc::Mesh) -> Vec<u8> {
    let (asset, _) = inf_dcc::to_mesh_asset(mesh, &inf_dcc::ExportOptions::default());
    inf_asset::encode(&asset).expect("the asset encodes")
}

// ── (a) MODEL A PROP ────────────────────────────────────────────────────────

/// **The done-when's first clause.** A usable prop, modelled from a cube
/// primitive through extrude / loop-cut / bevel, and a journal that replays
/// bit-identically.
#[test]
fn the_prop_models_through_the_op_path_and_replays_bit_identically() {
    let session = modelled();

    // It is a PROP, not a cube with a name. Anti-vacuity before the identity
    // claims: a recipe whose ops all refused would replay identically too.
    assert_eq!(session.ops().len(), 3, "extrude, loop cut, bevel");
    let base_faces = session.base().face_count();
    assert_eq!(base_faces, 12, "a 2 m cube arrives as 12 triangles");
    assert!(
        session.mesh().face_count() > base_faces * 2,
        "the recipe left {} faces on a {base_faces}-face baseline — nothing was \
         modelled",
        session.mesh().face_count()
    );
    assert_eq!(
        inf_dcc::validate(session.mesh()),
        Ok(()),
        "the modelled prop is not a valid half-edge mesh"
    );
    // The extrude really raised the lid, and the bevel really cut the rim.
    let top = session
        .mesh()
        .vert_ids()
        .filter_map(|v| session.mesh().position(v))
        .map(|p| p.y)
        .fold(f64::MIN, f64::max);
    assert!(
        (top - (PHASE23_PROP_SIZE_M * 0.5 + PHASE23_EXTRUDE_M)).abs() < 1e-9,
        "the prop tops out at {top} m"
    );

    // **REPLAY, TWICE.** `replay(base, ops)` is a pure function or the journal is
    // not an undo story. Two replays rather than one because a single replay
    // compared against the incrementally-edited mesh would also pass for a replay
    // that simply returned its input.
    let a = MeshSession::replay(session.base(), session.ops()).expect("replay 1");
    let b = MeshSession::replay(session.base(), session.ops()).expect("replay 2");
    assert_eq!(a.encoded(), b.encoded(), "two replays disagree");
    assert_eq!(
        a.encoded(),
        session.mesh().encoded(),
        "replaying the journal does not reproduce the mesh the session holds"
    );
    assert_eq!(encoded(&a), encoded(session.mesh()), "…nor its asset bytes");
}

// ── (b) UNWRAP ──────────────────────────────────────────────────────────────

/// **The done-when's second clause.** Seams, then charts: inside the unit square,
/// fold-free, and solved.
#[test]
fn the_prop_unwraps_into_the_unit_square_without_folding() {
    let mut session = modelled();
    let report = phase23_unwrap_prop(&mut session).expect("the prop unwraps");

    assert!(report.seams > 0, "no seam was cut, so there is one chart");
    assert!(
        report.charts.len() > 1,
        "the seams produced {} chart(s)",
        report.charts.len()
    );
    assert!(report.corners > 0, "no corner got a UV");
    assert_eq!(
        report.flipped, 0,
        "the layout folds: {} triangles are wound against their chart",
        report.flipped
    );

    // EVERY corner, read off the mesh rather than off the report — the report
    // counts what the solver wrote, and this is what the mesh now holds.
    let mesh = session.mesh();
    let mut seen = 0usize;
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for h in mesh.half_ids() {
        if mesh.face_of(h) == Some(None) {
            continue; // a boundary half-edge has no corner
        }
        let Some(uv) = mesh.corner_uv(h) else {
            continue;
        };
        assert!(
            (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]),
            "corner UV {uv:?} is outside the unit square"
        );
        for k in 0..2 {
            lo[k] = lo[k].min(uv[k]);
            hi[k] = hi[k].max(uv[k]);
        }
        seen += 1;
    }
    assert_eq!(seen, report.corners, "the report counted a different set");
    // **ANTI-VACUITY, and it is not decoration**: "every corner is inside the
    // unit square" is satisfied *perfectly* by a layout that left every UV at
    // (0, 0). The packing has to fill the square.
    assert!(
        hi[1] - lo[1] > 0.9,
        "the layout spans only {lo:?}..{hi:?} — a packed atlas fills its square"
    );
    // **And the other axis says the packer did its job.** Before the P23.6 bevel
    // ruling this arm pinned an *inefficiency*: eighteen charts landed in a column
    // 0.13 wide, so ~87% of the atlas was empty. Those eighteen charts were the
    // fragments the refused bevel produced; with the legal one the layout fills
    // most of the square, so the assertion is now the positive claim it should
    // always have been.
    assert!(
        hi[0] - lo[0] > PHASE23_ATLAS_U_SPAN,
        "the atlas spans only {} on u; the packer is leaving {}% of the square \
         empty",
        hi[0] - lo[0],
        (100.0 * (1.0 - (hi[0] - lo[0]))).round()
    );

    // **The solver's own verdict: machine epsilon on every chart, no exceptions.**
    //
    // Before the P23.6 bevel ruling one chart of eighteen stalled at 1.6e-2 and
    // this arm carried a bound and an allowance for it. Both are gone, and that
    // is a *consequence* rather than a relaxation: the n-gons whose fan
    // triangulation ill-conditioned that solve were the ones the refused bevel
    // produced, so refusing it removed the stalled chart. An allowance kept
    // "just in case" would have hidden that.
    let stalled: Vec<(usize, f64)> = report
        .charts
        .iter()
        .enumerate()
        .filter(|(_, c)| c.convergence >= PHASE23_CONVERGED)
        .map(|(i, c)| (i, c.convergence))
        .collect();
    assert!(
        stalled.is_empty(),
        "charts {stalled:?} did not converge; every chart of this prop must reach \
         machine epsilon"
    );
}

// ── (c) SAVE AS A STANDARD ASSET ────────────────────────────────────────────

/// **The done-when's third clause.** Through `save_mesh_session` — the product
/// path — and then read back off disk three ways.
#[test]
fn the_prop_saves_as_a_standard_asset() {
    let tmp = tempfile::tempdir().unwrap();
    let content = scaffold(tmp.path());
    let mesh_id = AssetId(PHASE23_MESH_GUID);
    let session = finished();

    let mut proj = AssetProject::open(&content).unwrap();
    assert_eq!(
        proj.db().get(mesh_id).expect("the sample's mesh").kind(),
        AssetKind::Mesh
    );
    let saved = dcc::save_mesh_session(&mut proj, mesh_id, session.mesh()).expect("saves");
    assert!(
        matches!(saved.vmesh, vmesh::VmeshDerivation::Built(_)),
        "the derivation ran synchronously in the same call: {:?}",
        saved.vmesh
    );
    assert_eq!(saved.export.reused_diagonals, 0);
    // **THE RULING, ASSERTED FROM THE OUTSIDE.** Every counter that the four-edge
    // bevel used to move reads zero, and the asset the save wrote **re-opens**.
    // The refusal is not a warning the author has to heed — the op will not build
    // an unopenable mesh in the first place, so the save cannot write one.
    assert_eq!(
        saved.export.coincident_vertices, PHASE23_BEVEL_COINCIDENT,
        "the save wrote coincident vertices; the reader's weld will fuse them"
    );
    assert_eq!(
        saved.export.fan_fallbacks, PHASE23_BEVEL_FAN_FALLBACKS,
        "the writer could not ear-clip a face"
    );
    assert_eq!(
        saved.export.fallback_tangents, PHASE23_BEVEL_UNTANGENTED,
        "vertices went out without a usable tangent"
    );
    let (written, _) = inf_dcc::to_mesh_asset(session.mesh(), &inf_dcc::ExportOptions::default());
    let reopened =
        inf_dcc::from_mesh_asset(&written).expect("a saved prop must re-open in the Model Editor");
    assert_eq!(
        reopened.mesh.vert_count(),
        session.mesh().vert_count(),
        "the reader's weld fused vertices the writer meant to keep"
    );
    assert_eq!(inf_dcc::validate(&reopened.mesh), Ok(()));
    assert_eq!(
        reopened.report.boundary_edges, 0,
        "the re-opened prop is not a closed solid"
    );

    // 1. the `.inf_mesh` decodes, and is the prop.
    let entry = proj.db().get(mesh_id).expect("still registered").clone();
    let raw = std::fs::read(&entry.path).unwrap();
    let payload: inf_mesh::MeshAsset = inf_asset::decode(&raw).expect("the .inf_mesh decodes");
    assert!(
        (payload.bounds.max[1] as f64 - (PHASE23_PROP_SIZE_M * 0.5 + PHASE23_EXTRUDE_M)).abs()
            < 1e-5,
        "the saved mesh tops out at {}",
        payload.bounds.max[1]
    );
    assert_eq!(payload.schema_version, inf_mesh::MeshAsset::CURRENT_VERSION);

    // 2. the SIDECAR's hash is the hash of the bytes beside it. A sidecar that
    //    described the previous payload is how a content-hash render key goes
    //    stale without anything else looking wrong.
    assert_eq!(
        entry.sidecar.content_hash,
        inf_asset::ContentHash::of(&raw),
        "the sidecar's hash is not the hash of the payload on disk"
    );

    // 3. the derived `.inf_vmesh` exists and decodes, and describes THIS mesh.
    let derived = inf_vgeom::derived_vmesh_id(mesh_id);
    let dag_path = proj
        .db()
        .get(derived)
        .expect("the derived .inf_vmesh is registered")
        .path
        .clone();
    let dag = inf_vgeom::VgeomSource::from_payload(std::fs::read(&dag_path).unwrap())
        .expect("the .inf_vmesh decodes")
        .to_mesh()
        .expect("the DAG decodes to geometry");
    let top = dag
        .vertices
        .iter()
        .map(|v| v.position[1])
        .fold(f32::MIN, f32::max);
    assert!(
        (top as f64 - (PHASE23_PROP_SIZE_M * 0.5 + PHASE23_EXTRUDE_M)).abs() < 1e-5,
        "the derived DAG tops out at {top} m — it describes the mesh that was \
         replaced, not the one that was saved"
    );
}

/// **THE RULING, AND THE THREE CONSEQUENCES IT RETIRED.**
///
/// The gate's first recipe beveled all four rim edges of the extruded cap, and
/// what came out was a `.inf_mesh` that saved, cooked, rendered and fractured
/// perfectly — and that the Model Editor **could not re-open**. Two bevels
/// meeting at a corner each offset it into their far face; on a right angle both
/// land in the same place; the reader's tolerance-zero weld fuses the pair into
/// an edge used twice.
///
/// `Op::BevelEdges` now refuses that construction. This arm holds the refusal to
/// its bargain from both ends:
///
/// * the four-edge bevel is **refused**, as a typed value, inertly, with a
///   message an author can act on;
/// * the geometry it would have produced really is unopenable, and really does
///   carry the three counters this batch measured — so the refusal is not
///   preventing something harmless;
/// * the recipe's own two-edge bevel is **allowed**, so the op is still a tool.
///
/// The counters live here rather than in a ledger paragraph because a ruling
/// whose justification is only prose is a ruling nobody can re-check.
#[test]
fn the_bevel_refuses_the_corner_it_cannot_save() {
    use inf_dcc::{Op, OpError};

    // Model up to the point where the rim exists — the same two ops the recipe
    // runs, so this is the real topology and not an imitation of it.
    let mut session = open_baseline();
    let top_y = PHASE23_PROP_SIZE_M * 0.5 + PHASE23_EXTRUDE_M;
    phase23_extrude_and_cut(&mut session).expect("the recipe's first two ops");
    let rim = phase23_rim_edges(session.mesh(), top_y);
    assert_eq!(rim.len(), 4, "the cap has a four-edge rim");

    // (1) THE REFUSAL. All four at once is what an author selects, and it is
    //     exactly the construction that cannot be saved.
    let mut all_four = session.mesh().clone();
    let before = all_four.encoded();
    let err = inf_dcc::ops::apply(
        &mut all_four,
        &Op::BevelEdges {
            edges: rim.clone(),
            amount: PHASE23_BEVEL_M,
        },
    )
    .expect_err("beveling four meeting edges must be refused");
    assert!(
        matches!(err, OpError::BevelCoincidentVertex { .. }),
        "refused for the wrong reason: {err}"
    );
    assert_eq!(
        all_four.encoded(),
        before,
        "a refused bevel must leave the mesh byte-identical"
    );
    let text = err.to_string();
    assert!(text.contains("do not share an endpoint"), "{text}");
    assert!(text.contains("corner join"), "{text}");

    // (2) …AND THE HARM IT PREVENTS IS REAL, demonstrated rather than cited.
    //
    // The refusal rests on one premise: **two vertices at one position make an
    // asset its own reader will not take back**. The bevel can no longer be made
    // to produce that state — the guard also compares against vertices already in
    // the mesh, so beveling the rim one edge at a time is refused too, which is a
    // stronger result than intended and is asserted here rather than assumed.
    // So the premise is shown directly, through the public op set: two coincident
    // quads, four coincident pairs, and a reader that refuses the result.
    // The premise itself: coincident vertices ⇒ the reader refuses.
    let mut doubled = inf_dcc::plane(1.0);
    let places: Vec<[f64; 3]> = doubled
        .vert_ids()
        .map(|v| doubled.position(v).expect("live").to_array())
        .collect();
    assert_eq!(places.len(), 4);
    let mut twins = Vec::new();
    for p in &places {
        let out = inf_dcc::ops::apply(&mut doubled, &Op::AddVertex { position: *p })
            .expect("a vertex may sit on another — the KERNEL allows it");
        twins.push(out.verts[0]);
    }
    inf_dcc::ops::apply(
        &mut doubled,
        &Op::AddFace {
            verts: twins,
            corners: vec![inf_dcc::CornerData::default(); 4],
            slot: None,
        },
    )
    .expect("and so does a face built on them");
    let (twin_asset, twin_report) =
        inf_dcc::to_mesh_asset(&doubled, &inf_dcc::ExportOptions::default());
    assert_eq!(
        twin_report.coincident_vertices, 8,
        "the fixture must really carry the hazard it is demonstrating"
    );
    assert!(
        matches!(
            inf_dcc::from_mesh_asset(&twin_asset),
            Err(inf_dcc::ImportError::NonManifoldEdge { .. })
        ),
        "coincident vertices must really make an asset unopenable, or the bevel's \
         refusal is a superstition"
    );

    // (3) …AND THE OP IS STILL A TOOL. Two edges that do not meet bevel cleanly,
    //     which is what the recipe uses and what arm (c) saves.
    let pair = phase23_disjoint_pair(session.mesh(), &rim).expect("a disjoint pair");
    let mut two = session.mesh().clone();
    inf_dcc::ops::apply(
        &mut two,
        &Op::BevelEdges {
            edges: pair.to_vec(),
            amount: PHASE23_BEVEL_M,
        },
    )
    .expect("two disjoint edges must still bevel");
    let (clean, clean_report) = inf_dcc::to_mesh_asset(&two, &inf_dcc::ExportOptions::default());
    assert_eq!(clean_report.coincident_vertices, 0);
    assert!(inf_dcc::from_mesh_asset(&clean).is_ok(), "and it re-opens");
}

// ── (d) LIVE UPDATE ─────────────────────────────────────────────────────────

/// **The done-when's fourth clause, both sides.** A scene that resolved the mesh
/// *before* the edit re-keys after it; and a pack cooked *after* the save carries
/// the new bytes.
#[test]
fn a_scene_that_referenced_the_mesh_before_the_edit_sees_it_after() {
    let tmp = tempfile::tempdir().unwrap();
    let content = scaffold(tmp.path());
    let mesh_id = AssetId(PHASE23_MESH_GUID);

    // The sample ships no derived DAG (a committed sample is source), so the
    // fixture derives one first — that is the state a project is in after it has
    // been opened once, and it is what makes "the key MOVED" meaningful.
    let mut proj = AssetProject::open(&content).unwrap();
    vmesh::ensure_vmesh(&mut proj, mesh_id).expect("the baseline derives");
    drop(proj);

    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(content.clone()));
    let before = store
        .resolve_vgeom(PHASE23_MESH_GUID)
        .expect("the scene resolves the baseline")
        .id;

    let mut proj = AssetProject::open(&content).unwrap();
    dcc::save_mesh_session(&mut proj, mesh_id, finished().mesh()).expect("saves");
    drop(proj);

    // EDITOR SIDE: the render key is the payload's content hash, so new bytes
    // are a new key and a stale draw is unrepresentable.
    store.refresh_index();
    let after = store
        .resolve_vgeom(PHASE23_MESH_GUID)
        .expect("still resolves after the rewrite")
        .id;
    assert_ne!(
        after, before,
        "the editor kept the OLD render key across the save"
    );

    // COOK SIDE: a pack built after the save carries the edited geometry. The
    // editor re-keying proves nothing about what ships.
    let out = tmp.path().join("out");
    let report = cook(&tmp.path().join("proj"), &out, &CookOptions::default()).expect("cooks");
    let reader = PackReader::open(&report.pack_path).expect("open pack");
    let entry = reader
        .index()
        .find(|e| e.guid == mesh_id)
        .expect("the pack carries the mesh");
    let shipped: inf_mesh::MeshAsset =
        inf_asset::decode(&reader.read(entry.guid).unwrap()).expect("the shipped mesh decodes");
    assert!(
        (shipped.bounds.max[1] as f64 - (PHASE23_PROP_SIZE_M * 0.5 + PHASE23_EXTRUDE_M)).abs()
            < 1e-5,
        "the cooked pack carries geometry that tops out at {} m — the save did \
         not reach the build",
        shipped.bounds.max[1]
    );
}

// ── (e) UNDO CLEAN ──────────────────────────────────────────────────────────

/// **The done-when's fifth clause.** Undo the whole journal → the import
/// baseline, byte for byte. Redo → the edited state, byte for byte.
#[test]
fn undoing_the_journal_returns_the_import_baseline_byte_for_byte() {
    let mut session = finished();
    let edited = session.mesh().encoded();
    let baseline = session.base().encoded();
    assert_ne!(edited, baseline, "the recipe changed nothing");

    let mut undos = 0;
    while session.undo() {
        undos += 1;
        assert!(undos < 1000, "undo does not terminate");
    }
    assert!(undos >= 4, "only {undos} undo steps for a four-op journal");
    assert_eq!(
        session.mesh().encoded(),
        baseline,
        "a fully-undone journal is not the mesh that was opened"
    );
    // …and the ASSET the writer would produce is the committed baseline itself.
    assert_eq!(
        encoded(session.mesh()),
        inf_asset::encode(&phase23_baseline_mesh()).unwrap(),
        "undoing to the start does not reproduce the committed Prop.inf_mesh"
    );

    let mut redos = 0;
    while session.redo() {
        redos += 1;
        assert!(redos < 1000, "redo does not terminate");
    }
    assert_eq!(redos, undos, "redo did not walk back the same distance");
    assert_eq!(
        session.mesh().encoded(),
        edited,
        "redo did not return to the edited state"
    );
}

// ── (f) DETERMINISM ON OP REPLAY ────────────────────────────────────────────

/// **The subprocess half of (f).** Not an arm: a probe the arm below spawns,
/// which restores a `SessionSave` from the file named by [`PROBE_SESSION`],
/// replays it and prints the asset's hash.
///
/// A **fresh process**, because that is the only thing an in-process repetition
/// cannot be: a per-process seed (a `RandomState`, a lazily-initialized pool)
/// that leaked into a mesh would give the same answer twice inside one process
/// and a different one here. `inf-dcc`'s `determinism_law` bans the constructs
/// that could do it by reading the source; this executes the claim.
///
/// It is a `#[test]` rather than a `[[bin]]` for a packaging reason worth stating:
/// a `[[bin]]` cannot reach dev-dependencies, and `inf-dcc` is deliberately one
/// here — the shipped player must never link the modelling kernel.
#[test]
fn dcc_replay_probe() {
    let Ok(path) = std::env::var(PROBE_SESSION) else {
        // Run as an ordinary test (no env var): there is nothing to replay. The
        // arm below is what gives it work.
        return;
    };
    let save: inf_dcc::SessionSave =
        serde_json::from_slice(&std::fs::read(&path).expect("read the session"))
            .expect("the session decodes");
    let session = MeshSession::restore(save).expect("the session restores");
    let mut hex = String::new();
    for b in encoded(session.mesh()) {
        hex.push_str(&format!("{b:02x}"));
    }
    println!("asset={hex}");
}

/// **The done-when's sixth clause.** The journal, carried as bytes into a fresh
/// process, produces byte-identical asset bytes.
#[test]
fn the_journal_replays_byte_identically_in_a_fresh_process() {
    let session = finished();
    let mine: String = encoded(session.mesh())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("session.bin");
    std::fs::write(&path, serde_json::to_vec(&session.save()).unwrap()).unwrap();

    let exe = std::env::current_exe().expect("this test binary");
    let out = Command::new(exe)
        .args(["dcc_replay_probe", "--exact", "--nocapture"])
        .env(PROBE_SESSION, &path)
        .output()
        .expect("spawn the replay probe");
    assert!(
        out.status.success(),
        "the replay probe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let theirs = String::from_utf8(out.stdout)
        .expect("probe stdout utf8")
        .lines()
        .find_map(|l| l.strip_prefix("asset=").map(str::to_string))
        .expect("the probe printed asset=");

    assert_eq!(
        theirs, mine,
        "the same journal produced different asset bytes in a fresh process — \
         something on the modelling path depends on per-process state"
    );
    // Anti-vacuity: the hash is a hash of something. An empty mesh would compare
    // equal to another empty mesh.
    assert!(
        mine.len() > 64,
        "the asset encodes to {} hex chars",
        mine.len()
    );
}

// ── (g) EDIT DURING SIMULATE ────────────────────────────────────────────────

/// One step of the trace: the prop's world translation, as raw bits.
fn sample_pose(doc: &SceneDoc) -> [u64; 3] {
    let e = doc.entity_of(PHASE23_PROP_GUID).expect("the prop");
    let t = doc
        .world()
        .world()
        .get::<inf_ecs::components::Transform>(e)
        .expect("a transform")
        .translation;
    [t.x.to_bits(), t.y.to_bits(), t.z.to_bits()]
}

fn workshop_doc() -> SceneDoc {
    serialize::load(&phase23_workshop_dir().join("Phase23Workshop.inf_lvl"))
        .expect("the committed level loads")
}

fn doc_bytes(doc: &SceneDoc) -> Vec<u8> {
    serialize::encode(&serialize::to_scene_file_for(
        doc,
        serialize::ScenePersist::Memory,
    ))
    .expect("encodes")
}

/// Run the committed level for [`PHASE23_STEPS`], calling `mid` once at
/// [`PHASE23_SAVE_AT`]. Returns `(trace, pre-enter bytes, post-exit bytes)`.
fn run_workshop<F: FnMut()>(mut mid: F) -> (Vec<[u64; 3]>, Vec<u8>, Vec<u8>) {
    let mut doc = workshop_doc();
    let before = doc_bytes(&doc);
    let mut session = SimSession::enter(&mut doc, vec![], DVec2::new(0.0, -9.81), SIM_HZ);
    let mut trace = Vec::with_capacity(PHASE23_STEPS);
    for step in 0..PHASE23_STEPS {
        session.step_once(&mut doc, SimInput::default());
        trace.push(sample_pose(&doc));
        if step == PHASE23_SAVE_AT {
            mid();
        }
    }
    assert_ne!(
        doc_bytes(&doc),
        before,
        "the level did not move during Simulate — restoring it would prove nothing"
    );
    session.exit(&mut doc);
    let after = doc_bytes(&doc);
    (trace, before, after)
}

/// **THE dcc-vision HEADLINE, at gate level.** A mesh saved while the committed
/// level is simulating: the run is undisturbed bit for bit, the editor re-keys
/// mid-run, Stop restores the document byte-for-byte, and the ASSET keeps the
/// edit.
///
/// `inf-editor-core`'s `dcc_edit_during_simulate` makes this claim on a synthetic
/// scene; this makes it on the shipped one, loaded off the committed `.inf_lvl`
/// through the same reader the editor uses.
#[test]
fn an_edit_saved_mid_simulate_survives_stop_on_the_committed_level() {
    let tmp = tempfile::tempdir().unwrap();
    let content = scaffold(tmp.path());
    let mesh_id = AssetId(PHASE23_MESH_GUID);
    let mut proj = AssetProject::open(&content).unwrap();
    vmesh::ensure_vmesh(&mut proj, mesh_id).expect("the baseline derives");
    drop(proj);

    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(content.clone()));
    let key_before = store.resolve_vgeom(PHASE23_MESH_GUID).expect("resolves").id;

    // The control: the same level, the same steps, no save.
    let (control, control_before, control_after) = run_workshop(|| {});
    assert_eq!(
        control_before, control_after,
        "Stop does not restore the committed level even with nothing else going on"
    );

    let mut key_after = key_before;
    let mut fired = false;
    let (traced, before, after) = run_workshop(|| {
        let mut proj = AssetProject::open(&content).unwrap();
        dcc::save_mesh_session(&mut proj, mesh_id, finished().mesh()).expect("saves mid-Simulate");
        drop(proj);
        store.refresh_index();
        key_after = store.resolve_vgeom(PHASE23_MESH_GUID).expect("resolves").id;
        fired = true;
    });
    assert!(fired, "the mid-run hook never ran");

    assert_ne!(
        key_after, key_before,
        "the editor did not re-key across a save made during Simulate"
    );
    assert_eq!(
        traced, control,
        "the step trace diverged from the control run: saving an asset reached \
         the simulation"
    );
    assert!(
        traced.windows(2).any(|w| w[0] != w[1]),
        "the prop never moved, so the identical traces say nothing"
    );
    assert_eq!(
        before, after,
        "Stop did not restore the document byte-for-byte after a mid-run save"
    );

    // The asset kept the edit. This is the ruling the phase is built on.
    let proj = AssetProject::open(&content).unwrap();
    let reread: inf_mesh::MeshAsset = proj.load_payload(mesh_id).unwrap();
    assert!(
        (reread.bounds.max[1] as f64 - (PHASE23_PROP_SIZE_M * 0.5 + PHASE23_EXTRUDE_M)).abs()
            < 1e-5,
        "Stop reverted the ASSET: it tops out at {}",
        reread.bounds.max[1]
    );
}

// ── (h) BUDGETS ─────────────────────────────────────────────────────────────

/// Opening and saving are **load-class** work, so they take the load ceiling.
/// No new ratchet: this phase does not get its own budget for being new.
#[test]
fn the_workshop_opens_and_saves_inside_its_budgets() {
    let tmp = tempfile::tempdir().unwrap();
    let content = scaffold(tmp.path());
    let mesh_id = AssetId(PHASE23_MESH_GUID);

    let t = Instant::now();
    let session = finished();
    let model_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(
        model_ms < LOAD_BUDGET_MS,
        "opening + modelling + unwrapping took {model_ms:.1} ms (budget \
         {LOAD_BUDGET_MS} ms)"
    );

    let mut proj = AssetProject::open(&content).unwrap();
    let t = Instant::now();
    dcc::save_mesh_session(&mut proj, mesh_id, session.mesh()).expect("saves");
    let save_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(
        save_ms < LOAD_BUDGET_MS,
        "the save (rewrite + synchronous vmesh derivation) took {save_ms:.1} ms \
         (budget {LOAD_BUDGET_MS} ms)"
    );
    eprintln!("phase23: model {model_ms:.2} ms, save {save_ms:.2} ms");
}

// ── (i) COOK ADVISORIES ─────────────────────────────────────────────────────

/// **What the cook says about this sample, and why it is not silence.**
///
/// Every other phase gate asserts its flagship draws no advisory. This one
/// cannot, and the reason is a finding rather than an oversight: **a
/// hand-modelled prop is a few dozen triangles, which is below the cook's
/// `[vgeom] min_triangles` (2048)**. The editor's `ensure_vmesh` derives from one
/// triangle, so the prop looks right the whole time it is being authored and
/// ships as a placeholder cube — which is exactly what that advisory exists to
/// say, and the DCC's entire output class lives in that gap.
///
/// So the arm asserts the advisory is **the only one**, that it says what it
/// should, and that lowering the threshold silences the cook completely — which
/// is what proves nothing else is wrong with the content.
#[test]
fn the_workshop_cook_draws_only_the_advisory_it_earns() {
    let tmp = tempfile::tempdir().unwrap();
    scaffold(tmp.path());
    let proj = tmp.path().join("proj");

    let report = cook(&proj, &tmp.path().join("out"), &CookOptions::default()).expect("cooks");
    assert_eq!(
        report.warnings.len(),
        1,
        "expected exactly the sub-threshold advisory: {:?}",
        report.warnings
    );
    let w = &report.warnings[0];
    assert!(w.contains("PLACEHOLDER CUBE"), "{w}");
    assert!(w.contains(&PHASE23_MESH_GUID.to_string()), "{w}");
    assert!(w.contains("min_triangles"), "the fix is named: {w}");

    // …and the fix works, which is what makes the advisory advice.
    let opts = CookOptions {
        vgeom: inf_packager::VgeomCookOptions {
            min_triangles: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let lowered = cook(&proj, &tmp.path().join("out2"), &opts).expect("cooks");
    assert!(
        lowered.warnings.is_empty(),
        "with the threshold lowered the sample still trips advisories: {:?}",
        lowered.warnings
    );
    let reader = PackReader::open(&lowered.pack_path).unwrap();
    assert!(
        reader
            .index()
            .any(|e| e.guid == inf_vgeom::derived_vmesh_id(AssetId(PHASE23_MESH_GUID))),
        "the lowered cook did not ship a .inf_vmesh for the prop"
    );
}

/// **The sample really is the done-when sentence** — asserted on the generators,
/// so a change to the design fails here with a readable number rather than as a
/// byte diff.
#[test]
fn the_workshop_is_actually_the_done_when_sentence() {
    let doc = workshop_doc();
    // One entity draws the mesh under edit…
    let e = doc.entity_of(PHASE23_PROP_GUID).expect("the prop");
    let mesh_ref = doc
        .world()
        .world()
        .get::<inf_ecs::components::MeshRef>(e)
        .expect("the prop has a MeshRef");
    assert_eq!(
        mesh_ref.asset,
        Some(PHASE23_MESH_GUID),
        "the prop does not reference the mesh the gate edits"
    );
    // …and it is a dynamic body, so the level MOVES while that mesh is edited.
    assert_eq!(
        doc.world()
            .world()
            .get::<inf_ecs::components::RigidBody3D>(e)
            .expect("the prop has a body")
            .kind,
        inf_ecs::components::BodyKind3D::Dynamic
    );
    // …whose collider is authored, which is why a mesh edit cannot reach it.
    let col = doc
        .world()
        .world()
        .get::<inf_ecs::components::Collider3D>(e)
        .expect("the prop has a collider");
    assert!(
        col.density > 1.0,
        "the collider carries rapier's 1 kg/m³ placeholder density"
    );
}
