//! Ring-2 command surface for the **Model Editor** (ROADMAP P23.4).
//!
//! The fourth instance of the template `GraphState`, `MaterialState` and
//! `PcgState` already are — a document map and a journal map keyed the same way
//! — with two deliberate differences, both consequences of what a mesh is:
//!
//! * **The journal IS the document.** `inf_dcc::MeshSession` owns the base mesh,
//!   the op list and the cursor, so there is no second copy of the mesh to keep
//!   in step with it. `DccDoc` holds only what the *session* does not: which
//!   asset it came from, the selection, the camera and the import report.
//! * **There is no registry.** A graph editor's palette is data; a modelling
//!   toolbar is buttons, and its "registry" is [`inf_editor_core::ipc::DccToolDto`],
//!   a closed enum the compiler checks.
//!
//! # The session is asset-scoped and the scene is never touched (P23.1 §2)
//!
//! `dcc_open` takes an asset id, `dcc_save` writes that asset, and nothing here
//! names `SceneState`. That is what makes the whole batch cheap: no schema move,
//! no third rung on the lock order, and an edit made while Simulate is running
//! survives Stop — because `SimSession::exit` reverts *the document*, and an
//! asset edit is not in it.
//!
//! # Save lives in Ring 1, and that is a correctness decision
//!
//! `dcc_save` is four lines that call [`inf_editor_core::dcc::save_mesh_session`].
//! It is written there rather than here because a `#[tauri::command]` cannot be
//! driven from a test on any CI leg: the first version inlined the rewrite and
//! the derivation into this file and "proved" them with a Ring-1 test that
//! inlined the same two calls — so deleting the derivation from *this* function
//! failed nothing, and the gate certified a pattern rather than the product.
//!
//! What that function guarantees, and this one only reports:
//! `rewrite_payload` alone leaves the derived `.inf_vmesh` describing the **old**
//! geometry, which the viewport then redraws with complete confidence. The pair
//! is therefore always (new payload, new DAG) or (new payload, no DAG) — never
//! (new payload, stale DAG) — and the one state where that can still fail is
//! reported in words that say what disk holds.
//!
//! Thumbnails need no invalidation: `ThumbnailCache` keys on the content hash, so
//! a rewritten payload is a cache miss by construction.

use std::collections::BTreeMap;
use std::sync::Mutex;

use inf_asset::{AssetId, AssetKind};
use inf_dcc::{
    canonical_edge, edge_loop, edge_ring, from_mesh_asset, op_preserves_ids, ExportReport, FaceId,
    HalfId, ImportReport, KnifePoint, MergeTarget, MeshSession, MirrorAxis, Op, OpError,
    SelectMode, SelectionSet, VertId,
};
use inf_editor_core::assets::vmesh;
use inf_editor_core::dcc::{
    self, DccOrientDto, DccPivotDto, GizmoDrag, GizmoInFlight, GizmoMode, OverlayStyle,
    PendingDrag, PickHit, PreviewCache, Projector, VertTransform,
};
use inf_editor_core::ipc::{
    DccApplyDto, DccDocDto, DccDragBeginDto, DccDragDto, DccExportDto, DccGarmentDto,
    DccGizmoModeDto, DccGroomDto, DccGroomResultDto, DccGroomStatDto, DccHistoryEntryDto,
    DccImportDto, DccModeDto, DccNewDto, DccPaintModeDto, DccPreviewDto, DccPrimitiveDto,
    DccSaveDto, DccSculptModeDto, DccSelectDto, DccToolDto, DccUnwrapDto, SculptFalloffDto,
};
use inf_editor_core::thumbnail::{encode_png_fast, PreviewView, Thumbnailer};
use tauri::{AppHandle, Emitter, State};

/// The preview edge length. 256², per the P23.2a measurement: the *render* is
/// 0.09 ms there and the transport is the real cost — 512² pushes ~1.4 MB of
/// base64 per frame through the webview bridge, which is ~42 MB/s at 30 fps.
const PREVIEW_SIZE: u32 = 256;

/// The brush footprint's ink — a cyan that neither the neutral grey surface
/// (`DCC_SURFACE_WGSL`) nor the amber selection can be confused with.
const BRUSH_RING: [u8; 3] = [64, 220, 210];

/// One open mesh document. The mesh itself lives in the journal.
struct DccDoc {
    id: String,
    asset: AssetId,
    name: String,
    mode: SelectMode,
    selection: SelectionSet,
    view: PreviewView,
    import: ImportReport,
    /// The generation the last successful save wrote. `dirty` is
    /// `generation != saved_generation`, which is exact rather than a flag that
    /// can drift: undoing back to the saved state correctly reads clean only if
    /// the stamps match, and they do not — so an undo past a save reads dirty,
    /// which is the honest answer (the mesh may be identical, the *journal* is
    /// not, and the panel says "unsaved edits", not "different bytes").
    saved_generation: Option<u64>,
    /// Vertices in the order they were picked — the knife's path.
    ///
    /// A `SelectionSet` is a `BTreeSet`, so it has no order and cannot carry a
    /// path. Rather than give the kernel an ordered selection nothing else wants,
    /// the *pick* records its own sequence here and the knife reads it.
    knife: Vec<VertId>,
    /// The last edge picked, for loop/ring select (which need a seed, not a set).
    last_edge: Option<HalfId>,
    /// The tessellation and mesh snapshot the preview draws, held against the
    /// generation that produced them — so an orbit costs a camera write and two
    /// `Arc` clones rather than a full export (P23.4 audit — M6).
    preview: PreviewCache,
    /// **The one pointer-drag slot** (P23.5): a sculpt stroke or a gizmo drag,
    /// never both. Uncommitted — the preview renders it from a scratch copy and
    /// the journal does not hear about it until the gesture settles.
    pending: Option<PendingDrag>,
    /// Snap for the drag in flight (metres / radians / ratio; `0` = off).
    drag_snap: f32,
    /// Which transform gizmo is armed, if any. Backend state because the *pivot*
    /// and the handles are drawn backend-side, and a panel-held copy would be a
    /// second opinion about which tool is active.
    gizmo: Option<GizmoMode>,
    /// **Where the gizmo sits** (Wave D). Hard-wired to the centroid through
    /// P24; now a document setting, because it is a statement about this
    /// document's selection and the panel already holds none of those.
    pivot: DccPivotDto,
    /// **Which way its axes point** (Wave D). `Quat::IDENTITY` at two sites
    /// until now, which is exactly why it is *one* field read by both.
    orient: DccOrientDto,
    /// The position of the last component picked — what
    /// [`DccPivotDto::ActiveElement`] means, and the one pivot that cannot be
    /// derived from the selection because a `BTreeSet` has no last element.
    active: Option<glam::DVec3>,
    /// Whether a marquee catches what it cannot see. Off by default: a box
    /// select that grabs the far side of a closed model is the commonest way to
    /// delete geometry by accident.
    xray: bool,
    /// **The live readout of the drag in flight** — the delta in metres, the
    /// angle in degrees, or the per-axis factor.
    ///
    /// Computed on `dcc_drag_move` and read by `doc_dto`, because the transform
    /// lives in `PendingDrag` and the panel holds no copy of it. This closes the
    /// P23 carried remainder "no rotate-gizmo angle readout" — for all three
    /// modes, since all three had the same nothing.
    readout: Option<String>,
}

impl DccDoc {
    /// **Where this document's gizmo sits**, honouring the pivot setting.
    ///
    /// One function, called by the gizmo drag, the numeric rotate and the
    /// numeric scale — because "the pivot" answered three times is three answers
    /// the first time somebody changes one of them. `None` when nothing is
    /// selected, which is what makes a drag with no selection refuse instead of
    /// silently transforming about the origin.
    fn pivot_point(&self, mesh: &inf_dcc::Mesh) -> Option<glam::DVec3> {
        dcc::gizmo_pivot_of(mesh, &self.selection, self.mode, self.pivot, self.active)
    }

    /// …and which way its axes point.
    fn gizmo_quat(&self, mesh: &inf_dcc::Mesh) -> glam::Quat {
        dcc::gizmo_orientation(mesh, &self.selection, self.mode, self.orient, self.view)
    }
}

struct DccStore {
    docs: BTreeMap<String, DccDoc>,
    journals: BTreeMap<String, MeshSession>,
}

pub struct DccState {
    inner: Mutex<DccStore>,
    /// This panel's own offscreen renderer, like `MaterialState`'s. Separate
    /// because a `Thumbnailer` owns its `PreviewSession`, and that session owns
    /// the uploaded geometry — sharing one with the material editor would have
    /// the two overwrite each other's vertex buffer on every alternation.
    preview: Mutex<Thumbnailer>,
}

impl Default for DccState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(DccStore {
                docs: BTreeMap::new(),
                journals: BTreeMap::new(),
            }),
            preview: Mutex::new(Thumbnailer::new(PREVIEW_SIZE)),
        }
    }
}

impl DccState {
    fn with<R>(&self, f: impl FnOnce(&mut DccStore) -> Result<R, String>) -> Result<R, String> {
        let mut store = self.inner.lock().map_err(|e| e.to_string())?;
        f(&mut store)
    }

    /// Drop a document and its journal together — the whole point of the
    /// two-map shape (`graph_close`'s rule, and the leak it fixed).
    ///
    /// Returns **whether a pointer drag was abandoned**. Not decoration: the
    /// abandon is the one deliberate exception to the orphan-settler doctrine
    /// (see [`dcc_close`]), and without a return value it is *unobservable* — the
    /// doc and the journal are gone either way, so a test could not tell a close
    /// that abandoned from one that settled, and the exception would be a comment
    /// rather than a decision. It is also what the log line reports, because an
    /// author whose strokes vanished with a panel deserves to find out why.
    fn close(&self, id: &str) -> Result<bool, String> {
        self.with(|s| {
            let abandoned = s.docs.remove(id).is_some_and(|d| {
                matches!(
                    d.pending,
                    Some(PendingDrag::Stroke(_)) | Some(PendingDrag::Gizmo(_))
                )
            });
            s.journals.remove(id);
            Ok(abandoned)
        })
    }

    /// Drop **every** open document — the project the sessions belong to is
    /// going away (round-2 finding R2.F15).
    ///
    /// A `MeshSession` is a base mesh plus its op journal plus its checkpoints,
    /// and every document here is keyed on an `AssetId` **in the open project's
    /// database**. `project::apply_open` re-roots that database and every
    /// viewport and never touched this store, so a Model Editor tab open across
    /// a switch kept a full session alive against an asset id that no longer
    /// resolves: every command still worked on it, only `dcc_save` failed, and
    /// each switch stranded another. The panel's unmount effect was the only
    /// remover, and a dock tab is not unmounted by opening a project.
    ///
    /// Returns how many documents were dropped, so the switch can say so — and
    /// so a test can tell a clear that ran from one that did not.
    pub fn clear_all(&self) -> usize {
        match self.inner.lock() {
            Ok(mut s) => {
                let n = s.docs.len();
                s.docs.clear();
                s.journals.clear();
                n
            }
            // A poisoned store is already unusable; reporting zero here is the
            // honest answer and the project switch is not the place to panic.
            Err(_) => 0,
        }
    }
}

impl DccStore {
    fn pair(&mut self, id: &str) -> Result<(&mut DccDoc, &mut MeshSession), String> {
        let doc = self
            .docs
            .get_mut(id)
            .ok_or_else(|| format!("no model document `{id}`"))?;
        let session = self.journals.get_mut(id).ok_or("missing journal")?;
        Ok((doc, session))
    }
}

fn mode_of(m: DccModeDto) -> SelectMode {
    match m {
        DccModeDto::Vert => SelectMode::Vert,
        DccModeDto::Edge => SelectMode::Edge,
        DccModeDto::Face => SelectMode::Face,
    }
}

fn mode_dto(m: SelectMode) -> DccModeDto {
    match m {
        SelectMode::Vert => DccModeDto::Vert,
        SelectMode::Edge => DccModeDto::Edge,
        SelectMode::Face => DccModeDto::Face,
    }
}

fn sculpt_mode_of(m: DccSculptModeDto) -> inf_dcc::SculptMode {
    match m {
        DccSculptModeDto::Draw => inf_dcc::SculptMode::Draw,
        DccSculptModeDto::Smooth => inf_dcc::SculptMode::Smooth,
        DccSculptModeDto::Flatten => inf_dcc::SculptMode::Flatten,
        DccSculptModeDto::Grab => inf_dcc::SculptMode::Grab,
    }
}

fn paint_mode_of(m: DccPaintModeDto) -> inf_dcc::PaintMode {
    match m {
        DccPaintModeDto::Add => inf_dcc::PaintMode::Add,
        DccPaintModeDto::Subtract => inf_dcc::PaintMode::Subtract,
        DccPaintModeDto::Replace => inf_dcc::PaintMode::Replace,
        DccPaintModeDto::Smooth => inf_dcc::PaintMode::Smooth,
    }
}

fn kernel_falloff_of(f: SculptFalloffDto) -> inf_dcc::SculptFalloff {
    match f {
        SculptFalloffDto::Smooth => inf_dcc::SculptFalloff::Smooth,
        SculptFalloffDto::Sphere => inf_dcc::SculptFalloff::Sphere,
        SculptFalloffDto::Linear => inf_dcc::SculptFalloff::Linear,
        SculptFalloffDto::Sharp => inf_dcc::SculptFalloff::Sharp,
    }
}

fn gizmo_mode_of(m: DccGizmoModeDto) -> GizmoMode {
    match m {
        DccGizmoModeDto::Translate => GizmoMode::Translate,
        DccGizmoModeDto::Rotate => GizmoMode::Rotate,
        DccGizmoModeDto::Scale => GizmoMode::Scale,
    }
}

fn gizmo_mode_dto(m: GizmoMode) -> DccGizmoModeDto {
    match m {
        GizmoMode::Translate => DccGizmoModeDto::Translate,
        GizmoMode::Rotate => DccGizmoModeDto::Rotate,
        GizmoMode::Scale => DccGizmoModeDto::Scale,
    }
}

fn falloff_of(f: SculptFalloffDto) -> inf_terrain::Falloff {
    match f {
        SculptFalloffDto::Smooth => inf_terrain::Falloff::Smooth,
        SculptFalloffDto::Sphere => inf_terrain::Falloff::Sphere,
        SculptFalloffDto::Linear => inf_terrain::Falloff::Linear,
        SculptFalloffDto::Sharp => inf_terrain::Falloff::Sharp,
    }
}

fn import_dto(r: &ImportReport) -> DccImportDto {
    DccImportDto {
        source_vertices: r.source_vertices as u32,
        welded_positions: r.welded_positions as u32,
        fan_splits: r.fan_splits as u32,
        degenerate_triangles_skipped: r.degenerate_triangles_skipped as u32,
        sharp_edges: r.sharp_edges as u32,
        boundary_edges: r.boundary_edges as u32,
        non_finite_values: r.non_finite_values as u32,
        duplicate_faces_dropped: r.duplicate_faces_dropped as u32,
        faces_reoriented: r.faces_reoriented as u32,
        non_manifold_splits: r.non_manifold_splits as u32,
        skin_conflicts: r.skin_conflicts as u32,
    }
}

fn export_dto(r: &ExportReport) -> DccExportDto {
    DccExportDto {
        submeshes: r.submeshes as u32,
        vertices: r.vertices as u32,
        triangles: r.triangles as u32,
        fan_fallbacks: r.fan_fallbacks as u32,
        optimized: r.optimized,
        optimize_skipped_skinned: r.optimize_skipped_skinned as u32,
        fallback_tangents: r.fallback_tangents as u32,
        coincident_vertices: r.coincident_vertices as u32,
        reused_diagonals: r.reused_diagonals as u32,
        non_finite_written: r.non_finite_written as u32,
        non_unit_normals_written: r.non_unit_normals_written as u32,
    }
}

fn doc_dto(doc: &DccDoc, session: &MeshSession) -> DccDocDto {
    let mesh = session.mesh();
    DccDocDto {
        id: doc.id.clone(),
        asset_id: doc.asset.to_string(),
        name: doc.name.clone(),
        mode: mode_dto(doc.mode),
        verts: mesh.vert_count() as u32,
        edges: mesh.edge_count() as u32,
        faces: mesh.face_count() as u32,
        selected: doc.selection.len(doc.mode) as u32,
        can_undo: session.can_undo(),
        can_redo: session.can_redo(),
        dirty: doc.saved_generation != Some(session.generation()),
        generation: session.generation(),
        import: import_dto(&doc.import),
        knife_points: doc.knife.len() as u32,
        dragging: doc.pending.is_some(),
        drag_points: match &doc.pending {
            Some(PendingDrag::Stroke(s)) => s.path.len() as u32,
            // A weight stroke is a stroke: "is this doing anything" has to be
            // answerable for it too, or the status bar reads 0 while the author
            // paints (P24.2).
            Some(PendingDrag::Weights(w)) => w.path.len() as u32,
            _ => 0,
        },
        gizmo: doc.gizmo.map(gizmo_mode_dto),
        selection_rev: dcc::selection_revision(&doc.selection),
        seams: inf_dcc::seam_count(mesh) as u32,
        charts: inf_dcc::charts(mesh).len() as u32,
        skin_joints: mesh.skin_binding().map(|b| b.joints),
        // ── Wave D ─────────────────────────────────────────────────────────
        material_slots: mesh.material_slots().to_vec(),
        pivot: doc.pivot,
        orient: doc.orient,
        xray: doc.xray,
        readout: doc.readout.clone(),
    }
}

/// A fresh document for `asset`, at the framing its own bounds imply.
///
/// One constructor for `dcc_open` and `dcc_new`, because two literals of a
/// fifteen-field struct is one struct that acquires a field on one path only —
/// which is how `pivot` would have shipped as "whatever `Default` says" in the
/// editor and "Median" in the creator.
fn new_doc(
    doc_id: String,
    asset: AssetId,
    name: String,
    mesh: &inf_dcc::Mesh,
    session: &MeshSession,
    import: ImportReport,
) -> DccDoc {
    DccDoc {
        id: doc_id,
        asset,
        name,
        mode: SelectMode::Face,
        selection: SelectionSet::new(session.generation()),
        view: dcc::frame(dcc::tessellate(mesh).bounds),
        import,
        saved_generation: Some(session.generation()),
        knife: Vec::new(),
        last_edge: None,
        preview: PreviewCache::new(),
        pending: None,
        drag_snap: 0.0,
        gizmo: None,
        pivot: DccPivotDto::default(),
        orient: DccOrientDto::default(),
        active: None,
        xray: false,
        readout: None,
    }
}

/// **Settle a drag that is still in flight, before doing anything else.**
///
/// The P21.3 orphan-settler doctrine (P23.5): a pointer-up is not guaranteed to
/// arrive — the panel can close, the tool can change, a detached window can lose
/// pointer capture — so every command that touches the journal calls this first
/// and the dabs already on screen become a real, undoable edit.
///
/// **Settle, not abandon**, everywhere except `dcc_close`, which frees the
/// session in the same call and therefore has nothing to settle *into*; see the
/// note there. The logic itself is `inf_editor_core::dcc::settle_drag`, in Ring 1,
/// because a `#[tauri::command]` cannot be driven from a test and this is the
/// rule the whole doctrine rests on.
#[must_use = "a refused stroke is a stroke the author can see on screen and the \
              file will not contain — say so (C4-34)"]
fn settle(doc: &mut DccDoc, session: &mut MeshSession) -> Option<String> {
    let pending = doc.pending.take();
    // The readout describes a drag in flight, so it goes when the drag does.
    // Here rather than at the eleven doors, for the same reason `pending` is
    // taken here: a state that survives its own gesture is the shape where a
    // status strip reads "+0.42 m on X" over a model nobody is touching.
    doc.readout = None;
    dcc::settle_drag(session, &mut doc.selection, doc.mode, pending)
}

/// Settle the stroke in flight and **say so if it was refused** (C4-34).
///
/// `settle` consumes `doc.pending` whether or not the stroke commits, and its
/// `Option<String>` was dropped at ten of the eleven doors — including
/// `dcc_save`, which returned a hardcoded `ok: true`. So a save could write the
/// asset *without* the dabs the author was looking at and report success, which
/// is the one thing the comment above that call forbids in those words.
///
/// `#[must_use]` on `settle` makes dropping the refusal a compile error under
/// `-D warnings`. This helper is the answer for a door whose DTO has nowhere to
/// put it: the Output Log is a real surface, and silence is not.
fn settle_reported(doc: &mut DccDoc, session: &mut MeshSession, door: &str) -> Option<String> {
    let refusal = settle(doc, session);
    if let Some(why) = &refusal {
        tracing::warn!("{door}: the stroke in flight was REFUSED and is not in the model — {why}");
    }
    refusal
}

/// Bring the selection up to the session's stamp, the only way a caller is
/// allowed to read it after anything may have moved.
fn sync(doc: &mut DccDoc, session: &MeshSession) {
    if doc.selection.sync(session.generation()) {
        doc.knife.clear();
        doc.last_edge = None;
    }
}

// ── commands ───────────────────────────────────────────────────────────────

/// Open a `.inf_mesh` asset for editing. Idempotent: a second open of the same
/// asset returns the document already in flight rather than discarding its
/// history.
#[tauri::command]
pub async fn dcc_open(
    app: AppHandle,
    asset_id: String,
    state: State<'_, DccState>,
    assets: State<'_, super::assets::AssetState>,
) -> Result<DccDocDto, String> {
    let id: AssetId = asset_id.parse().map_err(|e| format!("bad asset id: {e}"))?;
    let doc_id = format!("dcc:{id}");
    if let Some(existing) = state.with(|s| {
        Ok(s.docs
            .get(&doc_id)
            .and_then(|d| s.journals.get(&doc_id).map(|j| doc_dto(d, j))))
    })? {
        return Ok(existing);
    }

    let (payload, name) = assets.with_project(|proj| {
        let entry = proj.db().get(id).ok_or_else(|| format!("no asset {id}"))?;
        if entry.kind() != AssetKind::Mesh {
            return Err(format!("{} is not a mesh asset", entry.name));
        }
        let name = entry.name.clone();
        let payload: inf_mesh::MeshAsset = proj
            .load_payload(id)
            .map_err(|e| format!("read {name}: {e}"))?;
        Ok((payload, name))
    })?;

    let import = from_mesh_asset(&payload).map_err(|e| e.to_string())?;
    let session = MeshSession::new(import.mesh);
    let dto = state.with(|s| {
        // No counter: unlike the three graph stores this template came from,
        // a document id here is `dcc:{asset}` — the asset IS the identity, which
        // is what makes `dcc_open` idempotent. The inherited `counter` was
        // incremented on every open and read by nothing.
        let doc = new_doc(
            doc_id.clone(),
            id,
            name,
            session.mesh(),
            &session,
            import.report,
        );
        let dto = doc_dto(&doc, &session);
        s.docs.insert(doc_id.clone(), doc);
        s.journals.insert(doc_id.clone(), session);
        Ok(dto)
    })?;
    let _ = app.emit("dcc://sync", dto.id.clone());
    Ok(dto)
}

/// **Start a new model.** Mint a `.inf_mesh` from one of the kernel's
/// primitives, open a session on it, and reveal it in the drawer.
///
/// The gap Wave E named and could not close from where it was standing: *"there
/// is no way to start a new model at all"*. The kernel has had
/// `cube`/`plane`/`cylinder`/`torus` since P23.3 and Ring 2 has had no door,
/// which made a modelling engine a tool for editing other people's geometry.
///
/// Everything with a rule in it is in Ring 1 — `dcc::primitive_mesh` (the
/// parameter checks, NaN included) and `dcc::create_mesh_asset` (the atomic
/// write, the sidecar, and the **synchronous** `ensure_vmesh` the save path also
/// takes) — because a `#[tauri::command]` cannot be driven from a test. This
/// function is the wiring: build, write, open, announce.
#[tauri::command]
pub async fn dcc_new(
    app: AppHandle,
    spec: DccNewDto,
    state: State<'_, DccState>,
    assets: State<'_, super::assets::AssetState>,
) -> Result<DccDocDto, String> {
    let mesh = dcc::primitive_mesh(spec.primitive, spec.size_m, spec.segments, spec.rings)?;
    let name = spec
        .name
        .as_deref()
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| {
            match spec.primitive {
                DccPrimitiveDto::Cube => "Cube",
                DccPrimitiveDto::Plane => "Plane",
                DccPrimitiveDto::Cylinder => "Cylinder",
                DccPrimitiveDto::Torus => "Torus",
            }
            .to_string()
        });
    let (id, report) =
        assets.with_project(|proj| dcc::create_mesh_asset(proj, &name, &mesh).map_err(|e| e))?;
    // The writer's advisories are surfaced, not dropped, even though a generated
    // primitive cannot normally trip one — "cannot normally" is the class of
    // claim this codebase keeps paying for.
    for line in advisories(&report) {
        tracing::warn!("dcc_new({name}): {line}");
    }

    let doc_id = format!("dcc:{id}");
    let session = MeshSession::new(mesh);
    let dto = state.with(|s| {
        let doc = new_doc(
            doc_id.clone(),
            id,
            name,
            session.mesh(),
            &session,
            ImportReport::default(),
        );
        let dto = doc_dto(&doc, &session);
        s.docs.insert(doc_id.clone(), doc);
        s.journals.insert(doc_id.clone(), session);
        Ok(dto)
    })?;
    // Both fan-outs, for the same reason `dcc_save` sends both: the drawer has to
    // show the new asset and the viewport has to be able to place it.
    viewport_refresh_all(&app);
    let _ = app.emit("assets://changed", ());
    let _ = app.emit("dcc://sync", dto.id.clone());
    Ok(dto)
}

/// Tell every view the asset index moved.
///
/// A free function so `dcc_new` does not have to take a `ViewportState` it uses
/// for one line — and so the `Target::All` decision (this is a fact about the
/// *project*, not about one viewport — P23.2a's rule at every resolution point)
/// is written once.
fn viewport_refresh_all(app: &AppHandle) {
    use tauri::Manager;
    if let Some(vp) = app.try_state::<super::ViewportState>() {
        vp.refresh_asset_index(super::Target::All);
    }
}

/// Free a document and its journal, **by asset id** — the same key `dcc_open`
/// takes.
///
/// Symmetric on purpose. The document id is `"dcc:<assetId>"`, and that format is
/// constructed in exactly one place (`dcc_open`); a `dcc_close` taking the
/// document id made the *frontend* build it, which meant a panel that had to
/// close before its `open` resolved had no id to close with and simply did not
/// send the call — leaking the session, the mesh and the journal for the life of
/// the process. A close that needs nothing but the asset id can always be sent.
///
/// Idempotent: closing an asset with no open document is a no-op, which is what
/// lets the frontend send it speculatively during that race.
///
/// # This door ABANDONS a drag in flight, and every other door settles one
///
/// The orphan-settler doctrine (P21.3, applied to gestures in P23.5) says a
/// half-finished interaction is settled deliberately or abandoned deliberately —
/// never merely dropped. Every other command here settles, because the dabs are
/// the author's work and are already on screen.
///
/// **Close is the exception, and it has to be.** `DccState::close` frees the
/// `DccDoc` and the `MeshSession` together, in this call: an op applied first
/// would land in a journal that is destroyed a line later, so it could never be
/// undone, saved, or seen. A settle here is not a smaller loss than an abandon —
/// it is the same loss with a wasted `Mesh::transact` in front of it. Dropping
/// `pending` with the document is therefore the decision, not the oversight.
///
/// Two gates hold the pair of it, and the citation used to name a test that did
/// not exist: `a_close_abandons_a_drag_while_every_other_door_settles_it` is the
/// behavioural half (this door reports the abandon, and exactly one door does),
/// and `every_journal_door_declares_whether_it_settles_a_drag` is the structural
/// half - it reads the command list out of this file and fails until a new door
/// has *chosen* a policy, because the failing-open default was "does not settle".
#[tauri::command]
pub async fn dcc_close(asset_id: String, state: State<'_, DccState>) -> Result<(), String> {
    let id: AssetId = asset_id.parse().map_err(|e| format!("bad asset id: {e}"))?;
    if state.close(&format!("dcc:{id}"))? {
        tracing::info!(
            "the Model Editor for {id} closed with a drag in flight; it was \
             discarded (the session it would have been journalled into is gone)"
        );
    }
    Ok(())
}

/// Every open document.
#[tauri::command]
pub async fn dcc_list(state: State<'_, DccState>) -> Result<Vec<DccDocDto>, String> {
    state.with(|s| {
        Ok(s.docs
            .values()
            .filter_map(|d| s.journals.get(&d.id).map(|j| doc_dto(d, j)))
            .collect())
    })
}

/// Press a modelling tool against the current selection.
///
/// A refusal is a **value**: the kernel's typed error comes back as
/// `DccApplyDto::refusal`, the mesh is byte-identical, and nothing is journalled.
#[tauri::command]
pub async fn dcc_apply(
    app: AppHandle,
    id: String,
    tool: DccToolDto,
    state: State<'_, DccState>,
) -> Result<DccApplyDto, String> {
    let out = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // A tool press with a drag still in flight settles it first: the author's
        // dabs become a real edit, and the tool runs on the mesh they can see.
        let settled = settle_reported(doc, session, "dcc_apply");
        sync(doc, session);
        let ops = build_ops(doc, session, &tool)?;
        if ops.is_empty() {
            return Ok(DccApplyDto {
                ok: false,
                refusal: Some(settled.unwrap_or_else(|| "nothing is selected".to_string())),
                doc: doc_dto(doc, session),
            });
        }
        // Each op is journalled on its own, so an undo peels one tool press at a
        // time when a tool expands to several (a per-face batch, a merge chain).
        let mut refusal = None;
        for op in ops {
            let preserves = op_preserves_ids(&op);
            match session.apply(op) {
                Ok(outcome) => {
                    if preserves {
                        doc.selection.carry(session.generation(), session.mesh());
                    } else {
                        doc.selection
                            .adopt(session.generation(), &outcome, session.mesh());
                        doc.knife.clear();
                        doc.last_edge = None;
                    }
                }
                Err(e) => {
                    refusal = Some(e.to_string());
                    break;
                }
            }
        }
        // C4-34: a refused settle outranks a successful tool press — the stroke
        // the author is looking at is NOT in the model, and reporting `ok: true`
        // because the *next* op worked is the save that lied, one command early.
        let refusal = settled.or(refusal);
        Ok(DccApplyDto {
            ok: refusal.is_none(),
            refusal,
            doc: doc_dto(doc, session),
        })
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(out)
}

/// Turn a tool press plus the document's selection into kernel ops.
///
/// Where the "operands are the selection" rule lives. Returns an empty list when
/// the selection cannot supply operands, which the caller reports rather than
/// silently doing nothing.
fn build_ops(doc: &DccDoc, session: &MeshSession, tool: &DccToolDto) -> Result<Vec<Op>, String> {
    let mesh = session.mesh();
    let faces: Vec<FaceId> = doc.selection.faces().iter().copied().collect();
    let edges: Vec<HalfId> = doc.selection.edges().iter().copied().collect();
    let verts: Vec<VertId> = doc
        .selection
        .resolved_verts(mesh, doc.mode)
        .into_iter()
        .collect();
    Ok(match tool {
        DccToolDto::Extrude { distance } => {
            if faces.is_empty() {
                Vec::new()
            } else {
                vec![Op::ExtrudeFaces {
                    faces,
                    distance: *distance,
                }]
            }
        }
        DccToolDto::ExtrudeEdges { delta } => {
            if edges.is_empty() {
                Vec::new()
            } else {
                vec![Op::ExtrudeEdges {
                    edges,
                    delta: *delta,
                }]
            }
        }
        DccToolDto::Inset { amount, individual } => {
            if faces.is_empty() {
                Vec::new()
            } else {
                vec![Op::InsetFaces {
                    faces,
                    amount: *amount,
                    individual: *individual,
                }]
            }
        }
        DccToolDto::Bevel { amount, segments } => {
            if edges.is_empty() {
                Vec::new()
            } else {
                vec![Op::BevelEdges {
                    edges,
                    amount: *amount,
                    segments: *segments,
                }]
            }
        }
        DccToolDto::LoopCut { cuts } => match doc.last_edge.or_else(|| edges.first().copied()) {
            Some(half) => vec![Op::LoopCut { half, cuts: *cuts }],
            None => Vec::new(),
        },
        DccToolDto::Knife => {
            if doc.knife.len() < 2 {
                Vec::new()
            } else {
                vec![Op::Knife {
                    path: doc.knife.iter().copied().map(KnifePoint::Vertex).collect(),
                }]
            }
        }
        DccToolDto::Merge { center } => {
            if verts.len() < 2 {
                Vec::new()
            } else {
                vec![Op::MergeVerts {
                    verts,
                    target: if *center {
                        MergeTarget::Center
                    } else {
                        MergeTarget::Last
                    },
                }]
            }
        }
        DccToolDto::Subdivide => {
            if faces.is_empty() {
                Vec::new()
            } else {
                vec![Op::SubdivideFaces { faces }]
            }
        }
        DccToolDto::Mirror { axis, coord } => {
            let axis = match axis.as_str() {
                "x" | "X" => MirrorAxis::X,
                "y" | "Y" => MirrorAxis::Y,
                "z" | "Z" => MirrorAxis::Z,
                other => return Err(format!("unknown mirror axis `{other}`")),
            };
            vec![Op::Mirror {
                axis,
                coord: *coord,
            }]
        }
        // **Both translate tools go through `dcc::transform_ops`**, which is the
        // same function the dragged gizmo commits through (P23.5). That is the
        // whole of "the numeric path and the direct-manipulation path produce
        // identical ops": not two implementations kept in step by a test, but one
        // implementation with a test that says so at the product boundary
        // (`the_gizmo_and_the_numeric_tool_journal_the_same_op`).
        DccToolDto::Translate { delta } => dcc::transform_ops(
            mesh,
            &doc.selection,
            doc.mode,
            // A hard translate is pivot-free; the pivot is only read by the
            // rotate and scale forms.
            glam::DVec3::ZERO,
            VertTransform::Translate(glam::DVec3::from_array(*delta)),
            None,
        ),
        // One `TranslateVerts` per distinct weight, so the whole soft move is
        // ordinary journal entries — and so an author who undoes gets their shape
        // back rather than a partially-relaxed one.
        DccToolDto::SoftTranslate {
            delta,
            radius,
            falloff,
        } => dcc::transform_ops(
            mesh,
            &doc.selection,
            doc.mode,
            glam::DVec3::ZERO,
            VertTransform::Translate(glam::DVec3::from_array(*delta)),
            Some((*radius, falloff_of(*falloff))),
        ),
        // One op per edge, so an undo peels one mark at a time — the same rule
        // the per-face batch tools follow.
        DccToolDto::Seam { seam } => edges
            .into_iter()
            .map(|half| Op::SetEdgeSeam { half, seam: *seam })
            .collect(),
        DccToolDto::Delete => match doc.mode {
            SelectMode::Face => faces
                .into_iter()
                .map(|face| Op::RemoveFace { face })
                .collect(),
            // Removing a vertex requires it to be isolated, so deleting in vertex
            // mode drops the faces around it first — which is what every modeller
            // means by "delete these vertices".
            SelectMode::Vert | SelectMode::Edge => {
                let mut ops: Vec<Op> = Vec::new();
                let mut touched: std::collections::BTreeSet<FaceId> =
                    std::collections::BTreeSet::new();
                for &v in &verts {
                    for &h in mesh.vert_outgoing(v).unwrap_or(&[]) {
                        if let Some(Some(f)) = mesh.face_of(h) {
                            touched.insert(f);
                        }
                    }
                }
                ops.extend(touched.into_iter().map(|face| Op::RemoveFace { face }));
                ops.extend(verts.into_iter().map(|vert| Op::RemoveVertex { vert }));
                ops
            }
        },

        // ── the Wave-D verbs ───────────────────────────────────────────────
        //
        // Every one of these resolves its gesture in **Ring 1** and journals the
        // answer. A refusal comes back as `Err`, so the panel says why rather
        // than silently doing nothing — the shape `dcc_apply` already gives a
        // kernel `OpError`.
        DccToolDto::Dissolve => {
            if edges.is_empty() {
                Vec::new()
            } else {
                vec![Op::DissolveEdges { edges }]
            }
        }
        DccToolDto::Bridge => {
            if edges.is_empty() {
                Vec::new()
            } else {
                vec![Op::BridgeLoops {
                    pairs: dcc::bridge_pairs(mesh, &edges)?,
                }]
            }
        }
        DccToolDto::Flip => {
            // Nothing selected means "recalculate outside", which is what every
            // modeller's version of this button does.
            let scope = if faces.is_empty() {
                mesh.face_ids().collect()
            } else {
                faces
            };
            if scope.is_empty() {
                Vec::new()
            } else {
                vec![Op::FlipFaces { faces: scope }]
            }
        }
        DccToolDto::Shade { smooth, angle_deg } => {
            let (sharp, soft) = dcc::shade_edges(mesh, &faces, *smooth, *angle_deg);
            // Two ops at most — one for each answer — rather than one per edge.
            // Shade-smooth is one gesture and must be one undo step.
            let mut ops = Vec::new();
            if !sharp.is_empty() {
                ops.push(Op::SetEdgesSharp {
                    halfs: sharp,
                    sharp: true,
                });
            }
            if !soft.is_empty() {
                ops.push(Op::SetEdgesSharp {
                    halfs: soft,
                    sharp: false,
                });
            }
            ops
        }
        DccToolDto::Slide { t } => {
            let moves = dcc::slide_moves(mesh, &doc.selection, doc.mode, *t);
            if moves.is_empty() {
                Vec::new()
            } else {
                vec![Op::MoveVerts { moves }]
            }
        }
        DccToolDto::MergeByDistance { tolerance } => dcc::merge_clusters(mesh, &verts, *tolerance)?
            .into_iter()
            .map(|verts| Op::MergeVerts {
                verts,
                target: MergeTarget::Center,
            })
            .collect(),

        // **The other two thirds of the one door.** `Translate` above has gone
        // through `dcc::transform_ops` since P23.5 and had no caller; these two
        // complete the numeric box, and every one of the three produces exactly
        // the op the dragged gizmo produces because there is one function.
        DccToolDto::Rotate { axis, degrees } => {
            let Some(pivot) = doc.pivot_point(mesh) else {
                return Ok(Vec::new());
            };
            dcc::transform_ops(
                mesh,
                &doc.selection,
                doc.mode,
                pivot,
                VertTransform::Rotate {
                    axis: glam::DVec3::from_array(*axis),
                    // Degrees at the UI boundary, radians in the op (the units
                    // doctrine). `to_radians` is exact scaling, not trigonometry.
                    radians: degrees.to_radians(),
                },
                None,
            )
        }
        DccToolDto::Scale { factor } => {
            let Some(pivot) = doc.pivot_point(mesh) else {
                return Ok(Vec::new());
            };
            dcc::transform_ops(
                mesh,
                &doc.selection,
                doc.mode,
                pivot,
                VertTransform::Scale(glam::DVec3::from_array(*factor)),
                None,
            )
        }

        // ── material slots (Wave D) ────────────────────────────────────────
        DccToolDto::AssignSlot { slot } => faces
            .into_iter()
            .map(|face| Op::SetFaceSlot { face, slot: *slot })
            .collect(),
        DccToolDto::AutoSeam { angle_deg, replace } => {
            let cut = dcc::auto_seam_edges(mesh, *angle_deg);
            let mut ops = Vec::new();
            if *replace {
                let all: Vec<HalfId> = mesh
                    .half_ids()
                    .filter(|&h| inf_dcc::canonical_edge(mesh, h) == Some(h))
                    .collect();
                if !all.is_empty() {
                    ops.push(Op::SetEdgesSeam {
                        halfs: all,
                        seam: false,
                    });
                }
            }
            if !cut.is_empty() {
                ops.push(Op::SetEdgesSeam {
                    halfs: cut,
                    seam: true,
                });
            }
            ops
        }
        DccToolDto::AddSlots { names } => {
            let names: Vec<String> = names
                .iter()
                .map(|n| n.trim())
                .filter(|n| !n.is_empty())
                .map(str::to_string)
                .collect();
            if names.is_empty() {
                Vec::new()
            } else {
                vec![Op::AddMaterialSlots { names }]
            }
        }
    })
}

/// Run a selection command.
#[tauri::command]
pub async fn dcc_select(
    id: String,
    action: DccSelectDto,
    state: State<'_, DccState>,
) -> Result<DccDocDto, String> {
    state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // This door answers with a bare `DccDocDto`, which has nowhere to
        // carry a refusal; `settle_reported` has already said it out loud.
        let _ = settle_reported(doc, session, "dcc_select");
        sync(doc, session);
        let mesh = session.mesh();
        match action {
            DccSelectDto::Mode { mode } => {
                let to = mode_of(mode);
                doc.selection.convert(mesh, doc.mode, to);
                doc.mode = to;
            }
            DccSelectDto::All => {
                doc.selection.clear();
                doc.selection.invert(mesh, doc.mode);
            }
            DccSelectDto::None => {
                doc.selection.clear();
                doc.knife.clear();
            }
            DccSelectDto::Invert => doc.selection.invert(mesh, doc.mode),
            DccSelectDto::Grow => doc.selection.grow(mesh, doc.mode),
            DccSelectDto::Shrink => doc.selection.shrink(mesh, doc.mode),
            DccSelectDto::Linked => doc.selection.select_linked(mesh, doc.mode),
            DccSelectDto::Loop => {
                if let Some(h) = doc.last_edge {
                    for x in edge_loop(mesh, h) {
                        doc.selection.set_edge(mesh, x, true);
                    }
                }
            }
            DccSelectDto::Ring => {
                if let Some(h) = doc.last_edge {
                    for x in edge_ring(mesh, h) {
                        doc.selection.set_edge(mesh, x, true);
                    }
                }
            }
        }
        Ok(doc_dto(doc, session))
    })
}

/// Pick the component under a pointer, in the preview's own pixel space.
///
/// `size` must be the size the panel last asked [`dcc_preview`] for, because the
/// projection is built from it — the panel passes the same number to both, which
/// is why there is one constant on the frontend side rather than two literals.
#[tauri::command]
pub async fn dcc_pick(
    id: String,
    x: f64,
    y: f64,
    size: u32,
    additive: bool,
    state: State<'_, DccState>,
) -> Result<DccDocDto, String> {
    state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // This door answers with a bare `DccDocDto`, which has nowhere to
        // carry a refusal; `settle_reported` has already said it out loud.
        let _ = settle_reported(doc, session, "dcc_pick");
        sync(doc, session);
        let mesh = session.mesh();
        let proj = Projector::new(doc.view, size.max(1));
        let hit = dcc::pick(mesh, &proj, doc.mode, x as f32, y as f32);
        match hit {
            Some(PickHit::Vert(v)) => {
                if !additive {
                    doc.selection.clear();
                    doc.knife.clear();
                }
                let on = !doc.selection.contains_vert(v);
                doc.selection.set_vert(v, on);
                if on {
                    doc.knife.push(v);
                } else {
                    doc.knife.retain(|&x| x != v);
                }
            }
            Some(PickHit::Edge(h)) => {
                if !additive {
                    doc.selection.clear();
                }
                let on = !doc.selection.contains_edge(mesh, h);
                doc.selection.set_edge(mesh, h, on);
                doc.last_edge = canonical_edge(mesh, h);
            }
            Some(PickHit::Face(f)) => {
                if !additive {
                    doc.selection.clear();
                }
                let on = !doc.selection.contains_face(f);
                doc.selection.set_face(f, on);
            }
            // A click on empty space clears, which every modeller does and which
            // makes "I cannot deselect" impossible.
            None if !additive => {
                doc.selection.clear();
                doc.knife.clear();
            }
            None => {}
        }
        // **The active element** (Wave D): the position of what was just picked,
        // which is what `DccPivotDto::ActiveElement` means and the one pivot that
        // cannot be derived — a `BTreeSet` has no last member. Taken from the
        // *hit* rather than from the selection so it really is the thing clicked
        // and not the centroid of everything that happens to be selected.
        doc.active = match hit {
            Some(PickHit::Vert(v)) => mesh_position(session.mesh(), v),
            Some(PickHit::Edge(h)) => {
                let m = session.mesh();
                match (
                    m.origin(h).and_then(|v| m.position(v)),
                    m.dest(h).and_then(|v| m.position(v)),
                ) {
                    (Some(a), Some(b)) => Some((a + b) * 0.5),
                    _ => None,
                }
            }
            Some(PickHit::Face(f)) => face_centroid(session.mesh(), f),
            None => None,
        };
        Ok(doc_dto(doc, session))
    })
}

fn mesh_position(mesh: &inf_dcc::Mesh, v: inf_dcc::VertId) -> Option<glam::DVec3> {
    mesh.position(v).filter(|p| p.is_finite())
}

fn face_centroid(mesh: &inf_dcc::Mesh, f: inf_dcc::FaceId) -> Option<glam::DVec3> {
    let vs = mesh.face_verts(f)?;
    let mut acc = glam::DVec3::ZERO;
    let mut n = 0usize;
    for v in vs {
        if let Some(p) = mesh_position(mesh, v) {
            acc += p;
            n += 1;
        }
    }
    (n > 0).then(|| acc / n as f64)
}

// ── pointer drags: the sculpt brush and the component gizmo (P23.5) ────────

/// Arm or disarm the transform gizmo.
///
/// Backend state, like the camera and the selection: the handles are drawn and
/// hit-tested backend-side, so a panel-held copy of "which tool is on" would be a
/// second opinion about what the pixels under the pointer mean.
#[tauri::command]
pub async fn dcc_set_gizmo(
    id: String,
    mode: Option<DccGizmoModeDto>,
    state: State<'_, DccState>,
) -> Result<DccDocDto, String> {
    state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // This door answers with a bare `DccDocDto`, which has nowhere to
        // carry a refusal; `settle_reported` has already said it out loud.
        let _ = settle_reported(doc, session, "dcc_set_gizmo");
        doc.gizmo = mode.map(gizmo_mode_of);
        Ok(doc_dto(doc, session))
    })
}

/// **Set the gizmo's pivot, its orientation, and the x-ray toggle** (Wave D).
///
/// One door for the three, because they are one decision an author makes at
/// once and three doors would be three `dcc://sync` round trips for it. Every
/// argument is optional and `None` means "leave it", so the panel sends only what
/// changed.
#[tauri::command]
pub async fn dcc_set_view_opts(
    id: String,
    pivot: Option<DccPivotDto>,
    orient: Option<DccOrientDto>,
    xray: Option<bool>,
    state: State<'_, DccState>,
) -> Result<DccDocDto, String> {
    state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // This door answers with a bare `DccDocDto`, which has nowhere to
        // carry a refusal; `settle_reported` has already said it out loud.
        let _ = settle_reported(doc, session, "dcc_set_view_opts");
        if let Some(p) = pivot {
            doc.pivot = p;
        }
        if let Some(o) = orient {
            doc.orient = o;
        }
        if let Some(x) = xray {
            doc.xray = x;
        }
        Ok(doc_dto(doc, session))
    })
}

/// **Marquee select.** Everything the rectangle catches, in the current mode.
///
/// The rules and the visibility test are `dcc::pick_box`, in Ring 1 — including
/// the honest caveat that "visible" is back-face culling and not depth testing,
/// so the far side of a *fold* is still caught with x-ray off.
///
/// `additive` extends the selection rather than replacing it, matching
/// `dcc_pick`'s shift-click. A degenerate rectangle (a click that wobbled) is a
/// no-op rather than a "select nothing": clearing an author's whole selection
/// because their hand moved two pixels is the failure this guard exists for.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn dcc_box_select(
    id: String,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    size: u32,
    additive: bool,
    state: State<'_, DccState>,
) -> Result<DccDocDto, String> {
    state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // This door answers with a bare `DccDocDto`, which has nowhere to
        // carry a refusal; `settle_reported` has already said it out loud.
        let _ = settle_reported(doc, session, "dcc_box_select");
        sync(doc, session);
        let rect = dcc::BoxRect {
            x0: x0 as f32,
            y0: y0 as f32,
            x1: x1 as f32,
            y1: y1 as f32,
        };
        if rect.is_degenerate() {
            return Ok(doc_dto(doc, session));
        }
        let mesh = session.mesh();
        let proj = Projector::new(doc.view, size.max(1));
        let (verts, edges, faces) = dcc::pick_box(mesh, &proj, doc.mode, rect, doc.xray);
        if !additive {
            doc.selection.clear();
            doc.knife.clear();
        }
        for v in verts {
            doc.selection.set_vert(v, true);
        }
        for h in edges {
            doc.selection.set_edge(mesh, h, true);
            doc.last_edge = Some(h);
        }
        for f in faces {
            doc.selection.set_face(f, true);
        }
        // The marquee moves the "active element" to the selection it just made,
        // so a pivot set to `ActiveElement` does not keep pointing at whatever
        // was clicked three gestures ago.
        doc.active = dcc::gizmo_pivot(session.mesh(), &doc.selection, doc.mode);
        Ok(doc_dto(doc, session))
    })
}

/// Begin a pointer drag — a sculpt stroke or a gizmo drag.
///
/// **A miss is not a refusal.** `grabbed: false` means the pointer found neither
/// the surface (sculpt) nor a handle (gizmo), nothing is pending, and the panel
/// should treat the gesture as a camera orbit. That is what lets one pointer
/// button paint on the model and orbit off it without a modifier.
#[tauri::command]
pub async fn dcc_drag_begin(
    app: AppHandle,
    id: String,
    drag: DccDragDto,
    x: f64,
    y: f64,
    size: u32,
    state: State<'_, DccState>,
) -> Result<DccDragBeginDto, String> {
    let out = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // A second `begin` without an `end` — a lost pointer-up — settles the
        // first rather than discarding it.
        let settled = settle_reported(doc, session, "dcc_drag_begin");
        sync(doc, session);
        let proj = Projector::new(doc.view, size.max(1));
        let (px, py) = (x as f32, y as f32);
        let (grabbed, handle) = match drag {
            DccDragDto::Sculpt {
                mode,
                radius,
                strength,
                falloff,
            } => {
                // **Four lines that call Ring 1.** The radius floor and the
                // surface hit are decided in `dcc::begin_stroke`, where a test can
                // reach them — the P23.5 audit's mutation table found that with
                // the floor written here, deleting it failed nothing at all
                // (a `#[tauri::command]` is not drivable from any CI leg). Same
                // finding as P23.4's save, same fix.
                match dcc::begin_stroke(
                    session.mesh(),
                    &proj,
                    px,
                    py,
                    dcc::BrushSettings {
                        mode: sculpt_mode_of(mode),
                        radius,
                        strength,
                        falloff: kernel_falloff_of(falloff),
                    },
                ) {
                    Ok(Some(stroke)) => {
                        doc.pending = Some(PendingDrag::Stroke(stroke));
                        (true, None)
                    }
                    Ok(None) => (false, None),
                    Err(refusal) => {
                        return Ok(DccDragBeginDto {
                            grabbed: false,
                            handle: None,
                            refusal: Some(refusal),
                            doc: doc_dto(doc, session),
                        })
                    }
                }
            }
            DccDragDto::WeightPaint {
                joint,
                mode,
                radius,
                strength,
                falloff,
            } => {
                // Same four lines calling Ring 1, same reason: the radius floor,
                // the "is this mesh even skinned" check and the joint range live
                // in `dcc::begin_weight_stroke`, where a test can reach them.
                match dcc::begin_weight_stroke(
                    session.mesh(),
                    &proj,
                    px,
                    py,
                    dcc::WeightBrush {
                        joint: joint.min(u16::MAX as u32) as u16,
                        mode: paint_mode_of(mode),
                        radius,
                        strength,
                        falloff: kernel_falloff_of(falloff),
                    },
                ) {
                    Ok(Some(stroke)) => {
                        doc.pending = Some(PendingDrag::Weights(stroke));
                        (true, None)
                    }
                    Ok(None) => (false, None),
                    Err(refusal) => {
                        return Ok(DccDragBeginDto {
                            grabbed: false,
                            handle: None,
                            refusal: Some(refusal),
                            doc: doc_dto(doc, session),
                        })
                    }
                }
            }
            DccDragDto::Gizmo {
                mode,
                snap,
                soft_radius,
                falloff,
            } => {
                let gmode = gizmo_mode_of(mode);
                doc.gizmo = Some(gmode);
                let Some(pivot) = doc.pivot_point(session.mesh()) else {
                    return Ok(DccDragBeginDto {
                        grabbed: false,
                        handle: None,
                        refusal: Some(settled.unwrap_or_else(|| {
                            "nothing is selected, so the gizmo has no pivot".to_string()
                        })),
                        doc: doc_dto(doc, session),
                    });
                };
                // **The orientation is read ONCE and handed to both** the
                // hit-test and the drag. Those are the two sites that used to say
                // `Quat::IDENTITY` independently, which is exactly the shape
                // where one gets updated and the other does not — and the symptom
                // would be a gizmo that picks one axis and drags along another.
                let quat = doc.gizmo_quat(session.mesh());
                match (
                    dcc::pick_gizmo(&proj, doc.view, pivot, quat, gmode, px, py),
                    proj.ray(px, py),
                ) {
                    (Some(axis), Some((ro, rd))) => {
                        let origin =
                            glam::Vec3::new(pivot.x as f32, pivot.y as f32, pivot.z as f32);
                        doc.drag_snap = snap as f32;
                        doc.pending = Some(PendingDrag::Gizmo(Box::new(GizmoInFlight {
                            drag: GizmoDrag::begin(gmode, axis, quat, origin, ro, rd),
                            pivot,
                            // Zero until the first move: a click that never
                            // dragged must journal nothing.
                            xform: match gmode {
                                GizmoMode::Translate => VertTransform::Translate(glam::DVec3::ZERO),
                                GizmoMode::Rotate => VertTransform::Rotate {
                                    axis: glam::DVec3::Y,
                                    radians: 0.0,
                                },
                                GizmoMode::Scale => VertTransform::Scale(glam::DVec3::ONE),
                            },
                            soft: (soft_radius > 0.0).then(|| (soft_radius, falloff_of(falloff))),
                        })));
                        (true, Some(format!("{axis:?}")))
                    }
                    _ => (false, None),
                }
            }
        };
        Ok(DccDragBeginDto {
            grabbed,
            handle,
            // The PREVIOUS stroke's refusal, if there was one: this drag beginning
            // is not evidence that the last one landed (C4-34).
            refusal: settled,
            doc: doc_dto(doc, session),
        })
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(out)
}

/// Extend the drag in flight. Cheap on purpose: it appends a point (or updates a
/// transform) and returns nothing, because the panel's next `dcc_preview` is what
/// shows the result and a document round trip per pointer-move would double the
/// bridge traffic of a drag.
///
/// A move that finds nothing is **skipped**, not an error: a sculpt drag that
/// wanders off the silhouette and back must not tear the stroke in two, and a
/// gizmo ray that cannot be built leaves the last transform standing.
#[tauri::command]
pub async fn dcc_drag_move(
    id: String,
    x: f64,
    y: f64,
    size: u32,
    state: State<'_, DccState>,
) -> Result<(), String> {
    state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        let proj = Projector::new(doc.view, size.max(1));
        let (px, py) = (x as f32, y as f32);
        let snap = doc.drag_snap;
        match doc.pending.as_mut() {
            Some(PendingDrag::Stroke(stroke)) => {
                if let Some((face, hit)) = dcc::pick_surface(session.mesh(), &proj, px, py) {
                    stroke.path.push(hit);
                    if let Some(n) = inf_dcc::face_normal(session.mesh(), face) {
                        stroke.last_normal = n;
                    }
                }
            }
            Some(PendingDrag::Weights(stroke)) => {
                if let Some((face, hit)) = dcc::pick_surface(session.mesh(), &proj, px, py) {
                    stroke.path.push(hit);
                    if let Some(n) = inf_dcc::face_normal(session.mesh(), face) {
                        stroke.last_normal = n;
                    }
                }
            }
            Some(PendingDrag::Gizmo(g)) => {
                if let Some((ro, rd)) = proj.ray(px, py) {
                    g.xform = VertTransform::from_gizmo(g.drag.update(ro, rd, snap));
                    // **The readout** (Wave D). Written here rather than derived
                    // in `doc_dto`, because `doc.pending` is borrowed mutably in
                    // this arm and the alternative is a second match over the
                    // same enum in the projection — which is where two answers
                    // about one drag come from.
                    doc.readout = Some(dcc::drag_readout(&g.xform));
                }
            }
            None => {}
        }
        Ok(())
    })
}

/// Finish the drag: **one journal entry** for the whole gesture.
#[tauri::command]
pub async fn dcc_drag_end(
    app: AppHandle,
    id: String,
    state: State<'_, DccState>,
) -> Result<DccApplyDto, String> {
    let out = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        let refusal = settle_reported(doc, session, "dcc_drag_end");
        Ok(DccApplyDto {
            ok: refusal.is_none(),
            refusal,
            doc: doc_dto(doc, session),
        })
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(out)
}

/// Throw the drag in flight away — the author's explicit "no" (Escape).
///
/// The **only** deliberate abandon besides `dcc_close`, and it is deliberate in
/// the other direction: an author who presses Escape has said the gesture was a
/// mistake, and settling it anyway would make Escape mean "commit and then
/// undo".
#[tauri::command]
pub async fn dcc_drag_cancel(
    app: AppHandle,
    id: String,
    state: State<'_, DccState>,
) -> Result<DccDocDto, String> {
    let dto = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        doc.pending = None;
        Ok(doc_dto(doc, session))
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(dto)
}

/// Orbit / dolly the preview camera. Degrees and metres.
#[tauri::command]
pub async fn dcc_orbit(
    id: String,
    yaw_deg: f32,
    pitch_deg: f32,
    dolly: f32,
    state: State<'_, DccState>,
) -> Result<(), String> {
    // **The one door in this module that wrote unvalidated floats into state.**
    // `f32::clamp` PROPAGATES NaN (it returns `self` when neither comparison
    // fires — the C4-35 mechanism), so a single non-finite delta from the wire
    // poisoned the camera permanently: every later preview rendered a blank
    // image and nothing said why. Recoverable via `dcc_frame`, which is not a
    // thing an author would know to try.
    if !yaw_deg.is_finite() || !pitch_deg.is_finite() || !dolly.is_finite() {
        return Err("a camera move must be finite".into());
    }
    state.with(|s| {
        let (doc, _) = s.pair(&id)?;
        doc.view.yaw_deg += yaw_deg;
        doc.view.pitch_deg = (doc.view.pitch_deg + pitch_deg).clamp(-89.0, 89.0);
        // Multiplicative, so a wheel notch moves the same *proportion* whatever
        // the model's scale — the alternative is a dolly that crawls on a
        // building and teleports on a bolt.
        doc.view.distance = (doc.view.distance * (1.0 + dolly)).clamp(1e-3, 1e6);
        Ok(())
    })
}

/// Re-frame the camera on the whole mesh.
#[tauri::command]
pub async fn dcc_frame(id: String, state: State<'_, DccState>) -> Result<(), String> {
    state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        doc.view = dcc::frame(dcc::tessellate(session.mesh()).bounds);
        Ok(())
    })
}

/// Step the journal back one op.
#[tauri::command]
pub async fn dcc_undo(
    app: AppHandle,
    id: String,
    state: State<'_, DccState>,
) -> Result<DccDocDto, String> {
    let dto = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // Settle FIRST, so Ctrl+Z during a drag undoes that drag rather than the
        // edit before it and then losing the drag entirely.
        // This door answers with a bare `DccDocDto`, which has nowhere to
        // carry a refusal; `settle_reported` has already said it out loud.
        let _ = settle_reported(doc, session, "dcc_undo");
        session.undo();
        sync(doc, session);
        Ok(doc_dto(doc, session))
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(dto)
}

/// **The history**, as rows a panel can draw — the op list the Edit menu has
/// advertised since Phase 1 and never had behind it.
///
/// The whole journal, redo tail included, because a redo an author has not
/// pressed is still their work and hiding it makes the list lie about where the
/// cursor is.
#[tauri::command]
pub async fn dcc_history(
    id: String,
    state: State<'_, DccState>,
) -> Result<Vec<DccHistoryEntryDto>, String> {
    state.with(|s| {
        let (_, session) = s.pair(&id)?;
        let cursor = session.cursor();
        Ok(session
            .ops()
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let scalar = inf_dcc::op_scalar(op);
                let amendable = !matches!(inf_dcc::op_amendable(op), inf_dcc::Amendability::Never)
                    && scalar.is_some()
                    && i < cursor;
                DccHistoryEntryDto {
                    index: i as u32,
                    kind: inf_dcc::op_kind(op).to_string(),
                    applied: i < cursor,
                    value: scalar.map(|(v, _)| v),
                    unit: scalar.map(|(_, u)| u.to_string()).unwrap_or_default(),
                    amendable,
                    // A reason for every row that is not amendable — the Wave-E
                    // invariant "a route or a reason, never neither" at row
                    // scope. Three different situations, three different
                    // sentences, because "not amendable" would be the same
                    // useless answer to all of them.
                    reason: if amendable {
                        None
                    } else if i >= cursor {
                        Some("this edit is undone; redo it before changing it".into())
                    } else if scalar.is_none() {
                        Some(
                            "this edit has no single number to change — undo to it and \
                             make the edit you meant instead"
                                .into(),
                        )
                    } else {
                        Some(inf_dcc::why_never(op).to_string())
                    },
                }
            })
            .collect())
    })
}

/// **Re-parameterize an edit that is already in the past**, and re-derive
/// everything after it.
///
/// The wave's signature capability at the product boundary. Everything with a
/// rule in it is `inf_dcc::MeshSession::amend` — two gates, a full tail replay
/// and a topology comparison at every step — and this is the wiring: find the
/// op, put the new number in it, hand it over.
///
/// A refusal is a **value**: the kernel's typed `AmendError` comes back as
/// `DccApplyDto::refusal`, the session is byte-identical, and the panel says
/// why. That is the same shape `dcc_apply` already has, for the same reason.
#[tauri::command]
pub async fn dcc_amend(
    app: AppHandle,
    id: String,
    index: u32,
    value: f64,
    state: State<'_, DccState>,
) -> Result<DccApplyDto, String> {
    let out = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // Settle first, exactly as undo does: an amendment with a drag in flight
        // must not throw the drag away, and it must not be replayed *under* it.
        let settled = settle_reported(doc, session, "dcc_amend");
        let i = index as usize;
        let Some(op) = session.ops().get(i).cloned() else {
            return Ok(DccApplyDto {
                ok: false,
                refusal: Some(chain_refusals(
                    settled,
                    format!("there is no edit {index} in this history"),
                )),
                doc: doc_dto(doc, session),
            });
        };
        if !value.is_finite() {
            return Ok(DccApplyDto {
                ok: false,
                refusal: Some(chain_refusals(
                    settled,
                    format!("{value} is not a number this edit can take"),
                )),
                doc: doc_dto(doc, session),
            });
        }
        let Some(amended) = inf_dcc::with_scalar(&op, value) else {
            return Ok(DccApplyDto {
                ok: false,
                refusal: Some(chain_refusals(
                    settled,
                    format!("a {} has no single number to change", inf_dcc::op_kind(&op)),
                )),
                doc: doc_dto(doc, session),
            });
        };
        let refusal = match session.amend(i, amended) {
            Ok(()) => settled,
            Err(e) => Some(chain_refusals(settled, e.to_string())),
        };
        // The amendment moves the generation stamp, so the selection's ids are
        // stale by the crate's own rule even though the topology is proven
        // identical. `sync` is what says so rather than assuming.
        sync(doc, session);
        Ok(DccApplyDto {
            ok: refusal.is_none(),
            refusal,
            doc: doc_dto(doc, session),
        })
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(out)
}

/// Step the journal forward one op.
#[tauri::command]
pub async fn dcc_redo(
    app: AppHandle,
    id: String,
    state: State<'_, DccState>,
) -> Result<DccDocDto, String> {
    let dto = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // This door answers with a bare `DccDocDto`, which has nowhere to
        // carry a refusal; `settle_reported` has already said it out loud.
        let _ = settle_reported(doc, session, "dcc_redo");
        session.redo();
        sync(doc, session);
        Ok(doc_dto(doc, session))
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(dto)
}

/// Render one preview frame: the edit mesh, with the wireframe and selection
/// composited on top.
///
/// The geometry upload is keyed by **`PreviewCache::upload_stamp`**, so an orbit
/// re-renders without touching a vertex buffer (the P23.2a warm path) *and* a
/// live drag does. `encode_png_fast` is the encoder, per the same measurement:
/// at edit rate the default deflate is 98% of the frame.
///
/// Round-2 finding R2.F4: this used to send `session.generation()`, which is
/// precisely the number an uncommitted drag does not move — so `set_geometry`
/// early-returned on every sculpt / weight / gizmo frame and the shaded surface
/// stayed frozen at the pre-drag mesh while the composited wireframe tracked the
/// pointer. The cache is the only thing that knows which of its two slots it
/// just served, so the stamp is read from it rather than re-derived here.
#[tauri::command]
pub async fn dcc_preview(
    id: String,
    size: u32,
    state: State<'_, DccState>,
) -> Result<DccPreviewDto, String> {
    let size = size.clamp(64, 1024);
    // Everything CPU-side under the store lock, then render outside it: the GPU
    // call is the long one and nothing else in this module needs the store while
    // it runs.
    #[allow(clippy::type_complexity)]
    let (geo, mesh_copy, stamp, view, selection, mode, brush, gizmo) = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        sync(doc, session);
        // Cached on the generation stamp: an orbit re-renders without
        // re-tessellating and without cloning the mesh (P23.4 audit — M6). With
        // a drag in flight the uncommitted side channel takes over, keyed on the
        // drag's own shape so an orbit mid-stroke is still free
        // (`PreviewCache::get_with_pending`, and the cost is measured there).
        let (geo, mesh) =
            doc.preview
                .get_with_pending(session, &doc.selection, doc.mode, doc.pending.as_ref());
        // A brush ring while a stroke is live. **Only while it is live**: a hover
        // ring would need a pointer-move round trip the panel does not make, and
        // it is a stated remainder rather than a silently missing feature.
        let brush = match &doc.pending {
            Some(PendingDrag::Stroke(st)) => {
                st.path.last().map(|&c| (c, st.last_normal, st.radius))
            }
            // A weight stroke gets the same ring — it is the same gesture with
            // the same reach, and a brush that draws no footprint while it is
            // painting is the one an author cannot aim (P24.2).
            Some(PendingDrag::Weights(st)) => st
                .path
                .last()
                .map(|&c| (c, st.last_normal, st.brush.radius)),
            _ => None,
        };
        // The gizmo is drawn against the COMMITTED selection's pivot, which is
        // where the drag was begun — recomputing it from the scratch mesh would
        // make the handle chase the geometry it is moving.
        let gizmo = doc
            .gizmo
            .zip(dcc::gizmo_pivot(session.mesh(), &doc.selection, doc.mode))
            .map(|(g, p)| {
                let active = match &doc.pending {
                    Some(PendingDrag::Gizmo(d)) => Some(d.drag.axis),
                    _ => None,
                };
                (g, p, active)
            });
        Ok((
            geo,
            mesh,
            doc.preview.upload_stamp(),
            doc.view,
            doc.selection.clone(),
            doc.mode,
            brush,
            gizmo,
        ))
    })?;

    let mut prev = state.preview.lock().map_err(|e| e.to_string())?;
    let rendered = prev.render_geometry_view(
        &geo.verts,
        &geo.indices,
        stamp,
        dcc::DCC_SURFACE_WGSL,
        size,
        view,
    );
    drop(prev);

    let mut rgba = match rendered {
        Ok(Some(rgba)) => rgba,
        Ok(None) => {
            return Ok(DccPreviewDto {
                image: None,
                error: Some("no GPU adapter on this machine".into()),
                size,
            })
        }
        Err(msg) => {
            tracing::warn!("dcc_preview: {msg}");
            return Ok(DccPreviewDto {
                image: None,
                error: Some(msg),
                size,
            });
        }
    };
    let proj = Projector::new(view, size);
    dcc::draw_overlay(
        &mut rgba,
        size,
        &mesh_copy,
        &proj,
        &selection,
        mode,
        &OverlayStyle::default(),
    );
    // Handles under the brush ring: the ring follows the pointer and must never
    // be the thing hidden.
    if let Some((gmode, pivot, active)) = gizmo {
        dcc::draw_gizmo(&mut rgba, size, &proj, view, pivot, gmode, active);
    }
    if let Some((centre, normal, radius)) = brush {
        dcc::draw_brush_ring(&mut rgba, size, &proj, centre, normal, radius, BRUSH_RING);
    }
    let png = encode_png_fast(size, &rgba)?;
    Ok(DccPreviewDto {
        image: Some(format!(
            "data:image/png;base64,{}",
            super::assets::base64(&png)
        )),
        error: None,
        size,
    })
}

/// Write the edit mesh back to its asset.
///
/// Four lines that call [`inf_editor_core::dcc::save_mesh_session`] and render
/// its verdict. Everything the save actually guarantees — the payload rewrite,
/// the synchronous derivation, and the removal of a DAG that would otherwise
/// describe the geometry just replaced — is stated and tested there, because a
/// `#[tauri::command]` cannot be driven from a test (see the module docs).
#[tauri::command]
pub async fn dcc_save(
    app: AppHandle,
    id: String,
    state: State<'_, DccState>,
    assets: State<'_, super::assets::AssetState>,
    viewport: State<'_, super::ViewportState>,
) -> Result<DccSaveDto, String> {
    let (asset, mesh, generation, settled) = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // A save with a stroke in flight writes the stroke: the author can see it
        // on screen, so a file that did not contain it would be a save that lied.
        // **And when the stroke is REFUSED, the file does not contain it** — which
        // is what this door reported `ok: true` over (C4-34). The refusal now
        // rides out in `advisories` and decides `ok`.
        let settled = settle_reported(doc, session, "dcc_save");
        Ok((
            doc.asset,
            session.mesh().clone(),
            session.generation(),
            settled,
        ))
    })?;

    // ONE call, in Ring 1, where a test can reach it — see the module docs.
    let outcome = assets.with_project(|proj| {
        dcc::save_mesh_session(proj, asset, &mesh).map_err(|e| e.to_string())
    })?;
    let report = outcome.export;
    let vmesh = match outcome.vmesh {
        vmesh::VmeshDerivation::Built(_) => "built",
        vmesh::VmeshDerivation::Cached(_) => "cached",
        vmesh::VmeshDerivation::Skipped => "skipped",
    };

    state.with(|s| {
        if let Some(doc) = s.docs.get_mut(&id) {
            doc.saved_generation = Some(generation);
        }
        Ok(())
    })?;

    // Push the change out now rather than waiting for the debounced watcher: the
    // author pressed Save and the next thing they do is look at the viewport.
    // `Target::All` because this is a fact about the project, not about one
    // viewport (P23.2a's rule at every resolution point).
    viewport.refresh_asset_index(super::Target::All);
    let _ = app.emit("assets://changed", ());
    let _ = app.emit("dcc://sync", id);

    let mut advisories = advisories(&report);
    if let Some(why) = &settled {
        advisories.insert(
            0,
            format!(
                "the stroke in flight was REFUSED and is NOT in the file you just saved — {why}"
            ),
        );
    }
    Ok(DccSaveDto {
        ok: settled.is_none(),
        advisories,
        export: export_dto(&report),
        vmesh: vmesh.into(),
    })
}

/// Turn the writer's counters into sentences. The P16 advisory doctrine: a
/// silent hazard the author cannot see is worse than a line of text.
fn advisories(r: &ExportReport) -> Vec<String> {
    let mut out = Vec::new();
    if r.coincident_vertices > 0 {
        out.push(format!(
            "{} vertices share a position with another. Re-opening this mesh will \
             FUSE them — the reader's weld is exact. Move them apart or merge them \
             deliberately.",
            r.coincident_vertices
        ));
    }
    if r.reused_diagonals > 0 {
        out.push(format!(
            "{} triangulation diagonals had to repeat an edge the mesh already \
             has; re-opening may refuse those faces as non-manifold.",
            r.reused_diagonals
        ));
    }
    if r.fan_fallbacks > 0 {
        out.push(format!(
            "{} faces had no valid ear and fell back to a fan — they are \
             self-intersecting or collinear.",
            r.fan_fallbacks
        ));
    }
    if r.non_finite_written > 0 {
        out.push(format!(
            "{} vertices arrived from the source file with a non-finite value \
             and were written as zeroes — the saved mesh is readable and those \
             vertices sit at the origin.",
            r.non_finite_written
        ));
    }
    if r.non_unit_normals_written > 0 {
        out.push(format!(
            "{} written normals are not unit length (imported values passing \
             through).",
            r.non_unit_normals_written
        ));
    }
    out
}

/// Unwrap the mesh: cut at the seams, parameterize each chart, pack, and journal
/// the **result**.
///
/// The solver runs here and its output becomes one `Op::Unwrap`; replay does not
/// re-solve (see `inf_dcc::uv` for why that is a decision and not an
/// optimization). A refusal — a chart with no area, or no pair to pin — comes
/// back as a value with the chart index in it.
///
/// # Three phases, because the solve is seconds long (round-2 finding R2.F14)
///
/// `DccState::with` guards `docs` **and** `journals` for every open Model
/// Editor, and this was the one door that held it across a long kernel call:
/// `inf_dcc::unwrap`'s conjugate-gradient parameterization is seconds on a dense
/// mesh, so an unwrap froze every other Model Editor command *and* the Tauri
/// worker it ran on. The rule is stated 240 lines above (`dcc_preview`: CPU-side
/// under the lock, the long call outside it) and `dcc_save` follows it; this one
/// did not.
///
/// So: clone under the lock, solve on a blocking task, re-acquire and apply. The
/// re-acquire **re-checks the generation** and refuses if the mesh moved while
/// the solver ran — an `Op::Unwrap` carries the corner list it solved for, and
/// applying it to a mesh that has since been extruded would write UVs onto
/// corners that are not the ones measured. A refusal is a value, per P21's law.
#[tauri::command]
pub async fn dcc_unwrap(
    app: AppHandle,
    id: String,
    state: State<'_, DccState>,
) -> Result<DccUnwrapDto, String> {
    // Phase 1 — settle and take a copy, under the lock.
    let (mesh, solved_at, settled) = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        let settled = settle_reported(doc, session, "dcc_unwrap");
        sync(doc, session);
        Ok((session.mesh().clone(), session.generation(), settled))
    })?;

    // Phase 2 — the solve, off both the store lock and the async workers.
    let solved = tauri::async_runtime::spawn_blocking(move || {
        let seams = inf_dcc::seam_count(&mesh) as u32;
        (inf_dcc::unwrap(&mesh), seams)
    })
    .await
    .map_err(|e| format!("unwrap task failed: {e}"))?;
    let (solved, seams_before) = solved;

    // Phase 3 — apply, back under the lock.
    let out = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        if session.generation() != solved_at {
            let seams = inf_dcc::seam_count(session.mesh()) as u32;
            return Ok(unwrap_refused(
                doc,
                session,
                "the model changed while the unwrap was solving — nothing was written; \
                 run Unwrap again"
                    .to_string(),
                settled,
                seams,
            ));
        }
        match solved {
            Ok(unwrapped) => {
                let report = unwrapped.report;
                match session.apply(unwrapped.op) {
                    Ok(_) => {
                        // `Op::Unwrap` is a pure attribute write, so the ids the
                        // author has selected still name what they named.
                        doc.selection.carry(session.generation(), session.mesh());
                        Ok(DccUnwrapDto {
                            // C4-34: the unwrap succeeded, the stroke before it
                            // did not, and the file will not contain it.
                            ok: settled.is_none(),
                            refusal: settled,
                            charts: report.charts.len() as u32,
                            corners: report.corners as u32,
                            seams: report.seams as u32,
                            worst_residual: report.worst_residual,
                            worst_convergence: report.worst_convergence,
                            flipped: report.flipped as u32,
                            triangles: report.charts.iter().map(|c| c.triangles).sum::<usize>()
                                as u32,
                            doc: doc_dto(doc, session),
                        })
                    }
                    Err(e) => Ok(unwrap_refused(
                        doc,
                        session,
                        refusal_text(&e),
                        settled,
                        report.seams as u32,
                    )),
                }
            }
            Err(e) => Ok(unwrap_refused(
                doc,
                session,
                refusal_text(&e),
                settled,
                seams_before,
            )),
        }
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(out)
}

/// An unwrap that wrote nothing, as a value the panel prints (the P21 law).
///
/// `settled` is the **stroke-in-flight** refusal this door has already reported
/// to the log, and it is chained rather than replaced (round 3). Every one of
/// this command's three refusal exits dropped it: the success path carried it
/// (that is C4-34's own arm), and the three failures answered with their own
/// message alone — so an author whose sculpt stroke was discarded AND whose
/// unwrap then refused was told about the second and not the first, which is
/// the half that says their file will not contain what they were looking at.
/// `dcc_apply`'s `settled.or(refusal)` is the same decision one door over.
fn unwrap_refused(
    doc: &DccDoc,
    session: &MeshSession,
    why: String,
    settled: Option<String>,
    seams: u32,
) -> DccUnwrapDto {
    DccUnwrapDto {
        ok: false,
        refusal: Some(chain_refusals(settled, why)),
        charts: 0,
        corners: 0,
        seams,
        worst_residual: 0.0,
        worst_convergence: 0.0,
        flipped: 0,
        triangles: 0,
        doc: doc_dto(doc, session),
    }
}

/// Two refusals into the one slot a DTO has for them, **settle first**.
///
/// The order is the chronology: the stroke was discarded before the operation
/// refused, and the first sentence an author reads should be the one about work
/// they had already done. One `refusal` field can hold both because it is
/// prose — the panel prints it — and dropping either is the failure round 3
/// found at three of `dcc_unwrap`'s exits and two of `dcc_merge_asset`'s.
fn chain_refusals(settled: Option<String>, why: String) -> String {
    match settled {
        Some(first) => format!("{first} Also: {why}"),
        None => why,
    }
}

/// One frame of the **2D UV view**: the charts, the wireframe, the seams and the
/// shared selection, composited on the CPU and encoded like the 3D preview.
///
/// No GPU at all — a UV layout is flat lines on a flat square — which is why this
/// one cannot report "no adapter" and always has an image.
#[tauri::command]
pub async fn dcc_uv_preview(
    id: String,
    size: u32,
    state: State<'_, DccState>,
) -> Result<DccPreviewDto, String> {
    let size = size.clamp(64, 1024);
    let (mesh, selection, mode) = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        sync(doc, session);
        // The COMMITTED mesh, not the drag scratch: a sculpt moves positions and
        // leaves UVs alone, so a drag changes nothing in this view and rendering
        // the scratch would cost a tessellation for an identical picture.
        Ok((session.mesh().clone(), doc.selection.clone(), doc.mode))
    })?;
    let mut rgba = vec![0u8; (size as usize) * (size as usize) * 4];
    dcc::draw_uv_layout(
        &mut rgba,
        size,
        &mesh,
        &selection,
        mode,
        &dcc::UvStyle::default(),
    );
    let png = encode_png_fast(size, &rgba)?;
    Ok(DccPreviewDto {
        image: Some(format!(
            "data:image/png;base64,{}",
            super::assets::base64(&png)
        )),
        error: None,
        size,
    })
}

/// Drop another mesh asset into this document as a second component — the
/// drag-and-drop modularity seed (see [`inf_editor_core::dcc::merge_into`] for
/// exactly what v1 does and does not do).
///
/// The part lands **beside** the model rather than inside it: an offset of the
/// two bounding radii along +X, so an author can see what arrived and move it
/// deliberately. Dropping it at the pointer would need a ray-vs-nothing solve
/// (there is no ground plane in a model editor) and would put kit pieces inside
/// each other as often as not.
#[tauri::command]
pub async fn dcc_merge_asset(
    app: AppHandle,
    id: String,
    asset_id: String,
    state: State<'_, DccState>,
    assets: State<'_, super::assets::AssetState>,
) -> Result<DccApplyDto, String> {
    let incoming_id: AssetId = asset_id.parse().map_err(|e| format!("bad asset id: {e}"))?;
    let payload = assets.with_project(|proj| {
        let entry = proj
            .db()
            .get(incoming_id)
            .ok_or_else(|| format!("no asset {incoming_id}"))?;
        if entry.kind() != AssetKind::Mesh {
            return Err(format!("{} is not a mesh asset", entry.name));
        }
        proj.load_payload::<inf_mesh::MeshAsset>(incoming_id)
            .map_err(|e| e.to_string())
    })?;
    let incoming = from_mesh_asset(&payload).map_err(|e| e.to_string())?;

    let out = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        // **The SIXTH C4-34 door** (round-2 finding R2.F5; the count corrected
        // in round 3 — five doors drop the refusal, and this is the one that
        // must not). The comment that
        // stood here said this door "answers with a bare `DccDocDto`, which has
        // nowhere to carry a refusal" — and it does not: it answers with a
        // `DccApplyDto`, whose `refusal` field it populates twenty lines below
        // from a different cause. So the settle refusal was dropped by
        // `let _ =` (which satisfies `#[must_use]`) into a DTO that had a slot
        // for it, and the structural gate saw only that `settle_reported` had
        // been called. A merge that silently discarded the stroke in flight
        // reported plain success.
        let settled = settle_reported(doc, session, "dcc_merge_asset");
        sync(doc, session);
        let here = dcc::tessellate(session.mesh()).bounds;
        let there = dcc::tessellate(&incoming.mesh).bounds;
        let offset = glam::DVec3::new((here.max[0] - there.min[0]) as f64 + 0.25, 0.0, 0.0);
        // **`None` rig, deliberately** (P24.3). `merge_into` can carry a skin,
        // and this door cannot supply what that needs: a `.inf_mesh` records no
        // skeleton (`SkinBinding::skeleton` is `None` on import, by design — the
        // pairing lives in the scene's `SkeletalMesh`), so merging two RIGS means
        // the author naming both skeletons and an attach socket. That is the
        // Skeleton Editor's job, and it also needs a `SkeletonAsset` writer to
        // put the merged rig back. Until then this drops geometry and materials
        // — which is strictly more than P23.4 did, since the slot table now comes
        // with the part.
        match dcc::merge_into(session, &incoming.mesh, offset, None) {
            Ok(r) => {
                // `sync`, not a bare `selection.sync` — the module's own door
                // also clears the knife path and `last_edge`, which name ids the
                // merge has just renumbered. Re-implementing half of it here is
                // how a stale knife survived a merge.
                sync(doc, session);
                let merged = r.faces > 0;
                Ok(DccApplyDto {
                    // A refused stroke means the model on disk will not contain
                    // what the author was looking at, so it is not an `ok` merge
                    // even when the geometry landed.
                    ok: merged && settled.is_none(),
                    // Round 3: the `(false, _)` arm used to answer with the
                    // merge's own message alone and drop the settle refusal —
                    // the same defect R2.F5 closed at this door, in the branch
                    // beside the one it closed.
                    refusal: match (merged, settled) {
                        (false, s) => Some(chain_refusals(
                            s,
                            "the dropped mesh had no faces".to_string(),
                        )),
                        (true, Some(why)) => Some(why),
                        (true, None) => None,
                    },
                    doc: doc_dto(doc, session),
                })
            }
            Err(e) => Ok(DccApplyDto {
                ok: false,
                refusal: Some(chain_refusals(settled, refusal_text(&e))),
                doc: doc_dto(doc, session),
            }),
        }
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(out)
}

/// The content sub-folders authored garments and hairstyles land in.
const CLOTH_FOLDER: &str = "Cloth";
const HAIR_FOLDER: &str = "Hair";

/// Resolve an optional `.inf_skel` id into the skeleton whose bones become the
/// collision capsules.
///
/// A missing or mistyped id is a **refusal by name**, not a silent `None`: a
/// garment authored against "the character" and quietly given no capsules would
/// fall straight through its wearer the first time it ran, and nothing would have
/// said so.
fn resolve_rig(
    assets: &State<'_, super::assets::AssetState>,
    id: Option<&String>,
) -> Result<Option<inf_anim::Skeleton>, String> {
    let Some(id) = id else { return Ok(None) };
    let guid: AssetId = id.parse().map_err(|e| format!("bad skeleton id: {e}"))?;
    let asset = assets.with_project(|proj| {
        let entry = proj
            .db()
            .get(guid)
            .ok_or_else(|| format!("no asset {guid}"))?;
        if entry.kind() != AssetKind::Skeleton {
            return Err(format!("{} is not a skeleton asset", entry.name));
        }
        proj.load_payload::<inf_anim::SkeletonAsset>(guid)
            .map_err(|e| e.to_string())
    })?;
    Ok(Some(asset.skeleton))
}

/// A refusal, as a value the panel prints (the P21 law).
fn groom_refusal(message: String) -> DccGroomResultDto {
    DccGroomResultDto {
        ok: false,
        refusal: Some(message),
        asset: None,
        path: None,
        stats: Vec::new(),
    }
}

fn stat(label: &str, value: usize) -> DccGroomStatDto {
    DccGroomStatDto {
        label: label.to_string(),
        value: value.min(u32::MAX as usize) as u32,
    }
}

/// **Make a garment** out of the open mesh (P24.4): write a `.inf_cloth` under
/// `Content/Cloth` and return its GUID.
///
/// Four lines of Ring 2 around one Ring-1 call, for the reason this module's docs
/// give: the rule that decides what a garment IS lives in
/// `inf_editor_core::groom`, where a test on any CI leg can drive it.
///
/// The **selected vertices are the pins**, and they are resolved backend-side
/// from the document's own selection — a tool press's shape, not a second copy of
/// the selection in the panel.
///
/// The new asset depends on the source `.inf_mesh` (and on the `.inf_skel` when
/// one was named), so the cook's dependency closure carries them and deleting one
/// warns.
#[tauri::command]
pub async fn dcc_make_garment(
    app: AppHandle,
    id: String,
    spec: DccGarmentDto,
    state: State<'_, DccState>,
    assets: State<'_, super::assets::AssetState>,
) -> Result<DccGroomResultDto, String> {
    let rig = match resolve_rig(&assets, spec.skeleton.as_ref()) {
        Ok(r) => r,
        Err(e) => return Ok(groom_refusal(e)),
    };
    let (source, mesh, selection, doc_name, settled) = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        let settled = settle_reported(doc, session, "dcc_make_garment");
        Ok((
            doc.asset,
            session.mesh().clone(),
            doc.selection.clone(),
            doc.name.clone(),
            settled,
        ))
    })?;
    // C4-34: the garment is cut from a mesh that does NOT contain the stroke the
    // author can see. Refuse rather than build a garment they did not model.
    if let Some(why) = settled {
        return Ok(groom_refusal(format!(
            "the stroke in flight was refused, so the mesh this would be cut from is not \
             what is on screen — {why}"
        )));
    }

    let built = inf_editor_core::groom::garment_from_session(
        &mesh,
        &selection,
        *source.0.as_bytes(),
        inf_editor_core::groom::GarmentSpec {
            material: inf_anim::ClothMaterial {
                stretch_compliance: spec.stretch_compliance,
                bend_compliance: spec.bend_compliance,
                damping: spec.damping,
                thickness_m: spec.thickness_m,
                substeps: spec.substeps,
                iterations: spec.iterations,
            },
            body_radius_m: spec.body_radius_m,
        },
        rig.as_ref(),
    );
    let (asset, report) = match built {
        Ok(v) => v,
        Err(e) => return Ok(groom_refusal(e.to_string())),
    };

    let name = spec.name.unwrap_or_else(|| format!("{doc_name} Garment"));
    let mut deps = vec![source];
    if let Some(sk) = spec
        .skeleton
        .as_ref()
        .and_then(|s| s.parse::<AssetId>().ok())
    {
        deps.push(sk);
    }
    let (new_id, path) = assets.with_project(|proj| {
        let dir = proj.content_dir(CLOTH_FOLDER).map_err(|e| e.to_string())?;
        let id = proj
            .write_asset(&dir, &name, &asset, None, deps, None)
            .map_err(|e| e.to_string())?;
        let path = proj
            .db()
            .get(id)
            .map(|e| e.path.display().to_string())
            .unwrap_or_default();
        Ok((id, path))
    })?;
    super::assets::emit_changed(&app, &assets);

    Ok(DccGroomResultDto {
        ok: true,
        refusal: None,
        asset: Some(new_id.to_string()),
        path: Some(path),
        stats: vec![
            stat("particles", report.particles),
            stat("triangles", report.triangles),
            stat("stretch", report.stretch),
            stat("bend", report.bend),
            stat("pinned", report.pinned),
            stat("capsules", report.capsules),
        ],
    })
}

/// **Grow guides** out of the open mesh's face selection (P24.4): write an
/// `.inf_hair` under `Content/Hair` and return its GUID.
///
/// This is the control `HairAsset::UPGRADE_REMEDY` has been naming since the
/// strand batch landed — "Model Editor ▸ Hair ▸ Grow Guides" — and until now it
/// did not exist.
#[tauri::command]
pub async fn dcc_grow_hair(
    app: AppHandle,
    id: String,
    spec: DccGroomDto,
    state: State<'_, DccState>,
    assets: State<'_, super::assets::AssetState>,
) -> Result<DccGroomResultDto, String> {
    let rig = match resolve_rig(&assets, spec.skeleton.as_ref()) {
        Ok(r) => r,
        Err(e) => return Ok(groom_refusal(e)),
    };
    let (source, mesh, selection, doc_name, settled) = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        let settled = settle_reported(doc, session, "dcc_grow_hair");
        Ok((
            doc.asset,
            session.mesh().clone(),
            doc.selection.clone(),
            doc.name.clone(),
            settled,
        ))
    })?;
    // C4-34, as above: a scalp taken from a mesh missing the author's last stroke.
    if let Some(why) = settled {
        return Ok(groom_refusal(format!(
            "the stroke in flight was refused, so the scalp this would be grown on is not \
             what is on screen — {why}"
        )));
    }

    let built = inf_editor_core::groom::groom_from_session(
        &mesh,
        &selection,
        *source.0.as_bytes(),
        inf_editor_core::groom::GroomSpec {
            length_m: spec.length_m,
            segments: spec.segments,
            material: inf_anim::HairMaterial {
                segment_compliance: spec.segment_compliance,
                damping: spec.damping,
                thickness_m: spec.thickness_m,
                substeps: spec.substeps,
                iterations: 1,
                ribbon_width_m: spec.ribbon_width_m,
            },
            groom: inf_anim::HairGroom {
                clump_strength: spec.clump_strength,
                curl_radius_m: spec.curl_radius_m,
                curl_turns: spec.curl_turns,
            },
            clump_spacing_m: spec.clump_spacing_m,
            fallback_joint: spec.fallback_joint,
            body_radius_m: spec.body_radius_m,
        },
        rig.as_ref(),
    );
    let (asset, report) = match built {
        Ok(v) => v,
        Err(e) => return Ok(groom_refusal(e.to_string())),
    };

    let name = spec.name.unwrap_or_else(|| format!("{doc_name} Hair"));
    let mut deps = vec![source];
    if let Some(sk) = spec
        .skeleton
        .as_ref()
        .and_then(|s| s.parse::<AssetId>().ok())
    {
        deps.push(sk);
    }
    let (new_id, path) = assets.with_project(|proj| {
        let dir = proj.content_dir(HAIR_FOLDER).map_err(|e| e.to_string())?;
        let id = proj
            .write_asset(&dir, &name, &asset, None, deps, None)
            .map_err(|e| e.to_string())?;
        let path = proj
            .db()
            .get(id)
            .map(|e| e.path.display().to_string())
            .unwrap_or_default();
        Ok((id, path))
    })?;
    super::assets::emit_changed(&app, &assets);

    Ok(DccGroomResultDto {
        ok: true,
        refusal: None,
        asset: Some(new_id.to_string()),
        path: Some(path),
        stats: vec![
            stat("strands", report.strands),
            stat("particles", report.particles),
            stat("clumps", report.clumps),
            stat("skinnedRoots", report.skinned_roots),
            stat("capsules", report.capsules),
        ],
    })
}

fn refusal_text(e: &OpError) -> String {
    e.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seed a document directly (the real `dcc_open` needs a Tauri `State` and an
    /// asset project, neither of which a unit test can build).
    fn seed(state: &DccState, id: &str, mesh: inf_dcc::Mesh) {
        let session = MeshSession::new(mesh);
        state
            .with(|s| {
                // Through `new_doc`, the one constructor `dcc_open` and `dcc_new`
                // both use — so a field added to `DccDoc` cannot arrive with one
                // default in the product and another in the tests.
                let mut doc = new_doc(
                    id.to_string(),
                    AssetId::new(),
                    "T".into(),
                    session.mesh(),
                    &session,
                    ImportReport::default(),
                );
                doc.view = PreviewView::default();
                s.docs.insert(id.to_string(), doc);
                s.journals.insert(id.to_string(), session);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn close_frees_the_doc_and_journal() {
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        assert!(
            !state.close("dcc:1").unwrap(),
            "nothing was in flight, so nothing was abandoned"
        );
        state
            .with(|s| {
                assert!(s.docs.is_empty(), "doc freed");
                assert!(s.journals.is_empty(), "journal freed");
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn a_tool_with_nothing_selected_refuses_rather_than_doing_nothing() {
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                let ops = build_ops(doc, session, &DccToolDto::Extrude { distance: 1.0 })?;
                assert!(ops.is_empty(), "no faces selected, no ops");
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn the_tool_operands_are_the_selection() {
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                let f = session.mesh().face_ids().next().unwrap();
                doc.selection.set_face(f, true);
                let ops = build_ops(doc, session, &DccToolDto::Extrude { distance: 0.5 })?;
                assert_eq!(
                    ops,
                    vec![Op::ExtrudeFaces {
                        faces: vec![f],
                        distance: 0.5
                    }]
                );
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn deleting_vertices_removes_their_faces_first() {
        // `RemoveVertex` refuses while anything is attached, so a vertex-mode
        // delete that only emitted `RemoveVertex` would refuse every time. The
        // faces go first, which is what every modeller means by the word.
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::plane(2.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                doc.mode = SelectMode::Vert;
                let v = session.mesh().vert_ids().next().unwrap();
                doc.selection.set_vert(v, true);
                let ops = build_ops(doc, session, &DccToolDto::Delete)?;
                assert!(matches!(ops[0], Op::RemoveFace { .. }), "{ops:?}");
                assert!(
                    matches!(ops.last(), Some(Op::RemoveVertex { .. })),
                    "{ops:?}"
                );
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn a_soft_translate_scales_by_weight_and_is_one_op() {
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(2.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                doc.mode = SelectMode::Vert;
                let v = session.mesh().vert_ids().next().unwrap();
                doc.selection.set_vert(v, true);
                let ops = build_ops(
                    doc,
                    session,
                    &DccToolDto::SoftTranslate {
                        delta: [0.0, 1.0, 0.0],
                        radius: 4.0,
                        falloff: SculptFalloffDto::Linear,
                    },
                )?;
                // Wave D: the whole proportional drag is ONE op carrying a
                // weight table, not one op per distinct weight. Same numbers,
                // one undo press.
                assert_eq!(ops.len(), 1, "a soft drag is one op now: {ops:?}");
                let Op::MoveVerts { moves } = &ops[0] else {
                    panic!("{:?}", ops[0])
                };
                assert!(moves.len() > 1, "the neighbourhood moves too: {moves:?}");
                let full = moves.iter().filter(|(_, d)| d[1] == 1.0).count();
                assert_eq!(full, 1, "exactly one vertex moves at full weight");
                for (_, d) in moves {
                    assert!(d[1] > 0.0 && d[1] <= 1.0, "{d:?}");
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn the_save_advisories_say_what_the_counters_mean() {
        let clean = ExportReport::default();
        assert!(advisories(&clean).is_empty());
        let dirty = ExportReport {
            coincident_vertices: 2,
            reused_diagonals: 1,
            ..Default::default()
        };
        let said = advisories(&dirty);
        assert_eq!(said.len(), 2);
        assert!(said[0].contains("FUSE"), "{said:?}");
        assert!(said[1].contains("non-manifold"), "{said:?}");
    }

    #[test]
    fn dirty_is_the_generation_stamp_not_a_flag() {
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                assert!(
                    !doc_dto(doc, session).dirty,
                    "a freshly opened doc is clean"
                );
                let f = session.mesh().face_ids().next().unwrap();
                session
                    .apply(Op::ExtrudeFaces {
                        faces: vec![f],
                        distance: 0.5,
                    })
                    .unwrap();
                assert!(doc_dto(doc, session).dirty);
                doc.saved_generation = Some(session.generation());
                assert!(!doc_dto(doc, session).dirty);
                // And an undo past the save reads dirty again: the mesh may match
                // what is on disk, the JOURNAL does not, and the panel says
                // "unsaved edits" rather than "different bytes".
                session.undo();
                assert!(doc_dto(doc, session).dirty);
                Ok(())
            })
            .unwrap();
    }

    // ── P23.5: drags, settlers, and the gizmo/numeric equivalence ──────────

    /// A stroke over the first vertex of the seeded mesh, ready to settle.
    fn stroke_of(session: &MeshSession) -> PendingDrag {
        let start = session
            .mesh()
            .position(session.mesh().vert_ids().next().expect("a vertex"))
            .expect("live");
        PendingDrag::Stroke(inf_editor_core::dcc::StrokeInFlight {
            mode: inf_dcc::SculptMode::Draw,
            radius: 0.9,
            strength: 0.1,
            falloff: inf_dcc::SculptFalloff::Smooth,
            path: (0..5)
                .map(|i| start + glam::DVec3::X * (i as f64 * 0.1))
                .collect(),
            last_normal: glam::DVec3::Y,
        })
    }

    #[test]
    fn every_journal_door_settles_a_drag_in_flight() {
        // The orphan-settler doctrine, measured at the door rather than asserted
        // in a comment: a stroke still in flight when the author presses a tool,
        // changes selection, undoes or saves becomes ONE journalled edit first.
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                doc.pending = Some(stroke_of(session));
                let before = session.mesh().encoded();
                assert_eq!(settle(doc, session), None);
                assert_eq!(session.ops().len(), 1, "one entry for the whole stroke");
                assert!(doc.pending.is_none(), "and the slot is empty afterwards");
                assert_ne!(session.mesh().encoded(), before);
                // Settling again is a no-op, which is what makes calling it at
                // every door cheap enough to actually do.
                assert_eq!(settle(doc, session), None);
                assert_eq!(session.ops().len(), 1);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn a_settled_stroke_keeps_the_selection_it_was_painted_on() {
        // Every op a drag makes is `op_preserves_ids`, so the settle CARRIES the
        // selection. Dropping it would deselect the face under the brush on every
        // tool switch — which is what `sync` would have done if `settle` had not
        // been the thing that ran first.
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                let f = session.mesh().face_ids().next().unwrap();
                doc.selection.set_face(f, true);
                doc.pending = Some(stroke_of(session));
                let _ = settle_reported(doc, session, "dcc_grow_hair");
                sync(doc, session);
                assert_eq!(doc.selection.len(SelectMode::Face), 1);
                assert!(doc.selection.contains_face(f));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn a_close_abandons_a_drag_while_every_other_door_settles_it() {
        // The other half of the doctrine. `DccState::close` frees the doc and the
        // journal together, so an op applied first could never be undone, saved
        // or seen — the abandon is the decision. This test is what stops it
        // quietly becoming a settle, and what stops the settle above quietly
        // becoming an abandon.
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        let before = state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                doc.pending = Some(stroke_of(session));
                Ok(session.mesh().encoded())
            })
            .unwrap();
        assert!(
            state.close("dcc:1").unwrap(),
            "the close must REPORT the abandon — without that the decision is \
             unobservable, and a close that settled instead would pass this test"
        );
        state
            .with(|s| {
                assert!(s.docs.is_empty(), "doc freed");
                assert!(s.journals.is_empty(), "journal freed, drag and all");
                Ok(())
            })
            .unwrap();
        assert!(!before.is_empty());
    }

    /// What a command does about a drag that is still in flight.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum DragPolicy {
        /// Touches the journal, so it settles first.
        Settles,
        /// Deliberately discards the drag - the two documented exceptions.
        Abandons,
        /// Reads, renders or moves the camera; the drag is none of its business.
        NoJournal,
        /// The one door whose whole job is to EXTEND the drag.
        Extends,
    }

    /// This module's own source, read by the gate below.
    const SOURCE: &str = include_str!("dcc.rs");

    /// **Every command, and what it does about a pending drag.**
    ///
    /// Hand-written on purpose: a table derived from the code would agree with the
    /// code by construction and prove nothing. This is the decision, written down
    /// once, and the gate holds the code to it.
    const DRAG_POLICY: [(&str, DragPolicy); 27] = [
        ("dcc_open", DragPolicy::NoJournal),
        // Wave D. `dcc_new` touches no existing document at all — it mints one —
        // so there is nothing of the author's in flight for it to settle or
        // abandon. The other two are ordinary selection/setting doors and settle,
        // like `dcc_pick` and `dcc_set_gizmo` beside them.
        ("dcc_new", DragPolicy::NoJournal),
        ("dcc_box_select", DragPolicy::Settles),
        ("dcc_set_view_opts", DragPolicy::Settles),
        // The history panel. `dcc_history` READS and journals nothing;
        // `dcc_amend` rewrites the journal and therefore settles first, exactly
        // as undo does — an amendment with a drag in flight must neither throw
        // the drag away nor be replayed underneath it.
        ("dcc_history", DragPolicy::NoJournal),
        ("dcc_amend", DragPolicy::Settles),
        ("dcc_close", DragPolicy::Abandons),
        ("dcc_list", DragPolicy::NoJournal),
        ("dcc_apply", DragPolicy::Settles),
        ("dcc_select", DragPolicy::Settles),
        ("dcc_pick", DragPolicy::Settles),
        ("dcc_set_gizmo", DragPolicy::Settles),
        ("dcc_drag_begin", DragPolicy::Settles),
        ("dcc_drag_move", DragPolicy::Extends),
        ("dcc_drag_end", DragPolicy::Settles),
        ("dcc_drag_cancel", DragPolicy::Abandons),
        ("dcc_orbit", DragPolicy::NoJournal),
        ("dcc_frame", DragPolicy::NoJournal),
        ("dcc_undo", DragPolicy::Settles),
        ("dcc_redo", DragPolicy::Settles),
        ("dcc_preview", DragPolicy::NoJournal),
        ("dcc_save", DragPolicy::Settles),
        ("dcc_merge_asset", DragPolicy::Settles),
        ("dcc_unwrap", DragPolicy::Settles),
        ("dcc_uv_preview", DragPolicy::NoJournal),
        // P24.4. Both SETTLE: an authored garment is built out of the mesh as the
        // author sees it, and a sculpt stroke still in flight is part of what they
        // see. A door that read the journal past an unsettled drag would mint a
        // .inf_cloth of the shape the mesh had one gesture ago.
        ("dcc_make_garment", DragPolicy::Settles),
        ("dcc_grow_hair", DragPolicy::Settles),
    ];

    /// The lines of a function body, from its signature until its braces balance.
    fn body_of(source: &str, signature: &str) -> String {
        let lines: Vec<&str> = source.lines().collect();
        let start = lines
            .iter()
            .position(|l| l.contains(signature))
            .unwrap_or_else(|| panic!("no line contains `{signature}`"));
        let mut depth: i32 = 0;
        // **`opened` is load-bearing.** A signature whose arguments are on the
        // following lines has no `{` on its own line, so a "stop when the depth is
        // zero" rule stops on argument two and reports an empty body — which reads
        // as "this command does not settle" and fails a door that does.
        let mut opened = false;
        let mut out: Vec<&str> = Vec::new();
        for line in &lines[start..] {
            out.push(line);
            for ch in line.chars() {
                match ch {
                    '{' => {
                        depth += 1;
                        opened = true;
                    }
                    '}' => depth -= 1,
                    _ => {}
                }
            }
            if opened && depth <= 0 {
                break;
            }
        }
        out.join("\n")
    }

    #[test]
    fn every_journal_door_declares_whether_it_settles_a_drag() {
        // **The gate the settle/abandon doctrine did not have.** The rule was
        // eleven hand-written statements spread across twenty commands, and its
        // cited test did not exist: the real one called the private `settle()`
        // directly, which proves `settle` works and says *nothing* about which
        // doors call it. A twenty-first command defaults to NOT settling, and that
        // failure is silent, ordering-shaped, and only reachable when a pointer-up
        // goes missing.
        //
        // So the source is read: every `pub async fn` in this file must be in the
        // table above, and its body must match the policy it declared. A new door
        // fails this test until somebody decides what it does.
        let found: Vec<String> = SOURCE
            .lines()
            .filter_map(|l| l.trim().strip_prefix("pub async fn "))
            .filter_map(|l| l.split('(').next())
            .map(str::to_string)
            .collect();
        let mut a = found.clone();
        let mut b: Vec<String> = DRAG_POLICY.iter().map(|(n, _)| n.to_string()).collect();
        a.sort();
        b.sort();
        assert_eq!(
            a, b,
            "a command is missing from DRAG_POLICY (or the table names one that no \
             longer exists). A new door must CHOOSE."
        );

        for (name, policy) in DRAG_POLICY {
            let body = body_of(SOURCE, &format!("pub async fn {name}("));
            // C4-34: the spelling is `settle_reported`, the ONE door, because
            // `settle` is now `#[must_use]` and a caller that dropped its refusal
            // would not compile. Looking for the bare `settle(` would also match
            // the wrapper's own definition and pass for a door that never calls it.
            let settles = body.contains("settle_reported(doc, session");
            match policy {
                DragPolicy::Settles => assert!(
                    settles,
                    "{name} is declared `Settles` and never calls `settle`: a drag \
                     in flight when it runs is lost"
                ),
                DragPolicy::Abandons | DragPolicy::NoJournal | DragPolicy::Extends => {
                    assert!(
                        !settles,
                        "{name} is declared {policy:?} but calls `settle` - it has \
                         changed class without the table being told"
                    )
                }
            }
        }

        // The abandons are the exceptions, so there must be exactly two and they
        // must be the two that are documented as such.
        let abandons: Vec<&str> = DRAG_POLICY
            .iter()
            .filter(|(_, p)| *p == DragPolicy::Abandons)
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(
            abandons,
            vec!["dcc_close", "dcc_drag_cancel"],
            "the deliberate abandons are `close` (the journal goes with it) and \
             `cancel` (the author said no). A third is a decision, not a detail."
        );
        assert_eq!(
            DRAG_POLICY
                .iter()
                .filter(|(_, p)| *p == DragPolicy::Extends)
                .count(),
            1,
            "exactly one door exists to GROW a drag"
        );
    }

    #[test]
    fn the_gizmo_and_the_numeric_tool_journal_the_same_op() {
        // Deliverable 2's contract at the PRODUCT boundary, not just in Ring 1:
        // `build_ops` is what a toolbar press goes through and `PendingDrag::ops`
        // is what a released handle goes through, and for the same delta the two
        // must produce byte-equal journal entries. A gizmo that quietly emitted a
        // different op would make undo behave differently depending on how the
        // author moved something.
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                let f = session.mesh().face_ids().next().unwrap();
                doc.selection.set_face(f, true);
                let delta = [0.25, 0.0, 0.0];

                let numeric = build_ops(doc, session, &DccToolDto::Translate { delta })?;
                let pivot =
                    dcc::gizmo_pivot(session.mesh(), &doc.selection, doc.mode).expect("a pivot");
                let dragged = PendingDrag::Gizmo(Box::new(GizmoInFlight {
                    drag: GizmoDrag::begin(
                        GizmoMode::Translate,
                        inf_editor_core::dcc::GizmoAxis::X,
                        glam::Quat::IDENTITY,
                        glam::Vec3::ZERO,
                        glam::Vec3::new(0.0, 0.0, 5.0),
                        glam::Vec3::new(0.0, 0.0, -1.0),
                    ),
                    pivot,
                    xform: VertTransform::Translate(glam::DVec3::from_array(delta)),
                    soft: None,
                }))
                .ops(session.mesh(), &doc.selection, doc.mode);

                assert_eq!(numeric, dragged, "the twin tools disagree");
                assert_eq!(numeric.len(), 1);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn a_click_that_never_dragged_journals_nothing() {
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                for f in session.mesh().face_ids().collect::<Vec<_>>() {
                    doc.selection.set_face(f, true);
                }
                let pivot =
                    dcc::gizmo_pivot(session.mesh(), &doc.selection, doc.mode).expect("a pivot");
                doc.pending = Some(PendingDrag::Gizmo(Box::new(GizmoInFlight {
                    drag: GizmoDrag::begin(
                        GizmoMode::Translate,
                        inf_editor_core::dcc::GizmoAxis::X,
                        glam::Quat::IDENTITY,
                        glam::Vec3::ZERO,
                        glam::Vec3::new(0.0, 0.0, 5.0),
                        glam::Vec3::new(0.0, 0.0, -1.0),
                    ),
                    pivot,
                    xform: VertTransform::Translate(glam::DVec3::ZERO),
                    soft: None,
                })));
                assert_eq!(settle(doc, session), None);
                assert_eq!(session.ops().len(), 0, "an undo step for a click");
                assert!(!doc_dto(doc, session).dragging);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn the_document_reports_the_drag_the_panel_owes_an_end_for() {
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                assert!(!doc_dto(doc, session).dragging);
                doc.pending = Some(stroke_of(session));
                let dto = doc_dto(doc, session);
                assert!(dto.dragging);
                assert_eq!(dto.drag_points, 5);
                assert_eq!(dto.gizmo, None);
                doc.gizmo = Some(GizmoMode::Rotate);
                assert_eq!(
                    doc_dto(doc, session).gizmo,
                    Some(inf_editor_core::ipc::DccGizmoModeDto::Rotate)
                );
                Ok(())
            })
            .unwrap();
    }
    /// The asset tick's source, read by the gate below.
    const ASSETS_SOURCE: &str = include_str!("assets.rs");

    /// **The statement-level lines of a body**, comments and blanks removed.
    ///
    /// "Statement level" is literally the indentation of the construct's own
    /// scope: `indent` spaces and no more. That is what makes the check see a
    /// *condition* — wrapping a call in `if …` moves its line one level in and
    /// takes it out of this set — and it is why the positives below are an
    /// ordered comparison against a list rather than a `.contains()` over a whole
    /// body, which a comment satisfies (P23.5's M1).
    fn statements(body: &str, indent: usize) -> Vec<String> {
        let pad = " ".repeat(indent);
        body.lines()
            .skip(1) // the signature line itself
            .filter(|l| l.starts_with(&pad) && !l[indent..].starts_with(' '))
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with("//"))
            .collect()
    }

    /// A body with its `//` comments stripped, for the bans — so a ban cannot
    /// fire on prose and, more importantly, cannot be *satisfied* by it either.
    fn code_only(body: &str) -> String {
        body.lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// **What may make the save's invalidation conditional: nothing.**
    ///
    /// An ALLOWLIST, not a ban list. The first version of this gate banned four
    /// spellings (`SimState`, `sim_is_running`, `is_running`, `sim::`) and a fifth
    /// — `paused()`, `PlayState`, any freshly-named helper — walked straight
    /// through. What is checked instead is the whole statement *sequence*: a new
    /// statement in this door, of any name, fails until somebody adds it here on
    /// purpose. The P22 law one layer up — **a ban enumerates what you thought of,
    /// an allowlist what is allowed.**
    const DCC_SAVE_TAIL: [&str; 8] = [
        "viewport.refresh_asset_index(super::Target::All);",
        "let _ = app.emit(\"assets://changed\", ());",
        "let _ = app.emit(\"dcc://sync\", id);",
        // C4-34: the refused stroke becomes the first advisory and decides `ok`.
        // Two statements, added deliberately, which is what this pin is for.
        "let mut advisories = advisories(&report);",
        "if let Some(why) = &settled {",
        "}",
        "Ok(DccSaveDto {",
        "})",
    ];

    /// The tick's refresh is legitimately nested — inside `if outcome.index_stale`
    /// and then inside the state lookup — so its allowlist is the *guard chain*,
    /// pinned line by line. A third rung on that chain fails here, which is the
    /// one place an extra condition would look natural.
    const TICK_REFRESH_CHAIN: [&str; 3] = [
        "if outcome.index_stale {",
        "if let Some(viewport) = app.try_state::<super::ViewportState>() {",
        "viewport.refresh_asset_index(super::Target::All);",
    ];

    /// Module and concept substrings a refresh must not name. A backstop under the
    /// allowlists above, never the primary check.
    const SIM_TOKENS: [&str; 6] = ["sim", "Sim", "play", "Play", "paus", "Paus"];

    /// **The P23.6 headline's missing link, held where a test can reach it.**
    ///
    /// `dcc_edit_during_simulate.rs` executes both *ends* of the live-update chain
    /// — the save writes new bytes, and `EditorRenderAssets` re-keys on them while
    /// a `SimSession` is alive. The link between them is this function's push, and
    /// a `#[tauri::command]` cannot be driven from a test on any CI leg. So it is
    /// read instead.
    ///
    /// Three things are read, and each closes a hole the first version had:
    ///
    /// 1. **The save's tail is an exact statement sequence** at the body's own
    ///    indentation. Wrapping the push in any condition moves it a level in and
    ///    fails; inserting a statement before it fails; deleting it fails.
    /// 2. **The tick's guard chain is pinned line by line.** The old version used
    ///    `.contains()` over the whole body, which a comment satisfies.
    /// 3. **The bans run over code with comments stripped**, and they ban
    ///    substrings of module and concept names rather than four spellings — with
    ///    the allowlists doing the real work, because a ban alone only catches what
    ///    its author imagined.
    #[test]
    fn the_save_pushes_its_invalidation_unconditionally() {
        let save = body_of(SOURCE, "pub async fn dcc_save(");
        let tail = statements(&save, 4);
        let at = tail
            .iter()
            .position(|l| l.starts_with("viewport.refresh_asset_index"))
            .unwrap_or_else(|| {
                panic!(
                    "dcc_save has no statement-level `viewport.refresh_asset_index` \
                     - a viewport that is not told keeps drawing the geometry the \
                     author just replaced, and a conditional push is a skip waiting \
                     for a condition to be true. Statements seen: {tail:?}"
                )
            });
        assert_eq!(
            &tail[at..],
            &DCC_SAVE_TAIL[..],
            "dcc_save's tail is not the sequence DCC_SAVE_TAIL declares. A new \
             statement here is a DECISION - add it to the list."
        );

        // The tick's half, by BLOCK EXTRACTION rather than a whole-body scan:
        // `body_of` brace-balances from the guard's own line, so what is pinned is
        // that block and not whatever else the tick happens to contain.
        let tick = body_of(ASSETS_SOURCE, "fn spawn_tick(");
        assert!(
            tick.lines()
                .any(|l| l == "            if outcome.index_stale {"),
            "`if outcome.index_stale` is no longer a statement of the tick loop \
             itself - something else now decides whether the viewport is told"
        );
        let block = body_of(ASSETS_SOURCE, "if outcome.index_stale {");
        let chain: Vec<String> = block
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| l.starts_with("if ") || l.starts_with("viewport.refresh_asset_index"))
            .collect();
        assert_eq!(
            chain,
            TICK_REFRESH_CHAIN
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            "the asset tick's refresh guard chain changed. It is pinned line by \
             line because the nesting here is legitimate and therefore the one \
             place an extra condition would look natural."
        );

        for (name, body) in [("dcc_save", &save), ("spawn_tick", &tick)] {
            let code = code_only(body);
            for token in SIM_TOKENS {
                assert!(
                    !code.contains(token),
                    "{name} names `{token}`: its invalidation has been made to \
                     depend on play state. An asset edit is not in the document, so \
                     Simulate has no opinion about it - see \
                     docs/memos/p23-dcc-design.md section 3."
                );
            }
        }
    }

    #[test]
    fn a_selection_is_dropped_when_an_op_moves_the_stamp() {
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        state
            .with(|s| {
                let (doc, session) = s.pair("dcc:1")?;
                let f = session.mesh().face_ids().next().unwrap();
                doc.selection.set_face(f, true);
                doc.knife.push(session.mesh().vert_ids().next().unwrap());
                session
                    .apply(Op::SubdivideFaces { faces: vec![f] })
                    .unwrap();
                sync(doc, session);
                assert_eq!(doc.selection.len(SelectMode::Face), 0);
                assert!(doc.knife.is_empty(), "the knife path went with it");
                Ok(())
            })
            .unwrap();
    }

    /// **Round-2 finding R2.F5, as the class rather than the instance.**
    ///
    /// `settle_reported` is `#[must_use]`-shaped and **five** doors legitimately
    /// drop its answer with `let _ =`, because the DTO they return has nowhere
    /// to put it. `dcc_merge_asset` did the same over a comment saying exactly
    /// that — and it returns a `DccApplyDto`, which has a `refusal` field it
    /// populates twenty lines further down. The structural gate could not see
    /// it: it checked only that `settle_reported` had been *called*.
    ///
    /// **The count in this sentence was wrong** (round 3). It said "ten doors",
    /// and the list below — which the census holds the module to — has always
    /// had five; the closing arm called `dcc_merge_asset` "the eleventh door"
    /// on the same arithmetic. Corrected rather than deleted, because the
    /// numbers are the only part of a doc a reader can check.
    ///
    /// So the pin is the correspondence itself. Every door that drops the
    /// refusal must return a DTO that genuinely cannot carry one, and that is
    /// asserted against `inf-editor-core`'s own source rather than against a
    /// belief about it. Adding a `refusal` slot to `DccDocDto` later — or
    /// writing a new `let _ = settle_reported` in a door whose DTO has one —
    /// fails here.
    #[test]
    fn no_door_drops_a_refusal_into_a_dto_that_could_carry_it() {
        // The P22 CRLF law: a `.rs` read by a test is normalized first.
        let ipc = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../crates/inf-editor-core/src/ipc.rs"),
        )
        .expect("the DTO source is readable")
        .replace("\r\n", "\n");

        /// Whether `struct <name>` in `ipc.rs` declares a `refusal` field.
        fn carries_a_refusal(ipc: &str, dto: &str) -> bool {
            let header = format!("pub struct {dto} {{");
            let start = ipc
                .find(&header)
                .unwrap_or_else(|| panic!("ipc.rs declares no `{dto}`"))
                + header.len();
            // `"\n}"` — the same escapes a scripted edit ate elsewhere in this
            // wave. Behaviourally identical (the source above is normalized), and
            // spelled so the next reader does not have to count blank lines.
            let end = start + ipc[start..].find("\n}").expect("the struct closes");
            ipc[start..end].contains("pub refusal:")
        }

        // The doors that drop it, paired with what they answer with. A list of
        // PAIRS, so the two halves cannot drift apart silently.
        let dropping: &[(&str, &str)] = &[
            ("pub async fn dcc_select(", "DccDocDto"),
            ("pub async fn dcc_pick(", "DccDocDto"),
            ("pub async fn dcc_set_gizmo(", "DccDocDto"),
            ("pub async fn dcc_undo(", "DccDocDto"),
            ("pub async fn dcc_redo(", "DccDocDto"),
            // Wave D's two selection/setting doors, same shape and same reason:
            // a `DccDocDto` has no `refusal` field, and `settle_reported` has
            // already put the sentence in the Output Log.
            ("pub async fn dcc_box_select(", "DccDocDto"),
            ("pub async fn dcc_set_view_opts(", "DccDocDto"),
        ];
        for (signature, dto) in dropping {
            let body = code_only(&body_of(SOURCE, signature));
            assert!(
                body.contains("let _ = settle_reported("),
                "{signature} no longer drops the refusal — move it out of this list \
                 and carry it, which is strictly better"
            );
            assert!(
                !carries_a_refusal(&ipc, dto),
                "{signature} drops the settle refusal into a {dto}, which HAS a \
                 `refusal` field — this is R2.F5 exactly, at another door"
            );
        }

        // …and the census that stops the list above from being a subset: every
        // `let _ = settle_reported` in this module must be one of them. The
        // PRODUCTION half only — this test module quotes the spelling it bans,
        // and a census that counts its own arms is measuring itself.
        let production = SOURCE
            .split_once("#[cfg(test)]")
            .map(|(head, _)| head)
            .expect("this module has a test half");
        let dropped_calls = code_only(production)
            .matches("let _ = settle_reported(")
            .count();
        assert_eq!(
            dropped_calls,
            dropping.len(),
            "a door drops the settle refusal and is not in this list; each one is a \
             decision that its DTO cannot carry the refusal, and it has to be made"
        );

        // The sixth door, from the other side: it must NOT drop it.
        let merge = code_only(&body_of(SOURCE, "pub async fn dcc_merge_asset("));
        assert!(
            !merge.contains("let _ = settle_reported("),
            "dcc_merge_asset returns a DccApplyDto and must carry the refusal"
        );
        assert!(
            merge.contains("let settled = settle_reported("),
            "…by binding it"
        );
        assert!(
            carries_a_refusal(&ipc, "DccApplyDto"),
            "the slot it goes in"
        );
    }

    /// **Round 3: carrying a refusal on the success path is not carrying it.**
    ///
    /// R2.F5 gave `dcc_unwrap` and `dcc_merge_asset` a settle refusal to carry,
    /// and both carried it only where the operation *succeeded*. Every failure
    /// exit answered with its own message alone — so the author whose sculpt
    /// stroke was discarded and whose unwrap then refused was told the second
    /// thing and not the first, which is the one that says their file will not
    /// contain what they were looking at.
    ///
    /// The chaining rule is unit-testable (it is a pure function) and the call
    /// sites are not, so both halves are asserted: the rule's behaviour, and
    /// that every refusal exit of the two commands goes through it.
    #[test]
    fn every_refusal_exit_carries_the_settle_refusal_too() {
        // The rule: settle first (it happened first), both survive, and a door
        // with nothing to chain is unchanged.
        assert_eq!(chain_refusals(None, "boom".into()), "boom");
        let both = chain_refusals(Some("the stroke was discarded".into()), "boom".into());
        assert!(
            both.contains("the stroke was discarded") && both.contains("boom"),
            "both halves must reach the panel: {both}"
        );
        assert!(
            both.starts_with("the stroke was discarded"),
            "the earlier event reads first: {both}"
        );

        // The call sites, on the text — `dcc_unwrap` and `dcc_merge_asset` take
        // `State<'_, DccState>` and are not constructible here.
        let unwrap = code_only(&body_of(SOURCE, "pub async fn dcc_unwrap("));
        let refused = unwrap.matches("unwrap_refused(").count();
        assert_eq!(
            refused, 3,
            "dcc_unwrap has three refusal exits; the count is the thing this arm \
             would otherwise drift past"
        );
        for call in unwrap.split("unwrap_refused(").skip(1) {
            // The argument list, up to the call's own closing `))`. Every inner
            // `)` in these calls is followed by a `,`, so the first `))` is the
            // end of the list.
            let args = &call[..call.find("))").unwrap_or(call.len())];
            assert!(
                args.contains("settled"),
                "a refusal exit hands `unwrap_refused` no settle refusal: {args}"
            );
        }

        // …and the helper they hand it to must USE it. Passing an argument is
        // not carrying it: `unwrap_refused` could take `settled` and answer with
        // `why` alone, and every call site above would still read correctly.
        // (Measured: that mutation survived the first version of this arm.)
        let helper = code_only(&body_of(SOURCE, "fn unwrap_refused("));
        assert!(
            helper.contains("chain_refusals(settled, why)"),
            "unwrap_refused takes the settle refusal and must combine it: {helper}"
        );

        let merge = code_only(&body_of(SOURCE, "pub async fn dcc_merge_asset("));
        assert_eq!(
            merge.matches("chain_refusals(").count(),
            2,
            "the no-faces arm and the kernel-error arm both dropped it"
        );

        // …and the helper is the ONLY way a refusal is combined, so a third
        // door cannot invent a different sentence for the same situation.
        let production = SOURCE
            .split_once("#[cfg(test)]")
            .map(|(head, _)| head)
            .expect("this module has a test half");
        assert_eq!(
            code_only(production).matches("fn chain_refusals(").count(),
            1,
            "one rule, one spelling"
        );
    }

    /// **The source pin for R2.F4.** The defect is a *skipped upload*, and no
    /// instrument in this repo can see one: the preview command returns a PNG
    /// either way, the cache's counters say a scratch was tessellated (it was),
    /// and only a human looking at a stale surface knows. `PreviewCache`'s own
    /// arm holds the stamp's semantics; this holds the call site to using it.
    #[test]
    fn the_preview_uploads_on_the_cache_s_own_stamp() {
        let body = code_only(&body_of(SOURCE, "pub async fn dcc_preview("));
        assert!(
            body.contains("doc.preview.upload_stamp()"),
            "dcc_preview must key the vertex-buffer upload on the stamp the CACHE \
             served, not on the journal generation — a live drag does not move the \
             generation, which is the whole reason the scratch channel exists"
        );
        assert!(
            !body.contains("session.generation(),"),
            "…and must not pass the generation as the wire stamp, which is R2.F4"
        );
    }

    /// **The source pin for R2.F14.** The finding is that a seconds-long solve
    /// ran inside `DccState::with`, and a command taking `State<'_, DccState>`
    /// plus an `AppHandle` is not constructible in a unit test — the same reason
    /// every settle check in this module is a source gate. So the two properties
    /// are pinned on the text: the solve is **outside** the lock, and the apply
    /// re-checks the stamp it solved against.
    #[test]
    fn the_unwrap_solves_outside_the_store_lock() {
        let body = code_only(&body_of(SOURCE, "pub async fn dcc_unwrap("));
        assert!(
            body.contains("spawn_blocking"),
            "`inf_dcc::unwrap` is a conjugate-gradient solve, seconds on a dense \
             mesh; `dcc.rs` states the rule 240 lines above and `dcc_save` follows it"
        );
        // The solve must not appear inside a `state.with(` block. Both `with`
        // blocks close before it, so the call sits at the function's own
        // statement level — which is exactly what `statements` measures.
        let stmts = statements(&body_of(SOURCE, "pub async fn dcc_unwrap("), 4);
        assert!(
            stmts.iter().any(|l| l.contains("spawn_blocking")),
            "the solve must be a statement of the command, not of a closure holding \
             the store lock: {stmts:?}"
        );
        assert!(
            body.contains("session.generation() != solved_at"),
            "an Op::Unwrap carries the corner list it solved for; applying it to a \
             mesh that moved meanwhile writes UVs onto corners nobody measured"
        );
    }

    /// **Round-2 finding R2.F15**: a project switch strands every open session.
    ///
    /// Asserted as WORLD state (the store is empty) rather than as "the call was
    /// made", and the count is returned so the switch can log it — a clear that
    /// dropped nothing and one that dropped three are otherwise the same event.
    #[test]
    fn closing_a_project_drops_every_model_session() {
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        seed(&state, "dcc:2", inf_dcc::cube(2.0));
        state
            .with(|s| {
                assert_eq!(s.docs.len(), 2);
                assert_eq!(s.journals.len(), 2, "and a journal each");
                Ok(())
            })
            .unwrap();

        assert_eq!(state.clear_all(), 2, "both were dropped, and it says so");
        state
            .with(|s| {
                assert!(s.docs.is_empty());
                assert!(
                    s.journals.is_empty(),
                    "the journal is the session — leaving it is the whole leak"
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(state.clear_all(), 0, "and it is idempotent");
    }

    /// The **source pin** for the half above that has no observable behaviour:
    /// `apply_open` has to call it. Nothing in a running editor distinguishes a
    /// switch that cleared from one that did not until a save fails, which is
    /// why the campaign's rule puts a gate on the text.
    #[test]
    fn the_project_switch_reaches_the_session_clear() {
        let project = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/commands/project.rs"),
        )
        .expect("project.rs is readable");
        let body = code_only(&body_of(&project, "fn apply_open("));
        assert!(
            body.contains("dcc.clear_all()"),
            "apply_open re-roots the asset database and every viewport; the Model \
             Editor sessions are keyed on ids in that database and must go with them"
        );
    }
}
