//! Ring-2 command surface for the **Model Editor** (ROADMAP P23.4).
//!
//! The fourth instance of the template `GraphState`, `MaterialState` and
//! `PcgState` already are — a document map, a journal map keyed the same way and
//! a counter — with two deliberate differences, both consequences of what a mesh
//! is:
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
use inf_editor_core::dcc::{self, OverlayStyle, PickHit, PreviewCache, Projector};
use inf_editor_core::ipc::{
    DccApplyDto, DccDocDto, DccExportDto, DccImportDto, DccModeDto, DccPreviewDto, DccSaveDto,
    DccSelectDto, DccToolDto, SculptFalloffDto,
};
use inf_editor_core::thumbnail::{encode_png_fast, PreviewView, Thumbnailer};
use tauri::{AppHandle, Emitter, State};

/// The preview edge length. 256², per the P23.2a measurement: the *render* is
/// 0.09 ms there and the transport is the real cost — 512² pushes ~1.4 MB of
/// base64 per frame through the webview bridge, which is ~42 MB/s at 30 fps.
const PREVIEW_SIZE: u32 = 256;

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
}

struct DccStore {
    docs: BTreeMap<String, DccDoc>,
    journals: BTreeMap<String, MeshSession>,
    counter: u32,
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
                counter: 0,
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
    fn close(&self, id: &str) -> Result<(), String> {
        self.with(|s| {
            s.docs.remove(id);
            s.journals.remove(id);
            Ok(())
        })
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
    }
}

fn export_dto(r: &ExportReport) -> DccExportDto {
    DccExportDto {
        submeshes: r.submeshes as u32,
        vertices: r.vertices as u32,
        triangles: r.triangles as u32,
        fan_fallbacks: r.fan_fallbacks as u32,
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
    }
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
    let view = dcc::frame(dcc::tessellate(&import.mesh).bounds);
    let session = MeshSession::new(import.mesh);
    let dto = state.with(|s| {
        s.counter += 1;
        let doc = DccDoc {
            id: doc_id.clone(),
            asset: id,
            name,
            mode: SelectMode::Face,
            selection: SelectionSet::new(session.generation()),
            view,
            import: import.report,
            saved_generation: Some(session.generation()),
            knife: Vec::new(),
            last_edge: None,
            preview: PreviewCache::new(),
        };
        let dto = doc_dto(&doc, &session);
        s.docs.insert(doc_id.clone(), doc);
        s.journals.insert(doc_id.clone(), session);
        Ok(dto)
    })?;
    let _ = app.emit("dcc://sync", dto.id.clone());
    Ok(dto)
}

/// Free a document and its journal.
#[tauri::command]
pub async fn dcc_close(id: String, state: State<'_, DccState>) -> Result<(), String> {
    state.close(&id)
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
        sync(doc, session);
        let ops = build_ops(doc, session, &tool)?;
        if ops.is_empty() {
            return Ok(DccApplyDto {
                ok: false,
                refusal: Some("nothing is selected".into()),
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
        DccToolDto::Bevel { amount } => {
            if edges.is_empty() {
                Vec::new()
            } else {
                vec![Op::BevelEdges {
                    edges,
                    amount: *amount,
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
        DccToolDto::Translate { delta } => {
            if verts.is_empty() {
                Vec::new()
            } else {
                vec![Op::TranslateVerts {
                    verts,
                    delta: *delta,
                }]
            }
        }
        DccToolDto::SoftTranslate {
            delta,
            radius,
            falloff,
        } => {
            // One `TranslateVerts` per distinct weight, so the whole soft move is
            // ordinary journal entries — and so an author who undoes gets their
            // shape back rather than a partially-relaxed one.
            let weights = doc
                .selection
                .soft_weights(mesh, doc.mode, *radius, falloff_of(*falloff));
            if weights.is_empty() {
                return Ok(Vec::new());
            }
            let mut by_weight: BTreeMap<u64, Vec<VertId>> = BTreeMap::new();
            for (v, w) in weights {
                by_weight.entry(w.to_bits()).or_default().push(v);
            }
            by_weight
                .into_iter()
                .map(|(bits, verts)| {
                    let w = f64::from_bits(bits);
                    Op::TranslateVerts {
                        verts,
                        delta: [delta[0] * w, delta[1] * w, delta[2] * w],
                    }
                })
                .collect()
        }
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
        sync(doc, session);
        let mesh = session.mesh();
        let proj = Projector::new(doc.view, size.max(1));
        match dcc::pick(mesh, &proj, doc.mode, x as f32, y as f32) {
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
        Ok(doc_dto(doc, session))
    })
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
        session.undo();
        sync(doc, session);
        Ok(doc_dto(doc, session))
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(dto)
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
/// The geometry upload is keyed by the journal generation, so an orbit re-renders
/// without touching a vertex buffer (the P23.2a warm path). `encode_png_fast` is
/// the encoder, per the same measurement: at edit rate the default deflate is 98%
/// of the frame.
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
    let (geo, mesh_copy, stamp, view, selection, mode) = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        sync(doc, session);
        // Cached on the generation stamp: an orbit re-renders without
        // re-tessellating and without cloning the mesh (P23.4 audit — M6).
        let (geo, mesh) = doc.preview.get(session);
        Ok((
            geo,
            mesh,
            session.generation(),
            doc.view,
            doc.selection.clone(),
            doc.mode,
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
/// `rewrite_payload` **and** `ensure_vmesh`, under one project lock — see the
/// module docs for why the second one is not optional.
#[tauri::command]
pub async fn dcc_save(
    app: AppHandle,
    id: String,
    state: State<'_, DccState>,
    assets: State<'_, super::assets::AssetState>,
    viewport: State<'_, super::ViewportState>,
) -> Result<DccSaveDto, String> {
    let (asset, mesh, generation) = state.with(|s| {
        let (doc, session) = s.pair(&id)?;
        Ok((doc.asset, session.mesh().clone(), session.generation()))
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

    Ok(DccSaveDto {
        ok: true,
        advisories: advisories(&report),
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
            "{} vertices carry a non-finite value from the source file.",
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
        sync(doc, session);
        let here = dcc::tessellate(session.mesh()).bounds;
        let there = dcc::tessellate(&incoming.mesh).bounds;
        let offset = glam::DVec3::new((here.max[0] - there.min[0]) as f64 + 0.25, 0.0, 0.0);
        match dcc::merge_into(session, &incoming.mesh, offset) {
            Ok(faces) => {
                doc.selection.sync(session.generation());
                Ok(DccApplyDto {
                    ok: faces > 0,
                    refusal: (faces == 0).then(|| "the dropped mesh had no faces".to_string()),
                    doc: doc_dto(doc, session),
                })
            }
            Err(e) => Ok(DccApplyDto {
                ok: false,
                refusal: Some(refusal_text(&e)),
                doc: doc_dto(doc, session),
            }),
        }
    })?;
    let _ = app.emit("dcc://sync", id);
    Ok(out)
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
                s.docs.insert(
                    id.to_string(),
                    DccDoc {
                        id: id.to_string(),
                        asset: AssetId::new(),
                        name: "T".into(),
                        mode: SelectMode::Face,
                        selection: SelectionSet::new(session.generation()),
                        view: PreviewView::default(),
                        import: ImportReport::default(),
                        saved_generation: Some(session.generation()),
                        knife: Vec::new(),
                        last_edge: None,
                        preview: PreviewCache::new(),
                    },
                );
                s.journals.insert(id.to_string(), session);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn close_frees_the_doc_and_journal() {
        let state = DccState::default();
        seed(&state, "dcc:1", inf_dcc::cube(1.0));
        state.close("dcc:1").unwrap();
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
    fn a_soft_translate_scales_by_weight_and_stays_one_op_per_weight() {
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
                assert!(ops.len() > 1, "the neighbourhood moves too: {ops:?}");
                let full = ops
                    .iter()
                    .filter(|o| matches!(o, Op::TranslateVerts { delta, .. } if delta[1] == 1.0))
                    .count();
                assert_eq!(full, 1, "exactly one group moves at full weight");
                for op in &ops {
                    let Op::TranslateVerts { delta, .. } = op else {
                        panic!("{op:?}")
                    };
                    assert!(delta[1] > 0.0 && delta[1] <= 1.0, "{delta:?}");
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
}
