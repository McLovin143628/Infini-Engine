//! **The Model Editor's Ring-1 half** (P23.4): everything the panel needs that
//! is neither kernel topology nor Tauri.
//!
//! Four things live here, and they are here rather than in Ring 2 for the reason
//! [`crate::render_assets`] states: a Ring-2 command module is not reachable from
//! a test on any CI leg, and every one of these is a pure function with an exact
//! answer.
//!
//! 1. [`tessellate`] — the kernel mesh as drawable triangles.
//! 2. [`Projector`] — the ONE camera projection, shared by the picker and the
//!    overlay (§ below).
//! 3. [`pick`] — pointer → the component under it.
//! 4. [`draw_overlay`] — the wireframe and selection, composited into the RGBA
//!    the preview render handed back.
//!
//! # Why the preview draws what the SAVE would write
//!
//! [`tessellate`] goes through [`inf_dcc::to_mesh_asset`] — the same writer
//! [`crate::assets::AssetProject::rewrite_payload`] will be handed — rather than
//! a private triangulator. So the picture in the panel is the geometry that gets
//! saved, including its ear clipping, its corner splits and its derived normals.
//! A second triangulator would be a second answer to "what is this mesh", and the
//! two would disagree exactly where an n-gon is interesting.
//!
//! It is not free (ear clipping plus a tangent solve), which is why the caller
//! caches it against the session's generation stamp and re-runs it only when the
//! mesh moves — never on a camera orbit.
//!
//! # The overlay is CPU-composited, and that is the decision
//!
//! The alternative was a second GPU pipeline in [`crate::thumbnail::PreviewSession`]
//! drawing `LineList` against the same depth buffer. Rejected for three reasons,
//! in order of weight:
//!
//! * **One projection, or the panel lies.** Picking has to be CPU (there is no
//!   sub-object id buffer, and the P23.1 memo rules the viewport's ID pass out of
//!   this path deliberately). If the *highlight* were computed by a vertex shader
//!   and the *hit* by this module, they would be two answers to the same
//!   question, differing at exactly the sub-pixel margins where a user complains.
//!   Composited here, what lights up is what [`pick`] would have returned,
//!   because it is the same [`Projector`] and there is no second one to drift.
//! * **`PreviewSession` is shared with the material editor.** It is a proven,
//!   measured path (P23.2a: 0.34 ms warm at 512²); growing it a second pipeline,
//!   a second bind group and a depth-bias tuning problem to serve one panel is
//!   risk spent in the wrong place.
//! * **Occlusion comes free from the topology.** A GPU line pass needs the depth
//!   buffer to hide back-side wires. A half-edge mesh already knows both faces of
//!   every edge, so an edge whose two faces both point away is culled with a dot
//!   product and no depth at all — which a line pipeline cannot do at any price.
//!
//! **The honest limit**: that is back-face culling, not depth testing. An edge on
//! the near side of the model that happens to sit behind *another part* of the
//! same model still draws. On a convex-ish prop (which is what v1 models) the two
//! coincide; on a folded one the wireframe reads as mild x-ray. Fixing it means
//! reading the depth buffer back beside the colour — one more aligned copy per
//! frame — and it is a remainder, not a defect that needs hiding.
//!
//! # Units
//!
//! Positions are metres. Screen coordinates are **pixels of the square preview**,
//! origin top-left, which is the space the panel's pointer events arrive in.

use glam::{DVec3, Mat4, Vec3, Vec4Swizzles};
use inf_dcc::{FaceId, HalfId, Mesh, Op, SelectMode, SelectionSet, VertId};
use inf_mesh::{MeshAsset, MeshVertex};
use serde::{Deserialize, Serialize};

use crate::thumbnail::PreviewView;

/// The neutral lit surface the Model Editor's preview draws the edit mesh with.
///
/// [`crate::thumbnail::PreviewSession`] is a *material* preview: it compiles a
/// `material_surface(MatIn) -> Surface` and wraps it in a PBR shell. A modelling
/// preview wants no material at all, so it supplies a constant one and gets the
/// whole cached-pipeline path (P23.2a) with no new GPU code — the surface is one
/// cache key, compiled once for the session's life.
///
/// Deliberately mid-grey and slightly rough: a shape being modelled is read by
/// its silhouette and its shading gradient, and a coloured or shiny default
/// fights the wireframe drawn on top of it.
pub const DCC_SURFACE_WGSL: &str = "\
struct MatIn { uv: vec2<f32>, normal: vec3<f32>, world_pos: vec3<f32>, time: f32 };
struct Surface { base_color: vec3<f32>, metallic: f32, roughness: f32, emissive: vec3<f32> };
fn material_surface(mi: MatIn) -> Surface {
    var surf: Surface;
    surf.base_color = vec3<f32>(0.62, 0.63, 0.66);
    surf.metallic = 0.0;
    surf.roughness = 0.55;
    surf.emissive = vec3<f32>(0.0);
    return surf;
}
";

/// The edit mesh as triangles, plus the asset it came from.
pub struct EditGeometry {
    pub verts: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    /// The bounding box the framing camera is fitted to.
    pub bounds: inf_mesh::Aabb,
    /// **The kernel CORNER each written vertex was interned from** (Wave D),
    /// parallel to `verts` — or empty when the export could not supply one.
    ///
    /// This is what makes [`displace`] possible: a drag moves positions and
    /// leaves the topology alone, so the whole re-export was re-exporting the
    /// same connectivity. A corner and not merely its vertex, because the normal
    /// the writer emits is a property of the CORNER — authored, or the corner's
    /// smooth fan, which stops at sharp edges.
    pub sources: Vec<inf_dcc::HalfId>,
}

/// Tessellate the kernel mesh for drawing — through the real writer, so the
/// preview and the save agree by construction (see the module docs).
pub fn tessellate(mesh: &Mesh) -> EditGeometry {
    let (asset, _, sources) =
        inf_dcc::to_mesh_asset_sourced(mesh, &inf_dcc::ExportOptions::default());
    let (verts, indices) = flatten(&asset);
    EditGeometry {
        bounds: asset.bounds,
        verts,
        indices,
        // Flattened the same way the vertices are, so index `i` of `verts` and
        // index `i` of `sources` describe the same written vertex.
        sources: sources
            .map(|s| s.into_iter().flatten().collect())
            .unwrap_or_default(),
    }
}

/// **Re-derive a tessellation for a mesh whose vertices merely MOVED**, without
/// re-running the exporter.
///
/// The P23 ledger's named next lever, in the words it named it in: *"displacing
/// the cached vertex buffer in place rather than re-running the exporter, which
/// needs the writer to expose its corner→vertex map"*. The map is
/// [`inf_dcc::to_mesh_asset_sourced`]'s third return value; this is what it is
/// for.
///
/// `None` — meaning "fall back to a full `tessellate`" — when the geometry has
/// no source map (an optimized export), when a source id is no longer live, or
/// when a source position is not finite. **`None` is a value, not a failure**:
/// every caller can always tessellate.
///
/// # What is re-derived, and what is not
///
/// * **Positions** come straight from the mesh — exact, not approximated.
/// * **Normals** are re-derived by calling [`inf_dcc::corner_normal`] on the
///   corner each written vertex was interned from — the **writer's own rule**,
///   on the writer's own corners. They have to be re-derived at all because a
///   displaced surface with the old normals shades like the shape it used to
///   be, which is worse than a slow frame; and they have to be re-derived *that
///   way* because a vertex-average over the index buffer is a different number
///   on any hard-shaded mesh, where a derived normal stops at a sharp edge.
///   (This paragraph described the vertex-average for a while after the code
///   stopped doing it — which is the comment most likely to talk someone into
///   putting it back.)
/// * **UVs** are unchanged, because a position move does not move a UV.
/// * **Tangents are KEPT**, and that is the one approximation. A tangent is only
///   read by normal mapping and the Model Editor's preview surface
///   (`DCC_SURFACE_WGSL`) has no normal map, so a drag frame cannot show the
///   difference — and the committed frame, which is a full `tessellate`, is
///   exact. Stated rather than left for someone to find.
/// * **A face's own authored normals** are likewise kept: they are authored data
///   that a drag does not change, and the export policy writes them verbatim.
///
/// The result is only a *preview*. Nothing here reaches a file: a save goes
/// through `save_mesh_session`, which exports from the mesh.
pub fn displace(base: &EditGeometry, mesh: &Mesh) -> Option<EditGeometry> {
    if base.sources.len() != base.verts.len() || base.sources.is_empty() {
        return None;
    }
    let mut verts = base.verts.clone();
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for (v, &h) in verts.iter_mut().zip(&base.sources) {
        if !mesh.has_half(h) || mesh.is_boundary(h) != Some(false) {
            return None;
        }
        let p = mesh.origin(h).and_then(|x| mesh.position(x))?;
        if !p.is_finite() {
            return None;
        }
        v.position = [p.x as f32, p.y as f32, p.z as f32];
        for k in 0..3 {
            lo[k] = lo[k].min(v.position[k]);
            hi[k] = hi[k].max(v.position[k]);
        }
    }
    // **The writer's own normal rule, called on the writer's own corners.**
    // Not a vertex-average: a derived normal is the corner's SMOOTH FAN, which
    // stops at sharp edges, and an authored one is copied verbatim. Both live in
    // `inf_dcc::corner_normal`, and calling it is the difference between a
    // preview that matches the save and one that merely looks plausible.
    for (v, &h) in verts.iter_mut().zip(&base.sources) {
        let n = inf_dcc::corner_normal(mesh, h, inf_dcc::NormalPolicy::PreserveAuthored);
        let len = n.length();
        // A corner whose fan cancels out — a fold pinched flat by the drag —
        // keeps the normal it had. Zeroing it would make the preview go black
        // there, and the drag is the moment an author is looking hardest.
        if len.is_finite() && len > 1e-20 {
            let n = n / len;
            v.normal = [n.x as f32, n.y as f32, n.z as f32];
        }
    }
    Some(EditGeometry {
        verts,
        indices: base.indices.clone(),
        bounds: inf_mesh::Aabb { min: lo, max: hi },
        sources: base.sources.clone(),
    })
}

/// Concatenate an asset's submeshes into one vertex/index pair, rebasing each
/// submesh's indices. (The thumbnailer's `combined_geometry` does the same for
/// its own path; this one is not `pub` there.)
fn flatten(asset: &MeshAsset) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for sm in &asset.submeshes {
        let base = verts.len() as u32;
        verts.extend_from_slice(&sm.vertices);
        indices.extend(sm.indices.iter().map(|&i| i + base));
    }
    (verts, indices)
}

/// A camera that frames `bounds` from the default three-quarter direction.
///
/// The preview target is square, so the fit uses the bounding sphere rather than
/// the box: a long thin prop framed by its widest axis would leave the other two
/// as a sliver, and framed by its longest would push it off screen when rotated.
/// The sphere is rotation-invariant, which is what an orbiting preview needs.
pub fn frame(bounds: inf_mesh::Aabb) -> PreviewView {
    let mut view = PreviewView::default();
    if !bounds
        .min
        .iter()
        .chain(bounds.max.iter())
        .all(|c| c.is_finite())
    {
        return view;
    }
    let centre = Vec3::new(
        (bounds.min[0] + bounds.max[0]) * 0.5,
        (bounds.min[1] + bounds.max[1]) * 0.5,
        (bounds.min[2] + bounds.max[2]) * 0.5,
    );
    let radius = Vec3::new(
        bounds.max[0] - bounds.min[0],
        bounds.max[1] - bounds.min[1],
        bounds.max[2] - bounds.min[2],
    )
    .length()
        * 0.5;
    let radius = if radius > 1e-6 { radius } else { 1.0 };
    view.target = centre;
    // `sin(fov/2)` would be the exact fit; the margin factor covers the wireframe
    // and the fact that an orbit swings the far corner toward the camera.
    view.distance = radius / (view.fov_deg * 0.5).to_radians().sin() * 1.15;
    view.near = (radius * 0.01).max(1e-3);
    view.far = view.distance + radius * 4.0;
    view
}

/// One camera, resolved into the pixel space the panel's pointer events use.
///
/// Built once per frame and handed to **both** [`pick`] and [`draw_overlay`], so
/// the two cannot disagree about where anything is (see the module docs).
#[derive(Debug, Clone, Copy)]
pub struct Projector {
    view_proj: Mat4,
    /// The inverse, cached because **every** pointer interaction needs it: a
    /// sculpt dab, a gizmo grab and a gizmo update are all "pixel → world ray".
    /// Inverting per call would be three inversions per drag frame of a matrix
    /// that changes only when the camera does.
    inv_view_proj: Mat4,
    eye: Vec3,
    size: f32,
}

/// A point that survived projection: pixel position plus a depth to sort by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projected {
    pub x: f32,
    pub y: f32,
    /// Distance from the eye along the view direction, in metres. Smaller wins.
    pub depth: f32,
}

impl Projector {
    pub fn new(view: PreviewView, size: u32) -> Self {
        let (eye, view_proj) = view.view_proj();
        Self {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            eye,
            size: size.max(1) as f32,
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.eye
    }

    pub fn size(&self) -> f32 {
        self.size
    }

    /// The camera matrix itself — handed to `inf_render::gizmo`, which takes a
    /// `view_proj` and does its own analytic hit-testing with it.
    ///
    /// Exposed rather than re-derived so the gizmo's picking and this module's
    /// drawing are the same projection as [`pick`]'s: the P23.4 rule ("one
    /// projection, or the panel lies") does not stop being true because the thing
    /// being picked is a handle rather than a face.
    pub fn view_proj(&self) -> Mat4 {
        self.view_proj
    }

    /// The world-space ray through a pixel, as `(origin on the near plane,
    /// unit direction)`.
    ///
    /// **Built by inverting the projection**, not by re-deriving a camera basis
    /// from [`PreviewView`]'s fields. A hand-built basis would be a second answer
    /// to "where is the camera" and would drift from `view_proj` the first time
    /// either changed — the same reasoning that put `pick` and `draw_overlay`
    /// behind one `Projector`. The near plane is `z = 0` and the far plane
    /// `z = 1`: the preview's `perspective_rh` documents itself as the wgpu
    /// `[0, 1]` depth convention.
    ///
    /// `None` when the inverse is degenerate or the pixel maps to nothing finite.
    pub fn ray(&self, px: f32, py: f32) -> Option<(Vec3, Vec3)> {
        let ndc_x = px / self.size * 2.0 - 1.0;
        let ndc_y = 1.0 - py / self.size * 2.0;
        let unproject = |z: f32| -> Option<Vec3> {
            let p = self.inv_view_proj * glam::Vec4::new(ndc_x, ndc_y, z, 1.0);
            (p.w.is_finite() && p.w.abs() > 1e-12).then(|| p.xyz() / p.w)
        };
        let a = unproject(0.0)?;
        let b = unproject(1.0)?;
        let d = b - a;
        let len = d.length();
        (a.is_finite() && len.is_finite() && len > 1e-12).then(|| (a, d / len))
    }

    /// Project a world point. `None` when it is at or behind the eye plane —
    /// the clip case, which v1 drops rather than clipping a segment against the
    /// near plane (the preview always frames the whole mesh, so nothing is
    /// behind the camera unless the author has dollied inside their own model).
    pub fn point(&self, p: DVec3) -> Option<Projected> {
        let v = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
        let clip = self.view_proj * v.extend(1.0);
        // Spelled out rather than `!(w > eps)`: the negation is load-bearing,
        // because a NaN `w` compares false against everything and must be
        // dropped, not projected.
        if !(clip.w.is_finite() && clip.w > 1e-6) {
            return None;
        }
        let ndc = clip.xyz() / clip.w;
        if !ndc.x.is_finite() || !ndc.y.is_finite() {
            return None;
        }
        Some(Projected {
            // NDC y is up; pixel y is down.
            x: (ndc.x * 0.5 + 0.5) * self.size,
            y: (0.5 - ndc.y * 0.5) * self.size,
            depth: (v - self.eye).length(),
        })
    }
}

// ── picking ────────────────────────────────────────────────────────────────

/// What a pick landed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PickHit {
    Vert(VertId),
    /// The **canonical** half of the edge (`inf_dcc::canonical_edge`).
    Edge(HalfId),
    Face(FaceId),
}

/// The pixel radius a vertex or edge pick reaches.
///
/// Generous on purpose: a vertex is a point, and a point is unhittable. This is
/// the same number the overlay draws its vertex dots at scale from, so what looks
/// clickable is clickable.
pub const PICK_RADIUS_PX: f32 = 7.0;

/// The component under `(px, py)`, or `None`.
///
/// Vertices and edges are "nearest within [`PICK_RADIUS_PX`], ties broken by
/// depth"; faces are **containment**, ties broken by depth — a face is an area,
/// so a radius would let a click in open space grab the nearest polygon, which is
/// the behaviour that makes an author fight a modeller.
///
/// Back-facing components are eligible for vertices and edges (a wireframe worker
/// clicks through) but not for faces, which is the convention every DCC uses and
/// the reason a face pick feels solid.
pub fn pick(mesh: &Mesh, proj: &Projector, mode: SelectMode, px: f32, py: f32) -> Option<PickHit> {
    match mode {
        SelectMode::Vert => {
            let mut best: Option<(f32, f32, VertId)> = None;
            for v in mesh.vert_ids() {
                let Some(p) = mesh.position(v).and_then(|q| proj.point(q)) else {
                    continue;
                };
                let d = ((p.x - px).powi(2) + (p.y - py).powi(2)).sqrt();
                if d > PICK_RADIUS_PX {
                    continue;
                }
                if best.is_none_or(|(bd, bz, _)| (d, p.depth) < (bd, bz)) {
                    best = Some((d, p.depth, v));
                }
            }
            best.map(|(_, _, v)| PickHit::Vert(v))
        }
        SelectMode::Edge => {
            let mut best: Option<(f32, f32, HalfId)> = None;
            for h in mesh.half_ids() {
                let Some(c) = inf_dcc::canonical_edge(mesh, h) else {
                    continue;
                };
                if c != h {
                    continue; // once per undirected edge
                }
                let (Some(a), Some(b)) = (
                    mesh.origin(h).and_then(|v| mesh.position(v)),
                    mesh.dest(h).and_then(|v| mesh.position(v)),
                ) else {
                    continue;
                };
                let (Some(pa), Some(pb)) = (proj.point(a), proj.point(b)) else {
                    continue;
                };
                let d = segment_distance(px, py, pa.x, pa.y, pb.x, pb.y);
                if d > PICK_RADIUS_PX {
                    continue;
                }
                let z = (pa.depth + pb.depth) * 0.5;
                if best.is_none_or(|(bd, bz, _)| (d, z) < (bd, bz)) {
                    best = Some((d, z, c));
                }
            }
            best.map(|(_, _, h)| PickHit::Edge(h))
        }
        SelectMode::Face => {
            let mut best: Option<(f32, FaceId)> = None;
            for f in mesh.face_ids() {
                let Some(poly) = project_face(mesh, proj, f) else {
                    continue;
                };
                if !faces_eye(&poly) {
                    continue;
                }
                if !contains(&poly, px, py) {
                    continue;
                }
                let z = poly.iter().map(|p| p.depth).sum::<f32>() / poly.len() as f32;
                if best.is_none_or(|(bz, _)| z < bz) {
                    best = Some((z, f));
                }
            }
            best.map(|(_, f)| PickHit::Face(f))
        }
    }
}

fn project_face(mesh: &Mesh, proj: &Projector, f: FaceId) -> Option<Vec<Projected>> {
    let verts = mesh.face_verts(f)?;
    let mut out = Vec::with_capacity(verts.len());
    for v in verts {
        out.push(proj.point(mesh.position(v)?)?);
    }
    (out.len() >= 3).then_some(out)
}

/// **The facing rule, in one place.** A projected polygon faces the eye when its
/// signed area is negative: the pixel y axis points down, so an outward-wound
/// (CCW in world) loop comes out clockwise on screen.
///
/// Extracted because it was written out three times in this file — in the face
/// picker, in the overlay's fill and in `face_faces_eye` — while the module docs
/// above claimed there was "no second one to drift". Three copies of a sign
/// convention is three chances to get it backwards, and the two that disagree
/// would be the picker and the thing that draws the highlight.
fn faces_eye(poly: &[Projected]) -> bool {
    signed_area(poly) < 0.0
}

/// Twice the signed area of a projected polygon. Use [`faces_eye`] rather than
/// comparing this yourself.
fn signed_area(poly: &[Projected]) -> f32 {
    let n = poly.len();
    let mut acc = 0.0;
    for i in 0..n {
        let (a, b) = (&poly[i], &poly[(i + 1) % n]);
        acc += a.x * b.y - b.x * a.y;
    }
    acc
}

/// Even-odd containment — correct for non-convex n-gons, which the kernel has.
fn contains(poly: &[Projected], px: f32, py: f32) -> bool {
    let n = poly.len();
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (&poly[i], &poly[(i + n - 1) % n]);
        if (a.y > py) != (b.y > py) {
            let t = (py - a.y) / (b.y - a.y);
            if px < a.x + t * (b.x - a.x) {
                inside = !inside;
            }
        }
    }
    inside
}

fn segment_distance(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 1e-12 {
        (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (cx, cy) = (ax + dx * t, ay + dy * t);
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

// ── overlay ────────────────────────────────────────────────────────────────

/// The overlay's palette, in the panel's ink.
#[derive(Debug, Clone, Copy)]
pub struct OverlayStyle {
    pub wire: [u8; 3],
    pub selected: [u8; 3],
    /// Fill tint for a selected face, and how strongly it is blended (0–255).
    pub fill: [u8; 3],
    pub fill_alpha: u8,
    pub vert_radius: i32,
}

impl Default for OverlayStyle {
    fn default() -> Self {
        Self {
            wire: [24, 26, 30],
            selected: [255, 168, 46],
            fill: [255, 148, 32],
            fill_alpha: 90,
            vert_radius: 2,
        }
    }
}

/// Composite the wireframe and the selection into a rendered RGBA frame.
///
/// `rgba` is `size × size × 4`, tightly packed, as the preview readback hands it
/// back. Everything is drawn with the same [`Projector`] [`pick`] uses.
pub fn draw_overlay(
    rgba: &mut [u8],
    size: u32,
    mesh: &Mesh,
    proj: &Projector,
    selection: &SelectionSet,
    mode: SelectMode,
    style: &OverlayStyle,
) {
    let w = size as i32;
    if rgba.len() < (size as usize) * (size as usize) * 4 {
        return;
    }

    // Selected faces first, so wires land on top of their own tint.
    if mode == SelectMode::Face {
        for &f in selection.faces() {
            let Some(poly) = project_face(mesh, proj, f) else {
                continue;
            };
            if !faces_eye(&poly) {
                continue;
            }
            fill_polygon(rgba, w, &poly, style.fill, style.fill_alpha);
        }
    }

    // Edges. An edge whose two faces both point away from the eye is culled —
    // the half-edge structure knows both, which no line pipeline would.
    for h in mesh.half_ids() {
        let Some(c) = inf_dcc::canonical_edge(mesh, h) else {
            continue;
        };
        if c != h {
            continue;
        }
        if !edge_is_visible(mesh, proj, h) {
            continue;
        }
        let (Some(a), Some(b)) = (
            mesh.origin(h).and_then(|v| mesh.position(v)),
            mesh.dest(h).and_then(|v| mesh.position(v)),
        ) else {
            continue;
        };
        let (Some(pa), Some(pb)) = (proj.point(a), proj.point(b)) else {
            continue;
        };
        let hot = match mode {
            SelectMode::Edge => selection.contains_edge(mesh, h),
            SelectMode::Face => {
                // The border of the selection reads as its outline.
                [mesh.face_of(h), mesh.twin(h).and_then(|t| mesh.face_of(t))]
                    .into_iter()
                    .flatten()
                    .flatten()
                    .any(|f| selection.contains_face(f))
            }
            SelectMode::Vert => false,
        };
        let colour = if hot { style.selected } else { style.wire };
        line(rgba, w, pa.x, pa.y, pb.x, pb.y, colour);
    }

    if mode == SelectMode::Vert {
        for v in mesh.vert_ids() {
            let Some(p) = mesh.position(v).and_then(|q| proj.point(q)) else {
                continue;
            };
            let colour = if selection.contains_vert(v) {
                style.selected
            } else {
                style.wire
            };
            let r = if selection.contains_vert(v) {
                style.vert_radius + 1
            } else {
                style.vert_radius
            };
            dot(rgba, w, p.x, p.y, r, colour);
        }
    }
}

/// Is either face of this edge turned toward the eye? A boundary edge always is
/// (there is nothing behind it to hide it).
fn edge_is_visible(mesh: &Mesh, proj: &Projector, h: HalfId) -> bool {
    let Some(t) = mesh.twin(h) else { return false };
    let mut any_face = false;
    for x in [h, t] {
        match mesh.face_of(x) {
            Some(Some(f)) => {
                any_face = true;
                if face_faces_eye(mesh, proj, f) {
                    return true;
                }
            }
            Some(None) => return true, // a boundary edge is a silhouette
            None => {}
        }
    }
    !any_face
}

fn face_faces_eye(mesh: &Mesh, proj: &Projector, f: FaceId) -> bool {
    project_face(mesh, proj, f).is_some_and(|poly| faces_eye(&poly))
}

/// Even-odd scanline fill with a constant-alpha blend.
fn fill_polygon(rgba: &mut [u8], w: i32, poly: &[Projected], colour: [u8; 3], alpha: u8) {
    let (mut top, mut bottom) = (f32::MAX, f32::MIN);
    for p in poly {
        top = top.min(p.y);
        bottom = bottom.max(p.y);
    }
    let y0 = pixel(top.floor(), w).max(0);
    let y1 = pixel(bottom.ceil(), w).min(w - 1);
    let n = poly.len();
    let mut xs: Vec<f32> = Vec::with_capacity(n);
    for y in y0..=y1 {
        let sy = y as f32 + 0.5;
        xs.clear();
        for i in 0..n {
            let (a, b) = (&poly[i], &poly[(i + 1) % n]);
            if (a.y > sy) != (b.y > sy) {
                let t = (sy - a.y) / (b.y - a.y);
                xs.push(a.x + t * (b.x - a.x));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for pair in xs.chunks_exact(2) {
            let sx0 = pixel(pair[0].ceil(), w).max(0);
            let sx1 = pixel(pair[1].floor(), w).min(w - 1);
            for x in sx0..=sx1 {
                blend(rgba, w, x, y, colour, alpha);
            }
        }
    }
}

/// Bresenham, clipped by the per-pixel bounds check.
fn line(rgba: &mut [u8], w: i32, x0: f32, y0: f32, x1: f32, y1: f32, colour: [u8; 3]) {
    // **Saturating throughout.** A camera inside the model projects a vertex a
    // few pixels from the eye plane to a coordinate in the billions, and
    // `dx + dy` on two of those overflows — which is a panic in a debug build
    // and a wrapped, wrong picture in a release one. Clamped to a window a few
    // screens wide, the arithmetic cannot leave `i32` and every pixel is still
    // rejected by `blend`'s bounds check.
    let clamp = |v: f32| pixel(v, w);
    let (mut x, mut y) = (clamp(x0), clamp(y0));
    let (tx, ty) = (clamp(x1), clamp(y1));
    let dx = (tx - x).abs();
    let dy = -(ty - y).abs();
    let sx = if x < tx { 1 } else { -1 };
    let sy = if y < ty { 1 } else { -1 };
    let mut err = dx.saturating_add(dy);
    // A guard rather than a `loop`: a projection that produced an absurd
    // coordinate must not spin, and an off-screen segment costs its own length
    // and nothing more.
    let budget = dx.saturating_sub(dy).saturating_add(2);
    for _ in 0..budget.min(8 * w) {
        blend(rgba, w, x, y, colour, 255);
        if x == tx && y == ty {
            break;
        }
        let e2 = err.saturating_mul(2);
        if e2 >= dy {
            err = err.saturating_add(dy);
            x += sx;
        }
        if e2 <= dx {
            err = err.saturating_add(dx);
            y += sy;
        }
    }
}

/// A projected coordinate as a pixel index, clamped to a few screens either side.
///
/// `f32 as i32` saturates rather than wrapping (Rust guarantees it), but the
/// saturated value is `i32::MAX`, and *differences* of those overflow. Clamping
/// to a window keeps every intermediate small while leaving off-screen segments
/// pointing the way they pointed.
fn pixel(v: f32, w: i32) -> i32 {
    let limit = w.saturating_mul(4);
    if v.is_finite() {
        (v.round() as i32).clamp(-limit, limit)
    } else {
        0
    }
}

fn dot(rgba: &mut [u8], w: i32, cx: f32, cy: f32, r: i32, colour: [u8; 3]) {
    let (ix, iy) = (pixel(cx, w), pixel(cy, w));
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                blend(
                    rgba,
                    w,
                    ix.saturating_add(dx),
                    iy.saturating_add(dy),
                    colour,
                    255,
                );
            }
        }
    }
}

#[inline]
fn blend(rgba: &mut [u8], w: i32, x: i32, y: i32, colour: [u8; 3], alpha: u8) {
    if x < 0 || y < 0 || x >= w || y >= w {
        return;
    }
    let i = ((y as usize) * (w as usize) + x as usize) * 4;
    if alpha == 255 {
        rgba[i] = colour[0];
        rgba[i + 1] = colour[1];
        rgba[i + 2] = colour[2];
    } else {
        let a = alpha as u32;
        for k in 0..3 {
            let src = colour[k] as u32;
            let dst = rgba[i + k] as u32;
            rgba[i + k] = ((src * a + dst * (255 - a)) / 255) as u8;
        }
    }
    rgba[i + 3] = 255;
}

// ── the 2D UV view (P23.5) ─────────────────────────────────────────────────

/// The UV view's palette.
#[derive(Debug, Clone, Copy)]
pub struct UvStyle {
    pub background: [u8; 3],
    /// The `[0,1]²` boundary — the texture's own edge.
    pub border: [u8; 3],
    pub wire: [u8; 3],
    /// An edge whose 3D twin pair carries a seam flag. Drawn last and thickest,
    /// because a seam is the one thing in this view an author is looking *for*.
    pub seam: [u8; 3],
    pub selected: [u8; 3],
    pub fill: [u8; 3],
    pub fill_alpha: u8,
}

impl Default for UvStyle {
    fn default() -> Self {
        Self {
            background: [18, 19, 22],
            border: [70, 74, 82],
            wire: [126, 132, 142],
            seam: [255, 96, 72],
            selected: [255, 168, 46],
            fill: [255, 148, 32],
            fill_alpha: 70,
        }
    }
}

/// Draw the UV layout into an RGBA buffer.
///
/// # Why this is rasterized in Rust and not drawn on a `<canvas>`
///
/// The `SpriteSheetPanel` draws its frames in the frontend, and that is right for
/// it: a sprite sheet's rectangles *are* the document. A UV layout is not — it is
/// a **projection of the mesh**, and every question it answers ("is this edge a
/// seam", "is this corner selected", "which chart is this") is a question the
/// backend already owns and the frontend has no copy of. Shipping the answer
/// would mean shipping a polygon soup plus a seam set plus a selection per frame
/// and keeping a second renderer in step with `draw_overlay` — the same
/// two-answers-to-one-question shape the P23.4 ruling rejected for the 3D
/// overlay. So it goes down the identical path: composite here, `encode_png_fast`,
/// one `<img>`.
///
/// # Edges are drawn per CORNER PAIR, and that is what makes charts visible
///
/// A vertex on a seam has a different UV in each chart it belongs to (P23.3 §7a:
/// attributes live where seams live). Drawing "vertex UV to vertex UV" would
/// therefore need a vertex UV, which does not exist — and would stitch the charts
/// back together on screen. Walking each face loop and joining `uv(h)` to
/// `uv(next(h))` draws exactly what the writer will emit.
///
/// `selection` is the **same set the 3D view shows**, which is the whole of the
/// synchronization: there is one selection, in one document.
pub fn draw_uv_layout(
    rgba: &mut [u8],
    size: u32,
    mesh: &Mesh,
    selection: &SelectionSet,
    mode: SelectMode,
    style: &UvStyle,
) {
    let n = (size as usize) * (size as usize) * 4;
    if rgba.len() < n {
        return;
    }
    for px in rgba[..n].chunks_exact_mut(4) {
        px[0] = style.background[0];
        px[1] = style.background[1];
        px[2] = style.background[2];
        px[3] = 255;
    }
    let w = size as i32;
    let s = size as f32;
    // UV (0,0) is bottom-left; pixel y grows downward.
    let to_px = |uv: [f64; 2]| Projected {
        x: uv[0] as f32 * s,
        y: (1.0 - uv[1] as f32) * s,
        depth: 0.0,
    };

    // Selected faces first, so the wires land on their own tint.
    if mode == SelectMode::Face {
        for &f in selection.faces() {
            let Some(poly) = face_uv_polygon(mesh, f, &to_px) else {
                continue;
            };
            fill_polygon(rgba, w, &poly, style.fill, style.fill_alpha);
        }
    }

    // **Three passes, and the order is the priority order.** Every outline is
    // drawn plain first; then the frame; then the seams; then the selection. A
    // single pass that picked one colour per edge made a *selected seam* read as
    // a seam — which is the one edge the author is certainly looking at — and
    // made a chart packed against `u = 0` swallow the border. Collecting and
    // re-drawing costs one `Vec` of segments and removes both.
    let mut seams: Vec<(Projected, Projected)> = Vec::new();
    let mut hot: Vec<(Projected, Projected)> = Vec::new();
    for f in mesh.face_ids() {
        let Some(halfs) = mesh.face_loop(f) else {
            continue;
        };
        let count = halfs.len();
        for i in 0..count {
            let (h, next) = (halfs[i], halfs[(i + 1) % count]);
            let (Some(a), Some(b)) = (mesh.corner_uv(h), mesh.corner_uv(next)) else {
                continue;
            };
            let (pa, pb) = (to_px(a), to_px(b));
            line(rgba, w, pa.x, pa.y, pb.x, pb.y, style.wire);
            if mesh.is_seam(h) == Some(true) {
                seams.push((pa, pb));
            }
            let selected = match mode {
                SelectMode::Edge => selection.contains_edge(mesh, h),
                SelectMode::Face => selection.contains_face(f),
                SelectMode::Vert => false,
            };
            if selected {
                hot.push((pa, pb));
            }
        }
    }
    // **The unit square, drawn after the wires.** A chart packed hard against
    // `u = 0` or `v = 1` puts its outline exactly on the border, and drawing the
    // border first meant the wireframe swallowed it entirely — measured: 127 wire
    // pixels and not one border pixel left. The frame is the reference the whole
    // view is read against, so it goes on top of the thing it frames.
    for (a, b) in [
        ((0.0, 0.0), (s, 0.0)),
        ((s, 0.0), (s, s)),
        ((s, s), (0.0, s)),
        ((0.0, s), (0.0, 0.0)),
    ] {
        line(rgba, w, a.0, a.1, b.0, b.1, style.border);
    }

    for (pa, pb) in seams {
        // Two passes one pixel apart: the seam has to read at a glance in a view
        // that is otherwise all thin grey lines, and a second colour alone does
        // not survive being one pixel wide next to a bright selection.
        line(rgba, w, pa.x, pa.y, pb.x, pb.y, style.seam);
        line(rgba, w, pa.x, pa.y + 1.0, pb.x, pb.y + 1.0, style.seam);
    }
    for (pa, pb) in hot {
        line(rgba, w, pa.x, pa.y, pb.x, pb.y, style.selected);
    }

    if mode == SelectMode::Vert {
        for f in mesh.face_ids() {
            let Some(halfs) = mesh.face_loop(f) else {
                continue;
            };
            for h in halfs {
                let (Some(v), Some(uv)) = (mesh.origin(h), mesh.corner_uv(h)) else {
                    continue;
                };
                if !selection.contains_vert(v) {
                    continue;
                }
                let p = to_px(uv);
                dot(rgba, w, p.x, p.y, 3, style.selected);
            }
        }
    }
}

fn face_uv_polygon(
    mesh: &Mesh,
    f: FaceId,
    to_px: &impl Fn([f64; 2]) -> Projected,
) -> Option<Vec<Projected>> {
    let halfs = mesh.face_loop(f)?;
    let mut out = Vec::with_capacity(halfs.len());
    for h in halfs {
        out.push(to_px(mesh.corner_uv(h)?));
    }
    (out.len() >= 3).then_some(out)
}

// ── the surface point under the pointer (P23.5) ────────────────────────────

/// The point on the model under `(px, py)`, and the face it is on.
///
/// **The face is found by the picker, not by a separate ray-vs-triangle sweep.**
/// [`pick`] in [`SelectMode::Face`] already answers "which front-facing polygon
/// contains this pixel, nearest first"; asking a second time with a different
/// method would give a different answer along every silhouette, and a brush that
/// lands on a face other than the one that highlights is a brush an author cannot
/// aim. So this asks the picker, then intersects the ray with **that** face's
/// plane.
///
/// `None` when the pointer is over empty space, or the face is degenerate (no
/// normal), or the ray is parallel to it.
pub fn pick_surface(mesh: &Mesh, proj: &Projector, px: f32, py: f32) -> Option<(FaceId, DVec3)> {
    let PickHit::Face(f) = pick(mesh, proj, SelectMode::Face, px, py)? else {
        return None;
    };
    let n = inf_dcc::face_normal(mesh, f)?;
    let verts = mesh.face_verts(f)?;
    let mut centroid = DVec3::ZERO;
    for &v in &verts {
        centroid += mesh.position(v)?;
    }
    let centroid = centroid / verts.len() as f64;

    let (ro, rd) = proj.ray(px, py)?;
    let ro = DVec3::new(ro.x as f64, ro.y as f64, ro.z as f64);
    let rd = DVec3::new(rd.x as f64, rd.y as f64, rd.z as f64);
    let denom = n.dot(rd);
    if !(denom.is_finite() && denom.abs() > 1e-12) {
        return None;
    }
    let t = n.dot(centroid - ro) / denom;
    let hit = ro + rd * t;
    hit.is_finite().then_some((f, hit))
}

/// Draw the brush footprint: a ring of `radius` metres in the surface's tangent
/// plane at `centre`.
///
/// **In the tangent plane rather than facing the camera**, because that is what
/// the brush actually does — the influence is measured *on the surface* — and
/// because a screen-facing disc would need the camera's basis, which is a second
/// answer to "where is the camera" this module deliberately does not keep (see
/// [`Projector::ray`]).
///
/// **Honest limit**: the influence is *geodesic*, so on a folded or bumpy surface
/// the reach is shorter than the ring in some directions and the ring is an
/// upper bound, not an outline. That is the same relationship the terrain brush's
/// disc has to its own falloff, and drawing the true geodesic frontier would mean
/// running the Dijkstra once per preview frame for a decoration.
pub fn draw_brush_ring(
    rgba: &mut [u8],
    size: u32,
    proj: &Projector,
    centre: DVec3,
    normal: DVec3,
    radius: f64,
    colour: [u8; 3],
) {
    if rgba.len() < (size as usize) * (size as usize) * 4 {
        return;
    }
    if !(centre.is_finite() && normal.is_finite() && radius.is_finite() && radius > 0.0) {
        return;
    }
    let n = normal.normalize_or_zero();
    if n == DVec3::ZERO {
        return;
    }
    // Any tangent will do; picked off the least-aligned axis so the cross product
    // never degenerates.
    let seed = if n.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let u = n.cross(seed).normalize_or_zero();
    let v = n.cross(u);
    let w = size as i32;
    let mut prev: Option<Projected> = None;
    for i in 0..=RING_SEGMENTS {
        let a = i as f64 / RING_SEGMENTS as f64 * std::f64::consts::TAU;
        let p = centre + (u * a.cos() + v * a.sin()) * radius;
        let here = proj.point(p);
        if let (Some(a), Some(b)) = (prev, here) {
            line(rgba, w, a.x, a.y, b.x, b.y, colour);
        }
        prev = here;
    }
}

/// Segment count for every circle this module draws, and the number
/// `inf_render::gizmo::pick_axis` uses internally for its rotate rings. A ring
/// drawn with a different count than the one picked against is a ring that is
/// hittable where it is not painted.
const RING_SEGMENTS: usize = 48;

// ── the component gizmo (P23.5, the P23.4 deferral) ────────────────────────

pub use inf_render::gizmo::{GizmoAxis, GizmoDelta, GizmoDrag, GizmoMode};

/// A transform to apply to a selection, pivot-relative.
///
/// The **one** shape both the numeric tools and the dragged gizmo produce, so
/// "the gizmo and the number box do the same thing" is a property of the code and
/// not a claim about two code paths. [`transform_ops`] is where it becomes ops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VertTransform {
    Translate(DVec3),
    Rotate { axis: DVec3, radians: f64 },
    Scale(DVec3),
}

impl VertTransform {
    /// The pure-Rust half of "what a drag update means", lifted out of
    /// `inf_render`'s render-space types.
    ///
    /// `GizmoDelta` is `f32` because the viewport's gizmo works in render-local
    /// space after the floating-origin rebase. A DCC document is a single asset
    /// at its own origin, so the widening is exact and there is nothing to rebase
    /// — but the *conversion* has to happen somewhere, and here is better than in
    /// a command, where a test cannot reach it.
    pub fn from_gizmo(delta: GizmoDelta) -> Self {
        match delta {
            GizmoDelta::Translate(d) => VertTransform::Translate(d),
            GizmoDelta::Rotate { axis, radians } => VertTransform::Rotate {
                axis: DVec3::new(axis.x as f64, axis.y as f64, axis.z as f64),
                radians: radians as f64,
            },
            GizmoDelta::Scale(f) => {
                VertTransform::Scale(DVec3::new(f.x as f64, f.y as f64, f.z as f64))
            }
        }
    }

    /// Whether this transform would move anything at all — the test a commit uses
    /// to avoid journalling a click that did not drag.
    pub fn is_identity(&self) -> bool {
        match self {
            VertTransform::Translate(d) => *d == DVec3::ZERO,
            VertTransform::Rotate { radians, .. } => *radians == 0.0,
            VertTransform::Scale(f) => *f == DVec3::ONE,
        }
    }
}

/// **What the drag says it is doing**, in the units the author typed.
///
/// The P23 ledger carried "no rotate-gizmo angle readout"; all three gizmo modes
/// had the same nothing, so all three get this. In Ring 1 rather than in the
/// command because it is a *rule about presentation* — how many digits, which
/// units, what "no movement yet" reads as — and a rule written in a
/// `#[tauri::command]` is a rule no gate can see (memo §7c).
///
/// Units at the boundary (architecture rule 6 and the units doctrine): metres
/// for a translate, **degrees** for a rotate (radians live in the op), a bare
/// multiplier for a scale.
pub fn drag_readout(x: &VertTransform) -> String {
    match x {
        VertTransform::Translate(d) => {
            // The dominant axis by name, plus the magnitude — which is what an
            // author reads off a constrained drag. An unconstrained one shows all
            // three, because naming one of them would be a lie about the other
            // two.
            let (ax, mag) = dominant_axis(*d);
            if mag > 0.0 && d.length() - mag < 1e-9 {
                format!("{mag:+.4} m on {ax}")
            } else {
                format!("{:+.4}, {:+.4}, {:+.4} m", d.x, d.y, d.z)
            }
        }
        VertTransform::Rotate { axis, radians } => {
            let (ax, _) = dominant_axis(*axis);
            format!("{:+.2}° about {ax}", radians.to_degrees())
        }
        VertTransform::Scale(f) => {
            if (f.x - f.y).abs() < 1e-12 && (f.y - f.z).abs() < 1e-12 {
                format!("x{:.4}", f.x)
            } else {
                format!("x{:.4}, {:.4}, {:.4}", f.x, f.y, f.z)
            }
        }
    }
}

/// The axis a vector points most along, and its signed component there.
fn dominant_axis(v: DVec3) -> (&'static str, f64) {
    let a = v.abs();
    if a.x >= a.y && a.x >= a.z {
        ("X", v.x)
    } else if a.y >= a.z {
        ("Y", v.y)
    } else {
        ("Z", v.z)
    }
}

/// Turn a transform into journal ops, honouring soft-select weights.
///
/// `soft` is `Some((radius, falloff))` for a proportional transform and `None`
/// for a hard one.
///
/// # A hard transform is ONE PARAMETRIC op; a soft one is ONE RESULT op
///
/// The two halves are shaped differently on purpose, and Wave D is where the
/// difference became visible.
///
/// * **Hard** — one `TranslateVerts` / `RotateVerts` / `ScaleVerts` over the
///   resolved vertices in ascending id order. The op still *says what it is*
///   ("rotate 30° about this axis"), which is what makes it re-parameterizable by
///   [`inf_dcc::MeshSession::amend`] twelve steps later.
/// * **Soft** — one [`Op::MoveVerts`] carrying the finished per-vertex deltas.
///
/// The soft shape is the **weight-table op the P23 ledger named and did not
/// build** (`inf_dcc::ops::Op::MoveVerts`). What it replaces: "one op per
/// distinct weight", which was the right *shape* — ordinary journal entries, so
/// an undo gives the author their shape back — and the wrong *granularity*. A
/// 289-vertex plane with a 3 m radius produced **105 ops from one drag** before
/// `SOFT_WEIGHT_STEPS` capped it at 64, and 64 ops at `CHECKPOINT_INTERVAL = 32`
/// still takes two full mesh snapshots and evicts most of the eight-slot
/// checkpoint history, per drag. Now it is one op and one undo press, which is
/// what an author means by "I dragged that once".
///
/// **The quantization went with it.** `quantize_weight` existed to bound the op
/// count; the op count is now 1, so rounding the falloff to 64 steps buys nothing
/// and costs the author the exact curve they chose. It stays public and tested
/// because it is still the right rounding for anything that must bound a *count*;
/// it is simply not this path's problem any more.
///
/// The weight blends toward the identity, per kind, and the blend is unchanged:
/// * translate — the delta is scaled;
/// * rotate — the **angle** is scaled, so every vertex still travels on a circle
///   about the pivot rather than being dragged off one by a scaled chord;
/// * scale — the factor is lerped from `1`, so weight `0` is "unchanged" rather
///   than "collapsed onto the pivot".
///
/// The soft path computes its positions through [`inf_dcc::Rotation`] and
/// [`inf_dcc::scale_point`] — **the kernel's own maps**, not a second Rodrigues
/// written one ring up where the crate's portable-trig gate cannot read it.
pub fn transform_ops(
    mesh: &Mesh,
    selection: &SelectionSet,
    mode: SelectMode,
    pivot: DVec3,
    xform: VertTransform,
    soft: Option<(f64, inf_terrain::Falloff)>,
) -> Vec<Op> {
    let Some((radius, falloff)) = soft else {
        let verts: Vec<VertId> = selection.resolved_verts(mesh, mode).into_iter().collect();
        if verts.is_empty() {
            return Vec::new();
        }
        return vec![match xform {
            VertTransform::Translate(d) => Op::TranslateVerts {
                verts,
                delta: d.to_array(),
            },
            VertTransform::Rotate { axis, radians } => Op::RotateVerts {
                verts,
                pivot: pivot.to_array(),
                axis: axis.to_array(),
                radians,
            },
            VertTransform::Scale(f) => Op::ScaleVerts {
                verts,
                pivot: pivot.to_array(),
                factor: f.to_array(),
            },
        }];
    };

    // A rotation is prepared per weight, because the ANGLE is what the weight
    // scales. `Rotation::new` refuses a zero axis and an out-of-range angle
    // exactly as the op would; a refusal here means the whole drag journals
    // nothing, which is the same outcome the op's refusal produced.
    let mut moves: Vec<(VertId, [f64; 3])> = Vec::new();
    for (v, w) in selection.soft_weights(mesh, mode, radius, falloff) {
        if !(w.is_finite() && w > 0.0) {
            continue;
        }
        let Some(p) = mesh.position(v) else { continue };
        let to = match xform {
            VertTransform::Translate(d) => p + d * w,
            VertTransform::Rotate { axis, radians } => {
                match inf_dcc::Rotation::new(pivot.to_array(), axis.to_array(), radians * w) {
                    Ok(rot) => rot.apply(p),
                    Err(_) => return Vec::new(),
                }
            }
            VertTransform::Scale(f) => {
                inf_dcc::scale_point(p, pivot, DVec3::ONE + (f - DVec3::ONE) * w)
            }
        };
        let delta = to - p;
        if delta == DVec3::ZERO {
            continue;
        }
        moves.push((v, delta.to_array()));
    }
    if moves.is_empty() {
        return Vec::new();
    }
    // `soft_weights` walks a `BTreeMap`, so this is already ascending — sorted
    // anyway, because the op's wire convention is "sorted by vertex id" and a
    // convention nobody enforces is a convention that drifts.
    moves.sort_by_key(|&(v, _)| v);
    vec![Op::MoveVerts { moves }]
}

/// How many distinct weights a soft transform may use — and therefore the most
/// ops one drag can journal.
///
/// # Why a cap at all, and why this is the interim fix
///
/// "One op per distinct weight" is the right *shape* — a soft move is ordinary
/// journal entries, so an undo gives the author their shape back — and it is the
/// wrong *granularity*: geodesic distance is continuous, so a 289-vertex plane
/// with a 3 m radius produced **105 ops from one drag**, which at
/// `CHECKPOINT_INTERVAL = 32` takes about three full mesh snapshots and **evicts
/// the entire eight-slot checkpoint history**, per drag, for ~7 kB of journal.
/// The gate that was supposed to cover this asserted `ops.len() > 1` — it named
/// the defect as a feature.
///
/// Quantizing to 64 steps caps a drag at 64 ops. The weight is a *falloff*, so
/// the visual difference between `w` and `round(64w)/64` is at most 1/128 of the
/// drag — invisible on a brush that already has a soft edge — and the bound is
/// hard rather than statistical.
///
/// **The real fix is a weight table on a `Sculpt`-shaped transform op**, so one
/// drag is one entry however many weights it touches. That is a wire change with
/// a `SessionSave` bump behind it and it is ledgered, not done here: it should
/// happen the day sessions actually persist, when the ladder has to move anyway.
pub const SOFT_WEIGHT_STEPS: f64 = 64.0;

/// A soft-select weight, rounded to [`SOFT_WEIGHT_STEPS`] steps.
///
/// `round` on a non-negative finite `f64` is exactly specified, so this keeps the
/// op list a pure function of the mesh — the quantization bounds the journal
/// without weakening the determinism argument.
pub fn quantize_weight(w: f64) -> f64 {
    if !w.is_finite() {
        return 0.0;
    }
    (w.clamp(0.0, 1.0) * SOFT_WEIGHT_STEPS).round() / SOFT_WEIGHT_STEPS
}

// ── starting a model at all (Wave D) ───────────────────────────────────────

/// The folder a newly-minted mesh lands in, under the project's content root.
///
/// `Meshes`, beside `Materials` — `asset_create`'s convention, and the one place
/// an author will look for something they made rather than imported.
pub const NEW_MESH_FOLDER: &str = "Meshes";

pub use crate::ipc::DccPrimitiveDto;

/// The **size** limits a primitive is built inside.
///
/// Refusals rather than clamps, per the units doctrine: the limit belongs to the
/// author's intent, not to the generator, and silently building a different
/// shape is how a mesh nobody asked for ends up in a level.
pub const MIN_PRIMITIVE_M: f64 = 1e-4;
/// A kilometre. Past this a "primitive" is a terrain, and this engine has one.
pub const MAX_PRIMITIVE_M: f64 = 1_000.0;
/// The most segments a generated primitive may have around its major axis.
///
/// 256 × 256 on a torus is 65 536 quads, which is a real model and past the
/// point where the Model Editor's CPU picker is interactive (see the
/// live-sculpt ceiling). Refused rather than accepted-and-slow.
pub const MAX_PRIMITIVE_SEGMENTS: u32 = 256;

/// Build one of the kernel's primitives, with every parameter checked.
///
/// **NaN at the numeric door** (the Wave E precedent): a non-finite size is
/// refused here rather than stored, because a mesh whose bounds exclude its own
/// geometry prints as `NaN` in a Details panel and makes every later comparison
/// false — including the ones a later refusal would rely on.
pub fn primitive_mesh(
    kind: DccPrimitiveDto,
    size_m: f64,
    segments: u32,
    rings: u32,
) -> Result<Mesh, String> {
    let size = check_size("a primitive size", size_m)?;
    let seg = check_segments("segments", segments)?;
    let ring = check_segments("rings", rings)?;
    Ok(match kind {
        DccPrimitiveDto::Cube => inf_dcc::cube(size),
        DccPrimitiveDto::Plane => inf_dcc::plane(size),
        // A cylinder's `size` is its DIAMETER, so a "1 m cube" and a "1 m
        // cylinder" occupy the same box. Radius would make the two buttons mean
        // different things under one number.
        DccPrimitiveDto::Cylinder => inf_dcc::cylinder(size * 0.5, size, seg),
        // …and a torus's is its outer diameter, for the same reason: the minor
        // radius is a quarter of it, which is the proportion every DCC's default
        // torus has.
        DccPrimitiveDto::Torus => inf_dcc::torus(size * 0.375, size * 0.125, seg, ring),
    })
}

fn check_size(what: &str, v: f64) -> Result<f64, String> {
    if !v.is_finite() {
        return Err(format!("{what} must be a number, got {v}"));
    }
    if !(MIN_PRIMITIVE_M..=MAX_PRIMITIVE_M).contains(&v) {
        return Err(format!(
            "{what} must be between {MIN_PRIMITIVE_M} m and {MAX_PRIMITIVE_M} m, got {v}"
        ));
    }
    Ok(v)
}

fn check_segments(what: &str, v: u32) -> Result<usize, String> {
    if !(3..=MAX_PRIMITIVE_SEGMENTS).contains(&v) {
        return Err(format!(
            "{what} must be between 3 and {MAX_PRIMITIVE_SEGMENTS}, got {v}"
        ));
    }
    Ok(v as usize)
}

/// Mint a `.inf_mesh` from a kernel mesh and derive everything that hangs off it.
///
/// The **creation** twin of [`save_mesh_session`], and it goes through the same
/// two doors in the same order for the same reason: `write_asset` (atomic, with
/// its sidecar) and then `ensure_vmesh` **synchronously**, because the next thing
/// the author does is look at the viewport and a queued derivation is a window in
/// which the level draws nothing.
///
/// Returns the new asset id and the writer's advisory report — which the caller
/// must **surface**, not drop: a primitive is generated geometry and cannot
/// normally trip one, but "cannot normally" is exactly the class of claim this
/// codebase keeps paying for.
pub fn create_mesh_asset(
    project: &mut crate::assets::AssetProject,
    name: &str,
    mesh: &Mesh,
) -> Result<(inf_asset::AssetId, inf_dcc::ExportReport), String> {
    let (payload, report) = inf_dcc::to_mesh_asset(mesh, &inf_dcc::ExportOptions::default());
    let dir = project
        .content_dir(NEW_MESH_FOLDER)
        .map_err(|e| e.to_string())?;
    let id = project
        .write_asset(&dir, name, &payload, None, vec![], None)
        .map_err(|e| format!("write {name}: {e}"))?;
    // Same synchronous derivation the save path takes. A primitive is under
    // `min_triangles` so this Skips, and Skipping is correct — but calling it is
    // what makes the two paths one contract instead of two habits.
    crate::assets::vmesh::ensure_vmesh(project, id).map_err(|e| format!("derive {name}: {e}"))?;
    Ok((id, report))
}

// ── the Wave-D derived tools ───────────────────────────────────────────────
//
// Each of these turns a *gesture* into the ops the kernel takes, and each one is
// here rather than in a `#[tauri::command]` for the memo §7c LAW: a command
// cannot be driven from a test, so a rule written there is a rule no gate can
// see. They are also all **solvers whose answer is journalled**, on the
// `Op::Unwrap` precedent — an op that re-derived "which boundary edge pairs with
// which" would silently rewrite every recorded session the day the matcher
// improved.

/// The boundary loops among a set of half-edges, each in `next` order.
///
/// A boundary loop is a cycle of half-edges with `face == None`, exactly like a
/// face loop (the kernel's first decision: `twin` is total and `face` is the
/// `Option`). Loops come back sorted by their lowest member so the answer does
/// not depend on which edge the author clicked first.
pub fn boundary_loops(mesh: &Mesh, edges: &[HalfId]) -> Vec<Vec<HalfId>> {
    let wanted: std::collections::BTreeSet<HalfId> = edges
        .iter()
        .flat_map(|&h| {
            // An author selects an *undirected* edge; only one of its halves is
            // the boundary one.
            [Some(h), mesh.twin(h)].into_iter().flatten()
        })
        .filter(|&h| mesh.is_boundary(h) == Some(true))
        .collect();
    let mut seen: std::collections::BTreeSet<HalfId> = std::collections::BTreeSet::new();
    let mut loops: Vec<Vec<HalfId>> = Vec::new();
    for &start in &wanted {
        if seen.contains(&start) {
            continue;
        }
        let mut cycle = Vec::new();
        let mut h = start;
        loop {
            if !seen.insert(h) {
                break;
            }
            cycle.push(h);
            match mesh.next(h) {
                Some(n) if n != start => h = n,
                _ => break,
            }
        }
        loops.push(cycle);
    }
    loops.sort_by_key(|c| c.iter().copied().min());
    loops
}

/// Pair two boundary loops for a bridge, and refuse — as a value — when the
/// selection is not two loops of the same length.
///
/// The pairing rule, derived rather than guessed: quad `k` uses `a[k]` and
/// `b[m]`, and quad `k + 1` shares an edge with it only if `b[m']` **ends** where
/// `b[m]` starts. So loop B is walked **backwards**, which is the same statement
/// as "two rings facing each other have boundary loops running opposite ways".
/// The offset `m0` is chosen by nearest midpoint, so the bridge does not put a
/// twist in a ring the author lined up by eye.
pub fn bridge_pairs(mesh: &Mesh, edges: &[HalfId]) -> Result<Vec<(HalfId, HalfId)>, String> {
    let loops = boundary_loops(mesh, edges);
    if loops.len() != 2 {
        return Err(format!(
            "a bridge needs exactly two open borders; this selection touches {}",
            loops.len()
        ));
    }
    let (a, b) = (&loops[0], &loops[1]);
    if a.len() != b.len() {
        return Err(format!(
            "the two borders have {} and {} edges; a bridge needs them equal",
            a.len(),
            b.len()
        ));
    }
    let mid = |h: HalfId| -> DVec3 {
        let o = mesh.origin(h).and_then(|v| mesh.position(v));
        let d = mesh.dest(h).and_then(|v| mesh.position(v));
        match (o, d) {
            (Some(o), Some(d)) => (o + d) * 0.5,
            _ => DVec3::ZERO,
        }
    };
    let anchor = mid(a[0]);
    let mut best = (0usize, f64::INFINITY);
    for (m, &h) in b.iter().enumerate() {
        let d = (mid(h) - anchor).length_squared();
        // `<` and not `<=`, so ties go to the lower index and the answer is a
        // pure function of the mesh.
        if d < best.1 {
            best = (m, d);
        }
    }
    let n = a.len();
    Ok((0..n)
        .map(|k| (a[k], b[(best.0 + n - k % n) % n]))
        .collect())
}

/// Where an edge/vertex **slide** puts each selected vertex, as one
/// [`Op::MoveVerts`]'s worth of deltas.
///
/// The direction is the vertex's own **ring** edge — the incident edge whose far
/// endpoint is *not* selected — which is precisely what makes selecting an edge
/// loop and sliding it do what a modeller expects: the loop's own edges have both
/// endpoints selected and are skipped, so what is left is the perpendicular ring.
/// `t > 0` slides toward the lower-id ring neighbour and `t < 0` toward its most
/// opposite partner, so one slider covers both directions and the sign is stable
/// across a drag.
///
/// `t` is clamped to `[-1, 1]`: `1` lands exactly on the neighbour, and past it
/// the vertex would cross the ring edge and invert the quad. Clamped rather than
/// refused because this is a *drag*, and refusing the far end of a gesture is
/// refusing the gesture.
pub fn slide_moves(
    mesh: &Mesh,
    selection: &SelectionSet,
    mode: SelectMode,
    t: f64,
) -> Vec<(VertId, [f64; 3])> {
    if !t.is_finite() || t == 0.0 {
        return Vec::new();
    }
    let t = t.clamp(-1.0, 1.0);
    let selected: std::collections::BTreeSet<VertId> = selection.resolved_verts(mesh, mode);
    let mut moves: Vec<(VertId, [f64; 3])> = Vec::new();
    for &v in &selected {
        let Some(p) = mesh.position(v) else { continue };
        // The ring neighbours, in half-edge id order — deterministic.
        let mut ring: Vec<(HalfId, VertId, DVec3)> = Vec::new();
        for &h in mesh.vert_outgoing(v).unwrap_or(&[]) {
            let Some(w) = mesh.dest(h) else { continue };
            if selected.contains(&w) {
                continue;
            }
            let Some(q) = mesh.position(w) else { continue };
            if ring.iter().any(|&(_, x, _)| x == w) {
                continue;
            }
            ring.push((h, w, q - p));
        }
        if ring.is_empty() {
            continue;
        }
        ring.sort_by_key(|&(h, _, _)| h);
        let forward = ring[0].2;
        let dir = if t > 0.0 {
            forward
        } else {
            // The most opposite ring edge, so a negative `t` really is the other
            // way rather than an arbitrary second neighbour.
            let mut best = (ring[0].2, f64::INFINITY);
            for &(_, _, d) in &ring[1..] {
                let c = d.normalize_or_zero().dot(forward.normalize_or_zero());
                if c < best.1 {
                    best = (d, c);
                }
            }
            best.0
        };
        let delta = dir * t.abs();
        if delta == DVec3::ZERO || !delta.is_finite() {
            continue;
        }
        moves.push((v, delta.to_array()));
    }
    moves.sort_by_key(|&(v, _)| v);
    moves
}

/// Cluster selected vertices that are within `tolerance` metres of each other.
///
/// The **kernel's reader never welds by epsilon** (`WELD_TOLERANCE` is exactly
/// zero, and that is a law): a file that comes back smaller than it went in is
/// data loss nobody asked for. A *tool* is a different thing — the author asked,
/// the result is one journal entry per cluster, and undo puts it back. The law
/// already says so at `inf_dcc::build`'s tolerance note.
///
/// Deterministic: single-linkage over pairs enumerated in ascending id order,
/// and every cluster comes back sorted with the whole list sorted by first
/// member. The honest bound is the op count — **one `MergeVerts` per cluster** —
/// so a selection with a thousand duplicate pairs is a thousand undo steps. That
/// is the shape the kernel offers (`MergeVerts` fuses one set), and a
/// result-carrying batch op is a wire change this wave's single schema move is
/// already spent on.
pub fn merge_clusters(
    mesh: &Mesh,
    verts: &[VertId],
    tolerance: f64,
) -> Result<Vec<Vec<VertId>>, String> {
    if !(tolerance.is_finite() && tolerance >= 0.0) {
        return Err(format!(
            "a merge tolerance must be finite and ≥ 0, got {tolerance}"
        ));
    }
    let mut ids: Vec<VertId> = verts.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let n = ids.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let tol2 = tolerance * tolerance;
    // Positions resolved ONCE, up front: the pairwise loop below reads each of
    // them `n` times, and a vertex whose position is missing must be skipped in
    // both halves consistently rather than re-asked about.
    let places: Vec<Option<DVec3>> = ids.iter().map(|&v| mesh.position(v)).collect();
    for (i, pi) in places.iter().enumerate() {
        let Some(pi) = pi else { continue };
        for (j, pj) in places.iter().enumerate().skip(i + 1) {
            let Some(pj) = pj else { continue };
            if (*pi - *pj).length_squared() <= tol2 {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    // Union toward the LOWER root, so the forest is a pure
                    // function of the input rather than of the visit order.
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<VertId>> =
        std::collections::BTreeMap::new();
    for (i, &v) in ids.iter().enumerate() {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(v);
    }
    Ok(groups.into_values().filter(|g| g.len() > 1).collect())
}

/// The edges shade-smooth / shade-flat / auto-smooth marks, split into the set to
/// make **sharp** and the set to make **smooth**.
///
/// `faces` empty means "the whole mesh" — "recalculate shading" with nothing
/// selected is the gesture every modeller expects to apply everywhere.
///
/// `angle_deg` is the auto-smooth threshold: an edge whose two faces disagree by
/// more than it becomes sharp, the rest become smooth. The comparison is
/// `dot(n1, n2) < cos(threshold)` — **no `acos`**, which is both faster and the
/// only version the P14 portability law permits on a path that writes committed
/// content. `cos` itself comes from [`inf_math::pcos64`] for the same reason.
pub fn shade_edges(
    mesh: &Mesh,
    faces: &[FaceId],
    smooth: bool,
    angle_deg: Option<f64>,
) -> (Vec<HalfId>, Vec<HalfId>) {
    let scope: std::collections::BTreeSet<FaceId> = if faces.is_empty() {
        mesh.face_ids().collect()
    } else {
        faces.iter().copied().collect()
    };
    let mut edges: std::collections::BTreeSet<HalfId> = std::collections::BTreeSet::new();
    for &f in &scope {
        for h in mesh.face_loop(f).unwrap_or_default() {
            if let Some(c) = inf_dcc::canonical_edge(mesh, h) {
                edges.insert(c);
            }
        }
    }
    let Some(deg) = angle_deg else {
        let all: Vec<HalfId> = edges.into_iter().collect();
        return if smooth {
            (Vec::new(), all)
        } else {
            (all, Vec::new())
        };
    };
    let threshold = inf_math::pcos64(deg.clamp(0.0, 180.0) * std::f64::consts::PI / 180.0);
    let (mut sharp, mut soft) = (Vec::new(), Vec::new());
    for h in edges {
        let Some(t) = mesh.twin(h) else { continue };
        let n1 = mesh.face_of(h).flatten().map(|f| face_normal_of(mesh, f));
        let n2 = mesh.face_of(t).flatten().map(|f| face_normal_of(mesh, f));
        match (n1, n2) {
            (Some(a), Some(b)) if a != DVec3::ZERO && b != DVec3::ZERO => {
                if a.normalize().dot(b.normalize()) < threshold {
                    sharp.push(h);
                } else {
                    soft.push(h);
                }
            }
            // A boundary edge has no dihedral. It is marked SHARP, which is what
            // it looks like anyway: the surface really does end there.
            _ => sharp.push(h),
        }
    }
    (sharp, soft)
}

// ── UV-space editing (Wave D) ──────────────────────────────────────────────
//
// The UV pane has been a handler-less `<img>` since P23.5 — "UV-space dragging"
// is the carried remainder these three close. Everything here works in the pane's
// own pixel space, which is the inverse of `draw_uv_layout`'s `to_px` and is
// written once, here, so the picture and the pick cannot disagree (the P23.4
// rule that put `pick` and `draw_overlay` behind one `Projector`, applied to the
// second view).

/// The pane pixel a UV lands on. The inverse of [`uv_from_px`].
fn uv_to_px(uv: [f64; 2], size: f32) -> (f32, f32) {
    // UV (0,0) is bottom-left; pixel y grows downward — `draw_uv_layout`'s rule.
    (uv[0] as f32 * size, (1.0 - uv[1] as f32) * size)
}

/// The UV a pane pixel names.
pub fn uv_from_px(px: f32, py: f32, size: f32) -> [f64; 2] {
    [(px / size) as f64, (1.0 - py / size) as f64]
}

/// **Pick in the UV pane**: the vertex whose nearest corner is under the pointer.
///
/// Returns a *vertex*, not a corner, and that is the decision: the selection is
/// shared with the 3D view, so picking in UV space has to answer in the same
/// currency or the two pictures would disagree about what is selected. A vertex
/// on a seam has a corner in each chart it touches; clicking either one selects
/// the vertex, and a drag then moves **every** corner of it — which is what
/// "move this vertex in UV space" has to mean when the vertex is cut.
///
/// `radius_px` is the same 7-pixel reach [`pick`] uses, for the same reason.
pub fn pick_uv(mesh: &Mesh, size: u32, px: f32, py: f32) -> Option<VertId> {
    let s = size.max(1) as f32;
    let mut best: Option<(f32, VertId)> = None;
    for h in mesh.half_ids() {
        if mesh.is_boundary(h) != Some(false) {
            continue;
        }
        let (Some(uv), Some(v)) = (mesh.corner_uv(h), mesh.origin(h)) else {
            continue;
        };
        let (x, y) = uv_to_px(uv, s);
        let d = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
        if d > PICK_RADIUS_PX {
            continue;
        }
        // Ties break toward the lower vertex id, so a pick is a pure function of
        // the mesh rather than of iteration order.
        if best.is_none_or(|(bd, bv)| (d, v) < (bd, bv)) {
            best = Some((d, v));
        }
    }
    best.map(|(_, v)| v)
}

/// **Move the selection's corners in UV space**, as one
/// [`inf_dcc::Op::MoveUvs`]'s worth of values.
///
/// The delta arrives in **pane pixels** and is converted here, because the pane's
/// size is the panel's business and the op's units are the kernel's. A drag of
/// `size` pixels is a drag of one whole UV unit, which is what makes the gesture
/// feel like dragging the picture.
///
/// Every corner of every selected vertex moves — including the ones in other
/// charts. A vertex cut by a seam has a corner per chart, and moving one of them
/// and not the others is how a seam becomes a tear.
pub fn uv_move_corners(
    mesh: &Mesh,
    selection: &SelectionSet,
    mode: SelectMode,
    size: u32,
    dx_px: f64,
    dy_px: f64,
) -> Vec<(HalfId, [f64; 2])> {
    let s = size.max(1) as f64;
    let (du, dv) = (dx_px / s, -dy_px / s);
    if !(du.is_finite() && dv.is_finite()) || (du == 0.0 && dv == 0.0) {
        return Vec::new();
    }
    let verts = selection.resolved_verts(mesh, mode);
    let mut out: Vec<(HalfId, [f64; 2])> = Vec::new();
    for v in verts {
        for &h in mesh.vert_outgoing(v).unwrap_or(&[]) {
            if mesh.is_boundary(h) != Some(false) {
                continue;
            }
            let Some(uv) = mesh.corner_uv(h) else {
                continue;
            };
            let moved = [uv[0] + du, uv[1] + dv];
            if !(moved[0].is_finite() && moved[1].is_finite()) {
                continue;
            }
            out.push((h, moved));
        }
    }
    // Sorted by half-edge — the op's wire convention, and what makes two runs of
    // one drag produce one byte string.
    out.sort_by_key(|&(h, _)| h);
    out.dedup_by_key(|&mut (h, _)| h);
    out
}

/// **Auto-seam**: every edge whose two faces disagree by more than `angle_deg`,
/// plus every boundary edge.
///
/// The modeller's half of "smart UV project", and it is the same dihedral
/// measurement [`shade_edges`] makes — one rule, two uses, because "where does
/// this surface crease" has one answer and giving it two would let them drift.
/// A boundary edge is always a seam: the surface really does end there, and a
/// chart that walks off the edge of the mesh is not a chart.
///
/// The angle is degrees at the UI boundary; the comparison is
/// `dot(n1, n2) < cos(threshold)` — **no `acos`**, both because it is faster and
/// because the P14 portability law bans `std` trigonometry from anything that
/// writes committed content. `cos` itself is [`inf_math::pcos64`].
///
/// The **result** is what gets journalled (one [`inf_dcc::Op::SetEdgesSeam`]),
/// not the angle: an op that re-derived the edge set would change meaning the
/// day the measurement did — the `Op::Unwrap` precedent.
pub fn auto_seam_edges(mesh: &Mesh, angle_deg: f64) -> Vec<HalfId> {
    let threshold = inf_math::pcos64(angle_deg.clamp(0.0, 180.0) * std::f64::consts::PI / 180.0);
    let mut out: Vec<HalfId> = Vec::new();
    let mut seen: std::collections::BTreeSet<HalfId> = std::collections::BTreeSet::new();
    for h in mesh.half_ids() {
        let Some(c) = inf_dcc::canonical_edge(mesh, h) else {
            continue;
        };
        if !seen.insert(c) {
            continue;
        }
        let Some(t) = mesh.twin(c) else { continue };
        match (mesh.face_of(c).flatten(), mesh.face_of(t).flatten()) {
            (Some(f), Some(g)) => {
                let (a, b) = (face_normal_of(mesh, f), face_normal_of(mesh, g));
                if a == DVec3::ZERO || b == DVec3::ZERO {
                    continue;
                }
                if a.normalize().dot(b.normalize()) < threshold {
                    out.push(c);
                }
            }
            // A boundary edge. The surface ends here, so a chart must too.
            _ => out.push(c),
        }
    }
    out
}

/// A face's area-weighted normal, or zero for a degenerate face.
fn face_normal_of(mesh: &Mesh, f: FaceId) -> DVec3 {
    let verts = mesh.face_verts(f).unwrap_or_default();
    if verts.len() < 3 {
        return DVec3::ZERO;
    }
    // Newell's method, which is the one the kernel's exporter uses and the only
    // one that is right for a non-planar n-gon.
    let mut n = DVec3::ZERO;
    for i in 0..verts.len() {
        let (Some(a), Some(b)) = (
            mesh.position(verts[i]),
            mesh.position(verts[(i + 1) % verts.len()]),
        ) else {
            return DVec3::ZERO;
        };
        n.x += (a.y - b.y) * (a.z + b.z);
        n.y += (a.z - b.z) * (a.x + b.x);
        n.z += (a.x - b.x) * (a.y + b.y);
    }
    n
}

/// A **content revision of the selection** — the number a view keys on to know
/// its picture is stale.
///
/// # Why a hash and not a counter
///
/// A counter has to be bumped, and the P23.5 audit found what happens when the
/// thing keyed on is not the thing that changed: the UV pane keyed on the
/// *journal* generation, so picking a different face never refreshed it —
/// selecting is not a journal op — and `selected` is a **count**, so switching
/// from face A to face B left it reading `1` and even a count-keyed view would
/// have stayed stale. That falsified the panel's own "one `SelectionSet`, one
/// document" claim in the one place it was visible.
///
/// A hash over the set's contents cannot be forgotten by a new mutation path,
/// which is the property a counter does not have. FNV-1a over the generation and
/// the three id sets, in `BTreeSet` order.
pub fn selection_revision(selection: &SelectionSet) -> u64 {
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        acc ^= x;
        acc = acc.wrapping_mul(0x1000_0000_01b3);
    };
    mix(selection.generation());
    for v in selection.verts() {
        mix(v.0 as u64 | (1 << 32));
    }
    for e in selection.edges() {
        mix(e.0 as u64 | (2 << 32));
    }
    for f in selection.faces() {
        mix(f.0 as u64 | (3 << 32));
    }
    acc
}

/// The pivot a gizmo sits on: the **centroid of the selected vertices**.
///
/// Not the bounding-box centre: a centroid is what the transform ops are
/// pivot-relative to, and a box centre would put the visible handle somewhere the
/// rotation is not actually about on any asymmetric selection.
/// **Where the gizmo sits** — the first of the two constants P23.5 hard-wired.
///
/// Blender offers five; four of them are a statement about the selection and one
/// (*individual origins*) is a statement about the **op**, because it means "run
/// this transform once per element about its own centre". `transform_ops`
/// produces one op for one transform, so individual origins is a different shape
/// and is a named remainder rather than a fifth variant nobody can honour.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, ts_rs::TS,
)]
#[serde(rename_all = "camelCase")]
pub enum DccPivotDto {
    /// The mean of the selected vertices. What P23.5 hard-wired, and still the
    /// default because it is what a modeller means nine times in ten.
    #[default]
    Median,
    /// The centre of the selection's axis-aligned bounding box. Differs from the
    /// median whenever the vertices are unevenly distributed — which is the case
    /// an author reaches for this in.
    BoundingBox,
    /// The mesh's own origin. Rotating a whole object about it is the one thing
    /// a median pivot cannot express.
    WorldOrigin,
    /// The last component clicked. Blender's "active element", and the only
    /// pivot that depends on the *order* the selection was built in — which is
    /// why the document tracks it rather than deriving it.
    ActiveElement,
}

/// **Which way the gizmo's axes point** — the second hard-wired constant
/// (`Quat::IDENTITY`, at two sites).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, ts_rs::TS,
)]
#[serde(rename_all = "camelCase")]
pub enum DccOrientDto {
    /// World axes. What was hard-wired.
    #[default]
    Global,
    /// +Z along the selection's area-weighted normal — so "extrude along the
    /// blue handle" is "extrude along the surface" without the author computing
    /// anything. Blender's own convention for the normal orientation.
    Normal,
    /// The camera's basis: +Z toward the eye, +X right, +Y up. Dragging the red
    /// handle moves the selection across the screen.
    View,
}

/// The gizmo's rotation, for a pivot kind and the camera it is drawn against.
///
/// Handed to `inf_render::gizmo::pick_axis` and to `GizmoDrag::begin`, which are
/// the exact two sites that used to read `Quat::IDENTITY` — so a caller cannot
/// pick up one and forget the other, because the value comes from here.
pub fn gizmo_orientation(
    mesh: &Mesh,
    selection: &SelectionSet,
    mode: SelectMode,
    kind: DccOrientDto,
    view: PreviewView,
) -> glam::Quat {
    match kind {
        DccOrientDto::Global => glam::Quat::IDENTITY,
        DccOrientDto::Normal => {
            let n = selection_normal(mesh, selection, mode);
            match n {
                // `from_rotation_arc` is undefined for opposite vectors, so the
                // 180° case is spelled out rather than left to produce a NaN
                // quaternion the gizmo would then hit-test against.
                Some(n) if n.dot(Vec3::Z) < -0.999_999 => {
                    glam::Quat::from_axis_angle(Vec3::X, std::f32::consts::PI)
                }
                Some(n) => glam::Quat::from_rotation_arc(Vec3::Z, n),
                None => glam::Quat::IDENTITY,
            }
        }
        DccOrientDto::View => {
            let (eye, _) = view.view_proj();
            let fwd = (eye - view.target).normalize_or_zero();
            if fwd == Vec3::ZERO {
                return glam::Quat::IDENTITY;
            }
            let right = view.up.cross(fwd).normalize_or_zero();
            if right == Vec3::ZERO {
                return glam::Quat::IDENTITY;
            }
            glam::Quat::from_mat3(&glam::Mat3::from_cols(right, fwd.cross(right), fwd))
        }
    }
}

/// The unit average of the selected faces' normals, or `None` when there is no
/// usable direction.
///
/// Area-weighted through the same Newell sum the exporter and the kernel use, so
/// "the normal" means one thing in this codebase. In vertex or edge mode the
/// faces *touching* the selection are what is averaged — an author in vertex mode
/// pointing at a surface still means the surface.
fn selection_normal(mesh: &Mesh, selection: &SelectionSet, mode: SelectMode) -> Option<Vec3> {
    let mut acc = DVec3::ZERO;
    let faces: std::collections::BTreeSet<inf_dcc::FaceId> = if mode == SelectMode::Face {
        selection.faces().iter().copied().collect()
    } else {
        let verts = selection.resolved_verts(mesh, mode);
        let mut out = std::collections::BTreeSet::new();
        for v in verts {
            for &h in mesh.vert_outgoing(v).unwrap_or(&[]) {
                if let Some(Some(f)) = mesh.face_of(h) {
                    out.insert(f);
                }
            }
        }
        out
    };
    for f in faces {
        acc += face_normal_of(mesh, f);
    }
    let n = acc.normalize_or_zero();
    (n != DVec3::ZERO).then(|| Vec3::new(n.x as f32, n.y as f32, n.z as f32))
}

/// Where the gizmo sits, for a pivot kind.
///
/// `active` is the position of the last component the author clicked, which the
/// document tracks because it cannot be derived: a `BTreeSet` has no "last".
pub fn gizmo_pivot_of(
    mesh: &Mesh,
    selection: &SelectionSet,
    mode: SelectMode,
    kind: DccPivotDto,
    active: Option<DVec3>,
) -> Option<DVec3> {
    match kind {
        DccPivotDto::Median => gizmo_pivot(mesh, selection, mode),
        DccPivotDto::WorldOrigin => {
            // Still `None` on an empty selection: a gizmo with nothing to move is
            // not a gizmo, and drawing one at the origin would invite a drag that
            // journals nothing.
            gizmo_pivot(mesh, selection, mode).map(|_| DVec3::ZERO)
        }
        DccPivotDto::BoundingBox => {
            let verts = selection.resolved_verts(mesh, mode);
            let (mut lo, mut hi) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
            let mut any = false;
            for v in verts {
                if let Some(p) = mesh.position(v) {
                    if p.is_finite() {
                        lo = lo.min(p);
                        hi = hi.max(p);
                        any = true;
                    }
                }
            }
            any.then(|| (lo + hi) * 0.5)
        }
        // Falls back to the median when nothing has been clicked yet — the
        // alternative is a gizmo that vanishes the first time an author picks
        // this mode before picking a component.
        DccPivotDto::ActiveElement => active
            .filter(|p| p.is_finite())
            .or_else(|| gizmo_pivot(mesh, selection, mode)),
    }
}

/// A screen-space rectangle in the preview's own pixel space, origin top-left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl BoxRect {
    /// Normalized so `x0 <= x1` and `y0 <= y1` — a drag may go any direction.
    pub fn normalized(self) -> Self {
        Self {
            x0: self.x0.min(self.x1),
            y0: self.y0.min(self.y1),
            x1: self.x0.max(self.x1),
            y1: self.y0.max(self.y1),
        }
    }

    fn contains(&self, p: &Projected) -> bool {
        p.x >= self.x0 && p.x <= self.x1 && p.y >= self.y0 && p.y <= self.y1
    }

    /// A drag of a few pixels is a click that wobbled, not a marquee.
    pub fn is_degenerate(&self) -> bool {
        let n = self.normalized();
        (n.x1 - n.x0) < 2.0 && (n.y1 - n.y0) < 2.0
    }
}

/// **What a marquee catches** — box select, in the current component mode.
///
/// The rules are Blender's, and each one is a decision:
///
/// * a **vertex** is caught when its projection is inside the rectangle;
/// * an **edge** when *both* endpoints are — a rectangle clipping the middle of
///   a long edge has not selected it, it has crossed it;
/// * a **face** when *every* corner is.
///
/// `through` is the x-ray toggle. With it off, a component is caught only if it
/// is facing the eye — the same visibility test [`pick`] uses, because a marquee
/// that grabs the far side of a closed model is the single most common way to
/// delete geometry by accident. **It is back-face culling and not depth
/// testing** (the carried remainder the overlay names): the far side of a *fold*
/// still gets caught. Said here rather than discovered.
pub fn pick_box(
    mesh: &Mesh,
    proj: &Projector,
    mode: SelectMode,
    rect: BoxRect,
    through: bool,
) -> (Vec<VertId>, Vec<HalfId>, Vec<inf_dcc::FaceId>) {
    let rect = rect.normalized();
    let inside = |v: VertId| -> bool {
        mesh.position(v)
            .and_then(|p| proj.point(p))
            .is_some_and(|q| rect.contains(&q))
    };
    let mut verts = Vec::new();
    let mut edges = Vec::new();
    let mut faces = Vec::new();
    match mode {
        SelectMode::Vert => {
            for v in mesh.vert_ids() {
                if inside(v) && (through || vert_is_visible(mesh, proj, v)) {
                    verts.push(v);
                }
            }
        }
        SelectMode::Edge => {
            for h in mesh.half_ids() {
                // Once per undirected edge.
                if inf_dcc::canonical_edge(mesh, h) != Some(h) {
                    continue;
                }
                let (Some(a), Some(b)) = (mesh.origin(h), mesh.dest(h)) else {
                    continue;
                };
                if inside(a) && inside(b) && (through || edge_is_visible(mesh, proj, h)) {
                    edges.push(h);
                }
            }
        }
        SelectMode::Face => {
            for f in mesh.face_ids() {
                let vs = mesh.face_verts(f).unwrap_or_default();
                if !vs.is_empty()
                    && vs.iter().all(|&v| inside(v))
                    && (through || face_faces_eye(mesh, proj, f))
                {
                    faces.push(f);
                }
            }
        }
    }
    (verts, edges, faces)
}

/// A vertex is visible when **any** face around it faces the eye — a vertex on
/// a silhouette belongs to the near side too, and requiring all of them would
/// drop exactly the vertices an author is aiming at.
fn vert_is_visible(mesh: &Mesh, proj: &Projector, v: VertId) -> bool {
    let out = mesh.vert_outgoing(v).unwrap_or(&[]);
    if out.is_empty() {
        return true; // an isolated vertex has no facing to be wrong about
    }
    out.iter().any(|&h| match mesh.face_of(h) {
        Some(Some(f)) => face_faces_eye(mesh, proj, f),
        _ => true, // a boundary corner is always a candidate
    })
}

pub fn gizmo_pivot(mesh: &Mesh, selection: &SelectionSet, mode: SelectMode) -> Option<DVec3> {
    let verts = selection.resolved_verts(mesh, mode);
    if verts.is_empty() {
        return None;
    }
    let mut acc = DVec3::ZERO;
    let mut n = 0usize;
    for v in verts {
        if let Some(p) = mesh.position(v) {
            if p.is_finite() {
                acc += p;
                n += 1;
            }
        }
    }
    (n > 0).then(|| acc / n as f64)
}

/// The gizmo's world size at `pivot` — screen-constant, through
/// `inf_render`'s own rule so the handle is the same fraction of the frame here
/// as it is in the level viewport.
pub fn gizmo_size(proj: &Projector, view: PreviewView, pivot: DVec3) -> f32 {
    let p = Vec3::new(pivot.x as f32, pivot.y as f32, pivot.z as f32);
    inf_render::gizmo::gizmo_world_size(p, proj.eye(), view.fov_deg.to_radians())
}

/// Which handle is under `(px, py)`, through `inf_render::gizmo::pick_axis` — the
/// same analytic 11-pixel hit-test the level viewport's gizmo uses.
///
/// Reused rather than re-derived: the DCC's gizmo is the *same widget* on a
/// different set of things, and a second hit-tester would be a second feel.
pub fn pick_gizmo(
    proj: &Projector,
    view: PreviewView,
    pivot: DVec3,
    // The gizmo's rotation — `gizmo_orientation`'s answer, passed in rather than
    // assumed. It used to be `Quat::IDENTITY` here **and** at `GizmoDrag::begin`
    // one ring up, which is the shape where one site gets an orientation and the
    // other does not, and the symptom is a gizmo that picks one axis and drags
    // along another.
    quat: glam::Quat,
    mode: GizmoMode,
    px: f32,
    py: f32,
) -> Option<GizmoAxis> {
    let p = Vec3::new(pivot.x as f32, pivot.y as f32, pivot.z as f32);
    inf_render::gizmo::pick_axis(
        mode,
        p,
        quat,
        gizmo_size(proj, view, pivot),
        proj.view_proj(),
        glam::Vec2::new(px, py),
        proj.size(),
        proj.size(),
        false,
    )
}

/// Composite the gizmo handles into a rendered frame.
///
/// **Drawn to match `pick_axis`'s geometry exactly** — axis lines from the pivot
/// to `pivot + dir × size`, plane markers at `pivot + (u + v) × size × 0.35`, and
/// rotate rings of radius `size` sampled at [`RING_SEGMENTS`]. `inf_render`'s
/// `build_geometry` is the level viewport's twin of this and emits `DebugDraw`
/// lines into a GPU pass, which is exactly what this panel does not have (the
/// P23.4 ruling: the overlay is CPU-composited so what lights up is what `pick`
/// would return). The two therefore draw the same shape by two mechanisms, and
/// `the_gizmo_is_hittable_wherever_it_is_painted` is the gate that keeps this one
/// honest against the picker it shares.
pub fn draw_gizmo(
    rgba: &mut [u8],
    size: u32,
    proj: &Projector,
    view: PreviewView,
    pivot: DVec3,
    mode: GizmoMode,
    active: Option<GizmoAxis>,
) {
    if rgba.len() < (size as usize) * (size as usize) * 4 {
        return;
    }
    if !pivot.is_finite() {
        return;
    }
    let w = size as i32;
    let g = gizmo_size(proj, view, pivot);
    if !(g.is_finite() && g > 0.0) {
        return;
    }
    let origin = pivot;
    let colour = |axis: GizmoAxis| -> [u8; 3] {
        if active == Some(axis) {
            return [255, 217, 38]; // the amber highlight `build_geometry` uses
        }
        let c = axis.color();
        [
            (c[0] * 255.0) as u8,
            (c[1] * 255.0) as u8,
            (c[2] * 255.0) as u8,
        ]
    };
    let axis_dir = |axis: GizmoAxis| {
        let d = axis.dir();
        DVec3::new(d.x as f64, d.y as f64, d.z as f64)
    };
    let g = g as f64;

    if mode == GizmoMode::Rotate {
        for a in [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z] {
            let n = axis_dir(a);
            let (u, v) = plane_tangents(n);
            let mut prev: Option<Projected> = None;
            for i in 0..=RING_SEGMENTS {
                let t = i as f64 / RING_SEGMENTS as f64 * std::f64::consts::TAU;
                let p = proj.point(origin + (u * t.cos() + v * t.sin()) * g);
                if let (Some(x), Some(y)) = (prev, p) {
                    line(rgba, w, x.x, x.y, y.x, y.y, colour(a));
                }
                prev = p;
            }
        }
        return;
    }

    // Translate / scale: the three axis lines, then the three plane markers.
    for a in [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z] {
        let (Some(o), Some(tip)) = (proj.point(origin), proj.point(origin + axis_dir(a) * g))
        else {
            continue;
        };
        line(rgba, w, o.x, o.y, tip.x, tip.y, colour(a));
        // A solid tip so the end of a short handle is still visible and still
        // matches `pick_axis`, whose axis test is distance to the SEGMENT.
        dot(rgba, w, tip.x, tip.y, 2, colour(a));
    }
    for a in [GizmoAxis::PlaneX, GizmoAxis::PlaneY, GizmoAxis::PlaneZ] {
        let (u, v) = plane_tangents(axis_dir(a));
        let corner = origin + (u + v) * g * 0.35;
        if let Some(c) = proj.point(corner) {
            dot(rgba, w, c.x, c.y, 3, colour(a));
        }
    }
}

/// The two tangents of a plane whose normal is an axis direction — the same
/// choice `inf_render::gizmo`'s private `plane_tangents` makes, and it has to
/// be, or the plane handles are drawn in one corner and picked in another.
fn plane_tangents(normal: DVec3) -> (DVec3, DVec3) {
    if normal == DVec3::X {
        (DVec3::Y, DVec3::Z)
    } else if normal == DVec3::Y {
        (DVec3::X, DVec3::Z)
    } else {
        (DVec3::X, DVec3::Y)
    }
}

// ── a drag in flight, and the orphan-settler doctrine (P23.5) ──────────────

/// A brush stroke being drawn: the parameters, and the raw surface points the
/// pointer has visited so far.
///
/// The **raw** path, not the resampled dabs: resampling is arc-length, so
/// resampling incrementally as points arrive would produce a different dab set
/// than resampling the finished path once. One resample, at the end, in
/// [`StrokeInFlight::op`].
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeInFlight {
    pub mode: inf_dcc::SculptMode,
    pub radius: f64,
    pub strength: f64,
    pub falloff: inf_dcc::SculptFalloff,
    pub path: Vec<DVec3>,
    /// The surface normal at the most recent path point — **display state**, so
    /// [`draw_brush_ring`] can lay the footprint in the tangent plane without
    /// searching for the face again on every frame. Not part of the op: the
    /// stroke's arithmetic derives its own normals from the mesh.
    pub last_normal: DVec3,
}

impl StrokeInFlight {
    /// The journal entry this stroke has become, or `None` when it never touched
    /// the surface.
    pub fn op(&self) -> Option<Op> {
        if self.path.is_empty() {
            return None;
        }
        let dabs: Vec<[f64; 3]> = inf_dcc::stroke_dabs(&self.path, self.radius)
            .into_iter()
            .map(|p| p.to_array())
            .collect();
        if dabs.is_empty() {
            return None;
        }
        Some(Op::Sculpt {
            mode: self.mode,
            dabs,
            radius: self.radius,
            strength: self.strength,
            falloff: self.falloff,
        })
    }
}

/// **Begin a sculpt stroke** — the pointer ray, the radius floor and the surface
/// hit, in one place a test can reach.
///
/// Three answers, because there are three outcomes and the panel does something
/// different for each:
///
/// * `Err(reason)` — **refused**. The radius is below [`inf_dcc::MIN_BRUSH_RADIUS_M`],
///   which is a sentence the author needs to read.
/// * `Ok(None)` — **missed**. The pointer is off the model; the gesture becomes a
///   camera orbit and nothing is said.
/// * `Ok(Some(stroke))` — grabbed.
///
/// # Why this is Ring 1 and not four lines inside the command
///
/// The floor lived in `dcc_drag_begin` and the P23.5 audit's mutation table found
/// what that meant: deleting it failed **nothing**, because a `#[tauri::command]`
/// cannot be driven from a test on any CI leg. That is the same finding P23.4's
/// audit made about the save (§7c: *"a gate that inlines the code it is gating is
/// a copy, not a gate"*), and it has the same fix — the decision moves to where a
/// test can call it, and the command reports its verdict.
/// The parameters a sculpt stroke is begun with — the popover's state, as one
/// value so [`begin_stroke`] takes a brush rather than five loose numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrushSettings {
    pub mode: inf_dcc::SculptMode,
    /// Geodesic reach, **metres**. Refused below [`inf_dcc::MIN_BRUSH_RADIUS_M`].
    pub radius: f64,
    pub strength: f64,
    pub falloff: inf_dcc::SculptFalloff,
}

pub fn begin_stroke(
    mesh: &Mesh,
    proj: &Projector,
    px: f32,
    py: f32,
    brush: BrushSettings,
) -> Result<Option<StrokeInFlight>, String> {
    let BrushSettings {
        mode,
        radius,
        strength,
        falloff,
    } = brush;
    // **A refusal as a value, not a clamp.** A clamp would sculpt at a radius the
    // author did not ask for and never say so; the floor exists because `1e-12` is
    // one keystroke from `1e-1` in a number box, and at that radius the resampler
    // was asked for 1.2e13 dab positions — an out-of-memory abort taking the
    // unsaved session with it.
    if !(radius.is_finite() && radius >= inf_dcc::MIN_BRUSH_RADIUS_M) {
        return Err(format!(
            "a brush radius of {radius} m is below the {} m floor; a brush that \
             small cannot reach a second vertex on anything modelled in metres",
            inf_dcc::MIN_BRUSH_RADIUS_M
        ));
    }
    let Some((face, hit)) = pick_surface(mesh, proj, px, py) else {
        return Ok(None);
    };
    Ok(Some(StrokeInFlight {
        mode,
        radius,
        strength,
        falloff,
        path: vec![hit],
        last_normal: inf_dcc::face_normal(mesh, face).unwrap_or(DVec3::Y),
    }))
}

/// A gizmo drag being made: the handle, the pivot, and the transform the pointer
/// currently implies.
#[derive(Debug, Clone, Copy)]
pub struct GizmoInFlight {
    pub drag: GizmoDrag,
    pub pivot: DVec3,
    pub xform: VertTransform,
    /// `Some((radius, falloff))` when the drag is soft.
    pub soft: Option<(f64, inf_terrain::Falloff)>,
}

/// **The one pending-interaction slot on a document.**
///
/// Sculpt and the gizmo are the same gesture with different arithmetic —
/// pointer-down, a run of moves, pointer-up — so they share one slot, one
/// preview path and one settler. Two slots would be two places for a drag to be
/// forgotten, and forgetting a drag is exactly what the doctrine below exists to
/// prevent.
#[derive(Debug, Clone)]
pub enum PendingDrag {
    Stroke(StrokeInFlight),
    Gizmo(Box<GizmoInFlight>),
    /// A **weight-paint** stroke (P24.2). Third gesture, same slot, same settler
    /// — the reason the slot is one slot.
    Weights(WeightStrokeInFlight),
}

/// The parameters a weight stroke is begun with — [`BrushSettings`]'s twin for
/// the skin channel.
///
/// A separate struct rather than a mode on `BrushSettings` because the two share
/// only their radius: a sculpt brush needs a falloff and a displacement, a weight
/// brush needs a **joint**, and folding them would make "which joint" a field
/// that means nothing three quarters of the time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightBrush {
    /// The influence being painted — an index into the mesh's bound skeleton.
    pub joint: u16,
    pub mode: inf_dcc::PaintMode,
    /// Geodesic reach, **metres**. Refused below [`inf_dcc::MIN_BRUSH_RADIUS_M`],
    /// by the same door and for the same reason as a sculpt radius.
    pub radius: f64,
    /// Weight delta at full coverage, `[0, 1]`.
    pub strength: f64,
    pub falloff: inf_dcc::SculptFalloff,
}

/// A weight stroke being painted: the brush, and the raw surface points so far.
///
/// The **raw** path for the same reason [`StrokeInFlight`] keeps one — arc-length
/// resampling incrementally gives a different dab set than resampling the
/// finished path once.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightStrokeInFlight {
    pub brush: WeightBrush,
    pub path: Vec<DVec3>,
    /// Display state for the brush ring, exactly as on a sculpt stroke.
    pub last_normal: DVec3,
}

impl WeightStrokeInFlight {
    /// The journal entry this stroke has become.
    ///
    /// **Needs the mesh**, unlike [`StrokeInFlight::op`], because the op carries
    /// the stroke's *result* rather than its dabs — see [`inf_dcc::paint`] for
    /// why a weight stroke cannot be replayed from its parameters.
    pub fn op(&self, mesh: &Mesh) -> Option<Op> {
        if self.path.is_empty() {
            return None;
        }
        let dabs: Vec<[f64; 3]> = inf_dcc::stroke_dabs(&self.path, self.brush.radius)
            .into_iter()
            .map(|p| p.to_array())
            .collect();
        if dabs.is_empty() {
            return None;
        }
        // A refusal here is swallowed and reported as "nothing to journal": every
        // one of them was already checked at `begin_weight_stroke`, and a stroke
        // is not the place to learn that the mesh was unbound between pointer-down
        // and pointer-up. `settle_drag` is where a real refusal surfaces.
        inf_dcc::paint_weights(
            mesh,
            self.brush.joint,
            self.brush.mode,
            &dabs,
            self.brush.radius,
            self.brush.strength,
            self.brush.falloff,
        )
        .ok()
        .flatten()
    }
}

/// **Begin a weight-paint stroke.** The [`begin_stroke`] shape, on the skin
/// channel: three answers, and every refusal decided in Ring 1 where a test can
/// reach it.
///
/// * `Err(reason)` — the radius is below the floor, the mesh carries no skin, or
///   the joint is not one the binding has. All three are sentences an author
///   needs to read, and the last two are the ones a fresh mesh hits first.
/// * `Ok(None)` — the pointer missed the model; the gesture becomes an orbit.
/// * `Ok(Some(stroke))` — painting.
pub fn begin_weight_stroke(
    mesh: &Mesh,
    proj: &Projector,
    px: f32,
    py: f32,
    brush: WeightBrush,
) -> Result<Option<WeightStrokeInFlight>, String> {
    let Some(binding) = mesh.skin_binding() else {
        return Err(
            "this mesh carries no skin, so there is no influence to paint; bind it \
             to a skeleton first"
                .to_string(),
        );
    };
    if brush.joint as u32 >= binding.joints {
        return Err(format!(
            "joint {} is out of range: this mesh is bound to a skeleton with {} \
             joints",
            brush.joint, binding.joints
        ));
    }
    if !(brush.radius.is_finite() && brush.radius >= inf_dcc::MIN_BRUSH_RADIUS_M) {
        return Err(format!(
            "a brush radius of {} m is below the {} m floor; a brush that small \
             cannot reach a second vertex on anything modelled in metres",
            brush.radius,
            inf_dcc::MIN_BRUSH_RADIUS_M
        ));
    }
    let Some((face, hit)) = pick_surface(mesh, proj, px, py) else {
        return Ok(None);
    };
    Ok(Some(WeightStrokeInFlight {
        brush,
        path: vec![hit],
        last_normal: inf_dcc::face_normal(mesh, face).unwrap_or(DVec3::Y),
    }))
}

impl PendingDrag {
    /// The ops this drag would commit, given the document it belongs to.
    ///
    /// Empty when the gesture did nothing — a click with no drag, or a stroke
    /// whose pointer never found the surface. The caller journals nothing for an
    /// empty list, which is what makes "click on the model in sculpt mode" not
    /// produce an undo step.
    pub fn ops(&self, mesh: &Mesh, selection: &SelectionSet, mode: SelectMode) -> Vec<Op> {
        match self {
            PendingDrag::Stroke(s) => s.op().into_iter().collect(),
            PendingDrag::Weights(w) => w.op(mesh).into_iter().collect(),
            PendingDrag::Gizmo(g) => {
                if g.xform.is_identity() {
                    Vec::new()
                } else {
                    transform_ops(mesh, selection, mode, g.pivot, g.xform, g.soft)
                }
            }
        }
    }

    /// The mesh this drag would produce, for the live preview — **a scratch copy,
    /// never the document's**.
    ///
    /// Refusals here are swallowed on purpose: a preview frame is not the place
    /// to learn that an in-progress drag has overflowed, and the same ops go
    /// through the journal on pointer-up where the refusal *is* reported. What
    /// comes back on a refusal is the mesh as it stands, which is the honest
    /// picture of "nothing has been committed".
    pub fn scratch(
        &self,
        session: &inf_dcc::MeshSession,
        selection: &SelectionSet,
        mode: SelectMode,
    ) -> Mesh {
        let mut mesh = session.mesh().clone();
        for op in self.ops(session.mesh(), selection, mode) {
            if inf_dcc::ops::apply(&mut mesh, &op).is_err() {
                return session.mesh().clone();
            }
        }
        mesh
    }
}

/// **Settle a drag that is still in flight** — the P21.3 orphan-settler doctrine
/// applied to a gesture rather than to a terrain edit.
///
/// A pointer-up is not guaranteed to arrive: the panel can close, the tool can
/// change, the document can be saved or undone, or a detached window can lose
/// pointer capture. Every one of those doors calls this, and the ruling is
/// **settle, not abandon**: the dabs are the author's work, they are already
/// visible in the preview, and a gesture that silently evaporates is the failure
/// this codebase keeps paying for.
///
/// The **one exception is closing the document**, and it is not this function —
/// `dcc_close` drops the whole session in the same call, so committing an op into
/// a journal that is about to be freed is a write nobody can undo, save or see.
/// That door abandons, deliberately, and says so.
///
/// Returns the refusal text if the settle was attempted and refused; `None` when
/// there was nothing pending or it applied cleanly. **The selection is carried,
/// never dropped**: every op a drag produces is `op_preserves_ids`, so insisting
/// on a drop would deselect the face the author just sculpted.
pub fn settle_drag(
    session: &mut inf_dcc::MeshSession,
    selection: &mut SelectionSet,
    mode: SelectMode,
    pending: Option<PendingDrag>,
) -> Option<String> {
    let pending = pending?;
    let ops = pending.ops(session.mesh(), selection, mode);
    if ops.is_empty() {
        return None;
    }
    let mut refusal = None;
    for op in ops {
        debug_assert!(
            inf_dcc::op_preserves_ids(&op),
            "a drag must only produce ops that keep ids, or the selection has to drop"
        );
        match session.apply(op) {
            Ok(_) => selection.carry(session.generation(), session.mesh()),
            Err(e) => {
                refusal = Some(e.to_string());
                break;
            }
        }
    }
    refusal
}

// ── the save (M1) ──────────────────────────────────────────────────────────

/// What a save did.
#[derive(Debug, Clone)]
pub struct SaveOutcome {
    pub export: inf_dcc::ExportReport,
    pub vmesh: crate::assets::vmesh::VmeshDerivation,
}

/// Why a save could not finish, **and what the disk holds because of it**.
///
/// The error text is part of the contract, not decoration: a save is the one
/// operation whose failure leaves state behind, and an author who is told
/// "save failed" and nothing else does not know whether to try again or to
/// reopen.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SaveError {
    #[error("could not write {name}: {message}. Nothing on disk changed.")]
    Write { name: String, message: String },
    /// The payload landed and the derived `.inf_vmesh` could not be rebuilt.
    ///
    /// **The stale DAG is gone from disk** — verified, not assumed — so nothing
    /// can draw the old geometry. See [`save_mesh_session`] for why that is the
    /// chosen failure mode.
    #[error(
        "{name} was saved, but its meshlet DAG could not be rebuilt: {message}. \
         The stale DAG has been removed, so the mesh will draw as a placeholder \
         until the project is reopened."
    )]
    Derived { name: String, message: String },
    /// Something is **still on disk** at the derived id: the delete failed (a
    /// mapped file on Windows), or the artifact sitting there is *authored*
    /// rather than derived and is not this code's to remove.
    ///
    /// The only state in which something can still draw the previous geometry,
    /// and the reason the removal is checked against the filesystem: the
    /// database's own answer is unconditionally `Ok`, so a version of this that
    /// trusted it reported "removed, it will draw as a placeholder" over a file
    /// that was still there and still being drawn.
    #[error(
        "{name} was saved, but its meshlet DAG could not be rebuilt ({message}) \
         and the stale DAG could not be removed either. Until the project is \
         reopened, this mesh may draw its PREVIOUS geometry."
    )]
    Torn { name: String, message: String },
}

/// Write a kernel mesh back to its asset, and rebuild its derived `.inf_vmesh`.
///
/// **This is the product path.** It is Ring 1 rather than Ring 2 for the reason
/// this module exists at all: a `#[tauri::command]` cannot be driven from a test,
/// so a gate that "proves the save" by inlining the same two calls proves the
/// *pattern* and never the product — which is exactly what happened, and why
/// dropping `ensure_vmesh` from the command failed nothing.
///
/// # The failure contract, decided rather than assumed
///
/// `AssetProject` has a lock, not a transaction: `rewrite_payload` can complete
/// and `ensure_vmesh` can then fail, and the previous code claimed "no window in
/// which the two disagree" while leaving them disagreeing **permanently** — new
/// payload on disk, stale DAG beside it, the watcher re-keying on the new content
/// hash and the viewport drawing the old surface with complete confidence. That
/// is the precise failure the P23.1 memo opens by naming.
///
/// Three orderings were available and this is why it is this one:
///
/// * *Derive first, then write* would need the DAG built from a payload the
///   database has not seen (the planner reads the db to compute the source hash),
///   so it is not reachable without reshaping `assets::vmesh`.
/// * *Write, derive, and on failure leave it* is the status quo, and its bad
///   state is the one bad state that is invisible.
/// * **Write, derive, and on failure REMOVE the stale DAG** — this. The pair is
///   then always either (new payload, new DAG) or (new payload, no DAG), and "no
///   DAG" is a state the renderer already handles: `resolve_vgeom` misses and the
///   entity falls back to a placeholder. Visibly wrong beats confidently wrong,
///   and the next `ensure_vmesh` — the save the author will now retry, or the
///   project-open sweep — repairs it.
///
/// If the removal *also* fails, that is [`SaveError::Torn`], and the message says
/// exactly what disk holds. It is the only state in which something can still
/// draw the previous geometry, and it is named rather than hidden.
///
/// **"Fails" means the file is still there**, checked against the filesystem.
/// `AssetProject::delete` reports success unconditionally, so a version of this
/// that believed it would have told the author the DAG was gone while the
/// renderer went on drawing from it — which is the failure this whole function
/// exists to prevent, delivered by its own error path.
pub fn save_mesh_session(
    project: &mut crate::assets::AssetProject,
    asset: inf_asset::AssetId,
    mesh: &Mesh,
) -> Result<SaveOutcome, SaveError> {
    let name = project
        .db()
        .get(asset)
        .map(|e| e.name.clone())
        .unwrap_or_else(|| asset.to_string());
    let (payload, export) = inf_dcc::to_mesh_asset(mesh, &inf_dcc::ExportOptions::default());
    project
        .rewrite_payload(asset, &payload, vec![])
        .map_err(|e| SaveError::Write {
            name: name.clone(),
            message: e.to_string(),
        })?;
    match crate::assets::vmesh::ensure_vmesh(project, asset) {
        Ok(vmesh) => {
            // **`Skipped` is the third state, and it was the leak.** A mesh
            // edited down below the virtualization threshold derives nothing —
            // correctly, there is nothing to virtualize — and the DAG describing
            // the mesh it USED to be stays on disk. `resolve_vgeom` computes the
            // derived id rather than looking it up, so it finds that artifact and
            // draws the old geometry with total confidence: the exact pair this
            // function exists to make unreachable, arriving through the one
            // outcome that is not an error.
            //
            // Its removal is checked like any other. It used to be called for
            // effect and its answer dropped, which meant no path in this function
            // could observe a removal that failed.
            if matches!(vmesh, crate::assets::vmesh::VmeshDerivation::Skipped)
                && !crate::assets::vmesh::remove_derived_vmesh(project, asset)
            {
                return Err(SaveError::Torn {
                    name,
                    message: "this mesh no longer has enough geometry to \
                              virtualize, so its meshlet DAG is obsolete"
                        .into(),
                });
            }
            Ok(SaveOutcome { export, vmesh })
        }
        Err(e) => {
            let message = e.to_string();
            // Nothing may be left that draws the geometry we just replaced, and
            // `remove_derived_vmesh` answers with the filesystem rather than the
            // database — so this really is "is something still there", not "did
            // the bookkeeping return Ok" (which it always did).
            if crate::assets::vmesh::remove_derived_vmesh(project, asset) {
                Err(SaveError::Derived { name, message })
            } else {
                Err(SaveError::Torn { name, message })
            }
        }
    }
}

// ── the preview's geometry cache (M6) ──────────────────────────────────────

/// The tessellation and the mesh snapshot a preview frame needs, held against the
/// journal generation that produced them.
///
/// The module docs above promise the tessellation runs "only when the mesh moves
/// — never on a camera orbit", and until this type existed that was false: every
/// orbit frame re-ran the exporter (ear clipping, a tangent solve, a corner
/// intern) and cloned the whole mesh for the overlay, thirty times a second, for
/// a camera that moved 144 bytes. `GenCache`'s rule, one layer up: the key is
/// what actually invalidates, and here that is the generation stamp and nothing
/// else.
/// **What a live drag costs, measured** (P23.5). See
/// [`PreviewCache::get_with_pending`] for the numbers and the ruling.
#[derive(Default)]
pub struct PreviewCache {
    stamp: Option<u64>,
    geometry: Option<std::sync::Arc<EditGeometry>>,
    mesh: Option<std::sync::Arc<Mesh>>,
    tessellations: u64,
    /// The scratch half: the generation and the drag "shape" the uncommitted
    /// frame was built for.
    scratch_key: Option<(u64, usize)>,
    scratch_geometry: Option<std::sync::Arc<EditGeometry>>,
    scratch_mesh: Option<std::sync::Arc<Mesh>>,
    /// How many scratch frames fell back to a FULL tessellation (Wave D).
    scratch_tessellations: u64,
    /// How many scratch frames were built at all, fast path or slow.
    scratch_frames: u64,
    /// **The identity of the geometry the last get returned** — the key a
    /// vertex-buffer upload may be skipped on. See [`PreviewCache::upload_stamp`].
    upload_stamp: u64,
}

/// Fold a preview key into one integer, so the renderer's upload can be skipped
/// on equality without the caller carrying a pair.
///
/// FNV-1a over the two halves rather than a shift-and-or, because `step` is a
/// hash itself for a gizmo drag (see [`drag_step`]) and a packing would collide
/// on its high bits.
///
/// **`| 1`, because zero is not a stamp** (round 3).
/// `PreviewSession::geometry_stamp`'s own doc says `0` **is the built-in
/// sphere** — the value the session holds before anything has been uploaded —
/// and `set_geometry` skips the upload when the stamp it is handed equals the
/// one it holds. A fold that landed on zero would therefore leave the Model
/// Editor drawing the material preview's sphere until the next edit, which is
/// R2.F4's symptom arriving by a different route. One bit of a 64-bit hash buys
/// the guarantee outright, and it is cheaper than making the sentinel an
/// `Option` at a public signature four other callers pass through.
fn fold_key(generation: u64, step: u64) -> u64 {
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    for x in [generation, step] {
        acc ^= x;
        acc = acc.wrapping_mul(0x1000_0000_01b3);
    }
    acc | 1
}

/// The step half of a committed (no drag in flight) key. `u64::MAX` because a
/// real step is a path length or a `usize` hash and neither is asked to be
/// distinct from this — the fold is what makes them so.
const COMMITTED_STEP: u64 = u64::MAX;

impl PreviewCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// **The stamp a caller must give the renderer** for the geometry the last
    /// [`get`](PreviewCache::get) / [`get_with_pending`](PreviewCache::get_with_pending)
    /// returned (round-2 finding R2.F4).
    ///
    /// `PreviewSession::set_geometry` early-returns when the stamp it is handed
    /// equals the one it holds — that is the P23.2a warm path and it is what
    /// makes an orbit free. The Model Editor's command layer passed
    /// `session.generation()`, which is exactly the number an **uncommitted
    /// drag deliberately does not move**: this cache keys its scratch on
    /// `(generation, step)` *because* the generation cannot see a live stroke.
    /// So every sculpt / weight-paint / gizmo drag frame re-tessellated a fresh
    /// scratch mesh and then skipped the upload of it. The wireframe and brush
    /// ring (composited CPU-side, from the same geometry) tracked the pointer
    /// while the shaded surface stayed frozen at the pre-drag mesh until
    /// `dcc_drag_end` moved the generation.
    ///
    /// Reading it from the cache rather than recomputing it at the call site is
    /// the point: the cache is the only thing that knows which of its two slots
    /// it just served, and a second derivation of that is the class of bug this
    /// tree's laws forbid.
    pub fn upload_stamp(&self) -> u64 {
        self.upload_stamp
    }

    /// How many times the tessellator has actually run — the cache's observable
    /// state, and what the orbit gate asserts on. (`Thumbnailer::size` and
    /// `MeshSession::checkpoint_count` are the same idea: a number a test can
    /// hold the implementation to.)
    pub fn tessellations(&self) -> u64 {
        self.tessellations
    }

    /// The geometry and the mesh for `session`, tessellating only if the journal
    /// has moved since the last call.
    pub fn get(
        &mut self,
        session: &inf_dcc::MeshSession,
    ) -> (std::sync::Arc<EditGeometry>, std::sync::Arc<Mesh>) {
        let stamp = session.generation();
        if self.stamp != Some(stamp) || self.geometry.is_none() || self.mesh.is_none() {
            self.stamp = Some(stamp);
            self.geometry = Some(std::sync::Arc::new(tessellate(session.mesh())));
            self.mesh = Some(std::sync::Arc::new(session.mesh().clone()));
            self.tessellations += 1;
        }
        self.upload_stamp = fold_key(stamp, COMMITTED_STEP);
        (
            self.geometry.clone().expect("just filled"),
            self.mesh.clone().expect("just filled"),
        )
    }

    /// How many scratch frames fell back to a **full** tessellation.
    ///
    /// Wave D split this from [`PreviewCache::scratch_frames`], and the split is
    /// the measurement: since the drag path displaces the committed geometry in
    /// place, a working fast path leaves this at **zero** however many frames a
    /// drag produces. A gate that read only "how many frames were built" could
    /// not tell the two apart, which is exactly how a silently-disabled
    /// optimization survives.
    pub fn scratch_tessellations(&self) -> u64 {
        self.scratch_tessellations
    }

    /// How many uncommitted frames were built at all — the number the
    /// "an orbit during a drag costs nothing extra" claim is about.
    pub fn scratch_frames(&self) -> u64 {
        self.scratch_frames
    }

    /// The geometry and mesh a frame should draw, given a drag that has not been
    /// committed yet.
    ///
    /// # Wave D: the drag frame DISPLACES the committed one
    ///
    /// The P23 ledger's named next lever, in its own words — *"displacing the
    /// cached vertex buffer in place rather than re-running the exporter, which
    /// needs the writer to expose its corner→vertex map"* — and the carried
    /// remainder it retires is the **~100 000-vertex live-sculpt ceiling**.
    ///
    /// Every op a drag can be holding is id-preserving (a stroke, a gizmo
    /// transform, a weight table), so the scratch mesh has the committed mesh's
    /// topology and the full re-export was re-exporting the same connectivity.
    /// [`displace`] writes the positions and asks
    /// [`inf_dcc::corner_normal`] — the *writer's own rule* — for the normals,
    /// on the corners [`inf_dcc::to_mesh_asset_sourced`] hands back.
    ///
    /// **Measured on this machine, debug build**, by
    /// `live_drag_frame_cost_is_measured` (the same test and the same profile as
    /// the P23.5 figures below it, so the columns are comparable):
    ///
    /// | mesh | full `tessellate` | **displaced drag frame** |
    /// | --- | --- | --- |
    /// | 26 v / 48 tri | 0.21 ms | **0.01 ms** |
    /// | 1 538 v / 3 072 tri | 8.53 ms | **0.29 ms** |
    /// | 24 578 v / 49 152 tri | 141.98 ms | **4.35 ms** |
    ///
    /// **29–33× at both real sizes, and the scaling is linear in vertices rather
    /// than in the exporter's `BTreeMap` interning.** At 24 578 vertices a drag
    /// frame is 4.35 ms in a debug build — inside an interactive rate with the
    /// P23.2a render (0.09 ms) and encode (0.34 ms) on top — where the same frame
    /// used to cost 142 ms, i.e. 7 fps. Extrapolating the linear term, a hundred
    /// thousand vertices is ~18 ms debug and several times less in release, so
    /// the ceiling the P23.5 docs stated is **retired with a number rather than
    /// with an argument**.
    ///
    /// What it is NOT: a claim that the *committed* frame got faster. It did
    /// not — a pointer-up still exports, once, and the save always exports. What
    /// changed is that the frames *between* pointer-down and pointer-up stopped
    /// paying for it. `scratch_tessellations` counts the fallbacks and is
    /// asserted at **zero**, so a fast path that silently stopped working is a
    /// failing test rather than a slow editor.
    ///
    /// # The side channel, and why it used to be a full re-tessellation
    ///
    /// [`PreviewCache::get`]'s key is the journal generation, and an uncommitted
    /// drag *does not move it* — that is the whole point of not journalling until
    /// pointer-up. So a live drag needs a second channel, and v1's is the honest
    /// one: apply the pending ops to a **clone**, tessellate that, and key the
    /// result on `(generation, path length / drag step)` so a frame whose drag
    /// has not changed — an orbit mid-stroke, a re-render after a resize — is
    /// still free.
    ///
    /// **Measured on this machine**, by `live_drag_frame_cost_is_measured`, in a
    /// **debug** build (the profile CI and the editor's dev runs use; a release
    /// build is several times faster and the ratio is what matters here):
    ///
    /// | mesh | `tessellate` | clone + 1 dab + tessellate |
    /// | --- | --- | --- |
    /// | subdivided cube, 26 v / 48 tri | 0.17 ms | 0.11 ms |
    /// | subdivided cube, 1 538 v / 3 072 tri | 8.6 ms | 9.1 ms |
    ///
    /// **The clone and the stroke are free; the tessellation is the whole cost**
    /// — the two columns agree to within noise at both sizes, which is the useful
    /// finding, and it is exactly why Wave D attacked the tessellation and not
    /// the clone.
    ///
    /// Against the P23.2a budget — 0.09 ms to render at 256² and ~0.34 ms to
    /// encode — a drag frame on a small model is comfortable and a **1.5 k-vertex
    /// model is already the dominant cost at ~9 ms**, i.e. about 30 fps in a
    /// debug build and fine in release. Stated rather than hidden: **this path
    /// will not hold an interactive rate on a model of a hundred thousand
    /// vertices.** That was true when it was written and is **retired** by the
    /// section above — the lever it names is the one that was pulled. What is
    /// still *not* on the table is displacing on the GPU: the CPU picker could
    /// not see it, and the panel would go back to highlighting one thing and
    /// hitting another.
    pub fn get_with_pending(
        &mut self,
        session: &inf_dcc::MeshSession,
        selection: &SelectionSet,
        mode: SelectMode,
        pending: Option<&PendingDrag>,
    ) -> (std::sync::Arc<EditGeometry>, std::sync::Arc<Mesh>) {
        let Some(pending) = pending else {
            // Nothing in flight: the committed path, and the scratch is released
            // so a finished drag does not hold a second copy of the mesh for the
            // rest of the session.
            self.scratch_key = None;
            self.scratch_geometry = None;
            self.scratch_mesh = None;
            return self.get(session);
        };
        let step = match pending {
            PendingDrag::Stroke(s) => s.path.len(),
            // A weight stroke's shape is its path length, exactly as a sculpt
            // stroke's — it changes no position, so the scratch mesh it produces
            // is only ever re-tessellated when the stroke actually grew.
            PendingDrag::Weights(w) => w.path.len(),
            // A gizmo's "shape" is its current transform; hashing the bits keeps
            // the key one integer without pretending a float is a step count.
            PendingDrag::Gizmo(g) => drag_step(&g.xform),
        };
        let key = (session.generation(), step ^ selection_step(selection, mode));
        if self.scratch_key != Some(key)
            || self.scratch_geometry.is_none()
            || self.scratch_mesh.is_none()
        {
            let mesh = pending.scratch(session, selection, mode);
            // **The committed tessellation is the base, and the drag DISPLACES
            // it** (Wave D) — the P23 ledger's named next lever. Every op a drag
            // can be holding is id-preserving (a stroke, a gizmo transform, a
            // weight table), so the scratch mesh has the committed mesh's
            // topology and the whole re-export was re-exporting the same
            // connectivity. `displace` writes positions and re-derives normals
            // through the writer's own `corner_normal` instead.
            //
            // `self.get(session)` first, so the committed geometry exists and
            // this is a hit on the cache the panel already keeps. It also sets
            // `upload_stamp`, which the line below then overwrites with the
            // scratch key — the order matters and is why the call is here rather
            // than at the top of the function.
            let (committed, _) = self.get(session);
            let geo = match displace(&committed, &mesh) {
                Some(g) => g,
                None => {
                    // No source map, or the drag moved something off the map.
                    // A full tessellation is always correct; this is the
                    // fallback, and `scratch_tessellations` counts only it so a
                    // gate can tell the fast path from the slow one.
                    self.scratch_tessellations += 1;
                    tessellate(&mesh)
                }
            };
            self.scratch_key = Some(key);
            self.scratch_geometry = Some(std::sync::Arc::new(geo));
            self.scratch_mesh = Some(std::sync::Arc::new(mesh));
            self.scratch_frames += 1;
        }
        self.upload_stamp = fold_key(key.0, key.1 as u64);
        (
            self.scratch_geometry.clone().expect("just filled"),
            self.scratch_mesh.clone().expect("just filled"),
        )
    }
}

/// The **selection and the mode**, reduced to one integer for the scratch key
/// (round 3 — the carried half of R2.F4).
///
/// `PendingDrag::scratch` takes `(session, selection, mode)` and the key took
/// `(generation, step)`, so two of its three inputs were outside it. A gizmo
/// drag's ops are built from the selected ids: change the selection — or switch
/// Vert→Edge→Face, which changes *which* ids the same op set reads — while the
/// pointer is down and the drag's `step` hash does not move, because the
/// transform did not. The cache then serves the mesh built for the OLD
/// selection, and the shaded surface shows the previous elements moving while
/// the CPU-composited overlay draws the new ones.
///
/// Ids rather than a count: `{v1, v2}` and `{v3, v4}` are the same size and a
/// different drag. `BTreeSet` iteration is ordered, so the hash is a function of
/// the set and not of the order it was assembled in.
fn selection_step(selection: &SelectionSet, mode: SelectMode) -> usize {
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        acc ^= x;
        acc = acc.wrapping_mul(0x1000_0000_01b3);
    };
    mix(match mode {
        SelectMode::Vert => 1,
        SelectMode::Edge => 2,
        SelectMode::Face => 3,
    });
    mix(selection.generation());
    for v in selection.verts() {
        mix(v.index() as u64);
    }
    mix(0xffff_ffff_ffff_fff1);
    for e in selection.edges() {
        mix(e.index() as u64);
    }
    mix(0xffff_ffff_ffff_fff2);
    for f in selection.faces() {
        mix(f.index() as u64);
    }
    acc as usize
}

/// A gizmo transform reduced to one integer, so an unchanged drag is a cache hit.
///
/// Deliberately a **hash of the exact bits**, not a quantization: two transforms
/// that differ in the last ulp are different pictures, and a key that called them
/// equal would freeze the preview on a slow drag.
fn drag_step(xform: &VertTransform) -> usize {
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: f64| {
        acc ^= x.to_bits();
        acc = acc.wrapping_mul(0x1000_0000_01b3);
    };
    match xform {
        VertTransform::Translate(d) => {
            mix(1.0);
            mix(d.x);
            mix(d.y);
            mix(d.z);
        }
        VertTransform::Rotate { axis, radians } => {
            mix(2.0);
            mix(axis.x);
            mix(axis.y);
            mix(axis.z);
            mix(*radians);
        }
        VertTransform::Scale(f) => {
            mix(3.0);
            mix(f.x);
            mix(f.y);
            mix(f.z);
        }
    }
    acc as usize
}

// ── drag-and-drop modularity, v1 ───────────────────────────────────────────

/// **The tessellated mesh as a triangle soup**, for the auto-fit BVH (P24.3).
///
/// The hop `skel_fit_to_mesh` needed and that was missing: `inf_dcc::fit_template`
/// takes a `Bvh`, `Bvh::new` takes `Vec<Tri>`, and the only route from a
/// `CanonicalMesh` to triangles is [`tessellate`] — which yields the **drawn**
/// buffers (`verts` + a `u32` index list), because they are the same triangles
/// `to_mesh_asset` writes. So the conversion is a walk of `indices` in threes.
///
/// **Widened to `f64` here and not before**: `MeshVertex::position` is `f32`
/// because it is a render vertex, and the BVH is an `f64` spatial structure
/// (architecture rule 3). One widening, at the one seam, rather than an `f64`
/// vertex format nothing else wants.
///
/// A trailing partial triangle is **dropped** rather than padded: an index list
/// whose length is not a multiple of three is a malformed mesh, and inventing a
/// vertex for it would put a triangle in the hierarchy that the mesh does not
/// have. `chunks_exact` says exactly that.
///
/// # NOT for a mesh this engine GENERATED (SK1b, and its audit)
///
/// The widening above is exact for a mesh that **arrived** as an asset: its
/// kernel positions are widened `f32` and the round trip through `tessellate`
/// gives the same bits back. It is not exact for a mesh `inf_dcc` *generated*,
/// whose kernel positions are `f64` — and a visibility ray cast from an exact
/// kernel vertex then starts up to an ulp **outside** the surface this soup
/// describes and hits its own face at `t ~ 0`.
///
/// Measured on the SK1b starter body, 795 generated vertices: **349** unreached
/// through this soup against **35** through [`inf_dcc::mesh_soup`], which is the
/// same triangles in `f64`. Ten times as much of a character the weight solver
/// believes it cannot see, from a rounding step in the oracle.
/// `inf_dcc`'s `the_narrowed_oracle_cannot_see_a_third_of_a_generated_body` is
/// the arm, and SK1b's fifth decision is the sentence: *a visibility oracle must
/// be built in the space its rays are cast in.*
///
/// So: an author's imported model, yes ([`fit_bvh`](crate::character::fit_bvh),
/// `skinned_copy`). Anything `inf_dcc::body_mesh` or a grammar bake produced,
/// [`inf_dcc::mesh_soup`].
pub fn triangle_soup(geo: &EditGeometry) -> Vec<inf_dcc::Tri> {
    geo.indices
        .chunks_exact(3)
        .filter_map(|t| {
            let p = |i: u32| -> Option<DVec3> {
                geo.verts.get(i as usize).map(|v| {
                    DVec3::new(
                        v.position[0] as f64,
                        v.position[1] as f64,
                        v.position[2] as f64,
                    )
                })
            };
            Some(inf_dcc::Tri {
                a: p(t[0])?,
                b: p(t[1])?,
                c: p(t[2])?,
            })
        })
        .collect()
}

/// What a rig-carrying [`merge_into`] did.
///
/// Reported rather than inferred, because every number here is a thing that can
/// silently not happen: a merge that dropped the incoming weights, or landed
/// every face on slot `None`, looks exactly like a merge that had none to carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MergeReport {
    /// Faces added.
    pub faces: usize,
    /// Vertices added.
    pub verts: usize,
    /// Material slots **appended** to this document's table (never reused ones).
    pub slots_appended: usize,
    /// Incoming slots that matched an existing name and were **reused** rather
    /// than appended.
    pub slots_reused: usize,
    /// Vertices whose incoming weights were re-indexed by `joint_offset` and
    /// written onto the merged geometry.
    pub verts_reweighted: usize,
}

/// Append another mesh into this document as a **new connected component**,
/// offset by `offset` metres — the dcc-vision seed: assembling a prop, or a
/// character, out of kit pieces without leaving the Model Editor.
///
/// **What it does**: applies an `AddVertex` per incoming vertex and an `AddFace`
/// per incoming face, so the whole merge is ordinary journal entries — undo peels
/// it off, replay reproduces it byte for byte, and nothing special-cases it
/// anywhere. It has to be done against a live session rather than precomputed as
/// a `Vec<Op>`, because the ids the target mints come from its arena's free list;
/// that is the same property that makes replay work, and it is why there is no
/// pure `merge_ops` beside this.
///
/// # The rig comes with it (P24.3)
///
/// Two of P23.4's three stated non-goals are now closed, and both through ops
/// rather than through a mutation the journal cannot see:
///
/// * **Material slots.** `Op::AddMaterialSlots` (discriminant 30) appends the
///   incoming names this document does not already have, and each incoming slot
///   index is mapped to its name's index here. A name that already exists is
///   **reused**, not duplicated: two parts that both say "Default" are two parts
///   whose author meant one material. The kernel has no opinion about that (it
///   allows duplicates); the policy is here, where the author's intent is.
/// * **Skin.** If the incoming mesh is bound, its weights are re-indexed by the
///   `joint_offset` the caller got from `inf_anim::merge_skeletons` and written
///   with one `Op::AssignWeights`, after an `Op::BindSkin` widens this document's
///   binding to the merged skeleton's joint count. **The base document's weights
///   are never touched** — that is the append-only law, one level up from the
///   skeleton: torso joints keep their indices, so the torso's own weight table
///   and any IK chain authored against it survive by construction.
///
/// The third non-goal stands: **no welding or snapping** (the dropped part is a
/// separate shell that happens to be nearby; joining it is the author's
/// `MergeVerts`), **no instancing** (geometry is copied — that is the difference
/// between assembling and referencing, and referencing is what the *scene* is
/// for), and **no refusal recovery** (a merge that fails partway has journalled
/// what it applied; the caller undoes it).
///
/// # `rig` is `None` for a rigid merge
///
/// Passing `None` merges geometry and slots and leaves every new vertex rigid,
/// which is what a prop kit wants and what every pre-P24.3 caller got. Passing
/// `Some` is the character path, and the caller is expected to have merged the
/// *skeletons* first — this function cannot, because a `Mesh` holds a joint
/// count and no joint list.
pub fn merge_into(
    session: &mut inf_dcc::MeshSession,
    incoming: &Mesh,
    offset: DVec3,
    rig: Option<MergeRig>,
) -> Result<MergeReport, inf_dcc::OpError> {
    let mut report = MergeReport::default();

    // ── 1. the slot table, before any face names an index into it ──────────
    //
    // First, because `Op::AddFace` runs `check_slot` and an index past the end
    // is a refusal — a merge that added its faces first would have to add them
    // all on `None` and repaint afterwards, which is two journal entries per
    // face for the same result.
    let existing: Vec<String> = session.mesh().material_slots().to_vec();
    let mut slot_map: Vec<u32> = Vec::with_capacity(incoming.material_slots().len());
    let mut to_append: Vec<String> = Vec::new();
    for name in incoming.material_slots() {
        match existing.iter().position(|e| e == name) {
            Some(i) => {
                slot_map.push(i as u32);
                report.slots_reused += 1;
            }
            None => match to_append.iter().position(|e| e == name) {
                // The incoming table can itself repeat a name; appending it
                // twice would make two slots nothing can tell apart.
                Some(i) => slot_map.push((existing.len() + i) as u32),
                None => {
                    slot_map.push((existing.len() + to_append.len()) as u32);
                    to_append.push(name.clone());
                }
            },
        }
    }
    if !to_append.is_empty() {
        report.slots_appended = to_append.len();
        session.apply(Op::AddMaterialSlots { names: to_append })?;
    }

    // ── 2. the geometry ────────────────────────────────────────────────────
    let sources: Vec<VertId> = incoming.vert_ids().collect();
    let mut minted: std::collections::BTreeMap<VertId, VertId> = std::collections::BTreeMap::new();
    for &v in &sources {
        let Some(p) = incoming.position(v) else {
            continue;
        };
        let q = p + offset;
        if !q.is_finite() {
            continue;
        }
        let out = session.apply(Op::AddVertex {
            position: q.to_array(),
        })?;
        minted.insert(v, out.verts[0]);
    }
    report.verts = minted.len();

    for f in incoming.face_ids() {
        let Some(loop_halfs) = incoming.face_loop(f) else {
            continue;
        };
        let mut verts = Vec::with_capacity(loop_halfs.len());
        let mut corners = Vec::with_capacity(loop_halfs.len());
        let mut ok = true;
        for h in loop_halfs {
            match incoming.origin(h).and_then(|v| minted.get(&v).copied()) {
                Some(v) => verts.push(v),
                None => {
                    ok = false;
                    break;
                }
            }
            corners.push(inf_dcc::CornerData {
                uv: incoming.corner_uv(h).unwrap_or_default(),
                normal: None,
            });
        }
        if !ok || verts.len() < 3 {
            continue;
        }
        // The incoming face's slot, through the map built above. A face with no
        // slot keeps none; a slot the incoming table does not actually have (it
        // cannot, but the map is indexed) also keeps none rather than guessing.
        let slot = incoming
            .face_slot(f)
            .flatten()
            .and_then(|s| slot_map.get(s as usize).copied());
        session.apply(Op::AddFace {
            verts,
            corners,
            slot,
        })?;
        report.faces += 1;
    }

    // ── 3. the rig ─────────────────────────────────────────────────────────
    let Some(rig) = rig else {
        return Ok(report);
    };
    // Widen (or establish) this document's binding FIRST: `Op::AssignWeights`
    // refuses on an unbound mesh (`OpError::NotSkinned`) and refuses a joint
    // index past the binding's count, and the incoming indices are about to be
    // shifted past the base skeleton's.
    session.apply(Op::BindSkin {
        skeleton: rig.skeleton,
        joints: rig.joints,
    })?;
    let mut weights: Vec<(VertId, inf_dcc::VertWeights)> = Vec::new();
    for (&src, &dst) in &minted {
        let Some(w) = incoming.vert_weights(src) else {
            continue;
        };
        if w.is_rigid() && rig.joint_offset == 0 {
            // Nothing to say: an unweighted vertex on an unshifted rig is
            // already what `AddVertex` produced.
            continue;
        }
        let mut out = w;
        for (j, weight) in out.joints.iter_mut().zip(&w.weights) {
            // Zero-weight slots are parked on joint 0 by `VertWeights::normalized`
            // and carry no influence; shifting them would move them off 0 for no
            // reason and cost a range check against the merged skeleton.
            if *weight != 0.0 {
                *j += rig.joint_offset;
            }
        }
        weights.push((dst, out));
    }
    if !weights.is_empty() {
        // `AssignWeights` wants them sorted by vertex id; a `BTreeMap` walk is
        // sorted by the SOURCE id, and the minted ids come from a free list.
        weights.sort_by_key(|(v, _)| *v);
        report.verts_reweighted = weights.len();
        session.apply(Op::AssignWeights { weights })?;
    }
    Ok(report)
}

/// What [`merge_into`] needs to carry a skin, and nothing more.
///
/// Produced by the caller from `inf_anim::merge_skeletons`: `joint_offset` is
/// that function's own output, and `joints` is the merged skeleton's length. Kept
/// as a small POD rather than taking a `SkeletonAsset` because Ring 1 is where
/// the two crates meet and `inf_dcc` must not learn what a skeleton is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeRig {
    /// The merged skeleton's `.inf_skel` GUID as raw bytes, or `None` when the
    /// document does not know which skeleton it is bound to (the ordinary state
    /// for a mesh opened from a bare `.inf_mesh` — see `inf_dcc::SkinBinding`).
    pub skeleton: Option<[u8; 16]>,
    /// The merged skeleton's joint count.
    pub joints: u32,
    /// What to add to an incoming joint index — `inf_anim::SkeletonMerge::joint_offset`.
    pub joint_offset: u16,
}

/// **Mirror the mesh AND its joints** across an axis-aligned plane (P24.3).
///
/// `Op::Mirror` alone copies each vertex's weights verbatim, so a mirrored left
/// arm is still weighted to the left arm's bones. Fixing that needs to pair
/// `upper_arm_l` with `upper_arm_r`, which needs joint **names** — and
/// `inf_dcc::SkinBinding` carries a GUID and a joint **count**, deliberately, so
/// the kernel cannot do it. (Putting the names on the binding would change
/// `Mesh`'s shape and therefore `SessionSave`'s, which is a schema bump for a
/// convenience.)
///
/// So the pairing happens **here**, as a composition of two ops rather than as a
/// new one:
///
///  1. `Op::Mirror`, whose outcome reports exactly the vertices it created;
///  2. `Op::AssignWeights` over those vertices, with each influence's joint
///     swapped through `inf_anim::mirror_joint_map`.
///
/// Both carry values, so the journal replays the mirror as a fact and a later
/// build with a different `mirror_joint_map` cannot silently rewrite a saved
/// session — the `Op::Unwrap` doctrine, applied to the rig.
///
/// An **unbound** mesh takes the `Op::Mirror`-only path, byte for byte as before.
/// A bound mesh whose skeleton has a sided joint with no twin is **refused**, by
/// value, naming the joints: mirroring it would produce a limb weighted to the
/// wrong side and look correct doing it.
pub fn mirror_with_joints(
    session: &mut inf_dcc::MeshSession,
    axis: inf_dcc::MirrorAxis,
    coord: f64,
    skeleton: Option<&inf_anim::Skeleton>,
) -> Result<MirrorRigReport, MirrorRigError> {
    let bound = session.mesh().is_skinned();
    let map = match (bound, skeleton) {
        (true, Some(sk)) => {
            let unmatched = inf_anim::unmatched_sided_joints(sk);
            if !unmatched.is_empty() {
                return Err(MirrorRigError::UnmatchedJoints(unmatched));
            }
            Some(inf_anim::mirror_joint_map(sk))
        }
        // Rigid mesh, or a skinned one whose skeleton the caller could not
        // resolve. The second case is REPORTED rather than refused: the author
        // asked for a mirror, and a mirror with un-swapped weights is what this
        // op has always done.
        _ => None,
    };
    let out = session
        .apply(Op::Mirror { axis, coord })
        .map_err(MirrorRigError::Op)?;
    let mut report = MirrorRigReport {
        verts: out.verts.len(),
        faces: out.faces.len(),
        joints_swapped: 0,
        weights_unmapped: !bound || map.is_none(),
    };
    let Some(map) = map else {
        return Ok(report);
    };
    let mut weights: Vec<(VertId, inf_dcc::VertWeights)> = Vec::new();
    for &v in &out.verts {
        let Some(w) = session.mesh().vert_weights(v) else {
            continue;
        };
        let mut swapped = w;
        let mut moved = false;
        for (j, weight) in swapped.joints.iter_mut().zip(&w.weights) {
            if *weight == 0.0 {
                continue;
            }
            if let Some(&twin) = map.get(*j as usize) {
                if twin != *j {
                    *j = twin;
                    moved = true;
                }
            }
        }
        if moved {
            weights.push((v, swapped));
        }
    }
    if !weights.is_empty() {
        weights.sort_by_key(|(v, _)| *v);
        report.joints_swapped = weights.len();
        session
            .apply(Op::AssignWeights { weights })
            .map_err(MirrorRigError::Op)?;
    }
    Ok(report)
}

/// What [`mirror_with_joints`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MirrorRigReport {
    /// Vertices the mirror created.
    pub verts: usize,
    /// Faces the mirror created.
    pub faces: usize,
    /// Mirrored vertices whose influences moved to the other side's joints.
    pub joints_swapped: usize,
    /// **True when the weights were copied verbatim** — a rigid mesh (where that
    /// is the only meaning) or a skinned one whose skeleton the caller did not
    /// supply (where it is the pre-P24.3 behaviour, and worth saying out loud).
    pub weights_unmapped: bool,
}

/// Why [`mirror_with_joints`] refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MirrorRigError {
    /// The kernel refused the mirror itself.
    #[error("{0}")]
    Op(inf_dcc::OpError),
    /// The skeleton has sided joints with no opposite number, so a mirrored
    /// vertex has nowhere correct to send its influence.
    #[error(
        "these joints name a side but have no opposite number, so mirroring \
         would weight the copy to the wrong side: {0:?}"
    )]
    UnmatchedJoints(Vec<String>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_dcc::{cube, plane, MeshSession};

    fn projector(mesh: &Mesh, size: u32) -> Projector {
        Projector::new(frame(tessellate(mesh).bounds), size)
    }

    #[test]
    fn tessellation_is_what_the_writer_would_save() {
        // The claim in the module docs, asserted rather than implied: the drawn
        // triangles ARE the asset's triangles.
        let m = cube(1.0);
        let geo = tessellate(&m);
        let (asset, _) = inf_dcc::to_mesh_asset(&m, &inf_dcc::ExportOptions::default());
        assert_eq!(geo.verts.len(), asset.vertex_count());
        assert_eq!(geo.indices.len(), 36, "12 triangles");
        assert_eq!(geo.bounds.min, asset.bounds.min);
    }

    #[test]
    fn framing_fits_a_mesh_whatever_its_scale() {
        for size in [0.01f64, 1.0, 250.0] {
            let geo = tessellate(&cube(size));
            let view = frame(geo.bounds);
            let proj = Projector::new(view, 256);
            let m = cube(size);
            let mut seen = 0;
            for v in m.vert_ids() {
                let p = proj
                    .point(m.position(v).expect("live"))
                    .expect("in front of the camera");
                assert!(
                    (0.0..=256.0).contains(&p.x) && (0.0..=256.0).contains(&p.y),
                    "cube({size}) vertex projected to {p:?}"
                );
                seen += 1;
            }
            assert_eq!(seen, 8);
        }
    }

    #[test]
    fn a_pick_at_a_projected_vertex_returns_that_vertex() {
        let m = cube(1.0);
        let proj = projector(&m, 256);
        for v in m.vert_ids() {
            let p = proj.point(m.position(v).expect("live")).expect("visible");
            assert_eq!(
                pick(&m, &proj, SelectMode::Vert, p.x, p.y),
                Some(PickHit::Vert(v)),
                "clicking exactly on {v} must return {v}"
            );
        }
    }

    #[test]
    fn a_pick_in_open_space_returns_nothing() {
        let m = cube(1.0);
        let proj = projector(&m, 256);
        for mode in [SelectMode::Vert, SelectMode::Edge, SelectMode::Face] {
            assert_eq!(
                pick(&m, &proj, mode, 2.0, 2.0),
                None,
                "the far corner of the frame is empty in {mode:?} mode"
            );
        }
    }

    #[test]
    fn an_edge_pick_lands_on_the_edge_under_the_pointer() {
        let m = cube(1.0);
        let proj = projector(&m, 256);
        for h in m.half_ids() {
            let Some(c) = inf_dcc::canonical_edge(&m, h) else {
                continue;
            };
            if c != h {
                continue;
            }
            let a = proj
                .point(m.position(m.origin(h).expect("live")).expect("live"))
                .expect("visible");
            let b = proj
                .point(m.position(m.dest(h).expect("live")).expect("live"))
                .expect("visible");
            let (mx, my) = ((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
            let hit =
                pick(&m, &proj, SelectMode::Edge, mx, my).expect("something under the middle");
            // Two edges can cross in projection; what must hold is that the hit
            // is at least as close as the one we aimed at.
            let PickHit::Edge(got) = hit else {
                panic!("{hit:?}")
            };
            let ga = proj
                .point(m.position(m.origin(got).expect("live")).expect("live"))
                .expect("visible");
            let gb = proj
                .point(m.position(m.dest(got).expect("live")).expect("live"))
                .expect("visible");
            assert!(
                segment_distance(mx, my, ga.x, ga.y, gb.x, gb.y) <= 1e-3,
                "picked {got} which is not under the pointer"
            );
        }
    }

    #[test]
    fn a_face_pick_prefers_the_near_side_and_never_the_far_one() {
        // The whole point of the depth tie-break and the back-face rule: the
        // centre of a cube's screen has three faces stacked behind it, and a
        // modeller that hands back the one at the back is unusable.
        let m = cube(1.0);
        let proj = projector(&m, 256);
        let hit = pick(&m, &proj, SelectMode::Face, 128.0, 128.0).expect("the cube is in the way");
        let PickHit::Face(f) = hit else {
            panic!("{hit:?}")
        };
        let poly = project_face(&m, &proj, f).expect("visible");
        assert!(signed_area(&poly) < 0.0, "a back face was picked");
        let near = poly.iter().map(|p| p.depth).sum::<f32>() / poly.len() as f32;
        for other in m.face_ids() {
            if let Some(p) = project_face(&m, &proj, other) {
                if signed_area(&p) < 0.0 && contains(&p, 128.0, 128.0) {
                    let z = p.iter().map(|q| q.depth).sum::<f32>() / p.len() as f32;
                    assert!(near <= z + 1e-4, "{other} is nearer than the pick");
                }
            }
        }
    }

    #[test]
    fn face_picking_is_containment_not_proximity() {
        // A click just outside a plane's silhouette must miss. With a radius
        // rule it would grab the nearest polygon from open space, which is the
        // behaviour that makes an author fight a modeller.
        let m = plane(2.0);
        let proj = projector(&m, 256);
        let f = m.face_ids().next().expect("the quad");
        let poly = project_face(&m, &proj, f).expect("visible");
        let cx = poly.iter().map(|p| p.x).sum::<f32>() / poly.len() as f32;
        let cy = poly.iter().map(|p| p.y).sum::<f32>() / poly.len() as f32;
        assert_eq!(
            pick(&m, &proj, SelectMode::Face, cx, cy),
            Some(PickHit::Face(f)),
            "the centre of the quad is on the quad"
        );
        // Three pixels past its rightmost corner, on that corner's own row. A
        // proximity rule would take this; containment does not.
        let corner = poly
            .iter()
            .copied()
            .fold(poly[0], |a, p| if p.x > a.x { p } else { a });
        assert_eq!(
            pick(&m, &proj, SelectMode::Face, corner.x + 3.0, corner.y),
            None,
            "just outside the silhouette is outside"
        );
    }

    #[test]
    fn the_overlay_draws_the_selection_in_a_different_colour_than_the_wire() {
        let m = cube(1.0);
        let proj = projector(&m, 128);
        let style = OverlayStyle::default();
        let mut a = vec![200u8; 128 * 128 * 4];
        let mut b = a.clone();
        let empty = SelectionSet::new(1);
        draw_overlay(&mut a, 128, &m, &proj, &empty, SelectMode::Edge, &style);

        let mut sel = SelectionSet::new(1);
        // A VISIBLE edge: three of a cube's twelve are culled from any one angle
        // (`the_overlay_culls_edges_whose_faces_both_point_away`), and selecting
        // one of those would correctly change nothing — which would make this
        // test assert the opposite of what it means.
        let h = m
            .half_ids()
            .find(|&h| inf_dcc::canonical_edge(&m, h) == Some(h) && edge_is_visible(&m, &proj, h))
            .expect("a visible edge");
        sel.set_edge(&m, h, true);
        draw_overlay(&mut b, 128, &m, &proj, &sel, SelectMode::Edge, &style);

        assert_ne!(a, b, "selecting an edge must change the picture");
        let hot = b
            .chunks_exact(4)
            .filter(|p| p[0..3] == style.selected)
            .count();
        assert!(
            hot > 0,
            "the selected edge is drawn in the selection colour"
        );
        let cold = a
            .chunks_exact(4)
            .filter(|p| p[0..3] == style.selected)
            .count();
        assert_eq!(cold, 0, "an empty selection paints nothing hot");
    }

    #[test]
    fn the_overlay_culls_edges_whose_faces_both_point_away() {
        // The occlusion claim in the module docs, measured. A cube seen from a
        // corner shows 9 of its 12 edges; drawing all 12 is the x-ray a
        // wireframe must not be by default.
        let m = cube(1.0);
        let proj = projector(&m, 256);
        let visible = m
            .half_ids()
            .filter(|&h| inf_dcc::canonical_edge(&m, h) == Some(h))
            .filter(|&h| edge_is_visible(&m, &proj, h))
            .count();
        assert_eq!(visible, 9, "12 edges, 3 hidden behind the solid");

        // **Which** three, and which faces they hang off. A count of 9 is
        // inversion-symmetric — flip the facing rule and a cube still shows nine
        // edges, the nine on the FAR side — so the count alone certifies nothing.
        let facing: Vec<glam::DVec3> = m
            .face_ids()
            .filter(|&f| face_faces_eye(&m, &proj, f))
            .map(|f| {
                let vs = m.face_verts(f).expect("live");
                let c: glam::DVec3 = vs
                    .iter()
                    .map(|&v| m.position(v).expect("live"))
                    .sum::<glam::DVec3>()
                    / vs.len() as f64;
                c
            })
            .collect();
        assert_eq!(facing.len(), 3, "a cube shows three faces from a corner");
        // The default view looks from (+x, +y, +z), so the three visible faces
        // are the +X, +Y and +Z ones — and every one of their centroids is on the
        // camera's side of the origin.
        let eye = proj.eye();
        for c in &facing {
            let towards = glam::DVec3::new(eye.x as f64, eye.y as f64, eye.z as f64);
            assert!(
                c.dot(towards) > 0.0,
                "a face at {c:?} is on the far side and must not be front-facing"
            );
        }
        let mut axes: Vec<usize> = facing
            .iter()
            .map(|c| {
                let a = c.abs();
                if a.x >= a.y && a.x >= a.z {
                    0
                } else if a.y >= a.z {
                    1
                } else {
                    2
                }
            })
            .collect();
        axes.sort_unstable();
        assert_eq!(
            axes,
            vec![0, 1, 2],
            "one face per axis, all on the near side"
        );
    }

    #[test]
    fn what_the_overlay_paints_is_what_a_pick_would_return() {
        // The reason the overlay is composited here at all: ONE projector. Walk
        // every vertex, ask the picker what is under its drawn dot, and require
        // the answer to be that vertex.
        let m = cube(1.0);
        let proj = projector(&m, 256);
        let mut sel = SelectionSet::new(1);
        for v in m.vert_ids() {
            sel.set_vert(v, true);
        }
        let mut rgba = vec![10u8; 256 * 256 * 4];
        let style = OverlayStyle::default();
        draw_overlay(&mut rgba, 256, &m, &proj, &sel, SelectMode::Vert, &style);
        for v in m.vert_ids() {
            let p = proj.point(m.position(v).expect("live")).expect("visible");
            // Not "is the projected pixel hot" — a dot has a radius, so a mark
            // drawn three pixels away would still cover it and the gate would
            // pass on exactly the drift it exists to catch (measured). What is
            // asserted is that the mark's CENTROID is where the projection is.
            let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0usize);
            let r = style.vert_radius + 4;
            for dy in -r..=r {
                for dx in -r..=r {
                    let (x, y) = (p.x.round() as i32 + dx, p.y.round() as i32 + dy);
                    if x < 0 || y < 0 || x >= 256 || y >= 256 {
                        continue;
                    }
                    let i = ((y as usize) * 256 + x as usize) * 4;
                    if rgba[i..i + 3] == style.selected {
                        sx += x as f64;
                        sy += y as f64;
                        n += 1;
                    }
                }
            }
            assert!(n > 0, "{v} is not drawn at all near where it projects");
            let (cx, cy) = (sx / n as f64, sy / n as f64);
            assert!(
                (cx - p.x as f64).abs() < 1.0 && (cy - p.y as f64).abs() < 1.0,
                "{v} is drawn at ({cx:.2}, {cy:.2}) but projects to ({:.2}, {:.2})",
                p.x,
                p.y
            );
            assert_eq!(
                pick(&m, &proj, SelectMode::Vert, p.x, p.y),
                Some(PickHit::Vert(v))
            );
        }
    }

    #[test]
    fn drawing_never_writes_outside_the_buffer() {
        // A camera inside the model puts vertices behind the eye and coordinates
        // far off screen; the compositor must clip rather than panic or smear.
        let m = cube(1.0);
        let mut view = frame(tessellate(&m).bounds);
        view.distance = 0.05; // inside it
        let proj = Projector::new(view, 64);
        let mut rgba = vec![0u8; 64 * 64 * 4];
        let mut sel = SelectionSet::new(1);
        for f in m.face_ids() {
            sel.set_face(f, true);
        }
        for mode in [SelectMode::Vert, SelectMode::Edge, SelectMode::Face] {
            draw_overlay(
                &mut rgba,
                64,
                &m,
                &proj,
                &sel,
                mode,
                &OverlayStyle::default(),
            );
        }
        assert_eq!(rgba.len(), 64 * 64 * 4);
    }

    #[test]
    fn the_rasterizer_survives_coordinates_no_screen_could_hold() {
        // **The gate the overflow fix did not have.** Reverting `line` to
        // `dx + dy` / `2 * err` / `(dx - dy) + 2` failed NOTHING, while the
        // ledger listed it under gates that bite — so the fix was a claim.
        //
        // `f32 as i32` saturates (Rust guarantees it), and the saturated value is
        // `i32::MAX`; it is the DIFFERENCES of those that overflow, which is a
        // panic in a debug build and a wrapped, wrong picture in a release one.
        // A camera dollied inside a model produces coordinates like these for
        // real — the projection divides by a `w` a hair above zero.
        let mut rgba = vec![0u8; 32 * 32 * 4];
        let colour = [1u8, 2, 3];
        for (x0, y0, x1, y1) in [
            (-3.0e30f32, 0.0f32, 3.0e30f32, 5.0f32),
            (3.0e30, -3.0e30, -3.0e30, 3.0e30),
            (f32::NEG_INFINITY, 0.0, f32::INFINITY, 0.0),
            (f32::NAN, f32::NAN, 4.0, 4.0),
        ] {
            line(&mut rgba, 32, x0, y0, x1, y1, colour);
            dot(&mut rgba, 32, x0, y0, 3, colour);
        }
        // A polygon whose vertices are off in the same way must not hang the
        // scanline fill either.
        let far = |x: f32, y: f32| Projected { x, y, depth: 1.0 };
        fill_polygon(
            &mut rgba,
            32,
            &[
                far(-3.0e30, -3.0e30),
                far(3.0e30, -3.0e30),
                far(0.0, 3.0e30),
            ],
            colour,
            128,
        );
        assert_eq!(rgba.len(), 32 * 32 * 4, "the buffer is intact");
    }

    #[test]
    fn an_undersized_buffer_is_refused_rather_than_overrun() {
        let m = cube(1.0);
        let proj = projector(&m, 64);
        let mut rgba = vec![0u8; 16];
        let before = rgba.clone();
        draw_overlay(
            &mut rgba,
            64,
            &m,
            &proj,
            &SelectionSet::new(1),
            SelectMode::Edge,
            &OverlayStyle::default(),
        );
        assert_eq!(rgba, before);
    }

    #[test]
    fn an_orbit_never_re_tessellates() {
        // **The module's own contract, made checkable.** The docs at the top of
        // this file promise the tessellation runs "only when the mesh moves —
        // never on a camera orbit", and it was false: every frame re-ran the
        // exporter (ear clipping, a tangent solve, a corner intern) and cloned
        // the whole mesh, thirty times a second, for a camera that moved 144
        // bytes. A prose contract nothing measures is a comment.
        let mut s = MeshSession::new(cube(1.0));
        let mut cache = PreviewCache::new();
        let (geo0, mesh0) = cache.get(&s);
        for _ in 0..10 {
            let (g, m) = cache.get(&s);
            assert!(std::sync::Arc::ptr_eq(&g, &geo0), "a new tessellation");
            assert!(std::sync::Arc::ptr_eq(&m, &mesh0), "a new mesh clone");
        }
        assert_eq!(
            cache.tessellations(),
            1,
            "ten orbit frames, one tessellation"
        );

        // And it DOES re-run when the mesh moves, or the cache would be a very
        // fast way to draw the wrong thing.
        let f = s.mesh().face_ids().next().expect("a face");
        s.apply(inf_dcc::Op::SubdivideFaces { faces: vec![f] })
            .expect("subdivide");
        let (geo1, _) = cache.get(&s);
        assert!(!std::sync::Arc::ptr_eq(&geo1, &geo0));
        assert_eq!(cache.tessellations(), 2);
        assert!(geo1.indices.len() > geo0.indices.len());
    }

    // ── P23.5: rays, the surface point, the gizmo, and drags in flight ──────

    #[test]
    fn a_ray_through_a_projected_point_passes_back_through_that_point() {
        // The inverse of `Projector::point`, and the property that makes a dab
        // land where the pointer is. Asserted as a *distance from the ray*, which
        // is what a pick actually needs, rather than as an equality of two
        // matrices nobody looks at.
        let m = cube(1.0);
        let proj = projector(&m, 256);
        for v in m.vert_ids() {
            let p = m.position(v).expect("live");
            let s = proj.point(p).expect("visible");
            let (ro, rd) = proj.ray(s.x, s.y).expect("a ray through a visible pixel");
            let q = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
            let along = (q - ro).dot(rd);
            let off = (q - (ro + rd * along)).length();
            assert!(off < 1e-3, "{v} is {off} m off its own ray");
            assert!(along > 0.0, "{v} is behind the ray origin");
        }
        // A ray at the exact centre of the frame must aim at the target.
        let (_, rd) = proj.ray(128.0, 128.0).expect("a centre ray");
        assert!((rd.length() - 1.0).abs() < 1e-5, "the direction is unit");
    }

    #[test]
    fn a_surface_pick_lands_on_the_face_the_picker_returns_and_on_its_plane() {
        let m = cube(1.0);
        let proj = projector(&m, 256);
        let (f, hit) = pick_surface(&m, &proj, 128.0, 128.0).expect("the cube is in the way");
        assert_eq!(
            pick(&m, &proj, SelectMode::Face, 128.0, 128.0),
            Some(PickHit::Face(f)),
            "the surface point must be on the face the PICKER reports, or the \
             brush lands somewhere the highlight is not"
        );
        // On that face's plane, and inside its silhouette.
        let n = inf_dcc::face_normal(&m, f).expect("a normal");
        let verts = m.face_verts(f).expect("live");
        let c: DVec3 = verts
            .iter()
            .map(|&v| m.position(v).expect("live"))
            .sum::<DVec3>()
            / verts.len() as f64;
        assert!((hit - c).dot(n).abs() < 1e-9, "{hit:?} is off the plane");
        let back = proj
            .point(hit)
            .expect("the hit projects back into the frame");
        assert!(
            (back.x - 128.0).abs() < 0.5 && (back.y - 128.0).abs() < 0.5,
            "the hit re-projects to {back:?}, not to the pixel it came from"
        );
    }

    #[test]
    fn a_surface_pick_in_open_space_is_none() {
        let m = cube(1.0);
        let proj = projector(&m, 256);
        assert!(pick_surface(&m, &proj, 3.0, 3.0).is_none());
    }

    #[test]
    fn the_brush_ring_is_drawn_around_the_dab_and_nowhere_else() {
        let m = cube(1.0);
        let proj = projector(&m, 256);
        let (f, hit) = pick_surface(&m, &proj, 128.0, 128.0).expect("a hit");
        let n = inf_dcc::face_normal(&m, f).expect("a normal");
        let mut rgba = vec![0u8; 256 * 256 * 4];
        let colour = [7u8, 200, 190];
        draw_brush_ring(&mut rgba, 256, &proj, hit, n, 0.25, colour);

        let painted: Vec<(f64, f64)> = rgba
            .chunks_exact(4)
            .enumerate()
            .filter(|(_, p)| p[0..3] == colour)
            .map(|(i, _)| (((i % 256) as f64), ((i / 256) as f64)))
            .collect();
        assert!(!painted.is_empty(), "the ring must actually be drawn");
        // Its centroid is the dab, and every painted pixel is roughly one
        // projected radius away — a ring, not a disc and not a smear.
        let cx = painted.iter().map(|p| p.0).sum::<f64>() / painted.len() as f64;
        let cy = painted.iter().map(|p| p.1).sum::<f64>() / painted.len() as f64;
        assert!(
            (cx - 128.0).abs() < 3.0 && (cy - 128.0).abs() < 3.0,
            "the ring is centred at ({cx:.1}, {cy:.1}), not on the dab"
        );
        let radii: Vec<f64> = painted
            .iter()
            .map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt())
            .collect();
        let rmin = radii.iter().copied().fold(f64::MAX, f64::min);
        let rmax = radii.iter().copied().fold(0.0, f64::max);
        assert!(rmin > 2.0, "something was painted at the centre: {rmin}");
        assert!(
            rmax / rmin < 2.0,
            "the ring is not round in projection: {rmin}..{rmax}"
        );
        // A degenerate call paints nothing rather than panicking.
        let mut blank = vec![0u8; 256 * 256 * 4];
        draw_brush_ring(&mut blank, 256, &proj, hit, DVec3::ZERO, 0.25, colour);
        draw_brush_ring(&mut blank, 256, &proj, hit, n, f64::NAN, colour);
        assert!(blank.iter().all(|&b| b == 0));
    }

    #[test]
    fn the_gizmo_is_hittable_wherever_it_is_painted() {
        // The P23.4 rule, extended to the handles: the compositor draws the axis
        // to `pivot + dir × size` and `pick_axis` measures distance to that same
        // segment, so the end of every painted handle must be a hit. If the two
        // ever disagree the author drags nothing while looking straight at the
        // arrow.
        let m = cube(1.0);
        let view = frame(tessellate(&m).bounds);
        let proj = Projector::new(view, 256);
        let mut sel = SelectionSet::new(1);
        for f in m.face_ids() {
            sel.set_face(f, true);
        }
        let pivot = gizmo_pivot(&m, &sel, SelectMode::Face).expect("a pivot");
        assert!(pivot.length() < 1e-9, "a cube's centroid is its centre");

        for mode in [GizmoMode::Translate, GizmoMode::Scale] {
            let g = gizmo_size(&proj, view, pivot) as f64;
            for axis in [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z] {
                let d = axis.dir();
                let tip = pivot + DVec3::new(d.x as f64, d.y as f64, d.z as f64) * g;
                let s = proj.point(tip).expect("the tip is in frame");
                assert_eq!(
                    pick_gizmo(&proj, view, pivot, glam::Quat::IDENTITY, mode, s.x, s.y),
                    Some(axis),
                    "{mode:?}/{axis:?}: the painted tip is not hittable"
                );
            }
        }
        // A rotate ring is hittable on the ring, and open space is not a hit in
        // any mode.
        let g = gizmo_size(&proj, view, pivot) as f64;
        let ring = proj
            .point(pivot + DVec3::Y * g)
            .expect("a point on the X ring");
        assert!(pick_gizmo(
            &proj,
            view,
            pivot,
            glam::Quat::IDENTITY,
            GizmoMode::Rotate,
            ring.x,
            ring.y
        )
        .is_some());
        for mode in [GizmoMode::Translate, GizmoMode::Rotate, GizmoMode::Scale] {
            assert_eq!(
                pick_gizmo(&proj, view, pivot, glam::Quat::IDENTITY, mode, 2.0, 2.0),
                None,
                "{mode:?} claims a hit in the corner of the frame"
            );
        }
    }

    #[test]
    fn drawing_the_gizmo_marks_the_pixels_the_picker_answers_for() {
        let m = cube(1.0);
        let view = frame(tessellate(&m).bounds);
        let proj = Projector::new(view, 256);
        let mut sel = SelectionSet::new(1);
        for f in m.face_ids() {
            sel.set_face(f, true);
        }
        let pivot = gizmo_pivot(&m, &sel, SelectMode::Face).expect("a pivot");
        let mut rgba = vec![0u8; 256 * 256 * 4];
        draw_gizmo(
            &mut rgba,
            256,
            &proj,
            view,
            pivot,
            GizmoMode::Translate,
            Some(GizmoAxis::X),
        );
        // Something is painted, the active axis takes the amber highlight, and
        // every painted pixel is a place the picker answers for.
        let painted: Vec<usize> = rgba
            .chunks_exact(4)
            .enumerate()
            .filter(|(_, p)| p[0..3] != [0, 0, 0])
            .map(|(i, _)| i)
            .collect();
        assert!(
            painted.len() > 40,
            "the gizmo drew {} pixels",
            painted.len()
        );
        assert!(
            rgba.chunks_exact(4).any(|p| p[0..3] == [255, 217, 38]),
            "the active axis is not highlighted"
        );
        let hits = painted
            .iter()
            .filter(|&&i| {
                pick_gizmo(
                    &proj,
                    view,
                    pivot,
                    glam::Quat::IDENTITY,
                    GizmoMode::Translate,
                    (i % 256) as f32,
                    (i / 256) as f32,
                )
                .is_some()
            })
            .count();
        assert_eq!(
            hits,
            painted.len(),
            "{} painted pixels are not hittable",
            painted.len() - hits
        );
        // An undersized buffer is refused, like the overlay's.
        let mut small = vec![0u8; 16];
        let before = small.clone();
        draw_gizmo(
            &mut small,
            256,
            &proj,
            view,
            pivot,
            GizmoMode::Translate,
            None,
        );
        assert_eq!(small, before);
    }

    #[test]
    fn the_numeric_tool_and_the_gizmo_produce_identical_ops() {
        // Deliverable 2's contract, and the reason `transform_ops` exists at all:
        // the direct-manipulation twin must not be a second implementation. A
        // gizmo drag that resolves to a 0.25 m move along +X and a number box
        // that says 0.25 must journal the SAME op.
        let m = plane(2.0);
        let mut sel = SelectionSet::new(1);
        for f in m.face_ids() {
            sel.set_face(f, true);
        }
        let pivot = gizmo_pivot(&m, &sel, SelectMode::Face).expect("a pivot");
        let delta = DVec3::new(0.25, 0.0, 0.0);

        let numeric = transform_ops(
            &m,
            &sel,
            SelectMode::Face,
            pivot,
            VertTransform::Translate(delta),
            None,
        );
        let dragged = transform_ops(
            &m,
            &sel,
            SelectMode::Face,
            pivot,
            VertTransform::from_gizmo(GizmoDelta::Translate(delta)),
            None,
        );
        assert_eq!(numeric, dragged);
        assert_eq!(numeric.len(), 1, "a hard transform is ONE op: {numeric:?}");
        let verts: Vec<VertId> = sel
            .resolved_verts(&m, SelectMode::Face)
            .into_iter()
            .collect();
        assert_eq!(
            numeric,
            vec![Op::TranslateVerts {
                verts,
                delta: [0.25, 0.0, 0.0]
            }]
        );
    }

    /// Wave D: a whole proportional drag is **one** journal entry, and it still
    /// blends toward the identity in the same three ways.
    #[test]
    fn a_soft_transform_is_one_op_and_blends_toward_the_identity() {
        let m = cube(2.0);
        let mut sel = SelectionSet::new(1);
        let seed = m.vert_ids().next().expect("a vertex");
        sel.set_vert(seed, true);
        let pivot = gizmo_pivot(&m, &sel, SelectMode::Vert).expect("a pivot");
        let soft = Some((6.0, inf_terrain::Falloff::Linear));

        for xform in [
            VertTransform::Translate(DVec3::Y),
            VertTransform::Rotate {
                axis: DVec3::Y,
                radians: 0.5,
            },
            VertTransform::Scale(DVec3::splat(2.0)),
        ] {
            let ops = transform_ops(&m, &sel, SelectMode::Vert, pivot, xform, soft);
            assert_eq!(ops.len(), 1, "a soft drag is ONE op now: {ops:?}");
            let Op::MoveVerts { moves } = &ops[0] else {
                panic!("a soft drag journals a weight table: {:?}", ops[0]);
            };
            assert!(
                moves.len() > 1,
                "the neighbourhood moves too: {} vertices",
                moves.len()
            );
            // Sorted by vertex id — the op's wire convention.
            assert!(moves.windows(2).all(|w| w[0].0 < w[1].0), "{moves:?}");
            // …and it is a legal op on the mesh it was computed from.
            let mut probe = m.clone();
            inf_dcc::ops::apply(&mut probe, &ops[0]).expect("the soft drag applies");

            // The identity-blend, stated where it is a statement about NUMBERS.
            // Only for translate: a soft rotate and a soft scale are about the
            // selection's own pivot, so the selected vertex is the one that moves
            // LEAST (it is the pivot) — asserting "the seed moves most" there
            // would be asserting a falsehood that happens to be about the right
            // idea, which is how a vacuous gate is born.
            if let VertTransform::Translate(_) = xform {
                let len = |d: &[f64; 3]| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let seed_move = moves
                    .iter()
                    .find(|(v, _)| *v == seed)
                    .map(|(_, d)| len(d))
                    .expect("the selected vertex moves");
                assert!((seed_move - 1.0).abs() < 1e-12, "full weight is 1 m");
                for (v, d) in moves {
                    assert!(
                        len(d) <= seed_move + 1e-12,
                        "vertex {v} moved further than the selection did"
                    );
                }
                let at_full = moves
                    .iter()
                    .filter(|(_, d)| (len(d) - seed_move).abs() < 1e-12)
                    .count();
                assert_eq!(at_full, 1, "only the selection is at full weight");
            }
        }
    }

    #[test]
    fn a_soft_drag_over_a_dense_selection_is_still_one_op() {
        // The measurement this retires: a 289-vertex plane with a 3 m radius
        // produced **105 ops from one drag**, which `SOFT_WEIGHT_STEPS` capped at
        // 64 — still two full mesh snapshots at `CHECKPOINT_INTERVAL = 32` and
        // most of the 8-slot checkpoint history evicted, per drag. It is 1 now.
        let mut m = plane(2.0);
        for _ in 0..4 {
            let faces: Vec<_> = m.face_ids().collect();
            inf_dcc::ops::apply(&mut m, &Op::SubdivideFaces { faces }).expect("subdivides");
        }
        assert!(m.vert_count() >= 289, "{} verts", m.vert_count());
        // **Jittered, and that is what makes this a gate.** On a regular grid every
        // geodesic distance is a multiple of the spacing, so the *geometry* has
        // already quantized the weights and removing the quantization changes
        // nothing — measured: 24 ops either way. A real model's distances are all
        // distinct, which is the case the 105-op measurement came from and the case
        // the cap exists for. A deterministic integer jitter, so the fixture is
        // still a pure function of nothing.
        for (i, v) in m.vert_ids().collect::<Vec<_>>().into_iter().enumerate() {
            let p = m.position(v).expect("live");
            let d = ((i * 2_654_435_761) % 1_000) as f64 / 1_000.0;
            inf_dcc::ops::apply(
                &mut m,
                &Op::TranslateVerts {
                    verts: vec![v],
                    delta: [d * 0.01, 0.0, d * 0.013],
                },
            )
            .expect("jitters");
            let _ = p;
        }
        let mut sel = SelectionSet::new(1);
        sel.set_vert(m.vert_ids().next().expect("a vertex"), true);
        let ops = transform_ops(
            &m,
            &sel,
            SelectMode::Vert,
            DVec3::ZERO,
            VertTransform::Translate(DVec3::Y),
            Some((3.0, inf_terrain::Falloff::Linear)),
        );
        let moved = match ops.as_slice() {
            [Op::MoveVerts { moves }] => moves.len(),
            other => panic!("a soft drag journalled {} ops: {other:?}", other.len()),
        };
        println!("soft drag: {} verts, 1 op, {moved} moves", m.vert_count());
        // …and collapsing to one op did not throw the neighbourhood away. The
        // count is the whole neighbourhood now, not a quantized subset of it:
        // every distinct weight is carried, because there is no longer a reason
        // to round them together.
        assert!(moved > 100, "only {moved} vertices move");
    }

    // ── Wave D: starting a model, the pivot, the marquee ───────────────────

    /// Every primitive builds, is valid, and is the size it was asked for.
    #[test]
    fn every_primitive_builds_at_the_size_it_was_asked_for() {
        for kind in [
            DccPrimitiveDto::Cube,
            DccPrimitiveDto::Plane,
            DccPrimitiveDto::Cylinder,
            DccPrimitiveDto::Torus,
        ] {
            let m = primitive_mesh(kind, 2.0, 16, 8).expect("a primitive");
            assert_eq!(inf_dcc::validate(&m), Ok(()), "{kind:?}");
            assert!(m.face_count() > 0, "{kind:?} has no faces");
            // `size` is the BOUNDING BOX, so one number means the same box for
            // all four — which is the whole reason the dialog has one field.
            let (mut lo, mut hi) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
            for v in m.vert_ids() {
                let p = m.position(v).expect("live");
                lo = lo.min(p);
                hi = hi.max(p);
            }
            let extent = (hi - lo).max_element();
            assert!(
                (extent - 2.0).abs() < 0.02,
                "{kind:?} is {extent} across, not 2 m"
            );
        }
    }

    /// **NaN at the numeric door**, and every other refusal is a value with a
    /// sentence in it — not a clamp, and not a panic.
    #[test]
    fn a_primitive_refuses_a_size_that_is_not_a_size() {
        for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0, MAX_PRIMITIVE_M * 2.0] {
            let err =
                primitive_mesh(DccPrimitiveDto::Cube, bad, 16, 8).expect_err("{bad} is not a size");
            assert!(err.contains("size"), "{err}");
        }
        for bad in [0u32, 2, MAX_PRIMITIVE_SEGMENTS + 1] {
            assert!(primitive_mesh(DccPrimitiveDto::Cylinder, 1.0, bad, 8).is_err());
        }
    }

    /// The pivot really is four different answers, and the world-origin one is
    /// still `None` with nothing selected — a gizmo at the origin invites a drag
    /// that journals nothing.
    #[test]
    fn the_four_pivots_answer_differently() {
        let m = cube(2.0);
        let mut sel = SelectionSet::new(1);
        // Two vertices on one face, so the median and the bbox centre agree but
        // neither is the origin…
        let ids: Vec<VertId> = m.vert_ids().take(2).collect();
        for &v in &ids {
            sel.set_vert(v, true);
        }
        let median = gizmo_pivot_of(&m, &sel, SelectMode::Vert, DccPivotDto::Median, None)
            .expect("a median");
        let bbox = gizmo_pivot_of(&m, &sel, SelectMode::Vert, DccPivotDto::BoundingBox, None)
            .expect("a bbox");
        let origin = gizmo_pivot_of(&m, &sel, SelectMode::Vert, DccPivotDto::WorldOrigin, None)
            .expect("an origin");
        assert!(
            (median - bbox).length() < 1e-12,
            "two vertices: same answer"
        );
        assert_eq!(origin, DVec3::ZERO);
        assert!(median != DVec3::ZERO, "the fixture does not separate them");

        // …and a THREE-vertex selection separates the median from the bbox.
        sel.set_vert(m.vert_ids().nth(2).expect("a third"), true);
        let median = gizmo_pivot_of(&m, &sel, SelectMode::Vert, DccPivotDto::Median, None)
            .expect("a median");
        let bbox = gizmo_pivot_of(&m, &sel, SelectMode::Vert, DccPivotDto::BoundingBox, None)
            .expect("a bbox");
        assert!(
            (median - bbox).length() > 1e-9,
            "an uneven selection must separate them: {median} vs {bbox}"
        );

        // The active element is what was clicked, and falls back when nothing
        // has been.
        let active = DVec3::new(9.0, 9.0, 9.0);
        assert_eq!(
            gizmo_pivot_of(
                &m,
                &sel,
                SelectMode::Vert,
                DccPivotDto::ActiveElement,
                Some(active)
            ),
            Some(active)
        );
        assert_eq!(
            gizmo_pivot_of(&m, &sel, SelectMode::Vert, DccPivotDto::ActiveElement, None),
            Some(median)
        );
        // Nothing selected: no gizmo, whatever the pivot says.
        let empty = SelectionSet::new(1);
        for kind in [
            DccPivotDto::Median,
            DccPivotDto::BoundingBox,
            DccPivotDto::WorldOrigin,
        ] {
            assert_eq!(
                gizmo_pivot_of(&m, &empty, SelectMode::Vert, kind, None),
                None,
                "{kind:?}"
            );
        }
    }

    /// The normal orientation really points the blue axis at the surface.
    #[test]
    fn the_normal_orientation_aims_z_along_the_selection() {
        let m = cube(2.0);
        let view = PreviewView::default();
        let mut sel = SelectionSet::new(1);
        let top = m
            .face_ids()
            .max_by_key(|&f| (face_normal_of(&m, f).normalize_or_zero().y * 1e6) as i64)
            .expect("the +Y face");
        sel.set_face(top, true);
        let q = gizmo_orientation(&m, &sel, SelectMode::Face, DccOrientDto::Normal, view);
        let z = q * Vec3::Z;
        assert!(
            z.y > 0.99,
            "+Z should point along the face normal, got {z:?}"
        );
        // Global really is the identity, which is what the two sites used to
        // hard-code.
        assert_eq!(
            gizmo_orientation(&m, &sel, SelectMode::Face, DccOrientDto::Global, view),
            glam::Quat::IDENTITY
        );
        // View is a real basis: unit, right-handed, and not the identity.
        let v = gizmo_orientation(&m, &sel, SelectMode::Face, DccOrientDto::View, view);
        assert!((v.length() - 1.0).abs() < 1e-5);
        assert!(v.angle_between(glam::Quat::IDENTITY) > 0.1);
    }

    /// A marquee over the whole preview catches everything facing the eye — and
    /// **not** the far side, unless x-ray is on. That difference is the reason
    /// the flag exists.
    #[test]
    fn a_marquee_catches_the_near_side_and_x_ray_catches_both() {
        let m = cube(2.0);
        let view = frame(tessellate(&m).bounds);
        let proj = Projector::new(view, 256);
        let all = BoxRect {
            x0: 0.0,
            y0: 0.0,
            x1: 256.0,
            y1: 256.0,
        };
        let (_, _, near) = pick_box(&m, &proj, SelectMode::Face, all, false);
        let (_, _, both) = pick_box(&m, &proj, SelectMode::Face, all, true);
        assert_eq!(both.len(), 6, "x-ray catches every face of a cube");
        assert!(
            near.len() < both.len() && !near.is_empty(),
            "without x-ray the far side must be spared: {} of {}",
            near.len(),
            both.len()
        );
        // A rectangle over one corner catches less than everything.
        let corner = BoxRect {
            x0: 0.0,
            y0: 0.0,
            x1: 40.0,
            y1: 40.0,
        };
        let (_, _, few) = pick_box(&m, &proj, SelectMode::Face, corner, true);
        assert!(few.len() < 6, "a corner marquee caught the whole cube");
        // An edge needs BOTH endpoints inside — a rectangle that crosses one has
        // not selected it.
        let (_, edges_all, _) = pick_box(&m, &proj, SelectMode::Edge, all, true);
        assert_eq!(edges_all.len(), m.edge_count());
        // …and a degenerate rectangle is a click that wobbled.
        assert!(BoxRect {
            x0: 10.0,
            y0: 10.0,
            x1: 11.0,
            y1: 11.0
        }
        .is_degenerate());
        assert!(!all.is_degenerate());
    }

    /// The readout says what the drag is doing, in the units the author typed.
    #[test]
    fn the_drag_readout_names_the_axis_and_the_units() {
        let t = drag_readout(&VertTransform::Translate(DVec3::new(0.0, 0.42, 0.0)));
        assert!(
            t.contains("0.4200") && t.contains(" m") && t.contains('Y'),
            "{t}"
        );
        // An unconstrained drag names all three rather than lying about one.
        let t = drag_readout(&VertTransform::Translate(DVec3::new(1.0, 1.0, 0.0)));
        assert!(t.matches(',').count() == 2, "{t}");
        // Degrees at the boundary, not radians.
        let r = drag_readout(&VertTransform::Rotate {
            axis: DVec3::Y,
            radians: std::f64::consts::FRAC_PI_2,
        });
        assert!(
            r.contains("90.00") && r.contains('°') && r.contains('Y'),
            "{r}"
        );
        let s = drag_readout(&VertTransform::Scale(DVec3::splat(1.25)));
        assert_eq!(s, "x1.2500");
        let s = drag_readout(&VertTransform::Scale(DVec3::new(2.0, 1.0, 1.0)));
        assert!(s.contains("2.0000") && s.contains("1.0000"), "{s}");
    }

    // ── Wave D: the derived tools ──────────────────────────────────────────

    /// Two separate open rings — the shape an author actually bridges — must
    /// close into one solid, with every wall quad legal and no border left.
    ///
    /// **Not a cap-less cylinder**, and the difference is the test's whole point:
    /// a cylinder with its caps removed still has its *walls*, so bridging its
    /// two rims would ask for the wall edges a second time and refuse. The
    /// fixture has to be two rings with nothing between them.
    #[test]
    fn bridge_pairs_closes_two_open_rings_without_a_twist() {
        let mut m = Mesh::new();
        let k = vec![inf_dcc::CornerData::default(); 4];
        let ring = |m: &mut Mesh, y: f64| -> Vec<VertId> {
            [[0.0, y, 0.0], [1.0, y, 0.0], [1.0, y, 1.0], [0.0, y, 1.0]]
                .iter()
                .map(|p| {
                    let out =
                        inf_dcc::ops::apply(m, &Op::AddVertex { position: *p }).expect("a vertex");
                    out.verts[0]
                })
                .collect()
        };
        let lo = ring(&mut m, 0.0);
        let hi = ring(&mut m, 1.0);
        // Wound opposite ways, so their boundary loops run opposite ways — which
        // is what makes two rings *face* each other.
        inf_dcc::ops::apply(
            &mut m,
            &Op::AddFace {
                verts: lo.clone(),
                corners: k.clone(),
                slot: None,
            },
        )
        .expect("the floor");
        inf_dcc::ops::apply(
            &mut m,
            &Op::AddFace {
                verts: hi.iter().rev().copied().collect(),
                corners: k,
                slot: None,
            },
        )
        .expect("the lid");
        let border: Vec<HalfId> = m
            .half_ids()
            .filter(|&h| m.is_boundary(h) == Some(true))
            .collect();
        assert_eq!(boundary_loops(&m, &border).len(), 2, "two rings");
        let pairs = bridge_pairs(&m, &border).expect("a pairing");
        assert_eq!(pairs.len(), 4);
        inf_dcc::ops::apply(&mut m, &Op::BridgeLoops { pairs }).expect("the bridge applies");
        assert_eq!(inf_dcc::validate(&m), Ok(()));
        assert_eq!(m.face_count(), 6, "a closed box");
        assert!(
            m.half_ids().all(|h| m.is_boundary(h) == Some(false)),
            "the bridge closed the box"
        );
    }

    /// One open border is not two, and the refusal is a **value with a reason** —
    /// not a panic and not a silently-empty op list.
    #[test]
    fn bridging_one_border_refuses_with_a_reason() {
        let m = plane(2.0);
        let border: Vec<HalfId> = m
            .half_ids()
            .filter(|&h| m.is_boundary(h) == Some(true))
            .collect();
        let err = bridge_pairs(&m, &border).expect_err("one loop cannot be bridged");
        assert!(err.contains("exactly two"), "{err}");
    }

    /// Slide moves a selected loop along its **ring** edges. On a subdivided
    /// plane the middle row's ring edges run perpendicular to the row, so a slide
    /// moves the row sideways and never along itself.
    #[test]
    fn a_slide_moves_along_the_ring_edge_and_not_along_the_loop() {
        let mut m = plane(2.0);
        for _ in 0..2 {
            let faces: Vec<inf_dcc::FaceId> = m.face_ids().collect();
            inf_dcc::ops::apply(&mut m, &Op::SubdivideFaces { faces }).expect("subdivides");
        }
        // The row of vertices at z ≈ 0 (the plane spans x/z).
        let mut sel = SelectionSet::new(1);
        let mut row = 0;
        for v in m.vert_ids() {
            let p = m.position(v).expect("live");
            if p.z.abs() < 1e-9 {
                sel.set_vert(v, true);
                row += 1;
            }
        }
        assert!(row >= 3, "found only {row} vertices in the row");
        let moves = slide_moves(&m, &sel, SelectMode::Vert, 0.5);
        assert_eq!(moves.len(), row, "every selected vertex slides");
        for (v, d) in &moves {
            let d = DVec3::from_array(*d);
            assert!(d.length() > 0.0, "vertex {v} did not move");
            // The ring direction is ±z here; a move along x would mean the tool
            // picked a LOOP edge, which is the defect this test names.
            assert!(
                d.z.abs() > 1e-9 && d.x.abs() < 1e-9,
                "vertex {v} slid along the loop, not the ring: {d:?}"
            );
        }
        // …and reversing the sign really goes the other way.
        let back = slide_moves(&m, &sel, SelectMode::Vert, -0.5);
        for ((_, f), (_, b)) in moves.iter().zip(back.iter()) {
            assert!(
                DVec3::from_array(*f).dot(DVec3::from_array(*b)) < 0.0,
                "a negative slide went the same way"
            );
        }
    }

    #[test]
    fn merge_clusters_groups_only_what_is_inside_the_tolerance() {
        let mut m = cube(2.0);
        let ids: Vec<VertId> = m.vert_ids().collect();
        // Drag two vertices to within 1 mm of each other; leave the rest apart.
        let a = m.position(ids[0]).expect("live");
        let b = m.position(ids[1]).expect("live");
        inf_dcc::ops::apply(
            &mut m,
            &Op::TranslateVerts {
                verts: vec![ids[1]],
                delta: (a + DVec3::X * 0.0005 - b).to_array(),
            },
        )
        .expect("drags");
        let clusters = merge_clusters(&m, &ids, 0.001).expect("a clustering");
        assert_eq!(clusters.len(), 1, "one pair is close enough: {clusters:?}");
        assert_eq!(clusters[0], vec![ids[0], ids[1]]);
        // A zero tolerance groups only *exactly* coincident vertices, which a
        // cube has none of.
        assert!(merge_clusters(&m, &ids, 0.0)
            .expect("a clustering")
            .is_empty());
        assert!(merge_clusters(&m, &ids, f64::NAN).is_err());
    }

    /// **The UV pane's pick and its drag** — the pixel arithmetic is the inverse
    /// of the picture's, and the drag moves EVERY corner of a cut vertex.
    #[test]
    fn a_uv_pick_finds_the_corner_under_the_pointer_and_a_drag_moves_all_of_them() {
        // A cube, unwrapped, so its corners are spread over a real atlas.
        let mut m = cube(2.0);
        let seams: Vec<HalfId> = auto_seam_edges(&m, 40.0);
        inf_dcc::ops::apply(
            &mut m,
            &Op::SetEdgesSeam {
                halfs: seams,
                seam: true,
            },
        )
        .expect("marks");
        let u = inf_dcc::unwrap(&m).expect("unwraps");
        inf_dcc::ops::apply(&mut m, &u.op).expect("applies");

        // Round-trip the pixel mapping: a UV converted to pixels and back is the
        // UV. This is the arithmetic that makes the pick land where the picture
        // says it is.
        for uv in [[0.0, 0.0], [1.0, 1.0], [0.25, 0.75]] {
            let (x, y) = uv_to_px(uv, 256.0);
            let back = uv_from_px(x, y, 256.0);
            assert!((back[0] - uv[0]).abs() < 1e-6 && (back[1] - uv[1]).abs() < 1e-6);
        }

        // Pick at a corner's own pixel and get its vertex back.
        let h = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(false))
            .expect("a corner");
        let uv = m.corner_uv(h).expect("a uv");
        let v = m.origin(h).expect("live");
        let (x, y) = uv_to_px(uv, 256.0);
        assert_eq!(pick_uv(&m, 256, x, y), Some(v));
        // …and a pick in empty space finds nothing rather than the nearest thing.
        assert_eq!(pick_uv(&m, 256, -500.0, -500.0), None);

        // A drag moves EVERY corner of the selected vertex. On a fully-seamed
        // cube each vertex is cut into three charts, so this is the case where
        // moving only one corner would tear the seam.
        let mut sel = SelectionSet::new(1);
        sel.set_vert(v, true);
        let moves = uv_move_corners(&m, &sel, SelectMode::Vert, 256, 25.6, 0.0);
        let corners_at_v = m
            .vert_outgoing(v)
            .unwrap_or(&[])
            .iter()
            .filter(|&&h| m.is_boundary(h) == Some(false))
            .count();
        assert_eq!(
            moves.len(),
            corners_at_v,
            "a drag must move every corner of the vertex, not one of them"
        );
        assert!(corners_at_v > 1, "the fixture must have a CUT vertex");
        // 25.6 px of 256 is 0.1 of a UV unit, rightward.
        for (h, uv) in &moves {
            let was = m.corner_uv(*h).expect("a uv");
            assert!((uv[0] - was[0] - 0.1).abs() < 1e-9, "{uv:?} vs {was:?}");
            assert!((uv[1] - was[1]).abs() < 1e-12);
        }
        // Sorted by half-edge — the op's wire convention.
        assert!(moves.windows(2).all(|w| w[0].0 < w[1].0));
        // …and it applies.
        inf_dcc::ops::apply(&mut m, &Op::MoveUvs { corners: moves }).expect("the drag applies");

        // A zero drag is nothing, not an empty op.
        assert!(uv_move_corners(&m, &sel, SelectMode::Vert, 256, 0.0, 0.0).is_empty());
        assert!(uv_move_corners(&m, &sel, SelectMode::Vert, 256, f64::NAN, 0.0).is_empty());
    }

    /// Auto-seam cuts a cube into six charts and leaves a smooth cylinder's
    /// barrel whole — the two ends of the rule, on the two shapes that make it
    /// a rule rather than a threshold.
    #[test]
    fn auto_seam_cuts_the_creases_and_the_borders_and_nothing_else() {
        let cube = cube(2.0);
        let seams = auto_seam_edges(&cube, 40.0);
        assert_eq!(
            seams.len(),
            cube.edge_count(),
            "every cube edge is 90°, so every one is a seam"
        );
        // …and applying them really cuts it into six charts.
        let mut m = cube.clone();
        inf_dcc::ops::apply(
            &mut m,
            &Op::SetEdgesSeam {
                halfs: seams,
                seam: true,
            },
        )
        .expect("marks");
        assert_eq!(inf_dcc::charts(&m).len(), 6, "a cube cuts into six faces");

        // A 16-sided cylinder: the barrel is 22.5° and stays whole; the two cap
        // rims are 90° and cut.
        let c = inf_dcc::cylinder(0.5, 2.0, 16);
        let seams = auto_seam_edges(&c, 40.0);
        assert!(!seams.is_empty() && seams.len() < c.edge_count());
        let mut m = c.clone();
        inf_dcc::ops::apply(
            &mut m,
            &Op::SetEdgesSeam {
                halfs: seams,
                seam: true,
            },
        )
        .expect("marks");
        assert_eq!(
            inf_dcc::charts(&m).len(),
            3,
            "barrel plus two caps: {}",
            inf_dcc::charts(&m).len()
        );

        // A PLANE: no crease anywhere, so only its border is cut — which is the
        // arm that says the boundary rule is doing something rather than being
        // carried by the angle one.
        let p = plane(2.0);
        let seams = auto_seam_edges(&p, 40.0);
        assert_eq!(
            seams.len(),
            4,
            "a quad's four border edges, and nothing else"
        );
    }

    /// Auto-smooth splits a cube's 90° edges from a smooth cylinder's shallow
    /// ones, and it does it without `acos` anywhere.
    #[test]
    fn auto_smooth_creases_only_the_edges_over_the_threshold() {
        let m = cube(2.0);
        let (sharp, soft) = shade_edges(&m, &[], true, Some(30.0));
        assert_eq!(soft.len(), 0, "every cube edge is 90°");
        assert_eq!(sharp.len(), m.edge_count(), "…so all of them crease");

        // A 16-sided cylinder's wall edges are 22.5° apart — under the threshold.
        let c = inf_dcc::cylinder(0.5, 2.0, 16);
        let (sharp, soft) = shade_edges(&c, &[], true, Some(30.0));
        assert!(!soft.is_empty(), "the wall edges should stay smooth");
        assert!(
            !sharp.is_empty(),
            "the cap rim is 90° and must still crease"
        );
        assert_eq!(sharp.len() + soft.len(), c.edge_count());

        // …and with no threshold it is all-or-nothing.
        let (sharp, soft) = shade_edges(&m, &[], true, None);
        assert!(sharp.is_empty() && soft.len() == m.edge_count());
        let (sharp, soft) = shade_edges(&m, &[], false, None);
        assert!(soft.is_empty() && sharp.len() == m.edge_count());
    }

    #[test]
    fn a_quantized_weight_is_a_step_and_a_zero_weight_is_dropped() {
        assert_eq!(quantize_weight(1.0), 1.0);
        assert_eq!(quantize_weight(0.5), 0.5);
        // `f64::round` breaks ties away from zero, so exactly half a step rounds
        // UP — spelled out rather than discovered, because it decides whether the
        // outermost ring of a falloff moves at all.
        assert_eq!(
            quantize_weight(1.0 / 128.0),
            1.0 / 64.0,
            "half a step rounds up"
        );
        assert_eq!(quantize_weight(1.0 / 256.0), 0.0, "below half a step drops");
        assert_eq!(quantize_weight(1.0 / 64.0), 1.0 / 64.0);
        assert_eq!(quantize_weight(f64::NAN), 0.0);
        assert_eq!(quantize_weight(2.0), 1.0);
        // Every representable answer is a multiple of the step.
        for i in 0..=200 {
            let q = quantize_weight(i as f64 / 200.0);
            assert_eq!(q * SOFT_WEIGHT_STEPS, (q * SOFT_WEIGHT_STEPS).round());
        }
    }

    #[test]
    fn the_selection_revision_moves_when_the_selection_does_and_not_otherwise() {
        // **M3.** The UV pane keyed on the JOURNAL generation, which a selection
        // change does not move — and `selected` is a count, so face A and face B
        // both read `1`. Neither number can tell those two states apart; this one
        // has to.
        let m = cube(1.0);
        let faces: Vec<_> = m.face_ids().collect();
        let mut a = SelectionSet::new(7);
        a.set_face(faces[0], true);
        let mut b = SelectionSet::new(7);
        b.set_face(faces[1], true);
        assert_eq!(
            a.len(SelectMode::Face),
            b.len(SelectMode::Face),
            "the COUNT cannot tell these apart — that is the point"
        );
        assert_ne!(
            selection_revision(&a),
            selection_revision(&b),
            "A and B are different selections and must read differently"
        );
        // Stable for an unchanged set, and it moves for the generation too.
        let mut again = SelectionSet::new(7);
        again.set_face(faces[0], true);
        assert_eq!(selection_revision(&a), selection_revision(&again));
        let mut later = SelectionSet::new(8);
        later.set_face(faces[0], true);
        assert_ne!(selection_revision(&a), selection_revision(&later));
        // An empty selection is not the same as a one-face one.
        assert_ne!(
            selection_revision(&SelectionSet::new(7)),
            selection_revision(&a)
        );
    }

    #[test]
    fn a_transform_with_nothing_selected_produces_no_ops() {
        let m = plane(2.0);
        let sel = SelectionSet::new(1);
        assert!(transform_ops(
            &m,
            &sel,
            SelectMode::Face,
            DVec3::ZERO,
            VertTransform::Translate(DVec3::X),
            None
        )
        .is_empty());
        assert!(gizmo_pivot(&m, &sel, SelectMode::Face).is_none());
    }

    #[test]
    fn a_drag_in_flight_is_settled_and_the_selection_survives_it() {
        // **The orphan-settler ruling, tested rather than asserted in prose.** A
        // stroke whose pointer-up never arrives is committed by whichever door
        // notices — and because every op a drag makes preserves ids, the
        // selection is CARRIED, not dropped. A settle that dropped it would
        // deselect the face the author is sculpting on every tool switch.
        let mut s = MeshSession::new(cube(1.0));
        let mut sel = SelectionSet::new(s.generation());
        let f = s.mesh().face_ids().next().expect("a face");
        sel.set_face(f, true);
        let start = s
            .mesh()
            .position(s.mesh().vert_ids().next().expect("a vertex"))
            .expect("live");
        let pending = PendingDrag::Stroke(StrokeInFlight {
            mode: inf_dcc::SculptMode::Draw,
            radius: 0.8,
            strength: 0.1,
            falloff: inf_dcc::SculptFalloff::Smooth,
            path: (0..6)
                .map(|i| start + DVec3::X * (i as f64 * 0.1))
                .collect(),
            last_normal: DVec3::Y,
        });
        let before = s.mesh().encoded();
        assert_eq!(
            settle_drag(&mut s, &mut sel, SelectMode::Face, Some(pending)),
            None
        );
        assert_eq!(s.ops().len(), 1, "a whole stroke is ONE journal entry");
        assert_ne!(s.mesh().encoded(), before);
        assert_eq!(sel.generation(), s.generation(), "the selection kept up");
        assert!(sel.contains_face(f), "and it still names the same face");
        assert!(s.undo());
        assert_eq!(s.mesh().encoded(), before, "one undo takes the whole drag");
    }

    #[test]
    fn a_stroke_below_the_radius_floor_is_refused_and_a_miss_is_not() {
        // **The three answers**, and the reason this is Ring 1: the floor used to
        // live in the Tauri command, where deleting it failed nothing at all.
        //
        // A refusal and a miss must stay distinguishable, because the panel does
        // opposite things with them — a miss becomes a camera orbit, a refusal
        // becomes a sentence in the status bar.
        let m = cube(1.0);
        let proj = projector(&m, 256);
        let mode = inf_dcc::SculptMode::Draw;
        let falloff = inf_dcc::SculptFalloff::Smooth;

        for radius in [1.0e-4_f64, 1.0e-12, 0.0, -1.0, f64::NAN] {
            let brush = BrushSettings {
                mode,
                radius,
                strength: 0.1,
                falloff,
            };
            let err = begin_stroke(&m, &proj, 128.0, 128.0, brush)
                .expect_err("a radius below the floor must be refused");
            assert!(err.contains("floor"), "{err}");
        }
        // At the floor exactly, and above it, the pointer is allowed to grab.
        for radius in [inf_dcc::MIN_BRUSH_RADIUS_M, 0.25] {
            let brush = BrushSettings {
                mode,
                radius,
                strength: 0.1,
                falloff,
            };
            let stroke = begin_stroke(&m, &proj, 128.0, 128.0, brush)
                .expect("a legal radius is not refused")
                .expect("the pointer is on the cube");
            assert_eq!(stroke.path.len(), 1);
            assert!((stroke.last_normal.length() - 1.0).abs() < 1e-9);
        }
        // And a miss is `Ok(None)` — silent, not an error.
        let brush = BrushSettings {
            mode,
            radius: 0.25,
            strength: 0.1,
            falloff,
        };
        assert!(begin_stroke(&m, &proj, 3.0, 3.0, brush)
            .expect("a miss is not a refusal")
            .is_none());
    }

    #[test]
    fn the_radius_floor_and_the_dab_cap_are_one_decision() {
        // The two constants are sized against each other: at the floor, the cap
        // covers more than a metre of drag, so a legal stroke never meets it.
        // Asserted here as well as in the kernel because this is the door that
        // enforces the floor, and a change to either number has to fail both.
        let reach = inf_dcc::MAX_STROKE_DABS as f64
            * inf_dcc::DAB_SPACING_FRACTION
            * inf_dcc::MIN_BRUSH_RADIUS_M;
        assert!(reach > 1.0, "{reach} m of drag at the floor");
    }

    #[test]
    fn a_gesture_that_did_nothing_journals_nothing() {
        // A click with no drag must not produce an undo step. The gizmo's guard
        // is `VertTransform::is_identity`; the stroke's is an empty path.
        let mut s = MeshSession::new(cube(1.0));
        let mut sel = SelectionSet::new(s.generation());
        for f in s.mesh().face_ids() {
            sel.set_face(f, true);
        }
        let pivot = gizmo_pivot(s.mesh(), &sel, SelectMode::Face).expect("a pivot");
        let drag = GizmoDrag::begin(
            GizmoMode::Translate,
            GizmoAxis::X,
            glam::Quat::IDENTITY,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
        );
        for pending in [
            PendingDrag::Gizmo(Box::new(GizmoInFlight {
                drag,
                pivot,
                xform: VertTransform::Translate(DVec3::ZERO),
                soft: None,
            })),
            PendingDrag::Stroke(StrokeInFlight {
                mode: inf_dcc::SculptMode::Draw,
                radius: 0.5,
                strength: 0.1,
                falloff: inf_dcc::SculptFalloff::Smooth,
                path: Vec::new(),
                last_normal: DVec3::Y,
            }),
        ] {
            assert!(pending.ops(s.mesh(), &sel, SelectMode::Face).is_empty());
            assert_eq!(
                settle_drag(&mut s, &mut sel, SelectMode::Face, Some(pending)),
                None
            );
        }
        assert_eq!(s.ops().len(), 0, "nothing was journalled");
    }

    #[test]
    fn the_live_preview_re_tessellates_only_when_the_drag_moves() {
        // The scratch channel's own contract, made checkable the same way
        // `an_orbit_never_re_tessellates` made the committed one. Ten frames of a
        // motionless drag cost one tessellation; extending the path costs one
        // more; and letting go returns to the cached committed geometry.
        let s = MeshSession::new(cube(1.0));
        let sel = SelectionSet::new(s.generation());
        let mut cache = PreviewCache::new();
        let mut stroke = StrokeInFlight {
            mode: inf_dcc::SculptMode::Draw,
            radius: 0.8,
            strength: 0.1,
            falloff: inf_dcc::SculptFalloff::Smooth,
            path: vec![DVec3::new(0.5, 0.5, 0.5)],
            last_normal: DVec3::Y,
        };
        let pending = PendingDrag::Stroke(stroke.clone());
        let (geo0, _) = cache.get_with_pending(&s, &sel, SelectMode::Face, Some(&pending));
        for _ in 0..10 {
            let (g, _) = cache.get_with_pending(&s, &sel, SelectMode::Face, Some(&pending));
            assert!(std::sync::Arc::ptr_eq(&g, &geo0), "a new scratch frame");
        }
        assert_eq!(cache.scratch_frames(), 1, "ten frames, one scratch");
        // Wave D: and that one frame was DISPLACED, not re-exported. Zero is the
        // whole claim — a fast path that silently stopped working would read
        // `1` here and every other assertion in this test would still pass.
        assert_eq!(
            cache.scratch_tessellations(),
            0,
            "the drag frame re-ran the exporter instead of displacing"
        );
        assert_eq!(
            cache.tessellations(),
            1,
            "the committed geometry is tessellated ONCE, as the base the drag \
             displaces — it used to be zero because the scratch re-exported"
        );

        stroke.path.push(DVec3::new(0.6, 0.5, 0.5));
        let moved = PendingDrag::Stroke(stroke);
        let (geo1, _) = cache.get_with_pending(&s, &sel, SelectMode::Face, Some(&moved));
        assert!(!std::sync::Arc::ptr_eq(&geo1, &geo0));
        assert_eq!(cache.scratch_frames(), 2);
        assert_eq!(cache.scratch_tessellations(), 0, "still displacing");

        // Pointer-up: back to the committed cache, and the scratch is released.
        let (committed, _) = cache.get_with_pending(&s, &sel, SelectMode::Face, None);
        assert_eq!(cache.tessellations(), 1);
        assert_eq!(cache.scratch_frames(), 2, "no extra scratch on release");
        assert_eq!(cache.scratch_tessellations(), 0);
        assert!(!std::sync::Arc::ptr_eq(&committed, &geo1));
    }

    /// **Round-2 finding R2.F4**: the stamp the upload is skipped on has to move
    /// whenever the geometry does, and a live drag moves the geometry without
    /// moving the journal generation — that is the whole reason the scratch
    /// channel exists.
    ///
    /// The command layer sent `session.generation()`, so the renderer's
    /// `set_geometry` early-return fired on every drag frame and the shaded
    /// surface stayed frozen at the pre-drag mesh while the CPU-composited
    /// wireframe tracked the pointer. The arm asserts the property that was
    /// false: **distinct geometry, distinct stamp** — and that a frame which
    /// really did not move keeps its stamp, because a stamp that always changed
    /// would retire the warm path this cache exists for.
    #[test]
    fn the_upload_stamp_moves_whenever_the_geometry_does() {
        let s = MeshSession::new(cube(1.0));
        let sel = SelectionSet::new(s.generation());
        let mut cache = PreviewCache::new();

        let (_, _) = cache.get(&s);
        let committed = cache.upload_stamp();

        let mut stroke = StrokeInFlight {
            mode: inf_dcc::SculptMode::Draw,
            radius: 0.8,
            strength: 0.1,
            falloff: inf_dcc::SculptFalloff::Smooth,
            path: vec![DVec3::new(0.5, 0.5, 0.5)],
            last_normal: DVec3::Y,
        };
        let pending = PendingDrag::Stroke(stroke.clone());
        cache.get_with_pending(&s, &sel, SelectMode::Face, Some(&pending));
        let first = cache.upload_stamp();
        assert_ne!(
            first, committed,
            "the first drag frame draws a different mesh from the committed one, at the SAME \
             journal generation — this is the inequality `session.generation()` could not express"
        );

        // A frame where nothing moved keeps the stamp: the warm path survives.
        cache.get_with_pending(&s, &sel, SelectMode::Face, Some(&pending));
        assert_eq!(first, cache.upload_stamp(), "an orbit mid-stroke is free");

        // The stroke grows → new geometry → new stamp.
        stroke.path.push(DVec3::new(0.6, 0.5, 0.5));
        let moved = PendingDrag::Stroke(stroke);
        cache.get_with_pending(&s, &sel, SelectMode::Face, Some(&moved));
        let second = cache.upload_stamp();
        assert_ne!(second, first, "the stroke grew and the surface must follow");

        // Letting go returns to the committed geometry, which is a THIRD picture
        // and must not be skipped either.
        cache.get_with_pending(&s, &sel, SelectMode::Face, None);
        assert_eq!(
            cache.upload_stamp(),
            committed,
            "an abandoned drag returns to exactly the geometry it started from"
        );
        assert_ne!(cache.upload_stamp(), second);
    }

    /// **Round 3, the carried half of R2.F4**: the scratch key took two of the
    /// three inputs the scratch is built from.
    ///
    /// `PendingDrag::scratch(session, selection, mode)` reads the selected ids —
    /// a gizmo drag's ops ARE the selection transformed — and the key was
    /// `(generation, step)`. Change the selection while the pointer is down, or
    /// switch Vert→Edge→Face so the same op set reads different ids, and the
    /// transform has not moved: same step, same generation, cache hit, and the
    /// shaded surface keeps drawing the elements that are no longer selected.
    ///
    /// Asserted on the cache's own observable state (`scratch_tessellations`)
    /// as well as the stamp, so "it re-tessellated" is a measurement rather than
    /// an inference from the key having changed.
    #[test]
    fn the_scratch_key_sees_the_selection_and_the_mode() {
        let s = MeshSession::new(cube(1.0));
        let mut sel = SelectionSet::new(s.generation());
        let verts: Vec<_> = s.mesh().vert_ids().collect();
        assert!(verts.len() >= 2, "the fixture needs two vertices");
        sel.set_vert(verts[0], true);

        let pending = PendingDrag::Gizmo(Box::new(GizmoInFlight {
            drag: GizmoDrag::begin(
                GizmoMode::Translate,
                GizmoAxis::X,
                glam::Quat::IDENTITY,
                Vec3::ZERO,
                Vec3::new(0.0, 0.0, 5.0),
                Vec3::new(0.0, 0.0, -1.0),
            ),
            pivot: DVec3::ZERO,
            xform: VertTransform::Translate(DVec3::new(0.25, 0.0, 0.0)),
            soft: None,
        }));
        let mut cache = PreviewCache::new();
        cache.get_with_pending(&s, &sel, SelectMode::Vert, Some(&pending));
        let first = cache.upload_stamp();
        let tess = cache.scratch_frames();

        // Same transform, same generation — a different SELECTION.
        let mut other = SelectionSet::new(s.generation());
        other.set_vert(verts[1], true);
        cache.get_with_pending(&s, &other, SelectMode::Vert, Some(&pending));
        assert_ne!(
            cache.upload_stamp(),
            first,
            "the drag moved a different vertex and the preview served the old mesh"
        );
        assert_eq!(
            cache.scratch_frames(),
            tess + 1,
            "…and it really rebuilt the frame rather than only re-keying"
        );

        // Same transform, same selection — a different MODE. The ids are the
        // same integers and they name different elements.
        let second = cache.upload_stamp();
        cache.get_with_pending(&s, &other, SelectMode::Face, Some(&pending));
        assert_ne!(
            cache.upload_stamp(),
            second,
            "Vert and Face read the same id set as different geometry"
        );

        // And the warm path survives: an unchanged frame is still free.
        let third = cache.upload_stamp();
        let tess = cache.scratch_frames();
        cache.get_with_pending(&s, &other, SelectMode::Face, Some(&pending));
        assert_eq!(cache.upload_stamp(), third, "an orbit mid-drag is free");
        assert_eq!(cache.scratch_frames(), tess);
    }

    /// **Round 3**: `0` is `PreviewSession`'s built-in sphere, so no stamp this
    /// cache serves may be zero — a fold that landed there would skip the
    /// upload and draw a ball where the model is.
    ///
    /// Asserted as **ODDNESS**, not as `!= 0`. "This hash is never zero" is a
    /// claim about 2⁻⁶⁴ of the input space and no finite arm can check it: a
    /// test that tries a few thousand keys passes just as happily with the
    /// guarantee deleted, which is how a probabilistic property hides a
    /// mutation (measured — `acc & !1` survived the first version of this arm).
    /// Oddness is the mechanism `| 1` actually provides, it implies non-zero for
    /// every input at once, and it fails the moment the bit stops being set.
    #[test]
    fn no_stamp_this_cache_serves_is_the_sphere_sentinel() {
        for g in 0..64u64 {
            for step in [0u64, 1, 7, u64::MAX, COMMITTED_STEP] {
                let k = fold_key(g, step);
                assert_eq!(
                    k % 2,
                    1,
                    "fold_key({g}, {step}) = {k} is even, so some other input \
                     folds to 0 — the built-in sphere"
                );
                assert_ne!(k, 0);
            }
        }
        // …and the served stamp is the folded one, on both channels.
        let s = MeshSession::new(cube(1.0));
        let sel = SelectionSet::new(s.generation());
        let mut cache = PreviewCache::new();
        cache.get(&s);
        assert_eq!(cache.upload_stamp() % 2, 1);
        let pending = PendingDrag::Stroke(StrokeInFlight {
            mode: inf_dcc::SculptMode::Draw,
            radius: 0.8,
            strength: 0.1,
            falloff: inf_dcc::SculptFalloff::Smooth,
            path: vec![DVec3::new(0.5, 0.5, 0.5)],
            last_normal: DVec3::Y,
        });
        cache.get_with_pending(&s, &sel, SelectMode::Face, Some(&pending));
        assert_eq!(cache.upload_stamp() % 2, 1);
    }

    #[test]
    fn the_scratch_never_touches_the_document() {
        // The whole reason the live channel is a clone: a preview frame must not
        // be able to edit the mesh. Ten frames of a stroke, and the session's
        // bytes and journal are exactly what they were.
        let s = MeshSession::new(cube(1.0));
        let sel = SelectionSet::new(s.generation());
        let before = s.mesh().encoded();
        let mut cache = PreviewCache::new();
        let pending = PendingDrag::Stroke(StrokeInFlight {
            mode: inf_dcc::SculptMode::Draw,
            radius: 2.0,
            strength: 0.5,
            falloff: inf_dcc::SculptFalloff::Smooth,
            path: vec![DVec3::new(0.5, 0.5, 0.5), DVec3::new(0.5, 0.9, 0.5)],
            last_normal: DVec3::Y,
        });
        let (geo, scratch) = cache.get_with_pending(&s, &sel, SelectMode::Face, Some(&pending));
        assert_eq!(s.mesh().encoded(), before, "the document did not move");
        assert_eq!(s.ops().len(), 0, "and nothing was journalled");
        assert_ne!(
            scratch.encoded(),
            before,
            "…while the scratch DOES show the uncommitted dab"
        );
        assert!(geo.verts.len() >= 8);
    }

    #[test]
    fn live_drag_frame_cost_is_measured() {
        // **The number the P23.5 ruling rests on**, taken rather than assumed.
        // A live drag frame is a mesh clone plus the pending ops plus a full
        // re-tessellation through the writer, because the `PreviewCache`'s key is
        // the journal generation and an uncommitted drag deliberately does not
        // move it. The figures land in the docs on
        // `PreviewCache::get_with_pending`; the assertion here is loose on
        // purpose — it exists to catch an accidental O(n²), not to pin a machine.
        //
        // Two sizes, so the shape of the cost is visible and not just its value:
        // a scratch frame must be a small constant factor over the committed
        // tessellation at BOTH, or something in the path is superlinear.
        for subdivisions in [1usize, 4, 6] {
            let mut s = MeshSession::new(cube(1.0));
            for _ in 0..subdivisions {
                let faces: Vec<_> = s.mesh().face_ids().collect();
                s.apply(inf_dcc::Op::SubdivideFaces { faces })
                    .expect("subdivides");
            }
            let verts = s.mesh().vert_count();
            let sel = SelectionSet::new(s.generation());
            let pending = PendingDrag::Stroke(StrokeInFlight {
                mode: inf_dcc::SculptMode::Draw,
                radius: 0.3,
                strength: 0.05,
                falloff: inf_dcc::SculptFalloff::Smooth,
                path: vec![DVec3::new(0.5, 0.5, 0.5)],
                last_normal: DVec3::Y,
            });

            let t = std::time::Instant::now();
            let plain = tessellate(s.mesh());
            let committed_ms = t.elapsed().as_secs_f64() * 1000.0;

            // A fresh cache per frame, so this measures the SCRATCH path and not
            // the hit path `the_live_preview_re_tessellates_only_when_the_drag_
            // moves` already pins. The committed base is warmed first, exactly
            // as it is in the product — a drag always follows a committed frame.
            let frames = 5;
            let t = std::time::Instant::now();
            for _ in 0..frames {
                let mut c = PreviewCache::new();
                let _ = c.get(&s);
                let _ = c.get_with_pending(&s, &sel, SelectMode::Face, Some(&pending));
            }
            let warm_and_drag_ms = t.elapsed().as_secs_f64() * 1000.0 / frames as f64;
            // …and the drag frame ALONE, which is the number an author feels:
            // the committed frame is already in the cache by the time they press
            // the button.
            let mut c = PreviewCache::new();
            let _ = c.get(&s);
            let t = std::time::Instant::now();
            for i in 0..frames {
                // A different drag shape each time, or the cache would answer
                // from its key and this would measure a hash.
                let mut p = pending.clone();
                if let PendingDrag::Stroke(st) = &mut p {
                    st.path.push(DVec3::new(0.5 + i as f64 * 0.01, 0.5, 0.5));
                }
                let _ = c.get_with_pending(&s, &sel, SelectMode::Face, Some(&p));
            }
            let drag_ms = t.elapsed().as_secs_f64() * 1000.0 / frames as f64;
            assert_eq!(
                c.scratch_tessellations(),
                0,
                "the measurement fell back to the exporter, so it is measuring \
                 the path this change replaced"
            );

            // **And the PICK, at the same sizes** (audit).
            //
            // The wave re-carried "BVH-backed picking" unspent, on the grounds
            // that "the per-interaction cost is dominated by tessellation, not
            // by picking" — and the numbers it cited were all tessellation's.
            // A prescription refused by measurement has to be refused by a
            // measurement *of the thing prescribed* (the P25 law about
            // inference dressed as measurement), so here it is: `pick` is a
            // linear scan over every element, and this is what that costs at
            // the same three sizes the drag frame is measured at.
            let proj = projector(s.mesh(), 256);
            let picks = 16;
            let t = std::time::Instant::now();
            let mut hits = 0usize;
            for i in 0..picks {
                // Spread across the viewport, so the measurement is not one
                // lucky early-out repeated sixteen times.
                let px = 8.0 + (i % 4) as f32 * 60.0;
                let py = 8.0 + (i / 4) as f32 * 60.0;
                for mode in [SelectMode::Vert, SelectMode::Edge, SelectMode::Face] {
                    if pick(s.mesh(), &proj, mode, px, py).is_some() {
                        hits += 1;
                    }
                }
            }
            let pick_ms = t.elapsed().as_secs_f64() * 1000.0 / (picks * 3) as f64;
            assert!(
                hits > 0,
                "no pick hit anything at {verts} verts, so the timing is an \
                 early-out and not a scan"
            );

            println!(
                "live-drag frame cost: {verts} verts, {} tris | tessellate \
                 {committed_ms:.2} ms | cold (warm+drag) {warm_and_drag_ms:.2} ms | \
                 DISPLACED drag frame {drag_ms:.2} ms | pick {pick_ms:.3} ms",
                plain.indices.len() / 3
            );
            assert!(
                drag_ms < 2_000.0,
                "a scratch frame took {drag_ms:.1} ms on {verts} vertices — that \
                 is not a constant factor over the {committed_ms:.1} ms tessellation"
            );
        }
    }

    /// **The displaced frame IS the exported frame**, to the precision the
    /// approximation admits.
    ///
    /// The whole risk of the fast path is that it diverges from the slow one and
    /// nothing says so — a drag that previews a shape the save will not produce.
    /// So: apply a real transform, tessellate it the long way, displace the
    /// committed geometry the short way, and compare **vertex for vertex**.
    #[test]
    fn a_displaced_frame_matches_the_exported_one() {
        let mut s = MeshSession::new(cube(1.0));
        for _ in 0..2 {
            let faces: Vec<_> = s.mesh().face_ids().collect();
            s.apply(inf_dcc::Op::SubdivideFaces { faces })
                .expect("subdivides");
        }
        // **Round-tripped, so every face is a TRIANGLE** — and that is the
        // finding this fixture encodes rather than works around. The exporter
        // ear-clips an n-gon by its *geometry*, so moving a quad's corners can
        // flip which diagonal it picks; a displaced frame keeps the base's
        // triangulation and a re-export would choose again. On a triangle mesh
        // — which is what imported art is — the triangulation is the identity
        // and the two paths must agree exactly, which is a comparison worth
        // making. The n-gon difference is a preview detail (it is a *stabler*
        // picture, not a wrong one) and the committed frame re-exports anyway.
        let s = MeshSession::new(
            inf_dcc::from_mesh_asset(&inf_dcc::to_mesh_asset(s.mesh(), &Default::default()).0)
                .expect("a kernel export re-opens")
                .mesh,
        );
        let base = tessellate(s.mesh());
        assert_eq!(
            base.sources.len(),
            base.verts.len(),
            "the exporter must hand back a source per written vertex"
        );

        // Move half the vertices, so the comparison is over a mesh that really
        // changed shape rather than one that did not.
        let moved: Vec<inf_dcc::VertId> = s.mesh().vert_ids().step_by(2).collect();
        let mut after = s.mesh().clone();
        inf_dcc::ops::apply(
            &mut after,
            &inf_dcc::Op::TranslateVerts {
                verts: moved,
                delta: [0.0, 0.35, 0.1],
            },
        )
        .expect("translates");

        let slow = tessellate(&after);
        let fast = displace(&base, &after).expect("the fast path is available");
        assert_eq!(fast.verts.len(), slow.verts.len());
        assert_eq!(fast.indices, slow.indices, "the topology must not move");
        for (i, (f, sl)) in fast.verts.iter().zip(&slow.verts).enumerate() {
            for k in 0..3 {
                assert!(
                    (f.position[k] - sl.position[k]).abs() < 1e-6,
                    "vertex {i} position {k}: displaced {} vs exported {}",
                    f.position[k],
                    sl.position[k]
                );
                assert!(
                    (f.normal[k] - sl.normal[k]).abs() < 1e-4,
                    "vertex {i} normal {k}: displaced {:?} vs exported {:?}",
                    f.normal,
                    sl.normal
                );
            }
            assert_eq!(f.uv, sl.uv, "a position move does not move a UV");
        }
        for k in 0..3 {
            assert!((fast.bounds.min[k] - slow.bounds.min[k]).abs() < 1e-6);
            assert!((fast.bounds.max[k] - slow.bounds.max[k]).abs() < 1e-6);
        }
        assert!(
            base.sources.iter().all(|&h| s
                .mesh()
                .corner_normal(h)
                .expect("a live corner")
                .is_some()),
            "an imported mesh's normals are authored — that phase tested the \
             copied-verbatim branch, and the phase below tests the derived one"
        );

        // ── the DERIVED branch ─────────────────────────────────────────────
        //
        // Clear every authored normal, so the exporter falls back to its
        // smooth-fan rule and `displace` must re-accumulate. Without this the
        // test above would pass just as happily if `displace` never touched a
        // normal at all — the mutation that motivated splitting it in two.
        let mut derived = s.mesh().clone();
        for h in derived.half_ids().collect::<Vec<_>>() {
            if derived.is_boundary(h) == Some(false) {
                inf_dcc::ops::apply(
                    &mut derived,
                    &inf_dcc::Op::SetCornerNormal {
                        half: h,
                        normal: None,
                    },
                )
                .expect("clears");
            }
        }
        let base = tessellate(&derived);
        assert!(
            base.sources
                .iter()
                .all(|&h| derived.corner_normal(h).expect("a live corner").is_none()),
            "the fixture still has authored normals"
        );
        let mut after = derived.clone();
        let moved: Vec<inf_dcc::VertId> = derived.vert_ids().step_by(2).collect();
        inf_dcc::ops::apply(
            &mut after,
            &inf_dcc::Op::TranslateVerts {
                verts: moved,
                delta: [0.0, 0.35, 0.1],
            },
        )
        .expect("translates");
        let slow = tessellate(&after);
        let fast = displace(&base, &after).expect("the fast path is available");
        assert_eq!(fast.indices, slow.indices, "still a triangle mesh");
        let mut changed = 0usize;
        for (i, (f, sl)) in fast.verts.iter().zip(&slow.verts).enumerate() {
            for k in 0..3 {
                assert!(
                    (f.normal[k] - sl.normal[k]).abs() < 1e-4,
                    "vertex {i} normal {k}: displaced {:?} vs exported {:?}",
                    f.normal,
                    sl.normal
                );
            }
            if f.normal != base.verts[i].normal {
                changed += 1;
            }
        }
        assert!(
            changed > 0,
            "no derived normal moved — `displace` is not recomputing them"
        );
    }

    /// `displace` refuses — as a value — when it cannot be trusted, and every
    /// caller can always fall back to a full tessellation.
    #[test]
    fn displace_refuses_rather_than_guessing() {
        let s = MeshSession::new(cube(1.0));
        let mut base = tessellate(s.mesh());
        // No source map (an optimized export produces none).
        base.sources.clear();
        assert!(displace(&base, s.mesh()).is_none());
        // …and a source that is no longer live.
        let mut base = tessellate(s.mesh());
        base.sources[0] = inf_dcc::HalfId(9999);
        assert!(displace(&base, s.mesh()).is_none());
    }

    #[test]
    fn the_uv_view_draws_the_layout_the_writer_would_emit() {
        // Every claim the panel makes about this picture, measured: the charts
        // are inside the unit square, a seam is drawn in the seam colour, and the
        // shared selection lights up here as well as in the 3D view.
        let mut m = cube(1.0);
        // Cut the hard edges so there are six charts to look at.
        for h in m.half_ids().collect::<Vec<_>>() {
            if inf_dcc::canonical_edge(&m, h) != Some(h) {
                continue;
            }
            let (Some(Some(f)), Some(Some(g))) =
                (m.face_of(h), m.twin(h).and_then(|t| m.face_of(t)))
            else {
                continue;
            };
            let (Some(a), Some(b)) = (inf_dcc::face_normal(&m, f), inf_dcc::face_normal(&m, g))
            else {
                continue;
            };
            if a.dot(b) < 0.7 {
                m = {
                    let mut s = MeshSession::new(m);
                    s.apply(Op::SetEdgeSeam {
                        half: h,
                        seam: true,
                    })
                    .expect("marks");
                    s.mesh().clone()
                };
            }
        }
        assert_eq!(inf_dcc::seam_count(&m), 12);
        let out = inf_dcc::unwrap(&m).expect("unwraps");
        inf_dcc::ops::apply(&mut m, &out.op).expect("applies");

        let style = UvStyle::default();
        let mut rgba = vec![0u8; 128 * 128 * 4];
        let mut sel = SelectionSet::new(1);
        draw_uv_layout(&mut rgba, 128, &m, &sel, SelectMode::Face, &style);
        let seam_px = rgba
            .chunks_exact(4)
            .filter(|p| p[0..3] == style.seam)
            .count();
        assert!(seam_px > 20, "the seams are not drawn: {seam_px} pixels");
        assert!(
            rgba.chunks_exact(4).any(|p| p[0..3] == style.border),
            "the unit square is not drawn"
        );
        // Every one of this cube's twelve edges is a seam, so the seam pass
        // covers the whole outline: a wire pixel here would mean an edge had
        // been missed by the seam pass.
        assert_eq!(
            rgba.chunks_exact(4)
                .filter(|p| p[0..3] == style.wire)
                .count(),
            0,
            "an edge was drawn as a wire when all twelve are seams"
        );
        // Nothing selected paints nothing hot…
        assert_eq!(
            rgba.chunks_exact(4)
                .filter(|p| p[0..3] == style.selected)
                .count(),
            0
        );
        // …and the SAME selection the 3D view holds lights up here.
        let f = m.face_ids().next().expect("a face");
        sel.set_face(f, true);
        let mut hot = vec![0u8; 128 * 128 * 4];
        draw_uv_layout(&mut hot, 128, &m, &sel, SelectMode::Face, &style);
        assert_ne!(hot, rgba, "selecting a face must change the UV picture");
        assert!(
            hot.chunks_exact(4).any(|p| p[0..3] == style.selected),
            "a selected face's outline must beat the seam colour: the selection \
             is what the author is pointing at"
        );

        // An undersized buffer is refused, like every other compositor here.
        let mut small = vec![9u8; 16];
        let before = small.clone();
        draw_uv_layout(&mut small, 128, &m, &sel, SelectMode::Face, &style);
        assert_eq!(small, before);
    }

    #[test]
    fn a_mesh_with_no_uvs_still_draws_a_frame_rather_than_nothing() {
        // Opening the UV view before unwrapping is the normal first move, and it
        // must show the empty square rather than a blank panel that reads as a
        // broken render.
        let m = cube(1.0);
        let style = UvStyle::default();
        let mut rgba = vec![0u8; 64 * 64 * 4];
        draw_uv_layout(
            &mut rgba,
            64,
            &m,
            &SelectionSet::new(1),
            SelectMode::Vert,
            &style,
        );
        assert!(
            rgba.chunks_exact(4).any(|p| p[0..3] == style.border),
            "the unit square must be visible even with nothing unwrapped"
        );
        // NOT `any(wire)`: `cube()` gives every face the full unit square, so all
        // six outlines land exactly on the border and the frame (drawn last)
        // covers them. That is the honest picture of an un-unwrapped primitive —
        // six charts stacked on top of each other — and it is what the panel's
        // "unwrap first" prompt is for.
        assert!(rgba.chunks_exact(4).all(|p| p[3] == 255));
    }

    #[test]
    fn a_dropped_mesh_becomes_a_second_component_through_the_journal() {
        let mut s = MeshSession::new(cube(1.0));
        let before = s.mesh().face_count();
        let r = merge_into(&mut s, &cube(1.0), DVec3::new(4.0, 0.0, 0.0), None).expect("merges");
        assert_eq!(r.faces, 6);
        assert_eq!(r.verts, 8);
        // A rigid merge of a slot-less cube appends no slot and re-weights
        // nothing — the pre-P24.3 shape, unchanged.
        assert_eq!(r.slots_appended, 0);
        assert_eq!(r.verts_reweighted, 0);
        assert_eq!(s.mesh().face_count(), before + 6);
        assert_eq!(s.mesh().vert_count(), 16);
        assert_eq!(inf_dcc::validate(s.mesh()), Ok(()));
        // The whole merge is journal entries: one undo per op, and replay
        // reproduces it byte for byte.
        assert_eq!(s.ops().len(), 8 + 6);
        let replayed =
            MeshSession::replay(s.base(), &s.ops()[..s.cursor()]).expect("journalled ops replay");
        assert_eq!(replayed.encoded(), s.mesh().encoded());
        // And it really is a SECOND shell, not welded to the first.
        let mut sel = SelectionSet::new(s.generation());
        sel.set_face(s.mesh().face_ids().next().expect("a face"), true);
        sel.select_linked(s.mesh(), SelectMode::Face);
        assert_eq!(
            sel.len(SelectMode::Face),
            6,
            "linked reaches one shell only"
        );
    }

    // ── P24.3: the auto-fit hop ─────────────────────────────────

    /// **The conversion is a triangle per index triple, and the positions are the
    /// mesh's** — the two claims `triangle_soup` rests on.
    ///
    /// A cube is 6 quads, which `to_mesh_asset` triangulates to 12 triangles and
    /// 36 indices. Both numbers are asserted, because a conversion that dropped
    /// every other triple would still produce "some triangles".
    #[test]
    fn the_triangle_soup_is_one_tri_per_index_triple() {
        let geo = tessellate(&cube(2.0));
        assert_eq!(geo.indices.len(), 36, "a cube tessellates to 12 triangles");
        let tris = triangle_soup(&geo);
        assert_eq!(tris.len(), geo.indices.len() / 3);
        assert_eq!(tris.len(), 12);

        // Every corner the soup names is a corner of the mesh — so the positions
        // came from `verts`, not from an off-by-one into it.
        let (lo, hi) = (geo.bounds.min, geo.bounds.max);
        for t in &tris {
            for v in [t.a, t.b, t.c] {
                for k in 0..3 {
                    assert!(
                        v[k] >= lo[k] as f64 - 1e-9 && v[k] <= hi[k] as f64 + 1e-9,
                        "{v:?} is outside the mesh's own bounds {lo:?}..{hi:?}"
                    );
                }
            }
        }
        // …and no triangle is degenerate, which an index triple read as
        // `[i, i, i]` would be.
        for t in &tris {
            assert!(
                (t.b - t.a).cross(t.c - t.a).length() > 1e-12,
                "degenerate triangle {t:?}"
            );
        }
    }

    /// **A closest-point query through the converted BVH lands where the GEOMETRY
    /// says** — checked against a hand-computed answer, not against the BVH's own
    /// opinion.
    ///
    /// The fixture is `cube(2.0)`, whose bounds the tessellation reports
    /// independently of anything the hierarchy does. A point 10 m above the
    /// centre must project onto the top face: `y` at the bound, `x`/`z`
    /// unchanged at 0, distance exactly `10 − max_y`. A soup that mangled
    /// coordinates — swapped axes, kept `f32` precision, read the wrong vertex —
    /// fails one of the three.
    #[test]
    fn a_closest_point_through_the_converted_soup_is_hand_checkable() {
        let geo = tessellate(&cube(2.0));
        let top = geo.bounds.max[1] as f64;
        let bvh = inf_dcc::Bvh::new(triangle_soup(&geo));
        assert!(!bvh.is_empty());

        let q = DVec3::new(0.0, 10.0, 0.0);
        let hit = bvh.closest_point(q).expect("a non-empty hierarchy answers");
        assert!(
            (hit.point.y - top).abs() < 1e-9,
            "closest point y = {} but the top face is at {top}",
            hit.point.y
        );
        assert!(
            hit.point.x.abs() < 1e-9 && hit.point.z.abs() < 1e-9,
            "{:?}",
            hit.point
        );
        assert!(
            ((10.0 - top) - (q - hit.point).length()).abs() < 1e-9,
            "distance {} is not 10 - {top}",
            (q - hit.point).length()
        );

        // …and the hierarchy really encloses the solid, which is what
        // `FitReport::joints_inside` is computed from.
        assert!(bvh.contains(DVec3::ZERO), "the cube's centre is inside it");
        assert!(!bvh.contains(q));
    }

    /// The soup drops a trailing partial triangle rather than inventing a vertex
    /// for it — and an out-of-range index takes its whole triangle with it.
    #[test]
    fn a_malformed_index_list_loses_only_its_bad_triangles() {
        let mut geo = tessellate(&cube(2.0));
        geo.indices.push(0);
        geo.indices.push(1);
        assert_eq!(
            triangle_soup(&geo).len(),
            12,
            "a trailing pair must not become a triangle"
        );
        geo.indices.truncate(36);
        geo.indices[0] = u32::MAX;
        assert_eq!(
            triangle_soup(&geo).len(),
            11,
            "an out-of-range index must drop its triangle, not panic or fabricate"
        );
    }

    // ── P24.3: the merge carries the rig ──────────────────────────────────

    /// A skinned cube bound to `joints` bones, every vertex on `joint`.
    fn skinned_cube(joints: u32, joint: u16) -> Mesh {
        let mut s = MeshSession::new(cube(1.0));
        s.apply(Op::BindSkin {
            skeleton: None,
            joints,
        })
        .expect("binds");
        let weights: Vec<(VertId, inf_dcc::VertWeights)> = s
            .mesh()
            .vert_ids()
            .map(|v| (v, inf_dcc::VertWeights::from_pairs(&[(joint, 1.0)])))
            .collect();
        s.apply(Op::AssignWeights { weights }).expect("weights");
        s.mesh().clone()
    }

    /// A cube whose faces are painted with two named material slots.
    fn slotted_cube(names: &[&str]) -> Mesh {
        let mut s = MeshSession::new(cube(1.0));
        s.apply(Op::AddMaterialSlots {
            names: names.iter().map(|n| n.to_string()).collect(),
        })
        .expect("slots");
        let faces: Vec<FaceId> = s.mesh().face_ids().collect();
        for (i, f) in faces.iter().enumerate() {
            s.apply(Op::SetFaceSlot {
                face: *f,
                slot: Some((i % names.len()) as u32),
            })
            .expect("paints");
        }
        s.mesh().clone()
    }

    /// **The headline gate for modular assembly**: merging a skinned arm onto a
    /// skinned torso re-indexes the ARM's weights and leaves the TORSO's exactly
    /// where they were.
    ///
    /// The torso is weighted to joint 1 of a 2-joint rig; the arm to joint 1 of
    /// its own 2-joint rig, which after a merge is joint 3. A merge that dropped
    /// the remap would leave the arm on joint 1 — deforming with the torso, and
    /// looking plausible while doing it, which is why the assertion is on the
    /// weights and not on a count.
    #[test]
    fn merging_a_skinned_part_reindexes_its_weights_and_never_the_bases() {
        let mut s = MeshSession::new(skinned_cube(2, 1));
        let torso_verts: Vec<VertId> = s.mesh().vert_ids().collect();

        let r = merge_into(
            &mut s,
            &skinned_cube(2, 1),
            DVec3::new(4.0, 0.0, 0.0),
            Some(MergeRig {
                skeleton: None,
                joints: 4,
                joint_offset: 2,
            }),
        )
        .expect("merges");
        assert_eq!(r.verts, 8);
        assert_eq!(r.verts_reweighted, 8, "every merged vertex was re-indexed");

        // The base's weights are byte-for-byte what they were — the append-only
        // law, which is what lets an IK chain authored on the torso survive.
        for v in &torso_verts {
            let w = s.mesh().vert_weights(*v).expect("live");
            assert_eq!(w.influences(), vec![(1, 1.0)], "a base vertex moved");
        }
        // …and the merged part rides joint 1 + 2 = 3.
        let merged: Vec<VertId> = s
            .mesh()
            .vert_ids()
            .filter(|v| !torso_verts.contains(v))
            .collect();
        assert_eq!(merged.len(), 8);
        for v in merged {
            let w = s.mesh().vert_weights(v).expect("live");
            assert_eq!(
                w.influences(),
                vec![(3, 1.0)],
                "the merged part kept the base's joint indices"
            );
        }
        assert_eq!(
            s.mesh().skin_binding().map(|b| b.joints),
            Some(4),
            "the binding must widen to the merged skeleton"
        );
        assert_eq!(inf_dcc::validate(s.mesh()), Ok(()));
    }

    /// The rig rides the JOURNAL, not a side channel: replaying the recorded ops
    /// reproduces the merged document byte for byte, weights included.
    #[test]
    fn a_rig_carrying_merge_replays_byte_for_byte() {
        let mut s = MeshSession::new(skinned_cube(2, 1));
        merge_into(
            &mut s,
            &skinned_cube(2, 1),
            DVec3::new(4.0, 0.0, 0.0),
            Some(MergeRig {
                skeleton: None,
                joints: 4,
                joint_offset: 2,
            }),
        )
        .expect("merges");
        let replayed =
            MeshSession::replay(s.base(), &s.ops()[..s.cursor()]).expect("journalled ops replay");
        assert_eq!(replayed.encoded(), s.mesh().encoded());
        // …and undo peels the whole merge off, back to the torso alone.
        let ops = s.ops().len();
        for _ in 0..ops {
            s.undo();
        }
        assert_eq!(s.mesh().vert_count(), 8);
    }

    /// **Material slots come with the part**, and a name both parts already use
    /// is REUSED rather than duplicated — the policy this layer owns (the kernel
    /// allows duplicates on purpose).
    #[test]
    fn merging_carries_material_slots_and_reuses_matching_names() {
        let mut s = MeshSession::new(slotted_cube(&["Default", "Trim"]));
        let r = merge_into(
            &mut s,
            &slotted_cube(&["Default", "Glass"]),
            DVec3::new(4.0, 0.0, 0.0),
            None,
        )
        .expect("merges");
        assert_eq!(r.slots_reused, 1, "\"Default\" already existed");
        assert_eq!(r.slots_appended, 1, "\"Glass\" did not");
        assert_eq!(s.mesh().material_slots(), ["Default", "Trim", "Glass"]);

        // The incoming faces landed on the RIGHT slots: its slot 0 ("Default")
        // maps to 0 and its slot 1 ("Glass") to 2 — never to 1 ("Trim"), which
        // is the mis-paint an offset-only remap would produce.
        let painted: Vec<Option<u32>> = s
            .mesh()
            .face_ids()
            .filter_map(|f| s.mesh().face_slot(f))
            .collect();
        assert_eq!(painted.len(), 12);
        assert!(
            painted.contains(&Some(2)),
            "the appended slot was never used"
        );
        // Exactly the six base faces use slot 1; nothing merged does.
        assert_eq!(
            painted.iter().filter(|s| **s == Some(1)).count(),
            3,
            "the merged part was painted with the BASE's second material"
        );
    }

    /// A merge with no rig leaves every new vertex rigid and the binding alone —
    /// the prop-kit path, unchanged from P23.4.
    #[test]
    fn a_rigid_merge_never_touches_the_skin() {
        let mut s = MeshSession::new(skinned_cube(2, 1));
        let before = s.mesh().skin_binding();
        let r = merge_into(&mut s, &cube(1.0), DVec3::new(4.0, 0.0, 0.0), None).expect("merges");
        assert_eq!(r.verts_reweighted, 0);
        assert_eq!(s.mesh().skin_binding(), before);
        assert_eq!(inf_dcc::validate(s.mesh()), Ok(()));
    }

    // ── P24.3: Op::Mirror joins the joints ────────────────────────────────

    /// A 4-joint rig: root, then `upper_arm_l` / `upper_arm_r` and a `hand_l` —
    /// deliberately missing `hand_r`, so the unmatched arm can be tested too.
    fn sided_skeleton(with_hand_r: bool) -> inf_anim::Skeleton {
        let j = |name: &str, parent: Option<u16>| inf_anim::Joint {
            name: name.into(),
            parent,
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
            local_bind: inf_anim::JointTransform::IDENTITY,
        };
        let mut joints = vec![
            j("hips", None),
            j("upper_arm_l", Some(0)),
            j("upper_arm_r", Some(0)),
            j("hand_l", Some(1)),
        ];
        if with_hand_r {
            joints.push(j("hand_r", Some(2)));
        }
        inf_anim::Skeleton::new(joints).unwrap()
    }

    /// **The headline gate for the mirror**: a mirrored left arm ends up
    /// weighted to the RIGHT arm's joints.
    ///
    /// This is P24.2's ledgered `Op::Mirror` defect, closed one level up from the
    /// kernel — which still cannot do it, because `SkinBinding` carries a joint
    /// COUNT and no names.
    #[test]
    fn mirroring_a_skinned_mesh_swaps_its_left_and_right_joints() {
        let sk = sided_skeleton(true);
        // A cube entirely on `upper_arm_l` (joint 1), off the mirror plane.
        let mut s = MeshSession::new(skinned_cube(5, 1));
        let before: Vec<VertId> = s.mesh().vert_ids().collect();
        // Move it off x = 0 so the mirror really produces new vertices.
        s.apply(Op::TranslateVerts {
            verts: before.clone(),
            delta: [3.0, 0.0, 0.0],
        })
        .expect("translates");

        let r =
            mirror_with_joints(&mut s, inf_dcc::MirrorAxis::X, 0.0, Some(&sk)).expect("mirrors");
        assert_eq!(r.verts, 8, "the mirror produced no copies");
        assert!(!r.weights_unmapped);
        assert_eq!(r.joints_swapped, 8, "every mirrored vertex moved side");

        // The ORIGINAL vertices are untouched; the copies are on joint 2.
        for v in &before {
            assert_eq!(
                s.mesh().vert_weights(*v).unwrap().influences(),
                vec![(1, 1.0)],
                "the mirror re-weighted the SOURCE"
            );
        }
        let copies: Vec<VertId> = s
            .mesh()
            .vert_ids()
            .filter(|v| !before.contains(v))
            .collect();
        assert_eq!(copies.len(), 8);
        for v in copies {
            assert_eq!(
                s.mesh().vert_weights(v).unwrap().influences(),
                vec![(2, 1.0)],
                "a mirrored vertex is still weighted to the LEFT arm"
            );
        }
        assert_eq!(inf_dcc::validate(s.mesh()), Ok(()));
    }

    /// …and the swap rides the JOURNAL as values, so a later build with a
    /// different pairing rule cannot rewrite a saved session (the `Op::Unwrap`
    /// doctrine).
    #[test]
    fn the_mirror_joint_swap_is_journalled_as_values() {
        let sk = sided_skeleton(true);
        let mut s = MeshSession::new(skinned_cube(5, 1));
        let all: Vec<VertId> = s.mesh().vert_ids().collect();
        s.apply(Op::TranslateVerts {
            verts: all,
            delta: [3.0, 0.0, 0.0],
        })
        .unwrap();
        mirror_with_joints(&mut s, inf_dcc::MirrorAxis::X, 0.0, Some(&sk)).unwrap();

        // Two ops, in this order — a Mirror and an AssignWeights carrying the
        // swapped table.
        let tail = &s.ops()[s.ops().len() - 2..];
        assert!(matches!(tail[0], Op::Mirror { .. }));
        match &tail[1] {
            Op::AssignWeights { weights } => {
                assert_eq!(weights.len(), 8);
                assert!(
                    weights
                        .iter()
                        .all(|(_, w)| w.influences() == vec![(2, 1.0)]),
                    "the journalled weights are not the swapped ones"
                );
            }
            other => panic!("expected AssignWeights, got {other:?}"),
        }
        let replayed = MeshSession::replay(s.base(), &s.ops()[..s.cursor()]).expect("replays");
        assert_eq!(replayed.encoded(), s.mesh().encoded());
    }

    /// A sided joint with no twin **refuses**, by value, naming the joints —
    /// mirroring it would weight the copy to the wrong side and look right.
    #[test]
    fn mirroring_against_a_half_sided_rig_refuses_by_name() {
        let sk = sided_skeleton(false); // no `hand_r`
        let mut s = MeshSession::new(skinned_cube(5, 1));
        match mirror_with_joints(&mut s, inf_dcc::MirrorAxis::X, 0.0, Some(&sk)) {
            Err(MirrorRigError::UnmatchedJoints(names)) => {
                assert_eq!(names, vec!["hand_l".to_string()]);
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // A refusal must not have applied the mirror.
        assert_eq!(s.mesh().vert_count(), 8);
        assert!(s.ops().is_empty());
    }

    /// A **rigid** mesh takes the plain-mirror path and says so — byte for byte
    /// what `Op::Mirror` alone did before P24.3.
    #[test]
    fn mirroring_a_rigid_mesh_is_the_old_behaviour_and_is_reported() {
        let mut plain = MeshSession::new(cube(1.0));
        let all: Vec<VertId> = plain.mesh().vert_ids().collect();
        plain
            .apply(Op::TranslateVerts {
                verts: all.clone(),
                delta: [3.0, 0.0, 0.0],
            })
            .unwrap();
        let mut viaop = MeshSession::new(cube(1.0));
        viaop
            .apply(Op::TranslateVerts {
                verts: all,
                delta: [3.0, 0.0, 0.0],
            })
            .unwrap();

        let r = mirror_with_joints(&mut plain, inf_dcc::MirrorAxis::X, 0.0, None).expect("mirrors");
        viaop
            .apply(Op::Mirror {
                axis: inf_dcc::MirrorAxis::X,
                coord: 0.0,
            })
            .unwrap();
        assert!(r.weights_unmapped, "a rigid mirror maps no joints");
        assert_eq!(r.joints_swapped, 0);
        assert_eq!(
            plain.mesh().encoded(),
            viaop.mesh().encoded(),
            "the rigid path diverged from Op::Mirror alone"
        );
        assert_eq!(plain.ops().len(), viaop.ops().len());
    }

    /// A skinned mesh whose skeleton the caller could NOT resolve is mirrored
    /// with un-swapped weights and **reports it** — the pre-P24.3 behaviour, made
    /// visible rather than removed.
    #[test]
    fn a_skinned_mirror_without_a_skeleton_reports_unmapped_weights() {
        let mut s = MeshSession::new(skinned_cube(5, 1));
        let all: Vec<VertId> = s.mesh().vert_ids().collect();
        s.apply(Op::TranslateVerts {
            verts: all,
            delta: [3.0, 0.0, 0.0],
        })
        .unwrap();
        let r = mirror_with_joints(&mut s, inf_dcc::MirrorAxis::X, 0.0, None).expect("mirrors");
        assert!(
            r.weights_unmapped,
            "the caller must be told the copies kept the source's joints"
        );
        assert_eq!(r.joints_swapped, 0);
    }
}
