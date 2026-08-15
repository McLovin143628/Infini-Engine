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
}

/// Tessellate the kernel mesh for drawing — through the real writer, so the
/// preview and the save agree by construction (see the module docs).
pub fn tessellate(mesh: &Mesh) -> EditGeometry {
    let (asset, _) = inf_dcc::to_mesh_asset(mesh, &inf_dcc::ExportOptions::default());
    let (verts, indices) = flatten(&asset);
    EditGeometry {
        bounds: asset.bounds,
        verts,
        indices,
    }
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

/// Turn a transform into journal ops, honouring soft-select weights.
///
/// **One op per distinct weight**, which is the `SoftTranslate` shape P23.4
/// established and the reason `Op::RotateVerts` / `Op::ScaleVerts` do not carry a
/// weight table (see [`inf_dcc::xform`]). `soft` is `Some((radius, falloff))` for
/// a soft transform and `None` for a hard one; a hard transform therefore emits
/// exactly one op, over the resolved vertices in ascending id order.
///
/// The weight blends toward the identity, per kind:
/// * translate — the delta is scaled;
/// * rotate — the **angle** is scaled, so every vertex still travels on a circle
///   about the pivot rather than being dragged off one by a scaled chord;
/// * scale — the factor is lerped from `1`, so weight `0` is "unchanged" rather
///   than "collapsed onto the pivot".
pub fn transform_ops(
    mesh: &Mesh,
    selection: &SelectionSet,
    mode: SelectMode,
    pivot: DVec3,
    xform: VertTransform,
    soft: Option<(f64, inf_terrain::Falloff)>,
) -> Vec<Op> {
    let groups: std::collections::BTreeMap<u64, Vec<VertId>> = match soft {
        Some((radius, falloff)) => {
            let mut by_weight: std::collections::BTreeMap<u64, Vec<VertId>> =
                std::collections::BTreeMap::new();
            for (v, w) in selection.soft_weights(mesh, mode, radius, falloff) {
                let q = quantize_weight(w);
                if q <= 0.0 {
                    continue;
                }
                by_weight.entry(q.to_bits()).or_default().push(v);
            }
            by_weight
        }
        None => {
            let verts: Vec<VertId> = selection.resolved_verts(mesh, mode).into_iter().collect();
            if verts.is_empty() {
                Default::default()
            } else {
                [(1.0f64.to_bits(), verts)].into_iter().collect()
            }
        }
    };

    groups
        .into_iter()
        .map(|(bits, verts)| {
            let w = f64::from_bits(bits);
            match xform {
                VertTransform::Translate(d) => Op::TranslateVerts {
                    verts,
                    delta: (d * w).to_array(),
                },
                VertTransform::Rotate { axis, radians } => Op::RotateVerts {
                    verts,
                    pivot: pivot.to_array(),
                    axis: axis.to_array(),
                    radians: radians * w,
                },
                VertTransform::Scale(f) => Op::ScaleVerts {
                    verts,
                    pivot: pivot.to_array(),
                    factor: (DVec3::ONE + (f - DVec3::ONE) * w).to_array(),
                },
            }
        })
        .collect()
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
    mode: GizmoMode,
    px: f32,
    py: f32,
) -> Option<GizmoAxis> {
    let p = Vec3::new(pivot.x as f32, pivot.y as f32, pivot.z as f32);
    inf_render::gizmo::pick_axis(
        mode,
        p,
        glam::Quat::IDENTITY,
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
    scratch_tessellations: u64,
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

    /// How many times the **scratch** tessellator has run — the uncommitted half
    /// of the cache, counted separately so a gate can say "an orbit during a
    /// drag costs nothing extra" rather than assuming it.
    pub fn scratch_tessellations(&self) -> u64 {
        self.scratch_tessellations
    }

    /// The geometry and mesh a frame should draw, given a drag that has not been
    /// committed yet.
    ///
    /// # The side channel, and why it is a full re-tessellation
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
    /// finding and the one that says the scratch channel adds nothing of its own.
    ///
    /// Against the P23.2a budget — 0.09 ms to render at 256² and ~0.34 ms to
    /// encode — a drag frame on a small model is comfortable and a **1.5 k-vertex
    /// model is already the dominant cost at ~9 ms**, i.e. about 30 fps in a
    /// debug build and fine in release. Stated rather than hidden: **this path
    /// will not hold an interactive rate on a model of a hundred thousand
    /// vertices.** The next lever is displacing the cached vertex buffer in place
    /// rather than re-running the exporter, which needs the writer to expose its
    /// corner→vertex map — a remainder, not a defect. What is *not* on the table
    /// is displacing on the GPU: the CPU picker could not see it, and the panel
    /// would go back to highlighting one thing and hitting another.
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
            self.scratch_key = Some(key);
            self.scratch_geometry = Some(std::sync::Arc::new(tessellate(&mesh)));
            self.scratch_mesh = Some(std::sync::Arc::new(mesh));
            self.scratch_tessellations += 1;
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
                    pick_gizmo(&proj, view, pivot, mode, s.x, s.y),
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
        assert!(pick_gizmo(&proj, view, pivot, GizmoMode::Rotate, ring.x, ring.y).is_some());
        for mode in [GizmoMode::Translate, GizmoMode::Rotate, GizmoMode::Scale] {
            assert_eq!(
                pick_gizmo(&proj, view, pivot, mode, 2.0, 2.0),
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

    #[test]
    fn a_soft_transform_is_one_op_per_weight_and_blends_toward_the_identity() {
        let m = cube(2.0);
        let mut sel = SelectionSet::new(1);
        sel.set_vert(m.vert_ids().next().expect("a vertex"), true);
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
            assert!(ops.len() > 1, "the neighbourhood moves too: {ops:?}");
            // **The CAP, not merely "more than one".** The old assertion was
            // `> 1` and passed just as happily on the 105-op drag the audit
            // measured — it named the defect as a feature.
            assert!(
                ops.len() <= SOFT_WEIGHT_STEPS as usize,
                "a soft drag journalled {} ops; the quantization caps it at {}",
                ops.len(),
                SOFT_WEIGHT_STEPS
            );
            for op in &ops {
                match op {
                    // The weight scales the DELTA…
                    Op::TranslateVerts { delta, .. } => {
                        assert!(delta[1] > 0.0 && delta[1] <= 1.0, "{delta:?}")
                    }
                    // …the ANGLE (so every vertex still travels on a circle)…
                    Op::RotateVerts {
                        radians, pivot: p, ..
                    } => {
                        assert!(*radians > 0.0 && *radians <= 0.5, "{radians}");
                        assert_eq!(*p, pivot.to_array(), "every op shares the pivot");
                    }
                    // …and the FACTOR lerps from 1, not from 0.
                    Op::ScaleVerts { factor, .. } => {
                        assert!(factor[0] > 1.0 && factor[0] <= 2.0, "{factor:?}")
                    }
                    other => panic!("{other:?}"),
                }
            }
            // Exactly one group is at full weight — the selection itself.
            let full = ops
                .iter()
                .filter(|o| match o {
                    Op::TranslateVerts { delta, .. } => delta[1] == 1.0,
                    Op::RotateVerts { radians, .. } => *radians == 0.5,
                    Op::ScaleVerts { factor, .. } => factor[0] == 2.0,
                    _ => false,
                })
                .count();
            assert_eq!(full, 1, "{ops:?}");
        }
    }

    #[test]
    fn a_soft_drag_over_a_dense_selection_stays_inside_the_op_cap() {
        // The measurement the cap is sized against: a 289-vertex plane with a 3 m
        // radius produced **105 ops from one drag** before quantization, which is
        // ~3 full mesh snapshots at `CHECKPOINT_INTERVAL = 32` and evicts the
        // whole 8-slot checkpoint history — per drag.
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
        println!(
            "soft drag: {} verts, {} ops (cap {})",
            m.vert_count(),
            ops.len(),
            SOFT_WEIGHT_STEPS
        );
        assert!(
            ops.len() <= SOFT_WEIGHT_STEPS as usize,
            "{} ops from one drag",
            ops.len()
        );
        // …and the quantization did not throw the neighbourhood away.
        let moved: usize = ops
            .iter()
            .map(|o| match o {
                Op::TranslateVerts { verts, .. } => verts.len(),
                _ => 0,
            })
            .sum();
        assert!(moved > 100, "only {moved} vertices move");
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
        assert_eq!(cache.scratch_tessellations(), 1, "ten frames, one scratch");
        assert_eq!(
            cache.tessellations(),
            0,
            "and the committed path is untouched"
        );

        stroke.path.push(DVec3::new(0.6, 0.5, 0.5));
        let moved = PendingDrag::Stroke(stroke);
        let (geo1, _) = cache.get_with_pending(&s, &sel, SelectMode::Face, Some(&moved));
        assert!(!std::sync::Arc::ptr_eq(&geo1, &geo0));
        assert_eq!(cache.scratch_tessellations(), 2);

        // Pointer-up: back to the committed cache, and the scratch is released.
        let (committed, _) = cache.get_with_pending(&s, &sel, SelectMode::Face, None);
        assert_eq!(cache.tessellations(), 1);
        assert_eq!(
            cache.scratch_tessellations(),
            2,
            "no extra scratch on release"
        );
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
        let tess = cache.scratch_tessellations();

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
            cache.scratch_tessellations(),
            tess + 1,
            "…and it really re-tessellated rather than only re-keying"
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
        let tess = cache.scratch_tessellations();
        cache.get_with_pending(&s, &other, SelectMode::Face, Some(&pending));
        assert_eq!(cache.upload_stamp(), third, "an orbit mid-drag is free");
        assert_eq!(cache.scratch_tessellations(), tess);
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
        for subdivisions in [1usize, 4] {
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
            // moves` already pins.
            let frames = 5;
            let t = std::time::Instant::now();
            for _ in 0..frames {
                let mut c = PreviewCache::new();
                let _ = c.get_with_pending(&s, &sel, SelectMode::Face, Some(&pending));
            }
            let scratch_ms = t.elapsed().as_secs_f64() * 1000.0 / frames as f64;

            println!(
                "live-drag frame cost: {verts} verts, {} tris | tessellate \
                 {committed_ms:.2} ms | clone+apply+tessellate {scratch_ms:.2} ms",
                plain.indices.len() / 3
            );
            assert!(
                scratch_ms < 2_000.0,
                "a scratch frame took {scratch_ms:.1} ms on {verts} vertices — that \
                 is not a constant factor over the {committed_ms:.1} ms tessellation"
            );
        }
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
