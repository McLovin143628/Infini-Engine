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
#[derive(Default)]
pub struct PreviewCache {
    stamp: Option<u64>,
    geometry: Option<std::sync::Arc<EditGeometry>>,
    mesh: Option<std::sync::Arc<Mesh>>,
    tessellations: u64,
}

impl PreviewCache {
    pub fn new() -> Self {
        Self::default()
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
        (
            self.geometry.clone().expect("just filled"),
            self.mesh.clone().expect("just filled"),
        )
    }
}

// ── drag-and-drop modularity, v1 ───────────────────────────────────────────

/// Append another mesh into this document as a **new connected component**,
/// offset by `offset` metres — the dcc-vision seed: assembling a prop out of kit
/// pieces without leaving the Model Editor.
///
/// **What v1 does**: applies an `AddVertex` per incoming vertex and an `AddFace`
/// per incoming face, so the whole merge is ordinary journal entries — undo peels
/// it off, replay reproduces it byte for byte, and nothing special-cases it
/// anywhere. It has to be done against a live session rather than precomputed as
/// a `Vec<Op>`, because the ids the target mints come from its arena's free list;
/// that is the same property that makes replay work, and it is why there is no
/// pure `merge_ops` beside this.
///
/// **What v1 does NOT do**, stated rather than left to be discovered:
/// * **No welding or snapping.** The dropped part is a separate shell that
///   happens to be nearby. Joining it is the author's `MergeVerts`.
/// * **No material slots.** Every dropped face lands on slot `None`, because the
///   slot *table* is not an `Op` — remapping the incoming names into this mesh's
///   table would be a mutation outside the journal, and a journal with a hole in
///   it is worse than a merge that loses a material assignment.
/// * **No instancing.** The geometry is copied, so editing the source asset
///   afterwards does not update what was dropped. That is the difference between
///   assembling and referencing, and referencing is what the *scene* is for.
/// * **No refusal recovery.** A merge that fails partway has already journalled
///   what it applied; the caller undoes it. (Every op here is `AddVertex` or
///   `AddFace` onto fresh vertices, so the only way to fail is a non-finite
///   offset, which the caller rejects first.)
pub fn merge_into(
    session: &mut inf_dcc::MeshSession,
    incoming: &Mesh,
    offset: DVec3,
) -> Result<usize, inf_dcc::OpError> {
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
    let mut faces = 0usize;
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
        session.apply(Op::AddFace {
            verts,
            corners,
            slot: None,
        })?;
        faces += 1;
    }
    Ok(faces)
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

    #[test]
    fn a_dropped_mesh_becomes_a_second_component_through_the_journal() {
        let mut s = MeshSession::new(cube(1.0));
        let before = s.mesh().face_count();
        let faces = merge_into(&mut s, &cube(1.0), DVec3::new(4.0, 0.0, 0.0)).expect("merges");
        assert_eq!(faces, 6);
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
}
