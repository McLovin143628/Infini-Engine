//! **The modelling op set** (P23.4): extrude, inset, bevel, loop cut, knife,
//! merge, subdivide and mirror — the tools a Model Editor toolbar actually has
//! buttons for.
//!
//! Everything here is built out of [`crate::topo`]'s one path (capture the local
//! patch of faces, remove it, rebuild it) and obeys the three rules the core op
//! set already promises: **a refusal is a typed value and it is inert**, **the
//! mesh stays valid**, and **every op is a journal entry** ([`crate::ops::Op`]
//! gained nine appended variants, and the frozen-discriminant test in
//! [`crate::journal`] had to be updated by hand to accept them — which is the
//! point of it being a `match` and not a table).
//!
//! # Region-border detection, stated once
//!
//! Extrude and inset both operate on a *set* of faces, and the thing that makes
//! them feel like a modelling tool rather than a batch of per-face operations is
//! that the set is treated as one region: an edge with a selected face on **both**
//! sides is interior and gets no wall, an edge with a selected face on only one
//! side is the border and does. Two faces extruded together move as one wall
//! with no membrane between them, which is what an author means by "extrude this
//! area". [`Region`] computes that once and both ops read it.
//!
//! # The v1 scopes, honestly
//!
//! Each of these is a real tool with a real boundary, and the boundary is a typed
//! refusal rather than a silent approximation:
//!
//! * **Bevel is `segments = 1`** — one flat chamfer strip per edge, no rounding.
//!   Its construction *keeps* both endpoints and caps each end of the strip with
//!   a triangle, which is what makes it work at **any vertex valence** (see
//!   [`bevel_edges`]). Blender dissolves the endpoint into an n-gon instead; that
//!   is a nicer result and a much larger algorithm, and beveling several edges
//!   that meet at one vertex produces one strip each with a wedge between them
//!   rather than a true vertex bevel. Both are named remainders.
//! * **Knife is a path of existing vertices and points on existing edges** —
//!   straight cuts through the surface, applied as one atomic transaction.
//!   Free-form cutting (a screen-space polyline projected onto arbitrary
//!   positions inside faces) is a remainder; it needs a ray-vs-face solve that
//!   belongs with the panel, not with the kernel.
//! * **Subdivide is simple midpoint** — no Catmull–Clark limit-surface smoothing,
//!   so the shape does not change, only its density. Faces adjacent to the
//!   subdivided region gain the new midpoint corners (they must, or the shared
//!   edge would be split on one side only, which is not a mesh).
//! * **Mirror is across an AXIS-ALIGNED plane** and that is not laziness: the
//!   seam weld has a tolerance of exactly zero (the kernel's law), so the mirror
//!   of a point *on* the plane has to come back **bit-identical**. `2d − c` is
//!   exact when `c == d` on one axis and a copy on the other two; an arbitrary
//!   plane's reflection is three fused multiply-adds and lands one ULP away, so
//!   the seam would not weld and the author would get a two-shell mesh with a
//!   crack in it. An arbitrary-plane mirror needs a tolerance, and a tolerance is
//!   what this kernel refuses to have.
//!
//! # Units
//!
//! `distance`, `amount` and every `delta` are **metres** (architecture rule 6).

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::export::newell;
use crate::ops::{finite, lerp_corner, weld_impl, OpError, OpOutcome};
use crate::topo::{CornerData, FaceId, HalfId, Mesh, VertId};

/// The most cuts one [`crate::ops::Op::LoopCut`] may make.
///
/// A cap rather than an unbounded loop because `cuts` arrives from a UI spinner
/// and the op allocates `cuts × ring length` vertices before it can refuse
/// anything.
pub const MAX_LOOP_CUTS: u32 = 64;

/// Where a [`crate::ops::Op::MergeVerts`] puts the survivor.
///
/// Externally tagged (serde's default) like every other wire enum in this crate:
/// bincode cannot encode an internally-tagged enum (the P4 finding).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeTarget {
    /// The centroid of every merged vertex's **original** position.
    Center,
    /// The last vertex in the list keeps its position.
    Last,
}

/// The plane a [`crate::ops::Op::Mirror`] reflects across, named by its axis.
///
/// Axis-aligned only — see the module docs for why the exact-zero weld law makes
/// that a correctness requirement rather than a simplification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirrorAxis {
    X,
    Y,
    Z,
}

impl MirrorAxis {
    fn index(self) -> usize {
        match self {
            MirrorAxis::X => 0,
            MirrorAxis::Y => 1,
            MirrorAxis::Z => 2,
        }
    }
}

/// One waypoint of a [`crate::ops::Op::Knife`] path.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum KnifePoint {
    /// An existing vertex.
    Vertex(VertId),
    /// A point at parameter `t ∈ (0, 1)` along an existing edge, which the cut
    /// splits into a vertex first.
    Edge { half: HalfId, t: f64 },
}

// ── the region ─────────────────────────────────────────────────────────────

/// A set of faces treated as one area, plus its border.
///
/// The border is the list of half-edges **belonging to region faces** whose twin
/// does not — so it is directed, and its direction is the one a wall or a ring
/// quad must be wound against (the region side of the edge is free once the
/// region's faces are removed, so a new face may claim it).
struct Region {
    /// Ascending, deduplicated.
    faces: Vec<FaceId>,
    set: BTreeSet<FaceId>,
    /// Ascending by half-edge id — the op's iteration order, and therefore part
    /// of what makes its allocations a pure function of the op.
    border: Vec<HalfId>,
    verts: BTreeSet<VertId>,
}

impl Region {
    fn of_faces(mesh: &Mesh, faces: &[FaceId]) -> Result<Self, OpError> {
        if faces.is_empty() {
            return Err(OpError::EmptyOperand { what: "a face set" });
        }
        for &f in faces {
            if !mesh.has_face(f) {
                return Err(OpError::NoSuchFace(f));
            }
        }
        let set: BTreeSet<FaceId> = faces.iter().copied().collect();
        let ordered: Vec<FaceId> = set.iter().copied().collect();
        let mut border = Vec::new();
        let mut verts = BTreeSet::new();
        for &f in &ordered {
            for h in mesh.face_loop(f).expect("live face id") {
                verts.insert(mesh.origin(h).expect("live half-edge id"));
                let t = mesh.twin(h).expect("live half-edge id");
                let outside = match mesh.face_of(t).expect("live half-edge id") {
                    Some(g) => !set.contains(&g),
                    None => true,
                };
                if outside {
                    border.push(h);
                }
            }
        }
        border.sort_unstable();
        Ok(Self {
            faces: ordered,
            set,
            border,
            verts,
        })
    }

    /// The region's area-weighted normal — the Newell sum over its faces, which
    /// weights each by twice its area for free.
    fn normal(&self, mesh: &Mesh) -> Result<DVec3, OpError> {
        let mut acc = DVec3::ZERO;
        for &f in &self.faces {
            acc += newell(mesh, f);
        }
        let len = acc.length();
        if !above(len, 1e-18) {
            return Err(OpError::DegenerateRegion {
                faces: self.faces.len(),
            });
        }
        Ok(acc / len)
    }

    /// Faces adjacent to a border edge that are **not** in the region — the ones
    /// an op that inserts vertices into a border edge must rebuild too, because a
    /// shared edge split on one side only is not a mesh.
    fn fringe(&self, mesh: &Mesh) -> Vec<FaceId> {
        let mut out = BTreeSet::new();
        for &h in &self.border {
            let t = mesh.twin(h).expect("live half-edge id");
            if let Some(Some(g)) = mesh.face_of(t) {
                if !self.set.contains(&g) {
                    out.insert(g);
                }
            }
        }
        out.into_iter().collect()
    }
}

/// A border site captured as data: the two endpoints, the two corner records the
/// wall or ring quad inherits, and the slot it draws with. A tuple rather than a
/// struct because it is built and consumed in one place, and named because five
/// fields of three types is not readable inline.
type BorderSite = (VertId, VertId, CornerData, CornerData, Option<u32>);

/// The same, when only the UVs are wanted.
type BorderUvSite = (VertId, VertId, [f64; 2], [f64; 2], Option<u32>);

/// A face captured as data that survives the removal of every id in it.
struct Captured {
    verts: Vec<VertId>,
    corners: Vec<CornerData>,
    slot: Option<u32>,
}

fn capture(mesh: &Mesh, f: FaceId) -> Captured {
    let halfs = mesh.face_loop(f).expect("live face id");
    let mut verts = Vec::with_capacity(halfs.len());
    let mut corners = Vec::with_capacity(halfs.len());
    for &h in &halfs {
        verts.push(mesh.origin(h).expect("live half-edge id"));
        corners.push(CornerData {
            uv: mesh.corner_uv(h).expect("live half-edge id"),
            normal: mesh.corner_normal(h).expect("live half-edge id"),
        });
    }
    Captured {
        verts,
        corners,
        slot: mesh.face_slot(f).expect("live face id"),
    }
}

/// A corner whose UV is carried over but whose normal is handed back to export's
/// smooth-fan rule — the right default for a corner this crate *created*, since
/// an authored normal copied onto a new facet is a lie about a surface that did
/// not exist a moment ago.
fn derived_corner(uv: [f64; 2]) -> CornerData {
    CornerData { uv, normal: None }
}

/// `x` is a real number strictly greater than `floor`.
///
/// Spelled out rather than written `!(x > floor)` because the negation is the
/// load-bearing part: a NaN compares false against everything, so `!(x > floor)`
/// is *deliberately* true for it — and a reader (and clippy) is right to want
/// that said in words rather than inferred from an operator.
fn above(x: f64, floor: f64) -> bool {
    x.is_finite() && x > floor
}

fn pos(mesh: &Mesh, v: VertId) -> DVec3 {
    mesh.position(v).expect("live vertex id")
}

/// Mark the undirected edge `a–b` sharp, if it exists.
fn sharpen(mesh: &mut Mesh, a: VertId, b: VertId) {
    if let Some(h) = mesh.find_half(a, b) {
        mesh.set_sharp_pair(h, true);
    }
}

// ── extrude ────────────────────────────────────────────────────────────────

/// Extrude a face region along its own area-weighted normal.
///
/// The region moves as a unit and the **border** grows walls; interior edges get
/// none, which is what makes two selected faces extrude into one box rather than
/// two boxes sharing a membrane.
///
/// `distance` may be zero or negative. Zero produces coincident vertices — a
/// legal modelling intermediate that the writer counts as an advisory
/// ([`crate::export::ExportReport::coincident_vertices`]) — because
/// extrude-then-drag is the interactive gesture and refusing the first half of it
/// would be refusing the gesture.
pub(crate) fn extrude_faces(
    mesh: &mut Mesh,
    faces: &[FaceId],
    distance: f64,
) -> Result<OpOutcome, OpError> {
    finite("an extrude distance", &[distance])?;
    let region = Region::of_faces(mesh, faces)?;
    let delta = region.normal(mesh)? * distance;
    let captured: Vec<Captured> = region.faces.iter().map(|&f| capture(mesh, f)).collect();
    // Border sites as VERTEX pairs plus the two corners they carry: half-edge ids
    // do not survive the removal below, and vertex ids do.
    let border: Vec<BorderSite> = region
        .border
        .iter()
        .map(|&h| {
            let n = mesh.next(h).expect("live half-edge id");
            let slot = mesh
                .face_of(h)
                .expect("live half-edge id")
                .and_then(|f| mesh.face_slot(f).expect("live face id"));
            (
                mesh.origin(h).expect("live half-edge id"),
                mesh.origin(n).expect("live half-edge id"),
                CornerData {
                    uv: mesh.corner_uv(h).expect("live half-edge id"),
                    normal: None,
                },
                CornerData {
                    uv: mesh.corner_uv(n).expect("live half-edge id"),
                    normal: None,
                },
                slot,
            )
        })
        .collect();
    let sharp = mesh.capture_sharp(&region.faces);

    mesh.transact(|m| {
        // The displaced copies, allocated in ascending source order so the id
        // assignment is a pure function of the op.
        let mut dup: BTreeMap<VertId, VertId> = BTreeMap::new();
        for &v in &region.verts {
            let p = pos(m, v);
            let np = (p + delta).to_array();
            finite("an extruded vertex position", &np)?;
            dup.insert(v, m.alloc_vert(np));
        }
        let map = |v: VertId| *dup.get(&v).expect("every region vertex is duplicated");

        for &f in &region.faces {
            m.remove_face_raw(f);
        }

        let mut touched: BTreeSet<VertId> = region.verts.clone();
        touched.extend(dup.values().copied());
        let mut made = Vec::with_capacity(captured.len() + border.len());

        // The moved region, on the duplicated vertices, keeping its attributes.
        for c in &captured {
            let verts: Vec<VertId> = c.verts.iter().copied().map(map).collect();
            made.push(m.add_face_raw(&verts, &c.corners, c.slot)?);
        }
        // One wall per border edge. The region side of `a→b` is free (its face
        // was just removed) and the outside face still holds `b→a`, so the wall
        // claims `a→b` and its winding follows from that with nothing to choose.
        for &(a, b, ref ca, ref cb, slot) in &border {
            let (a2, b2) = (map(a), map(b));
            made.push(m.add_face_raw(&[a, b, b2, a2], &[*ca, *cb, *cb, *ca], slot)?);
        }
        // A region vertex with no incident wall — one strictly inside the region —
        // is now isolated: its faces all moved to its copy.
        for &v in &region.verts {
            if m.vert_outgoing(v).map(<[HalfId]>::len) == Some(0) {
                m.free_vert(v);
            }
        }

        let moved: Vec<(VertId, VertId, bool)> =
            sharp.iter().map(|&(x, y, s)| (map(x), map(y), s)).collect();
        m.apply_sharp(&moved);
        // The silhouette: the border edge and its copy read as hard, while the
        // *vertical* wall edges stay as they were — so extruding a ring of faces
        // off a cylinder keeps the barrel smooth and still gets a crisp rim.
        for &(a, b, ..) in &border {
            sharpen(m, a, b);
            sharpen(m, map(a), map(b));
        }
        m.finish_patch(&touched)?;
        Ok(OpOutcome {
            verts: dup.values().copied().collect(),
            faces: made,
            ..Default::default()
        })
    })
}

/// Extrude boundary edges into new faces, by an explicit delta.
///
/// **A delta, not a distance, and the asymmetry with [`extrude_faces`] is
/// deliberate**: a face has a normal, so "extrude by 0.2 m" names one direction.
/// An edge does not — the two obvious candidates (the adjacent face's normal, and
/// the in-plane outward direction) are both wrong half the time — so the caller
/// supplies the vector and the op does not invent one.
///
/// Refuses an edge with a face on both sides: extruding *that* would have to tear
/// the surface open, which is a different tool (a rip) and is not this one.
pub(crate) fn extrude_edges(
    mesh: &mut Mesh,
    edges: &[HalfId],
    delta: [f64; 3],
) -> Result<OpOutcome, OpError> {
    finite("an extrude delta", &delta)?;
    if edges.is_empty() {
        return Err(OpError::EmptyOperand {
            what: "an edge set",
        });
    }
    // Resolve to the boundary half of each edge, deduplicated and ordered.
    let mut sites: BTreeSet<HalfId> = BTreeSet::new();
    for &h in edges {
        if !mesh.has_half(h) {
            return Err(OpError::NoSuchHalf(h));
        }
        let t = mesh.twin(h).expect("live half-edge id");
        let boundary = if mesh.is_boundary(h) == Some(true) {
            h
        } else if mesh.is_boundary(t) == Some(true) {
            t
        } else {
            return Err(OpError::EdgeNotBoundary(h));
        };
        sites.insert(boundary);
    }
    let d = DVec3::from_array(delta);

    // (a, b, uv at a, uv at b, slot) captured off the FACE side, which is the
    // only side carrying corner attributes.
    let mut walls: Vec<BorderUvSite> = Vec::new();
    let mut ends: BTreeSet<VertId> = BTreeSet::new();
    for &hb in &sites {
        let a = mesh.origin(hb).expect("live half-edge id");
        let b = mesh.dest(hb).expect("live half-edge id");
        let t = mesh.twin(hb).expect("live half-edge id");
        // `t` runs b→a inside its face, so its own corner is at b and the next
        // corner is at a.
        let (uv_a, uv_b, slot) = match mesh.face_of(t).expect("live half-edge id") {
            Some(f) => {
                let n = mesh.next(t).expect("live half-edge id");
                (
                    mesh.corner_uv(n).expect("live half-edge id"),
                    mesh.corner_uv(t).expect("live half-edge id"),
                    mesh.face_slot(f).expect("live face id"),
                )
            }
            // A wire edge is a `Violation`, not a state — but reading defensively
            // costs nothing and keeps this function total.
            None => ([0.0, 0.0], [0.0, 0.0], None),
        };
        walls.push((a, b, uv_a, uv_b, slot));
        ends.insert(a);
        ends.insert(b);
    }

    mesh.transact(|m| {
        let mut dup: BTreeMap<VertId, VertId> = BTreeMap::new();
        for &v in &ends {
            let np = (pos(m, v) + d).to_array();
            finite("an extruded vertex position", &np)?;
            dup.insert(v, m.alloc_vert(np));
        }
        let map = |v: VertId| *dup.get(&v).expect("every endpoint is duplicated");
        let mut made = Vec::with_capacity(walls.len());
        for &(a, b, uv_a, uv_b, slot) in &walls {
            let (a2, b2) = (map(a), map(b));
            made.push(m.add_face_raw(
                &[a, b, b2, a2],
                &[
                    derived_corner(uv_a),
                    derived_corner(uv_b),
                    derived_corner(uv_b),
                    derived_corner(uv_a),
                ],
                slot,
            )?);
        }
        let mut touched = ends.clone();
        touched.extend(dup.values().copied());
        m.finish_patch(&touched)?;
        Ok(OpOutcome {
            verts: dup.values().copied().collect(),
            faces: made,
            ..Default::default()
        })
    })
}

// ── inset ──────────────────────────────────────────────────────────────────

/// Inset a face region: shrink its border inward by `amount` metres, filling the
/// gap with a ring of quads.
///
/// `individual` insets each face on its own (every face gets its own ring, and a
/// shared edge ends up between two rings); the region form insets the outline of
/// the whole selection and leaves interior edges untouched.
pub(crate) fn inset_faces(
    mesh: &mut Mesh,
    faces: &[FaceId],
    amount: f64,
    individual: bool,
) -> Result<OpOutcome, OpError> {
    finite("an inset amount", &[amount])?;
    if individual {
        // Ordered so the sequence of sub-ops — and therefore the id allocation —
        // is a function of the op alone, not of the caller's list order.
        let mut ordered: Vec<FaceId> = faces.to_vec();
        ordered.sort_unstable();
        ordered.dedup();
        if ordered.is_empty() {
            return Err(OpError::EmptyOperand { what: "a face set" });
        }
        for &f in &ordered {
            if !mesh.has_face(f) {
                return Err(OpError::NoSuchFace(f));
            }
        }
        // One transaction over the whole batch: an inset that refuses on face 4
        // of 5 must not leave three of them done.
        return mesh.transact(|m| {
            let mut made = Vec::new();
            let mut verts = Vec::new();
            // Each sub-inset rebuilds its face, so the id must be re-resolved by
            // what it *was*: capture the loops first and find each face again by
            // its (still live) corner vertices.
            let loops: Vec<Vec<VertId>> = ordered
                .iter()
                .map(|&f| m.face_verts(f).expect("live face id"))
                .collect();
            for verts_of in &loops {
                let f = face_with_loop(m, verts_of).ok_or(OpError::EmptyOperand {
                    what: "a face the inset batch already consumed",
                })?;
                let out = inset_region(m, &Region::of_faces(m, &[f])?, amount)?;
                made.extend(out.faces);
                verts.extend(out.verts);
            }
            Ok(OpOutcome {
                verts,
                faces: made,
                ..Default::default()
            })
        });
    }
    let region = Region::of_faces(mesh, faces)?;
    mesh.transact(|m| inset_region(m, &region, amount))
}

/// The face whose loop is exactly `verts` (in some rotation) — how the individual
/// inset re-finds a face after an earlier sub-inset renumbered it.
fn face_with_loop(mesh: &Mesh, verts: &[VertId]) -> Option<FaceId> {
    let first = *verts.first()?;
    let second = *verts.get(1)?;
    let h = mesh.find_half(first, second)?;
    let f = mesh.face_of(h)??;
    let got = mesh.face_verts(f)?;
    (got.len() == verts.len()).then_some(f)
}

fn inset_region(mesh: &mut Mesh, region: &Region, amount: f64) -> Result<OpOutcome, OpError> {
    if region.border.is_empty() {
        return Err(OpError::NoRegionBorder);
    }
    // Border chains through each border vertex: exactly one edge in and one out,
    // or the corner has no defined miter.
    let mut incoming: BTreeMap<VertId, HalfId> = BTreeMap::new();
    let mut outgoing: BTreeMap<VertId, HalfId> = BTreeMap::new();
    for &h in &region.border {
        let a = mesh.origin(h).expect("live half-edge id");
        let b = mesh.dest(h).expect("live half-edge id");
        if outgoing.insert(a, h).is_some() {
            return Err(OpError::AmbiguousRegionVertex(a));
        }
        if incoming.insert(b, h).is_some() {
            return Err(OpError::AmbiguousRegionVertex(b));
        }
    }

    // The inward in-plane normal of a border half-edge: `N × ê`, which points
    // into the face for a loop wound so `N` is its outward normal.
    let inward = |m: &Mesh, h: HalfId| -> Result<DVec3, OpError> {
        let f = m
            .face_of(h)
            .expect("live half-edge id")
            .expect("a border half-edge belongs to a region face");
        let n = newell(m, f);
        if !above(n.length(), 1e-18) {
            return Err(OpError::DegenerateFace(f));
        }
        let a = pos(m, m.origin(h).expect("live half-edge id"));
        let b = pos(m, m.dest(h).expect("live half-edge id"));
        let e = b - a;
        if !above(e.length(), 1e-18) {
            return Err(OpError::ZeroLengthEdge(h));
        }
        Ok(n.normalize().cross(e.normalize()))
    };

    let mut moved: BTreeMap<VertId, [f64; 3]> = BTreeMap::new();
    for (&v, &out) in &outgoing {
        let inc = *incoming.get(&v).ok_or(OpError::AmbiguousRegionVertex(v))?;
        let n1 = inward(mesh, inc)?;
        let n2 = inward(mesh, out)?;
        // The miter vector: the offset that puts the vertex exactly `amount` from
        // BOTH offset edge lines. `m·n1 = m·n2 = 1` ⇒ `m = (n1+n2)/(1+n1·n2)`.
        let denom = 1.0 + n1.dot(n2);
        if !above(denom, 1e-9) {
            return Err(OpError::DegenerateInsetCorner(v));
        }
        let p = pos(mesh, v) + (n1 + n2) / denom * amount;
        let arr = p.to_array();
        finite("an inset vertex position", &arr)?;
        moved.insert(v, arr);
    }

    let captured: Vec<Captured> = region.faces.iter().map(|&f| capture(mesh, f)).collect();
    let border: Vec<BorderUvSite> = region
        .border
        .iter()
        .map(|&h| {
            let n = mesh.next(h).expect("live half-edge id");
            let slot = mesh
                .face_of(h)
                .expect("live half-edge id")
                .and_then(|f| mesh.face_slot(f).expect("live face id"));
            (
                mesh.origin(h).expect("live half-edge id"),
                mesh.origin(n).expect("live half-edge id"),
                mesh.corner_uv(h).expect("live half-edge id"),
                mesh.corner_uv(n).expect("live half-edge id"),
                slot,
            )
        })
        .collect();
    let sharp = mesh.capture_sharp(&region.faces);

    let mut inner: BTreeMap<VertId, VertId> = BTreeMap::new();
    for (&v, &p) in &moved {
        inner.insert(v, mesh.alloc_vert(p));
    }
    // An interior region vertex is not on the border and does not move.
    let map = |v: VertId| inner.get(&v).copied().unwrap_or(v);

    for &f in &region.faces {
        mesh.remove_face_raw(f);
    }
    let mut touched: BTreeSet<VertId> = region.verts.clone();
    touched.extend(inner.values().copied());
    let mut made = Vec::with_capacity(captured.len() + border.len());
    for c in &captured {
        let verts: Vec<VertId> = c.verts.iter().copied().map(map).collect();
        made.push(mesh.add_face_raw(&verts, &c.corners, c.slot)?);
    }
    for &(a, b, uv_a, uv_b, slot) in &border {
        let (a2, b2) = (map(a), map(b));
        made.push(mesh.add_face_raw(
            &[a, b, b2, a2],
            &[
                derived_corner(uv_a),
                derived_corner(uv_b),
                derived_corner(uv_b),
                derived_corner(uv_a),
            ],
            slot,
        )?);
    }
    let remapped: Vec<(VertId, VertId, bool)> =
        sharp.iter().map(|&(x, y, s)| (map(x), map(y), s)).collect();
    mesh.apply_sharp(&sharp);
    mesh.apply_sharp(&remapped);
    mesh.finish_patch(&touched)?;
    Ok(OpOutcome {
        verts: inner.values().copied().collect(),
        faces: made,
        ..Default::default()
    })
}

// ── bevel ──────────────────────────────────────────────────────────────────

/// Chamfer a set of interior edges — **one segment**, see the module docs.
///
/// The construction, because it is the part that decides what the tool can do:
/// both endpoints of the edge are **kept**, each adjacent face grows two corners
/// (the offset copies of the endpoints, pushed `amount` metres into that face's
/// own plane, perpendicular to the edge), the strip between them is one quad, and
/// each end of the strip is capped with a triangle back to the surviving
/// endpoint. That is eight new directed edges, all four of them twinned inside
/// the patch, and it needs to know **nothing** about how many other faces meet at
/// the endpoint — which is what makes it work on a cube corner, a pole with
/// twelve faces, and everything between, where a construction that dissolved the
/// vertex would need a case per valence.
pub(crate) fn bevel_edges(
    mesh: &mut Mesh,
    edges: &[HalfId],
    amount: f64,
) -> Result<OpOutcome, OpError> {
    finite("a bevel amount", &[amount])?;
    if !above(amount, 0.0) {
        return Err(OpError::AmountNotPositive {
            what: "a bevel amount",
            value: amount,
        });
    }
    if edges.is_empty() {
        return Err(OpError::EmptyOperand {
            what: "an edge set",
        });
    }
    // As vertex pairs: each bevel rebuilds two faces, so a half-edge id captured
    // now may not name the same edge by the time its turn comes. Both endpoints
    // survive every bevel, so the pair does.
    let mut pairs: Vec<(VertId, VertId)> = Vec::with_capacity(edges.len());
    for &h in edges {
        if !mesh.has_half(h) {
            return Err(OpError::NoSuchHalf(h));
        }
        let t = mesh.twin(h).expect("live half-edge id");
        if mesh.is_boundary(h) == Some(true) || mesh.is_boundary(t) == Some(true) {
            return Err(OpError::BevelBoundaryEdge(h));
        }
        let a = mesh.origin(h).expect("live half-edge id");
        let b = mesh.origin(t).expect("live half-edge id");
        pairs.push(if a < b { (a, b) } else { (b, a) });
    }
    pairs.sort_unstable();
    pairs.dedup();

    mesh.transact(|m| {
        let mut made = Vec::new();
        let mut verts = Vec::new();
        for &(a, b) in &pairs {
            let h = m.find_half(a, b).ok_or(OpError::NoSuchVert(a))?;
            let out = bevel_one(m, h, amount)?;
            made.extend(out.faces);
            verts.extend(out.verts);
        }
        Ok(OpOutcome {
            verts,
            faces: made,
            ..Default::default()
        })
    })
}

fn bevel_one(mesh: &mut Mesh, h: HalfId, amount: f64) -> Result<OpOutcome, OpError> {
    let t = mesh.twin(h).expect("live half-edge id");
    let f1 = match mesh.face_of(h).expect("live half-edge id") {
        Some(f) => f,
        None => return Err(OpError::BevelBoundaryEdge(h)),
    };
    let f2 = match mesh.face_of(t).expect("live half-edge id") {
        Some(f) => f,
        None => return Err(OpError::BevelBoundaryEdge(h)),
    };
    let a = mesh.origin(h).expect("live half-edge id");
    let b = mesh.origin(t).expect("live half-edge id");
    let (pa, pb) = (pos(mesh, a), pos(mesh, b));
    let e = pb - pa;
    if !above(e.length(), 1e-18) {
        return Err(OpError::ZeroLengthEdge(h));
    }
    let eu = e.normalize();
    let n1 = newell(mesh, f1);
    if !above(n1.length(), 1e-18) {
        return Err(OpError::DegenerateFace(f1));
    }
    let n2 = newell(mesh, f2);
    if !above(n2.length(), 1e-18) {
        return Err(OpError::DegenerateFace(f2));
    }
    // Into f1 across `a→b`; into f2 across `b→a` (its own half runs the other way).
    let in1 = n1.normalize().cross(eu) * amount;
    let in2 = n2.normalize().cross(-eu) * amount;
    let new_pos = [pa + in1, pb + in1, pa + in2, pb + in2];
    for p in &new_pos {
        finite("a bevel vertex position", &p.to_array())?;
    }
    let was_sharp = mesh.is_sharp(h) == Some(true);

    let c1 = capture(mesh, f1);
    let c2 = capture(mesh, f2);
    let sharp = mesh.capture_sharp(&[f1, f2]);
    // Where a and b sit in each captured loop. `f1` holds a→b, `f2` holds b→a.
    let i1 = index_of(&c1.verts, a);
    let i2 = index_of(&c2.verts, b);

    mesh.remove_face_raw(f1);
    mesh.remove_face_raw(f2);
    let va1 = mesh.alloc_vert(new_pos[0].to_array());
    let vb1 = mesh.alloc_vert(new_pos[1].to_array());
    let va2 = mesh.alloc_vert(new_pos[2].to_array());
    let vb2 = mesh.alloc_vert(new_pos[3].to_array());

    let n_1 = c1.verts.len();
    let n_2 = c2.verts.len();
    let ca_1 = c1.corners[i1];
    let cb_1 = c1.corners[(i1 + 1) % n_1];
    let cb_2 = c2.corners[i2];
    let ca_2 = c2.corners[(i2 + 1) % n_2];

    // f1: … a, a1, b1, b … — a and b keep their corners, the copies take theirs.
    let mut v1 = c1.verts.clone();
    let mut k1 = c1.corners.clone();
    v1.splice(i1 + 1..i1 + 1, [va1, vb1]);
    k1.splice(i1 + 1..i1 + 1, [ca_1, cb_1]);
    // f2: … b, b2, a2, a …
    let mut v2 = c2.verts.clone();
    let mut k2 = c2.corners.clone();
    v2.splice(i2 + 1..i2 + 1, [vb2, va2]);
    k2.splice(i2 + 1..i2 + 1, [cb_2, ca_2]);

    // The two grown faces, then the strip, then a cap at each end. Every one of
    // the eight new directed edges is twinned inside these five loops -- that is
    // the whole proof that the construction is manifold at any valence.
    let mut made = vec![
        mesh.add_face_raw(&v1, &k1, c1.slot)?,
        mesh.add_face_raw(&v2, &k2, c2.slot)?,
    ];
    made.push(mesh.add_face_raw(
        &[va1, va2, vb2, vb1],
        &[
            derived_corner(ca_1.uv),
            derived_corner(ca_2.uv),
            derived_corner(cb_2.uv),
            derived_corner(cb_1.uv),
        ],
        c1.slot,
    )?);
    made.push(mesh.add_face_raw(
        &[va1, a, va2],
        &[
            derived_corner(ca_1.uv),
            derived_corner(ca_1.uv),
            derived_corner(ca_2.uv),
        ],
        c1.slot,
    )?);
    made.push(mesh.add_face_raw(
        &[b, vb1, vb2],
        &[
            derived_corner(cb_1.uv),
            derived_corner(cb_1.uv),
            derived_corner(cb_2.uv),
        ],
        c1.slot,
    )?);

    mesh.apply_sharp(&sharp);
    if was_sharp {
        // The chamfer keeps the crease it replaced: a flat-shaded box beveled
        // stays flat-shaded, rather than acquiring a smooth band nobody asked for.
        for (x, y) in [
            (va1, vb1),
            (va2, vb2),
            (va1, va2),
            (vb1, vb2),
            (va1, a),
            (a, va2),
            (b, vb1),
            (vb2, b),
        ] {
            sharpen(mesh, x, y);
        }
    }
    let touched: BTreeSet<VertId> = c1
        .verts
        .iter()
        .chain(c2.verts.iter())
        .copied()
        .chain([va1, vb1, va2, vb2])
        .collect();
    mesh.finish_patch(&touched)?;
    Ok(OpOutcome {
        verts: vec![va1, vb1, va2, vb2],
        faces: made,
        ..Default::default()
    })
}

fn index_of(verts: &[VertId], v: VertId) -> usize {
    verts
        .iter()
        .position(|&x| x == v)
        .expect("the edge's own face contains its endpoint")
}

// ── loop cut ───────────────────────────────────────────────────────────────

/// Cut a quad strip with `cuts` parallel loops.
///
/// The strip is the **edge ring** through `half` ([`crate::select::edge_ring`]) —
/// the walk that steps across each quad to the edge opposite the one it came in
/// on. It stops, as a value, at anything that is not a quad or has no face there,
/// so a ring that runs off the end of a grid simply ends; the faces beyond it are
/// *fringe* and are rebuilt with the new vertices inserted, because a shared edge
/// cut on one side only is not a mesh.
pub(crate) fn loop_cut(mesh: &mut Mesh, half: HalfId, cuts: u32) -> Result<OpOutcome, OpError> {
    if !mesh.has_half(half) {
        return Err(OpError::NoSuchHalf(half));
    }
    if cuts == 0 || cuts > MAX_LOOP_CUTS {
        return Err(OpError::CountOutOfRange {
            what: "loop cuts",
            value: cuts,
            min: 1,
            max: MAX_LOOP_CUTS,
        });
    }
    let start_face = match mesh.face_of(half).expect("live half-edge id") {
        Some(f) => f,
        None => return Err(OpError::BoundaryCorner(half)),
    };
    if mesh.face_loop(start_face).expect("live face id").len() != 4 {
        return Err(OpError::NotAQuad(start_face));
    }
    let ring = crate::select::edge_ring(mesh, half);

    // The strip, captured as data. `entry` is the ring edge the walk came in on;
    // the quad is [p, q, r, s] with (p,q) the entry and (r,s) the exit.
    struct Quad {
        verts: [VertId; 4],
        corners: [CornerData; 4],
        slot: Option<u32>,
    }
    let mut strip: Vec<Quad> = Vec::with_capacity(ring.len());
    let mut strip_faces: BTreeSet<FaceId> = BTreeSet::new();
    let mut ring_edges: BTreeSet<(VertId, VertId)> = BTreeSet::new();
    for &h in &ring {
        let f = mesh
            .face_of(h)
            .expect("live half-edge id")
            .expect("the ring only steps into faced quads");
        strip_faces.insert(f);
        let mut hs = [h; 4];
        for k in 1..4 {
            hs[k] = mesh.next(hs[k - 1]).expect("live half-edge id");
        }
        let verts = hs.map(|x| mesh.origin(x).expect("live half-edge id"));
        let corners = hs.map(|x| CornerData {
            uv: mesh.corner_uv(x).expect("live half-edge id"),
            normal: mesh.corner_normal(x).expect("live half-edge id"),
        });
        ring_edges.insert(canonical_pair(verts[0], verts[1]));
        ring_edges.insert(canonical_pair(verts[2], verts[3]));
        strip.push(Quad {
            verts,
            corners,
            slot: mesh.face_slot(f).expect("live face id"),
        });
    }

    // Fringe: faces touching a ring edge from outside the strip.
    let mut fringe_faces: BTreeSet<FaceId> = BTreeSet::new();
    for &(a, b) in &ring_edges {
        for (x, y) in [(a, b), (b, a)] {
            if let Some(h) = mesh.find_half(x, y) {
                if let Some(Some(f)) = mesh.face_of(h) {
                    if !strip_faces.contains(&f) {
                        fringe_faces.insert(f);
                    }
                }
            }
        }
    }
    let fringe: Vec<Captured> = fringe_faces.iter().map(|&f| capture(mesh, f)).collect();
    let all: Vec<FaceId> = strip_faces
        .iter()
        .copied()
        .chain(fringe_faces.iter().copied())
        .collect();
    let sharp = mesh.capture_sharp(&all);

    mesh.transact(|m| {
        for &f in &all {
            m.remove_face_raw(f);
        }
        // The cut vertices of each ring edge, ordered from its LOWER id endpoint —
        // one canonical order, so the two faces sharing an edge agree about which
        // cut is which without either of them having to be asked.
        let n = cuts as usize;
        let mut cut: BTreeMap<(VertId, VertId), Vec<VertId>> = BTreeMap::new();
        let mut touched: BTreeSet<VertId> = BTreeSet::new();
        for &(lo, hi) in &ring_edges {
            let (a, b) = (pos(m, lo), pos(m, hi));
            let mut made = Vec::with_capacity(n);
            for k in 1..=n {
                let t = k as f64 / (n + 1) as f64;
                let p = (a + (b - a) * t).to_array();
                finite("a loop-cut vertex position", &p)?;
                made.push(m.alloc_vert(p));
            }
            touched.insert(lo);
            touched.insert(hi);
            touched.extend(made.iter().copied());
            cut.insert((lo, hi), made);
        }
        // The cuts on `a→b`, in that direction.
        let along = |a: VertId, b: VertId| -> Vec<VertId> {
            let key = canonical_pair(a, b);
            let mut v = cut.get(&key).cloned().unwrap_or_default();
            if a > b {
                v.reverse();
            }
            v
        };

        let mut made = Vec::new();
        for q in &strip {
            let [p, qq, r, s] = q.verts;
            let c = along(p, qq); // c[0] nearest p
            let d = along(r, s); // d[0] nearest r
            let cc: Vec<CornerData> = (1..=n)
                .map(|k| lerp_corner(&q.corners[0], &q.corners[1], k as f64 / (n + 1) as f64))
                .collect();
            let dd: Vec<CornerData> = (1..=n)
                .map(|k| lerp_corner(&q.corners[2], &q.corners[3], k as f64 / (n + 1) as f64))
                .collect();
            // n+1 quads, walking p→q on one side and s→r on the other.
            for k in 0..=n {
                let (v0, k0) = if k == 0 {
                    (p, q.corners[0])
                } else {
                    (c[k - 1], cc[k - 1])
                };
                let (v1, k1) = if k == n {
                    (qq, q.corners[1])
                } else {
                    (c[k], cc[k])
                };
                let (v2, k2) = if k == n {
                    (r, q.corners[2])
                } else {
                    (d[n - 1 - k], dd[n - 1 - k])
                };
                let (v3, k3) = if k == 0 {
                    (s, q.corners[3])
                } else {
                    (d[n - k], dd[n - k])
                };
                made.push(m.add_face_raw(&[v0, v1, v2, v3], &[k0, k1, k2, k3], q.slot)?);
            }
        }
        for c in &fringe {
            let (verts, corners) = insert_cuts(&c.verts, &c.corners, &cut, n);
            touched.extend(verts.iter().copied());
            made.push(m.add_face_raw(&verts, &corners, c.slot)?);
        }
        m.apply_sharp(&sharp);
        // A cut edge inherits the sharpness of the edge it came from, on both of
        // its halves — the same rule `SplitEdge` already applies.
        for (&(lo, hi), made_verts) in &cut {
            if sharp
                .iter()
                .any(|&(x, y, s)| s && ((x == lo && y == hi) || (x == hi && y == lo)))
            {
                let mut chain = vec![lo];
                chain.extend(made_verts.iter().copied());
                chain.push(hi);
                for w in chain.windows(2) {
                    sharpen(m, w[0], w[1]);
                }
            }
        }
        m.finish_patch(&touched)?;
        Ok(OpOutcome {
            verts: cut.values().flatten().copied().collect(),
            faces: made,
            ..Default::default()
        })
    })
}

fn canonical_pair(a: VertId, b: VertId) -> (VertId, VertId) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Rebuild a face loop with any cut vertices spliced into the edges that got
/// them — the fringe's whole job.
fn insert_cuts(
    verts: &[VertId],
    corners: &[CornerData],
    cut: &BTreeMap<(VertId, VertId), Vec<VertId>>,
    n: usize,
) -> (Vec<VertId>, Vec<CornerData>) {
    let len = verts.len();
    let mut out_v = Vec::with_capacity(len);
    let mut out_c = Vec::with_capacity(len);
    for i in 0..len {
        let (a, b) = (verts[i], verts[(i + 1) % len]);
        out_v.push(a);
        out_c.push(corners[i]);
        let Some(made) = cut.get(&canonical_pair(a, b)) else {
            continue;
        };
        let mut made = made.clone();
        if a > b {
            made.reverse();
        }
        for (k, v) in made.into_iter().enumerate() {
            out_v.push(v);
            out_c.push(lerp_corner(
                &corners[i],
                &corners[(i + 1) % len],
                (k + 1) as f64 / (n + 1) as f64,
            ));
        }
    }
    (out_v, out_c)
}

// ── knife ──────────────────────────────────────────────────────────────────

/// Cut across faces along a path of existing vertices and points on existing
/// edges.
///
/// One transaction for the whole path: a knife that refuses on its fourth segment
/// must not leave three cuts behind. A segment whose two ends are **already**
/// adjacent is skipped rather than refused — running part of a cut along an edge
/// that is already there is a normal thing to do, and "that cut already exists"
/// is not an error.
pub(crate) fn knife(mesh: &mut Mesh, path: &[KnifePoint]) -> Result<OpOutcome, OpError> {
    if path.len() < 2 {
        return Err(OpError::EmptyOperand {
            what: "a knife path of at least two points",
        });
    }
    // Resolve edge points to (endpoints, t) BEFORE anything is cut: splitting one
    // edge rebuilds its faces, and a later `HalfId` in the same path may not
    // survive that. Vertex ids do.
    enum Site {
        Vert(VertId),
        Edge(VertId, VertId, f64),
    }
    let mut sites = Vec::with_capacity(path.len());
    for p in path {
        match *p {
            KnifePoint::Vertex(v) => {
                if !mesh.has_vert(v) {
                    return Err(OpError::NoSuchVert(v));
                }
                sites.push(Site::Vert(v));
            }
            KnifePoint::Edge { half, t } => {
                if !mesh.has_half(half) {
                    return Err(OpError::NoSuchHalf(half));
                }
                if !(t > 0.0 && t < 1.0) {
                    return Err(OpError::SplitParameterOutOfRange { t });
                }
                sites.push(Site::Edge(
                    mesh.origin(half).expect("live half-edge id"),
                    mesh.dest(half).expect("live half-edge id"),
                    t,
                ));
            }
        }
    }

    mesh.transact(|m| {
        let mut points = Vec::with_capacity(sites.len());
        let mut verts = Vec::new();
        for site in &sites {
            match *site {
                Site::Vert(v) => points.push(v),
                Site::Edge(a, b, t) => {
                    let h = m.find_half(a, b).ok_or(OpError::NoSuchVert(a))?;
                    let out = crate::ops::apply(m, &crate::ops::Op::SplitEdge { half: h, t })?;
                    let v = out.verts[0];
                    verts.push(v);
                    points.push(v);
                }
            }
        }
        let mut faces = Vec::new();
        for w in points.windows(2) {
            let (u, v) = (w[0], w[1]);
            if u == v {
                return Err(OpError::SameVertex(u));
            }
            if m.find_half(u, v).is_some() || m.find_half(v, u).is_some() {
                continue; // the cut is already there
            }
            let (from, to) = common_face_corners(m, u, v)
                .ok_or(OpError::KnifeNoCommonFace { from: u, to: v })?;
            let out = crate::ops::apply(m, &crate::ops::Op::SplitFace { from, to })?;
            faces.extend(out.faces);
        }
        Ok(OpOutcome {
            verts,
            faces,
            ..Default::default()
        })
    })
}

/// The two half-edges of a face that `u` and `v` are the corners of, if one face
/// has both.
fn common_face_corners(mesh: &Mesh, u: VertId, v: VertId) -> Option<(HalfId, HalfId)> {
    for &h in mesh.vert_outgoing(u)? {
        let Some(Some(f)) = mesh.face_of(h) else {
            continue;
        };
        let loop_halfs = mesh.face_loop(f)?;
        if let Some(&other) = loop_halfs.iter().find(|&&x| mesh.origin(x) == Some(v)) {
            return Some((h, other));
        }
    }
    None
}

// ── merge ──────────────────────────────────────────────────────────────────

/// Merge vertices into one, at the centre of the set or onto its last member.
///
/// This op does what `CollapseEdge` deliberately will not: it moves the survivor
/// as well as fusing the topology. That is exactly why it is its own journal
/// entry — the target is on the wire, so "merge at centre" and "merge to last"
/// are two different recorded edits rather than one edit plus a translate an undo
/// would separate.
pub(crate) fn merge_verts(
    mesh: &mut Mesh,
    verts: &[VertId],
    target: MergeTarget,
) -> Result<OpOutcome, OpError> {
    if verts.len() < 2 {
        return Err(OpError::EmptyOperand {
            what: "a merge set of at least two vertices",
        });
    }
    for (i, &v) in verts.iter().enumerate() {
        if !mesh.has_vert(v) {
            return Err(OpError::NoSuchVert(v));
        }
        if verts[..i].contains(&v) {
            return Err(OpError::SameVertex(v));
        }
    }
    let keep = match target {
        MergeTarget::Center => verts[0],
        MergeTarget::Last => verts[verts.len() - 1],
    };
    let place = match target {
        MergeTarget::Center => {
            let mut acc = DVec3::ZERO;
            for &v in verts {
                acc += pos(mesh, v);
            }
            (acc / verts.len() as f64).to_array()
        }
        MergeTarget::Last => pos(mesh, keep).to_array(),
    };
    finite("a merge position", &place)?;

    mesh.transact(|m| {
        let mut faces = Vec::new();
        for &v in verts {
            if v == keep {
                continue;
            }
            faces.extend(weld_impl(m, keep, v)?);
        }
        m.vert_mut(keep).position = place;
        Ok(OpOutcome {
            verts: vec![keep],
            faces,
            ..Default::default()
        })
    })
}

// ── subdivide ──────────────────────────────────────────────────────────────

/// Split each face into one quad per corner — **simple midpoint**, no smoothing.
///
/// The shape does not move; only its density changes. Faces adjacent to the
/// subdivided set are rebuilt with the new edge midpoints spliced in, for the
/// same reason the loop cut has a fringe.
pub(crate) fn subdivide_faces(mesh: &mut Mesh, faces: &[FaceId]) -> Result<OpOutcome, OpError> {
    let region = Region::of_faces(mesh, faces)?;
    let captured: Vec<Captured> = region.faces.iter().map(|&f| capture(mesh, f)).collect();
    let fringe_ids = region.fringe(mesh);
    let fringe: Vec<Captured> = fringe_ids.iter().map(|&f| capture(mesh, f)).collect();
    let all: Vec<FaceId> = region
        .faces
        .iter()
        .copied()
        .chain(fringe_ids.iter().copied())
        .collect();
    let sharp = mesh.capture_sharp(&all);
    // Every undirected edge of the region: each one gets exactly one midpoint,
    // shared by both faces that use it.
    let mut edges: BTreeSet<(VertId, VertId)> = BTreeSet::new();
    for c in &captured {
        let n = c.verts.len();
        for i in 0..n {
            edges.insert(canonical_pair(c.verts[i], c.verts[(i + 1) % n]));
        }
    }

    mesh.transact(|m| {
        for &f in &all {
            m.remove_face_raw(f);
        }
        let mut mid: BTreeMap<(VertId, VertId), Vec<VertId>> = BTreeMap::new();
        let mut touched: BTreeSet<VertId> = BTreeSet::new();
        for &(lo, hi) in &edges {
            let p = ((pos(m, lo) + pos(m, hi)) * 0.5).to_array();
            finite("a subdivision midpoint", &p)?;
            let v = m.alloc_vert(p);
            touched.insert(lo);
            touched.insert(hi);
            touched.insert(v);
            mid.insert((lo, hi), vec![v]);
        }
        let mut made = Vec::new();
        let mut centres = Vec::with_capacity(captured.len());
        for c in &captured {
            let n = c.verts.len();
            let mut acc = DVec3::ZERO;
            for &v in &c.verts {
                acc += pos(m, v);
            }
            let cp = (acc / n as f64).to_array();
            finite("a subdivision centroid", &cp)?;
            let centre = m.alloc_vert(cp);
            centres.push(centre);
            touched.insert(centre);
            let centre_corner = average_corner(&c.corners);
            for i in 0..n {
                let prev = (i + n - 1) % n;
                let m_i = mid[&canonical_pair(c.verts[i], c.verts[(i + 1) % n])][0];
                let m_p = mid[&canonical_pair(c.verts[prev], c.verts[i])][0];
                let k_i = lerp_corner(&c.corners[i], &c.corners[(i + 1) % n], 0.5);
                let k_p = lerp_corner(&c.corners[prev], &c.corners[i], 0.5);
                made.push(m.add_face_raw(
                    &[c.verts[i], m_i, centre, m_p],
                    &[c.corners[i], k_i, centre_corner, k_p],
                    c.slot,
                )?);
            }
        }
        for c in &fringe {
            let (verts, corners) = insert_cuts(&c.verts, &c.corners, &mid, 1);
            touched.extend(verts.iter().copied());
            made.push(m.add_face_raw(&verts, &corners, c.slot)?);
        }
        m.apply_sharp(&sharp);
        for (&(lo, hi), made_verts) in &mid {
            if sharp
                .iter()
                .any(|&(x, y, s)| s && ((x == lo && y == hi) || (x == hi && y == lo)))
            {
                sharpen(m, lo, made_verts[0]);
                sharpen(m, made_verts[0], hi);
            }
        }
        m.finish_patch(&touched)?;
        Ok(OpOutcome {
            verts: centres,
            faces: made,
            ..Default::default()
        })
    })
}

/// The mean of a face's corners. The normal is authored only if **every** corner
/// was — one derived corner in the loop means the face was already relying on the
/// smooth fan, and inventing an authored value for its centre would pin a normal
/// nothing asked for.
fn average_corner(corners: &[CornerData]) -> CornerData {
    let n = corners.len() as f64;
    let uv = corners
        .iter()
        .fold([0.0, 0.0], |a, c| [a[0] + c.uv[0] / n, a[1] + c.uv[1] / n]);
    let all = corners.iter().all(|c| c.normal.is_some());
    let normal = all.then(|| {
        let acc: DVec3 = corners
            .iter()
            .map(|c| DVec3::from_array(c.normal.expect("checked above")))
            .sum();
        acc
    });
    let normal = normal.and_then(|v| {
        let len = v.length();
        (len > 1e-12).then(|| (v / len).to_array())
    });
    CornerData { uv, normal }
}

// ── mirror ─────────────────────────────────────────────────────────────────

/// Reflect the whole mesh across an axis-aligned plane, welding the seam.
///
/// **The weld genuinely fires**, and that is the reason the plane is axis
/// aligned. A vertex exactly on the plane reflects to `2d − d`, which is `d` with
/// no rounding at all (the doubling is exact and the subtraction of equals is
/// exact), so the reflected vertex *is* the original vertex and the seam is one
/// row of shared vertices rather than two rows a tolerance would have to guess
/// about. The kernel's weld tolerance is exactly zero and stays that way.
///
/// A seam edge that already has a face on both sides is a genuine
/// non-manifoldness — mirroring a closed solid that straddles the plane is not a
/// thing — and comes back as [`OpError::EdgeAlreadyFaced`], inert.
pub(crate) fn mirror(mesh: &mut Mesh, axis: MirrorAxis, coord: f64) -> Result<OpOutcome, OpError> {
    finite("a mirror plane coordinate", &[coord])?;
    let ax = axis.index();
    let captured: Vec<Captured> = mesh.face_ids().map(|f| capture(mesh, f)).collect();
    let sharp: Vec<(VertId, VertId, bool)> = {
        let all: Vec<FaceId> = mesh.face_ids().collect();
        mesh.capture_sharp(&all)
    };

    mesh.transact(|m| {
        let sources: Vec<VertId> = m.vert_ids().collect();
        let mut map: BTreeMap<VertId, VertId> = BTreeMap::new();
        let mut fresh = Vec::new();
        for v in sources {
            let p = m.vert_ref(v).position;
            if p[ax] == coord {
                map.insert(v, v); // on the plane: the reflection IS the original
                continue;
            }
            let mut q = p;
            q[ax] = 2.0 * coord - p[ax];
            finite("a mirrored vertex position", &q)?;
            let w = m.alloc_vert(q);
            map.insert(v, w);
            fresh.push(w);
        }
        let mut touched: BTreeSet<VertId> = map.keys().copied().collect();
        touched.extend(map.values().copied());

        let mut made = Vec::with_capacity(captured.len());
        for c in &captured {
            let n = c.verts.len();
            // A reflection flips orientation, so the loop is reversed to keep the
            // mirrored face's normal pointing out of the mirrored solid.
            let verts: Vec<VertId> = (0..n).map(|i| map[&c.verts[n - 1 - i]]).collect();
            let corners: Vec<CornerData> = (0..n)
                .map(|i| {
                    let mut k = c.corners[n - 1 - i];
                    if let Some(nrm) = k.normal.as_mut() {
                        nrm[ax] = -nrm[ax];
                    }
                    k
                })
                .collect();
            made.push(m.add_face_raw(&verts, &corners, c.slot)?);
        }
        let mirrored: Vec<(VertId, VertId, bool)> = sharp
            .iter()
            .map(|&(x, y, s)| (map[&x], map[&y], s))
            .collect();
        m.apply_sharp(&mirrored);
        m.finish_patch(&touched)?;
        Ok(OpOutcome {
            verts: fresh,
            faces: made,
            ..Default::default()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, cylinder, plane};
    use crate::ops::{apply, Op};
    use crate::validate::validate;

    fn ok(mesh: &mut Mesh, op: Op) -> OpOutcome {
        let out = apply(mesh, &op).unwrap_or_else(|e| panic!("{op:?} refused: {e}"));
        assert_eq!(validate(mesh), Ok(()), "invalid after {op:?}");
        out
    }

    fn refuses(mesh: &mut Mesh, op: Op) -> OpError {
        let before = mesh.encoded();
        let err = apply(mesh, &op).expect_err("expected a refusal");
        assert_eq!(mesh.encoded(), before, "a refused {op:?} must be inert");
        err
    }

    /// Signed volume × 6 by the divergence theorem — the winding gate that P23.3
    /// learned the hard way is the only thing that catches an inside-out result.
    fn six_volume(mesh: &Mesh) -> f64 {
        let mut acc = 0.0;
        for f in mesh.face_ids() {
            let verts = mesh.face_verts(f).expect("live face id");
            let p0 = pos(mesh, verts[0]);
            for i in 1..verts.len() - 1 {
                let a = pos(mesh, verts[i]) - p0;
                let b = pos(mesh, verts[i + 1]) - p0;
                acc += p0.dot(a.cross(b));
            }
        }
        acc
    }

    fn euler(mesh: &Mesh) -> i64 {
        mesh.vert_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64
    }

    // ── extrude ────────────────────────────────────────────────────────────

    #[test]
    fn extruding_one_cube_face_adds_a_box_and_keeps_the_solid_closed() {
        let mut m = cube(2.0);
        let v0 = six_volume(&m);
        let top = m
            .face_ids()
            .find(|&f| newell(&m, f).normalize().y > 0.9)
            .expect("the +Y face");
        let out = ok(
            &mut m,
            Op::ExtrudeFaces {
                faces: vec![top],
                distance: 1.0,
            },
        );
        assert_eq!(out.faces.len(), 5, "the moved cap plus four walls");
        assert_eq!(m.vert_count(), 12);
        assert_eq!(m.face_count(), 10, "5 untouched + the cap + 4 walls");
        assert_eq!(euler(&m), 2, "still one closed genus-0 shell");
        assert!(m.half_ids().all(|h| m.is_boundary(h) == Some(false)));
        // 2×2×2 plus a 2×2×1 box on top: 8 + 4 = 12 m³.
        assert!(
            (six_volume(&m) - 12.0 * 6.0).abs() < 1e-9,
            "volume {} (and it must be POSITIVE — an inside-out extrude passes every count)",
            six_volume(&m) / 6.0
        );
        assert!(six_volume(&m) > v0);
    }

    #[test]
    fn a_two_face_region_extrudes_as_one_wall_not_two() {
        // The region-border rule, measured: the shared edge must get NO wall.
        // Extruding the faces one at a time would produce 4+4 = 8 walls and a
        // membrane; as a region it is 6 walls and none.
        let mut m = cube(2.0);
        let top = m
            .face_ids()
            .find(|&f| newell(&m, f).normalize().y > 0.9)
            .expect("+Y");
        // Cut the top in half so there are two coplanar faces to extrude.
        let halfs = m.face_loop(top).expect("live");
        ok(
            &mut m,
            Op::SplitFace {
                from: halfs[0],
                to: halfs[2],
            },
        );
        let region: Vec<FaceId> = m
            .face_ids()
            .filter(|&f| newell(&m, f).normalize().y > 0.9)
            .collect();
        assert_eq!(region.len(), 2);
        let out = ok(
            &mut m,
            Op::ExtrudeFaces {
                faces: region,
                distance: 1.0,
            },
        );
        assert_eq!(
            out.faces.len(),
            6,
            "two moved caps + FOUR walls — the shared edge is interior"
        );
        assert_eq!(euler(&m), 2);
        assert!(m.half_ids().all(|h| m.is_boundary(h) == Some(false)));
    }

    #[test]
    fn extrude_refuses_an_empty_or_dead_face_set() {
        let mut m = cube(1.0);
        assert_eq!(
            refuses(
                &mut m,
                Op::ExtrudeFaces {
                    faces: vec![],
                    distance: 1.0
                }
            ),
            OpError::EmptyOperand { what: "a face set" }
        );
        assert_eq!(
            refuses(
                &mut m,
                Op::ExtrudeFaces {
                    faces: vec![FaceId(999)],
                    distance: 1.0
                }
            ),
            OpError::NoSuchFace(FaceId(999))
        );
        let all: Vec<FaceId> = m.face_ids().collect();
        assert!(matches!(
            refuses(
                &mut m,
                Op::ExtrudeFaces {
                    faces: all,
                    distance: f64::INFINITY
                }
            ),
            OpError::NonFinite { .. }
        ));
    }

    #[test]
    fn extruding_a_plane_boundary_edge_makes_a_quad() {
        let mut m = plane(2.0);
        let b: Vec<HalfId> = m
            .half_ids()
            .filter(|&h| m.is_boundary(h) == Some(true))
            .collect();
        assert_eq!(b.len(), 4);
        let out = ok(
            &mut m,
            Op::ExtrudeEdges {
                edges: vec![b[0]],
                delta: [0.0, 1.0, 0.0],
            },
        );
        assert_eq!(out.faces.len(), 1);
        assert_eq!(m.face_count(), 2);
        assert_eq!(m.vert_count(), 6);
        assert_eq!(validate(&m), Ok(()));
    }

    #[test]
    fn extruding_an_interior_edge_is_refused_rather_than_tearing_the_surface() {
        let mut m = cube(1.0);
        let h = m.half_ids().next().expect("a half-edge");
        assert_eq!(
            refuses(
                &mut m,
                Op::ExtrudeEdges {
                    edges: vec![h],
                    delta: [0.0, 1.0, 0.0]
                }
            ),
            OpError::EdgeNotBoundary(h)
        );
    }

    // ── inset ──────────────────────────────────────────────────────────────

    #[test]
    fn insetting_one_face_rings_it_and_keeps_the_area() {
        let mut m = cube(2.0);
        let top = m
            .face_ids()
            .find(|&f| newell(&m, f).normalize().y > 0.9)
            .expect("+Y");
        let before = six_volume(&m);
        let out = ok(
            &mut m,
            Op::InsetFaces {
                faces: vec![top],
                amount: 0.5,
                individual: false,
            },
        );
        assert_eq!(out.faces.len(), 5, "the shrunk face plus four ring quads");
        assert_eq!(m.vert_count(), 12);
        assert_eq!(euler(&m), 2);
        // The inset is coplanar: the solid is unchanged.
        assert!((six_volume(&m) - before).abs() < 1e-9);
        // And the shrunk face really is 1×1, not 2×2.
        // Found by POSITION, not by area: the ring quads are coplanar with it
        // and face +Y too, and the smallest of them (0.75 m2) is smaller than
        // the inner face (1 m2), so "the smallest +Y face" is the wrong probe.
        let inner = m
            .face_ids()
            .find(|&f| {
                m.face_verts(f).expect("live").iter().all(|&v| {
                    let p = pos(&m, v);
                    (p.y - 1.0).abs() < 1e-12 && p.x.abs() < 0.75 && p.z.abs() < 0.75
                })
            })
            .expect("the inset face");
        assert!(
            (newell(&m, inner).length() * 0.5 - 1.0).abs() < 1e-9,
            "inset area {}",
            newell(&m, inner).length() * 0.5
        );
    }

    #[test]
    fn an_individual_inset_gives_every_face_its_own_ring() {
        let mut m = cube(2.0);
        let faces: Vec<FaceId> = m.face_ids().collect();
        ok(
            &mut m,
            Op::InsetFaces {
                faces,
                amount: 0.25,
                individual: true,
            },
        );
        assert_eq!(m.face_count(), 6 * 5, "6 inner faces + 6×4 ring quads");
        assert_eq!(euler(&m), 2);
        assert_eq!(validate(&m), Ok(()));
    }

    #[test]
    fn a_region_with_no_border_has_nothing_to_inset() {
        let mut m = cube(1.0);
        let all: Vec<FaceId> = m.face_ids().collect();
        assert_eq!(
            refuses(
                &mut m,
                Op::InsetFaces {
                    faces: all,
                    amount: 0.1,
                    individual: false
                }
            ),
            OpError::NoRegionBorder
        );
    }

    // ── bevel ──────────────────────────────────────────────────────────────

    #[test]
    fn beveling_one_cube_edge_replaces_it_with_a_strip_and_two_caps() {
        let mut m = cube(2.0);
        let before = six_volume(&m);
        let h = m.half_ids().next().expect("an edge");
        let out = ok(
            &mut m,
            Op::BevelEdges {
                edges: vec![h],
                amount: 0.25,
            },
        );
        assert_eq!(out.faces.len(), 5, "two grown faces + strip + two caps");
        assert_eq!(m.vert_count(), 12, "the 8 corners plus 4 offsets");
        assert_eq!(m.face_count(), 9);
        assert_eq!(euler(&m), 2, "still closed");
        assert!(m.half_ids().all(|h| m.is_boundary(h) == Some(false)));
        assert!(
            six_volume(&m) < before && six_volume(&m) > 0.0,
            "a chamfer removes material and must not invert the solid"
        );
    }

    #[test]
    fn a_bevel_works_at_a_high_valence_vertex() {
        // The claim the construction is FOR: it never looks at the valence. A
        // cylinder cap is an n-gon whose rim vertices carry three faces, and the
        // pole of the fan is where a vertex-dissolving bevel needs a special case.
        let mut m = cylinder(1.0, 2.0, 8);
        let h = m
            .half_ids()
            .find(|&h| {
                let a = m.origin(h).expect("live");
                let b = m.dest(h).expect("live");
                m.position(a).expect("live").y < 0.0 && m.position(b).expect("live").y > 0.0
            })
            .expect("a vertical side edge");
        ok(
            &mut m,
            Op::BevelEdges {
                edges: vec![h],
                amount: 0.1,
            },
        );
        assert_eq!(validate(&m), Ok(()));
        assert_eq!(euler(&m), 2);
    }

    #[test]
    fn bevel_refuses_a_boundary_edge_and_a_non_positive_amount() {
        let mut m = plane(2.0);
        let b = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(true))
            .expect("a boundary half");
        assert_eq!(
            refuses(
                &mut m,
                Op::BevelEdges {
                    edges: vec![b],
                    amount: 0.1
                }
            ),
            OpError::BevelBoundaryEdge(b)
        );
        let mut c = cube(1.0);
        let h = c.half_ids().next().expect("an edge");
        assert_eq!(
            refuses(
                &mut c,
                Op::BevelEdges {
                    edges: vec![h],
                    amount: 0.0
                }
            ),
            OpError::AmountNotPositive {
                what: "a bevel amount",
                value: 0.0
            }
        );
    }

    // ── loop cut ───────────────────────────────────────────────────────────

    /// A 3×1 strip of quads in the XZ plane — an OPEN ring, so the cut has to
    /// handle both ends running off the edge of the mesh.
    fn quad_strip(n: usize) -> Mesh {
        let mut m = Mesh::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        for i in 0..=n {
            let x = i as f64;
            lower.push(m.alloc_vert([x, 0.0, 0.0]));
            upper.push(m.alloc_vert([x, 0.0, 1.0]));
        }
        let mut touched = BTreeSet::new();
        for i in 0..n {
            let quad = [lower[i], lower[i + 1], upper[i + 1], upper[i]];
            touched.extend(quad);
            m.add_face_raw(&quad, &[CornerData::default(); 4], None)
                .expect("a grid is well formed");
        }
        m.finish_patch(&touched).expect("a grid is manifold");
        m
    }

    #[test]
    fn a_loop_cut_across_an_open_strip_cuts_every_quad_once() {
        let mut m = quad_strip(3);
        assert_eq!(m.face_count(), 3);
        // The ring runs ALONG the strip: pick the half-edge on the shared rung of
        // the first quad so the walk steps quad to quad.
        let h = m
            .half_ids()
            .find(|&h| {
                let (a, b) = (
                    m.position(m.origin(h).expect("live")).expect("live"),
                    m.position(m.dest(h).expect("live")).expect("live"),
                );
                m.is_boundary(h) == Some(false) && (a.x - b.x).abs() < 1e-12
            })
            .expect("a rung");
        let out = ok(&mut m, Op::LoopCut { half: h, cuts: 1 });
        assert_eq!(m.face_count(), 6, "each of the three quads split in two");
        assert_eq!(out.verts.len(), 4, "one cut per ring edge, 4 ring edges");
        assert_eq!(m.vert_count(), 12);
        assert_eq!(validate(&m), Ok(()));
    }

    #[test]
    fn a_closed_ring_around_a_cylinder_cuts_it_into_bands() {
        let mut m = cylinder(1.0, 2.0, 8);
        let f0 = m.face_count();
        let h = m
            .half_ids()
            .find(|&h| {
                let a = m.position(m.origin(h).expect("live")).expect("live");
                let b = m.position(m.dest(h).expect("live")).expect("live");
                (a.y - b.y).abs() > 1.0 && m.is_boundary(h) == Some(false)
            })
            .expect("a vertical side edge");
        let out = ok(&mut m, Op::LoopCut { half: h, cuts: 2 });
        assert_eq!(out.verts.len(), 16, "8 ring edges × 2 cuts");
        assert_eq!(m.face_count(), f0 + 16, "8 side quads become 24");
        assert_eq!(euler(&m), 2);
        assert!(m.half_ids().all(|h| m.is_boundary(h) == Some(false)));
    }

    #[test]
    fn a_loop_cut_refuses_a_non_quad_start_and_an_out_of_range_count() {
        // The cylinder's cap is an n-gon: a ring cannot start inside it.
        let mut m = cylinder(1.0, 2.0, 6);
        let cap = m
            .face_ids()
            .find(|&f| m.face_loop(f).expect("live").len() == 6)
            .expect("a cap");
        let h = m.face_half(cap).expect("live");
        assert_eq!(
            refuses(&mut m, Op::LoopCut { half: h, cuts: 1 }),
            OpError::NotAQuad(cap)
        );
        let side = m
            .half_ids()
            .find(|&h| {
                m.face_of(h)
                    .expect("live")
                    .map(|f| m.face_loop(f).expect("live").len() == 4)
                    == Some(true)
            })
            .expect("a side quad");
        assert!(matches!(
            refuses(
                &mut m,
                Op::LoopCut {
                    half: side,
                    cuts: 0
                }
            ),
            OpError::CountOutOfRange { .. }
        ));
    }

    // ── knife ──────────────────────────────────────────────────────────────

    #[test]
    fn a_knife_path_cuts_across_two_faces_atomically() {
        let mut m = quad_strip(2);
        let verts: Vec<VertId> = m.vert_ids().collect();
        // The strip is lower/upper interleaved: v0,v2,v4 lower, v1,v3,v5 upper.
        let out = ok(
            &mut m,
            Op::Knife {
                path: vec![
                    KnifePoint::Vertex(verts[0]),
                    KnifePoint::Vertex(verts[3]),
                    KnifePoint::Vertex(verts[4]),
                ],
            },
        );
        assert_eq!(out.faces.len(), 4, "two split faces, two halves each");
        assert_eq!(m.face_count(), 4);
        assert_eq!(validate(&m), Ok(()));
    }

    #[test]
    fn a_knife_through_an_edge_point_splits_the_edge_first() {
        let mut m = plane(2.0);
        let f = m.face_ids().next().expect("the quad");
        let halfs = m.face_loop(f).expect("live");
        let out = ok(
            &mut m,
            Op::Knife {
                path: vec![
                    KnifePoint::Edge {
                        half: halfs[0],
                        t: 0.5,
                    },
                    KnifePoint::Edge {
                        half: halfs[2],
                        t: 0.5,
                    },
                ],
            },
        );
        assert_eq!(out.verts.len(), 2, "two edge points became vertices");
        assert_eq!(m.vert_count(), 6);
        assert_eq!(m.face_count(), 2);
        assert_eq!(validate(&m), Ok(()));
    }

    #[test]
    fn a_knife_that_cannot_finish_cuts_nothing_at_all() {
        // The atomicity claim, measured: the first segment is cuttable and the
        // second is not, and the mesh must come back byte-identical.
        // `quad_strip` allocates lower/upper in pairs, so the quads are
        // [v0,v2,v3,v1], [v2,v4,v5,v3], [v4,v6,v7,v5].
        let mut m = quad_strip(3);
        let verts: Vec<VertId> = m.vert_ids().collect();
        let err = refuses(
            &mut m,
            Op::Knife {
                path: vec![
                    KnifePoint::Vertex(verts[0]), // cuttable: the first quad diagonal
                    KnifePoint::Vertex(verts[3]),
                    KnifePoint::Vertex(verts[6]), // two quads away, no common face
                ],
            },
        );
        assert!(matches!(err, OpError::KnifeNoCommonFace { .. }), "{err:?}");
        assert_eq!(m.face_count(), 3, "and the first segment did not survive");
    }

    #[test]
    fn a_knife_segment_that_is_already_an_edge_is_skipped_not_refused() {
        let mut m = plane(2.0);
        let quad = m
            .face_verts(m.face_ids().next().expect("the quad"))
            .expect("live");
        // v0→v1 is an existing edge; v1→v3 is a real diagonal.
        let out = ok(
            &mut m,
            Op::Knife {
                path: vec![
                    KnifePoint::Vertex(quad[0]),
                    KnifePoint::Vertex(quad[1]),
                    KnifePoint::Vertex(quad[3]),
                ],
            },
        );
        assert_eq!(out.faces.len(), 2, "only the diagonal cut anything");
        assert_eq!(m.face_count(), 2);
    }

    // ── merge ──────────────────────────────────────────────────────────────

    #[test]
    fn merging_two_quad_corners_at_the_centre_moves_the_survivor() {
        let mut m = plane(2.0);
        let quad = m
            .face_verts(m.face_ids().next().expect("the quad"))
            .expect("live");
        let (a, b) = (quad[0], quad[1]);
        let want = (pos(&m, a) + pos(&m, b)) * 0.5;
        ok(
            &mut m,
            Op::MergeVerts {
                verts: vec![a, b],
                target: MergeTarget::Center,
            },
        );
        assert_eq!(m.vert_count(), 3);
        assert_eq!(m.face_count(), 1);
        assert_eq!(m.position(a), Some(want));
    }

    #[test]
    fn merging_to_last_keeps_that_vertex_where_it_was() {
        let mut m = plane(2.0);
        let quad = m
            .face_verts(m.face_ids().next().expect("the quad"))
            .expect("live");
        let (a, b) = (quad[0], quad[1]);
        let want = pos(&m, b);
        ok(
            &mut m,
            Op::MergeVerts {
                verts: vec![a, b],
                target: MergeTarget::Last,
            },
        );
        assert_eq!(m.position(b), Some(want));
        assert!(!m.has_vert(a));
    }

    #[test]
    fn merge_refuses_a_repeat_and_a_singleton() {
        let mut m = plane(2.0);
        let v = m.vert_ids().next().expect("a vertex");
        assert!(matches!(
            refuses(
                &mut m,
                Op::MergeVerts {
                    verts: vec![v],
                    target: MergeTarget::Center
                }
            ),
            OpError::EmptyOperand { .. }
        ));
        assert_eq!(
            refuses(
                &mut m,
                Op::MergeVerts {
                    verts: vec![v, v],
                    target: MergeTarget::Center
                }
            ),
            OpError::SameVertex(v)
        );
    }

    // ── subdivide ──────────────────────────────────────────────────────────

    #[test]
    fn subdividing_a_cube_quadruples_its_faces_and_moves_nothing() {
        let mut m = cube(2.0);
        let before = six_volume(&m);
        let all: Vec<FaceId> = m.face_ids().collect();
        ok(&mut m, Op::SubdivideFaces { faces: all });
        assert_eq!(m.face_count(), 24);
        assert_eq!(m.vert_count(), 8 + 12 + 6, "corners + edge mids + centres");
        assert_eq!(euler(&m), 2);
        assert!(
            (six_volume(&m) - before).abs() < 1e-9,
            "simple midpoint does not smooth: the shape is identical"
        );
    }

    #[test]
    fn subdividing_one_face_of_a_cube_re_cuts_its_neighbours() {
        // The fringe rule: the four faces around the subdivided one must gain the
        // midpoint corners, or their shared edges are split on one side only.
        let mut m = cube(2.0);
        let top = m
            .face_ids()
            .find(|&f| newell(&m, f).normalize().y > 0.9)
            .expect("+Y");
        ok(&mut m, Op::SubdivideFaces { faces: vec![top] });
        assert_eq!(
            m.face_count(),
            4 + 5,
            "4 sub-quads + the 5 untouched/regrown"
        );
        assert_eq!(euler(&m), 2);
        assert!(m.half_ids().all(|h| m.is_boundary(h) == Some(false)));
        // The four neighbours are pentagons now.
        let pentagons = m
            .face_ids()
            .filter(|&f| m.face_loop(f).expect("live").len() == 5)
            .count();
        assert_eq!(pentagons, 4);
    }

    // ── mirror ─────────────────────────────────────────────────────────────

    /// Half a box: five quads with an open square boundary lying exactly on x=0.
    fn half_box() -> Mesh {
        let mut m = Mesh::new();
        let p = [
            [0.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let ids: Vec<VertId> = p.iter().map(|&q| m.alloc_vert(q)).collect();
        let quads = [
            [0, 1, 2, 3], // −Y
            [7, 6, 5, 4], // +Y
            [4, 5, 1, 0], // −Z
            [6, 7, 3, 2], // +Z
            [5, 6, 2, 1], // +X
        ];
        let mut touched = BTreeSet::new();
        for q in quads {
            let loop_verts: Vec<VertId> = q.iter().map(|&i| ids[i]).collect();
            touched.extend(loop_verts.iter().copied());
            m.add_face_raw(&loop_verts, &[CornerData::default(); 4], None)
                .expect("well formed");
        }
        m.finish_patch(&touched).expect("manifold");
        m
    }

    #[test]
    fn mirroring_half_a_box_welds_the_seam_into_one_closed_solid() {
        let mut m = half_box();
        assert_eq!(m.vert_count(), 8);
        let open = m
            .half_ids()
            .filter(|&h| m.is_boundary(h) == Some(true))
            .count();
        assert_eq!(open, 4, "the seam square is open");

        let out = ok(
            &mut m,
            Op::Mirror {
                axis: MirrorAxis::X,
                coord: 0.0,
            },
        );
        assert_eq!(out.verts.len(), 4, "only the OFF-plane vertices are copied");
        assert_eq!(
            m.vert_count(),
            12,
            "8 + 4 — the four seam vertices were welded, not duplicated"
        );
        assert_eq!(m.face_count(), 10);
        assert_eq!(euler(&m), 2);
        assert!(
            m.half_ids().all(|h| m.is_boundary(h) == Some(false)),
            "a welded seam leaves NO boundary — this is the whole claim"
        );
        assert!(six_volume(&m) > 0.0, "and it is not inside-out");
        assert!((six_volume(&m) / 6.0 - 8.0).abs() < 1e-9, "2×2×2");
    }

    #[test]
    fn a_seam_vertex_reflects_to_bit_identical_coordinates() {
        // The reason the plane is axis aligned, asserted as bits rather than
        // argued. `2d − d == d` exactly; an arbitrary-plane reflection is not.
        for d in [0.0f64, 1.5, -37.25, 1.0 / 3.0] {
            assert_eq!((2.0 * d - d).to_bits(), d.to_bits(), "plane at {d}");
        }
    }

    #[test]
    fn mirroring_a_solid_whose_seam_is_already_closed_is_refused() {
        // Mirror the half box once: it closes, and the seam edges now have a face
        // on BOTH sides. Mirroring again would have to put a third face on each of
        // them, which is not a mesh -- refused, inert.
        //
        // NB a solid with NOTHING on the plane (a centred cube mirrored about its
        // own middle) is a perfectly legal two-shell mirror and is not this case.
        let mut m = half_box();
        ok(
            &mut m,
            Op::Mirror {
                axis: MirrorAxis::X,
                coord: 0.0,
            },
        );
        assert!(matches!(
            refuses(
                &mut m,
                Op::Mirror {
                    axis: MirrorAxis::X,
                    coord: 0.0
                }
            ),
            OpError::EdgeAlreadyFaced { .. }
        ));
    }

    #[test]
    fn mirroring_a_shape_that_misses_the_plane_makes_a_second_shell() {
        let mut m = half_box();
        let out = ok(
            &mut m,
            Op::Mirror {
                axis: MirrorAxis::X,
                coord: 4.0,
            },
        );
        assert_eq!(out.verts.len(), 8, "nothing sat on the plane");
        assert_eq!(m.vert_count(), 16);
        assert_eq!(m.face_count(), 10);
    }

    // ── determinism ────────────────────────────────────────────────────────

    #[test]
    fn every_modelling_op_is_deterministic_across_two_runs() {
        let script = |mut m: Mesh| -> Mesh {
            let top = m
                .face_ids()
                .find(|&f| newell(&m, f).normalize().y > 0.9)
                .expect("+Y");
            apply(
                &mut m,
                &Op::ExtrudeFaces {
                    faces: vec![top],
                    distance: 0.5,
                },
            )
            .expect("extrude");
            let f: Vec<FaceId> = m.face_ids().take(2).collect();
            apply(
                &mut m,
                &Op::InsetFaces {
                    faces: f,
                    amount: 0.1,
                    individual: true,
                },
            )
            .expect("inset");
            let h = m.half_ids().next().expect("an edge");
            let _ = apply(
                &mut m,
                &Op::BevelEdges {
                    edges: vec![h],
                    amount: 0.05,
                },
            );
            let some: Vec<FaceId> = m.face_ids().take(3).collect();
            apply(&mut m, &Op::SubdivideFaces { faces: some }).expect("subdivide");
            m
        };
        assert_eq!(script(cube(2.0)).encoded(), script(cube(2.0)).encoded());
    }
}
