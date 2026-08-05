//! **Cook-time Voronoi pre-fracture** (P22.2): turning a `.inf_mesh` into a
//! chunk hierarchy a runtime can swap in when the asset breaks.
//!
//! This module is the whole geometry half of the fracture pipeline. It is
//! **Ring 0 and pure**: no physics, no renderer, no IO, no clock, no threads —
//! `fracture_mesh` is a function of `(mesh bytes, asset id, params)` and
//! nothing else, which is what makes the derived `.inf_fracture` byte-identical
//! from run to run and host to host. Runtime destruction (damage, the swap, the
//! structural solve) is P22.3; everything here is its *input*.
//!
//! # The pipeline, in order
//!
//! 1. **Seeds.** A [`Hash64`] stream (the SplitMix64 finalizer, the same mixer
//!    and the same counter-based discipline as `inf_pcg::hash`) is started from
//!    the mesh's own asset GUID plus the authored seed offset. Every site
//!    coordinate is a pure function of `(guid, seed, site index, axis, attempt)`
//!    — never of a stateful RNG, never of iteration order.
//! 2. **Sites.** `chunk_count` points, rejection-sampled into the mesh's convex
//!    hull inside its AABB (see [`Sites`]).
//! 3. **The hull.** A deterministic incremental convex hull over the mesh's
//!    vertices, its coplanar triangles merged into distinct **half-spaces**.
//! 4. **Cells.** Each chunk is the intersection of half-spaces: the six AABB
//!    planes, the bisectors against the other sites (a Voronoi cell by
//!    definition), and the hull planes that actually reach the cell. The
//!    polytope is recovered by [`polytope_from_halfspaces`].
//! 5. **Chunks.** Each polytope becomes vertices + indices + a hull point set
//!    for the collider + volume + centre of mass + the adjacency P22.3's
//!    structural graph is built from.
//!
//! # Honest scope of v1
//!
//! * **A concave mesh fractures as its convex hull.** Chunk geometry is
//!   `Voronoi cell ∩ hull`, so a mesh with a hollow, an archway or a re-entrant
//!   notch produces chunks that fill it in. This is not an oversight to be
//!   apologised for later: it is what makes every chunk **convex**, which is
//!   what makes it a `ColliderShape3D::ConvexHull` with a real mass instead of a
//!   massless trimesh. Fracturing the true surface means clipping the source
//!   triangles per cell and producing non-convex chunks that need convex
//!   *decomposition* before they can be dynamic bodies — a strictly larger
//!   problem, and a documented non-goal here.
//! * **The hull is simplified to at most [`HULL_PLANE_BUDGET`] half-spaces.**
//!   Keeping the largest-area faces produces a polytope that *contains* the true
//!   hull, so the union of the chunks is a superset of the mesh's hull and the
//!   volume gate reads `Σ chunks >= hull` rather than `==`. Bounding the plane
//!   count is what keeps the (deliberately simple, deliberately exact) vertex
//!   enumeration affordable.
//! * **Two levels, and only two.** Level 0 is the intact asset; level 1 is the
//!   `N` chunks. Fracturing a chunk again (a depth-2 hierarchy for progressive
//!   damage) is a documented non-goal for v1: it multiplies the cook, and
//!   nothing in P22.3 consumes it.
//! * **Interior faces get one generated material slot.** Exterior faces inherit
//!   a slot from the source vertices lying on their plane, which is an
//!   approximation for a hull-fractured exterior — see [`FractureAsset::slots`].
//!
//! # Determinism rules obeyed here
//!
//! Pure `f64`; no `HashMap` (and in fact no hashing container at all — every
//! association is a sorted `Vec` or a `BTreeMap`); no `std` trig (angular
//! ordering uses [`pseudo_angle`], which is exact division only — the P14 law
//! that `f32` `std` trig is not bit-portable, applied one register wider out of
//! caution); no iteration over an unordered collection; and no dependence on
//! the order sites are handed in (`fracture_from_sites` sorts them
//! canonically — the invariance is a gate, not a hope).

use std::collections::BTreeMap;

use glam::DVec3;
use inf_asset::{AssetId, AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

use crate::asset::{Aabb, MeshAsset, MeshVertex};

// ─────────────────────────────────────────────────────────────────────────────
// Derived asset identity
// ─────────────────────────────────────────────────────────────────────────────

/// The fixed salt XORed into a mesh GUID to derive its `.inf_fracture` GUID.
///
/// Same construction, and the same reasoning, as [`crate::VMESH_ID_SALT`]'s
/// neighbour in `inf-vgeom`: XOR with a constant is a bijection, so distinct
/// meshes always yield distinct fracture ids, and the salt makes a collision
/// with an *authored* asset id vanishingly unlikely (the cook guards the
/// remaining case). A runtime finds a mesh's fracture data by computing the id,
/// so no side index has to ship or stay in sync.
///
/// It lives **here**, in the Ring-0 crate that owns the `.inf_fracture` format,
/// rather than in the packager — the lesson `inf_vgeom::VMESH_ID_SALT` was moved
/// to Ring 0 to learn, taken before the second copy exists rather than after the
/// third.
pub const FRACTURE_ID_SALT: u128 = 0x2622_0002_4652_4143_8b17_4d9e_c05a_31f7;

/// Derive the deterministic `.inf_fracture` asset id for a given mesh id.
///
/// Involutive (`derived_fracture_id(derived_fracture_id(x)) == x`), because the
/// salt is XORed.
pub fn derived_fracture_id(mesh_id: AssetId) -> AssetId {
    AssetId(uuid::Uuid::from_u128(
        mesh_id.uuid().as_u128() ^ FRACTURE_ID_SALT,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic hashing (the `inf_pcg::hash` conventions, locally)
// ─────────────────────────────────────────────────────────────────────────────

/// The golden-ratio odd constant used to decorrelate successive mixed words.
const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64 finalizer: a full-avalanche 64→64 bit mix.
#[inline]
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A counter-based hasher: every draw is a pure function of the integers folded
/// into it, so site placement is independent of evaluation order.
///
/// A deliberate 30-line copy of `inf_pcg::hash::Hash64` rather than a dependency:
/// `inf-mesh` has no business depending on the PCG runtime, and the mixer is a
/// *specification* (the SplitMix64 finalizer constants) rather than an
/// implementation that could drift. The constants are asserted equal by
/// `splitmix_matches_the_pcg_mixer`.
#[derive(Clone, Copy, Debug)]
pub struct Hash64(u64);

impl Hash64 {
    /// Start a hash stream from `seed`.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Hash64(mix64(seed ^ GOLDEN))
    }
    /// Fold an unsigned word into the state.
    #[inline]
    pub fn mix_u64(self, v: u64) -> Self {
        Hash64(mix64(self.0 ^ v.wrapping_mul(GOLDEN)))
    }
    /// The raw 64-bit hash.
    #[inline]
    pub fn finish(self) -> u64 {
        self.0
    }
    /// A uniform `f64` in `[0, 1)` (53-bit mantissa precision).
    #[inline]
    pub fn unit(self) -> f64 {
        (self.0 >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Planes and convex polytopes
// ─────────────────────────────────────────────────────────────────────────────

/// An oriented plane. A point `p` is **inside** when `normal · p + d <= 0`, so a
/// convex polytope is simply the set of points inside every one of its planes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Unit outward normal.
    pub normal: DVec3,
    /// Plane offset: the plane is `normal · p + d = 0`.
    pub d: f64,
}

impl Plane {
    /// A plane with the given outward normal through `point`. Returns `None` if
    /// the normal cannot be normalized.
    pub fn through(point: DVec3, normal: DVec3) -> Option<Self> {
        let len = normal.length();
        if !len.is_finite() || len < 1e-30 {
            return None;
        }
        let n = normal / len;
        Some(Plane {
            normal: n,
            d: -n.dot(point),
        })
    }

    /// Signed distance from `p` to the plane: negative inside, positive outside.
    #[inline]
    pub fn distance(&self, p: DVec3) -> f64 {
        self.normal.dot(p) + self.d
    }
}

/// One face of a [`Polytope`]: the plane it lies in (by index into the polytope's
/// plane list) and its vertex loop, ordered counter-clockwise about the outward
/// normal.
#[derive(Clone, Debug, PartialEq)]
pub struct PolyFace {
    /// Index into the plane list the polytope was built from.
    pub plane: usize,
    /// Vertex indices, CCW about `plane`'s outward normal.
    pub loop_: Vec<usize>,
}

/// A bounded convex polytope recovered from a set of half-spaces.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Polytope {
    pub vertices: Vec<DVec3>,
    pub faces: Vec<PolyFace>,
}

impl Polytope {
    /// Enclosed volume, m³. Positive for the outward winding this module
    /// produces; the sign is itself a correctness signal (a flipped face would
    /// subtract).
    pub fn volume(&self) -> f64 {
        let mut v = 0.0;
        for f in &self.faces {
            let a = self.vertices[f.loop_[0]];
            for w in f.loop_[1..].windows(2) {
                let b = self.vertices[w[0]];
                let c = self.vertices[w[1]];
                v += a.dot(b.cross(c));
            }
        }
        v / 6.0
    }

    /// Volume centroid (centre of mass of the solid at uniform density).
    /// Falls back to the vertex average for a degenerate (zero-volume) body.
    pub fn centroid(&self) -> DVec3 {
        let mut vol = 0.0;
        let mut acc = DVec3::ZERO;
        for f in &self.faces {
            let a = self.vertices[f.loop_[0]];
            for w in f.loop_[1..].windows(2) {
                let b = self.vertices[w[0]];
                let c = self.vertices[w[1]];
                let tet = a.dot(b.cross(c)) / 6.0;
                vol += tet;
                // Tetra (origin, a, b, c) centroid.
                acc += (a + b + c) * (tet * 0.25);
            }
        }
        if vol.abs() > 1e-18 {
            acc / vol
        } else if self.vertices.is_empty() {
            DVec3::ZERO
        } else {
            self.vertices.iter().copied().sum::<DVec3>() / self.vertices.len() as f64
        }
    }

    /// Total surface area, m² — used to weight hull-plane simplification.
    pub fn area(&self) -> f64 {
        let mut a = 0.0;
        for f in &self.faces {
            a += self.face_area(f);
        }
        a
    }

    /// Area of one face, m².
    pub fn face_area(&self, f: &PolyFace) -> f64 {
        let o = self.vertices[f.loop_[0]];
        let mut acc = DVec3::ZERO;
        for w in f.loop_[1..].windows(2) {
            acc += (self.vertices[w[0]] - o).cross(self.vertices[w[1]] - o);
        }
        acc.length() * 0.5
    }
}

/// A monotone stand-in for `atan2(y, x)` mapped to `[0, 4)` — the "diamond
/// angle".
///
/// Used to order a face's vertices around its centroid. It is **exact division
/// only**: no `sin`/`cos`/`atan2` anywhere, because `std` trig is not
/// bit-portable across platforms (the P14 law, found on `f32` and applied here
/// out of caution) and a face whose vertices ordered differently on two hosts
/// would produce two different chunk meshes from one cook.
///
/// It is not the angle — it is a strictly increasing function of the angle,
/// which is all a sort needs.
#[inline]
pub fn pseudo_angle(x: f64, y: f64) -> f64 {
    let s = x.abs() + y.abs();
    if s == 0.0 {
        return 0.0;
    }
    let p = y / s;
    if x < 0.0 {
        2.0 - p
    } else if y < 0.0 {
        4.0 + p
    } else {
        p
    }
}

/// Recover the bounded convex polytope that is the intersection of `planes`.
///
/// **Exact vertex enumeration.** A vertex of the intersection is the solution of
/// some three planes that satisfies all the others, so every triple is solved
/// (Cramer's rule) and tested. That is `O(P⁴)` in the plane count, and it is the
/// right trade here: `P` is kept small by construction (six AABB planes, the
/// bisectors of a dozen sites, and only the hull planes that actually reach the
/// cell — see [`active_planes`]), the cook is offline, and the alternative —
/// incremental face clipping — trades a large constant factor for a body of
/// edge-case bookkeeping (cap-face chaining across coincident and on-plane
/// vertices) whose failures are silent holes in chunk geometry. If cook time
/// ever becomes the constraint, *that* is the documented optimization; it is not
/// a correctness fix.
///
/// `scale` is the world size of the problem (an AABB diagonal), used to set the
/// distance tolerances. Returns `None` when the intersection is empty or has no
/// volume.
pub fn polytope_from_halfspaces(planes: &[Plane], scale: f64) -> Option<Polytope> {
    if planes.len() < 4 {
        return None;
    }
    let eps = 1e-9 * scale.max(1.0);
    let merge = 1e-7 * scale.max(1.0);

    // ── vertices: every feasible triple intersection ────────────────────────
    let mut verts: Vec<DVec3> = Vec::new();
    for i in 0..planes.len() {
        for j in (i + 1)..planes.len() {
            for k in (j + 1)..planes.len() {
                let (a, b, c) = (planes[i], planes[j], planes[k]);
                let n1 = a.normal;
                let n2 = b.normal;
                let n3 = c.normal;
                let det = n1.dot(n2.cross(n3));
                // Near-parallel triples have no well-conditioned intersection.
                if det.abs() < 1e-10 {
                    continue;
                }
                let p = (n2.cross(n3) * -a.d + n3.cross(n1) * -b.d + n1.cross(n2) * -c.d) / det;
                if !p.is_finite() {
                    continue;
                }
                if planes.iter().any(|pl| pl.distance(p) > eps) {
                    continue;
                }
                // Merge duplicates (a vertex where four or more planes meet is
                // found once per triple). A linear scan, not a hashed key: the
                // vertex count is tens, and a quantized key would put two
                // arbitrarily close points in different buckets.
                if verts.iter().any(|v| v.distance(p) <= merge) {
                    continue;
                }
                verts.push(p);
            }
        }
    }
    if verts.len() < 4 {
        return None;
    }

    // ── faces: the vertices lying on each plane, ordered ────────────────────
    let mut faces: Vec<PolyFace> = Vec::new();
    for (pi, pl) in planes.iter().enumerate() {
        let on: Vec<usize> = (0..verts.len())
            .filter(|&vi| pl.distance(verts[vi]).abs() <= merge)
            .collect();
        if on.len() < 3 {
            continue; // a redundant plane, or one touching at an edge/point
        }
        let centre = on.iter().map(|&vi| verts[vi]).sum::<DVec3>() / on.len() as f64;
        // A basis on the plane. `u` points at the vertex farthest from the
        // centroid (never a near-zero vector); `w = n × u` completes a frame in
        // which increasing pseudo-angle is CCW about the outward normal, because
        // `u × w = n`.
        let far = on
            .iter()
            .copied()
            .max_by(|&x, &y| {
                let (dx, dy) = (
                    verts[x].distance_squared(centre),
                    verts[y].distance_squared(centre),
                );
                dx.partial_cmp(&dy)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(y.cmp(&x))
            })
            .expect("on.len() >= 3");
        let u = (verts[far] - centre).normalize_or_zero();
        if u == DVec3::ZERO {
            continue;
        }
        let w = pl.normal.cross(u);
        let mut ordered = on;
        ordered.sort_by(|&x, &y| {
            let key = |vi: usize| {
                let r = verts[vi] - centre;
                pseudo_angle(r.dot(u), r.dot(w))
            };
            key(x)
                .partial_cmp(&key(y))
                .unwrap_or(std::cmp::Ordering::Equal)
                // Ties (two vertices at the same angle) break on index so the
                // order is total and reproducible.
                .then(x.cmp(&y))
        });
        faces.push(PolyFace {
            plane: pi,
            loop_: ordered,
        });
    }
    if faces.len() < 4 {
        return None;
    }

    // Drop vertices no face uses, keeping the surviving order stable.
    let mut used = vec![false; verts.len()];
    for f in &faces {
        for &vi in &f.loop_ {
            used[vi] = true;
        }
    }
    let mut remap = vec![usize::MAX; verts.len()];
    let mut kept: Vec<DVec3> = Vec::with_capacity(verts.len());
    for (vi, &u) in used.iter().enumerate() {
        if u {
            remap[vi] = kept.len();
            kept.push(verts[vi]);
        }
    }
    for f in &mut faces {
        for vi in &mut f.loop_ {
            *vi = remap[*vi];
        }
    }

    let poly = Polytope {
        vertices: kept,
        faces,
    };
    // The winding is asserted, not assumed: a flipped face would make the
    // divergence-theorem volume negative, and a chunk with inside-out geometry
    // is exactly the defect this catches at the source.
    // NaN-safe: `<=` is false for NaN, so a non-finite volume is caught by the
    // explicit finiteness test rather than by the comparison's polarity.
    let volume = poly.volume();
    if !volume.is_finite() || volume <= 0.0 {
        return None;
    }
    Some(poly)
}

/// The planes of `candidates` that actually reach `cell` — i.e. those with at
/// least one cell vertex strictly outside them.
///
/// A convex cell entirely inside a half-space is unaffected by it, so dropping
/// those planes before the (quartic) enumeration is exact, not approximate. It
/// is what keeps a mesh with dozens of hull planes affordable: a chunk in the
/// middle of a wall is reached by three or four of them.
/// Returns **indices** into `candidates`, not the planes themselves, so the
/// caller can map a face back to the plane it came from without comparing two
/// `f64` planes for equality.
fn active_planes(cell: &Polytope, candidates: &[Plane], eps: f64) -> Vec<usize> {
    (0..candidates.len())
        .filter(|&k| {
            cell.vertices
                .iter()
                .any(|&v| candidates[k].distance(v) > eps)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Convex hull
// ─────────────────────────────────────────────────────────────────────────────

/// One triangular face of the incremental hull.
#[derive(Clone, Copy, Debug)]
struct HullTri {
    v: [usize; 3],
    plane: Plane,
}

/// The convex hull of `points` as a triangle list with outward-facing planes, or
/// `None` for a point set that bounds no volume (fewer than four points, or
/// collinear/coplanar ones).
///
/// A plain **incremental** hull: build a deterministic starting tetrahedron from
/// four extreme points, then add the remaining points in index order, deleting
/// the faces each one can see and stitching a cone to the horizon. `O(n · F)`,
/// which is the right shape for a cook, and every choice inside it (which
/// extreme point, which order, which tie-break) is by index, so two runs build
/// the same hull.
///
/// It is deliberately *ours* rather than parry's: `parry` is a runtime collision
/// library that `inf-physics` alone may name (the three-ring facade), and a cook
/// kernel that produced geometry a different library would disagree with is the
/// exact drift the ring rules exist to prevent.
pub fn convex_hull_faces(points: &[DVec3]) -> Option<Vec<HullFaceOut>> {
    if points.len() < 4 {
        return None;
    }
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for p in points {
        if !p.is_finite() {
            return None;
        }
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let extent = (hi - lo).length();
    if !extent.is_finite() || extent <= 0.0 {
        return None;
    }
    let eps = 1e-9 * extent;

    // ── a deterministic starting tetrahedron ────────────────────────────────
    let i0 = (0..points.len())
        .min_by(|&a, &b| cmp_point(points[a], points[b]).then(a.cmp(&b)))
        .expect("non-empty");
    let i1 = farthest(points, |p| p.distance(points[i0]))?;
    if points[i1].distance(points[i0]) <= eps {
        return None;
    }
    let axis = points[i1] - points[i0];
    let i2 = farthest(points, |p| (p - points[i0]).cross(axis).length())?;
    if (points[i2] - points[i0]).cross(axis).length() <= eps * extent {
        return None; // collinear
    }
    let base = Plane::through(
        points[i0],
        (points[i1] - points[i0]).cross(points[i2] - points[i0]),
    )?;
    let i3 = farthest(points, |p| base.distance(p).abs())?;
    if base.distance(points[i3]).abs() <= eps {
        return None; // coplanar
    }

    let mut tris: Vec<HullTri> = Vec::new();
    let seed = [i0, i1, i2, i3];
    let inner = (points[i0] + points[i1] + points[i2] + points[i3]) / 4.0;
    for drop in 0..4 {
        let t: Vec<usize> = (0..4).filter(|&i| i != drop).map(|i| seed[i]).collect();
        let (a, b, c) = (t[0], t[1], t[2]);
        let pl = Plane::through(
            points[a],
            (points[b] - points[a]).cross(points[c] - points[a]),
        )?;
        // Orient outward: the tetra's own interior point must be inside.
        if pl.distance(inner) > 0.0 {
            tris.push(HullTri {
                v: [a, c, b],
                plane: Plane {
                    normal: -pl.normal,
                    d: -pl.d,
                },
            });
        } else {
            tris.push(HullTri {
                v: [a, b, c],
                plane: pl,
            });
        }
    }

    // ── add the rest, in index order ────────────────────────────────────────
    for (pi, p) in points.iter().enumerate() {
        if seed.contains(&pi) {
            continue;
        }
        let visible: Vec<usize> = (0..tris.len())
            .filter(|&ti| tris[ti].plane.distance(*p) > eps)
            .collect();
        if visible.is_empty() {
            continue;
        }
        // Horizon: a directed edge of a visible face whose twin is NOT in the
        // visible set. `BTreeMap`, never a hash set — the horizon walk must be
        // reproducible.
        let mut dir: BTreeMap<(usize, usize), ()> = BTreeMap::new();
        for &ti in &visible {
            let v = tris[ti].v;
            for e in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                dir.insert(e, ());
            }
        }
        let horizon: Vec<(usize, usize)> = dir
            .keys()
            .copied()
            .filter(|&(a, b)| !dir.contains_key(&(b, a)))
            .collect();
        // Remove visible faces (descending, so indices stay valid).
        for &ti in visible.iter().rev() {
            tris.swap_remove(ti);
        }
        for (a, b) in horizon {
            if let Some(pl) =
                Plane::through(points[a], (points[b] - points[a]).cross(*p - points[a]))
            {
                tris.push(HullTri {
                    v: [a, b, pi],
                    plane: pl,
                });
            }
        }
        // `swap_remove` reorders the face list, which would make the result
        // depend on removal order. Restoring a canonical order after every point
        // is what keeps the hull a pure function of the input.
        tris.sort_by_key(|t| t.v);
    }
    if tris.len() < 4 {
        return None;
    }
    Some(
        tris.into_iter()
            .map(|t| HullFaceOut {
                v: t.v,
                plane: t.plane,
            })
            .collect(),
    )
}

/// A hull triangle: its three source-point indices and its outward plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullFaceOut {
    pub v: [usize; 3],
    pub plane: Plane,
}

/// Lexicographic point compare (x, then y, then z) — a total order with no
/// float-equality surprises beyond the ones `partial_cmp` already has.
fn cmp_point(a: DVec3, b: DVec3) -> std::cmp::Ordering {
    a.x.partial_cmp(&b.x)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
        .then(a.z.partial_cmp(&b.z).unwrap_or(std::cmp::Ordering::Equal))
}

/// Index of the point maximizing `f`, ties broken by the **lower** index.
fn farthest(points: &[DVec3], f: impl Fn(DVec3) -> f64) -> Option<usize> {
    (0..points.len()).max_by(|&a, &b| {
        f(points[a])
            .partial_cmp(&f(points[b]))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.cmp(&a))
    })
}

/// At most this many half-spaces are kept from a mesh's convex hull.
///
/// Hull faces are merged into distinct planes first (a box's twelve triangles
/// become six), so a boxy prop is unaffected; the budget only bites on curved
/// geometry. Keeping the **largest-area** planes yields a polytope that
/// *contains* the true hull, so the fracture never loses material — it rounds a
/// sphere up towards its circumscribed polytope. The number is a cook-cost
/// decision: vertex enumeration is quartic in the plane count, and 48 keeps a
/// twelve-chunk fracture of a dense mesh in the low seconds.
pub const HULL_PLANE_BUDGET: usize = 48;

/// Merge a hull's triangles into distinct half-spaces, largest total area first,
/// truncated to [`HULL_PLANE_BUDGET`].
///
/// Returns planes in a canonical order (first appearance in the hull's own
/// sorted face list) so the result is a pure function of the point set.
fn hull_halfspaces(points: &[DVec3], faces: &[HullFaceOut], extent: f64) -> Vec<Plane> {
    let cos_eps = 1.0 - 1e-9;
    let d_eps = 1e-7 * extent.max(1.0);
    // (plane, total area, first-appearance index)
    let mut groups: Vec<(Plane, f64, usize)> = Vec::new();
    for (fi, f) in faces.iter().enumerate() {
        let (a, b, c) = (points[f.v[0]], points[f.v[1]], points[f.v[2]]);
        let area = (b - a).cross(c - a).length() * 0.5;
        match groups.iter_mut().find(|(pl, _, _)| {
            pl.normal.dot(f.plane.normal) >= cos_eps && (pl.d - f.plane.d).abs() <= d_eps
        }) {
            Some(g) => g.1 += area,
            None => groups.push((f.plane, area, fi)),
        }
    }
    if groups.len() > HULL_PLANE_BUDGET {
        groups.sort_by(|x, y| {
            y.1.partial_cmp(&x.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.2.cmp(&y.2))
        });
        groups.truncate(HULL_PLANE_BUDGET);
        groups.sort_by_key(|g| g.2);
    }
    groups.into_iter().map(|g| g.0).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────

/// The default number of chunks a `Destructible` asks for.
///
/// Twelve: below about four pieces a break reads as a *cut* rather than a
/// shatter, and above about thirty the debris budget (P22.4) and the cook cost
/// dominate long before the extra pieces are legible in motion. It is a
/// starting point an author overrides per asset, not a claim about physics.
pub const DEFAULT_CHUNK_COUNT: u32 = 12;
/// Fewer than two chunks is not a fracture.
pub const MIN_CHUNK_COUNT: u32 = 2;
/// Vertex enumeration is quartic in the plane count and every site adds a
/// bisector, so the cap is a cook-cost ceiling rather than a design limit.
pub const MAX_CHUNK_COUNT: u32 = 64;

/// Clamp an authored chunk count into `[MIN_CHUNK_COUNT, MAX_CHUNK_COUNT]`.
pub fn clamp_chunk_count(n: u32) -> u32 {
    n.clamp(MIN_CHUNK_COUNT, MAX_CHUNK_COUNT)
}

/// A hull smaller than this is not worth fracturing: one cubic centimetre.
///
/// Below it the chunks are sub-millimetre, no player can tell them from the
/// intact asset, and each still costs a rigid body. The cook says so rather than
/// quietly emitting them.
pub const MIN_FRACTURE_VOLUME_M3: f64 = 1e-6;

/// What a `Destructible` asks the cook to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FractureParams {
    /// Authored seed **offset**, folded in beside the mesh's own GUID. Changing
    /// it re-shatters the same mesh a different way; leaving it alone means two
    /// cooks of the same content produce the same pieces.
    pub seed: u32,
    /// Target chunk count. Clamped by [`clamp_chunk_count`]; the produced count
    /// can still be lower when a site's cell misses the hull entirely.
    pub chunk_count: u32,
}

impl Default for FractureParams {
    fn default() -> Self {
        Self {
            seed: 0,
            chunk_count: DEFAULT_CHUNK_COUNT,
        }
    }
}

/// Why a mesh was **not** fractured. The `VmeshSkip` shape: a reason the cook can
/// turn into a precise advisory, not a failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FractureSkip {
    /// No triangles at all — there is nothing to break, and nothing to say
    /// about it beyond the `Destructible` that asked.
    NoGeometry,
    /// Real geometry, but flat or collinear: it bounds no volume, so it has no
    /// convex hull and no chunk could ever have a mass.
    Degenerate,
    /// A real solid, but smaller than [`MIN_FRACTURE_VOLUME_M3`].
    TooSmall {
        /// The measured hull volume, m³.
        volume_m3: f64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// The `.inf_fracture` payload
// ─────────────────────────────────────────────────────────────────────────────

/// One contiguous run of a chunk's index buffer drawn with a single material
/// slot.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChunkSection {
    /// Index into [`FractureAsset::slots`].
    pub material_slot: u32,
    /// First index into the chunk's index buffer.
    pub first_index: u32,
    /// Number of indices in the run (a multiple of three).
    pub index_count: u32,
}

/// One level-1 piece of a fractured mesh.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FractureChunk {
    /// Renderable geometry, in the **source mesh's local space** — so P22.3 can
    /// swap the intact mesh for the chunks without moving anything.
    pub vertices: Vec<MeshVertex>,
    /// Triangle indices into `vertices`.
    pub indices: Vec<u32>,
    /// Index runs by material slot, in ascending slot order.
    pub sections: Vec<ChunkSection>,
    /// The chunk's convex-hull point set, `f64`, local space — the collider
    /// (`inf_physics::d3::ColliderShape3D::ConvexHull`). Kept separate from
    /// `vertices` because a render vertex is `f32` and duplicated per face,
    /// while a collider wants the exact `f64` corner set once.
    pub hull_points: Vec<[f64; 3]>,
    /// Enclosed volume, **m³**. With the `Destructible`'s `density_kg_m3` this
    /// is the chunk's mass — which is the entire reason chunks are convex.
    pub volume_m3: f64,
    /// Volume centroid, local space, metres.
    pub center_of_mass: [f64; 3],
    /// Indices of the chunks sharing a Voronoi face with this one — P22.3's
    /// structural-integrity graph, precomputed. Sorted ascending, and symmetric
    /// across the whole asset.
    pub neighbors: Vec<u32>,
}

/// The `.inf_fracture` payload: a mesh's level-1 chunk set.
///
/// Two levels and only two: level 0 is the source `.inf_mesh` (which this never
/// modifies or copies), level 1 is [`chunks`](Self::chunks).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FractureAsset {
    /// On-disk schema version. See [`FractureAsset::CURRENT_VERSION`].
    pub schema_version: u32,
    /// GUID of the `.inf_mesh` this was derived from, as raw bytes — the
    /// provenance a tool needs to say *what broke*. (Raw bytes, not an
    /// `AssetId`, so the payload's wire form does not depend on `uuid`'s serde
    /// representation.)
    pub source_mesh: [u8; 16],
    /// The source mesh's local-space bounds.
    pub bounds: Aabb,
    /// The seed and count this asset was built with — so a cook can tell a
    /// stale derived asset from a current one without re-running the fracture.
    pub seed: u32,
    /// Material slot names: the source mesh's slots, followed by exactly one
    /// generated interior slot at index [`interior_slot`](Self::interior_slot).
    ///
    /// **Exterior slots are an approximation, deliberately.** A chunk's exterior
    /// faces are hull faces, not source triangles, so there is no exact slot to
    /// inherit; each exterior face takes the lowest slot among the source
    /// vertices lying on its plane (0 when none do). For the single-slot props
    /// v1 targets this is exact; for a multi-material mesh it is a reasonable
    /// guess that never invents a slot the mesh did not declare.
    pub slots: Vec<String>,
    /// Index into [`slots`](Self::slots) of the generated interior material —
    /// the freshly-exposed inside of the break, which is never one of the
    /// mesh's authored slots.
    pub interior_slot: u32,
    /// The chunks, in canonical (site) order.
    pub chunks: Vec<FractureChunk>,
}

impl FractureAsset {
    /// Schema v1 — the format as first shipped (P22.2).
    ///
    /// The decode ladder exists from day one *deliberately*: `migrate` below is
    /// where a v1 payload will be lifted when v2 appends something, and having
    /// the shape in place means the first growth is an edit rather than a
    /// redesign. The version is also written into the payload's first field, so
    /// a reader can branch on it before decoding the rest — the `.inf_lvl`
    /// discipline, one container down.
    pub const CURRENT_VERSION: u32 = 1;

    /// Total chunk volume, m³.
    pub fn total_volume_m3(&self) -> f64 {
        self.chunks.iter().map(|c| c.volume_m3).sum()
    }

    /// Every `(a, b)` adjacency pair with `a < b`, sorted — the edge list of
    /// P22.3's structural graph.
    pub fn adjacency_pairs(&self) -> Vec<(u32, u32)> {
        let mut out: Vec<(u32, u32)> = Vec::new();
        for (i, c) in self.chunks.iter().enumerate() {
            for &n in &c.neighbors {
                let (a, b) = (i as u32, n);
                if a < b {
                    out.push((a, b));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

impl AssetPayload for FractureAsset {
    const KIND: AssetKind = AssetKind::Fracture;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// The version ladder. Today it has one rung and rejects the future; a v2
    /// adds its `1 => lift` arm here rather than reinterpreting shorter bytes as
    /// the longer layout (bincode is positional — the law paid for four times
    /// over in the scene codec).
    fn migrate(self) -> inf_asset::Result<Self> {
        match self.schema_version {
            0 | 1 => Ok(self),
            found => Err(inf_asset::AssetError::SchemaTooNew {
                kind: Self::KIND.slug(),
                found,
                current: Self::SCHEMA_VERSION,
            }),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Site generation
// ─────────────────────────────────────────────────────────────────────────────

/// How many times a site is re-drawn while it lands outside the hull before the
/// draw is accepted anyway.
///
/// A site outside the hull produces a Voronoi cell that may miss the solid
/// entirely, costing a chunk. Rejection sampling fixes the common case (a hull
/// filling most of its AABB) cheaply; a bounded number of attempts keeps the
/// function total for a sliver-shaped mesh, where accepting an outside site
/// simply means fewer chunks than asked for — which the cook reports rather than
/// looping forever.
const SITE_ATTEMPTS: u64 = 64;

/// The fracture sites for a mesh: `count` points inside `hull_planes` where
/// possible, drawn from `bounds`, as a pure function of `(guid, seed, index)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Sites(pub Vec<DVec3>);

/// Generate the sites for one fracture. Public so the determinism gates can
/// permute them and check the result does not move.
pub fn sites_for(
    guid: AssetId,
    params: FractureParams,
    lo: DVec3,
    hi: DVec3,
    hull: &[Plane],
) -> Sites {
    let count = clamp_chunk_count(params.chunk_count) as u64;
    let id = guid.uuid().as_u128();
    let base = Hash64::new((id >> 64) as u64)
        .mix_u64(id as u64)
        .mix_u64(params.seed as u64);
    let span = hi - lo;
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count {
        let mut chosen = None;
        for attempt in 0..SITE_ATTEMPTS {
            let h = base.mix_u64(i).mix_u64(attempt);
            let p = lo
                + DVec3::new(
                    span.x * h.mix_u64(0).unit(),
                    span.y * h.mix_u64(1).unit(),
                    span.z * h.mix_u64(2).unit(),
                );
            if hull.iter().all(|pl| pl.distance(p) <= 0.0) {
                chosen = Some(p);
                break;
            }
            if chosen.is_none() {
                chosen = Some(p); // the fallback, overwritten by any hit
            }
        }
        out.push(chosen.expect("SITE_ATTEMPTS >= 1"));
    }
    Sites(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// The fracture itself
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-fracture `mesh` into a [`FractureAsset`], or say why it could not be.
///
/// Pure: the output is a function of `(mesh, guid, params)` alone, which is what
/// makes the derived asset byte-identical between cooks and between hosts.
pub fn fracture_mesh(
    mesh: &MeshAsset,
    guid: AssetId,
    params: FractureParams,
) -> Result<FractureAsset, FractureSkip> {
    // ── the source point cloud, with each point's material slot ─────────────
    let mut cloud: Vec<DVec3> = Vec::new();
    let mut slot_of: Vec<u32> = Vec::new();
    let mut triangles = 0usize;
    for sm in &mesh.submeshes {
        triangles += sm.triangle_count();
        let slot = sm.material_slot.unwrap_or(0);
        for v in &sm.vertices {
            cloud.push(DVec3::new(
                v.position[0] as f64,
                v.position[1] as f64,
                v.position[2] as f64,
            ));
            slot_of.push(slot);
        }
    }
    if triangles == 0 || cloud.len() < 4 {
        return Err(FractureSkip::NoGeometry);
    }

    let faces = convex_hull_faces(&cloud).ok_or(FractureSkip::Degenerate)?;
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for p in &cloud {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let extent = (hi - lo).length();
    let hull_planes = hull_halfspaces(&cloud, &faces, extent);

    // The six AABB planes are ALWAYS in the set, and that is load-bearing:
    // the hull's plane list may have been truncated to the budget, and an
    // intersection of a subset of a hull's half-spaces can be unbounded. The box
    // is what guarantees every cell is a bounded polytope.
    let box_planes = aabb_planes(lo, hi);
    let hull_poly = {
        let mut all = box_planes.to_vec();
        all.extend_from_slice(&hull_planes);
        polytope_from_halfspaces(&all, extent).ok_or(FractureSkip::Degenerate)?
    };
    let hull_volume = hull_poly.volume();
    if hull_volume < MIN_FRACTURE_VOLUME_M3 {
        return Err(FractureSkip::TooSmall {
            volume_m3: hull_volume,
        });
    }

    let sites = sites_for(guid, params, lo, hi, &hull_planes);
    let cells = fracture_from_sites(&sites, &box_planes, &hull_planes, extent);

    // ── material slots ──────────────────────────────────────────────────────
    let mut slots: Vec<String> = mesh.material_slots.clone();
    if slots.is_empty() {
        slots.push("Default".to_string());
    }
    let interior_slot = slots.len() as u32;
    slots.push(INTERIOR_SLOT_NAME.to_string());
    // Slot per *exterior* plane: the lowest slot among source vertices on it.
    let exterior_slot_of: Vec<u32> = box_planes
        .iter()
        .chain(hull_planes.iter())
        .map(|pl| {
            let tol = 1e-7 * extent.max(1.0);
            cloud
                .iter()
                .enumerate()
                .filter(|(_, p)| pl.distance(**p).abs() <= tol)
                .map(|(i, _)| slot_of[i])
                .min()
                .unwrap_or(0)
        })
        .collect();

    // ── chunks ──────────────────────────────────────────────────────────────
    let mut chunks: Vec<FractureChunk> = Vec::with_capacity(cells.len());
    for cell in &cells {
        chunks.push(build_chunk(
            cell,
            interior_slot,
            &exterior_slot_of,
            box_planes.len(),
            hull_planes.len(),
        ));
    }

    Ok(FractureAsset {
        schema_version: FractureAsset::CURRENT_VERSION,
        source_mesh: *guid.uuid().as_bytes(),
        bounds: mesh.bounds,
        seed: params.seed,
        slots,
        interior_slot,
        chunks,
    })
}

/// The generated material slot every freshly-exposed interior face uses.
pub const INTERIOR_SLOT_NAME: &str = "Fracture Interior";

/// One cell of the fracture: its polytope plus which of its faces came from
/// which plane source.
#[derive(Clone, Debug)]
pub struct Cell {
    /// The chunk's geometry. Face `plane` indices address
    /// `[box planes .., hull planes .., bisectors ..]`.
    pub poly: Polytope,
    /// For each bisector plane in this cell's plane list, the site it separates
    /// this one from — parallel to the bisector block of the plane list.
    pub bisector_sites: Vec<usize>,
    /// This cell's own site index.
    pub site: usize,
}

/// Build the cells for `sites`, each clipped to the box and the hull.
///
/// **Order-invariant by construction**: the sites are canonicalized (sorted
/// lexicographically) before anything geometric happens, so handing the same set
/// of points in a different order produces the same cells in the same order.
/// That is the property `permuted_sites_produce_identical_chunks` asserts, and
/// the sort is why it can.
pub fn fracture_from_sites(
    sites: &Sites,
    box_planes: &[Plane],
    hull_planes: &[Plane],
    extent: f64,
) -> Vec<Cell> {
    let mut ordered: Vec<(DVec3, usize)> = sites.0.iter().copied().zip(0..).collect();
    ordered.sort_by(|a, b| cmp_point(a.0, b.0).then(a.1.cmp(&b.1)));
    let pts: Vec<DVec3> = ordered.iter().map(|s| s.0).collect();

    let eps = 1e-9 * extent.max(1.0);
    let mut cells = Vec::new();
    for i in 0..pts.len() {
        // The Voronoi half-spaces: `p` is in cell i iff it is at least as close
        // to site i as to every other site, which is exactly the half-space on
        // site i's side of each bisector.
        let mut bisectors: Vec<Plane> = Vec::new();
        let mut bisector_sites: Vec<usize> = Vec::new();
        for j in 0..pts.len() {
            if i == j {
                continue;
            }
            let mid = (pts[i] + pts[j]) * 0.5;
            if let Some(pl) = Plane::through(mid, pts[j] - pts[i]) {
                bisectors.push(pl);
                bisector_sites.push(j);
            }
        }
        // Cheap cell first (box + bisectors) so the hull planes can be filtered
        // down to the ones that actually reach it before the quartic step.
        let mut cheap: Vec<Plane> = box_planes.to_vec();
        cheap.extend_from_slice(&bisectors);
        let Some(cell0) = polytope_from_halfspaces(&cheap, extent) else {
            continue;
        };
        let active = active_planes(&cell0, hull_planes, eps);

        let mut all: Vec<Plane> = box_planes.to_vec();
        let hull_used = active.len();
        all.extend(active.iter().map(|&k| hull_planes[k]));
        all.extend_from_slice(&bisectors);
        let Some(poly) = polytope_from_halfspaces(&all, extent) else {
            continue; // the cell misses the solid entirely
        };
        // Re-address the face plane indices onto the canonical
        // `[box, hull, bisector]` layout the chunk builder expects: the hull
        // block above holds only the ACTIVE planes, so a face on hull slot `k`
        // is really on `active[k]`.
        let mut poly = poly;
        for f in &mut poly.faces {
            f.plane = if f.plane < box_planes.len() {
                f.plane
            } else if f.plane < box_planes.len() + hull_used {
                box_planes.len() + active[f.plane - box_planes.len()]
            } else {
                box_planes.len() + hull_planes.len() + (f.plane - box_planes.len() - hull_used)
            };
        }
        cells.push(Cell {
            poly,
            bisector_sites,
            site: i,
        });
    }

    // Adjacency is derived from the surviving bisector faces of BOTH sides and
    // unioned, so a face detected from either cell counts. The two derivations
    // agreeing is asserted as a test (`adjacency_is_symmetric_from_both_sides`)
    // rather than made true by the union alone — a union makes symmetry vacuous,
    // and a vacuous check hides the real intrusion (the P19 law).
    cells
}

/// The six outward half-spaces of an axis-aligned box.
fn aabb_planes(lo: DVec3, hi: DVec3) -> [Plane; 6] {
    [
        Plane {
            normal: DVec3::NEG_X,
            d: lo.x,
        },
        Plane {
            normal: DVec3::X,
            d: -hi.x,
        },
        Plane {
            normal: DVec3::NEG_Y,
            d: lo.y,
        },
        Plane {
            normal: DVec3::Y,
            d: -hi.y,
        },
        Plane {
            normal: DVec3::NEG_Z,
            d: lo.z,
        },
        Plane {
            normal: DVec3::Z,
            d: -hi.z,
        },
    ]
}

/// Turn one cell into a renderable, collidable chunk.
fn build_chunk(
    cell: &Cell,
    interior_slot: u32,
    exterior_slot_of: &[u32],
    box_count: usize,
    hull_count: usize,
) -> FractureChunk {
    let poly = &cell.poly;
    // Group faces by the slot they draw with, so the index buffer is one
    // contiguous run per slot (ascending) and `sections` needs no sorting pass.
    let mut by_slot: BTreeMap<u32, Vec<&PolyFace>> = BTreeMap::new();
    let mut neighbors: Vec<u32> = Vec::new();
    for f in &poly.faces {
        let slot = if f.plane < box_count + hull_count {
            // A face on the box or on a hull plane is the ORIGINAL surface.
            exterior_slot_of.get(f.plane).copied().unwrap_or(0)
        } else {
            // A face on a bisector is freshly exposed by the break.
            let b = f.plane - box_count - hull_count;
            if let Some(&other) = cell.bisector_sites.get(b) {
                neighbors.push(other as u32);
            }
            interior_slot
        };
        by_slot.entry(slot).or_default().push(f);
    }
    neighbors.sort_unstable();
    neighbors.dedup();

    let mut vertices: Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut sections: Vec<ChunkSection> = Vec::new();
    for (slot, faces) in by_slot {
        let first = indices.len() as u32;
        for f in faces {
            let n = face_normal(poly, f);
            // A planar UV frame on the face: 1 UV unit == 1 metre, which keeps a
            // fracture-interior material at the same texel density as the rest of
            // the world (SI, architecture rule 6).
            let o = poly.vertices[f.loop_[0]];
            let u = (poly.vertices[f.loop_[1]] - o).normalize_or_zero();
            let w = n.cross(u);
            let base = vertices.len() as u32;
            for &vi in &f.loop_ {
                let p = poly.vertices[vi];
                let r = p - o;
                vertices.push(MeshVertex {
                    position: [p.x as f32, p.y as f32, p.z as f32],
                    normal: [n.x as f32, n.y as f32, n.z as f32],
                    uv: [r.dot(u) as f32, r.dot(w) as f32],
                    tangent: [u.x as f32, u.y as f32, u.z as f32, 1.0],
                });
            }
            for k in 1..(f.loop_.len() as u32 - 1) {
                indices.extend_from_slice(&[base, base + k, base + k + 1]);
            }
        }
        sections.push(ChunkSection {
            material_slot: slot,
            first_index: first,
            index_count: indices.len() as u32 - first,
        });
    }

    let com = poly.centroid();
    FractureChunk {
        vertices,
        indices,
        sections,
        hull_points: poly.vertices.iter().map(|v| [v.x, v.y, v.z]).collect(),
        volume_m3: poly.volume(),
        center_of_mass: [com.x, com.y, com.z],
        neighbors,
    }
}

/// A face's outward unit normal, taken from its own winding (so it is the normal
/// the geometry actually has, not the one its plane claims).
fn face_normal(poly: &Polytope, f: &PolyFace) -> DVec3 {
    let o = poly.vertices[f.loop_[0]];
    let mut acc = DVec3::ZERO;
    for w in f.loop_[1..].windows(2) {
        acc += (poly.vertices[w[0]] - o).cross(poly.vertices[w[1]] - o);
    }
    acc.normalize_or_zero()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::SubMesh;

    fn guid(n: u128) -> AssetId {
        AssetId(uuid::Uuid::from_u128(n))
    }

    /// A unit cube mesh centred on the origin, as 12 triangles.
    fn cube_mesh(half: f32) -> MeshAsset {
        let c = |x: f32, y: f32, z: f32| MeshVertex {
            position: [x * half, y * half, z * half],
            ..Default::default()
        };
        // 8 corners; the index list only has to *cover* the corners for the hull.
        let vertices = vec![
            c(-1.0, -1.0, -1.0),
            c(1.0, -1.0, -1.0),
            c(1.0, 1.0, -1.0),
            c(-1.0, 1.0, -1.0),
            c(-1.0, -1.0, 1.0),
            c(1.0, -1.0, 1.0),
            c(1.0, 1.0, 1.0),
            c(-1.0, 1.0, 1.0),
        ];
        let indices = vec![
            0, 1, 2, 0, 2, 3, // -z
            4, 6, 5, 4, 7, 6, // +z
            0, 4, 5, 0, 5, 1, // -y
            3, 2, 6, 3, 6, 7, // +y
            0, 3, 7, 0, 7, 4, // -x
            1, 5, 6, 1, 6, 2, // +x
        ];
        MeshAsset::new(
            vec![SubMesh {
                name: "cube".into(),
                vertices,
                indices,
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["Stone".into()],
        )
    }

    // ── the mixer ───────────────────────────────────────────────────────────

    /// The local `Hash64` is a copy of `inf_pcg::hash::Hash64` on purpose; the
    /// constants are what make the copy safe, so they are pinned to literals
    /// rather than to the other crate (which `inf-mesh` must not depend on).
    #[test]
    fn splitmix_matches_the_pcg_mixer() {
        assert_eq!(GOLDEN, 0x9e37_79b9_7f4a_7c15);
        // The two finalizer multipliers, pinned by their effect rather than by
        // reading them back: a typo in either moves these words, and moving them
        // would re-shatter every committed asset.
        assert_eq!(mix64(0), 0);
        assert_eq!(mix64(1), 0x5692_161d_100b_05e5);
        assert_eq!(mix64(u64::MAX), 0xb4d0_55fc_f2cb_bd7b);
        for i in 0..1000u64 {
            let u = Hash64::new(i).mix_u64(i * 7).unit();
            assert!((0.0..1.0).contains(&u), "u={u}");
        }
    }

    // ── polytope geometry ───────────────────────────────────────────────────

    #[test]
    fn a_box_recovers_as_a_box() {
        let lo = DVec3::new(-1.0, -2.0, -3.0);
        let hi = DVec3::new(1.0, 2.0, 3.0);
        let p = polytope_from_halfspaces(&aabb_planes(lo, hi), 8.0).unwrap();
        assert_eq!(p.vertices.len(), 8, "a box has eight corners");
        assert_eq!(p.faces.len(), 6);
        assert!(
            (p.volume() - 2.0 * 4.0 * 6.0).abs() < 1e-9,
            "{}",
            p.volume()
        );
        assert!(p.centroid().length() < 1e-9, "{:?}", p.centroid());
        for f in &p.faces {
            assert_eq!(f.loop_.len(), 4, "a box face is a quad");
            // The winding really is outward: the face normal from the loop and
            // the plane's own normal agree.
            let n = face_normal(&p, f);
            assert!(n.dot(aabb_planes(lo, hi)[f.plane].normal) > 0.999);
        }
    }

    /// Cutting a box with one plane through its centre halves it, and the two
    /// halves' volumes add back up — the conservation property every later gate
    /// rests on, at the smallest possible scale.
    #[test]
    fn a_half_space_cut_conserves_volume() {
        let (lo, hi) = (DVec3::splat(-1.0), DVec3::splat(1.0));
        let cut = Plane::through(DVec3::new(0.25, 0.0, 0.0), DVec3::X).unwrap();
        let flip = Plane {
            normal: -cut.normal,
            d: -cut.d,
        };
        let mut a = aabb_planes(lo, hi).to_vec();
        a.push(cut);
        let mut b = aabb_planes(lo, hi).to_vec();
        b.push(flip);
        let va = polytope_from_halfspaces(&a, 4.0).unwrap().volume();
        let vb = polytope_from_halfspaces(&b, 4.0).unwrap().volume();
        assert!((va - 1.25 * 2.0 * 2.0).abs() < 1e-9, "{va}");
        assert!((va + vb - 8.0).abs() < 1e-9, "{va} + {vb}");
    }

    /// Every cell of a Voronoi diagram clipped to a box tiles that box: the cell
    /// volumes sum to the box's, exactly enough for `f64`. This is the
    /// conservation gate for the cell machinery itself, with no hull involved.
    #[test]
    fn voronoi_cells_tile_their_box() {
        let (lo, hi) = (DVec3::new(-2.0, -1.0, -3.0), DVec3::new(2.0, 1.0, 3.0));
        let boxp = aabb_planes(lo, hi);
        let sites = sites_for(guid(0xBEEF), FractureParams::default(), lo, hi, &[]);
        let cells = fracture_from_sites(&sites, &boxp, &[], (hi - lo).length());
        assert_eq!(cells.len(), DEFAULT_CHUNK_COUNT as usize);
        let sum: f64 = cells.iter().map(|c| c.poly.volume()).sum();
        let want = 4.0 * 2.0 * 6.0;
        assert!(
            (sum - want).abs() < 1e-7 * want,
            "cells summed to {sum}, box is {want}"
        );
        // Every cell vertex really is closest to its own site — the definition
        // of a Voronoi cell, checked rather than assumed.
        let pts: Vec<DVec3> = {
            let mut o: Vec<(DVec3, usize)> = sites.0.iter().copied().zip(0..).collect();
            o.sort_by(|a, b| cmp_point(a.0, b.0).then(a.1.cmp(&b.1)));
            o.into_iter().map(|s| s.0).collect()
        };
        for c in &cells {
            for v in &c.poly.vertices {
                let mine = v.distance(pts[c.site]);
                for (j, p) in pts.iter().enumerate() {
                    assert!(
                        mine <= v.distance(*p) + 1e-6,
                        "vertex of cell {} is closer to site {j}",
                        c.site
                    );
                }
            }
        }
    }

    // ── the hull ────────────────────────────────────────────────────────────

    #[test]
    fn hull_of_a_cube_cloud_is_six_planes_and_the_right_volume() {
        let cloud: Vec<DVec3> = {
            let mut v = Vec::new();
            for sx in [-1.0, 1.0f64] {
                for sy in [-1.0, 1.0f64] {
                    for sz in [-1.0, 1.0f64] {
                        v.push(DVec3::new(sx, sy, sz));
                    }
                }
            }
            v.push(DVec3::ZERO); // interior point: absorbed
            v.push(DVec3::splat(0.3));
            v
        };
        let faces = convex_hull_faces(&cloud).expect("a cube cloud has a hull");
        assert_eq!(faces.len(), 12, "a cube hull triangulates into 12 faces");
        let planes = hull_halfspaces(&cloud, &faces, 2.0 * 3f64.sqrt());
        assert_eq!(
            planes.len(),
            6,
            "coplanar triangles merge into 6 half-spaces"
        );
        let poly = polytope_from_halfspaces(&planes, 4.0).unwrap();
        assert!((poly.volume() - 8.0).abs() < 1e-9, "{}", poly.volume());
    }

    /// Brute force is the only honest check for a hull: every source point must
    /// be inside every face's plane, and each face plane must be *supported*
    /// (some point actually on it). A hull that merely looked plausible would
    /// fail one of the two.
    #[test]
    fn hull_planes_support_the_cloud_and_contain_it() {
        // A lumpy cloud: a cube plus points pushed out along the diagonals.
        let mut cloud: Vec<DVec3> = Vec::new();
        for i in 0..40u64 {
            let h = Hash64::new(0x5EED).mix_u64(i);
            cloud.push(DVec3::new(
                h.mix_u64(0).unit() * 2.0 - 1.0,
                h.mix_u64(1).unit() * 2.0 - 1.0,
                h.mix_u64(2).unit() * 2.0 - 1.0,
            ));
        }
        let faces = convex_hull_faces(&cloud).unwrap();
        for f in &faces {
            let mut support = false;
            for p in &cloud {
                let d = f.plane.distance(*p);
                assert!(d <= 1e-9, "point outside a hull face by {d}");
                support |= d.abs() <= 1e-9;
            }
            assert!(support, "a hull face plane touches no point");
        }
        // …and the farthest point along a few directions is ON the hull, which
        // is what "the hull is not too big" means.
        for dir in [DVec3::X, DVec3::Y, DVec3::Z, DVec3::ONE.normalize()] {
            let best = cloud
                .iter()
                .map(|p| p.dot(dir))
                .fold(f64::NEG_INFINITY, f64::max);
            let hull_best = faces
                .iter()
                .flat_map(|f| f.v)
                .map(|i| cloud[i].dot(dir))
                .fold(f64::NEG_INFINITY, f64::max);
            assert!((best - hull_best).abs() < 1e-9, "{best} vs {hull_best}");
        }
    }

    #[test]
    fn degenerate_clouds_have_no_hull() {
        assert!(convex_hull_faces(&[]).is_none());
        assert!(convex_hull_faces(&[DVec3::ZERO; 8]).is_none());
        let collinear: Vec<DVec3> = (0..8).map(|i| DVec3::X * i as f64).collect();
        assert!(convex_hull_faces(&collinear).is_none());
        let coplanar: Vec<DVec3> = (0..5)
            .flat_map(|i| (0..5).map(move |j| DVec3::new(i as f64, 0.0, j as f64)))
            .collect();
        assert!(
            convex_hull_faces(&coplanar).is_none(),
            "a slab is not a solid"
        );
    }

    // ── the fracture ────────────────────────────────────────────────────────

    #[test]
    fn a_cube_fractures_into_chunks_that_conserve_its_volume() {
        let mesh = cube_mesh(0.5); // 1 m cube
        let f = fracture_mesh(&mesh, guid(1), FractureParams::default()).unwrap();
        assert_eq!(f.schema_version, FractureAsset::CURRENT_VERSION);
        assert!(f.chunks.len() >= 2, "got {} chunks", f.chunks.len());
        assert!(f.chunks.len() <= DEFAULT_CHUNK_COUNT as usize);

        // A cube's hull IS the cube, so the chunks tile it exactly.
        let sum = f.total_volume_m3();
        assert!(
            (sum - 1.0).abs() < 1e-7,
            "chunks summed to {sum} m3, want 1"
        );
        // Every chunk is a real solid with real mass.
        for (i, c) in f.chunks.iter().enumerate() {
            assert!(c.volume_m3 > 0.0, "chunk {i} has no volume");
            assert!(c.hull_points.len() >= 4, "chunk {i} has no collider hull");
            assert!(!c.indices.is_empty() && c.indices.len() % 3 == 0);
            // Sections cover the index buffer exactly once, in slot order.
            let covered: u32 = c.sections.iter().map(|s| s.index_count).sum();
            assert_eq!(covered as usize, c.indices.len(), "chunk {i} sections");
            let mut cursor = 0;
            let mut last_slot = None;
            for s in &c.sections {
                assert_eq!(s.first_index, cursor);
                cursor += s.index_count;
                if let Some(l) = last_slot {
                    assert!(s.material_slot > l, "sections must ascend by slot");
                }
                last_slot = Some(s.material_slot);
            }
            // The centre of mass is inside the chunk's own bounds.
            let com = DVec3::from_array(c.center_of_mass);
            let pts: Vec<DVec3> = c
                .hull_points
                .iter()
                .map(|p| DVec3::from_array(*p))
                .collect();
            let lo = pts
                .iter()
                .copied()
                .fold(DVec3::splat(f64::INFINITY), DVec3::min);
            let hi = pts
                .iter()
                .copied()
                .fold(DVec3::splat(f64::NEG_INFINITY), DVec3::max);
            assert!(com.cmpge(lo - 1e-9).all() && com.cmple(hi + 1e-9).all());
        }
    }

    /// The interior of a break is a **new** material, and the outside keeps the
    /// mesh's. A chunk with no interior face would mean nothing broke.
    #[test]
    fn interior_faces_get_the_generated_slot_and_exterior_faces_keep_the_meshs() {
        let mesh = cube_mesh(0.5);
        let f = fracture_mesh(&mesh, guid(2), FractureParams::default()).unwrap();
        assert_eq!(
            f.slots,
            vec!["Stone".to_string(), INTERIOR_SLOT_NAME.to_string()]
        );
        assert_eq!(f.interior_slot, 1);
        let mut saw_interior = 0;
        let mut saw_exterior = 0;
        for c in &f.chunks {
            for s in &c.sections {
                assert!(
                    (s.material_slot as usize) < f.slots.len(),
                    "a chunk names a slot the asset does not declare"
                );
                if s.material_slot == f.interior_slot {
                    saw_interior += 1;
                } else {
                    saw_exterior += 1;
                    assert_eq!(s.material_slot, 0, "the cube has one authored slot");
                }
            }
        }
        assert!(saw_interior >= f.chunks.len(), "every chunk has a cut face");
        assert!(saw_exterior > 0, "no chunk kept any original surface");
    }

    /// Adjacency is a graph, and P22.3's structural solve depends on all three
    /// of these being true.
    #[test]
    fn adjacency_is_symmetric_and_connected() {
        let mesh = cube_mesh(0.5);
        let f = fracture_mesh(&mesh, guid(3), FractureParams::default()).unwrap();
        let n = f.chunks.len();
        // Symmetric.
        for (i, c) in f.chunks.iter().enumerate() {
            for &j in &c.neighbors {
                assert_ne!(j as usize, i, "a chunk is not its own neighbour");
                assert!((j as usize) < n, "neighbour {j} out of range");
                assert!(
                    f.chunks[j as usize].neighbors.contains(&(i as u32)),
                    "chunk {i} lists {j} but not the other way round"
                );
            }
            assert!(
                c.neighbors.windows(2).all(|w| w[0] < w[1]),
                "sorted, deduped"
            );
        }
        // Connected: a convex solid cut by planes cannot fall into two pieces.
        let mut seen = vec![false; n];
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(i) = stack.pop() {
            for &j in &f.chunks[i].neighbors {
                if !seen[j as usize] {
                    seen[j as usize] = true;
                    stack.push(j as usize);
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "the chunk graph is disconnected");
        assert!(!f.adjacency_pairs().is_empty());
    }

    /// Two runs, byte for byte; a different seed really changes the break; and
    /// the same seed on a different mesh id does not.
    #[test]
    fn the_fracture_is_deterministic_and_seed_stable() {
        let mesh = cube_mesh(0.5);
        let a =
            inf_asset::encode(&fracture_mesh(&mesh, guid(7), FractureParams::default()).unwrap())
                .unwrap();
        let b =
            inf_asset::encode(&fracture_mesh(&mesh, guid(7), FractureParams::default()).unwrap())
                .unwrap();
        assert_eq!(a, b, "two runs of one fracture must be byte-identical");

        let other_seed = inf_asset::encode(
            &fracture_mesh(
                &mesh,
                guid(7),
                FractureParams {
                    seed: 1,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert_ne!(
            a, other_seed,
            "the authored seed offset must change the break"
        );

        let other_mesh =
            inf_asset::encode(&fracture_mesh(&mesh, guid(8), FractureParams::default()).unwrap())
                .unwrap();
        assert_ne!(a, other_mesh, "the mesh id must change the break");

        // And it round-trips through the payload codec unchanged.
        let back: FractureAsset = inf_asset::decode(&a).unwrap();
        assert_eq!(inf_asset::encode(&back).unwrap(), a);
    }

    /// The geometry must not depend on the order the sites arrive in — the
    /// property that makes a parallel or reordered producer safe.
    #[test]
    fn permuted_sites_produce_identical_chunks() {
        let (lo, hi) = (DVec3::splat(-1.0), DVec3::splat(1.0));
        let boxp = aabb_planes(lo, hi);
        let sites = sites_for(guid(0x51E5), FractureParams::default(), lo, hi, &[]);
        let straight = fracture_from_sites(&sites, &boxp, &[], (hi - lo).length());

        // A deterministic non-identity permutation: reverse, then rotate by 3.
        let mut permuted: Vec<DVec3> = sites.0.clone();
        permuted.reverse();
        permuted.rotate_left(3);
        assert_ne!(permuted, sites.0, "the permutation must actually permute");
        let shuffled = fracture_from_sites(&Sites(permuted), &boxp, &[], (hi - lo).length());

        assert_eq!(straight.len(), shuffled.len());
        for (a, b) in straight.iter().zip(&shuffled) {
            assert_eq!(a.poly, b.poly, "a permuted site set moved the geometry");
        }
    }

    /// Independently derived adjacency from each side must agree. Production
    /// takes the per-cell faces directly (no union), so this is the check that
    /// the two derivations really are one fact — the vacuity the P19 law warns
    /// about is avoided precisely by not unioning.
    #[test]
    fn adjacency_is_symmetric_from_both_sides() {
        let mesh = cube_mesh(0.75);
        let f = fracture_mesh(
            &mesh,
            guid(11),
            FractureParams {
                seed: 5,
                chunk_count: 16,
            },
        )
        .unwrap();
        let mut from_a: Vec<(u32, u32)> = Vec::new();
        let mut from_b: Vec<(u32, u32)> = Vec::new();
        for (i, c) in f.chunks.iter().enumerate() {
            for &j in &c.neighbors {
                if (i as u32) < j {
                    from_a.push((i as u32, j));
                } else {
                    from_b.push((j, i as u32));
                }
            }
        }
        from_a.sort_unstable();
        from_b.sort_unstable();
        assert_eq!(from_a, from_b, "the two sides disagree about a shared face");
        assert!(!from_a.is_empty());
    }

    /// A **curved, many-faced** mesh: the hull plane budget bites, and the
    /// stated bound is what the chunks must obey.
    ///
    /// Keeping the largest-area hull faces yields a polytope that *contains* the
    /// true hull, so `Σ chunks >= hull` is the direction of the guarantee — the
    /// fracture never loses material, it rounds a curved surface up towards its
    /// circumscribed polytope. The upper bound is the honest cost of that: a
    /// 48-plane approximation of a sphere over-states its volume by a few per
    /// cent, and the test pins how much rather than leaving it unsaid.
    #[test]
    fn a_curved_mesh_fractures_within_the_stated_hull_bound() {
        // A ~sphere: 300 points on the unit sphere, generated without trig by
        // normalizing hashed cube points (rejecting the degenerate centre).
        let mut vertices = Vec::new();
        let mut i = 0u64;
        while vertices.len() < 300 {
            let h = Hash64::new(0x5E_1A11).mix_u64(i);
            i += 1;
            let p = DVec3::new(
                h.mix_u64(0).unit() * 2.0 - 1.0,
                h.mix_u64(1).unit() * 2.0 - 1.0,
                h.mix_u64(2).unit() * 2.0 - 1.0,
            );
            if p.length() < 0.05 {
                continue;
            }
            let n = p.normalize();
            vertices.push(MeshVertex {
                position: [n.x as f32, n.y as f32, n.z as f32],
                ..Default::default()
            });
        }
        let n = vertices.len() as u32;
        let mesh = MeshAsset::new(
            vec![SubMesh {
                name: "ball".into(),
                vertices,
                // The index buffer only has to make `triangle_count` non-zero —
                // the fracture reads the vertex cloud, not the topology (which is
                // exactly the hull-only scope this test also documents).
                indices: (0..n).collect(),
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["Rock".into()],
        );

        let f = fracture_mesh(&mesh, guid(31), FractureParams::default()).unwrap();
        assert!(f.chunks.len() >= 2);

        // The true hull of the cloud, for the bound.
        let cloud: Vec<DVec3> = mesh.submeshes[0]
            .vertices
            .iter()
            .map(|v| {
                DVec3::new(
                    v.position[0] as f64,
                    v.position[1] as f64,
                    v.position[2] as f64,
                )
            })
            .collect();
        let faces = convex_hull_faces(&cloud).unwrap();
        let planes = hull_halfspaces(&cloud, &faces, 2.0);
        // The budget must genuinely BIND here, or the bound below would be
        // measuring nothing: a sphere's hull triangles are all on distinct
        // planes, so the unbudgeted count is in the hundreds.
        assert!(
            faces.len() > HULL_PLANE_BUDGET * 4,
            "this mesh is not curved enough to exercise the budget: {} faces",
            faces.len()
        );
        assert_eq!(
            planes.len(),
            HULL_PLANE_BUDGET,
            "the budget must be what limits the plane count"
        );
        assert_eq!(f.chunks.len(), DEFAULT_CHUNK_COUNT as usize);
        // The unbudgeted hull, for the comparison the budget is measured against.
        let exact: f64 = {
            let mut v = 0.0;
            for t in &faces {
                let (a, b, c) = (cloud[t.v[0]], cloud[t.v[1]], cloud[t.v[2]]);
                v += a.dot(b.cross(c));
            }
            v / 6.0
        };
        let sum = f.total_volume_m3();
        assert!(
            sum >= exact,
            "chunks {sum} lost material against hull {exact}"
        );
        assert!(
            sum <= exact * 1.25,
            "the 48-plane hull over-states the sphere by more than the stated 25%: \
             {sum} vs {exact}"
        );
    }

    // ── refusals ────────────────────────────────────────────────────────────

    #[test]
    fn refusals_are_values_with_reasons() {
        let empty = MeshAsset::new(vec![], vec![]);
        assert_eq!(
            fracture_mesh(&empty, guid(20), FractureParams::default()),
            Err(FractureSkip::NoGeometry)
        );

        // A flat quad: real triangles, no volume.
        let v = |x: f32, z: f32| MeshVertex {
            position: [x, 0.0, z],
            ..Default::default()
        };
        let flat = MeshAsset::new(
            vec![SubMesh {
                name: "flat".into(),
                vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)],
                indices: vec![0, 1, 2, 0, 2, 3],
                material_slot: None,
                skin: Vec::new(),
            }],
            vec![],
        );
        assert_eq!(
            fracture_mesh(&flat, guid(21), FractureParams::default()),
            Err(FractureSkip::Degenerate)
        );

        // A 1 mm cube: a real solid, below the size worth breaking.
        let tiny = cube_mesh(0.0005);
        match fracture_mesh(&tiny, guid(22), FractureParams::default()) {
            Err(FractureSkip::TooSmall { volume_m3 }) => {
                assert!(volume_m3 < MIN_FRACTURE_VOLUME_M3, "{volume_m3}");
            }
            other => panic!("expected TooSmall, got {other:?}"),
        }
    }

    #[test]
    fn chunk_counts_clamp_and_the_derived_id_is_a_bijection() {
        assert_eq!(clamp_chunk_count(0), MIN_CHUNK_COUNT);
        assert_eq!(clamp_chunk_count(9), 9);
        assert_eq!(clamp_chunk_count(10_000), MAX_CHUNK_COUNT);

        for n in [0u128, 1, 0xDEAD_BEEF, u64::MAX as u128] {
            let m = guid(n);
            let d = derived_fracture_id(m);
            assert_ne!(d, m);
            assert_eq!(derived_fracture_id(d), m, "the salt must be involutive");
        }
        // Distinct meshes never collide.
        assert_ne!(derived_fracture_id(guid(1)), derived_fracture_id(guid(2)));
    }
}
