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
//!
//!   **A volume sum cannot see shape error, so here is the number.** Σ chunks
//!   runs up to about **1.23 ×** the true hull on a quantized sphere, but
//!   individual chunk *surfaces* run much further out than that ratio suggests:
//!   measured at **0.48 m outside a 1 m sphere**. Where this shows up is
//!   visuals — a fractured ball's pieces are visibly faceted against the source
//!   surface, because chunk geometry is `Voronoi cell ∩ simplified hull` and
//!   never the source triangles. Boxy props (the v1 target) are unaffected: a
//!   crate's hull IS the crate, to within its own `f32` quantization.
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
/// Same construction, and the same reasoning, as `inf_vgeom::VMESH_ID_SALT`'s
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
    // The vertex sets already claimed by a face, so a plane COINCIDENT with an
    // earlier one contributes no second face.
    //
    // **This is a correctness fix, not a tidy-up.** `fracture_mesh` always feeds
    // coincident planes: any axis-aligned mesh face touches the AABB, so its hull
    // plane is bit-identical to a box plane, and both would produce the same
    // polygon. `volume()` integrates over the face list, so the same polygon
    // twice DOUBLES the reported volume — a 1 m cube measured 2.0 m³, which in
    // turn skews `MIN_FRACTURE_VOLUME_M3` by 2× and prints doubled numbers in an
    // author-facing advisory. Keyed on the vertex SET rather than on the plane, so
    // it catches coincidence from any source (box↔hull, hull↔bisector, or two
    // planes that merely resolve to the same polygon numerically).
    //
    // The FIRST plane in the canonical `[box, hull, bisector]` order wins, which
    // is also the right answer for classification: a bisector lying exactly on the
    // hull surface is not an interior cut, and the cell on its far side is outside
    // the solid and has no chunk to be adjacent to.
    let mut claimed: std::collections::BTreeSet<Vec<usize>> = std::collections::BTreeSet::new();
    for (pi, pl) in planes.iter().enumerate() {
        let on: Vec<usize> = (0..verts.len())
            .filter(|&vi| pl.distance(verts[vi]).abs() <= merge)
            .collect();
        if on.len() < 3 {
            continue; // a redundant plane, or one touching at an edge/point
        }
        // `on` is built in ascending vertex order, so it is already the set key.
        if !claimed.insert(on.clone()) {
            continue; // coincident with a plane already faced
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

/// The spacing of `f32` at magnitude 1 — `2⁻²³ ≈ 1.19e-7`.
///
/// **The single most load-bearing number in this module**, because a mesh vertex
/// is `[f32; 3]` ([`MeshVertex::position`]) and every point handed to the hull is
/// therefore already *quantized*. A coordinate of magnitude `X` carries an
/// absolute representation error up to `1.19e-7 · X` before any arithmetic
/// happens, so a perfectly planar face exported by any DCC tool arrives here
/// **provably non-planar** by about that much.
pub const F32_QUANTUM: f64 = 1.192_092_9e-7;

/// How many `f32` quanta of the point cloud's own extent count as "the same
/// plane" in the hull builder.
///
/// The original `1e-9 · extent` was **60× below the f32 quantum** and therefore
/// measured noise rather than geometry: every tessellated flat face read as a
/// forest of microscopic non-coplanar facets, which is how a 200-point cloud
/// produced 948 faces against a topological maximum of 396, how 800 points cost
/// 3.94 s, and how a 20 m tessellated cube fractured into two chunks totalling
/// 0.000 m³ against a truth of 8000.
///
/// Eight quanta is a plane-fit budget, not a fudge: three quantized vertices fix
/// a plane whose offset error is a small multiple of the vertex error, and a
/// factor of eight covers the amplification of a moderately slivery triangle
/// without swallowing real geometry. The trade is stated: a solid genuinely
/// thinner than eight `f32` quanta of its own bounding extent is refused as
/// degenerate — which is honest, because at that thickness `f32` cannot
/// represent it as non-flat in the first place.
pub const HULL_EPS_QUANTA: f64 = 8.0;

/// The tolerance the hull builder treats as "on the plane", for a cloud of the
/// given extent. See [`HULL_EPS_QUANTA`].
#[inline]
pub fn hull_epsilon(extent: f64) -> f64 {
    HULL_EPS_QUANTA * F32_QUANTUM * extent
}

/// How much looser the **certification** tolerance is than the visibility one.
///
/// The two cannot be the same number, and the reason is structural rather than
/// numerical. A point within [`hull_epsilon`] of a face is deliberately *not*
/// made visible, so it never becomes a vertex; the faces around it are then
/// replaced by cone faces fitted through *other* points, and the plane drifts
/// slightly away from it. The drift accumulates with the number of points that
/// were legitimately skipped, so it grows with cloud density: measured on
/// quantized unit spheres, the worst legitimately-skipped point ends up
/// **1.4 ×** the visibility tolerance outside its final face at 3000 points and
/// **15.9 ×** at 6000. (It was 274 × before insertion order became
/// farthest-first — that change is worth more than any tolerance.)
///
/// Thirty-two is the 6000-point measurement with 2 × headroom. **It is a
/// gross-error gate, not a claim of exactness**: on a 1 m object it still
/// certifies containment to about 90 µm, while the defects it exists to catch —
/// a point five units outside its face, a surface with a hole — are three to
/// four orders of magnitude beyond it. A hull that fails this is not
/// approximately right; it is wrong.
pub const HULL_CERT_SLACK: f64 = 32.0;

/// How many times the visible region may be grown to relieve a pinched horizon
/// before the build gives up. Three is generous: the measured cases settle on the
/// first relief round, and the region grows monotonically, so this is a tripwire
/// against an unforeseen cycle rather than a tuning knob.
const PINCH_RELIEF_ROUNDS: usize = 4;

/// The convex hull of `points` as a triangle list with outward-facing planes, or
/// `None` for a point set the builder cannot certify.
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
///
/// # It has a POST-CONDITION, and that is the point
///
/// An incremental hull under floating point can degrade in ways that are not
/// visible from the inside: it can leave a hole where a horizon walk went wrong,
/// leave a point outside a face it should have destroyed, or — with too small a
/// coplanarity epsilon — grow indefinitely on quantization noise. `tris.len() >=
/// 4` catches none of those, and every one of them ships as *plausible geometry*.
///
/// So the result is **certified before it is returned**:
///
/// 1. **Containment** — every input point lies inside every face plane, within
///    [`hull_epsilon`] × [`HULL_CERT_SLACK`]. A hull that does not contain its
///    own cloud is not a hull.
/// 2. **Watertightness** — every directed edge appears exactly once and its twin
///    appears exactly once, i.e. the triangle set is a closed orientable surface.
///    A hole here becomes a chunk with no volume.
/// 3. **Growth cap** — a convex polyhedron on `V` vertices has at most `2V − 4`
///    faces (Euler), so exceeding `2 · points.len()` proves the builder is
///    generating facets from noise rather than geometry, and it also bounds the
///    super-cubic blow-up that made an 800-point cloud take seconds.
///
/// A failure of any of the three returns `None`, which reaches the caller as the
/// existing [`FractureSkip::Degenerate`] refusal and, in a cook, as a named
/// advisory. **A refusal is a value; a plausible-looking wrong hull is not.**
///
/// # Known limit, stated
///
/// Verified on `f32`-quantized clouds up to **6000 fully-extreme points** (a
/// dense ball, where every point is a hull vertex — the worst case an incremental
/// hull has). A denser fully-convex cloud can exceed the certification and be
/// refused. That is a *false* refusal rather than a wrong asset, and it is the
/// right side of the trade; real destructible meshes are nowhere near it,
/// because most of a mesh's vertices are interior to its hull.
///
/// Certification costs one `O(n · F)` sweep on top of the `O(n · F)` build, so it
/// is a constant factor, and `F` is capped at `2n` by rule (3) above.
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
    let eps = hull_epsilon(extent);
    // Euler's bound for a convex polyhedron, used as a "the builder is generating
    // noise" tripwire rather than as a memory cap.
    let face_cap = 2 * points.len();

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
    // A point strictly inside the seed tetrahedron. The hull only ever GROWS, so
    // this point stays interior for the whole build — which is what makes it a
    // valid outward reference for every cone face added below.
    let inner = (points[i0] + points[i1] + points[i2] + points[i3]) / 4.0;
    for drop in 0..4 {
        let t: Vec<usize> = (0..4).filter(|&i| i != drop).map(|i| seed[i]).collect();
        let (a, b, c) = (t[0], t[1], t[2]);
        tris.push(oriented_tri(points, a, b, c, inner)?);
    }

    // ── add the rest, FARTHEST FIRST ────────────────────────────────────────
    //
    // Not input order. Adding the most extreme points first means each later
    // point is either comfortably inside (skipped in one test) or comfortably
    // outside — the "barely outside, nearly coplanar with several faces at once"
    // case that pinches the horizon is what raw input order manufactures. The
    // order is a deterministic function of the cloud (distance from the seed
    // tetrahedron's centre, descending, ties broken by index), so the hull stays
    // a pure function of its input.
    let mut order: Vec<usize> = (0..points.len()).filter(|i| !seed.contains(i)).collect();
    order.sort_by(|&a, &b| {
        points[b]
            .distance_squared(inner)
            .partial_cmp(&points[a].distance_squared(inner))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    for pi in order {
        let p = &points[pi];
        // ── the visible region ──────────────────────────────────────────────
        //
        // **A connected region whose boundary is a cycle — not a predicate.**
        //
        // The naive `distance > eps` set is what a textbook writes, and it is
        // wrong the moment a point is *coplanar* with part of the hull, which is
        // the normal case for a tessellated flat face — i.e. for every crate,
        // wall and floor anyone will ever mark destructible. The point then sees
        // some faces of that flat region and not others, so the visible set is
        // disconnected, annular, or **pinched** (a figure-eight boundary touching
        // itself at a vertex — measured at 8 horizon edges over 7 distinct start
        // vertices on a 6000-point quantized ball). Its boundary is then not a
        // single cycle, and the cone stitched to it is non-manifold: a hull with
        // a hole, which clips every Voronoi cell to nothing and ships chunks of
        // zero volume.
        //
        // Two repairs, in order, both of which GROW the region — never drop the
        // point, because a dropped point stays outside the hull and the
        // containment post-condition would then (correctly) refuse an ordinary
        // dense mesh with a `Degenerate` advisory that is simply untrue:
        //
        // 1. **Coplanar absorption.** Flood-fill across shared edges, absorbing
        //    any adjacent face the point is not strictly *behind* (`> -eps`).
        //    This is what makes a tessellated flat face behave as one face.
        // 2. **Pinch relief.** If the boundary still touches itself at a vertex,
        //    absorb every face incident to that vertex and re-fill. The region
        //    grows monotonically, so this terminates; the bound below is a
        //    tripwire, not a policy.
        let mut vis: Vec<bool> = tris.iter().map(|t| t.plane.distance(*p) > eps).collect();
        if !vis.iter().any(|&v| v) {
            continue;
        }
        // Undirected edge → the faces on it, for the flood fill. `BTreeMap`,
        // never a hash map: the walk must be reproducible.
        let mut edge_faces: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (ti, t) in tris.iter().enumerate() {
            for e in [(t.v[0], t.v[1]), (t.v[1], t.v[2]), (t.v[2], t.v[0])] {
                edge_faces
                    .entry((e.0.min(e.1), e.0.max(e.1)))
                    .or_default()
                    .push(ti);
            }
        }
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut visible: Vec<usize> = Vec::new();
        let mut settled = false;
        for _ in 0..PINCH_RELIEF_ROUNDS {
            // (1) coplanar absorption
            let mut stack: Vec<usize> = (0..tris.len()).filter(|&i| vis[i]).collect();
            while let Some(ti) = stack.pop() {
                let t = tris[ti];
                for e in [(t.v[0], t.v[1]), (t.v[1], t.v[2]), (t.v[2], t.v[0])] {
                    let key = (e.0.min(e.1), e.0.max(e.1));
                    for &nb in edge_faces.get(&key).into_iter().flatten() {
                        if !vis[nb] && tris[nb].plane.distance(*p) > -eps {
                            vis[nb] = true;
                            stack.push(nb);
                        }
                    }
                }
            }
            visible = (0..tris.len()).filter(|&i| vis[i]).collect();
            if visible.len() == tris.len() {
                // The point would consume the entire hull, which cannot happen
                // for a point of a bounded cloud and means the arithmetic has
                // lost the shape.
                return None;
            }
            // Horizon: a directed edge of a visible face whose twin is NOT in the
            // visible set. `BTreeMap`, never a hash set — the walk must be
            // reproducible.
            let mut dir: BTreeMap<(usize, usize), ()> = BTreeMap::new();
            for &ti in &visible {
                let v = tris[ti].v;
                for e in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                    dir.insert(e, ());
                }
            }
            horizon = dir
                .keys()
                .copied()
                .filter(|&(a, b)| !dir.contains_key(&(b, a)))
                .collect();
            // A single closed cycle: every vertex starts exactly one horizon edge
            // and ends exactly one.
            let mut starts: BTreeMap<usize, u32> = BTreeMap::new();
            let mut ends: BTreeMap<usize, u32> = BTreeMap::new();
            for &(a, b) in &horizon {
                *starts.entry(a).or_default() += 1;
                *ends.entry(b).or_default() += 1;
            }
            if horizon.len() >= 3
                && starts.values().all(|&n| n == 1)
                && ends.values().all(|&n| n == 1)
                && starts.len() == horizon.len()
                && ends.len() == horizon.len()
            {
                settled = true;
                break;
            }
            // (2) pinch relief — absorb everything touching an offending vertex.
            let pinched: std::collections::BTreeSet<usize> = starts
                .iter()
                .chain(ends.iter())
                .filter(|(_, &n)| n != 1)
                .map(|(&v, _)| v)
                .collect();
            if pinched.is_empty() {
                break;
            }
            let mut grew = false;
            for ti in 0..tris.len() {
                if !vis[ti] && tris[ti].v.iter().any(|v| pinched.contains(v)) {
                    vis[ti] = true;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        if !settled {
            // The region could not be made into a disk. Refusing here is the
            // honest answer: the alternative is a non-manifold cone, and the
            // caller turns this into a named advisory rather than a silent
            // half-built hull.
            return None;
        }
        // Remove visible faces (descending, so indices stay valid).
        for &ti in visible.iter().rev() {
            tris.swap_remove(ti);
        }
        for (a, b) in horizon {
            // **Oriented against the seed interior point, not against the
            // horizon's winding.** The horizon edge's direction is only as
            // trustworthy as the faces it came from, and one inward-facing cone
            // face silently inverts a region of the hull — which then contains
            // nothing and clips every cell to nothing.
            if let Some(t) = oriented_tri(points, a, b, pi, inner) {
                tris.push(t);
            }
        }
        if tris.len() > face_cap {
            return None;
        }
        // `swap_remove` reorders the face list, which would make the result
        // depend on removal order. Restoring a canonical order after every point
        // is what keeps the hull a pure function of the input.
        tris.sort_by_key(|t| t.v);
    }
    if tris.len() < 4 {
        return None;
    }

    let faces: Vec<HullFaceOut> = tris
        .into_iter()
        .map(|t| HullFaceOut {
            v: t.v,
            plane: t.plane,
        })
        .collect();
    hull_is_certified(points, &faces, eps).then_some(faces)
}

/// The hull post-condition, as a function of the result — so it can be run
/// against a **mutated** face list and shown to reject it.
///
/// Extracted rather than inlined for exactly that reason: a post-condition that
/// only ever sees correct input is a claim, not a check
/// (`the_post_condition_rejects_mutated_hulls` mutates a clone — the P18
/// discipline). Both halves are pure and allocation-light.
///
/// 1. **Watertight** — every directed edge appears exactly once and its twin
///    appears exactly once. A hole, a fin or a doubled facet fails.
/// 2. **Containment** — no input point sits outside any face plane by more than
///    [`hull_epsilon`] × [`HULL_CERT_SLACK`].
pub fn hull_is_certified(points: &[DVec3], faces: &[HullFaceOut], eps: f64) -> bool {
    if faces.len() < 4 {
        return false;
    }
    let mut edges: BTreeMap<(usize, usize), u32> = BTreeMap::new();
    for t in faces {
        for e in [(t.v[0], t.v[1]), (t.v[1], t.v[2]), (t.v[2], t.v[0])] {
            *edges.entry(e).or_default() += 1;
        }
    }
    for (&(a, b), &n) in &edges {
        if n != 1 || edges.get(&(b, a)).copied().unwrap_or(0) != 1 {
            return false;
        }
    }
    let cert = eps * HULL_CERT_SLACK;
    for t in faces {
        for p in points {
            if t.plane.distance(*p) > cert {
                return false;
            }
        }
    }
    true
}

/// A hull triangle on `(a, b, c)` wound so its plane faces **away** from
/// `interior`. Returns `None` for a degenerate triangle.
fn oriented_tri(
    points: &[DVec3],
    a: usize,
    b: usize,
    c: usize,
    interior: DVec3,
) -> Option<HullTri> {
    let pl = Plane::through(
        points[a],
        (points[b] - points[a]).cross(points[c] - points[a]),
    )?;
    if pl.distance(interior) > 0.0 {
        Some(HullTri {
            v: [a, c, b],
            plane: Plane {
                normal: -pl.normal,
                d: -pl.d,
            },
        })
    } else {
        Some(HullTri {
            v: [a, b, c],
            plane: pl,
        })
    }
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
    // The SAME tolerance the hull builder treats as "on the plane". It has to be:
    // a tessellated flat face arrives as triangles whose fitted plane offsets
    // differ by up to `hull_epsilon(extent)`, so a smaller merge tolerance here
    // would leave them as dozens of distinct half-spaces, blow the plane budget
    // on what is geometrically one face, and truncate a box down to a wedge.
    let d_eps = hull_epsilon(extent).max(f64::MIN_POSITIVE);
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
    /// The geometry was fine but the chunking produced fewer than
    /// [`MIN_CHUNK_COUNT`] pieces — every site's cell missed the solid, or all
    /// but one did.
    ///
    /// A refusal rather than an asset: a `.inf_fracture` with no chunks encodes,
    /// packs and loads perfectly while making its mesh unbreakable, and nothing
    /// downstream can tell that from a mesh nobody marked destructible.
    TooFewChunks {
        /// How many chunks came out.
        produced: u32,
        /// How many were asked for (post-clamp).
        requested: u32,
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
    /// The authored seed offset this asset was built with — so a tool can tell a
    /// stale derived asset from a current one without re-running the fracture.
    pub seed: u32,
    /// How many chunks were **asked for** (post-clamp), against
    /// [`chunks`](Self::chunks)`.len()`, which is how many the geometry actually
    /// yielded.
    ///
    /// They differ whenever a site's Voronoi cell missed the solid — legal, and
    /// invisible without this field, which is why it exists: a shipped asset that
    /// says "12" and carries 5 is the only evidence that a level's break will be
    /// coarser than its author asked for. The cook turns a large shortfall into
    /// an advisory; this is what makes that checkable from the pack afterwards.
    pub requested_chunks: u32,
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
                let i = i as u32;
                if i == n {
                    continue; // a chunk is not its own neighbour
                }
                // **Canonicalised, not filtered.** The old form kept a pair
                // only when the LOWER index listed it, so a one-sided edge —
                // chunk 7 naming chunk 3 while chunk 3 does not name 7 —
                // vanished silently, and a consumer that prices bonds would
                // charge 0 J to break it.
                //
                // `prune_faceless_adjacency` now makes the cook's own output
                // symmetric, so this is belt to that braces: it costs one `min`
                // and one `max`, and it keeps the reader correct for an asset
                // written by an older cook or by a tool that is not this one.
                out.push((i.min(n), i.max(n)));
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
    //
    // The tolerance is [`hull_epsilon`], the same `f32`-quantization budget the
    // hull builder uses, and for the same reason: a source vertex "on" a hull
    // plane is only on it to within the quantization of its own coordinates, so a
    // tighter tolerance would find no vertices on a 20 m crate's faces and drop
    // every exterior face back to slot 0.
    let slot_tol = hull_epsilon(extent);
    let exterior_slot_of: Vec<u32> = box_planes
        .iter()
        .chain(hull_planes.iter())
        .map(|pl| {
            let tol = slot_tol;
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
    let remap = site_to_chunk(&cells);
    let mut chunks: Vec<FractureChunk> = Vec::with_capacity(cells.len());
    for cell in &cells {
        chunks.push(build_chunk(
            cell,
            interior_slot,
            &exterior_slot_of,
            box_planes.len(),
            hull_planes.len(),
            &remap,
        ));
    }
    // **Drop adjacency edges with no face behind them** (P22.3 re-audit). Must
    // run AFTER every chunk exists, because the measurement needs both hulls. See
    // `prune_faceless_adjacency` for why a faceless edge is a phantom load path
    // rather than merely a mispriced bond.
    {
        let (lo, hi) = (mesh.bounds.min, mesh.bounds.max);
        let extent_m = (((hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2])) as f64).max(1.0e-3);
        prune_faceless_adjacency(&mut chunks, extent_m);
    }

    // **A fracture with fewer than two pieces is not a fracture.** The chunk
    // count is enforced on the OUTPUT, not just clamped on the input: before
    // this, a hull the cells all missed produced `Ok(FractureAsset { chunks: []
    // })`, which encodes to 68 perfectly valid bytes and packs silently — a mesh
    // that ships fracture data and cannot break, which is worse than one that
    // ships none, because the advisory that would have said so never fires.
    let requested = clamp_chunk_count(params.chunk_count);
    if chunks.len() < MIN_CHUNK_COUNT as usize {
        return Err(FractureSkip::TooFewChunks {
            produced: chunks.len() as u32,
            requested,
        });
    }

    Ok(FractureAsset {
        schema_version: FractureAsset::CURRENT_VERSION,
        source_mesh: *guid.uuid().as_bytes(),
        bounds: mesh.bounds,
        seed: params.seed,
        requested_chunks: requested,
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

    // Adjacency is NOT derived here: a cell knows the SITE on the far side of
    // each of its bisector faces, and a chunk index is a position in this vector
    // — two numbering systems that coincide only when no cell was dropped, which
    // is exactly the case a cube-shaped fixture tests and nothing else does. The
    // site→chunk remap lives in [`site_to_chunk`] and is applied in `build_chunk`.
    cells
}

/// Map each cell's **site** index to its **chunk** index — the position it will
/// occupy in the shipped `chunks` vector.
///
/// The two numbering systems diverge whenever a site produced no cell: a site
/// outside the mesh's hull, or one whose Voronoi cell misses the solid entirely,
/// is dropped by [`fracture_from_sites`] and every later chunk shifts down.
/// Publishing site indices as `neighbors` therefore ships a structural graph with
/// out-of-range entries, self-loops and asymmetry — measured on a diagonal plank
/// as 1 out-of-range, 3 self-referential and 4 asymmetric, and on a flat pane
/// as a single chunk naming neighbour 7 in a one-element array. P22.3's support
/// solve indexes straight into that array.
///
/// A dropped site simply has no entry, so a face against it contributes no edge —
/// which is correct: there is no chunk on the other side.
fn site_to_chunk(cells: &[Cell]) -> BTreeMap<usize, u32> {
    cells
        .iter()
        .enumerate()
        .map(|(ci, c)| (c.site, ci as u32))
        .collect()
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
    site_to_chunk: &BTreeMap<usize, u32>,
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
            // The far side is a SITE index; `neighbors` is a list of CHUNK
            // indices. A site whose cell was dropped has no chunk and so
            // contributes no edge — see `site_to_chunk`.
            if let Some(&chunk) = cell
                .bisector_sites
                .get(b)
                .and_then(|site| site_to_chunk.get(site))
            {
                neighbors.push(chunk);
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

/// Fraction of the source mesh's extent within which two chunks' hull corners
/// count as **the same corner**.
///
/// A shared Voronoi face's corners are the intersection of the *same* three
/// half-space planes in both cells, so the two hulls carry them at coordinates
/// that agree to within the arithmetic that produced them. `1e-4 x 10 m = 1 mm`
/// clears that comfortably while staying far below any real geometric feature —
/// two genuinely distinct corners of a metre-scale chunk are never a millimetre
/// apart.
pub const FACE_PLANE_EPS_FRACTION: f64 = 1.0e-4;

impl FractureAsset {
    /// The source mesh's largest extent, metres — the scale every geometric
    /// tolerance in this module is a fraction of.
    pub fn extent_m(&self) -> f64 {
        let (lo, hi) = (self.bounds.min, self.bounds.max);
        let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        (d[0].max(d[1]).max(d[2]) as f64).max(1.0e-3)
    }

    /// The area, m², of the face chunks `a` and `b` share — `0.0` when they share
    /// no face at all.
    ///
    /// # Why it lives here rather than in the consumer
    ///
    /// Two callers need the same answer for different reasons, and they are in
    /// different crates. The **cook** ([`fracture_mesh`]) uses it to decide
    /// whether an adjacency edge is real: an edge with no face behind it is a
    /// phantom load path, because the same `neighbors` graph the pricing reads is
    /// the graph the structural solve propagates support along — a chunk held up
    /// by a neighbour it never touches. The **runtime**
    /// (`inf_physics::d3::fracture`) uses it to price the bond. One
    /// implementation, in the crate that owns the geometry, called by both.
    ///
    /// # How, and why not the obvious way
    ///
    /// The obvious measurement is "the chunks are Voronoi cells, so their shared
    /// face lies on the perpendicular bisector of their two centres — sum `a`'s
    /// triangles on that plane". **It is wrong, and it fails silently.** A Voronoi
    /// face bisects the two *sites*, and a cell's `center_of_mass` is its volume
    /// centroid, which is not its site: clip a cell against the source mesh's hull
    /// and the centroid slides away from the seed.
    ///
    /// So the face is found from the **geometry the two chunks agree on**: their
    /// `hull_points` are `f64`, and a shared face's corners are the intersection
    /// of the same three planes in both cells. The common points are the face's
    /// polygon; fit a plane through them, order them around their centroid, and
    /// sum the fan. Working in `f64` hull points rather than `f32` render vertices
    /// is the other half of that — the vertices a chunk draws with are quantised,
    /// and quantisation noise is the size of the tolerance this needs.
    pub fn shared_face_area_m2(&self, a: u32, b: u32) -> f64 {
        let (Some(ca), Some(cb)) = (self.chunks.get(a as usize), self.chunks.get(b as usize))
        else {
            return 0.0;
        };
        shared_face_area_between(ca, cb, self.extent_m())
    }
}

/// [`FractureAsset::shared_face_area_m2`] over two chunks directly — the form the
/// cook needs, before there is an asset to ask.
pub fn shared_face_area_between(a: &FractureChunk, b: &FractureChunk, extent_m: f64) -> f64 {
    let eps = extent_m * FACE_PLANE_EPS_FRACTION;
    let eps2 = eps * eps;
    let common: Vec<DVec3> = a
        .hull_points
        .iter()
        .map(|p| DVec3::from_array(*p))
        .filter(|p| {
            b.hull_points
                .iter()
                .any(|q| (DVec3::from_array(*q) - *p).length_squared() <= eps2)
        })
        .collect();
    polygon_area_m2(&common)
}

/// The area of the convex polygon through `points`, which are assumed coplanar
/// and in convex position but in **no particular order** — a shared face's corner
/// set as the two hulls happen to list it.
///
/// Fewer than three points bound no area. The plane's normal is taken from the
/// widest available triangle rather than from the first one, because a face's
/// corner list can start with three nearly-collinear points and a normal fitted
/// to those is noise.
fn polygon_area_m2(points: &[DVec3]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let centre = points.iter().copied().sum::<DVec3>() / points.len() as f64;
    let u = {
        let mut best = (0.0_f64, DVec3::X);
        for p in points {
            let d = *p - centre;
            let l = d.length();
            if l > best.0 {
                best = (l, d);
            }
        }
        if best.0 <= 0.0 {
            return 0.0;
        }
        best.1 / best.0
    };
    let normal = {
        let mut best = (0.0_f64, DVec3::Y);
        for p in points {
            let c = u.cross(*p - centre);
            let l = c.length();
            if l > best.0 {
                best = (l, c);
            }
        }
        if best.0 <= 0.0 {
            return 0.0; // collinear: no area
        }
        best.1 / best.0
    };
    let v = normal.cross(u);
    // Angle-order around the centre, then sum the shoelace. Ties break by the
    // input index, so the order is a function of the geometry alone.
    let mut planar: Vec<(u64, usize, f64, f64)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let d = *p - centre;
            let (x, y) = (d.dot(u), d.dot(v));
            (total_order_key(y.atan2(x)), i, x, y)
        })
        .collect();
    planar.sort_by_key(|(k, i, _, _)| (*k, *i));
    let mut twice = 0.0;
    for i in 0..planar.len() {
        let (_, _, x0, y0) = planar[i];
        let (_, _, x1, y1) = planar[(i + 1) % planar.len()];
        twice += x0 * y1 - x1 * y0;
    }
    (twice * 0.5).abs()
}

/// Map an `f64` onto a `u64` whose unsigned ordering matches the float's total
/// ordering — so a sort key is exact and needs no float comparator.
fn total_order_key(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits & (1 << 63) != 0 {
        !bits
    } else {
        bits ^ (1 << 63)
    }
}

/// **Drop every adjacency edge with no real face behind it, and make what
/// survives symmetric** (P22.3 re-audit).
///
/// # Why an edge can be faceless at all
///
/// `build_chunk` lists a neighbour for each bisector face the *clipped* polytope
/// still has, which already removes the common case — a face clipped away
/// entirely by the source hull leaves no face and so no edge. What it cannot
/// remove is a **sliver**: a bisector face reduced by clipping to a degenerate
/// wisp with corners but no area. That is still a face to the loop above, and it
/// becomes an adjacency edge.
///
/// # Why a faceless edge is worse than a mispriced bond
///
/// The `neighbors` graph is read twice downstream, and only one of the readers is
/// about strength. `inf_physics::d3::fracture` prices bonds from it — a bond with
/// no measurable face falls back to an estimate, and the estimate
/// (`min(volume)^(2/3)`) is close to the size of the chunk's *largest* real faces,
/// so a phantom bond is priced as one of the strongest in the structure. The
/// other reader is the **support solve**, which propagates "this chunk is held up"
/// along exactly these edges — so a phantom edge is a phantom load path, and a
/// chunk hangs in the air held by a neighbour it never touches.
///
/// Neither reader can detect it: an area of zero and an area the measurement
/// could not find are the same observation. So it is fixed here, in the producer,
/// where the geometry is still around to be asked.
///
/// Symmetry falls out for free, and is worth having: the two cells' clips can
/// disagree about a sliver, so an edge listed by one side and not the other is
/// exactly the asymmetric case the runtime used to canonicalise around.
fn prune_faceless_adjacency(chunks: &mut [FractureChunk], extent_m: f64) {
    // The area below which a "face" is arithmetic noise rather than geometry: the
    // square of the corner-matching tolerance. At a 2 m extent that is 4e-8 m² —
    // a fortieth of a square millimetre.
    let eps = extent_m * FACE_PLANE_EPS_FRACTION;
    let min_area = eps * eps;
    let n = chunks.len();
    // Every unordered pair either side lists, measured once.
    let mut keep: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
    let mut seen: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
    for i in 0..n {
        for k in 0..chunks[i].neighbors.len() {
            let j = chunks[i].neighbors[k];
            if j as usize >= n || j == i as u32 {
                continue;
            }
            let pair = ((i as u32).min(j), (i as u32).max(j));
            if !seen.insert(pair) {
                continue;
            }
            if shared_face_area_between(
                &chunks[pair.0 as usize],
                &chunks[pair.1 as usize],
                extent_m,
            ) > min_area
            {
                keep.insert(pair);
            }
        }
    }
    for (i, c) in chunks.iter_mut().enumerate() {
        let i = i as u32;
        c.neighbors = keep
            .iter()
            .filter_map(|&(a, b)| {
                if a == i {
                    Some(b)
                } else if b == i {
                    Some(a)
                } else {
                    None
                }
            })
            .collect();
        c.neighbors.sort_unstable();
        c.neighbors.dedup();
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

    /// **The ordinary destructible asset.** A cube of side `side`, every face
    /// tessellated into a `grid × grid` quad patch, rotated off every axis, and
    /// stored as `[f32; 3]` positions — which is to say, put through exactly what
    /// a DCC export does to a crate.
    ///
    /// Each of those four steps is one the naive builder could not survive:
    /// tessellation multiplies the coplanar points, rotation stops the faces
    /// aligning with the AABB (so the hull planes are no longer bit-identical to
    /// the box planes), and the `f32` round trip makes every "flat" face provably
    /// non-planar by ~1.19e-7 of its magnitude.
    fn tessellated_crate(side: f64, grid: usize) -> MeshAsset {
        // A fixed off-axis rotation with rational components: `sqrt` is
        // IEEE-correctly-rounded and therefore bit-portable, unlike `sin`/`cos`.
        let q = glam::DQuat::from_xyzw(1.0, 2.0, 3.0, 4.0).normalize();
        let h = side * 0.5;
        let mut vertices = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        // Six faces: axis, sign.
        for axis in 0..3usize {
            for sign in [-1.0, 1.0f64] {
                let base = vertices.len() as u32;
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                for i in 0..=grid {
                    for j in 0..=grid {
                        let mut p = [0.0f64; 3];
                        p[axis] = sign * h;
                        p[u] = -h + 2.0 * h * (i as f64 / grid as f64);
                        p[v] = -h + 2.0 * h * (j as f64 / grid as f64);
                        let r = q * DVec3::new(p[0], p[1], p[2]);
                        vertices.push(MeshVertex {
                            // The f32 round trip is the whole point of the fixture.
                            position: [r.x as f32, r.y as f32, r.z as f32],
                            ..Default::default()
                        });
                    }
                }
                let n = (grid + 1) as u32;
                for i in 0..grid as u32 {
                    for j in 0..grid as u32 {
                        let a = base + i * n + j;
                        indices.extend_from_slice(&[a, a + 1, a + n + 1, a, a + n + 1, a + n]);
                    }
                }
            }
        }
        MeshAsset::new(
            vec![SubMesh {
                name: "crate".into(),
                vertices,
                indices,
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["Wood".into()],
        )
    }

    /// A box of the given half-extents, rotated by `q`, as 8 corners — the
    /// "hull fills a small fraction of its AABB" shape when it is long, thin and
    /// diagonal.
    fn rotated_box(half: DVec3, q: glam::DQuat) -> MeshAsset {
        let mut vertices = Vec::new();
        for sx in [-1.0, 1.0f64] {
            for sy in [-1.0, 1.0f64] {
                for sz in [-1.0, 1.0f64] {
                    let r = q * DVec3::new(sx * half.x, sy * half.y, sz * half.z);
                    vertices.push(MeshVertex {
                        position: [r.x as f32, r.y as f32, r.z as f32],
                        ..Default::default()
                    });
                }
            }
        }
        let n = vertices.len() as u32;
        MeshAsset::new(
            vec![SubMesh {
                name: "plank".into(),
                vertices,
                indices: (0..n).collect(),
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["Pine".into()],
        )
    }

    /// Assert the structural invariants every shipped chunk set must satisfy.
    /// Returned so callers can also assert content-specific numbers.
    fn assert_chunk_set_is_sane(f: &FractureAsset) {
        let n = f.chunks.len();
        assert!(
            n >= MIN_CHUNK_COUNT as usize,
            "{n} chunks is not a fracture"
        );
        for (i, c) in f.chunks.iter().enumerate() {
            assert!(c.volume_m3 > 0.0, "chunk {i} has no volume");
            assert!(c.volume_m3.is_finite(), "chunk {i} volume is not finite");
            assert!(c.hull_points.len() >= 4, "chunk {i} has no collider hull");
            assert!(
                !c.indices.is_empty() && c.indices.len() % 3 == 0,
                "chunk {i} geometry"
            );
            assert!(
                c.center_of_mass.iter().all(|v| v.is_finite()),
                "chunk {i} centre of mass"
            );
            let covered: u32 = c.sections.iter().map(|s| s.index_count).sum();
            assert_eq!(covered as usize, c.indices.len(), "chunk {i} sections");
            // Adjacency is a CHUNK-index graph: in range, never self, sorted,
            // deduped, and symmetric.
            for &j in &c.neighbors {
                assert!(
                    (j as usize) < n,
                    "chunk {i} names neighbour {j} in a {n}-chunk asset"
                );
                assert_ne!(j as usize, i, "chunk {i} is its own neighbour");
                assert!(
                    f.chunks[j as usize].neighbors.contains(&(i as u32)),
                    "chunk {i} lists {j} but not the other way round"
                );
            }
            assert!(
                c.neighbors.windows(2).all(|w| w[0] < w[1]),
                "chunk {i} neighbours are not sorted+deduped"
            );
        }
        assert!(!f.adjacency_pairs().is_empty(), "no structural graph");
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

    // ── the f32 reality regression suite (P22.2 audit round) ────────────────

    /// **THE headline regression.** A rotated, `f32`-quantized, tessellated crate
    /// — the most ordinary destructible asset imaginable — must fracture into
    /// sane chunks with sane volumes.
    ///
    /// Every probe the audit measured is here as a `(side, grid)` row. Before the
    /// epsilon fix, `side = 20, grid = 5` shipped **two chunks totalling
    /// 0.000 m³** against a truth of 8000, and `grid = 9` cost seconds in the
    /// builder; the naive `1e-9 · extent` tolerance was 60× below the `f32`
    /// quantum, so every tessellated flat face read as non-coplanar noise.
    #[test]
    fn a_rotated_quantized_tessellated_crate_fractures_sanely() {
        for &(side, grid) in &[(1.0f64, 5usize), (20.0, 5), (20.0, 9), (0.5, 3)] {
            let mesh = tessellated_crate(side, grid);
            let f = fracture_mesh(&mesh, guid(0xC7A7), FractureParams::default())
                .unwrap_or_else(|e| panic!("side={side} grid={grid} refused: {e:?}"));
            assert_chunk_set_is_sane(&f);

            let truth = side * side * side;
            let sum = f.total_volume_m3();
            // The chunks tile the mesh's convex hull, which for a crate IS the
            // crate — to within the `f32` quantization of its own corners.
            assert!(
                (sum - truth).abs() <= truth * 1e-4,
                "side={side} grid={grid}: chunks summed to {sum} m3, truth {truth}"
            );
            // …and it really did break into pieces, not survive as one lump.
            assert!(
                f.chunks.len() >= DEFAULT_CHUNK_COUNT as usize / 2,
                "side={side} grid={grid}: only {} chunks",
                f.chunks.len()
            );
            assert_eq!(f.requested_chunks, DEFAULT_CHUNK_COUNT);
        }
    }

    /// **The plank**: a long thin box rotated onto the diagonal, so its hull
    /// fills a small fraction of its AABB and most sites land outside the solid.
    /// That is the shape that makes site indices and chunk indices diverge —
    /// which is what shipped a structural graph with out-of-range entries,
    /// self-loops and asymmetry until the remap landed.
    #[test]
    fn a_diagonal_plank_drops_sites_and_still_ships_a_clean_graph() {
        let q = glam::DQuat::from_xyzw(1.0, 2.0, 3.0, 4.0).normalize();
        let (hx, hy) = (2.0, 0.02);
        let mesh = rotated_box(DVec3::new(hx, hy, hy), q);
        let f = fracture_mesh(
            &mesh,
            guid(0xB1A4),
            FractureParams {
                seed: 0,
                chunk_count: 32,
            },
        )
        .unwrap();

        // The fixture has to actually exercise the divergence, or the test is
        // measuring the cube case again. Measured: the hull fills 0.209% of the
        // AABB and 21 of the 32 sites are dropped.
        let hull_vol = 2.0 * hx * 2.0 * hy * 2.0 * hy;
        let b = f.bounds;
        let aabb_vol = ((b.max[0] - b.min[0]) as f64)
            * ((b.max[1] - b.min[1]) as f64)
            * ((b.max[2] - b.min[2]) as f64);
        assert!(
            hull_vol < aabb_vol * 0.01,
            "the plank's hull fills {:.3}% of its AABB — not a divergent fixture",
            100.0 * hull_vol / aabb_vol
        );
        assert!(
            f.chunks.len() + 8 <= f.requested_chunks as usize,
            "too few sites dropped to exercise the remap: {} of {}",
            f.chunks.len(),
            f.requested_chunks
        );

        // …and every neighbour index is still a CHUNK index: in range, never
        // self, symmetric. Before the remap this fixture shipped an out-of-range
        // entry, three self-loops and four asymmetric edges.
        assert_chunk_set_is_sane(&f);
        let sum = f.total_volume_m3();
        assert!(
            (sum - hull_vol).abs() <= hull_vol * 1e-6,
            "chunks summed to {sum} m3, plank is {hull_vol}"
        );
    }

    /// **The pane**: a zero-thickness sheet is refused, and a thin-but-real one
    /// either fractures cleanly or refuses — never ships a one-chunk asset whose
    /// single chunk names a neighbour that does not exist.
    #[test]
    fn panes_refuse_or_ship_a_real_graph_never_a_dangling_one() {
        let flat = rotated_box(
            DVec3::new(1.0, 0.0, 1.0),
            glam::DQuat::from_xyzw(1.0, 2.0, 3.0, 4.0).normalize(),
        );
        assert_eq!(
            fracture_mesh(&flat, guid(0xFA7E), FractureParams::default()),
            Err(FractureSkip::Degenerate),
            "a zero-thickness pane bounds no volume"
        );

        // 2 mm thick: real, but thin enough that most cells miss it.
        let thin = rotated_box(
            DVec3::new(1.0, 0.001, 1.0),
            glam::DQuat::from_xyzw(1.0, 2.0, 3.0, 4.0).normalize(),
        );
        match fracture_mesh(&thin, guid(0xFA7F), FractureParams::default()) {
            Ok(f) => assert_chunk_set_is_sane(&f),
            Err(FractureSkip::TooFewChunks {
                produced,
                requested,
            }) => {
                assert!(produced < MIN_CHUNK_COUNT, "{produced} of {requested}");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// The hull's post-condition is what makes the refusal trustworthy, so it is
    /// tested by **mutation**: hulls that violate containment, watertightness or
    /// the growth cap must be rejected, and the honest one must not be.
    #[test]
    fn the_hull_post_condition_rejects_what_the_old_check_accepted() {
        let cloud: Vec<DVec3> = {
            let mut v = Vec::new();
            for sx in [-1.0, 1.0f64] {
                for sy in [-1.0, 1.0f64] {
                    for sz in [-1.0, 1.0f64] {
                        v.push(DVec3::new(sx, sy, sz));
                    }
                }
            }
            v
        };
        let good = convex_hull_faces(&cloud).expect("a cube cloud has a hull");
        assert_eq!(good.len(), 12);

        // (1) CONTAINMENT — a point outside a face plane. The old check
        //     (`tris.len() >= 4`) accepted this shape every time.
        let mut escaped = cloud.clone();
        escaped.push(DVec3::new(0.0, 0.0, 5.0));
        let hull = convex_hull_faces(&escaped).expect("still a hull");
        for f in &hull {
            for p in &escaped {
                assert!(
                    f.plane.distance(*p) <= hull_epsilon(10.0),
                    "the certified hull does not contain its own cloud"
                );
            }
        }

        // (2) WATERTIGHT — every directed edge exactly once, twin present.
        let mut edges: BTreeMap<(usize, usize), u32> = BTreeMap::new();
        for f in &good {
            for e in [(f.v[0], f.v[1]), (f.v[1], f.v[2]), (f.v[2], f.v[0])] {
                *edges.entry(e).or_default() += 1;
            }
        }
        assert!(edges.values().all(|&n| n == 1), "a doubled directed edge");
        assert!(
            edges.keys().all(|&(a, b)| edges.contains_key(&(b, a))),
            "an unpaired edge — the surface has a hole"
        );

        // (3) OUTWARD — every face plane has the cloud's centroid inside it. A
        //     single inverted cone face makes a region of the hull contain
        //     nothing, which clips every cell to nothing.
        let c = cloud.iter().copied().sum::<DVec3>() / cloud.len() as f64;
        for f in &good {
            assert!(f.plane.distance(c) < 0.0, "an inward-facing hull face");
        }

        // (4) The certification really is a gate: it accepts the honest hull.
        assert!(hull_is_certified(&cloud, &good, hull_epsilon(3.5)));

        // (5) The growth cap holds on a cloud that used to explode: 200 points on
        //     a sphere produced 948 faces against a topological maximum of 396.
        let mut sphere: Vec<DVec3> = Vec::new();
        let mut i = 0u64;
        while sphere.len() < 200 {
            let h = Hash64::new(0xB0A7).mix_u64(i);
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
            // Quantize like a real mesh, which is what made this explode.
            sphere.push(DVec3::new(
                n.x as f32 as f64,
                n.y as f32 as f64,
                n.z as f32 as f64,
            ));
        }
        let hull = convex_hull_faces(&sphere).expect("a sphere cloud has a hull");
        assert!(
            hull.len() <= 2 * sphere.len() - 4,
            "{} faces exceeds Euler's 2V-4 = {}",
            hull.len(),
            2 * sphere.len() - 4
        );
    }

    /// **Mutation evidence for the post-condition.** Each of the three defects
    /// the old `tris.len() >= 4` check accepted is manufactured by mutating a
    /// *clone* of a correct hull, and the certification must reject every one.
    ///
    /// Without this the post-condition would only ever be shown agreeing with
    /// correct input, which is a claim rather than a check — the P18 "mutate a
    /// clone" discipline, applied to geometry instead of a meshlet DAG.
    #[test]
    fn the_post_condition_rejects_mutated_hulls() {
        let cloud: Vec<DVec3> = {
            let mut v = Vec::new();
            for sx in [-1.0, 1.0f64] {
                for sy in [-1.0, 1.0f64] {
                    for sz in [-1.0, 1.0f64] {
                        v.push(DVec3::new(sx, sy, sz));
                    }
                }
            }
            v
        };
        let extent = (DVec3::splat(2.0)).length();
        let eps = hull_epsilon(extent);
        let good = convex_hull_faces(&cloud).expect("a cube cloud has a hull");
        assert!(
            hull_is_certified(&cloud, &good, eps),
            "the honest hull must pass, or the gate is just a rejector"
        );

        // (0) A DEGENERATE FACE SET. Both clauses below are universally
        //     quantified over faces and edges, so an EMPTY list satisfies them
        //     vacuously — without the arity guard, "no faces at all" certifies as
        //     a hull. Found by deleting each clause in turn and checking the suite
        //     noticed: this was the one deletion nothing caught.
        assert!(
            !hull_is_certified(&cloud, &[], eps),
            "an empty face set must not certify — every other clause is vacuous on it"
        );
        assert!(
            !hull_is_certified(&cloud, &good[..3], eps),
            "three faces cannot bound a solid"
        );

        // (1) A HOLE: drop one triangle. The surface is no longer closed, and a
        //     chunk clipped against it has no volume.
        let mut holed = good.clone();
        holed.remove(3);
        assert!(
            !hull_is_certified(&cloud, &holed, eps),
            "a hull with a hole must be refused"
        );

        // (2) A FIN / inverted facet: reverse one triangle's winding. Its edges
        //     now duplicate its neighbours' instead of pairing with them.
        let mut flipped = good.clone();
        flipped[5].v.swap(1, 2);
        assert!(
            !hull_is_certified(&cloud, &flipped, eps),
            "an inverted facet must be refused"
        );

        // (3) NON-CONTAINMENT: leave the hull alone and add a point outside it.
        //     This is the shape that shipped a "hull" the mesh stuck out of.
        let mut escaped = cloud.clone();
        escaped.push(DVec3::new(0.0, 0.0, 5.0));
        assert!(
            !hull_is_certified(&escaped, &good, eps),
            "a hull that does not contain its cloud must be refused"
        );

        // …and the tolerance is a tolerance, not a hole in the gate: a point
        // outside by less than the certification slack is still accepted.
        let mut nudged = cloud.clone();
        nudged.push(DVec3::new(0.0, 0.0, 1.0 + eps * (HULL_CERT_SLACK * 0.5)));
        assert!(hull_is_certified(&nudged, &good, eps));
    }

    /// Coincident planes must contribute ONE face, not two.
    ///
    /// `fracture_mesh` always feeds some: an axis-aligned mesh face's hull plane
    /// is bit-identical to a box plane. Before the dedup, `volume()` integrated
    /// the same polygon twice and a 1 m cube measured **2.0 m³** — which then
    /// skewed the sub-threshold refusal by 2× and printed a doubled number in the
    /// author-facing advisory.
    #[test]
    fn coincident_planes_are_not_integrated_twice() {
        let (lo, hi) = (DVec3::splat(-0.5), DVec3::splat(0.5));
        let mut doubled = aabb_planes(lo, hi).to_vec();
        doubled.extend_from_slice(&aabb_planes(lo, hi)); // the exact hazard
        let p = polytope_from_halfspaces(&doubled, 2.0).unwrap();
        assert_eq!(p.faces.len(), 6, "twelve planes, six faces");
        assert!((p.volume() - 1.0).abs() < 1e-12, "{}", p.volume());
        assert!((p.area() - 6.0).abs() < 1e-12, "{}", p.area());
        assert!(p.centroid().length() < 1e-12);

        // …and the whole-mesh path agrees: an axis-aligned 1 m cube's hull
        // measures 1 m³, which is what `MIN_FRACTURE_VOLUME_M3` is compared to.
        let mesh = cube_mesh(0.5);
        let f = fracture_mesh(&mesh, guid(0xC0DE), FractureParams::default()).unwrap();
        assert!((f.total_volume_m3() - 1.0).abs() < 1e-6);
    }

    /// A `f32`-quantized sphere cloud of `n` points — the curved case, where the
    /// hull plane budget bites.
    fn sphere_cloud(n: usize, seed: u64) -> Vec<DVec3> {
        let mut v = Vec::new();
        let mut i = 0u64;
        while v.len() < n {
            let h = Hash64::new(seed).mix_u64(i);
            i += 1;
            let p = DVec3::new(
                h.mix_u64(0).unit() * 2.0 - 1.0,
                h.mix_u64(1).unit() * 2.0 - 1.0,
                h.mix_u64(2).unit() * 2.0 - 1.0,
            );
            if p.length() < 0.05 {
                continue;
            }
            let u = p.normalize();
            v.push(DVec3::new(
                u.x as f32 as f64,
                u.y as f32 as f64,
                u.z as f32 as f64,
            ));
        }
        v
    }

    fn mesh_of(cloud: &[DVec3], slot: &str) -> MeshAsset {
        let vertices: Vec<MeshVertex> = cloud
            .iter()
            .map(|p| MeshVertex {
                position: [p.x as f32, p.y as f32, p.z as f32],
                ..Default::default()
            })
            .collect();
        let n = vertices.len() as u32;
        MeshAsset::new(
            vec![SubMesh {
                name: "ball".into(),
                vertices,
                indices: (0..n).collect(),
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec![slot.to_string()],
        )
    }

    /// The most the budgeted hull may exceed the true one, measured.
    ///
    /// Ratios across quantized unit spheres: **1.0712** at 300 points, **1.2295**
    /// at 1000, **1.1940** at 3000, **1.2089** at 6000. The pin is the worst of
    /// those with headroom. It is not a target — it is the stated price of
    /// [`HULL_PLANE_BUDGET`], and lowering the budget raises it.
    const BUDGETED_HULL_MAX_RATIO: f64 = 1.30;

    /// **The budget's cost, stated where it can fail.**
    ///
    /// `Σ chunks >= true hull` is a **direction witness, not a gate**: keeping the
    /// largest-area planes yields a polytope that *contains* the true hull, so the
    /// inequality is true by construction and could never fail. It is asserted to
    /// document the direction, and the arm that can actually fail is the **upper**
    /// bound — pinned above the worst measured tessellation, not the friendliest
    /// one, and cross-checked against a lower bound so a regression that made the
    /// budget keep every plane (or none) would be caught in one direction or the
    /// other.
    ///
    /// **A volume sum cannot see shape error, and this is where that shows up.**
    /// Σ chunks can sit 20% above the hull while individual chunk *surfaces* run
    /// far further out: measured at **0.48 m outside a 1 m sphere**. That is the
    /// v1 hull scope made numeric — a fractured sphere's pieces are visibly
    /// faceted against the source surface, because chunk geometry is
    /// `Voronoi cell ∩ simplified hull` and never the source triangles. It is
    /// invisible to any volume assertion, so it is measured here explicitly.
    #[test]
    fn the_hull_budget_costs_what_it_says_it_costs() {
        let mut worst_ratio = 0.0f64;
        let mut worst_shape = 0.0f64;
        for &n in &[300usize, 1000, 3000] {
            let cloud = sphere_cloud(n, 0x5E_1A11);
            let mesh = mesh_of(&cloud, "Rock");
            let faces = convex_hull_faces(&cloud).expect("a sphere cloud has a hull");
            assert!(
                faces.len() > HULL_PLANE_BUDGET * 4,
                "n={n} is not curved enough to bind the budget: {} faces",
                faces.len()
            );
            assert_eq!(
                hull_halfspaces(&cloud, &faces, 2.0).len(),
                HULL_PLANE_BUDGET,
                "n={n}: the budget must be what limits the plane count"
            );
            // The true hull volume, by direct integration over its triangles.
            let exact: f64 = faces
                .iter()
                .map(|t| cloud[t.v[0]].dot(cloud[t.v[1]].cross(cloud[t.v[2]])))
                .sum::<f64>()
                / 6.0;

            let f = fracture_mesh(&mesh, guid(31), FractureParams::default()).unwrap();
            assert_chunk_set_is_sane(&f);
            let sum = f.total_volume_m3();

            // The direction witness (true by construction — see the docs).
            assert!(sum >= exact, "n={n}: chunks {sum} lost material vs {exact}");
            // The arm that can fail.
            let ratio = sum / exact;
            assert!(
                ratio <= BUDGETED_HULL_MAX_RATIO,
                "n={n}: the {HULL_PLANE_BUDGET}-plane hull over-states the sphere by \
                 {ratio:.4}×, past the stated {BUDGETED_HULL_MAX_RATIO}"
            );
            worst_ratio = worst_ratio.max(ratio);

            // Shape error: the farthest a shipped chunk vertex sits outside the
            // TRUE hull. A volume sum cannot see this.
            for c in &f.chunks {
                for hp in &c.hull_points {
                    let p = DVec3::from_array(*hp);
                    let d = faces
                        .iter()
                        .map(|t| t.plane.distance(p))
                        .fold(f64::NEG_INFINITY, f64::max);
                    worst_shape = worst_shape.max(d);
                }
            }
        }
        // The bound is not slack: at least one tessellation must actually cost
        // more than 15%, or the budget stopped binding and the pin above is
        // measuring nothing.
        assert!(
            worst_ratio > 1.15,
            "no tessellation cost more than {worst_ratio:.4}× — is the budget \
             still binding?"
        );
        // …and the shape error the volume sum cannot see, pinned on a unit sphere.
        assert!(
            (0.30..=0.55).contains(&worst_shape),
            "chunk surfaces run {worst_shape:.3} m outside a 1 m sphere; the v1 \
             hull scope statement quotes ~0.48 m"
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

#[cfg(test)]
mod derived_id_salt {
    /// **The two derived-asset salts must differ.**
    ///
    /// `derived_fracture_id` and `inf_vgeom::derived_vmesh_id` are both
    /// bijections on a mesh's own GUID, and the cook stores their outputs in ONE
    /// pack index. If the salts ever coincided, a mesh's `.inf_vmesh` and its
    /// `.inf_fracture` would collide on a single key and one would silently
    /// overwrite the other — a wall that renders as its meshlet DAG and breaks
    /// into the DAG's bytes, or the reverse.
    ///
    /// The two constants live in two crates that do not depend on each other, so
    /// nothing but this test relates them. It is cheap, and the failure it
    /// prevents is un-debuggable.
    #[test]
    fn the_fracture_and_vmesh_salts_are_distinct() {
        for n in [0u128, 1, 0xDEAD_BEEF, u128::MAX >> 1] {
            let mesh = inf_asset::AssetId(uuid::Uuid::from_u128(n));
            assert_ne!(
                super::derived_fracture_id(mesh),
                inf_vgeom::derived_vmesh_id(mesh),
                "the fracture and vmesh derived ids collide for {mesh:?} — one \
                 would overwrite the other in the pack index"
            );
        }
    }
}
