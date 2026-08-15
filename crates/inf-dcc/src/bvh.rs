//! A **bounding-volume hierarchy over triangles** (P24.2): closest-point,
//! nearest-hit and inside/outside, deterministically.
//!
//! # Why it lives here and not in `inf-core`
//!
//! `inf-core` is the job system, and everything in the tree depends on it — so a
//! BVH there would put a modelling structure in the *simulation's* dependency
//! cone for no benefit. Nothing in the fixed step wants one: this exists to point
//! a skeleton at a mesh and to ask a mesh whether a point is inside it, which are
//! authoring questions. `inf-dcc` is a **dev**-dependency of `inf-player`, so the
//! shipped runtime does not link a byte of it, and that property is worth
//! keeping.
//!
//! # Determinism, and what it costs
//!
//! Two runs on the same triangles build the same tree and answer the same
//! queries, on any target:
//!
//! * **The split is a sort, and the sort is total.** Triangles are ordered by
//!   `f64::total_cmp` on their centroid along the split axis, tie-broken by their
//!   original index — so equal centroids (a symmetric mesh has many) cannot be
//!   ordered by whichever the sort happened to see first.
//! * **The traversal order is fixed.** `closest_point` visits the nearer child
//!   first, and "nearer" is decided with `total_cmp` plus a left-first tie-break,
//!   so a point equidistant from both boxes takes the same branch every time.
//! * **Only `+ - * /` and comparisons.** No trigonometry (the P14 law); `sqrt`
//!   is used freely — IEEE-754 specifies it exactly.
//!
//! The cost is that the split is a **median**, not a SAH: a surface-area
//! heuristic evaluates a cost function whose ties are broken by floating
//! comparison, and its benefit is query throughput on scenes far larger than a
//! character mesh. A median split is `O(n log n)` to build, balanced by
//! construction, and its worst case is a bad tree rather than a wrong answer.
//!
//! # Inside/outside is a WINDING DEPTH, not a parity
//!
//! [`Bvh::contains`] counts *signed* crossings — `+1` leaving a solid, `−1`
//! entering one — and asks whether the total is at least 1. Parity (odd = in)
//! is the textbook answer and it is wrong for the meshes this is pointed at: a
//! character built as a union of overlapping parts (a torso box and an arm box
//! that interpenetrate) gives a point inside *both* an even crossing count, and
//! parity reports it as outside. The signed sum reports 2, which is the number of
//! solids containing it, and `>= 1` is the question actually being asked.
//!
//! Rays are deliberately **not axis-aligned**: an axis-aligned ray through an
//! axis-aligned box hits its edges exactly, and an exact edge hit is counted
//! twice or zero times depending on which side of the epsilon it lands. Three
//! skew directions, majority vote.

use glam::DVec3;

use crate::topo::Mesh;

/// Leaves hold at most this many triangles.
///
/// Four is the usual figure for a median-split tree: below it the node overhead
/// dominates, above it a leaf test costs more than the box test that would have
/// pruned it.
pub const LEAF_TRIANGLES: usize = 4;

/// Depth beyond which a node becomes a leaf whatever its size.
///
/// A median split halves the count, so a balanced tree over `n` triangles is
/// `log2(n)` deep and this is never reached in practice. It exists so a
/// pathological input — thousands of triangles sharing one centroid, which a
/// median split cannot separate — costs a big leaf rather than an unbounded
/// recursion.
pub const MAX_DEPTH: u32 = 64;

/// The three ray directions [`Bvh::contains`] votes over.
///
/// Skew on purpose (see the module docs) and fixed, so containment is a pure
/// function of the geometry. The components are small integers over 100 so the
/// values are exactly representable and identical on every target.
const VOTE_DIRECTIONS: [[f64; 3]; 3] = [[1.0, 0.13, 0.29], [0.17, 1.0, 0.31], [0.23, 0.37, 1.0]];

/// One triangle, in the order it was given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tri {
    pub a: DVec3,
    pub b: DVec3,
    pub c: DVec3,
}

impl Tri {
    /// The unnormalized outward normal (`(b−a) × (c−a)`), i.e. twice the area
    /// vector. Unnormalized deliberately: every consumer here only needs its
    /// *sign* against a direction, and normalizing a degenerate triangle would
    /// produce a NaN where a zero is the honest answer.
    pub fn normal(&self) -> DVec3 {
        (self.b - self.a).cross(self.c - self.a)
    }

    fn centroid(&self) -> DVec3 {
        (self.a + self.b + self.c) / 3.0
    }
}

/// A node: its box, and either two children or a run of triangles.
#[derive(Debug, Clone, Copy)]
struct Node {
    min: DVec3,
    max: DVec3,
    /// Leaf: index of the first triangle. Interior: index of the LEFT child.
    first: u32,
    /// Leaf: triangle count (> 0). Interior: 0.
    count: u32,
    /// Interior: index of the right child. Unused in a leaf.
    right: u32,
}

/// The closest point on a mesh to a query point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosestHit {
    /// The point itself, on the surface.
    pub point: DVec3,
    /// Distance from the query point (metres).
    pub distance: f64,
    /// Index into [`Bvh::triangles`].
    pub triangle: usize,
}

/// Where a ray first meets the surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Ray parameter (`origin + t · direction`), always `> 0`.
    pub t: f64,
    /// Index into [`Bvh::triangles`].
    pub triangle: usize,
}

/// A median-split BVH over a fixed triangle list.
#[derive(Debug, Clone)]
pub struct Bvh {
    nodes: Vec<Node>,
    /// Triangles in **leaf order** — the build permutes them so a leaf is a
    /// contiguous run.
    tris: Vec<Tri>,
    /// `source[i]` is the index `tris[i]` had in the caller's list, so a hit can
    /// be reported against the caller's own numbering.
    source: Vec<u32>,
}

impl Bvh {
    /// Build from a triangle list. An empty list gives an empty hierarchy whose
    /// queries all answer "nothing" rather than panicking.
    pub fn new(tris: Vec<Tri>) -> Self {
        let n = tris.len();
        if n == 0 {
            return Self {
                nodes: Vec::new(),
                tris,
                source: Vec::new(),
            };
        }
        // Order is decided on an index list first, then applied — so the split
        // never reads a half-permuted array.
        let mut order: Vec<u32> = (0..n as u32).collect();
        let centroids: Vec<DVec3> = tris.iter().map(Tri::centroid).collect();
        let mut nodes: Vec<Node> = Vec::with_capacity(2 * n);
        nodes.push(Node {
            min: DVec3::ZERO,
            max: DVec3::ZERO,
            first: 0,
            count: 0,
            right: 0,
        });
        build(
            &Src {
                tris: &tris,
                centroids: &centroids,
            },
            &mut order,
            &mut nodes,
            0,
            (0, n),
            0,
        );
        let ordered: Vec<Tri> = order.iter().map(|&i| tris[i as usize]).collect();
        Self {
            nodes,
            tris: ordered,
            source: order,
        }
    }

    /// Build from a kernel mesh, triangulating each face as a fan.
    ///
    /// A fan is enough here and would not be enough at export: this structure
    /// answers *proximity* questions, and a fan over a non-convex n-gon can only
    /// misreport the interior of that n-gon's own concavity — whereas an exported
    /// asset with a bad fan is geometry an author has to live with. `export`
    /// therefore ear-clips and this does not.
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let mut tris = Vec::new();
        for f in mesh.face_ids() {
            let Some(verts) = mesh.face_verts(f) else {
                continue;
            };
            let ps: Vec<DVec3> = verts
                .iter()
                .filter_map(|&v| mesh.position(v))
                .filter(|p| p.is_finite())
                .collect();
            for i in 1..ps.len().saturating_sub(1) {
                tris.push(Tri {
                    a: ps[0],
                    b: ps[i],
                    c: ps[i + 1],
                });
            }
        }
        Self::new(tris)
    }

    /// The triangles, in leaf order.
    pub fn triangles(&self) -> &[Tri] {
        &self.tris
    }

    /// The index a triangle had in the caller's original list.
    pub fn source_index(&self, triangle: usize) -> Option<usize> {
        self.source.get(triangle).map(|&i| i as usize)
    }

    /// Whether the hierarchy covers no triangles.
    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    /// The bounds of every triangle, or `None` when empty.
    pub fn bounds(&self) -> Option<(DVec3, DVec3)> {
        self.nodes.first().map(|r| (r.min, r.max))
    }

    /// The closest point on the surface to `p`.
    ///
    /// Branch and bound: a node whose box is already further than the best hit
    /// cannot contain a better one. The nearer child is visited first so the
    /// bound tightens early, and "nearer" is a `total_cmp` with a left-first
    /// tie-break, so the traversal is the same on every run.
    pub fn closest_point(&self, p: DVec3) -> Option<ClosestHit> {
        if self.tris.is_empty() {
            return None;
        }
        let mut best: Option<ClosestHit> = None;
        let mut stack: Vec<(u32, f64)> = vec![(0, 0.0)];
        while let Some((idx, bound)) = stack.pop() {
            if let Some(b) = &best {
                if bound >= b.distance * b.distance {
                    continue;
                }
            }
            let node = self.nodes[idx as usize];
            if node.count > 0 {
                for k in 0..node.count as usize {
                    let ti = node.first as usize + k;
                    let q = closest_on_triangle(p, &self.tris[ti]);
                    let d = (q - p).length();
                    let better = match &best {
                        None => true,
                        // The tie-break makes two equidistant triangles resolve
                        // by INDEX rather than by traversal order, so a symmetric
                        // mesh (every one of them) reports the same triangle
                        // twice running.
                        Some(b) => d < b.distance || (d == b.distance && ti < b.triangle),
                    };
                    if better {
                        best = Some(ClosestHit {
                            point: q,
                            distance: d,
                            triangle: ti,
                        });
                    }
                }
                continue;
            }
            let (l, r) = (node.first, node.right);
            let dl = aabb_distance2(p, self.nodes[l as usize]);
            let dr = aabb_distance2(p, self.nodes[r as usize]);
            // Pushed far-first so the near child is popped first.
            if dl.total_cmp(&dr).is_gt() {
                stack.push((l, dl));
                stack.push((r, dr));
            } else {
                stack.push((r, dr));
                stack.push((l, dl));
            }
        }
        best
    }

    /// The first surface `origin + t · direction` meets, for `t > `[`RAY_EPSILON`].
    ///
    /// `direction` need not be normalized; `t` is in units of `direction`.
    pub fn first_hit(&self, origin: DVec3, direction: DVec3) -> Option<RayHit> {
        let mut best: Option<RayHit> = None;
        self.for_each_hit(origin, direction, |t, ti, _| {
            let better = match &best {
                None => true,
                Some(b) => t < b.t || (t == b.t && ti < b.triangle),
            };
            if better {
                best = Some(RayHit { t, triangle: ti });
            }
        });
        best
    }

    /// The **winding depth** at `p` along `direction`: how many solids the point
    /// is inside of, counted as signed crossings (see the module docs).
    pub fn winding_depth(&self, p: DVec3, direction: DVec3) -> i32 {
        let mut depth = 0i32;
        self.for_each_hit(p, direction, |_, _, exiting| {
            depth += if exiting { 1 } else { -1 };
        });
        depth
    }

    /// Whether `p` is inside the surface — a majority of three skew rays saying
    /// [`Bvh::winding_depth`] is at least 1.
    ///
    /// The vote is not decoration. A single ray that grazes an edge counts a
    /// crossing twice or not at all, and a character mesh made of boxes has a lot
    /// of edges; two of three directions have to agree before a point is called
    /// inside.
    pub fn contains(&self, p: DVec3) -> bool {
        let inside = VOTE_DIRECTIONS
            .iter()
            .filter(|d| self.winding_depth(p, DVec3::from_array(**d)) >= 1)
            .count();
        inside >= 2
    }

    /// Walk every triangle the ray meets, in traversal order, handing the
    /// callback `(t, triangle, exiting)`.
    fn for_each_hit(&self, origin: DVec3, direction: DVec3, mut f: impl FnMut(f64, usize, bool)) {
        if self.tris.is_empty() || !direction.is_finite() || direction == DVec3::ZERO {
            return;
        }
        // No reciprocal is precomputed: `slab_hit` takes the direction itself and
        // branches per axis, because a zero component's reciprocal is the ∞ that
        // used to produce a NaN and cull a real hit. See `slab_hit`.
        let mut stack: Vec<u32> = vec![0];
        while let Some(idx) = stack.pop() {
            let node = self.nodes[idx as usize];
            if !slab_hit(origin, direction, node) {
                continue;
            }
            if node.count > 0 {
                for k in 0..node.count as usize {
                    let ti = node.first as usize + k;
                    let tri = &self.tris[ti];
                    if let Some(t) = ray_triangle(origin, direction, tri) {
                        f(t, ti, tri.normal().dot(direction) > 0.0);
                    }
                }
                continue;
            }
            stack.push(node.right);
            stack.push(node.first);
        }
    }
}

/// The smallest `t` a hit is allowed to have.
///
/// A ray fired *from* a point on the surface — which the medial refinement in
/// [`crate::autofit`] does constantly — would otherwise re-hit the triangle it
/// started on at `t ≈ 0`, and that self-hit is a crossing the winding depth would
/// count. In metres, against a character whose smallest feature is centimetres.
pub const RAY_EPSILON: f64 = 1e-9;

/// The immutable half of a build: the triangles and their centroids, bundled so
/// the recursion carries one reference instead of two.
struct Src<'a> {
    tris: &'a [Tri],
    centroids: &'a [DVec3],
}

fn build(
    src: &Src,
    order: &mut [u32],
    nodes: &mut Vec<Node>,
    node: usize,
    range: (usize, usize),
    depth: u32,
) {
    let (from, to) = range;
    let (tris, centroids) = (src.tris, src.centroids);
    let (mut min, mut max) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    let (mut cmin, mut cmax) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    for &i in &order[from..to] {
        let t = &tris[i as usize];
        for p in [t.a, t.b, t.c] {
            min = min.min(p);
            max = max.max(p);
        }
        let c = centroids[i as usize];
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }
    nodes[node].min = min;
    nodes[node].max = max;

    let count = to - from;
    if count <= LEAF_TRIANGLES || depth >= MAX_DEPTH {
        nodes[node].first = from as u32;
        nodes[node].count = count as u32;
        return;
    }
    // The widest axis of the CENTROID bounds, ties by axis order — splitting on
    // the widest extent of the boxes would let one huge triangle choose the axis
    // for a thousand small ones.
    let ext = cmax - cmin;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };
    order[from..to].sort_by(|&a, &b| {
        let (ca, cb) = (
            centroids[a as usize].to_array()[axis],
            centroids[b as usize].to_array()[axis],
        );
        // Total order, then the ORIGINAL index — a symmetric mesh has many
        // triangles sharing a centroid coordinate, and leaving them to the sort's
        // own stability would make the tree depend on the input order in a way
        // nothing states.
        ca.total_cmp(&cb).then_with(|| a.cmp(&b))
    });
    let mid = from + count / 2;

    let left = nodes.len() as u32;
    nodes.push(blank());
    let right = nodes.len() as u32;
    nodes.push(blank());
    nodes[node].first = left;
    nodes[node].count = 0;
    nodes[node].right = right;
    build(src, order, nodes, left as usize, (from, mid), depth + 1);
    build(src, order, nodes, right as usize, (mid, to), depth + 1);
}

fn blank() -> Node {
    Node {
        min: DVec3::ZERO,
        max: DVec3::ZERO,
        first: 0,
        count: 0,
        right: 0,
    }
}

/// Squared distance from `p` to a node's box (zero inside).
fn aabb_distance2(p: DVec3, node: Node) -> f64 {
    let d = (node.min - p).max(DVec3::ZERO) + (p - node.max).max(DVec3::ZERO);
    d.dot(d)
}

/// Slab test: does the ray meet the box at any `t > `[`RAY_EPSILON`]?
///
/// # The branch-free form was WRONG, and it culled real hits
///
/// This used to be `(min − origin) * inv` with `inv = 1/direction`, relying on
/// `min`/`max` to sort the pair. The comment that stood here claimed the only
/// hazard — a NaN from `0 · ∞` when a direction component is zero *and* the
/// origin sits exactly on that slab's plane — could not arise because
/// [`Bvh::for_each_hit`] "refuses a zero direction outright". **It refuses a
/// zero *vector*, not a zero *component*.** A direction like `(0, 0, −1)` has
/// two zero components, `1/0 = ∞`, and an origin whose `y` is exactly the box's
/// `min.y` gives `(min.y − origin.y) · ∞ = 0 · ∞ = NaN`. glam's `min`/`max` are
/// branch forms (`if a < b`), and a NaN compares false against everything, so it
/// propagates into **both** bounds and `enter <= exit` reads false: the node is
/// culled and every triangle under it disappears.
///
/// Measured by the P24.2 audit: 727 divergences from brute force over 31 043
/// real auto-fit ray candidates, 29 of them returning `None` where a hit exists.
/// `Bvh::contains` happened to be immune (its three vote directions have no zero
/// component by construction, which was luck rather than design);
/// [`crate::heat`]'s visibility rays were exposed — 564 of 1 544 tube queries
/// have an exactly-zero component, because a vertex and a bone on a
/// bilaterally-symmetric model share a coordinate constantly.
///
/// # The form that replaces it
///
/// Per-axis, with the zero-direction case handled **explicitly**: a ray parallel
/// to a slab either starts inside that slab (and the axis constrains nothing) or
/// it never enters it. No reciprocal is taken for a zero component, so no `∞`
/// reaches a multiply, so no `NaN` can be produced. The one remaining
/// `0 · ∞` — a *denormal* direction component whose reciprocal overflows, with
/// the origin exactly on the plane — is folded to `0.0`, which is its limit: the
/// ray is at that plane *now*.
///
/// The cost is three branches on a path that had none. Measured against the
/// alternative of keeping the fast form and pre-nudging origins: a nudge is a
/// tolerance, and it would have to be chosen against a model's scale.
fn slab_hit(origin: DVec3, direction: DVec3, node: Node) -> bool {
    let (o, d) = (origin.to_array(), direction.to_array());
    let (lo, hi) = (node.min.to_array(), node.max.to_array());
    let mut enter = RAY_EPSILON;
    let mut exit = f64::INFINITY;
    for axis in 0..3 {
        if d[axis] == 0.0 {
            // Parallel to this slab: it constrains nothing if the origin is
            // already between the planes, and excludes everything if not.
            if o[axis] < lo[axis] || o[axis] > hi[axis] {
                return false;
            }
            continue;
        }
        let inv = 1.0 / d[axis];
        let mut t0 = (lo[axis] - o[axis]) * inv;
        let mut t1 = (hi[axis] - o[axis]) * inv;
        // `0 · ∞`, from a denormal direction component with the origin exactly on
        // the plane. Its limit is 0 — the ray is at that plane now.
        if t0.is_nan() {
            t0 = 0.0;
        }
        if t1.is_nan() {
            t1 = 0.0;
        }
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        enter = enter.max(t0);
        exit = exit.min(t1);
        if enter > exit {
            return false;
        }
    }
    enter <= exit
}

/// Möller–Trumbore, **two-sided**: the winding depth needs the crossings on both
/// facings, and it reads the facing off the triangle's own normal rather than
/// off the determinant's sign.
fn ray_triangle(origin: DVec3, dir: DVec3, tri: &Tri) -> Option<f64> {
    let e1 = tri.b - tri.a;
    let e2 = tri.c - tri.a;
    let h = dir.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1e-15 {
        return None; // parallel, or a degenerate triangle
    }
    let inv = 1.0 / det;
    let s = origin - tri.a;
    let u = s.dot(h) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > RAY_EPSILON).then_some(t)
}

/// The closest point of a triangle to `p` — Ericson's barycentric region test.
///
/// Branch-free of transcendentals and exact on the degenerate cases: a triangle
/// collapsed to a point returns that point rather than a NaN, which matters
/// because a mesh handed to this may carry one.
pub fn closest_on_triangle(p: DVec3, tri: &Tri) -> DVec3 {
    let (a, b, c) = (tri.a, tri.b, tri.c);
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        return if denom != 0.0 {
            a + ab * (d1 / denom)
        } else {
            a
        };
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        return if denom != 0.0 {
            a + ac * (d2 / denom)
        } else {
            a
        };
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        return if denom != 0.0 {
            b + (c - b) * ((d4 - d3) / denom)
        } else {
            b
        };
    }
    let denom = va + vb + vc;
    if denom == 0.0 {
        return a; // fully degenerate
    }
    let inv = 1.0 / denom;
    a + ab * (vb * inv) + ac * (vc * inv)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::build::{cube, cylinder};

    /// An axis-aligned box as 12 outward-wound triangles.
    pub(crate) fn box_tris(min: DVec3, max: DVec3) -> Vec<Tri> {
        let v = |i: usize| {
            DVec3::new(
                if i & 1 == 0 { min.x } else { max.x },
                if i & 2 == 0 { min.y } else { max.y },
                if i & 4 == 0 { min.z } else { max.z },
            )
        };
        // Each face as two triangles, wound so the normal points out.
        let quads: [[usize; 4]; 6] = [
            [0, 2, 3, 1], // -z
            [4, 5, 7, 6], // +z
            [0, 1, 5, 4], // -y
            [2, 6, 7, 3], // +y
            [0, 4, 6, 2], // -x
            [1, 3, 7, 5], // +x
        ];
        let mut out = Vec::with_capacity(12);
        for q in quads {
            out.push(Tri {
                a: v(q[0]),
                b: v(q[1]),
                c: v(q[2]),
            });
            out.push(Tri {
                a: v(q[0]),
                b: v(q[2]),
                c: v(q[3]),
            });
        }
        out
    }

    #[test]
    fn a_box_is_wound_outward_and_contains_its_own_middle() {
        let tris = box_tris(DVec3::splat(-1.0), DVec3::splat(1.0));
        let bvh = Bvh::new(tris);
        assert!(bvh.contains(DVec3::ZERO));
        assert!(bvh.contains(DVec3::new(0.9, 0.9, 0.9)));
        for p in [
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, -3.0, 0.0),
            DVec3::new(1.5, 1.5, 1.5),
        ] {
            assert!(!bvh.contains(p), "{p:?} is outside the box");
        }
    }

    /// **The reason it is a winding depth and not a parity.** Two overlapping
    /// boxes: a point in the overlap crosses an even number of surfaces, so
    /// parity calls it outside and the signed sum calls it doubly inside.
    #[test]
    fn a_point_inside_two_overlapping_solids_is_inside() {
        let mut tris = box_tris(DVec3::new(-1.0, -1.0, -1.0), DVec3::new(1.0, 1.0, 1.0));
        tris.extend(box_tris(
            DVec3::new(-0.5, -0.5, -0.5),
            DVec3::new(2.0, 0.5, 0.5),
        ));
        let bvh = Bvh::new(tris);
        let p = DVec3::new(0.0, 0.0, 0.0);
        assert_eq!(
            bvh.winding_depth(p, DVec3::from_array(VOTE_DIRECTIONS[0])),
            2,
            "the point is inside BOTH boxes; a parity test reports it outside"
        );
        assert!(bvh.contains(p));
        // …and the part of the second box that sticks out is still inside.
        assert!(bvh.contains(DVec3::new(1.5, 0.0, 0.0)));
        assert!(!bvh.contains(DVec3::new(1.5, 0.9, 0.0)));
    }

    #[test]
    fn closest_point_on_a_box_face_is_the_projection() {
        let bvh = Bvh::new(box_tris(DVec3::splat(-1.0), DVec3::splat(1.0)));
        let hit = bvh.closest_point(DVec3::new(0.0, 5.0, 0.0)).unwrap();
        assert!((hit.point - DVec3::new(0.0, 1.0, 0.0)).length() < 1e-12);
        assert!((hit.distance - 4.0).abs() < 1e-12);
        // From inside, the closest surface is the nearest face.
        let hit = bvh.closest_point(DVec3::new(0.9, 0.0, 0.0)).unwrap();
        assert!((hit.distance - 0.1).abs() < 1e-12, "{hit:?}");
    }

    /// The exhaustive control: on every query point, the hierarchy agrees with a
    /// brute-force scan of every triangle. This is what makes the pruning
    /// trustworthy — a wrong bound produces a *plausible* answer, not a crash.
    #[test]
    fn the_hierarchy_agrees_with_brute_force() {
        let mesh = cylinder(0.6, 1.7, 24);
        let bvh = Bvh::from_mesh(&mesh);
        assert!(bvh.triangles().len() > 40, "a real fixture, not two faces");
        let mut checked = 0;
        for i in -3..=3 {
            for j in -3..=3 {
                for k in -3..=3 {
                    let p = DVec3::new(i as f64 * 0.4, j as f64 * 0.4, k as f64 * 0.4);
                    let brute = bvh
                        .triangles()
                        .iter()
                        .map(|t| (closest_on_triangle(p, t) - p).length())
                        .fold(f64::INFINITY, f64::min);
                    let hit = bvh.closest_point(p).unwrap();
                    assert!(
                        (hit.distance - brute).abs() < 1e-12,
                        "at {p:?}: bvh {} vs brute {brute}",
                        hit.distance
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 343);
    }

    #[test]
    fn building_twice_gives_the_same_tree_and_the_same_answers() {
        let mesh = cube(2.0);
        let a = Bvh::from_mesh(&mesh);
        let b = Bvh::from_mesh(&mesh);
        assert_eq!(a.source, b.source, "the permutation must be reproducible");
        for i in -2..=2 {
            let p = DVec3::new(i as f64 * 0.7, 0.3, -0.2);
            assert_eq!(a.closest_point(p), b.closest_point(p));
            assert_eq!(a.contains(p), b.contains(p));
        }
    }

    /// The split ordering is total: triangles whose centroids tie on the split
    /// axis are ordered by their original index, so feeding the SAME triangles in
    /// a different order gives the same *set* of leaves rather than a tree that
    /// depends on input order for its answers.
    #[test]
    fn a_shuffled_input_answers_identically() {
        let tris = box_tris(DVec3::splat(-1.0), DVec3::splat(1.0));
        let mut shuffled = tris.clone();
        shuffled.reverse();
        let a = Bvh::new(tris);
        let b = Bvh::new(shuffled);
        for i in -3..=3 {
            for j in -3..=3 {
                let p = DVec3::new(i as f64 * 0.5, j as f64 * 0.5, 0.25);
                assert_eq!(
                    a.closest_point(p).map(|h| h.distance),
                    b.closest_point(p).map(|h| h.distance)
                );
                assert_eq!(a.contains(p), b.contains(p));
            }
        }
    }

    #[test]
    fn an_empty_hierarchy_answers_nothing_rather_than_panicking() {
        let bvh = Bvh::new(Vec::new());
        assert!(bvh.is_empty());
        assert_eq!(bvh.closest_point(DVec3::ZERO), None);
        assert_eq!(bvh.first_hit(DVec3::ZERO, DVec3::X), None);
        assert!(!bvh.contains(DVec3::ZERO));
        assert_eq!(bvh.bounds(), None);
    }

    #[test]
    fn a_ray_from_the_surface_does_not_hit_the_triangle_it_left() {
        let bvh = Bvh::new(box_tris(DVec3::splat(-1.0), DVec3::splat(1.0)));
        // Straight off the +x face, outward: the only surfaces ahead are none.
        let from = DVec3::new(1.0, 0.0, 0.0);
        assert_eq!(
            bvh.first_hit(from, DVec3::X),
            None,
            "the face the ray starts on must not be its own first hit"
        );
        // …and inward it crosses the far face exactly once.
        let hit = bvh.first_hit(from, -DVec3::X).expect("crosses the box");
        assert!((hit.t - 2.0).abs() < 1e-9, "{hit:?}");
    }

    /// **The P24.2 audit's minimal repro, enshrined** (blocker B2).
    ///
    /// A ray straight down `−z` at an origin whose `y` is *exactly* the box's
    /// `min.y`. Under the old branch-free slab test `1/0 = ∞` and
    /// `(min.y − origin.y) · ∞ = 0 · ∞ = NaN`, which glam's comparison-form
    /// `min`/`max` propagated into both bounds, so `enter <= exit` read false and
    /// the root node was culled: `None`, on a ray that plainly crosses the box.
    ///
    /// The third arm is the one that made it a *blocker* rather than a curiosity:
    /// nudging the origin by a picometre restores the hit, so the failure is
    /// invisible to any test whose numbers are not exactly aligned — and a
    /// bilaterally-symmetric character model aligns coordinates constantly.
    #[test]
    fn a_ray_whose_origin_sits_on_a_slab_plane_is_not_culled() {
        // min.y is exactly 0.0, and so is the ray origin's y.
        let bvh = Bvh::new(box_tris(
            DVec3::new(-1.0, 0.0, -1.0),
            DVec3::new(1.0, 2.0, 1.0),
        ));
        let origin = DVec3::new(0.0, 0.0, 5.0);
        let dir = DVec3::new(0.0, 0.0, -1.0);

        let hit = bvh
            .first_hit(origin, dir)
            .expect("the ray crosses the box; the slab test must not cull it");
        assert!((hit.t - 4.0).abs() < 1e-12, "{hit:?}");

        // Brute force agrees — the authority the sweep below generalizes.
        let brute = bvh
            .triangles()
            .iter()
            .filter_map(|t| ray_triangle(origin, dir, t))
            .fold(f64::INFINITY, f64::min);
        assert!((brute - 4.0).abs() < 1e-12, "brute force says {brute}");

        // …and the picometre nudge that used to be the difference now changes
        // nothing, which is what "no longer alignment-sensitive" means.
        let nudged = bvh
            .first_hit(origin + DVec3::new(0.0, 1e-12, 0.0), dir)
            .expect("still a hit");
        assert!((nudged.t - hit.t).abs() < 1e-9, "{nudged:?} vs {hit:?}");

        // The other two axes, and the winding depth that reads off them.
        for (o, d) in [
            (DVec3::new(-5.0, 0.0, 0.0), DVec3::X),
            (DVec3::new(0.0, 0.0, -5.0), DVec3::Z),
            (DVec3::new(-1.0, 5.0, 0.0), -DVec3::Y),
        ] {
            assert!(
                bvh.first_hit(o, d).is_some(),
                "an axis-aligned ray from {o:?} along {d:?} was culled"
            );
        }
    }

    /// **The audit's regression instrument**: the hierarchy agrees with brute
    /// force over a *candidate sweep*, and the divergence counter is asserted to
    /// be zero rather than merely "no panic".
    ///
    /// Every origin here is drawn from the coordinates the geometry itself uses —
    /// face planes, corners, midpoints — because that is where the defect lived:
    /// a sweep over pseudo-random floats would have hit an exact plane with
    /// probability zero and certified the broken code. Directions include every
    /// signed axis (two zero components each), every face diagonal (one zero
    /// component) and a skew control.
    #[test]
    fn no_ray_diverges_from_brute_force_over_the_aligned_candidate_sweep() {
        let mut tris = box_tris(DVec3::new(-1.0, 0.0, -1.0), DVec3::new(1.0, 2.0, 1.0));
        tris.extend(box_tris(
            DVec3::new(0.5, 0.5, -0.5),
            DVec3::new(2.5, 1.0, 0.5),
        ));
        let bvh = Bvh::new(tris);

        // Coordinates the geometry is built from — the alignment the bug needed.
        let coords = [-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 2.5];
        let dirs: Vec<DVec3> = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, -1.0],
            [-1.0, 0.0, 1.0],
            [0.37, -0.81, 0.29],
        ]
        .into_iter()
        .map(DVec3::from_array)
        .collect();

        let mut candidates = 0usize;
        let mut real_hits = 0usize;
        let mut divergences = 0usize;
        let mut missed_entirely = 0usize;
        let mut worst: Option<(DVec3, DVec3, Option<f64>, f64)> = None;
        for &x in &coords {
            for &y in &coords {
                for &z in &coords {
                    let origin = DVec3::new(x, y, z);
                    for &dir in &dirs {
                        candidates += 1;
                        let mine = bvh.first_hit(origin, dir).map(|h| h.t);
                        // The authority: every triangle, no hierarchy.
                        let brute = bvh
                            .triangles()
                            .iter()
                            .filter_map(|t| ray_triangle(origin, dir, t))
                            .fold(f64::INFINITY, f64::min);
                        let brute = brute.is_finite().then_some(brute);
                        if brute.is_some() {
                            real_hits += 1;
                        }
                        let agrees = match (mine, brute) {
                            (None, None) => true,
                            (Some(a), Some(b)) => (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                            _ => false,
                        };
                        if !agrees {
                            divergences += 1;
                            if brute.is_some() && mine.is_none() {
                                missed_entirely += 1;
                            }
                            if worst.is_none() {
                                worst = Some((origin, dir, mine, brute.unwrap_or(f64::NAN)));
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            (divergences, missed_entirely),
            (0, 0),
            "{divergences} of {candidates} candidates diverge from brute force \
             ({missed_entirely} returning None where a hit exists); first: {worst:?}"
        );
        // NOT VACUOUS: the sweep is big, and most of it really does cross the
        // geometry — an agreement over rays that all miss proves nothing.
        assert!(candidates > 5_000, "{candidates} candidates is too few");
        assert!(
            real_hits > candidates / 4,
            "only {real_hits} of {candidates} candidates hit anything; two \
             agreeing `None`s are not a measurement"
        );
    }

    #[test]
    fn a_zero_direction_ray_is_a_value_not_a_hang() {
        let bvh = Bvh::new(box_tris(DVec3::splat(-1.0), DVec3::splat(1.0)));
        assert_eq!(bvh.first_hit(DVec3::ZERO, DVec3::ZERO), None);
        assert_eq!(bvh.winding_depth(DVec3::ZERO, DVec3::ZERO), 0);
        assert_eq!(bvh.first_hit(DVec3::ZERO, DVec3::splat(f64::NAN)), None);
    }
}
