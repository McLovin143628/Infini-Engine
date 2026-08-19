//! **UV seams, LSCM unwrap and deterministic packing** (P23.5).
//!
//! Three stages, and the whole module is a pure function of the mesh:
//!
//! 1. **Charts** — the connected components of the face graph after the seam
//!    edges are cut ([`charts`]). A chart is a set of faces; a *chart vertex* is
//!    a `(chart, VertId)` pair, so a vertex on a seam appears once in each chart
//!    that touches it and the cut really is a cut.
//! 2. **Parameterize** — least-squares conformal maps (LSCM, Lévy et al. 2002)
//!    per chart, assembled deterministically and solved with a **fixed-iteration**
//!    conjugate gradient whose residual is *reported*, not hidden ([`solve_chart`]).
//! 3. **Pack** — shelf packing by decreasing height into `[0, 1]²`, with the
//!    tie-breaks written down ([`pack`]).
//!
//! # The result is journalled, the solver is not
//!
//! [`unwrap`] returns the computed per-corner UVs and the caller applies them as
//! one [`crate::ops::Op::Unwrap`]. **Replay does not re-solve.** The solver is
//! deterministic today — everything below is `BTreeMap`-ordered `f64` arithmetic
//! over `+`, `-`, `*`, `/` and `sqrt`, with no transcendental and no hash
//! container — and that is *still* not a reason to make an op mean "whatever this
//! build's solver says". The meshopt lesson (P18), applied before it can bite: a
//! journal is replayed by a different build than wrote it, so an op defined by a
//! computation is an op whose recorded effect changes when the computation
//! improves. Carrying the values makes an unwrap a **fact**, and the solver a
//! tool that produced it once.
//!
//! # Honest convergence
//!
//! The CG runs a **fixed** [`CG_ITERATIONS`] iterations — a fixed count is what
//! makes the answer a pure function of the input rather than of a tolerance
//! comparison that could stop one iteration earlier on a different rounding. The
//! price is that it may stop short of convergence, so
//! [`ChartReport::residual`] carries `‖Ax − b‖ / ‖b‖` and
//! [`UnwrapReport::worst_residual`] surfaces the worst of them. **A big residual
//! is a distorted chart, not an error**, and the caller shows the number rather
//! than pretending.
//!
//! # LSCM in one paragraph
//!
//! Each triangle is laid out isometrically in its own plane; conformality asks
//! that the map from those local coordinates to `(u, v)` satisfy Cauchy–Riemann.
//! Writing `W_j` for the complex edge terms of triangle `t` scaled by
//! `1/√(2·area)`, the energy is `Σ_t |Σ_j W_j (u_j + i·v_j)|²`, which is a
//! linear least-squares system in `(u, v)`. It has a 4-dimensional null space
//! (translation, rotation, scale of the plane), so **two vertices are pinned**;
//! everything else is free, which is what makes LSCM a *free-boundary* method and
//! why it does not need a boundary to be mapped to a circle first.
//!
//! # Units
//!
//! Positions are metres; UVs are dimensionless and land in `[0, 1]²`.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec2;
#[cfg(test)]
use glam::DVec3;

use crate::ops::{Op, OpError};
use crate::topo::{FaceId, HalfId, Mesh, VertId};

/// The **floor** on conjugate-gradient iterations per chart.
///
/// The count is not a tolerance test — a `while residual > tol` loop makes the
/// iteration count depend on a floating comparison and therefore makes the answer
/// depend on rounding, which on committed content is the same class of hazard as
/// a non-portable `sin`. It is [`cg_iterations`], an **integer function of the
/// chart's size**, which keeps the determinism argument exactly as it was: two
/// runs on the same mesh do the same number of iterations because they have the
/// same number of unknowns.
///
/// # Why this stopped being the whole story
///
/// It was a flat 256, justified against the 4-to-16-vertex fixtures in this
/// file's tests — and the P23.5 audit measured what that meant one toolbar button
/// away. A plane subdivided five times is 1 089 vertices, which an author reaches
/// by clicking Subdivide five times; at 256 iterations its residual was 5.7e-2,
/// its edge lengths were wrong by 361%, and **two triangles were folded — on a
/// flat square, whose exact conformal map is a similarity**. Six subdivisions
/// folded 338. Every one of them converges to ~1e-14 by 1024.
///
/// A solver that silently folds geometry at a size its own UI can produce is not
/// a slow solver, it is a wrong one, and reporting the residual does not fix it
/// because the fold is already in the file.
pub const CG_ITERATIONS: usize = 256;

/// The **ceiling** on conjugate-gradient iterations per chart.
///
/// CG on `2n` unknowns terminates in at most `2n` steps in exact arithmetic, so
/// `4n` is twice the theoretical bound and the cap is only reached by charts far
/// past what this editor is for. It exists so a pathological mesh costs a bounded
/// wait rather than an unbounded one.
pub const CG_ITERATION_CAP: usize = 8192;

/// Iterations for a chart with `free` unconstrained vertices — `4·free`, clamped
/// between [`CG_ITERATIONS`] and [`CG_ITERATION_CAP`].
///
/// Integer arithmetic on a count, so it is a pure function of the mesh and not of
/// a rounding: the determinism gates are unchanged by it.
pub fn cg_iterations(free: usize) -> usize {
    free.saturating_mul(4)
        .clamp(CG_ITERATIONS, CG_ITERATION_CAP)
}

/// The gap left between packed charts, as a fraction of the packed square.
///
/// Not zero: two charts sharing an exact UV boundary bleed into each other under
/// bilinear filtering and any mip level above the first, which is the classic
/// "seam line on the texture" and is a property of the *layout*, not of the
/// texture.
pub const PACK_MARGIN: f64 = 0.004;

/// One chart's parameterization verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct ChartReport {
    /// Faces in the chart.
    pub faces: usize,
    /// Distinct vertices in the chart.
    pub verts: usize,
    /// The two vertices pinned to fix the null space, in the order they were
    /// pinned — see [`pin_pair`] for the rule.
    pub pinned: [VertId; 2],
    /// **How distorted the chart is**: `‖Ax − b‖ / ‖b‖` of the conformal system.
    ///
    /// Zero for a developable chart mapped perfectly; non-zero for a chart that
    /// *cannot* be flattened without stretching, however well the solver did. A
    /// saddle at its own optimum reads ~4e-2 and there is nothing wrong with it.
    pub residual: f64,
    /// **Whether the solve finished**: `‖MᵀMx − Mᵀb‖ / ‖Mᵀb‖`, the residual of
    /// the normal equations, which goes to zero **iff** CG converged — regardless
    /// of how flattenable the chart was.
    ///
    /// The two numbers were one field until the P23.5 audit, and conflating them
    /// gave the same reading to opposite causes: a *failed* flat plane read 5.7e-2
    /// and a *converged* saddle read 4.1e-2, so the panel offered the same advice
    /// for "the solver stopped early" and "this shape is not developable". Split,
    /// each one answers its own question.
    pub convergence: f64,
    /// **Triangles whose UV winding is opposite the chart's majority** — the
    /// fold count.
    ///
    /// A conformal solve can converge perfectly on a chart that is not a disk and
    /// hand back a parameterization that overlaps itself; the residual cannot see
    /// it, because an overlapping map can be a genuine least-squares optimum.
    /// Measured on this crate's own seam recipe: a cylinder folds 20 triangles and
    /// a torus **285 of 576**. Counted rather than refused — the author's next
    /// move is another seam, and refusing the unwrap would take away the picture
    /// that shows them where to put it.
    pub flipped: usize,
    /// Triangles in the chart, so `flipped` has a denominator.
    pub triangles: usize,

    // ── the Wave-D instrument (see the module docs' collapse section) ──────
    /// **Triangles with no UV area once the layout is packed and rounded to
    /// `f32`** — the number the P25 defect is about, per chart rather than in
    /// aggregate.
    ///
    /// Measured *after* packing and *in `f32`*, which is the whole point: the
    /// solver's own output is `f64` and packing is a uniform affine map, so a
    /// triangle that reads healthy in either of those can still land on three
    /// identical `f32` UVs in the file. `flipped` is measured before packing
    /// because a winding cannot change under a positive scale; this cannot
    /// borrow that argument, because *resolution* is exactly what the scale
    /// changes.
    ///
    /// **An average hides a station** (the P25 law): a chart-wide collapse and a
    /// scattering of slivers look identical in a global fraction and nothing
    /// alike here.
    pub degenerate_uv: usize,
    /// The chart's packed UV bounding-box diagonal.
    ///
    /// The instrument's other half. `pack` scales **every** chart by one factor,
    /// so a chart whose UV extent is orders of magnitude below the largest one's
    /// comes out of packing spanning a handful of `f32` steps — and its
    /// triangles are then degenerate for a reason that has nothing to do with
    /// its 3D shape.
    pub uv_extent: f64,
    /// The 3D distance between the two pinned vertices.
    ///
    /// [`pin_pair`] takes the *farthest* vertex from the lowest id, so this is a
    /// chart's 3D diameter and cannot be small unless the chart is. Recorded to
    /// **retire** the ill-conditioned-pins hypothesis with a number rather than
    /// leaving it as a suspicion.
    pub pin_distance: f64,
    /// The chart's total 3D area — the denominator for "healthy 3D area, no UV
    /// area".
    pub area3: f64,
    /// **The conformal solve collapsed this chart onto a line and a planar
    /// projection replaced it** (Wave D).
    ///
    /// The P25 defect's fix and its instrument in one field. See `planar_uv`:
    /// on a chart far from developable, the two-pin LSCM energy is genuinely
    /// minimized by squashing everything onto the line through the pins, and the
    /// solve **converges** to it. A projection is a worse map and an infinitely
    /// better one, because every triangle has area and can therefore be
    /// textured.
    ///
    /// A non-zero count is not a failure — it is a chart the author should cut
    /// smaller, and the advisory says so.
    pub projected: bool,
}

/// What an unwrap did.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnwrapReport {
    pub charts: Vec<ChartReport>,
    /// The worst per-chart [`ChartReport::residual`] — how badly the *worst* chart
    /// has to stretch. A property of the geometry, not of the solver.
    pub worst_residual: f64,
    /// The worst per-chart [`ChartReport::convergence`] — whether the *solver*
    /// finished. A property of the solver, not of the geometry.
    pub worst_convergence: f64,
    /// Folded triangles across every chart (see [`ChartReport::flipped`]).
    pub flipped: usize,
    /// Triangles with no UV area in the written `f32` layout, across every chart
    /// (see [`ChartReport::degenerate_uv`]). The aggregate the P25 measurement
    /// was taken in — kept, because a caller wants one number, and now with the
    /// per-chart split beside it that says which kind of zero it is.
    pub degenerate_uv: usize,
    /// Charts the conformal solve collapsed onto a line, which a planar
    /// projection replaced (see [`ChartReport::projected`]).
    pub projected: usize,
    /// Corners written.
    pub corners: usize,
    /// Seam edges the charts were cut along.
    pub seams: usize,
}

/// The computed layout plus its verdict. The op is what gets journalled; the
/// report is what the panel shows.
#[derive(Debug, Clone, PartialEq)]
pub struct Unwrapped {
    pub op: Op,
    pub report: UnwrapReport,
}

/// The **charts**: connected components of the face graph, where two faces are
/// adjacent only across an edge that is *not* a seam.
///
/// Returned sorted — each chart's faces ascending, and the charts themselves
/// ordered by their lowest face id — because everything downstream (the pin
/// rule, the packing order, the op's corner list) has to be a pure function of
/// the mesh and not of a traversal's whim.
pub fn charts(mesh: &Mesh) -> Vec<Vec<FaceId>> {
    let mut seen: BTreeSet<FaceId> = BTreeSet::new();
    let mut out: Vec<Vec<FaceId>> = Vec::new();
    for f in mesh.face_ids() {
        if seen.contains(&f) {
            continue;
        }
        let mut component: BTreeSet<FaceId> = [f].into_iter().collect();
        let mut stack = vec![f];
        while let Some(g) = stack.pop() {
            let Some(loop_halfs) = mesh.face_loop(g) else {
                continue;
            };
            for h in loop_halfs {
                if mesh.is_seam(h) == Some(true) {
                    continue; // the cut
                }
                let Some(t) = mesh.twin(h) else { continue };
                if let Some(Some(n)) = mesh.face_of(t) {
                    if component.insert(n) {
                        stack.push(n);
                    }
                }
            }
        }
        seen.extend(component.iter().copied());
        out.push(component.into_iter().collect());
    }
    out
}

/// How many undirected edges are marked as seams.
pub fn seam_count(mesh: &Mesh) -> usize {
    mesh.half_ids()
        .filter(|&h| {
            mesh.is_seam(h) == Some(true) && crate::select::canonical_edge(mesh, h) == Some(h)
        })
        .count()
}

/// **The pinned pair**, and the rule, written down because a solver's answer
/// depends on it:
///
/// * `p0` is the chart's **lowest [`VertId`]** — an arena index, but a stable one
///   within a journal generation, and the op carries the *result* anyway;
/// * `p1` is the chart vertex at the greatest Euclidean distance from `p0`, with
///   ties broken by the lowest id.
///
/// Two rounds of "farthest point" would approximate the diameter more closely and
/// cost another `O(n)`; one round is enough for what the pins are *for*, which is
/// to fix the translation/rotation/scale null space with a well-separated,
/// deterministically-chosen pair. A pair that is nearly coincident makes the
/// system ill-conditioned, which is why this is a farthest point and not "the two
/// lowest ids".
///
/// `None` when the chart has fewer than two vertices, or every candidate is at
/// the same place as `p0` — a degenerate chart, refused by the caller.
pub fn pin_pair(mesh: &Mesh, verts: &BTreeSet<VertId>) -> Option<[VertId; 2]> {
    let p0 = *verts.iter().next()?;
    let a = mesh.position(p0)?;
    let mut best: Option<(f64, VertId)> = None;
    for &v in verts {
        if v == p0 {
            continue;
        }
        let Some(p) = mesh.position(v) else { continue };
        let d = (p - a).length();
        if !d.is_finite() {
            continue;
        }
        if best.is_none_or(|(bd, _)| d > bd) {
            best = Some((d, v));
        }
    }
    let (d, p1) = best?;
    (d > 1e-12).then_some([p0, p1])
}

/// How flat a chart's parameterization has to be before it counts as
/// **collapsed** — total `|UV area|` below this fraction of its bounding box's
/// **diagonal squared**.
///
/// The denominator is the diagonal and not the box's area, and that is the
/// difference between a detector that works and one that does not: a chart
/// squashed onto a line has a bounding box squashed with it, so `area / (w·h)`
/// stays near ½ for a segment. `area / diag²` is the chart's aspect ratio.
///
/// `1e-4` is an aspect worse than about 1 : 10 000 — a chart that thin carries no
/// texels a bake could read whatever the solver meant by it, and no legitimate
/// parameterization of a surface patch is anywhere near it. Measured on the P25
/// scan fixture; see the ledger for the before/after counts.
pub const COLLAPSE_AREA_FRACTION: f64 = 1e-4;

/// **The fallback for a chart LSCM collapsed onto a line** — a planar
/// projection along the chart's own area-weighted normal.
///
/// # Why a chart collapses, and why this is the answer
///
/// A two-pin LSCM fixes translation, rotation and scale by pinning two
/// vertices, and then minimizes conformal error over everything else. On a chart
/// that is far from developable — a scan patch with real Gaussian curvature,
/// residual 0.5 to 0.8 where a flat chart reads 1e-14 — the minimum of that
/// energy is genuinely a map that squashes the whole chart onto the line through
/// its two pins: shrinking the area costs nothing the energy measures and buys a
/// large reduction in the conformal term. The solve **converges** (measured:
/// 1e-16) to a **degenerate** answer.
///
/// That is the P25 defect, and it is why "healthy 3D area, no UV area" was the
/// reading: the 3D chart is fine, the map is a segment. The bounding box is not
/// small either — a collapsed chart spans a normal-looking `uv_extent` because a
/// segment has length — which is exactly why a per-chart *extent* instrument
/// alone would have missed it and the per-chart *area* one finds it.
///
/// A planar projection is worse than a good conformal map and infinitely better
/// than a segment: every triangle has area, so every triangle can be textured.
/// It is what a box-projection unwrap would have given the chart in the first
/// place. Deterministic: the normal is the same Newell sum the rest of the crate
/// uses, and the tangent is built from whichever world axis is *least* aligned
/// with it, which is a comparison and not a choice.
fn planar_uv(
    mesh: &Mesh,
    verts: &BTreeSet<VertId>,
    tris: &[[VertId; 3]],
) -> Option<BTreeMap<VertId, DVec2>> {
    let mut n = glam::DVec3::ZERO;
    for t in tris {
        let (a, b, c) = (
            mesh.position(t[0])?,
            mesh.position(t[1])?,
            mesh.position(t[2])?,
        );
        n += (b - a).cross(c - a);
    }
    let len = n.length();
    // A chart whose triangle normals cancel exactly — a closed shell cut into
    // one chart — has no plane to project onto. `+Y` is as good as any other
    // answer and is at least an answer; the alternative is refusing an unwrap
    // over a chart the author can fix with one more seam.
    let n = if len.is_finite() && len > 1e-18 {
        n / len
    } else {
        glam::DVec3::Y
    };
    let a = n.abs();
    let axis = if a.x <= a.y && a.x <= a.z {
        glam::DVec3::X
    } else if a.y <= a.z {
        glam::DVec3::Y
    } else {
        glam::DVec3::Z
    };
    let t = axis.cross(n);
    let tl = t.length();
    if !(tl.is_finite() && tl > 1e-18) {
        return None;
    }
    let t = t / tl;
    let b = n.cross(t);
    let mut out = BTreeMap::new();
    for &v in verts {
        let p = mesh.position(v)?;
        if !p.is_finite() {
            return None;
        }
        out.insert(v, DVec2::new(p.dot(t), p.dot(b)));
    }
    Some(out)
}

/// The total `|UV area|` of a chart's triangles, and its bounding box's area.
fn uv_area_and_box(tris: &[[VertId; 3]], uv: &BTreeMap<VertId, DVec2>) -> (f64, f64) {
    let mut area = 0.0f64;
    for t in tris {
        let (Some(a), Some(b), Some(c)) = (uv.get(&t[0]), uv.get(&t[1]), uv.get(&t[2])) else {
            continue;
        };
        area += ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs() * 0.5;
    }
    let (mut lo, mut hi) = (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN));
    for p in uv.values() {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let d = hi - lo;
    // **The diagonal SQUARED, not the box's area.** A chart squashed onto a line
    // has a bounding box that is squashed with it, so `area / (w·h)` stays near
    // 0.5 for a segment and the detector reads healthy — measured, and it is why
    // the first version of this fix caught only a fraction of the charts. The
    // diagonal is a length the collapse does NOT shrink, so `area / diag²` is the
    // chart's aspect ratio and goes to zero exactly when it should.
    (
        area,
        if d.is_finite() {
            d.length_squared()
        } else {
            0.0
        },
    )
}

/// A chart's faces, triangulated by a fan for the purposes of the solve.
///
/// A fan, not the writer's ear clipping: LSCM's energy is a sum over triangles
/// and is not sensitive to *which* triangulation of a planar-ish polygon is used,
/// while the writer's ear clipping exists to produce non-overlapping output
/// triangles and is considerably more expensive. What matters here is that the
/// choice is deterministic and that a degenerate triangle is *counted*, not
/// silently skipped — a chart made entirely of them has no parameterization and
/// is refused.
fn triangles(mesh: &Mesh, faces: &[FaceId]) -> (Vec<[VertId; 3]>, usize) {
    let mut tris = Vec::new();
    let mut degenerate = 0usize;
    for &f in faces {
        let Some(loop_verts) = mesh.face_verts(f) else {
            continue;
        };
        for i in 1..loop_verts.len().saturating_sub(1) {
            let t = [loop_verts[0], loop_verts[i], loop_verts[i + 1]];
            match tri_area2(mesh, t) {
                Some(a) if a > 1e-24 => tris.push(t),
                _ => degenerate += 1,
            }
        }
    }
    (tris, degenerate)
}

/// Twice the area of a triangle, or `None` if a position is missing/non-finite.
fn tri_area2(mesh: &Mesh, t: [VertId; 3]) -> Option<f64> {
    let a = mesh.position(t[0])?;
    let b = mesh.position(t[1])?;
    let c = mesh.position(t[2])?;
    let n = (b - a).cross(c - a);
    n.is_finite().then(|| n.length())
}

/// A triangle laid out isometrically in its own plane: `(0,0)`, `(|ab|, 0)`, and
/// `c` resolved against that basis.
fn local_frame(mesh: &Mesh, t: [VertId; 3]) -> Option<[DVec2; 3]> {
    let a = mesh.position(t[0])?;
    let b = mesh.position(t[1])?;
    let c = mesh.position(t[2])?;
    let ab = b - a;
    let len = ab.length();
    if !(len.is_finite() && len > 1e-18) {
        return None;
    }
    let x = ab / len;
    let ac = c - a;
    let px = ac.dot(x);
    let perp = ac - x * px;
    let py = perp.length();
    if !(px.is_finite() && py.is_finite()) {
        return None;
    }
    Some([DVec2::ZERO, DVec2::new(len, 0.0), DVec2::new(px, py)])
}

/// Solve one chart's LSCM system.
///
/// Returns `(uv per chart vertex, distortion residual, convergence residual)`.
/// The system is built as `M x = r`
/// with `x` the `2·free` unknowns; the pinned columns move to the right-hand
/// side, and it is solved in the least-squares sense by conjugate gradient on the
/// normal equations `MᵀM x = Mᵀr`. `MᵀM` is never formed: CG only needs the
/// product `Mᵀ(M p)`, which keeps the work proportional to the number of
/// triangles rather than to the square of the vertex count.
#[allow(clippy::type_complexity)]
fn solve_chart(
    mesh: &Mesh,
    tris: &[[VertId; 3]],
    verts: &BTreeSet<VertId>,
    pins: [VertId; 2],
) -> Option<(BTreeMap<VertId, DVec2>, f64, f64)> {
    // Deterministic indexing: ascending vertex id, pinned vertices excluded.
    let free: Vec<VertId> = verts
        .iter()
        .copied()
        .filter(|v| *v != pins[0] && *v != pins[1])
        .collect();
    let index: BTreeMap<VertId, usize> = free.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let n = free.len();

    // The pins: `p0` at the origin, `p1` at (|p0p1|, 0) so the chart starts at
    // the model's own scale. The packing normalizes afterwards, so the scale is
    // cosmetic — but a pin pair at (0,0) and (1,0) on a 40 m chart makes the
    // free coordinates enormous relative to the pinned ones, which is a needless
    // conditioning problem.
    let d = (mesh.position(pins[1])? - mesh.position(pins[0])?).length();
    let pinned: BTreeMap<VertId, DVec2> = [
        (pins[0], DVec2::ZERO),
        (
            pins[1],
            DVec2::new(if d.is_finite() && d > 0.0 { d } else { 1.0 }, 0.0),
        ),
    ]
    .into_iter()
    .collect();

    // Rows: two per triangle (the real and imaginary halves of the Cauchy-Riemann
    // residual). Each row is a sparse list of (column, coefficient) plus a
    // constant folded from the pinned columns.
    struct Row {
        cols: Vec<(usize, f64)>,
        rhs: f64,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(tris.len() * 2);

    for &t in tris {
        let frame = local_frame(mesh, t)?;
        let area2 = (frame[1].x - frame[0].x) * (frame[2].y - frame[0].y)
            - (frame[2].x - frame[0].x) * (frame[1].y - frame[0].y);
        if !(area2.is_finite() && area2.abs() > 1e-24) {
            continue;
        }
        let scale = 1.0 / area2.abs().sqrt();
        // W_j = (x_{j+2} - x_{j+1}) + i (y_{j+2} - y_{j+1}), cyclically.
        let w: [DVec2; 3] = [
            (frame[2] - frame[1]) * scale,
            (frame[0] - frame[2]) * scale,
            (frame[1] - frame[0]) * scale,
        ];
        // Real row:  Σ (Wr_j u_j - Wi_j v_j) = 0
        // Imag row:  Σ (Wi_j u_j + Wr_j v_j) = 0
        let mut real = Row {
            cols: Vec::with_capacity(6),
            rhs: 0.0,
        };
        let mut imag = Row {
            cols: Vec::with_capacity(6),
            rhs: 0.0,
        };
        for j in 0..3 {
            let v = t[j];
            let (wr, wi) = (w[j].x, w[j].y);
            match index.get(&v) {
                Some(&k) => {
                    real.cols.push((2 * k, wr));
                    real.cols.push((2 * k + 1, -wi));
                    imag.cols.push((2 * k, wi));
                    imag.cols.push((2 * k + 1, wr));
                }
                None => {
                    let p = *pinned.get(&v)?;
                    // Moves to the right-hand side with a sign flip.
                    real.rhs -= wr * p.x - wi * p.y;
                    imag.rhs -= wi * p.x + wr * p.y;
                }
            }
        }
        rows.push(real);
        rows.push(imag);
    }

    let mut x = vec![0.0f64; 2 * n];
    if n > 0 && !rows.is_empty() {
        // r = Mᵀ(rhs - M x0) with x0 = 0  ⇒  r = Mᵀ rhs.
        let mul = |v: &[f64]| -> Vec<f64> {
            rows.iter()
                .map(|row| row.cols.iter().map(|&(c, a)| a * v[c]).sum::<f64>())
                .collect()
        };
        let mul_t = |v: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0f64; 2 * n];
            for (i, row) in rows.iter().enumerate() {
                for &(c, a) in &row.cols {
                    out[c] += a * v[i];
                }
            }
            out
        };
        let rhs: Vec<f64> = rows.iter().map(|r| r.rhs).collect();
        let b = mul_t(&rhs);
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rs = dot(&r, &r);
        for _ in 0..cg_iterations(n) {
            if !(rs.is_finite() && rs > 0.0) {
                break;
            }
            let ap = mul_t(&mul(&p));
            let pap = dot(&p, &ap);
            if !(pap.is_finite() && pap.abs() > 1e-300) {
                break;
            }
            let alpha = rs / pap;
            for i in 0..x.len() {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rs_new = dot(&r, &r);
            if !rs_new.is_finite() {
                break;
            }
            let beta = rs_new / rs;
            for i in 0..p.len() {
                p[i] = r[i] + beta * p[i];
            }
            rs = rs_new;
        }
        // **Two residuals, because there are two questions.**
        //
        // `residual` is of the ORIGINAL system `M x = rhs`: how far the map is
        // from conformal, i.e. how much this chart has to stretch. It is what an
        // author can see, and it is non-zero for a shape that simply is not
        // developable however perfectly the solve went.
        //
        // `convergence` is of the NORMAL equations `MᵀM x = Mᵀb`, whose residual
        // goes to zero if and only if CG finished — regardless of how
        // flattenable the chart was. Until the P23.5 audit only the first was
        // reported, and the two failures it cannot tell apart were measured: a
        // *failed* flat plane read 5.7e-2 and a *converged* saddle read 4.1e-2,
        // and the panel gave the same advice for both.
        let relative = |num: f64, den: f64| if den > 1e-300 { num / den } else { num };
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let ax = mul(&x);
        let residual = relative(
            norm(&ax.iter().zip(&rhs).map(|(a, b)| a - b).collect::<Vec<_>>()),
            norm(&rhs),
        );
        let normal = mul_t(&ax);
        let convergence = relative(
            norm(
                &normal
                    .iter()
                    .zip(&b)
                    .map(|(a, c)| a - c)
                    .collect::<Vec<_>>(),
            ),
            norm(&b),
        );
        if !x.iter().all(|c| c.is_finite()) {
            return None;
        }
        let mut uv: BTreeMap<VertId, DVec2> = pinned.clone();
        for (i, &v) in free.iter().enumerate() {
            uv.insert(v, DVec2::new(x[2 * i], x[2 * i + 1]));
        }
        return Some((uv, residual, convergence));
    }

    Some((pinned, 0.0, 0.0))
}

/// **Triangles whose UV winding opposes the chart's majority** — the folds.
///
/// The sign of the 2D cross product (`perp_dot`) of each triangle's UV edges is
/// its winding; a chart that lays down flat has one sign throughout. The majority
/// is the reference rather than a fixed orientation, because LSCM's null space
/// includes a reflection and a chart mirrored as a whole is not folded.
///
/// Zero-area UV triangles are counted as neither: they have no winding to
/// disagree with, and calling them folds would report a fold on every degenerate
/// sliver a legal mesh is allowed to contain.
fn count_flipped(tris: &[[VertId; 3]], uv: &BTreeMap<VertId, DVec2>) -> usize {
    let signed = |t: &[VertId; 3]| -> Option<f64> {
        let (a, b, c) = (uv.get(&t[0])?, uv.get(&t[1])?, uv.get(&t[2])?);
        let v = (*b - *a).perp_dot(*c - *a);
        v.is_finite().then_some(v)
    };
    let (mut pos, mut neg) = (0usize, 0usize);
    for t in tris {
        match signed(t) {
            Some(v) if v > 0.0 => pos += 1,
            Some(v) if v < 0.0 => neg += 1,
            _ => {}
        }
    }
    pos.min(neg)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// **Shelf packing by decreasing height**, into `[0, 1]²`.
///
/// The rule, in full, because every tie-break in it has to be a pure function of
/// the input:
///
/// 1. each chart is translated so its own bounding box starts at the origin;
/// 2. charts are sorted by **decreasing height**, ties by decreasing width, ties
///    by ascending chart index — so no two charts can ever compare equal;
/// 3. the bin width is `√(Σ area)` widened to fit the widest chart, which makes
///    the packed region roughly square without a search;
/// 4. charts are laid left to right on a shelf; a chart that does not fit starts
///    a new shelf at the height of the tallest so far;
/// 5. everything is scaled by `1 / max(used width, used height)` so the result
///    is inside the unit square, and clamped — the clamp is for `f64` rounding
///    at the boundary, not a substitute for the scaling.
///
/// Takes and returns per-chart UV maps in chart order.
pub fn pack(charts: &mut [BTreeMap<VertId, DVec2>]) {
    if charts.is_empty() {
        return;
    }
    // 1. Normalize each chart to its own origin, and measure it.
    let mut boxes: Vec<(f64, f64)> = Vec::with_capacity(charts.len());
    for chart in charts.iter_mut() {
        let (mut lo, mut hi) = (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN));
        for p in chart.values() {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        if !(lo.is_finite() && hi.is_finite()) {
            boxes.push((0.0, 0.0));
            continue;
        }
        for p in chart.values_mut() {
            *p -= lo;
        }
        let d = hi - lo;
        boxes.push((d.x.max(0.0), d.y.max(0.0)));
    }

    // 2. Decreasing height, then decreasing width, then chart index.
    let mut order: Vec<usize> = (0..charts.len()).collect();
    order.sort_by(|&a, &b| {
        boxes[b]
            .1
            .total_cmp(&boxes[a].1)
            .then(boxes[b].0.total_cmp(&boxes[a].0))
            .then(a.cmp(&b))
    });

    // 3. A bin wide enough to be roughly square and to hold the widest chart.
    let area: f64 = boxes.iter().map(|(w, h)| w * h).sum();
    let widest = boxes.iter().map(|(w, _)| *w).fold(0.0f64, f64::max);
    let bin = area.max(0.0).sqrt().max(widest).max(1e-12);
    let margin = bin * PACK_MARGIN;

    // 4. Shelves.
    let (mut cx, mut cy, mut shelf_h, mut used_w, mut used_h) = (0.0f64, 0.0f64, 0.0f64, 0.0, 0.0);
    let mut offsets: Vec<DVec2> = vec![DVec2::ZERO; charts.len()];
    for &i in &order {
        let (w, h) = boxes[i];
        if cx > 0.0 && cx + w > bin {
            cy += shelf_h + margin;
            cx = 0.0;
            shelf_h = 0.0;
        }
        offsets[i] = DVec2::new(cx, cy);
        cx += w + margin;
        shelf_h = shelf_h.max(h);
        used_w = f64::max(used_w, cx);
        used_h = f64::max(used_h, cy + h);
    }

    // 5. Into the unit square.
    let extent = used_w.max(used_h).max(1e-12);
    let scale = 1.0 / extent;
    for (i, chart) in charts.iter_mut().enumerate() {
        let off = offsets[i];
        for p in chart.values_mut() {
            let q = (*p + off) * scale;
            *p = DVec2::new(q.x.clamp(0.0, 1.0), q.y.clamp(0.0, 1.0));
        }
    }
}

/// **Unwrap the whole mesh**: cut at the seams, parameterize each chart, pack the
/// charts, and hand back the corner UVs as one journalled op.
///
/// Refuses — as a value — when a chart cannot be parameterized: fewer than two
/// distinct vertex positions to pin, or no non-degenerate triangle at all. The
/// error names the chart index, because "unwrap failed" on a forty-chart model is
/// not something an author can act on.
pub fn unwrap(mesh: &Mesh) -> Result<Unwrapped, OpError> {
    let chart_faces = charts(mesh);
    let mut per_chart: Vec<BTreeMap<VertId, DVec2>> = Vec::with_capacity(chart_faces.len());
    let mut reports: Vec<ChartReport> = Vec::with_capacity(chart_faces.len());
    // The triangulation each chart was solved over, kept so the post-pack
    // instrument measures the same triangles the solve did.
    let mut chart_tris: Vec<Vec<[VertId; 3]>> = Vec::with_capacity(chart_faces.len());

    for (ci, faces) in chart_faces.iter().enumerate() {
        let verts: BTreeSet<VertId> = faces
            .iter()
            .filter_map(|&f| mesh.face_verts(f))
            .flatten()
            .collect();
        let (tris, _degenerate) = triangles(mesh, faces);
        if tris.is_empty() {
            return Err(OpError::DegenerateChart {
                chart: ci,
                why: "every triangle in it has zero area",
            });
        }
        let Some(pins) = pin_pair(mesh, &verts) else {
            return Err(OpError::DegenerateChart {
                chart: ci,
                why: "its vertices are all at the same position, so there is no \
                      pair to pin",
            });
        };
        let Some((uv, residual, convergence)) = solve_chart(mesh, &tris, &verts, pins) else {
            return Err(OpError::DegenerateChart {
                chart: ci,
                why: "the conformal solve produced a non-finite coordinate",
            });
        };
        // **THE COLLAPSE FALLBACK** (Wave D) — see `planar_uv` for the mechanism
        // and for why a converged solve can hand back a segment.
        let (uv_area, uv_box) = uv_area_and_box(&tris, &uv);
        let (uv, projected) = if uv_area <= uv_box * COLLAPSE_AREA_FRACTION {
            match planar_uv(mesh, &verts, &tris) {
                Some(flat) => (flat, true),
                // No plane either — the chart's normals cancel exactly AND the
                // tangent degenerates. Keep the conformal answer: a segment is a
                // bad UV layout and refusing the whole unwrap over one chart
                // would take away the picture that shows the author where to cut.
                None => (uv, false),
            }
        } else {
            (uv, false)
        };
        // Counted BEFORE packing: packing translates and scales uniformly, which
        // cannot change a winding, but counting on the solver's own output keeps
        // the number a statement about the solve.
        let flipped = count_flipped(&tris, &uv);
        let pin_distance = match (mesh.position(pins[0]), mesh.position(pins[1])) {
            (Some(a), Some(b)) => (b - a).length(),
            _ => 0.0,
        };
        let area3: f64 = tris
            .iter()
            .filter_map(|&t| tri_area2(mesh, t))
            .map(|a2| a2 * 0.5)
            .sum();
        reports.push(ChartReport {
            faces: faces.len(),
            verts: verts.len(),
            pinned: pins,
            residual,
            convergence,
            flipped,
            triangles: tris.len(),
            // Filled after packing — the two fields that are statements about
            // what lands in the FILE rather than about the solve.
            degenerate_uv: 0,
            uv_extent: 0.0,
            pin_distance,
            area3,
            projected,
        });
        per_chart.push(uv);
        chart_tris.push(tris);
    }

    pack(&mut per_chart);

    // **The Wave-D instrument, measured where the defect lives.** After packing
    // and in `f32`, because that is the domain the asset is written in and the
    // domain the collapse happens in — see `ChartReport::degenerate_uv`.
    for (ci, chart) in per_chart.iter().enumerate() {
        let (mut lo, mut hi) = (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN));
        for p in chart.values() {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        reports[ci].uv_extent = if lo.is_finite() && hi.is_finite() {
            (hi - lo).length()
        } else {
            0.0
        };
        reports[ci].degenerate_uv = chart_tris[ci]
            .iter()
            .filter(|t| {
                let uv = |v: &VertId| {
                    chart
                        .get(v)
                        .map(|p| (p.x as f32, p.y as f32))
                        .unwrap_or((0.0, 0.0))
                };
                let (a, b, c) = (uv(&t[0]), uv(&t[1]), uv(&t[2]));
                let area2 = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
                !(area2.abs() > 0.0)
            })
            .count();
    }

    // Per-CORNER, keyed by half-edge, ordered by half-edge — the op's wire form.
    // A vertex on a seam has a different UV in each chart it belongs to, which is
    // exactly what the corner storage exists for (P23.3 §7a: attributes live
    // where seams live).
    let mut corners: BTreeMap<HalfId, [f64; 2]> = BTreeMap::new();
    for (ci, faces) in chart_faces.iter().enumerate() {
        for &f in faces {
            let Some(loop_halfs) = mesh.face_loop(f) else {
                continue;
            };
            for h in loop_halfs {
                let Some(v) = mesh.origin(h) else { continue };
                if let Some(p) = per_chart[ci].get(&v) {
                    corners.insert(h, [p.x, p.y]);
                }
            }
        }
    }

    let worst = |f: fn(&ChartReport) -> f64| {
        reports
            .iter()
            .map(f)
            .fold(0.0f64, |a, b| if b > a { b } else { a })
    };
    let report = UnwrapReport {
        degenerate_uv: reports.iter().map(|r| r.degenerate_uv).sum(),
        projected: reports.iter().filter(|r| r.projected).count(),
        corners: corners.len(),
        seams: seam_count(mesh),
        worst_residual: worst(|r| r.residual),
        worst_convergence: worst(|r| r.convergence),
        flipped: reports.iter().map(|r| r.flipped).sum(),
        charts: reports,
    };
    Ok(Unwrapped {
        op: Op::Unwrap {
            corners: corners.into_iter().collect(),
        },
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, cylinder, plane};
    use crate::journal::MeshSession;
    use crate::ops::apply;
    use crate::topo::CornerData;
    use crate::topo::FaceId;
    use crate::validate::validate;

    /// An `n × n` grid of unit quads in XZ.
    fn grid(n: usize) -> Mesh {
        let mut m = Mesh::new();
        let mut ids = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                ids.push(m.alloc_vert([i as f64, 0.0, j as f64]));
            }
        }
        let at = |i: usize, j: usize| ids[j * (n + 1) + i];
        let mut touched = BTreeSet::new();
        for j in 0..n {
            for i in 0..n {
                let quad = [at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)];
                touched.extend(quad);
                m.add_face_raw(&quad, &[CornerData::default(); 4], None)
                    .expect("a grid is well formed");
            }
        }
        m.finish_patch(&touched).expect("a grid is manifold");
        m
    }

    /// Mark every edge whose two faces disagree in normal by more than 45° — a
    /// cheap "cut the hard edges" that turns a cube into six charts.
    fn seam_the_hard_edges(m: &mut Mesh) -> usize {
        let mut marked = 0;
        for h in m.half_ids().collect::<Vec<_>>() {
            if crate::select::canonical_edge(m, h) != Some(h) {
                continue;
            }
            let (Some(Some(f)), Some(Some(g))) =
                (m.face_of(h), m.twin(h).and_then(|t| m.face_of(t)))
            else {
                continue;
            };
            let (Some(a), Some(b)) = (
                crate::sculpt::face_normal(m, f),
                crate::sculpt::face_normal(m, g),
            ) else {
                continue;
            };
            if a.dot(b) < 0.7 {
                m.set_seam_pair(h, true);
                marked += 1;
            }
        }
        marked
    }

    #[test]
    fn a_mesh_with_no_seams_is_one_chart_and_a_cut_cube_is_six() {
        let m = cube(1.0);
        assert_eq!(charts(&m).len(), 1, "no seams, one chart");
        assert_eq!(seam_count(&m), 0);

        let mut cut = cube(1.0);
        let marked = seam_the_hard_edges(&mut cut);
        assert_eq!(marked, 12, "a cube has twelve hard edges");
        assert_eq!(seam_count(&cut), 12);
        let cs = charts(&cut);
        assert_eq!(cs.len(), 6, "one chart per face: {cs:?}");
        assert_eq!(cs.iter().map(Vec::len).sum::<usize>(), 6);
        assert_eq!(validate(&cut), Ok(()));
    }

    #[test]
    fn the_chart_count_is_the_seam_cut_component_count() {
        // The property stated directly: cut ONE edge ring of a grid and the chart
        // count follows the topology rather than the geometry.
        let mut m = grid(4);
        // The column of edges at x = 2: cutting it does not separate the grid
        // (the rows still connect around), so it is still ONE chart.
        for h in m.half_ids().collect::<Vec<_>>() {
            let a = m.position(m.origin(h).expect("live")).expect("live");
            let b = m.position(m.dest(h).expect("live")).expect("live");
            if (a.x - 2.0).abs() < 1e-12 && (b.x - 2.0).abs() < 1e-12 {
                m.set_seam_pair(h, true);
            }
        }
        assert_eq!(seam_count(&m), 4);
        assert_eq!(
            charts(&m).len(),
            2,
            "a full-height cut splits the grid in two"
        );
    }

    #[test]
    fn a_planar_chart_unwraps_to_an_isometry_up_to_scale() {
        // The strongest statement available about a conformal map: on a FLAT
        // chart the exact answer is a similarity, so the residual is ~0 and every
        // triangle's UV shape matches its 3D shape. A solver that quietly did
        // nothing would still land in [0,1]² and still be "deterministic".
        let m = grid(3);
        let out = unwrap(&m).expect("a flat grid unwraps");
        assert_eq!(out.report.charts.len(), 1);
        assert!(
            out.report.worst_residual < 1e-9,
            "a planar chart is exactly conformal, got {}",
            out.report.worst_residual
        );
        let Op::Unwrap { corners } = &out.op else {
            panic!("{:?}", out.op)
        };
        let uv: BTreeMap<HalfId, DVec2> = corners
            .iter()
            .map(|&(h, p)| (h, DVec2::from_array(p)))
            .collect();
        // Angle preservation, measured as the ratio of UV edge length to 3D edge
        // length being constant across the chart.
        let mut ratios: Vec<f64> = Vec::new();
        for f in m.face_ids() {
            let halfs = m.face_loop(f).expect("live");
            for w in halfs.windows(2) {
                let (h0, h1) = (w[0], w[1]);
                let p0 = m.position(m.origin(h0).expect("live")).expect("live");
                let p1 = m.position(m.origin(h1).expect("live")).expect("live");
                let d3 = (p1 - p0).length();
                let d2 = (uv[&h1] - uv[&h0]).length();
                if d3 > 1e-9 {
                    ratios.push(d2 / d3);
                }
            }
        }
        assert!(ratios.len() > 20);
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        assert!(mean > 1e-6, "the chart collapsed to a point");
        for r in &ratios {
            assert!(
                (r / mean - 1.0).abs() < 1e-6,
                "a planar unwrap must be a similarity: {r} vs mean {mean}"
            );
        }
    }

    #[test]
    fn every_corner_lands_in_the_unit_square() {
        for mut m in [cube(1.0), grid(3), plane(2.0)] {
            seam_the_hard_edges(&mut m);
            let out = unwrap(&m).expect("unwraps");
            let Op::Unwrap { corners } = &out.op else {
                panic!()
            };
            assert!(!corners.is_empty());
            for &(h, [u, v]) in corners {
                assert!(
                    (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v),
                    "{h} landed at ({u}, {v})"
                );
            }
        }
    }

    #[test]
    fn every_corner_of_every_face_gets_a_uv() {
        // The vacuity guard: a solve that produced UVs for one chart and dropped
        // the rest would still satisfy "everything is in [0,1]²".
        let mut m = cube(2.0);
        seam_the_hard_edges(&mut m);
        let out = unwrap(&m).expect("unwraps");
        let expected: usize = m
            .face_ids()
            .map(|f| m.face_loop(f).map(|l| l.len()).unwrap_or(0))
            .sum();
        assert_eq!(out.report.corners, expected, "one UV per face corner");
        assert_eq!(out.report.seams, 12);
        assert_eq!(out.report.charts.len(), 6);
    }

    #[test]
    fn two_runs_of_the_solve_are_bit_identical() {
        // The determinism gate, on the bits rather than on a tolerance: the
        // assembly is `BTreeMap`-ordered and the CG runs a fixed count, so the
        // same mesh must produce the same `f64`s.
        let mut m = cube(1.0);
        seam_the_hard_edges(&mut m);
        let a = unwrap(&m).expect("a");
        let b = unwrap(&m).expect("b");
        assert_eq!(a.op, b.op);
        assert_eq!(a.report, b.report);
        let cfg = bincode::config::standard();
        assert_eq!(
            bincode::serde::encode_to_vec(&a.op, cfg).unwrap(),
            bincode::serde::encode_to_vec(&b.op, cfg).unwrap(),
        );
    }

    #[test]
    fn a_seam_gives_the_two_sides_different_uvs_at_the_same_vertex() {
        // What corners are FOR. A cube vertex belongs to three charts after the
        // hard edges are cut, and the three corners at it must not agree — if
        // they did, the cut would not be a cut.
        let mut m = cube(1.0);
        seam_the_hard_edges(&mut m);
        let out = unwrap(&m).expect("unwraps");
        let Op::Unwrap { corners } = &out.op else {
            panic!()
        };
        let uv: BTreeMap<HalfId, [f64; 2]> = corners.iter().copied().collect();
        let v = m.vert_ids().next().expect("a vertex");
        let at: Vec<[f64; 2]> = m
            .vert_outgoing(v)
            .expect("live")
            .iter()
            .filter_map(|h| uv.get(h).copied())
            .collect();
        assert_eq!(at.len(), 3, "three faces meet at a cube corner");
        assert!(
            at.iter().any(|p| p != &at[0]),
            "every chart put the corner in the same place: {at:?}"
        );
    }

    #[test]
    fn a_seam_survives_the_ops_that_rebuild_the_patch_it_lives_on() {
        // **The gate the widened `capture_edge_flags` exists for.** Every
        // structural op captures the flags of the edges it is about to destroy
        // and re-applies them to the rebuilt patch; sharpness had that since
        // P23.3 and the seam needed exactly the same treatment at exactly the
        // same dozen sites. Without it an author's UV cut would quietly heal the
        // first time they bevelled near it, and nothing would say so.
        //
        // Split, loop cut and subdivide are the three that *divide* the seam
        // edge, so they also have to hand the flag to BOTH pieces.
        let seamed = |m: &Mesh| seam_count(m);

        // 1. A split of the seam edge itself: one seam becomes two.
        let mut m = grid(2);
        let h = m
            .half_ids()
            .find(|&h| {
                let a = m.position(m.origin(h).expect("live")).expect("live");
                let b = m.position(m.dest(h).expect("live")).expect("live");
                (a.x - 1.0).abs() < 1e-12 && (b.x - 1.0).abs() < 1e-12
            })
            .expect("an interior column edge");
        apply(
            &mut m,
            &Op::SetEdgeSeam {
                half: h,
                seam: true,
            },
        )
        .expect("marks");
        assert_eq!(seamed(&m), 1);
        apply(&mut m, &Op::SplitEdge { half: h, t: 0.5 }).expect("splits");
        assert_eq!(seamed(&m), 2, "both halves of a split seam are seams");
        assert_eq!(validate(&m), Ok(()));

        // 2. Subdivide: the seam is divided and every other edge is untouched.
        let mut m = grid(2);
        let h = m
            .half_ids()
            .find(|&h| {
                let a = m.position(m.origin(h).expect("live")).expect("live");
                let b = m.position(m.dest(h).expect("live")).expect("live");
                (a.x - 1.0).abs() < 1e-12 && (b.x - 1.0).abs() < 1e-12
            })
            .expect("an interior column edge");
        apply(
            &mut m,
            &Op::SetEdgeSeam {
                half: h,
                seam: true,
            },
        )
        .expect("marks");
        let faces: Vec<FaceId> = m.face_ids().collect();
        apply(&mut m, &Op::SubdivideFaces { faces }).expect("subdivides");
        assert_eq!(seamed(&m), 2, "a subdivided seam edge stays a seam");
        assert_eq!(validate(&m), Ok(()));

        // 3. **A face rebuild that does NOT divide the seam edge.** This is the
        // one that goes through `apply_edge_flags` rather than through a split's
        // own inheritance, and it is the arm the widened capture exists for:
        // dropping `rec.seam` from `apply_edge_flags` failed nothing until this
        // case was written (found by mutation).
        let mut m = plane(2.0);
        let f = m.face_ids().next().expect("the quad");
        let halfs = m.face_loop(f).expect("live");
        apply(
            &mut m,
            &Op::SetEdgeSeam {
                half: halfs[0],
                seam: true,
            },
        )
        .expect("marks");
        assert_eq!(seamed(&m), 1);
        apply(
            &mut m,
            &Op::SplitFace {
                from: halfs[0],
                to: halfs[2],
            },
        )
        .expect("splits the quad");
        assert_eq!(
            seamed(&m),
            1,
            "the seam edge was rebuilt with the face and must come back a seam"
        );
        assert_eq!(validate(&m), Ok(()));

        // 4. An op that rebuilds a *neighbouring* patch must not lose it either.
        let mut m = grid(3);
        let h = m
            .half_ids()
            .find(|&h| {
                let a = m.position(m.origin(h).expect("live")).expect("live");
                let b = m.position(m.dest(h).expect("live")).expect("live");
                (a.x - 1.0).abs() < 1e-12 && (b.x - 1.0).abs() < 1e-12
            })
            .expect("an interior column edge");
        apply(
            &mut m,
            &Op::SetEdgeSeam {
                half: h,
                seam: true,
            },
        )
        .expect("marks");
        let f = m.face_ids().next().expect("a face");
        apply(
            &mut m,
            &Op::ExtrudeFaces {
                faces: vec![f],
                distance: 0.3,
            },
        )
        .expect("extrudes");
        assert!(seamed(&m) >= 1, "an extrude nearby dropped the seam");
        assert_eq!(validate(&m), Ok(()));
    }

    #[test]
    fn a_degenerate_chart_refuses_as_a_value() {
        // A face whose vertices are collinear has no area, so it has no
        // parameterization. Refused with the chart index, not a bare error.
        let mut m = Mesh::new();
        let v: Vec<VertId> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
            .iter()
            .map(|p| m.alloc_vert(*p))
            .collect();
        m.add_face_raw(&v, &[CornerData::default(); 3], None)
            .expect("a collinear triangle is topologically fine");
        m.finish_patch(&v.iter().copied().collect())
            .expect("manifold");
        match unwrap(&m) {
            Err(OpError::DegenerateChart { chart, why }) => {
                assert_eq!(chart, 0);
                assert!(why.contains("zero area"), "{why}");
            }
            other => panic!("a zero-area chart was not refused: {other:?}"),
        }

        // And a chart whose vertices are all at one point has no pin pair.
        let mut m = plane(2.0);
        for v in m.vert_ids().collect::<Vec<_>>() {
            m.vert_mut(v).position = [0.0; 3];
        }
        assert!(matches!(unwrap(&m), Err(OpError::DegenerateChart { .. })));
    }

    #[test]
    fn the_pin_rule_is_the_one_written_down() {
        // `p0` is the lowest id and `p1` is the farthest from it. Asserted
        // against a hand answer, because the pins are what fix the solution's
        // null space and a change to them changes every UV in the file.
        let m = grid(2);
        let verts: BTreeSet<VertId> = m.vert_ids().collect();
        let [p0, p1] = pin_pair(&m, &verts).expect("a pin pair");
        assert_eq!(p0, *verts.iter().next().unwrap(), "p0 is the lowest id");
        let a = m.position(p0).expect("live");
        let far = (m.position(p1).expect("live") - a).length();
        for v in &verts {
            let d = (m.position(*v).expect("live") - a).length();
            assert!(d <= far + 1e-12, "{v} is farther than the pinned {p1}");
        }
    }

    #[test]
    fn an_unwrap_is_one_journal_entry_and_one_undo_step() {
        let mut m = cube(1.0);
        seam_the_hard_edges(&mut m);
        let out = unwrap(&m).expect("unwraps");
        let mut s = MeshSession::new(m);
        let before = s.mesh().encoded();
        s.apply(out.op).expect("applies");
        assert_eq!(s.ops().len(), 1);
        assert_ne!(s.mesh().encoded(), before);
        assert_eq!(validate(s.mesh()), Ok(()));
        assert!(s.undo());
        assert_eq!(s.mesh().encoded(), before);
        // And it replays, byte for byte, WITHOUT re-solving.
        assert!(s.redo());
        let replayed =
            MeshSession::replay(s.base(), &s.ops()[..s.cursor()]).expect("journalled ops replay");
        assert_eq!(replayed.encoded(), s.mesh().encoded());
    }

    #[test]
    fn an_unwrap_naming_a_dead_corner_refuses_inertly() {
        let mut m = plane(2.0);
        let before = m.encoded();
        let boundary = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(true))
            .expect("a boundary half");
        for (op, want) in [
            (
                Op::Unwrap {
                    corners: vec![(HalfId(9_999), [0.5, 0.5])],
                },
                OpError::UnwrapCornerMissing(HalfId(9_999)),
            ),
            (
                Op::Unwrap {
                    corners: vec![(boundary, [0.5, 0.5])],
                },
                OpError::UnwrapCornerMissing(boundary),
            ),
        ] {
            assert_eq!(apply(&mut m, &op), Err(want));
            assert_eq!(m.encoded(), before, "the refusal is inert");
        }
        // A non-finite UV is refused at the same door, and also inertly — even
        // when a legal corner comes first in the list.
        let good = m
            .half_ids()
            .find(|&h| m.is_boundary(h) == Some(false))
            .expect("a corner");
        assert!(matches!(
            apply(
                &mut m,
                &Op::Unwrap {
                    corners: vec![(good, [0.25, 0.25]), (good, [f64::NAN, 0.0])],
                },
            ),
            Err(OpError::NonFinite { .. })
        ));
        assert_eq!(m.encoded(), before);
    }

    #[test]
    fn packing_separates_charts_and_keeps_the_margin() {
        // Two unit squares must not overlap and must not touch: a shared UV
        // boundary bleeds under filtering, which is the seam line on a texture.
        let mut charts = vec![
            [
                (VertId(0), DVec2::new(0.0, 0.0)),
                (VertId(1), DVec2::new(1.0, 0.0)),
                (VertId(2), DVec2::new(1.0, 1.0)),
                (VertId(3), DVec2::new(0.0, 1.0)),
            ]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
            [
                (VertId(4), DVec2::new(5.0, 5.0)),
                (VertId(5), DVec2::new(6.0, 5.0)),
                (VertId(6), DVec2::new(6.0, 6.0)),
                (VertId(7), DVec2::new(5.0, 6.0)),
            ]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        ];
        pack(&mut charts);
        let bounds = |c: &BTreeMap<VertId, DVec2>| {
            let (mut lo, mut hi) = (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN));
            for p in c.values() {
                lo = lo.min(*p);
                hi = hi.max(*p);
            }
            (lo, hi)
        };
        let (lo0, hi0) = bounds(&charts[0]);
        let (lo1, hi1) = bounds(&charts[1]);
        for (lo, hi) in [(lo0, hi0), (lo1, hi1)] {
            assert!(lo.x >= 0.0 && lo.y >= 0.0 && hi.x <= 1.0 && hi.y <= 1.0);
        }
        let disjoint_x = hi0.x < lo1.x || hi1.x < lo0.x;
        let disjoint_y = hi0.y < lo1.y || hi1.y < lo0.y;
        assert!(
            disjoint_x || disjoint_y,
            "the two charts overlap: {lo0:?}..{hi0:?} and {lo1:?}..{hi1:?}"
        );
    }

    #[test]
    fn packing_orders_by_decreasing_height_with_written_down_tie_breaks() {
        // The shelf rule, measured: the tallest chart is placed first, so it is
        // the one at the origin. A sort that was not total (equal heights AND
        // equal widths) would make the layout depend on the sort's stability,
        // which is a property of the standard library and not of this rule.
        let mk = |w: f64, h: f64, base: u32| -> BTreeMap<VertId, DVec2> {
            [
                (VertId(base), DVec2::ZERO),
                (VertId(base + 1), DVec2::new(w, h)),
            ]
            .into_iter()
            .collect()
        };
        let mut charts = vec![mk(1.0, 0.5, 0), mk(0.5, 2.0, 10), mk(1.0, 0.5, 20)];
        pack(&mut charts);
        // The tall one (index 1) is placed first, at the origin.
        assert_eq!(charts[1][&VertId(10)], DVec2::ZERO);
        // The two identical ones are separated deterministically, lower index
        // first, so neither is at the origin and they differ.
        assert_ne!(charts[0][&VertId(0)], charts[2][&VertId(20)]);
        // Running it again on the same input gives the same layout.
        let mut again = vec![mk(1.0, 0.5, 0), mk(0.5, 2.0, 10), mk(1.0, 0.5, 20)];
        pack(&mut again);
        assert_eq!(charts, again);
    }

    #[test]
    fn unwrapping_changes_the_tangents_the_writer_derives() {
        // **The integration the whole UV half is for.** P23.3's writer derives
        // MikkTSpace-class tangents from the UVs and falls back to the constant
        // `TANGENT_FALLBACK` when a mesh has none — which is what a kernel mesh
        // with all-zero corner UVs has. So an unwrap must move every written
        // tangent off that constant, and the ones it writes must be real: finite,
        // unit, with a ±1 handedness.
        //
        // Without this the unwrap could be writing UVs nothing downstream reads,
        // and every other test in this module would still pass.
        let mut m = cube(1.0);
        seam_the_hard_edges(&mut m);
        // The primitive generators write real UVs, so the "no UVs" state has to
        // be made: this is the mesh an author gets from a source file with no
        // texture coordinates, which is the case the unwrap exists to fix.
        for h in m.half_ids().collect::<Vec<_>>() {
            m.half_mut(h).uv = [0.0, 0.0];
        }
        let opts = crate::export::ExportOptions::default();

        let (before, before_report) = crate::export::to_mesh_asset(&m, &opts);
        assert_eq!(
            before_report.fallback_tangents,
            before.vertex_count(),
            "a mesh with no UVs takes the constant fallback everywhere"
        );
        assert!(before
            .submeshes
            .iter()
            .flat_map(|s| &s.vertices)
            .all(|v| v.tangent == crate::export::TANGENT_FALLBACK));

        let out = unwrap(&m).expect("unwraps");
        apply(&mut m, &out.op).expect("applies");
        let (after, after_report) = crate::export::to_mesh_asset(&m, &opts);

        assert_eq!(
            after_report.fallback_tangents, 0,
            "every vertex now has a UV, so none of them needs the fallback"
        );
        let mut moved = 0usize;
        for v in after.submeshes.iter().flat_map(|s| &s.vertices) {
            if v.tangent != crate::export::TANGENT_FALLBACK {
                moved += 1;
            }
            let t = DVec3::new(
                v.tangent[0] as f64,
                v.tangent[1] as f64,
                v.tangent[2] as f64,
            );
            assert!(t.is_finite(), "a derived tangent is not finite: {t:?}");
            assert!(
                (t.length() - 1.0).abs() < 1e-5,
                "a derived tangent is not unit: {t:?}"
            );
            assert!(
                v.tangent[3] == 1.0 || v.tangent[3] == -1.0,
                "{:?}",
                v.tangent
            );
        }
        assert_eq!(
            moved,
            after.vertex_count(),
            "every tangent must have moved off the constant"
        );
    }

    /// A plane subdivided `n` times — the shape a toolbar reaches by clicking
    /// Subdivide `n` times, and the fixture the B1 blocker was measured on.
    fn subdivided_plane(n: usize) -> Mesh {
        let mut m = plane(2.0);
        for _ in 0..n {
            let faces: Vec<FaceId> = m.face_ids().collect();
            apply(&mut m, &Op::SubdivideFaces { faces }).expect("subdivides");
        }
        m
    }

    #[test]
    fn the_solver_converges_at_the_sizes_a_toolbar_can_reach() {
        // **B1.** A flat square's exact conformal map is a similarity, so the
        // residual, the fold count and the convergence must ALL be ~0 whatever
        // its density — and at a flat 256 iterations they were not: 1 089
        // vertices gave a 5.7e-2 residual, 361% edge error and TWO FOLDED
        // TRIANGLES on a flat square, and 4 225 vertices folded 338. A solver
        // that folds geometry at a size its own UI produces in five clicks is not
        // slow, it is wrong.
        for n in [4usize, 5, 6] {
            let m = subdivided_plane(n);
            let verts = m.vert_count();
            let out = unwrap(&m).expect("a flat plane unwraps");
            let c = &out.report.charts[0];
            println!(
                "subdivide x{n}: {verts} verts, {} tris, {} iters | residual {:.3e} \
                 convergence {:.3e} flipped {}",
                c.triangles,
                cg_iterations(c.verts.saturating_sub(2)),
                c.residual,
                c.convergence,
                c.flipped
            );
            assert_eq!(
                out.report.flipped, 0,
                "x{n} ({verts} verts) folded {} triangles ON A FLAT SQUARE",
                out.report.flipped
            );
            assert!(
                out.report.worst_residual < 1e-6,
                "x{n} ({verts} verts) residual {:.3e}",
                out.report.worst_residual
            );
            assert!(
                out.report.worst_convergence < 1e-6,
                "x{n} ({verts} verts) convergence {:.3e}",
                out.report.worst_convergence
            );
        }
    }

    #[test]
    fn the_iteration_count_is_an_integer_function_of_the_chart() {
        // The determinism argument, unchanged by making the count adaptive: it is
        // integer arithmetic on a count, not a tolerance test on a float.
        assert_eq!(cg_iterations(0), CG_ITERATIONS, "the floor holds");
        assert_eq!(
            cg_iterations(10),
            CG_ITERATIONS,
            "small charts take the floor"
        );
        assert_eq!(cg_iterations(1_000), 4_000);
        assert_eq!(
            cg_iterations(usize::MAX),
            CG_ITERATION_CAP,
            "and it saturates"
        );
        const { assert!(CG_ITERATION_CAP >= CG_ITERATIONS) };
    }

    #[test]
    fn a_chart_that_cannot_lie_flat_is_counted_rather_than_called_converged() {
        // **M4/M5.** A cylinder cut only at its rim is a tube: conformal, but not
        // a disk, so the solve converges and the map still overlaps itself. The
        // fold count is the only signal that sees it, and the two residuals have
        // to disagree — convergence low, distortion not — or the panel gives the
        // same advice for "the solver stopped early" and "this shape is not
        // developable".
        let m = cylinder(1.0, 2.0, 12);
        let out = unwrap(&m).expect("unwraps");
        println!(
            "cylinder: charts {} residual {:.3e} convergence {:.3e} flipped {}/{}",
            out.report.charts.len(),
            out.report.worst_residual,
            out.report.worst_convergence,
            out.report.flipped,
            out.report.charts.iter().map(|c| c.triangles).sum::<usize>()
        );
        assert!(
            out.report.flipped > 0,
            "a tube cannot lie flat; nothing detected the fold"
        );
        assert!(
            out.report.worst_convergence < 1e-6,
            "…and the SOLVER finished: {:.3e}",
            out.report.worst_convergence
        );
        // The flat control: same code path, nothing folded, both numbers ~0.
        let flat = unwrap(&subdivided_plane(2)).expect("unwraps");
        assert_eq!(flat.report.flipped, 0);
    }

    #[test]
    fn an_empty_mesh_unwraps_to_nothing_rather_than_refusing() {
        let m = Mesh::new();
        let out = unwrap(&m).expect("an empty mesh has no charts to fail on");
        assert_eq!(out.report.charts.len(), 0);
        assert_eq!(out.report.corners, 0);
        assert_eq!(out.op, Op::Unwrap { corners: vec![] });
    }

    // ── Wave D: the LSCM collapse, diagnosed ───────────────────────────────

    /// **THE DIAGNOSIS.** P25 measured 953 of 995 locally-degenerate UV
    /// triangles carrying *healthy 3D area*, called the mechanism unknown, and
    /// carried it. This is the mechanism, and it was in the report all along —
    /// one column to the left.
    ///
    /// A zero-UV-area triangle with healthy 3D area is **the crease of a fold**.
    /// An LSCM solve that overlaps itself has a Jacobian determinant that changes
    /// sign across the chart; by continuity it passes through zero, and the
    /// triangles straddling that locus are flat in UV and perfectly ordinary in
    /// 3D. That is the P25 reading exactly. The fixture it was measured on folds
    /// **1 225** triangles and reports **995** degenerate — one number is not a
    /// mystery beside the other, it is the same event counted twice.
    ///
    /// Two hypotheses this retires, with numbers rather than argument:
    ///
    /// * **ill-conditioned pins.** [`pin_pair`] takes the vertex FARTHEST from
    ///   the lowest id, so `pin_distance` is the chart's 3D diameter by
    ///   construction and cannot be small unless the chart is. Asserted below on
    ///   the very charts that collapse.
    /// * **an unconverged solve.** `convergence` is at machine epsilon on the
    ///   collapsed charts, so the solver finished; what it finished with was an
    ///   overlapping map, which is a genuine least-squares optimum for a chart
    ///   that is not a disk.
    ///
    /// The **third** — the packer's uniform scale putting a small chart under
    /// `f32` resolution — is real arithmetic and is *not* what happened here. It
    /// has its own test below, which measures how far away it is.
    ///
    /// The consequence for the gate: `degenerate_uv <= flipped` is the sharp
    /// statement — *every zero-area triangle is accounted for by a fold* — and it
    /// is what replaces a wholesale 20% fraction. The remedy is unchanged and
    /// already documented (`min_chart_faces`, more seams); what changes is that
    /// an *unexplained* collapse is now a failing test.
    #[test]
    fn every_degenerate_uv_triangle_is_the_crease_of_a_fold() {
        // A tube: conformal, not a disk, so it folds — the crate's own fixture
        // for "the solve converged and the map still overlaps itself".
        let m = cylinder(1.0, 2.0, 12);
        let out = unwrap(&m).expect("unwraps");
        let mut any_folded = false;
        for (i, c) in out.report.charts.iter().enumerate() {
            println!(
                "chart {i}: tris {} flipped {} degenerate_uv {} uv_extent {:.3e} area3 {:.3e} pin {:.3e} conv {:.2e}",
                c.triangles,
                c.flipped,
                c.degenerate_uv,
                c.uv_extent,
                c.area3,
                c.pin_distance,
                c.convergence
            );
            // **THE CLAIM**, per chart: a degeneracy is never unexplained.
            assert!(
                c.degenerate_uv <= c.flipped,
                "chart {i} has {} zero-area UV triangles and only {} folds — that is a collapse the fold count cannot account for, which is the defect this test exists to catch",
                c.degenerate_uv,
                c.flipped
            );
            if c.flipped > 0 {
                any_folded = true;
                // …and it is not the pins, and not the solver.
                assert!(
                    c.pin_distance > 1e-6,
                    "chart {i}'s pins are coincident after all: {}",
                    c.pin_distance
                );
                assert!(
                    c.convergence < 1e-6,
                    "chart {i} did not converge: {}",
                    c.convergence
                );
                assert!(c.area3 > 1e-9, "chart {i} has no 3D area to begin with");
            }
        }
        assert!(
            any_folded,
            "the fixture must actually fold, or this is vacuous"
        );
        assert!(
            out.report.degenerate_uv <= out.report.flipped,
            "{} degenerate of {} folded",
            out.report.degenerate_uv,
            out.report.flipped
        );

        // The control: a chart that does NOT fold has no degeneracies at all,
        // which is what makes the inequality above a statement rather than a
        // tautology satisfied by zero.
        let flat = unwrap(&subdivided_plane(3)).expect("unwraps");
        assert_eq!(flat.report.flipped, 0);
        assert_eq!(
            flat.report.degenerate_uv, 0,
            "a chart with no fold produced a zero-area UV triangle — the diagnosis is wrong"
        );
        assert!(flat.report.charts.iter().all(|c| c.uv_extent > 0.1));
    }

    /// **The collapse fallback, as an invariant**: after [`unwrap`], no chart is
    /// entirely degenerate.
    ///
    /// A chart every one of whose triangles has zero UV area is a chart mapped
    /// onto a line — it carries no texels at all, so nothing downstream can
    /// texture it, and it is the P25 defect. A *few* degenerate triangles inside
    /// a chart with area are fold creases and slivers, which is a different and
    /// much smaller thing.
    ///
    /// Stated over every fixture this module has, including the two that fold.
    #[test]
    fn no_chart_survives_unwrap_mapped_onto_a_line() {
        for (name, m) in [
            ("plane", subdivided_plane(3)),
            ("cylinder", cylinder(1.0, 2.0, 12)),
            ("torus", crate::build::torus(1.0, 0.3, 12, 8)),
            ("cube", cube(2.0)),
        ] {
            let out = unwrap(&m).expect("unwraps");
            for (i, c) in out.report.charts.iter().enumerate() {
                assert!(
                    c.triangles == 0 || c.degenerate_uv < c.triangles,
                    "{name} chart {i}: all {} triangles have zero UV area — the \
                     chart is a segment and the collapse fallback did not fire",
                    c.triangles
                );
            }
        }
    }

    /// The fallback's own arithmetic, on a chart that would collapse: a planar
    /// projection gives **every** triangle area, and it is a pure function of
    /// the mesh.
    #[test]
    fn a_planar_projection_gives_every_triangle_area_and_is_deterministic() {
        let m = subdivided_plane(2);
        let verts: BTreeSet<VertId> = m.vert_ids().collect();
        let faces: Vec<FaceId> = m.face_ids().collect();
        let (tris, _) = triangles(&m, &faces);
        let a = planar_uv(&m, &verts, &tris).expect("a plane has a plane");
        let b = planar_uv(&m, &verts, &tris).expect("twice");
        assert_eq!(a, b, "the projection is not a pure function of the mesh");
        let (area, diag2) = uv_area_and_box(&tris, &a);
        assert!(
            area > diag2 * COLLAPSE_AREA_FRACTION,
            "the projection itself reads as collapsed: {area} vs {diag2}"
        );
        for t in &tris {
            let (p, q, r) = (a[&t[0]], a[&t[1]], a[&t[2]]);
            let d = (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
            assert!(d.abs() > 1e-12, "a projected triangle has no area");
        }
    }

    /// The third hypothesis, measured and set aside: the packer's **uniform
    /// scale** really can put a chart under `f32` resolution, and the ratio it
    /// takes is far past anything a merged-chart scan produces.
    ///
    /// [`pack`] normalizes each chart to its own origin and then scales **every**
    /// chart by one factor, so a chart whose UV extent is orders of magnitude
    /// below the atlas's comes out spanning a handful of `f32` steps. Kept as a
    /// test rather than as a paragraph because it is the arithmetic that says
    /// *how far* — the day someone packs a millimetre detail beside a
    /// forty-metre wall, this is the number they need.
    #[test]
    fn the_packers_uniform_scale_is_a_hazard_at_a_ratio_no_merged_chart_scan_reaches() {
        for ratio in [1e2, 1e4] {
            let mut mesh = Mesh::new();
            grid_patch(&mut mesh, DVec3::ZERO, 40.0, 3);
            grid_patch(&mut mesh, DVec3::new(200.0, 0.0, 0.0), 40.0 / ratio, 3);
            let out = unwrap(&mesh).expect("it unwraps");
            let extents: Vec<f64> = out.report.charts.iter().map(|c| c.uv_extent).collect();
            let degenerate: usize = out.report.charts.iter().map(|c| c.degenerate_uv).sum();
            println!("ratio {ratio:.0e}: uv extents {extents:?} degenerate {degenerate}");
            assert_eq!(
                degenerate, 0,
                "a {ratio:.0e} size ratio already collapses a chart — the packer hazard is closer than the diagnosis says and the ledger is wrong"
            );
        }
    }

    /// A disconnected square patch of `n × n` quads at `origin`, `size` metres
    /// across, added through the op path.
    fn grid_patch(mesh: &mut Mesh, origin: DVec3, size: f64, n: usize) -> Vec<FaceId> {
        let step = size / n as f64;
        let mut ids = Vec::new();
        let mut faces = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let p = origin + DVec3::new(i as f64 * step, 0.0, j as f64 * step);
                let out = crate::ops::apply(
                    mesh,
                    &Op::AddVertex {
                        position: p.to_array(),
                    },
                )
                .expect("a vertex");
                ids.push(out.verts[0]);
            }
        }
        let at = |i: usize, j: usize| ids[j * (n + 1) + i];
        for j in 0..n {
            for i in 0..n {
                let out = crate::ops::apply(
                    mesh,
                    &Op::AddFace {
                        verts: vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)],
                        corners: vec![crate::topo::CornerData::default(); 4],
                        slot: None,
                    },
                )
                .expect("a face");
                faces.push(out.faces[0]);
            }
        }
        faces
    }
}
