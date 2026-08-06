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

/// Conjugate-gradient iterations per chart. **Fixed, not tolerance-driven.**
///
/// A `while residual > tol` loop makes the iteration count depend on a floating
/// comparison, and therefore makes the answer depend on rounding — on a solver
/// whose output is committed content, that is the same class of hazard as a
/// non-portable `sin`. 256 is comfortably past convergence for the chart sizes a
/// panel-scale model produces (the residual on the fixtures here is below 1e-10);
/// where it is not, the number is reported.
pub const CG_ITERATIONS: usize = 256;

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
    /// `‖Ax − b‖ / ‖b‖` after [`CG_ITERATIONS`]. Reported, never hidden.
    pub residual: f64,
}

/// What an unwrap did.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnwrapReport {
    pub charts: Vec<ChartReport>,
    /// The worst per-chart residual — the one number that answers "did the solve
    /// actually converge".
    pub worst_residual: f64,
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
/// Returns `(uv per chart vertex, residual)`. The system is built as `M x = r`
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
) -> Option<(BTreeMap<VertId, DVec2>, f64)> {
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
        for _ in 0..CG_ITERATIONS {
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
        // The reported residual is of the ORIGINAL system `M x = rhs`, not of the
        // normal equations: `‖MᵀM x − Mᵀb‖` can be tiny while the map is badly
        // distorted, and the number an author reads has to be about the thing
        // they can see.
        let ax = mul(&x);
        let num: f64 = ax
            .iter()
            .zip(&rhs)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let den: f64 = rhs.iter().map(|b| b * b).sum::<f64>().sqrt();
        let residual = if den > 1e-300 { num / den } else { num };
        if !x.iter().all(|c| c.is_finite()) {
            return None;
        }
        let mut uv: BTreeMap<VertId, DVec2> = pinned.clone();
        for (i, &v) in free.iter().enumerate() {
            uv.insert(v, DVec2::new(x[2 * i], x[2 * i + 1]));
        }
        return Some((uv, residual));
    }

    Some((pinned, 0.0))
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
        let Some((uv, residual)) = solve_chart(mesh, &tris, &verts, pins) else {
            return Err(OpError::DegenerateChart {
                chart: ci,
                why: "the conformal solve produced a non-finite coordinate",
            });
        };
        reports.push(ChartReport {
            faces: faces.len(),
            verts: verts.len(),
            pinned: pins,
            residual,
        });
        per_chart.push(uv);
    }

    pack(&mut per_chart);

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

    let worst = reports
        .iter()
        .map(|r| r.residual)
        .fold(0.0f64, |a, b| if b > a { b } else { a });
    let report = UnwrapReport {
        corners: corners.len(),
        seams: seam_count(mesh),
        charts: reports,
        worst_residual: worst,
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
    use crate::build::{cube, plane};
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

    #[test]
    fn an_empty_mesh_unwraps_to_nothing_rather_than_refusing() {
        let m = Mesh::new();
        let out = unwrap(&m).expect("an empty mesh has no charts to fail on");
        assert_eq!(out.report.charts.len(), 0);
        assert_eq!(out.report.corners, 0);
        assert_eq!(out.op, Op::Unwrap { corners: vec![] });
    }
}
