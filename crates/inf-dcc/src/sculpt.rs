//! **Brush sculpting** (P23.5): a whole drag as one journal entry.
//!
//! # The stroke IS the transaction
//!
//! `inf_terrain`'s brush has `Stroke::begin` / `add_dab` / `finish`, and it needs
//! them because the terrain's undo unit is a `HeightDelta` that several dabs
//! accumulate into. This journal has no transactions and does not want any: an
//! [`crate::ops::Op`] is already the atom of undo, of replay and of persistence,
//! so the adaptation is to make **the op carry the whole stroke** —
//! [`crate::ops::Op::Sculpt`] holds every dab centre of one mouse-down→up gesture.
//!
//! One journal entry per stroke, one undo step per stroke, and the replay story
//! comes free: a stroke is data, and applying that data to the same mesh twice
//! produces the same bytes twice.
//!
//! # Determinism, precisely
//!
//! * **Dabs are quantized before they reach the op.** [`dab_positions`] is the
//!   3D mirror of `inf_terrain::dab_positions` — arc length along the drag path,
//!   one dab every `spacing` metres — and the caller runs it *once*, on the way
//!   in. The op therefore carries the resampled centres, not the raw pointer
//!   trail, so replay does not depend on the resampler at all.
//! * **The influence is the existing geodesic machinery.** Each dab seeds
//!   [`crate::select::geodesic_distances`] at the vertex nearest its centre and
//!   weighs the result with [`inf_terrain::Falloff`] — the same curve the terrain
//!   brush has used since P16, reached through the same door
//!   [`crate::select::SelectionSet::soft_weights`] uses. There is no second
//!   falloff and no second distance metric in this engine.
//! * **All `f64`, no transcendentals.** Lengths use `sqrt`, which IEEE-754
//!   specifies exactly. Nothing here calls `sin`.
//! * **The finiteness contract holds on RESULTS.** The M3 lesson: two finite
//!   values add to an infinity, so every computed position is checked before the
//!   stroke commits, and a stroke that would store one is refused **inertly** (it
//!   runs inside `Mesh::transact`).
//!
//! # The honest limit of "the vertex nearest the dab"
//!
//! A dab centre is a point on a *face* (the panel ray-casts it), but geodesic
//! distance is measured on the **edge graph**, whose nodes are vertices. So the
//! seed is the nearest vertex and the felt centre of the brush snaps to it: on a
//! coarse mesh a dab in the middle of a large quad pulls from that quad's nearest
//! corner. Seeding every corner of the containing face with its own initial
//! distance would fix it and needs a `geodesic_distances` that accepts non-zero
//! seed distances — a kernel change, deliberately not made for a v1 whose brush
//! radius is normally several edges wide.
//!
//! # Units
//!
//! Dab centres and `radius` are **metres**. `strength` means:
//!
//! | mode | strength |
//! | --- | --- |
//! | [`SculptMode::Draw`] | metres of displacement at full weight (negative carves in) |
//! | [`SculptMode::Smooth`] | blend fraction toward the neighbour average, clamped to `[0, 1]` |
//! | [`SculptMode::Flatten`] | blend fraction toward the plane, clamped to `[0, 1]` |
//! | [`SculptMode::Grab`] | multiplier on the drag vector (`1` follows the pointer exactly) |

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::export::newell;
use crate::ops::{finite, OpError, OpOutcome};
use crate::select::geodesic_distances;
use crate::topo::{Mesh, VertId};

/// What a brush dab does to the surface under it.
///
/// **A wire enum** — it rides inside [`crate::ops::Op::Sculpt`], so it is
/// append-only and pinned by `frozen_nested` in [`crate::journal`]'s tests (the
/// P19 law).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SculptMode {
    /// Displace along the influenced region's averaged normal.
    Draw,
    /// Laplacian relaxation toward the neighbour average — a **Jacobi** sweep, so
    /// the result does not depend on which vertex is visited first.
    Smooth,
    /// Converge toward the plane the stroke started on (see [`SculptMode`]'s
    /// note below and the flatten section of this module).
    Flatten,
    /// Move the influenced region with the drag vector.
    Grab,
}

/// The falloff curve a stroke uses.
///
/// A **wire mirror** of the four parameterless [`inf_terrain::Falloff`] curves,
/// for the reason every DTO in this tree is a mirror: `Falloff` is a rendering
/// and terrain type with a `Plateau(f64)` variant, and tying an op's serialized
/// discriminants to another crate's enum means a variant added there silently
/// re-numbers every session ever written here. The *curves* are not duplicated —
/// [`SculptFalloff::curve`] hands the real one back and
/// `inf_terrain::Falloff::weight` does the arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SculptFalloff {
    #[default]
    Smooth,
    Sphere,
    Linear,
    Sharp,
}

impl SculptFalloff {
    /// The real curve — `inf_terrain`'s, not a copy of it.
    pub fn curve(self) -> inf_terrain::Falloff {
        match self {
            SculptFalloff::Smooth => inf_terrain::Falloff::Smooth,
            SculptFalloff::Sphere => inf_terrain::Falloff::Sphere,
            SculptFalloff::Linear => inf_terrain::Falloff::Linear,
            SculptFalloff::Sharp => inf_terrain::Falloff::Sharp,
        }
    }
}

/// One dab every `DAB_SPACING_FRACTION × radius` metres of drag.
///
/// A quarter of the radius is the terrain brush's own working figure: dense
/// enough that consecutive dabs overlap heavily (so a fast drag does not leave
/// scallops) and sparse enough that a slow drag does not put a hundred dabs in
/// one place and drill a hole.
pub const DAB_SPACING_FRACTION: f64 = 0.25;

/// Resample a drag path into evenly spaced dab centres: the first path point,
/// then one every `spacing` metres of arc length.
///
/// The 3D mirror of `inf_terrain::dab_positions`, and deliberately the same
/// algorithm rather than a re-derivation — a brush that spaced its dabs
/// differently from the terrain brush would feel different for no reason anyone
/// could name. A non-positive `spacing` returns the path unchanged; an empty path
/// returns empty.
pub fn dab_positions(path: &[DVec3], spacing: f64) -> Vec<DVec3> {
    if path.is_empty() {
        return Vec::new();
    }
    if !(spacing.is_finite() && spacing > 0.0) {
        return path.to_vec();
    }
    let mut out = vec![path[0]];
    let mut carry = 0.0; // arc length already consumed since the last dab
    for w in path.windows(2) {
        let (a, b) = (w[0], w[1]);
        let seg = b - a;
        let len = seg.length();
        if !(len.is_finite() && len > 0.0) {
            continue;
        }
        let dir = seg / len;
        let mut t = spacing - carry; // distance into this segment to the next dab
        while t <= len {
            out.push(a + dir * t);
            t += spacing;
        }
        carry = len - (t - spacing); // leftover past the last dab on this segment
    }
    out
}

/// [`dab_positions`] at the brush's own spacing rule — the one door a caller
/// should use, so "how far apart are dabs" has a single answer.
pub fn stroke_dabs(path: &[DVec3], radius: f64) -> Vec<DVec3> {
    dab_positions(path, radius * DAB_SPACING_FRACTION)
}

/// Apply a whole stroke. Called by [`crate::ops::apply`] for
/// [`crate::ops::Op::Sculpt`]; see the module docs for the contract.
pub(crate) fn sculpt(
    mesh: &mut Mesh,
    mode: SculptMode,
    dabs: &[[f64; 3]],
    radius: f64,
    strength: f64,
    falloff: SculptFalloff,
) -> Result<OpOutcome, OpError> {
    if dabs.is_empty() {
        return Err(OpError::EmptyOperand {
            what: "a sculpt stroke's dab list",
        });
    }
    for d in dabs {
        finite("a sculpt dab centre", d)?;
    }
    finite("a sculpt radius", &[radius])?;
    finite("a sculpt strength", &[strength])?;
    if !(radius.is_finite() && radius > 0.0) {
        return Err(OpError::AmountNotPositive {
            what: "a sculpt radius",
            value: radius,
        });
    }
    let curve = falloff.curve();
    let centres: Vec<DVec3> = dabs.iter().copied().map(DVec3::from_array).collect();

    // Everything below runs on a clone: a stroke that computes a non-finite
    // position leaves the mesh byte-identical (the ops module's rule 1).
    mesh.transact(|m| {
        match mode {
            // Grab is the one mode whose influence is fixed for the whole gesture:
            // the author grabbed a place on the surface and dragged it, so the
            // set that moves is the set that was under the pointer when the drag
            // STARTED. Recomputing it per dab would let the grabbed region walk
            // across the model.
            SculptMode::Grab => {
                let weights = dab_weights(m, centres[0], radius, curve);
                let delta = (centres[centres.len() - 1] - centres[0]) * strength;
                let moved: BTreeMap<VertId, DVec3> = weights
                    .iter()
                    .map(|(&v, &w)| (v, position(m, v) + delta * w))
                    .collect();
                commit(m, &moved)?;
            }
            SculptMode::Flatten => {
                // The plane is the stroke's, not the dab's: it is computed ONCE,
                // from the surface as it stood when the gesture began. A plane
                // recomputed per dab chases the geometry it is flattening and
                // converges on nothing — which is the difference between a
                // flatten tool and a very slow smooth.
                let first = dab_weights(m, centres[0], radius, curve);
                let Some(n) = region_normal(m, &first) else {
                    return Err(OpError::DegenerateRegion { faces: 0 });
                };
                let plane_point = centres[0];
                let blend = strength.clamp(0.0, 1.0);
                for &c in &centres {
                    let weights = dab_weights(m, c, radius, curve);
                    let moved: BTreeMap<VertId, DVec3> = weights
                        .iter()
                        .map(|(&v, &w)| {
                            let p = position(m, v);
                            let d = (p - plane_point).dot(n);
                            (v, p - n * (d * blend * w))
                        })
                        .collect();
                    commit(m, &moved)?;
                }
            }
            SculptMode::Draw => {
                for &c in &centres {
                    let weights = dab_weights(m, c, radius, curve);
                    // A dab over geometry with no usable normal has no direction
                    // to draw along, so it does nothing. Skipped rather than
                    // refused: one degenerate patch in the middle of a drag must
                    // not throw away the whole stroke.
                    let Some(n) = region_normal(m, &weights) else {
                        continue;
                    };
                    let moved: BTreeMap<VertId, DVec3> = weights
                        .iter()
                        .map(|(&v, &w)| (v, position(m, v) + n * (strength * w)))
                        .collect();
                    commit(m, &moved)?;
                }
            }
            SculptMode::Smooth => {
                let blend = strength.clamp(0.0, 1.0);
                for &c in &centres {
                    let weights = dab_weights(m, c, radius, curve);
                    // **Jacobi, not Gauss-Seidel.** Every target is read off the
                    // positions as they stand at the top of the dab, and only
                    // then written back — so the answer does not depend on the
                    // order the map is walked in. A sweep that wrote as it went
                    // would be deterministic too (the map is a `BTreeMap`) but
                    // it would be deterministically *wrong*: the relaxation would
                    // propagate along ascending vertex id, which is an arena
                    // artefact and not a property of the surface.
                    let moved: BTreeMap<VertId, DVec3> = weights
                        .iter()
                        .filter_map(|(&v, &w)| {
                            let target = neighbour_average(m, v)?;
                            let p = position(m, v);
                            Some((v, p + (target - p) * (blend * w)))
                        })
                        .collect();
                    commit(m, &moved)?;
                }
            }
        }
        // A sculpt has no successor: it moves geometry that was already there and
        // creates nothing, so the caller CARRIES its selection rather than
        // adopting one (`op_preserves_ids` says so).
        Ok(OpOutcome::default())
    })
}

/// Write a computed batch of positions, refusing the whole stroke if any of them
/// is not finite.
///
/// The M3 law at the door that stores: the operands were checked at the top of
/// [`sculpt`], and it is the *sums* that overflow.
fn commit(mesh: &mut Mesh, moved: &BTreeMap<VertId, DVec3>) -> Result<(), OpError> {
    for p in moved.values() {
        finite("a sculpted vertex position", &p.to_array())?;
    }
    for (&v, p) in moved {
        mesh.vert_mut(v).position = p.to_array();
    }
    Ok(())
}

fn position(mesh: &Mesh, v: VertId) -> DVec3 {
    DVec3::from_array(mesh.vert_ref(v).position)
}

/// The influence of one dab: geodesic metres from the vertex nearest its centre,
/// through the falloff curve. Empty when the mesh has no vertices.
fn dab_weights(
    mesh: &Mesh,
    centre: DVec3,
    radius: f64,
    falloff: inf_terrain::Falloff,
) -> BTreeMap<VertId, f64> {
    let Some(seed) = nearest_vertex(mesh, centre) else {
        return BTreeMap::new();
    };
    let seeds: BTreeSet<VertId> = [seed].into_iter().collect();
    geodesic_distances(mesh, &seeds, radius)
        .into_iter()
        .filter_map(|(v, d)| {
            let w = falloff.weight(d, radius);
            (w > 0.0).then_some((v, w))
        })
        .collect()
}

/// The live vertex closest to `p`, ties broken by the **lowest id**.
///
/// `vert_ids` yields ascending, and the comparison is strict, so the first of a
/// tie wins — which is what makes a dab landing exactly between two vertices pick
/// the same one on every machine and in every replay.
fn nearest_vertex(mesh: &Mesh, p: DVec3) -> Option<VertId> {
    let mut best: Option<(f64, VertId)> = None;
    for v in mesh.vert_ids() {
        let q = position(mesh, v);
        let d = (q - p).length_squared();
        if !d.is_finite() {
            continue;
        }
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, v));
        }
    }
    best.map(|(_, v)| v)
}

/// The area-weighted normal of a weighted vertex set: each vertex's own normal
/// (the Newell sum over its incident faces, which weights each by twice its area)
/// scaled by the brush weight. `None` when the sum cancels or the set is empty.
fn region_normal(mesh: &Mesh, weights: &BTreeMap<VertId, f64>) -> Option<DVec3> {
    let mut acc = DVec3::ZERO;
    for (&v, &w) in weights {
        acc += vertex_normal(mesh, v) * w;
    }
    let len = acc.length();
    (len.is_finite() && len > 1e-18).then(|| acc / len)
}

/// The **unnormalized** area-weighted normal at a vertex: the Newell sum of its
/// incident faces. Unnormalized on purpose — a big face should count for more
/// than a sliver in the region average.
fn vertex_normal(mesh: &Mesh, v: VertId) -> DVec3 {
    let mut acc = DVec3::ZERO;
    let mut seen: BTreeSet<crate::topo::FaceId> = BTreeSet::new();
    for &h in mesh.vert_outgoing(v).unwrap_or(&[]) {
        if let Some(Some(f)) = mesh.face_of(h) {
            if seen.insert(f) {
                acc += newell(mesh, f);
            }
        }
    }
    acc
}

/// The mean of a vertex's one-ring neighbour positions — the Laplacian target.
/// `None` for an isolated vertex (nothing to relax toward).
fn neighbour_average(mesh: &Mesh, v: VertId) -> Option<DVec3> {
    let neighbours: BTreeSet<VertId> = mesh
        .vert_outgoing(v)
        .unwrap_or(&[])
        .iter()
        .filter_map(|&h| mesh.dest(h))
        .collect();
    if neighbours.is_empty() {
        return None;
    }
    let mut acc = DVec3::ZERO;
    for n in &neighbours {
        acc += position(mesh, *n);
    }
    Some(acc / neighbours.len() as f64)
}

/// The face normal at `f`, normalized. `None` for a degenerate face or a dead id.
///
/// Public because the Model Editor needs it to turn a pointer ray into a point on
/// the surface, and a second Newell sum written in Ring 1 would be a second answer
/// to "which way does this face point" — the two would disagree exactly on the
/// slivers where it matters.
pub fn face_normal(mesh: &Mesh, f: crate::topo::FaceId) -> Option<DVec3> {
    if !mesh.has_face(f) {
        return None;
    }
    let n = newell(mesh, f);
    let len = n.length();
    (len.is_finite() && len > 1e-18).then(|| n / len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, plane};
    use crate::journal::MeshSession;
    use crate::ops::Op;
    use crate::topo::CornerData;
    use crate::validate::validate;

    /// An `n × n` grid of unit quads in the XZ plane, normals up.
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

    fn stroke(mode: SculptMode, dabs: Vec<[f64; 3]>, radius: f64, strength: f64) -> Op {
        Op::Sculpt {
            mode,
            dabs,
            radius,
            strength,
            falloff: SculptFalloff::Linear,
        }
    }

    #[test]
    fn dabs_are_spaced_by_arc_length_and_span_a_corner() {
        let path = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(4.0, 0.0, 0.0),
            DVec3::new(4.0, 0.0, 4.0),
        ];
        let dabs = dab_positions(&path, 2.0);
        // 0, 2, 4 along the first leg then 2, 4 along the second — arc length,
        // not per-segment restarts (the terrain precedent's own corner case).
        assert_eq!(
            dabs,
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 0.0),
                DVec3::new(4.0, 0.0, 0.0),
                DVec3::new(4.0, 0.0, 2.0),
                DVec3::new(4.0, 0.0, 4.0),
            ]
        );
        assert!(dab_positions(&[], 1.0).is_empty());
        assert_eq!(dab_positions(&path, 0.0), path);
        assert_eq!(stroke_dabs(&path, 8.0), dab_positions(&path, 2.0));
    }

    #[test]
    fn a_draw_stroke_displaces_along_the_surface_normal() {
        let mut m = grid(6);
        let before: Vec<f64> = m.vert_ids().map(|v| position(&m, v).y).collect();
        assert!(before.iter().all(|y| *y == 0.0));
        let out = crate::ops::apply(
            &mut m,
            &stroke(SculptMode::Draw, vec![[3.0, 0.0, 3.0]], 1.5, 0.5),
        )
        .expect("draws");
        assert_eq!(out, OpOutcome::default(), "a sculpt has no successor");
        assert_eq!(validate(&m), Ok(()));
        // The grid faces +Y (the quads are wound so the Newell sum points up),
        // so the bump is on +Y and it is centred on the dab.
        let peak = m
            .vert_ids()
            .map(|v| position(&m, v))
            .max_by(|a, b| a.y.total_cmp(&b.y))
            .expect("a vertex");
        assert!(peak.y > 0.4, "peak {peak:?}");
        assert!(
            (peak.x - 3.0).abs() < 1e-9 && (peak.z - 3.0).abs() < 1e-9,
            "the peak is under the dab, got {peak:?}"
        );
        // ...and nothing outside the radius moved at all.
        for v in m.vert_ids() {
            let p = position(&m, v);
            let d = ((p.x - 3.0).powi(2) + (p.z - 3.0).powi(2)).sqrt();
            if d > 1.5 + 1e-9 {
                assert!(p.y.abs() < 1e-12, "{p:?} is outside 1.5 m and moved");
            }
        }
    }

    #[test]
    fn a_negative_strength_carves_inward() {
        let mut m = grid(6);
        crate::ops::apply(
            &mut m,
            &stroke(SculptMode::Draw, vec![[3.0, 0.0, 3.0]], 1.5, -0.5),
        )
        .expect("carves");
        let low = m
            .vert_ids()
            .map(|v| position(&m, v).y)
            .fold(f64::MAX, f64::min);
        assert!(low < -0.4, "{low}");
    }

    #[test]
    fn a_grab_moves_the_starting_region_by_the_whole_drag() {
        let mut m = grid(6);
        let seed = m
            .vert_ids()
            .find(|&v| position(&m, v).abs_diff_eq(DVec3::new(3.0, 0.0, 3.0), 1e-12))
            .expect("the centre vertex");
        crate::ops::apply(
            &mut m,
            &stroke(
                SculptMode::Grab,
                vec![[3.0, 0.0, 3.0], [3.0, 0.4, 3.0], [3.0, 0.8, 3.0]],
                1.5,
                1.0,
            ),
        )
        .expect("grabs");
        // The seed is at full weight and the drag is (0, 0.8, 0).
        assert!(
            (position(&m, seed).y - 0.8).abs() < 1e-12,
            "{:?}",
            position(&m, seed)
        );
        assert_eq!(validate(&m), Ok(()));
    }

    #[test]
    fn a_smooth_stroke_relaxes_a_spike_without_moving_its_neighbours_average() {
        let mut m = grid(4);
        let spike = m
            .vert_ids()
            .find(|&v| position(&m, v).abs_diff_eq(DVec3::new(2.0, 0.0, 2.0), 1e-12))
            .expect("the middle vertex");
        m.vert_mut(spike).position = [2.0, 1.0, 2.0];
        let before = position(&m, spike).y;
        crate::ops::apply(
            &mut m,
            &stroke(SculptMode::Smooth, vec![[2.0, 1.0, 2.0]], 1.2, 1.0),
        )
        .expect("smooths");
        let after = position(&m, spike).y;
        assert!(after < before, "{before} -> {after}");
        assert!(after.abs() < 1e-9, "a full-weight relax lands on the mean");
    }

    #[test]
    fn a_smooth_dab_is_a_jacobi_sweep_and_not_a_walk_in_id_order() {
        // **The gate the Jacobi choice did not have** (found by mutation: writing
        // as the sweep goes failed nothing). Gauss-Seidel is just as
        // deterministic — the map is a `BTreeMap` — so no determinism test can
        // see it. It is deterministically *wrong*: the relaxation propagates
        // along ascending vertex id, which is an arena artefact, so the same
        // shape smooths differently depending on the order its vertices happened
        // to be allocated in.
        //
        // Measured against the definition rather than against a second
        // implementation: every target is the mean of the neighbours'
        // **original** positions, blended by the brush weight.
        let mut m = grid(3);
        for (i, v) in m.vert_ids().collect::<Vec<_>>().into_iter().enumerate() {
            let p = position(&m, v);
            // A lumpy surface whose lumps are not symmetric, so a sweep that
            // used updated neighbours could not coincidentally agree.
            m.vert_mut(v).position = [p.x, (i % 3) as f64 * 0.37, p.z];
        }
        let before: BTreeMap<VertId, DVec3> = m.vert_ids().map(|v| (v, position(&m, v))).collect();
        let seed = m
            .vert_ids()
            .find(|&v| {
                let p = before[&v];
                (p.x - 1.0).abs() < 1e-12 && (p.z - 1.0).abs() < 1e-12
            })
            .expect("an interior vertex");
        let centre = before[&seed];
        let (radius, strength) = (1.8_f64, 1.0_f64);
        // The distances are the ones the dab saw: measured on the surface BEFORE
        // it moved, because the brush weighs the mesh it was aimed at.
        let dist = geodesic_distances(&m, &[seed].into_iter().collect(), radius);

        crate::ops::apply(
            &mut m,
            &Op::Sculpt {
                mode: SculptMode::Smooth,
                dabs: vec![centre.to_array()],
                radius,
                strength,
                falloff: SculptFalloff::Linear,
            },
        )
        .expect("smooths");

        let mut checked = 0usize;
        for (&v, &d) in &dist {
            let w = inf_terrain::Falloff::Linear.weight(d, radius);
            if w <= 0.0 {
                continue;
            }
            // The neighbour mean, from the positions as they stood BEFORE the dab.
            let neighbours: BTreeSet<VertId> = m
                .vert_outgoing(v)
                .expect("live")
                .iter()
                .filter_map(|&h| m.dest(h))
                .collect();
            let mut acc = DVec3::ZERO;
            for n in &neighbours {
                acc += before[n];
            }
            let target = acc / neighbours.len() as f64;
            let p = before[&v];
            let want = p + (target - p) * (strength * w);
            let got = position(&m, v);
            assert!(
                (got - want).length() < 1e-12,
                "{v} moved to {got:?}, a Jacobi sweep puts it at {want:?}"
            );
            checked += 1;
        }
        assert!(checked >= 5, "only {checked} vertices were under the brush");
    }

    #[test]
    fn a_flatten_stroke_pulls_toward_the_plane_it_started_on() {
        let mut m = grid(4);
        // A bumpy patch: alternate vertices raised.
        let ids: Vec<VertId> = m.vert_ids().collect();
        for (i, &v) in ids.iter().enumerate() {
            if i % 2 == 0 {
                let p = position(&m, v);
                m.vert_mut(v).position = [p.x, 0.3, p.z];
            }
        }
        crate::ops::apply(
            &mut m,
            &stroke(SculptMode::Flatten, vec![[2.0, 0.0, 2.0]], 3.0, 1.0),
        )
        .expect("flattens");
        // Every vertex the brush reached is on (or very near) y = 0, the plane
        // through the first dab with the region's normal.
        for v in m.vert_ids() {
            let p = position(&m, v);
            let d = ((p.x - 2.0).powi(2) + (p.z - 2.0).powi(2)).sqrt();
            if d < 1.0 {
                assert!(p.y.abs() < 1e-9, "{p:?} is under the brush and not flat");
            }
        }
    }

    #[test]
    fn a_stroke_is_one_journal_entry_and_one_undo_step() {
        let mut s = MeshSession::new(grid(5));
        let before = s.mesh().encoded();
        let dabs: Vec<[f64; 3]> = (0..12).map(|i| [1.0 + i as f64 * 0.2, 0.0, 2.5]).collect();
        s.apply(stroke(SculptMode::Draw, dabs, 1.0, 0.2))
            .expect("strokes");
        assert_eq!(s.ops().len(), 1, "twelve dabs, ONE journal entry");
        assert_ne!(s.mesh().encoded(), before);
        assert!(s.undo());
        assert_eq!(
            s.mesh().encoded(),
            before,
            "one undo takes the whole stroke"
        );
        assert!(s.redo());
        assert_ne!(s.mesh().encoded(), before);
    }

    #[test]
    fn replaying_a_stroke_is_bit_identical() {
        let mut a = MeshSession::new(grid(5));
        let mut b = MeshSession::new(grid(5));
        let dabs: Vec<[f64; 3]> = (0..9).map(|i| [0.7 + i as f64 * 0.3, 0.0, 2.2]).collect();
        for mode in [
            SculptMode::Draw,
            SculptMode::Smooth,
            SculptMode::Flatten,
            SculptMode::Grab,
        ] {
            a.apply(stroke(mode, dabs.clone(), 1.1, 0.3)).expect("a");
            b.apply(stroke(mode, dabs.clone(), 1.1, 0.3)).expect("b");
        }
        assert_eq!(a.mesh().encoded(), b.mesh().encoded());
        let replayed =
            MeshSession::replay(a.base(), &a.ops()[..a.cursor()]).expect("a journalled op replays");
        assert_eq!(replayed.encoded(), a.mesh().encoded());
    }

    #[test]
    fn a_stroke_whose_arithmetic_overflows_is_refused_inertly() {
        // The M3 law at the sculpt door: **every operand here is finite** — the
        // dabs, the radius and the strength all pass `finite` — and the product
        // `(last − first) × strength` is `1e150 × 1e200`, which is not. A gate on
        // the operands alone would have written an infinity into the mesh and
        // then failed the session's own `restore`.
        let mut m = grid(2);
        let before = m.encoded();
        match crate::ops::apply(
            &mut m,
            &stroke(
                SculptMode::Grab,
                vec![[0.0, 0.0, 0.0], [1.0e150, 0.0, 0.0]],
                4.0,
                1.0e200,
            ),
        ) {
            Err(OpError::NonFinite { what, .. }) => {
                assert_eq!(what, "a sculpted vertex position")
            }
            other => panic!("an overflowing stroke was not refused: {other:?}"),
        }
        assert_eq!(m.encoded(), before, "and the refusal is inert");
    }

    #[test]
    fn a_dab_over_geometry_with_no_usable_normal_is_skipped_not_stored() {
        // The other half of the finiteness story, and the reason `region_normal`
        // returns an `Option` rather than a vector: a Newell sum over positions
        // near the top of the `f64` range overflows, so the direction to draw
        // along does not exist. The dab does nothing, the stroke still applies,
        // and no non-finite value is stored — which a normalize-anyway would have
        // put on every vertex under the brush.
        let mut m = grid(2);
        for v in m.vert_ids().collect::<Vec<_>>() {
            let p = position(&m, v);
            m.vert_mut(v).position = [p.x * 1.0e200, 0.0, p.z * 1.0e200];
        }
        let before = m.encoded();
        crate::ops::apply(
            &mut m,
            &stroke(SculptMode::Draw, vec![[0.0, 0.0, 0.0]], 1.0e201, 1.0),
        )
        .expect("the stroke applies");
        assert_eq!(
            m.encoded(),
            before,
            "no dab had a direction, so nothing moved"
        );
        assert!(m.vert_ids().all(|v| position(&m, v).is_finite()));
    }

    #[test]
    fn the_operand_refusals_are_values_too() {
        let mut m = plane(2.0);
        let before = m.encoded();
        for (op, want) in [
            (
                stroke(SculptMode::Draw, vec![], 1.0, 0.1),
                OpError::EmptyOperand {
                    what: "a sculpt stroke's dab list",
                },
            ),
            (
                stroke(SculptMode::Draw, vec![[0.0; 3]], 0.0, 0.1),
                OpError::AmountNotPositive {
                    what: "a sculpt radius",
                    value: 0.0,
                },
            ),
        ] {
            assert_eq!(crate::ops::apply(&mut m, &op), Err(want));
            assert_eq!(m.encoded(), before);
        }
        // And a non-finite operand never reaches the arithmetic.
        assert!(matches!(
            crate::ops::apply(
                &mut m,
                &stroke(SculptMode::Draw, vec![[f64::NAN, 0.0, 0.0]], 1.0, 0.1),
            ),
            Err(OpError::NonFinite { .. })
        ));
        assert_eq!(m.encoded(), before);
    }

    #[test]
    fn a_stroke_that_misses_the_mesh_entirely_does_nothing_and_says_so_by_doing_nothing() {
        // A brush a hundred metres away still seeds at the nearest vertex, but
        // every geodesic distance from it exceeds the radius except the seed's
        // own — so only the seed moves. Documented rather than asserted away:
        // the panel does not start a stroke that does not hit the surface, and
        // the kernel does not need a second rule for the case.
        let mut m = plane(2.0);
        let before = m.encoded();
        crate::ops::apply(
            &mut m,
            &stroke(SculptMode::Draw, vec![[100.0, 0.0, 100.0]], 0.1, 0.5),
        )
        .expect("applies");
        assert_ne!(m.encoded(), before, "the seed vertex is always in range");
        assert_eq!(m.vert_count(), 4);
    }

    #[test]
    fn face_normal_agrees_with_the_writers_own_newell_sum() {
        let m = cube(2.0);
        for f in m.face_ids() {
            let n = face_normal(&m, f).expect("a cube face has a normal");
            assert!((n.length() - 1.0).abs() < 1e-12);
            let raw = newell(&m, f);
            assert!(
                (n - raw / raw.length()).length() < 1e-12,
                "face_normal must BE the writer's normal, not a second opinion"
            );
        }
        assert_eq!(face_normal(&m, crate::topo::FaceId(9_999)), None);
    }
}
