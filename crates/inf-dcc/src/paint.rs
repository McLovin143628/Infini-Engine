//! **The weight-paint brush** (P24.2): one stroke, one journal entry.
//!
//! The sculpt doctrine, on the skin channel. [`crate::sculpt`] made a whole
//! mouse-down→up gesture into a single [`crate::ops::Op`] because the journal has
//! no transactions and does not want any — an op *is* the atom of undo, replay
//! and persistence — and a weight stroke is the same gesture with different
//! arithmetic.
//!
//! # What the op carries, and why it is not `Op::Sculpt`
//!
//! [`crate::ops::Op::AssignWeights`], carrying the **result**: the weights the
//! stroke produced, per vertex. Not the dabs.
//!
//! That is the opposite of `Op::Sculpt`, which carries its dab centres and
//! recomputes the displacement on replay, and the difference is deliberate.
//! A sculpt dab's effect is a function of the *mesh* alone (a position and a
//! falloff), so replaying the dabs reproduces it. A weight dab's effect is a
//! function of the mesh **and of every weight already on it** — painting joint 3
//! to 0.6 rescales whatever else was there — so replaying the dabs against a
//! different starting table gives a different answer. Carrying the values makes a
//! paint stroke a fact, exactly as [`crate::ops::Op::Unwrap`] makes an unwrap
//! one.
//!
//! It also means the brush and the [heat solve](crate::heat) go through one door:
//! both compute a weight table and both journal it as `AssignWeights`, so undo,
//! replay and persistence cannot tell them apart and nothing has to.
//!
//! # Coverage is the MAXIMUM over the dabs, not the sum
//!
//! A slow drag lays more dabs over the same place than a fast one. Summing their
//! influence would make the stroke's strength depend on the author's mouse speed
//! — the classic paint-tool defect — so a vertex's coverage is the largest
//! influence any dab gave it, and `strength` means what it says: at coverage 1,
//! the weight moves by `strength`.
//!
//! # Units
//!
//! `radius` is **geodesic metres**, measured across the surface by
//! [`crate::select::geodesic_distances`] — the same reach the sculpt brush and
//! the soft select use, so a brush does not paint through a thin wall. `strength`
//! is a weight delta in `[0, 1]`.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;

use crate::ops::{finite, Op, OpError};
use crate::sculpt::{SculptFalloff, MAX_STROKE_DABS, MIN_BRUSH_RADIUS_M};
use crate::select::geodesic_distances;
use crate::skin::VertWeights;
use crate::topo::{Mesh, VertId};

/// What a weight dab does to the influence under it.
///
/// **Not a wire enum.** It never reaches a journal — the op carries the stroke's
/// *result*, not its parameters — so unlike [`SculptFalloff`] it is not
/// freeze-pinned and may be extended freely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaintMode {
    /// Raise the joint's share toward 1.
    #[default]
    Add,
    /// Lower it toward 0. The rest of the vertex's influences take up the slack,
    /// in proportion (see [`VertWeights::painted`]).
    Subtract,
    /// Move it toward `strength` itself — "make this exactly 0.5".
    Replace,
    /// Move it toward the average of its one-ring neighbours' shares of the same
    /// joint. The relaxation a hard-edged auto-solve needs at the seams.
    Smooth,
}

/// Turn a finished stroke into the journal entry it became.
///
/// Refusals are values, checked before anything is computed: an unbound mesh, a
/// joint the binding does not have, an empty or oversized dab list, a
/// non-positive radius, a non-finite anything.
///
/// Returns `Ok(None)` when the stroke reached no vertex at all — a dab in the air,
/// or a radius smaller than the gap to the nearest vertex. That is not a refusal
/// (nothing was wrong) and it must not become an empty journal entry, which would
/// be an undo step that does nothing.
pub fn paint_weights(
    mesh: &Mesh,
    joint: u16,
    mode: PaintMode,
    dabs: &[[f64; 3]],
    radius: f64,
    strength: f64,
    falloff: SculptFalloff,
) -> Result<Option<Op>, OpError> {
    let Some(binding) = mesh.skin_binding() else {
        return Err(OpError::NotSkinned);
    };
    if joint as u32 >= binding.joints {
        return Err(OpError::CountOutOfRange {
            what: "a weight-paint joint index",
            value: joint as u32,
            min: 0,
            max: binding.joints.saturating_sub(1),
        });
    }
    if dabs.is_empty() {
        return Err(OpError::EmptyOperand {
            what: "a weight stroke's dab list",
        });
    }
    // The same guard `crate::sculpt` carries and for the same reason: every dab
    // is a Dijkstra, and an op is data that can arrive from a restored session.
    if dabs.len() > MAX_STROKE_DABS {
        return Err(OpError::CountOutOfRange {
            what: "a weight stroke's dab count",
            value: dabs.len().min(u32::MAX as usize) as u32,
            min: 1,
            max: MAX_STROKE_DABS as u32,
        });
    }
    for d in dabs {
        finite("a weight dab centre", d)?;
    }
    finite("a weight brush radius", &[radius])?;
    finite("a weight brush strength", &[strength])?;
    if !(radius.is_finite() && radius >= MIN_BRUSH_RADIUS_M) {
        return Err(OpError::AmountNotPositive {
            what: "a weight brush radius",
            value: radius,
        });
    }

    let curve = falloff.curve();
    // Coverage: the LARGEST influence any dab gave each vertex (see the module
    // docs). `BTreeMap`, so the write order below is vertex order.
    let mut coverage: BTreeMap<VertId, f64> = BTreeMap::new();
    for d in dabs {
        let centre = DVec3::from_array(*d);
        let Some(seed) = nearest_vertex(mesh, centre) else {
            continue;
        };
        // **A dab that is not on the model reaches nothing.**
        //
        // `nearest_vertex` snaps — the sculpt brush's documented rule, and the
        // right one there because a sculpt dab is ray-cast onto the surface
        // before it is ever an op. A weight table is a *file*, and an op is data
        // that can arrive from a restored session, so a dab 500 m away would
        // otherwise snap to whichever vertex happened to be nearest and paint it.
        // Euclidean distance is a lower bound on geodesic distance, so this can
        // only reject dabs the geodesic walk would have rejected anyway.
        if mesh
            .position(seed)
            .is_some_and(|p| (p - centre).length() > radius)
        {
            continue;
        }
        let seeds: BTreeSet<VertId> = [seed].into_iter().collect();
        for (v, dist) in geodesic_distances(mesh, &seeds, radius) {
            let w = curve.weight(dist, radius);
            if w > 0.0 {
                let slot = coverage.entry(v).or_insert(0.0);
                if w > *slot {
                    *slot = w;
                }
            }
        }
    }
    if coverage.is_empty() {
        return Ok(None);
    }

    let strength = strength.clamp(0.0, 1.0);
    let mut weights: Vec<(VertId, VertWeights)> = Vec::with_capacity(coverage.len());
    for (&v, &cover) in &coverage {
        let current = mesh.vert_weights(v).expect("live vertex id");
        let have_weight = current.weight_of(joint);
        let target = match mode {
            PaintMode::Add => have_weight + (strength * cover) as f32,
            PaintMode::Subtract => have_weight - (strength * cover) as f32,
            PaintMode::Replace => have_weight + (strength as f32 - have_weight) * cover as f32,
            PaintMode::Smooth => {
                let avg_weight = neighbour_share(mesh, v, joint).unwrap_or(have_weight);
                have_weight + (avg_weight - have_weight) * (strength * cover) as f32
            }
        };
        let painted = current.painted(joint, target.clamp(0.0, 1.0));
        if painted != current {
            weights.push((v, painted));
        }
    }
    if weights.is_empty() {
        // Every vertex was already where the stroke wanted it — painting 1.0 onto
        // a vertex already at 1.0. No entry, so no undo step that does nothing.
        return Ok(None);
    }
    Ok(Some(Op::AssignWeights { weights }))
}

/// The mean share of `joint` over `v`'s one-ring, or `None` when it has no
/// neighbours.
fn neighbour_share(mesh: &Mesh, v: VertId, joint: u16) -> Option<f32> {
    let out = mesh.vert_outgoing(v)?;
    let mut total_weight = 0.0f32;
    let mut n = 0u32;
    for &h in out {
        let Some(other) = mesh.dest(h) else { continue };
        let Some(w) = mesh.vert_weights(other) else {
            continue;
        };
        total_weight += w.weight_of(joint);
        n += 1;
    }
    (n > 0).then(|| total_weight / n as f32)
}

/// The live vertex closest to `p`, ties broken by the **lowest id** — the same
/// rule (and the same reason) as `crate::sculpt`'s.
fn nearest_vertex(mesh: &Mesh, p: DVec3) -> Option<VertId> {
    let mut best: Option<(f64, VertId)> = None;
    for v in mesh.vert_ids() {
        let Some(q) = mesh.position(v) else { continue };
        let d = (q - p).length_squared();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, v));
        }
    }
    best.map(|(_, v)| v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::plane;
    use crate::ops::apply;
    use crate::skin::SkinBinding;
    use crate::validate::validate;

    /// A subdivided plane, bound to a 4-joint rig, with every vertex on joint 0.
    fn canvas() -> Mesh {
        let mut m = plane(4.0);
        for _ in 0..3 {
            let faces: Vec<_> = m.face_ids().collect();
            apply(&mut m, &Op::SubdivideFaces { faces }).unwrap();
        }
        apply(
            &mut m,
            &Op::BindSkin {
                skeleton: None,
                joints: 4,
            },
        )
        .unwrap();
        m
    }

    fn stroke(m: &mut Mesh, joint: u16, mode: PaintMode, at: [f64; 3], radius: f64, s: f64) {
        let op = paint_weights(m, joint, mode, &[at], radius, s, SculptFalloff::Linear)
            .expect("the stroke is legal")
            .expect("the stroke reached the surface");
        apply(m, &op).expect("the op applies");
        assert_eq!(validate(m), Ok(()));
    }

    #[test]
    fn a_stroke_is_one_op_carrying_its_result() {
        let mut m = canvas();
        let op = paint_weights(
            &m,
            2,
            PaintMode::Add,
            &[[0.0, 0.0, 0.0]],
            1.0,
            0.5,
            SculptFalloff::Linear,
        )
        .unwrap()
        .unwrap();
        // The op is the RESULT, not the dabs — so a replay against a different
        // starting table cannot drift.
        let Op::AssignWeights { weights } = &op else {
            panic!("a paint stroke journals its weight table: {op:?}");
        };
        assert!(!weights.is_empty());
        assert!(
            weights.windows(2).all(|w| w[0].0 < w[1].0),
            "the table must be sorted by vertex, deterministically"
        );
        apply(&mut m, &op).unwrap();
        // Replay from the base reproduces it exactly.
        let mut twin = canvas();
        apply(&mut twin, &op).unwrap();
        assert_eq!(m.canonical(), twin.canonical());
    }

    #[test]
    fn add_raises_the_joint_and_the_rest_take_the_slack() {
        let mut m = canvas();
        let centre = [0.0, 0.0, 0.0];
        stroke(&mut m, 2, PaintMode::Add, centre, 1.2, 0.75);
        let hit = nearest_vertex(&m, DVec3::ZERO).unwrap();
        let w = m.vert_weights(hit).unwrap();
        assert!(w.weight_of(2) > 0.5, "{w:?}");
        assert!(w.is_normalized(), "{w:?}");
        // Joint 0 (which held the whole vertex) gave up exactly the difference.
        assert!(
            (w.weight_of(0) + w.weight_of(2) - 1.0).abs() < 1e-5,
            "{w:?}"
        );

        // Subtract puts it back.
        stroke(&mut m, 2, PaintMode::Subtract, centre, 1.2, 1.0);
        let w = m.vert_weights(hit).unwrap();
        assert_eq!(w.weight_of(2), 0.0, "{w:?}");
        assert!(w.is_normalized(), "{w:?}");
    }

    /// **The falloff is real**: the vertex under the centre gets more than one at
    /// the rim, and one outside the radius gets nothing at all.
    #[test]
    fn the_brush_falls_off_with_geodesic_distance() {
        let mut m = canvas();
        stroke(&mut m, 1, PaintMode::Add, [0.0, 0.0, 0.0], 1.0, 1.0);
        let mut weight_by_distance: Vec<(u64, f32)> = m
            .vert_ids()
            .filter_map(|v| {
                let p = m.position(v)?;
                let w = m.vert_weights(v)?.weight_of(1);
                Some((p.length().to_bits(), w))
            })
            .collect();
        weight_by_distance.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        let near = weight_by_distance
            .first()
            .expect("a vertex near the centre")
            .1;
        let far = weight_by_distance.last().expect("a vertex at the corner").1;
        assert!(
            near > 0.9,
            "the centre should be nearly all joint 1: {near}"
        );
        assert_eq!(far, 0.0, "a corner 2.8 m away is outside a 1 m brush");
        // …and the count in between is non-trivial, so "falls off" is a curve and
        // not a step at the boundary.
        let partial = weight_by_distance
            .iter()
            .filter(|(_, w)| *w > 0.0 && *w < 0.9)
            .count();
        assert!(
            partial >= 3,
            "only {partial} vertices are partially painted"
        );
    }

    /// Coverage is the MAX over dabs, so a slow drag paints the same as a fast
    /// one over the same ground.
    #[test]
    fn a_repeated_dab_paints_no_harder_than_a_single_one() {
        let one = paint_weights(
            &canvas(),
            1,
            PaintMode::Add,
            &[[0.0, 0.0, 0.0]],
            1.0,
            0.5,
            SculptFalloff::Linear,
        )
        .unwrap()
        .unwrap();
        let many = paint_weights(
            &canvas(),
            1,
            PaintMode::Add,
            &[[0.0, 0.0, 0.0]; 8],
            1.0,
            0.5,
            SculptFalloff::Linear,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            one, many,
            "the stroke's effect must not depend on dab count"
        );
    }

    #[test]
    fn smooth_pulls_a_spike_toward_its_neighbours() {
        let mut m = canvas();
        // A hard spike: one vertex fully on joint 3, its neighbours on joint 0.
        let spike = nearest_vertex(&m, DVec3::ZERO).unwrap();
        apply(
            &mut m,
            &Op::AssignWeights {
                weights: vec![(spike, VertWeights::from_pairs(&[(3, 1.0)]))],
            },
        )
        .unwrap();
        let before = m.vert_weights(spike).unwrap().weight_of(3);
        stroke(&mut m, 3, PaintMode::Smooth, [0.0, 0.0, 0.0], 0.8, 1.0);
        let after = m.vert_weights(spike).unwrap().weight_of(3);
        assert!(
            after < before,
            "smooth must relax the spike ({before} → {after})"
        );
        // …and it did not simply erase it: the neighbours came UP.
        let neighbour_up = neighbour_share(&m, spike, 3).unwrap();
        assert!(neighbour_up > 0.0, "the neighbours took some of the spike");
    }

    #[test]
    fn every_refusal_is_a_value() {
        let m = canvas();
        let dab = [[0.0, 0.0, 0.0]];
        // Not skinned.
        let mut rigid = plane(2.0);
        assert_eq!(
            paint_weights(
                &rigid,
                0,
                PaintMode::Add,
                &dab,
                1.0,
                0.5,
                SculptFalloff::Linear
            ),
            Err(OpError::NotSkinned)
        );
        rigid.set_skin_binding(Some(SkinBinding {
            skeleton: None,
            joints: 2,
        }));
        // A joint the binding does not have.
        assert_eq!(
            paint_weights(
                &rigid,
                5,
                PaintMode::Add,
                &dab,
                1.0,
                0.5,
                SculptFalloff::Linear
            ),
            Err(OpError::CountOutOfRange {
                what: "a weight-paint joint index",
                value: 5,
                min: 0,
                max: 1,
            })
        );
        // No dabs, too many dabs, a radius below the floor, a NaN.
        assert!(matches!(
            paint_weights(&m, 0, PaintMode::Add, &[], 1.0, 0.5, SculptFalloff::Linear),
            Err(OpError::EmptyOperand { .. })
        ));
        let flood = vec![[0.0, 0.0, 0.0]; MAX_STROKE_DABS + 1];
        assert!(matches!(
            paint_weights(
                &m,
                0,
                PaintMode::Add,
                &flood,
                1.0,
                0.5,
                SculptFalloff::Linear
            ),
            Err(OpError::CountOutOfRange { .. })
        ));
        assert!(matches!(
            paint_weights(&m, 0, PaintMode::Add, &dab, 0.0, 0.5, SculptFalloff::Linear),
            Err(OpError::AmountNotPositive { .. })
        ));
        assert!(matches!(
            paint_weights(
                &m,
                0,
                PaintMode::Add,
                &[[f64::NAN, 0.0, 0.0]],
                1.0,
                0.5,
                SculptFalloff::Linear
            ),
            Err(OpError::NonFinite { .. })
        ));
    }

    /// A stroke that changes nothing produces **no op** — not an empty one, which
    /// would be an undo step that does nothing.
    #[test]
    fn a_stroke_that_changes_nothing_journals_nothing() {
        let m = canvas();
        // Joint 0 already holds every vertex; adding more cannot move it.
        assert_eq!(
            paint_weights(
                &m,
                0,
                PaintMode::Add,
                &[[0.0, 0.0, 0.0]],
                1.0,
                1.0,
                SculptFalloff::Linear
            ),
            Ok(None)
        );
        // …and a dab far off the surface reaches nothing.
        assert_eq!(
            paint_weights(
                &m,
                1,
                PaintMode::Add,
                &[[500.0, 0.0, 0.0]],
                0.01,
                1.0,
                SculptFalloff::Linear
            ),
            Ok(None)
        );
    }

    #[test]
    fn a_stroke_preserves_ids_so_a_selection_survives_it() {
        let m = canvas();
        let op = paint_weights(
            &m,
            1,
            PaintMode::Add,
            &[[0.0, 0.0, 0.0]],
            1.0,
            0.5,
            SculptFalloff::Linear,
        )
        .unwrap()
        .unwrap();
        assert!(
            crate::select::op_preserves_ids(&op),
            "a weight stroke must keep the selection the author painted with"
        );
    }
}
