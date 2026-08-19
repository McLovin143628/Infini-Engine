//! **Pivot-relative vertex transforms** (P23.5): the rotate and scale halves of
//! the component gizmo.
//!
//! [`crate::ops::Op::TranslateVerts`] already existed, because a numeric nudge
//! only needs a delta. A dragged gizmo needs the other two, and both of them need
//! a **pivot** — a rotation without one is not a transform of a selection, it is a
//! transform of the world origin, and scaling a face about `(0,0,0)` moves it
//! across the model.
//!
//! # Weighting is the CALLER's job, and that is deliberate
//!
//! Soft select blends a *fraction* of the transform into each vertex. Rather than
//! give these ops a weight table — which would put a `BTreeMap<VertId, f64>` on
//! the wire and make the op's size proportional to the neighbourhood — the caller
//! groups vertices by weight and emits **one op per distinct weight**, with the
//! angle / factor already scaled. That is exactly what `SoftTranslate` has done
//! since P23.4, so the three tools share one shape, and `inf_editor_core::dcc`'s
//! `transform_ops` is the single function all of them go through.
//!
//! # Rodrigues, through `inf_math`'s portable trig
//!
//! The P14 LAW: `sin`/`cos` from `std` are not bit-identical across targets, and a
//! rotated vertex is **committed content** — it lands in a `.inf_mesh` and in a
//! journal two machines are claimed to replay identically. So the rotation is
//! built by hand from [`inf_math::psin64`] / [`inf_math::pcos64`] rather than from
//! `DQuat::from_axis_angle`, whose `sin_cos` no source grep in this crate would
//! ever see. `crate::journal`'s determinism gate bans the constructors that would
//! reintroduce it.
//!
//! # Units
//!
//! Pivot and positions are **metres**; `radians` is radians; `factor` is a
//! dimensionless per-axis multiplier (`1` = unchanged, negative mirrors).

use std::collections::BTreeMap;

use glam::DVec3;
use inf_math::{pcos64, psin64};

use crate::ops::{finite, OpError, OpOutcome};
use crate::topo::{Mesh, VertId};

/// The largest rotation angle this kernel accepts, radians: **2^52**.
///
/// Not a taste limit. Above `2^52` consecutive `f64`s are more than one radian
/// apart, so the stored value cannot represent the angle its author meant and
/// reducing it modulo 2π returns a number with no information in it. The
/// alternative to refusing is what shipped before the P23.5 audit: `psin64` and
/// `pcos64` both fall to exactly zero past ~2e16 and Rodrigues becomes an axis
/// projection — finite, accepted, and geometry-destroying.
pub const MAX_ROTATION_RADIANS: f64 = 4_503_599_627_370_496.0;

/// Rotate vertices about `pivot` around `axis` by `radians`.
pub(crate) fn rotate_verts(
    mesh: &mut Mesh,
    verts: &[VertId],
    pivot: [f64; 3],
    axis: [f64; 3],
    radians: f64,
) -> Result<OpOutcome, OpError> {
    let rot = Rotation::new(pivot, axis, radians)?;
    write_all(mesh, verts, |p| rot.apply(p))
}

/// **The point map [`crate::ops::Op::RotateVerts`] applies**, prepared once.
///
/// Public since Wave D, and for a reason worth stating: a caller that has to
/// *pre-compute* soft-weighted results — `inf_editor_core::dcc`'s
/// `transform_ops`, which now collapses a whole proportional drag into one
/// [`crate::ops::Op::MoveVerts`] — needs the same rotation the op would have
/// applied. Reimplementing Rodrigues one ring up would put a **second** copy of
/// the portability law's trig in the tree, in a crate the kernel's own
/// determinism gate does not read. One door, and the op goes through it too.
///
/// The trig is evaluated once in [`Rotation::new`], not per point, so this is
/// also strictly cheaper than the shape it replaced.
#[derive(Debug, Clone, Copy)]
pub struct Rotation {
    pivot: DVec3,
    /// The **unit** axis.
    axis: DVec3,
    /// `sin θ` and `cos θ`, renormalized — see the construction.
    s: f64,
    c: f64,
}

impl Rotation {
    /// Prepare a rotation, refusing a zero axis or an angle past
    /// [`MAX_ROTATION_RADIANS`] exactly as the op does.
    pub fn new(pivot: [f64; 3], axis: [f64; 3], radians: f64) -> Result<Self, OpError> {
        finite("a rotation pivot", &pivot)?;
        finite("a rotation axis", &axis)?;
        finite("a rotation angle", &[radians])?;
        let k = DVec3::from_array(axis);
        let len = k.length();
        if !(len.is_finite() && len > 1e-12) {
            return Err(OpError::ZeroAxis { axis });
        }
        let k = k / len;
        let pivot = DVec3::from_array(pivot);
        // **The angle is bounded before the polynomials see it** (P23.5 audit).
        //
        // The failure this closes: past ~2e16 `psin64` and `pcos64` both return
        // exactly zero, and Rodrigues with `s = c = 0` is `pivot + k·(k·r)` — an axis
        // **projection**. Every coordinate stays finite, so it was accepted, and
        // `validate` passed because it audits topology and not geometry: θ = 1e100
        // returned a quad whose four vertices were collinear. `Op::RotateVerts` is
        // public API and rides in a session save, so a mistyped exponent in a numeric
        // box was a data-loss bug.
        //
        // The `is_finite` conjunct is not decoration: a NaN compares false against
        // every bound and would otherwise be admitted by an `<=` test.
        if !(radians.is_finite() && radians.abs() <= MAX_ROTATION_RADIANS) {
            return Err(OpError::AngleOutOfRange {
                radians,
                limit: MAX_ROTATION_RADIANS,
            });
        }

        // # The audit prescribed a `mod 2π` fold here, and the measurement says no
        //
        // The prescription was to reduce before calling the polynomials, on the
        // grounds that it closes both the collapse and a ~5.7e-10 accuracy droop at
        // θ = 1e6. Measured across `[6.5, 4.5e15]`, it closes **neither**:
        //
        // | θ | angle error raw | reduced |
        // | --- | --- | --- |
        // | 1e6 | 5.99e-10 | 6.16e-10 |
        // | 1e12 | 3.37e-5 | 4.98e-5 |
        // | 1e15 | 6.87e-2 | 1.53e-2 |
        //
        // The fold moves the error around and does not remove it, because at those
        // magnitudes the error is the **input's own resolution** — consecutive `f64`s
        // at 1e12 are 1.2e-4 radians apart — and no reduction recovers a digit that
        // was never stored. What the fold *does* improve is `|s² + c²| − 1` before
        // normalization (2.5e-2 → 1.7e-10 at 1e15), and that is precisely the
        // quantity the renormalization below already fixes.
        //
        // So it is not here. The collapse is closed by the bound above and by the
        // refusal below, both of which are gated; a third mechanism that no
        // measurement supports and no test can distinguish is code that will be
        // maintained for ever on the strength of a comment.
        let theta = radians;
        // Rodrigues: r·cos θ + (k × r)·sin θ + k·(k · r)(1 − cos θ).
        //
        // **The pair is renormalized, and that is the difference between "a rotation
        // by roughly θ" and "roughly a rotation".** `psin64` is a degree-11 Taylor
        // polynomial with an endpoint error near 5.7e-8, so `s² + c²` is not exactly
        // 1 and the raw matrix is a rotation *composed with a slight scale*: a
        // quarter-turn of a 1 m vertex came back 56 nm short, and repeated drags
        // would shrink a selection with nothing to tell the author why.
        //
        // Dividing by `√(s² + c²)` — `sqrt` is exactly specified by IEEE-754 and
        // therefore still bit-portable — makes the transform an exact rotation to
        // `f64` rounding. What remains is an **angle** error of at most ~6e-8 rad
        // (≈ 3.4e-6 degrees), which is the honest cost of the portability law and is
        // four orders below anything a modeller can express.
        let (s, c) = {
            let (s, c) = (psin64(theta), pcos64(theta));
            let n = (s * s + c * c).sqrt();
            if n.is_finite() && n > 1e-12 {
                (s / n, c / n)
            } else {
                // **Unreachable below the bound, and kept anyway.** Swept across
                // fifteen decades by `no_angle_inside_the_limit_degenerates_the_
                // sine_cosine_pair`, the worst `|s, c|` inside the limit is 0.968 —
                // so the renormalization above always has something to work with.
                //
                // It is a refusal rather than a fallthrough because the fallthrough is
                // exactly how the collapse got in: a degenerate pair used to be
                // *accepted*, and Rodrigues with `s = c = 0` is a projection onto the
                // axis, not a rotation. Two mechanisms hold one failure, and the
                // mutation table records that each closes it alone.
                return Err(OpError::AngleOutOfRange {
                    radians,
                    limit: MAX_ROTATION_RADIANS,
                });
            }
        };
        Ok(Self {
            pivot,
            axis: k,
            s,
            c,
        })
    }

    /// Rodrigues: `r·cos θ + (k × r)·sin θ + k·(k · r)(1 − cos θ)`, about the
    /// pivot.
    pub fn apply(&self, p: DVec3) -> DVec3 {
        let (k, s, c) = (self.axis, self.s, self.c);
        let r = p - self.pivot;
        self.pivot + r * c + k.cross(r) * s + k * (k.dot(r) * (1.0 - c))
    }
}

/// The point map [`crate::ops::Op::ScaleVerts`] applies — the companion to
/// [`Rotation::apply`], public for the same one-door reason.
pub fn scale_point(p: DVec3, pivot: DVec3, factor: DVec3) -> DVec3 {
    pivot + (p - pivot) * factor
}

/// Scale vertices about `pivot` by a per-axis `factor`.
///
/// A zero or negative component is **allowed**: flattening a selection onto a
/// plane and mirroring it are both things an author asks a scale handle for, and
/// refusing them would mean the gizmo could not do what dragging past the pivot
/// visibly does.
pub(crate) fn scale_verts(
    mesh: &mut Mesh,
    verts: &[VertId],
    pivot: [f64; 3],
    factor: [f64; 3],
) -> Result<OpOutcome, OpError> {
    finite("a scale pivot", &pivot)?;
    finite("a scale factor", &factor)?;
    let pivot = DVec3::from_array(pivot);
    let f = DVec3::from_array(factor);
    write_all(mesh, verts, |p| scale_point(p, pivot, f))
}

/// Check every id, compute every new position, check every RESULT, then write.
///
/// The M3 law and the `TranslateVerts` precedent in one helper: a refusal is
/// inert without needing a `Mesh::transact`, because nothing is written until the
/// last check has passed.
fn write_all(
    mesh: &mut Mesh,
    verts: &[VertId],
    f: impl Fn(DVec3) -> DVec3,
) -> Result<OpOutcome, OpError> {
    for &v in verts {
        if !mesh.has_vert(v) {
            return Err(OpError::NoSuchVert(v));
        }
    }
    // A `BTreeMap` rather than a `Vec`: a caller that names the same vertex twice
    // must not transform it twice, and the map makes that impossible instead of
    // making it the caller's problem. (`TranslateVerts` adds a delta and is
    // idempotent-adjacent; a rotate applied twice is a different angle.)
    let mut moved: BTreeMap<VertId, [f64; 3]> = BTreeMap::new();
    for &v in verts {
        let p = DVec3::from_array(mesh.vert_ref(v).position);
        moved.insert(v, f(p).to_array());
    }
    for p in moved.values() {
        finite("a transformed vertex position", p)?;
    }
    for (v, p) in moved {
        mesh.vert_mut(v).position = p;
    }
    Ok(OpOutcome::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, plane};
    use crate::ops::{apply, Op};
    use crate::validate::validate;

    fn positions(m: &Mesh) -> Vec<DVec3> {
        m.vert_ids()
            .map(|v| DVec3::from_array(m.vert_ref(v).position))
            .collect()
    }

    #[test]
    fn a_quarter_turn_about_y_maps_x_onto_minus_z() {
        // Right-handed: +Y rotation takes +X to −Z. Checked against a hand
        // answer rather than against a second implementation.
        let mut m = Mesh::new();
        let v = m.alloc_vert([1.0, 0.0, 0.0]);
        apply(
            &mut m,
            &Op::RotateVerts {
                verts: vec![v],
                pivot: [0.0; 3],
                axis: [0.0, 1.0, 0.0],
                radians: std::f64::consts::FRAC_PI_2,
            },
        )
        .expect("rotates");
        let p = DVec3::from_array(m.vert_ref(v).position);
        // 1e-6, not 1e-12, and the number is the CONTRACT: `psin64` is accurate
        // to ~6e-8 rad, so a 1 m lever arm lands within ~60 nm of the exact
        // answer. Asserting f64 equality here would be asserting that the
        // portability law was not obeyed.
        assert!((p - DVec3::new(0.0, 0.0, -1.0)).length() < 1e-6, "{p:?}");
    }

    #[test]
    fn a_rotation_is_about_the_pivot_and_not_the_origin() {
        let mut m = Mesh::new();
        let v = m.alloc_vert([3.0, 0.0, 0.0]);
        apply(
            &mut m,
            &Op::RotateVerts {
                verts: vec![v],
                pivot: [2.0, 0.0, 0.0],
                axis: [0.0, 1.0, 0.0],
                radians: std::f64::consts::PI,
            },
        )
        .expect("rotates");
        let p = DVec3::from_array(m.vert_ref(v).position);
        assert!((p - DVec3::new(1.0, 0.0, 0.0)).length() < 1e-6, "{p:?}");
    }

    #[test]
    fn a_rotation_preserves_lengths_from_the_pivot() {
        let mut m = cube(2.0);
        let pivot = DVec3::new(0.3, -0.2, 0.9);
        let before: Vec<f64> = positions(&m)
            .iter()
            .map(|p| (*p - pivot).length())
            .collect();
        let all: Vec<VertId> = m.vert_ids().collect();
        apply(
            &mut m,
            &Op::RotateVerts {
                verts: all,
                pivot: pivot.to_array(),
                axis: [0.3, 0.9, -0.4],
                radians: 0.87,
            },
        )
        .expect("rotates");
        let after: Vec<f64> = positions(&m)
            .iter()
            .map(|p| (*p - pivot).length())
            .collect();
        // **Exactly** here, not to 1e-6: renormalizing the sine/cosine pair is
        // what makes the transform an isometry, so a length error above f64
        // rounding means that renormalization was dropped — which is precisely
        // the defect the first version of this shipped.
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-14 * a.max(1.0), "{a} != {b}");
        }
        assert_eq!(validate(&m), Ok(()));
    }

    #[test]
    fn a_scale_is_per_axis_and_about_the_pivot() {
        let mut m = Mesh::new();
        let v = m.alloc_vert([2.0, 4.0, 6.0]);
        apply(
            &mut m,
            &Op::ScaleVerts {
                verts: vec![v],
                pivot: [1.0, 1.0, 1.0],
                factor: [2.0, 0.5, -1.0],
            },
        )
        .expect("scales");
        let p = DVec3::from_array(m.vert_ref(v).position);
        assert!((p - DVec3::new(3.0, 2.5, -4.0)).length() < 1e-12, "{p:?}");
    }

    #[test]
    fn a_zero_factor_flattens_rather_than_refusing() {
        // Documented behaviour: dragging a scale handle onto the pivot is how an
        // author flattens a selection, and a refusal there would make the visible
        // result of the drag unreachable.
        let mut m = plane(2.0);
        let all: Vec<VertId> = m.vert_ids().collect();
        apply(
            &mut m,
            &Op::ScaleVerts {
                verts: all,
                pivot: [0.0; 3],
                factor: [1.0, 1.0, 0.0],
            },
        )
        .expect("flattens");
        assert!(positions(&m).iter().all(|p| p.z == 0.0));
        assert_eq!(validate(&m), Ok(()), "a flat quad is still a valid mesh");
    }

    #[test]
    fn a_zero_axis_is_refused_as_a_value_and_the_refusal_is_inert() {
        let mut m = plane(2.0);
        let before = m.encoded();
        let all: Vec<VertId> = m.vert_ids().collect();
        assert_eq!(
            apply(
                &mut m,
                &Op::RotateVerts {
                    verts: all,
                    pivot: [0.0; 3],
                    axis: [0.0; 3],
                    radians: 1.0,
                },
            ),
            Err(OpError::ZeroAxis { axis: [0.0; 3] })
        );
        assert_eq!(m.encoded(), before);
    }

    #[test]
    fn a_huge_angle_never_becomes_an_axis_projection() {
        // **The collapse.** Past ~2e16 both `psin64` and `pcos64` return exactly
        // zero, and Rodrigues with `s = c = 0` is `pivot + k·(k·r)` — a projection
        // onto the axis. Every coordinate stays finite so it was ACCEPTED, and
        // `validate` passed because it audits topology and not geometry: a quad
        // came back collinear from a public op that rides in a session save.
        //
        // The shape of the check is "the quad still has area", because that is
        // what a projection destroys and what no existing gate looked at.
        let area = |m: &Mesh, v: &[VertId]| {
            let p: Vec<DVec3> = v
                .iter()
                .map(|&x| DVec3::from_array(m.vert_ref(x).position))
                .collect();
            (p[1] - p[0]).cross(p[2] - p[0]).length()
        };
        for theta in [1.0e17_f64, 1.0e100, -1.0e30, f64::MAX] {
            let mut m = plane(2.0);
            let v: Vec<VertId> = m.vert_ids().collect();
            let before = area(&m, &v);
            let err = apply(
                &mut m,
                &Op::RotateVerts {
                    verts: v.clone(),
                    pivot: [0.0; 3],
                    axis: [0.0, 1.0, 0.0],
                    radians: theta,
                },
            )
            .expect_err("an angle past the limit must refuse");
            assert!(
                matches!(err, OpError::AngleOutOfRange { .. }),
                "{theta}: {err:?}"
            );
            assert!(
                (area(&m, &v) - before).abs() < 1e-12,
                "{theta} flattened the quad: {before} -> {}",
                area(&m, &v)
            );
        }
    }

    #[test]
    fn a_large_but_legal_angle_is_reduced_and_stays_accurate() {
        // The other half: angles inside the limit are *reduced* rather than
        // refused, so `1e6 + π/2` is still a rotation and still lands where it
        // should. Before the reduction the accuracy drooped to ~5.7e-10 here,
        // because `pcos64(x)` is `psin64(x + π/2)` and both the internal range
        // reduction and that addition lose precision on a large argument.
        //
        // Measured by composition, since `atan2` is on this crate's ban list:
        // rotating by θ and then by −θ must return the vertex exactly where it
        // started.
        for theta in [
            std::f64::consts::TAU * 1_000.0 + 0.7,
            1.0e6,
            1.0e12,
            -1.0e9,
            MAX_ROTATION_RADIANS,
        ] {
            let mut m = Mesh::new();
            let v = m.alloc_vert([1.0, 0.0, 0.0]);
            for sign in [1.0, -1.0] {
                apply(
                    &mut m,
                    &Op::RotateVerts {
                        verts: vec![v],
                        pivot: [0.0; 3],
                        axis: [0.0, 1.0, 0.0],
                        radians: theta * sign,
                    },
                )
                .unwrap_or_else(|e| panic!("{theta} refused: {e}"));
                let p = DVec3::from_array(m.vert_ref(v).position);
                assert!(
                    (p.length() - 1.0).abs() < 1e-12,
                    "{theta}: the radius moved to {}",
                    p.length()
                );
            }
            // **The round trip closes to the resolution of the angle itself.**
            // At θ = 1e12 consecutive `f64`s are 1.2e-4 radians apart, so the
            // *input* does not specify the rotation more precisely than that and
            // no reduction can recover what was never there. The tolerance is
            // therefore the angle's own ulp, not a constant — and where the ulp
            // is itself of radian scale the check is skipped rather than passed
            // vacuously, because a 4-radian tolerance on a unit circle asserts
            // nothing at all (the P19 law). The isometry above holds at every
            // magnitude, which is the invariant that matters.
            let ulp = theta.abs().max(1.0) * f64::EPSILON;
            let tolerance = 40.0 * ulp + 1e-6;
            if tolerance < 0.1 {
                let p = DVec3::from_array(m.vert_ref(v).position);
                assert!(
                    (p - DVec3::X).length() < tolerance,
                    "{theta} then -{theta} landed at {p:?} (tolerance {tolerance:.2e})"
                );
            }
        }
    }

    #[test]
    fn no_angle_inside_the_limit_degenerates_the_sine_cosine_pair() {
        // **What makes the bound sufficient**, swept rather than assumed. The
        // collapse is `s = c = 0`, which the renormalization cannot rescue —
        // dividing by zero is what produced the axis projection. The bound is only
        // the right bound if the pair stays away from the origin everywhere below
        // it, so that is measured across fifteen decades and both signs, not
        // argued from where the polynomial "should" be well behaved.
        let mut worst = f64::MAX;
        let mut worst_at = 0.0f64;
        let mut theta = 1.0e-3_f64;
        while theta <= MAX_ROTATION_RADIANS {
            for signed in [theta, -theta, theta * 1.7, theta * 3.3] {
                if signed.abs() > MAX_ROTATION_RADIANS {
                    continue;
                }
                let (s, c) = (psin64(signed), pcos64(signed));
                let n = (s * s + c * c).sqrt();
                assert!(
                    n.is_finite() && n > 1e-12,
                    "the pair degenerated at {signed}: |s,c| = {n}"
                );
                if n < worst {
                    worst = n;
                    worst_at = signed;
                }
            }
            theta *= 1.7;
        }
        println!("worst |s,c| inside the limit: {worst:.6} at theta = {worst_at:.3e}");
        assert!(worst > 0.9, "the pair got within {worst} of degenerate");

        // And just past the limit it really does collapse — which is the reason
        // the limit exists and not a hypothetical.
        let (s, c) = (psin64(1.0e17), pcos64(1.0e17));
        assert!(
            (s * s + c * c).sqrt() <= 1e-12,
            "the collapse this bound exists for did not reproduce: ({s}, {c})"
        );
    }

    #[test]
    fn an_angle_inside_one_turn_is_left_bit_identical() {
        // The reduction is for the range that was broken. Everything an author
        // can express goes through untouched — asserted on the BITS, because
        // "close enough" is what a silent behaviour change looks like.
        for theta in [0.0_f64, 0.7, -0.5, std::f64::consts::PI, -3.0, 6.2] {
            let mut a = Mesh::new();
            let va = a.alloc_vert([1.0, 2.0, 3.0]);
            apply(
                &mut a,
                &Op::RotateVerts {
                    verts: vec![va],
                    pivot: [0.25, 0.0, -0.5],
                    axis: [0.3, 0.9, -0.4],
                    radians: theta,
                },
            )
            .expect("rotates");
            // The arithmetic, spelled out.
            let (s0, c0) = (inf_math::psin64(theta), inf_math::pcos64(theta));
            let n = (s0 * s0 + c0 * c0).sqrt();
            let (s0, c0) = (s0 / n, c0 / n);
            let k = DVec3::new(0.3, 0.9, -0.4).normalize();
            let pivot = DVec3::new(0.25, 0.0, -0.5);
            let r = DVec3::new(1.0, 2.0, 3.0) - pivot;
            let want = pivot + r * c0 + k.cross(r) * s0 + k * (k.dot(r) * (1.0 - c0));
            assert_eq!(
                a.vert_ref(va).position,
                want.to_array(),
                "the arithmetic moved a bit at theta = {theta}"
            );
        }
    }

    #[test]
    fn a_dead_id_refuses_the_whole_batch_before_anything_moves() {
        let mut m = plane(2.0);
        let before = m.encoded();
        let mut verts: Vec<VertId> = m.vert_ids().collect();
        verts.push(VertId(9_999));
        for op in [
            Op::RotateVerts {
                verts: verts.clone(),
                pivot: [0.0; 3],
                axis: [0.0, 1.0, 0.0],
                radians: 1.0,
            },
            Op::ScaleVerts {
                verts: verts.clone(),
                pivot: [0.0; 3],
                factor: [2.0; 3],
            },
        ] {
            assert_eq!(apply(&mut m, &op), Err(OpError::NoSuchVert(VertId(9_999))));
            assert_eq!(m.encoded(), before);
        }
    }

    #[test]
    fn a_transform_that_computes_a_non_finite_position_is_refused() {
        // Every operand finite, the result not — the M3 law, met at the two new
        // doors that store computed positions.
        let huge = 1.0e308_f64;
        let mut m = Mesh::new();
        let v = m.alloc_vert([huge, 0.0, 0.0]);
        let before = m.encoded();
        match apply(
            &mut m,
            &Op::ScaleVerts {
                verts: vec![v],
                pivot: [-huge, 0.0, 0.0],
                factor: [10.0, 1.0, 1.0],
            },
        ) {
            Err(OpError::NonFinite { what, .. }) => {
                assert_eq!(what, "a transformed vertex position")
            }
            other => panic!("an overflowing scale was not refused: {other:?}"),
        }
        assert_eq!(m.encoded(), before);
    }

    #[test]
    fn naming_a_vertex_twice_transforms_it_once() {
        // A rotate is not additive, so a duplicated id in the operand list would
        // be a double rotation — a defect the caller could not see and the op
        // makes impossible.
        let mut m = Mesh::new();
        let v = m.alloc_vert([1.0, 0.0, 0.0]);
        apply(
            &mut m,
            &Op::RotateVerts {
                verts: vec![v, v, v],
                pivot: [0.0; 3],
                axis: [0.0, 1.0, 0.0],
                radians: std::f64::consts::FRAC_PI_2,
            },
        )
        .expect("rotates");
        let p = DVec3::from_array(m.vert_ref(v).position);
        assert!((p - DVec3::new(0.0, 0.0, -1.0)).length() < 1e-6, "{p:?}");
    }

    #[test]
    fn the_rotation_is_an_isometry_even_though_its_angle_is_approximate() {
        // The two halves of the portability trade, stated as one test rather than
        // left implicit in a tolerance: the shape is preserved to `f64` rounding,
        // and the accumulated *angle* error over a full turn is under 1e-6 rad. A
        // `sin_cos` from `std` would swap those — an exact angle on this machine,
        // and a mesh two machines disagree about.
        //
        // Measured by composition rather than with `atan2`, which is on this
        // crate's own ban list: four quarter-turns must land back on the start.
        let mut m = Mesh::new();
        let v = m.alloc_vert([1.0, 0.0, 0.0]);
        for _ in 0..4 {
            apply(
                &mut m,
                &Op::RotateVerts {
                    verts: vec![v],
                    pivot: [0.0; 3],
                    axis: [0.0, 1.0, 0.0],
                    radians: std::f64::consts::FRAC_PI_2,
                },
            )
            .expect("rotates");
            let p = DVec3::from_array(m.vert_ref(v).position);
            assert!(
                (p.length() - 1.0).abs() < 1e-14,
                "an isometry must not change the radius: {p:?}"
            );
        }
        let p = DVec3::from_array(m.vert_ref(v).position);
        assert!(
            (p - DVec3::new(1.0, 0.0, 0.0)).length() < 1e-6,
            "four quarter-turns must close the circle, got {p:?}"
        );
    }
}
