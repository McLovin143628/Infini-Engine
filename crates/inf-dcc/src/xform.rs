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

/// Rotate vertices about `pivot` around `axis` by `radians`.
pub(crate) fn rotate_verts(
    mesh: &mut Mesh,
    verts: &[VertId],
    pivot: [f64; 3],
    axis: [f64; 3],
    radians: f64,
) -> Result<OpOutcome, OpError> {
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
        let (s, c) = (psin64(radians), pcos64(radians));
        let n = (s * s + c * c).sqrt();
        if n.is_finite() && n > 1e-12 {
            (s / n, c / n)
        } else {
            (s, c)
        }
    };
    write_all(mesh, verts, |p| {
        let r = p - pivot;
        pivot + r * c + k.cross(r) * s + k * (k.dot(r) * (1.0 - c))
    })
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
    write_all(mesh, verts, |p| pivot + (p - pivot) * f)
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
