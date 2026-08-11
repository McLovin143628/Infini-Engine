//! Similarity alignment — how a gauge-free reconstruction is compared with a
//! ground truth at all.
//!
//! A reconstruction is only determined up to a **similarity**: a rotation, a
//! translation and a uniform scale. Two solvers can agree perfectly about the
//! shape of a scene and disagree about every number in it. Comparing a
//! reconstruction with known poses therefore means first finding the similarity
//! that best maps one onto the other, and only then measuring what is left over.
//! That leftover is the error; anything measured without this step is measuring
//! the gauge.
//!
//! [`similarity_from_points`] is the closed-form least-squares solution
//! (Umeyama 1991): the optimal rotation comes from the SVD of the
//! cross-covariance, the scale from the ratio of the aligned variance to the
//! source variance, and the translation from the centroids. It is exact, not
//! iterative, and calls no transcendental.

use glam::{DMat3, DQuat, DVec3};

use crate::camera::Pose;
use crate::linalg::{mat3_at, svd3};

/// A rigid transform with a uniform scale: `y = scale * rotation * x +
/// translation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    /// Uniform scale factor (dimensionless).
    pub scale: f64,
    /// Rotation.
    pub rotation: DMat3,
    /// Translation, in the **target's** units.
    pub translation: DVec3,
}

impl Similarity {
    /// The identity.
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            rotation: DMat3::IDENTITY,
            translation: DVec3::ZERO,
        }
    }

    /// Map a point.
    #[inline]
    pub fn apply(&self, p: DVec3) -> DVec3 {
        self.rotation.mul_vec3(p) * self.scale + self.translation
    }

    /// Map a **camera pose**, which transforms contravariantly: a camera at
    /// centre `C` with rotation `R_cam` becomes a camera at `apply(C)` with
    /// rotation `R_cam * rotationᵀ`.
    pub fn apply_pose(&self, pose: &Pose) -> Pose {
        let centre = self.apply(pose.center());
        let rotation = pose.rotation_matrix().mul_mat3(&self.rotation.transpose());
        let translation = -rotation.mul_vec3(centre);
        Pose::from_rt(rotation, translation)
    }
}

/// The least-squares similarity mapping `source` onto `target` (Umeyama).
///
/// Returns `None` for fewer than three points, or for a source cloud with no
/// spread (every point on top of every other), where the scale is undefined.
pub fn similarity_from_points(source: &[DVec3], target: &[DVec3]) -> Option<Similarity> {
    assert_eq!(source.len(), target.len(), "point clouds must line up");
    let n = source.len();
    if n < 3 {
        return None;
    }
    let inv_n = 1.0 / n as f64;
    let mut mu_s = DVec3::ZERO;
    let mut mu_t = DVec3::ZERO;
    for i in 0..n {
        mu_s += source[i];
        mu_t += target[i];
    }
    mu_s *= inv_n;
    mu_t *= inv_n;

    let mut var_s = 0.0f64;
    let mut cov = [[0.0f64; 3]; 3];
    for i in 0..n {
        let ds = source[i] - mu_s;
        let dt = target[i] - mu_t;
        var_s += ds.length_squared();
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += dt[r] * ds[c];
            }
        }
    }
    var_s *= inv_n;
    if var_s <= 1e-18 {
        return None;
    }
    for row in cov.iter_mut() {
        for v in row.iter_mut() {
            *v *= inv_n;
        }
    }

    let sigma = crate::linalg::mat3_from_rows(cov[0], cov[1], cov[2]);
    let (u, s, v) = svd3(&sigma);
    // The reflection guard: if the covariance is left-handed, the last singular
    // direction must be flipped or the "rotation" is a mirror — which fits a
    // mirrored reconstruction perfectly and reports zero error for it.
    let flip = u.determinant() * v.determinant() < 0.0;
    let rotation = if flip {
        let u2 = DMat3::from_cols(u.col(0), u.col(1), -u.col(2));
        u2.mul_mat3(&v.transpose())
    } else {
        u.mul_mat3(&v.transpose())
    };
    let trace_ds = if flip {
        s.x + s.y - s.z
    } else {
        s.x + s.y + s.z
    };
    let scale = trace_ds / var_s;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let translation = mu_t - rotation.mul_vec3(mu_s) * scale;
    Some(Similarity {
        scale,
        rotation,
        translation,
    })
}

/// The angle between two rotations, in **degrees**.
///
/// The one transcendental call in this crate, and it is
/// [`inf_math::patan2_64`] — the bit-portable one — because a metric that
/// disagreed between a Windows and a Linux CI leg would make every bound in the
/// gate platform-dependent.
///
/// `atan2` of the quaternion difference's vector and scalar parts, **not**
/// `acos` of the dot product. That is not a stylistic preference: `acos` near
/// its argument's limit loses half its significant digits, so `2 acos(|dot|)`
/// reports around `2e-6` degrees for two rotations that are bit-identical up to
/// rounding. Measured, on this module's own round-trip test, before the form was
/// changed. The `atan2` form is accurate all the way down to zero, which matters
/// because "the pose error is below a bound" is the gate's headline claim.
pub fn rotation_angle_deg(a: DQuat, b: DQuat) -> f64 {
    // The rotation carrying `a` to `b`.
    let d = b * a.inverse();
    let vector = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
    // |w| rather than w picks the shorter of the two arcs.
    2.0 * inf_math::patan2_64(vector, d.w.abs()).to_degrees()
}

/// Errors of an aligned reconstruction against a ground truth.
///
/// # Two rotation errors, and which one to believe
///
/// [`PoseError::max_aligned_rotation_deg`] measures each camera's rotation after
/// the similarity fitted to the camera *centres* has been applied. That is the
/// intuitive metric, and for a handful of cameras it is **dominated by the
/// alignment's own uncertainty**: the similarity's rotation is estimated from a
/// few centres, so a centre residual of `e` over a configuration of radius `R`
/// leaves the alignment rotation uncertain by roughly `atan(e / R)`, and every
/// camera inherits that same error. Measured on the P25.1 gate fixture: six
/// cameras, centre residuals of 12 mm over a 2 m radius, and an aligned rotation
/// error of 0.31 degrees of which most is the alignment's.
///
/// [`PoseError::max_relative_rotation_deg`] is the **gauge-free** answer: for
/// every pair of cameras it compares the *relative* rotation `Rⱼ Rᵢᵀ` between
/// the estimate and the truth. No alignment is involved, so nothing is
/// inherited, and it is the number to gate a reconstruction's rotational
/// accuracy on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseError {
    /// Worst camera-centre error, in the **target's** units.
    pub max_center: f64,
    /// Mean camera-centre error, in the **target's** units.
    pub mean_center: f64,
    /// Worst rotation error after aligning centres, in **degrees**.
    pub max_aligned_rotation_deg: f64,
    /// Mean rotation error after aligning centres, in **degrees**.
    pub mean_aligned_rotation_deg: f64,
    /// Worst **relative** rotation error over all camera pairs, in **degrees**.
    /// Gauge-free.
    pub max_relative_rotation_deg: f64,
    /// Mean relative rotation error over all camera pairs, in **degrees**.
    pub mean_relative_rotation_deg: f64,
    /// The similarity that was applied before measuring the centres.
    pub alignment: Similarity,
}

/// Align `estimated` onto `truth` by their camera centres and report what is
/// left. The two slices must be the same cameras in the same order.
pub fn pose_error(estimated: &[Pose], truth: &[Pose]) -> Option<PoseError> {
    assert_eq!(estimated.len(), truth.len(), "pose lists must line up");
    let src: Vec<DVec3> = estimated.iter().map(|p| p.center()).collect();
    let dst: Vec<DVec3> = truth.iter().map(|p| p.center()).collect();
    let alignment = similarity_from_points(&src, &dst)?;

    let mut max_center = 0.0f64;
    let mut sum_center = 0.0f64;
    let mut max_rot = 0.0f64;
    let mut sum_rot = 0.0f64;
    for (est, want) in estimated.iter().zip(truth) {
        let aligned = alignment.apply_pose(est);
        let dc = (aligned.center() - want.center()).length();
        let dr = rotation_angle_deg(aligned.rotation, want.rotation);
        max_center = max_center.max(dc);
        sum_center += dc;
        max_rot = max_rot.max(dr);
        sum_rot += dr;
    }

    let mut max_rel = 0.0f64;
    let mut sum_rel = 0.0f64;
    let mut pairs = 0usize;
    for i in 0..estimated.len() {
        for j in (i + 1)..estimated.len() {
            let got = estimated[j].rotation * estimated[i].rotation.inverse();
            let want = truth[j].rotation * truth[i].rotation.inverse();
            let d = rotation_angle_deg(got, want);
            max_rel = max_rel.max(d);
            sum_rel += d;
            pairs += 1;
        }
    }

    let n = estimated.len() as f64;
    Some(PoseError {
        max_center,
        mean_center: sum_center / n,
        max_aligned_rotation_deg: max_rot,
        mean_aligned_rotation_deg: sum_rot / n,
        max_relative_rotation_deg: max_rel,
        mean_relative_rotation_deg: if pairs == 0 {
            0.0
        } else {
            sum_rel / pairs as f64
        },
        alignment,
    })
}

/// Debug spelling of a similarity, for assertion messages.
pub fn similarity_debug(s: &Similarity) -> String {
    format!(
        "scale {:.6}, t ({:.4}, {:.4}, {:.4}), R [{:.4} {:.4} {:.4}]",
        s.scale,
        s.translation.x,
        s.translation.y,
        s.translation.z,
        mat3_at(&s.rotation, 0, 0),
        mat3_at(&s.rotation, 0, 1),
        mat3_at(&s.rotation, 0, 2),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Hash64;

    fn cloud(n: usize) -> Vec<DVec3> {
        (0..n)
            .map(|i| {
                let h = Hash64::new(0xa11).mix_usize(i);
                DVec3::new(
                    -2.0 + 4.0 * h.mix_u64(1).unit(),
                    -1.0 + 2.5 * h.mix_u64(2).unit(),
                    1.0 + 3.0 * h.mix_u64(3).unit(),
                )
            })
            .collect()
    }

    fn planted() -> Similarity {
        // Asymmetric on purpose: a rotation about a slanted axis and an
        // off-centre translation, so a transposed rotation or a swapped axis
        // does not accidentally fit.
        let f = DVec3::new(0.3, 0.8, -0.5).normalize();
        let u = {
            let raw = DVec3::new(-0.2, 0.4, 0.9);
            (raw - f * f.dot(raw)).normalize()
        };
        let r = DMat3::from_cols(f, u, f.cross(u));
        Similarity {
            scale: 2.75,
            rotation: r,
            translation: DVec3::new(-1.5, 0.75, 3.25),
        }
    }

    #[test]
    fn umeyama_recovers_a_planted_similarity() {
        let src = cloud(24);
        let want = planted();
        let dst: Vec<DVec3> = src.iter().map(|&p| want.apply(p)).collect();
        let got = similarity_from_points(&src, &dst).expect("a similarity");
        assert!(
            (got.scale - want.scale).abs() < 1e-12,
            "scale {} vs {} ({})",
            got.scale,
            want.scale,
            similarity_debug(&got)
        );
        for p in &src {
            assert!(
                (got.apply(*p) - want.apply(*p)).length() < 1e-10,
                "mapping disagrees at {p}"
            );
        }
    }

    #[test]
    fn umeyama_does_not_fit_a_mirror() {
        // Anti-vacuity: reflect the target. A solver without the reflection
        // guard reports a perfect fit for a mirrored reconstruction, which is
        // exactly the failure a photogrammetry gate must never miss.
        let src = cloud(24);
        let dst: Vec<DVec3> = src.iter().map(|p| DVec3::new(p.x, p.y, -p.z)).collect();
        let got = similarity_from_points(&src, &dst).expect("a similarity");
        assert!(
            (got.rotation.determinant() - 1.0).abs() < 1e-12,
            "the fit is a reflection: det = {}",
            got.rotation.determinant()
        );
        let residual: f64 = src
            .iter()
            .zip(&dst)
            .map(|(s, d)| (got.apply(*s) - *d).length())
            .sum();
        assert!(
            residual > 1e-3,
            "a mirror was fitted exactly: residual {residual}"
        );
    }

    #[test]
    fn too_few_or_degenerate_points_refuse() {
        assert!(similarity_from_points(&cloud(2), &cloud(2)).is_none());
        let flat = vec![DVec3::ONE; 6];
        assert!(similarity_from_points(&flat, &cloud(6)).is_none());
    }

    #[test]
    fn rotation_angle_is_zero_for_equal_rotations_and_180_for_opposites() {
        let q = DQuat::from_axis_angle(DVec3::new(0.3, 0.5, 0.8).normalize(), 0.7);
        assert!(rotation_angle_deg(q, q) < 1e-7);
        // A quaternion and its negation are the same rotation.
        let neg = DQuat::from_xyzw(-q.x, -q.y, -q.z, -q.w);
        assert!(rotation_angle_deg(q, neg) < 1e-7);
        let flipped = DQuat::from_axis_angle(DVec3::X, std::f64::consts::PI) * q;
        assert!(
            (rotation_angle_deg(q, flipped) - 180.0).abs() < 1e-6,
            "got {}",
            rotation_angle_deg(q, flipped)
        );
    }

    #[test]
    fn pose_error_is_zero_for_a_similarity_transformed_reconstruction() {
        let truth: Vec<Pose> = [
            (DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 5.0)),
            (DVec3::new(1.2, 0.3, -0.4), DVec3::new(0.1, 0.0, 5.0)),
            (DVec3::new(2.1, -0.2, 0.5), DVec3::new(-0.1, 0.1, 5.0)),
            (DVec3::new(-1.4, 0.6, 0.1), DVec3::new(0.2, -0.1, 5.0)),
        ]
        .iter()
        .map(|&(eye, at)| Pose::look_at(eye, at, DVec3::new(0.03, -1.0, 0.02)))
        .collect();
        let s = planted();
        let estimated: Vec<Pose> = truth.iter().map(|p| s.apply_pose(p)).collect();
        // `estimated` differs from `truth` by a similarity, so after alignment
        // the error must vanish.
        let err = pose_error(&estimated, &truth).expect("alignable");
        assert!(
            err.max_center < 1e-9,
            "centre error {} survived",
            err.max_center
        );
        assert!(
            err.max_aligned_rotation_deg < 1e-6,
            "rotation error {} survived",
            err.max_aligned_rotation_deg
        );
        assert!(
            err.max_relative_rotation_deg < 1e-6,
            "relative rotation error {} survived",
            err.max_relative_rotation_deg
        );
    }

    #[test]
    fn pose_error_sees_a_real_error() {
        // Anti-vacuity for the test above: nudge one camera and the metric must
        // move, or it is measuring nothing.
        let truth: Vec<Pose> = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.2, 0.3, -0.4),
            DVec3::new(2.1, -0.2, 0.5),
            DVec3::new(-1.4, 0.6, 0.1),
        ]
        .iter()
        .map(|&eye| Pose::look_at(eye, DVec3::new(0.0, 0.0, 5.0), DVec3::new(0.03, -1.0, 0.02)))
        .collect();
        let mut estimated = truth.clone();
        estimated[2] = estimated[2].retract(DVec3::new(0.05, 0.0, 0.0), DVec3::new(0.1, 0.0, 0.0));
        let err = pose_error(&estimated, &truth).expect("alignable");
        assert!(
            err.max_center > 0.01,
            "a 0.1-unit displacement went unnoticed: {}",
            err.max_center
        );
        assert!(
            err.max_aligned_rotation_deg > 0.5,
            "a 0.05 rad twist went unnoticed by the aligned metric"
        );
        assert!(
            err.max_relative_rotation_deg > 2.0,
            "a 0.05 rad twist went unnoticed by the gauge-free metric: {}",
            err.max_relative_rotation_deg
        );
    }

    #[test]
    fn apply_pose_keeps_the_camera_looking_at_the_same_content() {
        let pose = Pose::look_at(
            DVec3::new(1.0, 0.5, -2.0),
            DVec3::new(0.0, 0.0, 3.0),
            DVec3::new(0.0, -1.0, 0.0),
        );
        let s = planted();
        let moved = s.apply_pose(&pose);
        let x = DVec3::new(0.4, -0.2, 2.5);
        // The transformed camera sees the transformed point in the same
        // direction (the depth scales, the ray does not move).
        let before = pose.transform(x);
        let after = moved.transform(s.apply(x));
        assert!(
            (before.normalize() - after.normalize()).length() < 1e-10,
            "the viewing ray moved: {before} -> {after}"
        );
        assert!((after.length() / before.length() - s.scale).abs() < 1e-9);
    }
}
