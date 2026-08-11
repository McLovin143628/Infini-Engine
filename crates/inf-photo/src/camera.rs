//! The camera model: **pinhole plus two radial distortion coefficients**, in
//! `f64` end to end.
//!
//! # The model
//!
//! A world point `X` becomes a pixel in four steps:
//!
//! 1. **Pose.** `p = R X + t`, where `(R, t)` is the *world-to-camera* rigid
//!    transform carried by [`Pose`]. The camera looks down `+Z` in its own
//!    frame; a point is in front of the camera exactly when `p.z > 0`.
//! 2. **Perspective divide.** `n = (p.x / p.z, p.y / p.z)` — the *normalized*
//!    image plane, dimensionless.
//! 3. **Radial distortion.** `d = n * (1 + k1 r² + k2 r⁴)` with `r² = |n|²`.
//! 4. **Intrinsics.** `pixel = (fx * d.x + cx, fy * d.y + cy)`.
//!
//! # Units
//!
//! * `fx`, `fy`, `cx`, `cy` are **pixels**. `width`/`height` are pixels.
//! * `k1`, `k2` are **dimensionless** — they multiply powers of the normalized
//!   radius, which is itself dimensionless, so they do not scale with resolution
//!   and can be carried across a resize that also scales `fx/fy/cx/cy`.
//! * [`Pose::translation`] is in **reconstruction units**, not metres. See the
//!   crate docs on gauge: structure from motion is scale-ambiguous, so the world
//!   frame is the first camera and the unit is the baseline to the second.
//!
//! # Why only `k1`/`k2`
//!
//! Two radial terms fit ordinary rectilinear lenses to well under a pixel across
//! the frame. Tangential (`p1`/`p2`) terms model sensor–lens misalignment, and a
//! third radial term matters mostly for wide angles; both are additions to this
//! struct plus rows in [`project_with_jacobian`], and neither is here. A
//! fisheye needs a *different* model (equidistant, not polynomial-radial) and is
//! not expressible by adding coefficients — that is a `CameraModel` enum, and
//! this crate does not have one yet.

use glam::{DMat3, DQuat, DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::linalg::{mat3_at, mat3_from_rows, normalize_or};

/// How many fixed-point iterations [`Intrinsics::undistort`] runs.
///
/// The inverse of a polynomial radial distortion has no closed form, so it is
/// solved by the standard fixed-point iteration `n <- d / (1 + k1 r² + k2 r⁴)`
/// evaluated at the current estimate. For the distortion magnitudes real lenses
/// exhibit this converges to below 1e-12 of a normalized unit in well under ten
/// steps. The count is **fixed**, never "iterate until converged", so undistortion
/// costs the same and returns the same bits on every call.
pub const UNDISTORT_ITERATIONS: usize = 12;

/// Pinhole intrinsics with two radial distortion coefficients.
///
/// These are an **input** to the reconstruction and are never refined by it —
/// see the crate docs' "what this is not". Callers get them from a calibration,
/// from EXIF, or from an assumed field of view.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Intrinsics {
    /// Image width in **pixels**.
    pub width: u32,
    /// Image height in **pixels**.
    pub height: u32,
    /// Focal length along x, in **pixels**.
    pub fx: f64,
    /// Focal length along y, in **pixels**.
    pub fy: f64,
    /// Principal point x, in **pixels**.
    pub cx: f64,
    /// Principal point y, in **pixels**.
    pub cy: f64,
    /// First radial distortion coefficient (**dimensionless**).
    pub k1: f64,
    /// Second radial distortion coefficient (**dimensionless**).
    pub k2: f64,
}

impl Intrinsics {
    /// A distortion-free pinhole.
    pub fn pinhole(width: u32, height: u32, fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            k1: 0.0,
            k2: 0.0,
        }
    }

    /// The same camera with radial distortion.
    pub fn with_radial(mut self, k1: f64, k2: f64) -> Self {
        self.k1 = k1;
        self.k2 = k2;
        self
    }

    /// A centred pinhole with a square pixel and the given focal length in
    /// **pixels** — the honest default when nothing is known but the image size.
    ///
    /// The principal point is `((w - 1) / 2, (h - 1) / 2)`, **not** `(w/2,
    /// h/2)`, because this crate is on the **pixel-centre** convention: pixel
    /// `x` is centred at the continuous coordinate `x`, so a `w`-wide sensor
    /// occupies `[-0.5, w - 0.5]` and its centre is `(w - 1) / 2`. Writing
    /// `w/2` puts the optical axis half a pixel off the sensor centre in both
    /// axes, which is the same principal-point shift the synthetic renderer
    /// carried before it was corrected — and with radial distortion in the
    /// model, which is centred on the principal point, no camera rotation can
    /// absorb it.
    pub fn centred(width: u32, height: u32, focal_px: f64) -> Self {
        Self::pinhole(
            width,
            height,
            focal_px,
            focal_px,
            (width as f64 - 1.0) * 0.5,
            (height as f64 - 1.0) * 0.5,
        )
    }

    /// The mean focal length in pixels — the scale that converts a pixel
    /// threshold into a normalized-plane one.
    #[inline]
    pub fn mean_focal(&self) -> f64 {
        0.5 * (self.fx + self.fy)
    }

    /// Apply radial distortion to a normalized point.
    #[inline]
    pub fn distort(&self, n: DVec2) -> DVec2 {
        let r2 = n.x * n.x + n.y * n.y;
        let f = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        n * f
    }

    /// Invert radial distortion by [`UNDISTORT_ITERATIONS`] fixed-point steps.
    #[inline]
    pub fn undistort(&self, d: DVec2) -> DVec2 {
        if self.k1 == 0.0 && self.k2 == 0.0 {
            return d;
        }
        let mut n = d;
        for _ in 0..UNDISTORT_ITERATIONS {
            let r2 = n.x * n.x + n.y * n.y;
            let f = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
            if f.abs() < 1e-12 {
                return d;
            }
            n = d / f;
        }
        n
    }

    /// Normalized (undistorted) point -> **pixel**.
    #[inline]
    pub fn to_pixel(&self, n: DVec2) -> DVec2 {
        let d = self.distort(n);
        DVec2::new(self.fx * d.x + self.cx, self.fy * d.y + self.cy)
    }

    /// **Pixel** -> normalized (undistorted) point.
    #[inline]
    pub fn to_normalized(&self, px: DVec2) -> DVec2 {
        let d = DVec2::new((px.x - self.cx) / self.fx, (px.y - self.cy) / self.fy);
        self.undistort(d)
    }

    /// Project a **camera-space** point to a pixel. `None` when the point is at
    /// or behind the camera plane.
    #[inline]
    pub fn project(&self, p: DVec3) -> Option<DVec2> {
        if !p.z.is_finite() || p.z <= 1e-9 {
            return None;
        }
        Some(self.to_pixel(DVec2::new(p.x / p.z, p.y / p.z)))
    }

    /// Project a **camera-space** point and return the 2x3 Jacobian of the pixel
    /// with respect to that camera-space point.
    ///
    /// The Jacobian rows are `d(px)/d(p)` and `d(py)/d(p)`. It is the whole
    /// chain — perspective divide **and** distortion — because the bundle
    /// adjuster minimises in pixels, and a Jacobian that ignored distortion
    /// would make every step slightly wrong in exactly the direction the
    /// distortion bends.
    pub fn project_with_jacobian(&self, p: DVec3) -> Option<(DVec2, [[f64; 3]; 2])> {
        if !p.z.is_finite() || p.z <= 1e-9 {
            return None;
        }
        let inv_z = 1.0 / p.z;
        let xn = p.x * inv_z;
        let yn = p.y * inv_z;

        // d(n)/d(p): the perspective divide.
        let dn = [[inv_z, 0.0, -xn * inv_z], [0.0, inv_z, -yn * inv_z]];

        // d(distorted)/d(n), scaled by the focal lengths.
        let r2 = xn * xn + yn * yn;
        let f = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        // d(f)/d(r²) = k1 + 2 k2 r², and d(r²)/d(xn) = 2 xn.
        let dfdr2 = self.k1 + 2.0 * self.k2 * r2;
        let dd = [
            [
                self.fx * (f + 2.0 * xn * xn * dfdr2),
                self.fx * (2.0 * xn * yn * dfdr2),
            ],
            [
                self.fy * (2.0 * xn * yn * dfdr2),
                self.fy * (f + 2.0 * yn * yn * dfdr2),
            ],
        ];

        let mut j = [[0.0f64; 3]; 2];
        for (r, jr) in j.iter_mut().enumerate() {
            for (c, slot) in jr.iter_mut().enumerate() {
                *slot = dd[r][0] * dn[0][c] + dd[r][1] * dn[1][c];
            }
        }
        let pixel = DVec2::new(self.fx * xn * f + self.cx, self.fy * yn * f + self.cy);
        Some((pixel, j))
    }

    /// Is this pixel inside the image?
    ///
    /// On the **pixel-centre** convention a `w`-wide sensor occupies the
    /// continuous interval `[-0.5, w - 0.5]`: pixel `0` reaches half a pixel to
    /// the left of its own centre, and pixel `w - 1` half a pixel to the right.
    /// Testing `0 <= x < w` instead would reject real coverage at the left edge
    /// and accept half a pixel of nothing at the right one.
    #[inline]
    pub fn contains(&self, px: DVec2) -> bool {
        px.x >= -0.5
            && px.y >= -0.5
            && px.x <= self.width as f64 - 0.5
            && px.y <= self.height as f64 - 0.5
    }

    /// Canonical bytes, for the reconstruction's determinism surface.
    pub fn write_canonical(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        for v in [self.fx, self.fy, self.cx, self.cy, self.k1, self.k2] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
}

/// A rigid **world-to-camera** transform: `p_camera = rotation * X_world + translation`.
///
/// Stored as a unit quaternion plus a translation. The quaternion is kept
/// **sign-canonical** (`w >= 0`) by every constructor and by
/// [`Pose::retract`], because `q` and `-q` are the same rotation but not the
/// same bytes, and this type's bytes are compared.
///
/// Deliberately **not** `Serialize`: glam's `DQuat`/`DVec3` carry no serde
/// impls under the workspace pin (adding the feature would put serde in every
/// crate that touches glam), and nothing in P25.1 writes a pose to a file. The
/// byte surface that matters is [`Pose::write_canonical`], which is explicit
/// about its layout. When P25.3 starts persisting reconstructions this becomes a
/// schema decision, with a `schema_version` and a migration, like every other
/// asset in the engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Unit quaternion taking world directions into camera directions.
    pub rotation: DQuat,
    /// Camera-space position of the world origin.
    pub translation: DVec3,
}

impl Default for Pose {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Pose {
    /// The identity pose: the camera *is* the world frame. This is exactly what
    /// the first registered camera is set to, and what pins the gauge.
    pub const IDENTITY: Pose = Pose {
        rotation: DQuat::from_xyzw(0.0, 0.0, 0.0, 1.0),
        translation: DVec3::ZERO,
    };

    /// Build from a rotation matrix and a translation, normalising and
    /// sign-canonicalising the quaternion.
    pub fn from_rt(r: DMat3, t: DVec3) -> Self {
        Self {
            rotation: canonical_quat(DQuat::from_mat3(&r)),
            translation: t,
        }
    }

    /// Build from a quaternion and a translation.
    pub fn from_qt(q: DQuat, t: DVec3) -> Self {
        Self {
            rotation: canonical_quat(q),
            translation: t,
        }
    }

    /// The rotation as a matrix.
    #[inline]
    pub fn rotation_matrix(&self) -> DMat3 {
        DMat3::from_quat(self.rotation)
    }

    /// World point -> camera-space point.
    #[inline]
    pub fn transform(&self, world: DVec3) -> DVec3 {
        self.rotation * world + self.translation
    }

    /// The camera's **centre** in world space: `-Rᵀ t`.
    #[inline]
    pub fn center(&self) -> DVec3 {
        self.rotation.inverse() * (-self.translation)
    }

    /// The camera's viewing direction (`+Z` of the camera frame) in world space.
    #[inline]
    pub fn forward(&self) -> DVec3 {
        self.rotation.inverse() * DVec3::Z
    }

    /// The pose of a camera at `eye` looking at `target`, with `up` breaking the
    /// roll. Camera `+Z` points at the target; `+X` is right and `+Y` is down,
    /// matching the image convention where `y` grows downward.
    ///
    /// No trigonometry: the frame is three cross products. Roll is expressed by
    /// tilting `up`, which is why the synthetic dataset can vary roll without
    /// calling `sin`.
    pub fn look_at(eye: DVec3, target: DVec3, up: DVec3) -> Self {
        let z = normalize_or(target - eye, DVec3::Z);
        let x = normalize_or(up.cross(z), DVec3::X);
        let y = z.cross(x);
        // Rows of R are the camera axes expressed in world coordinates.
        let r = mat3_from_rows([x.x, x.y, x.z], [y.x, y.y, y.z], [z.x, z.y, z.z]);
        let t = -(r.mul_vec3(eye));
        Self::from_rt(r, t)
    }

    /// Apply a **local** 6-DOF increment: rotation `omega` (an axis-times-angle
    /// 3-vector, applied on the left in camera space) then translation `delta`.
    ///
    /// The rotation retraction is the **normalized quaternion** one:
    /// `dq = normalize(1, omega/2)`, `q <- dq * q`. It is a valid retraction —
    /// smooth, the identity at zero, with the identity differential at zero — so
    /// Levenberg–Marquardt converges exactly as it does under the exponential
    /// map, and it calls **no transcendental function**. That is the whole
    /// reason it is used here rather than Rodrigues; see the crate docs.
    ///
    /// The corresponding update to the translation is `t <- dR t + delta`, which
    /// is what makes `p' = dR (R X + t) + delta` — the perturbation model every
    /// Jacobian in [`crate::bundle`] is written against.
    pub fn retract(&self, omega: DVec3, delta: DVec3) -> Self {
        let dq = DQuat::from_xyzw(0.5 * omega.x, 0.5 * omega.y, 0.5 * omega.z, 1.0).normalize();
        let rotation = (dq * self.rotation).normalize();
        let translation = dq * self.translation + delta;
        Self {
            rotation: canonical_quat(rotation),
            translation,
        }
    }

    /// Scale the reconstruction about the world origin: centres and
    /// translations scale, rotations do not. Reprojection is **exactly**
    /// invariant under this (the perspective divide cancels `s`), which is what
    /// makes it a free gauge projection.
    #[inline]
    pub fn scaled(&self, s: f64) -> Self {
        Self {
            rotation: self.rotation,
            translation: self.translation * s,
        }
    }

    /// The relative pose taking `self`'s camera frame into `other`'s.
    pub fn relative_to(&self, other: &Pose) -> Pose {
        let q = other.rotation * self.rotation.inverse();
        let t = other.translation - q * self.translation;
        Pose::from_qt(q, t)
    }

    /// Canonical bytes, for the reconstruction's determinism surface.
    pub fn write_canonical(&self, out: &mut Vec<u8>) {
        for v in [
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
            self.rotation.w,
            self.translation.x,
            self.translation.y,
            self.translation.z,
        ] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
}

/// Normalize and force `w >= 0`, so a rotation has exactly one representation.
#[inline]
pub fn canonical_quat(q: DQuat) -> DQuat {
    let q = if q.length_squared() > 0.0 {
        q.normalize()
    } else {
        DQuat::IDENTITY
    };
    if q.w < 0.0 {
        DQuat::from_xyzw(-q.x, -q.y, -q.z, -q.w)
    } else {
        q
    }
}

/// The 3x4 projection matrix `[R | t]` of a pose, row-major.
pub fn projection_rows(pose: &Pose) -> [[f64; 4]; 3] {
    let r = pose.rotation_matrix();
    let t = pose.translation;
    [
        [mat3_at(&r, 0, 0), mat3_at(&r, 0, 1), mat3_at(&r, 0, 2), t.x],
        [mat3_at(&r, 1, 0), mat3_at(&r, 1, 1), mat3_at(&r, 1, 2), t.y],
        [mat3_at(&r, 2, 0), mat3_at(&r, 2, 1), mat3_at(&r, 2, 2), t.z],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Intrinsics {
        Intrinsics::centred(640, 480, 520.0).with_radial(-0.11, 0.03)
    }

    #[test]
    fn distortion_round_trips_across_the_frame() {
        let k = cam();
        let mut worst = 0.0f64;
        for iy in 0..17 {
            for ix in 0..17 {
                let px = DVec2::new(
                    ix as f64 * (k.width as f64 - 1.0) / 16.0,
                    iy as f64 * (k.height as f64 - 1.0) / 16.0,
                );
                let n = k.to_normalized(px);
                let back = k.to_pixel(n);
                worst = worst.max((back - px).length());
            }
        }
        assert!(
            worst < 1e-9,
            "pixel -> normalized -> pixel drifted {worst} px"
        );
    }

    #[test]
    fn undistort_is_the_identity_without_coefficients() {
        let k = Intrinsics::centred(320, 240, 300.0);
        let d = DVec2::new(0.31, -0.22);
        assert_eq!(k.undistort(d), d);
    }

    #[test]
    fn the_projection_jacobian_matches_finite_differences() {
        let k = cam();
        let points = [
            DVec3::new(0.2, -0.1, 3.0),
            DVec3::new(-1.4, 0.9, 5.5),
            DVec3::new(0.02, 0.03, 1.1),
        ];
        for p in points {
            let (px, j) = k.project_with_jacobian(p).expect("in front of the camera");
            assert!((px - k.project(p).unwrap()).length() < 1e-12);
            let h = 1e-6;
            for c in 0..3 {
                let mut up = p;
                let mut dn = p;
                up[c] += h;
                dn[c] -= h;
                let fd = (k.project(up).unwrap() - k.project(dn).unwrap()) / (2.0 * h);
                assert!(
                    (fd.x - j[0][c]).abs() < 1e-4 * (1.0 + fd.x.abs()),
                    "d(px)/d(p{c}): analytic {} vs numeric {}",
                    j[0][c],
                    fd.x
                );
                assert!(
                    (fd.y - j[1][c]).abs() < 1e-4 * (1.0 + fd.y.abs()),
                    "d(py)/d(p{c}): analytic {} vs numeric {}",
                    j[1][c],
                    fd.y
                );
            }
        }
    }

    #[test]
    fn the_default_principal_point_is_the_sensor_centre_on_the_pixel_centre_convention() {
        // The crate's convention, asserted where it is easiest to get wrong.
        // Pixel `x` is centred at the continuous coordinate `x`, so the four
        // corner *pixels* of a w x h sensor are (0,0), (w-1,0), (0,h-1) and
        // (w-1,h-1), and the sensor's centre is their centroid.
        let (w, h) = (640u32, 480u32);
        let k = Intrinsics::centred(w, h, 500.0);
        let corners = [
            DVec2::new(0.0, 0.0),
            DVec2::new(w as f64 - 1.0, 0.0),
            DVec2::new(0.0, h as f64 - 1.0),
            DVec2::new(w as f64 - 1.0, h as f64 - 1.0),
        ];
        let centroid = corners.iter().fold(DVec2::ZERO, |a, b| a + *b) / 4.0;
        assert_eq!(DVec2::new(k.cx, k.cy), centroid, "principal point off centre");
        // The principal ray lands on it, and it is not the `w/2` answer — the
        // half pixel this is about.
        let axis = k.to_pixel(DVec2::ZERO);
        assert!((axis - centroid).length() < 1e-12, "principal ray at {axis}");
        assert!(
            (k.cx - w as f64 * 0.5).abs() > 0.4,
            "the principal point is still the pixel-interval answer"
        );
        // And `contains` agrees with it: the sensor reaches half a pixel past
        // the outermost pixel centres and no further.
        assert!(k.contains(DVec2::new(-0.5, -0.5)));
        assert!(k.contains(DVec2::new(w as f64 - 0.5, h as f64 - 0.5)));
        assert!(!k.contains(DVec2::new(-0.51, 0.0)));
        assert!(!k.contains(DVec2::new(w as f64 - 0.49, 0.0)));
    }

    #[test]
    fn a_point_behind_the_camera_does_not_project() {
        let k = cam();
        assert!(k.project(DVec3::new(0.1, 0.1, -2.0)).is_none());
        assert!(k.project(DVec3::new(0.1, 0.1, 0.0)).is_none());
        assert!(k
            .project_with_jacobian(DVec3::new(0.0, 0.0, -1e-3))
            .is_none());
    }

    #[test]
    fn look_at_puts_the_target_on_the_principal_ray() {
        let eye = DVec3::new(1.5, 2.0, -3.0);
        let target = DVec3::new(-0.5, 0.25, 1.0);
        let pose = Pose::look_at(eye, target, DVec3::new(0.03, -1.0, 0.01));
        let p = pose.transform(target);
        assert!(p.z > 0.0, "the target must be in front, got z = {}", p.z);
        assert!(
            p.x.abs() < 1e-12 && p.y.abs() < 1e-12,
            "target off-axis: {p}"
        );
        assert!(
            (pose.center() - eye).length() < 1e-12,
            "centre drifted: {}",
            pose.center()
        );
        assert!((pose.rotation_matrix().determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn retract_perturbs_the_camera_point_the_way_the_jacobian_says() {
        // p' = dR (R X + t) + delta. At small omega, dR v ≈ v + omega × v.
        let pose = Pose::look_at(
            DVec3::new(0.4, -1.0, 2.0),
            DVec3::ZERO,
            DVec3::new(0.0, -1.0, 0.05),
        );
        let x = DVec3::new(0.3, 0.7, -0.2);
        let p = pose.transform(x);
        let omega = DVec3::new(1e-6, -2e-6, 3e-7);
        let delta = DVec3::new(4e-7, 1e-7, -2e-7);
        let moved = pose.retract(omega, delta).transform(x);
        let predicted = p + omega.cross(p) + delta;
        assert!(
            (moved - predicted).length() < 1e-11,
            "retraction differential wrong: {moved} vs {predicted}"
        );
    }

    #[test]
    fn retract_stays_a_unit_sign_canonical_quaternion_under_a_huge_step() {
        let pose = Pose::look_at(DVec3::new(3.0, 1.0, -2.0), DVec3::ZERO, DVec3::NEG_Y);
        let far = pose.retract(DVec3::new(50.0, -30.0, 12.0), DVec3::new(9.0, -4.0, 1.0));
        assert!((far.rotation.length() - 1.0).abs() < 1e-12);
        assert!(far.rotation.w >= 0.0, "quaternion sign not canonical");
        assert!((far.rotation_matrix().determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn scaling_leaves_every_projection_untouched() {
        // The gauge ruling depends on this being exact, not approximate.
        let pose = Pose::look_at(DVec3::new(1.0, 0.5, -4.0), DVec3::ZERO, DVec3::NEG_Y);
        let k = cam();
        let x = DVec3::new(0.6, -0.3, 0.9);
        let s = 3.7;
        let a = k.project(pose.transform(x)).unwrap();
        let b = k.project(pose.scaled(s).transform(x * s)).unwrap();
        assert!(
            (a - b).length() < 1e-9,
            "scale changed the projection: {a} vs {b}"
        );
    }

    #[test]
    fn relative_to_composes() {
        let a = Pose::look_at(DVec3::new(2.0, 0.0, -3.0), DVec3::ZERO, DVec3::NEG_Y);
        let b = Pose::look_at(DVec3::new(-1.0, 1.0, -2.5), DVec3::X, DVec3::NEG_Y);
        let rel = a.relative_to(&b);
        let x = DVec3::new(0.2, -0.4, 0.7);
        let via = rel.transform(a.transform(x));
        assert!(
            (via - b.transform(x)).length() < 1e-12,
            "{via} vs {}",
            b.transform(x)
        );
    }

    #[test]
    fn canonical_bytes_distinguish_poses_and_repeat() {
        let a = Pose::look_at(DVec3::new(1.0, 0.0, -1.0), DVec3::ZERO, DVec3::NEG_Y);
        let b = Pose::look_at(DVec3::new(1.0, 0.0, -1.0001), DVec3::ZERO, DVec3::NEG_Y);
        let (mut ba, mut bb, mut ba2) = (Vec::new(), Vec::new(), Vec::new());
        a.write_canonical(&mut ba);
        b.write_canonical(&mut bb);
        a.write_canonical(&mut ba2);
        assert_eq!(ba, ba2);
        assert_ne!(ba, bb);
        assert_eq!(ba.len(), 7 * 8);
    }
}
