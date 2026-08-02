//! Cascaded-shadow-map math (P13.3b): the pure, GPU-free half of the CSM pass.
//!
//! Splitting the view frustum into cascades, fitting a stable orthographic light
//! projection to each slice, and texel-snapping it are all deterministic pure
//! functions so they unit-test without a device. [`crate::passes::shadow`] consumes
//! them to build the per-cascade `view_proj` matrices; the receiver shaders sample
//! the resulting depth array (see `shaders/mesh.wgsl`).
//!
//! ## Conventions (documented choices)
//!
//! * **Cascades:** [`SHADOW_CASCADES`] (3), each a `SHADOW_RESOLUTION`² depth slice
//!   of one `Depth32Float` array texture.
//! * **Split scheme:** the *practical split* blend of the logarithmic and uniform
//!   schemes (Zhang et al.), weighted by λ ([`crate::ShadowSettings::lambda`],
//!   default 0.7): near cascades get more resolution, far ones cover more ground.
//! * **Forward-Z shadow depth:** unlike the reverse-Z **camera** projection, the
//!   shadow ortho uses a conventional forward-Z range (near → 0, far → 1) via
//!   [`glam::Mat4::orthographic_rh`], compared `LessEqual`. The two projections are
//!   independent; keeping the shadow map forward-Z simplifies the bias reasoning
//!   and the comparison-sampler direction (a receiver is lit when its light-space
//!   depth is ≤ the stored nearest-caster depth + bias).
//! * **Stability:** each cascade is fit to the *bounding sphere* of its frustum
//!   slice (rotation-invariant, so the covered area doesn't pulse as the camera
//!   turns) and the sphere centre is **snapped to the cascade's texel grid** in
//!   light space (so shadows don't swim under camera translation).

use glam::{Mat4, Vec3};

/// Number of shadow cascades (finest = 0).
pub const SHADOW_CASCADES: usize = 3;
/// Per-cascade shadow-map resolution (square). 2048² default; the whole array is
/// `SHADOW_RESOLUTION² × SHADOW_CASCADES` `Depth32Float`.
pub const SHADOW_RESOLUTION: u32 = 2048;

/// The `SHADOW_CASCADES` cascade **far** distances (metres) along the view
/// direction, blended between the logarithmic and uniform split schemes by
/// `lambda` (0 = uniform, 1 = logarithmic). The last element is `far`. `near`/`far`
/// are the shadow range (camera near → [`crate::ShadowSettings::max_distance`]).
pub fn cascade_splits(near: f32, far: f32, lambda: f32) -> [f32; SHADOW_CASCADES] {
    let near = near.max(1e-3);
    let far = far.max(near + 1e-3);
    let lambda = lambda.clamp(0.0, 1.0);
    let mut out = [0.0; SHADOW_CASCADES];
    let n = SHADOW_CASCADES as f32;
    for (i, o) in out.iter_mut().enumerate() {
        let p = (i as f32 + 1.0) / n;
        let log = near * (far / near).powf(p);
        let uni = near + (far - near) * p;
        *o = lambda * log + (1.0 - lambda) * uni;
    }
    out
}

/// The 8 world-space (render-local) corners of the view-frustum slice between view
/// distances `d0` and `d1` (near four, then far four). Built analytically from the
/// camera basis + FOV + aspect so it works with the reverse-infinite projection
/// (whose far plane is at infinity and can't be unprojected).
pub fn frustum_slice_corners(
    eye: Vec3,
    forward: Vec3,
    up: Vec3,
    fov_y: f32,
    aspect: f32,
    d0: f32,
    d1: f32,
) -> [Vec3; 8] {
    let f = forward.normalize_or_zero();
    let r = f.cross(up).normalize_or_zero();
    let u = r.cross(f).normalize_or_zero();
    let tan = (fov_y * 0.5).tan();
    let plane = |d: f32| {
        let h = tan * d;
        let w = h * aspect;
        let c = eye + f * d;
        [
            c + u * h - r * w,
            c + u * h + r * w,
            c - u * h - r * w,
            c - u * h + r * w,
        ]
    };
    let n = plane(d0);
    let fa = plane(d1);
    [n[0], n[1], n[2], n[3], fa[0], fa[1], fa[2], fa[3]]
}

/// Bounding sphere `(centre, radius)` of a point set (the frustum-slice corners).
/// A cheap centroid + max-radius fit — exact enough for a shadow cascade and
/// rotation-invariant (the whole point of fitting a sphere rather than an AABB).
pub fn bounding_sphere(points: &[Vec3]) -> (Vec3, f32) {
    if points.is_empty() {
        return (Vec3::ZERO, 1.0);
    }
    let center = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let radius = points
        .iter()
        .map(|p| (*p - center).length())
        .fold(0.0_f32, f32::max)
        .max(1e-3);
    (center, radius)
}

/// The stable orthographic `view_proj` for one cascade fit to a frustum-slice
/// bounding sphere at render-local `center` with `radius`, cast from a directional
/// light whose **unit direction toward the light** is `light_dir_to`.
///
/// Returns `(view_proj, texel_world)` where `texel_world = 2·radius / resolution`
/// is the world size of one shadow texel (drives the normal-offset bias). The
/// sphere centre is snapped to the texel grid in light space so the shadow is
/// stable under camera motion.
pub fn cascade_matrix(
    light_dir_to: Vec3,
    center: Vec3,
    radius: f32,
    resolution: u32,
) -> (Mat4, f32) {
    let light_fwd = (-light_dir_to).normalize_or_zero();
    let light_fwd = if light_fwd.length_squared() < 1e-6 {
        Vec3::NEG_Y
    } else {
        light_fwd
    };
    // A world up that isn't parallel to the light direction.
    let up = if light_fwd.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let right = light_fwd.cross(up).normalize_or_zero();
    let true_up = right.cross(light_fwd).normalize_or_zero();

    let res = resolution.max(1) as f32;
    let texel = 2.0 * radius / res;
    // Snap the centre to the texel grid along the light's right/up axes.
    let sx = (center.dot(right) / texel).round() * texel;
    let sy = (center.dot(true_up) / texel).round() * texel;
    let sz = center.dot(light_fwd);
    let snapped = right * sx + true_up * sy + light_fwd * sz;

    // Pull the light "eye" back so casters behind the slice still fit the near side.
    let pullback = radius;
    let eye = snapped - light_fwd * (radius + pullback);
    let view = glam::camera::rh::view::look_to_mat4(eye, light_fwd, true_up);
    // Forward-Z ortho (wgpu NDC z ∈ [0,1]): near → 0, far → 1.
    let ortho = glam::camera::rh::proj::directx::orthographic(
        -radius,
        radius,
        -radius,
        radius,
        0.0,
        2.0 * radius + pullback,
    );
    (ortho * view, texel)
}

/// The **cascade blend weight** (P18.4): how much of the *next* cascade a receiver
/// at view distance `dist` mixes in, given the current cascade's range
/// `[near_edge, far]` and a blend band of `blend` × that range.
///
/// `0` everywhere except the last `blend` fraction of the cascade, ramping linearly
/// to `1` exactly at `far` — so the transition is continuous in `dist`, and the two
/// cascades agree at the hand-over point instead of switching mid-surface. `blend =
/// 0` returns `0` everywhere, which is the pre-P18.4 hard switch exactly.
///
/// Mirrors the arithmetic in `shadow_factor` (`shaders/env_lighting.wgsl`); it
/// lives here so the continuity property is unit-tested with no GPU, which is the
/// only way that property runs on every CI leg.
pub fn cascade_blend_weight(dist: f32, near_edge: f32, far: f32, blend: f32) -> f32 {
    let blend = blend.clamp(0.0, 0.5);
    if blend <= 0.0 {
        return 0.0;
    }
    let band = (far - near_edge).max(1e-4) * blend;
    ((dist - (far - band)) / band.max(1e-4)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the P13 deferral this closes: the shadow factor must be
    /// **continuous** across a cascade boundary, not a step.
    #[test]
    fn cascade_blend_is_continuous_and_reaches_both_ends() {
        let (near_edge, far, blend) = (0.1f32, 10.0, 0.1);
        // Zero well inside the cascade, one exactly at the hand-over.
        assert_eq!(cascade_blend_weight(0.5, near_edge, far, blend), 0.0);
        assert_eq!(cascade_blend_weight(8.0, near_edge, far, blend), 0.0);
        assert!((cascade_blend_weight(far, near_edge, far, blend) - 1.0).abs() < 1e-6);
        // Clamped past the end (the next cascade owns it from here).
        assert_eq!(cascade_blend_weight(50.0, near_edge, far, blend), 1.0);

        // Continuous: no step larger than the sample spacing's own slope anywhere
        // across the band, INCLUDING at the band's opening edge (the seam a naive
        // `if dist > threshold` implementation would leave).
        let mut prev = cascade_blend_weight(0.0, near_edge, far, blend);
        let mut d = 0.0f32;
        while d <= far {
            let w = cascade_blend_weight(d, near_edge, far, blend);
            assert!((0.0..=1.0).contains(&w));
            assert!(w + 1e-6 >= prev, "not monotonic at {d}");
            assert!(
                (w - prev) < 0.05,
                "discontinuity of {} at distance {d}",
                w - prev
            );
            prev = w;
            d += 0.01;
        }
        assert!(prev > 0.99, "the band never reached the next cascade");
    }

    #[test]
    fn zero_blend_is_the_pre_p18_4_hard_switch() {
        for d in [0.0f32, 5.0, 9.999, 10.0, 100.0] {
            assert_eq!(
                cascade_blend_weight(d, 0.1, 10.0, 0.0),
                0.0,
                "blend 0 must never mix the next cascade in (at {d})"
            );
        }
        // ...and a degenerate cascade range cannot divide by zero.
        assert!(cascade_blend_weight(5.0, 10.0, 10.0, 0.1).is_finite());
    }

    #[test]
    fn splits_are_monotonic_within_range() {
        let s = cascade_splits(0.1, 60.0, 0.7);
        assert!(s[0] > 0.1 && s[0] < s[1] && s[1] < s[2]);
        assert!((s[SHADOW_CASCADES - 1] - 60.0).abs() < 1e-3, "last == far");
        // λ=0 → uniform splits (evenly spaced).
        let u = cascade_splits(0.0, 30.0, 0.0);
        assert!((u[0] - 10.0).abs() < 1e-3 && (u[1] - 20.0).abs() < 1e-3);
    }

    #[test]
    fn bounding_sphere_contains_corners() {
        let corners = frustum_slice_corners(
            Vec3::new(0.0, 2.0, 5.0),
            Vec3::NEG_Z,
            Vec3::Y,
            60f32.to_radians(),
            16.0 / 9.0,
            1.0,
            10.0,
        );
        let (c, r) = bounding_sphere(&corners);
        for p in &corners {
            assert!((*p - c).length() <= r + 1e-3, "corner outside sphere");
        }
    }

    #[test]
    fn cascade_matrix_maps_slice_into_clip() {
        let corners = frustum_slice_corners(
            Vec3::new(0.0, 3.0, 8.0),
            Vec3::NEG_Z,
            Vec3::Y,
            60f32.to_radians(),
            1.0,
            0.1,
            20.0,
        );
        let (c, r) = bounding_sphere(&corners);
        let light = Vec3::new(0.4, 0.8, 0.3).normalize();
        let (vp, texel) = cascade_matrix(light, c, r, SHADOW_RESOLUTION);
        assert!(texel > 0.0);
        // Every slice corner projects inside the shadow clip box (xy ∈ [-1,1],
        // z ∈ [0,1]) — the cascade actually covers its frustum slice.
        for p in &corners {
            let ndc = vp.project_point3(*p);
            assert!(ndc.x.abs() <= 1.01 && ndc.y.abs() <= 1.01, "xy {ndc:?}");
            assert!(ndc.z >= -0.01 && ndc.z <= 1.01, "z {ndc:?}");
        }
    }

    #[test]
    fn texel_snapping_is_stable_under_subtexel_motion() {
        // Two centres a fraction of a texel apart snap to the same grid → identical
        // matrices (no shadow swimming).
        let r = 10.0;
        let texel = 2.0 * r / SHADOW_RESOLUTION as f32;
        let light = Vec3::new(0.3, 0.9, 0.2).normalize();
        let c0 = Vec3::new(1.0, 0.0, 1.0);
        let c1 = c0 + Vec3::X * (texel * 0.2);
        let (m0, _) = cascade_matrix(light, c0, r, SHADOW_RESOLUTION);
        let (m1, _) = cascade_matrix(light, c1, r, SHADOW_RESOLUTION);
        let d: f32 = m0
            .to_cols_array()
            .iter()
            .zip(m1.to_cols_array())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(d < 1e-4, "sub-texel motion changed the matrix (swim): {d}");
    }
}
