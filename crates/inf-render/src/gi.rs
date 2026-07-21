//! Dynamic global-illumination math (P13.3b): the pure, GPU-free half of the
//! real-time single-bounce diffuse GI.
//!
//! The layout of the camera-centred voxel volume and the probe grid, the
//! deterministic golden-spiral ray directions, and the L1 spherical-harmonic basis
//! all live here as pure functions so they unit-test without a device and stay
//! bit-identical to the compute shaders ([`crate::passes::gi`], `shaders/gi_*.wgsl`)
//! that mirror them.
//!
//! ## The scheme (what the shaders implement)
//!
//! 1. **Voxelize** a [`GI_DIM`]³ volume centred on the camera, covering
//!    [`crate::GiSettings::extent`] metres: each voxel stores an albedo + binary
//!    occupancy (packed RGBA8 in a storage buffer — portable, no 3D storage-texture
//!    feature). v1 voxelizes rigid mesh instances **analytically** (a `MeshInstance`
//!    is a unit cube transformed by its TRS → a box; a voxel is solid when its
//!    centre maps inside the box in instance-local space). Terrain/skinned
//!    voxelization is a documented follow-up.
//! 2. **March** [`PROBE_DIMS`] probes (16×8×16) across the same volume: each casts
//!    `rays` fixed golden-spiral directions; a ray that hits occupancy gathers
//!    `albedo × sun_visibility(hit)` (single bounce; sun visibility is a second
//!    march toward the sun), a ray that misses gathers a sky-gradient radiance. The
//!    result is projected to **L1 SH** (4 coeffs × RGB) per probe.
//! 3. **Sample** in the lit passes: the ambient term becomes the trilinearly
//!    probe-interpolated `SH-evaluate(normal)` (× intensity × SSAO), replacing the
//!    hemispheric constant when GI is on.

use glam::Vec3;

/// Voxel grid resolution per axis (64³ volume).
pub const GI_DIM: u32 = 64;
/// Probe grid dimensions `[x, y, z]` (16×8×16 = 2048 probes). Fewer probes
/// vertically since scenes are wider than tall.
pub const PROBE_DIMS: [u32; 3] = [16, 8, 16];

/// Total probe count.
pub const fn probe_count() -> u32 {
    PROBE_DIMS[0] * PROBE_DIMS[1] * PROBE_DIMS[2]
}

/// Flat probe index for grid coordinate `(x, y, z)` (`x` fastest). Mirrors the
/// compute shader's `probe_index`.
pub fn probe_index(x: u32, y: u32, z: u32) -> u32 {
    (z * PROBE_DIMS[1] + y) * PROBE_DIMS[0] + x
}

/// Flat voxel index for `(x, y, z)` in the [`GI_DIM`]³ grid (`x` fastest). Mirrors
/// the compute shader's `voxel_index`.
pub fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    (z * GI_DIM + y) * GI_DIM + x
}

/// The `i`-th of `n` golden-spiral (Fibonacci-sphere) unit directions — an even,
/// deterministic spread over the sphere with no temporal jitter (v1 determinism).
/// Mirrors `spiral_dir` in `shaders/gi_probes.wgsl`.
pub fn golden_spiral_dir(i: u32, n: u32) -> Vec3 {
    let n = n.max(1) as f32;
    let i = i as f32;
    // Golden angle.
    let phi = std::f32::consts::PI * (3.0 - (5.0_f32).sqrt());
    let y = 1.0 - 2.0 * (i + 0.5) / n; // (-1, 1)
    let r = (1.0 - y * y).max(0.0).sqrt();
    let theta = phi * i;
    Vec3::new(theta.cos() * r, y, theta.sin() * r)
}

/// Real L1 spherical-harmonic basis evaluated in direction `d`
/// `[Y₀₀, Y₁₋₁, Y₁₀, Y₁₁]` = `[0.282095, 0.488603·y, 0.488603·z, 0.488603·x]`.
/// Mirrors `sh_basis` in the shader.
pub fn sh_l1_basis(d: Vec3) -> [f32; 4] {
    [0.282095, 0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x]
}

/// The render-local minimum corner of the camera-centred volume: `eye − extent/2`
/// on every axis (the voxel grid and the probe grid share it).
pub fn volume_min(eye_local: Vec3, extent: f32) -> Vec3 {
    eye_local - Vec3::splat(extent * 0.5)
}

/// World size of one voxel (`extent / GI_DIM`).
pub fn voxel_size(extent: f32) -> f32 {
    extent / GI_DIM as f32
}

/// Render-local position of probe `(x, y, z)`: probes sit at the **cell corners*
/// spanning the whole extent (so the outermost probes are on the volume faces).
pub fn probe_position(x: u32, y: u32, z: u32, vol_min: Vec3, extent: f32) -> Vec3 {
    let frac = |i: u32, n: u32| {
        if n <= 1 {
            0.5
        } else {
            i as f32 / (n - 1) as f32
        }
    };
    vol_min
        + Vec3::new(
            frac(x, PROBE_DIMS[0]),
            frac(y, PROBE_DIMS[1]),
            frac(z, PROBE_DIMS[2]),
        ) * extent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spiral_dirs_are_unit_and_spread() {
        let n = 48;
        let mut sum = Vec3::ZERO;
        for i in 0..n {
            let d = golden_spiral_dir(i, n);
            assert!((d.length() - 1.0).abs() < 1e-4, "dir {i} not unit");
            sum += d;
        }
        // An even sphere cover roughly cancels out.
        assert!(sum.length() / (n as f32) < 0.15, "not well spread: {sum:?}");
    }

    #[test]
    fn spiral_is_deterministic() {
        for i in 0..48 {
            assert_eq!(golden_spiral_dir(i, 48), golden_spiral_dir(i, 48));
        }
    }

    #[test]
    fn sh_basis_constant_term() {
        // The l=0 term is direction-independent.
        assert!((sh_l1_basis(Vec3::X)[0] - 0.282095).abs() < 1e-6);
        assert!((sh_l1_basis(Vec3::Y)[0] - 0.282095).abs() < 1e-6);
        // l=1 terms pick up the matching axis.
        assert!((sh_l1_basis(Vec3::Y)[1] - 0.488603).abs() < 1e-6);
        assert!((sh_l1_basis(Vec3::X)[3] - 0.488603).abs() < 1e-6);
    }

    #[test]
    fn probe_and_voxel_indices_are_bounded_and_unique() {
        // Probe corners span the whole grid, endpoints on the faces.
        let vmin = Vec3::ZERO;
        let p0 = probe_position(0, 0, 0, vmin, 40.0);
        let p1 = probe_position(
            PROBE_DIMS[0] - 1,
            PROBE_DIMS[1] - 1,
            PROBE_DIMS[2] - 1,
            vmin,
            40.0,
        );
        assert!(p0.abs_diff_eq(Vec3::ZERO, 1e-4));
        assert!(p1.abs_diff_eq(Vec3::splat(40.0), 1e-4));

        assert_eq!(probe_index(0, 0, 0), 0);
        assert_eq!(probe_count(), 16 * 8 * 16);
        // x is the fastest axis.
        assert_eq!(probe_index(1, 0, 0), 1);
        assert_eq!(probe_index(0, 1, 0), PROBE_DIMS[0]);
        assert_eq!(voxel_index(1, 0, 0), 1);
        assert_eq!(voxel_index(0, 0, 1), GI_DIM * GI_DIM);
    }

    #[test]
    fn volume_is_camera_centred() {
        let eye = Vec3::new(5.0, 2.0, -3.0);
        let vmin = volume_min(eye, 40.0);
        // The eye sits at the volume centre.
        assert!((vmin + Vec3::splat(20.0)).abs_diff_eq(eye, 1e-4));
        assert!((voxel_size(40.0) - 40.0 / 64.0).abs() < 1e-6);
    }
}
