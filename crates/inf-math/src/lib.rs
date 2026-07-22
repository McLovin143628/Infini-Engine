//! Math: f64 world / f32 render split, floating-origin rebasing (glam-based).
//!
//! Architecture rule 3 (docs/ROADMAP.md, CLAUDE.md): world-space positions are
//! `DVec3`; the GPU only ever sees f32 coordinates *relative to a floating
//! origin* that follows the camera. At 1 km from the origin an f32 has ~0.06 mm
//! of resolution — plenty — while raw f32 world coordinates at 100 km would
//! jitter by ~8 mm per ULP. The origin snaps to a coarse grid so the grid
//! shader (1 m / 10 m lines) stays exactly aligned across rebases.

pub use glam;

pub mod portable;
pub use portable::{pcos, pcos64, psin, psin64};

pub mod spline;
pub use spline::{
    arc_length_lut, eval as eval_spline, eval_at_distance, eval_catmull_rom, eval_linear,
    lut_length, ArcLenSample, SplineInterp,
};

use glam::{DVec3, Mat4, Quat, Vec3};

/// Rebase when the focus (camera) strays further than this from the origin.
/// At 1024 m the render-local f32 error is still far below one pixel.
pub const REBASE_DISTANCE: f64 = 1024.0;

/// Origins snap to multiples of this (metres). A multiple of both editor grid
/// spacings (1 m and 10 m) so grid lines land on identical local coordinates
/// before and after a rebase — no visual crawl.
pub const ORIGIN_SNAP: f64 = 10.0;

/// The floating origin: a snapped world-space anchor near the camera. All
/// render-local coordinates are `world - origin`, cast to f32.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FloatingOrigin {
    origin: DVec3,
}

impl FloatingOrigin {
    /// Create an origin snapped to [`ORIGIN_SNAP`] near `anchor`.
    pub fn new(anchor: DVec3) -> Self {
        Self {
            origin: snap(anchor),
        }
    }

    pub fn origin(&self) -> DVec3 {
        self.origin
    }

    /// World → render-local (f32). The subtraction happens in f64.
    pub fn to_render(&self, world: DVec3) -> Vec3 {
        (world - self.origin).as_vec3()
    }

    /// Render-local (f32) → world.
    pub fn to_world(&self, local: Vec3) -> DVec3 {
        self.origin + local.as_dvec3()
    }

    /// Re-anchor near `focus` if it drifted past [`REBASE_DISTANCE`].
    /// Returns `true` when the origin moved (GPU-side transforms must be
    /// re-uploaded against the new origin).
    pub fn maybe_rebase(&mut self, focus: DVec3) -> bool {
        if (focus - self.origin).length() <= REBASE_DISTANCE {
            return false;
        }
        let next = snap(focus);
        if next == self.origin {
            return false;
        }
        self.origin = next;
        true
    }

    /// Model matrix for a world-space TRS, expressed in render-local space:
    /// rotation/scale are f32-safe as-is, only the translation needs the f64
    /// subtraction against the origin.
    pub fn model_matrix(&self, translation: DVec3, rotation: Quat, scale: Vec3) -> Mat4 {
        Mat4::from_scale_rotation_translation(scale, rotation, self.to_render(translation))
    }
}

fn snap(v: DVec3) -> DVec3 {
    (v / ORIGIN_SNAP).round() * ORIGIN_SNAP
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_snaps_to_grid() {
        let o = FloatingOrigin::new(DVec3::new(1234.9, -7.2, 55.1));
        let origin = o.origin();
        for c in origin.to_array() {
            assert_eq!(c % ORIGIN_SNAP, 0.0, "origin {origin:?} not snapped");
        }
        assert!((origin - DVec3::new(1230.0, -10.0, 60.0)).length() < 1e-9);
    }

    #[test]
    fn round_trip_is_exact_near_origin() {
        let o = FloatingOrigin::new(DVec3::new(1_000_000.0, 0.0, -2_500_000.0));
        let world = o.origin() + DVec3::new(12.5, 3.25, -700.125);
        // Sub-metre offsets with power-of-two fractions survive the f32 cast.
        assert_eq!(o.to_world(o.to_render(world)), world);
    }

    #[test]
    fn far_from_true_origin_precision_holds() {
        // 10 000 km out: raw f32 world coords would quantize to ~1 m steps;
        // render-local coords must still resolve millimetres.
        let anchor = DVec3::new(1e7, 0.0, 1e7);
        let o = FloatingOrigin::new(anchor);
        let a = o.origin() + DVec3::new(1.0, 0.0, 0.0);
        let b = o.origin() + DVec3::new(1.001, 0.0, 0.0);
        let delta = (o.to_render(b) - o.to_render(a)).length();
        assert!((delta - 0.001).abs() < 1e-6, "lost precision: {delta}");
    }

    #[test]
    fn rebase_only_past_threshold() {
        let mut o = FloatingOrigin::new(DVec3::ZERO);
        assert!(!o.maybe_rebase(DVec3::new(REBASE_DISTANCE - 1.0, 0.0, 0.0)));
        assert_eq!(o.origin(), DVec3::ZERO);
        assert!(o.maybe_rebase(DVec3::new(REBASE_DISTANCE + 50.0, 0.0, 0.0)));
        assert_eq!(o.origin().x % ORIGIN_SNAP, 0.0);
        assert!((o.origin().x - (REBASE_DISTANCE + 50.0)).abs() <= ORIGIN_SNAP);
    }

    #[test]
    fn model_matrix_matches_f64_reference() {
        let o = FloatingOrigin::new(DVec3::new(5000.0, 0.0, -3000.0));
        let t = o.origin() + DVec3::new(2.0, 1.0, -4.0);
        let r = Quat::from_rotation_y(0.7);
        let s = Vec3::splat(2.0);
        let m = o.model_matrix(t, r, s);
        let p_local = m.transform_point3(Vec3::new(0.5, 0.5, 0.5));
        // f64 reference: rotate/scale the corner, offset by world translation,
        // then subtract the origin.
        let corner = (r * (Vec3::new(0.5, 0.5, 0.5) * s)).as_dvec3();
        let expect = (t + corner - o.origin()).as_vec3();
        assert!((p_local - expect).length() < 1e-4);
    }
}
