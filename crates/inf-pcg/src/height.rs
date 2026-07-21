//! The terrain seam the PCG runtime samples against.
//!
//! Scattering needs to ask the terrain two questions at a world `(x, z)`: what is
//! the ground height, and what is the surface normal? Both are optional — a query
//! outside the terrain's authored extent returns `None`, and the scatter kernel
//! skips such candidates.
//!
//! ## Why a local trait (the seam)
//!
//! `inf-terrain` is being built concurrently and defines its own
//! `trait HeightSource { fn height(&self, x, z) -> Option<f64>;
//! fn normal(&self, x, z) -> Option<DVec3>; }` — the **same shape** as this
//! trait. This crate deliberately does **not** depend on `inf-terrain` yet, so it
//! defines its own [`HeightProvider`]. The one-line bridge the orchestrator adds
//! next batch (in `inf-pcg`, which owns this trait, once `inf-terrain` is a dep)
//! is a blanket impl:
//!
//! ```ignore
//! impl<T: inf_terrain::HeightSource + ?Sized> HeightProvider for T {
//!     fn height(&self, x: f64, z: f64) -> Option<f64> {
//!         inf_terrain::HeightSource::height(self, x, z)
//!     }
//!     fn normal(&self, x: f64, z: f64) -> Option<glam::DVec3> {
//!         inf_terrain::HeightSource::normal(self, x, z)
//!     }
//! }
//! ```
//!
//! Until then, samplers/scatter are tested against procedural height functions
//! (see [`FnHeight`] and the `sine_hills` test fixture).

use glam::DVec3;

/// A read-only terrain height/normal field. `Send + Sync` so it can back a
/// `DensityField` evaluated in parallel over scatter cells.
pub trait HeightProvider: Send + Sync {
    /// Ground height at world `(x, z)`, or `None` outside the terrain extent.
    fn height(&self, x: f64, z: f64) -> Option<f64>;
    /// Unit surface normal at world `(x, z)`, or `None` outside the extent.
    fn normal(&self, x: f64, z: f64) -> Option<DVec3>;
}

/// A reference to a provider is itself a provider (lets `&dyn HeightProvider`
/// satisfy the trait, so the sampler tree can borrow one provider).
impl<H: HeightProvider + ?Sized> HeightProvider for &H {
    fn height(&self, x: f64, z: f64) -> Option<f64> {
        (**self).height(x, z)
    }
    fn normal(&self, x: f64, z: f64) -> Option<DVec3> {
        (**self).normal(x, z)
    }
}

/// An analytic height field defined by a closure `(x, z) -> Option<f64>`, with
/// the normal computed by central differences. Handy for tests, previews, and
/// any purely procedural terrain. The closure must be `Send + Sync`.
pub struct FnHeight<F> {
    f: F,
    /// Central-difference step (world units) for the numerical normal.
    eps: f64,
}

impl<F> FnHeight<F>
where
    F: Fn(f64, f64) -> Option<f64> + Send + Sync,
{
    /// Wrap `f` with the default 0.1-unit normal step.
    pub fn new(f: F) -> Self {
        Self { f, eps: 0.1 }
    }

    /// Wrap `f` with a custom central-difference step.
    pub fn with_eps(f: F, eps: f64) -> Self {
        Self { f, eps }
    }
}

impl<F> HeightProvider for FnHeight<F>
where
    F: Fn(f64, f64) -> Option<f64> + Send + Sync,
{
    fn height(&self, x: f64, z: f64) -> Option<f64> {
        (self.f)(x, z)
    }

    fn normal(&self, x: f64, z: f64) -> Option<DVec3> {
        let e = self.eps;
        let hx0 = (self.f)(x - e, z)?;
        let hx1 = (self.f)(x + e, z)?;
        let hz0 = (self.f)(x, z - e)?;
        let hz1 = (self.f)(x, z + e)?;
        // Tangents: dP/dx = (2e, hx1-hx0, 0), dP/dz = (0, hz1-hz0, 2e).
        // Normal = normalize(dP/dz × dP/dx) → points +Y for flat ground.
        let dhdx = (hx1 - hx0) / (2.0 * e);
        let dhdz = (hz1 - hz0) / (2.0 * e);
        Some(DVec3::new(-dhdx, 1.0, -dhdz).normalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_ground_normal_points_up() {
        let h = FnHeight::new(|_, _| Some(5.0));
        assert_eq!(h.height(1.0, 2.0), Some(5.0));
        let n = h.normal(1.0, 2.0).unwrap();
        assert!((n - DVec3::Y).length() < 1e-9, "n={n:?}");
    }

    #[test]
    fn slope_tilts_normal_away_from_up() {
        // A 45° ramp in +x: height = x.
        let h = FnHeight::new(|x, _| Some(x));
        let n = h.normal(3.0, 0.0).unwrap();
        // Normal must lean in −x and stay unit length.
        assert!(n.x < 0.0);
        assert!((n.length() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn out_of_extent_is_none() {
        let h = FnHeight::new(|x, _| if x >= 0.0 { Some(0.0) } else { None });
        assert!(h.height(-1.0, 0.0).is_none());
        // Normal needs all four neighbours; a None neighbour propagates.
        assert!(h.normal(0.0, 0.0).is_none());
    }
}
