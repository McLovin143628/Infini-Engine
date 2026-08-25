//! The terrain seam the PCG runtime samples against.
//!
//! Scattering needs to ask the terrain two questions at a world `(x, z)`: what is
//! the ground height, and what is the surface normal? Both are optional — a query
//! outside the terrain's authored extent returns `None`, and the scatter kernel
//! skips such candidates.
//!
//! ## Why a local trait (the seam)
//!
//! `inf-terrain` defines its own
//! `trait HeightSource { fn height(&self, x, z) -> Option<f64>;
//! fn normal(&self, x, z) -> Option<DVec3>; }` — the **same shape** as this
//! trait. This crate keeps its own [`HeightProvider`] (samplers/scatter code
//! against it) and bridges the two with the [`TerrainHeight`] newtype: wrap any
//! `inf_terrain::HeightSource` and it becomes a `HeightProvider`.
//!
//! A *blanket* `impl<T: HeightSource> HeightProvider for T` (as an earlier sketch
//! proposed) does not compile: it collides with the reference-forwarding
//! `impl<H: HeightProvider> HeightProvider for &H` (a fundamental-type coherence
//! conflict), and that reference impl is load-bearing for the sampler tree. See
//! [`TerrainHeight`] for the full rationale.
//!
//! Procedural height functions (see [`FnHeight`]) remain available for tests and
//! purely-procedural terrain.

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

/// Bridges an [`inf_terrain::HeightSource`] into a [`HeightProvider`] so a real
/// terrain (`inf_terrain::TerrainData`, a procedural field, …) drives scatter.
///
/// ## Why a newtype, not the blanket impl the module docs sketched
///
/// The module docs above proposed a *blanket* `impl<T: inf_terrain::HeightSource>
/// HeightProvider for T`. That does **not** compile here: it collides (E0119)
/// with [`impl<H: HeightProvider> HeightProvider for &H`](HeightProvider) — the
/// reference-forwarding impl — because `&T` is a *fundamental* type, so the
/// coherence checker must assume `inf-terrain` could later add
/// `impl<H: HeightSource> HeightSource for &H`, which would make some `&H` match
/// *both* impls. And that reference impl is load-bearing: [`SamplerDef::build`]
/// (crate::rules) feeds a `&dyn HeightProvider` straight into `SlopeFilter` /
/// `AltitudeFilter` (whose `H` must be `HeightProvider`), so dropping it to admit
/// the blanket would break the sampler tree.
///
/// (Notably the blanket does *not* conflict with the concrete
/// [`FnHeight`] impl — the compiler can prove a local type isn't a
/// `HeightSource`. Only the fundamental-`&T` case forces the conflict.)
///
/// So the bridge is this explicit newtype instead: wrap a `HeightSource` value
/// and it becomes a `HeightProvider`; share it across filters + the scatter
/// kernel by borrowing the wrapper (`&TerrainHeight<_>` is itself a
/// `HeightProvider` via the reference impl).
///
/// ```
/// use inf_pcg::height::{HeightProvider, TerrainHeight};
/// let terrain = inf_terrain::TerrainData::new(8, 1.0); // empty ⇒ height None
/// let provider = TerrainHeight(terrain);
/// assert_eq!(provider.height(0.0, 0.0), None);
/// ```
pub struct TerrainHeight<T>(pub T);

impl<T: inf_terrain::HeightSource + Send + Sync> HeightProvider for TerrainHeight<T> {
    fn height(&self, x: f64, z: f64) -> Option<f64> {
        inf_terrain::HeightSource::height(&self.0, x, z)
    }
    fn normal(&self, x: f64, z: f64) -> Option<DVec3> {
        inf_terrain::HeightSource::normal(&self.0, x, z)
    }
}

/// The default central-difference step [`FnHeight::new`] computes its numerical
/// normal with, in **world metres**.
///
/// Named rather than written as a literal because it is a *reach*: a slope query
/// at `(x, z)` reads the terrain at `x ± eps` and `z ± eps`, so anything that
/// evaluates a region tile by tile has to know how far past a tile's own box the
/// evaluation can look. See [`BiomeBinding::refresh_resident`](crate::BiomeBinding::refresh_resident).
pub const FN_HEIGHT_NORMAL_EPS: f64 = 0.1;

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
    /// Wrap `f` with the default [`FN_HEIGHT_NORMAL_EPS`] normal step.
    ///
    /// **The constant rather than the literal** (the I7b audit).
    /// [`scatter_reach_m`](crate::scatter_reach_m) adds `FN_HEIGHT_NORMAL_EPS` to
    /// the reach it bounds a per-tile evaluation by, and it was bounding a
    /// literal it did not share: editing this `0.1` would have widened how far a
    /// slope query reads without widening the neighbourhood the memo keys on,
    /// which is first sight reintroduced by a number nobody would connect to it.
    /// One door.
    pub fn new(f: F) -> Self {
        Self {
            f,
            eps: FN_HEIGHT_NORMAL_EPS,
        }
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
    use glam::DVec2;

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

    #[test]
    fn terrain_height_bridges_a_real_terrain_data() {
        // A `TerrainData` wrapped in `TerrainHeight` answers `HeightProvider`
        // queries by delegating to its `HeightSource` impl.
        let mut data = inf_terrain::TerrainData::new(8, 1.0);
        data.author_tile((0, 0), |x, z| 0.5 * x + 0.25 * z);
        let provider = TerrainHeight(data);
        // On the authored tile → Some, matching the source function.
        let h = provider.height(2.0, 2.0).expect("on authored terrain");
        assert!((h - (0.5 * 2.0 + 0.25 * 2.0)).abs() < 1e-4, "h={h}");
        assert!(provider.normal(2.0, 2.0).is_some());
        // Far outside any authored tile → None (the extent boundary).
        assert!(provider.height(1e6, 1e6).is_none());
    }

    #[test]
    fn scatter_on_real_terrain_only_in_valid_zones() {
        use crate::sampler::{AltitudeFilter, Constant, DensityField, Multiply, SlopeFilter};
        use crate::scatter::{scatter_region, Region, RotationMode, ScatterParams};

        // Author a sine-hill terrain across several tiles via `write_region`.
        let mut data = inf_terrain::TerrainData::new(16, 1.0);
        let span = data.tile_span();
        let extent = span * 3.0;
        let hill = |x: f64, z: f64| 20.0 * (x * 0.03).sin() * (z * 0.03).cos();
        data.write_region(DVec2::ZERO, DVec2::splat(extent), hill);
        let terrain = TerrainHeight(data);

        // Accept only a mid-altitude band on gentle-to-moderate slopes. Both
        // filters read the same terrain (shared by borrowing the wrapper).
        let slope = SlopeFilter {
            height: &terrain,
            min_deg: 0.0,
            max_deg: 25.0,
            feather_deg: 0.0,
        };
        let alt = AltitudeFilter {
            height: &terrain,
            min: 2.0,
            max: 12.0,
            feather: 0.0,
        };
        let density = Multiply(&slope, &alt);

        let params = ScatterParams {
            seed: 77,
            cell_size: 8.0,
            base_density: 1.0,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (1.0, 1.0),
            rotation: RotationMode::RandomYaw,
            altitude_offset: 0.0,
        };
        // Region extends past the authored footprint → some candidates are
        // off-terrain (height `None`) and must be skipped.
        let region = Region::from_xz(-5.0, -5.0, extent + 5.0, extent + 5.0);
        let out = scatter_region(&params, &density, &terrain, region);

        assert!(!out.is_empty(), "some instances land in the valid zone");
        for inst in &out {
            let (x, z) = (inst.pos.x, inst.pos.z);
            // On real, authored ground.
            let h = terrain
                .height(x, z)
                .expect("instance sits on authored terrain");
            assert!((inst.pos.y - h).abs() < 1e-9, "instance is on the ground");
            // Every placed instance satisfies BOTH filters (product was > 0).
            assert!(slope.density(x, z) > 0.0, "slope band satisfied at {x},{z}");
            assert!(
                alt.density(x, z) > 0.0,
                "altitude band satisfied at {x},{z}"
            );
            assert!(
                (2.0..=12.0).contains(&h),
                "height {h} within the altitude band"
            );
        }

        // The filters genuinely exclude: an all-pass scatter over the same terrain
        // places strictly more (the hill has out-of-band heights + steep slopes).
        let all = scatter_region(&params, &Constant(1.0), &terrain, region);
        assert!(
            all.len() > out.len(),
            "filters reject some on-terrain candidates: all={} filtered={}",
            all.len(),
            out.len()
        );
    }
}
