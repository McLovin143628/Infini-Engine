//! Density fields: the scalar `[0, 1]` weight that drives where scattering may
//! place instances.
//!
//! A [`DensityField`] answers `density(x, z) -> [0, 1]` at any world position.
//! The scatter kernel treats that value as an **acceptance probability**, so a
//! field of `0.0` places nothing and a field of `1.0` fills every candidate slot.
//! Fields are pure and composable:
//!
//! * **sources** — [`Constant`], [`Noise`], [`MaskImage`];
//! * **terrain filters** — [`SlopeFilter`], [`AltitudeFilter`] (each reads a
//!   [`HeightProvider`]);
//! * **combinators** — [`Multiply`], [`Max`], [`Min`], [`Invert`].
//!
//! All filters use a **feather**: a soft ramp of the given width on each side of
//! the accepted band (via [`smoothstep`]) so edges are not a hard cutoff. A
//! feather of `0` gives a crisp step.

use glam::DVec3;

use crate::height::HeightProvider;
use crate::noise::ValueNoise;

/// A pure scalar field over the world XZ plane, valued in `[0, 1]`. `Send + Sync`
/// so a field can be evaluated in parallel across scatter cells.
pub trait DensityField: Send + Sync {
    /// The density at world `(x, z)`, clamped to `[0, 1]`.
    fn density(&self, x: f64, z: f64) -> f64;
}

/// A boxed field is a field (lets the [`SamplerDef`](crate::rules::SamplerDef)
/// tree build a `Box<dyn DensityField>`).
impl<T: DensityField + ?Sized> DensityField for Box<T> {
    fn density(&self, x: f64, z: f64) -> f64 {
        (**self).density(x, z)
    }
}

/// A reference to a field is a field.
impl<T: DensityField + ?Sized> DensityField for &T {
    fn density(&self, x: f64, z: f64) -> f64 {
        (**self).density(x, z)
    }
}

/// `smoothstep(edge0, edge1, x)` — 0 below `edge0`, 1 above `edge1`, a C¹ Hermite
/// ramp between. Degenerate (`edge0 == edge1`) becomes a hard step.
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A feathered band window: `1.0` for `value ∈ [min, max]`, ramping to `0.0`
/// across `feather` on each side. `feather <= 0` gives a hard `[min, max]` gate.
pub fn feather_window(value: f64, min: f64, max: f64, feather: f64) -> f64 {
    if feather <= 0.0 {
        return if value >= min && value <= max {
            1.0
        } else {
            0.0
        };
    }
    let lo = smoothstep(min - feather, min, value);
    let hi = 1.0 - smoothstep(max, max + feather, value);
    lo.min(hi).clamp(0.0, 1.0)
}

/// A uniform density everywhere (clamped into `[0, 1]`).
#[derive(Debug, Clone, Copy)]
pub struct Constant(pub f64);

impl DensityField for Constant {
    fn density(&self, _x: f64, _z: f64) -> f64 {
        self.0.clamp(0.0, 1.0)
    }
}

/// fBm value-noise density (see [`ValueNoise`]).
#[derive(Debug, Clone, Copy)]
pub struct Noise(pub ValueNoise);

impl DensityField for Noise {
    fn density(&self, x: f64, z: f64) -> f64 {
        self.0.sample(x, z)
    }
}

/// Accepts only where the terrain slope (degrees from horizontal) falls in
/// `[min_deg, max_deg]`, feathered by `feather_deg`. Off-terrain positions
/// (`normal` is `None`) score `0`.
pub struct SlopeFilter<H> {
    pub height: H,
    pub min_deg: f64,
    pub max_deg: f64,
    pub feather_deg: f64,
}

impl<H: HeightProvider> DensityField for SlopeFilter<H> {
    fn density(&self, x: f64, z: f64) -> f64 {
        match self.height.normal(x, z) {
            Some(n) => {
                let up_dot = n.normalize().dot(DVec3::Y).clamp(-1.0, 1.0);
                let slope_deg = up_dot.acos().to_degrees();
                feather_window(slope_deg, self.min_deg, self.max_deg, self.feather_deg)
            }
            None => 0.0,
        }
    }
}

/// Accepts only where the terrain height falls in `[min, max]`, feathered by
/// `feather`. Off-terrain positions score `0`.
pub struct AltitudeFilter<H> {
    pub height: H,
    pub min: f64,
    pub max: f64,
    pub feather: f64,
}

impl<H: HeightProvider> DensityField for AltitudeFilter<H> {
    fn density(&self, x: f64, z: f64) -> f64 {
        match self.height.height(x, z) {
            Some(h) => feather_window(h, self.min, self.max, self.feather),
            None => 0.0,
        }
    }
}

/// A grayscale bitmap mask stretched over a world rectangle, bilinearly sampled.
/// Bytes are row-major (`data[row * width + col]`, row index increasing with
/// `z`); each byte maps `0..=255` → `0.0..=1.0`. Positions outside the rectangle
/// score `0`.
#[derive(Debug, Clone)]
pub struct MaskImage {
    /// World rect as `[min_x, min_z, max_x, max_z]`.
    pub rect: [f64; 4],
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl MaskImage {
    #[inline]
    fn texel(&self, col: i64, row: i64) -> f64 {
        let w = self.width as i64;
        let h = self.height as i64;
        let c = col.clamp(0, w - 1);
        let r = row.clamp(0, h - 1);
        let idx = (r * w + c) as usize;
        self.data.get(idx).copied().unwrap_or(0) as f64 / 255.0
    }
}

impl DensityField for MaskImage {
    fn density(&self, x: f64, z: f64) -> f64 {
        let [min_x, min_z, max_x, max_z] = self.rect;
        if self.width == 0
            || self.height == 0
            || max_x <= min_x
            || max_z <= min_z
            || x < min_x
            || x >= max_x
            || z < min_z
            || z >= max_z
        {
            return 0.0;
        }
        // World → continuous texel coordinates (pixel centres at integers).
        let fu = (x - min_x) / (max_x - min_x) * (self.width as f64 - 1.0);
        let fv = (z - min_z) / (max_z - min_z) * (self.height as f64 - 1.0);
        let u0 = fu.floor();
        let v0 = fv.floor();
        let tu = fu - u0;
        let tv = fv - v0;
        let (u0, v0) = (u0 as i64, v0 as i64);
        let c00 = self.texel(u0, v0);
        let c10 = self.texel(u0 + 1, v0);
        let c01 = self.texel(u0, v0 + 1);
        let c11 = self.texel(u0 + 1, v0 + 1);
        let top = c00 + (c10 - c00) * tu;
        let bot = c01 + (c11 - c01) * tu;
        (top + (bot - top) * tv).clamp(0.0, 1.0)
    }
}

/// Product of two fields (`a · b`) — the intersection combinator.
pub struct Multiply<A, B>(pub A, pub B);

impl<A: DensityField, B: DensityField> DensityField for Multiply<A, B> {
    fn density(&self, x: f64, z: f64) -> f64 {
        (self.0.density(x, z) * self.1.density(x, z)).clamp(0.0, 1.0)
    }
}

/// Maximum of two fields (`max(a, b)`) — the union combinator.
pub struct Max<A, B>(pub A, pub B);

impl<A: DensityField, B: DensityField> DensityField for Max<A, B> {
    fn density(&self, x: f64, z: f64) -> f64 {
        self.0.density(x, z).max(self.1.density(x, z))
    }
}

/// Minimum of two fields (`min(a, b)`).
pub struct Min<A, B>(pub A, pub B);

impl<A: DensityField, B: DensityField> DensityField for Min<A, B> {
    fn density(&self, x: f64, z: f64) -> f64 {
        self.0.density(x, z).min(self.1.density(x, z))
    }
}

/// Inverts a field (`1 − a`).
pub struct Invert<A>(pub A);

impl<A: DensityField> DensityField for Invert<A> {
    fn density(&self, x: f64, z: f64) -> f64 {
        (1.0 - self.0.density(x, z)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::height::FnHeight;

    #[test]
    fn constant_clamps() {
        assert_eq!(Constant(0.5).density(9.0, -3.0), 0.5);
        assert_eq!(Constant(2.0).density(0.0, 0.0), 1.0);
        assert_eq!(Constant(-1.0).density(0.0, 0.0), 0.0);
    }

    #[test]
    fn feather_window_edges() {
        // Hard gate (no feather).
        assert_eq!(feather_window(5.0, 0.0, 10.0, 0.0), 1.0);
        assert_eq!(feather_window(-1.0, 0.0, 10.0, 0.0), 0.0);
        // Feathered: exactly at the band edges → 1, at the outer feather → 0,
        // halfway across the ramp → 0.5.
        assert!((feather_window(0.0, 0.0, 10.0, 2.0) - 1.0).abs() < 1e-9);
        assert!((feather_window(10.0, 0.0, 10.0, 2.0) - 1.0).abs() < 1e-9);
        assert!(feather_window(-2.0, 0.0, 10.0, 2.0).abs() < 1e-9);
        assert!((feather_window(-1.0, 0.0, 10.0, 2.0) - 0.5).abs() < 1e-9);
        assert!((feather_window(11.0, 0.0, 10.0, 2.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn slope_filter_selects_flat_ground() {
        // Flat plane → slope 0°.
        let flat = SlopeFilter {
            height: FnHeight::new(|_, _| Some(0.0)),
            min_deg: 0.0,
            max_deg: 10.0,
            feather_deg: 0.0,
        };
        assert_eq!(flat.density(1.0, 1.0), 1.0);

        // 45° ramp (height = x) → slope 45°, excluded by a [0,10]° band.
        let ramp = SlopeFilter {
            height: FnHeight::new(|x, _| Some(x)),
            min_deg: 0.0,
            max_deg: 10.0,
            feather_deg: 0.0,
        };
        assert_eq!(ramp.density(3.0, 0.0), 0.0);

        // …and accepted by a band that includes 45°.
        let ramp_ok = SlopeFilter {
            height: FnHeight::new(|x, _| Some(x)),
            min_deg: 30.0,
            max_deg: 60.0,
            feather_deg: 0.0,
        };
        assert!((ramp_ok.density(3.0, 0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn slope_filter_off_terrain_is_zero() {
        let f = SlopeFilter {
            height: FnHeight::new(|_, _| None),
            min_deg: 0.0,
            max_deg: 90.0,
            feather_deg: 0.0,
        };
        assert_eq!(f.density(0.0, 0.0), 0.0);
    }

    #[test]
    fn altitude_filter_band() {
        // Height ramps with z; accept z in [10,20].
        let f = AltitudeFilter {
            height: FnHeight::new(|_, z| Some(z)),
            min: 10.0,
            max: 20.0,
            feather: 0.0,
        };
        assert_eq!(f.density(0.0, 5.0), 0.0);
        assert_eq!(f.density(0.0, 15.0), 1.0);
        assert_eq!(f.density(0.0, 25.0), 0.0);
    }

    #[test]
    fn mask_bilinear_and_bounds() {
        // 2×2 mask over [0,10]×[0,10]: left column 0, right column 255.
        let mask = MaskImage {
            rect: [0.0, 0.0, 10.0, 10.0],
            width: 2,
            height: 2,
            data: vec![0, 255, 0, 255],
        };
        // Left edge (col 0) → 0, right edge (col 1 at x=10 excluded → use ~9.99).
        assert!(mask.density(0.0, 0.0).abs() < 1e-9);
        assert!((mask.density(9.999, 5.0) - 1.0).abs() < 1e-2);
        // Midpoint between the columns → ~0.5.
        assert!((mask.density(5.0, 5.0) - 0.5).abs() < 0.05);
        // Outside the rect → 0.
        assert_eq!(mask.density(-1.0, 5.0), 0.0);
        assert_eq!(mask.density(11.0, 5.0), 0.0);
    }

    #[test]
    fn combinators() {
        let half = Constant(0.5);
        let quarter = Constant(0.25);
        assert!((Multiply(half, quarter).density(0.0, 0.0) - 0.125).abs() < 1e-9);
        assert_eq!(Max(half, quarter).density(0.0, 0.0), 0.5);
        assert_eq!(Min(half, quarter).density(0.0, 0.0), 0.25);
        assert_eq!(Invert(quarter).density(0.0, 0.0), 0.75);
    }
}
