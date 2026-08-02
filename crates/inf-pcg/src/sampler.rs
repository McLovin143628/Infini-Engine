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
//! * **terrain-layer masks** (P19.3) — [`DataMapMask`] over the P19.1 erosion
//!   maps and [`BiomeMask`] over the P19.2 painted ids (each reads a
//!   [`TerrainFields`]);
//! * **combinators** — [`Multiply`], [`Max`], [`Min`], [`Invert`].
//!
//! All filters use a **feather**: a soft ramp of the given width on each side of
//! the accepted band (via [`smoothstep`]) so edges are not a hard cutoff. A
//! feather of `0` gives a crisp step.

use glam::DVec3;

use crate::fields::TerrainFields;
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

/// A **normalized** read of one P19.1 erosion data map (P19.3).
///
/// The stored accumulators are raw and monotone — flow in m³, deposition and wear
/// in metres — so a `[0, 1]` density has to name the window it divides by. That
/// window lives on the sampler (and on the `mask.*` node), never on the terrain:
/// the P19.1 doctrine is that *normalization is a view, never the storage*, and
/// two masks over the same terrain may legitimately want different windows.
///
/// `min` maps to `0`, `max` maps to `1`, and the result is clamped, so a mask is
/// a linear window, not a threshold. A degenerate window (`max <= min`) becomes a
/// hard step at `min` — the honest reading of "everything above here". Positions
/// off the terrain score `0`.
pub struct DataMapMask<F> {
    pub fields: F,
    pub kind: inf_terrain::DataMapKind,
    /// Raw value mapping to density `0`.
    pub min: f64,
    /// Raw value mapping to density `1`.
    pub max: f64,
}

impl<F: TerrainFields> DensityField for DataMapMask<F> {
    fn density(&self, x: f64, z: f64) -> f64 {
        match self.fields.data_map(self.kind, x, z) {
            Some(v) => {
                if self.max <= self.min {
                    return if v >= self.min { 1.0 } else { 0.0 };
                }
                ((v - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
            }
            None => 0.0,
        }
    }
}

/// Cap on how many terrain samples the [`BiomeMask`] feather search may walk per
/// axis, in each direction.
///
/// The search is `O(k²)` probes per candidate in the worst case (a point deep
/// inside a biome finds nothing and scans the whole disc), so the radius is
/// bounded rather than trusted: a feather wider than `MAX_FEATHER_SAMPLES ·
/// spacing` saturates instead of costing quadratically more. At the 1–2 m
/// spacings terrain actually ships with, 64 samples is a 64–128 m blend — far
/// past any authored border.
pub const MAX_FEATHER_SAMPLES: i32 = 64;

/// A **biome-id membership mask** (P19.3): `1.0` where the terrain is painted
/// with `id`, `0.0` everywhere else, ramped across `feather` metres of border.
///
/// # The soft consumer P19.2 promised
///
/// P19.2 stores biome ids **crisply** on purpose — an id is categorical, and
/// feathering the storage would throw away exactly the information a blend needs
/// (the ROADMAP P19.2 block argues this at length). This is the other half: the
/// consumer feathers, reading the crisp ids.
///
/// # How the ramp is computed
///
/// Membership is a step function on the sample lattice, so the ramp is driven by
/// **distance to the nearest unlike sample** — the closest lattice point whose id
/// is not `id`, including *off-terrain* (a terrain edge is a border like any
/// other). That distance is found by an expanding ring search over the lattice,
/// stopped as soon as no further ring can beat the best hit, and capped at
/// [`MAX_FEATHER_SAMPLES`].
///
/// The boundary itself lies about **half a sample** inside the nearest unlike
/// point, so that half-spacing is subtracted before the ramp — otherwise every
/// mask would read `smoothstep(spacing)` right at its own edge instead of `0`.
/// The ramp is [`smoothstep`], so the blend is C¹ and **monotone in distance**:
/// deeper inside is never less dense.
///
/// `feather <= 0` is the crisp mask — no search at all, and the common case.
pub struct BiomeMask<F> {
    pub fields: F,
    /// The biome id this mask selects. Never
    /// [`UNASSIGNED_BIOME`](inf_terrain::UNASSIGNED_BIOME) in a binding (id `0`
    /// scatters nothing), but a graph may name it and it behaves like any other.
    pub id: u8,
    /// Border blend width in **metres**. `0` (or less) is a hard edge.
    pub feather: f64,
}

impl<F: TerrainFields> BiomeMask<F> {
    /// Distance in metres to the nearest lattice sample whose id is not
    /// [`id`](Self::id) (off-terrain counts), or `None` when none exists within
    /// `radius` samples. Squared distances are compared so the search does one
    /// `sqrt` at the end and stays exact.
    fn nearest_unlike(&self, x: f64, z: f64, radius: i32, spacing: f64) -> Option<f64> {
        let mut best_sq = i64::MAX;
        for r in 1..=radius {
            // Every sample on ring `r` is at least `r · spacing` away, so once the
            // best hit is that close no further ring can improve on it.
            if (r as i64) * (r as i64) >= best_sq {
                break;
            }
            for dj in -r..=r {
                for di in -r..=r {
                    // Ring only — the interior was scanned by earlier rounds.
                    if di.abs() != r && dj.abs() != r {
                        continue;
                    }
                    let d_sq = (di as i64) * (di as i64) + (dj as i64) * (dj as i64);
                    if d_sq >= best_sq {
                        continue;
                    }
                    let px = x + di as f64 * spacing;
                    let pz = z + dj as f64 * spacing;
                    if self.fields.biome_id(px, pz) != Some(self.id) {
                        best_sq = d_sq;
                    }
                }
            }
        }
        (best_sq != i64::MAX).then(|| (best_sq as f64).sqrt() * spacing)
    }
}

impl<F: TerrainFields> DensityField for BiomeMask<F> {
    fn density(&self, x: f64, z: f64) -> f64 {
        if self.fields.biome_id(x, z) != Some(self.id) {
            return 0.0;
        }
        if self.feather <= 0.0 {
            return 1.0;
        }
        let spacing = self.fields.sample_spacing();
        if spacing <= 0.0 || spacing.is_nan() {
            return 1.0;
        }
        let radius = ((self.feather / spacing).ceil() as i32).clamp(1, MAX_FEATHER_SAMPLES);
        match self.nearest_unlike(x, z, radius, spacing) {
            // The border sits half a sample inside the nearest unlike point.
            Some(d) => smoothstep(0.0, self.feather, (d - 0.5 * spacing).max(0.0)),
            // Nothing unlike within the capped radius ⇒ fully interior.
            None => 1.0,
        }
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

    // ── P19.3 terrain-layer masks ───────────────────────────────────────────

    /// A synthetic layer source with **known values**, so the mask maths is
    /// pinned without dragging a whole terrain in.
    struct Synth {
        /// `biome_id(x, z)` = `id_at(x)`; `None` outside `[0, 32)`.
        spacing: f64,
    }

    impl TerrainFields for Synth {
        fn data_map(&self, kind: inf_terrain::DataMapKind, x: f64, _z: f64) -> Option<f64> {
            if !(0.0..32.0).contains(&x) {
                return None;
            }
            Some(match kind {
                inf_terrain::DataMapKind::Flow => x * 100.0,
                inf_terrain::DataMapKind::Deposition => 4.0,
                inf_terrain::DataMapKind::Wear => 0.0,
            })
        }
        fn biome_id(&self, x: f64, _z: f64) -> Option<u8> {
            // Two half-planes meeting at x = 16, on a 1 m lattice.
            (0.0..32.0)
                .contains(&x)
                .then_some(if x < 16.0 { 1 } else { 2 })
        }
        fn sample_spacing(&self) -> f64 {
            self.spacing
        }
    }

    /// The data-map mask is a **linear window over raw values**, clamped, and it
    /// scores `0` off the terrain — the P19.1 "normalization is a view" rule made
    /// concrete.
    #[test]
    fn data_map_mask_normalizes_a_raw_window() {
        let f = Synth { spacing: 1.0 };
        let m = DataMapMask {
            fields: &f,
            kind: inf_terrain::DataMapKind::Flow,
            min: 500.0,
            max: 1500.0,
        };
        // Raw flow at x is 100x: 5 → 500 (the window floor), 10 → 1000 (half),
        // 15 → 1500 (the ceiling), 20 → 2000 (clamped).
        assert_eq!(m.density(5.0, 0.0), 0.0);
        assert!((m.density(10.0, 0.0) - 0.5).abs() < 1e-12);
        assert_eq!(m.density(15.0, 0.0), 1.0);
        assert_eq!(m.density(20.0, 0.0), 1.0, "above the window clamps to 1");
        assert_eq!(m.density(2.0, 0.0), 0.0, "below the window clamps to 0");
        // Off the authored extent scores 0 (fails closed).
        assert_eq!(m.density(-1.0, 0.0), 0.0);
        assert_eq!(m.density(99.0, 0.0), 0.0);
        // A different channel reads a different raw value through the same node.
        let dep = DataMapMask {
            fields: &f,
            kind: inf_terrain::DataMapKind::Deposition,
            min: 0.0,
            max: 8.0,
        };
        assert!((dep.density(1.0, 0.0) - 0.5).abs() < 1e-12);
        // A degenerate window is a hard step at `min`, not a divide by zero.
        let step = DataMapMask {
            fields: &f,
            kind: inf_terrain::DataMapKind::Flow,
            min: 1000.0,
            max: 1000.0,
        };
        assert_eq!(step.density(9.0, 0.0), 0.0);
        assert_eq!(step.density(10.0, 0.0), 1.0);
        // With no layer source at all, a mask places nothing.
        let none = DataMapMask {
            fields: crate::fields::NoFields,
            kind: inf_terrain::DataMapKind::Flow,
            min: 0.0,
            max: 1.0,
        };
        assert_eq!(none.density(10.0, 0.0), 0.0);
    }

    /// The biome mask is crisp at `feather = 0`, feathers to a **monotone** ramp
    /// otherwise, and treats the terrain edge as a border.
    #[test]
    fn biome_mask_is_crisp_then_feathers_monotonically() {
        let f = Synth { spacing: 1.0 };
        let crisp = BiomeMask {
            fields: &f,
            id: 1,
            feather: 0.0,
        };
        assert_eq!(crisp.density(0.0, 0.0), 1.0);
        assert_eq!(
            crisp.density(15.0, 0.0),
            1.0,
            "hard edge right at the border"
        );
        assert_eq!(crisp.density(16.0, 0.0), 0.0);
        assert_eq!(crisp.density(-1.0, 0.0), 0.0, "off-terrain is outside");

        let soft = BiomeMask {
            fields: &f,
            id: 1,
            feather: 4.0,
        };
        // Outside is still exactly 0 — the feather never leaks across the line.
        assert_eq!(soft.density(16.0, 0.0), 0.0);
        assert_eq!(soft.density(20.0, 0.0), 0.0);
        // Deep inside (further than the feather from BOTH borders) is exactly 1.
        assert_eq!(soft.density(8.0, 0.0), 1.0);
        // The band ramps monotonically from the border inward.
        let mut prev = -1.0;
        for step in 0..=10 {
            let x = 15.0 - step as f64 * 0.5;
            let d = soft.density(x, 0.0);
            assert!(d >= prev - 1e-12, "fell at x={x}: {d} < {prev}");
            assert!((0.0..=1.0).contains(&d));
            prev = d;
        }
        // The nearest unlike sample to x = 15 is x = 16, one spacing away, and
        // the border sits half a spacing in ⇒ the ramp is barely off zero there.
        assert!(
            soft.density(15.0, 0.0) < 0.05,
            "{}",
            soft.density(15.0, 0.0)
        );
        // The terrain's own edge feathers too (x = 0 is one sample from nothing).
        assert!(soft.density(0.0, 0.0) < 0.05);
        // A mask for an id nobody painted is empty everywhere.
        let absent = BiomeMask {
            fields: &f,
            id: 9,
            feather: 4.0,
        };
        assert_eq!(absent.density(8.0, 0.0), 0.0);
        // With no layer source, nothing is inside anything.
        let none = BiomeMask {
            fields: crate::fields::NoFields,
            id: 1,
            feather: 4.0,
        };
        assert_eq!(none.density(0.0, 0.0), 0.0);
    }

    /// The feather is in **metres**, so a coarser lattice must not change where
    /// the ramp lands — and the search radius is capped, not trusted.
    #[test]
    fn the_feather_is_metric_and_the_search_radius_is_capped() {
        // The same 16 m border and the same 4 m feather, sampled on a 1 m and a
        // 2 m lattice. The blend is a *world-space* width, so both saturate by
        // 4 m in and both are mid-ramp 1.5 m in — only the number of probe rings
        // differs. (The values inside the band are not identical: a coarser
        // lattice estimates the border half a coarser sample in, which is the
        // honest resolution limit, not a unit bug.)
        for spacing in [1.0, 2.0] {
            let m = BiomeMask {
                fields: &Synth { spacing },
                id: 1,
                feather: 4.0,
            };
            assert_eq!(m.density(20.0, 0.0), 0.0, "outside, spacing {spacing}");
            assert_eq!(
                m.density(11.0, 0.0),
                1.0,
                "4 m inside must saturate at spacing {spacing}"
            );
            let band = m.density(14.0, 0.0);
            assert!(
                band > 0.0 && band < 1.0,
                "1.5 m inside must be mid-ramp at spacing {spacing}, got {band}"
            );
        }
        // An absurd feather saturates at the cap instead of scanning forever.
        let huge = BiomeMask {
            fields: &Synth { spacing: 1.0 },
            id: 1,
            feather: 1.0e9,
        };
        // Everything inside is < 1 (nothing is 1e9 m from a border) but finite
        // and computed — the cap bounds the work, it does not change the answer's
        // shape.
        let d = huge.density(8.0, 0.0);
        assert!((0.0..1.0).contains(&d), "d={d}");
        assert_eq!(MAX_FEATHER_SAMPLES, 64);
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
