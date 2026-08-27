//! **The detail band** (wave TER2b) — the relief the survey cannot hold, put
//! there by design and said to be design.
//!
//! # What was missing
//!
//! The island's grid is 1 m a sample and its finest source is a z15 terrarium
//! pixel — **3.11 m** of ground at 49.34 N. Every build has printed that ratio in
//! the `source.upsampled` advisory and said, correctly, that detail below it is
//! interpolation. What it was, precisely, is *bilinear* interpolation: a raster
//! at 3.11 m a pixel cannot represent a feature shorter than **6.22 m**, so the
//! whole band from the grid's own Nyquist (2 m) up to the source's (6.22 m) was
//! not merely coarse — it was **empty**, and the ground below six metres was
//! exactly a plane through four survey samples.
//!
//! This stage fills that band with a portable fBm, and nothing else. It writes
//! **nothing above the source's Nyquist**, so no survey metre is overwritten.
//!
//! # The band, derived rather than authored
//!
//! [`DetailBand::of`] takes the two pitches the plan already carries and answers
//! the base wavelength, the octave count and the amplitude ceiling. There is no
//! recipe field for any of it, and that is a decision: the recipe schema is
//! `deny_unknown_fields` with an **exact-equality** version check and no migrate
//! function, so a v2 → v3 bump invalidates every committed recipe on disk for
//! three numbers that are a pure function of two the recipe already states.
//! The seed is the recipe's own, salted.
//!
//! The octave count is counted by **halving**, not by a logarithm: `log2` is on
//! the libm ban list and this stage writes committed bytes.
//!
//! # Where it runs, and why exactly there
//!
//! [`BuildStep::Detail`](crate::BuildStep::Detail) sits between **Roads** and
//! **Pyramid**, and that slot is the whole design:
//!
//! * The road **grade audit** and the road **mesh** are both built in the Roads
//!   step against `data.height_at`. Running after them means neither can move,
//!   whatever this stage writes.
//! * The committed **water** design is derived in Hydrology, three steps earlier,
//!   so the stream network, the lakes and their drift comparison are untouched.
//! * The **pyramid** is built from `data` immediately after, so the coarse levels
//!   see the detail rather than disagreeing with level 0.
//!
//! # And where it does NOT run
//!
//! Four exclusions, each because something downstream measures that ground:
//!
//! | excluded | why |
//! |---|---|
//! | at and below the waterline, fading in over [`SHORE_FADE_M`] | the carve puts the shore *at* the sea level and the shelf at its stated depth; both are asserted to within a metre, and a beach with a metre of noise in it is a beach that is sometimes underwater |
//! | road corridors, fading out over one corridor width | the corridor was *levelled* by the carve so the road could sit on it; the grade audit measures the 1 m ground the road sits on |
//! | stream channels, fading out over one channel width | a cut bed with noise in it is a bed the water does not follow |
//! | site pads, fading out over one radius | a terrace with bumps is not a terrace |
//!
//! Every one is a **fade**, not a cut: a hard mask edge is a visible crease, and
//! a crease along every road is worse than no detail at all.
//!
//! # Portability
//!
//! `inf_terrain::fbm_signed` hashes integers and touches only `floor`,
//! multiply, add and compare — no transcendental — and this module adds `sqrt`
//! (in the slope gradient) and `inf_math::portable::patan2_64` (the same door
//! `CoarseHeights::slope_deg` and `inf_island::splat` already go through). The
//! crate's libm-ban table covers it; `two_builds_of_one_recipe_produce_the_same_terrain`
//! is the arm that proves the whole stage is a pure function of its inputs.

use glam::DVec2;

use inf_terrain::TerrainData;

use crate::shape::{Coastline, SegmentIndex};

/// The most vertical relief the detail band may add or remove, in metres, on the
/// roughest ground it ever reaches.
///
/// A ceiling on the *modulated* amplitude, so almost nothing sees it: gentle
/// ground takes [`AMPLITUDE_FLOOR_FRACTION`] of it. 1.5 m over a 6.2 m base
/// wavelength is a broken rock face, which is what the steep end of the
/// modulation is describing.
pub const MAX_AMPLITUDE_M: f64 = 1.5;

/// What fraction of [`MAX_AMPLITUDE_M`] the flattest ground takes.
///
/// Not zero: a dead-flat plain is the one thing real ground never is, and 15 cm
/// of undulation over six metres is the difference between a meadow and a table.
pub const AMPLITUDE_FLOOR_FRACTION: f64 = 0.10;

/// Slope, in degrees, at and below which the amplitude is at its floor.
pub const SLOPE_LO_DEG: f64 = 3.0;
/// Slope, in degrees, at and above which the amplitude is at its ceiling.
pub const SLOPE_HI_DEG: f64 = 35.0;

/// Height above the waterline, in metres, over which the detail fades in.
///
/// The shoreline arm walks the coastline's own vertices and asserts every one is
/// within 2.5 m of the waterline; the beach itself is [`crate::recipe::SeaSpec`]'s
/// `beach_rise_m` tall. Six metres clears both with room, and the fade means the
/// ground does not step at the top of it.
pub const SHORE_FADE_M: f64 = 6.0;

/// How far past a masked feature's own half-width the detail fades back in, as a
/// multiple of that half-width.
///
/// One: a road corridor is fully excluded inside its width and fully detailed one
/// width outside it. Wider would erase the ground beside every road; narrower
/// leaves a ridge along it.
pub const FADE_WIDTHS: f64 = 1.0;

/// The most octaves the band may carry, whatever the pitches say.
///
/// A guard rather than a tuning knob: the octave count is derived from two
/// numbers a recipe states, and a recipe stating a 1 mm grid against a 1 km source
/// would otherwise ask for twenty octaves of noise per sample over fifty million
/// samples.
pub const MAX_OCTAVES: u32 = 6;

/// Salt mixed into the recipe seed, so the detail band is decorrelated from every
/// other hashed decision the build makes off the same number.
const DETAIL_SALT: u64 = 0x7E7A_11D3;

/// **The band this build writes into** — derived from the two pitches, never
/// authored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailBand {
    /// Wavelength of the base octave, in metres — the SOURCE's Nyquist, i.e. the
    /// shortest feature the survey could have represented.
    pub base_wavelength_m: f64,
    /// Wavelength of the finest octave, in metres.
    pub finest_wavelength_m: f64,
    /// Octaves, at lacunarity 2.
    pub octaves: u32,
}

impl DetailBand {
    /// The band for a grid of `grid_m_per_sample` over a source of
    /// `source_m_per_px`.
    ///
    /// Returns `None` when the grid is **not** finer than the source: there is
    /// then no empty band to fill, and inventing relief the survey could have
    /// carried would be overwriting it. That is the same test the
    /// `source.upsampled` advisory makes, at the same threshold, so a build that
    /// prints the advisory is exactly a build that gets detail.
    pub fn of(grid_m_per_sample: f64, source_m_per_px: f64) -> Option<Self> {
        // `is_finite` first, then an ordinary comparison. Written this way rather
        // than as `!(x > 0.0)` because clippy is right that a negated comparison
        // on a partially-ordered type hides the third case — and here the third
        // case is real: a recipe whose grid or source pitch decodes as NaN or as
        // infinity must get no band rather than an infinite one.
        if !grid_m_per_sample.is_finite()
            || !source_m_per_px.is_finite()
            || grid_m_per_sample <= 0.0
            || source_m_per_px <= 0.0
        {
            return None;
        }
        if source_m_per_px / grid_m_per_sample <= crate::source::UPSAMPLE_ADVISORY_RATIO {
            return None;
        }
        // Both Nyquists: the coarse end is what the SOURCE could not represent,
        // the fine end is what the GRID cannot represent. Everything between the
        // two is empty by construction and is what this band fills.
        let base = 2.0 * source_m_per_px;
        let finest_possible = 2.0 * grid_m_per_sample;
        let mut octaves = 1u32;
        let mut w = base;
        // Halving, not `log2` — the libm law reaches committed bytes.
        while w * 0.5 >= finest_possible && octaves < MAX_OCTAVES {
            w *= 0.5;
            octaves += 1;
        }
        Some(Self {
            base_wavelength_m: base,
            finest_wavelength_m: w,
            octaves,
        })
    }

    /// Cycles per world metre of the base octave — `fbm_signed`'s `frequency`.
    pub fn base_frequency(&self) -> f64 {
        1.0 / self.base_wavelength_m
    }
}

/// What the detail stage was given to respect.
///
/// Borrowed rather than owned for `CarvePlan`'s reason: every one of these is a
/// live structure the build already holds, and copying a signed-distance field to
/// pass it one step later would be a megabyte for nothing.
pub struct DetailPlan<'a> {
    /// The recipe's seed. Salted here, so the band is not a scaled copy of any
    /// other hashed decision.
    pub seed: u64,
    /// The waterline the shore fade is measured from.
    pub sea_level_m: f64,
    /// The band. `None` disables the whole stage.
    pub band: Option<DetailBand>,
    /// The carve's coastline — used for `is_land`, so a submerged sample is never
    /// touched however high the fade would have let it be.
    pub coast: &'a Coastline,
    /// The road corridor, if the design has roads.
    pub corridor: Option<&'a SegmentIndex>,
    /// The corridor's own half-width in metres.
    pub corridor_half_m: f64,
    /// The stream channels.
    pub channels: Option<&'a SegmentIndex>,
    /// The widest channel's half-width in metres.
    pub channel_half_m: f64,
    /// Site pads as `(centre, radius, datum)` — the carve's own list.
    pub pads: &'a [(DVec2, f64, f64)],
}

/// What one detail pass did — every number the ledger prints.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DetailStats {
    /// Level-0 samples walked.
    pub samples: u64,
    /// Samples the masks let through at all (weight > 0).
    pub written: u64,
    /// Samples refused because they are at or under the waterline.
    pub masked_water: u64,
    /// Samples refused because they are inside a road corridor.
    pub masked_road: u64,
    /// Samples refused because they are inside a stream channel.
    pub masked_channel: u64,
    /// Samples refused because they are on a site pad.
    pub masked_pad: u64,
    /// The largest absolute displacement written, in metres.
    pub max_abs_m: f64,
    /// The mean absolute displacement over the samples that took one.
    pub mean_abs_m: f64,
    /// The band that was used (`None` ⇒ the stage was inert).
    pub band: Option<DetailBand>,
}

impl DetailStats {
    /// Whether the stage wrote anything at all.
    pub fn is_inert(&self) -> bool {
        self.written == 0
    }
}

/// `0` below `lo`, `1` above `hi`, smooth in between (`t·t·(3−2t)`).
///
/// Smooth rather than linear because the derivative is what a shading normal
/// reads, and a linear ramp puts a visible crease at both ends of every mask.
#[inline]
fn ramp(lo: f64, hi: f64, v: f64) -> f64 {
    // A degenerate or unordered band is a STEP at `hi`, not a divide by zero.
    // Spelled positively (clippy's `neg_cmp_op_on_partial_ord`) because the case
    // it makes visible — one of the three being NaN — is one this really has to
    // answer: a mask width that arrived as NaN must exclude nothing rather than
    // silently multiplying the whole island's amplitude by NaN.
    if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
        return if v >= hi { 1.0 } else { 0.0 };
    }
    let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The mask weight a segment index contributes at `p`: `0` inside `half`, ramping
/// to `1` at `half · (1 + FADE_WIDTHS)`.
#[inline]
fn segment_weight(index: Option<&SegmentIndex>, half_m: f64, p: DVec2) -> f64 {
    let (Some(ix), true) = (index, half_m > 0.0) else {
        return 1.0;
    };
    match ix.nearest(p) {
        Some(n) => ramp(half_m, half_m * (1.0 + FADE_WIDTHS), n.distance_m),
        None => 1.0,
    }
}

/// The slope in degrees at one sample of one tile, by central difference on the
/// 1 m grid, reading the neighbour tile across a border.
///
/// A near-copy of `splat::slope_deg_at` and deliberately not shared with it: that
/// one takes a `&TerrainTile` borrowed out of the same `TerrainData` this stage
/// needs `&mut`, and the two walks read at different points in the build. What is
/// shared is the *rule* — one central difference, one `patan2_64` — and the two
/// are pinned against each other by
/// [`tests::the_slope_rule_agrees_with_the_splat_walk`].
fn slope_deg_at(data: &TerrainData, origin: DVec2, mps: f64, i: u32, j: u32, res: u32) -> f64 {
    let world = |di: f64, dj: f64| {
        DVec2::new(
            origin.x + (f64::from(i) + di) * mps,
            origin.y + (f64::from(j) + dj) * mps,
        )
    };
    let here = data.height_at(world(0.0, 0.0)).unwrap_or(0.0);
    let h = |di: f64, dj: f64| data.height_at(world(di, dj)).unwrap_or(here);
    let _ = res;
    let dx = (h(1.0, 0.0) - h(-1.0, 0.0)) / (2.0 * mps);
    let dz = (h(0.0, 1.0) - h(0.0, -1.0)) / (2.0 * mps);
    let g = (dx * dx + dz * dz).sqrt();
    inf_math::portable::patan2_64(g, 1.0).to_degrees()
}

/// How much of the ceiling amplitude this biome takes.
///
/// A multiplier on the slope term rather than a replacement for it: a forest
/// floor on a 40° slope is still broken ground. The four that move are the four
/// whose *surface* is a fact about the biome and not about the terrain under it.
fn biome_amplitude(id: u8) -> f64 {
    match crate::biome::IslandBiome::from_id(id) {
        // Cut flat, drained and driven on. A ploughed field is smooth at six
        // metres whatever the hill it sits on does.
        Some(crate::biome::IslandBiome::Farmland) => 0.25,
        Some(crate::biome::IslandBiome::Urban) => 0.20,
        // Sand under water and over it: wave-worked, and the shore fade is
        // already taking most of it.
        Some(crate::biome::IslandBiome::Beach) => 0.35,
        // Scree, frost-shattered rock and no soil to smooth it.
        Some(crate::biome::IslandBiome::Alpine) => 1.0,
        // Roots, hollows and blowdown.
        Some(crate::biome::IslandBiome::Forest) => 0.9,
        // Everything else, including the unclassified: the slope term alone.
        _ => 0.7,
    }
}

/// **Write the detail band into every level-0 tile.**
///
/// Deterministic and order-independent: the displacement at a world position is a
/// pure function of `(seed, band, that position)` and the masks, so the tile walk
/// order cannot reach the result. It is a *read-then-write* per tile — the slope
/// is measured on the ground as the previous steps left it, for every sample,
/// before any sample of that tile is moved — so a tile's own detail cannot feed
/// back into its own slope term. Across a tile border the neighbour may already
/// have been written; the slope term is a soft amplitude multiplier over a 32°
/// ramp, so the difference that makes is fractions of a millimetre, and it is
/// stated rather than hidden.
pub fn apply_detail(data: &mut TerrainData, plan: &DetailPlan<'_>) -> DetailStats {
    let mut st = DetailStats {
        band: plan.band,
        ..DetailStats::default()
    };
    let Some(band) = plan.band else {
        return st;
    };
    let res = data.tile_resolution();
    let mps = data.meters_per_sample();
    let n = (res * res) as usize;
    if n == 0 {
        return st;
    }
    let seed = plan.seed ^ DETAIL_SALT;
    let freq = band.base_frequency();
    let mut coords: Vec<(i32, i32)> = data.tiles().map(|(c, _)| *c).collect();
    // Sorted, so the walk is a function of the tile SET and not of the map's
    // iteration order. Nothing here depends on order, and that is exactly why it
    // must be impossible for a future term to.
    coords.sort();
    let mut buf: Vec<f32> = Vec::with_capacity(n);
    let mut sum_abs = 0.0f64;

    for c in coords {
        let origin = data.tile_origin_xz(c);
        buf.clear();
        {
            let Some(tile) = data.get_tile(c) else {
                continue;
            };
            for j in 0..res {
                for i in 0..res {
                    let p =
                        DVec2::new(origin.x + f64::from(i) * mps, origin.y + f64::from(j) * mps);
                    let h = f64::from(tile.sample(res, i, j));
                    st.samples += 1;

                    // ── the masks, cheapest and most decisive first ──
                    if !plan.coast.is_land(p) || h <= plan.sea_level_m {
                        st.masked_water += 1;
                        buf.push(tile.sample(res, i, j));
                        continue;
                    }
                    let mut w = ramp(plan.sea_level_m, plan.sea_level_m + SHORE_FADE_M, h);
                    if w <= 0.0 {
                        st.masked_water += 1;
                        buf.push(tile.sample(res, i, j));
                        continue;
                    }
                    let road = segment_weight(plan.corridor, plan.corridor_half_m, p);
                    if road <= 0.0 {
                        st.masked_road += 1;
                        buf.push(tile.sample(res, i, j));
                        continue;
                    }
                    let chan = segment_weight(plan.channels, plan.channel_half_m, p);
                    if chan <= 0.0 {
                        st.masked_channel += 1;
                        buf.push(tile.sample(res, i, j));
                        continue;
                    }
                    let mut pad = 1.0f64;
                    for (centre, radius, _) in plan.pads {
                        if *radius > 0.0 {
                            let d = (p - *centre).length();
                            pad = pad.min(ramp(*radius, *radius * (1.0 + FADE_WIDTHS), d));
                        }
                    }
                    if pad <= 0.0 {
                        st.masked_pad += 1;
                        buf.push(tile.sample(res, i, j));
                        continue;
                    }
                    w *= road * chan * pad;

                    // ── the amplitude ──
                    let slope = slope_deg_at(data, origin, mps, i, j, res);
                    let slope_t = ramp(SLOPE_LO_DEG, SLOPE_HI_DEG, slope);
                    let shape =
                        AMPLITUDE_FLOOR_FRACTION + (1.0 - AMPLITUDE_FLOOR_FRACTION) * slope_t;
                    let biome = biome_amplitude(tile.biome_sample(res, i, j));
                    let amp = MAX_AMPLITUDE_M * shape * biome * w;

                    let d = amp * inf_terrain::fbm_signed(seed, freq, band.octaves, p.x, p.y);
                    if d != 0.0 {
                        st.written += 1;
                        let a = d.abs();
                        sum_abs += a;
                        if a > st.max_abs_m {
                            st.max_abs_m = a;
                        }
                    }
                    buf.push((h + d) as f32);
                }
            }
        }
        let Some(tile) = data.get_tile_mut(c) else {
            continue;
        };
        for j in 0..res {
            for i in 0..res {
                tile.set_sample(res, i, j, buf[(j * res + i) as usize]);
            }
        }
    }
    if st.written > 0 {
        st.mean_abs_m = sum_abs / st.written as f64;
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The island's own numbers: a 1 m grid over a 3.11 m z15 pixel gives two
    /// octaves running 6.22 m down to 3.11 m — and **nothing below 2 m**, which
    /// is the grid's own Nyquist and the line between "detail" and "aliasing".
    #[test]
    fn the_island_band_runs_from_the_sources_nyquist_to_the_grids() {
        let b = DetailBand::of(1.0, 3.11).expect("a 3.1x upsample gets a band");
        assert_eq!(b.octaves, 2);
        assert!((b.base_wavelength_m - 6.22).abs() < 1e-12);
        assert!((b.finest_wavelength_m - 3.11).abs() < 1e-12);
        assert!(
            b.finest_wavelength_m >= 2.0,
            "the finest octave is {} m, under the 1 m grid's 2 m Nyquist -- that \
             is aliasing, not detail",
            b.finest_wavelength_m
        );
    }

    /// A grid no finer than its source gets **no band at all**, at exactly the
    /// ratio the `source.upsampled` advisory fires at. The two must agree: a
    /// build that says "detail below X is the design's" and adds none, or one
    /// that adds detail and never says so, is the same defect either way round.
    #[test]
    fn a_grid_no_finer_than_its_source_is_left_alone() {
        assert!(DetailBand::of(1.0, 1.0).is_none());
        assert!(
            DetailBand::of(1.0, 1.05).is_none(),
            "exactly at the advisory"
        );
        assert!(DetailBand::of(1.0, 1.06).is_some(), "just past it");
        assert!(DetailBand::of(4.0, 3.11).is_none(), "a COARSER grid");
        assert!(DetailBand::of(0.0, 3.11).is_none());
        assert!(DetailBand::of(1.0, 0.0).is_none());
    }

    /// The octave guard bites before the count runs away, and the finest octave
    /// never crosses the grid's Nyquist even when it does.
    #[test]
    fn the_octave_count_is_guarded_and_never_aliases() {
        let b = DetailBand::of(0.001, 1000.0).expect("a huge upsample");
        assert_eq!(b.octaves, MAX_OCTAVES);
        let b = DetailBand::of(1.0, 20.0).expect("a 20x upsample");
        // 40 -> 20 -> 10 -> 5 -> 2.5, and 1.25 would be under the 2 m Nyquist.
        assert_eq!(b.octaves, 5);
        assert!(b.finest_wavelength_m >= 2.0);
    }

    /// The masks are FADES. A hard cut would put a crease along every road and
    /// every stream, which is more visible than the detail is.
    #[test]
    fn every_mask_ramps_rather_than_cutting() {
        // Dead inside the half width, dead ON it, and full at one width past it.
        assert_eq!(ramp(10.0, 20.0, 0.0), 0.0);
        assert_eq!(ramp(10.0, 20.0, 10.0), 0.0);
        assert_eq!(ramp(10.0, 20.0, 20.0), 1.0);
        // …and strictly monotone in between, with a zero derivative at both ends
        // (which is what `t*t*(3-2t)` buys over a linear ramp).
        let mid = ramp(10.0, 20.0, 15.0);
        assert!((mid - 0.5).abs() < 1e-12);
        assert!(ramp(10.0, 20.0, 11.0) < 0.06, "the low end is nearly flat");
        assert!(ramp(10.0, 20.0, 19.0) > 0.94, "and so is the high end");
    }

    /// A biome multiplier is a multiplier: bounded, positive, and never a way to
    /// exceed the stage's own ceiling.
    #[test]
    fn no_biome_can_exceed_the_amplitude_ceiling() {
        for id in 0u8..=8 {
            let a = biome_amplitude(id);
            assert!(
                a > 0.0 && a <= 1.0,
                "biome {id} takes {a} of the ceiling, which is not a fraction"
            );
        }
        // …and the whole product, at its worst: full slope, richest biome, no
        // mask. That is the number the ledger prints as the ceiling.
        let worst =
            MAX_AMPLITUDE_M * (AMPLITUDE_FLOOR_FRACTION + (1.0 - AMPLITUDE_FLOOR_FRACTION)) * 1.0;
        assert!((worst - MAX_AMPLITUDE_M).abs() < 1e-12);
    }

    /// The displacement is a pure function of world position: the same point
    /// answers the same metres however the caller reached it.
    #[test]
    fn the_displacement_is_a_pure_function_of_the_world_position() {
        let b = DetailBand::of(1.0, 3.11).unwrap();
        let f = b.base_frequency();
        for (x, z) in [(0.0, 0.0), (123.5, -987.25), (-4000.0, 4000.0)] {
            let a = inf_terrain::fbm_signed(7 ^ DETAIL_SALT, f, b.octaves, x, z);
            let c = inf_terrain::fbm_signed(7 ^ DETAIL_SALT, f, b.octaves, x, z);
            assert_eq!(a, c);
            assert!((-1.0..=1.0).contains(&a), "fbm answered {a}");
        }
        // …and a different seed is a different island.
        assert_ne!(
            inf_terrain::fbm_signed(7 ^ DETAIL_SALT, f, b.octaves, 10.0, 10.0),
            inf_terrain::fbm_signed(8 ^ DETAIL_SALT, f, b.octaves, 10.0, 10.0)
        );
    }

    /// The slope rule is the splat walk's rule. Both measure one central
    /// difference on the 1 m grid through `patan2_64`; if they ever disagreed,
    /// a sample could be rock in the paint and meadow in the relief.
    #[test]
    fn the_slope_rule_agrees_with_the_splat_walk() {
        let res = 9u32;
        let mut data = TerrainData::new(res, 1.0);
        // A 30-degree ramp along +x: dh/dx = tan(30) = 0.5773502691896257.
        let g = 0.577_350_269_189_625_7f64;
        let mut heights = vec![0.0f32; (res * res) as usize];
        for j in 0..res {
            for i in 0..res {
                heights[(j * res + i) as usize] = (f64::from(i) * g) as f32;
            }
        }
        data.insert_tile(
            (0, 0),
            inf_terrain::TerrainTile::from_heights(res, glam::DVec3::ZERO, heights)
                .expect("a full height buffer builds a tile"),
        )
        .expect("the tile inserts");
        let mine = slope_deg_at(&data, DVec2::ZERO, 1.0, 4, 4, res);
        // 1e-4 and not 1e-12: a tile stores `f32`, so a 30-degree ramp built in
        // `f64` and read back is right to about a millionth of a degree. Tighter
        // would be asserting the storage format rather than the rule.
        assert!(
            (mine - 30.0).abs() < 1e-4,
            "the ramp measures {mine} degrees, not 30"
        );
        // Flat ground is zero, not a small number that a ramp would round up.
        let mut flat = TerrainData::new(res, 1.0);
        flat.insert_tile(
            (0, 0),
            crate::terrain::flat_tile(res, glam::DVec3::ZERO, 12.0),
        )
        .expect("the tile inserts");
        assert_eq!(slope_deg_at(&flat, DVec2::ZERO, 1.0, 4, 4, res), 0.0);
    }
}
