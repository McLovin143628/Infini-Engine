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
//! **It fills the coarse part of the empty band, not all of it** (TER2b audit).
//! Octaves halve from the source's Nyquist and the last one that clears the
//! grid's is kept, so on the island the band runs 6.22 m → 3.11 m and the
//! remaining 3.11 m → 2 m stays as empty as it was: one octave of the 1.64 the
//! upsample opens. Filling it would mean an octave at 1.56 m against a 2 m
//! Nyquist, which is aliasing rather than detail — the arm
//! [`tests::the_island_band_runs_from_the_sources_nyquist_to_the_grids`] is the
//! one that refuses it, and the honest statement is "the band is filled down to
//! the finest octave the grid can carry".
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
/// Not zero: a dead-flat plain is the one thing real ground never is, and a
/// decimetre of undulation over six metres is the difference between a meadow and
/// a table.
///
/// **A decimetre and not fifteen centimetres** (TER2b audit): the slope floor is
/// `0.10 × 1.5 m`, and then [`biome_amplitude`] multiplies it, and the least
/// specific biome — the `_` arm every unclassified sample takes — is `0.7`. So a
/// flat plain's ceiling is `0.105 m`, measured at **0.086 m** of actual peak
/// displacement on a flat two-tile fixture, because a two-octave fBm does not
/// reach its own bound. The first write-up quoted the product of two of the three
/// terms.
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

/// **How far a mask index must answer for the fade to exist at all.**
///
/// A [`SegmentIndex`] answers `None` past its own reach, and `None` is read here
/// as "far away, take all the detail". So an index built to the feature's *own*
/// half-width has no fade in it whatever [`FADE_WIDTHS`] says: the weight is `0`
/// out to the half-width, the query stops answering one metre later, and the
/// ground jumps to full amplitude in a single sample. That is the crease along
/// every road this stage exists to avoid, and it is what the TER2b audit measured
/// — `0.000 m` of relief at 7 m from a corridor centreline and `0.127 m` at 8 m,
/// where the ramp should have given `0.007`.
///
/// So an index the detail stage queries is built through **this** function, and
/// the carve keeps building its own at the half-width, which is the right reach
/// for levelling ground.
pub fn fade_reach_m(half_m: f64) -> f64 {
    half_m * (1.0 + FADE_WIDTHS)
}

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
/// to `1` at [`fade_reach_m(half)`](fade_reach_m).
///
/// **The index must answer out to `fade_reach_m(half_m)`**, because a `None` is
/// read here as "past the fade, take everything" and a short index makes that a
/// lie for the whole fade band. `apply_detail` debug-asserts it; `build.rs`
/// constructs both of the detail stage's indices through `fade_reach_m` for the
/// same reason.
#[inline]
fn segment_weight(index: Option<&SegmentIndex>, half_m: f64, p: DVec2) -> f64 {
    let (Some(ix), true) = (index, half_m > 0.0) else {
        return 1.0;
    };
    match ix.nearest(p) {
        Some(n) => ramp(half_m, fade_reach_m(half_m), n.distance_m),
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
/// really are compared, sample for sample and to the bit, by
/// [`tests::the_slope_rule_agrees_with_the_splat_walk`], which calls
/// [`crate::splat::slope_deg_at`] rather than describing it. (The TER2b audit
/// found that arm asserting only that a 30° ramp measures 30°, which the splat
/// walk is not needed for and cannot fail.)
fn slope_deg_at(data: &TerrainData, origin: DVec2, mps: f64, i: u32, j: u32) -> f64 {
    let world = |di: f64, dj: f64| {
        DVec2::new(
            origin.x + (f64::from(i) + di) * mps,
            origin.y + (f64::from(j) + dj) * mps,
        )
    };
    let here = data.height_at(world(0.0, 0.0)).unwrap_or(0.0);
    let h = |di: f64, dj: f64| data.height_at(world(di, dj)).unwrap_or(here);
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

/// What one sample's masks and amplitude answered — the whole of a sample's fate,
/// so that a sample computed in one pass can be *spent* in another without
/// recomputing it or double-counting it in the stats.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Outcome {
    Water,
    Road,
    Channel,
    Pad,
    /// Metres to add to the sample's stored height (exactly `0.0` when the fBm is).
    Moved(f64),
}

/// A sample on the outermost ring of its tile — the only samples whose central
/// difference reaches across a tile border.
///
/// An interior sample at `i = 1` reads `i = 0` and `i = 2`, both of its own tile;
/// only `0` and `res − 1` (in either axis) read a neighbour. That is why the ring
/// is the exact set that has to be settled before any tile is written.
#[inline]
fn is_rim(i: u32, j: u32, res: u32) -> bool {
    i == 0 || j == 0 || i + 1 == res || j + 1 == res
}

/// The masks, the amplitude and the displacement at one sample of one tile.
///
/// Reads `data` and never writes it, so a caller may run it over every tile
/// before any tile has moved — which is exactly what [`apply_detail`]'s first
/// pass does for the rim.
#[allow(clippy::too_many_arguments)]
fn outcome_at(
    data: &TerrainData,
    tile: &inf_terrain::TerrainTile,
    plan: &DetailPlan<'_>,
    band: DetailBand,
    seed: u64,
    freq: f64,
    origin: DVec2,
    mps: f64,
    res: u32,
    i: u32,
    j: u32,
) -> Outcome {
    let p = DVec2::new(origin.x + f64::from(i) * mps, origin.y + f64::from(j) * mps);
    let h = f64::from(tile.sample(res, i, j));

    // ── the masks, cheapest and most decisive first ──
    if !plan.coast.is_land(p) || h <= plan.sea_level_m {
        return Outcome::Water;
    }
    let mut w = ramp(plan.sea_level_m, plan.sea_level_m + SHORE_FADE_M, h);
    if w <= 0.0 {
        return Outcome::Water;
    }
    let road = segment_weight(plan.corridor, plan.corridor_half_m, p);
    if road <= 0.0 {
        return Outcome::Road;
    }
    let chan = segment_weight(plan.channels, plan.channel_half_m, p);
    if chan <= 0.0 {
        return Outcome::Channel;
    }
    let mut pad = 1.0f64;
    for (centre, radius, _) in plan.pads {
        if *radius > 0.0 {
            let d = (p - *centre).length();
            pad = pad.min(ramp(*radius, fade_reach_m(*radius), d));
        }
    }
    if pad <= 0.0 {
        return Outcome::Pad;
    }
    w *= road * chan * pad;

    // ── the amplitude ──
    let slope = slope_deg_at(data, origin, mps, i, j);
    let slope_t = ramp(SLOPE_LO_DEG, SLOPE_HI_DEG, slope);
    let shape = AMPLITUDE_FLOOR_FRACTION + (1.0 - AMPLITUDE_FLOOR_FRACTION) * slope_t;
    let biome = biome_amplitude(tile.biome_sample(res, i, j));
    let amp = MAX_AMPLITUDE_M * shape * biome * w;
    Outcome::Moved(amp * inf_terrain::fbm_signed(seed, freq, band.octaves, p.x, p.y))
}

/// **Write the detail band into every level-0 tile.**
///
/// Deterministic and order-independent: the displacement at a world position is a
/// pure function of `(seed, band, that position)` and the masks, so neither the
/// tile walk order nor the tile *partition* can reach the result.
///
/// # Why it is two passes, and what the second one would otherwise cost
///
/// The amplitude reads a **slope**, and a slope is a central difference over the
/// neighbouring metre. Inside a tile that is safe by construction — the walk is
/// read-then-write, so a tile's own detail cannot feed its own slope term. Across
/// a tile border it is not: adjacent tiles **share a row of samples**
/// (`tile_span = (res − 1) · mps`), so the same world position is stored twice,
/// and a one-pass walk computes the second copy's amplitude against a neighbour
/// it has already displaced. The two copies then disagree, and a heightfield
/// whose two tiles answer differently for one metre of ground is a **crack**.
///
/// Measured before the fix, on a two-tile ramp at 1 m a sample: **0.382 m** of
/// disagreement on a 20° slope and **0.498 m** on 35°, against a carried note
/// that called it "fractions of a millimetre". It is a first-class seam, not a
/// rounding remark.
///
/// So pass one settles the **rim** — [`is_rim`], the only samples whose
/// difference leaves the tile — over ground no tile has moved yet, and pass two
/// spends those answers while computing the interiors, which cannot reach out of
/// their tile at all. Both copies of a shared sample are computed in pass one
/// from identical neighbourhoods, so they are identical to the bit.
pub fn apply_detail(data: &mut TerrainData, plan: &DetailPlan<'_>) -> DetailStats {
    let mut st = DetailStats {
        band: plan.band,
        ..DetailStats::default()
    };
    let Some(band) = plan.band else {
        return st;
    };
    // The fade is a lie if the index stops answering inside it — see
    // `fade_reach_m`, and the TER2b audit that measured the crease it leaves.
    debug_assert!(
        plan.corridor
            .is_none_or(|ix| ix.reach_m() >= fade_reach_m(plan.corridor_half_m) - 1e-9),
        "the corridor index answers only to {:?} m and the fade runs to {} m",
        plan.corridor.map(|ix| ix.reach_m()),
        fade_reach_m(plan.corridor_half_m)
    );
    debug_assert!(
        plan.channels
            .is_none_or(|ix| ix.reach_m() >= fade_reach_m(plan.channel_half_m) - 1e-9),
        "the channel index answers only to {:?} m and the fade runs to {} m",
        plan.channels.map(|ix| ix.reach_m()),
        fade_reach_m(plan.channel_half_m)
    );
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

    // ── PASS ONE: the rim, on ground no tile has moved ──────────────────────
    let mut rims: std::collections::BTreeMap<(i32, i32), Vec<Outcome>> =
        std::collections::BTreeMap::new();
    for &c in &coords {
        let origin = data.tile_origin_xz(c);
        let Some(tile) = data.get_tile(c) else {
            continue;
        };
        let mut ring = Vec::with_capacity(4 * res as usize);
        for j in 0..res {
            for i in 0..res {
                if is_rim(i, j, res) {
                    ring.push(outcome_at(
                        data, tile, plan, band, seed, freq, origin, mps, res, i, j,
                    ));
                }
            }
        }
        rims.insert(c, ring);
    }

    // ── PASS TWO: the interiors, and the writes ─────────────────────────────
    let mut buf: Vec<f32> = Vec::with_capacity(n);
    let mut sum_abs = 0.0f64;
    for c in coords {
        let origin = data.tile_origin_xz(c);
        buf.clear();
        {
            let Some(tile) = data.get_tile(c) else {
                continue;
            };
            let ring = rims.get(&c);
            // Walked in the same `(j, i)` order pass one filled the ring in, so
            // the cursor and the ring stay in step without storing a key.
            let mut cursor = 0usize;
            for j in 0..res {
                for i in 0..res {
                    st.samples += 1;
                    let out = if is_rim(i, j, res) {
                        let o = ring.and_then(|r| r.get(cursor)).copied();
                        cursor += 1;
                        match o {
                            Some(o) => o,
                            None => outcome_at(
                                data, tile, plan, band, seed, freq, origin, mps, res, i, j,
                            ),
                        }
                    } else {
                        outcome_at(data, tile, plan, band, seed, freq, origin, mps, res, i, j)
                    };
                    match out {
                        Outcome::Water => {
                            st.masked_water += 1;
                            buf.push(tile.sample(res, i, j));
                        }
                        Outcome::Road => {
                            st.masked_road += 1;
                            buf.push(tile.sample(res, i, j));
                        }
                        Outcome::Channel => {
                            st.masked_channel += 1;
                            buf.push(tile.sample(res, i, j));
                        }
                        Outcome::Pad => {
                            st.masked_pad += 1;
                            buf.push(tile.sample(res, i, j));
                        }
                        Outcome::Moved(d) => {
                            if d != 0.0 {
                                st.written += 1;
                                let a = d.abs();
                                sum_abs += a;
                                if a > st.max_abs_m {
                                    st.max_abs_m = a;
                                }
                            }
                            buf.push((f64::from(tile.sample(res, i, j)) + d) as f32);
                        }
                    }
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

    /// Two tiles carrying a ramp of `grad` in +x, sharing their border column
    /// (`tile_span = (res − 1) · mps`, so the shared metre is stored twice).
    fn two_tiles(res: u32, mps: f64, grad: f64, base: f64) -> TerrainData {
        let mut data = TerrainData::new(res, mps);
        let span = (f64::from(res) - 1.0) * mps;
        for c in [(0i32, 0i32), (1, 0)] {
            let origin = glam::DVec3::new(f64::from(c.0) * span, 0.0, f64::from(c.1) * span);
            let mut h = vec![0.0f32; (res * res) as usize];
            for j in 0..res {
                for i in 0..res {
                    let x = origin.x + f64::from(i) * mps;
                    h[(j * res + i) as usize] = (base + grad * x) as f32;
                }
            }
            data.insert_tile(
                (c.0, c.1),
                inf_terrain::TerrainTile::from_heights(res, origin, h).expect("a full buffer"),
            )
            .expect("the tile inserts");
        }
        data
    }

    /// A coastline that puts everything in a large square on land.
    fn all_land(half: f64) -> Coastline {
        Coastline::new(
            vec![vec![
                DVec2::new(-half, -half),
                DVec2::new(half, -half),
                DVec2::new(half, half),
                DVec2::new(-half, half),
            ]],
            DVec2::splat(-half * 2.0),
            DVec2::splat(half * 2.0),
            4.0,
        )
    }

    fn plain_plan<'a>(coast: &'a Coastline, band: Option<DetailBand>) -> DetailPlan<'a> {
        DetailPlan {
            seed: 7,
            sea_level_m: 0.0,
            band,
            coast,
            corridor: None,
            corridor_half_m: 0.0,
            channels: None,
            channel_half_m: 0.0,
            pads: &[],
        }
    }

    /// The slope rule is the splat walk's rule. Both measure one central
    /// difference on the 1 m grid through `patan2_64`; if they ever disagreed,
    /// a sample could be rock in the paint and meadow in the relief.
    ///
    /// **It calls both** (wave TER2b audit). The arm this replaces asserted only
    /// that a 30° ramp measures 30°, which is a fact about arithmetic that the
    /// splat walk is not needed for and cannot falsify — a cited pin that did not
    /// exist. The comparison below runs over **every** sample of two tiles,
    /// including the border rows, where the two implementations differ most
    /// (`splat` reads its own tile's buffer for interior samples and
    /// `TerrainData::height_at` across a border; `detail` goes through `height_at`
    /// for all four).
    #[test]
    fn the_slope_rule_agrees_with_the_splat_walk() {
        let res = 9u32;
        let mps = 1.0;
        // A 30-degree ramp along +x: dh/dx = tan(30) = 0.5773502691896257.
        let data = two_tiles(res, mps, 0.577_350_269_189_625_7, 100.0);
        let mut compared = 0usize;
        let mut border = 0usize;
        for c in [(0i32, 0i32), (1, 0)] {
            let origin = data.tile_origin_xz(c);
            let tile = data.get_tile(c).expect("the tile is there");
            for j in 0..res {
                for i in 0..res {
                    let mine = slope_deg_at(&data, origin, mps, i, j);
                    let theirs = crate::splat::slope_deg_at(&data, tile, origin, res, mps, i, j);
                    assert_eq!(
                        mine.to_bits(),
                        theirs.to_bits(),
                        "tile {c:?} sample ({i}, {j}): the detail walk reads {mine} degrees \
                         and the splat walk reads {theirs}"
                    );
                    compared += 1;
                    if is_rim(i, j, res) {
                        border += 1;
                    }
                }
            }
        }
        // NOT VACUOUS: the comparison covered the border rows, and the ramp is
        // steep enough that a one-sided difference would have shown.
        assert_eq!(compared, 2 * (res * res) as usize);
        assert!(border > 0);
        let mid = slope_deg_at(&data, data.tile_origin_xz((0, 0)), mps, 4, 4);
        assert!(
            (mid - 30.0).abs() < 1e-4,
            "the ramp measures {mid} degrees, not 30"
        );
        // Flat ground is zero, not a small number that a ramp would round up.
        let mut flat = TerrainData::new(res, 1.0);
        flat.insert_tile(
            (0, 0),
            crate::terrain::flat_tile(res, glam::DVec3::ZERO, 12.0),
        )
        .expect("the tile inserts");
        assert_eq!(slope_deg_at(&flat, DVec2::ZERO, 1.0, 4, 4), 0.0);
    }

    /// **A shared tile edge takes one displacement** (wave TER2b audit).
    ///
    /// Adjacent tiles store the same metre of ground twice, so the detail stage
    /// has to answer identically for both copies or the heightfield has a crack
    /// in it every 256 m. The one-pass walk did not: the second tile's rim read a
    /// slope off a neighbour it had already displaced, and the amplitude ramp
    /// turned that into **0.382 m** of disagreement on a 20° slope and **0.498 m**
    /// on 35° — measured, against a carried note that said "fractions of a
    /// millimetre".
    ///
    /// The anti-vacuity half is the second assertion: the stage has to have moved
    /// the ground *at all* on this fixture, and by more than the tolerance the
    /// first assertion allows, or agreeing at the edge is agreeing about nothing.
    #[test]
    fn a_shared_tile_edge_takes_one_displacement() {
        let res = 33u32;
        let mps = 1.0;
        let coast = all_land(4000.0);
        // Gradients rather than degrees, in the fixture AND in the messages: the
        // crate's libm-ban table reads test lines too and `atan` is on it. It
        // caught the first draft of this arm — the third unprompted catch this
        // campaign, and the first inside an audit's own repair.
        for (grad, deg) in [
            (0.0f64, 0.0),
            (0.1, 5.7),
            (0.363_970_234_266_202_36, 20.0),
            (0.700_207_538_209_699_5, 35.0),
        ] {
            let mut data = two_tiles(res, mps, grad, 100.0);
            let st = apply_detail(&mut data, &plain_plan(&coast, DetailBand::of(mps, 3.11)));
            let a = data.get_tile((0, 0)).expect("tile a");
            let b = data.get_tile((1, 0)).expect("tile b");
            let mut worst = 0.0f64;
            for j in 0..res {
                let ha = f64::from(a.sample(res, res - 1, j));
                let hb = f64::from(b.sample(res, 0, j));
                worst = worst.max((ha - hb).abs());
            }
            assert_eq!(
                worst, 0.0,
                "the shared column disagrees by {worst} m on the {deg}-degree ramp \
                 (gradient {grad}) -- the two tiles store the same metre of ground \
                 and must answer the same height for it"
            );
            // …and the stage really did displace this fixture.
            assert!(
                st.written > 0 && st.max_abs_m > 0.05,
                "nothing moved on the {deg}-degree ramp ({} samples, worst {} m), \
                 so the agreement above is vacuous",
                st.written,
                st.max_abs_m
            );
        }
    }

    /// **A mask index that stops at the half-width cuts instead of fading**
    /// (wave TER2b audit).
    ///
    /// [`SegmentIndex`] answers `None` past its own reach and `segment_weight`
    /// reads `None` as "far away, take everything". So the whole `t·t·(3−2t)` ramp
    /// is dead unless the index answers out to [`fade_reach_m`] — which is what
    /// the shipped build passes now, and did not before.
    ///
    /// Both indices are built here, so the arm carries its own mutation: the
    /// half-width one steps a full unit in one metre and the faded one does not.
    #[test]
    fn a_mask_index_must_answer_past_the_fade_or_the_fade_is_a_cut() {
        let half = 7.0f64;
        let line = vec![vec![
            crate::shape::Vertex3 {
                xz: DVec2::new(-500.0, 0.0),
                y: 0.0,
            },
            crate::shape::Vertex3 {
                xz: DVec2::new(500.0, 0.0),
                y: 0.0,
            },
        ]];
        let short = SegmentIndex::new(&line, half);
        let faded = SegmentIndex::new(&line, fade_reach_m(half));
        let w = |ix: &SegmentIndex, d: f64| segment_weight(Some(ix), half, DVec2::new(0.0, d));

        // THE DEFECT, kept as the control: dead at the half-width and full one
        // sample later, which is a wall along every road.
        assert_eq!(w(&short, half), 0.0);
        assert_eq!(w(&short, half + 1.0), 1.0);

        // THE RULE: zero on the half-width, one at the far edge, and no step
        // bigger than a few per cent anywhere in between.
        assert_eq!(w(&faded, half), 0.0);
        assert_eq!(w(&faded, fade_reach_m(half)), 1.0);
        let mut worst_step = 0.0f64;
        let mut prev = w(&faded, 0.0);
        let mut d = 0.0f64;
        while d <= 3.0 * half {
            d += 0.01;
            let now = w(&faded, d);
            worst_step = worst_step.max((now - prev).abs());
            prev = now;
        }
        assert!(
            worst_step < 0.01,
            "the faded mask still steps by {worst_step} in one centimetre"
        );
    }

    /// …and the stage **refuses** a short index rather than quietly cutting.
    ///
    /// The arm above pins the rule; this one pins the WIRING, which is where the
    /// defect actually lived — `build.rs` handed `apply_detail` an index built to
    /// the corridor's own half-width for a whole wave. A `debug_assert` is the
    /// right shape for it: every CI leg builds the island fixture in `dev`, so a
    /// build that regresses the reach fails loudly there rather than shipping a
    /// crease.
    #[test]
    #[should_panic(expected = "the corridor index answers only to")]
    fn a_short_corridor_index_is_refused_rather_than_cut() {
        let half = 7.0f64;
        let line = vec![vec![
            crate::shape::Vertex3 {
                xz: DVec2::new(-500.0, 0.0),
                y: 0.0,
            },
            crate::shape::Vertex3 {
                xz: DVec2::new(500.0, 0.0),
                y: 0.0,
            },
        ]];
        let short = SegmentIndex::new(&line, half);
        let coast = all_land(4000.0);
        let mut data = two_tiles(9, 1.0, 0.1, 100.0);
        let mut plan = plain_plan(&coast, DetailBand::of(1.0, 3.11));
        plan.corridor = Some(&short);
        plan.corridor_half_m = half;
        apply_detail(&mut data, &plan);
    }
}
