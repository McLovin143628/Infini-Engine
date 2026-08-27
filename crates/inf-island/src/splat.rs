//! **The island's ground splat** (wave TER2a, clause 2): the per-sample
//! `[u8; 4]` layer weights the terrain shader blends its four materials by.
//!
//! # What this closes
//!
//! Before this module the island wrote biome **ids** and nothing else, so every
//! one of its 51.4 million samples shipped `inf_terrain::DEFAULT_WEIGHT`
//! (`[255, 0, 0, 0]`) — 100 % of layer 0 everywhere. The four `TerrainLayer`s
//! were declared and three of them were unreachable, and the terrain shader's
//! whole splat path (`sample_weights` → the four-way albedo blend → the Wave-T
//! per-layer virtual-material branch) resolved to one constant colour. The
//! island's ground was a flat colour with a procedural grain over it.
//!
//! # The three inputs, and which resolution each speaks at
//!
//! | input | lattice | why |
//! |---|---|---|
//! | biome | the classification's own 8 m lattice ([`crate::DERIVATION_PITCH_M`]), read **bilinearly** | ids stay NEAREST by ruling — a biome id is categorical and a half-forest is not a biome. The *weights* are what blends, so the feather lives here and nowhere else. |
//! | slope | the **1 m** sample grid, central difference | a cliff face is rock whatever biome it stands in, and at 8 m a 60 m cliff is eight cells wide. This is the only term that can put rock on a face the classifier called forest. |
//! | elevation | the **1 m** sample grid, the sample's own height | the shore line and the treeline are the two places a metre is visible, and the coarse lattice quantises both to 8 m steps. |
//!
//! So the *boundaries between biomes* are feathered at 8 m and the *slope and
//! shore* detail is at 1 m — which is the resolution each fact actually has.
//! Claiming 1 m feathering off an 8 m classification would be inventing it.
//!
//! # Portability
//!
//! Every arithmetic operation here is `f64` add / multiply / compare / `sqrt` /
//! `floor`, plus exactly one transcendental — [`inf_math::portable::patan2_64`],
//! for the slope angle, which is the same door [`CoarseHeights::slope_deg`]
//! already goes through so the fine rule and the coarse classification measure
//! an angle the same way. There is no `sin`, `cos`, `tan`, `powf`,
//! `exp` or `cbrt` on this path — the P14 law, whose proof for this crate is
//! `two_builds_of_one_recipe_produce_the_same_terrain` plus
//! `tests/portable_math_law.rs`. `f64::sqrt` is correctly rounded by IEEE-754
//! and is the same exemption `inf_terrain::erosion` already takes.
//!
//! # The invariant
//!
//! Every written weight sums to **exactly 255**. That is the splat's contract —
//! `sample_weights` in `terrain.wgsl` renormalises defensively, but a mask that
//! did not sum would darken or brighten the ground by however much it missed by,
//! and the renormalisation would hide it. [`quantize_255`] absorbs the rounding
//! residual on the dominant channel and then walks the rest deterministically,
//! and [`SplatStats::sum_violations`] counts what it could not fix (which is
//! zero, and is asserted rather than assumed).

use glam::DVec2;
use inf_terrain::{TerrainData, SPLAT_LAYERS};

use crate::biome::IslandBiome;
use crate::terrain::CoarseHeights;

/// Layer 0 — open grass. Plain, meadow, the grass half of a forest floor, and
/// the ground a settlement stands on.
pub const LAYER_GRASS: usize = 0;
/// Layer 1 — rock and scree. Alpine ground and every face over the slope band.
pub const LAYER_ROCK: usize = 1;
/// Layer 2 — forest floor: needle duff, leaf litter and the bare soil under it.
pub const LAYER_FOREST_FLOOR: usize = 2;
/// Layer 3 — sand and shingle. The beach band and the sea floor beyond it.
pub const LAYER_SAND: usize = 3;

/// The slope band, in **degrees**, over which a surface becomes bare rock.
///
/// The low fencepost is `rock_deg − ROCK_FEATHER_DEG` and the high one is
/// `rock_deg`, so the recipe's own rock angle is where rock reaches full
/// strength rather than where it starts — a face at exactly the author's number
/// is rock, and the twelve degrees below it are the transition. The
/// classification's `rock_deg` fencepost is the same number, so the fine rule
/// and the coarse ids agree about where rock is instead of fighting.
pub const ROCK_FEATHER_DEG: f64 = 12.0;

/// How far above the beach band the sand fades out, as a multiple of the
/// recipe's `beach_m`.
///
/// The classifier calls everything within `beach_m` of the water line Beach; this
/// carries the sand a little past it so the last of it dies in the grass rather
/// than at an 8 m fencepost.
pub const SAND_FADE_MULT: f64 = 1.6;

/// How far **below** the treeline the alpine rock starts creeping in, in metres.
pub const ALPINE_RAMP_M: f64 = 70.0;

/// One coarse cell's four layer weights, before any per-sample term.
type Mix = [f64; SPLAT_LAYERS];

/// What a biome's ground is made of.
///
/// Every row sums to 1. The rows are the island's own design decision and the
/// only place it is written down — a classifier answers *which biome*, and this
/// answers *what that biome's ground looks like*.
///
/// **Farmland is the honest compromise.** The splat is four channels wide and
/// this island binds grass, rock, forest floor and sand, so worked soil has no
/// channel of its own; it is carried as grass over a strong forest-floor
/// (soil) component, which is what a pasture reads as. A fifth ground set
/// (`Ground_Soil`) is authored and committed all the same — see the wave's
/// ledger, and the routed item beside it.
pub fn biome_mix(id: u8) -> Mix {
    match IslandBiome::from_id(id) {
        Some(IslandBiome::Beach) => mix(0.00, 0.00, 0.00, 1.00),
        Some(IslandBiome::Plain) => mix(0.92, 0.00, 0.08, 0.00),
        Some(IslandBiome::Meadow) => mix(0.80, 0.12, 0.08, 0.00),
        Some(IslandBiome::Farmland) => mix(0.70, 0.00, 0.30, 0.00),
        Some(IslandBiome::Forest) => mix(0.28, 0.02, 0.70, 0.00),
        Some(IslandBiome::Alpine) => mix(0.14, 0.84, 0.02, 0.00),
        Some(IslandBiome::Urban) => mix(0.55, 0.30, 0.15, 0.00),
        // `UNASSIGNED_BIOME` is the sea and the ground the classifier had
        // nothing to say about. Sand: a sea floor reads as sand, and an
        // unclassified metre next to a beach reads better as beach than as an
        // abrupt lawn.
        None => mix(0.00, 0.00, 0.00, 1.00),
    }
}

const fn mix(g: f64, r: f64, f: f64, s: f64) -> Mix {
    [g, r, f, s]
}

/// The coarse lattice of layer weights the fine walk interpolates.
///
/// Deliberately the **same lattice** [`CoarseHeights`] is built on, so a cell
/// index means the same thing in both and the two never drift apart by a half
/// cell.
#[derive(Clone, Debug)]
pub struct SplatField {
    pub min: DVec2,
    pub pitch: f64,
    pub nx: usize,
    pub nz: usize,
    /// Row-major `nx × nz`, each entry summing to 1.
    pub w: Vec<Mix>,
}

impl SplatField {
    /// Build the field from a classification, one cell at a time.
    ///
    /// `biome_at(i, j)` answers the id the classifier gave that cell — the same
    /// call the id stamp makes, so the two cannot disagree about what is where.
    pub fn of(coarse: &CoarseHeights, mut biome_at: impl FnMut(usize, usize) -> u8) -> Self {
        let mut w = Vec::with_capacity(coarse.nx * coarse.nz);
        for j in 0..coarse.nz {
            for i in 0..coarse.nx {
                w.push(biome_mix(biome_at(i, j)));
            }
        }
        Self {
            min: coarse.min,
            pitch: coarse.pitch,
            nx: coarse.nx,
            nz: coarse.nz,
            w,
        }
    }

    /// The bilinearly-interpolated mix at a world position.
    ///
    /// This is the whole feather: four neighbouring cells' mixes, weighted by
    /// where inside their square the sample falls. A sample sitting exactly on a
    /// cell reads that cell's mix unchanged, so the field is an interpolation of
    /// the classification rather than a smoothing of it.
    pub fn at(&self, p: DVec2) -> Mix {
        if self.nx == 0 || self.nz == 0 {
            return mix(1.0, 0.0, 0.0, 0.0);
        }
        let fx = ((p.x - self.min.x) / self.pitch).clamp(0.0, (self.nx - 1) as f64);
        let fz = ((p.y - self.min.y) / self.pitch).clamp(0.0, (self.nz - 1) as f64);
        let i0 = fx.floor() as usize;
        let j0 = fz.floor() as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.nz - 1);
        let tx = fx - i0 as f64;
        let tz = fz - j0 as f64;
        let a = &self.w[j0 * self.nx + i0];
        let b = &self.w[j0 * self.nx + i1];
        let c = &self.w[j1 * self.nx + i0];
        let d = &self.w[j1 * self.nx + i1];
        let mut out = [0.0f64; SPLAT_LAYERS];
        for k in 0..SPLAT_LAYERS {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bot = c[k] + (d[k] - c[k]) * tx;
            out[k] = top + (bot - top) * tz;
        }
        out
    }
}

/// The recipe numbers the per-sample rules read.
///
/// Passed as a record rather than a `&IslandRecipe` so the rule is testable
/// without a recipe file and so the three numbers it actually depends on are
/// visible at the call site.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplatRules {
    /// The water line, world metres.
    pub sea_level_m: f64,
    /// `[biomes] beach_m` — how far above the water line the shore band reaches.
    pub beach_m: f64,
    /// `[biomes] rock_deg` — the slope at which ground is bare rock.
    pub rock_deg: f64,
    /// `[biomes] alpine_m` — the treeline.
    pub alpine_m: f64,
}

impl SplatRules {
    /// The rules a recipe implies.
    pub fn of(recipe: &crate::recipe::IslandRecipe) -> Self {
        Self {
            sea_level_m: recipe.sea.level_m,
            beach_m: recipe.biomes.beach_m,
            rock_deg: recipe.biomes.rock_deg,
            alpine_m: recipe.biomes.alpine_m,
        }
    }
}

/// What the stamp did, for the build log and the report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SplatStats {
    /// Samples written.
    pub samples: u64,
    /// Samples whose **dominant** channel is this layer, in layer order.
    pub dominant: [u64; SPLAT_LAYERS],
    /// Samples where more than one channel carries at least a tenth — the
    /// measure of how much of the island is actually *blending* rather than
    /// picking a layer. A splat that never blends is a paint-by-numbers map.
    pub blended: u64,
    /// Samples the slope rule pushed at least half way to rock.
    pub rock_by_slope: u64,
    /// Weights that did not sum to 255. **Asserted zero**; a non-zero reading is
    /// a defect in [`quantize_255`], not a tolerance.
    pub sum_violations: u64,
}

impl SplatStats {
    /// The fraction of written samples whose dominant channel is `layer`.
    pub fn dominant_fraction(&self, layer: usize) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        self.dominant[layer.min(SPLAT_LAYERS - 1)] as f64 / self.samples as f64
    }

    /// The fraction of written samples carrying a real blend.
    pub fn blended_fraction(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        self.blended as f64 / self.samples as f64
    }

    /// The one-line summary the build log prints.
    pub fn summary(&self) -> String {
        format!(
            "splat {} samples: grass {:.1} %, rock {:.1} %, forest floor {:.1} %, \
             sand {:.1} %; {:.1} % blended, {} rock by slope",
            self.samples,
            self.dominant_fraction(LAYER_GRASS) * 100.0,
            self.dominant_fraction(LAYER_ROCK) * 100.0,
            self.dominant_fraction(LAYER_FOREST_FLOOR) * 100.0,
            self.dominant_fraction(LAYER_SAND) * 100.0,
            self.blended_fraction() * 100.0,
            self.rock_by_slope
        )
    }
}

/// Round four real weights to `u8` with an **exact** sum of 255.
///
/// The residual lands on the largest channel — the one whose intent dominates —
/// and anything a clamp could not absorb is walked off deterministically. The
/// twin of `inf_terrain::splat`'s private quantizer, which cannot be reached
/// from here; the *rule* is the same and
/// `the_quantizer_always_sums_to_255` measures this one over its whole domain
/// rather than trusting the resemblance.
pub fn quantize_255(w: Mix) -> [u8; 4] {
    // Normalise first: the per-sample rules lerp toward pure-layer targets, so
    // the sum is 1 by construction — but a NaN or a negative from a degenerate
    // recipe must not become a wrap.
    let mut clean = [0.0f64; SPLAT_LAYERS];
    let mut total = 0.0f64;
    for k in 0..SPLAT_LAYERS {
        let v = if w[k].is_finite() { w[k].max(0.0) } else { 0.0 };
        clean[k] = v;
        total += v;
    }
    // `total` is a sum of finite non-negatives by the loop above, so it is never
    // NaN here and the plain comparison is the whole test.
    if total <= 0.0 {
        return inf_terrain::DEFAULT_WEIGHT;
    }
    let mut out = [0i32; SPLAT_LAYERS];
    let mut best = 0usize;
    for k in 0..SPLAT_LAYERS {
        out[k] = (clean[k] / total * 255.0).round().clamp(0.0, 255.0) as i32;
        if clean[k] > clean[best] {
            best = k;
        }
    }
    let sum: i32 = out.iter().sum();
    out[best] = (out[best] + (255 - sum)).clamp(0, 255);
    let mut sum: i32 = out.iter().sum();
    let mut guard = 0usize;
    while sum != 255 && guard < 16 {
        let idx = guard % SPLAT_LAYERS;
        let step = if sum < 255 { 1 } else { -1 };
        let nv = (out[idx] + step).clamp(0, 255);
        if nv != out[idx] {
            out[idx] = nv;
            sum += step;
        }
        guard += 1;
    }
    [out[0] as u8, out[1] as u8, out[2] as u8, out[3] as u8]
}

/// Smoothstep between two fenceposts. Degenerate (`hi <= lo`) is a hard step at
/// `lo`, which is the only answer that does not divide by zero.
fn ramp(lo: f64, hi: f64, x: f64) -> f64 {
    if hi <= lo {
        return if x >= lo { 1.0 } else { 0.0 };
    }
    crate::shape::smooth01((x - lo) / (hi - lo))
}

/// Move `w` a fraction `t` of the way toward a pure layer.
fn toward(w: &mut Mix, layer: usize, t: f64) {
    let t = t.clamp(0.0, 1.0);
    if t <= 0.0 {
        return;
    }
    for (k, wk) in w.iter_mut().enumerate() {
        let target = if k == layer { 1.0 } else { 0.0 };
        *wk += (target - *wk) * t;
    }
}

/// The slope of the terrain at sample `(i, j)` of tile `coord`, in degrees.
///
/// Interior samples read the tile's own buffer; the four border rows read the
/// **neighbouring tile** through [`TerrainData::height_at`], so the central
/// difference is genuinely central everywhere and the weights carry no seam at
/// a tile edge. A one-sided difference at the shared row would have given the
/// two tiles two different answers for the same metre of ground, and the splat
/// would show a one-texel line every 256 m.
/// `pub(crate)` for one reason (wave TER2b audit): `detail::slope_deg_at` claims
/// to answer the same rule, and a claim that two functions agree is worth nothing
/// unless a test can call **both**. See
/// `detail::tests::the_slope_rule_agrees_with_the_splat_walk`.
pub(crate) fn slope_deg_at(
    data: &TerrainData,
    tile: &inf_terrain::TerrainTile,
    origin: DVec2,
    res: u32,
    mps: f64,
    i: u32,
    j: u32,
) -> f64 {
    let last = res - 1;
    let world = |di: f64, dj: f64| DVec2::new(origin.x + di * mps, origin.y + dj * mps);
    let h = |di: i32, dj: i32| -> f64 {
        let ii = i as i32 + di;
        let jj = j as i32 + dj;
        if ii >= 0 && jj >= 0 && ii <= last as i32 && jj <= last as i32 {
            f64::from(tile.sample(res, ii as u32, jj as u32))
        } else {
            // Off this tile: the neighbour's answer, or this sample's own height
            // at the world's edge (a one-sided difference there, which is the
            // honest answer when there is no neighbour).
            data.height_at(world(f64::from(ii), f64::from(jj)))
                .unwrap_or_else(|| f64::from(tile.sample(res, i, j)))
        }
    };
    let dx = (h(1, 0) - h(-1, 0)) / (2.0 * mps);
    let dz = (h(0, 1) - h(0, -1)) / (2.0 * mps);
    let g = (dx * dx + dz * dz).sqrt();
    inf_math::portable::patan2_64(g, 1.0).to_degrees()
}

/// **Write per-sample splat weights into every level-0 tile.**
///
/// Runs inside `BuildStep::Biomes` — the terrain-stamp step — immediately after
/// the ids, off the same classification. It is a second walk rather than a
/// second step on purpose: the step list is frozen and the fixture counts it
/// (`the_build_covers_every_recipe_step_exactly_once`), and the splat is the
/// same decision as the id, expressed as a blend.
///
/// # Weights are level-0 only
///
/// The pyramid's own rule (`inf_terrain::pyramid`, the layer-reduction table)
/// is that splat weights are **not carried** above L0 — a coarse tile is built
/// flat and never gets a weights buffer, and `pyramid.rs` asserts
/// `weights_are_default()` on every coarse tile it emits. Nothing here changes
/// that, and [`crate::terrain::build_asset`] builds the pyramid from the same
/// `TerrainData` this walked, so a coarse LOD page still shades off layer 0's
/// colour. That is the existing bound, restated because this is the wave that
/// makes it visible.
pub fn stamp_splat(data: &mut TerrainData, field: &SplatField, rules: SplatRules) -> SplatStats {
    let res = data.tile_resolution();
    let mps = data.meters_per_sample();
    let n = (res * res) as usize;
    if n == 0 {
        return SplatStats::default();
    }
    let coords: Vec<(i32, i32)> = data.tiles().map(|(c, _)| *c).collect();
    let mut st = SplatStats::default();
    let mut buf: Vec<[u8; 4]> = Vec::with_capacity(n);

    let rock_lo = rules.rock_deg - ROCK_FEATHER_DEG;
    let sand_lo = rules.beach_m * 0.5;
    let sand_hi = rules.beach_m * SAND_FADE_MULT;
    let alpine_lo = rules.alpine_m - ALPINE_RAMP_M;

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
                    let height = f64::from(tile.sample(res, i, j));
                    let mut w = field.at(p);

                    // 1. SLOPE — a face is rock whatever grows around it. The one
                    //    term the 8 m classification cannot express.
                    let slope = slope_deg_at(data, tile, origin, res, mps, i, j);
                    let rock_t = ramp(rock_lo, rules.rock_deg, slope);
                    toward(&mut w, LAYER_ROCK, rock_t);
                    if rock_t >= 0.5 {
                        st.rock_by_slope += 1;
                    }

                    // 2. ELEVATION, above the treeline — scree and bare rock
                    //    creeping in below the line and owning the ground above
                    //    it. Gated on the slope term so a flat alpine bench keeps
                    //    a little of its meadow.
                    let alp_t = ramp(alpine_lo, rules.alpine_m, height) * 0.85;
                    toward(&mut w, LAYER_ROCK, alp_t);

                    // 3. ELEVATION, at the water line — sand, and only where the
                    //    ground is gentle enough to hold it. A sea cliff is rock
                    //    down to the water.
                    let above = height - rules.sea_level_m;
                    let sand_t = (1.0 - ramp(sand_lo, sand_hi, above)) * (1.0 - rock_t);
                    toward(&mut w, LAYER_SAND, sand_t);

                    let q = quantize_255(w);
                    let sum: u32 = q.iter().map(|&v| u32::from(v)).sum();
                    if sum != 255 {
                        st.sum_violations += 1;
                    }
                    let mut best = 0usize;
                    let mut carriers = 0u32;
                    for k in 0..SPLAT_LAYERS {
                        if q[k] > q[best] {
                            best = k;
                        }
                        if q[k] >= 26 {
                            carriers += 1;
                        }
                    }
                    st.dominant[best] += 1;
                    if carriers >= 2 {
                        st.blended += 1;
                    }
                    st.samples += 1;
                    buf.push(q);
                }
            }
        }
        let Some(tile) = data.get_tile_mut(c) else {
            continue;
        };
        tile.ensure_weights(res).copy_from_slice(&buf);
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every biome's mix sums to one, and the four layers are all reachable —
    /// a row that summed to 0.9 would darken that biome's ground by a tenth
    /// after the quantizer renormalised it, and nothing downstream would say so.
    #[test]
    fn every_biome_mix_sums_to_one_and_every_layer_is_reachable() {
        let mut reached = [false; SPLAT_LAYERS];
        for id in 0u8..=8 {
            let m = biome_mix(id);
            let s: f64 = m.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-12,
                "biome {id} mixes to {s}, not 1: {m:?}"
            );
            for (k, v) in m.iter().enumerate() {
                if *v >= 0.5 {
                    reached[k] = true;
                }
            }
        }
        assert_eq!(
            reached, [true; SPLAT_LAYERS],
            "a declared layer is dominant in no biome, so the island can never \
             show it: {reached:?}"
        );
        // …and the seven named biomes are seven DISTINCT grounds. Two biomes
        // that mix identically would make the classification invisible.
        let rows: Vec<Mix> = IslandBiome::ALL.iter().map(|b| biome_mix(b.id())).collect();
        for a in 0..rows.len() {
            for b in (a + 1)..rows.len() {
                assert_ne!(
                    rows[a],
                    rows[b],
                    "{:?} and {:?} paint the same ground",
                    IslandBiome::ALL[a],
                    IslandBiome::ALL[b]
                );
            }
        }
    }

    /// **The splat invariant, over the whole domain the rules can produce.**
    ///
    /// Not a spot check: a lattice over all four channels plus the degenerate
    /// rows (all zero, one NaN, a negative) — every one sums to exactly 255.
    #[test]
    fn the_quantizer_always_sums_to_255() {
        let steps = [0.0, 0.0007, 0.13, 0.25, 0.5, 0.7331, 1.0, 3.0];
        let mut n = 0u64;
        for &a in &steps {
            for &b in &steps {
                for &c in &steps {
                    for &d in &steps {
                        let q = quantize_255([a, b, c, d]);
                        let s: u32 = q.iter().map(|&v| u32::from(v)).sum();
                        assert_eq!(s, 255, "[{a},{b},{c},{d}] -> {q:?} sums to {s}");
                        n += 1;
                    }
                }
            }
        }
        assert_eq!(n, 4096, "the sweep did not run");
        // The degenerate rows answer the documented default rather than a wrap.
        assert_eq!(quantize_255([0.0; 4]), inf_terrain::DEFAULT_WEIGHT);
        assert_eq!(quantize_255([f64::NAN; 4]), inf_terrain::DEFAULT_WEIGHT);
        assert_eq!(
            quantize_255([-1.0, -2.0, -3.0, -4.0]),
            inf_terrain::DEFAULT_WEIGHT
        );
        // A single channel is that channel, at full strength.
        assert_eq!(quantize_255([0.0, 1.0, 0.0, 0.0]), [0, 255, 0, 0]);
        // And a NaN beside a real channel does not poison it.
        assert_eq!(quantize_255([f64::NAN, 1.0, 0.0, 0.0]), [0, 255, 0, 0]);
    }

    /// The field interpolates rather than smooths: a sample **on** a cell reads
    /// that cell's mix unchanged, and a sample half way between two reads their
    /// average. The property the whole "ids stay nearest, weights blend" ruling
    /// rests on.
    #[test]
    fn the_field_is_an_interpolation_of_the_classification() {
        let coarse = CoarseHeights {
            min: DVec2::ZERO,
            pitch: 8.0,
            nx: 2,
            nz: 2,
            h: vec![0.0; 4],
            known: vec![true; 4],
        };
        // (0,0) forest, everything else beach.
        let f = SplatField::of(&coarse, |i, j| {
            if i == 0 && j == 0 {
                IslandBiome::Forest.id()
            } else {
                IslandBiome::Beach.id()
            }
        });
        let on = f.at(DVec2::new(0.0, 0.0));
        assert_eq!(
            on,
            biome_mix(IslandBiome::Forest.id()),
            "on-cell is not exact"
        );
        let far = f.at(DVec2::new(8.0, 8.0));
        assert_eq!(far, biome_mix(IslandBiome::Beach.id()));
        // Half way along the x edge between forest and beach.
        let half = f.at(DVec2::new(4.0, 0.0));
        let fm = biome_mix(IslandBiome::Forest.id());
        let bm = biome_mix(IslandBiome::Beach.id());
        for k in 0..SPLAT_LAYERS {
            assert!(
                (half[k] - (fm[k] + bm[k]) * 0.5).abs() < 1e-12,
                "channel {k}: {half:?} is not the midpoint of {fm:?} and {bm:?}"
            );
        }
        // …and the midpoint really is a BLEND: two channels carry weight there.
        let carriers = half.iter().filter(|v| **v >= 0.1).count();
        assert!(carriers >= 2, "the boundary did not feather: {half:?}");
    }

    /// **Slope beats biome.** A forest on a 60° face paints rock, and the same
    /// forest on flat ground does not — the term the 8 m classification cannot
    /// express, measured on a real `TerrainData` through the real stamp.
    #[test]
    fn a_cliff_is_rock_whatever_grows_around_it() {
        let res = 33u32;
        let mps = 1.0;
        let mut data = TerrainData::new(res, mps);
        // A ramp in +X: flat for the first half, then 2 m of rise per metre
        // (63.4 degrees) for the second.
        data.author_tile((0, 0), |x, _z| {
            if x < 16.0 {
                100.0
            } else {
                100.0 + (x - 16.0) * 2.0
            }
        });
        let coarse = CoarseHeights {
            min: DVec2::ZERO,
            pitch: 8.0,
            nx: 5,
            nz: 5,
            h: vec![100.0; 25],
            known: vec![true; 25],
        };
        let field = SplatField::of(&coarse, |_, _| IslandBiome::Forest.id());
        let rules = SplatRules {
            sea_level_m: 0.0,
            beach_m: 14.0,
            rock_deg: 36.0,
            alpine_m: 620.0,
        };
        let st = stamp_splat(&mut data, &field, rules);
        assert_eq!(st.samples, u64::from(res * res));
        assert_eq!(st.sum_violations, 0, "the invariant broke");
        let tile = data.get_tile((0, 0)).expect("the tile");
        let flat = tile.weight_sample(res, 4, 16);
        let steep = tile.weight_sample(res, 28, 16);
        assert!(
            flat[LAYER_FOREST_FLOOR] > flat[LAYER_ROCK],
            "flat forest painted rock: {flat:?}"
        );
        assert!(
            steep[LAYER_ROCK] > 200,
            "a 63-degree face did not paint rock: {steep:?}"
        );
        assert!(
            st.rock_by_slope > 0,
            "no sample was pushed to rock by slope"
        );
        // And the whole tile obeys the invariant, sample for sample.
        for j in 0..res {
            for i in 0..res {
                let w = tile.weight_sample(res, i, j);
                let s: u32 = w.iter().map(|&v| u32::from(v)).sum();
                assert_eq!(s, 255, "({i},{j}) sums to {s}");
            }
        }
    }

    /// **The weights carry no seam at a tile edge.** Two tiles share a row of
    /// samples, and a one-sided slope difference there would give the two
    /// different weights for the same metre of ground. Measured across a real
    /// two-tile boundary on a ramp that crosses it.
    #[test]
    fn a_shared_tile_edge_paints_one_answer() {
        let res = 9u32;
        let mps = 1.0;
        let mut data = TerrainData::new(res, mps);
        let h = |x: f64| 50.0 + x * 0.9;
        for c in [(0, 0), (1, 0)] {
            data.author_tile(c, |x, _z| h(x));
        }
        let coarse = CoarseHeights {
            min: DVec2::ZERO,
            pitch: 8.0,
            nx: 4,
            nz: 4,
            h: vec![50.0; 16],
            known: vec![true; 16],
        };
        let field = SplatField::of(&coarse, |_, _| IslandBiome::Plain.id());
        let rules = SplatRules {
            sea_level_m: 0.0,
            beach_m: 14.0,
            rock_deg: 36.0,
            alpine_m: 620.0,
        };
        stamp_splat(&mut data, &field, rules);
        let a = data.get_tile((0, 0)).expect("tile a");
        let b = data.get_tile((1, 0)).expect("tile b");
        for j in 0..res {
            assert_eq!(
                a.weight_sample(res, res - 1, j),
                b.weight_sample(res, 0, j),
                "row {j} of the shared edge disagrees"
            );
        }
        // ANTI-VACUITY: the ramp really is steep enough for the slope term to be
        // doing something at that edge — otherwise both sides read the biome mix
        // and the arm would pass over a defect it cannot see.
        let slope = slope_deg_at(&data, a, data.tile_origin_xz((0, 0)), res, mps, res - 1, 4);
        assert!(
            slope > 40.0,
            "the fixture is too flat to test the seam: {slope}"
        );
    }

    /// The shore rule: sand at the water line, grass a few metres up, and a
    /// **gradient** between them rather than a step.
    #[test]
    fn the_shore_fades_from_sand_into_the_ground_behind_it() {
        let res = 65u32;
        let mps = 1.0;
        let mut data = TerrainData::new(res, mps);
        data.author_tile((0, 0), |x, _z| x * 0.5);
        let coarse = CoarseHeights {
            min: DVec2::ZERO,
            pitch: 8.0,
            nx: 9,
            nz: 9,
            h: vec![10.0; 81],
            known: vec![true; 81],
        };
        let field = SplatField::of(&coarse, |_, _| IslandBiome::Plain.id());
        let rules = SplatRules {
            sea_level_m: 0.0,
            beach_m: 14.0,
            rock_deg: 36.0,
            alpine_m: 620.0,
        };
        stamp_splat(&mut data, &field, rules);
        let tile = data.get_tile((0, 0)).expect("the tile");
        let sand_at = |i: u32| tile.weight_sample(res, i, 32)[LAYER_SAND];
        assert!(
            sand_at(2) > 200,
            "the water line is not sand: {}",
            sand_at(2)
        );
        assert!(sand_at(60) < 10, "the high ground is sand: {}", sand_at(60));
        // Monotone, and with real intermediate values — a step would jump from
        // 255 to 0 with nothing in between.
        let mut intermediate = 0;
        for i in 0..res {
            if (10..=245).contains(&sand_at(i)) {
                intermediate += 1;
            }
        }
        assert!(
            intermediate >= 8,
            "the shore is a step, not a fade: {intermediate} intermediate samples"
        );
        for i in 1..res {
            assert!(
                sand_at(i) <= sand_at(i - 1),
                "sand rose going inland at {i}: {} after {}",
                sand_at(i),
                sand_at(i - 1)
            );
        }
    }

    /// The stamp is a **pure function of its inputs**: two runs over one
    /// terrain produce byte-identical weights. The crate-level byte-identity arm
    /// covers the whole build; this one isolates the new walk so a
    /// non-determinism introduced here is named here.
    #[test]
    fn two_stamps_of_one_terrain_agree_byte_for_byte() {
        let res = 33u32;
        let build = || {
            let mut data = TerrainData::new(res, 1.0);
            data.author_tile((0, 0), |x, z| 20.0 + (x * 0.31) + (z * 0.17));
            let coarse = CoarseHeights {
                min: DVec2::ZERO,
                pitch: 8.0,
                nx: 5,
                nz: 5,
                h: vec![20.0; 25],
                known: vec![true; 25],
            };
            let field = SplatField::of(&coarse, |i, j| {
                IslandBiome::ALL[(i + j) % IslandBiome::ALL.len()].id()
            });
            let st = stamp_splat(
                &mut data,
                &field,
                SplatRules {
                    sea_level_m: 0.0,
                    beach_m: 14.0,
                    rock_deg: 36.0,
                    alpine_m: 620.0,
                },
            );
            let w: Vec<[u8; 4]> = data.get_tile((0, 0)).expect("tile").weights().to_vec();
            (st, w)
        };
        let (a, wa) = build();
        let (b, wb) = build();
        assert_eq!(a, b, "the stats moved between two runs");
        assert_eq!(wa, wb, "the weights moved between two runs");
        // ANTI-VACUITY: the fixture really did paint something other than the
        // default, or two runs of nothing would agree perfectly.
        assert!(
            wa.iter().any(|w| *w != inf_terrain::DEFAULT_WEIGHT),
            "the stamp wrote the default everywhere"
        );
    }

    /// `SplatRules::of` reads the four recipe numbers it names and no others —
    /// a rule that silently read a different field would be invisible until an
    /// author changed the one it was supposed to read.
    #[test]
    fn the_rules_come_off_the_recipe_fields_they_name() {
        let recipe = crate::recipe::IslandRecipe::parse(
            &crate::recipe::tests::tiny_recipe_text(),
            std::path::Path::new("."),
        )
        .expect("the tiny recipe parses");
        let r = SplatRules::of(&recipe);
        assert_eq!(r.sea_level_m, recipe.sea.level_m);
        assert_eq!(r.beach_m, recipe.biomes.beach_m);
        assert_eq!(r.rock_deg, recipe.biomes.rock_deg);
        assert_eq!(r.alpine_m, recipe.biomes.alpine_m);
    }
}
