//! The terrain **LOD pyramid** (P16.3): deterministic 2:1 downsampling of height
//! tiles, so far terrain never needs full-resolution residency.
//!
//! # The construction
//!
//! Level 0 is the authored terrain: `res × res` samples per tile, spaced
//! `mps` metres apart, tile span `(res − 1) · mps`, shared edges between
//! neighbours. Level `n + 1` keeps the same `res` but **doubles the spacing**, so
//! one coarse tile covers exactly the 2 × 2 block of fine tiles
//! `(2TX + a, 2TZ + b)` and spans `2 ×` the world.
//!
//! Write the 2 × 2 fine block as one combined index space `i ∈ [0, 2(res − 1)]`
//! (fine tile `a` contributes `i ∈ [a(res − 1), a(res − 1) + res − 1]`, the two
//! halves overlapping in exactly the shared edge sample). Coarse sample `I` sits
//! at combined index `i = 2I`, and `I ∈ [0, res − 1]` covers `i ∈ [0, 2(res−1)]`
//! exactly. So the coarse tile is the fine block **decimated 2:1** — every coarse
//! sample *is* a fine sample, bit-for-bit.
//!
//! # Why decimation and not a filter
//!
//! A tent/box filter would need fine samples from *outside* the 2 × 2 block at the
//! block's own edges. Terrain is **sparse** (only authored tiles exist), so those
//! neighbours may simply not be there, and clamping instead would make coarse tile
//! `(TX, TZ)`'s far edge disagree with coarse tile `(TX + 1, TZ)`'s near edge —
//! a crack in the LOD mesh, exactly the seam the shared-edge tiling exists to
//! prevent. Decimation is the only 2:1 reduction that keeps the shared-edge
//! invariant with no neighbour fetch: the coarse shared sample is *the same fine
//! sample* both coarse tiles decimate, so **edge continuity is inherited from the
//! fine level, exactly** (`shared_edges_survive_downsampling`). It is also trivially
//! deterministic — no floating-point reduction order to get wrong. A filtered
//! (neighbour-aware) variant is a documented follow-up for the outer rings.
//!
//! ## The bit-exactness caveat: uniform height anchors
//!
//! "Bit-exact" holds while the fine tiles across a seam share one `f64`
//! `origin.y`. A coarse tile takes its anchor from one tile of its 2 × 2 block and
//! rebases the rest as `(fine.origin.y + fine_offset) − coarse.origin.y` in `f64`
//! before storing an `f32`; if two neighbouring coarse blocks pick *different*
//! anchors, the same world height can round to `f32` differently on either side
//! and the shared sample can disagree by an ULP. Every terrain this engine
//! creates today anchors at `origin.y == 0.0` — [`TerrainData::author_tile`],
//! `write_region`, the heightmap import and `TerrainTile::flat` all do — so the
//! property is exact in practice and the test asserts equality, not a tolerance.
//! If multi-anchor terrains ever ship (a planetary patch rebasing per region), the
//! coarse anchor must be derived from the *block coordinate* rather than from
//! whichever fine tile happens to be present, so it agrees across seams by
//! construction. Tracked as a P16.4+ follow-up.
//!
//! # Why weights and biomes stay full-res
//!
//! Only **heights** are pyramided. Splat weights are an authoring-resolution paint
//! layer: at the distance an LOD-`n` tile is drawn, one weight sample is far below
//! a screen texel, and decimating an already-quantized `[u8; 4]` blend that must
//! stay normalized to ≈ 255 either drifts the normalization or needs a
//! renormalizing filter — for pixels nobody can resolve. Coarse tiles therefore
//! carry the sparse [`DEFAULT_WEIGHT`](crate::DEFAULT_WEIGHT) default (empty
//! buffer), which also keeps the pyramid's byte cost to heights alone; the
//! renderer samples the level-0 weight page for near rings, where splat detail is
//! actually visible. Biome/foliage layers follow the same rule for the same reason.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;

use crate::data::TerrainData;
use crate::tile::TerrainTile;

/// Default cap on generated coarse levels. 8 levels is a 256 × reduction in
/// world-space tile count — a 16 k × 16 k source (4096 level-0 tiles at 256²)
/// collapses to a handful long before the cap.
pub const DEFAULT_MAX_PYRAMID_LEVELS: u32 = 8;

/// Default stop condition: once a level holds this many tiles or fewer, the whole
/// level fits in residency at once and a coarser one buys nothing.
pub const DEFAULT_MIN_PYRAMID_TILES: usize = 4;

/// Knobs for [`build_pyramid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyramidOptions {
    /// Maximum number of **coarse** levels to generate (level 0 is never counted).
    pub max_levels: u32,
    /// Stop as soon as a generated level has at most this many tiles.
    pub min_tiles: usize,
}

impl Default for PyramidOptions {
    fn default() -> Self {
        Self {
            max_levels: DEFAULT_MAX_PYRAMID_LEVELS,
            min_tiles: DEFAULT_MIN_PYRAMID_TILES,
        }
    }
}

/// One generated **coarse** LOD level (`lod ≥ 1`).
///
/// Tiles keep the terrain's `tile_resolution`; only the spacing changes, so a
/// level-`n` tile spans `2ⁿ ×` a level-0 tile.
#[derive(Debug, Clone, PartialEq)]
pub struct PyramidLevel {
    /// The level index (`1` for the first coarse level).
    pub lod: u32,
    /// World units between adjacent samples **at this level**
    /// (`level0_mps · 2^lod`).
    pub meters_per_sample: f64,
    /// The level's tiles, keyed by this level's own grid coordinate.
    pub tiles: BTreeMap<(i32, i32), TerrainTile>,
}

impl PyramidLevel {
    /// World size of one tile edge at this level.
    #[inline]
    pub fn tile_span(&self, tile_resolution: u32) -> f64 {
        (tile_resolution as f64 - 1.0) * self.meters_per_sample
    }
}

/// The **shape** of one pyramid level: which coarse tiles exist and at what
/// spacing, with no tile data.
///
/// [`plan_pyramid`] computes the whole shape from the level-0 *coordinate set*
/// alone, which is what lets a partial rebuild ([`crate::writeback`]) know
/// exactly which levels and which tiles a full [`build_pyramid`] would have
/// produced without decimating a single sample.
#[derive(Debug, Clone, PartialEq)]
pub struct PyramidPlanLevel {
    /// The level index (`1` for the first coarse level).
    pub lod: u32,
    /// World units between adjacent samples at this level (`level0 · 2^lod`).
    pub meters_per_sample: f64,
    /// The level's tile coordinates.
    pub coords: BTreeSet<(i32, i32)>,
}

/// The coarse coordinates one level up from `coords` (floor-halved, deduplicated).
///
/// The *only* place the fine→coarse grouping rule lives, so
/// [`downsample_tiles`], [`plan_pyramid`] and the partial rebuild cannot drift
/// apart about which fine tiles share a parent.
pub fn coarsen_coords(coords: &BTreeSet<(i32, i32)>) -> BTreeSet<(i32, i32)> {
    coords
        .iter()
        .map(|&(tx, tz)| (tx.div_euclid(2), tz.div_euclid(2)))
        .collect()
}

/// Plan the pyramid a [`build_pyramid`] over this level-0 coordinate set would
/// produce: which levels exist, their spacing, and their tile coordinates.
///
/// Coordinates alone decide the shape — the stop conditions
/// ([`PyramidOptions::min_tiles`] / [`max_levels`](PyramidOptions::max_levels))
/// are counts, and the fine→coarse grouping is pure arithmetic — so this is a
/// cheap, allocation-light mirror of the real build. [`build_pyramid`] is
/// implemented *on top of it*, which is what makes the mirror exact rather than a
/// second implementation that has to be kept in sync.
pub fn plan_pyramid(
    level0: &BTreeSet<(i32, i32)>,
    meters_per_sample: f64,
    opts: PyramidOptions,
) -> Vec<PyramidPlanLevel> {
    let mut out = Vec::new();
    if level0.len() <= opts.min_tiles || opts.max_levels == 0 {
        return out;
    }
    let mut cur = coarsen_coords(level0);
    let mut mps = meters_per_sample;
    for lod in 1..=opts.max_levels {
        mps *= 2.0;
        let stop = cur.len() <= opts.min_tiles || lod == opts.max_levels;
        let next = if stop {
            BTreeSet::new()
        } else {
            coarsen_coords(&cur)
        };
        out.push(PyramidPlanLevel {
            lod,
            meters_per_sample: mps,
            coords: std::mem::replace(&mut cur, next),
        });
        if stop {
            break;
        }
    }
    out
}

/// Build the coarse LOD levels of `data`, from level 1 upward.
///
/// Returns levels in ascending `lod` order (never including level 0 — that *is*
/// `data`). Generation stops at [`PyramidOptions::max_levels`], or as soon as a
/// level has at most [`PyramidOptions::min_tiles`] tiles. A terrain that is
/// already that small yields an empty pyramid.
///
/// Pure and deterministic: the output is a function of the input tiles alone
/// (`BTreeMap` iteration order + decimation, no floating-point reduction).
pub fn build_pyramid(data: &TerrainData, opts: PyramidOptions) -> Vec<PyramidLevel> {
    let res = data.tile_resolution();
    let level0: BTreeMap<(i32, i32), &TerrainTile> = data.tiles().map(|(&c, t)| (c, t)).collect();
    let coords: BTreeSet<(i32, i32)> = level0.keys().copied().collect();
    let plan = plan_pyramid(&coords, data.meters_per_sample(), opts);
    let mut levels: Vec<PyramidLevel> = Vec::with_capacity(plan.len());

    // Level 1 decimates the borrowed level-0 map; every level after decimates the
    // owned one it just produced (moved out, never cloned).
    for step in plan {
        // The *input* level's spacing: level 0's for the first step, else the
        // previous level's (this level's, halved).
        let src_mps = step.meters_per_sample * 0.5;
        let tiles = match levels.last() {
            None => downsample_tiles(res, src_mps, &level0),
            Some(prev) => downsample_tiles(res, src_mps, &prev.tiles),
        };
        debug_assert_eq!(
            tiles.keys().copied().collect::<BTreeSet<_>>(),
            step.coords,
            "plan_pyramid disagreed with downsample_tiles at lod {}",
            step.lod
        );
        levels.push(PyramidLevel {
            lod: step.lod,
            meters_per_sample: step.meters_per_sample,
            tiles,
        });
    }
    levels
}

/// Decimate one level 2:1 into the next coarser one.
///
/// `mps` is the **input** level's metres-per-sample (the output's is `2 · mps`);
/// `res` is the terrain-wide tile resolution, unchanged by the reduction. Generic
/// over `Borrow<TerrainTile>` so level 1 can decimate a map of *borrowed* level-0
/// tiles without cloning the whole authored terrain.
pub fn downsample_tiles<T: std::borrow::Borrow<TerrainTile>>(
    res: u32,
    mps: f64,
    tiles: &BTreeMap<(i32, i32), T>,
) -> BTreeMap<(i32, i32), TerrainTile> {
    let coarse_coords = coarsen_coords(&tiles.keys().copied().collect());
    coarse_coords
        .into_iter()
        .map(|c| (c, downsample_block(res, mps, c, tiles)))
        .collect()
}

/// Decimate the single 2 × 2 fine block under coarse coordinate `coarse` into one
/// coarse tile — **the** reduction kernel, and the reason a partial pyramid
/// rebuild is byte-equal to a full one.
///
/// `tiles` needs only to answer for the block's four members
/// `(2·cx + a, 2·cz + b), a,b ∈ {0,1}`; anything else in the map is ignored. So a
/// full rebuild passes the whole level ([`downsample_tiles`]) and a write-back
/// passes a four-entry map ([`crate::writeback`]) — same code, same bytes.
///
/// `mps` is the **input** level's metres-per-sample (the output's is `2 · mps`).
/// A block with no present tile at all yields a flat tile anchored at `0`.
pub fn downsample_block<T: std::borrow::Borrow<TerrainTile>>(
    res: u32,
    mps: f64,
    coarse: (i32, i32),
    tiles: &BTreeMap<(i32, i32), T>,
) -> TerrainTile {
    let res = res.max(2);
    let coarse_span = (res as f64 - 1.0) * mps * 2.0;
    let (cx, cz) = coarse;
    // The `f64` height anchor: the first present tile of the 2×2 block in a
    // fixed scan order, so the choice is deterministic and the coarse `f32`
    // offsets stay in the same numeric neighbourhood as the fine ones (the
    // precision doctrine — never rebase a planetary anchor onto 0).
    let anchor_y = BLOCK_SCAN
        .iter()
        .find_map(|&(a, b)| tiles.get(&(2 * cx + a, 2 * cz + b)))
        .map(|t| t.borrow().origin.y)
        .unwrap_or(0.0);
    let mut out = TerrainTile::flat(
        res,
        DVec3::new(cx as f64 * coarse_span, anchor_y, cz as f64 * coarse_span),
    );
    for j in 0..res {
        for i in 0..res {
            // Coarse sample (i, j) is combined fine index (2i, 2j).
            let h = combined_sample(tiles, res, (cx, cz), 2 * i, 2 * j);
            let off = h.map(|h| h - anchor_y).unwrap_or(0.0);
            out.set_sample(res, i, j, off as f32);
        }
    }
    out
}

/// The 2 × 2 fine-block scan order, fixed so every choice made over it is
/// deterministic.
const BLOCK_SCAN: [(i32, i32); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

/// Absolute world height at combined fine index `(i, j)` of the 2 × 2 block under
/// coarse coordinate `(cx, cz)`, or `None` when every candidate fine tile is
/// absent (a coverage hole — the caller writes a flat `0` offset there).
///
/// Combined index `i ∈ [0, 2(res − 1)]` maps to fine sub-tile `a` and local index
/// `li`; at `i == res − 1` **both** sub-tiles hold the sample (that is the shared
/// edge), so both are tried, `a = 0` first.
fn combined_sample<T: std::borrow::Borrow<TerrainTile>>(
    tiles: &BTreeMap<(i32, i32), T>,
    res: u32,
    (cx, cz): (i32, i32),
    i: u32,
    j: u32,
) -> Option<f64> {
    for (a, li) in axis_candidates(res, i) {
        for (b, lj) in axis_candidates(res, j) {
            if let Some(t) = tiles.get(&(2 * cx + a, 2 * cz + b)) {
                return Some(t.borrow().world_height(res, li, lj));
            }
        }
    }
    None
}

/// `(sub_tile, local_index)` candidates for one combined-index axis value, most
/// preferred first. One candidate everywhere except the shared edge, which has two.
fn axis_candidates(res: u32, i: u32) -> impl Iterator<Item = (i32, u32)> {
    let edge = res - 1;
    let primary = if i < edge { (0, i) } else { (1, i - edge) };
    let shared = (i == edge).then_some((0, edge));
    // The shared-edge sample belongs to the LOW sub-tile first (so a lone
    // `(2cx, ·)` tile still contributes its far edge), then the high one.
    shared.into_iter().chain(std::iter::once(primary))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TerrainData;
    use glam::DVec2;

    /// A terrain of `n × n` tiles authored from one **polynomial** world height
    /// function — seamless by construction, and bit-portable (no `std` trig; the
    /// P14 law).
    fn poly_terrain(res: u32, mps: f64, n: i32) -> TerrainData {
        let mut t = TerrainData::new(res, mps);
        for tz in 0..n {
            for tx in 0..n {
                t.author_tile((tx, tz), height_fn);
            }
        }
        t
    }

    fn height_fn(x: f64, z: f64) -> f64 {
        x * 0.25 - z * 0.125 + x * z * 0.001 + 5.0
    }

    #[test]
    fn pyramid_is_deterministic_across_rebuilds() {
        let t = poly_terrain(5, 2.0, 8);
        let a = build_pyramid(&t, PyramidOptions::default());
        let b = build_pyramid(&t, PyramidOptions::default());
        assert_eq!(a, b, "the pyramid is a pure function of the input tiles");
        assert!(!a.is_empty(), "64 tiles must produce coarse levels");
        // …and independent of insertion order: authoring the same tiles in a
        // different sequence yields the identical pyramid (BTreeMap ordering).
        let mut shuffled = TerrainData::new(5, 2.0);
        for tz in (0..8).rev() {
            for tx in (0..8).rev() {
                shuffled.author_tile((tx, tz), height_fn);
            }
        }
        assert_eq!(build_pyramid(&shuffled, PyramidOptions::default()), a);
    }

    #[test]
    fn levels_halve_tile_count_and_double_spacing() {
        let t = poly_terrain(5, 2.0, 8); // 64 tiles
        let p = build_pyramid(&t, PyramidOptions::default());
        assert_eq!(
            p.iter().map(|l| l.lod).collect::<Vec<_>>(),
            vec![1, 2],
            "8×8 → 4×4 → 2×2, stopping as soon as a level has ≤ 4 tiles"
        );
        assert_eq!(p[0].tiles.len(), 16);
        assert_eq!(p[1].tiles.len(), 4);
        assert_eq!(p[0].meters_per_sample, 4.0);
        assert_eq!(p[1].meters_per_sample, 8.0);
        // A coarse tile spans exactly 2^lod level-0 tiles.
        assert_eq!(p[0].tile_span(5), t.tile_span() * 2.0);
        assert_eq!(p[1].tile_span(5), t.tile_span() * 4.0);
    }

    #[test]
    fn small_terrain_yields_no_pyramid() {
        let t = poly_terrain(5, 1.0, 2); // 4 tiles == min_tiles
        assert!(build_pyramid(&t, PyramidOptions::default()).is_empty());
    }

    #[test]
    fn max_levels_caps_generation() {
        let t = poly_terrain(5, 1.0, 8);
        let p = build_pyramid(
            &t,
            PyramidOptions {
                max_levels: 1,
                ..Default::default()
            },
        );
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].lod, 1);
    }

    #[test]
    fn coarse_samples_are_exactly_the_fine_samples_they_decimate() {
        // Decimation, not filtering: every coarse sample must be BIT-identical to
        // the level-0 sample it was taken from (no averaging, no interpolation).
        let res = 5;
        let edge = res - 1;
        let t = poly_terrain(res, 2.0, 4);
        let p = build_pyramid(&t, PyramidOptions::default());
        let l1 = &p[0];
        for (&(cx, cz), tile) in &l1.tiles {
            for j in 0..res {
                for i in 0..res {
                    // Combined fine index (2i, 2j) → sub-tile + local index (the
                    // shared-edge sample resolves onto the LOW sub-tile).
                    let (a, li) = if 2 * i <= edge {
                        (0, 2 * i)
                    } else {
                        (1, 2 * i - edge)
                    };
                    let (b, lj) = if 2 * j <= edge {
                        (0, 2 * j)
                    } else {
                        (1, 2 * j - edge)
                    };
                    let fine = t.get_tile((2 * cx + a, 2 * cz + b)).unwrap();
                    assert_eq!(
                        tile.world_height(res, i, j),
                        fine.world_height(res, li, lj),
                        "coarse ({cx},{cz}) sample ({i},{j}) is not the fine sample"
                    );
                }
            }
        }
        // And the decimated level agrees with a world-space query of the source.
        let probe = DVec2::new(l1.tile_span(res), l1.meters_per_sample * 2.0);
        assert!(t.height_at(probe).is_some());
    }

    #[test]
    fn shared_edges_survive_downsampling() {
        // The seam guarantee, inherited level by level: a coarse tile's far edge
        // samples are bit-identical to its +X / +Z neighbour's near edge samples.
        let res = 5;
        let t = poly_terrain(res, 2.0, 8);
        for level in build_pyramid(&t, PyramidOptions::default()) {
            for (&(cx, cz), tile) in &level.tiles {
                if let Some(right) = level.tiles.get(&(cx + 1, cz)) {
                    for j in 0..res {
                        assert_eq!(
                            tile.world_height(res, res - 1, j),
                            right.world_height(res, 0, j),
                            "lod {} X-seam at ({cx},{cz}) row {j}",
                            level.lod
                        );
                    }
                }
                if let Some(down) = level.tiles.get(&(cx, cz + 1)) {
                    for i in 0..res {
                        assert_eq!(
                            tile.world_height(res, i, res - 1),
                            down.world_height(res, i, 0),
                            "lod {} Z-seam at ({cx},{cz}) col {i}",
                            level.lod
                        );
                    }
                }
            }
        }
    }

    /// [`plan_pyramid`] must describe **exactly** the levels [`build_pyramid`]
    /// produces — the invariant the partial write-back rebuild leans on to know
    /// which coarse tiles a full rebuild would have emitted without emitting them.
    #[test]
    fn the_plan_matches_the_built_pyramid() {
        for n in [1, 2, 3, 5, 8, 9, 16] {
            for opts in [
                PyramidOptions::default(),
                PyramidOptions {
                    max_levels: 1,
                    min_tiles: 4,
                },
                PyramidOptions {
                    max_levels: 8,
                    min_tiles: 1,
                },
                PyramidOptions {
                    max_levels: 0,
                    min_tiles: 4,
                },
            ] {
                let t = poly_terrain(5, 2.0, n);
                let built = build_pyramid(&t, opts);
                let coords: BTreeSet<(i32, i32)> = t.tiles().map(|(&c, _)| c).collect();
                let plan = plan_pyramid(&coords, t.meters_per_sample(), opts);
                assert_eq!(plan.len(), built.len(), "{n}×{n} tiles, {opts:?}");
                for (p, b) in plan.iter().zip(&built) {
                    assert_eq!(p.lod, b.lod);
                    assert_eq!(p.meters_per_sample, b.meters_per_sample);
                    assert_eq!(
                        p.coords,
                        b.tiles.keys().copied().collect::<BTreeSet<_>>(),
                        "lod {} coords disagree for {n}×{n}",
                        p.lod
                    );
                }
            }
        }
    }

    /// The reduction kernel is shared: decimating one block in isolation is
    /// **bit-identical** to the same tile inside a whole-level decimation.
    #[test]
    fn one_block_decimates_identically_to_a_whole_level() {
        let t = poly_terrain(5, 2.0, 4);
        let fine: BTreeMap<(i32, i32), &TerrainTile> = t.tiles().map(|(&c, ti)| (c, ti)).collect();
        let whole = downsample_tiles(5, 2.0, &fine);
        for (&coarse, expect) in &whole {
            // Feed the kernel ONLY the four block members, as a write-back does.
            let block: BTreeMap<(i32, i32), TerrainTile> = BLOCK_SCAN
                .iter()
                .filter_map(|&(a, b)| {
                    let c = (2 * coarse.0 + a, 2 * coarse.1 + b);
                    fine.get(&c).map(|t| (c, (*t).clone()))
                })
                .collect();
            assert_eq!(
                &downsample_block(5, 2.0, coarse, &block),
                expect,
                "block {coarse:?} differs from the whole-level reduction"
            );
        }
    }

    #[test]
    fn negative_coordinates_group_with_floor_semantics() {
        // Grouping must FLOOR, not truncate toward zero: tile -1 belongs to coarse
        // block -1 (floor(-1/2) = -1), while a plain `-1 / 2` would say 0 and merge
        // it with the positive side. -3 and -4 both floor to -2.
        let mut t = TerrainData::new(5, 1.0);
        for c in [(-4, -4), (-3, -3), (-1, -1), (0, 0)] {
            t.author_tile(c, height_fn);
        }
        let fine: BTreeMap<(i32, i32), &TerrainTile> =
            t.tiles().map(|(&c, tile)| (c, tile)).collect();
        let coarse = downsample_tiles(5, 1.0, &fine);
        let keys: Vec<_> = coarse.keys().copied().collect();
        assert_eq!(keys, vec![(-2, -2), (-1, -1), (0, 0)]);
    }

    #[test]
    fn coarse_tiles_keep_the_sparse_weight_default() {
        // Weights are level-0-only (see the module docs): a coarse tile must
        // serialize exactly like a never-painted one.
        let mut t = poly_terrain(5, 1.0, 4);
        t.get_tile_mut((0, 0))
            .unwrap()
            .set_weight_sample(5, 2, 2, [10, 20, 30, 195]);
        for level in build_pyramid(&t, PyramidOptions::default()) {
            for tile in level.tiles.values() {
                assert!(
                    tile.weights_are_default(),
                    "lod {} leaked weights",
                    level.lod
                );
            }
        }
    }

    #[test]
    fn coverage_holes_fill_flat_without_panicking() {
        // A 2×2 coarse block with only one fine tile still produces a tile; the
        // uncovered quadrants read as flat offsets at the anchor.
        let mut t = TerrainData::new(5, 1.0);
        for c in [(0, 0), (4, 0), (0, 4), (4, 4), (8, 8)] {
            t.author_tile(c, height_fn);
        }
        let p = build_pyramid(&t, PyramidOptions::default());
        assert!(!p.is_empty());
        let l1 = &p[0];
        let tile = l1.tiles.get(&(0, 0)).expect("block (0,0) exists");
        // Sample (0,0) came from the real fine tile (h(0,0) = 5); the far corner
        // had no source tile at all and reads as a flat offset.
        assert_eq!(tile.sample(5, 0, 0), height_fn(0.0, 0.0) as f32);
        assert_eq!(tile.sample(5, 4, 4), 0.0, "hole reads flat");
    }
}
