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
//! # Per-layer reduction rules (P19.3)
//!
//! A tile carries four parallel layers and they do **not** reduce the same way,
//! because they do not mean the same thing. The rule per layer, stated once:
//!
//! | layer | rule | why |
//! |---|---|---|
//! | heights (`f32` offsets) | **decimate** — the coarse sample *is* the fine sample | a point value on a lattice; anything else cracks the LOD mesh (above) |
//! | splat weights (`[u8; 4]`) | **not carried** (sparse default) | an authoring-resolution paint layer nobody can resolve at LOD `n`; see below |
//! | data map **flow** (`m³`) | **sum** the footprint | *extensive*: a coarse cell covers 4 fine cells' area, so it shipped their total water |
//! | data maps **deposition / wear** (`m`) | **mean** the footprint | *intensive*: metres of height change is a per-area value, and the mean is the one that survives a level |
//! | biome ids (`u8`) | **majority vote**, ties to the **lowest id** | categorical: the midpoint of biome 3 and biome 7 is one of the two |
//!
//! ## The footprint, and why the shared-edge ring degenerates
//!
//! A coarse sample `(I, J)` reduces the fine samples it covers, expressed in the
//! block's combined index space: `{2I, 2I+1} × {2J, 2J+1}` — the **four children**
//! (two per axis) the 2:1 reduction assigns to it, the first of which is the very
//! sample the heights decimate.
//!
//! **Except on the shared-edge ring** (`I ∈ {0, res−1}` or `J ∈ {0, res−1}`),
//! where the window degenerates to the single sample — i.e. the ring *decimates,
//! exactly like the heights*. That is not an optimization, it is the seam
//! guarantee: a coarse tile's far edge and its `+X` neighbour's near edge are the
//! **same fine world sample**, so both reduce to bit-identical bytes only if
//! neither aggregates. Any non-degenerate window at the ring would have to reach
//! fine samples from *outside* the 2 × 2 block — which terrain sparsity forbids
//! (the block may be all that exists) and which would break the invariant the
//! partial write-back rebuild rests on ([`downsample_block`] is fed only the
//! block's four members). It is the same trade the heights already make, one
//! layer up. On the far edge it costs nothing extra: clipping `2I+1` at
//! `2(res−1)` *is* the degenerate window.
//!
//! The ring's window is **not** uniformly one sample: a corner reduces 1 child, a
//! non-corner edge sample reduces 2 (one axis degenerate, one not), the interior
//! reduces 4. Only the **mean** channels are unaffected by that (a mean of 1 or 2
//! equal-area children is still the right per-area value, and a categorical id on
//! the ring is simply *correct* — it is the id painted there). Only **flow**, the
//! summed channel, loses anything.
//!
//! **What flow loses, measured.** Per axis the windows cover combined indices
//! `{0} ∪ {2 … 2res−2}`, i.e. `2res−2` of the block's `2res−1` — the single
//! combined index **`1`** falls in no window (`I = 0` degenerated to `{0}` and
//! `I = 1` starts at `{2, 3}`). Every other odd index *is* covered: `2res−3`
//! belongs to `I = res−2`'s window `{2res−4, 2res−3}`. So the uncovered fraction
//! of the fine block is `1 − ((2res−2)/(2res−1))²` — **0.39 % at `res = 256`**
//! — and that is the whole of flow's under-count. Stated, not hidden.
//!
//! ## Why flow **sums** and the metre channels **average**
//!
//! [`DataMapKind`](crate::DataMapKind) defines all three as raw, monotone
//! **accumulators**, but they are not the same *kind* of quantity, and the
//! reduction has to follow the dimension rather than the word "accumulator":
//!
//! * **flow** is `Σ dt · outflow` in **m³** — a volume the cell shipped. It is
//!   **extensive**: double the area, double the number. A coarse cell stands for
//!   four fine cells' area, so it shipped their **total**, and summing is what
//!   keeps `Σ over a world region` invariant across levels — the property a
//!   flow-accumulation mask actually reads. (Its per-level *value* therefore
//!   scales ×4, which is correct: so did the area.)
//! * **deposition** and **wear** are **metres** of height gained and lost — a
//!   height change, i.e. a per-area **intensity** (P19.1's own definition says to
//!   multiply by the cell area `l²` to get a volume). Doubling the area does not
//!   double a height. The value that survives a level is the **mean**, and it
//!   preserves the volume integral too: `mean(h) · 4A == Σ(h) · A`. **Summing
//!   these would 4×-inflate them every level**, so a mask thresholded at level 0
//!   would stop matching at level 1 — which is exactly backwards from the reason
//!   flow sums.
//!
//! (P19.1's own remainder note stated this split — "sum, not average, for flow;
//! mean for the metre channels" — and P19.3 briefly got it wrong in both the code
//! and the argument before the audit caught it. The dimension is the rule.)
//!
//! ## Why the splat weights still stay full-res
//!
//! Splat weights are an authoring-resolution paint layer: at the distance an
//! LOD-`n` tile is drawn, one weight sample is far below a screen texel, and
//! decimating an already-quantized `[u8; 4]` blend that must stay normalized to
//! ≈ 255 either drifts the normalization or needs a renormalizing filter — for
//! pixels nobody can resolve. Coarse tiles therefore carry the sparse
//! [`DEFAULT_WEIGHT`](crate::DEFAULT_WEIGHT) default (empty buffer); the renderer
//! samples the level-0 weight page for near rings, where splat detail is actually
//! visible.
//!
//! ## Sparsity survives
//!
//! A block whose members carry no maps (never eroded) or no ids (never painted)
//! produces a coarse tile on the **same sparse default** — the reduction is skipped
//! wholesale, so an un-eroded, unpainted terrain's pyramid costs exactly what it
//! did before P19.3, byte for byte. A reduction that happens to come out all-zero
//! is likewise dropped back to the default rather than materialized.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;

use crate::data::TerrainData;
use crate::tile::{
    DataMapKind, TerrainTile, DATA_MAP_CHANNELS, DEFAULT_DATA_MAP, UNASSIGNED_BIOME,
};

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
    reduce_layers(&mut out, res, coarse, tiles);
    out
}

/// The **categorical / integral** layers of a coarse tile (P19.3): biome ids by
/// majority vote, erosion data maps by sum, over each coarse sample's footprint.
/// See the module docs for the rules, the footprint, and the shared-edge ring.
///
/// Sparsity is preserved on both ends: a block whose members carry no ids / no
/// maps is skipped entirely (so an unpainted, un-eroded pyramid is byte-identical
/// to the pre-P19.3 one), and a reduction that comes out entirely default is
/// dropped back to the sparse default rather than materialized.
fn reduce_layers<T: std::borrow::Borrow<TerrainTile>>(
    out: &mut TerrainTile,
    res: u32,
    coarse: (i32, i32),
    tiles: &BTreeMap<(i32, i32), T>,
) {
    let members: Vec<&TerrainTile> = BLOCK_SCAN
        .iter()
        .filter_map(|&(a, b)| tiles.get(&(2 * coarse.0 + a, 2 * coarse.1 + b)))
        .map(|t| t.borrow())
        .collect();
    let any_maps = members.iter().any(|t| !t.maps_are_default());
    let any_biomes = members.iter().any(|t| !t.biomes_are_default());
    if !any_maps && !any_biomes {
        return;
    }

    let mut wrote_map = false;
    let mut wrote_biome = false;
    for j in 0..res {
        for i in 0..res {
            let (window, n) = footprint(res, i, j);
            let window = &window[..n];
            if any_maps {
                // One pass, two reductions: sum every channel, then divide the
                // **intensive** ones by the number of children that answered.
                // Fixed scan order ⇒ the `f32` sum is deterministic, and the
                // divisor is an exact small integer, so the mean is too.
                let mut acc = DEFAULT_DATA_MAP;
                let mut children = 0u32;
                for &(fi, fj) in window {
                    if let Some(texel) = combined_map(tiles, res, coarse, fi, fj) {
                        for c in 0..DATA_MAP_CHANNELS {
                            acc[c] += texel[c];
                        }
                        children += 1;
                    }
                }
                if children > 0 {
                    for kind in DataMapKind::ALL {
                        if !kind.is_extensive() {
                            acc[kind.channel()] /= children as f32;
                        }
                    }
                }
                if acc != DEFAULT_DATA_MAP {
                    out.set_map_texel(res, i, j, acc);
                    wrote_map = true;
                }
            }
            if any_biomes {
                let mut ids = [UNASSIGNED_BIOME; 4];
                let mut n_ids = 0usize;
                for &(fi, fj) in window {
                    if let Some(id) = combined_biome(tiles, res, coarse, fi, fj) {
                        ids[n_ids] = id;
                        n_ids += 1;
                    }
                }
                let best = majority(&ids[..n_ids]);
                if best != UNASSIGNED_BIOME {
                    out.set_biome_sample(res, i, j, best);
                    wrote_biome = true;
                }
            }
        }
    }
    // An all-default reduction must not cost per-sample bytes.
    if !wrote_map {
        out.clear_maps();
    }
    if !wrote_biome {
        out.clear_biomes();
    }
}

/// The winning biome id of a footprint: the **most frequent**, ties broken by the
/// **lowest id**. An empty footprint (every candidate fine tile absent) is
/// [`UNASSIGNED_BIOME`].
///
/// Id `0` votes like any other value — a coarse texel over mostly-unpainted
/// ground honestly reads *unassigned*, and a tie that includes `0` resolves to it
/// by the same lowest-id rule rather than by a special case.
fn majority(ids: &[u8]) -> u8 {
    let mut best = UNASSIGNED_BIOME;
    let mut best_count = 0usize;
    for &candidate in ids {
        let count = ids.iter().filter(|&&x| x == candidate).count();
        // Strictly-greater keeps the FIRST maximum; scanning candidates in
        // ascending order would too, but the footprint is not sorted, so the tie
        // is broken on the id explicitly.
        if count > best_count || (count == best_count && candidate < best) {
            best_count = count;
            best = candidate;
        }
    }
    best
}

/// The fine **combined** indices coarse sample `(i, j)` reduces: its 2 × 2
/// footprint `{2i, 2i+1} × {2j, 2j+1}`, degenerating to the single decimated
/// sample on the shared-edge ring (`i` or `j` ∈ `{0, res−1}`) so the seam stays
/// bit-identical between neighbouring coarse tiles. See the module docs.
///
/// Returns a fixed array plus its used length so the hot loop allocates nothing;
/// the scan order is fixed, which is what makes the `f32` sum deterministic.
fn footprint(res: u32, i: u32, j: u32) -> ([(u32, u32); 4], usize) {
    let edge = res - 1;
    let axis = |k: u32| -> ([u32; 2], usize) {
        if k == 0 || k == edge {
            ([2 * k, 2 * k], 1)
        } else {
            ([2 * k, 2 * k + 1], 2)
        }
    };
    let (xs, nx) = axis(i);
    let (zs, nz) = axis(j);
    let mut out = [(0, 0); 4];
    let mut n = 0;
    for &fj in zs.iter().take(nz) {
        for &fi in xs.iter().take(nx) {
            out[n] = (fi, fj);
            n += 1;
        }
    }
    (out, n)
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

/// One fine tile's value at combined index `(i, j)` of the 2 × 2 block, resolved
/// exactly like [`combined_sample`] (same sub-tile preference, same shared-edge
/// handling) but for a **layer** rather than the height. `None` when every
/// candidate fine tile is absent — the caller drops that sample from its
/// reduction rather than counting a zero it never measured.
fn combined_layer<T, V>(
    tiles: &BTreeMap<(i32, i32), T>,
    res: u32,
    (cx, cz): (i32, i32),
    i: u32,
    j: u32,
    read: impl Fn(&TerrainTile, u32, u32) -> V,
) -> Option<V>
where
    T: std::borrow::Borrow<TerrainTile>,
{
    for (a, li) in axis_candidates(res, i) {
        for (b, lj) in axis_candidates(res, j) {
            if let Some(t) = tiles.get(&(2 * cx + a, 2 * cz + b)) {
                return Some(read(t.borrow(), li, lj));
            }
        }
    }
    None
}

/// The erosion data-map texel at combined index `(i, j)` of the block.
fn combined_map<T: std::borrow::Borrow<TerrainTile>>(
    tiles: &BTreeMap<(i32, i32), T>,
    res: u32,
    coarse: (i32, i32),
    i: u32,
    j: u32,
) -> Option<[f32; DATA_MAP_CHANNELS]> {
    combined_layer(tiles, res, coarse, i, j, |t, li, lj| {
        t.map_texel(res, li, lj)
    })
}

/// The biome id at combined index `(i, j)` of the block.
fn combined_biome<T: std::borrow::Borrow<TerrainTile>>(
    tiles: &BTreeMap<(i32, i32), T>,
    res: u32,
    coarse: (i32, i32),
    i: u32,
    j: u32,
) -> Option<u8> {
    combined_layer(tiles, res, coarse, i, j, |t, li, lj| {
        t.biome_sample(res, li, lj)
    })
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

    // ── P19.3 per-layer reduction rules ─────────────────────────────────────

    /// The tie-break is stated, not incidental: **lowest id wins**, and the
    /// footprint's order must not change the answer.
    #[test]
    fn the_majority_vote_breaks_ties_on_the_lowest_id() {
        assert_eq!(
            majority(&[]),
            UNASSIGNED_BIOME,
            "an empty window is unassigned"
        );
        assert_eq!(majority(&[7]), 7);
        assert_eq!(majority(&[7, 7, 3, 1]), 7, "a plain majority wins");
        // 2 v 2 → the lower id.
        assert_eq!(majority(&[7, 7, 3, 3]), 3);
        assert_eq!(majority(&[3, 3, 7, 7]), 3, "…in either order");
        // Four distinct ids are a four-way tie → the lowest.
        assert_eq!(majority(&[9, 4, 2, 6]), 2);
        // Id 0 votes like any other value, including into a tie.
        assert_eq!(majority(&[0, 0, 5, 5]), UNASSIGNED_BIOME);
        assert_eq!(
            majority(&[0, 5, 5, 5]),
            5,
            "…but does not outvote a majority"
        );
    }

    /// The footprint is the coarse sample's 2 × 2 fine block, degenerating to the
    /// decimated sample on the shared-edge ring — the seam rule, in one place.
    #[test]
    fn the_footprint_degenerates_on_the_shared_edge_ring() {
        let res = 5;
        let win = |i, j| {
            let (w, n) = footprint(res, i, j);
            w[..n].to_vec()
        };
        assert_eq!(win(2, 2), vec![(4, 4), (5, 4), (4, 5), (5, 5)], "interior");
        assert_eq!(win(0, 0), vec![(0, 0)], "near corner decimates");
        assert_eq!(win(4, 4), vec![(8, 8)], "far corner decimates");
        assert_eq!(win(0, 2), vec![(0, 4), (0, 5)], "an edge column is 1 × 2");
        assert_eq!(win(2, 4), vec![(4, 8), (5, 8)], "…and an edge row is 2 × 1");
        // Every window contains the sample the heights decimate.
        for j in 0..res {
            for i in 0..res {
                assert!(win(i, j).contains(&(2 * i, 2 * j)), "({i},{j})");
            }
        }
    }

    /// **Biome ids majority-vote, and the vote is deterministic.** A block painted
    /// with a 3:1 mix per coarse sample reduces to the dominant id, and two
    /// reductions of the same input agree bit for bit.
    #[test]
    fn coarse_biome_ids_are_a_deterministic_majority_vote() {
        let res = 5;
        let mut t = poly_terrain(res, 1.0, 2);
        // Paint the whole 2×2 block: id 4 everywhere, then id 9 on the odd
        // combined columns of one tile so a couple of interior windows read 2:2.
        for coord in t.tiles().map(|(&c, _)| c).collect::<Vec<_>>() {
            let tile = t.get_tile_mut(coord).unwrap();
            for j in 0..res {
                for i in 0..res {
                    tile.set_biome_sample(res, i, j, 4);
                }
            }
        }
        // Fine tile (1,1)'s samples 1..=3 become id 9 → those coarse windows tie.
        {
            let tile = t.get_tile_mut((1, 1)).unwrap();
            for j in 1..4 {
                for i in 1..4 {
                    tile.set_biome_sample(res, i, j, 9);
                }
            }
        }
        let fine: BTreeMap<(i32, i32), &TerrainTile> = t.tiles().map(|(&c, x)| (c, x)).collect();
        let a = downsample_block(res, 1.0, (0, 0), &fine);
        let b = downsample_block(res, 1.0, (0, 0), &fine);
        assert_eq!(a, b, "the reduction is a pure function of its input");
        assert!(!a.biomes_are_default(), "a painted block must carry ids");
        // Every reduced id is one that was actually present.
        for j in 0..res {
            for i in 0..res {
                assert!(
                    matches!(a.biome_sample(res, i, j), 4 | 9),
                    "({i},{j}) invented an id"
                );
            }
        }
        // The near corner decimates: it is fine tile (0,0) sample (0,0) = id 4.
        assert_eq!(a.biome_sample(res, 0, 0), 4);
        // A window entirely inside the id-9 patch votes 9 (combined (5,5)+(6,6)
        // are all id 9 → sample (2,2) of the second sub-tile row).
        assert_eq!(a.biome_sample(res, 3, 3), 9, "the id-9 patch survives");
    }

    /// **THE COARSE-LOD HOLE REMAINDER, PINNED** (P21.2 audit).
    ///
    /// [`downsample_block`] reduces heights, biome ids and erosion data maps —
    /// and **not** the per-sample hole mask. A coarse pyramid page is therefore
    /// hole-free however thoroughly carved the level-0 block under it is, and the
    /// two consequences are shipped behaviour rather than theory:
    ///
    /// * a clipmap ring far enough out to draw a coarse page draws **ground over
    ///   a cave** — `terrain.wgsl` discards on the page's own mask and the page
    ///   has none, so a cave mouth closes up as the camera walks away from it;
    /// * `inf_render::RenderTerrain::seam_sample` reads the residency floor (the
    ///   coarsest level, so that lighting cannot depend on camera history), so on
    ///   a streamed terrain the seam blend cannot apply the poison rule either —
    ///   which is why `RenderTerrain::seam_holes_are_known` exists and why
    ///   `apply_seam` carries a mask-free veto for the case it answers `false`.
    ///
    /// This asserts the **current** behaviour, not the wanted one, and that is the
    /// whole point of writing it: a previous audit round claimed this was "pinned
    /// as a remainder" when nothing anywhere asserted it, and a cited gate that
    /// does not exist is worse than no claim. Propagating holes upward is a
    /// **decision**, not an oversight — a coarse sample covering four fine ones
    /// could be holed if any child is (a cave mouth grows by a whole coarse
    /// sample), if all are (it vanishes until it is 2ⁿ samples wide), or by
    /// majority like biome ids — and each answer draws a different distant
    /// silhouette. Whoever makes it deletes this test on the way past.
    #[test]
    fn a_coarse_page_carries_no_hole_mask() {
        let res = 5;
        let mut t = poly_terrain(res, 1.0, 2);
        // Carve the ENTIRE 2×2 block through: if anything were going to survive
        // the reduction, an all-holed input is where it would.
        for coord in t.tiles().map(|(&c, _)| c).collect::<Vec<_>>() {
            let tile = t.get_tile_mut(coord).unwrap();
            for j in 0..res {
                for i in 0..res {
                    tile.set_hole(res, i, j, true);
                }
            }
            assert!(tile.has_holes(), "the fixture must really be carved");
        }

        let fine: BTreeMap<(i32, i32), &TerrainTile> = t.tiles().map(|(&c, x)| (c, x)).collect();
        let coarse = downsample_block(res, 1.0, (0, 0), &fine);
        assert!(
            coarse.holes_are_default() && !coarse.has_holes(),
            "the reduction grew a hole mask — that is a deliberate change, and \
             this test is where it gets argued for"
        );
        for j in 0..res {
            for i in 0..res {
                assert!(!coarse.is_hole(res, i, j), "({i},{j})");
            }
        }
        // …and the heights DID reduce, so this is a statement about the mask and
        // not about a fixture that produced nothing.
        assert_eq!(
            coarse.sample(res, 0, 0),
            fine[&(0, 0)].sample(res, 0, 0),
            "the near corner must decimate — otherwise nothing was reduced at all"
        );
        // The whole pyramid, not just the kernel: every coarse level is hole-free.
        for level in build_pyramid(&t, PyramidOptions::default()) {
            for (coord, tile) in &level.tiles {
                assert!(
                    !tile.has_holes(),
                    "lod {} tile {coord:?} carries holes",
                    level.lod
                );
            }
        }
    }

    /// **The dimension is the rule: flow sums, the metre channels average.**
    ///
    /// The property that separates them is what a **uniform field** does across a
    /// level. `deposition` and `wear` are metres of height moved — an *intensive*
    /// per-area value — so a uniformly-1 m field must still read 1 m one level up;
    /// summing would report 4 m, and 16 m the level after, so a mask thresholded
    /// at level 0 would stop matching. `flow` is m³ of water shipped — *extensive*
    /// — so a uniformly-1 m³ field must read 4 m³ one level up, because the coarse
    /// cell covers four cells' worth of area.
    ///
    /// P19.3 shipped `sum` for all three for one commit; this test is what the
    /// rule looks like when it follows the dimension instead of the word
    /// "accumulator".
    #[test]
    fn data_maps_reduce_by_dimension_sum_flow_average_metres() {
        let res = 5;
        let mut t = poly_terrain(res, 1.0, 2);
        // A uniform field of exactly 1 in every channel makes each rule readable
        // straight off the reduced tile, with no arithmetic to mis-copy.
        for coord in t.tiles().map(|(&c, _)| c).collect::<Vec<_>>() {
            let tile = t.get_tile_mut(coord).unwrap();
            for j in 0..res {
                for i in 0..res {
                    tile.set_map_texel(res, i, j, [1.0, 1.0, 1.0]);
                }
            }
        }
        let fine: BTreeMap<(i32, i32), &TerrainTile> = t.tiles().map(|(&c, x)| (c, x)).collect();
        let c = downsample_block(res, 1.0, (0, 0), &fine);
        assert!(!c.maps_are_default());

        let dep = DataMapKind::Deposition.channel();
        let wear = DataMapKind::Wear.channel();
        let flow = DataMapKind::Flow.channel();

        // **A uniform metre field survives the level, everywhere** — interior
        // (4 children), edge (2), corner (1). That is the whole point of a mean.
        for &(i, j, label) in &[
            (2u32, 2u32, "interior (4 children)"),
            (0, 2, "edge column (2 children)"),
            (2, 0, "edge row (2 children)"),
            (0, 0, "near corner (1 child)"),
            (res - 1, res - 1, "far corner (1 child)"),
        ] {
            let texel = c.map_texel(res, i, j);
            assert_eq!(texel[dep], 1.0, "deposition moved at {label}");
            assert_eq!(texel[wear], 1.0, "wear moved at {label}");
        }

        // **Flow scales with the area it now stands for** — 4× interior, 2× on a
        // one-axis-degenerate edge, 1× on the decimated corner.
        assert_eq!(
            c.map_texel(res, 2, 2)[flow],
            4.0,
            "interior flow is the TOTAL"
        );
        assert_eq!(
            c.map_texel(res, 0, 2)[flow],
            2.0,
            "an edge covers two cells"
        );
        assert_eq!(c.map_texel(res, 0, 0)[flow], 1.0, "a corner decimates");

        // The two rules really are different — a test that passed under "sum
        // everything" would not distinguish them.
        assert_ne!(
            c.map_texel(res, 2, 2)[flow],
            c.map_texel(res, 2, 2)[dep],
            "the extensive and intensive channels must not reduce alike"
        );
        assert!(DataMapKind::Flow.is_extensive());
        assert!(!DataMapKind::Deposition.is_extensive());
        assert!(!DataMapKind::Wear.is_extensive());

        // Non-uniform values reduce by the same two rules.
        let mut varied = poly_terrain(res, 1.0, 2);
        let cells = (res - 1) as i32;
        for coord in varied.tiles().map(|(&c, _)| c).collect::<Vec<_>>() {
            let tile = varied.get_tile_mut(coord).unwrap();
            for j in 0..res {
                for i in 0..res {
                    // Keyed on the GLOBAL sample index, which for block (0,0) is
                    // exactly the combined index — so the expected children below
                    // can be read straight off the window.
                    let g = (coord.0 * cells + i as i32) + (coord.1 * cells + j as i32);
                    tile.set_map_texel(res, i, j, [g as f32, g as f32, 0.0]);
                }
            }
        }
        let vf: BTreeMap<(i32, i32), &TerrainTile> = varied.tiles().map(|(&c, x)| (c, x)).collect();
        let v = downsample_block(res, 1.0, (0, 0), &vf);
        // Interior sample (2,2) ⇒ combined children (4,4),(5,4),(4,5),(5,5), whose
        // `gx + gz` values are 8, 9, 9, 10.
        let t22 = v.map_texel(res, 2, 2);
        assert_eq!(t22[flow], 36.0, "sum of 8+9+9+10");
        assert_eq!(t22[dep], 9.0, "…and their mean");

        // Pure, in both channels.
        assert_eq!(c, downsample_block(res, 1.0, (0, 0), &fine), "pure");
    }

    /// **Flow's volume integral survives the level; the metre channels' area
    /// integral does.** The two conservation statements are different because the
    /// dimensions are, and this is the pair a consumer actually relies on.
    #[test]
    fn each_reduction_conserves_the_integral_its_dimension_names() {
        let res = 5;
        let mut t = poly_terrain(res, 1.0, 2);
        for coord in t.tiles().map(|(&c, _)| c).collect::<Vec<_>>() {
            let tile = t.get_tile_mut(coord).unwrap();
            for j in 0..res {
                for i in 0..res {
                    tile.set_map_texel(res, i, j, [1.0, 3.0, 0.0]);
                }
            }
        }
        let fine: BTreeMap<(i32, i32), &TerrainTile> = t.tiles().map(|(&c, x)| (c, x)).collect();
        let c = downsample_block(res, 1.0, (0, 0), &fine);

        // Interior only, so the ring's degenerate windows do not muddy the sum.
        let flow = DataMapKind::Flow.channel();
        let dep = DataMapKind::Deposition.channel();
        let mut coarse_flow = 0.0f64;
        let mut coarse_dep_volume = 0.0f64;
        // A coarse cell has 4× the area of a fine one.
        const COARSE_AREA: f64 = 4.0;
        for j in 1..res - 1 {
            for i in 1..res - 1 {
                coarse_flow += c.map_texel(res, i, j)[flow] as f64;
                coarse_dep_volume += c.map_texel(res, i, j)[dep] as f64 * COARSE_AREA;
            }
        }
        // Those (res−2)² interior coarse samples stand for 4 fine samples each.
        let fine_count = ((res - 2) * (res - 2) * 4) as f64;
        assert_eq!(
            coarse_flow, fine_count,
            "flow's VOLUME total must survive the level"
        );
        assert_eq!(
            coarse_dep_volume,
            3.0 * fine_count,
            "deposition's height × area (its volume) must survive the level"
        );
    }

    /// **Shared-edge continuity for the categorical and integral layers.** The
    /// heights' seam guarantee now extends to the ids and the maps: a coarse
    /// tile's far edge is bit-identical to its `+X` / `+Z` neighbour's near edge,
    /// which is exactly what the ring's degenerate window buys.
    #[test]
    fn coarse_layers_keep_the_shared_edge() {
        let res = 5;
        let mut t = poly_terrain(res, 1.0, 8);
        // A world-position-derived id + map so neighbouring tiles genuinely agree
        // on their shared samples (the level-0 seam invariant) and the reduction
        // has something to disagree about.
        let coords: Vec<(i32, i32)> = t.tiles().map(|(&c, _)| c).collect();
        for coord in coords {
            let span = (res - 1) as i32;
            let tile = t.get_tile_mut(coord).unwrap();
            for j in 0..res {
                for i in 0..res {
                    let gx = coord.0 * span + i as i32;
                    let gz = coord.1 * span + j as i32;
                    let id = 1 + (gx.rem_euclid(3) + 3 * gz.rem_euclid(2)) as u8;
                    tile.set_biome_sample(res, i, j, id);
                    tile.set_map_texel(res, i, j, [gx as f32, gz as f32, 1.0]);
                }
            }
        }
        for level in build_pyramid(&t, PyramidOptions::default()) {
            for (&(cx, cz), tile) in &level.tiles {
                if let Some(right) = level.tiles.get(&(cx + 1, cz)) {
                    for j in 0..res {
                        assert_eq!(
                            tile.biome_sample(res, res - 1, j),
                            right.biome_sample(res, 0, j),
                            "lod {} X-seam biome at ({cx},{cz}) row {j}",
                            level.lod
                        );
                        assert_eq!(
                            tile.map_texel(res, res - 1, j),
                            right.map_texel(res, 0, j),
                            "lod {} X-seam maps at ({cx},{cz}) row {j}",
                            level.lod
                        );
                    }
                }
                if let Some(down) = level.tiles.get(&(cx, cz + 1)) {
                    for i in 0..res {
                        assert_eq!(
                            tile.biome_sample(res, i, res - 1),
                            down.biome_sample(res, i, 0),
                            "lod {} Z-seam biome at ({cx},{cz}) col {i}",
                            level.lod
                        );
                        assert_eq!(
                            tile.map_texel(res, i, res - 1),
                            down.map_texel(res, i, 0),
                            "lod {} Z-seam maps at ({cx},{cz}) col {i}",
                            level.lod
                        );
                    }
                }
            }
        }
    }

    /// **The seam guarantee is conditional on coverage, and that is inherited.**
    ///
    /// A block member that simply is not there contributes nothing to its window
    /// (`combined_layer` answers `None` and the sample is dropped, rather than
    /// counting a zero nobody measured). At a seam where the two coarse tiles have
    /// *different* coverage, the shared sample can therefore disagree — the left
    /// tile's far edge needs fine tile `2cx+1`, the right tile's near edge needs
    /// `2cx+2`, and if only one exists only one side can answer.
    ///
    /// This is exactly the heights' own behaviour (`coverage_holes_fill_flat…`):
    /// a coverage hole reads as the default on the side that cannot reach it. It
    /// is asserted here rather than left implicit, because "the seam is
    /// bit-identical" is a claim this crate now makes for four layers and it is
    /// worth knowing precisely where it stops.
    #[test]
    fn a_coverage_hole_makes_the_seam_default_on_the_side_that_cannot_reach_it() {
        let res = 5;
        let mut t = TerrainData::new(res, 1.0);
        // Fine tiles 0 and 2 exist; 1 and 3 do not. Coarse block 0 covers fine
        // {0,1}, coarse block 1 covers fine {2,3}.
        // (Two tiles deep in Z so the hole is purely an X-axis coverage gap.)
        for tx in [0, 2] {
            for tz in 0..2 {
                t.author_tile((tx, tz), height_fn);
                let tile = t.get_tile_mut((tx, tz)).unwrap();
                for j in 0..res {
                    for i in 0..res {
                        tile.set_biome_sample(res, i, j, 6);
                        tile.set_map_texel(res, i, j, [2.0, 4.0, 0.0]);
                    }
                }
            }
        }
        let fine: BTreeMap<(i32, i32), &TerrainTile> = t.tiles().map(|(&c, x)| (c, x)).collect();
        let left = downsample_block(res, 1.0, (0, 0), &fine);
        let right = downsample_block(res, 1.0, (1, 0), &fine);

        // The right block's near edge reaches fine tile 2 and answers.
        assert_eq!(right.biome_sample(res, 0, 2), 6);
        assert_eq!(right.map_texel(res, 0, 2)[0], 4.0, "two children, summed");
        // The left block's far edge needs fine tile 1, which does not exist, so
        // it reads the sparse default — the documented, inherited behaviour.
        assert_eq!(
            left.biome_sample(res, res - 1, 2),
            UNASSIGNED_BIOME,
            "an unreachable seam sample must read the DEFAULT, not a guess"
        );
        assert_eq!(left.map_texel(res, res - 1, 2), DEFAULT_DATA_MAP);
        // …and it is still deterministic, which is what actually matters.
        assert_eq!(left, downsample_block(res, 1.0, (0, 0), &fine));

        // With full coverage the same seam agrees exactly (the positive control).
        let mut full = t.clone();
        for tz in 0..2 {
            full.author_tile((1, tz), height_fn);
            let tile = full.get_tile_mut((1, tz)).unwrap();
            for j in 0..res {
                for i in 0..res {
                    tile.set_biome_sample(res, i, j, 6);
                    tile.set_map_texel(res, i, j, [2.0, 4.0, 0.0]);
                }
            }
        }
        let ff: BTreeMap<(i32, i32), &TerrainTile> = full.tiles().map(|(&c, x)| (c, x)).collect();
        let l2 = downsample_block(res, 1.0, (0, 0), &ff);
        let r2 = downsample_block(res, 1.0, (1, 0), &ff);
        for j in 0..res {
            assert_eq!(l2.biome_sample(res, res - 1, j), r2.biome_sample(res, 0, j));
            assert_eq!(l2.map_texel(res, res - 1, j), r2.map_texel(res, 0, j));
        }
    }

    /// The reduction rules must not cost an un-eroded, unpainted terrain a single
    /// byte — the sparse defaults survive the pyramid, exactly as before P19.3.
    #[test]
    fn a_clean_terrain_still_pyramids_to_sparse_layers() {
        let t = poly_terrain(5, 1.0, 8);
        for level in build_pyramid(&t, PyramidOptions::default()) {
            for tile in level.tiles.values() {
                assert!(tile.maps_are_default(), "lod {} leaked maps", level.lod);
                assert!(tile.biomes_are_default(), "lod {} leaked ids", level.lod);
            }
        }
        // …and a block whose members carry buffers that happen to be all-default
        // is dropped back to sparse rather than materialized.
        let mut zeroed = poly_terrain(5, 1.0, 4);
        for coord in zeroed.tiles().map(|(&c, _)| c).collect::<Vec<_>>() {
            let tile = zeroed.get_tile_mut(coord).unwrap();
            tile.ensure_maps(5);
            tile.ensure_biomes(5);
        }
        for level in build_pyramid(&zeroed, PyramidOptions::default()) {
            for tile in level.tiles.values() {
                assert!(tile.maps_are_default());
                assert!(tile.biomes_are_default());
            }
        }
    }

    /// The write-back's four-member block still reduces **bit-identically** to a
    /// whole-level reduction with the layers live — the invariant that lets a
    /// partial pyramid rebuild skip the rest of the terrain.
    #[test]
    fn one_block_with_layers_decimates_identically_to_a_whole_level() {
        let res = 5;
        let mut t = poly_terrain(res, 2.0, 4);
        let coords: Vec<(i32, i32)> = t.tiles().map(|(&c, _)| c).collect();
        for coord in coords {
            let tile = t.get_tile_mut(coord).unwrap();
            for j in 0..res {
                for i in 0..res {
                    tile.set_biome_sample(
                        res,
                        i,
                        j,
                        1 + ((i + j + coord.0.unsigned_abs()) % 3) as u8,
                    );
                    tile.set_map_texel(res, i, j, [(i + j) as f32, 1.0, 0.25]);
                }
            }
        }
        let fine: BTreeMap<(i32, i32), &TerrainTile> = t.tiles().map(|(&c, ti)| (c, ti)).collect();
        let whole = downsample_tiles(res, 2.0, &fine);
        for (&coarse, expect) in &whole {
            let block: BTreeMap<(i32, i32), TerrainTile> = BLOCK_SCAN
                .iter()
                .filter_map(|&(a, b)| {
                    let c = (2 * coarse.0 + a, 2 * coarse.1 + b);
                    fine.get(&c).map(|ti| (c, (*ti).clone()))
                })
                .collect();
            assert_eq!(
                &downsample_block(res, 2.0, coarse, &block),
                expect,
                "block {coarse:?} differs from the whole-level reduction"
            );
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
