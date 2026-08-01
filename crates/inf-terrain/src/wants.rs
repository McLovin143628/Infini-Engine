//! **Wants**: the pure, deterministic functions that decide *which* tiles a
//! consumer needs resident (P16.3b2).
//!
//! [`residency`](crate::residency) deliberately owns no policy — it executes an
//! explicit set of wanted keys. This module is the policy, kept equally pure: a
//! camera position (or a list of sim positions) plus a tile index in, a
//! [`BTreeSet<TileKey>`] out. No I/O, no clock, no job pool, no interior
//! mutability — so every rule below is unit-testable without a GPU, a pack, or a
//! frame loop, and two runs over the same inputs produce the identical set.
//!
//! # THE DETERMINISM DOCTRINE (non-negotiable)
//!
//! A fixed-step simulation's results must **never** depend on camera-driven
//! residency, or the replay / PIE-==-shipping gates die. There are therefore two
//! residency drivers and they are kept strictly apart:
//!
//! * **Sim wants** ([`sim_wants`]) — deterministic. Level-0 tiles required by sim
//!   *consumers* (terrain height queries through the `terrain.height_at` host
//!   seam, physics heightfield colliders), computed from **sim state alone**
//!   (entity positions + a fixed margin), loaded **synchronously at the
//!   fixed-step boundary before the step runs**, in sorted order. A sim height
//!   query must always see the same tile bytes regardless of camera, frame rate,
//!   or load timing.
//! * **Render wants** ([`render_wants`]) — free. The camera-driven quadtree cut
//!   used for *rendering*, loaded on the job pool and applied at a deterministic
//!   point in the frame, and **never observable by the sim**.
//!
//! The split is enforced *structurally*, not by convention: the two want sets
//! feed two **different [`TerrainData`](crate::TerrainData) instances**. The
//! sim's lives in the ECS `Terrain` component (the only thing `height_at` can
//! reach); the render-resident set lives in the streamer
//! ([`crate::stream::TerrainStreamer`]) outside the world entirely. The sim
//! cannot see a camera-paged tile because there is no reference from the world to
//! the set that holds it. The (bounded) duplication of tiles wanted by both is
//! the price, and it is paid on purpose.
//!
//! # The quadtree cut (the B2 contract)
//!
//! `inf_render::passes::terrain` is hole-free under *any* residency set but only
//! *artefact*-free under a **quadtree cut**: for every patch of ground exactly one
//! resident page covers it — a coarse page is resident precisely where its four
//! children are not. [`render_wants`] produces exactly that by construction: it
//! walks the pyramid top-down and either **keeps** a node or **replaces it by all
//! of its children that exist**. A parent and its children are therefore never
//! both in the output, and children enter and leave as complete sets.
//!
//! [`advance_cut`] preserves the same invariant while *converging* toward a new
//! target over several frames: every intermediate state is itself a valid cut
//! (one refinement/coarsening step is applied atomically per node), so a moving
//! camera never publishes a "three of four children beside their parent" set even
//! transiently, and the per-frame load batch stays bounded.
//!
//! # Hysteresis
//!
//! A camera hovering exactly on a refinement threshold would otherwise page a
//! node's children in and out every frame. The refine test is therefore
//! **asymmetric**: a node that was coarse last frame refines only inside
//! `r · (1 − h)`, while a node that was already refined stays refined out to
//! `r · (1 + h)`. The dead band `[r(1−h), r(1+h)]` costs one extra level of
//! detail at its outer edge and buys a thrash-free hover.

use std::collections::BTreeSet;

use glam::DVec2;

use crate::residency::{tile_range, TileStore};
use crate::tile::TileKey;

// ── grid geometry ───────────────────────────────────────────────────────────

/// The tile grid's world-space geometry: everything the want rules need to turn a
/// [`TileKey`] into a world-space box.
///
/// Level-`n` tile `(TX, TZ)` spans
/// `[TX·Sₙ, (TX+1)·Sₙ] × [TZ·Sₙ, (TZ+1)·Sₙ]` where `Sₙ = S₀ · 2ⁿ` and
/// `S₀ = (resolution − 1) · meters_per_sample` — the same convention
/// [`TerrainData::tile_span`](crate::TerrainData::tile_span) and the renderer's
/// [`RenderTerrain::tile_span_at`] use, so the three agree by construction.
///
/// [`RenderTerrain::tile_span_at`]: https://docs.rs/inf-render
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileGrid {
    level0_span: f64,
}

impl TileGrid {
    /// From a terrain's level-0 configuration.
    pub fn new(tile_resolution: u32, meters_per_sample: f64) -> Self {
        Self::from_span((tile_resolution.max(2) as f64 - 1.0) * meters_per_sample)
    }

    /// From an already-computed level-0 tile span (clamped to a positive length —
    /// a zero span would divide by zero in every coordinate mapping).
    pub fn from_span(level0_span: f64) -> Self {
        Self {
            level0_span: if level0_span.is_finite() && level0_span > 0.0 {
                level0_span
            } else {
                1.0
            },
        }
    }

    /// World edge length of a level-0 tile.
    #[inline]
    pub fn level0_span(&self) -> f64 {
        self.level0_span
    }

    /// World edge length of a tile at `lod` (`S₀ · 2^lod`).
    #[inline]
    pub fn span_at(&self, lod: u32) -> f64 {
        self.level0_span * (1u64 << lod.min(62)) as f64
    }

    /// The `(min, max)` world-XZ corners of `key`'s footprint.
    pub fn bounds(&self, key: TileKey) -> (DVec2, DVec2) {
        let s = self.span_at(key.lod);
        let lo = DVec2::new(key.coord.0 as f64 * s, key.coord.1 as f64 * s);
        (lo, lo + DVec2::splat(s))
    }

    /// The `lod`-level grid coordinate containing world XZ `p` (floored, so
    /// negative coordinates land on the tile *below* rather than toward zero).
    pub fn coord_at(&self, lod: u32, p: DVec2) -> (i32, i32) {
        let s = self.span_at(lod);
        ((p.x / s).floor() as i32, (p.y / s).floor() as i32)
    }

    /// Shortest world-XZ distance from `p` to `key`'s footprint; `0` inside it.
    pub fn distance_to(&self, key: TileKey, p: DVec2) -> f64 {
        let (lo, hi) = self.bounds(key);
        let d = (lo - p).max(p - hi).max(DVec2::ZERO);
        d.length()
    }
}

// ── the availability index ──────────────────────────────────────────────────

/// What a want rule may ask about the *available* tiles.
///
/// Deliberately tiny and read-only: the rules must never be able to load, evict
/// or mutate anything, which is what keeps them pure. [`TileCatalog`] is the
/// concrete implementation built once from a [`TileStore`]; tests implement it
/// directly over a hand-written key set.
pub trait TileIndex {
    /// Whether `key` exists in the source.
    fn has(&self, key: TileKey) -> bool;

    /// The coarsest LOD level present (`0` for a level-0-only source).
    fn max_lod(&self) -> u32;

    /// Every key at exactly `lod`, ascending. Used to seed the top-down walk.
    fn keys_at_lod(&self, lod: u32) -> Vec<TileKey>;
}

/// The keys a tile source can serve, indexed for the want rules.
///
/// Built once (from a `.inf_terrain` directory, a memory store, or a literal key
/// list) and then queried per frame — the directory is never re-parsed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TileCatalog {
    keys: BTreeSet<TileKey>,
    max_lod: u32,
}

impl TileCatalog {
    /// From an explicit key set.
    pub fn from_keys(keys: impl IntoIterator<Item = TileKey>) -> Self {
        let keys: BTreeSet<TileKey> = keys.into_iter().collect();
        let max_lod = keys.iter().map(|k| k.lod).max().unwrap_or(0);
        Self { keys, max_lod }
    }

    /// From everything a store can serve.
    pub fn from_store(store: &dyn TileStore) -> Self {
        Self::from_keys(store.tile_keys())
    }

    /// Number of indexed keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The indexed keys, ascending.
    pub fn keys(&self) -> impl Iterator<Item = TileKey> + '_ {
        self.keys.iter().copied()
    }

    /// How many keys sit at exactly `lod`.
    pub fn count_at_lod(&self, lod: u32) -> usize {
        self.range_at(lod).count()
    }

    fn range_at(&self, lod: u32) -> impl Iterator<Item = &TileKey> {
        self.keys.range(
            TileKey::new(lod, (i32::MIN, i32::MIN))..=TileKey::new(lod, (i32::MAX, i32::MAX)),
        )
    }
}

impl TileIndex for TileCatalog {
    fn has(&self, key: TileKey) -> bool {
        self.keys.contains(&key)
    }

    fn max_lod(&self) -> u32 {
        self.max_lod
    }

    fn keys_at_lod(&self, lod: u32) -> Vec<TileKey> {
        self.range_at(lod).copied().collect()
    }
}

// ── render wants: the camera-driven quadtree cut ────────────────────────────

/// Tuning for [`render_wants`].
#[derive(Debug, Clone, PartialEq)]
pub struct RenderWantsParams {
    /// `radii_per_lod[L]` is the world-space distance within which a **lod-`L`
    /// node is refined into its four children**. Beyond the slice the last entry
    /// is extrapolated by doubling (matching the node-size doubling per level).
    /// Index 0 is unused (a level-0 tile has nothing finer).
    pub radii_per_lod: Vec<f64>,
    /// Refinement dead band as a fraction of the radius (see the module docs).
    /// `0` disables hysteresis; values are clamped into `[0, 0.9]`.
    pub hysteresis: f64,
    /// Drop a node entirely beyond this distance (`0` = never). A *whole* node is
    /// dropped, never part of one, so the cut property survives — the result is a
    /// legal hole, exactly like unauthored ground.
    pub max_distance: f64,
}

impl RenderWantsParams {
    /// A geometric ladder: a lod-`L` node refines within `lod0_radius · 2^L`.
    ///
    /// This is the ladder the renderer's clipmap already assumes — ring `r` reads
    /// asset LOD `r`, and both double per step — so the streamed cut and the
    /// drawn rings stay in step.
    pub fn geometric(lod0_radius: f64, levels: u32) -> Self {
        Self {
            radii_per_lod: (0..levels.max(1))
                .map(|l| lod0_radius * (1u64 << l.min(62)) as f64)
                .collect(),
            hysteresis: DEFAULT_HYSTERESIS,
            max_distance: 0.0,
        }
    }

    /// The refine radius for a lod-`L` node.
    pub fn refine_radius(&self, lod: u32) -> f64 {
        if self.radii_per_lod.is_empty() {
            return 0.0;
        }
        let n = self.radii_per_lod.len();
        if (lod as usize) < n {
            return self.radii_per_lod[lod as usize];
        }
        // Extrapolate by doubling from the last entry.
        let extra = lod as usize - (n - 1);
        self.radii_per_lod[n - 1] * (1u64 << extra.min(62)) as f64
    }

    fn clamped_hysteresis(&self) -> f64 {
        self.hysteresis.clamp(0.0, 0.9)
    }
}

/// Default refinement dead band (±15 %).
pub const DEFAULT_HYSTERESIS: f64 = 0.15;

impl Default for RenderWantsParams {
    /// Four levels of a geometric ladder anchored at three level-0 tile spans —
    /// but a caller normally builds this from its own grid via
    /// [`geometric`](Self::geometric).
    fn default() -> Self {
        Self::geometric(3.0, 4)
    }
}

/// Every **strict** ancestor of every key in `set`, up to `max_lod` — i.e. the
/// nodes that `set` has *refined*.
///
/// A cut contains a node's descendants exactly when the node was refined, so this
/// turns "was this node refined last frame?" into one `BTreeSet` lookup instead
/// of a subtree walk.
fn refined_nodes(set: &BTreeSet<TileKey>, max_lod: u32) -> BTreeSet<TileKey> {
    let mut out = BTreeSet::new();
    for &k in set {
        let mut cur = k;
        while cur.lod < max_lod {
            cur = cur.parent();
            if !out.insert(cur) {
                break; // this chain was already walked
            }
        }
    }
    out
}

/// The camera-driven **quadtree cut**: which pages the renderer wants resident.
///
/// Walks the pyramid from its coarsest level down, keeping a node or replacing it
/// by *all* of its children that exist in `available`. The result is a cut by
/// construction (no parent coexists with a child), which is exactly the B2
/// contract the terrain pass documents.
///
/// `prev_cut` is the previously published cut and is used **only** for hysteresis
/// (see the module docs); pass an empty set for a stateless evaluation.
///
/// Pure: the output is a function of `(camera_xz, grid, params, available,
/// prev_cut)` alone, and is returned in a `BTreeSet` so iteration order is
/// deterministic.
pub fn render_wants(
    camera_xz: DVec2,
    grid: TileGrid,
    params: &RenderWantsParams,
    available: &dyn TileIndex,
    prev_cut: &BTreeSet<TileKey>,
) -> BTreeSet<TileKey> {
    let root_lod = available.max_lod();
    let refined_prev = refined_nodes(prev_cut, root_lod);
    let h = params.clamped_hysteresis();

    let mut out = BTreeSet::new();
    let mut stack = available.keys_at_lod(root_lod);
    while let Some(node) = stack.pop() {
        let dist = grid.distance_to(node, camera_xz);
        if params.max_distance > 0.0 && dist > params.max_distance {
            continue; // a whole node dropped — a legal hole, never a partial one
        }
        if node.lod == 0 {
            out.insert(node);
            continue;
        }
        let kids: Vec<TileKey> = node
            .children()
            .into_iter()
            .filter(|c| available.has(*c))
            .collect();
        if kids.is_empty() {
            out.insert(node); // nothing finer exists — this page is the detail
            continue;
        }
        let r = params.refine_radius(node.lod);
        let refine = if refined_prev.contains(&node) {
            dist <= r * (1.0 + h) // already refined: stay refined out to r(1+h)
        } else {
            dist < r * (1.0 - h) // was coarse: come inside r(1−h) to refine
        };
        if refine {
            stack.extend(kids);
        } else {
            out.insert(node);
        }
    }
    out
}

// ── sim wants: deterministic, level-0 only ──────────────────────────────────

/// The **level-0** tiles the fixed-step simulation needs resident: everything
/// within `margin_m` of any `position`.
///
/// Deterministic by construction — a function of sim state and one constant, with
/// no camera, no frame rate and no load history anywhere in it. The caller loads
/// these **synchronously at the fixed-step boundary, before the step runs**, so a
/// height query inside the step can never observe a partially-paged terrain.
///
/// # `positions` must be terrain OBSERVERS, not every entity
///
/// The result's size is `O(positions × margin²)` and — by doctrine — is **never
/// clamped to a budget**: a missing render page costs detail, a missing sim page
/// changes the simulation. That makes the caller's choice of `positions` the only
/// thing bounding sim residency, so it must pass the entities that can actually
/// *read* the terrain in a step (Blueprint actors, character movers, physics
/// bodies) and **not** every entity that happens to have a transform. Feeding it
/// a level's worth of static decorative props asks for most of the level-0 grid,
/// and there is no downstream ceiling to catch it. See
/// `inf_player::terrain_stream::observes_terrain` for the shipped predicate, and
/// its `SIM_WANT_SOFT_CEILING` for the advisory diagnostic that fires when the
/// result grows anyway.
///
/// `margin_m` must be at least the terrain's `meters_per_sample`:
/// [`normal_at`](crate::TerrainData::normal_at) central-differences one sample
/// either side of the query point, and
/// [`height_at`](crate::TerrainData::height_at) can resolve a point exactly on a
/// shared edge onto the *previous* tile. A margin below one sample spacing would
/// let those neighbour lookups fall off the resident set and silently change a
/// slope into a one-sided one.
pub fn sim_wants(grid: TileGrid, positions: &[DVec2], margin_m: f64) -> BTreeSet<TileKey> {
    let m = margin_m.max(0.0);
    let mut out = BTreeSet::new();
    for &p in positions {
        let lo = grid.coord_at(0, p - DVec2::splat(m));
        let hi = grid.coord_at(0, p + DVec2::splat(m));
        out.extend(tile_range(0, lo, hi));
    }
    out
}

/// The **level-0** tiles one brush dab needs resident before it may run: the
/// brush disk's footprint plus a **one-tile margin** (P16.4b).
///
/// Literally [`sim_wants`] with `margin_m = radius + one level-0 tile span`, and
/// deliberately so — an edit is the editor's fixed-step boundary. A dab must see
/// the same bytes whatever the camera happens to have paged in, so it names its
/// footprint the same deterministic, camera-free way the simulation names its
/// own, and the caller loads the result **synchronously before applying**.
///
/// # Why a whole tile of margin and not one sample
///
/// The disk's own AABB is not enough for the neighbourhood ops. `Smooth` extracts
/// a [`HeightRegion`](crate::HeightRegion) with a one-ring margin and reads a
/// 3 × 3 neighbourhood per sample; `normal_at` central-differences one sample
/// either side; and `height_at` resolves a point exactly on a shared edge onto
/// the *previous* tile. Any of those can reach a sample outside the footprint,
/// and a non-resident neighbour reads as **unauthored ground**, which silently
/// turns a slope into a one-sided one or excludes a real neighbour from a mean.
/// A tile is the granularity residency actually has, so one tile is the smallest
/// honest margin — and it is what the sim uses for the same reason.
pub fn brush_wants(grid: TileGrid, center: DVec2, radius: f64) -> BTreeSet<TileKey> {
    sim_wants(grid, &[center], radius.max(0.0) + grid.level0_span())
}

// ── budget clamp ────────────────────────────────────────────────────────────

/// Bring a cut under a tile budget by **collapsing its finest groups into their
/// parents, farthest from the camera first** — "coarsest-first keep".
///
/// The result is still a cut (a complete sibling group is replaced by exactly one
/// parent), so the renderer's contract survives the clamp; detail is what is
/// spent, never coverage. A group whose parent is not in `available`, or that has
/// only one existing sibling (collapsing it would not shrink the set), is skipped —
/// and when a whole level offers no collapsible group the search moves **up** a
/// level rather than giving up, so coverage is never dropped while detail is still
/// available to spend.
///
/// Only when no level anywhere can fold are the farthest tiles dropped outright,
/// as a last resort — a legal hole, and the only alternative to blowing the budget.
///
/// **Sim wants are never clamped.** Correctness first: a missing render page costs
/// detail, a missing sim page changes the simulation.
pub fn clamp_cut(
    cut: &BTreeSet<TileKey>,
    grid: TileGrid,
    camera_xz: DVec2,
    available: &dyn TileIndex,
    max_tiles: usize,
) -> BTreeSet<TileKey> {
    if max_tiles == 0 || cut.len() <= max_tiles {
        return cut.clone();
    }
    let mut out = cut.clone();
    // Collapse complete sibling groups, finest level first, farthest first.
    loop {
        if out.len() <= max_tiles {
            return out;
        }
        // Walk levels from the finest present **upward** looking for a collapsible
        // group, instead of giving up when the finest level has none. A level can
        // easily be all singletons — a sparse terrain edge, or a cut that already
        // collapsed everything collapsible there — while a coarser level still has
        // a full sibling set to fold. Stopping at the finest level would fall
        // through to the drop fallback and spend *coverage* while detail was still
        // available to spend, which inverts the whole policy.
        let levels: BTreeSet<u32> = out.iter().map(|k| k.lod).collect();
        let mut collapsed = false;
        for lod in levels {
            let mut groups: std::collections::BTreeMap<TileKey, Vec<TileKey>> = Default::default();
            for &k in out.iter().filter(|k| k.lod == lod) {
                groups.entry(k.parent()).or_default().push(k);
            }
            // A one-child group would not shrink the set (1 → 1), so it is not a
            // collapse candidate; the parent must exist to be paged in at all.
            //
            // The third clause is the **cut-contract guard**. Because this walk can
            // fold at a level that is not the finest one present, a sibling group at
            // `lod` can have a fourth sibling that is not here — it was refined, and
            // its own descendants sit further down in `out`. Folding the group into
            // `parent` would then publish `parent` *above* those descendants, i.e. a
            // page coexisting with its own children: exactly the shape
            // `render_wants` and `advance_cut` are built never to emit, and the one
            // the terrain pass's `superseded` leans on. No in-repo call site can
            // reach it today (every cut this module produces is refined uniformly),
            // but `clamp_cut` is `pub` and takes an arbitrary cut, so the invariant
            // is enforced rather than assumed.
            let best = groups
                .into_iter()
                .filter(|(parent, kids)| {
                    kids.len() > 1
                        && available.has(*parent)
                        && !out
                            .iter()
                            .any(|k| k.lod < lod && is_descendant_of(*k, *parent))
                })
                .max_by(|(a, _), (b, _)| {
                    grid.distance_to(*a, camera_xz)
                        .total_cmp(&grid.distance_to(*b, camera_xz))
                        .then_with(|| a.cmp(b))
                });
            if let Some((parent, kids)) = best {
                for k in kids {
                    out.remove(&k);
                }
                out.insert(parent);
                collapsed = true;
                break;
            }
        }
        if !collapsed {
            break; // nothing anywhere can fold — the drop fallback is all that is left
        }
    }
    // Last resort: drop the farthest tiles until we fit.
    let mut ordered: Vec<TileKey> = out.iter().copied().collect();
    ordered.sort_by(|a, b| {
        grid.distance_to(*b, camera_xz)
            .total_cmp(&grid.distance_to(*a, camera_xz))
            .then_with(|| a.cmp(b))
    });
    for k in ordered {
        if out.len() <= max_tiles {
            break;
        }
        out.remove(&k);
    }
    out
}

// ── incremental convergence ─────────────────────────────────────────────────

/// One bounded step of `current` toward `target`, **preserving the cut property
/// at every intermediate state**.
///
/// Each node of `current` is handled atomically: kept, refined into all of its
/// existing children, or (with its whole sibling group) coarsened into its parent.
/// Because every action replaces a complete set, the output is a valid cut even
/// when the budget stops the walk half-way — so the renderer never sees three of
/// four children beside their parent, not even for one frame.
///
/// `max_new_tiles` bounds how many **not-yet-resident** tiles the step may
/// introduce (`0` = unlimited); `resident` reports which keys are already in
/// memory, so re-entering a page that is still resident is free. Nodes are
/// considered nearest-camera-first, so the detail the player is looking at
/// converges before the horizon does.
///
/// This is what replaces timing-driven asynchrony: the batch is bounded (no
/// whole-cut hitch) yet its membership is a pure function of the inputs, so a
/// scripted camera path reproduces the identical resident-set trace run to run —
/// which a "whatever finished loading this frame" scheme could never do.
pub fn advance_cut(
    current: &BTreeSet<TileKey>,
    target: &BTreeSet<TileKey>,
    grid: TileGrid,
    camera_xz: DVec2,
    available: &dyn TileIndex,
    resident: &dyn Fn(TileKey) -> bool,
    max_new_tiles: usize,
) -> BTreeSet<TileKey> {
    if current == target {
        return target.clone();
    }
    let max_lod = available.max_lod();
    let target_refined = refined_nodes(target, max_lod);
    let mut budget = if max_new_tiles == 0 {
        usize::MAX
    } else {
        max_new_tiles
    };
    let mut out: BTreeSet<TileKey> = BTreeSet::new();
    let mut handled: BTreeSet<TileKey> = BTreeSet::new();

    let mut order: Vec<TileKey> = current.iter().copied().collect();
    order.sort_by(|a, b| {
        grid.distance_to(*a, camera_xz)
            .total_cmp(&grid.distance_to(*b, camera_xz))
            .then_with(|| a.cmp(b))
    });

    for node in order {
        if handled.contains(&node) {
            continue;
        }
        if target.contains(&node) {
            out.insert(node);
            continue;
        }
        // Coarsening: the target serves this ground with an ancestor. Replace the
        // node's entire sibling group under that ancestor in one move.
        if let Some(anc) = ancestor_in(node, target, max_lod) {
            let group: Vec<TileKey> = current
                .iter()
                .copied()
                .filter(|m| is_descendant_of(*m, anc))
                .collect();
            for m in &group {
                handled.insert(*m);
            }
            let cost = usize::from(!resident(anc));
            if cost <= budget {
                budget -= cost;
                out.insert(anc);
            } else {
                out.extend(group); // can't afford it yet — hold the finer set
            }
            continue;
        }
        // Refinement: the target has descendants under this node. Step one level.
        if target_refined.contains(&node) && node.lod > 0 {
            let kids: Vec<TileKey> = node
                .children()
                .into_iter()
                .filter(|c| available.has(*c))
                .collect();
            let cost = kids.iter().filter(|c| !resident(**c)).count();
            if !kids.is_empty() && cost <= budget {
                budget -= cost;
                out.extend(kids);
            } else {
                out.insert(node);
            }
            continue;
        }
        // The target does not cover this ground at all (culled, or the terrain
        // changed): drop it.
    }

    // Anything the target wants whose ground `current` never covered — the first
    // sync, or newly-reachable terrain — enters directly, budget permitting.
    let out_refined = refined_nodes(&out, max_lod);
    for &t in target {
        if out.contains(&t) || out_refined.contains(&t) || ancestor_in(t, &out, max_lod).is_some() {
            continue;
        }
        let cost = usize::from(!resident(t));
        if cost > budget {
            continue;
        }
        budget -= cost;
        out.insert(t);
    }
    out
}

/// The nearest strict ancestor of `key` present in `set`, if any.
fn ancestor_in(key: TileKey, set: &BTreeSet<TileKey>, max_lod: u32) -> Option<TileKey> {
    let mut cur = key;
    while cur.lod < max_lod {
        cur = cur.parent();
        if set.contains(&cur) {
            return Some(cur);
        }
    }
    None
}

/// Whether `key` lies strictly under `ancestor` in the pyramid.
fn is_descendant_of(key: TileKey, ancestor: TileKey) -> bool {
    if key.lod >= ancestor.lod {
        return false;
    }
    let mut cur = key;
    while cur.lod < ancestor.lod {
        cur = cur.parent();
    }
    cur == ancestor
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full `n × n` level-0 grid plus every coarse level up to a `1 × 1` root
    /// (`n` a power of two), so the pyramid is complete and every refinement is
    /// legal — the shape the cut properties are stated against.
    fn full_pyramid(n: i32) -> TileCatalog {
        let mut keys = Vec::new();
        let mut side = n;
        let mut lod = 0;
        while side >= 1 {
            for tz in 0..side {
                for tx in 0..side {
                    keys.push(TileKey::new(lod, (tx, tz)));
                }
            }
            if side == 1 {
                break;
            }
            side /= 2;
            lod += 1;
        }
        TileCatalog::from_keys(keys)
    }

    fn grid() -> TileGrid {
        TileGrid::new(17, 1.0) // level-0 span = 16 m
    }

    /// Every ground patch is covered by **exactly one** page: no key in the set
    /// has an ancestor or a descendant also in the set.
    fn assert_is_cut(set: &BTreeSet<TileKey>, max_lod: u32) {
        for &k in set {
            assert!(
                ancestor_in(k, set, max_lod).is_none(),
                "{k:?} coexists with an ancestor in {set:?}"
            );
        }
    }

    /// The renderer's steady-state contract: never three of four children beside
    /// their parent. (Implied by `assert_is_cut`, asserted separately because it
    /// is the property the terrain pass names.)
    fn assert_no_partial_sibling_set(set: &BTreeSet<TileKey>) {
        for &k in set {
            if k.lod == 0 {
                continue;
            }
            let resident_kids = k.children().iter().filter(|c| set.contains(c)).count();
            assert_eq!(
                resident_kids, 0,
                "{k:?} is resident beside {resident_kids} of its children"
            );
        }
    }

    #[test]
    fn grid_geometry_matches_the_terrain_convention() {
        let g = grid();
        assert_eq!(g.level0_span(), 16.0);
        assert_eq!(g.span_at(0), 16.0);
        assert_eq!(g.span_at(3), 128.0);
        let (lo, hi) = g.bounds(TileKey::new(1, (2, -1)));
        assert_eq!(lo, DVec2::new(64.0, -32.0));
        assert_eq!(hi, DVec2::new(96.0, 0.0));
        // Inside ⇒ zero; outside ⇒ the box distance.
        assert_eq!(
            g.distance_to(TileKey::lod0((0, 0)), DVec2::new(8.0, 8.0)),
            0.0
        );
        assert_eq!(
            g.distance_to(TileKey::lod0((0, 0)), DVec2::new(-6.0, 8.0)),
            6.0
        );
        // Floor semantics on the negative side (never truncate toward zero).
        assert_eq!(g.coord_at(0, DVec2::new(-0.5, -16.5)), (-1, -2));
    }

    #[test]
    fn render_wants_is_always_a_quadtree_cut() {
        let cat = full_pyramid(8); // lods 0..=3
        let g = grid();
        let params = RenderWantsParams::geometric(24.0, 4);
        // Sweep the camera across the whole terrain (and off its edges).
        for i in -4..20 {
            for j in -4..20 {
                let cam = DVec2::new(i as f64 * 8.0, j as f64 * 8.0);
                let cut = render_wants(cam, g, &params, &cat, &BTreeSet::new());
                assert!(!cut.is_empty(), "camera {cam:?} produced no pages");
                assert_is_cut(&cut, cat.max_lod());
                assert_no_partial_sibling_set(&cut);
            }
        }
    }

    #[test]
    fn render_wants_is_deterministic_and_camera_driven() {
        let cat = full_pyramid(8);
        let g = grid();
        let params = RenderWantsParams::geometric(24.0, 4);
        let cam = DVec2::new(20.0, 20.0);
        let a = render_wants(cam, g, &params, &cat, &BTreeSet::new());
        let b = render_wants(cam, g, &params, &cat, &BTreeSet::new());
        assert_eq!(a, b, "the same inputs must produce the same set");

        // Near the camera the cut reaches level 0; far away it stays coarse.
        assert!(
            a.iter().any(|k| k.lod == 0),
            "the cut refines to level 0 under the camera: {a:?}"
        );
        assert!(
            a.iter().any(|k| k.lod > 0),
            "the cut keeps coarse pages away from the camera: {a:?}"
        );
        let near = a
            .iter()
            .filter(|k| g.distance_to(**k, cam) == 0.0)
            .map(|k| k.lod)
            .min()
            .unwrap();
        assert_eq!(near, 0, "the tile under the camera is full resolution");

        // Moving the camera moves the fine region.
        let far = render_wants(DVec2::new(120.0, 120.0), g, &params, &cat, &BTreeSet::new());
        assert_ne!(a, far);
    }

    #[test]
    fn hysteresis_kills_the_hover_thrash() {
        let cat = full_pyramid(8);
        let g = grid();
        let params = RenderWantsParams {
            hysteresis: 0.25,
            ..RenderWantsParams::geometric(24.0, 4)
        };
        // Park the camera exactly on a lod-1 node's refine threshold and jitter it
        // inside the dead band: the published cut must not change at all.
        let node = TileKey::new(1, (2, 2));
        let r = params.refine_radius(1);
        let (lo, _) = g.bounds(node);
        let on_edge = DVec2::new(lo.x - r, lo.y + g.span_at(1) * 0.5);

        let settled = render_wants(on_edge, g, &params, &cat, &BTreeSet::new());
        let mut cut = settled.clone();
        let mut seen = BTreeSet::new();
        for k in 0..24 {
            // ±10 % of the radius — inside the ±25 % dead band.
            let wobble = if k % 2 == 0 { 0.1 } else { -0.1 } * r;
            let cam = on_edge + DVec2::new(wobble, 0.0);
            cut = render_wants(cam, g, &params, &cat, &cut);
            seen.insert(cut.clone());
            assert_is_cut(&cut, cat.max_lod());
        }
        assert_eq!(
            seen.len(),
            1,
            "a hovering camera published {} distinct cuts — hysteresis is not holding",
            seen.len()
        );

        // Without hysteresis the same wobble DOES flip the node (the property is
        // real, not an artefact of the fixture).
        let bare = RenderWantsParams {
            hysteresis: 0.0,
            ..params.clone()
        };
        let mut flips = BTreeSet::new();
        let mut c = render_wants(on_edge, g, &bare, &cat, &BTreeSet::new());
        for k in 0..8 {
            let wobble = if k % 2 == 0 { 0.1 } else { -0.1 } * r;
            c = render_wants(on_edge + DVec2::new(wobble, 0.0), g, &bare, &cat, &c);
            flips.insert(c.clone());
        }
        assert!(
            flips.len() > 1,
            "the fixture must straddle a real threshold"
        );
    }

    #[test]
    fn render_wants_handles_negative_coordinates() {
        // A pyramid over the NEGATIVE quadrant: floor-semantics grouping must keep
        // parents and children aligned (a truncating `/2` would split -1 from -2).
        let mut keys = Vec::new();
        for tz in -8..0 {
            for tx in -8..0 {
                keys.push(TileKey::lod0((tx, tz)));
            }
        }
        for lod in 1..=3u32 {
            let side: i32 = 8 >> lod;
            for tz in -side..0 {
                for tx in -side..0 {
                    keys.push(TileKey::new(lod, (tx, tz)));
                }
            }
        }
        let cat = TileCatalog::from_keys(keys);
        let g = grid();
        let params = RenderWantsParams::geometric(24.0, 4);
        let cam = DVec2::new(-20.0, -20.0);
        let cut = render_wants(cam, g, &params, &cat, &BTreeSet::new());
        assert_is_cut(&cut, cat.max_lod());
        assert_no_partial_sibling_set(&cut);
        assert!(cut.iter().all(|k| k.coord.0 < 0 && k.coord.1 < 0));
        assert!(cut.iter().any(|k| k.lod == 0), "refines under the camera");
        // Every parent/child link really is the floor-halved one.
        for &k in &cut {
            let p = k.parent();
            assert_eq!(p.coord.0, k.coord.0.div_euclid(2));
            assert_eq!(p.coord.1, k.coord.1.div_euclid(2));
        }
    }

    #[test]
    fn max_distance_drops_whole_nodes_not_partial_ones() {
        let cat = full_pyramid(8);
        let g = grid();
        let params = RenderWantsParams {
            max_distance: 40.0,
            ..RenderWantsParams::geometric(24.0, 4)
        };
        let cam = DVec2::new(8.0, 8.0);
        let cut = render_wants(cam, g, &params, &cat, &BTreeSet::new());
        assert!(!cut.is_empty());
        assert_is_cut(&cut, cat.max_lod());
        assert!(cut.iter().all(|k| g.distance_to(*k, cam) <= 40.0));
        // Strictly fewer pages than the unlimited cut.
        let all = render_wants(
            cam,
            g,
            &RenderWantsParams::geometric(24.0, 4),
            &cat,
            &BTreeSet::new(),
        );
        assert!(cut.len() < all.len());
    }

    #[test]
    fn a_source_with_no_pyramid_wants_every_level_zero_tile() {
        // max_lod == 0 ⇒ the walk starts and ends at level 0 (the pre-P16.3
        // "fully resident" behaviour), and it is still trivially a cut.
        let cat = TileCatalog::from_keys(tile_range(0, (0, 0), (3, 3)));
        let cut = render_wants(
            DVec2::ZERO,
            grid(),
            &RenderWantsParams::geometric(24.0, 4),
            &cat,
            &BTreeSet::new(),
        );
        assert_eq!(cut.len(), 16);
        assert!(cut.iter().all(|k| k.lod == 0));
        assert_is_cut(&cut, 0);
    }

    #[test]
    fn sim_wants_is_level_zero_only_and_camera_free() {
        let g = grid(); // 16 m tiles
        let pos = [DVec2::new(8.0, 8.0)];
        let w = sim_wants(g, &pos, 1.0);
        assert_eq!(w, tile_range(0, (0, 0), (0, 0)));
        assert!(w.iter().all(|k| k.is_lod0()));

        // A margin that straddles a boundary pulls in the neighbours.
        let w = sim_wants(g, &[DVec2::new(15.5, 15.5)], 1.0);
        assert_eq!(w, tile_range(0, (0, 0), (1, 1)));

        // Several positions union; the result is order-independent.
        let a = sim_wants(g, &[DVec2::new(8.0, 8.0), DVec2::new(40.0, 8.0)], 1.0);
        let b = sim_wants(g, &[DVec2::new(40.0, 8.0), DVec2::new(8.0, 8.0)], 1.0);
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
    }

    /// A brush dab wants its footprint **plus one whole tile** — the sim-wants
    /// shape, so a neighbourhood op (Smooth's 3 × 3 read, `normal_at`'s central
    /// difference, `height_at`'s shared-edge fallback) can never reach a
    /// non-resident tile and read it as unauthored ground (P16.4b).
    #[test]
    fn brush_wants_is_the_footprint_plus_one_tile() {
        let g = grid(); // 16 m tiles
                        // A dab well inside tile (1,1) with a tiny radius still pulls the ring of
                        // neighbours in, because the margin is a full tile span.
        let w = brush_wants(g, DVec2::new(24.0, 24.0), 0.5);
        assert_eq!(w, tile_range(0, (0, 0), (2, 2)));
        assert!(w.iter().all(|k| k.is_lod0()), "level 0 only");

        // A radius spanning two tiles widens the range by exactly that much.
        let w = brush_wants(g, DVec2::new(24.0, 24.0), 32.0);
        assert_eq!(w, tile_range(0, (-2, -2), (4, 4)));

        // It IS sim_wants with a one-tile-widened margin — the shapes must not
        // drift apart.
        assert_eq!(
            brush_wants(g, DVec2::new(-7.0, 40.0), 5.0),
            sim_wants(g, &[DVec2::new(-7.0, 40.0)], 5.0 + g.level0_span())
        );
        // A degenerate radius still asks for the tile under the cursor + its ring.
        assert_eq!(
            brush_wants(g, DVec2::new(24.0, 24.0), -3.0),
            tile_range(0, (0, 0), (2, 2))
        );
    }

    #[test]
    fn sim_wants_handles_negative_positions() {
        let g = grid();
        // Exactly on the origin boundary: the shared-edge neighbour on the -X/-Z
        // side must be included, or `height_at`'s edge fallback goes dark.
        let w = sim_wants(g, &[DVec2::ZERO], 1.0);
        assert_eq!(w, tile_range(0, (-1, -1), (0, 0)));
        let w = sim_wants(g, &[DVec2::new(-20.0, -33.0)], 2.0);
        assert!(w.contains(&TileKey::lod0((-2, -3))));
        assert!(w.iter().all(|k| k.coord.0 < 0 && k.coord.1 < 0));
    }

    #[test]
    fn clamp_collapses_coarsest_first_and_stays_a_cut() {
        let cat = full_pyramid(8);
        let g = grid();
        let cam = DVec2::new(8.0, 8.0);
        let full = render_wants(
            cam,
            g,
            &RenderWantsParams::geometric(24.0, 4),
            &cat,
            &BTreeSet::new(),
        );
        assert!(full.len() > 8);

        let clamped = clamp_cut(&full, g, cam, &cat, 8);
        assert!(clamped.len() <= 8, "budget honoured: {}", clamped.len());
        assert_is_cut(&clamped, cat.max_lod());
        assert_no_partial_sibling_set(&clamped);
        // Detail was spent, not coverage: the clamped set is coarser on average.
        let mean =
            |s: &BTreeSet<TileKey>| s.iter().map(|k| k.lod as f64).sum::<f64>() / s.len() as f64;
        assert!(
            mean(&clamped) > mean(&full),
            "clamping must coarsen ({} → {})",
            mean(&full),
            mean(&clamped)
        );
        // Deterministic.
        assert_eq!(clamped, clamp_cut(&full, g, cam, &cat, 8));
        // A budget of 0 (or one already satisfied) is a pass-through.
        assert_eq!(clamp_cut(&full, g, cam, &cat, 0), full);
        assert_eq!(clamp_cut(&full, g, cam, &cat, full.len()), full);
        // The nearest ground keeps the most detail: the tile under the camera is
        // still finer than the farthest page.
        let near_lod = clamped
            .iter()
            .filter(|k| g.distance_to(**k, cam) == 0.0)
            .map(|k| k.lod)
            .min()
            .unwrap();
        let far_lod = clamped
            .iter()
            .map(|k| (g.distance_to(*k, cam), k.lod))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap()
            .1;
        assert!(near_lod <= far_lod, "near {near_lod} vs far {far_lod}");
    }

    /// The clamp must look **up** the pyramid for a collapsible group before it
    /// resorts to dropping coverage.
    ///
    /// Fixture: the finest level (0) holds two tiles that are each the only child
    /// of an absent parent — un-collapsible twice over — while level 1 holds a full
    /// sibling set whose level-2 parent exists. The old finest-level-only search
    /// gave up at level 0 and dropped a page; the level walk folds the level-1
    /// group instead and keeps every bit of ground.
    #[test]
    fn clamp_walks_up_to_a_collapsible_level_before_dropping_coverage() {
        let g = grid();
        let cam = DVec2::new(8.0, 8.0);
        // Two lonely fine tiles (parents deliberately NOT in the catalog) …
        let lonely = [TileKey::lod0((0, 0)), TileKey::lod0((100, 100))];
        // … and a complete level-1 sibling set under an existing level-2 parent.
        let group = [
            TileKey::new(1, (10, 10)),
            TileKey::new(1, (11, 10)),
            TileKey::new(1, (10, 11)),
            TileKey::new(1, (11, 11)),
        ];
        let parent = TileKey::new(2, (5, 5));
        let cat =
            TileCatalog::from_keys(lonely.iter().chain(group.iter()).copied().chain([parent]));
        for k in &lonely {
            assert!(!cat.has(k.parent()), "the fixture needs an absent parent");
        }

        let cut: BTreeSet<TileKey> = lonely.iter().chain(group.iter()).copied().collect();
        assert_eq!(cut.len(), 6);
        assert_is_cut(&cut, cat.max_lod());

        let clamped = clamp_cut(&cut, g, cam, &cat, 5);
        assert!(clamped.len() <= 5, "budget honoured: {}", clamped.len());
        assert_is_cut(&clamped, cat.max_lod());

        // NOTHING was dropped: the level-1 group folded into its parent instead.
        assert!(
            clamped.contains(&parent),
            "the coarser level's group must collapse: {clamped:?}"
        );
        assert!(group.iter().all(|k| !clamped.contains(k)));
        for k in &lonely {
            assert!(
                clamped.contains(k),
                "{k:?} was dropped even though a coarser level could still fold"
            );
        }
        // Coverage is conserved: every original page is still served by itself or
        // by an ancestor now in the set.
        for &k in &cut {
            assert!(
                clamped.contains(&k) || ancestor_in(k, &clamped, cat.max_lod()).is_some(),
                "{k:?} lost its coverage"
            );
        }
        assert_eq!(clamped, clamp_cut(&cut, g, cam, &cat, 5), "deterministic");
    }

    /// The clamp must **refuse** a fold that would strand a refined sibling's
    /// descendants under the parent it is about to publish.
    ///
    /// Adversarial shape (unreachable from this module's own producers, but
    /// `clamp_cut` is `pub` and takes an arbitrary cut): three of a level-2 node's
    /// four children sit at level 1, while the fourth was refined into its own
    /// level-0 children — and that fourth child is **absent from the catalog**, so
    /// the level-0 group cannot fold and the walk moves up to level 1. Folding the
    /// three level-1 siblings into the level-2 parent would publish that parent on
    /// top of the level-0 pages still in the set: a page coexisting with its own
    /// descendants, the one thing a cut may never contain.
    #[test]
    fn clamp_refuses_a_fold_that_would_strand_descendants() {
        let g = grid();
        let cam = DVec2::ZERO;
        let parent = TileKey::new(2, (0, 0));
        // Three of the parent's four level-1 children …
        let siblings = [
            TileKey::new(1, (0, 0)),
            TileKey::new(1, (1, 0)),
            TileKey::new(1, (0, 1)),
        ];
        // … the fourth, (1,1), is refined into its level-0 children instead.
        let refined_away = TileKey::new(1, (1, 1));
        let grandkids = refined_away.children();
        assert!(grandkids.iter().all(|k| k.lod == 0));
        assert!(grandkids.iter().all(|k| is_descendant_of(*k, parent)));

        // The catalog deliberately OMITS the refined-away child, so the level-0
        // group has no existing parent to fold into and the walk must climb.
        let cat = TileCatalog::from_keys(siblings.iter().copied().chain(grandkids).chain([parent]));
        assert!(
            !cat.has(refined_away),
            "the fixture needs the 4th child absent"
        );

        let cut: BTreeSet<TileKey> = siblings.iter().copied().chain(grandkids).collect();
        assert_eq!(cut.len(), 7);
        assert_is_cut(&cut, cat.max_lod());

        let clamped = clamp_cut(&cut, g, cam, &cat, 6);
        assert!(clamped.len() <= 6, "budget honoured: {}", clamped.len());

        // THE GUARD: the level-2 parent must not have been published over the
        // level-0 pages still present.
        assert!(
            !clamped.contains(&parent),
            "the fold was allowed and stranded descendants under {parent:?}: {clamped:?}"
        );
        // The result is still a genuine cut, and still no partial sibling set.
        assert_is_cut(&clamped, cat.max_lod());
        assert_no_partial_sibling_set(&clamped);
        assert_eq!(clamped, clamp_cut(&cut, g, cam, &cat, 6), "deterministic");

        // The guard is NARROW, not a blanket refusal: with the stranded
        // descendants gone, the very same group folds happily.
        let clean: BTreeSet<TileKey> = siblings.iter().copied().collect();
        let folded = clamp_cut(&clean, g, cam, &cat, 2);
        assert!(
            folded.contains(&parent) && folded.len() == 1,
            "the guard must not block a legitimate fold: {folded:?}"
        );
        assert_is_cut(&folded, cat.max_lod());
    }

    #[test]
    fn advance_converges_and_every_step_is_a_cut() {
        let cat = full_pyramid(8);
        let g = grid();
        let params = RenderWantsParams::geometric(24.0, 4);
        let cam = DVec2::new(8.0, 8.0);
        let target = render_wants(cam, g, &params, &cat, &BTreeSet::new());

        // Start from the root level (what the streamer seeds with) and converge.
        let mut cur: BTreeSet<TileKey> = cat.keys_at_lod(cat.max_lod()).into_iter().collect();
        let mut resident = cur.clone();
        let mut steps = 0;
        while cur != target {
            let next = advance_cut(&cur, &target, g, cam, &cat, &|k| resident.contains(&k), 4);
            assert_ne!(next, cur, "the walk stalled at {cur:?}");
            assert_is_cut(&next, cat.max_lod());
            assert_no_partial_sibling_set(&next);
            resident.extend(next.iter().copied());
            cur = next;
            steps += 1;
            assert!(steps < 200, "did not converge");
        }
        assert!(steps > 1, "an unlimited jump would defeat the amortization");

        // Now walk back to a distant camera: coarsening is equally cut-preserving.
        let cam2 = DVec2::new(400.0, 400.0);
        let target2 = render_wants(cam2, g, &params, &cat, &target);
        let mut steps = 0;
        while cur != target2 {
            let next = advance_cut(&cur, &target2, g, cam2, &cat, &|k| resident.contains(&k), 4);
            assert_ne!(next, cur, "the walk stalled at {cur:?}");
            assert_is_cut(&next, cat.max_lod());
            assert_no_partial_sibling_set(&next);
            resident.extend(next.iter().copied());
            cur = next;
            steps += 1;
            assert!(steps < 200, "did not converge back");
        }
    }

    #[test]
    fn advance_is_deterministic_and_budget_bounded() {
        let cat = full_pyramid(8);
        let g = grid();
        let cam = DVec2::new(8.0, 8.0);
        let target = render_wants(
            cam,
            g,
            &RenderWantsParams::geometric(24.0, 4),
            &cat,
            &BTreeSet::new(),
        );
        let cur: BTreeSet<TileKey> = cat.keys_at_lod(cat.max_lod()).into_iter().collect();
        let resident = cur.clone();
        let nothing_resident = |_: TileKey| false;

        let a = advance_cut(&cur, &target, g, cam, &cat, &|k| resident.contains(&k), 4);
        let b = advance_cut(&cur, &target, g, cam, &cat, &|k| resident.contains(&k), 4);
        assert_eq!(a, b, "one step is a pure function of its inputs");

        // The budget really bounds the newly-introduced tiles.
        let step = advance_cut(&cur, &target, g, cam, &cat, &nothing_resident, 4);
        let new = step.difference(&cur).count();
        assert!(new <= 4, "{new} new tiles exceeded the budget of 4");

        // An unlimited budget lands on the target in one step from a cut that only
        // needs refinement one level deep.
        let one_level: BTreeSet<TileKey> = cat.keys_at_lod(1).into_iter().collect();
        let all_zero: BTreeSet<TileKey> = cat.keys_at_lod(0).into_iter().collect();
        let jump = advance_cut(&one_level, &all_zero, g, cam, &cat, &nothing_resident, 0);
        assert_eq!(jump, all_zero);
    }

    #[test]
    fn resident_tiles_are_free_in_the_budget() {
        // Re-entering a page that never left memory must not consume budget —
        // otherwise a camera oscillating across a threshold would crawl.
        let cat = full_pyramid(4);
        let g = grid();
        let cam = DVec2::ZERO;
        let parents: BTreeSet<TileKey> = cat.keys_at_lod(1).into_iter().collect();
        let kids: BTreeSet<TileKey> = cat.keys_at_lod(0).into_iter().collect();
        let everything_resident = |_: TileKey| true;
        let step = advance_cut(&parents, &kids, g, cam, &cat, &everything_resident, 1);
        assert_eq!(step, kids, "already-resident children cost nothing");
    }

    #[test]
    fn ancestor_and_descendant_helpers_agree() {
        let k = TileKey::lod0((-3, 5));
        let p = k.parent();
        assert!(is_descendant_of(k, p));
        assert!(is_descendant_of(k, p.parent()));
        assert!(!is_descendant_of(p, k));
        assert!(!is_descendant_of(k, k));
        assert!(p.children().contains(&k));
        let set: BTreeSet<TileKey> = [p.parent()].into_iter().collect();
        assert_eq!(ancestor_in(k, &set, 8), Some(p.parent()));
        assert_eq!(ancestor_in(k, &BTreeSet::new(), 8), None);
    }
}
