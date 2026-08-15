//! **Wants**: the pure, deterministic functions that decide *which* chunks a
//! consumer needs resident (P21.2).
//!
//! [`residency`](crate::residency) deliberately owns no policy — it executes an
//! explicit set of wanted keys. This module is the policy, kept equally pure: a
//! camera position plus a chunk index in, a [`BTreeSet<ChunkKey>`] out. No I/O, no
//! clock, no job pool, no interior mutability — so every rule below is
//! unit-testable without a GPU, a pack or a frame loop, and two runs over the same
//! inputs produce the identical set.
//!
//! It is deliberately shaped like `inf_terrain::wants` / `inf_terrain::stream`,
//! down to the vocabulary (`wants` → `clamp` → `advance` → a budget struct),
//! because it is the same problem one dimension up. Anything a reader already
//! knows about terrain streaming transfers here; the three genuine differences are
//! called out where they occur:
//!
//! 1. **There is no pyramid.** A terrain page has a coarser parent to fall back
//!    on, so its budget clamp spends *detail* and keeps *coverage*. A voxel chunk
//!    has no coarser form, so [`clamp_wants`] can only drop whole chunks —
//!    farthest first. What it spends is reach, and the loss is honest: a dropped
//!    chunk reads as empty space, exactly like one that was never carved.
//! 2. **The want set is a sphere, not a quadtree cut.** There is no
//!    "no partial sibling set" invariant to preserve, so every intermediate state
//!    of [`advance_wants`] is trivially legal and the batch bound is the only
//!    thing the incremental step buys.
//! 3. **Volumes are three-dimensional.** A radius that is generous in XZ is
//!    ruinous in Y for a deep shaft, which is why the radius is measured against
//!    the chunk's own box in 3-D rather than against its XZ footprint.
//!
//! # THE DETERMINISM DOCTRINE (restated where it is implemented)
//!
//! A fixed-step simulation's results must **never** depend on camera-driven
//! residency, or the replay / PIE-==-shipping gates die. Everything in this module
//! is the *render* driver: it is fed a camera and it feeds a store that no ECS
//! entity references. The simulation's own voxel set is a separate, camera-free
//! map seeded from sim state alone (`SimSession::set_voxel_volumes` /
//! `RuntimeSim::set_voxel_volumes`, read by [`crate::ground`]), and the two are
//! structurally apart for exactly the reason `inf_terrain::wants` states: there is
//! no reference from the world to the set the camera fills.
//!
//! # Hysteresis
//!
//! A camera hovering exactly on the radius would otherwise page a chunk in and out
//! every frame. The test is therefore **asymmetric**: a chunk that was not wanted
//! last sync is taken only inside `r · (1 − h)`, while one that was already wanted
//! is kept out to `r · (1 + h)`. The dead band `[r(1−h), r(1+h)]` costs a ring of
//! chunks at its outer edge and buys a thrash-free hover — verbatim the terrain
//! rule, and it has to be, because a re-paged chunk costs a **re-mesh of its whole
//! 3 × 3 × 3 neighbourhood** here rather than one buffer upload.

use std::collections::BTreeSet;

use glam::DVec3;

use crate::chunk::{ChunkKey, CHUNK_DIM};
use crate::residency::ChunkStore;

// ── grid geometry ───────────────────────────────────────────────────────────

/// A volume's chunk grid in world space: everything the want rules need to turn a
/// [`ChunkKey`] into a world box.
///
/// Chunk `(x, y, z)` spans `origin + (x, y, z)·S … origin + (x+1, y+1, z+1)·S`
/// where `S = CHUNK_DIM · voxel_size_m` — the same convention
/// [`VoxelData::chunk_origin_world`](crate::VoxelData::chunk_origin_world) uses, so
/// the two agree by construction rather than by review.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkGrid {
    span: f64,
    origin: DVec3,
}

impl ChunkGrid {
    /// From a volume's voxel scale and world anchor.
    ///
    /// The scale is clamped to a positive finite length for the reason
    /// [`VoxelData::new`](crate::VoxelData::new) clamps it: a zero or NaN span
    /// divides by zero in every coordinate mapping, and it is reachable from the
    /// Details grid.
    pub fn new(voxel_size_m: f64, origin: DVec3) -> Self {
        let vs = if voxel_size_m.is_finite() && voxel_size_m > 0.0 {
            voxel_size_m
        } else {
            0.5
        };
        Self {
            span: CHUNK_DIM as f64 * vs,
            origin: if origin.is_finite() {
                origin
            } else {
                DVec3::ZERO
            },
        }
    }

    /// World edge length of one chunk.
    #[inline]
    pub fn span(&self) -> f64 {
        self.span
    }

    /// World anchor of chunk `(0, 0, 0)`'s lowest corner.
    #[inline]
    pub fn origin(&self) -> DVec3 {
        self.origin
    }

    /// The `(min, max)` world corners of `key`'s box.
    pub fn bounds(&self, key: ChunkKey) -> (DVec3, DVec3) {
        let lo = self.origin + DVec3::new(key.x as f64, key.y as f64, key.z as f64) * self.span;
        (lo, lo + DVec3::splat(self.span))
    }

    /// The chunk containing world position `p` (floored, so negative coordinates
    /// land on the chunk *below* rather than toward zero).
    pub fn coord_at(&self, p: DVec3) -> ChunkKey {
        let g = (p - self.origin) / self.span;
        ChunkKey::new(floor_i32(g.x), floor_i32(g.y), floor_i32(g.z))
    }

    /// Shortest world distance from `p` to `key`'s box; `0` inside it.
    ///
    /// Box distance rather than centre distance, deliberately: a centre metric
    /// would make a chunk's want depend on a point that is up to `span·√3/2` from
    /// the surface the camera is actually looking at, and at the default half-metre
    /// scale that is 7 m of error on an 8 m chunk.
    pub fn distance_to(&self, key: ChunkKey, p: DVec3) -> f64 {
        let (lo, hi) = self.bounds(key);
        (lo - p).max(p - hi).max(DVec3::ZERO).length()
    }
}

/// Floor a `f64` to `i32`, saturating rather than wrapping — a position that far
/// out names no real chunk and simply never matches one.
#[inline]
fn floor_i32(v: f64) -> i32 {
    if !v.is_finite() {
        return 0;
    }
    v.floor().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

// ── the availability index ──────────────────────────────────────────────────

/// What a want rule may ask about the *available* chunks.
///
/// Deliberately tiny and read-only — the rules must never be able to load, evict
/// or mutate anything, which is what keeps them pure. [`ChunkCatalog`] is the
/// concrete implementation built once from a [`ChunkStore`]; tests implement it
/// directly over a hand-written key set. (The mirror of
/// `inf_terrain::wants::TileIndex`, minus the two pyramid methods it has no
/// analogue for.)
pub trait ChunkIndex {
    /// Whether `key` exists in the source.
    fn has(&self, key: ChunkKey) -> bool;

    /// Every key the source can serve, ascending.
    fn keys(&self) -> Vec<ChunkKey>;
}

/// The keys a chunk source can serve, indexed for the want rules.
///
/// Built **once** per bound volume (a `.inf_voxel` directory is parsed at open and
/// never re-read) and then queried per sync, which is what makes the per-frame cost
/// a distance test per available chunk rather than a directory walk.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkCatalog {
    keys: BTreeSet<ChunkKey>,
}

impl ChunkCatalog {
    /// From an explicit key set.
    pub fn from_keys(keys: impl IntoIterator<Item = ChunkKey>) -> Self {
        Self {
            keys: keys.into_iter().collect(),
        }
    }

    /// From everything a store can serve.
    pub fn from_store(store: &dyn ChunkStore) -> Self {
        Self::from_keys(store.chunk_keys())
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl ChunkIndex for ChunkCatalog {
    fn has(&self, key: ChunkKey) -> bool {
        self.keys.contains(&key)
    }

    fn keys(&self) -> Vec<ChunkKey> {
        self.keys.iter().copied().collect()
    }
}

// ── the camera-driven want set ──────────────────────────────────────────────

/// Default want dead band (±15 %) — the same number `inf_terrain::wants` uses, and
/// for the same reason.
pub const DEFAULT_HYSTERESIS: f64 = 0.15;

/// Tuning for [`chunk_wants`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelWantsParams {
    /// World-space distance within which a chunk is wanted resident, **metres**.
    /// Measured from the camera to the chunk's box (see [`ChunkGrid::distance_to`]).
    pub radius_m: f64,
    /// Want dead band as a fraction of the radius (see the module docs). `0`
    /// disables hysteresis; values are clamped into `[0, 0.9]`.
    pub hysteresis: f64,
}

impl VoxelWantsParams {
    /// A radius in metres with the default dead band.
    pub fn new(radius_m: f64) -> Self {
        Self {
            radius_m: radius_m.max(0.0),
            hysteresis: DEFAULT_HYSTERESIS,
        }
    }

    fn clamped_hysteresis(&self) -> f64 {
        self.hysteresis.clamp(0.0, 0.9)
    }
}

impl Default for VoxelWantsParams {
    /// **96 m** — twelve chunks at the default half-metre voxel scale.
    ///
    /// Sized against what a carved volume *is* rather than against a view
    /// distance: a cave is enclosed, so the far side of a chunk 100 m away is
    /// behind rock and contributes nothing, while the surface a player is standing
    /// on has to be there before the step that reads it. Wide enough that a cave
    /// system authored at human scale is simply resident, tight enough that a
    /// kilometre-long tunnel is not.
    fn default() -> Self {
        Self::new(96.0)
    }
}

/// The camera-driven want set: which chunks a consumer wants resident.
///
/// `prev` is the previously published set and is used **only** for hysteresis
/// (see the module docs); pass an empty set for a stateless evaluation.
///
/// Pure: the output is a function of `(camera, grid, params, available, prev)`
/// alone, and it comes back in a `BTreeSet` so iteration order is deterministic.
///
/// Cost is one box-distance test per **available** chunk, not per chunk in the
/// radius. That is the right way round for this content: a `.inf_voxel`'s
/// directory is the carved chunks only — the air around a tunnel was never
/// authored — so the catalog is the smaller of the two sets by construction, and
/// a large radius costs nothing extra.
pub fn chunk_wants(
    camera: DVec3,
    grid: ChunkGrid,
    params: &VoxelWantsParams,
    available: &dyn ChunkIndex,
    prev: &BTreeSet<ChunkKey>,
) -> BTreeSet<ChunkKey> {
    let r = params.radius_m.max(0.0);
    let h = params.clamped_hysteresis();
    let (take, keep) = (r * (1.0 - h), r * (1.0 + h));
    let mut out = BTreeSet::new();
    for key in available.keys() {
        let d = grid.distance_to(key, camera);
        let wanted = if prev.contains(&key) {
            d <= keep
        } else {
            d < take
        };
        if wanted {
            out.insert(key);
        }
    }
    out
}

// ── budget ──────────────────────────────────────────────────────────────────

/// The residency budget a camera-driven voxel sync respects — the chunk mirror of
/// `inf_terrain::StreamBudget`.
///
/// # Which budget a load is held against (the §8 law, restated here)
///
/// Paging chunks is a **load**, and a load is measured against
/// `inf_player::budget::LOAD_BUDGET_MS`, **never** `inf_core::FRAME_BUDGET_MS`.
/// The reasoning is written out in full at that constant and is not repeated
/// here, only its consequence for this struct: 33 ms is what a *frame* gets —
/// a thing that must happen thirty times a second forever — while paging a cave
/// in happens once per approach. Asserting a decode-plus-mesh batch against the
/// frame budget would be a hardware claim on a shared CI runner roughly 4× slower
/// than developer hardware, and §8 budgets are unbounded-growth tripwires, not
/// hardware claims.
///
/// What keeps the *frame* honest is not a timing assertion at all, it is
/// [`max_loads_per_sync`](Self::max_loads_per_sync): the batch is bounded by
/// construction, so no single sync can decode a whole volume however far the
/// camera teleported. And what keeps *residency* honest is
/// [`max_resident_chunks`](Self::max_resident_chunks), which is a set-size
/// tripwire — the kind of assertion that has teeth on a noisy runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxelStreamBudget {
    /// Ceiling on resident chunks **per volume** (`0` = unlimited). Over it, the
    /// want set is trimmed farthest-first ([`clamp_wants`]).
    pub max_resident_chunks: usize,
    /// Ceiling on how many not-yet-resident chunks one sync may bring in
    /// (`0` = unlimited). Bounds the per-sync decode + re-mesh cost.
    pub max_loads_per_sync: usize,
}

impl Default for VoxelStreamBudget {
    /// 4 096 resident chunks (≈ 80 MB of `f32` distances at 16³, ≈ 2 km of
    /// half-metre tunnel) and 32 loads per sync.
    ///
    /// The load bound is looser than terrain's 16 because a chunk is ~20 KB
    /// against a tile's ~66 KB and because a voxel volume, unlike a terrain, has
    /// no coarse page to draw while the fine one arrives — a chunk that has not
    /// landed is a *hole in the cave*, so converging quickly matters more here.
    fn default() -> Self {
        Self {
            max_resident_chunks: 4096,
            max_loads_per_sync: 32,
        }
    }
}

/// Bring a want set under a chunk budget by **dropping the farthest chunks**.
///
/// The one place this module cannot mirror `inf_terrain::wants::clamp_cut`: a
/// terrain page has a coarser parent to collapse into, so terrain spends detail
/// and never coverage. A chunk has no coarser form, so the only lever is reach.
/// Farthest-first is therefore not a preference, it is the whole policy — and the
/// result is a legal set, because a dropped chunk reads as empty space exactly
/// like one that was never carved.
///
/// Ties (two chunks equidistant from the camera) break on the key, so the result
/// is a pure function of its inputs.
pub fn clamp_wants(
    wants: &BTreeSet<ChunkKey>,
    grid: ChunkGrid,
    camera: DVec3,
    max_chunks: usize,
) -> BTreeSet<ChunkKey> {
    if max_chunks == 0 || wants.len() <= max_chunks {
        return wants.clone();
    }
    // DECORATE-SORT-UNDECORATE (lens 3 P29, Hardening Wave G). `sort_by` calls
    // its comparator O(n log n) times and evaluates `distance_to` TWICE in each
    // one, and `distance_to` builds a box and takes a `sqrt` — so the key was
    // computed ~2 n log n times for n distinct values. Computing it once per
    // key and sorting the pairs is the same order (`total_cmp` on the same
    // `f64`, tie-broken by the same key comparison, both pure functions of the
    // input) at n evaluations.
    // Nearest first, key-ordered on a tie.
    let mut ordered: Vec<(f64, ChunkKey)> = wants
        .iter()
        .map(|k| (grid.distance_to(*k, camera), *k))
        .collect();
    ordered.sort_by(|(da, a), (db, b)| da.total_cmp(db).then_with(|| a.cmp(b)));
    ordered.truncate(max_chunks);
    ordered.into_iter().map(|(_, k)| k).collect()
}

/// One bounded step of `current` toward `target`.
///
/// Everything `target` drops leaves immediately (an eviction costs nothing and
/// frees memory now), while at most `max_new_chunks` **not-yet-resident** chunks
/// are taken in, nearest-camera-first — so the cave converges under the player
/// before it converges at the far end of the tunnel. `resident` reports which keys
/// are already in memory, so re-entering a chunk that never left is free.
///
/// This is what replaces timing-driven asynchrony: the batch is bounded (no
/// whole-volume hitch) yet its membership is a pure function of the inputs, so a
/// scripted camera path reproduces the identical resident-set trace run to run —
/// which a "whatever finished decoding this frame" scheme could never do.
pub fn advance_wants(
    current: &BTreeSet<ChunkKey>,
    target: &BTreeSet<ChunkKey>,
    grid: ChunkGrid,
    camera: DVec3,
    resident: &dyn Fn(ChunkKey) -> bool,
    max_new_chunks: usize,
) -> BTreeSet<ChunkKey> {
    if current == target {
        return target.clone();
    }
    // Everything the target still wants and `current` already had, keeps.
    let mut out: BTreeSet<ChunkKey> = current.intersection(target).copied().collect();
    let mut budget = if max_new_chunks == 0 {
        usize::MAX
    } else {
        max_new_chunks
    };
    // DECORATE-SORT-UNDECORATE (lens 3 P29, Hardening Wave G). `sort_by` calls
    // its comparator O(n log n) times and evaluates `distance_to` TWICE in each
    // one, and `distance_to` builds a box and takes a `sqrt` — so the key was
    // computed ~2 n log n times for n distinct values. Computing it once per
    // key and sorting the pairs is the same order (`total_cmp` on the same
    // `f64`, tie-broken by the same key comparison, both pure functions of the
    // input) at n evaluations.
    let mut incoming: Vec<(f64, ChunkKey)> = target
        .difference(current)
        .map(|k| (grid.distance_to(*k, camera), *k))
        .collect();
    incoming.sort_by(|(da, a), (db, b)| da.total_cmp(db).then_with(|| a.cmp(b)));
    for (_, key) in incoming {
        // A chunk still in memory costs nothing to re-enter, so it never consumes
        // the batch — otherwise a camera oscillating across the dead band would
        // crawl for chunks it never actually let go of.
        let cost = usize::from(!resident(key));
        if cost > budget {
            continue;
        }
        budget -= cost;
        out.insert(key);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense `n³` catalog anchored at chunk `(0, 0, 0)`.
    fn cube(n: i32) -> ChunkCatalog {
        let mut keys = Vec::new();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    keys.push(ChunkKey::new(x, y, z));
                }
            }
        }
        ChunkCatalog::from_keys(keys)
    }

    /// 1 m voxels ⇒ a 16 m chunk, so every number below is readable.
    fn grid() -> ChunkGrid {
        ChunkGrid::new(1.0, DVec3::ZERO)
    }

    #[test]
    fn grid_geometry_matches_the_volume_convention() {
        let g = grid();
        assert_eq!(g.span(), 16.0);
        let (lo, hi) = g.bounds(ChunkKey::new(2, -1, 0));
        assert_eq!(lo, DVec3::new(32.0, -16.0, 0.0));
        assert_eq!(hi, DVec3::new(48.0, 0.0, 16.0));
        // Inside ⇒ zero; outside ⇒ the box distance.
        assert_eq!(
            g.distance_to(ChunkKey::new(0, 0, 0), DVec3::splat(8.0)),
            0.0
        );
        assert_eq!(
            g.distance_to(ChunkKey::new(0, 0, 0), DVec3::new(-6.0, 8.0, 8.0)),
            6.0
        );
        // Floor semantics on the negative side (never truncate toward zero).
        assert_eq!(
            g.coord_at(DVec3::new(-0.5, -16.5, 0.0)),
            ChunkKey::new(-1, -2, 0)
        );
        // An anchored grid moves with its volume.
        let a = ChunkGrid::new(0.5, DVec3::new(100.0, 0.0, 0.0));
        assert_eq!(a.span(), 8.0);
        assert_eq!(
            a.bounds(ChunkKey::new(0, 0, 0)).0,
            DVec3::new(100.0, 0.0, 0.0)
        );
        // A degenerate scale falls back rather than dividing by zero downstream.
        assert_eq!(ChunkGrid::new(0.0, DVec3::ZERO).span(), 8.0);
        assert_eq!(ChunkGrid::new(f64::NAN, DVec3::ZERO).span(), 8.0);
    }

    #[test]
    fn wants_are_a_deterministic_camera_driven_sphere() {
        let cat = cube(4);
        let g = grid();
        let params = VoxelWantsParams {
            radius_m: 20.0,
            hysteresis: 0.0,
        };
        let cam = DVec3::splat(8.0); // the centre of chunk (0,0,0)
        let a = chunk_wants(cam, g, &params, &cat, &BTreeSet::new());
        let b = chunk_wants(cam, g, &params, &cat, &BTreeSet::new());
        assert_eq!(a, b, "the same inputs must produce the same set");
        assert!(
            a.contains(&ChunkKey::new(0, 0, 0)),
            "the camera's own chunk"
        );
        assert!(
            !a.contains(&ChunkKey::new(3, 3, 3)),
            "the far corner is out"
        );
        assert!(a.len() < cat.len() && !a.is_empty(), "{}", a.len());
        // Every wanted chunk really is inside the radius, and every unwanted one
        // outside it — the rule, not just the count.
        for key in cat.keys() {
            assert_eq!(
                a.contains(&key),
                g.distance_to(key, cam) < 20.0,
                "{key:?} at {}",
                g.distance_to(key, cam)
            );
        }
        // Moving the camera moves the set.
        assert_ne!(
            a,
            chunk_wants(DVec3::splat(56.0), g, &params, &cat, &BTreeSet::new())
        );
        // A zero radius wants NOTHING — not even the chunk the camera is standing
        // in, because the test is `d < take` and distance 0 is not < 0. Stated the
        // way round the assertion actually falls: a caller that wants "just my own
        // chunk" asks for a radius, not for zero.
        let none = chunk_wants(
            cam,
            g,
            &VoxelWantsParams {
                radius_m: 0.0,
                hysteresis: 0.0,
            },
            &cat,
            &BTreeSet::new(),
        );
        assert!(none.is_empty());
    }

    #[test]
    fn hysteresis_kills_the_hover_thrash() {
        let cat = cube(4);
        let g = grid();
        let params = VoxelWantsParams {
            radius_m: 24.0,
            hysteresis: 0.25,
        };
        // Park the camera so a chunk sits exactly on the radius and jitter inside
        // the dead band: the published set must not change at all.
        let key = ChunkKey::new(2, 0, 0);
        let (lo, _) = g.bounds(key);
        let on_edge = DVec3::new(lo.x - params.radius_m, 8.0, 8.0);

        let mut set = chunk_wants(on_edge, g, &params, &cat, &BTreeSet::new());
        let mut seen = BTreeSet::new();
        for k in 0..24 {
            let wobble = if k % 2 == 0 { 0.1 } else { -0.1 } * params.radius_m;
            set = chunk_wants(
                on_edge + DVec3::new(wobble, 0.0, 0.0),
                g,
                &params,
                &cat,
                &set,
            );
            seen.insert(set.clone());
        }
        assert_eq!(
            seen.len(),
            1,
            "a hovering camera published {} distinct want sets — hysteresis is not \
             holding, and every flip costs a 3×3×3 re-mesh",
            seen.len()
        );

        // Without hysteresis the same wobble DOES flip it (the property is real,
        // not an artefact of the fixture).
        let bare = VoxelWantsParams {
            hysteresis: 0.0,
            ..params
        };
        let mut flips = BTreeSet::new();
        let mut s = chunk_wants(on_edge, g, &bare, &cat, &BTreeSet::new());
        for k in 0..8 {
            let wobble = if k % 2 == 0 { 0.1 } else { -0.1 } * bare.radius_m;
            s = chunk_wants(on_edge + DVec3::new(wobble, 0.0, 0.0), g, &bare, &cat, &s);
            flips.insert(s.clone());
        }
        assert!(
            flips.len() > 1,
            "the fixture must straddle a real threshold"
        );
    }

    #[test]
    fn the_clamp_drops_the_farthest_chunks_and_is_deterministic() {
        let cat = cube(4);
        let g = grid();
        let cam = DVec3::splat(8.0);
        let full: BTreeSet<ChunkKey> = cat.keys().into_iter().collect();
        let clamped = clamp_wants(&full, g, cam, 10);
        assert_eq!(clamped.len(), 10);
        assert_eq!(clamped, clamp_wants(&full, g, cam, 10), "deterministic");
        // The kept set is the nearest one: every kept chunk is at most as far as
        // every dropped one.
        let worst_kept = clamped
            .iter()
            .map(|k| g.distance_to(*k, cam))
            .fold(0.0f64, f64::max);
        for key in full.difference(&clamped) {
            assert!(
                g.distance_to(*key, cam) >= worst_kept,
                "{key:?} was dropped while a farther chunk was kept"
            );
        }
        assert!(clamped.contains(&ChunkKey::new(0, 0, 0)));
        // A budget of 0 (or one already satisfied) is a pass-through.
        assert_eq!(clamp_wants(&full, g, cam, 0), full);
        assert_eq!(clamp_wants(&full, g, cam, full.len()), full);
    }

    #[test]
    fn advance_converges_bounded_and_deterministically() {
        let cat = cube(4);
        let g = grid();
        let cam = DVec3::splat(8.0);
        let target: BTreeSet<ChunkKey> = cat.keys().into_iter().collect();
        let nothing_resident = |_: ChunkKey| false;

        let mut cur = BTreeSet::new();
        let mut steps = 0;
        while cur != target {
            let next = advance_wants(&cur, &target, g, cam, &nothing_resident, 8);
            assert!(next.len() - cur.len() <= 8, "the batch bound was exceeded");
            assert_ne!(next, cur, "the walk stalled at {} chunks", cur.len());
            cur = next;
            steps += 1;
            assert!(steps < 100, "did not converge");
        }
        assert!(steps > 1, "an unlimited jump would defeat the amortization");

        // One step is a pure function of its inputs …
        let a = advance_wants(&BTreeSet::new(), &target, g, cam, &nothing_resident, 8);
        let b = advance_wants(&BTreeSet::new(), &target, g, cam, &nothing_resident, 8);
        assert_eq!(a, b);
        // … and it takes the NEAREST chunks first.
        assert!(a.contains(&ChunkKey::new(0, 0, 0)));
        assert!(!a.contains(&ChunkKey::new(3, 3, 3)));

        // Evictions are never held back by the load budget.
        let shrunk: BTreeSet<ChunkKey> = [ChunkKey::new(0, 0, 0)].into_iter().collect();
        assert_eq!(
            advance_wants(&target, &shrunk, g, cam, &nothing_resident, 1),
            shrunk
        );
        // Already-resident chunks are free.
        let everything_resident = |_: ChunkKey| true;
        assert_eq!(
            advance_wants(&BTreeSet::new(), &target, g, cam, &everything_resident, 1),
            target
        );
    }

    #[test]
    fn a_catalog_is_built_from_a_store() {
        use crate::chunk::VoxelChunk;
        use crate::residency::MemoryChunkStore;
        let mut store = MemoryChunkStore::new();
        for key in crate::residency::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
            store.insert(key, &VoxelChunk::solid(1)).unwrap();
        }
        let cat = ChunkCatalog::from_store(&store);
        assert_eq!(cat.len(), 8);
        assert!(cat.has(ChunkKey::new(1, 1, 1)));
        assert!(!cat.has(ChunkKey::new(2, 0, 0)));
        assert!(ChunkCatalog::default().is_empty());
        // Ascending, which is what makes every want set deterministic.
        let keys = cat.keys();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }
}
