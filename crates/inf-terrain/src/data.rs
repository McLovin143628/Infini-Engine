//! [`TerrainData`]: a paged `f64` world-space heightfield.
//!
//! A sparse `BTreeMap<(i32, i32), TerrainTile>` of fixed-resolution height tiles
//! — only authored tiles cost memory, so a 16 km² (or a planetary patch) terrain
//! stores just the pages that carry data. Determinism is structural: the
//! `BTreeMap` fixes iteration/serialization order, and the flat serde form is
//! portable across bincode/JSON.
//!
//! As of P16.3 the tile map is the **resident** set rather than "the terrain": it
//! is grown and shrunk against a [`TileStore`] by an explicit set of wanted
//! `(coord, lod)` pairs, coarse LOD tiles ride beside it in a second map, and
//! every tile carries a monotone version + a dirty mark. See [`crate::residency`]
//! for the contract; nothing about the level-0 API changed, and a fully-resident
//! terrain behaves exactly as it did before.

use std::collections::{BTreeMap, BTreeSet};

use glam::{DVec2, DVec3};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::residency::{ResidencyReport, TileStore};
use crate::tile::{TerrainTile, TileKey};
use crate::HeightSource;

/// Default samples per tile side (256 × 256).
pub const DEFAULT_TILE_RESOLUTION: u32 = 256;
/// Default world units between samples (1 m).
pub const DEFAULT_METERS_PER_SAMPLE: f64 = 1.0;

/// A paged heightfield in `f64` world space.
///
/// Tiles are square blocks of `tile_resolution × tile_resolution` samples spaced
/// `meters_per_sample` apart. Tile grid coordinate `(tx, tz)` places its sample
/// `(0, 0)` at world `(tx · span, ·, tz · span)` where `span = (resolution − 1) ·
/// mps` — so a tile's **last** sample row/column coincides with the **first** of
/// the next tile (shared edges), and authoring both tiles from the same world
/// function keeps them seamless (the boundary-continuity guarantee).
///
/// ## Residency (P16.3)
///
/// `tiles` is the resident **level-0** set — the authored, editable heightfield
/// every query, brush and `.inf_lvl` write already meant. Three companions ride
/// beside it and are **not** serialized (they are runtime streaming state, not
/// authored content, so `.inf_lvl` bytes are untouched):
///
/// * `coarse` — resident LOD ≥ 1 tiles from the pyramid, kept apart so no
///   level-0 code path can ever accidentally see a downsampled tile;
/// * `versions` — the change stamp of each **resident** tile, drawn from one
///   **process-global** monotone counter (see [`TerrainData::tile_version`]);
/// * `dirty` — tiles mutated since the last drain, for write-back.
///
/// `PartialEq` compares only the **authored** state (resolution, spacing, level-0
/// tiles). Two terrains holding the same authored data are equal regardless of
/// which coarse pages happen to be paged in or how many times a tile has been
/// touched — those are derived caches, and letting them into equality would make
/// a scene look dirty for having streamed.
#[derive(Clone, Debug)]
pub struct TerrainData {
    tile_resolution: u32,
    meters_per_sample: f64,
    tiles: BTreeMap<(i32, i32), TerrainTile>,
    coarse: BTreeMap<TileKey, TerrainTile>,
    /// Change stamps for **resident** tiles only — pruned on evict/remove, so the
    /// ledger is bounded by residency and a terrain that pages a planet forever
    /// never grows a map entry per tile it has ever seen. Stamps are drawn from
    /// [`NEXT_TILE_VERSION`], which is *process-global* — see there for why.
    versions: BTreeMap<TileKey, u64>,
    dirty: BTreeSet<TileKey>,
}

/// The single monotone source **every** tile stamp in the process is drawn from.
///
/// Deliberately global rather than per-[`TerrainData`]. A per-instance counter
/// makes two different terrains mint the *same* stamps — two freshly loaded
/// levels both walk their tile `BTreeMap` in order and hand out 1, 2, 3, … — and
/// a consumer that caches by tile key alone (the renderer's GPU tile cache does
/// exactly that) would then see "version 1 == cached version 1" after a
/// File → Open and keep displaying the **previous level's** height texels
/// wherever the two levels' tile coordinates coincide.
///
/// One global counter makes a stamp unique across all `TerrainData` instances for
/// the life of the process, so "same stamp" always means "same content", and
/// every documented property survives unchanged: it only ever increases (so a
/// re-paged tile's stamp still strictly exceeds any it held before), `0` is still
/// never a live stamp, and the per-terrain ledger is still pruned on evict. At
/// one stamp per nanosecond a `u64` lasts ~584 years, so wraparound is not a
/// concern; `Relaxed` is sufficient because the only requirement is uniqueness,
/// not ordering against other memory.
static NEXT_TILE_VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl PartialEq for TerrainData {
    /// Authored state only — see the type docs.
    fn eq(&self, other: &Self) -> bool {
        self.tile_resolution == other.tile_resolution
            && self.meters_per_sample == other.meters_per_sample
            && self.tiles == other.tiles
    }
}

impl Default for TerrainData {
    fn default() -> Self {
        Self::new(DEFAULT_TILE_RESOLUTION, DEFAULT_METERS_PER_SAMPLE)
    }
}

impl TerrainData {
    /// An empty terrain with the given tile resolution (clamped to ≥ 2 so a tile
    /// spans at least one cell) and metres-per-sample (clamped to > 0).
    pub fn new(tile_resolution: u32, meters_per_sample: f64) -> Self {
        Self {
            tile_resolution: tile_resolution.max(2),
            meters_per_sample: if meters_per_sample > 0.0 {
                meters_per_sample
            } else {
                DEFAULT_METERS_PER_SAMPLE
            },
            tiles: BTreeMap::new(),
            coarse: BTreeMap::new(),
            versions: BTreeMap::new(),
            dirty: BTreeSet::new(),
        }
    }

    /// Samples per tile side.
    #[inline]
    pub fn tile_resolution(&self) -> u32 {
        self.tile_resolution
    }

    /// World units between adjacent samples.
    #[inline]
    pub fn meters_per_sample(&self) -> f64 {
        self.meters_per_sample
    }

    /// World size of one tile edge: `(resolution − 1) · mps`. Also the world-space
    /// stride between adjacent tile origins (so edges are shared, not doubled).
    #[inline]
    pub fn tile_span(&self) -> f64 {
        (self.tile_resolution as f64 - 1.0) * self.meters_per_sample
    }

    /// Number of authored tiles.
    #[inline]
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// `true` when no tile is authored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// World position of tile `(tx, tz)`'s sample `(0, 0)` in the XZ plane.
    #[inline]
    pub fn tile_origin_xz(&self, coord: (i32, i32)) -> DVec2 {
        let span = self.tile_span();
        DVec2::new(coord.0 as f64 * span, coord.1 as f64 * span)
    }

    /// The tile grid coordinate containing world `(x, z)` (floored). Points on a
    /// shared edge resolve to the tile on the `+X`/`+Z` side (`local` = 0).
    #[inline]
    pub fn tile_coord_of(&self, x: f64, z: f64) -> (i32, i32) {
        let span = self.tile_span();
        ((x / span).floor() as i32, (z / span).floor() as i32)
    }

    /// Iterate authored tiles in deterministic (grid-coordinate) order.
    pub fn tiles(&self) -> impl Iterator<Item = (&(i32, i32), &TerrainTile)> {
        self.tiles.iter()
    }

    pub fn get_tile(&self, coord: (i32, i32)) -> Option<&TerrainTile> {
        self.tiles.get(&coord)
    }

    /// Mutable access to a resident level-0 tile.
    ///
    /// **Touching is mutating**: the tile's version is bumped and it is marked
    /// dirty on the way out, because a `&mut` handle cannot tell the caller's
    /// intent. Every sculpt / paint / erosion path funnels through here (or
    /// [`get_or_create_tile`](Self::get_or_create_tile)), so "bump only the tiles
    /// under the brush" falls out for free.
    pub fn get_tile_mut(&mut self, coord: (i32, i32)) -> Option<&mut TerrainTile> {
        if !self.tiles.contains_key(&coord) {
            return None;
        }
        self.touch(TileKey::lod0(coord));
        self.tiles.get_mut(&coord)
    }

    /// True when `(tx, tz)` is authored.
    pub fn has_tile(&self, coord: (i32, i32)) -> bool {
        self.tiles.contains_key(&coord)
    }

    /// Get `(tx, tz)`, creating it flat (offsets `0`, `origin.y = 0`) if absent,
    /// and return a mutable handle. The horizontal origin is derived from the
    /// coordinate, so tiles stay grid-aligned.
    /// (Mutating access — bumps the tile's version and marks it dirty; see
    /// [`get_tile_mut`](Self::get_tile_mut).)
    pub fn get_or_create_tile(&mut self, coord: (i32, i32)) -> &mut TerrainTile {
        let res = self.tile_resolution;
        let o = self.tile_origin_xz(coord);
        self.touch(TileKey::lod0(coord));
        self.tiles
            .entry(coord)
            .or_insert_with(|| TerrainTile::flat(res, DVec3::new(o.x, 0.0, o.y)))
    }

    /// Insert (or replace) a tile at `coord`. Returns the tile back if its height
    /// buffer length doesn't match `resolution²` (rejected). The horizontal origin
    /// is overwritten to keep the tile grid-aligned; the caller's `origin.y` (the
    /// `f64` height anchor) is preserved.
    pub fn insert_tile(
        &mut self,
        coord: (i32, i32),
        mut tile: TerrainTile,
    ) -> Result<(), TerrainTile> {
        if tile.heights().len() != (self.tile_resolution * self.tile_resolution) as usize {
            return Err(tile);
        }
        let o = self.tile_origin_xz(coord);
        tile.origin.x = o.x;
        tile.origin.z = o.y;
        self.touch(TileKey::lod0(coord));
        self.tiles.insert(coord, tile);
        Ok(())
    }

    /// Remove `(tx, tz)`, returning it if it existed. An authoring delete —
    /// version-bumping and dirty-marking, unlike a residency
    /// [`evict`](Self::evict_tile), which drops a page without touching either.
    pub fn remove_tile(&mut self, coord: (i32, i32)) -> Option<TerrainTile> {
        let gone = self.tiles.remove(&coord);
        if gone.is_some() {
            let key = TileKey::lod0(coord);
            // The deletion still needs writing back, but the tile is no longer
            // resident, so it holds no stamp (the bounded-ledger rule).
            self.dirty.insert(key);
            self.versions.remove(&key);
        }
        gone
    }

    /// Author an entire tile from a world-space height function `f(x, z) → metres`
    /// (creating the tile if needed). Every sample — including the edges shared
    /// with neighbours — is written from `f`, so neighbouring tiles authored the
    /// same way stay seamless. `origin.y` stays `0`, i.e. the `f32` offset carries
    /// the full height.
    pub fn author_tile<F: FnMut(f64, f64) -> f64>(&mut self, coord: (i32, i32), mut f: F) {
        let res = self.tile_resolution;
        let mps = self.meters_per_sample;
        let o = self.tile_origin_xz(coord);
        let tile = self.get_or_create_tile(coord);
        tile.origin = DVec3::new(o.x, 0.0, o.y);
        for j in 0..res {
            for i in 0..res {
                let wx = o.x + i as f64 * mps;
                let wz = o.y + j as f64 * mps;
                tile.set_sample(res, i, j, f(wx, wz) as f32);
            }
        }
    }

    /// Bulk region write — the seam P10.2 brushes build on. For every authored (or
    /// newly created) tile sample whose world XZ falls in `[min, max]`, write
    /// `offset = f(x, z) − origin.y`. Samples on a shared edge belong to two tiles
    /// and are written in both from the same `f`, so the write stays seamless.
    pub fn write_region<F: FnMut(f64, f64) -> f64>(&mut self, min: DVec2, max: DVec2, mut f: F) {
        let (min, max) = (min.min(max), min.max(max));
        let c0 = self.tile_coord_of(min.x, min.y);
        let c1 = self.tile_coord_of(max.x, max.y);
        let res = self.tile_resolution;
        let mps = self.meters_per_sample;
        for tz in c0.1..=c1.1 {
            for tx in c0.0..=c1.0 {
                let coord = (tx, tz);
                let o = self.tile_origin_xz(coord);
                let tile = self.get_or_create_tile(coord);
                let base_y = tile.origin.y;
                for j in 0..res {
                    let wz = o.y + j as f64 * mps;
                    if wz < min.y || wz > max.y {
                        continue;
                    }
                    for i in 0..res {
                        let wx = o.x + i as f64 * mps;
                        if wx < min.x || wx > max.x {
                            continue;
                        }
                        tile.set_sample(res, i, j, (f(wx, wz) - base_y) as f32);
                    }
                }
            }
        }
    }

    /// Candidate `(tile_index, local_sample)` pairs along one axis for world
    /// coordinate `w`. Normally one pair (the floored tile); exactly on a shared
    /// edge (`local == 0`) it also yields the previous tile's far edge (`res − 1`),
    /// so a query on a single tile's far edge still resolves against it.
    #[inline]
    fn axis_candidates(&self, w: f64) -> [(i32, f64); 2] {
        let span = self.tile_span();
        let mps = self.meters_per_sample;
        let res = self.tile_resolution as f64;
        let th = (w / span).floor() as i32;
        let u = (w - th as f64 * span) / mps; // in [0, res-1)
        if u <= 1e-9 {
            [(th, u), (th - 1, res - 1.0)]
        } else {
            [(th, u), (th, u)]
        }
    }

    /// Bilinearly-sampled absolute world height at world `(x, z)`, or `None` when
    /// no authored tile contains the point. Exact at sample points (including a
    /// single tile's far edge, which resolves back onto that tile).
    pub fn height_at(&self, world_xz: DVec2) -> Option<f64> {
        let res = self.tile_resolution;
        let xs = self.axis_candidates(world_xz.x);
        let zs = self.axis_candidates(world_xz.y);
        // Prefer the floored tile; fall back to the shared-edge neighbour.
        for &(tx, u) in xs.iter().take(if xs[0].0 == xs[1].0 { 1 } else { 2 }) {
            for &(tz, v) in zs.iter().take(if zs[0].0 == zs[1].0 { 1 } else { 2 }) {
                if let Some(tile) = self.tiles.get(&(tx, tz)) {
                    let u = u.clamp(0.0, (res - 1) as f64);
                    let v = v.clamp(0.0, (res - 1) as f64);
                    let i0 = u.floor() as u32;
                    let j0 = v.floor() as u32;
                    let i1 = (i0 + 1).min(res - 1);
                    let j1 = (j0 + 1).min(res - 1);
                    let fx = u - i0 as f64;
                    let fz = v - j0 as f64;
                    let h00 = tile.sample(res, i0, j0) as f64;
                    let h10 = tile.sample(res, i1, j0) as f64;
                    let h01 = tile.sample(res, i0, j1) as f64;
                    let h11 = tile.sample(res, i1, j1) as f64;
                    let hx0 = h00 + (h10 - h00) * fx;
                    let hx1 = h01 + (h11 - h01) * fx;
                    return Some(tile.origin.y + hx0 + (hx1 - hx0) * fz);
                }
            }
        }
        None
    }

    /// Surface normal (unit, +Y up) at world `(x, z)` via central differences, or
    /// `None` when the containing tile is unauthored. Missing neighbours (terrain
    /// edge) fall back to the centre height (a one-sided slope).
    pub fn normal_at(&self, world_xz: DVec2) -> Option<DVec3> {
        let e = self.meters_per_sample;
        let c = self.height_at(world_xz)?;
        let hl = self.height_at(world_xz - DVec2::new(e, 0.0)).unwrap_or(c);
        let hr = self.height_at(world_xz + DVec2::new(e, 0.0)).unwrap_or(c);
        let hd = self.height_at(world_xz - DVec2::new(0.0, e)).unwrap_or(c);
        let hu = self.height_at(world_xz + DVec2::new(0.0, e)).unwrap_or(c);
        let dhdx = (hr - hl) / (2.0 * e);
        let dhdz = (hu - hd) / (2.0 * e);
        Some(DVec3::new(-dhdx, 1.0, -dhdz).normalize())
    }
}

// ── residency, versions, dirty tracking (P16.3) ─────────────────────────────

impl TerrainData {
    /// Stamp `key` with the next version — the single monotone allocator every
    /// residency and mutation path draws from ([`NEXT_TILE_VERSION`], global to
    /// the process so stamps are unique across terrains, not just within one).
    #[inline]
    fn stamp(&mut self, key: TileKey) {
        let v = NEXT_TILE_VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        self.versions.insert(key, v);
    }

    /// Stamp `key` **and** mark it dirty — the funnel every *mutation* goes
    /// through (streaming a page in stamps without dirtying).
    #[inline]
    fn touch(&mut self, key: TileKey) {
        self.stamp(key);
        self.dirty.insert(key);
    }

    /// A resident tile's change stamp; **`0` for a tile that is not resident**
    /// (never loaded, evicted, or deleted).
    ///
    /// Stamps come from one **process-global** counter ([`NEXT_TILE_VERSION`])
    /// that only ever increases, so:
    ///
    /// * a stamp is strictly greater than every stamp handed out before it —
    ///   including any this tile held in an earlier residency, so a re-paged tile
    ///   can never be mistaken for the contents a GPU cache still holds;
    /// * a stamp is unique across **every** `TerrainData` in the process, so a
    ///   key-addressed cache cannot serve one terrain's tile as another's after a
    ///   level switch (see [`NEXT_TILE_VERSION`] for the failure it prevents);
    /// * `0` is never a live stamp, so a consumer that treats "0 or changed" as
    ///   "re-upload" is conservatively correct for a non-resident tile;
    /// * the ledger is **bounded by residency** — evicting or deleting a tile
    ///   drops its entry, so paging across a planet costs no permanent memory.
    ///
    /// Stamps are therefore **not reproducible across runs** — they are runtime
    /// cache identity, never content. Nothing serialized, hashed, or compared for
    /// equality may read one (`PartialEq` for `TerrainData` deliberately ignores
    /// them).
    #[inline]
    pub fn tile_version(&self, key: TileKey) -> u64 {
        self.versions.get(&key).copied().unwrap_or(0)
    }

    /// How many tiles currently hold a change stamp.
    ///
    /// The bounded-ledger invariant made observable: this tracks the resident set,
    /// never the set of tiles ever streamed. A residency test asserts it, and a
    /// streaming budget can read it without walking the map.
    pub fn version_ledger_len(&self) -> usize {
        self.versions.len()
    }

    /// Whether `key` is currently resident (level 0 or coarse).
    pub fn is_resident(&self, key: TileKey) -> bool {
        if key.is_lod0() {
            self.tiles.contains_key(&key.coord)
        } else {
            self.coarse.contains_key(&key)
        }
    }

    /// A resident tile at any level.
    pub fn resident_tile(&self, key: TileKey) -> Option<&TerrainTile> {
        if key.is_lod0() {
            self.tiles.get(&key.coord)
        } else {
            self.coarse.get(&key)
        }
    }

    /// Every resident key across all levels, ascending (level-0 first).
    pub fn resident_keys(&self) -> Vec<TileKey> {
        self.tiles
            .keys()
            .map(|&c| TileKey::lod0(c))
            .chain(self.coarse.keys().copied())
            .collect()
    }

    /// Number of resident coarse (LOD ≥ 1) tiles. Level-0 residency is
    /// [`tile_count`](Self::tile_count).
    pub fn coarse_tile_count(&self) -> usize {
        self.coarse.len()
    }

    /// Iterate resident coarse tiles in ascending key order.
    pub fn coarse_tiles(&self) -> impl Iterator<Item = (&TileKey, &TerrainTile)> {
        self.coarse.iter()
    }

    /// Make `key` resident from `store`.
    ///
    /// Returns `Ok(true)` when the tile is resident afterwards. An **already
    /// resident** tile is left untouched (never re-read over live edits) and
    /// reports `true`; a tile the store cannot serve reports `false`. `Err` only
    /// for a corrupt blob — the tile then stays non-resident.
    pub fn request_tile<S: TileStore + ?Sized>(
        &mut self,
        key: TileKey,
        store: &S,
    ) -> Result<bool, String> {
        if self.is_resident(key) {
            return Ok(true);
        }
        let Some(tile) = store.load_tile(key)? else {
            return Ok(false);
        };
        self.insert_resident(key, tile);
        Ok(true)
    }

    /// Make an **already-decoded** tile resident — the seam a streamer that
    /// decodes off the main thread applies its batch through (P16.3b2).
    ///
    /// Identical in every respect to what [`request_tile`](Self::request_tile)
    /// does after its own decode: the tile is stamped (a consumer must see it as
    /// changed) but **not** marked dirty (paging a page in is not an edit, so it
    /// schedules no write-back), and a level-0 tile's horizontal origin is
    /// re-derived from its coordinate so it stays grid-aligned.
    pub fn insert_resident_tile(&mut self, key: TileKey, tile: TerrainTile) {
        self.insert_resident(key, tile);
    }

    /// Insert a tile as resident **without** marking it dirty: streaming a page in
    /// is not an edit, so it must not schedule a write-back. Its version still
    /// bumps (the contents changed from the consumer's point of view).
    fn insert_resident(&mut self, key: TileKey, mut tile: TerrainTile) {
        self.stamp(key);
        if key.is_lod0() {
            let o = self.tile_origin_xz(key.coord);
            tile.origin.x = o.x;
            tile.origin.z = o.y;
            self.tiles.insert(key.coord, tile);
        } else {
            self.coarse.insert(key, tile);
        }
    }

    /// Drop `key` from residency, returning whether it was resident.
    ///
    /// A pure paging operation: it does **not** mark anything dirty (there is
    /// nothing to write back that was not already marked). Evicting a tile with
    /// unsaved edits silently loses them — [`sync_residency`](Self::sync_residency)
    /// is the safe door, which refuses to.
    pub fn evict_tile(&mut self, key: TileKey) -> bool {
        let had = if key.is_lod0() {
            self.tiles.remove(&key.coord).is_some()
        } else {
            self.coarse.remove(&key).is_some()
        };
        if had {
            // Bounded ledger: a non-resident tile has no stamp (see
            // [`tile_version`](Self::tile_version)). Reloading it draws a fresh,
            // strictly greater one, so pruning here costs no correctness.
            self.versions.remove(&key);
        }
        had
    }

    /// Make every key in `wants` resident, **evicting nothing** — the additive
    /// half of [`sync_residency`](Self::sync_residency).
    ///
    /// The editor's edit working set (P16.4b) pages this way rather than through
    /// `sync_residency`: a brush footprint is a *different* set on every dab, so
    /// reconciling against it would evict the tiles the previous dab just edited
    /// and, worse, would leave the tiles an undo step has to write `before` into
    /// non-resident — where `revert_delta` would recreate them **flat** and lose
    /// every sample outside the recorded patch. Growing the set and never
    /// shrinking it is what makes undo across a stroke exact; the working set is
    /// released wholesale when the document closes.
    ///
    /// Reports only `loaded` / `missing` / `failed` (nothing is ever evicted or
    /// retained), and an already-resident tile is never re-read over live edits.
    pub fn request_tiles<S: TileStore + ?Sized>(
        &mut self,
        wants: &BTreeSet<TileKey>,
        store: &S,
    ) -> ResidencyReport {
        let mut report = ResidencyReport::default();
        for &key in wants {
            if self.is_resident(key) {
                continue;
            }
            match store.load_tile(key) {
                Ok(Some(tile)) => {
                    self.insert_resident(key, tile);
                    report.loaded.push(key);
                }
                Ok(None) => report.missing.push(key),
                Err(e) => report.failed.push((key, e)),
            }
        }
        report
    }

    /// Reconcile residency against an explicit set of wanted `(coord, lod)` keys.
    ///
    /// Loads every want the store can serve, drops every resident tile that is no
    /// longer wanted — **except** dirty ones, which are retained and reported so
    /// their edits can be written back first. There is no camera and no budget
    /// policy here by design (see [`crate::residency`]): the caller computes the
    /// wants, this executes them.
    pub fn sync_residency<S: TileStore + ?Sized>(
        &mut self,
        wants: &BTreeSet<TileKey>,
        store: &S,
    ) -> ResidencyReport {
        let mut report = ResidencyReport::default();

        // Evict first, so a sliding window frees its trailing pages before paging
        // in the leading ones (peak residency stays at the window size).
        for key in self.resident_keys() {
            if wants.contains(&key) {
                continue;
            }
            if self.dirty.contains(&key) {
                report.retained_dirty.push(key);
                continue;
            }
            if self.evict_tile(key) {
                report.evicted.push(key);
            }
        }

        for &key in wants {
            if self.is_resident(key) {
                continue;
            }
            match store.load_tile(key) {
                Ok(Some(tile)) => {
                    self.insert_resident(key, tile);
                    report.loaded.push(key);
                }
                Ok(None) => report.missing.push(key),
                Err(e) => report.failed.push((key, e)),
            }
        }
        report
    }

    /// Tiles mutated since the last drain, ascending.
    pub fn dirty_tiles(&self) -> Vec<TileKey> {
        self.dirty.iter().copied().collect()
    }

    /// Whether any tile awaits write-back.
    pub fn has_dirty_tiles(&self) -> bool {
        !self.dirty.is_empty()
    }

    /// Take the dirty set, leaving it empty — the write-back seam. Exactly the
    /// tiles mutated since the previous drain, ascending, with no duplicates.
    pub fn drain_dirty(&mut self) -> Vec<TileKey> {
        std::mem::take(&mut self.dirty).into_iter().collect()
    }

    /// Clear the dirty set **without** reading it (a fresh load is not an edit).
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// Clear `key`'s write-back mark **only if** its stamp still equals
    /// `version` — the concurrent-edit guard for a write-back that ran with the
    /// document unlocked.
    ///
    /// Returns whether the mark was cleared. A save stages the dirty tiles, drops
    /// the document lock for a multi-second whole-payload rewrite, and re-locks to
    /// clear the marks; anything the user sculpted in that window has a *newer*
    /// stamp than the bytes that actually reached disk, so clearing it would drop
    /// an edit on the floor. `false` therefore means "still dirty, deliberately" —
    /// the next save writes it.
    ///
    /// A deleted tile is handled by the same rule: it holds no stamp
    /// ([`tile_version`](Self::tile_version) reports `0`), so staging records `0`
    /// and a re-creation in the write window bumps it to something non-zero.
    pub fn clear_dirty_if_unchanged(&mut self, key: TileKey, version: u64) -> bool {
        if self.tile_version(key) != version {
            return false;
        }
        self.dirty.remove(&key)
    }
}

impl HeightSource for TerrainData {
    fn height(&self, x: f64, z: f64) -> Option<f64> {
        self.height_at(DVec2::new(x, z))
    }

    fn normal(&self, x: f64, z: f64) -> Option<DVec3> {
        self.normal_at(DVec2::new(x, z))
    }
}

// ── serde: portable flat form + length validation ───────────────────────────

/// Wire form: config scalars + a flat `(x, z, tile)` sequence (a native
/// `BTreeMap` with tuple keys doesn't encode in JSON; the flat sequence is
/// portable across bincode/JSON and deterministic via `BTreeMap` iteration).
#[derive(Serialize, Deserialize)]
struct TerrainDataRaw {
    tile_resolution: u32,
    meters_per_sample: f64,
    #[serde(default)]
    tiles: Vec<(i32, i32, TerrainTile)>,
}

impl Serialize for TerrainData {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let raw = TerrainDataRaw {
            tile_resolution: self.tile_resolution,
            meters_per_sample: self.meters_per_sample,
            tiles: self
                .tiles
                .iter()
                .map(|(&(x, z), t)| (x, z, t.clone()))
                .collect(),
        };
        raw.serialize(s)
    }
}

impl<'de> Deserialize<'de> for TerrainData {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = TerrainDataRaw::deserialize(d)?;
        let mut data = TerrainData::new(raw.tile_resolution, raw.meters_per_sample);
        let expect = (data.tile_resolution * data.tile_resolution) as usize;
        for (x, z, tile) in raw.tiles {
            if tile.heights().len() != expect {
                return Err(serde::de::Error::custom(format!(
                    "terrain tile ({x},{z}) has {} samples, expected {expect}",
                    tile.heights().len()
                )));
            }
            // Splat weights are stored sparsely: absent (len 0) means the uniform
            // default (layer 0). When present they must match the height
            // resolution exactly, so a corrupt/mismatched weight buffer is caught
            // on load like a bad height buffer.
            let wl = tile.weights_len();
            if wl != 0 && wl != expect {
                return Err(serde::de::Error::custom(format!(
                    "terrain tile ({x},{z}) has {wl} weight samples, expected {expect} or 0"
                )));
            }
            data.tiles.insert((x, z), tile);
            // A **resident tile always holds a stamp** (P16.3): a freshly loaded
            // tile is resident, so it draws one here exactly like a streamed-in or
            // authored one. Without this a deserialized terrain would report
            // version `0` for every tile — "no stamp" — and a GPU tile cache,
            // which conservatively re-uploads an unstamped page, would re-upload
            // the entire heightfield every single frame. Loading is not an edit,
            // so it stamps but does **not** dirty (no write-back is scheduled),
            // and the serialized bytes are untouched.
            data.stamp(TileKey::lod0((x, z)));
        }
        Ok(data)
    }
}
