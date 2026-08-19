//! Tile **residency** (P16.3): which tiles are in memory right now, and where the
//! cold ones come from.
//!
//! A planet-scale terrain cannot hold every tile at once, so [`TerrainData`] stops
//! being "the terrain" and becomes "the *resident* terrain": a working set that a
//! caller grows and shrinks against a cold store.
//!
//! # The contract
//!
//! * **[`TileStore`]** is the cold side — anything that can hand back a tile's
//!   canonical blob bytes for a [`TileKey`]. Two implementations ship here: an
//!   in-memory [`MemoryTileStore`] (tests, an editor scratch buffer) and the
//!   asset-backed [`TerrainAssetReader`], which
//!   slices the bytes out of a `.inf_terrain` payload — a pack mapping included —
//!   with no decode and no copy.
//! * **The caller computes the wants.** `TerrainData::sync_residency` takes an
//!   explicit set of wanted `(coord, lod)` pairs. There is deliberately **no
//!   camera, no radius and no budget policy in Ring 0**: the viewport host and the
//!   player each own their own view-dependent selection, and this layer only
//!   executes it. That keeps residency testable as a pure set operation.
//! * **Absent ≡ non-resident.** A query against a tile that is not resident
//!   returns `None`, exactly like a tile that was never authored — `height_at`
//!   and `normal_at` are untouched. Streaming can therefore never *change* an
//!   answer, only make one available.
//! * **Per-tile versions.** Every mutation stamps the tile it touched from a
//!   single **process-global** monotone counter (a sculpt or paint stroke stamps
//!   only the tiles under the brush), so a GPU tile cache re-uploads a page when —
//!   and only when — its contents changed. The counter never decreases, so a
//!   re-paged tile's stamp is strictly greater than any it held before and a stale
//!   cache entry can never be mistaken for a fresh one. It is global rather than
//!   per-terrain so that two *different* terrains can never mint the same stamp
//!   for the same tile coordinate — a key-addressed cache would otherwise serve
//!   level A's tiles after a switch to level B (see `NEXT_TILE_VERSION`). The
//!   *ledger* is pruned on evict, so it stays bounded by residency rather than by
//!   everything ever streamed.
//! * **Dirty set.** The same mutations mark the tile dirty; a caller
//!   [`drains`](TerrainData::drain_dirty) the set to write edits back to the cold
//!   store. `sync_residency` refuses to evict a dirty tile — unsaved authoring is
//!   never dropped on the floor.

use std::collections::{BTreeMap, BTreeSet};

use crate::asset::{decode_tile_at, TerrainAssetReader, TERRAIN_ASSET_SCHEMA_VERSION};
use crate::data::TerrainData;
use crate::tile::{TerrainTile, TileKey};

/// The cold side of residency: a source of tile blobs keyed by [`TileKey`].
///
/// Implementors hand back the tile's **canonical bincode blob** (the same bytes a
/// tile writes into a `.inf_lvl` or a `.inf_terrain`), borrowed for as long as the
/// store — so an asset-backed store can serve straight out of an mmap'd pack.
pub trait TileStore {
    /// The tile's blob bytes, or `None` when the store has no such tile.
    fn tile_bytes(&self, key: TileKey) -> Option<&[u8]>;

    /// Every key the store can serve, in ascending [`TileKey`] order.
    fn tile_keys(&self) -> Vec<TileKey>;

    /// Whether the store can serve `key`.
    fn contains_tile(&self, key: TileKey) -> bool {
        self.tile_bytes(key).is_some()
    }

    /// The `.inf_terrain` schema version the store's blobs were **written at** —
    /// which is what says how to decode them (bincode is positional; see
    /// [`decode_tile_at`]).
    ///
    /// Defaults to the current version, which is right for every store that
    /// encodes its own blobs (a [`MemoryTileStore`] staged from live tiles). An
    /// asset-backed store overrides it with its payload's own stamp so a v1/v2
    /// `.inf_terrain` still pages in.
    fn tile_schema_version(&self) -> u32 {
        TERRAIN_ASSET_SCHEMA_VERSION
    }

    /// Decode a tile, or `None` when absent. `Err` only for a corrupt blob.
    fn load_tile(&self, key: TileKey) -> Result<Option<TerrainTile>, String> {
        match self.tile_bytes(key) {
            None => Ok(None),
            Some(bytes) => decode_tile_at(bytes, self.tile_schema_version()).map(Some),
        }
    }
}

/// An in-memory [`TileStore`] over pre-encoded blobs.
///
/// The simple cold store: an editor scratch buffer, a test fixture, or the
/// write-back staging area a dirty drain flushes into.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryTileStore {
    tiles: BTreeMap<TileKey, Vec<u8>>,
}

impl MemoryTileStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage a tile, encoding it to its canonical blob. Replaces any previous one.
    pub fn insert(&mut self, key: TileKey, tile: &TerrainTile) -> Result<(), String> {
        let bytes = crate::asset::encode_tile(tile).map_err(|e| e.to_string())?;
        self.tiles.insert(key, bytes);
        Ok(())
    }

    /// Stage already-encoded blob bytes.
    pub fn insert_bytes(&mut self, key: TileKey, bytes: Vec<u8>) {
        self.tiles.insert(key, bytes);
    }

    /// Drop a tile.
    pub fn remove(&mut self, key: TileKey) -> Option<Vec<u8>> {
        self.tiles.remove(&key)
    }

    /// Number of staged tiles.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Stage every level-0 tile of `data` (the "spill the whole terrain into a
    /// cold store" helper the editor's write-back path uses).
    pub fn stage_level0(&mut self, data: &TerrainData) -> Result<(), String> {
        for (&coord, tile) in data.tiles() {
            self.insert(TileKey::lod0(coord), tile)?;
        }
        Ok(())
    }
}

impl TileStore for MemoryTileStore {
    fn tile_bytes(&self, key: TileKey) -> Option<&[u8]> {
        self.tiles.get(&key).map(|v| v.as_slice())
    }

    fn tile_keys(&self) -> Vec<TileKey> {
        self.tiles.keys().copied().collect()
    }
}

/// The asset-backed store: a `.inf_terrain` payload served in place.
///
/// `tile_bytes` is a binary search plus a sub-slice of the payload — so when the
/// payload is itself a borrowed `PackReader::read_ref` slice of an mmap, a tile
/// reaches the caller with **zero copies and zero decodes**.
impl<B: AsRef<[u8]>> TileStore for TerrainAssetReader<B> {
    fn tile_bytes(&self, key: TileKey) -> Option<&[u8]> {
        TerrainAssetReader::tile_bytes(self, key)
    }

    fn tile_keys(&self) -> Vec<TileKey> {
        self.keys().collect()
    }

    /// The payload's own stamp — an older asset's blobs hold the older tile
    /// layout, and this is what makes them keep paging in (P19.1).
    fn tile_schema_version(&self) -> u32 {
        self.header().schema_version
    }
}

/// What one [`TerrainData::sync_residency`] pass did.
///
/// Every list is in ascending [`TileKey`] order, so a report is a deterministic
/// function of `(resident set, wants, store)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidencyReport {
    /// Tiles brought in from the store.
    pub loaded: Vec<TileKey>,
    /// Tiles dropped because they were no longer wanted.
    pub evicted: Vec<TileKey>,
    /// Wanted tiles the store could not serve (a coverage hole, not an error).
    pub missing: Vec<TileKey>,
    /// Unwanted tiles that were **kept** because they carry unsaved edits. Drain
    /// the dirty set and write them back, then sync again to release them.
    pub retained_dirty: Vec<TileKey>,
    /// Tiles whose blob failed to decode, with the reason. The tile is left
    /// non-resident; a corrupt page never becomes silent geometry.
    pub failed: Vec<(TileKey, String)>,
}

impl ResidencyReport {
    /// `true` when nothing moved (no load, no evict, no failure).
    pub fn is_noop(&self) -> bool {
        self.loaded.is_empty() && self.evicted.is_empty() && self.failed.is_empty()
    }

    /// Number of tiles that changed residency.
    pub fn churn(&self) -> usize {
        self.loaded.len() + self.evicted.len()
    }
}

/// Convenience: the set of keys a rectangular tile range wants at one LOD — the
/// shape a caller's view-dependent selection usually reduces to. (Still *caller*
/// policy: this is arithmetic, not a camera.)
pub fn tile_range(lod: u32, min: (i32, i32), max: (i32, i32)) -> BTreeSet<TileKey> {
    let (x0, x1) = (min.0.min(max.0), min.0.max(max.0));
    let (z0, z1) = (min.1.min(max.1), min.1.max(max.1));
    let mut out = BTreeSet::new();
    for tz in z0..=z1 {
        for tx in x0..=x1 {
            out.insert(TileKey::new(lod, (tx, tz)));
        }
    }
    out
}

/// **Make a world-space rectangle answerable** — page every level-0 tile that
/// covers `[min, max]` in world XZ into `data`, from `store`.
///
/// # Why this exists (island phase, IB-1)
///
/// Residency is a *view-dependent* policy everywhere else in this engine: the
/// viewport and the player each compute wants from a camera or a streaming
/// source, and Ring 0 only executes them. There is one caller that is neither,
/// and it is the one a 50 km² world depends on most — **procedural generation**.
/// A scatter volume does not look at the terrain from anywhere; it asks for
/// heights over a *known rectangle*, once, and bakes the answer. Before this door
/// it had no way to ask a cold store anything, so a scatter over an asset-backed
/// terrain evaluated against `Some(0.0)`: measured at **929 of 929 instances at
/// exactly sea level**, with a different instance count from the same graph
/// because the slope and height masks were reading a plane.
///
/// # It is a want set, not a policy
///
/// The rectangle is the caller's; the arithmetic is here so that the two hosts
/// cannot each round it differently. Level 0 only, deliberately: PCG places
/// content on the ground an author authored, and a coarse pyramid level is a
/// *rendering* approximation of that ground. Reading LOD 2 would put a tree
/// somewhere no player will ever stand.
///
/// **Additive** ([`TerrainData::request_tiles`]): nothing is evicted, an already
/// resident tile is never re-read over live edits, and a tile the store does not
/// have is reported rather than invented. So calling this over a terrain that is
/// already fully resident — an *authored*, inline-tile terrain — is a no-op, and
/// the authored and streamed paths converge on one answer instead of two.
///
/// The caller is responsible for `data` already having the store's grid geometry
/// (`tile_resolution`, `meters_per_sample`): a `TerrainData` configured
/// differently from the asset it is paging from would place every tile it loaded
/// at the wrong world coordinate. `TerrainData::insert_resident` does not check
/// this and cannot — the store has no opinion about the grid it is being read
/// into.
pub fn page_region<S: TileStore + ?Sized>(
    data: &mut TerrainData,
    store: &S,
    min: glam::DVec2,
    max: glam::DVec2,
) -> ResidencyReport {
    let (min, max) = (min.min(max), min.max(max));
    // Non-finite input is a refusal, not a panic and not a full page-in: the
    // rectangle comes from a `PcgVolume`'s authored extent, `f64::NAN.floor() as
    // i32` is 0 in Rust, and a silent (0,0)..(0,0) would look like a working
    // scatter over one tile. NaN at doors (the standing law).
    if !min.is_finite() || !max.is_finite() {
        return ResidencyReport::default();
    }
    let c0 = data.tile_coord_of(min.x, min.y);
    let c1 = data.tile_coord_of(max.x, max.y);
    let wants = tile_range(0, c0, c1);
    data.request_tiles(&wants, store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::build_terrain_asset;
    use crate::pyramid::{build_pyramid, PyramidOptions};
    use glam::DVec2;

    fn height_fn(x: f64, z: f64) -> f64 {
        x * 0.25 - z * 0.125 + 5.0
    }

    fn terrain_4x4() -> TerrainData {
        let mut t = TerrainData::new(5, 2.0);
        for tz in 0..4 {
            for tx in 0..4 {
                t.author_tile((tx, tz), height_fn);
            }
        }
        t
    }

    #[test]
    fn memory_store_serves_encoded_tiles() {
        let t = terrain_4x4();
        let mut store = MemoryTileStore::new();
        store.stage_level0(&t).unwrap();
        assert_eq!(store.len(), 16);
        assert_eq!(store.tile_keys().len(), 16);
        let key = TileKey::lod0((2, 1));
        assert!(store.contains_tile(key));
        assert_eq!(
            store.load_tile(key).unwrap().unwrap(),
            *t.get_tile((2, 1)).unwrap()
        );
        assert!(store.load_tile(TileKey::lod0((9, 9))).unwrap().is_none());
    }

    #[test]
    fn asset_reader_is_a_tile_store() {
        let t = terrain_4x4();
        let p = build_pyramid(&t, PyramidOptions::default());
        let asset = build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap();
        let reader = asset.reader();
        let store: &dyn TileStore = &reader;
        assert_eq!(store.tile_keys().len(), reader.tile_count());
        assert!(store.contains_tile(TileKey::lod0((0, 0))));
        assert!(store.contains_tile(TileKey::new(1, (0, 0))));
        assert_eq!(
            store.load_tile(TileKey::lod0((3, 3))).unwrap().unwrap(),
            *t.get_tile((3, 3)).unwrap()
        );
    }

    /// Stamps are unique across **`TerrainData` instances**, not merely within
    /// one (P16.3b1). A per-instance counter would hand two freshly loaded
    /// terrains the identical 1, 2, 3, … over the identical tile coordinates, and
    /// a consumer that caches by tile key alone — the renderer's GPU tile cache —
    /// would keep serving the first terrain's contents after a level switch.
    #[test]
    fn distinct_terrains_mint_disjoint_stamps() {
        let t = terrain_4x4();
        let mut store = MemoryTileStore::new();
        store.stage_level0(&t).unwrap();
        let wants = tile_range(0, (0, 0), (1, 1));

        // Two independent terrains page in the SAME tile coordinates.
        let mut a = TerrainData::new(5, 2.0);
        let mut b = TerrainData::new(5, 2.0);
        a.sync_residency(&wants, &store);
        b.sync_residency(&wants, &store);

        let stamps_a: BTreeSet<u64> = wants.iter().map(|&k| a.tile_version(k)).collect();
        let stamps_b: BTreeSet<u64> = wants.iter().map(|&k| b.tile_version(k)).collect();
        assert_eq!(
            stamps_a.len(),
            wants.len(),
            "stamps unique within a terrain"
        );
        assert_eq!(stamps_b.len(), wants.len());
        assert!(
            stamps_a.is_disjoint(&stamps_b),
            "two terrains minted overlapping stamps {stamps_a:?} / {stamps_b:?} — a \
             key-addressed cache would serve stale tiles across a level switch"
        );
        // The later terrain's stamps are strictly greater, tile for tile.
        for &key in &wants {
            assert!(b.tile_version(key) > a.tile_version(key));
        }

        // The same holds for two terrains **deserialized** from identical bytes —
        // the File → Open shape, where every tile coordinate necessarily matches.
        let json = serde_json::to_string(&t).unwrap();
        let c: TerrainData = serde_json::from_str(&json).unwrap();
        let d: TerrainData = serde_json::from_str(&json).unwrap();
        assert_eq!(c, d, "identical authored content …");
        for (&coord, _) in c.tiles() {
            let key = TileKey::lod0(coord);
            assert!(
                d.tile_version(key) > c.tile_version(key) && c.tile_version(key) > 0,
                "… must still carry distinct, live stamps at {coord:?}"
            );
        }
    }

    #[test]
    fn version_ledger_is_bounded_and_strictly_increasing() {
        let t = terrain_4x4();
        let mut store = MemoryTileStore::new();
        store.stage_level0(&t).unwrap();
        let mut live = TerrainData::new(5, 2.0);

        let key = TileKey::lod0((0, 0));
        assert_eq!(live.tile_version(key), 0, "never-loaded ⇒ no stamp");

        // Page a window in and out repeatedly: stamps strictly increase every
        // time, and the ledger never outgrows the resident set.
        let wants = tile_range(0, (0, 0), (1, 1));
        let mut last = 0;
        for _ in 0..4 {
            live.sync_residency(&wants, &store);
            let v = live.tile_version(key);
            assert!(v > last, "reload stamp {v} must exceed the previous {last}");
            last = v;
            assert_eq!(live.version_ledger_len(), 4, "one stamp per resident tile");

            live.sync_residency(&BTreeSet::new(), &store);
            assert_eq!(live.tile_version(key), 0, "evicted ⇒ stamp pruned");
            assert_eq!(live.version_ledger_len(), 0, "ledger bounded by residency");
        }

        // A mutation stamps only the tile under it.
        live.sync_residency(&wants, &store);
        let others: Vec<u64> = [(1, 0), (0, 1), (1, 1)]
            .iter()
            .map(|&c| live.tile_version(TileKey::lod0(c)))
            .collect();
        let before = live.tile_version(key);
        live.get_tile_mut((0, 0)).unwrap().set_sample(5, 1, 1, 7.0);
        assert!(live.tile_version(key) > before, "the edited tile restamps");
        for (&c, &v) in [(1, 0), (0, 1), (1, 1)].iter().zip(&others) {
            assert_eq!(
                live.tile_version(TileKey::lod0(c)),
                v,
                "an untouched tile keeps its stamp"
            );
        }

        // An authoring delete drops the stamp but still schedules a write-back.
        live.remove_tile((1, 0));
        assert_eq!(live.tile_version(TileKey::lod0((1, 0))), 0);
        assert!(live.dirty_tiles().contains(&TileKey::lod0((1, 0))));
    }

    #[test]
    fn evict_then_request_restores_identical_bytes() {
        let t = terrain_4x4();
        let mut store = MemoryTileStore::new();
        store.stage_level0(&t).unwrap();
        let mut live = t.clone();

        let probe = DVec2::new(1.0, 1.0); // inside tile (0,0)
        let before = live.height_at(probe).expect("resident");

        // Evict → the query goes dark exactly like an unauthored tile, and the
        // tile's stamp is pruned (a non-resident tile has no version).
        let resident_stamp = live.tile_version(TileKey::lod0((0, 0)));
        assert!(live.evict_tile(TileKey::lod0((0, 0))));
        assert_eq!(live.tile_version(TileKey::lod0((0, 0))), 0);
        assert!(!live.is_resident(TileKey::lod0((0, 0))));
        assert!(live.height_at(probe).is_none(), "non-resident ≡ absent");
        assert!(live.normal_at(probe).is_none());

        // Re-request → identical tile, identical answer.
        assert!(live.request_tile(TileKey::lod0((0, 0)), &store).unwrap());
        assert!(
            live.tile_version(TileKey::lod0((0, 0))) > resident_stamp,
            "a re-paged tile takes a strictly greater stamp than it ever held"
        );
        assert_eq!(live.get_tile((0, 0)), t.get_tile((0, 0)));
        assert_eq!(live.height_at(probe), Some(before));
        assert_eq!(
            crate::asset::encode_tile(live.get_tile((0, 0)).unwrap()).unwrap(),
            store.tile_bytes(TileKey::lod0((0, 0))).unwrap(),
        );
    }

    #[test]
    fn sync_residency_loads_evicts_and_reports() {
        let t = terrain_4x4();
        let mut store = MemoryTileStore::new();
        store.stage_level0(&t).unwrap();
        let mut live = TerrainData::new(5, 2.0);

        let wants = tile_range(0, (0, 0), (1, 1));
        let r = live.sync_residency(&wants, &store);
        assert_eq!(r.loaded.len(), 4);
        assert!(r.evicted.is_empty() && r.missing.is_empty());
        assert_eq!(live.tile_count(), 4);

        // Idempotent: syncing the same wants moves nothing.
        let r = live.sync_residency(&wants, &store);
        assert!(r.is_noop(), "{r:?}");

        // Slide the window: two columns in, two out.
        let next = tile_range(0, (1, 0), (2, 1));
        let r = live.sync_residency(&next, &store);
        assert_eq!(r.loaded, vec![TileKey::lod0((2, 0)), TileKey::lod0((2, 1))]);
        assert_eq!(
            r.evicted,
            vec![TileKey::lod0((0, 0)), TileKey::lod0((0, 1))]
        );
        assert_eq!(r.churn(), 4);
        assert_eq!(
            live.resident_keys(),
            next.iter().copied().collect::<Vec<_>>()
        );

        // A want the store cannot serve is reported, not an error.
        let mut hole = next.clone();
        hole.insert(TileKey::lod0((40, 40)));
        let r = live.sync_residency(&hole, &store);
        assert_eq!(r.missing, vec![TileKey::lod0((40, 40))]);
        assert!(!live.is_resident(TileKey::lod0((40, 40))));
    }

    #[test]
    fn sync_residency_never_evicts_a_dirty_tile() {
        let t = terrain_4x4();
        let mut store = MemoryTileStore::new();
        store.stage_level0(&t).unwrap();
        let mut live = TerrainData::new(5, 2.0);
        live.sync_residency(&tile_range(0, (0, 0), (1, 1)), &store);

        // Sculpt one tile → dirty.
        live.get_tile_mut((0, 0)).unwrap().set_sample(5, 1, 1, 99.0);
        assert!(live.dirty_tiles().contains(&TileKey::lod0((0, 0))));

        // Want nothing: the clean tiles go, the dirty one stays.
        let r = live.sync_residency(&BTreeSet::new(), &store);
        assert_eq!(r.retained_dirty, vec![TileKey::lod0((0, 0))]);
        assert_eq!(r.evicted.len(), 3);
        assert!(live.is_resident(TileKey::lod0((0, 0))));
        assert_eq!(live.get_tile((0, 0)).unwrap().sample(5, 1, 1), 99.0);

        // Write back + drain, then it releases.
        let drained = live.drain_dirty();
        assert_eq!(drained, vec![TileKey::lod0((0, 0))]);
        for key in drained {
            store
                .insert(key, live.get_tile(key.coord).unwrap())
                .unwrap();
        }
        let r = live.sync_residency(&BTreeSet::new(), &store);
        assert_eq!(r.evicted, vec![TileKey::lod0((0, 0))]);
        assert!(live.is_empty());
        // The written-back edit survives the round trip.
        assert_eq!(
            store
                .load_tile(TileKey::lod0((0, 0)))
                .unwrap()
                .unwrap()
                .sample(5, 1, 1),
            99.0
        );
    }

    #[test]
    fn coarse_levels_are_resident_beside_level_zero() {
        let t = terrain_4x4();
        let p = build_pyramid(&t, PyramidOptions::default());
        let asset = build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap();
        let reader = asset.reader();

        let mut live = TerrainData::new(5, 2.0);
        let mut wants = tile_range(0, (0, 0), (0, 0));
        wants.extend(tile_range(1, (0, 0), (1, 1)));
        let r = live.sync_residency(&wants, &reader);
        assert_eq!(r.loaded.len(), 5);

        // Level-0 residency is the authored heightfield …
        assert_eq!(live.tile_count(), 1, "tile_count stays level-0 semantics");
        assert!(live.height_at(DVec2::new(1.0, 1.0)).is_some());
        // … coarse tiles live beside it and never affect a height query.
        assert_eq!(live.coarse_tile_count(), 4);
        let coarse = live.resident_tile(TileKey::new(1, (0, 0))).unwrap();
        assert_eq!(coarse, p[0].tiles.get(&(0, 0)).unwrap());
        assert!(live.height_at(DVec2::new(1000.0, 1000.0)).is_none());
    }

    #[test]
    fn tile_range_is_inclusive_and_normalizes() {
        let a = tile_range(2, (1, -1), (2, 0));
        assert_eq!(a.len(), 4);
        assert!(a.contains(&TileKey::new(2, (1, -1))));
        assert!(a.contains(&TileKey::new(2, (2, 0))));
        // Reversed bounds describe the same rectangle.
        assert_eq!(tile_range(2, (2, 0), (1, -1)), a);
    }
}
