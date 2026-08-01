//! The **streamer**: the stateful half of camera-driven tile residency
//! (P16.3b2) — cold stores over a cooked pack or a loose file, plus the
//! [`TerrainStreamer`] that drives a render-resident working set toward the cut
//! [`wants`](crate::wants) computes.
//!
//! # THE DETERMINISM DOCTRINE (restated where it is implemented)
//!
//! The fixed-step sim's results must **never** depend on camera-driven
//! residency. Two drivers, structurally separated:
//!
//! * **[`sync_sim`](TerrainStreamer::sync_sim)** — deterministic. Takes the
//!   caller's *sim* wants (level-0 only, computed from sim state by
//!   [`sim_wants`](crate::wants::sim_wants)), loads them **synchronously in
//!   sorted order** into the [`TerrainData`] the *simulation* reads — the one
//!   living in the ECS `Terrain` component. Called at the fixed-step boundary,
//!   **before** the step runs.
//! * **[`sync_render`](TerrainStreamer::sync_render)** — free. Camera-driven,
//!   loaded on the [`inf_core`] job pool, applied at the caller's render-sync
//!   point into [`render_data`](TerrainStreamer::render_data) — a **second,
//!   private** [`TerrainData`] that the ECS world holds no reference to.
//!
//! That last sentence is the enforcement: it is not a convention a future edit
//! can quietly break, it is the absence of a path. `terrain.height_at` reaches
//! the component's data; the component's data is only ever grown by `sync_sim`;
//! `sync_sim` only ever sees sim state. A camera cannot change a height query
//! because there is no reference from the world to the set the camera fills.
//! Tiles wanted by both drivers are held twice — bounded (sim wants are a small
//! level-0 neighbourhood) and paid on purpose.
//!
//! # Why the render loads are a *bounded deterministic batch*
//!
//! Loads run on the job pool, but the batch's **membership is a pure function of
//! the want set**, never of what happened to finish in time: each
//! [`sync_render`] advances the published cut by at most
//! [`StreamBudget::max_loads_per_sync`] not-yet-resident pages
//! ([`advance_cut`](crate::wants::advance_cut)), and every intermediate state is
//! itself a valid quadtree cut. That buys the two things timing-driven
//! asynchrony was wanted for — no whole-cut hitch, work spread across frames —
//! while keeping the property the replay / PIE-==-shipping gates need: a scripted
//! camera path reproduces a byte-identical resident-set trace run to run.
//! (`inf_core`'s pool exposes no `spawn`, only fork-join primitives, so
//! frame-straddling loads would have needed a bespoke thread — which would have
//! traded that determinism away for nothing this batch needs.)

use std::collections::BTreeSet;

use glam::DVec2;
use inf_asset::{AssetId, PackReader};

use crate::asset::{TerrainAssetError, TerrainAssetHeader, TerrainAssetReader, TileDirEntry};
use crate::data::TerrainData;
use crate::residency::{ResidencyReport, TileStore};
use crate::tile::{TerrainTile, TileKey};
use crate::wants::{
    advance_cut, clamp_cut, render_wants, RenderWantsParams, TileCatalog, TileGrid, TileIndex,
};

// ── cold stores ─────────────────────────────────────────────────────────────

/// A [`TileStore`] over a `.inf_terrain` entry of a **cooked pack**.
///
/// The header + tile directory are parsed once at construction; every
/// [`tile_bytes`](TileStore::tile_bytes) after that is a binary search plus a
/// sub-slice of [`PackReader::read_ref`]'s borrowed mapping — **no decode, no
/// copy, no allocation**, which is the entire reason terrain cooks uncompressed
/// ([`PackWriter::compresses_kind`](inf_asset::PackWriter::compresses_kind)).
///
/// A pack that nonetheless stores the entry compressed (an older cook, or a
/// hand-built pack) still works: the payload is decoded **once** at construction
/// into an owned buffer and served from there. Slower to open, identical
/// afterwards — the fallback exists so a stale pack degrades in speed, never in
/// correctness.
pub struct PackTileStore {
    backing: Backing,
    header: TerrainAssetHeader,
    dir: Vec<TileDirEntry>,
}

enum Backing {
    /// Zero-copy: slice straight out of the pack mapping on every fetch.
    Pack(std::sync::Arc<PackReader>, AssetId),
    /// The compressed-entry fallback: the decoded payload, owned.
    Owned(Vec<u8>),
}

impl PackTileStore {
    /// Open the `.inf_terrain` asset `guid` inside `reader`.
    pub fn open(reader: std::sync::Arc<PackReader>, guid: AssetId) -> Result<Self, String> {
        let payload = reader
            .read_ref(guid)
            .map_err(|e| format!("read terrain asset {guid}: {e}"))?;
        let borrowed = matches!(payload, std::borrow::Cow::Borrowed(_));
        let (header, dir) = {
            let r = TerrainAssetReader::new(payload.as_ref())
                .map_err(|e| format!("terrain asset {guid}: {e}"))?;
            (*r.header(), r.directory().to_vec())
        };
        let backing = if borrowed {
            drop(payload);
            Backing::Pack(reader, guid)
        } else {
            Backing::Owned(payload.into_owned())
        };
        Ok(Self {
            backing,
            header,
            dir,
        })
    }

    /// The asset's decoded header (grid config + level count).
    #[inline]
    pub fn header(&self) -> &TerrainAssetHeader {
        &self.header
    }

    /// The payload bytes, borrowed from whichever backing holds them.
    fn payload(&self) -> Option<&[u8]> {
        match &self.backing {
            // Verified once at construction; a compressed entry took the Owned
            // arm, so this is always the borrowed mapping.
            Backing::Pack(reader, guid) => match reader.read_ref(*guid) {
                Ok(std::borrow::Cow::Borrowed(b)) => Some(b),
                _ => None,
            },
            Backing::Owned(v) => Some(v.as_slice()),
        }
    }
}

impl std::fmt::Debug for PackTileStore {
    /// Summarizes; never dumps the (possibly gigabyte) payload.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackTileStore")
            .field(
                "backing",
                &match &self.backing {
                    Backing::Pack(_, guid) => format!("pack:{guid}"),
                    Backing::Owned(v) => format!("owned:{} bytes", v.len()),
                },
            )
            .field("tile_count", &self.dir.len())
            .field("lod_levels", &self.header.lod_levels)
            .finish()
    }
}

impl TileStore for PackTileStore {
    fn tile_bytes(&self, key: TileKey) -> Option<&[u8]> {
        let i = self.dir.binary_search_by(|e| e.key.cmp(&key)).ok()?;
        let e = &self.dir[i];
        let bytes = self.payload()?;
        // Bounds were validated when the directory was parsed.
        bytes.get(e.offset as usize..(e.offset + e.len) as usize)
    }

    fn tile_keys(&self) -> Vec<TileKey> {
        self.dir.iter().map(|e| e.key).collect()
    }
}

/// A [`TileStore`] over a **loose** `.inf_terrain` file — the editor's cold side.
///
/// The whole payload is read into memory once (a loose asset is being authored,
/// not shipped, so there is no mapping to page out of) and then served exactly
/// like the packed one: binary search plus a sub-slice, no decode per fetch.
pub type FileTileStore = TerrainAssetReader<Vec<u8>>;

/// Open a loose `.inf_terrain` as a [`FileTileStore`].
pub fn open_file_tile_store(path: &std::path::Path) -> std::io::Result<FileTileStore> {
    let asset = crate::asset::read_terrain_asset(path)?;
    TerrainAssetReader::new(asset.into_bytes())
        .map_err(|e: TerrainAssetError| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ── budget + diagnostics ────────────────────────────────────────────────────

/// The residency budget a [`TerrainStreamer`] respects.
///
/// **Render wants are clamped; sim wants never are.** A render page short of
/// budget costs detail; a sim page short of budget changes the simulation, so
/// correctness wins and the sim's working set is allowed to exceed the ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamBudget {
    /// Ceiling on **render-resident** tiles (`0` = unlimited). Over it, the
    /// target cut is coarsened farthest-first
    /// ([`clamp_cut`](crate::wants::clamp_cut)).
    pub max_resident_tiles: usize,
    /// Ceiling on how many not-yet-resident pages one
    /// [`sync_render`](TerrainStreamer::sync_render) may bring in (`0` =
    /// unlimited). Bounds the per-frame load cost without ever publishing a
    /// partial sibling set.
    pub max_loads_per_sync: usize,
}

impl Default for StreamBudget {
    /// 1024 resident pages (≈ 268 MB of `f32` heights at 256², ≈ 1 MB at 16²) and
    /// 16 loads per sync.
    fn default() -> Self {
        Self {
            max_resident_tiles: 1024,
            max_loads_per_sync: 16,
        }
    }
}

/// Streaming counters for the debug/stats path.
///
/// Ring 0 so the player and the editor viewport can both name one type; plain
/// numbers so a caller can log [`summary`](Self::summary), fold it into a
/// diagnostics report, or ratchet it in CI.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TerrainStreamStats {
    /// Render-resident level-0 pages.
    pub resident_level0: usize,
    /// Render-resident coarse (LOD ≥ 1) pages.
    pub resident_coarse: usize,
    /// Level-0 pages resident for the **sim** (the ECS component's set).
    pub sim_resident_level0: usize,
    /// Wanted render pages not yet resident (the convergence backlog).
    pub pending_loads: usize,
    /// Pages loaded since the streamer was created (render + sim).
    pub loads: u64,
    /// Pages evicted since the streamer was created (render + sim).
    pub evictions: u64,
    /// Pages whose blob failed to decode (never silently becomes geometry).
    pub failed: u64,
    /// Bytes of resident tile data (heights + materialized splat weights), both
    /// working sets.
    pub bytes_resident: u64,
    /// Whether the last render sync had to clamp its cut to the tile budget.
    pub budget_clamped: bool,
}

impl TerrainStreamStats {
    /// A one-line human summary for the Output Log / a `stats` dump.
    pub fn summary(&self) -> String {
        format!(
            "terrain streaming: {} resident ({} L0 + {} coarse), sim {} L0, {} pending, \
             {} loads / {} evictions / {} failed, {:.1} MiB{}",
            self.resident_level0 + self.resident_coarse,
            self.resident_level0,
            self.resident_coarse,
            self.sim_resident_level0,
            self.pending_loads,
            self.loads,
            self.evictions,
            self.failed,
            self.bytes_resident as f64 / (1024.0 * 1024.0),
            if self.budget_clamped {
                " [budget-clamped]"
            } else {
                ""
            }
        )
    }

    /// Fold another terrain's counters in (a scene with several streamed
    /// terrains reports one total).
    pub fn merge(&mut self, other: &Self) {
        self.resident_level0 += other.resident_level0;
        self.resident_coarse += other.resident_coarse;
        self.sim_resident_level0 += other.sim_resident_level0;
        self.pending_loads += other.pending_loads;
        self.loads += other.loads;
        self.evictions += other.evictions;
        self.failed += other.failed;
        self.bytes_resident += other.bytes_resident;
        self.budget_clamped |= other.budget_clamped;
    }
}

/// Bytes of tile data resident in `data` (heights + any materialized weights).
fn resident_bytes(data: &TerrainData) -> u64 {
    let one = |t: &crate::tile::TerrainTile| {
        (std::mem::size_of_val(t.heights()) + t.weights_len() * 4) as u64
    };
    data.tiles().map(|(_, t)| one(t)).sum::<u64>()
        + data.coarse_tiles().map(|(_, t)| one(t)).sum::<u64>()
}

// ── the streamer ────────────────────────────────────────────────────────────

/// Drives one terrain's residency: a private render-resident working set plus the
/// published quadtree cut, and the synchronous sim-side sync (see the module
/// docs for why those are two different `TerrainData`s).
pub struct TerrainStreamer {
    grid: TileGrid,
    catalog: TileCatalog,
    params: RenderWantsParams,
    budget: StreamBudget,
    /// The **render**-resident working set. Deliberately private and outside the
    /// ECS world — the sim has no path to it.
    render: TerrainData,
    /// The currently published cut (also the hysteresis state).
    cut: BTreeSet<TileKey>,
    /// The last target cut, so `pending_loads` reports the real backlog.
    target: BTreeSet<TileKey>,
    /// Bytes held by the **sim's** working set, sampled at the last
    /// [`sync_sim`](Self::sync_sim) — the render sync folds it into
    /// `bytes_resident` without needing the component.
    sim_bytes: u64,
    /// Keys whose blob failed to decode, **excluded from every subsequent want**
    /// until [`clear_failed`](Self::clear_failed) — see there for why.
    blocked: BTreeSet<TileKey>,
    /// Keys the editor has pinned into the render set because they carry unsaved
    /// edits ([`pin_tile`](Self::pin_tile)) — never evicted, never re-read from
    /// the store. Empty in the shipped player, which never edits anything.
    pinned: BTreeSet<TileKey>,
    stats: TerrainStreamStats,
}

/// A [`TileIndex`] view that hides keys which failed to decode.
///
/// A corrupt page is then indistinguishable, to the cut rules, from a page that
/// was never authored: its parent simply keeps covering that ground instead of
/// subdividing into a hole. That is both the correct visual outcome and the thing
/// that stops the streamer re-requesting it forever.
struct BlockedIndex<'a> {
    inner: &'a TileCatalog,
    blocked: &'a BTreeSet<TileKey>,
}

impl TileIndex for BlockedIndex<'_> {
    fn has(&self, key: TileKey) -> bool {
        self.inner.has(key) && !self.blocked.contains(&key)
    }

    fn max_lod(&self) -> u32 {
        self.inner.max_lod()
    }

    fn keys_at_lod(&self, lod: u32) -> Vec<TileKey> {
        let mut keys = self.inner.keys_at_lod(lod);
        keys.retain(|k| !self.blocked.contains(k));
        keys
    }
}

impl TerrainStreamer {
    /// A streamer over `store`'s tiles.
    ///
    /// The catalog (which keys exist, and the coarsest level) is read once here;
    /// the grid comes from the caller's level-0 configuration, which must match
    /// the asset's — [`from_store`](Self::from_store) takes both from the store.
    pub fn new(
        grid: TileGrid,
        catalog: TileCatalog,
        params: RenderWantsParams,
        budget: StreamBudget,
        tile_resolution: u32,
        meters_per_sample: f64,
    ) -> Self {
        // Seed the published cut with the pyramid's **coarsest** level: it is
        // small by construction (`build_pyramid` stops at `min_tiles`), covers the
        // whole terrain, and gives the first frame something to draw while the
        // finer levels converge.
        let cut: BTreeSet<TileKey> = catalog.keys_at_lod(catalog.max_lod()).into_iter().collect();
        Self {
            grid,
            catalog,
            params,
            budget,
            render: TerrainData::new(tile_resolution, meters_per_sample),
            target: cut.clone(),
            cut,
            sim_bytes: 0,
            blocked: BTreeSet::new(),
            pinned: BTreeSet::new(),
            stats: TerrainStreamStats::default(),
        }
    }

    /// A streamer configured entirely from a store's own header/catalog.
    pub fn from_store(
        store: &dyn TileStore,
        tile_resolution: u32,
        meters_per_sample: f64,
        params: RenderWantsParams,
        budget: StreamBudget,
    ) -> Self {
        let grid = TileGrid::new(tile_resolution, meters_per_sample);
        Self::new(
            grid,
            TileCatalog::from_store(store),
            params,
            budget,
            tile_resolution,
            meters_per_sample,
        )
    }

    /// The **render**-resident working set — what a `project_terrain` mirror
    /// draws for an asset-backed terrain.
    #[inline]
    pub fn render_data(&self) -> &TerrainData {
        &self.render
    }

    /// The currently published render cut (ascending).
    #[inline]
    pub fn cut(&self) -> &BTreeSet<TileKey> {
        &self.cut
    }

    /// The tile grid this streamer maps keys through.
    #[inline]
    pub fn grid(&self) -> TileGrid {
        self.grid
    }

    /// The available-tile index.
    #[inline]
    pub fn catalog(&self) -> &TileCatalog {
        &self.catalog
    }

    /// The streaming counters.
    #[inline]
    pub fn stats(&self) -> &TerrainStreamStats {
        &self.stats
    }

    /// Replace the residency budget (a settings change).
    pub fn set_budget(&mut self, budget: StreamBudget) {
        self.budget = budget;
    }

    /// Replace the render-wants tuning.
    pub fn set_params(&mut self, params: RenderWantsParams) {
        self.params = params;
    }

    /// **SIM (deterministic).** Reconcile the *simulation's* residency — the
    /// `TerrainData` living in the ECS component — against `wants`.
    ///
    /// Synchronous and serial, in ascending key order, with no camera anywhere in
    /// the inputs. Call this at the fixed-step boundary **before** the step runs,
    /// so every height query inside the step sees the same bytes regardless of
    /// frame rate, camera, or load timing.
    ///
    /// `wants` must come from [`sim_wants`](crate::wants::sim_wants) (or another
    /// pure function of sim state). Never pass a camera-derived set here.
    ///
    /// Keys already known to be corrupt are dropped from `wants` before the sync.
    /// That is **not** a clamp and changes nothing the sim can observe: a blocked
    /// key is by definition one whose blob does not decode, so `sync_residency`
    /// would have left it non-resident anyway — identical state, minus a doomed
    /// decode every step. (Which driver first discovered the corruption is
    /// therefore unobservable too, so sharing one block set with the camera-driven
    /// path does not leak the camera into the simulation.)
    pub fn sync_sim(
        &mut self,
        data: &mut TerrainData,
        wants: &BTreeSet<TileKey>,
        store: &dyn TileStore,
    ) -> ResidencyReport {
        let wants: BTreeSet<TileKey> = if self.blocked.is_empty() {
            wants.clone()
        } else {
            wants.difference(&self.blocked).copied().collect()
        };
        let report = data.sync_residency(&wants, store);
        self.stats.loads += report.loaded.len() as u64;
        self.stats.evictions += report.evicted.len() as u64;
        for (key, err) in &report.failed {
            self.block(*key, err, "sim");
        }
        self.stats.sim_resident_level0 = data.tile_count();
        self.refresh_bytes(data);
        report
    }

    /// **EDIT (editor-only, additive).** Page `wants` into an external working set
    /// without evicting anything, and count the loads.
    ///
    /// The editor's sculpt/paint path (P16.4b) drives this from
    /// [`brush_wants`](crate::wants::brush_wants) before every dab: the target is
    /// the ECS component's `TerrainData` — the **authoritative editable working
    /// set** for an asset-backed terrain — and it must only ever grow, because the
    /// undo record of a stroke can only be replayed against tiles that are still
    /// resident (see [`TerrainData::request_tiles`]).
    ///
    /// Synchronous and serial, like [`sync_sim`](Self::sync_sim), and for the same
    /// reason: an edit must see the same bytes whatever the camera has paged in.
    /// Keys known to be corrupt are dropped from `wants` first, so a bad blob
    /// costs one doomed decode rather than one per dab.
    pub fn page_in(
        &mut self,
        data: &mut TerrainData,
        wants: &BTreeSet<TileKey>,
        store: &dyn TileStore,
    ) -> ResidencyReport {
        let wants: BTreeSet<TileKey> = if self.blocked.is_empty() {
            wants.clone()
        } else {
            wants.difference(&self.blocked).copied().collect()
        };
        let report = data.request_tiles(&wants, store);
        self.stats.loads += report.loaded.len() as u64;
        for (key, err) in &report.failed {
            self.block(*key, err, "edit");
        }
        self.stats.sim_resident_level0 = data.tile_count();
        self.refresh_bytes(data);
        report
    }

    /// **Pin an edited tile into the render working set** so a live sculpt stroke
    /// is visible while it is being made (P16.4b).
    ///
    /// The document's `TerrainData` is the authority for an asset-backed terrain's
    /// edits, but the render projection draws *this* working set, which is paged
    /// from the store and therefore still holds the pre-edit bytes. Pushing the
    /// authored tile in here — and **pinning** it, so the camera cannot evict the
    /// tile the user is currently painting on — is what closes that gap without
    /// giving the renderer a second heightfield to reason about.
    ///
    /// A pinned tile is never re-read from the store either (residency skips
    /// tiles it already holds), so disk content can never win over an unsaved
    /// edit. Release them with [`unpin_tiles`](Self::unpin_tiles) once the store
    /// itself carries the edits — i.e. after a successful save.
    pub fn pin_tile(&mut self, key: TileKey, tile: TerrainTile) {
        self.render.insert_resident_tile(key, tile);
        self.pinned.insert(key);
    }

    /// Whether `key` is pinned against eviction.
    pub fn is_pinned(&self, key: TileKey) -> bool {
        self.pinned.contains(&key)
    }

    /// Number of pinned (edited, unsaved) tiles.
    pub fn pinned_len(&self) -> usize {
        self.pinned.len()
    }

    /// Release every pin, so the camera may page those tiles normally again.
    ///
    /// The tiles stay resident until the next sync decides otherwise — dropping
    /// them here would blink a hole into the terrain the user just saved.
    pub fn unpin_tiles(&mut self) {
        self.pinned.clear();
    }

    /// Unpin **and drop** one tile: the page no longer exists as authored ground.
    ///
    /// The delete case — an authoring `remove_tile`, or the undo of a brush that
    /// authored new ground from nothing. `unpin_tiles` alone would leave the tile
    /// resident and drawn, a phantom the camera can never evict because the store
    /// has nothing to page over it. Returns whether it was resident.
    pub fn unpin_and_evict(&mut self, key: TileKey) -> bool {
        self.pinned.remove(&key);
        let had = self.render.evict_tile(key);
        self.cut.remove(&key);
        had
    }

    /// Adopt a new tile catalog for a store that was **refreshed in place** — the
    /// P16.4b save write-back, which rewrites the `.inf_terrain` a live stream is
    /// pointed at and then reopens it.
    ///
    /// The published cut is intersected with the new catalog (a tile the rewrite
    /// dropped must stop being published), and reseeded from the coarsest level if
    /// that empties it. Everything else — resident pages, budget, params — is
    /// deliberately kept, so the terrain the user is flying over does not re-page
    /// just because it was saved. Pair with [`clear_failed`](Self::clear_failed):
    /// the old "this blob is corrupt" verdicts describe bytes that no longer exist.
    pub fn refresh_catalog(&mut self, catalog: TileCatalog) {
        self.cut.retain(|k| catalog.has(*k));
        if self.cut.is_empty() {
            self.cut = catalog.keys_at_lod(catalog.max_lod()).into_iter().collect();
        }
        self.target.clone_from(&self.cut);
        self.catalog = catalog;
    }

    /// The tile ceiling the **cut** may spend, given the pins that will be
    /// resident beside it.
    ///
    /// `max_resident_tiles` is documented (P16.3b2) as a hard bound on render
    /// residency, and pinned tiles are render-resident — so the pins have to come
    /// out of the same allowance rather than sitting outside it. Only pins that
    /// are *not* already in `cut` are charged; a pinned tile the cut also wants is
    /// paid for once.
    ///
    /// Two documented edges:
    /// * `0` means unlimited and stays unlimited (subtracting from it would turn
    ///   "no ceiling" into "a tiny one").
    /// * When the pins alone meet or exceed the ceiling, the cut still gets **1**
    ///   tile rather than none — a terrain with no published cut draws nothing at
    ///   all. Residency is then `pins + 1`, which is the one place the bound is
    ///   exceeded, and it is exceeded by unsaved authoring the editor must not
    ///   drop. Pins only ever exist in the editor; the shipped player never pins.
    fn pin_ceiling(&self, cut: &BTreeSet<TileKey>) -> usize {
        let max = self.budget.max_resident_tiles;
        if max == 0 || self.pinned.is_empty() {
            return max;
        }
        let charged = self.pinned.iter().filter(|k| !cut.contains(k)).count();
        max.saturating_sub(charged).max(1)
    }

    /// Record `key` as undecodable: warn **once**, count it once, and hide it from
    /// every future want.
    ///
    /// Without this the target cut is recomputed from the catalog each frame, so a
    /// corrupt page is re-requested forever — a doomed decode per frame, an
    /// unbounded `failed` counter, and a `warn!` per frame that buries every other
    /// diagnostic. Blocking it makes a corrupt page behave exactly like unauthored
    /// ground, and `pending_loads` drains.
    ///
    /// # What the ground under a blocked page looks like
    ///
    /// The blocked key is hidden from the index, so the cut treats it as a tile
    /// that was never authored — and that is *not* the same as "the parent covers
    /// it". [`render_wants`](crate::wants::render_wants) refines a node whenever
    /// **any** child exists, so:
    ///
    /// * **all four children blocked** → no child exists, the parent is kept, and
    ///   the footprint really is covered at coarser detail;
    /// * **one to three blocked** (the common case) → the surviving siblings are
    ///   published and the parent is not, so the blocked quadrant is a **hole** —
    ///   the same legal, deterministic hole unauthored ground has always produced.
    ///
    /// Deterministic either way: which pages decode is a property of the bytes, not
    /// of timing, so every run sees the identical hole.
    fn block(&mut self, key: TileKey, err: &str, driver: &str) {
        if self.blocked.insert(key) {
            self.stats.failed += 1;
            tracing::warn!(
                "inf-terrain: {driver} tile {key:?} failed to decode ({err}) — excluded from \
                 streaming until the store is reopened. Its footprint reads as unauthored \
                 ground: a deterministic hole unless all four siblings are blocked too, in \
                 which case the coarser parent covers it."
            );
        }
    }

    /// Keys currently excluded from streaming because their blob failed to decode.
    pub fn blocked_keys(&self) -> &BTreeSet<TileKey> {
        &self.blocked
    }

    /// Forget every blocked key, so corrupt pages are retried.
    ///
    /// **The structural path is a fresh streamer**: every shipped call site
    /// (`inf_player::terrain_stream`, `inf_editor_core::terrain_stream`) rebuilds
    /// the streamer along with the store when content is reloaded, which discards
    /// this set with everything else derived from the old bytes. This method exists
    /// for the narrower case of a store refreshed **in place** — a re-import that
    /// rewrites the `.inf_terrain` a live `FileTileStore` is pointed at — where the
    /// bytes changed but the streamer did not, and the old verdict no longer
    /// describes them.
    ///
    /// Nothing clears the set implicitly: within one set of bytes a blob that
    /// failed to decode will fail again, and retrying it spends a frame's work to
    /// reach the same answer.
    pub fn clear_failed(&mut self) {
        self.blocked.clear();
    }

    /// **RENDER (camera-driven).** Advance the published cut one bounded step
    /// toward the camera's target and reconcile the private render-resident set.
    ///
    /// Call at one fixed point in the frame (the render-sync point). Loads run on
    /// the job pool but the batch is a pure function of the want set (see the
    /// module docs), so the resident-set trace is reproducible run to run.
    pub fn sync_render(
        &mut self,
        camera_xz: DVec2,
        store: &(dyn TileStore + Sync),
    ) -> ResidencyReport {
        // Every want rule sees the catalog MINUS the keys known to be corrupt, so a
        // failed page is never re-requested and its parent covers that ground.
        let index = BlockedIndex {
            inner: &self.catalog,
            blocked: &self.blocked,
        };
        let raw = render_wants(camera_xz, self.grid, &self.params, &index, &self.cut);
        // Pinned tiles are resident and cannot be evicted, so the ceiling the CUT
        // may spend is the budget MINUS the pins that are not already in it (see
        // `pin_ceiling`). Without this the pins would sit outside the budget and
        // `max_resident_tiles` would stop being the hard bound B2 documents.
        let ceiling = self.pin_ceiling(&raw);
        let target = clamp_cut(&raw, self.grid, camera_xz, &index, ceiling);
        self.stats.budget_clamped = target.len() < raw.len();
        // (the published-cut clamp below can also set this)

        let stepped = advance_cut(
            &self.cut,
            &target,
            self.grid,
            camera_xz,
            &index,
            &|k| self.render.is_resident(k),
            self.budget.max_loads_per_sync,
        );
        // The budget bounds **residency**, so it must bound the published cut and
        // not merely the target: one `advance_cut` step replaces a node by up to
        // four children, which can transiently overshoot a target that was itself
        // inside the ceiling. Clamping here (farthest-first, so the detail the
        // camera is looking at survives) makes the ceiling a hard invariant rather
        // than a steady-state aspiration.
        let ceiling = self.pin_ceiling(&stepped);
        let next = clamp_cut(
            &stepped,
            self.grid,
            camera_xz,
            &BlockedIndex {
                inner: &self.catalog,
                blocked: &self.blocked,
            },
            ceiling,
        );
        self.stats.budget_clamped |= next.len() < stepped.len();

        let mut report = ResidencyReport::default();

        // Evict first, so a sliding cut frees its trailing pages before paging in
        // the leading ones (peak residency stays at the cut size) — the same order
        // `sync_residency` documents.
        for key in self.render.resident_keys() {
            // A pinned tile carries an unsaved edit (P16.4b): evicting it would
            // blink the stroke the user is making back to the on-disk bytes.
            if self.pinned.contains(&key) {
                continue;
            }
            if !next.contains(&key) && self.render.evict_tile(key) {
                report.evicted.push(key);
            }
        }

        // Load the new pages as one deterministic parallel batch, then apply them
        // in ascending key order.
        let to_load: Vec<TileKey> = next
            .iter()
            .copied()
            .filter(|k| !self.render.is_resident(*k))
            .collect();
        for (key, outcome) in to_load.iter().copied().zip(load_tiles(store, &to_load)) {
            match outcome {
                Ok(Some(tile)) => {
                    self.render.insert_resident_tile(key, tile);
                    report.loaded.push(key);
                }
                Ok(None) => report.missing.push(key),
                Err(e) => {
                    self.block(key, &e, "render");
                    report.failed.push((key, e));
                }
            }
        }

        // A page the store could not serve must not stay in the published cut. A
        // corrupt one is additionally BLOCKED above, so the next target is computed
        // without it and `pending_loads` really drains instead of re-requesting it
        // forever. Either way its absence is a legal hole — exactly like
        // unauthored ground, with its coarser parent covering the footprint.
        self.cut = next
            .into_iter()
            .filter(|k| self.render.is_resident(*k))
            .collect();
        self.target = target;

        self.stats.loads += report.loaded.len() as u64;
        self.stats.evictions += report.evicted.len() as u64;
        self.stats.resident_level0 = self.render.tile_count();
        self.stats.resident_coarse = self.render.coarse_tile_count();
        self.stats.pending_loads = self
            .target
            .iter()
            .filter(|k| !self.render.is_resident(**k))
            .count();
        self.stats.bytes_resident = resident_bytes(&self.render) + self.sim_bytes;
        report
    }

    /// Remember the sim side's byte count so `bytes_resident` covers both working
    /// sets without the render sync needing the component.
    fn refresh_bytes(&mut self, sim: &TerrainData) {
        self.sim_bytes = resident_bytes(sim);
        self.stats.bytes_resident = resident_bytes(&self.render) + self.sim_bytes;
    }
}

/// Decode `keys` (already ascending) into tiles.
///
/// Runs on the [`inf_core`] job pool for a batch worth parallelizing; the pool's
/// `parallel_map_ref` is a **deterministic in-order pure map**, so the output is
/// a function of the input alone and the caller can apply it in key order. Small
/// batches — and the browser player, which has no thread pool — take the serial
/// path and produce the identical result.
fn load_tiles(
    store: &(dyn TileStore + Sync),
    keys: &[TileKey],
) -> Vec<Result<Option<crate::tile::TerrainTile>, String>> {
    // wasm32 has no thread pool (rayon would build one and panic in a browser),
    // so the web player always takes the serial path — which produces the
    // identical result, `parallel_map_ref` being a pure in-order map.
    #[cfg(not(target_arch = "wasm32"))]
    {
        /// Below this many pages the pool's dispatch costs more than the decode.
        const PARALLEL_THRESHOLD: usize = 4;
        if keys.len() >= PARALLEL_THRESHOLD {
            return inf_core::parallel_map_ref(keys, |k| store.load_tile(*k));
        }
    }
    keys.iter().map(|k| store.load_tile(*k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::build_terrain_asset;
    use crate::pyramid::{build_pyramid, PyramidOptions};
    use crate::residency::MemoryTileStore;
    use crate::wants::sim_wants;

    fn height_fn(x: f64, z: f64) -> f64 {
        x * 0.25 - z * 0.125 + x * z * 0.001 + 5.0
    }

    /// An 8 × 8-tile terrain (res 5, 2 m spacing ⇒ 8 m tile span) with a full
    /// pyramid — big enough for two coarse levels.
    fn fixture() -> (TerrainData, FileTileStore) {
        let mut t = TerrainData::new(5, 2.0);
        for tz in 0..8 {
            for tx in 0..8 {
                t.author_tile((tx, tz), height_fn);
            }
        }
        let p = build_pyramid(&t, PyramidOptions::default());
        let asset = build_terrain_asset(&t, &p).unwrap();
        let store = TerrainAssetReader::new(asset.into_bytes()).unwrap();
        (t, store)
    }

    fn streamer(store: &FileTileStore) -> TerrainStreamer {
        TerrainStreamer::from_store(
            store,
            5,
            2.0,
            RenderWantsParams::geometric(12.0, 4),
            StreamBudget {
                max_resident_tiles: 0,
                max_loads_per_sync: 0,
            },
        )
    }

    #[test]
    fn render_sync_converges_to_the_cut_and_reports() {
        let (_t, store) = fixture();
        let mut s = streamer(&store);
        assert!(s.catalog().max_lod() >= 2, "need a real pyramid");

        let cam = DVec2::new(8.0, 8.0);
        let r = s.sync_render(cam, &store);
        assert!(!r.loaded.is_empty());
        assert!(r.failed.is_empty() && r.missing.is_empty());
        assert!(!s.render_data().is_empty() || s.render_data().coarse_tile_count() > 0);

        // Converge (an unlimited budget still takes a few steps: `advance_cut`
        // refines one level per node per call).
        for _ in 0..8 {
            s.sync_render(cam, &store);
        }
        assert_eq!(s.stats().pending_loads, 0, "the backlog drains");

        // Every published page is really resident, and the cut has fine pages near
        // the camera and coarse ones away from it.
        for &k in s.cut() {
            assert!(s.render_data().is_resident(k), "{k:?} published but absent");
        }
        assert!(s.cut().iter().any(|k| k.lod == 0));
        assert!(s.cut().iter().any(|k| k.lod > 0));

        // Idempotent once settled.
        let before = s.cut().clone();
        let r = s.sync_render(cam, &store);
        assert!(r.is_noop(), "{r:?}");
        assert_eq!(s.cut(), &before);
    }

    #[test]
    fn render_streaming_is_reproducible_run_to_run() {
        // THE GATE PROPERTY, at unit scale: two independent streamers walked over
        // the same scripted camera path publish the identical cut every step.
        let (_t, store) = fixture();
        let path: Vec<DVec2> = (0..40)
            .map(|i| DVec2::new(i as f64 * 1.7, 8.0 + (i % 7) as f64 * 2.3))
            .collect();
        let run = || {
            let mut s = TerrainStreamer::from_store(
                &store,
                5,
                2.0,
                RenderWantsParams::geometric(12.0, 4),
                StreamBudget {
                    max_resident_tiles: 64,
                    max_loads_per_sync: 3,
                },
            );
            let mut trace = Vec::new();
            for &cam in &path {
                s.sync_render(cam, &store);
                trace.push(s.cut().clone());
            }
            trace
        };
        assert_eq!(run(), run(), "the resident-set trace must be reproducible");
    }

    #[test]
    fn the_sim_never_sees_a_camera_paged_tile() {
        // THE DOCTRINE, as an executable assertion: run the identical sim wants
        // under two wildly different camera paths and compare the SIM's residency
        // and its height answers.
        let (_t, store) = fixture();
        let grid = TileGrid::new(5, 2.0);
        let walk: Vec<DVec2> = (0..24).map(|i| DVec2::new(2.0 + i as f64, 6.0)).collect();

        let run = |cameras: &dyn Fn(usize) -> DVec2| {
            let mut s = streamer(&store);
            let mut sim = TerrainData::new(5, 2.0);
            let mut heights = Vec::new();
            let mut residency = Vec::new();
            for (i, &p) in walk.iter().enumerate() {
                // Fixed-step boundary: sim wants, synchronous.
                let wants = sim_wants(grid, &[p], 2.0);
                s.sync_sim(&mut sim, &wants, &store);
                heights.push(sim.height_at(p));
                residency.push(sim.resident_keys());
                // Render-sync point: camera wants, into the OTHER working set.
                s.sync_render(cameras(i), &store);
            }
            (heights, residency)
        };

        let a = run(&|i| DVec2::new(i as f64 * 3.0, i as f64 * 3.0));
        let b = run(&|i| DVec2::new(60.0 - i as f64 * 2.0, 4.0));
        assert_eq!(a.0, b.0, "the sim's height answers moved with the camera");
        assert_eq!(a.1, b.1, "the sim's residency moved with the camera");
        assert!(
            a.0.iter().all(|h| h.is_some()),
            "the walk stayed on terrain"
        );
    }

    #[test]
    fn sim_wants_are_loaded_synchronously_and_never_clamped() {
        // A budget of ONE render page must not starve the sim: correctness first.
        let (_t, store) = fixture();
        let mut s = TerrainStreamer::from_store(
            &store,
            5,
            2.0,
            RenderWantsParams::geometric(12.0, 4),
            StreamBudget {
                max_resident_tiles: 1,
                max_loads_per_sync: 1,
            },
        );
        let grid = TileGrid::new(5, 2.0);
        let mut sim = TerrainData::new(5, 2.0);
        // A wide margin ⇒ many level-0 tiles wanted at once.
        let wants = sim_wants(grid, &[DVec2::new(20.0, 20.0)], 12.0);
        assert!(wants.len() > 4);
        let r = s.sync_sim(&mut sim, &wants, &store);
        assert_eq!(
            r.loaded.len(),
            wants.len(),
            "every sim want lands in ONE call"
        );
        assert!(wants.iter().all(|k| sim.is_resident(*k)));

        // The render side, meanwhile, really is clamped.
        s.sync_render(DVec2::new(20.0, 20.0), &store);
        assert!(
            s.render_data().tile_count() + s.render_data().coarse_tile_count() <= 2,
            "the render budget is not being honoured"
        );
        assert_eq!(s.stats().sim_resident_level0, sim.tile_count());
    }

    #[test]
    fn budget_clamp_is_reported_and_bounded() {
        let (_t, store) = fixture();
        let mut s = TerrainStreamer::from_store(
            &store,
            5,
            2.0,
            RenderWantsParams::geometric(12.0, 4),
            StreamBudget {
                max_resident_tiles: 6,
                max_loads_per_sync: 0,
            },
        );
        let cam = DVec2::new(16.0, 16.0);
        for _ in 0..12 {
            s.sync_render(cam, &store);
        }
        let resident = s.render_data().tile_count() + s.render_data().coarse_tile_count();
        assert!(resident <= 6, "budget blown: {resident} pages");
        assert!(s.stats().budget_clamped, "the clamp must be reported");
        assert!(s.stats().bytes_resident > 0);
        assert!(!s.stats().summary().is_empty());
    }

    /// A full-pyramid store held in memory, so a test can doctor one tile's bytes.
    fn memory_pyramid_store() -> MemoryTileStore {
        let (_t, asset) = fixture();
        let mut mem = MemoryTileStore::new();
        for key in asset.keys() {
            mem.insert_bytes(key, asset.tile_bytes(key).expect("indexed").to_vec());
        }
        mem
    }

    /// A **corrupt page is blocked, not retried forever** (P16.3b2 audit).
    ///
    /// The target cut is recomputed from the catalog every sync, so without a
    /// failed-key set a page whose blob does not decode is re-requested every
    /// single frame: a doomed decode per frame, an unbounded `failed` counter and a
    /// `warn!` per frame. Blocking makes it behave exactly like unauthored ground.
    #[test]
    fn a_corrupt_tile_is_blocked_once_and_never_re_requested() {
        let mut mem = memory_pyramid_store();
        let corrupt = TileKey::lod0((3, 3));
        assert!(mem.contains_tile(corrupt));
        // Garbage that decodes to nothing sane (a truncated, mistyped blob).
        mem.insert_bytes(corrupt, vec![0xFF; 3]);

        let mut s = TerrainStreamer::from_store(
            &mem,
            5,
            2.0,
            RenderWantsParams::geometric(12.0, 4),
            StreamBudget {
                max_resident_tiles: 0,
                max_loads_per_sync: 0,
            },
        );
        // Park the camera right on top of the corrupt page so the cut wants it.
        let cam = DVec2::new(26.0, 26.0);

        // Converge. The failure is discovered exactly once.
        let mut failed_trace = Vec::new();
        for _ in 0..12 {
            s.sync_render(cam, &mem);
            failed_trace.push(s.stats().failed);
        }
        assert_eq!(
            s.stats().failed,
            1,
            "the failure was counted more than once"
        );
        assert!(
            failed_trace.windows(2).all(|w| w[0] <= w[1]),
            "monotone by construction"
        );
        assert_eq!(
            *failed_trace.last().unwrap(),
            failed_trace[failed_trace.len() - 2],
            "`failed` is still growing — the page is being re-requested"
        );
        assert!(s.blocked_keys().contains(&corrupt));

        // The backlog really drains, and the corrupt page never enters the cut …
        assert_eq!(s.stats().pending_loads, 0, "the backlog never drained");
        assert!(!s.cut().contains(&corrupt));
        assert!(!s.render_data().is_resident(corrupt));
        // … while its parent keeps covering that ground (a hole, not a gap in the
        // cut invariant), and the cut is still a valid cut.
        assert!(
            s.cut().iter().any(|k| k.lod > 0),
            "coarse coverage should remain"
        );
        for &k in s.cut() {
            let mut anc = k;
            for _ in 0..8 {
                anc = anc.parent();
                assert!(!s.cut().contains(&anc), "{k:?} coexists with {anc:?}");
            }
        }

        // Settled: no churn at all once blocked.
        let before = s.cut().clone();
        let r = s.sync_render(cam, &mem);
        assert!(r.is_noop(), "{r:?}");
        assert_eq!(s.cut(), &before);
        assert_eq!(s.stats().failed, 1);

        // A store reopen is the documented way back: `clear_failed` retries it.
        s.clear_failed();
        assert!(s.blocked_keys().is_empty());
        s.sync_render(cam, &mem);
        assert_eq!(s.stats().failed, 2, "a cleared key is retried exactly once");
        assert!(s.blocked_keys().contains(&corrupt));
    }

    /// The SIM path blocks a corrupt page too, and the block is observationally
    /// free: a blocked key is one that could not decode, so the sim sees the same
    /// non-resident tile either way — just without a doomed decode per fixed step.
    #[test]
    fn a_corrupt_tile_does_not_stall_the_sim_path() {
        let mut mem = memory_pyramid_store();
        let corrupt = TileKey::lod0((1, 1));
        mem.insert_bytes(corrupt, vec![0xFF; 3]);

        let mut s = TerrainStreamer::from_store(
            &mem,
            5,
            2.0,
            RenderWantsParams::geometric(12.0, 4),
            StreamBudget::default(),
        );
        let grid = TileGrid::new(5, 2.0);
        let mut sim = TerrainData::new(5, 2.0);
        let wants = sim_wants(grid, &[DVec2::new(10.0, 10.0)], 6.0);
        assert!(wants.contains(&corrupt));

        for _ in 0..6 {
            s.sync_sim(&mut sim, &wants, &mem);
        }
        assert_eq!(s.stats().failed, 1, "warned/counted once, not per step");
        assert!(
            !sim.is_resident(corrupt),
            "a corrupt page never goes resident"
        );
        // Every OTHER wanted page is resident — one bad blob does not poison the
        // neighbourhood.
        for &k in wants.iter().filter(|k| **k != corrupt) {
            assert!(sim.is_resident(k), "{k:?} should have loaded");
        }
    }

    /// **Settling under a BINDING ceiling** (P16.3b2 audit): `advance_cut` refines
    /// nearest-first while `clamp_cut` collapses farthest-first, so the two could
    /// in principle chase each other forever — refine here, fold there, repeat.
    /// With a stationary camera the published cut must reach a fixed point and stay
    /// there.
    #[test]
    fn a_binding_budget_settles_with_a_stationary_camera() {
        let (_t, store) = fixture();
        // Deliberately binding: the unclamped cut for this camera is far larger.
        const CEILING: usize = 10;
        let mut s = TerrainStreamer::from_store(
            &store,
            5,
            2.0,
            RenderWantsParams::geometric(12.0, 4),
            StreamBudget {
                max_resident_tiles: CEILING,
                max_loads_per_sync: 3,
            },
        );
        let cam = DVec2::new(24.0, 24.0);

        let mut churn = Vec::new();
        let mut cuts = Vec::new();
        for _ in 0..60 {
            let r = s.sync_render(cam, &store);
            churn.push(r.churn());
            cuts.push(s.cut().clone());
            assert!(
                s.cut().len() <= CEILING,
                "budget blown mid-flight: {}",
                s.cut().len()
            );
        }
        assert!(s.stats().budget_clamped, "the ceiling must actually bind");

        // It converged: churn is zero for the whole tail, and the cut stopped
        // moving. (A refine/fold cycle would show a nonzero periodic churn.)
        let settle_by = 40;
        assert!(
            churn[settle_by..].iter().all(|&c| c == 0),
            "cut never settled — churn tail {:?}",
            &churn[settle_by..]
        );
        assert!(
            cuts[settle_by..].windows(2).all(|w| w[0] == w[1]),
            "the published cut is still oscillating"
        );
        assert!(
            churn[..settle_by].iter().any(|&c| c > 0),
            "the fixture never streamed anything"
        );
        // The settled cut is a real cut that spends detail, not coverage.
        let settled = s.cut();
        assert!(!settled.is_empty());
        for &k in settled {
            let mut anc = k;
            for _ in 0..8 {
                anc = anc.parent();
                assert!(!settled.contains(&anc), "{k:?} coexists with {anc:?}");
            }
        }
        assert!(settled.iter().any(|k| k.lod > 0), "clamping should coarsen");
    }

    /// **Pins are inside the budget (P16.4b audit).** An editor pinning unsaved
    /// tiles into the render set must not silently turn `max_resident_tiles` from
    /// a hard bound into an aspiration: the cut's allowance shrinks by the pins it
    /// does not already contain, so total residency stays under the ceiling.
    #[test]
    fn pinned_tiles_are_charged_against_the_residency_budget() {
        let (t, store) = fixture();
        const CEILING: usize = 10;
        let mut s = TerrainStreamer::from_store(
            &store,
            5,
            2.0,
            RenderWantsParams::geometric(12.0, 4),
            StreamBudget {
                max_resident_tiles: CEILING,
                max_loads_per_sync: 3,
            },
        );
        let cam = DVec2::new(24.0, 24.0);
        for _ in 0..60 {
            s.sync_render(cam, &store);
        }
        let unpinned_resident = s.render_data().resident_keys().len();
        assert!(unpinned_resident <= CEILING);

        // Pin four edited tiles far from the camera (so the cut never wants them).
        let pins: Vec<TileKey> = [(0, 0), (1, 0), (0, 1), (1, 1)]
            .into_iter()
            .map(TileKey::lod0)
            .collect();
        for &k in &pins {
            let tile = t.get_tile(k.coord).expect("fixture tile").clone();
            s.pin_tile(k, tile);
        }
        assert_eq!(s.pinned_len(), pins.len());

        for _ in 0..60 {
            s.sync_render(cam, &store);
            let resident = s.render_data().resident_keys().len();
            assert!(
                resident <= CEILING,
                "pins pushed residency to {resident} over the {CEILING} ceiling"
            );
        }
        // The pins survived every sync — they are unsaved authoring.
        for &k in &pins {
            assert!(s.render_data().is_resident(k), "{k:?} was evicted");
            assert!(s.is_pinned(k));
        }
        // …and the cut really did give ground for them.
        assert!(s.cut().len() < unpinned_resident + pins.len());

        // Releasing the pins hands the allowance back.
        s.unpin_tiles();
        assert_eq!(s.pinned_len(), 0);
        for _ in 0..60 {
            s.sync_render(cam, &store);
        }
        assert!(s.render_data().resident_keys().len() <= CEILING);

        // A ceiling of 0 still means UNLIMITED even with pins outstanding.
        let mut open = TerrainStreamer::from_store(
            &store,
            5,
            2.0,
            RenderWantsParams::geometric(12.0, 4),
            StreamBudget {
                max_resident_tiles: 0,
                max_loads_per_sync: 3,
            },
        );
        for &k in &pins {
            open.pin_tile(k, t.get_tile(k.coord).unwrap().clone());
        }
        for _ in 0..60 {
            open.sync_render(cam, &store);
        }
        assert!(
            !open.stats().budget_clamped,
            "an unlimited budget must never clamp, pins or not"
        );
        assert!(
            open.render_data().resident_keys().len() >= s.render_data().resident_keys().len(),
            "the unlimited streamer paged less than the clamped one"
        );
        for &k in &pins {
            assert!(open.is_pinned(k));
        }
    }

    /// A deleted tile is unpinned **and dropped**: `unpin_tiles` alone would leave
    /// a phantom page the camera can never evict (nothing in the store pages over
    /// it) — the undo-of-an-authoring-op case.
    #[test]
    fn unpin_and_evict_drops_a_deleted_tile() {
        let (t, store) = fixture();
        let mut s = streamer(&store);
        let key = TileKey::lod0((0, 0));
        s.pin_tile(key, t.get_tile(key.coord).unwrap().clone());
        assert!(s.render_data().is_resident(key) && s.is_pinned(key));

        assert!(s.unpin_and_evict(key));
        assert!(
            !s.render_data().is_resident(key),
            "phantom tile left behind"
        );
        assert!(!s.is_pinned(key));
        assert!(!s.cut().contains(&key), "a dropped tile stays published");
        // Idempotent.
        assert!(!s.unpin_and_evict(key));
    }

    #[test]
    fn stats_merge_and_summarize() {
        let mut a = TerrainStreamStats {
            resident_level0: 2,
            loads: 5,
            ..Default::default()
        };
        let b = TerrainStreamStats {
            resident_coarse: 3,
            evictions: 1,
            budget_clamped: true,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.resident_level0, 2);
        assert_eq!(a.resident_coarse, 3);
        assert_eq!(a.loads, 5);
        assert_eq!(a.evictions, 1);
        assert!(a.budget_clamped);
        assert!(a.summary().contains("terrain streaming"));
    }

    #[test]
    fn memory_store_backs_the_streamer_too() {
        // The streamer is store-agnostic (the trait is the seam): a level-0-only
        // memory store yields a level-0-only cut.
        let (t, _asset) = fixture();
        let mut mem = MemoryTileStore::new();
        mem.stage_level0(&t).unwrap();
        let mut s = TerrainStreamer::from_store(
            &mem,
            5,
            2.0,
            RenderWantsParams::geometric(12.0, 4),
            StreamBudget::default(),
        );
        assert_eq!(s.catalog().max_lod(), 0);
        for _ in 0..4 {
            s.sync_render(DVec2::new(8.0, 8.0), &mem);
        }
        assert_eq!(s.render_data().coarse_tile_count(), 0);
        assert_eq!(s.render_data().tile_count(), 64);
    }
}
