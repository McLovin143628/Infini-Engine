//! Editor-side terrain streaming (P16.3b2): the viewport's half of the sim/render
//! want split.
//!
//! The shipped player streams from a cooked pack; the editor streams the **loose**
//! `.inf_terrain` sitting in the project's content root. Everything else is the
//! same Ring-0 machinery ([`inf_terrain::wants`] + [`inf_terrain::stream`]), so
//! the two hosts page identically — which is what makes the PIE-==-shipping
//! parity gate meaningful for streamed terrain.
//!
//! # THE DETERMINISM DOCTRINE, in the editor
//!
//! The editor has no fixed step outside **Simulate**, so it computes **no sim
//! wants**: the only driver here is [`sync_render`](EditorTerrainStreams::sync_render),
//! from the editor camera. Its pages land in the streamer's private working set,
//! never in the document's `Terrain.data` — which is exactly why an editor camera
//! move cannot dirty the document, cannot change a `height_at` answer, and cannot
//! desync a Simulate session from a shipped run. (When Simulate runs, its
//! `RuntimeSim` drives its own sim wants against the world it owns, exactly as the
//! player does.)
//!
//! # Why this lives in Ring 1 and not in the viewport host
//!
//! `inf_viewport::host` is compiled only on Windows and macOS, so logic placed
//! there is invisible to Linux CI. Keeping the streaming *policy* here — platform
//! neutral, GPU free — means the unit tests below run on all three OSes and the
//! host is left with nothing but the call sites.
//!
//! # Editing a streamed terrain
//!
//! Not yet supported: sculpt/paint against an asset-backed terrain would have to
//! write tiles back through the residency dirty set and re-emit the `.inf_terrain`,
//! which arrives with the P16.4 import-wizard work. Until then the tools reject
//! the stroke with a status message rather than silently editing a page that is
//! about to be evicted — see [`STREAMED_TERRAIN_EDIT_REJECTION`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{DVec2, DVec3};
use inf_ecs::components::Terrain;
use inf_terrain::stream::{StreamBudget, TerrainStreamStats, TerrainStreamer};
use inf_terrain::wants::{RenderWantsParams, TileCatalog, TileGrid, TileIndex};
use inf_terrain::{FileTileStore, TerrainData};
use uuid::Uuid;

/// The status message the sculpt/paint tools surface when a stroke targets an
/// asset-backed (streamed) terrain. Public so the tools and their tests name one
/// string.
pub const STREAMED_TERRAIN_EDIT_REJECTION: &str =
    "Sculpt and Paint cannot edit a streamed (.inf_terrain) terrain yet — \
     editing streamed terrain arrives with the Terrain Import wizard (P16.4).";

/// Refinement radius of a level-0 node, in level-0 tile spans.
///
/// MIRROR of `inf_player::terrain_stream::RENDER_LOD0_RADIUS_TILES` — the editor
/// viewport and the shipped player must page the same cut from the same camera,
/// or "what you see is what ships" stops being true for terrain detail.
pub const RENDER_LOD0_RADIUS_TILES: f64 = 2.5;

/// One streamed terrain entity in the open document.
struct EditorStream {
    /// The `.inf_terrain` asset GUID.
    asset: Uuid,
    store: Arc<FileTileStore>,
    streamer: TerrainStreamer,
    /// The entity's world translation at the last projection.
    translation: DVec3,
    /// A clone of the `Terrain` component **minus its (empty) working set** —
    /// the layers + macro variation a re-projection needs without the document.
    /// Cheap: a streamed terrain's `data` carries no tiles.
    component: Terrain,
}

/// Every streamed terrain the viewport is currently drawing.
#[derive(Default)]
pub struct EditorTerrainStreams {
    /// Where loose `.inf_terrain` assets are looked up. `None` (the default)
    /// disables streaming entirely — an inline terrain is unaffected either way.
    content_root: Option<PathBuf>,
    /// Asset GUID → loose file, rebuilt when the root changes and, lazily, when
    /// a lookup misses (see [`EditorTerrainStreams::resolve`]).
    index: HashMap<Uuid, PathBuf>,
    /// Asset GUIDs a rescan has already failed to find, so a genuinely dangling
    /// reference costs **one** directory walk rather than one per frame. Cleared
    /// whenever the index is rebuilt.
    rescanned_for: std::collections::HashSet<Uuid>,
    /// Entity GUID → its stream.
    streams: HashMap<Uuid, EditorStream>,
    stats: TerrainStreamStats,
    budget: StreamBudget,
}

impl EditorTerrainStreams {
    /// An empty set with the default residency budget.
    pub fn new() -> Self {
        Self {
            budget: StreamBudget::default(),
            ..Default::default()
        }
    }

    /// Point streaming at a project's content root (or `None` to disable it).
    ///
    /// Rescans the loose `.inf_terrain` index and drops every live stream, so a
    /// project switch can never serve the previous project's pages.
    pub fn set_content_root(&mut self, root: Option<PathBuf>) {
        self.streams.clear();
        self.stats = TerrainStreamStats::default();
        self.index = match &root {
            Some(dir) => terrain_paths_by_guid(dir),
            None => HashMap::new(),
        };
        self.rescanned_for.clear();
        self.content_root = root;
    }

    /// Rebuild the loose-asset index **without** disturbing live streams.
    ///
    /// The index is a snapshot taken when the content root was set, so an asset
    /// written *after* it — exactly what the P16.4 import wizard does — would
    /// otherwise never be found. Ring 2 calls this when a terrain import
    /// finishes. Unlike [`set_content_root`](Self::set_content_root) it keeps
    /// every resident page, so refreshing after an import does not re-page a
    /// terrain the user is already flying over.
    pub fn refresh_index(&mut self) {
        let Some(dir) = self.content_root.clone() else {
            return;
        };
        self.index = terrain_paths_by_guid(&dir);
        self.rescanned_for.clear();
    }

    /// Number of indexed loose `.inf_terrain` assets.
    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    /// The active content root.
    pub fn content_root(&self) -> Option<&Path> {
        self.content_root.as_deref()
    }

    /// Replace the residency budget (applies to every live stream).
    pub fn set_budget(&mut self, budget: StreamBudget) {
        self.budget = budget;
        for s in self.streams.values_mut() {
            s.streamer.set_budget(budget);
        }
    }

    /// Whether `entity` is currently streamed (as opposed to inline).
    pub fn is_streamed(&self, entity: Uuid) -> bool {
        self.streams.contains_key(&entity)
    }

    /// Number of live streams.
    pub fn len(&self) -> usize {
        self.streams.len()
    }
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Merged counters for the diagnostics dump.
    pub fn stats(&self) -> &TerrainStreamStats {
        &self.stats
    }

    /// The render-resident working set for `entity` (what the viewport projects).
    pub fn render_data(&self, entity: Uuid) -> Option<&TerrainData> {
        self.streams.get(&entity).map(|s| s.streamer.render_data())
    }

    /// The cached component + translation a re-projection needs.
    pub fn projection_inputs(&self, entity: Uuid) -> Option<(&Terrain, &TerrainData, DVec3)> {
        let s = self.streams.get(&entity)?;
        Some((&s.component, s.streamer.render_data(), s.translation))
    }

    /// The published render cut for `entity` (the trace a test compares).
    pub fn cut(&self, entity: Uuid) -> Option<&std::collections::BTreeSet<inf_terrain::TileKey>> {
        self.streams.get(&entity).map(|s| s.streamer.cut())
    }

    /// Look `asset` up in the loose-file index, **rescanning once** if it misses.
    ///
    /// The index is a snapshot of the content root taken when the root was set.
    /// An asset written after that — a terrain the import wizard just produced —
    /// is therefore absent from it, and without this the entity the wizard spawns
    /// draws nothing until the project is reopened. A miss triggers **one** rescan
    /// per asset GUID (`rescanned_for`), so a reference that is genuinely dangling
    /// costs a single directory walk instead of one every frame.
    fn resolve(&mut self, asset: Uuid, entity: Uuid) -> Option<PathBuf> {
        if let Some(path) = self.index.get(&asset) {
            return Some(path.clone());
        }
        let root = self.content_root.clone()?;
        if !self.rescanned_for.insert(asset) {
            return None; // already looked for this one, still absent
        }
        self.index = terrain_paths_by_guid(&root);
        match self.index.get(&asset) {
            Some(path) => {
                tracing::info!(
                    "inf-editor-core: terrain asset {asset} appeared after the index was built \
                     — rescanned the content root"
                );
                Some(path.clone())
            }
            None => {
                tracing::warn!(
                    "inf-viewport: terrain {entity} references .inf_terrain {asset}, which is \
                     not in the content root — drawing its inline data"
                );
                None
            }
        }
    }

    /// Ensure `entity`'s stream exists and is current, returning whether it
    /// streams (i.e. whether the viewport should draw the streamer's working set
    /// instead of the component's).
    ///
    /// Called from the document projection: cheap and idempotent for an existing
    /// stream (it refreshes the cached component + translation), and a no-op for
    /// an inline terrain or an unresolvable asset ref.
    pub fn ensure(
        &mut self,
        entity: Uuid,
        terrain: &Terrain,
        translation: DVec3,
        eye: DVec3,
    ) -> bool {
        let Some(asset) = terrain.asset else {
            self.streams.remove(&entity);
            return false;
        };
        if let Some(existing) = self.streams.get_mut(&entity) {
            if existing.asset == asset {
                existing.translation = translation;
                existing.component = lightweight_component(terrain);
                return true;
            }
            self.streams.remove(&entity); // the ref was repointed
        }
        let Some(path) = self.resolve(asset, entity) else {
            return false;
        };
        let store = match inf_terrain::open_file_tile_store(&path) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                tracing::error!("inf-viewport: open {}: {e}", path.display());
                return false;
            }
        };
        let header = *store.header();
        let catalog = TileCatalog::from_store(store.as_ref());
        let grid = TileGrid::new(header.tile_resolution, header.meters_per_sample);
        let params = RenderWantsParams::geometric(
            RENDER_LOD0_RADIUS_TILES * grid.level0_span(),
            catalog.max_lod() + 1,
        );
        let mut streamer = TerrainStreamer::new(
            grid,
            catalog,
            params,
            self.budget,
            header.tile_resolution,
            header.meters_per_sample,
        );
        // Page the seed cut in immediately so the first frame draws something.
        streamer.sync_render(
            DVec2::new(eye.x, eye.z) - DVec2::new(translation.x, translation.z),
            store.as_ref(),
        );
        tracing::info!(
            "inf-viewport: streaming terrain {entity} from {} ({} page(s))",
            path.display(),
            streamer.catalog().len()
        );
        self.streams.insert(
            entity,
            EditorStream {
                asset,
                store,
                streamer,
                translation,
                component: lightweight_component(terrain),
            },
        );
        self.refresh_stats();
        true
    }

    /// Drop every stream except `keep` (the terrain the viewport is drawing).
    ///
    /// The viewport draws one terrain (first visible wins), so a stream for an
    /// entity that is no longer projected is dead memory; releasing it also
    /// releases its `.inf_terrain` payload.
    pub fn retain_only(&mut self, keep: Option<Uuid>) {
        self.streams.retain(|k, _| Some(*k) == keep);
        self.refresh_stats();
    }

    /// **The render-sync point.** Advance every stream's camera-driven cut.
    ///
    /// Returns `true` when any stream's published cut changed, i.e. when the
    /// caller must re-project the render terrain. Camera-driven and therefore
    /// deliberately *not* reflected anywhere in the document.
    pub fn sync_render(&mut self, eye_world: DVec3) -> bool {
        if self.streams.is_empty() {
            return false;
        }
        let mut changed = false;
        // Deterministic order (entity GUID), so a multi-terrain scene behaves the
        // same on every run.
        let mut keys: Vec<Uuid> = self.streams.keys().copied().collect();
        keys.sort();
        for k in keys {
            let Some(s) = self.streams.get_mut(&k) else {
                continue;
            };
            let cam =
                DVec2::new(eye_world.x, eye_world.z) - DVec2::new(s.translation.x, s.translation.z);
            let store = s.store.clone();
            let report = s.streamer.sync_render(cam, store.as_ref());
            changed |= !report.is_noop();
        }
        self.refresh_stats();
        changed
    }

    fn refresh_stats(&mut self) {
        let mut merged = TerrainStreamStats::default();
        let mut keys: Vec<Uuid> = self.streams.keys().copied().collect();
        keys.sort();
        for k in keys {
            merged.merge(self.streams[&k].streamer.stats());
        }
        self.stats = merged;
    }
}

/// A `Terrain` carrying everything a projection needs **except** the heightfield:
/// layers, macro variation, grid config and the asset ref. A streamed terrain's
/// document-side `data` is empty anyway, so this is a cheap clone that also
/// documents which fields the re-projection actually reads.
fn lightweight_component(terrain: &Terrain) -> Terrain {
    Terrain {
        meters_per_sample: terrain.meters_per_sample,
        tile_resolution: terrain.tile_resolution,
        data: TerrainData::new(terrain.tile_resolution, terrain.meters_per_sample),
        layers: terrain.layers,
        macro_variation: terrain.macro_variation,
        asset: terrain.asset,
    }
}

/// Index every loose `.inf_terrain` **under** `dir` by its asset GUID, read from
/// the sibling inf_asset `.toml` sidecar.
///
/// **Recursive**, unlike its player counterpart
/// (`inf_player::level::terrain_paths_by_guid_from_dir`), and deliberately so:
/// the player's scans a single sample-level *directory*, while this one scans a
/// whole project **content root**, which has folders — the P16.4 import wizard
/// writes to `<Content>/Terrain/`. Flat scanning would leave a freshly imported
/// terrain unstreamable, which is the entire point of the import.
///
/// Deterministic (each directory's entries are path-sorted before descending),
/// files without a readable sidecar are skipped, and the editor's own
/// dot-directories (`.inf` import cache, `.infinity` settings) are not walked.
pub fn terrain_paths_by_guid(dir: &Path) -> HashMap<Uuid, PathBuf> {
    let mut out = HashMap::new();
    let mut files = Vec::new();
    collect_terrain_files(dir, 0, &mut files);
    for p in files {
        match inf_asset::AssetSidecar::load(&p) {
            Ok(side) => {
                out.insert(side.guid.uuid(), p);
            }
            Err(_) => tracing::warn!(
                "inf-editor-core: .inf_terrain without a sidecar {}",
                p.display()
            ),
        }
    }
    out
}

/// Depth cap for [`terrain_paths_by_guid`]'s walk — deep enough for any content
/// layout, shallow enough that a symlink loop cannot hang project open.
const MAX_CONTENT_DEPTH: u32 = 16;

fn collect_terrain_files(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    if depth > MAX_CONTENT_DEPTH {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            let hidden = path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.starts_with('.'));
            if !hidden {
                collect_terrain_files(&path, depth + 1, out);
            }
        } else if path.extension().and_then(|s| s.to_str()) == Some("inf_terrain") {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samples::{
        streamed_terrain_camera_a, streamed_terrain_scene, write_streamed_terrain_asset,
        STREAMED_TERRAIN_ASSET_GUID, STREAMED_TERRAIN_TERRAIN_GUID,
    };

    /// A temp content root holding the generated streamed-terrain asset, plus the
    /// scene's `Terrain` component.
    fn fixture() -> (tempfile::TempDir, Terrain) {
        let dir = tempfile::tempdir().unwrap();
        write_streamed_terrain_asset(dir.path()).unwrap();
        let doc = streamed_terrain_scene();
        let world = doc.world();
        let e = world.entity_of(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        let terrain = world.world().get::<Terrain>(e).unwrap().clone();
        assert_eq!(terrain.asset, Some(STREAMED_TERRAIN_ASSET_GUID));
        assert!(terrain.data.is_empty(), "the doc ships no tiles");
        (dir, terrain)
    }

    #[test]
    fn without_a_content_root_nothing_streams() {
        let (_dir, terrain) = fixture();
        let mut s = EditorTerrainStreams::new();
        assert!(!s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::ZERO
        ));
        assert!(s.is_empty());
        assert!(!s.sync_render(DVec3::ZERO));
    }

    #[test]
    fn an_inline_terrain_never_streams() {
        let (dir, mut terrain) = fixture();
        terrain.asset = None;
        let mut s = EditorTerrainStreams::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert!(!s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::ZERO
        ));
        assert!(s.is_empty(), "inline terrain must be untouched");
    }

    #[test]
    fn a_streamed_terrain_pages_from_the_content_root() {
        let (dir, terrain) = fixture();
        let mut s = EditorTerrainStreams::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert!(s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::new(64.0, 40.0, 64.0)
        ));
        assert!(s.is_streamed(STREAMED_TERRAIN_TERRAIN_GUID));

        // The seed cut is already resident, so the first frame draws something.
        let data = s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        assert!(data.tile_count() + data.coarse_tile_count() > 0);

        // Converge, then check the cut has both fine and coarse pages.
        for _ in 0..24 {
            s.sync_render(DVec3::new(64.0, 40.0, 64.0));
        }
        let cut = s.cut(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        assert!(cut.iter().any(|k| k.lod == 0), "no fine pages: {cut:?}");
        assert!(cut.iter().any(|k| k.lod > 0), "no coarse pages: {cut:?}");
        // Every published page is really resident, and the cut is a real cut.
        let data = s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        for &k in cut {
            assert!(data.is_resident(k));
            let mut anc = k;
            for _ in 0..8 {
                anc = anc.parent();
                assert!(!cut.contains(&anc), "{k:?} coexists with {anc:?}");
            }
        }
        assert!(s.stats().loads > 0);
        assert!(s.stats().bytes_resident > 0);
    }

    /// **The doctrine, editor side**: the camera drives the streamer's working set
    /// and NEVER the document's component data.
    #[test]
    fn the_camera_never_touches_the_documents_terrain_data() {
        let (dir, terrain) = fixture();
        let mut s = EditorTerrainStreams::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::ZERO,
        );
        for step in 0..40 {
            s.sync_render(streamed_terrain_camera_a(step));
        }
        // The component handed in is untouched (it is borrowed, never written).
        assert!(
            terrain.data.is_empty(),
            "the editor camera wrote pages into the DOCUMENT's terrain"
        );
        // …while the streamer's own set is full of them.
        let data = s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        assert!(data.tile_count() + data.coarse_tile_count() > 0);
        // And no page was ever marked dirty: streaming is not authoring, so the
        // document can never look modified for having been looked at.
        assert!(!data.has_dirty_tiles());
    }

    /// Two independent runs over the same camera path publish the same cut trace —
    /// the editor mirror of the player's headless determinism gate.
    #[test]
    fn editor_streaming_is_reproducible() {
        let (dir, terrain) = fixture();
        let run = || {
            let mut s = EditorTerrainStreams::new();
            s.set_content_root(Some(dir.path().to_path_buf()));
            s.ensure(
                STREAMED_TERRAIN_TERRAIN_GUID,
                &terrain,
                DVec3::ZERO,
                DVec3::ZERO,
            );
            let mut trace = Vec::new();
            for step in 0..40 {
                s.sync_render(streamed_terrain_camera_a(step));
                trace.push(s.cut(STREAMED_TERRAIN_TERRAIN_GUID).unwrap().clone());
            }
            trace
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "the editor's resident-set trace is not reproducible");
        assert!(
            a.iter().collect::<std::collections::BTreeSet<_>>().len() > 1,
            "the camera never paged anything"
        );
    }

    #[test]
    fn changing_the_content_root_drops_every_stream() {
        let (dir, terrain) = fixture();
        let mut s = EditorTerrainStreams::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert!(s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::ZERO
        ));
        // A project switch must never serve the previous project's pages.
        let other = tempfile::tempdir().unwrap();
        s.set_content_root(Some(other.path().to_path_buf()));
        assert!(s.is_empty());
        assert!(!s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::ZERO
        ));
        assert_eq!(s.stats(), &TerrainStreamStats::default());
    }

    #[test]
    fn retain_only_releases_dead_streams() {
        let (dir, terrain) = fixture();
        let mut s = EditorTerrainStreams::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::ZERO,
        );
        assert_eq!(s.len(), 1);
        s.retain_only(Some(Uuid::from_u128(0xDEAD)));
        assert!(s.is_empty());
    }

    #[test]
    fn the_index_reads_asset_guids_from_sidecars() {
        let (dir, _terrain) = fixture();
        let index = terrain_paths_by_guid(dir.path());
        assert_eq!(index.len(), 1);
        assert!(index.contains_key(&STREAMED_TERRAIN_ASSET_GUID));
        // A directory with no assets indexes empty rather than failing.
        let empty = tempfile::tempdir().unwrap();
        assert!(terrain_paths_by_guid(empty.path()).is_empty());
        assert!(terrain_paths_by_guid(Path::new("no/such/dir")).is_empty());
    }

    /// The import wizard writes into `<Content>/Terrain/`, so a flat scan of the
    /// content root would leave a freshly imported terrain unstreamable — the
    /// one thing the whole import exists to avoid (P16.4a).
    #[test]
    fn the_index_finds_terrain_in_content_subfolders() {
        let root = tempfile::tempdir().unwrap();
        write_streamed_terrain_asset(&root.path().join("Terrain")).unwrap();
        let index = terrain_paths_by_guid(root.path());
        assert_eq!(index.len(), 1, "a subfolder terrain must be indexed");
        let path = &index[&STREAMED_TERRAIN_ASSET_GUID];
        assert!(
            path.ends_with("Terrain/World.inf_terrain")
                || path.ends_with("Terrain\\World.inf_terrain")
        );

        // …and the editor's own dot-directories are not walked (an import cache
        // can hold copies whose sidecars would shadow the real asset).
        write_streamed_terrain_asset(&root.path().join(".inf/import-cache")).unwrap();
        assert_eq!(terrain_paths_by_guid(root.path()).len(), 1);
    }

    /// **The import-then-walk gate (P16.4a audit).** The index is built when the
    /// content root is set; the wizard writes its asset *after* that. Without
    /// rescan-on-miss, "Add to Scene" would spawn an entity that draws nothing
    /// until the project is reopened — the one thing the import exists to avoid.
    #[test]
    fn a_terrain_written_after_the_index_still_streams() {
        let root = tempfile::tempdir().unwrap();
        let mut s = EditorTerrainStreams::new();
        // Index the (empty) content root FIRST — this is the stale snapshot.
        s.set_content_root(Some(root.path().to_path_buf()));
        assert_eq!(s.index_len(), 0);

        // …then import lands the asset, exactly as the wizard does.
        write_streamed_terrain_asset(&root.path().join("Terrain")).unwrap();

        let doc = streamed_terrain_scene();
        let world = doc.world();
        let e = world.entity_of(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        let terrain = world.world().get::<Terrain>(e).unwrap().clone();

        // No reopen, no `set_content_root`: the miss rescans and finds it.
        assert!(
            s.ensure(
                STREAMED_TERRAIN_TERRAIN_GUID,
                &terrain,
                DVec3::ZERO,
                DVec3::new(64.0, 40.0, 64.0)
            ),
            "a terrain written after the index was built must still stream"
        );
        assert_eq!(s.index_len(), 1);
        let data = s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        assert!(data.tile_count() + data.coarse_tile_count() > 0);
    }

    /// …and a reference that really is dangling costs ONE walk, not one a frame.
    #[test]
    fn a_dangling_asset_reference_rescans_only_once() {
        let root = tempfile::tempdir().unwrap();
        let doc = streamed_terrain_scene();
        let world = doc.world();
        let e = world.entity_of(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        let terrain = world.world().get::<Terrain>(e).unwrap().clone();

        let mut s = EditorTerrainStreams::new();
        s.set_content_root(Some(root.path().to_path_buf()));
        // First miss rescans (and still finds nothing).
        assert!(!s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::ZERO
        ));
        // Now write it — but the GUID is already on the "looked, absent" list, so
        // the next `ensure` must NOT walk the directory again…
        write_streamed_terrain_asset(root.path()).unwrap();
        assert!(!s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::ZERO
        ));
        // …until something explicitly refreshes the index (what Ring 2 does when
        // an import finishes).
        s.refresh_index();
        assert!(s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::ZERO
        ));
    }

    /// `refresh_index` keeps live streams (unlike a content-root change), so an
    /// import never re-pages a terrain the user is already flying over.
    #[test]
    fn refreshing_the_index_keeps_live_streams() {
        let (dir, terrain) = fixture();
        let mut s = EditorTerrainStreams::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::new(64.0, 40.0, 64.0),
        );
        assert_eq!(s.len(), 1);
        s.refresh_index();
        assert_eq!(s.len(), 1, "refresh_index must not drop live streams");
        assert!(s.is_streamed(STREAMED_TERRAIN_TERRAIN_GUID));
    }

    /// A terrain streamed out of a content SUBFOLDER pages exactly like one in
    /// the root — the end of the import wizard's "walk it immediately" path.
    #[test]
    fn a_terrain_in_a_subfolder_streams() {
        let root = tempfile::tempdir().unwrap();
        write_streamed_terrain_asset(&root.path().join("Terrain")).unwrap();
        let doc = streamed_terrain_scene();
        let world = doc.world();
        let e = world.entity_of(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        let terrain = world.world().get::<Terrain>(e).unwrap().clone();

        let mut s = EditorTerrainStreams::new();
        s.set_content_root(Some(root.path().to_path_buf()));
        assert!(s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::new(64.0, 40.0, 64.0)
        ));
        let data = s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        assert!(data.tile_count() + data.coarse_tile_count() > 0);
    }
}
