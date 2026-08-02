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
//! # Editing a streamed terrain (P16.4b)
//!
//! Supported, and the authority lives in the **document**, not here: a brush dab
//! pages its footprint into the ECS component's `TerrainData`
//! ([`page_brush_footprint`](EditorTerrainStreams::page_brush_footprint)) and
//! sculpts it exactly as it sculpts an inline terrain, while
//! [`overlay_document_edits`](EditorTerrainStreams::overlay_document_edits) pins
//! the edited tiles into *this* render working set so the stroke is visible while
//! it is made. The whole design note — why the component and not the streamer,
//! what it costs, and what undo/save do — is at the top of
//! [`crate::terrain_edit`], which also owns the save-time write-back.
//!
//! Note the doctrine above is unchanged by that: the **camera** still never
//! writes the document. Only an edit gesture pages into it.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{DVec2, DVec3};
use inf_ecs::components::Terrain;
use inf_terrain::stream::{StreamBudget, TerrainStreamStats, TerrainStreamer};
use inf_terrain::wants::{RenderWantsParams, TileCatalog, TileGrid, TileIndex};
use inf_terrain::{FileTileStore, TerrainData, TileKey};
use uuid::Uuid;

use crate::scene::SceneDoc;

/// The status message the sculpt/paint tools surface when a stroke targets an
/// asset-backed terrain whose `.inf_terrain` cannot be written (a read-only file,
/// or an asset that is not in the content root at all). Public so the tools and
/// their tests name one string.
///
/// Replaces P16.3b2's blanket "streamed terrain — tools disabled": a streamed
/// terrain *is* editable now, so the only remaining refusal is the honest one —
/// there is nowhere to save the edits to.
pub const STREAMED_TERRAIN_READONLY_REJECTION: &str =
    "This terrain's .inf_terrain asset is read-only, so Sculpt and Paint have \
     nowhere to save to. Make the file writable (or reimport it into the project's \
     Content folder) and try again.";

/// The status message a brush surfaces when the cursor is over **real streamed
/// ground that is only paged at coarse detail**.
///
/// A stroke needs a resident level-0 page: that is what the brush writes and what
/// the undo record is taken against. Distant terrain is covered by a coarse
/// pyramid tile instead, so the raycast finds nothing and the stroke silently
/// does not start — which reads as a broken tool. Saying why (and that it is
/// fixed by flying closer) costs one line and turns a mystery into an instruction.
/// Only raised when the asset really does have ground there
/// ([`covers_level0`](EditorTerrainStreams::covers_level0)); clicking past the
/// edge of the terrain stays silent.
pub const STREAMED_TERRAIN_COARSE_REJECTION: &str =
    "That ground is streamed in at low detail right now, so there is no full-resolution \
     page under the cursor to sculpt. Fly closer and try again.";

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
    /// The loose file the store was opened from — kept so a save can refresh the
    /// store **in place** without re-resolving the asset (P16.4b).
    path: PathBuf,
    store: Arc<FileTileStore>,
    streamer: TerrainStreamer,
    /// The entity's world translation at the last projection.
    translation: DVec3,
    /// A clone of the `Terrain` component **minus its (empty) working set** —
    /// the layers + macro variation a re-projection needs without the document.
    /// Cheap: a streamed terrain's `data` carries no tiles.
    component: Terrain,
    /// Which document tiles are currently mirrored into the render set, and at
    /// what document stamp — so [`EditorTerrainStreams::overlay_document_edits`]
    /// can run every frame and copy nothing when nothing changed.
    overlaid: BTreeMap<TileKey, u64>,
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
                path,
                store,
                streamer,
                translation,
                component: lightweight_component(terrain),
                overlaid: BTreeMap::new(),
            },
        );
        self.refresh_stats();
        true
    }

    /// Release **every** stream: its resident pages, its edit pins, and its
    /// `Arc<FileTileStore>` (i.e. the whole `.inf_terrain` payload it holds).
    ///
    /// The document-switch door. `File ▸ Open` / `File ▸ New` replace the document
    /// wholesale, so every stream keyed on the *old* document's entity GUIDs is
    /// dead memory — and any tile pinned for an unsaved edit in the old document
    /// would otherwise stay pinned forever, immune to eviction, holding a payload
    /// nothing references. Unlike [`set_content_root`](Self::set_content_root)
    /// this keeps the loose-asset index, because the project has not changed.
    pub fn clear(&mut self) {
        self.streams.clear();
        self.stats = TerrainStreamStats::default();
    }

    /// Drop every stream whose entity is **not** in `keep` — the terrains the
    /// viewport is currently drawing (P16.6: all of them, not just the first).
    ///
    /// A stream for an entity that is no longer projected is dead memory;
    /// releasing it also releases its `.inf_terrain` payload and any tile it
    /// pinned for an unsaved edit. Takes the live set as an iterator because the
    /// caller (the viewport host's projection) produces it as one — and because
    /// passing the *single* live terrain was exactly the shape that made this
    /// function evict a second terrain's payload every frame.
    pub fn retain_only(&mut self, keep: impl IntoIterator<Item = Uuid>) {
        let keep: std::collections::BTreeSet<Uuid> = keep.into_iter().collect();
        self.streams.retain(|k, _| keep.contains(k));
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

    // ── editing (P16.4b) ─────────────────────────────────────────────────
    //
    // See `crate::terrain_edit` for the design note. These three are the only
    // places the streamer and the document meet, and each one is one-directional:
    // `page_brush_footprint` moves store → document, `overlay_document_edits`
    // moves document → render set, and `reload_store` re-points the cold side at
    // the file a save just rewrote.

    /// The loose `.inf_terrain` `entity` streams from.
    pub fn asset_path(&self, entity: Uuid) -> Option<&Path> {
        self.streams.get(&entity).map(|s| s.path.as_path())
    }

    /// Whether `entity`'s terrain can be **edited**: it streams, and its
    /// `.inf_terrain` is a writable file.
    ///
    /// The tools gate on this rather than on "is it streamed" (P16.4b): a
    /// streamed terrain is editable, but one whose asset is read-only has nowhere
    /// to save to, and refusing the stroke up front beats letting the user sculpt
    /// for ten minutes and discover it at Ctrl+S. A missing file also reads as
    /// not-writable — the same honest refusal.
    pub fn is_editable(&self, entity: Uuid) -> bool {
        let Some(s) = self.streams.get(&entity) else {
            return false;
        };
        match std::fs::metadata(&s.path) {
            Ok(m) => !m.permissions().readonly(),
            Err(_) => false,
        }
    }

    /// **Page a brush footprint into the document, synchronously, before the dab.**
    ///
    /// The wants are [`inf_terrain::brush_wants`] — the disk plus a one-tile
    /// margin, i.e. literally the sim-wants shape, because an edit is the
    /// editor's fixed-step boundary and must not depend on what the camera
    /// happened to have paged in. Loading is **additive**: the document's working
    /// set only ever grows within a session, so an undo step can always be
    /// replayed against tiles that are still resident.
    ///
    /// Returns how many tiles were newly paged in, or `None` when `entity` is not
    /// streamed (an inline terrain needs no paging and the caller should just
    /// sculpt).
    pub fn page_brush_footprint(
        &mut self,
        entity: Uuid,
        doc: &mut SceneDoc,
        center_local: DVec2,
        radius: f64,
    ) -> Option<usize> {
        let s = self.streams.get_mut(&entity)?;
        let wants = inf_terrain::brush_wants(s.streamer.grid(), center_local, radius);
        let store = s.store.clone();
        let streamer = &mut s.streamer;
        let loaded = doc.with_terrain_data_mut(entity, |data| {
            streamer.page_in(data, &wants, store.as_ref()).loaded.len()
        })?;
        self.refresh_stats();
        Some(loaded)
    }

    /// **Mirror the document's *unsaved* level-0 tiles into the render working
    /// set**, so a live stroke is visible while it is being made.
    ///
    /// The document is the authority; this set is what the viewport draws. A
    /// mirrored tile is *pinned*, so the camera cannot evict it and the store can
    /// never re-page pre-edit bytes over an unsaved edit.
    ///
    /// Deliberately keyed on the **dirty** set rather than the whole working set:
    /// a clean document tile is byte-identical to the store's, so mirroring it
    /// would pin a page the streamer is perfectly capable of managing and grow
    /// render residency by everything the user has ever brushed. Dirty tiles are
    /// exactly the ones where the document and the disk disagree.
    ///
    /// Cheap to call every frame: a tile is only copied when its document stamp
    /// changed, and stamps only move when something actually wrote to the tile.
    /// Returns how many tiles were (re)mirrored.
    pub fn overlay_document_edits(&mut self, entity: Uuid, doc: &SceneDoc) -> usize {
        let Some((data, _)) = doc.terrain_data_and_origin(entity) else {
            return 0;
        };
        let Some(s) = self.streams.get_mut(&entity) else {
            return 0;
        };
        let mut mirrored = 0;
        for key in data.dirty_tiles() {
            if !key.is_lod0() {
                continue;
            }
            // A dirty key with no resident tile is an authoring DELETE — either a
            // `remove_tile` or the undo of a brush that authored new ground from
            // nothing. The page must be **dropped and unpinned**, not merely
            // forgotten: leaving it resident leaves a phantom tile the camera can
            // never evict, because the store has nothing to page over it.
            let Some(tile) = data.get_tile(key.coord) else {
                if s.overlaid.remove(&key).is_some() || s.streamer.is_pinned(key) {
                    s.streamer.unpin_and_evict(key);
                    mirrored += 1;
                }
                continue;
            };
            let stamp = data.tile_version(key);
            if s.overlaid.get(&key) == Some(&stamp) && s.streamer.is_pinned(key) {
                continue;
            }
            s.streamer.pin_tile(key, tile.clone());
            s.overlaid.insert(key, stamp);
            mirrored += 1;
        }
        if mirrored > 0 {
            self.refresh_stats();
        }
        mirrored
    }

    /// Whether the **asset** has authored level-0 ground at terrain-local `xz`,
    /// regardless of what is currently resident.
    ///
    /// The discriminator behind the "fly closer" tool status (P16.4b): a brush
    /// raycast that misses can mean two very different things, and the tools must
    /// only nag about one of them. `true` = there is real ground here, it is just
    /// paged at coarse detail, so the stroke can succeed after the camera closes
    /// in. `false` = the user clicked past the edge of the terrain, which is not a
    /// problem and must not produce a message.
    pub fn covers_level0(&self, entity: Uuid, xz: DVec2) -> bool {
        let Some(s) = self.streams.get(&entity) else {
            return false;
        };
        let coord = s.streamer.grid().coord_at(0, xz);
        s.streamer.catalog().has(TileKey::lod0(coord))
    }

    /// Number of tiles currently pinned into `entity`'s render set because they
    /// carry unsaved edits (diagnostics + tests).
    pub fn pinned_tiles(&self, entity: Uuid) -> usize {
        self.streams
            .get(&entity)
            .map(|s| s.streamer.pinned_len())
            .unwrap_or(0)
    }

    /// **Refresh a stream's cold store in place** after a save rewrote its
    /// `.inf_terrain` — the live stream keeps flying.
    ///
    /// Reopens the loose file, adopts its new catalog (the rewrite may have added
    /// or removed tiles), clears the "this blob is corrupt" verdicts — they
    /// describe bytes that no longer exist — and releases the edit pins, because
    /// the store now *is* the edits. Deliberately not
    /// [`set_content_root`](Self::set_content_root): that drops every stream and
    /// re-pages the terrain the user is looking at, which is a visible hitch for
    /// no reason. This is the `refresh_index` idea one level down.
    pub fn reload_store(&mut self, entity: Uuid) -> Result<(), String> {
        let Some(s) = self.streams.get_mut(&entity) else {
            return Ok(()); // not streamed — nothing to refresh
        };
        let store = inf_terrain::open_file_tile_store(&s.path)
            .map_err(|e| format!("reopen {}: {e}", s.path.display()))?;
        s.streamer.refresh_catalog(TileCatalog::from_store(&store));
        s.streamer.clear_failed();
        s.streamer.unpin_tiles();
        s.overlaid.clear();
        s.store = Arc::new(store);
        self.refresh_stats();
        Ok(())
    }

    /// [`reload_store`](Self::reload_store) for every live stream, logging rather
    /// than failing the save when one cannot be reopened.
    pub fn reload_stores(&mut self) {
        let mut keys: Vec<Uuid> = self.streams.keys().copied().collect();
        keys.sort();
        for k in keys {
            if let Err(e) = self.reload_store(k) {
                tracing::warn!("inf-editor-core: terrain store refresh failed: {e}");
            }
        }
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
        biome_set: terrain.biome_set,
        // P19.3's biome population is NOT carried: it is a derived cache the
        // projector reads off the DOCUMENT's component, never off this streaming
        // mirror, and copying it here would duplicate a potentially huge instance
        // list once per streamed terrain per document change.
        biome_population: Vec::new(),
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

    /// **P16.6.** Two streamed terrains projected at once: `retain_only` must keep
    /// **both**, and drop only what really left the projection.
    ///
    /// The pre-P16.6 signature took the single live terrain, so a viewport drawing
    /// two of them released one whole `.inf_terrain` payload every frame and paged
    /// it back in the next — the coupling P16.4b's audit note flagged as
    /// single-terrain-shaped.
    #[test]
    fn retain_only_keeps_every_live_stream() {
        let (dir, terrain) = fixture();
        let second = Uuid::from_u128(0x5EC0_11D2);
        let mut s = EditorTerrainStreams::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        // Two entities streaming from the same asset — different terrains as far
        // as the manager is concerned (streams are keyed by ENTITY).
        assert!(s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::ZERO
        ));
        assert!(s.ensure(second, &terrain, DVec3::new(1000.0, 0.0, 0.0), DVec3::ZERO));
        assert_eq!(s.len(), 2);

        // Both live ⇒ both kept.
        s.retain_only([STREAMED_TERRAIN_TERRAIN_GUID, second]);
        assert_eq!(s.len(), 2, "a live stream was released");

        // One leaves the projection ⇒ exactly that one is released.
        s.retain_only([second]);
        assert_eq!(s.len(), 1);
        assert!(s.render_data(second).is_some());
        assert!(s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).is_none());

        // Nothing live ⇒ everything released.
        s.retain_only(std::iter::empty());
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

    // ── editing (P16.4b) ─────────────────────────────────────────────────

    /// A live stream over the generated asset, plus its document.
    fn edit_fixture() -> (
        tempfile::TempDir,
        crate::scene::SceneDoc,
        EditorTerrainStreams,
    ) {
        let dir = tempfile::tempdir().unwrap();
        write_streamed_terrain_asset(dir.path()).unwrap();
        let doc = streamed_terrain_scene();
        let terrain = {
            let world = doc.world();
            let e = world.entity_of(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
            world.world().get::<Terrain>(e).unwrap().clone()
        };
        let mut s = EditorTerrainStreams::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert!(s.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::new(64.0, 40.0, 64.0)
        ));
        (dir, doc, s)
    }

    /// **Tool enablement.** A streamed terrain whose asset is writable is
    /// editable; the same terrain with a read-only asset is not.
    #[test]
    fn editability_follows_the_assets_write_permission() {
        let (dir, _doc, s) = edit_fixture();
        assert!(
            s.is_editable(STREAMED_TERRAIN_TERRAIN_GUID),
            "a normal loose asset must be editable"
        );
        assert!(
            !s.is_editable(Uuid::from_u128(0xDEAD)),
            "a non-streamed entity is never 'editable' here"
        );
        assert_eq!(
            s.asset_path(STREAMED_TERRAIN_TERRAIN_GUID)
                .and_then(|p| p.file_name()),
            Some(std::ffi::OsStr::new("World.inf_terrain"))
        );

        // Flip the file read-only: the tools must refuse rather than let a stroke
        // be lost at save time.
        let path = dir.path().join("World.inf_terrain");
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&path, perms).unwrap();
        assert!(!s.is_editable(STREAMED_TERRAIN_TERRAIN_GUID));
        // Restore so the tempdir can be removed on every platform.
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        perms.set_readonly(false);
        std::fs::set_permissions(&path, perms).unwrap();
    }

    /// **Footprint paging.** A dab's wants land in the DOCUMENT, not in the
    /// streamer's render set, and paging alone dirties nothing.
    #[test]
    fn paging_a_footprint_fills_the_document_without_dirtying_it() {
        let (_dir, mut doc, mut s) = edit_fixture();
        assert!(!doc.is_dirty());
        let center = DVec2::new(100.0, 100.0);
        let loaded = s
            .page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, center, 6.0)
            .expect("streamed");
        assert!(loaded > 0, "nothing paged in");

        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        assert_eq!(data.tile_count(), loaded);
        assert!(!data.has_dirty_tiles(), "paging is not authoring");
        assert!(!doc.is_dirty(), "paging must not dirty the document");

        // Idempotent: paging the same footprint again loads nothing.
        assert_eq!(
            s.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, center, 6.0),
            Some(0)
        );
        // …and the working set only GROWS: a footprint elsewhere adds tiles
        // without dropping the first one's.
        let far = DVec2::new(180.0, 60.0);
        assert!(
            s.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, far, 6.0)
                .unwrap()
                > 0
        );
        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        assert!(data.tile_count() > loaded);
        assert!(data.is_resident(inf_terrain::TileKey::lod0(
            data.tile_coord_of(center.x, center.y)
        )));
    }

    /// **The live-stroke overlay.** An edited tile is mirrored into the render set
    /// and **pinned**, so a camera move cannot evict the stroke back to the
    /// on-disk bytes.
    #[test]
    fn an_edited_tile_is_pinned_into_the_render_set() {
        let (_dir, mut doc, mut s) = edit_fixture();
        let center = DVec2::new(100.0, 100.0);
        s.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, center, 6.0);
        assert_eq!(
            s.overlay_document_edits(STREAMED_TERRAIN_TERRAIN_GUID, &doc),
            0
        );
        assert_eq!(s.pinned_tiles(STREAMED_TERRAIN_TERRAIN_GUID), 0);

        let mut stroke = inf_terrain::Stroke::begin();
        doc.sculpt_apply_dab(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &mut stroke,
            inf_terrain::BrushOp::Raise,
            inf_terrain::BrushParams::new(center, 6.0, 25.0),
        );
        let mirrored = s.overlay_document_edits(STREAMED_TERRAIN_TERRAIN_GUID, &doc);
        assert!(mirrored > 0, "the edit was not mirrored");
        assert_eq!(s.pinned_tiles(STREAMED_TERRAIN_TERRAIN_GUID), mirrored);

        let key = inf_terrain::TileKey::lod0(
            doc.terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
                .unwrap()
                .0
                .tile_coord_of(center.x, center.y),
        );
        let edited = s
            .render_data(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .height_at(center);
        assert_eq!(
            edited,
            doc.terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
                .unwrap()
                .0
                .height_at(center),
            "the render set does not show the edit"
        );

        // Fly far away for a long time: the cut moves, but the pinned page holds.
        for step in 0..40 {
            s.sync_render(streamed_terrain_camera_a(step));
        }
        assert!(s
            .render_data(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .is_resident(key));
        assert_eq!(
            s.render_data(STREAMED_TERRAIN_TERRAIN_GUID)
                .unwrap()
                .height_at(center),
            edited,
            "the camera paged the pre-edit bytes back over an unsaved edit"
        );
        // A second overlay pass with nothing new copies nothing.
        assert_eq!(
            s.overlay_document_edits(STREAMED_TERRAIN_TERRAIN_GUID, &doc),
            0
        );
    }

    /// **The biome twin of the pinning gate (P19.2).**
    ///
    /// This is the only test in the tree that puts an edit under *genuine
    /// eviction pressure*, and the biome layer has a render-side consumer the
    /// height twin cannot exercise: the per-tile **id buffer** the Biomes overlay
    /// uploads as an `R8Uint` texture. A stroke that survived in the document but
    /// got paged over in the render set would leave the overlay tinting the
    /// on-disk ids — a stale picture of a level the author just repainted, with
    /// nothing in the document wrong to point at.
    ///
    /// It also pins the *negative* half, which matters more for a categorical
    /// layer than for heights: ground the brush never claimed must still read
    /// `UNASSIGNED_BIOME` after the camera has walked away and back, or the
    /// overlay paints biomes nobody authored.
    #[test]
    fn an_edited_biome_tile_is_pinned_into_the_render_set() {
        let (_dir, mut doc, mut s) = edit_fixture();
        let center = DVec2::new(100.0, 100.0);
        s.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, center, 6.0);
        assert_eq!(
            s.overlay_document_edits(STREAMED_TERRAIN_TERRAIN_GUID, &doc),
            0
        );
        assert_eq!(s.pinned_tiles(STREAMED_TERRAIN_TERRAIN_GUID), 0);

        let mut stroke = inf_terrain::BiomeStroke::begin(4);
        doc.biome_apply_dab(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &mut stroke,
            inf_terrain::BrushParams::new(center, 6.0, 1.0),
        );
        assert!(doc.edit_commit_biome(STREAMED_TERRAIN_TERRAIN_GUID, stroke));

        let mirrored = s.overlay_document_edits(STREAMED_TERRAIN_TERRAIN_GUID, &doc);
        assert!(mirrored > 0, "the biome edit was not mirrored");
        assert_eq!(s.pinned_tiles(STREAMED_TERRAIN_TERRAIN_GUID), mirrored);

        let key = inf_terrain::TileKey::lod0(
            doc.terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
                .unwrap()
                .0
                .tile_coord_of(center.x, center.y),
        );
        assert_eq!(
            s.render_data(STREAMED_TERRAIN_TERRAIN_GUID)
                .unwrap()
                .biome_at(center),
            Some(4),
            "the render set does not show the painted biome"
        );
        // A point well outside the 6 m brush — but still inside a mirrored tile —
        // must NOT have been claimed. The id buffer the overlay uploads is
        // materialized now, so "unpainted" has to survive materialization, not
        // merely sparsity.
        let far = center + DVec2::new(10.0, 0.0);
        assert_eq!(
            s.render_data(STREAMED_TERRAIN_TERRAIN_GUID)
                .unwrap()
                .biome_at(far),
            Some(inf_terrain::UNASSIGNED_BIOME),
            "the brush claimed ground outside its radius"
        );

        // Fly far away for a long time: the cut moves, but the pinned page holds.
        for step in 0..40 {
            s.sync_render(streamed_terrain_camera_a(step));
        }
        let render = s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        assert!(render.is_resident(key));
        assert_eq!(
            render.biome_at(center),
            Some(4),
            "the camera paged the pre-edit ids back over an unsaved biome stroke"
        );
        assert_eq!(render.biome_at(far), Some(inf_terrain::UNASSIGNED_BIOME));
        // A second overlay pass with nothing new copies nothing.
        assert_eq!(
            s.overlay_document_edits(STREAMED_TERRAIN_TERRAIN_GUID, &doc),
            0
        );
    }

    /// **In-place store refresh (gate f).** Reopening the store after a save keeps
    /// the live stream flying: its cut survives, its untouched resident tiles are
    /// byte-stable, and the pins are released.
    #[test]
    fn refreshing_a_store_in_place_keeps_untouched_tiles_byte_stable() {
        let (dir, mut doc, mut s) = edit_fixture();
        for step in 0..24 {
            s.sync_render(streamed_terrain_camera_a(step));
        }
        let before_cut = s.cut(STREAMED_TERRAIN_TERRAIN_GUID).unwrap().clone();
        let before: Vec<(inf_terrain::TileKey, Vec<u8>)> = {
            let data = s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
            data.resident_keys()
                .into_iter()
                .filter_map(|k| {
                    data.resident_tile(k)
                        .map(|t| (k, inf_terrain::asset::encode_tile(t).unwrap()))
                })
                .collect()
        };
        assert!(before.len() > 1);

        // Edit ONE tile far from the camera and save it back.
        let center = DVec2::new(200.0, 200.0);
        s.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, center, 6.0);
        let mut stroke = inf_terrain::Stroke::begin();
        doc.sculpt_apply_dab(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &mut stroke,
            inf_terrain::BrushOp::Raise,
            inf_terrain::BrushParams::new(center, 6.0, 20.0),
        );
        doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke);
        s.overlay_document_edits(STREAMED_TERRAIN_TERRAIN_GUID, &doc);
        let edited_keys: Vec<inf_terrain::TileKey> = doc
            .terrain_dirty_tiles(STREAMED_TERRAIN_TERRAIN_GUID)
            .into_iter()
            .collect();
        assert!(!edited_keys.is_empty());
        crate::terrain_edit::flush_terrain_edits(&mut doc, dir.path());

        // THE REFRESH: the stream survives it.
        s.reload_store(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        assert_eq!(s.len(), 1, "the live stream must not be dropped");
        assert_eq!(
            s.cut(STREAMED_TERRAIN_TERRAIN_GUID).unwrap(),
            &before_cut,
            "the published cut moved"
        );
        assert_eq!(
            s.pinned_tiles(STREAMED_TERRAIN_TERRAIN_GUID),
            0,
            "the store carries the edits now; the pins must be released"
        );

        // …and every tile the edit did not touch is byte-identical.
        let data = s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        for (key, bytes) in &before {
            if edited_keys.contains(key) {
                continue;
            }
            let tile = data.resident_tile(*key).expect("still resident");
            assert_eq!(
                &inf_terrain::asset::encode_tile(tile).unwrap(),
                bytes,
                "{key:?} changed across an in-place store refresh"
            );
        }
        // A further camera pass still works against the new bytes.
        for step in 24..40 {
            s.sync_render(streamed_terrain_camera_a(step));
        }
        assert!(s.stats().failed == 0);
    }

    /// **The document-switch leak (P16.4b audit).** Opening another level while a
    /// streamed terrain carries unsaved edits must release the stream, its pins
    /// and its `.inf_terrain` payload — nothing else ever would: `retain_only`
    /// keys on the *live* terrain GUID, and a pin is only released by a successful
    /// save of a terrain still in the document.
    #[test]
    fn replacing_the_document_releases_streams_and_pins() {
        let (_dir, mut doc, mut s) = edit_fixture();
        let center = DVec2::new(100.0, 100.0);
        s.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, center, 6.0);
        let mut stroke = inf_terrain::Stroke::begin();
        doc.sculpt_apply_dab(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &mut stroke,
            inf_terrain::BrushOp::Raise,
            inf_terrain::BrushParams::new(center, 6.0, 20.0),
        );
        assert!(s.overlay_document_edits(STREAMED_TERRAIN_TERRAIN_GUID, &doc) > 0);
        assert!(s.pinned_tiles(STREAMED_TERRAIN_TERRAIN_GUID) > 0);
        assert_eq!(s.len(), 1);

        // File ▸ Open / File ▸ New: the document is replaced wholesale.
        s.clear();
        assert!(s.is_empty(), "the dead stream still holds its payload");
        assert_eq!(s.pinned_tiles(STREAMED_TERRAIN_TERRAIN_GUID), 0);
        assert!(s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).is_none());
        assert_eq!(s.stats(), &TerrainStreamStats::default());
        // The loose-asset index survives (the PROJECT did not change), so the new
        // document's terrains resolve without a rescan.
        assert_eq!(s.index_len(), 1);

        // …and the viewport's per-projection retention does the same for a terrain
        // that simply stopped being the drawn one.
        let (_dir2, mut doc2, mut s2) = edit_fixture();
        s2.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc2, center, 6.0);
        s2.retain_only(Some(Uuid::from_u128(0xFEED)));
        assert!(s2.is_empty());
        assert_eq!(s2.pinned_tiles(STREAMED_TERRAIN_TERRAIN_GUID), 0);
    }

    /// **The phantom tile (P16.4b audit).** Undoing a brush that authored ground
    /// from nothing removes the tile from the document — the render set must drop
    /// **and unpin** it, or it stays drawn forever (nothing in the store can page
    /// over a tile the store does not have).
    #[test]
    fn undoing_an_authoring_op_drops_the_phantom_tile() {
        let (_dir, mut doc, mut s) = edit_fixture();
        // Well outside the generated 16×16-tile (256 m) extent.
        let center = DVec2::new(300.0, 300.0);
        s.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, center, 8.0);
        let mut stroke = inf_terrain::Stroke::begin();
        doc.sculpt_apply_dab(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &mut stroke,
            inf_terrain::BrushOp::Raise,
            inf_terrain::BrushParams::new(center, 8.0, 20.0),
        );
        assert!(doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke));
        s.overlay_document_edits(STREAMED_TERRAIN_TERRAIN_GUID, &doc);

        let authored: Vec<inf_terrain::TileKey> = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0
            .tiles()
            .map(|(&c, _)| inf_terrain::TileKey::lod0(c))
            .filter(|k| k.coord.0 >= 16 || k.coord.1 >= 16)
            .collect();
        assert!(!authored.is_empty(), "Raise must author new ground");
        let render = s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        for &k in &authored {
            assert!(render.is_resident(k), "new ground {k:?} was not mirrored");
        }

        // UNDO: the tiles the stroke created are removed from the document.
        assert!(doc.undo());
        s.overlay_document_edits(STREAMED_TERRAIN_TERRAIN_GUID, &doc);
        let render = s.render_data(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        for &k in &authored {
            assert!(
                !render.is_resident(k),
                "{k:?} is a phantom: undone in the document, still drawn"
            );
        }
        assert_eq!(
            s.pinned_tiles(STREAMED_TERRAIN_TERRAIN_GUID),
            0,
            "a removed tile must not stay pinned"
        );
        for &k in &authored {
            assert!(!s.cut(STREAMED_TERRAIN_TERRAIN_GUID).unwrap().contains(&k));
        }
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
