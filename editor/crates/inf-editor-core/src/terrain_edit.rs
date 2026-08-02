//! Editing an **asset-backed (streamed) terrain** in the editor (P16.4b): the
//! authority model, and the save path that writes it back.
//!
//! # THE DESIGN NOTE — one authoritative editable working set
//!
//! **Decision: the ECS `Terrain` component's `TerrainData` *is* the editable
//! working set for a streamed terrain, exactly as it already is for an inline
//! one.** Resident tiles page **into the component** on demand — stamped, never
//! dirtied — the instant a brush needs them; brushes, deltas and undo then run
//! against the component with no idea that the tiles arrived from a
//! `.inf_terrain` rather than from the `.inf_lvl`; and on save the component's
//! dirty set is merged back into the asset. The camera-driven streamer keeps its
//! own private render working set and keeps feeding the render projection, with
//! the document's edited tiles overlaid on top of it so a stroke is visible while
//! it is being made. **Authority:** the component. **Undo:** the existing
//! `EditCommand::SculptTerrain` / `PaintSplat` against the component — *zero* new
//! undo machinery, because the thing being edited is the thing undo already
//! knows how to edit. **Save:** Ctrl+S drains the component's dirty tiles into
//! the `.inf_terrain` through [`inf_terrain::rewrite_terrain_asset`]; the
//! `.inf_lvl` never persists a streamed terrain's working set (see
//! [`crate::scene::serialize`]), so the level stays kilobytes and the asset stays
//! the single source of truth.
//!
//! The alternative — **editing the streamer's render working set** — loses, and
//! not marginally. That set is camera-driven: it is evicted the moment the user
//! flies away, so a stroke's tiles can vanish mid-gesture and an undo step
//! recorded against them would replay into tiles that are no longer there
//! (`revert_delta` would recreate them *flat* and silently destroy every sample
//! outside the recorded patch). It also holds *coarse* pyramid tiles beside the
//! level-0 ones, so "the tile under the brush" stops being a single well-defined
//! thing. And it is owned by the viewport host, which is compiled only on Windows
//! and macOS and locks the document every frame — so undo, autosave and save
//! would all have to reach across a thread boundary into platform-gated code to
//! find the authored bytes. Every one of those problems is *already solved* for
//! the component: the dirty set refuses eviction, the level-0 map is exactly the
//! authored heightfield, and `SceneDoc` is where undo and save already live.
//!
//! A third option — a *separate* editor-side edit buffer, distinct from both —
//! was rejected for adding a third copy of the truth (component, streamer,
//! buffer) with no owner, and for making `terrain_data_and_origin`, the erosion
//! bake, the PCG height source and every future terrain consumer ask "which one
//! do I mean?".
//!
//! # What paging into the document costs, and what it does not
//!
//! The P16.3b2 doctrine — *the editor camera never writes the document* —
//! survives intact and is worth restating precisely: **only an edit gesture pages
//! into the component.** Flying around still cannot dirty the document, cannot
//! change a `height_at` answer, and cannot desync a Simulate session. What
//! changed is that a *brush* may now page, synchronously, before it applies —
//! which is the same rule the fixed-step simulation follows
//! ([`inf_terrain::sim_wants`]), for the same reason.
//!
//! The working set only ever **grows** within a session
//! ([`TerrainData::request_tiles`](inf_terrain::TerrainData::request_tiles)
//! never evicts), because an undo step can only be replayed against tiles that
//! are still resident. Its ceiling is therefore "everything the user brushed",
//! released when the document closes. For a session that sculpts a wide area at
//! 256² tiles this is real memory, and bounding it — by spilling clean,
//! undo-covered tiles back to the store — is the documented follow-up.
//!
//! # Autosave does **not** write assets
//!
//! Asset writes are explicit. A debounced autosave firing a whole-payload
//! `.inf_terrain` rewrite every few seconds would be both a performance
//! catastrophe and a correctness one (it would silently commit edits the user has
//! not chosen to keep). So autosave leaves the terrain alone and instead records
//! that unsaved terrain edits *existed*, next to the crash-recovery file — see
//! [`crate::scene::serialize::write_recovery_terrain_note`]. Recovery then warns
//! honestly that the recovered level's terrain is the **last saved** asset, which
//! is the truth and is better than a silent partial restore.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use inf_terrain::writeback::TerrainEdits;
use inf_terrain::{PyramidOptions, TileKey};
use uuid::Uuid;

use crate::scene::SceneDoc;

/// One streamed terrain's pending write-back, lifted out of the document.
///
/// Owned and self-contained on purpose: the caller stages under whatever lock
/// guards the document, **releases it**, and then does the slow whole-payload
/// rewrite — the same "encode under the lock, write outside it" rule the level
/// save already follows, and for the same reason (the viewport locks the same
/// document every frame).
#[derive(Debug, Clone)]
pub struct StagedTerrainEdit {
    /// The terrain entity.
    pub entity: Uuid,
    /// The `.inf_terrain` asset GUID its tiles belong to.
    pub asset: Uuid,
    /// The level-0 changes to fold in.
    pub edits: TerrainEdits,
    /// **The concurrent-edit guard**: every staged tile's change stamp at the
    /// moment it was lifted out of the document.
    ///
    /// The rewrite runs with the document **unlocked** and can take seconds on a
    /// large asset, so the user can keep sculpting while it runs. Clearing the
    /// whole dirty set afterwards would throw those edits away — they were never
    /// written. Marking is therefore per key and conditional on the stamp still
    /// matching ([`inf_terrain::TerrainData::clear_dirty_if_unchanged`]): a tile
    /// touched during the write window stays dirty and the next save writes it.
    pub stamps: Vec<(TileKey, u64)>,
}

/// One asset a write-back pass rewrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenTerrain {
    /// The terrain entity.
    pub entity: Uuid,
    /// The `.inf_terrain` that was rewritten.
    pub path: PathBuf,
    /// The level-0 tiles that actually reached disk, with the stamp each carried
    /// when it was staged — exactly what may now be un-marked.
    pub tiles: Vec<(TileKey, u64)>,
}

/// A terrain whose edits could **not** be written, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnwrittenTerrain {
    /// The terrain entity, whose edits are still dirty (and therefore still
    /// protected from eviction, still reported unsaved, and still retried).
    pub entity: Uuid,
    /// The `.inf_terrain` asset GUID it references.
    pub asset: Uuid,
    /// Human-readable reason, for the save toast / Output Log.
    pub reason: String,
}

/// What one write-back pass did, per asset.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TerrainFlushReport {
    /// Each asset rewritten, with the tiles that landed.
    pub written: Vec<WrittenTerrain>,
    /// Terrains that had edits but could not be written — a missing asset, an IO
    /// error, a corrupt source. **Never silently dropped**: a save that could not
    /// write terrain is not a successful save of that terrain, and the caller
    /// must say so.
    pub unwritten: Vec<UnwrittenTerrain>,
}

impl TerrainFlushReport {
    /// `true` when nothing was written and nothing failed.
    pub fn is_empty(&self) -> bool {
        self.written.is_empty() && self.unwritten.is_empty()
    }

    /// `true` when at least one terrain's edits could not be written.
    pub fn has_failures(&self) -> bool {
        !self.unwritten.is_empty()
    }

    /// Total level-0 tiles folded into assets.
    pub fn tiles(&self) -> usize {
        self.written.iter().map(|w| w.tiles.len()).sum()
    }

    /// A one-line summary for the Output Log / the save toast.
    pub fn summary(&self) -> String {
        format!(
            "terrain write-back: {} tile(s) into {} asset(s){}",
            self.tiles(),
            self.written.len(),
            if self.unwritten.is_empty() {
                String::new()
            } else {
                format!(", {} FAILED", self.unwritten.len())
            }
        )
    }
}

/// The pyramid options a `.inf_terrain` write-back re-plans with **when the asset
/// does not say** — i.e. the fallback for a schema-**v1** payload.
///
/// # How this stopped being a limitation (P16.6)
///
/// v1's header did not record the options its pyramid was built with, so every
/// write-back re-planned with the defaults. The common case was fine (the sample
/// generator hard-codes the defaults and the wizard defaults to them), but the
/// wizard *exposes* `max_pyramid_levels` / `min_pyramid_tiles`, so an asset
/// imported with non-default knobs was silently re-planned to the **default
/// shape** on its first save — still a correct, byte-deterministic pyramid, but
/// not the shape its author chose, and a total rebuild rather than a partial one.
///
/// **Schema v2 records them** (`inf_terrain::asset`), so
/// [`write_terrain_edits`] re-plans with the asset's *own* options and a v2 asset
/// can no longer be reshaped by a save. This constant is what a v1 asset falls
/// back to, and it is the only case [`warn_on_pyramid_reshape`] can still fire
/// for. (Inferring the options back out of a v1 asset was considered and rejected
/// again: the two stop conditions are indistinguishable after the fact, and every
/// inference rule that preserves a capped asset's depth also refuses to deepen a
/// terrain that genuinely grew. A v1 asset is upgraded to v2 by that first save,
/// recording the defaults — which is the most honest thing that can be known
/// about it, and it means the warning fires at most once per asset.)
pub const WRITE_BACK_PYRAMID: PyramidOptions = PyramidOptions {
    max_levels: inf_terrain::pyramid::DEFAULT_MAX_PYRAMID_LEVELS,
    min_tiles: inf_terrain::pyramid::DEFAULT_MIN_PYRAMID_TILES,
};

/// The options a write-back of `source` should re-plan its pyramid with: the ones
/// the asset **records** (schema v2), else [`WRITE_BACK_PYRAMID`] (schema v1, which
/// did not record them).
pub fn write_back_pyramid<B: AsRef<[u8]>>(
    source: &inf_terrain::TerrainAssetReader<B>,
) -> PyramidOptions {
    source.pyramid_options().unwrap_or(WRITE_BACK_PYRAMID)
}

/// Warn when a **v1** `source`'s pyramid does not have the depth
/// [`WRITE_BACK_PYRAMID`] would have given it — i.e. when this save is about to
/// re-shape an asset whose options were never recorded (see there).
///
/// P16.6 narrowed this to exactly the case it can still describe. A v2 asset
/// carries its own options, so the rewrite re-plans to the shape its author chose
/// and there is nothing to warn about; a v1 asset that already matches the
/// defaults is not being reshaped either. What is left — a v1 asset that genuinely
/// disagrees with the defaults — is the one situation where a save really does
/// change the asset's shape, and it stays loud.
///
/// Cheap: it plans over the source's **own** level-0 coordinate set, which is
/// coordinate arithmetic with no tile data touched at all.
fn warn_on_pyramid_reshape<B: AsRef<[u8]>>(
    source: &inf_terrain::TerrainAssetReader<B>,
    path: &Path,
) {
    if source.pyramid_options().is_some() {
        return; // schema v2: the asset says, so nothing is re-shaped
    }
    let level0: std::collections::BTreeSet<(i32, i32)> = source
        .keys()
        .filter(|k| k.is_lod0())
        .map(|k| k.coord)
        .collect();
    let want = inf_terrain::plan_pyramid(&level0, source.meters_per_sample(), WRITE_BACK_PYRAMID)
        .len() as u32;
    let have = source.lod_levels().saturating_sub(1);
    if want != have {
        tracing::warn!(
            "inf-editor-core: {} has {have} coarse LOD level(s) but the default pyramid options \
             would give it {want} — this is a schema-v1 .inf_terrain, which did not record the \
             settings it was imported with, so saving terrain edits will re-plan it to the \
             default shape (and write it as v2, recording those defaults). Re-import it if you \
             need its original pyramid settings; see terrain_edit::WRITE_BACK_PYRAMID.",
            path.display()
        );
    }
}

/// Lift every streamed terrain's pending edits out of `doc` (empty when nothing
/// is dirty — a save that changed no terrain stages nothing and writes nothing).
pub fn stage_terrain_edits(doc: &SceneDoc) -> Vec<StagedTerrainEdit> {
    let mut out = Vec::new();
    for entity in doc.streamed_terrain_entities() {
        let Some(asset) = doc.terrain_asset_of(entity) else {
            continue;
        };
        let dirty: Vec<TileKey> = doc.terrain_dirty_tiles(entity);
        if dirty.is_empty() {
            continue;
        }
        let Some((data, _)) = doc.terrain_data_and_origin(entity) else {
            continue;
        };
        let edits = TerrainEdits::from_dirty(data, &dirty);
        if edits.is_empty() {
            continue;
        }
        // The stamp of every staged tile, so the mark step can tell "this is the
        // tile I wrote" from "the user re-sculpted it while I was writing".
        let stamps: Vec<(TileKey, u64)> = edits
            .touched()
            .into_iter()
            .map(|coord| {
                let key = TileKey::lod0(coord);
                (key, data.tile_version(key))
            })
            .collect();
        out.push(StagedTerrainEdit {
            entity,
            asset,
            edits,
            stamps,
        });
    }
    out
}

/// Merge `staged` into the loose `.inf_terrain` assets under `content_root` and
/// rewrite them **atomically**.
///
/// Does no document work at all (it takes none), so it is safe to call with every
/// lock released. Each asset goes through the one sanctioned writer,
/// [`inf_terrain::write_terrain_asset`], and its `inf_asset` sidecar is restamped
/// over exactly the bytes written so the cook packs what is on disk.
/// **Infallible by design.** A failure is *per terrain* and lands in
/// [`TerrainFlushReport::unwritten`] rather than aborting the pass: one
/// unreadable asset must not stop the others from being written, and — more
/// importantly — a failure has to reach the user as "this terrain did not save"
/// rather than as a save that quietly did less than it claimed.
pub fn write_terrain_edits(
    staged: &[StagedTerrainEdit],
    content_root: &Path,
) -> TerrainFlushReport {
    let mut report = TerrainFlushReport::default();
    if staged.is_empty() {
        return report;
    }
    // One directory walk for the whole save, however many terrains it covers.
    let index: BTreeMap<Uuid, PathBuf> = crate::terrain_stream::terrain_paths_by_guid(content_root)
        .into_iter()
        .collect();

    for item in staged {
        let mut fail = |reason: String| {
            tracing::warn!(
                "inf-editor-core: terrain {} could not be written back — {reason}. Its edits \
                 stay in memory (still unsaved, still retried by the next save).",
                item.entity
            );
            report.unwritten.push(UnwrittenTerrain {
                entity: item.entity,
                asset: item.asset,
                reason,
            });
        };
        let Some(path) = index.get(&item.asset) else {
            fail(format!(
                "its .inf_terrain {} is not under {}",
                item.asset,
                content_root.display()
            ));
            continue;
        };
        let source = match inf_terrain::open_file_tile_store(path) {
            Ok(s) => s,
            Err(e) => {
                fail(format!("open {}: {e}", path.display()));
                continue;
            }
        };
        warn_on_pyramid_reshape(&source, path);
        // P16.6: re-plan with the options the asset RECORDS (schema v2), falling
        // back to the defaults only for a v1 payload that never recorded any.
        let opts = write_back_pyramid(&source);
        let rewritten = match inf_terrain::rewrite_terrain_asset(&source, &item.edits, opts) {
            Ok(Some(a)) => a,
            // Nothing to write (an empty edit set slipped through) — not a
            // failure, and nothing to un-mark either.
            Ok(None) => continue,
            Err(e) => {
                fail(format!("rewrite {}: {e}", path.display()));
                continue;
            }
        };
        // Drop the old payload before the rename: the store owns a whole copy of
        // the previous image, and there is no reason to hold two plus the new one.
        drop(source);
        let bytes = match inf_terrain::write_terrain_asset(path, &rewritten) {
            Ok(b) => b,
            Err(e) => {
                fail(format!("write {}: {e}", path.display()));
                continue;
            }
        };
        if let Err(e) = inf_asset::AssetSidecar::new(
            inf_asset::AssetId(item.asset),
            inf_asset::AssetKind::Terrain,
            inf_asset::ContentHash::of(bytes),
        )
        .save(path)
        {
            // The payload landed but its hash did not: report it rather than
            // clearing the marks over a half-written asset pair.
            fail(format!("write sidecar for {}: {e}", path.display()));
            continue;
        }
        report.written.push(WrittenTerrain {
            entity: item.entity,
            path: path.clone(),
            tiles: item.stamps.clone(),
        });
    }
    report
}

/// Clear the write-back marks of every tile `report` says actually reached disk —
/// **and only if it has not been re-edited since it was staged**.
///
/// Two separate protections, both load-bearing:
///
/// * A terrain whose write *failed* keeps every mark, so its edits stay protected
///   from eviction, keep reporting as unsaved, and are retried by the next save.
/// * A *tile* the user sculpted during the (unlocked, possibly multi-second)
///   rewrite keeps its mark, because the bytes on disk are older than the tile in
///   memory. Clearing the whole dirty set would silently discard exactly the edits
///   made while the user was waiting for the save.
///
/// Returns how many marks were cleared.
pub fn mark_terrain_edits_saved(doc: &mut SceneDoc, report: &TerrainFlushReport) -> usize {
    report
        .written
        .iter()
        .map(|w| doc.terrain_mark_written_back(w.entity, &w.tiles))
        .sum()
}

/// Stage → write → mark, for a caller that can hold the document across the
/// whole operation (tests, the CLI). The editor's save path splits the three so
/// the disk IO happens with the document lock released.
pub fn flush_terrain_edits(doc: &mut SceneDoc, content_root: &Path) -> TerrainFlushReport {
    let staged = stage_terrain_edits(doc);
    let report = write_terrain_edits(&staged, content_root);
    mark_terrain_edits_saved(doc, &report);
    report
}

/// A human-readable note about unsaved streamed-terrain edits, or `None` when
/// there are none — what autosave records beside the crash-recovery file.
pub fn unsaved_terrain_note(doc: &SceneDoc) -> Option<String> {
    let mut lines = Vec::new();
    let mut total = 0usize;
    for entity in doc.streamed_terrain_entities() {
        let n = doc.terrain_dirty_tiles(entity).len();
        if n == 0 {
            continue;
        }
        total += n;
        lines.push(format!("  terrain {entity}: {n} tile(s)"));
    }
    if lines.is_empty() {
        return None;
    }
    Some(format!(
        "{total} unsaved terrain tile(s) were in memory when this recovery file was written.\n\
         Terrain edits live in the .inf_terrain asset and are only written by an explicit save, \
         so the recovered level's terrain is the LAST SAVED asset — these edits are gone:\n{}",
        lines.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samples::{
        streamed_terrain_data, streamed_terrain_scene, write_streamed_terrain_asset,
        STREAMED_TERRAIN_ASSET_GUID, STREAMED_TERRAIN_MPS, STREAMED_TERRAIN_RESOLUTION,
        STREAMED_TERRAIN_TERRAIN_GUID,
    };
    use crate::terrain_stream::EditorTerrainStreams;
    use glam::{DVec2, DVec3};
    use inf_terrain::{brush_wants, BrushOp, BrushParams, Stroke, TileGrid};

    /// A content root holding the generated `.inf_terrain`, plus the scene.
    fn fixture() -> (tempfile::TempDir, SceneDoc, EditorTerrainStreams) {
        let dir = tempfile::tempdir().unwrap();
        write_streamed_terrain_asset(dir.path()).unwrap();
        let doc = streamed_terrain_scene();
        let mut streams = EditorTerrainStreams::new();
        streams.set_content_root(Some(dir.path().to_path_buf()));
        (dir, doc, streams)
    }

    fn grid() -> TileGrid {
        TileGrid::new(STREAMED_TERRAIN_RESOLUTION, STREAMED_TERRAIN_MPS)
    }

    /// A tile's canonical blob — the byte-identity a restore is judged on.
    fn blob_of(tile: &inf_terrain::TerrainTile) -> Vec<u8> {
        inf_terrain::asset::encode_tile(tile).unwrap()
    }

    /// Page the brush footprint in, then lay one dab — the host's gesture, minus
    /// the raycast.
    fn dab(
        doc: &mut SceneDoc,
        streams: &mut EditorTerrainStreams,
        stroke: &mut Stroke,
        center: DVec2,
        radius: f64,
        op: BrushOp,
    ) {
        let paged =
            streams.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, doc, center, radius);
        assert!(paged.is_some(), "the terrain must be streamed");
        doc.sculpt_apply_dab(
            STREAMED_TERRAIN_TERRAIN_GUID,
            stroke,
            op,
            BrushParams::new(center, radius, 4.0),
        );
    }

    fn ensure_stream(doc: &SceneDoc, streams: &mut EditorTerrainStreams) {
        let world = doc.world();
        let e = world.entity_of(STREAMED_TERRAIN_TERRAIN_GUID).unwrap();
        let terrain = world
            .world()
            .get::<inf_ecs::components::Terrain>(e)
            .unwrap()
            .clone();
        assert!(streams.ensure(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &terrain,
            DVec3::ZERO,
            DVec3::new(64.0, 40.0, 64.0)
        ));
    }

    /// **Footprint residency (deliverable 1).** A dab over ground that is not
    /// resident pages exactly the footprint + one tile of margin into the
    /// DOCUMENT, then edits it.
    #[test]
    fn a_dab_pages_its_footprint_before_editing() {
        let (_dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);

        // Nothing is in the document's working set yet — that is the whole point
        // of a streamed terrain.
        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        assert!(data.is_empty(), "the document starts with no tiles");

        let center = DVec2::new(100.0, 100.0);
        let radius = 6.0;
        let mut stroke = Stroke::begin();
        dab(
            &mut doc,
            &mut streams,
            &mut stroke,
            center,
            radius,
            BrushOp::Raise,
        );

        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        let wants = brush_wants(grid(), center, radius);
        for key in &wants {
            assert!(
                data.is_resident(*key),
                "{key:?} in the footprint was not paged in"
            );
        }
        // …and nothing beyond it: paging is footprint-shaped, not whole-terrain.
        assert_eq!(
            data.tile_count(),
            wants.len(),
            "paged more than the footprint + margin"
        );
        assert!(doc.has_unsaved_terrain_edits());
    }

    /// **Equivalence (gate a).** The same brush script over a streamed terrain and
    /// over the identical terrain inline produces **byte-identical** level-0
    /// tiles. Editing an asset-backed terrain is not a different kind of editing.
    #[test]
    fn sculpting_streamed_equals_sculpting_inline() {
        let (_dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);

        // The script: a few dabs of different ops across a tile seam.
        let script: Vec<(DVec2, f64, BrushOp)> = vec![
            (DVec2::new(100.0, 100.0), 9.0, BrushOp::Raise),
            (DVec2::new(108.0, 100.0), 9.0, BrushOp::Raise),
            (DVec2::new(104.0, 106.0), 7.0, BrushOp::Lower),
            (
                DVec2::new(104.0, 102.0),
                11.0,
                BrushOp::Smooth { iterations: 2 },
            ),
        ];

        let mut stroke = Stroke::begin();
        for &(c, r, op) in &script {
            dab(&mut doc, &mut streams, &mut stroke, c, r, op);
        }
        doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke);

        // The inline twin: the SAME authored terrain, fully in memory.
        let mut inline = streamed_terrain_data();
        inline.clear_dirty();
        let mut inline_stroke = Stroke::begin();
        for &(c, r, op) in &script {
            inline_stroke.add_dab(&mut inline, op, BrushParams::new(c, r, 4.0));
        }

        let (streamed, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        assert!(streamed.tile_count() > 0);
        for (&coord, tile) in streamed.tiles() {
            assert_eq!(
                Some(tile),
                inline.get_tile(coord),
                "tile {coord:?} differs between the streamed and inline edits"
            );
        }
    }

    /// **Save write-back (deliverable 2 / gate b).** Ctrl+S folds the dirty tiles
    /// into the asset; a fresh streamer over the rewritten file serves the edited
    /// bytes; and the whole image equals a full rebuild of the equivalent terrain.
    #[test]
    fn saving_writes_the_edits_back_and_a_fresh_stream_reads_them() {
        let (dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);

        let center = DVec2::new(100.0, 100.0);
        let mut stroke = Stroke::begin();
        dab(
            &mut doc,
            &mut streams,
            &mut stroke,
            center,
            9.0,
            BrushOp::Raise,
        );
        doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke);

        let expect = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0
            .height_at(center)
            .unwrap();

        let report = flush_terrain_edits(&mut doc, dir.path());
        assert_eq!(report.written.len(), 1);
        assert!(report.tiles() > 0);
        assert!(!doc.has_unsaved_terrain_edits(), "the marks must clear");

        // A FRESH streamer over the rewritten asset (the reload path).
        let mut fresh = EditorTerrainStreams::new();
        fresh.set_content_root(Some(dir.path().to_path_buf()));
        let mut fresh_doc = streamed_terrain_scene();
        ensure_stream(&fresh_doc, &mut fresh);
        // Page the same footprint (no dab — just residency) and read it back.
        fresh.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut fresh_doc, center, 9.0);
        let (data, _) = fresh_doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        assert_eq!(
            data.height_at(center),
            Some(expect),
            "the reloaded asset does not serve the saved edit"
        );
    }

    /// **The biome twin of the write-back gate (P19.2).**
    ///
    /// Biome ids ride *inside* the tile blob, so nothing in this module mentions
    /// them — `TerrainEdits::from_dirty` clones whole tiles and carries every
    /// layer for free. That is exactly why it needs a test: "for free" is a claim
    /// about a seam (`get_tile_mut` stamps and dirties), and a future optimization
    /// that wrote ids through an immutable back door would drop every biome stroke
    /// on save without a single existing test noticing.
    ///
    /// Three claims in one script: a biome-only edit **schedules** a write-back;
    /// the rewritten asset **serves** the ids back to a fresh streamer; and the
    /// coarse pyramid pages stay **unpainted** (a decimated tile is a streaming
    /// page, not authored content — the same rule the weights and the data maps
    /// keep).
    #[test]
    fn a_biome_only_edit_writes_back_and_a_fresh_stream_reads_it() {
        use inf_terrain::BiomeStroke;

        let (dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);

        let center = DVec2::new(100.0, 100.0);
        let radius = 9.0;
        assert!(
            streams
                .page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, center, radius)
                .is_some(),
            "the terrain must be streamed"
        );

        let heights_before: Vec<f64> = {
            let (data, _) = doc
                .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
                .unwrap();
            data.tiles()
                .flat_map(|(_, t)| t.heights().iter().map(|&h| h as f64))
                .collect()
        };

        let mut stroke = BiomeStroke::begin(3);
        doc.biome_apply_dab(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &mut stroke,
            BrushParams::new(center, radius, 1.0),
        );
        assert!(
            doc.edit_commit_biome(STREAMED_TERRAIN_TERRAIN_GUID, stroke),
            "the stroke must record one undo step"
        );
        assert!(
            doc.has_unsaved_terrain_edits(),
            "a biome-only edit must schedule a write-back"
        );

        // Not one height moved — this really is a biome-only edit.
        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        let heights_after: Vec<f64> = data
            .tiles()
            .flat_map(|(_, t)| t.heights().iter().map(|&h| h as f64))
            .collect();
        assert_eq!(heights_before, heights_after, "a biome edit moved a height");
        assert!(!data.biomes_are_default(), "…but the ids did move");
        assert_eq!(
            data.biome_at(center),
            Some(3),
            "the brush centre must carry the painted id"
        );

        let report = flush_terrain_edits(&mut doc, dir.path());
        assert_eq!(report.written.len(), 1);
        assert!(report.tiles() > 0);
        assert!(!doc.has_unsaved_terrain_edits(), "the marks must clear");

        // A FRESH streamer over the rewritten asset serves the ids back.
        let mut fresh = EditorTerrainStreams::new();
        fresh.set_content_root(Some(dir.path().to_path_buf()));
        let mut fresh_doc = streamed_terrain_scene();
        ensure_stream(&fresh_doc, &mut fresh);
        fresh.page_brush_footprint(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &mut fresh_doc,
            center,
            radius,
        );
        let (reloaded, _) = fresh_doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        assert_eq!(
            reloaded.biome_at(center),
            Some(3),
            "the reloaded asset does not serve the saved biome ids"
        );

        // The rewritten asset's COARSE pages carry no ids: a decimated tile is a
        // streaming page, never authored content.
        let bytes = std::fs::read(dir.path().join("World.inf_terrain")).unwrap();
        let asset = inf_terrain::TerrainAssetReader::new(bytes.as_slice()).unwrap();
        let mut coarse_seen = 0usize;
        for key in asset.keys().filter(|k| !k.is_lod0()) {
            let tile = asset.tile(key).unwrap().expect("directory entry");
            assert!(tile.biomes_are_default(), "coarse {key:?} leaked biome ids");
            coarse_seen += 1;
        }
        assert!(coarse_seen > 0, "the fixture must have a pyramid to check");
    }

    /// **Undo of a biome stroke restores the saved bytes exactly (P19.2).** The
    /// blob-level twin of the splat/data-map undo gates: a stroke that
    /// materialized a tile's sparse id buffer must undo back to a tile that
    /// encodes byte-identically to the one that was never painted.
    #[test]
    fn undoing_a_biome_stroke_restores_byte_identical_tiles() {
        use inf_terrain::BiomeStroke;

        let (_dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);
        let center = DVec2::new(100.0, 100.0);
        streams.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, center, 9.0);

        let before: Vec<((i32, i32), Vec<u8>)> = {
            let (data, _) = doc
                .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
                .unwrap();
            data.tiles().map(|(&c, t)| (c, blob_of(t))).collect()
        };

        let mut stroke = BiomeStroke::begin(7);
        doc.biome_apply_dab(
            STREAMED_TERRAIN_TERRAIN_GUID,
            &mut stroke,
            BrushParams::new(center, 9.0, 1.0),
        );
        assert!(doc.edit_commit_biome(STREAMED_TERRAIN_TERRAIN_GUID, stroke));

        doc.undo();
        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        let after: Vec<((i32, i32), Vec<u8>)> =
            data.tiles().map(|(&c, t)| (c, blob_of(t))).collect();
        assert_eq!(
            before, after,
            "undo did not restore the tiles byte-for-byte"
        );
        assert!(
            data.biomes_are_default(),
            "the materialized id buffers must be dropped back to the sparse default"
        );
        // THE RULE (asserted by the height twin for the same reason): an undo is
        // an edit against the asset. The restored tiles are dirty again, so the
        // next save writes the restoration back — otherwise undoing a stroke
        // would leave the painted ids on disk while the document showed clean
        // ground, and only a reload would reveal it.
        assert!(
            doc.has_unsaved_terrain_edits(),
            "undo-restored tiles must be re-marked dirty"
        );
        assert!(!doc
            .terrain_dirty_tiles(STREAMED_TERRAIN_TERRAIN_GUID)
            .is_empty());

        // …and redo reproduces the stroke.
        doc.redo();
        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        assert_eq!(data.biome_at(center), Some(7));
    }

    /// **The no-dirty save (gate g).** A save with no terrain edits does not
    /// rewrite the asset at all — the file's bytes *and* its modification time are
    /// untouched.
    #[test]
    fn a_save_with_no_terrain_edits_skips_the_asset_write() {
        let (dir, mut doc, _streams) = fixture();
        let path = dir.path().join("World.inf_terrain");
        let before_bytes = std::fs::read(&path).unwrap();
        let before_mtime = std::fs::metadata(&path).unwrap().modified().unwrap();

        assert!(stage_terrain_edits(&doc).is_empty(), "nothing to stage");
        let report = flush_terrain_edits(&mut doc, dir.path());
        assert!(report.is_empty(), "a clean save must write nothing");

        assert_eq!(std::fs::read(&path).unwrap(), before_bytes);
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            before_mtime,
            "the asset was rewritten by a save that changed no terrain"
        );
    }

    /// **Authoring extends the asset (gate e).** A Raise past the authored extent
    /// creates a tile, and after a save it is in the asset's directory.
    #[test]
    fn an_authoring_op_extends_the_asset_on_save() {
        let (dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);
        let path = dir.path().join("World.inf_terrain");
        let before = inf_terrain::open_file_tile_store(&path)
            .unwrap()
            .tile_count();

        // The generated terrain is 16×16 tiles of 16 m => it ends at 256 m.
        let center = DVec2::new(300.0, 300.0);
        let mut stroke = Stroke::begin();
        dab(
            &mut doc,
            &mut streams,
            &mut stroke,
            center,
            8.0,
            BrushOp::Raise,
        );
        doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke);

        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        let authored: Vec<(i32, i32)> = data.tiles().map(|(&c, _)| c).collect();
        assert!(
            authored.iter().any(|&(tx, tz)| tx >= 16 || tz >= 16),
            "Raise must have authored ground outside the asset's extent"
        );

        flush_terrain_edits(&mut doc, dir.path());
        let store = inf_terrain::open_file_tile_store(&path).unwrap();
        assert!(store.tile_count() > before, "the directory did not grow");
        for &c in &authored {
            if c.0 >= 16 || c.1 >= 16 {
                assert!(
                    store.tile_bytes(TileKey::lod0(c)).is_some(),
                    "new tile {c:?} is not in the saved asset"
                );
            }
        }
    }

    /// **A neighbourhood op never authors past the extent.** Smooth/Flatten only
    /// touch existing ground, so brushing off the edge of a streamed terrain adds
    /// no tiles to the asset.
    #[test]
    fn neighbourhood_ops_never_grow_the_asset() {
        let (dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);
        let path = dir.path().join("World.inf_terrain");
        let before = inf_terrain::open_file_tile_store(&path)
            .unwrap()
            .tile_count();

        let mut stroke = Stroke::begin();
        dab(
            &mut doc,
            &mut streams,
            &mut stroke,
            DVec2::new(300.0, 300.0),
            8.0,
            BrushOp::Smooth { iterations: 1 },
        );
        assert!(
            !doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke),
            "smoothing empty space must change nothing"
        );
        flush_terrain_edits(&mut doc, dir.path());
        assert_eq!(
            inf_terrain::open_file_tile_store(&path)
                .unwrap()
                .tile_count(),
            before
        );
    }

    /// **Undo across residency (deliverable 3 / gate c).** 50 sculpt steps on a
    /// streamed terrain undo to a **byte-identical** terrain — and every tile an
    /// undo restored is re-marked dirty, because an undo is itself an edit against
    /// the asset.
    #[test]
    fn fifty_step_undo_restores_byte_identical_streamed_terrain() {
        let (dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);

        // Page the working area in and snapshot the pristine bytes.
        let base = DVec2::new(96.0, 96.0);
        streams.page_brush_footprint(STREAMED_TERRAIN_TERRAIN_GUID, &mut doc, base, 40.0);
        let pristine: Vec<((i32, i32), Vec<u8>)> = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0
            .tiles()
            .map(|(&c, t)| (c, blob_of(t)))
            .collect();
        assert!(!pristine.is_empty());

        for step in 0..50 {
            let c = base + DVec2::new((step % 7) as f64 * 3.0, (step / 7) as f64 * 3.0);
            let op = if step % 3 == 0 {
                BrushOp::Lower
            } else {
                BrushOp::Raise
            };
            let mut stroke = Stroke::begin();
            dab(&mut doc, &mut streams, &mut stroke, c, 5.0, op);
            assert!(doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke));
        }
        for _ in 0..50 {
            assert!(doc.undo());
        }

        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        for (coord, bytes) in &pristine {
            let tile = data.get_tile(*coord).expect("tile still resident");
            assert_eq!(
                &blob_of(tile),
                bytes,
                "tile {coord:?} did not restore byte-identically"
            );
        }
        // THE RULE: an undo is an edit against the asset, so the restored tiles
        // are dirty again and the next save writes the restoration back.
        assert!(
            doc.has_unsaved_terrain_edits(),
            "undo-restored tiles must be re-marked dirty"
        );
        assert!(!doc
            .terrain_dirty_tiles(STREAMED_TERRAIN_TERRAIN_GUID)
            .is_empty());

        // …and saving after the undo restores the asset to its original bytes —
        // the WHOLE FILE, not merely the tiles the test happened to look at: the
        // pyramid ancestors the strokes dirtied must decimate back to exactly what
        // they were, or the coarse rings would silently keep the sculpt.
        let path = dir.path().join("World.inf_terrain");
        let file_before = std::fs::read(&path).unwrap();
        flush_terrain_edits(&mut doc, dir.path());
        let store = inf_terrain::open_file_tile_store(&path).unwrap();
        for (coord, bytes) in &pristine {
            assert_eq!(
                store.tile_bytes(TileKey::lod0(*coord)).unwrap(),
                bytes.as_slice(),
                "the saved asset did not return to its pre-stroke bytes at {coord:?}"
            );
        }
        drop(store);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            file_before,
            "50 strokes + 50 undos + a save did not reproduce the .inf_terrain byte for byte"
        );
    }

    /// Redo replays the stroke, and the round trip is stable.
    #[test]
    fn undo_then_redo_returns_the_edited_terrain() {
        let (_dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);
        let center = DVec2::new(100.0, 100.0);
        let mut stroke = Stroke::begin();
        dab(
            &mut doc,
            &mut streams,
            &mut stroke,
            center,
            9.0,
            BrushOp::Raise,
        );
        assert!(doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke));

        let after = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0
            .height_at(center);
        assert!(doc.undo());
        let undone = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0
            .height_at(center);
        assert_ne!(after, undone);
        assert!(doc.redo());
        assert_eq!(
            doc.terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
                .unwrap()
                .0
                .height_at(center),
            after
        );
    }

    /// A terrain whose asset is missing from the content root is **reported**, and
    /// its edits stay dirty (so the next save retries) rather than being dropped.
    #[test]
    fn an_unresolvable_asset_is_reported_and_keeps_its_edits() {
        let (_dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);
        let mut stroke = Stroke::begin();
        dab(
            &mut doc,
            &mut streams,
            &mut stroke,
            DVec2::new(100.0, 100.0),
            9.0,
            BrushOp::Raise,
        );
        doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke);

        let empty = tempfile::tempdir().unwrap();
        let report = flush_terrain_edits(&mut doc, empty.path());
        assert_eq!(report.unwritten.len(), 1);
        assert_eq!(report.unwritten[0].entity, STREAMED_TERRAIN_TERRAIN_GUID);
        assert!(report.unwritten[0].reason.contains("not under"));
        assert!(report.has_failures());
        assert!(report.written.is_empty());
        assert!(
            doc.has_unsaved_terrain_edits(),
            "a failed write must not clear the write-back marks"
        );

        // THE STARVATION CONDITION (P16.4b audit). The level itself saved, so the
        // document is CLEAN — but terrain edits survive only in memory. Both Ring-2
        // gates key on exactly this pair: autosave must still fire (so the recovery
        // note is written) and the recovery file must NOT be cleared.
        doc.mark_saved();
        assert!(!doc.is_dirty(), "the level saved");
        assert!(
            doc.has_unsaved_terrain_edits(),
            "…while terrain edits are still only in memory"
        );
        assert!(
            !(!doc.is_dirty() && !doc.has_unsaved_terrain_edits()),
            "the autosave early-out would fire and starve the recovery note"
        );
        assert!(
            unsaved_terrain_note(&doc).is_some(),
            "…and there must be a note for it to write"
        );
    }

    /// **THE PLAY → STOP REGRESSION (P16.4b audit, blocking).**
    ///
    /// `SimSession::enter` snapshots the world and `exit` restores it wholesale.
    /// If that snapshot went through the *file* projection it would arrive with a
    /// streamed terrain's working set **stripped**, so Play → Stop would silently
    /// delete every unsaved sculpt — and the surviving undo stack would then
    /// replay height deltas into tiles `revert_delta` recreates *flat*, which is
    /// the exact corruption the design note at the top of this module warns about.
    #[test]
    fn play_then_stop_preserves_unsaved_streamed_terrain_edits() {
        use crate::simulate::SimSession;

        let (_dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);

        let center = DVec2::new(100.0, 100.0);
        let mut stroke = Stroke::begin();
        dab(
            &mut doc,
            &mut streams,
            &mut stroke,
            center,
            9.0,
            BrushOp::Raise,
        );
        assert!(doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke));

        // The pre-play truth: the edited bytes, the dirty set, and the height.
        let before: Vec<((i32, i32), Vec<u8>)> = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0
            .tiles()
            .map(|(&c, t)| (c, blob_of(t)))
            .collect();
        let dirty_before = doc.terrain_dirty_tiles(STREAMED_TERRAIN_TERRAIN_GUID);
        let height_before = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0
            .height_at(center);
        assert!(!dirty_before.is_empty() && !before.is_empty());

        // Play → (a step) → Stop.
        let mut session = SimSession::enter(&mut doc, Vec::new(), DVec2::new(0.0, -9.81), 60.0);
        session.step_once(&mut doc, Default::default());
        session.exit(&mut doc);

        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        let after: Vec<((i32, i32), Vec<u8>)> =
            data.tiles().map(|(&c, t)| (c, blob_of(t))).collect();
        assert_eq!(after, before, "Play → Stop changed the terrain's bytes");
        assert_eq!(data.height_at(center), height_before);
        assert_eq!(
            doc.terrain_dirty_tiles(STREAMED_TERRAIN_TERRAIN_GUID),
            dirty_before,
            "Play → Stop dropped the write-back marks — the edits would never save"
        );
        assert!(doc.has_unsaved_terrain_edits());

        // …and the undo stack still works against the restored tiles: undoing the
        // stroke must reach the ORIGINAL heightfield, not a flat recreation.
        assert!(doc.undo());
        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap();
        let undone = data.height_at(center).expect("tile still resident");
        assert_ne!(Some(undone), height_before);
        assert!(
            (undone - crate::samples::streamed_terrain_height(center.x, center.y)).abs() < 1e-3,
            "undo after Play → Stop landed on {undone}, not the authored surface — the tiles \
             were recreated flat"
        );
    }

    /// **The concurrent-edit guard (P16.4b audit).** A save stages its tiles, then
    /// rewrites the asset with the document **unlocked**. Anything sculpted during
    /// that window was never written, so its mark must survive — clearing the
    /// whole dirty set would silently discard exactly the edits the user made
    /// while waiting for the save.
    #[test]
    fn a_save_keeps_the_marks_of_tiles_edited_while_it_was_writing() {
        let (dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);

        // Edit tile A and stage it — this is the "save begins" moment.
        let a = DVec2::new(40.0, 40.0);
        let mut stroke = Stroke::begin();
        dab(&mut doc, &mut streams, &mut stroke, a, 5.0, BrushOp::Raise);
        doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke);
        let staged = stage_terrain_edits(&doc);
        assert_eq!(staged.len(), 1);
        let staged_keys: Vec<TileKey> = staged[0].stamps.iter().map(|(k, _)| *k).collect();
        assert!(!staged_keys.is_empty());

        // …the user keeps sculpting while the (slow, unlocked) rewrite runs: a
        // FRESH tile B, and a re-touch of one of the staged tiles A.
        let b = DVec2::new(200.0, 200.0);
        let mut stroke = Stroke::begin();
        dab(&mut doc, &mut streams, &mut stroke, b, 5.0, BrushOp::Raise);
        doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke);
        let mut stroke = Stroke::begin();
        dab(&mut doc, &mut streams, &mut stroke, a, 5.0, BrushOp::Lower);
        doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke);

        // The rewrite lands (it only ever knew about the staged tiles), then marks.
        let report = write_terrain_edits(&staged, dir.path());
        assert_eq!(report.written.len(), 1);
        mark_terrain_edits_saved(&mut doc, &report);

        // Every tile touched after staging is STILL dirty — including the staged
        // ones that were re-edited, whose on-disk bytes are now stale.
        let still = doc.terrain_dirty_tiles(STREAMED_TERRAIN_TERRAIN_GUID);
        assert!(
            !still.is_empty(),
            "the whole dirty set was cleared — edits made during the write are gone"
        );
        let b_coord = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0
            .tile_coord_of(b.x, b.y);
        assert!(
            still.contains(&TileKey::lod0(b_coord)),
            "a tile first edited during the write lost its mark"
        );
        let a_coord = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0
            .tile_coord_of(a.x, a.y);
        assert!(
            still.contains(&TileKey::lod0(a_coord)),
            "a staged tile re-edited during the write lost its mark, so its newer \
             contents would never reach the asset"
        );

        // The next save writes exactly those, and then the document is clean.
        let report = flush_terrain_edits(&mut doc, dir.path());
        assert!(!report.written.is_empty());
        assert!(!doc.has_unsaved_terrain_edits());
        // And the asset really carries the second round of edits.
        let store =
            inf_terrain::open_file_tile_store(&dir.path().join("World.inf_terrain")).unwrap();
        let mut fresh = inf_terrain::TerrainData::new(
            STREAMED_TERRAIN_RESOLUTION,
            crate::samples::STREAMED_TERRAIN_MPS,
        );
        fresh.request_tiles(
            &[TileKey::lod0(a_coord), TileKey::lod0(b_coord)]
                .into_iter()
                .collect(),
            &store,
        );
        let live = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0;
        assert_eq!(fresh.height_at(a), live.height_at(a));
        assert_eq!(fresh.height_at(b), live.height_at(b));
    }

    /// A streamed terrain entity that is **deleted and undone** comes back with
    /// its unsaved working set intact — the guard on the strip rule, which must
    /// stay out of `record_of` (undo's snapshot) and live only in the file
    /// projection.
    #[test]
    fn deleting_and_undoing_a_streamed_terrain_keeps_its_working_set() {
        let (_dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);
        let center = DVec2::new(100.0, 100.0);
        let mut stroke = Stroke::begin();
        dab(
            &mut doc,
            &mut streams,
            &mut stroke,
            center,
            9.0,
            BrushOp::Raise,
        );
        doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke);

        let before: Vec<((i32, i32), Vec<u8>)> = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .unwrap()
            .0
            .tiles()
            .map(|(&c, t)| (c, blob_of(t)))
            .collect();
        let dirty_before = doc.terrain_dirty_tiles(STREAMED_TERRAIN_TERRAIN_GUID);

        doc.edit_delete(&[STREAMED_TERRAIN_TERRAIN_GUID]);
        assert!(doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .is_none());
        assert!(doc.undo(), "the delete must be undoable");

        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .expect("the terrain came back");
        let after: Vec<((i32, i32), Vec<u8>)> =
            data.tiles().map(|(&c, t)| (c, blob_of(t))).collect();
        assert_eq!(
            after, before,
            "undoing a streamed-terrain delete lost its working set"
        );
        assert_eq!(
            doc.terrain_dirty_tiles(STREAMED_TERRAIN_TERRAIN_GUID),
            dirty_before,
            "…and its write-back marks"
        );
    }

    /// The assets this engine actually produces round-trip their **own** pyramid
    /// shape through [`WRITE_BACK_PYRAMID`] — so the common case really is a
    /// partial recompute, not a silent re-plan. (See there for the non-default
    /// import case, which `warn_on_pyramid_reshape` makes loud.)
    #[test]
    fn a_default_import_writes_back_at_its_own_pyramid_depth() {
        let (dir, _doc, _streams) = fixture();
        let path = dir.path().join("World.inf_terrain");
        let source = inf_terrain::open_file_tile_store(&path).unwrap();
        let level0: std::collections::BTreeSet<(i32, i32)> = source
            .keys()
            .filter(|k| k.is_lod0())
            .map(|k| k.coord)
            .collect();
        let planned =
            inf_terrain::plan_pyramid(&level0, source.meters_per_sample(), WRITE_BACK_PYRAMID);
        assert_eq!(
            planned.len() as u32,
            source.lod_levels() - 1,
            "the write-back would re-plan a default asset's pyramid to a different depth"
        );
        assert!(planned.len() >= 2, "the fixture must have a real pyramid");
    }

    /// **P16.6.** A schema-v2 asset carries its own pyramid options, so a
    /// write-back re-plans with *those* — the non-default import can no longer be
    /// silently reshaped by a save.
    #[test]
    fn a_v2_asset_writes_back_with_its_own_recorded_options() {
        use inf_terrain::PyramidOptions;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Custom.inf_terrain");
        // A deliberately non-default shape: cap at two coarse levels.
        let opts = PyramidOptions {
            max_levels: 2,
            min_tiles: 1,
        };
        let data = inf_editor_core_samples_terrain();
        let pyramid = inf_terrain::build_pyramid(&data, opts);
        let asset = inf_terrain::build_terrain_asset(&data, &pyramid, opts).unwrap();
        inf_terrain::write_terrain_asset(&path, &asset).unwrap();

        let source = inf_terrain::open_file_tile_store(&path).unwrap();
        assert_eq!(
            source.pyramid_options(),
            Some(opts),
            "a v2 asset must report its own options"
        );
        assert_eq!(
            super::write_back_pyramid(&source),
            opts,
            "the write-back must re-plan with the recorded options, not the defaults"
        );
        assert_ne!(opts, super::WRITE_BACK_PYRAMID, "the fixture must differ");
        // …and the depth really is the capped one, not the default one.
        assert_eq!(source.lod_levels(), 1 + pyramid.len() as u32);
        assert_eq!(pyramid.len(), 2);

        // Round trip a rewrite: the new image keeps the recorded options, so the
        // NEXT save re-plans identically (the drift v2 exists to end).
        let mut edited = data.clone();
        inf_terrain::apply_brush(
            &mut edited,
            inf_terrain::BrushOp::Raise,
            inf_terrain::BrushParams::new(DVec2::new(20.0, 20.0), 8.0, 3.0),
        );
        let edits =
            inf_terrain::writeback::TerrainEdits::from_dirty(&edited, &edited.dirty_tiles());
        let out =
            inf_terrain::rewrite_terrain_asset(&source, &edits, super::write_back_pyramid(&source))
                .unwrap()
                .unwrap();
        assert_eq!(out.reader().pyramid_options(), Some(opts));
        assert_eq!(out.reader().lod_levels(), source.lod_levels());
    }

    /// A 16 × 16-tile heightfield at the sample's grid — big enough that the
    /// pyramid options visibly change its depth.
    fn inf_editor_core_samples_terrain() -> inf_terrain::TerrainData {
        crate::samples::streamed_terrain_data()
    }

    /// The autosave note names the terrain and the tile count — the honest
    /// recovery warning (deliverable 2).
    #[test]
    fn the_recovery_note_reports_unsaved_terrain_edits() {
        let (_dir, mut doc, mut streams) = fixture();
        ensure_stream(&doc, &mut streams);
        assert!(unsaved_terrain_note(&doc).is_none(), "clean ⇒ no note");

        let mut stroke = Stroke::begin();
        dab(
            &mut doc,
            &mut streams,
            &mut stroke,
            DVec2::new(100.0, 100.0),
            9.0,
            BrushOp::Raise,
        );
        doc.edit_commit_sculpt(STREAMED_TERRAIN_TERRAIN_GUID, stroke);
        let note = unsaved_terrain_note(&doc).expect("a note");
        assert!(note.contains(&STREAMED_TERRAIN_TERRAIN_GUID.to_string()));
        assert!(note.contains("LAST SAVED"));
        assert_eq!(
            doc.terrain_asset_of(STREAMED_TERRAIN_TERRAIN_GUID),
            Some(STREAMED_TERRAIN_ASSET_GUID)
        );
    }
}
