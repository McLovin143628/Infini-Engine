//! Editing a **voxel volume** in the editor (P21.2 deliverable 7): where the
//! carved chunks live, and the save path that writes them back.
//!
//! [`crate::terrain_edit`]'s twin, deliberately down to the function names —
//! stage / write / mark-saved / flush / unsaved-note — because the two halves of
//! one Ctrl+S must not develop two different sets of rules about what a failed
//! asset write means.
//!
//! # THE DESIGN NOTE — the working set is the shared store, not the document
//!
//! A streamed *terrain*'s working set is the ECS `Terrain` component's
//! `TerrainData`, and that is what makes `terrain_edit` a module about
//! `SceneDoc`. A *voxel volume*'s cannot be: **scene schema v19 is frozen**, so
//! a `.inf_lvl` has no slot for SDF chunks and the `VoxelVolume` component
//! carries an asset reference and nothing else. The chunks live in
//! [`EditorVoxelVolumes`], the shared store the viewport pages and the undo
//! history reaches through a [`SharedVoxelVolumes`] handle (see there for why
//! that, and not a move into the document or a replay journal).
//!
//! So this module takes the store where `terrain_edit` takes the document, and
//! every other rule survives unchanged:
//!
//! * **Authority:** the store's `VoxelData`. It is what the brush cuts, what the
//!   mesher reads, and what a save writes.
//! * **Undo:** `EditCommand::CarveVoxels`, already reversible against the same
//!   store — *zero* new undo machinery.
//! * **Save:** Ctrl+S drains each volume's dirty chunks into its `.inf_voxel`
//!   through [`inf_voxel::rewrite_voxel_asset`], which copies every untouched
//!   chunk's blob **verbatim**. The `.inf_lvl` never carries voxel data at all,
//!   so the level stays kilobytes and the asset stays the single source of truth.
//! * **Autosave does not write assets.** Same reasoning, same consequence: a
//!   debounced whole-payload `.inf_voxel` rewrite every few seconds would commit
//!   cuts the author has not chosen to keep. Autosave records that unsaved carve
//!   edits *existed* instead — [`unsaved_voxel_note`], written beside the
//!   crash-recovery file next to the terrain's.
//!
//! # The heightfield half rides the TERRAIN write-back, on purpose
//!
//! A carve that breaks the surface opens a **hole mask** on the terrain's tiles,
//! and holes are a tile layer: `TerrainEdits::from_dirty` clones whole tiles, so
//! the mouths reach the `.inf_terrain` through the write-back that was already
//! there, with nothing in this file mentioning them. That is a claim about a
//! seam rather than an obvious truth — `terrain_edit`'s
//! `a_carve_writes_its_cave_mouths_into_the_asset` is what pins it, exactly as
//! the biome twin pins the P19.2 layer.
//!
//! Which is also why the two flushes are **ordered** in the Ring-2 save path:
//! terrain first, then voxels. Nothing in either write reads the other, but a
//! cave and its mouth are one edit to the author, and a save that reported the
//! rock written while the ground failed would be describing half a cut.
//!
//! # [`inline_hole_warnings`] — the defensive half
//!
//! The carve tools refuse to open a mouth on an inline terrain, because schema
//! v19 pins `TerrainData`'s wire form at a layout with no hole mask. This finds a
//! document that arrived in the lossy state anyway (an older session, a
//! hand-edited level, a terrain converted back to inline after a carve) and says
//! so **before** the save seals every mouth in it. The rule itself is
//! [`inf_voxel::inline_hole_advisory`] — Ring 0, pure, unit-tested on all three
//! OSes; this is the save path's call site, and the viewport's projection is the
//! other one.

use std::path::{Path, PathBuf};

use inf_voxel::{ChunkKey, VoxelEdits};
use uuid::Uuid;

use crate::scene::SceneDoc;
use crate::voxel_store::EditorVoxelVolumes;

/// One volume's pending write-back, lifted out of the shared store.
///
/// Owned and self-contained, for [`crate::terrain_edit::StagedTerrainEdit`]'s
/// reason: the caller stages under the store lock, **releases it**, and then does
/// the slow whole-payload rewrite. The viewport thread takes the same lock every
/// frame to mesh and to page, so disk IO must never be held under it.
#[derive(Debug, Clone)]
pub struct StagedVoxelEdit {
    /// The `VoxelVolume` entity whose chunks moved.
    pub entity: Uuid,
    /// The `.inf_voxel` asset GUID those chunks belong to.
    pub asset: Uuid,
    /// The chunk changes to fold in.
    pub edits: VoxelEdits,
    /// **The concurrent-edit guard**: every staged chunk's change stamp at the
    /// moment it was lifted out of the store.
    ///
    /// The rewrite runs unlocked, so the author can keep carving while it runs.
    /// Clearing the whole dirty set afterwards would throw those cuts away — they
    /// were never written. Marking is therefore per key and conditional on the
    /// stamp still matching
    /// ([`EditorVoxelVolumes::mark_written_back`]): a chunk carved during the
    /// write window stays dirty and the next save writes it.
    pub stamps: Vec<(ChunkKey, u64)>,
}

/// One asset a write-back pass rewrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenVoxel {
    /// The volume entity.
    pub entity: Uuid,
    /// The `.inf_voxel` that was rewritten.
    pub path: PathBuf,
    /// The chunks that actually reached disk, with the stamp each carried when it
    /// was staged — exactly what may now be un-marked.
    pub chunks: Vec<(ChunkKey, u64)>,
}

/// A volume whose carve edits could **not** be written, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnwrittenVoxel {
    /// The volume entity, whose chunks are still dirty (and therefore still
    /// protected from eviction, still reported unsaved, and still retried).
    pub entity: Uuid,
    /// The `.inf_voxel` asset GUID it references.
    pub asset: Uuid,
    /// Human-readable reason, for the save toast / Output Log.
    pub reason: String,
}

/// What one write-back pass did, per asset.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VoxelFlushReport {
    /// Each asset rewritten, with the chunks that landed.
    pub written: Vec<WrittenVoxel>,
    /// Volumes that had carve edits but could not be written — a missing asset,
    /// an IO error, a corrupt source. **Never silently dropped**: a save that
    /// could not write a cave is not a successful save of that cave, and the
    /// caller must say so.
    pub unwritten: Vec<UnwrittenVoxel>,
}

impl VoxelFlushReport {
    /// `true` when nothing was written and nothing failed.
    pub fn is_empty(&self) -> bool {
        self.written.is_empty() && self.unwritten.is_empty()
    }

    /// `true` when at least one volume's carve edits could not be written.
    pub fn has_failures(&self) -> bool {
        !self.unwritten.is_empty()
    }

    /// Total chunks folded into assets.
    pub fn chunks(&self) -> usize {
        self.written.iter().map(|w| w.chunks.len()).sum()
    }

    /// A one-line summary for the Output Log / the save toast.
    pub fn summary(&self) -> String {
        format!(
            "voxel write-back: {} chunk(s) into {} asset(s){}",
            self.chunks(),
            self.written.len(),
            if self.unwritten.is_empty() {
                String::new()
            } else {
                format!(", {} FAILED", self.unwritten.len())
            }
        )
    }
}

/// Lift every loaded volume's pending carve edits out of `volumes` (empty when
/// nothing is dirty — a save that carved nothing stages nothing and writes
/// nothing).
///
/// Takes `&EditorVoxelVolumes` and not `&mut`: staging **reads**. The dirty marks
/// survive until [`mark_voxel_edits_saved`] retires the ones that really landed,
/// which is what makes a failed write retry instead of vanish.
/// ([`VoxelEdits::from_dirty`] drains, which is right for a Ring-0 caller that
/// owns the volume and wrong for a save that can fail.)
pub fn stage_voxel_edits(volumes: &EditorVoxelVolumes) -> Vec<StagedVoxelEdit> {
    let mut out = Vec::new();
    for (entity, asset) in volumes.bound_volumes() {
        let dirty = volumes.dirty_chunks(entity);
        if dirty.is_empty() {
            continue;
        }
        let Some(slot) = volumes.slot(entity) else {
            continue;
        };
        let mut edits = VoxelEdits::default();
        let mut stamps = Vec::with_capacity(dirty.len());
        for key in dirty {
            // Resident ⇒ carved; dirty-but-absent ⇒ deleted, which is precisely
            // what `VoxelData::remove_chunk` leaves behind. The two halves fall
            // out of one walk, exactly as `TerrainEdits::from_dirty` splits on
            // residency — and the second half is why `sync_residency` refuses to
            // evict a dirty chunk: an evicted one would arrive here as a
            // *deletion* and the save would erase the carved cave from the asset,
            // not merely lose its unsaved carving. Nothing in the editor deletes a
            // chunk today (P21.3's excavation is the first door that will), so
            // this branch is currently reached only by that refusal failing —
            // which `flying_away_retains_an_unsaved_carve_rather_than_evicting_it`
            // is the gate on.
            match slot.data.get_chunk(key) {
                Some(c) => {
                    edits.changed.insert(key, c.clone());
                }
                None => {
                    edits.removed.insert(key);
                }
            }
            // The stamp of every staged chunk, so the mark step can tell "this is
            // the chunk I wrote" from "the author re-carved it while I was
            // writing". A deleted chunk stamps `0`, and a re-created one stamps
            // above it, so the guard holds through a delete → re-carve too.
            stamps.push((key, slot.data.chunk_version(key)));
        }
        if edits.is_empty() {
            continue;
        }
        out.push(StagedVoxelEdit {
            entity,
            asset,
            edits,
            stamps,
        });
    }
    out
}

/// Merge `staged` into the loose `.inf_voxel` assets under `content_root` and
/// rewrite them **atomically**.
///
/// Does no store work at all (it takes none), so it is safe to call with every
/// lock released. Each asset goes through the one sanctioned writer,
/// [`inf_voxel::write_voxel_asset`], and its `inf_asset` sidecar is restamped
/// over exactly the bytes written so the cook packs what is on disk.
/// **Infallible by design.** A failure is *per volume* and lands in
/// [`VoxelFlushReport::unwritten`] rather than aborting the pass: one unreadable
/// asset must not stop the others from being written, and — more importantly — a
/// failure has to reach the user as "this cave did not save" rather than as a
/// save that quietly did less than it claimed.
pub fn write_voxel_edits(staged: &[StagedVoxelEdit], content_root: &Path) -> VoxelFlushReport {
    let mut report = VoxelFlushReport::default();
    if staged.is_empty() {
        return report;
    }
    // One directory walk for the whole save, however many volumes it covers.
    let index = crate::voxel_store::voxel_paths_by_guid(content_root);

    for item in staged {
        let mut fail = |reason: String| {
            tracing::warn!(
                "inf-editor-core: voxel volume {} could not be written back — {reason}. Its \
                 carve edits stay in memory (still unsaved, still retried by the next save).",
                item.entity
            );
            report.unwritten.push(UnwrittenVoxel {
                entity: item.entity,
                asset: item.asset,
                reason,
            });
        };
        let Some(path) = index.get(&item.asset) else {
            fail(format!(
                "its .inf_voxel {} is not under {}",
                item.asset,
                content_root.display()
            ));
            continue;
        };
        let source = match inf_voxel::read_voxel_asset(path) {
            Ok(a) => a,
            Err(e) => {
                fail(format!("open {}: {e}", path.display()));
                continue;
            }
        };
        let rewritten = match inf_voxel::rewrite_voxel_asset(&source.reader(), &item.edits) {
            Ok(Some(a)) => a,
            // Nothing to write (an empty edit set slipped through) — not a
            // failure, and nothing to un-mark either.
            Ok(None) => continue,
            Err(e) => {
                fail(format!("rewrite {}: {e}", path.display()));
                continue;
            }
        };
        // Drop the source image before the rename: it is a whole copy of the
        // previous payload, and there is no reason to hold two plus the new one.
        drop(source);
        let bytes = match inf_voxel::write_voxel_asset(path, &rewritten) {
            Ok(b) => b,
            Err(e) => {
                fail(format!("write {}: {e}", path.display()));
                continue;
            }
        };
        if let Err(e) = inf_asset::AssetSidecar::new(
            inf_asset::AssetId(item.asset),
            inf_asset::AssetKind::VoxelVolume,
            inf_asset::ContentHash::of(bytes),
        )
        .save(path)
        {
            // The payload landed but its hash did not: report it rather than
            // clearing the marks over a half-written asset pair.
            fail(format!("write sidecar for {}: {e}", path.display()));
            continue;
        }
        report.written.push(WrittenVoxel {
            entity: item.entity,
            path: path.clone(),
            chunks: item.stamps.clone(),
        });
    }
    report
}

/// Clear the write-back marks of every chunk `report` says actually reached disk
/// — **and only if it has not been re-carved since it was staged**.
///
/// Two separate protections, both load-bearing and both `terrain_edit`'s:
///
/// * A volume whose write *failed* keeps every mark, so its edits stay reported
///   as unsaved and are retried by the next save.
/// * A *chunk* the author carved during the (unlocked, possibly multi-second)
///   rewrite keeps its mark, because the bytes on disk are older than the chunk
///   in memory.
///
/// Returns how many marks were cleared.
pub fn mark_voxel_edits_saved(
    volumes: &mut EditorVoxelVolumes,
    report: &VoxelFlushReport,
) -> usize {
    report
        .written
        .iter()
        .map(|w| volumes.mark_written_back(w.entity, &w.chunks))
        .sum()
}

/// Stage → write → mark, for a caller that can hold the store across the whole
/// operation (tests, the CLI). The editor's save path splits the three so the
/// disk IO happens with the store lock released.
pub fn flush_voxel_edits(
    volumes: &mut EditorVoxelVolumes,
    content_root: &Path,
) -> VoxelFlushReport {
    let staged = stage_voxel_edits(volumes);
    let report = write_voxel_edits(&staged, content_root);
    mark_voxel_edits_saved(volumes, &report);
    report
}

/// A human-readable note about unsaved carve edits, or `None` when there are
/// none — what autosave records beside the crash-recovery file, next to
/// [`crate::terrain_edit::unsaved_terrain_note`].
pub fn unsaved_voxel_note(volumes: &EditorVoxelVolumes) -> Option<String> {
    let mut lines = Vec::new();
    let mut total = 0usize;
    for (entity, _asset) in volumes.bound_volumes() {
        let n = volumes.dirty_chunks(entity).len();
        if n == 0 {
            continue;
        }
        total += n;
        lines.push(format!("  volume {entity}: {n} chunk(s)"));
    }
    if lines.is_empty() {
        return None;
    }
    Some(format!(
        "{total} unsaved voxel chunk(s) were in memory when this recovery file was written.\n\
         Carve edits live in the .inf_voxel asset and are only written by an explicit save, \
         so the recovered level's caves are the LAST SAVED asset — these cuts are gone:\n{}",
        lines.join("\n")
    ))
}

/// Every terrain in `doc` that is carrying cave mouths its container cannot
/// persist, as the line to show the author — **empty on every document written
/// before P21.2, and on every one the carve tools produced**.
///
/// The belt-and-braces half of the ruling. `CarveVerdict::RefusedInline` stops
/// the state being *created*; this reports a document that arrived in it, at the
/// one moment the loss actually happens — the save. Each line names the terrain
/// (so a multi-terrain world says *which*) and then the Ring-0 advisory, which
/// names the fix.
///
/// `asset_backed` is answered per terrain from `Terrain.asset.is_some()`, which
/// is exactly the question `.inf_terrain`-vs-`.inf_lvl` persistence turns on.
pub fn inline_hole_warnings(doc: &SceneDoc) -> Vec<String> {
    let mut out = Vec::new();
    for &guid in doc.order() {
        let Some((data, _)) = doc.terrain_data_and_origin(guid) else {
            continue;
        };
        let backed = doc.terrain_asset_of(guid).is_some();
        if let Some(note) = inf_voxel::inline_hole_advisory(data, backed) {
            out.push(format!("{}: {}", terrain_label(doc, guid), note.message()));
        }
    }
    out
}

/// How a terrain is named to the author: its `Name`, else its GUID.
///
/// The name and not the GUID, because the advisory's whole job is to point at a
/// thing in the Outliner — and the GUID fallback is there because an entity with
/// no `Name` must still be identifiable rather than anonymous.
fn terrain_label(doc: &SceneDoc, guid: Uuid) -> String {
    doc.entity_of(guid)
        .and_then(|e| doc.world().name_of(e))
        .map(str::to_owned)
        .unwrap_or_else(|| guid.to_string())
}

/// The **document** half of the Voxel tool's readout: what this level can be
/// carved into, and what it would refuse (P21.2 deliverable 5).
///
/// Camera-independent by construction — it asks the same questions
/// `SceneDoc::carve_verdict` does *except* "does this particular cut reach a
/// surface", which needs a pick. That is what lets a toolbar show the answer
/// before the first click instead of after the first refusal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VoxelLevelStatus {
    /// `VoxelVolume` entities in the document — the things a carve can target.
    pub volumes: usize,
    /// Terrains that can persist a cave mouth.
    pub asset_backed_terrains: usize,
    /// Terrains that cannot, by name: a surface-crossing cut over any of these
    /// is refused **whole**, so the author needs to know which one to convert.
    pub inline_terrains: Vec<String>,
    /// [`inline_hole_warnings`] — the document is already in the lossy state.
    pub advisories: Vec<String>,
}

/// Read [`VoxelLevelStatus`] off `doc`.
pub fn level_status(doc: &SceneDoc) -> VoxelLevelStatus {
    let mut out = VoxelLevelStatus::default();
    let world = doc.world();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        if world
            .world()
            .get::<inf_ecs::components::VoxelVolume>(e)
            .is_some()
        {
            out.volumes += 1;
        }
        // A terrain is anything with a `Terrain` component, asset-backed or not —
        // `terrain_data_and_origin` is the same door the verdict walks, so the
        // two cannot disagree about what counts.
        if doc.terrain_data_and_origin(guid).is_some() {
            if doc.terrain_asset_of(guid).is_some() {
                out.asset_backed_terrains += 1;
            } else {
                out.inline_terrains.push(terrain_label(doc, guid));
            }
        }
    }
    out.advisories = inline_hole_warnings(doc);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use inf_ecs::components::VoxelVolume;
    use inf_voxel::{
        VoxelChunk, VoxelData, VoxelOp, VoxelShape, VoxelStreamBudget, VoxelWantsParams,
    };

    /// Voxel edge length of the fixture volumes, metres.
    const VOXEL_M: f64 = 0.5;
    const ASSET: Uuid = Uuid::from_u128(0x2107_0001);
    const ENTITY: Uuid = Uuid::from_u128(0x2107_0002);

    /// A solid 2 × 2 × 2-chunk block of rock as a loose `.inf_voxel` + sidecar —
    /// the same fixture `tests/voxel_carve.rs` carves, written the way the
    /// editor's own writers leave it.
    fn write_rock(dir: &Path, name: &str, guid: Uuid) -> PathBuf {
        let mut v = VoxelData::new(VOXEL_M);
        for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
            v.insert_chunk(key, VoxelChunk::solid(1));
        }
        let asset = inf_voxel::build_voxel_asset(&v).unwrap();
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        let bytes = inf_voxel::write_voxel_asset(&path, &asset).unwrap();
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(guid),
            inf_asset::AssetKind::VoxelVolume,
            inf_asset::ContentHash::of(bytes),
        )
        .save(&path)
        .unwrap();
        path
    }

    /// A store rooted at a fresh content dir with the rock volume bound **and
    /// fully paged** (binding pages nothing — P21.2).
    fn fixture() -> (tempfile::TempDir, EditorVoxelVolumes) {
        let dir = tempfile::tempdir().unwrap();
        write_rock(dir.path(), "Rock.inf_voxel", ASSET);
        let mut store = EditorVoxelVolumes::new();
        store.set_content_root(Some(dir.path().to_path_buf()));
        assert!(store.ensure(ENTITY, &VoxelVolume::from_asset(ASSET)));
        let report = store.sync_camera(
            DVec3::splat(4.0),
            &VoxelWantsParams {
                radius_m: 1000.0,
                hysteresis: 0.0,
            },
            VoxelStreamBudget::default(),
        );
        assert!(report.loaded > 0, "the fixture volume must page in");
        assert!(
            !store.has_unsaved_edits(),
            "paging is not editing — a freshly loaded volume is clean"
        );
        (dir, store)
    }

    /// The cut every test below uses: a ball inside the rock block.
    fn cut() -> VoxelOp {
        VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(4.0, 4.0, 4.0),
            radius_m: 1.5,
        })
    }

    /// Solid samples across the volume — the "is the rock gone?" probe.
    fn solid_count(store: &EditorVoxelVolumes, entity: Uuid) -> usize {
        let data = &store.slot(entity).unwrap().data;
        data.chunks()
            .map(|(_, c)| {
                (0..inf_voxel::CHUNK_DIM)
                    .flat_map(|k| {
                        (0..inf_voxel::CHUNK_DIM)
                            .flat_map(move |j| (0..inf_voxel::CHUNK_DIM).map(move |i| (i, j, k)))
                    })
                    .filter(|&(i, j, k)| c.sample(i, j, k) < 0.0)
                    .count()
            })
            .sum()
    }

    /// **The save write-back gate.** Ctrl+S folds the carved chunks into the
    /// asset, and a fresh store over the rewritten file serves the cave back.
    #[test]
    fn saving_writes_the_carve_back_and_a_fresh_store_reads_it() {
        let (dir, mut store) = fixture();
        let before = solid_count(&store, ENTITY);

        assert!(store.carve(ENTITY, &cut()).is_some());
        assert!(store.has_unsaved_edits(), "a carve owes a save");
        let carved = solid_count(&store, ENTITY);
        assert!(carved < before, "the fixture cut removed no rock");

        let report = flush_voxel_edits(&mut store, dir.path());
        assert_eq!(report.written.len(), 1);
        assert!(report.chunks() > 0);
        assert!(!report.has_failures());
        assert!(!store.has_unsaved_edits(), "the marks must clear");
        assert!(report.summary().contains("chunk(s)"));

        // A FRESH store over the rewritten asset (the reload path).
        let mut fresh = EditorVoxelVolumes::new();
        fresh.set_content_root(Some(dir.path().to_path_buf()));
        assert!(fresh.ensure(ENTITY, &VoxelVolume::from_asset(ASSET)));
        fresh.sync_camera(
            DVec3::splat(4.0),
            &VoxelWantsParams {
                radius_m: 1000.0,
                hysteresis: 0.0,
            },
            VoxelStreamBudget::default(),
        );
        assert_eq!(
            solid_count(&fresh, ENTITY),
            carved,
            "the reloaded asset does not serve the saved carve"
        );
    }

    /// **The no-dirty save.** A save with no carve edits does not rewrite the
    /// asset at all — the file's bytes *and* its modification time are untouched.
    #[test]
    fn a_save_with_no_carve_edits_skips_the_asset_write() {
        let (dir, mut store) = fixture();
        let path = dir.path().join("Rock.inf_voxel");
        let before_bytes = std::fs::read(&path).unwrap();
        let before_mtime = std::fs::metadata(&path).unwrap().modified().unwrap();

        assert!(stage_voxel_edits(&store).is_empty(), "nothing to stage");
        let report = flush_voxel_edits(&mut store, dir.path());
        assert!(report.is_empty(), "a clean save must write nothing");

        assert_eq!(std::fs::read(&path).unwrap(), before_bytes);
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            before_mtime,
            "the asset was rewritten by a save that carved nothing"
        );
    }

    /// **The verbatim gate, at the editor's own door.** A carve touches a few
    /// chunks; every chunk it did not touch comes out of the save byte-for-byte
    /// identical. Ring 0 pins the rewrite; this pins that the *staging* names
    /// only what moved — a `from_dirty` that over-reported would re-encode the
    /// whole cave on every save and nothing else would notice.
    #[test]
    fn untouched_chunks_are_not_rewritten_by_a_save() {
        let (dir, mut store) = fixture();
        let path = dir.path().join("Rock.inf_voxel");
        let before = inf_voxel::read_voxel_asset(&path).unwrap();

        assert!(store.carve(ENTITY, &cut()).is_some());
        let staged = stage_voxel_edits(&store);
        assert_eq!(staged.len(), 1);
        let touched: Vec<ChunkKey> = staged[0].edits.changed.keys().copied().collect();
        assert!(!touched.is_empty());
        assert!(
            touched.len() < before.reader().chunk_count(),
            "the fixture cut must leave some chunks alone, or this proves nothing"
        );

        flush_voxel_edits(&mut store, dir.path());
        let after = inf_voxel::read_voxel_asset(&path).unwrap();
        assert_eq!(after.reader().chunk_count(), before.reader().chunk_count());
        for key in before.reader().keys() {
            if touched.contains(&key) {
                assert_ne!(
                    after.reader().chunk_bytes(key),
                    before.reader().chunk_bytes(key),
                    "{key:?} was carved and did not change on disk"
                );
                continue;
            }
            assert_eq!(
                after.reader().chunk_bytes(key),
                before.reader().chunk_bytes(key),
                "{key:?} was re-encoded by a save that never touched it"
            );
        }
    }

    /// A volume whose asset is missing from the content root is **reported**, and
    /// its edits stay dirty (so the next save retries) rather than being dropped.
    #[test]
    fn an_unresolvable_asset_is_reported_and_keeps_its_edits() {
        let (_dir, mut store) = fixture();
        assert!(store.carve(ENTITY, &cut()).is_some());

        let empty = tempfile::tempdir().unwrap();
        let report = flush_voxel_edits(&mut store, empty.path());
        assert_eq!(report.unwritten.len(), 1);
        assert_eq!(report.unwritten[0].entity, ENTITY);
        assert_eq!(report.unwritten[0].asset, ASSET);
        assert!(report.unwritten[0].reason.contains("not under"));
        assert!(report.has_failures());
        assert!(report.written.is_empty());
        assert!(
            store.has_unsaved_edits(),
            "a failed write must not clear the write-back marks"
        );
        assert!(report.summary().contains("FAILED"));
    }

    /// **The concurrent-edit guard.** A save stages its chunks, then rewrites the
    /// asset with the store **unlocked**. Anything carved during that window was
    /// never written, so its mark must survive — clearing the whole dirty set
    /// would silently discard exactly the cuts the author made while waiting.
    #[test]
    fn a_save_keeps_the_marks_of_chunks_carved_while_it_was_writing() {
        let (dir, mut store) = fixture();

        // Carve near the origin corner and stage it — "the save begins".
        assert!(store
            .carve(
                ENTITY,
                &VoxelOp::carve(VoxelShape::Sphere {
                    center: DVec3::new(1.0, 1.0, 1.0),
                    radius_m: 1.0,
                })
            )
            .is_some());
        let staged = stage_voxel_edits(&store);
        assert_eq!(staged.len(), 1);
        let staged_keys: Vec<ChunkKey> = staged[0].stamps.iter().map(|(k, _)| *k).collect();
        assert!(!staged_keys.is_empty());

        // …the author keeps carving while the (slow, unlocked) rewrite runs: a
        // FAR corner (fresh chunks) and a re-cut of one staged chunk.
        assert!(store
            .carve(
                ENTITY,
                &VoxelOp::carve(VoxelShape::Sphere {
                    center: DVec3::new(20.0, 20.0, 20.0),
                    radius_m: 1.0,
                })
            )
            .is_some());
        assert!(store
            .carve(
                ENTITY,
                &VoxelOp::carve(VoxelShape::Sphere {
                    center: DVec3::new(1.0, 1.0, 1.0),
                    radius_m: 1.5,
                })
            )
            .is_some());

        // The rewrite lands (it only ever knew about the staged chunks), then marks.
        let report = write_voxel_edits(&staged, dir.path());
        assert_eq!(report.written.len(), 1);
        let cleared = mark_voxel_edits_saved(&mut store, &report);

        let still = store.dirty_chunks(ENTITY);
        assert!(
            !still.is_empty(),
            "the whole dirty set was cleared — {cleared} mark(s) went, and the cuts \
             made during the write are gone"
        );
        for key in &staged_keys {
            assert!(
                still.contains(key),
                "{key:?} was re-carved during the write and lost its mark, so its \
                 newer contents would never reach the asset"
            );
        }

        // The next save writes exactly those, and then the store is clean.
        let report = flush_voxel_edits(&mut store, dir.path());
        assert!(!report.written.is_empty());
        assert!(!store.has_unsaved_edits());
        // …and the asset really carries the second round of cuts.
        let mut fresh = EditorVoxelVolumes::new();
        fresh.set_content_root(Some(dir.path().to_path_buf()));
        assert!(fresh.ensure(ENTITY, &VoxelVolume::from_asset(ASSET)));
        fresh.sync_camera(
            DVec3::splat(8.0),
            &VoxelWantsParams {
                radius_m: 1000.0,
                hysteresis: 0.0,
            },
            VoxelStreamBudget::default(),
        );
        assert_eq!(solid_count(&fresh, ENTITY), solid_count(&store, ENTITY));
    }

    /// **Flying away must not lose — or DELETE — an unsaved carve.**
    ///
    /// The staging split is "dirty and resident ⇒ carved, dirty and absent ⇒
    /// deleted", which is the right reading and a loaded gun: a camera pass that
    /// evicted a dirty chunk would not merely drop the unsaved cut, it would
    /// stage the chunk as **removed** and the next save would erase that chunk
    /// from the `.inf_voxel` — the carved cave, not merely the carving.
    /// `sync_residency` refuses to evict a dirty chunk for exactly that reason;
    /// this is the editor-level proof that the refusal survives the store's own
    /// camera door, which is the only one the viewport ever calls.
    #[test]
    fn flying_away_retains_an_unsaved_carve_rather_than_evicting_it() {
        let (dir, mut store) = fixture();
        let path = dir.path().join("Rock.inf_voxel");
        let chunks_before = inf_voxel::read_voxel_asset(&path)
            .unwrap()
            .reader()
            .chunk_count();

        assert!(store.carve(ENTITY, &cut()).is_some());
        let dirty = store.dirty_chunks(ENTITY);
        assert!(!dirty.is_empty());

        // The author flies to the far side of the world with a tight radius: the
        // carved chunks are no longer wanted by anything.
        let report = store.sync_camera(
            DVec3::splat(100_000.0),
            &VoxelWantsParams {
                radius_m: 8.0,
                hysteresis: 0.0,
            },
            VoxelStreamBudget::default(),
        );
        assert!(
            report.retained_dirty > 0,
            "the camera pass wanted none of the carved chunks and evicted them anyway"
        );
        assert!(
            report.evicted > 0,
            "…and it really did evict the clean ones"
        );
        assert_eq!(store.dirty_chunks(ENTITY), dirty, "a mark was dropped");
        for &key in &dirty {
            assert!(
                store.slot(ENTITY).unwrap().data.is_resident(key),
                "{key:?} kept its mark but lost its chunk — the carve is gone and the \
                 save would stage a DELETION"
            );
        }

        // …and the save still stages them as CHANGED, never as removed.
        let staged = stage_voxel_edits(&store);
        assert_eq!(staged.len(), 1);
        assert!(
            staged[0].edits.removed.is_empty(),
            "an evicted-then-staged chunk would erase itself from the asset"
        );
        let report = flush_voxel_edits(&mut store, dir.path());
        assert_eq!(report.written.len(), 1);
        let after = inf_voxel::read_voxel_asset(&path).unwrap();
        assert_eq!(
            after.reader().chunk_count(),
            chunks_before,
            "the save deleted chunks the author only flew away from"
        );
    }

    /// The autosave note names the volume and the chunk count — the honest
    /// recovery warning, verbatim the terrain's.
    #[test]
    fn the_recovery_note_reports_unsaved_carve_edits() {
        let (_dir, mut store) = fixture();
        assert!(unsaved_voxel_note(&store).is_none(), "clean ⇒ no note");

        assert!(store.carve(ENTITY, &cut()).is_some());
        let note = unsaved_voxel_note(&store).expect("a note");
        assert!(note.contains(&ENTITY.to_string()));
        assert!(note.contains("LAST SAVED"));
        assert_eq!(
            store.unsaved_chunk_count(),
            store.dirty_chunks(ENTITY).len()
        );
    }

    /// **The tool readout's document half, and the defensive advisory.**
    ///
    /// A level reports how many volumes it holds and which terrains could not
    /// keep a cave mouth — and the advisory fires only once the document is
    /// *actually* in the lossy state, not merely capable of reaching it. The two
    /// must stay distinct: "a breakthrough here would be refused" is a warning
    /// about the next gesture, "this level is carrying mouths it cannot save" is
    /// a report of damage already done.
    #[test]
    fn level_status_names_the_refusing_terrain_and_only_then_the_advisory() {
        use crate::ipc::SpawnKind;
        use inf_ecs::components::Terrain;

        let mut doc = SceneDoc::new();
        assert_eq!(
            level_status(&doc),
            VoxelLevelStatus::default(),
            "an empty level has nothing to say"
        );

        // A flat INLINE terrain at y = 4 (the carve fixture's ground) …
        let ground = doc.edit_create(SpawnKind::Terrain, "Ground", None);
        {
            let e = doc.entity_of(ground).unwrap();
            let mut t = doc.world_mut().world_mut().get_mut::<Terrain>(e).unwrap();
            t.data = inf_terrain::TerrainData::new(33, 1.0);
            t.data.author_tile((0, 0), |_, _| 4.0);
            t.asset = None;
        }
        // … and one volume to carve.
        let volume = doc.edit_create(SpawnKind::Empty, "Cave", None);
        {
            let e = doc.entity_of(volume).unwrap();
            doc.world_mut()
                .world_mut()
                .entity_mut(e)
                .insert(VoxelVolume::from_asset(ASSET));
        }

        let s = level_status(&doc);
        assert_eq!(s.volumes, 1);
        assert_eq!(s.asset_backed_terrains, 0);
        assert_eq!(s.inline_terrains, vec!["Ground".to_string()]);
        assert!(
            s.advisories.is_empty(),
            "nothing is carved yet — a level that COULD refuse is not a level that lost anything"
        );

        // Open a mouth behind the tools' back: the stale-document state.
        let origin = doc.terrain_data_and_origin(ground).unwrap().1;
        doc.with_terrain_data_mut(ground, |data| {
            inf_voxel::apply_surface_cut(
                data,
                origin,
                &VoxelShape::Sphere {
                    center: DVec3::new(8.0, 4.0, 8.0),
                    radius_m: 3.0,
                },
                true,
            )
        });
        let s = level_status(&doc);
        assert_eq!(s.advisories.len(), 1, "{:?}", s.advisories);
        assert!(
            s.advisories[0].starts_with("Ground: "),
            "the advisory must name the terrain: {}",
            s.advisories[0]
        );
        assert!(
            s.advisories[0].contains(".inf_terrain"),
            "…and the fix: {}",
            s.advisories[0]
        );
        assert_eq!(inline_hole_warnings(&doc), s.advisories);

        // Making the terrain asset-backed silences both halves at once — the
        // mouths are now persistable, so there is nothing to refuse and nothing
        // to lose.
        {
            let e = doc.entity_of(ground).unwrap();
            doc.world_mut()
                .world_mut()
                .get_mut::<Terrain>(e)
                .unwrap()
                .asset = Some(Uuid::from_u128(0x2107_00AA));
        }
        let s = level_status(&doc);
        assert!(s.advisories.is_empty());
        assert!(s.inline_terrains.is_empty());
        assert_eq!(s.asset_backed_terrains, 1);
    }

    /// `bound_volumes` is a **deterministic** order, so two saves of one session
    /// stage the same assets in the same sequence.
    #[test]
    fn bound_volumes_is_sorted_and_pairs_each_entity_with_its_asset() {
        let (dir, mut store) = fixture();
        let second = Uuid::from_u128(0x2107_0003);
        let second_entity = Uuid::from_u128(0x2107_0000); // sorts BEFORE `ENTITY`
        write_rock(dir.path(), "Rock2.inf_voxel", second);
        store.refresh_index();
        assert!(store.ensure(second_entity, &VoxelVolume::from_asset(second)));

        assert_eq!(
            store.bound_volumes(),
            vec![(second_entity, second), (ENTITY, ASSET)]
        );
    }
}
