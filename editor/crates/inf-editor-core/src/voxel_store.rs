//! **The interactive viewport's voxel store** (P21.1): the loose-file half of
//! what `inf_player::voxel::VoxelRegistry` is for a cooked pack.
//!
//! # Why it lives in Ring 1 and not in the viewport host
//!
//! Verbatim the reasoning that put [`crate::terrain_stream`] and
//! [`crate::render_assets`] here: `inf_viewport::host` is
//! `#[cfg(any(windows, target_os = "macos"))]`, so anything written there is
//! invisible to the Linux CI leg — the leg most likely to be the one a
//! contributor's PR runs first. Keeping the index, the resolution rule and the
//! failure policy here (platform-neutral, GPU-free) means the tests below run on
//! all three OSes and the host is left with nothing but call sites.
//!
//! # What is here, and what deliberately is not
//!
//! Here: finding the `.inf_voxel` a `VoxelVolume.asset` names, reading it, and
//! handing the bytes to the Ring-0 [`inf_voxel::VoxelVolumes`] store — which owns
//! everything after the bytes (parsing, residency, meshing, invalidation), so the
//! editor and the shipped player cannot mesh the same field differently.
//!
//! Not here: **any camera policy**. It lives in Ring 0 (`inf_voxel::wants`) and is
//! executed in Ring 0 (`inf_voxel::VoxelVolumes::sync_camera`); this file only
//! forwards the viewport's eye position to it (P21.2). Same reason as everything
//! else above — a radius, a dead band and a budget written once in the editor and
//! once in the player is two policies, and the shipped build would stream a cave
//! differently from its preview.
//!
//! # Resolution is by sidecar GUID, and a miss rescans once
//!
//! A `.inf_voxel` carries its GUID in its `inf_asset` sidecar, exactly as a
//! `.inf_terrain` does, so the index is a walk of the content root. A miss
//! triggers **one** rescan before giving up — an asset written after the project
//! opened (a fresh carve saved by a tool) must resolve without a reopen, which is
//! the same gap `terrain_stream` closed at P16.4a.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use glam::DVec3;
use inf_ecs::components::VoxelVolume;
use inf_voxel::{
    OpReport, VolumeSlot, VoxelDelta, VoxelDeltaBuilder, VoxelOp, VoxelStreamBudget,
    VoxelStreamReport, VoxelVolumes, VoxelWantsParams,
};
use uuid::Uuid;

/// Depth cap on the content-root walk — deep enough for any content layout,
/// shallow enough that a symlink loop cannot hang project open. Mirrors
/// `terrain_stream`'s cap for the same reason.
const MAX_CONTENT_DEPTH: u32 = 16;

/// Every loose `.inf_voxel` under `dir`, keyed by the GUID in its sidecar.
///
/// A payload without a sidecar is skipped with a warning rather than guessed at:
/// the file name is not an identity, and inventing one would make a level's
/// `VoxelVolume.asset` resolve to whatever happened to be lying next to it.
pub fn voxel_paths_by_guid(dir: &Path) -> HashMap<Uuid, PathBuf> {
    let mut out = HashMap::new();
    let mut files = Vec::new();
    collect_voxel_files(dir, 0, &mut files);
    for p in files {
        match inf_asset::AssetSidecar::load(&p) {
            Ok(side) => {
                out.insert(side.guid.uuid(), p);
            }
            Err(_) => tracing::warn!(
                "inf-editor-core: .inf_voxel without a sidecar {}",
                p.display()
            ),
        }
    }
    out
}

fn collect_voxel_files(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    if depth > MAX_CONTENT_DEPTH {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    // Sorted, so the index is a deterministic function of the tree rather than of
    // the filesystem's enumeration order.
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            let hidden = path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.starts_with('.'));
            if !hidden {
                collect_voxel_files(&path, depth + 1, out);
            }
        } else if path.extension().and_then(|s| s.to_str()) == Some("inf_voxel") {
            out.push(path);
        }
    }
}

/// The **one** editable voxel working set, shared by the two things that both
/// have to be able to move it (P21.2).
///
/// The viewport thread pages, meshes and carves it; the document's undo history
/// has to be able to put a carve back — and `SceneDoc::undo` runs on the Ring-2
/// command thread, which has no viewport. The two alternatives were both worse:
///
/// * **Move the store into `SceneDoc`.** Every projection would then need
///   `&mut SceneDoc` (binding and meshing are `&mut` acts), which is exactly the
///   signature the shipped player's projector does not have — and the two
///   projectors are held byte-identical by `projector_mirror`.
/// * **Journal the voxel half of an undo for the viewport to replay next frame.**
///   A single Ctrl+Z would then revert the ground now and the rock one frame
///   later, which is a visible tear across the cave mouth the two layers share.
///
/// So the handle is shared, and the lock order is fixed everywhere: **document
/// first, volumes second.** Both the viewport loop and the Ring-2 undo command
/// already take the document lock before they reach this one.
pub type SharedVoxelVolumes = Arc<Mutex<EditorVoxelVolumes>>;

/// A fresh [`SharedVoxelVolumes`] — the one constructor, so nobody has to spell
/// the `Arc<Mutex<…>>` out and no second wrapper shape can appear.
pub fn shared_volumes() -> SharedVoxelVolumes {
    Arc::new(Mutex::new(EditorVoxelVolumes::new()))
}

/// The editor's loaded voxel volumes, indexed over a project's content root.
#[derive(Default)]
pub struct EditorVoxelVolumes {
    /// Where loose `.inf_voxel` assets are looked up. `None` (the default)
    /// disables the whole store — a document with no volumes is unaffected either
    /// way, which is what keeps a viewport with no project byte-identical to its
    /// pre-P21.1 self.
    content_root: Option<PathBuf>,
    index: HashMap<Uuid, PathBuf>,
    volumes: VoxelVolumes,
    /// Assets that failed to resolve or load. Kept so the warning is logged
    /// **once** rather than on every frame of a projection that runs at 60 Hz —
    /// the difference between a diagnosable log and an unusable one.
    ///
    /// Cleared only by [`set_content_root`](EditorVoxelVolumes::set_content_root)
    /// and [`refresh_index`](EditorVoxelVolumes::refresh_index) — the two
    /// **external** events that can genuinely change the answer. It is emphatically
    /// NOT cleared by the miss-path rescan: doing so let two unresolvable assets
    /// un-suppress each other forever, and each rescan walks the whole content
    /// root and TOML-parses every sidecar under it.
    failed: HashSet<Uuid>,
    /// How many times the content root has been walked. The engagement counter
    /// behind the "logged once" claim: a suppression bug shows up here as an
    /// unbounded number and nowhere else, because the *behaviour* (a volume that
    /// does not draw) is identical either way.
    scans: usize,
}

impl EditorVoxelVolumes {
    pub fn new() -> Self {
        Self::default()
    }

    /// Point the store at a project's content root (or `None` to disable it).
    ///
    /// Drops every loaded volume: a different root is a different project, and a
    /// GUID that resolved under the old one says nothing about the new one.
    pub fn set_content_root(&mut self, root: Option<PathBuf>) {
        self.volumes.clear();
        self.failed.clear();
        self.index = match &root {
            Some(dir) => {
                self.scans += 1;
                voxel_paths_by_guid(dir)
            }
            None => HashMap::new(),
        };
        self.content_root = root;
    }

    /// Rebuild the index after the content database changed, **keeping** loaded
    /// volumes (their bytes are already in memory; a rewritten asset re-binds
    /// through [`ensure`](Self::ensure) when its GUID changes, and a rewrite under
    /// the same GUID is a P21.3 concern the carve path will drive explicitly).
    pub fn refresh_index(&mut self) {
        self.rescan();
        // An EXTERNAL refresh is the one event that can turn a permanent failure
        // back into a success (an import finished, the watcher saw a write), so it
        // is the one that may forget them.
        self.failed.clear();
    }

    /// Re-walk the content root **without** forgetting past failures — the
    /// miss-path rescan.
    ///
    /// Split from [`refresh_index`](Self::refresh_index) because clearing
    /// `failed` here is a thrash bug, not a nicety: with two unresolvable assets,
    /// A's rescan forgets B's failure, B's rescan forgets A's, and the projection
    /// walks the whole content root and TOML-parses every `.inf_voxel` sidecar
    /// under it **twice per frame, forever**. A single-asset test cannot see it —
    /// which is exactly why one existed and the bug survived it.
    fn rescan(&mut self) {
        let Some(dir) = self.content_root.clone() else {
            return;
        };
        self.scans += 1;
        self.index = voxel_paths_by_guid(&dir);
    }

    /// How many times the content root has been walked since this store was
    /// created — the bound the "logged once" claim is actually asserted against.
    pub fn scan_count(&self) -> usize {
        self.scans
    }

    pub fn content_root(&self) -> Option<&Path> {
        self.content_root.as_deref()
    }

    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    /// Make sure `entity`'s volume is loaded, and report whether anything is
    /// drawable afterwards.
    ///
    /// A volume with no `asset` releases whatever it had loaded and reports
    /// `false` — an author who cleared the reference must see the cave disappear,
    /// not keep the last one that resolved.
    pub fn ensure(&mut self, entity: Uuid, volume: &VoxelVolume) -> bool {
        let Some(asset) = volume.asset else {
            self.volumes.remove(entity.as_u128());
            return false;
        };
        if self.volumes.is_bound(entity.as_u128(), asset.as_u128()) {
            return true;
        }
        if self.content_root.is_none() || self.failed.contains(&asset) {
            return false;
        }
        // A miss rescans ONCE — an asset written after the project opened must
        // resolve without a reopen — and `failed` (which `rescan` deliberately
        // does NOT clear) suppresses every retry after that.
        if !self.index.contains_key(&asset) {
            self.rescan();
        }
        let Some(path) = self.index.get(&asset).cloned() else {
            tracing::warn!("inf-editor-core: no .inf_voxel for asset {asset}");
            self.failed.insert(asset);
            return false;
        };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("inf-editor-core: read {}: {e}", path.display());
                self.failed.insert(asset);
                return false;
            }
        };
        match self
            .volumes
            .ensure(entity.as_u128(), asset.as_u128(), &bytes)
        {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!("inf-editor-core: bad .inf_voxel {}: {e}", path.display());
                self.failed.insert(asset);
                false
            }
        }
    }

    /// Record where the document has placed `entity`'s volume in the world, so
    /// residency is measured from the camera to the cave the player can see rather
    /// than to the asset's authoring anchor. Idempotent; forwards to Ring 0.
    pub fn place(&mut self, entity: Uuid, translation: DVec3) -> bool {
        self.volumes.place(entity.as_u128(), translation)
    }

    /// **The camera-driven residency pass** (P21.2): page every bound volume
    /// against the viewport eye, then re-mesh what moved.
    ///
    /// A pure forward to `inf_voxel::VoxelVolumes::sync_camera` — the radius, the
    /// dead band, the budget and the eviction rule all live in Ring 0, so the
    /// editor and the shipped player stream the same cave the same way. The only
    /// editor-local part is *when* it is called (the viewport's per-frame sync
    /// point) and *where the eye comes from* (the editor camera, which the
    /// simulation cannot see — the determinism seam).
    ///
    /// Decode failures come back on the report rather than going to a log inside
    /// Ring 0; they are warned about here, **once per asset**, on the same
    /// suppression path an unresolvable reference uses — a projection runs at input
    /// rate and a per-frame warning is an unusable log.
    pub fn sync_camera(
        &mut self,
        eye_world: DVec3,
        params: &VoxelWantsParams,
        budget: VoxelStreamBudget,
    ) -> VoxelStreamReport {
        let report = self.volumes.sync_camera(eye_world, params, budget);
        for (key, err) in &report.failed {
            tracing::warn!(
                "inf-editor-core: voxel chunk ({}, {}, {}) failed to decode: {err}",
                key.x,
                key.y,
                key.z
            );
        }
        report
    }

    /// The loaded slot for `entity` — its resident chunks and meshed surface.
    pub fn slot(&self, entity: Uuid) -> Option<&VolumeSlot> {
        self.volumes.get(entity.as_u128())
    }

    // ── carving (P21.2) ──────────────────────────────────────────────────
    //
    // Three calls, and every one of them ends in `resync` — the Ring-0 re-mesh
    // of whatever the edit moved. A carve that skipped it would change the field
    // and leave the *previous* surface on screen until an unrelated streaming
    // pass happened to rebuild it, which reads as a brush that works
    // intermittently.

    /// Apply one carve/fill dab to `entity`'s volume, accumulating its undo
    /// record into `builder`, and re-mesh what moved.
    ///
    /// `None` when the entity has no loaded volume — an unbound reference, an
    /// asset that did not resolve, or a volume the store has released. That is a
    /// refusal, not an empty edit: a carve into a volume nobody loaded would
    /// report success and change nothing.
    ///
    /// One `builder` per **stroke**, not per dab (see
    /// [`inf_voxel::VoxelDeltaBuilder`]): overlapping dabs then merge into a
    /// single reversible record whose `before` is the state at the start of the
    /// drag.
    pub fn carve_into(
        &mut self,
        entity: Uuid,
        op: &VoxelOp,
        builder: &mut VoxelDeltaBuilder,
    ) -> Option<OpReport> {
        let slot = self.volumes.get_mut(entity.as_u128())?;
        let report = slot.data.apply_op_into(op, builder);
        if !report.is_noop() {
            // Re-mesh now rather than at commit: the whole point of a live brush
            // is that the cave appears under the cursor as it is dug.
            self.volumes.resync(entity.as_u128());
        }
        Some(report)
    }

    /// Close a stroke: compress `builder` into the reversible [`VoxelDelta`],
    /// read against the volume's **current** chunks. `None` when the volume is
    /// gone (the stroke has nothing left to describe).
    pub fn finish_carve(&self, entity: Uuid, builder: &VoxelDeltaBuilder) -> Option<VoxelDelta> {
        let slot = self.volumes.get(entity.as_u128())?;
        Some(builder.finalize(&slot.data))
    }

    /// The one-shot form of [`carve_into`](Self::carve_into) +
    /// [`finish_carve`](Self::finish_carve) — a scripted or single-click cut.
    pub fn carve(&mut self, entity: Uuid, op: &VoxelOp) -> Option<(OpReport, VoxelDelta)> {
        let mut builder = VoxelDeltaBuilder::new();
        let report = self.carve_into(entity, op, &mut builder)?;
        let delta = self.finish_carve(entity, &builder)?;
        Some((report, delta))
    }

    /// Redo a carve: replay its `after` samples into `entity`'s volume and
    /// re-mesh. Returns whether the volume was there to write to.
    pub fn apply_delta(&mut self, entity: Uuid, delta: &VoxelDelta) -> bool {
        let Some(slot) = self.volumes.get_mut(entity.as_u128()) else {
            return false;
        };
        Self::report_skipped("redo", entity, slot.data.apply_delta(delta));
        self.volumes.resync(entity.as_u128());
        true
    }

    /// Undo a carve: replay its `before` samples (dropping the chunks it
    /// materialized from nothing) and re-mesh, leaving the volume
    /// byte-identical to what it was before the op.
    pub fn revert_delta(&mut self, entity: Uuid, delta: &VoxelDelta) -> bool {
        let Some(slot) = self.volumes.get_mut(entity.as_u128()) else {
            return false;
        };
        Self::report_skipped("undo", entity, slot.data.revert_delta(delta));
        self.volumes.resync(entity.as_u128());
        true
    }

    /// Surface a malformed-patch count from a replay.
    ///
    /// Ring 0 returns it rather than swallowing it because a skipped patch leaves
    /// the volume in a *partial* state — half a carve put back — and the one
    /// thing worse than a broken undo is a silent one. It cannot happen with a
    /// delta this editor produced; it can with one that survived a corrupt
    /// history, and then the log is the only witness.
    fn report_skipped(what: &str, entity: Uuid, skipped: usize) {
        if skipped > 0 {
            tracing::warn!(
                "inf-editor-core: voxel {what} on {entity} skipped {skipped} malformed patch(es) \
                 — the volume is in a partial state"
            );
        }
    }

    /// Resident chunks across every loaded volume (a streaming readout).
    pub fn chunk_count(&self) -> usize {
        self.volumes.chunk_count()
    }

    /// Wanted-but-not-yet-resident chunks — the convergence backlog.
    pub fn pending_loads(&self) -> usize {
        self.volumes.pending_loads()
    }

    /// Release every volume whose entity is no longer in `live` (the document
    /// changed), so a deleted entity's chunks and meshes do not outlive it.
    pub fn retain_only(&mut self, live: impl IntoIterator<Item = Uuid>) {
        let keep: BTreeSet<u128> = live.into_iter().map(|g| g.as_u128()).collect();
        self.volumes.retain_only(&keep);
    }

    /// Drop everything (a document close / level switch).
    pub fn clear(&mut self) {
        self.volumes.clear();
        self.failed.clear();
    }

    pub fn len(&self) -> usize {
        self.volumes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty()
    }

    /// Meshed triangles across every loaded volume (a status readout).
    pub fn triangle_count(&self) -> usize {
        self.volumes.triangle_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_voxel::{ChunkKey, VoxelChunk, VoxelData};

    /// Page every bound volume in. The fixtures below are a couple of chunks
    /// around the world origin, so one wide-radius pass is the whole volume — and
    /// it has to be an explicit call, because P21.2 made binding and paging two
    /// different acts (see `inf_voxel::store`).
    fn page_everything(s: &mut EditorVoxelVolumes) -> inf_voxel::VoxelStreamReport {
        s.sync_camera(
            DVec3::splat(8.0),
            &VoxelWantsParams {
                radius_m: 1000.0,
                hysteresis: 0.0,
            },
            VoxelStreamBudget::default(),
        )
    }

    /// A carved-out `.inf_voxel` written into `dir` with a sidecar, so the index
    /// can find it exactly as the editor's own writers would leave it.
    fn write_asset(dir: &Path, name: &str, guid: Uuid, radius: f64) {
        let mut v = VoxelData::new(0.5);
        for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
            let b = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|i, j, k| {
                    DVec3::new(
                        (b[0] + i as i32) as f64,
                        (b[1] + j as i32) as f64,
                        (b[2] + k as i32) as f64,
                    )
                    .distance(DVec3::splat(16.0))
                        - radius
                }),
            );
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
    }

    #[test]
    fn without_a_content_root_nothing_loads() {
        let mut s = EditorVoxelVolumes::new();
        assert!(s.content_root().is_none());
        assert!(!s.ensure(
            Uuid::from_u128(1),
            &VoxelVolume::from_asset(Uuid::from_u128(9))
        ));
        assert!(s.is_empty());
    }

    #[test]
    fn a_volume_with_no_asset_releases_what_it_had() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0xA1);
        write_asset(dir.path(), "Cave.inf_voxel", asset, 10.0);
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));

        let e = Uuid::from_u128(1);
        assert!(s.ensure(e, &VoxelVolume::from_asset(asset)));
        assert_eq!(s.len(), 1);
        assert!(s.slot(e).is_some());
        assert_eq!(s.triangle_count(), 0, "binding pages nothing (P21.2)");
        assert!(page_everything(&mut s).loaded > 0);
        assert!(s.triangle_count() > 0);

        // Clearing the reference must make the cave disappear, not keep the last
        // one that resolved.
        assert!(!s.ensure(e, &VoxelVolume::default()));
        assert!(s.slot(e).is_none());
        assert!(s.is_empty());
    }

    #[test]
    fn a_volume_loads_from_the_content_root_and_stays_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0xB2);
        write_asset(&dir.path().join("Caves"), "Deep.inf_voxel", asset, 12.0);
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert_eq!(s.index_len(), 1, "the walk must reach subfolders");

        let e = Uuid::from_u128(7);
        let volume = VoxelVolume::from_asset(asset);
        assert!(s.ensure(e, &volume));
        page_everything(&mut s);
        let tris = s.slot(e).unwrap().meshes.triangle_count();
        assert!(tris > 0);
        // A second ensure is a no-op — no filesystem access, same slot.
        assert!(s.ensure(e, &volume));
        assert_eq!(s.slot(e).unwrap().meshes.triangle_count(), tris);
        assert_eq!(s.len(), 1);
        // …and so is a second camera pass from the same eye: a settled volume
        // pages nothing and moves no mesh stamp.
        assert!(page_everything(&mut s).is_noop());
    }

    /// An asset written **after** the project opened still resolves: the miss
    /// rescans once. This is the P16.4a gap, closed here at birth.
    #[test]
    fn an_asset_written_after_the_index_still_loads() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert_eq!(s.index_len(), 0);

        let asset = Uuid::from_u128(0xC3);
        write_asset(dir.path(), "Late.inf_voxel", asset, 10.0);
        // No reopen, no `set_content_root`: the miss rescans and finds it.
        assert!(s.ensure(Uuid::from_u128(1), &VoxelVolume::from_asset(asset)));
        assert_eq!(s.index_len(), 1);
    }

    /// **TWO** unresolvable assets fail quietly after **one rescan each** — a
    /// projection runs at input rate, and each rescan walks the whole content root
    /// and TOML-parses every sidecar under it.
    ///
    /// Two, not one, deliberately: with a single asset a `failed` set that is
    /// cleared by the miss-path rescan is indistinguishable from one that is not,
    /// because there is nobody to un-suppress it. Two assets clear each other and
    /// the walk count runs away — which is the only observable difference, since
    /// the *behaviour* (neither volume draws) is identical either way.
    #[test]
    fn two_unresolvable_assets_do_not_un_suppress_each_other() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        let baseline = s.scan_count(); // the walk `set_content_root` itself did

        let (a, b) = (Uuid::from_u128(0xDEAD), Uuid::from_u128(0xBEEF));
        let (va, vb) = (VoxelVolume::from_asset(a), VoxelVolume::from_asset(b));
        for _ in 0..30 {
            assert!(!s.ensure(Uuid::from_u128(1), &va));
            assert!(!s.ensure(Uuid::from_u128(2), &vb));
        }
        assert!(s.is_empty());
        assert_eq!(
            s.scan_count() - baseline,
            2,
            "30 projections over two unresolvable assets walked the content root \
             {} times — the miss-path rescan is clearing the `failed` set, so the \
             two assets keep un-suppressing each other",
            s.scan_count() - baseline
        );

        // …and an EXTERNAL refresh clears the suppression, so a later import is
        // picked up — the affordance the miss-path rescan must not duplicate.
        write_asset(dir.path(), "Found.inf_voxel", a, 10.0);
        s.refresh_index();
        assert!(s.ensure(Uuid::from_u128(1), &va));
        // The still-missing one is re-suppressed after exactly one more walk.
        let after = s.scan_count();
        for _ in 0..10 {
            assert!(!s.ensure(Uuid::from_u128(2), &vb));
        }
        assert_eq!(s.scan_count(), after + 1);
    }

    /// **THE BIND-PARITY REGRESSION (M2).** A volume whose chunks mesh to **no
    /// surface** is still a *bound* volume, and must stay bound.
    ///
    /// The editor used to build its live set from whatever the projection pushed,
    /// and `project_voxel` returns `None` for a volume with no drawable chunks —
    /// so an all-air (or all-rock) asset was released at the end of every
    /// projection and re-read from disk, re-parsed and re-meshed on the next one.
    /// A gizmo drag bumps the document version per input event, so that is a disk
    /// read per mouse move. The shipped player was never affected: it built its
    /// live set from the *wants*.
    ///
    /// Pinned here rather than in the viewport host because the host is
    /// `#[cfg(any(windows, macos))]` and this is the layer the rule actually lives
    /// at. The structural half — that neither projector binds inside its
    /// projection loop — is `projector_mirror`'s.
    #[test]
    fn a_volume_that_meshes_to_nothing_stays_bound() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0xA17);
        // A radius the sphere never reaches inside the authored chunks: every
        // sample is outside, so the field has no sign change and no surface.
        write_asset(dir.path(), "Airy.inf_voxel", asset, -1.0);
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        let baseline = s.scan_count();

        let e = Uuid::from_u128(1);
        let volume = VoxelVolume::from_asset(asset);
        assert!(s.ensure(e, &volume), "an all-air asset still BINDS");
        page_everything(&mut s);
        let slot = s.slot(e).expect("…and keeps its slot");
        assert_eq!(
            slot.meshes.triangle_count(),
            0,
            "the fixture must genuinely produce no surface, or this proves nothing"
        );
        assert!(slot.data.chunk_count() > 0, "it did load real chunks");

        // Re-ensure at the rate a gizmo drag bumps the document. Nothing is
        // re-read, re-parsed or re-meshed, and the store never walks the root.
        for _ in 0..50 {
            assert!(s.ensure(e, &volume));
        }
        assert_eq!(s.len(), 1, "the volume was released and re-read");
        assert_eq!(
            s.scan_count(),
            baseline,
            "a bound volume must not touch the filesystem again"
        );
    }

    /// A store with no content root never walks anything at all.
    #[test]
    fn a_rootless_store_never_walks() {
        let mut s = EditorVoxelVolumes::new();
        for _ in 0..5 {
            assert!(!s.ensure(
                Uuid::from_u128(1),
                &VoxelVolume::from_asset(Uuid::from_u128(9))
            ));
        }
        assert_eq!(s.scan_count(), 0);
    }

    #[test]
    fn changing_the_content_root_drops_every_volume() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0xE4);
        write_asset(dir.path(), "Cave.inf_voxel", asset, 10.0);
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert!(s.ensure(Uuid::from_u128(1), &VoxelVolume::from_asset(asset)));
        assert_eq!(s.len(), 1);

        let other = tempfile::tempdir().unwrap();
        s.set_content_root(Some(other.path().to_path_buf()));
        assert!(s.is_empty(), "a different root is a different project");
        assert_eq!(s.index_len(), 0);
    }

    #[test]
    fn retain_only_releases_dead_volumes() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0xF5);
        write_asset(dir.path(), "Cave.inf_voxel", asset, 10.0);
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        let volume = VoxelVolume::from_asset(asset);
        let (a, b) = (Uuid::from_u128(1), Uuid::from_u128(2));
        assert!(s.ensure(a, &volume) && s.ensure(b, &volume));
        assert_eq!(s.len(), 2);

        s.retain_only([a]);
        assert_eq!(s.len(), 1);
        assert!(s.slot(a).is_some() && s.slot(b).is_none());
        s.clear();
        assert!(s.is_empty());
    }

    /// A corrupt payload is reported and skipped — never drawn as an empty cave.
    #[test]
    fn a_corrupt_payload_is_reported_and_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0x0BAD);
        let path = dir.path().join("Broken.inf_voxel");
        std::fs::write(&path, b"this is not a voxel asset at all, not even close").unwrap();
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(asset),
            inf_asset::AssetKind::VoxelVolume,
            inf_asset::ContentHash::of(b"x"),
        )
        .save(&path)
        .unwrap();

        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert_eq!(s.index_len(), 1, "it IS indexed — it just does not parse");
        assert!(!s.ensure(Uuid::from_u128(1), &VoxelVolume::from_asset(asset)));
        assert!(s.is_empty());
    }
}
