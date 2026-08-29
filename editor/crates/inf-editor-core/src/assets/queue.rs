//! An async import queue with progress events.
//!
//! Imports (glTF decode, texture compression, mip generation, chunked heightmap
//! tiling) are heavy, so they run on a background worker rather than blocking the
//! editor. Callers `submit` a job and drain [`ImportProgress`] events on a tick;
//! Ring 2 forwards those to the webview as `assets://import`. The worker shares
//! the [`AssetProject`] behind a mutex with the command layer, so imported
//! assets appear in the same database the Content Drawer reads.
//!
//! # Two job shapes, one channel (P16.4)
//!
//! A file import is fire-and-forget: it either produces assets or it fails. A
//! **terrain** import can run for minutes over a 16 k source, so it also emits
//! [`ImportProgress::Progress`] ticks and can be [`cancel`](ImportQueue::cancel)led
//! mid-flight. Both travel the same channel and the same `assets://import` event,
//! because the frontend's job model — id, phase, optional error — is identical;
//! only the payload of a tick differs.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use inf_asset::{AssetChange, AssetId, AssetWatcher};

use super::terrain_import::{self, CancelToken, TerrainImportSettings};
use super::AssetProject;

/// A progress event for one import job.
#[derive(Debug, Clone)]
pub enum ImportProgress {
    /// The worker picked up job `id` for `source`.
    Started { id: u64, source: PathBuf },
    /// Job `id` has finished `done` of `total` units of work (terrain imports:
    /// tiles written, across every LOD level).
    Progress {
        id: u64,
        source: PathBuf,
        done: u64,
        total: u64,
        /// A short label for the stage (`"tiles"`, or `"lod{n}"`).
        stage: String,
    },
    /// Job `id` produced these assets (primary first, if any).
    Finished {
        id: u64,
        source: PathBuf,
        produced: Vec<AssetId>,
        primary: Option<AssetId>,
        cached: bool,
        /// **Non-fatal import advisories** (P26.5) — the P26.1 dimension pair
        /// plus the tail-cost badge, in the shape `ImportOutcome::advisories`
        /// already carried.
        ///
        /// They were raised into `tracing` at the import and reached the Output
        /// Log; nothing put them where the person who just dropped a file is
        /// looking. That is the P26.1 ledger's deferred *badging*: an author
        /// importing forty 128² decals is told, once, at the moment they can
        /// still say no.
        ///
        /// **Empty on a cache hit**, deliberately and not by omission: the
        /// advisories were said when the bytes were first imported, and
        /// re-announcing them on every drop of an unchanged file is how a badge
        /// becomes noise.
        advisories: Vec<String>,
    },
    /// Job `id` failed (or was cancelled — the error text says which).
    Failed {
        id: u64,
        source: PathBuf,
        error: String,
    },
}

impl ImportProgress {
    pub fn job_id(&self) -> u64 {
        match self {
            ImportProgress::Started { id, .. }
            | ImportProgress::Progress { id, .. }
            | ImportProgress::Finished { id, .. }
            | ImportProgress::Failed { id, .. } => *id,
        }
    }
}

enum Job {
    /// Route an external file through the generic importer.
    File {
        id: u64,
        source: PathBuf,
        dest: PathBuf,
    },
    /// Chunk a heightmap into a `.inf_terrain` asset (P16.4).
    Terrain {
        id: u64,
        source: PathBuf,
        settings: Box<TerrainImportSettings>,
        /// The level's geo-anchor at the moment the job was queued (Wave G).
        ///
        /// **Captured, not looked up.** The build runs for minutes with no
        /// project lock held; reading the anchor out of a live document when the
        /// job reaches the front of the queue would make where the terrain lands
        /// depend on whether the author edited World Settings in the meantime.
        anchor: Option<Box<inf_math::geo::GeoAnchor>>,
        name: Option<String>,
        cancel: CancelToken,
    },
    /// Derive a `.inf_vmesh` for every `.inf_mesh` that lacks a current one
    /// (P18.3) — the project-open sweep that covers content imported before the
    /// editor could derive meshlet DAGs at all.
    ///
    /// It rides the import queue rather than the background tick because the tick
    /// must never block on the project mutex — it carries the very progress events
    /// a long job reports. But riding the queue is not permission to *hold* the
    /// mutex either: `build_vgeom` on a real character runs for seconds, and a
    /// first open of a large project would freeze every asset command, the tick,
    /// and this job's own Cancel behind it. So the worker runs the P16.4a shape —
    /// **plan under a short lock, build holding nothing, re-acquire briefly to
    /// register** — one mesh at a time, checking `cancel` between each.
    Vmeshes { id: u64, cancel: CancelToken },
}

impl Job {
    fn id(&self) -> u64 {
        match self {
            Job::File { id, .. } | Job::Terrain { id, .. } | Job::Vmeshes { id, .. } => *id,
        }
    }
    fn source(&self) -> PathBuf {
        match self {
            Job::File { source, .. } | Job::Terrain { source, .. } => source.clone(),
            // The sweep has no source file; it reports under a stable pseudo-path
            // so the frontend's job model (id + source + phase) needs no new shape.
            Job::Vmeshes { .. } => PathBuf::from(VMESH_SWEEP_SOURCE),
        }
    }
}

/// The pseudo-source path the project-wide vmesh sweep reports under.
pub const VMESH_SWEEP_SOURCE: &str = "<derive meshlet DAGs>";

/// One drain of the progress channel.
#[derive(Debug, Default)]
pub struct PollBatch {
    /// The events, in the order the worker produced them.
    pub events: Vec<ImportProgress>,
    /// A job finished in this batch, so an asset the viewport resolves by GUID
    /// may now be on disk (a terrain, a mesh and its derived DAG, the project-open
    /// vmesh sweep) and the loose-asset index is stale. Recorded here rather than
    /// derived from the asset database, because the tick that reads this must
    /// never block on the project mutex.
    pub index_stale: bool,
}

/// The queue: a worker thread + a progress channel.
pub struct ImportQueue {
    tx: Sender<Job>,
    events: Receiver<ImportProgress>,
    next_id: u64,
    worker: Option<JoinHandle<()>>,
    /// Cancellation tokens for cancellable (terrain) jobs, kept until the job
    /// reports terminal.
    cancels: HashMap<u64, CancelToken>,
    /// Which in-flight job ids are terrain imports (see [`PollBatch`]).
    terrain_jobs: HashSet<u64>,
}

impl ImportQueue {
    /// Spawn the worker over a shared project.
    pub fn spawn(project: Arc<Mutex<AssetProject>>) -> Self {
        let (tx, rx) = channel::<Job>();
        let (etx, erx) = channel::<ImportProgress>();
        let worker = std::thread::Builder::new()
            .name("asset-import".into())
            .spawn(move || worker_loop(rx, etx, project))
            .expect("spawn import worker");
        Self {
            tx,
            events: erx,
            next_id: 1,
            worker: Some(worker),
            cancels: HashMap::new(),
            terrain_jobs: HashSet::new(),
        }
    }

    /// Queue a file import; returns the job id (referenced by later progress events).
    pub fn submit(&mut self, source: PathBuf, dest: PathBuf) -> u64 {
        let id = self.take_id();
        // If the worker has gone away the send fails; the caller sees no events.
        let _ = self.tx.send(Job::File { id, source, dest });
        id
    }

    /// Queue the project-wide `.inf_vmesh` derivation sweep (P18.3). Returns the
    /// job id; progress ticks and the terminal event arrive on the same channel as
    /// any import, so the frontend needs no new case.
    ///
    /// **Cancellable**, and that is not a nicety: closing or switching a project
    /// drops the `ImportQueue`, whose `Drop` joins the worker. Without a token a
    /// half-finished sweep of a large project would hold that join — and therefore
    /// the whole editor — for as long as the remaining meshes take to build.
    pub fn submit_vmesh_sweep(&mut self) -> u64 {
        let id = self.take_id();
        let cancel = CancelToken::new();
        self.cancels.insert(id, cancel.clone());
        let _ = self.tx.send(Job::Vmeshes { id, cancel });
        id
    }

    /// Queue a chunked heightmap → `.inf_terrain` import (P16.4). Returns the job
    /// id; progress ticks and the terminal event arrive on the same channel.
    ///
    /// `anchor` is the level's [`inf_math::geo::GeoAnchor`] (Wave G). It is what
    /// turns `TerrainImportSettings::use_georeference` from a recorded
    /// preference into a placed terrain — without it the importer has nothing to
    /// subtract and the asset lands at the world origin however the flag is set.
    pub fn submit_terrain(
        &mut self,
        source: PathBuf,
        settings: TerrainImportSettings,
        anchor: Option<inf_math::geo::GeoAnchor>,
        name: Option<String>,
    ) -> u64 {
        let id = self.take_id();
        let cancel = CancelToken::new();
        self.cancels.insert(id, cancel.clone());
        self.terrain_jobs.insert(id);
        let _ = self.tx.send(Job::Terrain {
            id,
            source,
            settings: Box::new(settings),
            anchor: anchor.map(Box::new),
            name,
            cancel,
        });
        id
    }

    /// Ask a cancellable job to stop. Returns `false` for an unknown or
    /// already-finished job. The job still reports a terminal event
    /// (`Failed { error: "…cancelled" }`), so the frontend's state machine needs
    /// no special case.
    pub fn cancel(&self, id: u64) -> bool {
        match self.cancels.get(&id) {
            Some(token) => {
                token.cancel();
                true
            }
            None => false,
        }
    }

    fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Drain all progress available right now (non-blocking).
    pub fn poll(&mut self) -> PollBatch {
        let events: Vec<ImportProgress> = self.events.try_iter().collect();
        let mut index_stale = false;
        for ev in &events {
            let terminal = matches!(
                ev,
                ImportProgress::Finished { .. } | ImportProgress::Failed { .. }
            );
            if terminal {
                let id = ev.job_id();
                self.cancels.remove(&id);
                self.terrain_jobs.remove(&id);
            }
            // P18.3 generalized the old terrain-only flag: **any** finished job can
            // have written an asset the viewport resolves by GUID (a terrain, a
            // mesh and its derived DAG, or the sweep), and the index it resolves
            // through is a snapshot that predates all of them. Refreshing after a
            // job that produced nothing streamable costs one directory walk;
            // missing one leaves a freshly imported mesh drawing a placeholder
            // forever, which is precisely the bug P18.3 exists to delete.
            if matches!(ev, ImportProgress::Finished { .. }) {
                index_stale = true;
            }
        }
        PollBatch {
            events,
            index_stale,
        }
    }

    /// Block for the next event (tests).
    pub fn recv(&self) -> Option<ImportProgress> {
        self.events.recv().ok()
    }
}

impl Drop for ImportQueue {
    fn drop(&mut self) {
        // Cancel anything still running so the join below cannot block behind a
        // multi-minute terrain import, then drop the sender to end the loop.
        for token in self.cancels.values() {
            token.cancel();
        }
        // (Replace tx with a dead channel so the worker's recv() returns Err.)
        let (dead, _) = channel();
        self.tx = dead;
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn worker_loop(rx: Receiver<Job>, etx: Sender<ImportProgress>, project: Arc<Mutex<AssetProject>>) {
    while let Ok(job) = rx.recv() {
        let id = job.id();
        let source = job.source();
        let _ = etx.send(ImportProgress::Started {
            id,
            source: source.clone(),
        });
        let event = match job {
            Job::File { dest, .. } => {
                let result = {
                    let mut proj = lock(&project);
                    proj.import_file(&source, &dest)
                };
                match result {
                    Ok(out) => ImportProgress::Finished {
                        id,
                        source,
                        produced: out.produced,
                        primary: out.primary,
                        cached: out.cached,
                        advisories: out.advisories,
                    },
                    Err(e) => ImportProgress::Failed {
                        id,
                        source,
                        error: e.to_string(),
                    },
                }
            }
            Job::Terrain {
                settings,
                anchor,
                name,
                cancel,
                ..
            } => {
                // PHASE 1 — build, with NO project lock held. A 16 k import runs
                // for minutes here; taking the shared mutex around it would
                // freeze every asset command, the progress tick that carries
                // these very events, and the wizard's own Cancel button.
                let result = {
                    let etx = etx.clone();
                    let src = source.clone();
                    let mut on_progress = |p: inf_terrain::ImportProgress| {
                        let _ = etx.send(ImportProgress::Progress {
                            id,
                            source: src.clone(),
                            done: p.tiles_done,
                            total: p.tiles_total,
                            stage: if p.lod == 0 {
                                "tiles".to_string()
                            } else {
                                format!("lod{}", p.lod)
                            },
                        });
                    };
                    terrain_import::build_anchored(
                        &source,
                        &settings,
                        anchor.as_deref(),
                        &mut on_progress,
                        &cancel,
                    )
                }
                // PHASE 2 — commit, holding the lock for the few filesystem +
                // database operations it actually needs. The cancellation flag is
                // checked BEFORE the lock, so a job cancelled while it was
                // building neither registers anything nor waits on a mutex some
                // other command may be holding.
                .and_then(|built| {
                    if cancel.is_cancelled() {
                        return Err(inf_asset::AssetError::Import("import cancelled".into()));
                    }
                    let mut proj = lock(&project);
                    terrain_import::commit(&mut proj, built, name.as_deref(), None, &cancel)
                });
                match result {
                    Ok(out) => ImportProgress::Finished {
                        id,
                        source,
                        produced: vec![out.asset],
                        primary: Some(out.asset),
                        cached: false,
                        // A heightmap is not a `.inf_tex` and takes no tile
                        // geometry, so the three TEXTURE advisories cannot
                        // apply to it — but the L7.H7 outside-the-project note
                        // does, and this line hard-coded it away (round-2
                        // finding R2.F7). It is the one importer where an
                        // outside-the-project source is the normal case.
                        advisories: out.advisories,
                    },
                    Err(e) => ImportProgress::Failed {
                        id,
                        source,
                        error: e.to_string(),
                    },
                }
            }
            // The project-open meshlet-DAG sweep (P18.3). Reports the derived
            // assets as `produced` with no `primary` — nothing here is a headline
            // the UI should reveal, but the Content Drawer must learn they exist.
            //
            // PHASE 1 — plan, under a short lock. Database reads only; meshes whose
            // derived asset is already current are not planned at all, so a
            // re-opened project does almost nothing here and nothing at all below.
            Job::Vmeshes { cancel, .. } => {
                let plans = {
                    let proj = lock(&project);
                    super::vmesh::plan_sweep(&proj)
                };
                let total = plans.len() as u64;
                let tick = |done: u64, stage: &str| {
                    let _ = etx.send(ImportProgress::Progress {
                        id,
                        source: source.clone(),
                        done,
                        total,
                        stage: stage.to_string(),
                    });
                };
                // "0 of N": phase 1 is over and the lock is free again. The only
                // tick that reports the plan rather than the work.
                tick(0, "planned");

                let mut produced = Vec::new();
                for (k, plan) in plans.iter().enumerate() {
                    if cancel.is_cancelled() {
                        break;
                    }
                    // PHASE 2 — build, with NO project lock held. This is where the
                    // seconds go, and holding the mutex across it would be the
                    // P16.4a stall through a new door.
                    let payload = match super::vmesh::build_vmesh(plan) {
                        Ok(Some(p)) => p,
                        // Nothing to virtualize, or an unreadable mesh: skip it.
                        // One bad asset must not fail a whole project's sweep.
                        Ok(None) => {
                            tick(k as u64 + 1, "built");
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "inf-editor-core: vmesh derivation failed for {}: {e}",
                                plan.mesh
                            );
                            tick(k as u64 + 1, "built");
                            continue;
                        }
                    };
                    // Reported BEFORE the commit on purpose: this tick is the
                    // observable proof that the heavy phase ran without the
                    // project, which is the property the contention gate pins.
                    tick(k as u64 + 1, "built");

                    // PHASE 3 — commit: one atomic write + a database insert.
                    //
                    // `try_lock` in a cancel-checking loop rather than a blocking
                    // `lock`. A worker parked on the mutex cannot see a
                    // cancellation, so a project held by some other command would
                    // make Cancel — and therefore closing or switching the project,
                    // which joins this thread — wait for that command to finish.
                    // Spinning at 2 ms costs nothing next to a `build_vgeom`.
                    loop {
                        if cancel.is_cancelled() {
                            break;
                        }
                        let mut proj = match project.try_lock() {
                            Ok(p) => p,
                            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
                            Err(std::sync::TryLockError::WouldBlock) => {
                                std::thread::sleep(std::time::Duration::from_millis(2));
                                continue;
                            }
                        };
                        match super::vmesh::commit_vmesh(&mut proj, plan, &payload) {
                            Ok(asset) => produced.push(asset),
                            Err(e) => tracing::warn!(
                                "inf-editor-core: vmesh commit failed for {}: {e}",
                                plan.mesh
                            ),
                        }
                        break;
                    }
                }
                ImportProgress::Finished {
                    id,
                    source,
                    produced,
                    primary: None,
                    cached: false,
                    // A derivation job re-reads assets that were already
                    // imported; whatever they had to be advised about was said
                    // then.
                    advisories: Vec::new(),
                }
            }
        };
        let _ = etx.send(event);
    }
}

/// The project mutex, recovering from a poisoned lock rather than panicking the
/// worker (an import failure must never take the editor's asset system with it).
fn lock(project: &Arc<Mutex<AssetProject>>) -> std::sync::MutexGuard<'_, AssetProject> {
    match project.lock() {
        Ok(p) => p,
        Err(poisoned) => poisoned.into_inner(),
    }
}

// ── the background tick ─────────────────────────────────────────────────────

/// What one [`tick`] found. Ring 2 turns this into events; nothing here names
/// Tauri, which is what makes the tick testable.
#[derive(Debug, Default)]
pub struct TickOutcome {
    /// Import events to forward on `assets://import`.
    pub events: Vec<ImportProgress>,
    /// The content database changed → emit `assets://changed`.
    pub content_changed: bool,
    /// A streamable asset appeared or changed → the viewport's loose-asset index
    /// must be refreshed, or a freshly imported terrain/mesh resolves to nothing
    /// and the entity the wizard (or a drag-drop) just spawned draws empty.
    pub index_stale: bool,
    /// The content version to publish, or `None` when the project was busy.
    pub version: Option<u64>,
    /// **InfiniScript files that changed on disk** (SCRIPT1b), as `(asset GUID,
    /// path)` — **one entry per script, ordered by GUID**.
    ///
    /// The hot-reload path: Ring 2 compiles each through
    /// `inf_script::source::compile_path` — the one file door — and swaps the
    /// class into a running Simulate, or prints the diagnostics and leaves the
    /// previous good program running.
    ///
    /// Resolved to a GUID **here**, while the database lock is already held, so
    /// the caller does not take it a second time to answer a question this pass
    /// already knew.
    ///
    /// # One save is one entry, whatever the platform's watcher does
    ///
    /// A single `std::fs::write` is **three** inotify events on Linux (create,
    /// modify-data, close-write) and **one** `ReadDirectoryChangesW`
    /// notification on Windows. `notify-debouncer-full` coalesces a burst in
    /// *time*, not across event *kinds*, so one drain really does hand this pass
    /// the same file three times — measured: CI reported
    /// `[(e723334b…, Counter.infini) x3]` on ubuntu for one save that Windows
    /// reported once. Carried through one-for-one that is three compiles and
    /// three class swaps per Ctrl+S, which is the operating system's event shape
    /// deciding how often a running world is re-lowered.
    ///
    /// So the pass collects into a map keyed by **GUID**, which is the right key
    /// rather than a convenient one: the GUID *is* the script's identity here (an
    /// actor binds through `ActorClass(Uuid)` — the SCRIPT1b audit's HIGH), so
    /// "the same script twice" is exactly "the same key twice". The order that
    /// falls out is GUID order — a total order this repository controls — rather
    /// than the order the operating system happened to report, and it is the
    /// order `SimSession::apply_pending_classes` then drains in.
    ///
    /// The sidecar `pin_script_ids` writes cannot add a fourth event:
    /// [`AssetWatcher`] drops every `.toml` before it sends
    /// (`inf_asset::is_sidecar`, *"so we don't process an asset twice per
    /// save"*), and the pin writes at all only when no sidecar exists yet.
    ///
    /// # The path is the database's, not the watcher's
    ///
    /// `AssetEntry::path` has been through `inf_asset`'s one `normalize` door —
    /// the same canonicalization its `by_path` index is keyed with — so a macOS
    /// watcher reporting `/private/var/folders/…` for a root the editor opened
    /// as `/var/folders/…` (the `/var` → `/private/var` symlink) yields **one**
    /// spelling here instead of two. It is a handle to the bytes Ring 2 opens,
    /// beside the GUID that is the identity.
    pub scripts: Vec<(uuid::Uuid, std::path::PathBuf)>,
}

/// Extensions whose external change invalidates the viewport's loose-asset index
/// — the set `inf_editor_core::render_assets` and `terrain_stream` resolve
/// through. An edit to anything else (a material, a texture, a level) reaches the
/// viewport through the document instead.
const INDEXED_ASSET_EXTENSIONS: [&str; 5] = [
    "inf_terrain",
    "inf_vmesh",
    "inf_mesh",
    "inf_skel",
    "inf_anim",
];

/// One pass of the editor's background import/watch tick.
///
/// # It never blocks on the project
///
/// The tick runs on its own thread but is reached through the same state the
/// asset commands use, so blocking here parks the whole asset subsystem — and,
/// worse, blocks the very channel a long import reports progress on, including
/// the wizard's Cancel. Every project access is therefore a `try_lock`: a tick
/// that loses the race publishes no version (`None`) and leaves the watcher's
/// batch queued for the next one, 120 ms later. A version that is one tick stale
/// is harmless — the frontend re-fetches the snapshot on the event either way.
pub fn tick(
    queue: &mut ImportQueue,
    project: &Arc<Mutex<AssetProject>>,
    watcher: Option<&AssetWatcher>,
) -> TickOutcome {
    let batch = queue.poll();
    let mut out = TickOutcome {
        content_changed: batch
            .events
            .iter()
            .any(|e| matches!(e, ImportProgress::Finished { .. })),
        index_stale: batch.index_stale,
        scripts: Vec::new(),
        events: batch.events,
        version: None,
    };

    // One short, non-blocking visit to the project: apply the watcher's batch and
    // read the version. If it is busy (a commit, a snapshot), both wait a tick.
    if let Ok(mut proj) = project.try_lock() {
        if let Some(w) = watcher {
            let changes = w.drain();
            if !changes.is_empty() {
                // **One entry per script, keyed by the identity, not by the
                // event** — see [`TickOutcome::scripts`]. A drain carries one
                // save as three events on Linux and one on Windows; a `Vec` of
                // pushes would carry that difference into how many times the
                // editor recompiles and swaps a class.
                let mut scripts: BTreeMap<uuid::Uuid, PathBuf> = BTreeMap::new();
                for c in changes {
                    let path = match &c {
                        AssetChange::Upserted(p) | AssetChange::Removed(p) => p.clone(),
                    };
                    if path
                        .extension()
                        .and_then(|s| s.to_str())
                        .is_some_and(|e| INDEXED_ASSET_EXTENSIONS.contains(&e))
                    {
                        out.index_stale = true;
                    }
                    // Both verdicts are reported (C4-39/C4-44). A dropped
                    // rescan error leaves a *stale* database entry while
                    // `bump()` and `content_changed` still fire, so the drawer
                    // re-reads and shows the old asset as current; a removal
                    // that matched nothing means the entry is still indexed and
                    // still counting as a live referrer for a file that is gone.
                    // SCRIPT1b: a `.infini` that changed is a program that
                    // changed. Collected AFTER the rescan below so the GUID
                    // reported is the one the database now holds — a file
                    // created this pass has no entry until it is scanned.
                    let is_script = inf_script::is_script_path(&path);
                    match c {
                        AssetChange::Upserted(p) => {
                            if let Err(e) = proj.db_mut().rescan_path(&p) {
                                tracing::warn!(
                                    "content watcher: {} changed but could not be re-read \
                                     ({e}); the database still holds its previous entry",
                                    p.display()
                                );
                            }
                            if is_script {
                                // **Pin the identity before reading it**
                                // (SCRIPT1b audit). A `.infini` with no sidecar
                                // is registered under a GUID derived from its
                                // CONTENT HASH, so a script created while the
                                // editor is running would be renamed by its own
                                // second save — and hot reload, PIE and the cook
                                // all bind by that GUID. `super::pin_script_ids`
                                // is the same door `AssetProject::open` uses;
                                // once a sidecar is on disk, `rescan_path` reads
                                // the id back instead of re-deriving it.
                                super::pin_script_ids(proj.db());
                                if let Some(entry) = proj.db().get_by_path(&p) {
                                    // The DATABASE's normalized path, not the
                                    // watcher's raw one, and one value per GUID.
                                    scripts.insert(entry.id().0, entry.path.clone());
                                }
                            }
                        }
                        AssetChange::Removed(p) => {
                            if proj.db_mut().remove_path(&p).is_none() {
                                tracing::debug!(
                                    "content watcher: {} was removed but was not in the \
                                     database under that path",
                                    p.display()
                                );
                            }
                        }
                    }
                }
                proj.bump();
                out.content_changed = true;
                out.scripts = scripts.into_iter().collect();
            }
        }
        out.version = Some(proj.version());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_terrain::{encode_png16, HeightImage};

    /// **The watcher tells the tick that a program changed** — the one link in
    /// the hot-reload chain that had no arm behind it (SCRIPT1b audit).
    ///
    /// The chain is: this function collects `.infini` changes into
    /// [`TickOutcome::scripts`]; Ring 2's asset thread calls
    /// `hot_reload_scripts(&app, &outcome.scripts)`; that compiles each through
    /// the file door and calls `SimSession::reload_class`; and the session
    /// drains the queue at the top of a fixed step. Every link but the **first**
    /// was gated — the Ring-2 call and its body by `dcc.rs`'s source pins, the
    /// swap by `script_hot_reload.rs`. A `scripts` field that silently stayed
    /// empty would make the whole feature dead in the editor while every one of
    /// those arms went on passing, which is this repository's most-repaired
    /// shape.
    ///
    /// Anti-vacuous in both directions: a `.infini` must arrive **under the GUID
    /// a level's `ActorClass` is bound to** — that is, the one the database held
    /// *before* the edit, which is the assertion that found the wave's HIGH —
    /// and a change to something that is not a script must not arrive at all.
    ///
    /// # The two platform exposures this arm found (the SCRIPT1b CI-red fix)
    ///
    /// Written against Windows, it reddened on both other runners, and neither
    /// red was about the arm's tolerance:
    ///
    /// * **ubuntu** reported the same file **three times** from one
    ///   `std::fs::write` — one drain, three inotify kinds. The count is the
    ///   claim (three compiles and three class swaps per save is a real cost),
    ///   so [`TickOutcome::scripts`] now collapses a drain by GUID and the
    ///   `== 1` here is asserted on every platform;
    /// * **macOS** disagreed about the *path*: the watcher reports
    ///   `/private/var/folders/…` where the tempdir is `/var/folders/…`. The
    ///   GUID is the identity and it matched; the path is a handle to bytes, so
    ///   it is asserted by **opening it** and then pinned through one
    ///   canonicalization door rather than compared raw.
    ///
    /// Un-fix mutations, and what each platform can say about them. Reporting
    /// the **watcher's** path instead of the database's (`p` for `entry.path`)
    /// reddens here on *Windows*, where `normalize`'s `fs::canonicalize` returns
    /// the `\\?\` verbatim form and the raw event path does not — so the macOS
    /// exposure is armed on a platform that has no `/private/var` symlink.
    /// Un-collapsing the drain (a `Vec` push for the map insert) reddens on
    /// **ubuntu only**, and cannot be reproduced here: the probe below measures
    /// this platform delivering **one** change event for a save (three rapid
    /// saves were also tried, and also arrived as one). The pre-fix tree *is*
    /// that mutant, and CI reddened it at this line with `left: 3, right: 1`.
    #[test]
    fn a_changed_script_reaches_the_tick_outcome_with_its_guid() {
        use std::time::{Duration, Instant};

        const V2: &str = "on tick(dt)\n  debug.print(\"v2\")\nend\n";

        let dir = tempfile::tempdir().unwrap();
        let scripts = dir.path().join("Scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        let path = scripts.join("Counter.infini");
        std::fs::write(&path, "on tick(dt)\n  debug.print(\"v1\")\nend\n").unwrap();
        // A non-script neighbour, so "only scripts arrive" is a claim about a
        // file that really changes rather than about one that does not exist.
        let other = dir.path().join("Notes.txt");
        std::fs::write(&other, "v1").unwrap();

        let project = Arc::new(Mutex::new(AssetProject::open(dir.path()).unwrap()));
        let guid = lock(&project)
            .db()
            .get_by_path(&path)
            .expect("the scan registered the script")
            .id()
            .0;

        let watcher = inf_asset::AssetWatcher::watch(dir.path(), Duration::from_millis(50))
            .expect("the watcher starts");
        // A second watcher over the same root, drained beside the tick and never
        // asserted on. It is how many raw change events **this platform**
        // delivered, and it is printed rather than asserted because that number
        // is a fact about the operating system, not about the editor: CI
        // measured ubuntu reporting ONE `fs::write` as three events, and this
        // machine reports THREE writes as one. The collapse exists precisely so
        // the tick's answer does not depend on which of those a runner is.
        let probe = inf_asset::AssetWatcher::watch(dir.path(), Duration::from_millis(50))
            .expect("the probe watcher starts");
        let mut queue = ImportQueue::spawn(project.clone());
        // ONE save — the shape that reddened ubuntu, where it arrives as three
        // events. Three rapid saves were tried here first, to give a
        // single-event platform a duplicate to collapse; the probe below
        // measured them arriving as **one** change event on Windows anyway, so
        // the extra writes bought nothing and are not pretended to.
        std::fs::write(&path, V2).unwrap();
        std::fs::write(&other, "v2").unwrap();

        let deadline = Instant::now() + Duration::from_secs(20);
        let mut seen: Vec<(uuid::Uuid, std::path::PathBuf)> = Vec::new();
        let mut saw_a_change = false;
        let mut raw = 0usize;
        let count_raw = |w: &inf_asset::AssetWatcher| {
            w.drain()
                .iter()
                .filter(|c| inf_script::is_script_path(c.path()))
                .count()
        };
        while Instant::now() < deadline && seen.is_empty() {
            let out = tick(&mut queue, &project, Some(&watcher));
            raw += count_raw(&probe);
            saw_a_change |= out.content_changed;
            seen = out.scripts;
            std::thread::yield_now();
        }
        // The probe keeps its own debounce timer, started a moment after the
        // watcher's, so it is usually still holding the batch when the tick
        // wins the race — give it a bounded chance rather than print a zero
        // that says nothing about the platform. A whole batch arrives at once.
        let probe_deadline = Instant::now() + Duration::from_secs(5);
        while raw == 0 && Instant::now() < probe_deadline {
            raw += count_raw(&probe);
            std::thread::sleep(Duration::from_millis(2));
        }
        println!(
            "this platform's watcher delivered {raw} raw change event(s) for one \
             save; the tick reported {}. On ubuntu CI the same save arrived as 3 \
             — the difference this collapse exists to keep out of how often a \
             running world is re-lowered. A platform printing 1 here cannot \
             falsify the collapse; ubuntu can, and did",
            seen.len()
        );
        assert!(saw_a_change, "the watcher never reported anything at all");
        assert_eq!(
            seen.len(),
            1,
            "the tick reported {seen:?} — a `.infini` that changed must arrive \
             here ONCE per save, whatever the platform's watcher does, and \
             nothing else may. Linux delivers three inotify events for one \
             `fs::write` (create, modify-data, close-write) where Windows \
             delivers one; carried one-for-one that is three compiles and three \
             class swaps for one Ctrl+S, so the drain is collapsed by GUID"
        );
        assert_eq!(
            seen[0].0, guid,
            "editing the script CHANGED its asset id. A level binds a script \
             through `ActorClass(Uuid)`, so hot reload would queue the class \
             under a GUID no actor holds and swap nothing — and PIE, Simulate \
             and the cook would then resolve that binding to None and run the \
             actor with no gameplay logic at all. `pin_script_ids` is what \
             stops the id being re-derived from bytes that moved"
        );

        // The path is the SECONDARY claim, and what it owes is that Ring 2 can
        // OPEN it — `hot_reload_scripts` hands it straight to `compile_path` —
        // so it is asserted by reading it back rather than by matching a string.
        assert_eq!(
            std::fs::read_to_string(&seen[0].1).expect("the reported path opens"),
            V2,
            "the path handed to `compile_path` must open as the EDIT that was \
             just saved, or hot reload recompiles something else"
        );
        // …and it is the database's own normalized spelling of that file.
        // Canonicalized on BOTH sides at one door: on macOS the watcher reports
        // `/private/var/folders/…` for a tempdir handed out as `/var/folders/…`
        // (the `/var` → `/private/var` symlink), so a raw tempdir path is the
        // one spelling that is wrong there. `AssetEntry::path` goes through
        // `inf_asset`'s `normalize`, which is `fs::canonicalize` when the file
        // exists — so this compares two doors' answers rather than restating one.
        assert_eq!(
            seen[0].1,
            std::fs::canonicalize(&path).expect("the script is on disk"),
            "the tick must report the path the DATABASE holds for that GUID"
        );
    }

    #[test]
    fn submitting_a_bad_source_reports_failure() {
        let dir = tempfile::tempdir().unwrap();
        let project = Arc::new(Mutex::new(AssetProject::open(dir.path()).unwrap()));
        let mut queue = ImportQueue::spawn(project);
        let dest = dir.path().join("imported");
        let id = queue.submit(PathBuf::from("does-not-exist.png"), dest);

        // First a Started, then a Failed for this job.
        let mut saw_started = false;
        let mut saw_failed = false;
        for _ in 0..2 {
            match queue.recv().unwrap() {
                ImportProgress::Started { id: jid, .. } => {
                    assert_eq!(jid, id);
                    saw_started = true;
                }
                ImportProgress::Failed { id: jid, .. } => {
                    assert_eq!(jid, id);
                    saw_failed = true;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert!(saw_started && saw_failed);
    }

    fn write_png(dir: &std::path::Path, w: u32, h: u32) -> PathBuf {
        let samples = (0..w as u64 * h as u64)
            .map(|i| ((i * 6367) % 65536) as u16)
            .collect();
        let bytes = encode_png16(&HeightImage {
            width: w,
            height: h,
            samples,
        })
        .unwrap();
        let path = dir.join("Heights.png");
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn a_terrain_job_reports_progress_then_finishes() {
        let src = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), 129, 129);
        let project = Arc::new(Mutex::new(AssetProject::open(dir.path()).unwrap()));
        let mut queue = ImportQueue::spawn(project.clone());
        let id = queue.submit_terrain(
            png,
            TerrainImportSettings {
                tile_resolution: 33,
                meters_per_sample: 8.0,
                ..Default::default()
            },
            None,
            Some("World".into()),
        );

        let mut ticks = 0usize;
        let produced;
        loop {
            match queue.recv().expect("worker alive") {
                ImportProgress::Started { id: j, .. } => assert_eq!(j, id),
                ImportProgress::Progress {
                    id: j, done, total, ..
                } => {
                    assert_eq!(j, id);
                    assert!(done <= total && total > 0);
                    ticks += 1;
                }
                ImportProgress::Finished {
                    id: j,
                    primary,
                    advisories,
                    ..
                } => {
                    assert_eq!(j, id);
                    // **Round-2 finding R2.F7.** This arm hard-coded
                    // `advisories: Vec::new()` under a comment reasoning only
                    // about TEXTURE advisories, so the channel to the Content
                    // Drawer's badge posted empty — and the terrain wizard is
                    // the one importer where an outside-the-project source is
                    // the NORM, which is exactly what L7.H7's advisory says.
                    // The heightmap here lives in its own tempdir, outside the
                    // project, so the note must arrive with the finish.
                    assert_eq!(advisories.len(), 1, "{advisories:?}");
                    assert!(
                        advisories[0].contains("outside the project"),
                        "{advisories:?}"
                    );
                    produced = primary;
                    break;
                }
                ImportProgress::Failed { error, .. } => panic!("terrain import failed: {error}"),
            }
        }
        assert!(ticks > 1, "no progress ticks");
        let asset = produced.expect("a terrain asset");
        assert!(lock(&project).db().contains(asset));
    }

    /// A georeferenced GeoTIFF whose top-left sample sits at a stated easting
    /// and northing in UTM zone 10N.
    fn write_geotiff(dir: &std::path::Path, n: u32, east: f64, north: f64) -> PathBuf {
        use tiff::encoder::{colortype, TiffEncoder};
        use tiff::tags::Tag;

        let samples: Vec<f32> = (0..n * n).map(|i| (i % 97) as f32).collect();
        let path = dir.join("Dem.tif");
        let file = std::fs::File::create(&path).unwrap();
        {
            let mut enc = TiffEncoder::new(std::io::BufWriter::new(file)).unwrap();
            let mut img = enc.new_image::<colortype::Gray32Float>(n, n).unwrap();
            img.rows_per_strip(8).unwrap();
            {
                let d = img.encoder();
                d.write_tag(Tag::Unknown(33550), &[10.0f64, 10.0, 0.0][..])
                    .unwrap();
                d.write_tag(
                    Tag::Unknown(33922),
                    &[0.0f64, 0.0, 0.0, east, north, 0.0][..],
                )
                .unwrap();
                // model type = projected, CRS = EPSG:32610, units = metres.
                d.write_tag(
                    Tag::Unknown(34735),
                    &[
                        1u16, 1, 0, 3, 1024, 0, 1, 1, 3072, 0, 1, 32610, 3076, 0, 1, 9001,
                    ][..],
                )
                .unwrap();
            }
            img.write_data(&samples).unwrap();
        }
        path
    }

    /// **The anchor has to REACH the payload** (Wave-G audit A3).
    ///
    /// `TerrainImportSettings::use_georeference` was recorded in the sidecar and
    /// read by `world_origin`, and `build_anchored` threaded an anchor into
    /// `ChunkedImportOptions::world_origin`, and `chunked.rs` wrote it into the
    /// header — every link was correct and the chain had no first link. The only
    /// caller of `build_anchored` was `build`, which passes `None`, so *every*
    /// asset this editor could produce carried `origin == DVec3::ZERO`. Turning
    /// the flag on made it worse than leaving it off: it suppressed the "imported
    /// at the world origin" advisory while still importing at the world origin.
    ///
    /// This asserts the **world** — the bytes of the built payload — rather than
    /// the report or the settings, because the settings were right the whole
    /// time.
    ///
    /// Un-fix mutation: pass `None` for the anchor at the `build_anchored` call
    /// in `worker_loop` and the origin below comes back as zero.
    #[test]
    fn a_queued_georeferenced_import_places_the_payload_against_the_level_anchor() {
        let src = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        // The DEM's top-left sample is 2 km east and 3 km NORTH of the anchor.
        let tif = write_geotiff(src.path(), 65, 493_000.0, 5_462_000.0);
        let anchor = inf_math::geo::GeoAnchor {
            enabled: true,
            crs: "EPSG:32610".into(),
            origin_easting_m: 491_000.0,
            origin_northing_m: 5_459_000.0,
            origin_height_m: 0.0,
            origin_latitude_deg: 49.28,
            origin_longitude_deg: -123.12,
            grid_convergence_deg: 0.0,
            vertical_datum: "EGM2008".into(),
        };
        let settings = TerrainImportSettings {
            tile_resolution: 17,
            meters_per_sample: 10.0,
            float_meters: true,
            center: false,
            use_georeference: true,
            ..Default::default()
        };

        let project = Arc::new(Mutex::new(AssetProject::open(dir.path()).unwrap()));
        let mut queue = ImportQueue::spawn(project.clone());
        let id = queue.submit_terrain(tif, settings, Some(anchor), Some("Placed".into()));

        let asset = loop {
            match queue.recv().expect("worker alive") {
                ImportProgress::Finished { id: j, primary, .. } => {
                    assert_eq!(j, id);
                    break primary.expect("an asset");
                }
                ImportProgress::Failed { error, .. } => panic!("import failed: {error}"),
                _ => {}
            }
        };

        let path = lock(&project)
            .db()
            .get(asset)
            .expect("registered")
            .path
            .clone();
        let bytes = std::fs::read(&path).expect("the payload was written");
        let reader = inf_terrain::TerrainAssetReader::new(&bytes[..]).expect("a valid payload");
        let origin = reader.header().origin;
        assert!(
            (origin.x - 2_000.0).abs() < 1e-6,
            "2 km east of the anchor is +2000 X; the header says {origin:?}. A \
             zero origin means the anchor never reached the importer and the \
             terrain will not line up with any vector layer from the same survey"
        );
        assert!(
            (origin.z + 3_000.0).abs() < 1e-6,
            "3 km north is -3000 Z (north is -Z, the solar frame); got {origin:?}"
        );
        assert_eq!(origin.y, 0.0, "the height datum belongs to the samples");

        // …and the same import WITHOUT an anchor lands at the world origin and
        // says so, rather than landing there silently.
        let dir2 = tempfile::tempdir().unwrap();
        let tif2 = write_geotiff(src.path(), 65, 493_000.0, 5_462_000.0);
        let project2 = Arc::new(Mutex::new(AssetProject::open(dir2.path()).unwrap()));
        let mut queue2 = ImportQueue::spawn(project2.clone());
        let id2 = queue2.submit_terrain(
            tif2,
            TerrainImportSettings {
                tile_resolution: 17,
                meters_per_sample: 10.0,
                float_meters: true,
                center: false,
                use_georeference: true,
                ..Default::default()
            },
            None,
            Some("Unplaced".into()),
        );
        let advisories = loop {
            match queue2.recv().expect("worker alive") {
                ImportProgress::Finished {
                    id: j, advisories, ..
                } => {
                    assert_eq!(j, id2);
                    break advisories;
                }
                ImportProgress::Failed { error, .. } => panic!("{error}"),
                _ => {}
            }
        };
        assert!(
            advisories
                .iter()
                .any(|a| a.contains("still placed at the world origin")),
            "asking for placement and not getting it must be REPORTED: {advisories:?}"
        );
    }

    /// **The lock-freedom gate (P16.4a audit).** A terrain import must make
    /// progress — and stay cancellable — while something else holds the project
    /// mutex, because the editor's asset commands and this very progress tick all
    /// take it. The test *is* the contention: it holds the lock for the whole
    /// import and asserts events keep arriving and Cancel still lands.
    #[test]
    fn a_terrain_import_progresses_and_cancels_while_the_project_is_locked() {
        let src = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        // Big enough to emit many tile rows, small enough for CI.
        let png = write_png(src.path(), 257, 257);
        let project = Arc::new(Mutex::new(AssetProject::open(dir.path()).unwrap()));
        let mut queue = ImportQueue::spawn(project.clone());

        // Hold the project for the entire import.
        let guard = project.lock().expect("uncontended at this point");

        let id = queue.submit_terrain(
            png,
            TerrainImportSettings {
                tile_resolution: 17,
                meters_per_sample: 4.0,
                ..Default::default()
            },
            None,
            Some("Locked".into()),
        );

        // Progress flows: if `build` took the project lock, this would deadlock.
        let mut ticks = 0usize;
        while ticks < 3 {
            match queue.recv().expect("worker alive") {
                ImportProgress::Progress { id: j, .. } => {
                    assert_eq!(j, id);
                    ticks += 1;
                }
                ImportProgress::Started { .. } => {}
                other => panic!("import ended before 3 progress ticks: {other:?}"),
            }
        }

        // Cancel lands while the lock is still held…
        assert!(queue.cancel(id), "the job is cancellable mid-build");

        // …and the job reports terminally WITHOUT ever taking the project (the
        // commit phase checks the flag before the lock). Still holding `guard`.
        loop {
            match queue.recv().expect("worker alive") {
                ImportProgress::Failed { error, .. } => {
                    assert!(error.contains("cancelled"), "got {error}");
                    break;
                }
                ImportProgress::Finished { .. } => {
                    panic!("a cancelled job committed while the project was locked")
                }
                _ => {}
            }
        }
        assert!(guard.db().is_empty(), "a cancelled job registered an asset");
        drop(guard);
    }

    /// The tick is the thing Ring 2 runs on a timer; it must never block on the
    /// project either, or it stops carrying the progress events above.
    #[test]
    fn the_tick_carries_events_without_blocking_on_a_busy_project() {
        let src = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), 129, 129);
        let project = Arc::new(Mutex::new(AssetProject::open(dir.path()).unwrap()));
        let mut queue = ImportQueue::spawn(project.clone());
        let id = queue.submit_terrain(
            png,
            TerrainImportSettings {
                tile_resolution: 17,
                ..Default::default()
            },
            None,
            Some("Ticked".into()),
        );

        // Busy project → the tick still returns, just without a version.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let mut saw_progress = false;
        {
            let _guard = project.lock().unwrap();
            while !saw_progress && std::time::Instant::now() < deadline {
                let out = tick(&mut queue, &project, None);
                assert!(out.version.is_none(), "the tick read a locked project");
                assert!(
                    !out.events
                        .iter()
                        .any(|e| matches!(e, ImportProgress::Finished { .. })),
                    "a job committed while the project was locked"
                );
                saw_progress = out
                    .events
                    .iter()
                    .any(|e| matches!(e, ImportProgress::Progress { .. }));
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        assert!(saw_progress, "the tick never carried a progress event");

        // Once free, the tick reports the finish, the version, and — crucially —
        // that a terrain landed, which is what refreshes the viewport's index.
        let mut finished = false;
        let mut terrain = false;
        while !finished && std::time::Instant::now() < deadline {
            let out = tick(&mut queue, &project, None);
            terrain |= out.index_stale;
            if out.events.iter().any(|e| match e {
                ImportProgress::Finished { id: j, .. } => *j == id,
                ImportProgress::Failed { error, .. } => panic!("import failed: {error}"),
                _ => false,
            }) {
                finished = true;
                assert!(out.content_changed);
                assert!(out.version.is_some());
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(finished, "the import never finished");
        assert!(terrain, "a finished terrain job must flag index_stale");
    }

    /// A grid mesh heavy enough that `build_vgeom` is not instantaneous — the
    /// sweep has to be observably *doing work* for the contention test below to
    /// mean anything.
    fn heavy_mesh(n: u32) -> inf_mesh::MeshAsset {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                vertices.push(inf_mesh::MeshVertex {
                    position: [i as f32, ((i * j) % 11) as f32 * 0.2, j as f32],
                    normal: [0.0, 1.0, 0.0],
                    uv: [i as f32 / n as f32, j as f32 / n as f32],
                    ..Default::default()
                });
            }
        }
        let idx = |i: u32, j: u32| j * (n + 1) + i;
        for j in 0..n {
            for i in 0..n {
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        inf_mesh::MeshAsset::new(
            vec![inf_mesh::SubMesh {
                name: "grid".into(),
                vertices,
                indices,
                material_slot: None,
                skin: Vec::new(),
            }],
            vec![],
        )
    }

    /// **The lock-freedom gate for the vmesh sweep (P18.3 audit).** The same
    /// property `a_terrain_import_progresses_and_cancels_while_the_project_is_locked`
    /// pins, through the door P18.3 added: `build_vgeom` runs for seconds per mesh,
    /// and doing it under the project mutex would freeze every asset command, the
    /// progress tick, and this job's own Cancel — the P16.4a stall, re-introduced.
    ///
    /// The test *is* the contention: it holds the project for the whole sweep and
    /// asserts progress keeps arriving and Cancel still lands.
    #[test]
    fn a_vmesh_sweep_progresses_and_cancels_while_the_project_is_locked() {
        let dir = tempfile::tempdir().unwrap();
        let project = Arc::new(Mutex::new(AssetProject::open(dir.path()).unwrap()));
        {
            let mut proj = lock(&project);
            let d = proj.content_dir("Meshes").unwrap();
            for k in 0..6 {
                proj.write_asset(
                    &d,
                    &format!("M{k}"),
                    &heavy_mesh(40 + k),
                    None,
                    vec![],
                    None,
                )
                .unwrap();
            }
        }
        let mut queue = ImportQueue::spawn(project.clone());
        let id = queue.submit_vmesh_sweep();

        // Wait for the "planned" tick before contending: it is emitted once PHASE 1
        // has released the lock, so what follows tests the BUILD phase rather than
        // racing the plan phase (grabbing the mutex first would simply park the
        // worker in phase 1 and prove nothing).
        let total = loop {
            match queue.recv().expect("worker alive") {
                ImportProgress::Progress {
                    id: j,
                    total,
                    ref stage,
                    ..
                } if stage == "planned" => {
                    assert_eq!(j, id);
                    break total;
                }
                ImportProgress::Started { .. } | ImportProgress::Progress { .. } => {}
                other => panic!("the sweep ended before planning finished: {other:?}"),
            }
        };
        assert!(
            total >= 6,
            "the sweep planned {total} meshes, expected >= 6"
        );

        // Now hold the project across the build phase.
        let guard = project.lock().expect("phase 1 released it");

        // A "built" tick still arrives: `build_vgeom` ran to completion while this
        // thread held the project. If the build phase took the lock, this deadlocks
        // — which is precisely the P16.4a failure, re-introduced through a new door.
        loop {
            match queue.recv().expect("worker alive") {
                ImportProgress::Progress {
                    id: j, ref stage, ..
                } if stage == "built" => {
                    assert_eq!(j, id);
                    break;
                }
                ImportProgress::Progress { .. } => {}
                other => panic!("the sweep ended before a contended build: {other:?}"),
            }
        }

        // Cancel lands while the lock is still held…
        assert!(queue.cancel(id), "the sweep is cancellable mid-build");

        // …and the job reports terminally WITHOUT ever taking the project: the
        // commit phase polls `try_lock` and re-checks the flag, so it is never
        // parked somewhere a cancellation cannot reach it. Still holding `guard`.
        loop {
            match queue.recv().expect("worker alive") {
                ImportProgress::Finished { produced, .. } => {
                    assert!(
                        produced.is_empty(),
                        "a cancelled sweep committed while the project was locked"
                    );
                    break;
                }
                ImportProgress::Failed { error, .. } => panic!("sweep failed: {error}"),
                _ => {}
            }
        }
        assert!(
            guard
                .db()
                .iter()
                .all(|e| e.kind() != inf_asset::AssetKind::MeshletMesh),
            "a cancelled sweep registered a derived asset"
        );
        drop(guard);
    }

    /// Uncontended, the sweep does its job: every mesh gains a derived DAG, and a
    /// second sweep is all cache hits (it plans nothing, so it reports nothing).
    #[test]
    fn an_uncontended_sweep_derives_every_mesh_then_goes_quiet() {
        let dir = tempfile::tempdir().unwrap();
        let project = Arc::new(Mutex::new(AssetProject::open(dir.path()).unwrap()));
        {
            let mut proj = lock(&project);
            let d = proj.content_dir("Meshes").unwrap();
            for k in 0..3 {
                proj.write_asset(&d, &format!("M{k}"), &heavy_mesh(8 + k), None, vec![], None)
                    .unwrap();
            }
        }
        let mut queue = ImportQueue::spawn(project.clone());

        let run = |queue: &mut ImportQueue| loop {
            match queue.recv().expect("worker alive") {
                ImportProgress::Finished { produced, .. } => return produced,
                ImportProgress::Failed { error, .. } => panic!("sweep failed: {error}"),
                _ => {}
            }
        };
        queue.submit_vmesh_sweep();
        assert_eq!(run(&mut queue).len(), 3, "all three derived");
        queue.submit_vmesh_sweep();
        assert!(run(&mut queue).is_empty(), "second sweep is all hits");
    }

    #[test]
    fn cancelling_a_terrain_job_reports_it_and_registers_nothing() {
        let src = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), 129, 129);
        let project = Arc::new(Mutex::new(AssetProject::open(dir.path()).unwrap()));
        let mut queue = ImportQueue::spawn(project.clone());
        let id = queue.submit_terrain(png, TerrainImportSettings::default(), None, None);
        assert!(queue.cancel(id), "the job is cancellable");
        assert!(!queue.cancel(id + 1), "an unknown job is not");

        // The job still reports terminally, so the UI needs no special case. It
        // may also have raced to completion before the flag was seen — either way
        // nothing is left half-written.
        loop {
            match queue.recv().expect("worker alive") {
                ImportProgress::Failed { error, .. } => {
                    assert!(error.contains("cancelled"), "got {error}");
                    assert!(lock(&project).db().is_empty());
                    break;
                }
                ImportProgress::Finished { .. } => break,
                _ => {}
            }
        }
    }
}
