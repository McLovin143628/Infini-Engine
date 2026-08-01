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

use std::collections::{HashMap, HashSet};
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
        name: Option<String>,
        cancel: CancelToken,
    },
}

impl Job {
    fn id(&self) -> u64 {
        match self {
            Job::File { id, .. } | Job::Terrain { id, .. } => *id,
        }
    }
    fn source(&self) -> &PathBuf {
        match self {
            Job::File { source, .. } | Job::Terrain { source, .. } => source,
        }
    }
}

/// One drain of the progress channel.
#[derive(Debug, Default)]
pub struct PollBatch {
    /// The events, in the order the worker produced them.
    pub events: Vec<ImportProgress>,
    /// A **terrain** job finished in this batch, so a new `.inf_terrain` is on
    /// disk. Recorded here rather than derived from the asset database, because
    /// the tick that reads this must never block on the project mutex.
    pub terrain_finished: bool,
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

    /// Queue a chunked heightmap → `.inf_terrain` import (P16.4). Returns the job
    /// id; progress ticks and the terminal event arrive on the same channel.
    pub fn submit_terrain(
        &mut self,
        source: PathBuf,
        settings: TerrainImportSettings,
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
        let mut terrain_finished = false;
        for ev in &events {
            let terminal = matches!(
                ev,
                ImportProgress::Finished { .. } | ImportProgress::Failed { .. }
            );
            if terminal {
                let id = ev.job_id();
                self.cancels.remove(&id);
                let was_terrain = self.terrain_jobs.remove(&id);
                if was_terrain && matches!(ev, ImportProgress::Finished { .. }) {
                    terrain_finished = true;
                }
            }
        }
        PollBatch {
            events,
            terrain_finished,
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
        let source = job.source().clone();
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
                    terrain_import::build(&source, &settings, &mut on_progress, &cancel)
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
                    },
                    Err(e) => ImportProgress::Failed {
                        id,
                        source,
                        error: e.to_string(),
                    },
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
    /// A `.inf_terrain` appeared or changed → the viewport's terrain index must
    /// be refreshed, or a freshly imported terrain resolves to nothing and the
    /// entity the wizard just spawned draws empty.
    pub terrain_changed: bool,
    /// The content version to publish, or `None` when the project was busy.
    pub version: Option<u64>,
}

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
        terrain_changed: batch.terrain_finished,
        events: batch.events,
        version: None,
    };

    // One short, non-blocking visit to the project: apply the watcher's batch and
    // read the version. If it is busy (a commit, a snapshot), both wait a tick.
    if let Ok(mut proj) = project.try_lock() {
        if let Some(w) = watcher {
            let changes = w.drain();
            if !changes.is_empty() {
                for c in changes {
                    let path = match &c {
                        AssetChange::Upserted(p) | AssetChange::Removed(p) => p.clone(),
                    };
                    if path.extension().and_then(|s| s.to_str()) == Some("inf_terrain") {
                        out.terrain_changed = true;
                    }
                    match c {
                        AssetChange::Upserted(p) => {
                            let _ = proj.db_mut().rescan_path(&p);
                        }
                        AssetChange::Removed(p) => {
                            proj.db_mut().remove_path(&p);
                        }
                    }
                }
                proj.bump();
                out.content_changed = true;
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
                ImportProgress::Finished { id: j, primary, .. } => {
                    assert_eq!(j, id);
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
            terrain |= out.terrain_changed;
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
        assert!(terrain, "a finished terrain job must flag terrain_changed");
    }

    #[test]
    fn cancelling_a_terrain_job_reports_it_and_registers_nothing() {
        let src = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), 129, 129);
        let project = Arc::new(Mutex::new(AssetProject::open(dir.path()).unwrap()));
        let mut queue = ImportQueue::spawn(project.clone());
        let id = queue.submit_terrain(png, TerrainImportSettings::default(), None);
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
