//! An async import queue with progress events.
//!
//! Imports (glTF decode, texture compression, mip generation) are heavy, so they
//! run on a background worker rather than blocking the editor. Callers `submit`
//! a source path + destination and drain [`ImportProgress`] events on a tick;
//! Ring 2 forwards those to the webview as `assets://import`. The worker shares
//! the [`AssetProject`] behind a mutex with the command layer, so imported
//! assets appear in the same database the Content Drawer reads.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use inf_asset::AssetId;

use super::AssetProject;

/// A progress event for one import job.
#[derive(Debug, Clone)]
pub enum ImportProgress {
    /// The worker picked up job `id` for `source`.
    Started { id: u64, source: PathBuf },
    /// Job `id` produced these assets (primary first, if any).
    Finished {
        id: u64,
        source: PathBuf,
        produced: Vec<AssetId>,
        primary: Option<AssetId>,
        cached: bool,
    },
    /// Job `id` failed.
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
            | ImportProgress::Finished { id, .. }
            | ImportProgress::Failed { id, .. } => *id,
        }
    }
}

struct Job {
    id: u64,
    source: PathBuf,
    dest: PathBuf,
}

/// The queue: a worker thread + a progress channel.
pub struct ImportQueue {
    tx: Sender<Job>,
    events: Receiver<ImportProgress>,
    next_id: u64,
    worker: Option<JoinHandle<()>>,
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
        }
    }

    /// Queue an import; returns the job id (referenced by later progress events).
    pub fn submit(&mut self, source: PathBuf, dest: PathBuf) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        // If the worker has gone away the send fails; the caller sees no events.
        let _ = self.tx.send(Job { id, source, dest });
        id
    }

    /// Drain all progress available right now (non-blocking).
    pub fn poll(&self) -> Vec<ImportProgress> {
        self.events.try_iter().collect()
    }

    /// Block for the next event (tests).
    pub fn recv(&self) -> Option<ImportProgress> {
        self.events.recv().ok()
    }
}

impl Drop for ImportQueue {
    fn drop(&mut self) {
        // Dropping the sender ends the worker loop; then join it.
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
        let _ = etx.send(ImportProgress::Started {
            id: job.id,
            source: job.source.clone(),
        });
        let result = {
            let mut proj = match project.lock() {
                Ok(p) => p,
                Err(poisoned) => poisoned.into_inner(),
            };
            proj.import_file(&job.source, &job.dest)
        };
        let event = match result {
            Ok(out) => ImportProgress::Finished {
                id: job.id,
                source: job.source,
                produced: out.produced,
                primary: out.primary,
                cached: out.cached,
            },
            Err(e) => ImportProgress::Failed {
                id: job.id,
                source: job.source,
                error: e.to_string(),
            },
        };
        let _ = etx.send(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
