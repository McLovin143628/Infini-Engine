//! Content-directory watching.
//!
//! Wraps `notify` + `notify-debouncer-full` so the editor gets a small stream of
//! coalesced [`AssetChange`]s instead of the raw firehose (editors and syncs
//! touch files in bursts). The consumer (Ring 2) drains the channel on a tick
//! and feeds each change to [`AssetDb::rescan_path`](crate::AssetDb::rescan_path)
//! / [`remove_path`](crate::AssetDb::remove_path), then emits `assets://changed`.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

use notify::{EventKind, RecursiveMode};
use notify_debouncer_full::{new_debouncer, DebounceEventResult};

use crate::error::Result;
use crate::sidecar::is_sidecar;

/// A coalesced change under the content root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetChange {
    /// A payload was created or modified — rescan it.
    Upserted(PathBuf),
    /// A payload was removed — drop it from the index.
    Removed(PathBuf),
}

impl AssetChange {
    pub fn path(&self) -> &Path {
        match self {
            AssetChange::Upserted(p) | AssetChange::Removed(p) => p,
        }
    }
}

/// A live recursive watcher over a content root. Dropping it stops watching.
pub struct AssetWatcher {
    // Kept alive for its background thread; field is otherwise unused.
    _debouncer: notify_debouncer_full::Debouncer<
        notify::RecommendedWatcher,
        notify_debouncer_full::RecommendedCache,
    >,
    rx: Receiver<AssetChange>,
}

impl AssetWatcher {
    /// Start watching `root` recursively. Changes arrive debounced by `debounce`
    /// (250 ms is a good default).
    pub fn watch(root: &Path, debounce: Duration) -> Result<Self> {
        let (tx, rx) = channel::<AssetChange>();

        let mut debouncer = new_debouncer(debounce, None, move |res: DebounceEventResult| {
            let events = match res {
                Ok(events) => events,
                Err(errs) => {
                    for e in errs {
                        tracing::warn!("asset watch error: {e}");
                    }
                    return;
                }
            };
            for ev in events {
                let removed = matches!(ev.kind, EventKind::Remove(_));
                for path in &ev.paths {
                    // Sidecars are followed via their payload; ignore them so we
                    // don't process an asset twice per save.
                    if is_sidecar(path) {
                        continue;
                    }
                    let change = if removed {
                        AssetChange::Removed(path.clone())
                    } else {
                        AssetChange::Upserted(path.clone())
                    };
                    // Receiver hung up → nothing to do.
                    let _ = tx.send(change);
                }
            }
        })
        .map_err(|e| crate::error::AssetError::Import(format!("watcher: {e}")))?;

        debouncer
            .watch(root, RecursiveMode::Recursive)
            .map_err(|e| crate::error::AssetError::Import(format!("watch {root:?}: {e}")))?;

        Ok(Self {
            _debouncer: debouncer,
            rx,
        })
    }

    /// Drain all changes available right now (non-blocking).
    pub fn drain(&self) -> Vec<AssetChange> {
        self.rx.try_iter().collect()
    }

    /// Block for the next change (used in tests / dedicated threads).
    pub fn recv(&self) -> Option<AssetChange> {
        self.rx.recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watcher_constructs_on_a_real_dir() {
        let dir = tempfile::tempdir().unwrap();
        // Construction spins up the OS watch + debounce thread; dropping stops it.
        let w = AssetWatcher::watch(dir.path(), Duration::from_millis(50)).unwrap();
        assert!(w.drain().is_empty(), "no changes yet");
    }
}
