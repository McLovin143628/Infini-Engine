//! The **null console** backend (ROADMAP §9, P14.4 item 1).
//!
//! A complete, dependency-free implementation of every [`inf_platform`] HAL seam
//! as an in-memory / no-op / deterministic mock. Its job is to *prove the seam
//! holds*: if the engine's platform needs can be satisfied by a backend that has
//! no OS, no GPU, no clock, and no devices, then a real console backend (a private
//! out-of-tree crate) is an additive implementation of the same traits rather than
//! an engine refactor.
//!
//! ## The out-of-tree private-backend pattern (demonstrated by this crate)
//!
//! This crate lives under `platforms/`, *not* `crates/`, and is **not** a
//! dependency of any engine crate — it is selected at the top level (a player
//! build, a test) exactly as a console backend would be. A real PS5/Xbox/Switch
//! backend is the same shape in a **separate private repository** (NDA SDKs cannot
//! live here): `platforms/inf-platform-<console>/` implementing these traits,
//! added to that private workspace and selected by the cook target. Nothing in
//! Ring 0 names it; the seam is the only contract. See
//! `docs/memos/p14-console-hal-audit.md`.
//!
//! ## Determinism
//!
//! The clock is a *fake* monotonic counter advanced only by [`NullClock::advance`]
//! — it never consults a wall clock, so a null-backed run is bit-reproducible
//! (the §2.5 rule extended to the platform layer). Threads run their closure
//! **inline** on `spawn`, so ordering is deterministic too.

use std::collections::BTreeMap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use inf_platform::{
    Clock, DisplayTarget, FileSystem, Hal, InputBackend, InputEvent, JoinToken, PresentMode,
    SaveDataStore, SurfaceSize, ThreadSpawner,
};

/// A deterministic fake clock: monotonic nanoseconds advanced explicitly.
#[derive(Default)]
pub struct NullClock {
    nanos: AtomicU64,
}

impl NullClock {
    /// Advance the fake monotonic clock by `d` (the only way time moves).
    pub fn advance(&self, d: Duration) {
        self.nanos.fetch_add(d.as_nanos() as u64, Ordering::SeqCst);
    }
}

impl Clock for NullClock {
    fn now_monotonic(&self) -> Duration {
        Duration::from_nanos(self.nanos.load(Ordering::SeqCst))
    }

    fn now_unix_nanos(&self) -> u128 {
        // A fixed epoch (2020-01-01T00:00:00Z in ns) + the monotonic offset, so
        // there is no dependence on the host wall clock.
        const FIXED_EPOCH_NS: u128 = 1_577_836_800_000_000_000;
        FIXED_EPOCH_NS + self.nanos.load(Ordering::SeqCst) as u128
    }
}

/// An in-memory filesystem (a flat `path -> bytes` map with directory listing).
#[derive(Default)]
pub struct MemFileSystem {
    files: Mutex<BTreeMap<String, Vec<u8>>>,
}

impl MemFileSystem {
    fn norm(path: &str) -> String {
        path.split('/')
            .filter(|s| !s.is_empty() && *s != ".")
            .collect::<Vec<_>>()
            .join("/")
    }
}

impl FileSystem for MemFileSystem {
    fn read(&self, path: &str) -> io::Result<Vec<u8>> {
        let key = Self::norm(path);
        self.files
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, key))
    }

    fn write(&self, path: &str, data: &[u8]) -> io::Result<()> {
        self.files
            .lock()
            .unwrap()
            .insert(Self::norm(path), data.to_vec());
        Ok(())
    }

    fn exists(&self, path: &str) -> bool {
        self.files.lock().unwrap().contains_key(&Self::norm(path))
    }

    fn remove(&self, path: &str) -> io::Result<()> {
        self.files.lock().unwrap().remove(&Self::norm(path));
        Ok(())
    }

    fn list(&self, path: &str) -> io::Result<Vec<String>> {
        let prefix = Self::norm(path);
        let files = self.files.lock().unwrap();
        let mut children = std::collections::BTreeSet::new();
        for key in files.keys() {
            let rest = if prefix.is_empty() {
                Some(key.as_str())
            } else {
                key.strip_prefix(&prefix).and_then(|r| r.strip_prefix('/'))
            };
            if let Some(rest) = rest {
                if let Some(first) = rest.split('/').next() {
                    if !first.is_empty() {
                        children.insert(first.to_string());
                    }
                }
            }
        }
        Ok(children.into_iter().collect())
    }
}

/// In-memory save-data slots.
#[derive(Default)]
pub struct MemSaveStore {
    slots: Mutex<BTreeMap<String, Vec<u8>>>,
}

impl SaveDataStore for MemSaveStore {
    fn save(&self, slot: &str, data: &[u8]) -> io::Result<()> {
        self.slots
            .lock()
            .unwrap()
            .insert(slot.to_string(), data.to_vec());
        Ok(())
    }

    fn load(&self, slot: &str) -> io::Result<Option<Vec<u8>>> {
        Ok(self.slots.lock().unwrap().get(slot).cloned())
    }

    fn list_slots(&self) -> io::Result<Vec<String>> {
        Ok(self.slots.lock().unwrap().keys().cloned().collect())
    }

    fn remove(&self, slot: &str) -> io::Result<()> {
        self.slots.lock().unwrap().remove(slot);
        Ok(())
    }
}

/// A join token for an inline-run "thread" (already finished).
pub struct InlineJoin;

impl JoinToken for InlineJoin {
    fn join(self: Box<Self>) {}
}

/// A thread spawner that runs the closure **inline** (deterministic ordering).
#[derive(Default)]
pub struct InlineThreadSpawner;

impl ThreadSpawner for InlineThreadSpawner {
    fn spawn(&self, _name: &str, f: Box<dyn FnOnce() + Send + 'static>) -> Box<dyn JoinToken> {
        f();
        Box::new(InlineJoin)
    }
}

/// A headless display target (no swapchain).
pub struct NullDisplay {
    size: SurfaceSize,
}

impl Default for NullDisplay {
    fn default() -> Self {
        Self {
            size: SurfaceSize {
                width: 1280,
                height: 720,
                scale_factor: 1.0,
            },
        }
    }
}

impl DisplayTarget for NullDisplay {
    fn surface_size(&self) -> SurfaceSize {
        self.size
    }

    fn present_mode(&self) -> PresentMode {
        PresentMode::Headless
    }

    fn is_headless(&self) -> bool {
        true
    }
}

/// An input backend with no devices and no events.
#[derive(Default)]
pub struct NullInputBackend;

impl InputBackend for NullInputBackend {
    fn poll(&self) -> Vec<InputEvent> {
        Vec::new()
    }

    fn connected_gamepads(&self) -> usize {
        0
    }
}

/// Assemble a fully-mock HAL. Returns the [`Hal`] plus the concrete [`NullClock`]
/// (so a test can advance time) — the clock inside the `Hal` is the same `Arc`.
pub fn null_hal() -> (Hal, Arc<NullClock>) {
    let clock = Arc::new(NullClock::default());
    let hal = Hal::new(
        clock.clone(),
        Arc::new(MemFileSystem::default()),
        Arc::new(MemSaveStore::default()),
        Arc::new(InlineThreadSpawner),
        Arc::new(NullDisplay::default()),
        Arc::new(NullInputBackend),
    );
    (hal, clock)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_platform::{Lifecycle, REQUIRED_SEAMS};

    // ── the P14.4 audit assertion: the null backend implements EVERY required
    //    seam, so the seam is satisfiable by a backend with no OS/GPU/devices ──

    #[test]
    fn null_backend_satisfies_every_required_seam() {
        let (hal, clock) = null_hal();

        // Clock: fake monotonic, advances only when told, never a wall clock.
        assert_eq!(hal.clock.now_monotonic(), Duration::ZERO);
        clock.advance(Duration::from_millis(16));
        assert_eq!(hal.clock.now_monotonic(), Duration::from_millis(16));
        assert!(hal.clock.now_unix_nanos() > 0);

        // FileSystem: in-memory round trip + listing.
        hal.fs.write("levels/intro.inf_lvl", b"payload").unwrap();
        assert!(hal.fs.exists("levels/intro.inf_lvl"));
        assert_eq!(hal.fs.read("levels/intro.inf_lvl").unwrap(), b"payload");
        assert_eq!(hal.fs.list("levels").unwrap(), vec!["intro.inf_lvl"]);
        hal.fs.remove("levels/intro.inf_lvl").unwrap();
        assert!(!hal.fs.exists("levels/intro.inf_lvl"));

        // SaveDataStore: slot round trip.
        hal.save.save("slot0", b"progress").unwrap();
        assert_eq!(hal.save.load("slot0").unwrap(), Some(b"progress".to_vec()));
        assert_eq!(hal.save.list_slots().unwrap(), vec!["slot0"]);

        // ThreadSpawner: runs the closure (inline, deterministic).
        let flag = Arc::new(AtomicU64::new(0));
        let f = flag.clone();
        hal.threads
            .spawn("worker", Box::new(move || f.store(1, Ordering::SeqCst)))
            .join();
        assert_eq!(flag.load(Ordering::SeqCst), 1);

        // DisplayTarget: headless, reports geometry.
        assert!(hal.display.is_headless());
        assert_eq!(hal.display.surface_size().width, 1280);
        assert_eq!(hal.display.present_mode(), PresentMode::Headless);

        // InputBackend: no devices, no events.
        assert!(hal.input.poll().is_empty());
        assert_eq!(hal.input.connected_gamepads(), 0);

        // Every named seam is exercised above.
        assert_eq!(REQUIRED_SEAMS.len(), 6);
    }

    #[test]
    fn lifecycle_sink_shape_compiles() {
        // The null backend also stands in for the TRC lifecycle sink; a real
        // console routes dashboard suspend/resume here.
        struct Sink;
        impl inf_platform::LifecycleSink for Sink {
            fn on_lifecycle(&self, _event: Lifecycle) {}
        }
        let s: Box<dyn inf_platform::LifecycleSink> = Box::new(Sink);
        s.on_lifecycle(Lifecycle::Suspending);
        s.on_lifecycle(Lifecycle::Resumed);
    }

    #[test]
    fn mem_fs_nested_listing() {
        let fs = MemFileSystem::default();
        fs.write("a/b/c.txt", b"1").unwrap();
        fs.write("a/d.txt", b"2").unwrap();
        assert_eq!(fs.list("a").unwrap(), vec!["b", "d.txt"]);
        assert_eq!(fs.list("a/b").unwrap(), vec!["c.txt"]);
    }
}
