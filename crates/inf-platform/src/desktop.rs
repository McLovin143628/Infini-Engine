//! The `std`-backed desktop backend for the HAL seams.
//!
//! This is the real backend Windows/macOS/Linux shipped games use for the
//! non-GPU seams (the GPU/window seam is still owned by `inf-render`/`winit`
//! today — see the audit). It exists in-repo both to serve desktop and to prove
//! the seam is implementable by *two* backends (desktop + null), which is the
//! console-portability claim in miniature.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::{
    Clock, DisplayTarget, FileSystem, Hal, InputBackend, InputEvent, JoinToken, PresentMode,
    SaveDataStore, SurfaceSize, ThreadSpawner,
};

/// `std::time`-backed clock.
pub struct StdClock {
    start: Instant,
}

impl Default for StdClock {
    fn default() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Clock for StdClock {
    fn now_monotonic(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    fn now_unix_nanos(&self) -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }
}

/// `std::fs`-backed filesystem rooted at a directory. Virtual paths are joined
/// onto `root`; `..` escapes are rejected.
pub struct StdFileSystem {
    root: PathBuf,
}

impl StdFileSystem {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn resolve(&self, path: &str) -> io::Result<PathBuf> {
        let mut out = self.root.clone();
        for seg in path.split('/') {
            if seg.is_empty() || seg == "." {
                continue;
            }
            if seg == ".." {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "path escapes vfs root",
                ));
            }
            out.push(seg);
        }
        Ok(out)
    }
}

impl FileSystem for StdFileSystem {
    fn read(&self, path: &str) -> io::Result<Vec<u8>> {
        std::fs::read(self.resolve(path)?)
    }

    fn write(&self, path: &str, data: &[u8]) -> io::Result<()> {
        let p = self.resolve(path)?;
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(p, data)
    }

    fn exists(&self, path: &str) -> bool {
        self.resolve(path).map(|p| p.exists()).unwrap_or(false)
    }

    fn remove(&self, path: &str) -> io::Result<()> {
        std::fs::remove_file(self.resolve(path)?)
    }

    fn list(&self, path: &str) -> io::Result<Vec<String>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(self.resolve(path)?)? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                out.push(name.to_string());
            }
        }
        out.sort();
        Ok(out)
    }
}

/// Save-data backed by a directory of `<slot>.save` files (desktop convention;
/// console backends replace this with the platform save API).
pub struct DirSaveStore {
    dir: PathBuf,
}

impl DirSaveStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn slot_path(&self, slot: &str) -> PathBuf {
        self.dir.join(format!("{slot}.save"))
    }
}

impl SaveDataStore for DirSaveStore {
    fn save(&self, slot: &str, data: &[u8]) -> io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        std::fs::write(self.slot_path(slot), data)
    }

    fn load(&self, slot: &str) -> io::Result<Option<Vec<u8>>> {
        let p = self.slot_path(slot);
        if p.exists() {
            Ok(Some(std::fs::read(p)?))
        } else {
            Ok(None)
        }
    }

    fn list_slots(&self) -> io::Result<Vec<String>> {
        let mut out = Vec::new();
        if !self.dir.exists() {
            return Ok(out);
        }
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(slot) = name.strip_suffix(".save") {
                out.push(slot.to_string());
            }
        }
        out.sort();
        Ok(out)
    }

    fn remove(&self, slot: &str) -> io::Result<()> {
        let p = self.slot_path(slot);
        if p.exists() {
            std::fs::remove_file(p)
        } else {
            Ok(())
        }
    }
}

/// A `std::thread::JoinHandle` wrapper.
pub struct StdJoin(std::thread::JoinHandle<()>);

impl JoinToken for StdJoin {
    fn join(self: Box<Self>) {
        let _ = self.0.join();
    }
}

/// `std::thread`-backed spawner (named threads).
#[derive(Default)]
pub struct StdThreadSpawner;

impl ThreadSpawner for StdThreadSpawner {
    fn spawn(&self, name: &str, f: Box<dyn FnOnce() + Send + 'static>) -> Box<dyn JoinToken> {
        let handle = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(f)
            .expect("failed to spawn HAL thread");
        Box::new(StdJoin(handle))
    }
}

/// A fixed-geometry desktop display target (the renderer still owns the real
/// swapchain today; this reports the geometry a console backend would supply).
pub struct DesktopDisplay {
    pub size: SurfaceSize,
    pub present_mode: PresentMode,
}

impl DisplayTarget for DesktopDisplay {
    fn surface_size(&self) -> SurfaceSize {
        self.size
    }

    fn present_mode(&self) -> PresentMode {
        self.present_mode
    }

    fn is_headless(&self) -> bool {
        matches!(self.present_mode, PresentMode::Headless)
    }
}

/// An input backend that yields no events (the real desktop input comes through
/// `inf-input`/`winit`/`gilrs`; this satisfies the seam for headless desktop).
#[derive(Default)]
pub struct NullInput;

impl InputBackend for NullInput {
    fn poll(&self) -> Vec<InputEvent> {
        Vec::new()
    }

    fn connected_gamepads(&self) -> usize {
        0
    }
}

/// Assemble a desktop HAL rooted at `content_root` (game data + save dir), at a
/// given window size.
pub fn host_hal(content_root: impl AsRef<Path>, size: SurfaceSize) -> Hal {
    let root = content_root.as_ref().to_path_buf();
    Hal::new(
        Arc::new(StdClock::default()),
        Arc::new(StdFileSystem::new(root.clone())),
        Arc::new(DirSaveStore::new(root.join("Saves"))),
        Arc::new(StdThreadSpawner),
        Arc::new(DesktopDisplay {
            size,
            present_mode: PresentMode::Fifo,
        }),
        Arc::new(NullInput),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn std_fs_round_trips_and_rejects_escapes() {
        let dir = std::env::temp_dir().join(format!("inf-plat-{}", std::process::id()));
        let fs = StdFileSystem::new(&dir);
        fs.write("a/b.txt", b"hi").unwrap();
        assert!(fs.exists("a/b.txt"));
        assert_eq!(fs.read("a/b.txt").unwrap(), b"hi");
        assert_eq!(fs.list("a").unwrap(), vec!["b.txt".to_string()]);
        assert!(fs.read("../escape").is_err());
        fs.remove("a/b.txt").unwrap();
        assert!(!fs.exists("a/b.txt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clock_is_monotonic() {
        let c = StdClock::default();
        let a = c.now_monotonic();
        let b = c.now_monotonic();
        assert!(b >= a);
        assert!(c.now_unix_nanos() > 0);
    }

    #[test]
    fn thread_spawner_runs_the_closure() {
        let s = StdThreadSpawner;
        let (tx, rx) = std::sync::mpsc::channel();
        let t = s.spawn("t", Box::new(move || tx.send(7u32).unwrap()));
        t.join();
        assert_eq!(rx.recv().unwrap(), 7);
    }
}
