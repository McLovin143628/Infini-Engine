//! Platform HAL traits: the **console seam** (ROADMAP §2.3.4, §9, P14.4).
//!
//! Every OS/GPU/file/input/time dependency a shipped game touches is expressed
//! here as a trait so a platform backend is an *additive* crate rather than a
//! refactor. Desktop uses [`desktop`] (a `std`-backed backend that lives in this
//! repo); consoles use private out-of-tree crates that implement the same traits
//! (see `platforms/`); tests and the P14.4 audit use the `inf-platform-null`
//! mock backend.
//!
//! This crate is deliberately **`std`-only** (no third-party deps): it is the
//! narrowest, most portable layer in Ring 0, and a console toolchain must be able
//! to build it before anything else.
//!
//! ## Honesty note (the audit's spine)
//!
//! Standing up this seam does **not** mean the engine already routes through it.
//! Today most Ring-0 crates reach for `std::fs`, `std::time::Instant`,
//! `std::thread`, and `wgpu`/`winit` directly. This crate defines the *target*
//! shape; `docs/memos/p14-console-hal-audit.md` is the gap table naming exactly
//! where the engine still bypasses it and what seam work each gap needs.

use std::io;
use std::sync::Arc;
use std::time::Duration;

pub mod desktop;

// ───────────────────────────────── time ────────────────────────────────────

/// The monotonic + wall clock seam. The fixed-step sim never reads a wall clock
/// (it is driven by caller-supplied deltas — see `inf-runtime`), but frame pacing,
/// timestamps, and profiling do; a console backend can drive these from its own
/// platform timer, and the null backend fakes a deterministic monotonic clock.
pub trait Clock: Send + Sync {
    /// Monotonic time since the backend started. Never decreases.
    fn now_monotonic(&self) -> Duration;

    /// Wall-clock time as Unix nanoseconds (may be faked / fixed on the null
    /// backend or a devkit with no RTC).
    fn now_unix_nanos(&self) -> u128;
}

// ─────────────────────────────── filesystem ────────────────────────────────

/// The virtual filesystem seam. Desktop maps this onto `std::fs`; consoles map it
/// onto their title-storage / content-mount APIs (read-only game data vs a
/// separate writable user area). Paths are backend-relative virtual paths, always
/// `/`-separated, so gameplay code never bakes in an OS path convention.
pub trait FileSystem: Send + Sync {
    fn read(&self, path: &str) -> io::Result<Vec<u8>>;
    fn write(&self, path: &str, data: &[u8]) -> io::Result<()>;
    fn exists(&self, path: &str) -> bool;
    fn remove(&self, path: &str) -> io::Result<()>;
    /// Immediate child names of a directory (not recursive), sorted.
    fn list(&self, path: &str) -> io::Result<Vec<String>>;
}

// ──────────────────────────── save-data (TRC) ──────────────────────────────

/// Console save-data is **not** a file path — it is a slot-oriented, often
/// asynchronous, quota-checked, corruption-guarded store (a TRC/TCR category).
/// Modelling it as its own seam keeps that requirement visible instead of hiding
/// it behind `FileSystem::write`.
pub trait SaveDataStore: Send + Sync {
    fn save(&self, slot: &str, data: &[u8]) -> io::Result<()>;
    fn load(&self, slot: &str) -> io::Result<Option<Vec<u8>>>;
    fn list_slots(&self) -> io::Result<Vec<String>>;
    fn remove(&self, slot: &str) -> io::Result<()>;
}

// ───────────────────────────────── threads ─────────────────────────────────

/// A join token for a backend-spawned worker. Object-safe (`self: Box<Self>`) so
/// backends can return their own handle type behind `dyn`.
pub trait JoinToken: Send {
    /// Block until the worker finishes.
    fn join(self: Box<Self>);
}

/// The OS-thread spawning seam. `inf-core` owns the *compute* job pool (rayon);
/// this seam is for the handful of long-lived background threads (asset watcher,
/// audio mixer, IO pump) whose creation a console constrains (named cores,
/// affinity, priority). The null backend runs the closure inline (deterministic).
pub trait ThreadSpawner: Send + Sync {
    fn spawn(&self, name: &str, f: Box<dyn FnOnce() + Send + 'static>) -> Box<dyn JoinToken>;
}

// ────────────────────────────── display / surface ──────────────────────────

/// Present mode hint for the swapchain (the GPU seam's one tunable that leaks to
/// gameplay/perf tiers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentMode {
    /// Wait for vblank (no tearing).
    Fifo,
    /// Present immediately (may tear).
    Immediate,
    /// Fixed cadence, headless render target (no display).
    Headless,
}

/// Physical surface geometry the renderer configures its swapchain against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSize {
    pub width: u32,
    pub height: u32,
    pub scale_factor: f64,
}

/// The window/GPU-surface seam. This is the **least-abstracted** seam today: the
/// renderer still creates its `wgpu::Surface` directly from a `winit`/HWND handle
/// rather than through this trait (see the audit gap table). What the trait pins
/// down now is the backend-neutral geometry + present policy a console backend
/// must supply; wiring `inf-render` to accept a `DisplayTarget` instead of a
/// concrete window is the named follow-up.
pub trait DisplayTarget: Send + Sync {
    fn surface_size(&self) -> SurfaceSize;
    fn present_mode(&self) -> PresentMode;
    /// `true` when no real display/swapchain backs this target (null backend,
    /// headless CI, dedicated server).
    fn is_headless(&self) -> bool;
}

// ───────────────────────────────── input ───────────────────────────────────

/// A backend-neutral input event. `inf-input` maps these onto action/axis bindings;
/// a console backend produces them from its native pad/HID stack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputEvent {
    /// Keyboard/scancode key (desktop). `code` is a backend scancode.
    Key { code: u32, pressed: bool },
    /// Gamepad face/shoulder/dpad button.
    Button {
        gamepad: u8,
        button: u32,
        pressed: bool,
    },
    /// Gamepad analog axis in `[-1, 1]`.
    Axis { gamepad: u8, axis: u32, value: f32 },
}

/// The input seam. The null backend reports no devices and an empty event stream.
pub trait InputBackend: Send + Sync {
    /// Drain pending input events (call once per frame).
    fn poll(&self) -> Vec<InputEvent>;
    /// Number of connected gamepads.
    fn connected_gamepads(&self) -> usize;
}

// ─────────────────────────────── lifecycle (TRC) ───────────────────────────

/// App lifecycle transitions a console mandates handling (suspend on dashboard /
/// resume / constrained / quit). Desktop rarely fires these; a console TRC does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifecycle {
    Suspending,
    Resumed,
    /// Reduced-resource "constrained" state (backgrounded overlay).
    Constrained,
    Quitting,
}

/// A sink the platform backend calls on lifecycle transitions so the engine can
/// checkpoint save-data, pause audio, and release the GPU appropriately.
pub trait LifecycleSink: Send + Sync {
    fn on_lifecycle(&self, event: Lifecycle);
}

// ───────────────────────────────── aggregate ───────────────────────────────

/// The assembled hardware-abstraction layer a shipped game runs against: one
/// handle bundling every platform seam. Constructed by a backend
/// ([`desktop::host_hal`] here, `inf_platform_null::null_hal` in tests, a private
/// crate on console) and threaded into the runtime.
#[derive(Clone)]
pub struct Hal {
    pub clock: Arc<dyn Clock>,
    pub fs: Arc<dyn FileSystem>,
    pub save: Arc<dyn SaveDataStore>,
    pub threads: Arc<dyn ThreadSpawner>,
    pub display: Arc<dyn DisplayTarget>,
    pub input: Arc<dyn InputBackend>,
}

impl Hal {
    /// Assemble a HAL from its seams.
    pub fn new(
        clock: Arc<dyn Clock>,
        fs: Arc<dyn FileSystem>,
        save: Arc<dyn SaveDataStore>,
        threads: Arc<dyn ThreadSpawner>,
        display: Arc<dyn DisplayTarget>,
        input: Arc<dyn InputBackend>,
    ) -> Self {
        Self {
            clock,
            fs,
            save,
            threads,
            display,
            input,
        }
    }
}

/// The set of seams a backend must implement — used by the audit as the checklist
/// of "does this platform supply every capability the engine needs".
pub const REQUIRED_SEAMS: &[&str] = &[
    "Clock",
    "FileSystem",
    "SaveDataStore",
    "ThreadSpawner",
    "DisplayTarget",
    "InputBackend",
];
