//! The C ABI shared between the host (editor / runtime) and game-logic
//! plugins (cdylibs).
//!
//! Doctrine (ROADMAP §5, Spike C): **no Rust types cross the boundary** —
//! only `#[repr(C)]` structs of fn pointers, raw byte buffers, and integer
//! status codes. Rust's ABI is unstable across compiler versions and
//! `panic = unwind` across `extern "C"` aborts the process, so every plugin
//! entry point is a `catch_unwind` wrapper (see [`crate::guest`]) that
//! converts panics into [`STATUS_PANICKED`].
//!
//! Buffers are moved with host-owned callbacks ([`WriteFn`], [`LogFn`]) so
//! **no allocation ever crosses the boundary** in either direction — the
//! plugin writes into host memory through a host fn pointer, and nothing
//! needs a cross-module `free`.

use std::os::raw::c_void;

/// Bumped whenever anything in this file changes shape. The host refuses to
/// load a plugin whose entry reports a different version.
pub const ABI_VERSION: u32 = 1;

/// The single symbol a plugin must export:
/// `extern "C" fn() -> *const PluginVTable`.
pub const ENTRY_SYMBOL: &[u8] = b"infinity_plugin_entry\0";

/// Call completed normally.
pub const STATUS_OK: u32 = 0;
/// The plugin code panicked; the panic was caught on the plugin side of the
/// boundary. The host must treat the component as broken (disable it) but
/// the process is fine.
pub const STATUS_PANICKED: u32 = 1;
/// The call failed without panicking (e.g. serialization error).
pub const STATUS_ERROR: u32 = 2;

pub const LOG_INFO: u32 = 0;
pub const LOG_WARN: u32 = 1;
pub const LOG_ERROR: u32 = 2;

/// Host-provided sink for serialized bytes. May be called multiple times per
/// serialize call; the host appends.
pub type WriteFn = unsafe extern "C" fn(ctx: *mut c_void, data: *const u8, len: usize);

/// Host-provided log sink (`level` is one of the `LOG_*` constants; `msg` is
/// UTF-8, not NUL-terminated).
pub type LogFn = unsafe extern "C" fn(ctx: *mut c_void, level: u32, msg: *const u8, len: usize);

/// One hot-reloadable component: `(name, schema_hash, create/destroy,
/// serialize/deserialize, tick)`. All state pointers are opaque to the host
/// and must only be passed back to fns from the *same* vtable.
#[repr(C)]
pub struct ComponentVTable {
    /// UTF-8, not NUL-terminated, `'static` inside the plugin image.
    pub name: *const u8,
    pub name_len: usize,
    /// Hash of the component's state schema descriptor. A mismatch across a
    /// reload means "migration happened" (the self-describing state format
    /// defaults new fields); it is reported, not fatal.
    pub schema_hash: u64,
    /// Allocate a default-initialized state. Null on failure.
    pub create: unsafe extern "C" fn() -> *mut c_void,
    /// Free a state previously produced by `create`/`deserialize` of this
    /// same vtable.
    pub destroy: unsafe extern "C" fn(state: *mut c_void),
    /// Stream the state through `write`. Returns a `STATUS_*` code.
    pub serialize:
        unsafe extern "C" fn(state: *mut c_void, write: WriteFn, write_ctx: *mut c_void) -> u32,
    /// Build a state from serialized bytes (possibly written by an *older*
    /// version of this component — unknown fields are defaulted). Null on
    /// failure.
    pub deserialize: unsafe extern "C" fn(data: *const u8, len: usize) -> *mut c_void,
    /// Advance the state by `dt` seconds. Returns a `STATUS_*` code; on
    /// `STATUS_PANICKED` the plugin has already logged the panic message via
    /// `log`.
    pub tick:
        unsafe extern "C" fn(state: *mut c_void, dt: f64, log: LogFn, log_ctx: *mut c_void) -> u32,
}

/// What [`ENTRY_SYMBOL`] returns. Lives for the lifetime of the plugin image
/// (which is forever — the host never unloads).
#[repr(C)]
pub struct PluginVTable {
    pub abi_version: u32,
    /// Plugin-defined version string (UTF-8, not NUL-terminated).
    pub version: *const u8,
    pub version_len: usize,
    pub component_count: usize,
    pub components: *const ComponentVTable,
}

pub type EntryFn = unsafe extern "C" fn() -> *const PluginVTable;
