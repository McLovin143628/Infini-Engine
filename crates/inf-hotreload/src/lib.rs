//! Hot reload of game-logic dylibs (Spike C, docs/ROADMAP.md §5).
//!
//! Split by side of the C ABI boundary:
//!
//! - [`abi`] — the `#[repr(C)]` contract both sides share. No Rust types
//!   ever cross it.
//! - [`guest`] — what plugin crates use: the [`guest::HotComponent`] trait,
//!   panic-catching `extern "C"` shims, and [`export_plugin!`].
//! - [`host`] *(feature `host`, default on; plugins build with
//!   `default-features = false`)* — dylib loading with the content-addressed
//!   shadow-copy dance, and [`host::HotWorld`] which owns instances, ticks
//!   them with panic containment, and carries state across reloads.
//!
//! Scoping note: the Blueprint interpreter is the primary iteration loop;
//! hot reload is the compiled-preview tier (P6.6) and PIE runs scripts in a
//! subprocess (Spike D) — this crate is the middle tier, not a safety-
//! critical single point of failure.

pub mod abi;
pub mod guest;
#[cfg(feature = "host")]
pub mod host;

#[cfg(feature = "host")]
pub use host::{
    HostError, HotWorld, InstanceId, InstanceStatus, LogRecord, Plugin, PluginHost, ReloadReport,
};
