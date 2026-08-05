//! Cook + bundle pipeline: asset packs, blueprint compilation, per-platform
//! bundling (ROADMAP P9.2).
//!
//! This crate is the **cook**: it resolves a project's asset dependency closure,
//! validates its blueprints, rewrites its levels to the runtime schema, and
//! writes a single content-addressed [`inf_asset::PackWriter`]-built
//! `content.inf_pack` plus a deterministic [`CookManifest`]. The standalone
//! player ([`inf-player`](../inf_player/index.html)) is the pack consumer.
//!
//! Per-platform *bundling* — [`export`] assembling a runnable desktop folder
//! (renamed player exe + pack + manifest + launch config) — is P9.5 and layers on
//! top of this cook output ([`bundle`]).

pub mod blueprint;
pub mod bundle;
pub mod cook;
pub mod error;
pub mod manifest;
pub mod mods;
pub mod targets;

pub use bundle::{export, ExportOptions, ExportReport, ExportTarget, WindowConfig};
pub use cook::{
    cook, derived_fracture_id, derived_partition_id, derived_vmesh_id, CookOptions, CookReport,
    FractureCookOptions, VgeomCookOptions, DEFAULT_PACK_NAME,
};
pub use error::{CookError, Result};
pub use manifest::{CookManifest, MANIFEST_FILE, MANIFEST_SCHEMA_VERSION};
pub use mods::{
    build_mod_wasm, generate_mod_crate, wasm_target_installed, GeneratedModCrate, ModBuildOptions,
    ModBuildOutcome,
};
pub use targets::{
    export_android, export_web, AndroidExportOptions, AndroidExportReport, WebExportOptions,
    WebExportReport,
};
