//! Cook + bundle pipeline: asset packs, blueprint compilation, per-platform
//! bundling (ROADMAP P9.2).
//!
//! This crate is the **cook**: it resolves a project's asset dependency closure,
//! validates its blueprints, rewrites its levels to the runtime schema, and
//! writes a single content-addressed [`inf_asset::PackWriter`]-built
//! `content.inf_pack` plus a deterministic [`CookManifest`]. The standalone
//! player ([`inf-player`](../inf_player/index.html)) is the pack consumer.
//!
//! Per-platform *bundling* (exe/installer/.app/AppImage) is P9.5 and layers on
//! top of this cook output later.

pub mod blueprint;
pub mod cook;
pub mod error;
pub mod manifest;

pub use cook::{cook, CookOptions, CookReport, DEFAULT_PACK_NAME};
pub use error::{CookError, Result};
pub use manifest::{CookManifest, MANIFEST_FILE, MANIFEST_SCHEMA_VERSION};
