//! The `mod.toml` manifest beside each `.wasm` (P14.5, deliverable 3).
//!
//! A mod's capabilities are declared explicitly and out-of-band in a TOML file
//! next to its `.wasm`, so granting is a data decision an auditor can read, not
//! something the mod binary asks for. **A missing manifest means no capabilities
//! at all** (deny-by-default all the way down): an unlabelled mod loads but can
//! reach nothing.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::caps::ModCaps;

/// A parsed `mod.toml`. Only the capability grant is load-bearing today; `name`
/// is cosmetic (for logs/UI).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModManifest {
    /// Human-readable mod name (defaults to the wasm file stem when absent).
    #[serde(default)]
    pub name: Option<String>,
    /// The explicit capability grant. Absent table → [`ModCaps::NONE`].
    #[serde(default)]
    pub caps: ModCaps,
}

impl ModManifest {
    /// Parse a `mod.toml` from text.
    pub fn parse(text: &str) -> Result<Self, ManifestError> {
        toml::from_str(text).map_err(|e| ManifestError::Parse(e.to_string()))
    }

    /// Load the `mod.toml` sitting beside `wasm_path` (same directory, named
    /// `mod.toml`). A missing file is **not** an error: it yields a manifest
    /// with [`ModCaps::NONE`] and the wasm file stem as the name.
    pub fn load_beside(wasm_path: &Path) -> Result<Self, ManifestError> {
        let manifest_path = wasm_path
            .parent()
            .map(|d| d.join("mod.toml"))
            .unwrap_or_else(|| Path::new("mod.toml").to_path_buf());
        match std::fs::read_to_string(&manifest_path) {
            Ok(text) => Self::parse(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ModManifest {
                name: wasm_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned()),
                caps: ModCaps::NONE,
            }),
            Err(e) => Err(ManifestError::Io(e.to_string())),
        }
    }
}

/// A failure reading or parsing a `mod.toml`.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("reading mod.toml: {0}")]
    Io(String),
    #[error("parsing mod.toml: {0}")]
    Parse(String),
}
