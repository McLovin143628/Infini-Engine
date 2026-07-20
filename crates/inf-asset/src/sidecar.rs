//! The TOML sidecar shared by every asset.
//!
//! One sidecar sits beside each payload (`foo.inf_mesh` → `foo.inf_mesh.toml`)
//! and holds the metadata the database is built from: the stable GUID, the kind,
//! the payload content hash, the import source + settings, and the explicit
//! dependency list (which other assets this one references — the edges of the
//! dependency graph). It is the human-readable, version-controlled face of an
//! asset; the payload is the machine face.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::hash::ContentHash;
use crate::id::AssetId;
use crate::kind::AssetKind;

/// Sidecar format version, independent of any payload's schema version.
pub const SIDECAR_SCHEMA_VERSION: u32 = 1;

/// Metadata beside every asset payload. Field order is fixed and `toml`
/// serializes map keys sorted, so a given sidecar re-emits byte-identically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssetSidecar {
    /// Sidecar-format version (see [`SIDECAR_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// The stable asset GUID.
    pub guid: AssetId,
    /// What kind of asset this is.
    pub kind: AssetKind,
    /// xxh3-128 of the payload bytes — the change + integrity signal, and the
    /// key thumbnails and the import cache are stored under.
    pub content_hash: ContentHash,
    /// The original import source, relative to the project root, if this asset
    /// was produced by importing an external file (glTF/PNG/…). `None` for
    /// authored-in-editor assets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// GUIDs this asset references (its outgoing dependency edges). Sorted for
    /// determinism.
    #[serde(default)]
    pub dependencies: Vec<AssetId>,
    /// Free-form user tags for search/filtering. Sorted for determinism.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Importer settings (opaque here; each importer owns its own schema and
    /// (de)serializes into this table). Reimport must honor these.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub import: Option<toml::Table>,
}

impl AssetSidecar {
    /// A fresh sidecar for a newly-produced asset.
    pub fn new(guid: AssetId, kind: AssetKind, content_hash: ContentHash) -> Self {
        Self {
            schema_version: SIDECAR_SCHEMA_VERSION,
            guid,
            kind,
            content_hash,
            source: None,
            dependencies: Vec::new(),
            tags: Vec::new(),
            import: None,
        }
    }

    /// Canonicalize collection fields so serialization is deterministic
    /// regardless of insertion order.
    pub fn normalize(&mut self) {
        self.dependencies.sort_unstable();
        self.dependencies.dedup();
        self.tags.sort_unstable();
        self.tags.dedup();
    }

    /// Serialize to deterministic pretty TOML.
    pub fn to_toml(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.normalize();
        Ok(toml::to_string_pretty(&copy)?)
    }

    /// Parse from TOML text.
    pub fn from_toml(text: &str) -> Result<Self> {
        Ok(toml::from_str(text)?)
    }

    /// Write the sidecar next to `payload_path`.
    pub fn save(&self, payload_path: &Path) -> Result<()> {
        if let Some(parent) = payload_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(sidecar_path(payload_path), self.to_toml()?)?;
        Ok(())
    }

    /// Read the sidecar beside `payload_path`.
    pub fn load(payload_path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(sidecar_path(payload_path))?;
        Self::from_toml(&text)
    }
}

/// `foo.inf_mesh` → `foo.inf_mesh.toml`.
pub fn sidecar_path(payload_path: &Path) -> PathBuf {
    let mut s = payload_path.as_os_str().to_os_string();
    s.push(".toml");
    PathBuf::from(s)
}

/// True if `path` is itself a sidecar (`*.toml` beside a payload). The watcher
/// uses this to avoid double-processing a payload and its sidecar.
pub fn is_sidecar(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("toml")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_path_appends_toml() {
        assert_eq!(
            sidecar_path(Path::new("a/Hero.inf_mesh")),
            PathBuf::from("a/Hero.inf_mesh.toml")
        );
    }

    #[test]
    fn toml_round_trips_and_is_deterministic() {
        let mut s = AssetSidecar::new(AssetId::new(), AssetKind::Mesh, ContentHash::of(b"x"));
        s.tags = vec!["b".into(), "a".into(), "a".into()];
        s.dependencies = vec![AssetId::new(), AssetId::new()];
        let t1 = s.to_toml().unwrap();
        let t2 = s.to_toml().unwrap();
        assert_eq!(t1, t2, "re-emit is byte-identical");
        // tags de-duped and sorted after normalize-on-emit.
        assert!(t1.contains("\"a\""));
        let back = AssetSidecar::from_toml(&t1).unwrap();
        assert_eq!(back.guid, s.guid);
        assert_eq!(back.kind, AssetKind::Mesh);
        assert_eq!(back.tags, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn save_load_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let payload = dir.path().join("Hero.inf_mesh");
        std::fs::write(&payload, b"payload").unwrap();
        let s = AssetSidecar::new(AssetId::new(), AssetKind::Mesh, ContentHash::of(b"payload"));
        s.save(&payload).unwrap();
        assert!(sidecar_path(&payload).exists());
        let back = AssetSidecar::load(&payload).unwrap();
        assert_eq!(back.guid, s.guid);
    }
}
