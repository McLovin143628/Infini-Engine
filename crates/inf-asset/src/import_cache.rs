//! The import cache.
//!
//! Importing (glTF → mesh, PNG → mip'd/compressed texture) is expensive, so its
//! output is cached by the **content hash of the source bytes combined with the
//! import settings**. Reimporting an unchanged file with unchanged settings is a
//! hash lookup, not a decode — the property that keeps a large project's rescan
//! fast (ROADMAP §3: "imports are cached by content hash and processed in
//! parallel").
//!
//! The cache is a directory of hash-named entries plus a small manifest mapping
//! `import key → produced asset id`. It is pure content-addressing: a stale
//! entry is simply never looked up again.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::hash::ContentHash;
use crate::id::AssetId;

/// The key an import is cached under: source content + a hash of the settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImportKey(pub ContentHash);

impl ImportKey {
    /// Build a key from raw source bytes and already-serialized settings bytes.
    pub fn new(source_bytes: &[u8], settings_bytes: &[u8]) -> Self {
        Self(ContentHash::of_parts(&[source_bytes, settings_bytes]))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Manifest {
    /// import-key hex → produced asset id.
    entries: HashMap<String, AssetId>,
}

/// A content-addressed import cache rooted at a directory (usually
/// `<project>/.inf/import-cache`).
#[derive(Debug)]
pub struct ImportCache {
    dir: PathBuf,
    manifest: Manifest,
}

impl ImportCache {
    /// Open (or create) a cache at `dir`, loading any existing manifest.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let manifest = match std::fs::read_to_string(dir.join("manifest.json")) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => Manifest::default(),
        };
        Ok(Self { dir, manifest })
    }

    fn manifest_path(&self) -> PathBuf {
        self.dir.join("manifest.json")
    }

    /// The produced asset id for a key, if this import is already cached.
    pub fn get(&self, key: ImportKey) -> Option<AssetId> {
        self.manifest.entries.get(&key.0.to_hex()).copied()
    }

    /// True if a byte artifact is stored for `key`.
    pub fn has_artifact(&self, key: ImportKey) -> bool {
        self.artifact_path(key).exists()
    }

    /// Where the cached payload bytes for `key` live.
    pub fn artifact_path(&self, key: ImportKey) -> PathBuf {
        self.dir.join(format!("{}.bin", key.0.to_hex()))
    }

    /// Record that `key` produced `id`, and stash its payload bytes.
    pub fn put(&mut self, key: ImportKey, id: AssetId, artifact: &[u8]) -> Result<()> {
        std::fs::write(self.artifact_path(key), artifact)?;
        self.manifest.entries.insert(key.0.to_hex(), id);
        self.flush()
    }

    /// Read cached payload bytes for `key`.
    pub fn read_artifact(&self, key: ImportKey) -> Result<Vec<u8>> {
        Ok(std::fs::read(self.artifact_path(key))?)
    }

    fn flush(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| crate::error::AssetError::Import(format!("cache manifest: {e}")))?;
        std::fs::write(self.manifest_path(), json)?;
        Ok(())
    }

    /// Number of cached imports.
    pub fn len(&self) -> usize {
        self.manifest.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.manifest.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_depends_on_source_and_settings() {
        let a = ImportKey::new(b"src", b"settingsA");
        let b = ImportKey::new(b"src", b"settingsB");
        let c = ImportKey::new(b"src", b"settingsA");
        assert_ne!(a, b, "different settings → different key");
        assert_eq!(a, c, "same inputs → same key");
    }

    #[test]
    fn put_then_get_hits_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let key = ImportKey::new(b"gltf-bytes", b"{}");
        let id = AssetId::new();
        {
            let mut cache = ImportCache::open(dir.path()).unwrap();
            assert!(cache.get(key).is_none());
            cache.put(key, id, b"imported-payload").unwrap();
            assert_eq!(cache.get(key), Some(id));
        }
        // Reopen: manifest + artifact persist.
        let cache = ImportCache::open(dir.path()).unwrap();
        assert_eq!(cache.get(key), Some(id));
        assert!(cache.has_artifact(key));
        assert_eq!(cache.read_artifact(key).unwrap(), b"imported-payload");
    }
}
