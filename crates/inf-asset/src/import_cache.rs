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
    ///
    /// **Absent is not the same as unreadable** (C4-38). An absent manifest is
    /// an empty cache and always was. An unreadable one — truncated by a crash
    /// mid-write, or a merge conflict — used to become an empty cache too, and
    /// the next `put` wrote that emptiness back over the real file: the key→GUID
    /// map is gone, so the next import **mints new GUIDs for everything** and
    /// every reference into the previously-imported assets dangles. It is now an
    /// error, and the file is left alone until a human acts on it. The idiom is
    /// `inf_audio::MixerConfig::load_or_default`'s.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("manifest.json");
        let manifest = match std::fs::read_to_string(&path) {
            Ok(s) => serde_json::from_str(&s).map_err(|e| {
                crate::error::AssetError::Import(format!(
                    "import cache manifest {} is unreadable ({e}); it is left untouched so the \
                     key→GUID map is not overwritten — repair or delete it to re-import",
                    path.display()
                ))
            })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Manifest::default(),
            Err(e) => return Err(e.into()),
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
        crate::atomic::write_atomically(&self.artifact_path(key), artifact)?;
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
        crate::atomic::write_atomically(&self.manifest_path(), json)?;
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

    /// **A corrupt manifest is refused; an absent one is an empty cache** — the
    /// two arms side by side, because collapsing them is the defect (C4-38).
    ///
    /// The corrupt half also proves the file is *left alone*: the whole harm was
    /// `open` reporting an empty map that the next `put` then wrote back over
    /// the real one.
    #[test]
    fn a_corrupt_manifest_is_refused_while_an_absent_one_is_empty() {
        // Absent → empty cache, no error.
        let absent = tempfile::tempdir().unwrap();
        let cache = ImportCache::open(absent.path()).unwrap();
        assert!(cache.is_empty(), "an absent manifest is an empty cache");

        // Present but unreadable → an error, and the bytes stay put.
        let corrupt = tempfile::tempdir().unwrap();
        let key = ImportKey::new(b"src", b"{}");
        let id = AssetId::new();
        {
            let mut cache = ImportCache::open(corrupt.path()).unwrap();
            cache.put(key, id, b"payload").unwrap();
        }
        let manifest = corrupt.path().join("manifest.json");
        let damaged = b"{\"entries\": {\"abc\": ".to_vec(); // truncated mid-write
        std::fs::write(&manifest, &damaged).unwrap();

        let err = ImportCache::open(corrupt.path())
            .expect_err("an unreadable manifest must not read as an empty cache");
        assert!(
            err.to_string().contains("unreadable"),
            "the error must name the condition: {err}"
        );
        assert_eq!(
            std::fs::read(&manifest).unwrap(),
            damaged,
            "the damaged manifest must be left exactly as it was found"
        );
    }
}
