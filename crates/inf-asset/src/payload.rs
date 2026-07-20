//! The dual-format asset rule, generically (ROADMAP §3).
//!
//! Every asset is two files on disk:
//!   * `<name>.<ext>`       — a **bincode payload** (fast, compact runtime load);
//!   * `<name>.<ext>.toml`  — a **deterministic TOML sidecar** (git-diffable
//!     metadata: GUID, schema version, content hash, dependencies, tags,
//!     import settings).
//!
//! A payload type opts in by implementing [`AssetPayload`]: it declares its
//! kind, its current `SCHEMA_VERSION`, and a `migrate` step. [`encode`] /
//! [`decode`] then give byte-deterministic bincode with the newer-than-current
//! guard applied on load.

use serde::{de::DeserializeOwned, Serialize};

use crate::error::{AssetError, Result};
use crate::kind::AssetKind;

/// The shared bincode configuration. `standard()` is fixed-endian and
/// deterministic, so re-encoding an unchanged payload is byte-identical.
pub fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

/// A versioned, serializable asset body.
///
/// Implementors are the concrete schema structs living in the domain crates
/// (`MeshAsset` in inf-mesh, `TextureAsset`/`MaterialAsset` in inf-material,
/// data-asset structs here). The database itself treats payloads as opaque
/// bytes — this trait is used by the *producers/consumers* of a given kind.
pub trait AssetPayload: Serialize + DeserializeOwned {
    /// The kind this payload represents.
    const KIND: AssetKind;
    /// The current on-disk schema version. Bump on any breaking layout change.
    const SCHEMA_VERSION: u32;

    /// The version stored in `self`. Schemas keep an explicit `schema_version`
    /// field so [`decode`] can detect and migrate old files.
    fn schema_version(&self) -> u32;

    /// Upgrade an older decoded value to [`Self::SCHEMA_VERSION`]. The default
    /// accepts equal-or-older and rejects newer; override to chain real
    /// migrations (v1→v2→…).
    fn migrate(self) -> Result<Self>
    where
        Self: Sized,
    {
        let found = self.schema_version();
        if found > Self::SCHEMA_VERSION {
            return Err(AssetError::SchemaTooNew {
                kind: Self::KIND.slug(),
                found,
                current: Self::SCHEMA_VERSION,
            });
        }
        Ok(self)
    }
}

/// Encode a payload to deterministic bincode bytes.
pub fn encode<T: AssetPayload>(value: &T) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(value, bincode_config())
        .map_err(|e| AssetError::Encode(e.to_string()))
}

/// Decode a payload, running its migration to the current schema.
pub fn decode<T: AssetPayload>(bytes: &[u8]) -> Result<T> {
    let (value, _): (T, usize) = bincode::serde::decode_from_slice(bytes, bincode_config())
        .map_err(|e| AssetError::Decode(e.to_string()))?;
    value.migrate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
    struct Foo {
        schema_version: u32,
        n: u32,
    }
    impl AssetPayload for Foo {
        const KIND: AssetKind = AssetKind::Table;
        const SCHEMA_VERSION: u32 = 2;
        fn schema_version(&self) -> u32 {
            self.schema_version
        }
    }

    #[test]
    fn encode_is_deterministic_and_round_trips() {
        let foo = Foo {
            schema_version: 2,
            n: 7,
        };
        let a = encode(&foo).unwrap();
        let b = encode(&foo).unwrap();
        assert_eq!(a, b, "re-encoding is byte-identical");
        assert_eq!(decode::<Foo>(&a).unwrap(), foo);
    }

    #[test]
    fn decode_rejects_newer_schema() {
        let future = Foo {
            schema_version: 99,
            n: 1,
        };
        let bytes = encode(&future).unwrap();
        assert!(matches!(
            decode::<Foo>(&bytes),
            Err(AssetError::SchemaTooNew { .. })
        ));
    }
}
