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

    /// What a user must do to bring a payload an **older** build wrote up to the
    /// current schema, phrased as an instruction.
    ///
    /// Read only by [`AssetError::SchemaTooOld`], which is the one message a user
    /// ever sees about a stale file — so the default is the remedy that is true
    /// for every *imported* kind, and a kind with a better door (a generator, a
    /// wizard) overrides it. A refusal that does not say what to do is a refusal
    /// nobody can act on, which is the same as no message at all.
    const UPGRADE_REMEDY: &'static str = "re-import it from its source file";

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

/// The leading `schema_version` of an asset payload, read **without decoding the
/// rest of it**.
///
/// Every [`AssetPayload`] keeps `schema_version` as its **first** field precisely
/// so this is possible; under `bincode::config::standard()` that is a leading
/// varint, and this decodes exactly that one integer.
///
/// `None` for bytes that are not an asset payload at all (empty, or a malformed
/// varint) — the caller then has nothing better to say than the decoder's own
/// error, which is correct.
pub fn peek_schema_version(bytes: &[u8]) -> Option<u32> {
    let (v, _): (u32, usize) = bincode::decode_from_slice(bytes, bincode_config()).ok()?;
    Some(v)
}

/// Decode a payload, running its migration to the current schema.
///
/// # Both directions of the version ladder are **named** errors
///
/// A payload from a **newer** build decodes structurally and is rejected by
/// [`AssetPayload::migrate`] ([`AssetError::SchemaTooNew`]). A payload from an
/// **older** one usually does not get that far: bincode is positional, so a field
/// appended at the tail since the file was written is a short read and the
/// decoder fails before `migrate` runs. Reporting that as
/// `Decode("UnexpectedEnd")` tells a user neither what happened nor what to do —
/// so the head's `schema_version` is peeked and the failure becomes
/// [`AssetError::SchemaTooOld`], carrying the type's own
/// [`UPGRADE_REMEDY`](AssetPayload::UPGRADE_REMEDY).
///
/// Generic on purpose: this covers every bincode asset kind at once, because the
/// hazard is the format's, not any one schema's.
pub fn decode<T: AssetPayload>(bytes: &[u8]) -> Result<T> {
    match bincode::serde::decode_from_slice::<T, _>(bytes, bincode_config()) {
        Ok((value, _)) => value.migrate(),
        Err(e) => {
            if let Some(found) = peek_schema_version(bytes) {
                if found < T::SCHEMA_VERSION {
                    return Err(AssetError::SchemaTooOld {
                        kind: T::KIND.slug(),
                        found,
                        current: T::SCHEMA_VERSION,
                        remedy: T::UPGRADE_REMEDY,
                    });
                }
            }
            Err(AssetError::Decode(e.to_string()))
        }
    }
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

    /// **The older direction is a NAMED error, and it names the remedy.**
    ///
    /// A payload written before a tail field was appended is a short read, so it
    /// never reaches `migrate`. Modelled the way it really happens: a shadow
    /// struct with the older shape, encoded, then decoded through the current one.
    #[test]
    fn decode_names_an_older_schema_and_says_what_to_do() {
        #[derive(Serialize)]
        struct FooV1 {
            schema_version: u32,
        }
        let bytes =
            bincode::serde::encode_to_vec(&FooV1 { schema_version: 1 }, bincode_config()).unwrap();
        match decode::<Foo>(&bytes) {
            Err(AssetError::SchemaTooOld {
                kind,
                found,
                current,
                remedy,
            }) => {
                assert_eq!((kind, found, current), (AssetKind::Table.slug(), 1, 2));
                assert!(
                    !remedy.is_empty(),
                    "a refusal with no remedy is unactionable"
                );
            }
            other => panic!("expected SchemaTooOld, got {other:?}"),
        }
        // …and the message a user reads carries the remedy verbatim.
        let msg = decode::<Foo>(&bytes).unwrap_err().to_string();
        assert!(msg.contains(Foo::UPGRADE_REMEDY), "{msg}");
    }

    /// Bytes that are not a payload at all still report the decoder's own error —
    /// the peek must not invent a version story for garbage.
    #[test]
    fn garbage_is_still_a_decode_error() {
        assert!(peek_schema_version(&[]).is_none());
        assert!(matches!(decode::<Foo>(&[]), Err(AssetError::Decode(_))));
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
