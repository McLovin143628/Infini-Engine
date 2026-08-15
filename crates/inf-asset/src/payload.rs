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
    ///
    /// # This is also the structural door (round-2 finding B2)
    ///
    /// A payload on disk is bytes somebody else wrote — the Content Drawer
    /// scans every loose file under the project root — so `migrate` is the one
    /// place a decoded body is inspected before a dozen readers trust it. Wave
    /// B's unit U6 made that argument for `TextureAsset` and `AnimClipAsset`;
    /// round 2 swept the remaining implementors and dispositioned each by
    /// **whether any production consumer uses a field as an index, an
    /// allocation size, a divisor or a raw FFI argument with no bound of its
    /// own**:
    ///
    /// | payload | disposition |
    /// |---|---|
    /// | `MeshAsset` | **structural migrate added** — its index buffer reaches `meshopt`'s raw FFI from two consumers |
    /// | `SkeletonAsset` | **structural migrate added** — `Skeleton`'s field is private and serde bypasses `Skeleton::new`, so a parent index past the joint list panicked `pose::global_transforms` on every posed character |
    /// | `VgeomMesh` | **structural migrate added** — safe only by the convention that everything goes through `VgeomSource`; `triangle()` documents its own panic |
    /// | `StateMachineAsset` | **structural migrate added at P29.1** — v2 gave `.inf_sm` a recursive condition tree and nested sub-machines, which the fixed step walks; the row used to read "safe — every `entry`/`from`/`to` read is `.min()`-clamped", which was true of v1 and is not the question v2 asks |
    /// | `ClothAsset`, `HairAsset` | safe — `validate()` runs unconditionally in `seed()`, and a garment that fails is skipped rather than simulated |
    /// | `StructAsset`, `EnumAsset` | safe — no field of theirs is an index, a size or a divisor |
    /// | `TableAsset` | safe — rows are walked with `.iter()`; a row/column disagreement is C4-42's import advisory, never an unchecked `row[col]` |
    /// | `MaterialAsset`, `DerivedMaterial` | safe — GUIDs and PBR floats; the one divided factor is clamped in `mesh.wgsl` |
    /// | `AudioAsset` | safe — `sample_rate`/`duration_secs` have no production reader at all; playback re-parses the original bytes |
    /// | `MaterialInstance` | safe — `parent` is a GUID and both resolvers carry the `depth > 16` guard |
    ///
    /// A new implementor answers the same question before it takes the default.
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

/// How far above a reader's own `SCHEMA_VERSION` a peeked version may sit and
/// still be *plausibly* a version rather than a coincidence.
///
/// A leading varint that decodes is not evidence of much — any bytes have one.
/// The peek exists to turn a real version skew into an actionable message, and a
/// schema does not jump by more than a handful of steps between two builds a user
/// has both of. Beyond this the honest answer is "these are not asset bytes",
/// which the decoder's own error already says.
const PLAUSIBLE_VERSION_LEAD: u32 = 8;

/// The leading `schema_version` of an asset payload, read **without decoding the
/// rest of it** — `None` when the head is not plausibly one.
///
/// Every [`AssetPayload`] keeps `schema_version` as its **first** field precisely
/// so this is possible; under `bincode::config::standard()` that is a leading
/// varint, and this decodes exactly that one integer.
///
/// # It refuses to invent a version (P24.1 re-audit F3)
///
/// The first cut returned whatever varint it found, and *every* byte sequence has
/// a leading varint. `[0x00]` came back as version **0** — a version no schema has
/// ever had, since they all start at 1 — and `[0x01, 0xAB, 0xCD, 0xEF]` as version
/// **1**, which then produced a confident `SchemaTooOld` naming a remedy for a file
/// whose problem is not its age. A wrong diagnosis is worse than none: it sends a
/// user to re-import an asset that is simply corrupt.
///
/// So: `version >= 1`, and no further above `current` than
/// [`PLAUSIBLE_VERSION_LEAD`]. `current` is the reader's own
/// [`AssetPayload::SCHEMA_VERSION`]; pass `None` when there is no type in hand and
/// only the floor applies.
///
/// **A bound, not a proof.** A zero-filled buffer is still a *structurally valid*
/// encoding of most schemas — it only fails here because its version is 0. See the
/// ROADMAP's P24 block for the "v>=1 floor at decode time" ledger entry.
pub fn peek_schema_version(bytes: &[u8], current: Option<u32>) -> Option<u32> {
    let (v, _): (u32, usize) = bincode::decode_from_slice(bytes, bincode_config()).ok()?;
    if v == 0 {
        return None; // no schema has ever been v0
    }
    if let Some(current) = current {
        if v > current.saturating_add(PLAUSIBLE_VERSION_LEAD) {
            return None; // a version that far ahead is not a version
        }
    }
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
            if let Some(found) = peek_schema_version(bytes, Some(T::SCHEMA_VERSION)) {
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

    /// The pre-tail shape of `Foo` — a real shadow struct, so "what v1 was" is
    /// written down rather than derived from the current encoder.
    #[derive(Serialize)]
    struct FooV1 {
        schema_version: u32,
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

    /// **Bytes that are not a payload get the decoder's own error, never a
    /// version story** (P24.1 re-audit F3).
    ///
    /// Every case below returned a confident `SchemaTooOld` before the floor and
    /// the plausibility bound existed — a wrong diagnosis, which sends a user to
    /// re-import an asset whose problem is not its age.
    ///
    /// **Renamed at P24.2**, because the old name (`garbage_is_still_a_decode
    /// _error`) claimed more than the body asserts and more than is true: the
    /// zero-filled case *decodes*, as `Foo { schema_version: 0, n: 0 }`, and
    /// `bincode_cannot_tell_a_zero_filled_file_from_an_empty_asset` below is
    /// where that bound is written down. What every case here really shares is
    /// the property this test was built to hold — the decoder never tells a
    /// **version story** about bytes that carry none, whatever else it does with
    /// them. The name now says that and nothing more.
    #[test]
    fn garbage_never_gets_a_version_story() {
        let cases: [(&str, &[u8]); 4] = [
            ("empty", &[]),
            // A lone zero byte: varint 0, a version no schema has ever had.
            ("a lone zero", &[0x00]),
            // A zero-filled buffer, which is what a truncated write leaves behind.
            ("zero-filled", &[0u8; 32]),
            // A version implausibly far ahead: not a skew, just not asset bytes.
            ("an implausible version", &[250, 0xFF, 0xFF]),
        ];
        // The **trailing-garbage** case (`[0x01, 0xAB, 0xCD, 0xEF]`) is deliberately
        // NOT here, and measuring it is why: it does not reach the peek at all —
        // it *decodes*, as `Foo { schema_version: 1, n: 171 }`. Its head is a
        // v1-shaped head, indistinguishable from a genuinely truncated v1 file, so
        // refusing it would switch off the diagnosis the peek exists to give. It
        // is pinned in `bincode_cannot_tell_a_zero_filled_file_from_an_empty_asset`
        // below, as the bound it actually is.
        for (label, bytes) in cases {
            assert!(
                peek_schema_version(bytes, Some(Foo::SCHEMA_VERSION)).is_none(),
                "{label}: the peek invented a version"
            );
            // Whatever `decode` does with these, it must never be `SchemaTooOld`:
            // that error names a remedy, and a remedy for the wrong problem is
            // worse than the decoder's own message.
            assert!(
                !matches!(decode::<Foo>(bytes), Err(AssetError::SchemaTooOld { .. })),
                "{label}: reported a version story for bytes that carry none"
            );
        }
        // …and a REAL older version is still recognised, or the bound above has
        // simply switched the feature off.
        let v1 =
            bincode::serde::encode_to_vec(&FooV1 { schema_version: 1 }, bincode_config()).unwrap();
        assert_eq!(peek_schema_version(&v1, Some(Foo::SCHEMA_VERSION)), Some(1));
    }

    /// **The bound of the "both directions are named errors" claim** — measured,
    /// so the ledger entry is a fact and not a worry (P24.1 re-audit).
    ///
    /// Two inputs that are not assets at all still *decode*, because bincode is a
    /// length-and-order format with no self-description: it has no way to say
    /// "these bytes were never an asset". So:
    ///
    ///  * a **zero-filled** buffer — what a truncated or pre-allocated write
    ///    leaves — decodes as a perfectly valid `schema_version = 0` asset with
    ///    every field at its zero;
    ///  * a valid leading varint over rubbish decodes as a valid *older* asset,
    ///    and is indistinguishable from a genuinely truncated one. That is why
    ///    `peek_schema_version` does **not** try to reject it: those bytes are a
    ///    v1-shaped head, and refusing them would switch off the very diagnosis
    ///    the peek exists to give.
    ///
    /// Closing the first one is a `schema_version >= 1` floor inside `migrate`,
    /// which is a behaviour change for every kind at once and belongs to a
    /// deliberate bump — ledgered in the ROADMAP's P24 block, not smuggled in
    /// here. This test is what will fail the day it lands, which is the point.
    #[test]
    fn bincode_cannot_tell_a_zero_filled_file_from_an_empty_asset() {
        let zeros = decode::<Foo>(&[0u8; 32]).expect("a zero-filled buffer decodes today");
        assert_eq!(
            zeros,
            Foo {
                schema_version: 0,
                n: 0
            },
            "the bound moved — if a v>=1 floor landed in `migrate`, retire the \
             ROADMAP ledger entry along with this assertion"
        );
        let over_garbage =
            decode::<Foo>(&[0x01, 0xAB, 0xCD, 0xEF]).expect("a v1-shaped head decodes today");
        assert_eq!(over_garbage.schema_version, 1);
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
