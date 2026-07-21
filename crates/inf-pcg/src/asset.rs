//! The `.inf_pcg` asset payload — the on-disk envelope for a [`PcgDocument`].
//!
//! This mirrors the shape of the engine's asset payloads (see
//! `inf_material::MaterialInstance` and `inf_asset::AssetPayload`): a
//! `schema_version` field, deterministic **bincode** encoding, and a `migrate`
//! step that rejects newer-than-current files.
//!
//! ## inf-asset integration (wired — P10 glue batch)
//!
//! [`PcgAssetPayload`] implements [`inf_asset::AssetPayload`]
//! (`KIND = AssetKind::Pcg`, `SCHEMA_VERSION = CURRENT_VERSION`), so it rides the
//! engine's single dual-format code path. The inherent [`PcgAssetPayload::encode`]
//! / [`PcgAssetPayload::decode`] now **defer to** [`inf_asset::encode`] /
//! [`inf_asset::decode`] — byte-identical to the old self-contained path (both use
//! `bincode::config::standard()`), just unified — while keeping the crate-local
//! [`PcgError`] surface its callers already expect.

use serde::{Deserialize, Serialize};

use crate::rules::PcgDocument;

/// Errors from encoding/decoding a `.inf_pcg` payload.
#[derive(Debug, thiserror::Error)]
pub enum PcgError {
    /// bincode serialization failed.
    #[error("pcg encode failed: {0}")]
    Encode(String),
    /// bincode deserialization failed.
    #[error("pcg decode failed: {0}")]
    Decode(String),
    /// The stored schema version is newer than this build understands.
    #[error("pcg schema version {found} is newer than supported {current}")]
    SchemaTooNew { found: u32, current: u32 },
}

/// The versioned on-disk body of a `.inf_pcg` asset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcgAssetPayload {
    pub schema_version: u32,
    #[serde(default)]
    pub document: PcgDocument,
}

impl PcgAssetPayload {
    /// The current on-disk schema version.
    pub const CURRENT_VERSION: u32 = 1;
    /// The stable UI/filter slug this kind registers as in `inf-asset`.
    pub const KIND_SLUG: &'static str = "pcg";
    /// The canonical file extension (without the dot).
    pub const EXTENSION: &'static str = "inf_pcg";

    /// Wrap `document` at the current schema version.
    pub fn new(document: PcgDocument) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            document,
        }
    }

    /// Encode to deterministic bincode bytes (defers to [`inf_asset::encode`], the
    /// engine-wide codec — byte-identical to the former self-contained path).
    pub fn encode(&self) -> Result<Vec<u8>, PcgError> {
        inf_asset::encode(self).map_err(|e| PcgError::Encode(e.to_string()))
    }

    /// Decode from bincode bytes, running the schema migration (defers to
    /// [`inf_asset::decode`]). A newer-than-current schema surfaces as
    /// [`PcgError::SchemaTooNew`], preserving this crate's error contract.
    pub fn decode(bytes: &[u8]) -> Result<Self, PcgError> {
        inf_asset::decode::<Self>(bytes).map_err(|e| match e {
            inf_asset::AssetError::SchemaTooNew { found, current, .. } => {
                PcgError::SchemaTooNew { found, current }
            }
            other => PcgError::Decode(other.to_string()),
        })
    }

    /// Upgrade an older decoded payload to [`Self::CURRENT_VERSION`]. Rejects
    /// newer-than-current. (v1 is the first version — this is the migration stub
    /// future `vN → vN+1` chains hang off, per ROADMAP §3.)
    pub fn migrate(self) -> Result<Self, PcgError> {
        if self.schema_version > Self::CURRENT_VERSION {
            return Err(PcgError::SchemaTooNew {
                found: self.schema_version,
                current: Self::CURRENT_VERSION,
            });
        }
        // No breaking layout changes yet; equal-or-older is accepted as-is.
        Ok(self)
    }
}

/// The dual-format asset-rule integration: a `.inf_pcg` document is a first-class
/// engine payload. `decode` (via [`inf_asset::decode`]) applies the trait's
/// newer-than-current guard using this `SCHEMA_VERSION`.
impl inf_asset::AssetPayload for PcgAssetPayload {
    const KIND: inf_asset::AssetKind = inf_asset::AssetKind::Pcg;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::ValueNoise;
    use crate::rules::{PcgKind, PcgRule, SamplerDef};
    use crate::scatter::ScatterParams;
    use uuid::Uuid;

    fn demo() -> PcgAssetPayload {
        let rule = PcgRule {
            name: "grass".into(),
            sampler: SamplerDef::Noise(ValueNoise::default()),
            scatter: ScatterParams::default(),
            kinds: vec![PcgKind::mesh(Uuid::from_u128(42))],
        };
        PcgAssetPayload::new(PcgDocument::single_layer("ground", vec![rule]))
    }

    #[test]
    fn encode_is_deterministic_and_round_trips() {
        let p = demo();
        let a = p.encode().unwrap();
        let b = p.encode().unwrap();
        assert_eq!(a, b, "re-encoding is byte-identical");
        assert_eq!(PcgAssetPayload::decode(&a).unwrap(), p);
    }

    #[test]
    fn decode_rejects_newer_schema() {
        let mut p = demo();
        p.schema_version = 99;
        let bytes = p.encode().unwrap();
        assert!(matches!(
            PcgAssetPayload::decode(&bytes),
            Err(PcgError::SchemaTooNew {
                found: 99,
                current: 1
            })
        ));
    }

    #[test]
    fn kind_constants() {
        assert_eq!(PcgAssetPayload::EXTENSION, "inf_pcg");
        assert_eq!(PcgAssetPayload::KIND_SLUG, "pcg");
    }

    #[test]
    fn kind_agrees_with_inf_asset_registry() {
        use inf_asset::AssetKind;
        // The payload's slug/extension consts match the registered `AssetKind::Pcg`,
        // and the extension classifies back to that kind (the round trip).
        assert_eq!(AssetKind::Pcg.slug(), PcgAssetPayload::KIND_SLUG);
        assert_eq!(AssetKind::Pcg.extension(), Some(PcgAssetPayload::EXTENSION));
        assert_eq!(AssetKind::Pcg.label(), "PCG Graph");
        assert_eq!(
            AssetKind::from_extension(PcgAssetPayload::EXTENSION),
            AssetKind::Pcg
        );
    }

    #[test]
    fn payload_round_trips_via_inf_asset_codec() {
        use inf_asset::AssetPayload;

        let p = demo();
        // The engine-wide codec round-trips the payload …
        let bytes = inf_asset::encode(&p).unwrap();
        assert_eq!(inf_asset::decode::<PcgAssetPayload>(&bytes).unwrap(), p);
        // … and is byte-identical to the inherent (deferring) encode.
        assert_eq!(bytes, p.encode().unwrap());

        // The trait's newer-than-current guard fires through the shared codec.
        let mut future = p;
        future.schema_version = 99;
        let bytes = inf_asset::encode(&future).unwrap();
        assert!(matches!(
            inf_asset::decode::<PcgAssetPayload>(&bytes),
            Err(inf_asset::AssetError::SchemaTooNew { .. })
        ));
        // The KIND const is wired.
        assert_eq!(PcgAssetPayload::KIND, inf_asset::AssetKind::Pcg);
    }
}
