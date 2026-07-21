//! The `.inf_pcg` asset payload — the on-disk envelope for a [`PcgDocument`].
//!
//! This mirrors the shape of the engine's asset payloads (see
//! `inf_material::MaterialInstance` and `inf_asset::AssetPayload`): a
//! `schema_version` field, deterministic **bincode** encoding, and a `migrate`
//! step that rejects newer-than-current files.
//!
//! ## inf-asset handoff (deliberately NOT wired this batch)
//!
//! The engine's `AssetPayload` trait requires `const KIND: AssetKind`, and
//! `AssetKind` (in `inf-asset`) has **no `Pcg` variant** yet. Adding it means
//! editing `inf-asset`, which is outside this batch's file boundary. So this
//! payload is self-contained (its own `encode`/`decode`) and carries the slug +
//! extension it *will* register as. The orchestrator's follow-up:
//!
//! 1. add `AssetKind::Pcg` → `"inf_pcg"` / slug `"pcg"` in `inf-asset`;
//! 2. add `inf-asset` as a dep and `impl inf_asset::AssetPayload for
//!    PcgAssetPayload` (KIND = `AssetKind::Pcg`, SCHEMA_VERSION =
//!    `CURRENT_VERSION`), then this module's `encode`/`decode` can defer to
//!    `inf_asset::{encode, decode}` for one code path across the engine.

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

    /// The shared, deterministic bincode configuration (fixed-endian `standard`,
    /// matching `inf_asset::bincode_config` so re-encoding is byte-identical).
    fn config() -> impl bincode::config::Config {
        bincode::config::standard()
    }

    /// Encode to deterministic bincode bytes.
    pub fn encode(&self) -> Result<Vec<u8>, PcgError> {
        bincode::serde::encode_to_vec(self, Self::config())
            .map_err(|e| PcgError::Encode(e.to_string()))
    }

    /// Decode from bincode bytes, running the schema migration.
    pub fn decode(bytes: &[u8]) -> Result<Self, PcgError> {
        let (value, _): (Self, usize) = bincode::serde::decode_from_slice(bytes, Self::config())
            .map_err(|e| PcgError::Decode(e.to_string()))?;
        value.migrate()
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
}
