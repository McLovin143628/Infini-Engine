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
///
/// ## What a `.inf_pcg` stores (P10.5b decision)
///
/// The editor's node graph is the authored source of truth — the volume
/// evaluates by [`lower_graph`](crate::graph::lower_graph)ing it into a
/// [`PcgDocument`] on demand, so the graph must survive a save (unlike materials,
/// whose `.inf_mat` graph-persistence is still a documented follow-up — PCG does
/// it here).
///
/// The graph is stored as a **JSON string** ([`graph_json`](Self::graph_json)),
/// not as a nested `inf_graph::Graph`: that model is wire-tuned for JSON
/// (`#[serde(skip_serializing_if)]` on `NodeUi::color` / `GraphDoc::viewport`),
/// which **desyncs a non-self-describing bincode stream**. Serializing the graph
/// to JSON first (then embedding the string in the bincode payload) sidesteps
/// that entirely and keeps the whole payload deterministic. The legacy
/// `document` field is **kept** (v1 payloads carried only it) so old files still
/// decode: when `graph_json` is `Some`, it is the source of truth; otherwise the
/// stored `document` is used directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcgAssetPayload {
    pub schema_version: u32,
    /// The authored editor graph as JSON (source of truth from v2 on). `None` on
    /// a v1 payload, which carried only the lowered `document`. JSON — not a
    /// nested `Graph` — because that model's `skip_serializing_if` fields are not
    /// bincode-safe (see the type docs).
    #[serde(default)]
    pub graph_json: Option<String>,
    /// The lowered runtime model. From v2 this is a convenience mirror of the
    /// graph; on a v1 payload it is the only content.
    #[serde(default)]
    pub document: PcgDocument,
}

impl PcgAssetPayload {
    /// The current on-disk schema version. v2 (P10.5b) appended the editor
    /// `graph`; v1 payloads (document-only) still decode via `#[serde(default)]`.
    pub const CURRENT_VERSION: u32 = 2;
    /// The stable UI/filter slug this kind registers as in `inf-asset`.
    pub const KIND_SLUG: &'static str = "pcg";
    /// The canonical file extension (without the dot).
    pub const EXTENSION: &'static str = "inf_pcg";

    /// Wrap a lowered `document` at the current schema version (no editor graph).
    pub fn new(document: PcgDocument) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            graph_json: None,
            document,
        }
    }

    /// Wrap an authored editor graph (serialized to JSON — the source of truth)
    /// plus its lowered `document` mirror, at the current schema version.
    ///
    /// **The write-side twin of the read-side hole** (C4-42): `to_string(..).ok()`
    /// dropped the authored graph on a serialization failure while `encode()`
    /// went on reporting success, so the asset saved *looked* fine and had lost
    /// the thing it exists to store. Now a failure is a refusal; see
    /// [`try_from_graph`](Self::try_from_graph).
    pub fn from_graph(graph: &inf_graph::Graph, document: PcgDocument) -> Self {
        Self::try_from_graph(graph, document).unwrap_or_else(|_| Self {
            schema_version: Self::CURRENT_VERSION,
            graph_json: None,
            document: PcgDocument::default(),
        })
    }

    /// Wrap an authored graph, refusing rather than silently dropping it.
    pub fn try_from_graph(
        graph: &inf_graph::Graph,
        document: PcgDocument,
    ) -> Result<Self, PcgError> {
        Ok(Self {
            schema_version: Self::CURRENT_VERSION,
            graph_json: Some(
                serde_json::to_string(graph)
                    .map_err(|e| PcgError::Encode(format!("graph will not serialize: {e}")))?,
            ),
            document,
        })
    }

    /// Parse the stored editor graph, if present.
    ///
    /// **`None` means "this payload carries no graph"** — see
    /// [`try_graph`](Self::try_graph) for why the distinction matters.
    pub fn graph(&self) -> Option<inf_graph::Graph> {
        self.try_graph().ok().flatten()
    }

    /// Parse the stored editor graph, distinguishing **absent** from **broken**
    /// (C4-42).
    ///
    /// `graph()` used to map a *parse failure* of `graph_json` onto the same
    /// `None` that legitimately means "a v1 document-only payload". Every
    /// consumer reads `None` as "use the lowered `document` mirror", and that
    /// mirror **structurally cannot carry grammar or building passes** — so a
    /// corrupt `graph_json` made every procedural building and grammar pass
    /// silently disappear from the generated world, and their meshes drop out of
    /// the pack's dependency closure with it.
    ///
    /// `Ok(None)` is the honest absence; `Err` is the corruption, and a caller
    /// that reads it can say so.
    pub fn try_graph(&self) -> Result<Option<inf_graph::Graph>, PcgError> {
        let Some(json) = self.graph_json.as_ref() else {
            return Ok(None);
        };
        serde_json::from_str(json)
            .map(Some)
            .map_err(|e| PcgError::Decode(format!("graph_json will not parse: {e}")))
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
    /// newer-than-current. v1→v2 is purely additive (the `graph` field defaults
    /// to `None`, so a v1 document-only payload lifts cleanly), so equal-or-older
    /// is accepted as-is — this stays the stub future breaking chains hang off.
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
                current: 2
            })
        ));
    }

    #[test]
    fn graph_round_trips_and_is_the_source_of_truth() {
        use crate::graph::{lower_graph, pcg_registry};
        use inf_graph::Graph;

        // An authored graph → its lowered document, both stored in the payload.
        let reg = pcg_registry();
        let mut g = Graph::empty();
        let scat = g.insert("scatter.scatter", inf_graph::NodeUi::default());
        let out = g.insert("output.pcg", inf_graph::NodeUi::default());
        g.links.push(inf_graph::Link {
            from: scat,
            from_port: "out".into(),
            to: out,
            to_port: "scatter".into(),
        });
        let doc = lower_graph(&g, &reg).document;
        let payload = PcgAssetPayload::from_graph(&g, doc);
        let bytes = payload.encode().unwrap();
        let back = PcgAssetPayload::decode(&bytes).unwrap();
        assert_eq!(back, payload);
        assert_eq!(back.graph(), Some(g));
        assert_eq!(back.schema_version, 2);
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
