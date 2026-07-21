//! Scene model: the **runtime** `.inf_lvl` reader (P9.2 · deliverable 3).
//!
//! The editor authors levels through `inf_editor_core::scene::serialize`
//! (Ring 1). The shipped **player** (Ring 2, engine-side) must load the very
//! same `.inf_lvl` bytes without depending on any editor crate — so this Ring-0
//! crate owns a **decode-only** reader that produces the runtime level
//! representation ([`RuntimeLevel`]).
//!
//! # Why this is byte-compatible with the editor
//!
//! `.inf_lvl` is a deterministic `bincode` payload of concrete, `serde`-derived
//! [`inf_ecs`] component types (`Transform`, `MeshRef`, `Sprite`, `Tilemap`, …).
//! `bincode` is not self-describing, so byte-compatibility is purely a matter of
//! mirroring the field layout. This reader reuses the *same* `inf_ecs` component
//! types and reproduces the editor's record layout exactly (both schema
//! versions), so editor-written bytes decode field-for-field. The cross-tests
//! prove it against the committed sample/template levels and the frozen v1
//! fixture (the editor crate cannot be a dependency here — a ring inversion — so
//! we assert against committed bytes, which is the stronger guarantee anyway).
//!
//! The editor keeps its authoring codec (and its byte-determinism/undo tests)
//! untouched; this is the read side of the same wire format.
//!
//! # Schema versions (kept in lockstep with the editor)
//!
//! * **v1** — 3D only: transform + mesh/material/light/camera.
//! * **v2** — appends the five 2D component slots (sprite / tilemap / nine-slice
//!   / text / 2D light). A v1 payload is decoded through its frozen record and
//!   lifted with those slots defaulted — never by reinterpreting the shorter
//!   byte stream. [`encode`] always writes the current schema, so cooking a v1
//!   level **rewrites it to v2** (the P9.2 "rewrite the level payload for
//!   runtime" step).

use inf_ecs::components::{
    Camera, Light, Light2D, Material, MeshRef, NineSlice, Sprite, Text2D, Tilemap, Transform,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The current on-disk `.inf_lvl` schema (matches the editor's `SCHEMA_VERSION`).
pub const SCHEMA_VERSION: u32 = 2;

/// A failure decoding or encoding a `.inf_lvl`.
#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(String),
    #[error("encode: {0}")]
    Encode(String),
    /// The payload's leading schema version is newer than this build understands.
    #[error("scene schema v{found} is newer than this build (v{current})")]
    SchemaTooNew { found: u32, current: u32 },
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, SceneError>;

/// One entity's persisted state — the runtime record.
///
/// This is the **schema-v2** wire layout: field order and the `#[serde(default)]`
/// markers mirror the editor's `EntityRecord` byte-for-byte so the same bincode
/// payload decodes here. Component slots are `Option`s (a slot is `Some` when the
/// entity carries that component).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEntity {
    /// Stable entity GUID (what selection/hierarchy reference across a reload).
    pub guid: Uuid,
    /// Display / Outliner name.
    pub name: String,
    /// Hierarchy parent GUID, if any.
    pub parent: Option<Uuid>,
    /// Local transform (translation m, rotation euler-deg, scale).
    pub transform: Transform,
    /// Self-visibility toggle.
    pub visible: bool,
    /// Renderable mesh (primitive; asset-mesh variant is a later phase).
    pub mesh: Option<MeshRef>,
    /// PBR material parameter block.
    pub material: Option<Material>,
    /// Light source.
    pub light: Option<Light>,
    /// Scene camera.
    pub camera: Option<Camera>,
    // ── v2 (P8.2b) 2D component slots ─────────────────────────────────────
    #[serde(default)]
    pub sprite: Option<Sprite>,
    #[serde(default)]
    pub tilemap: Option<Tilemap>,
    #[serde(default)]
    pub nine_slice: Option<NineSlice>,
    #[serde(default)]
    pub text2d: Option<Text2D>,
    #[serde(default)]
    pub light_2d: Option<Light2D>,
}

/// A decoded level ready for the runtime to instantiate.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeLevel {
    /// The level title.
    pub title: String,
    /// Entities in creation order (parents precede children).
    pub entities: Vec<RuntimeEntity>,
}

impl RuntimeLevel {
    /// Decode a `.inf_lvl` payload (any supported schema version).
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        decode(bytes)
    }

    /// Encode to the **current** schema (v2) — a deterministic bincode payload.
    pub fn encode(&self) -> Result<Vec<u8>> {
        encode(self)
    }

    /// Load and decode a `.inf_lvl` file.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        Self::decode(&std::fs::read(path)?)
    }

    /// Number of entities.
    pub fn len(&self) -> usize {
        self.entities.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// The entity with the given GUID, if present.
    pub fn entity(&self, guid: Uuid) -> Option<&RuntimeEntity> {
        self.entities.iter().find(|e| e.guid == guid)
    }
}

// ── wire records (serde layouts mirroring the editor codec) ─────────────────

fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

/// Just the leading `schema_version` — decoded first (bincode reads fields in
/// order and stops) to select the right versioned record.
#[derive(Deserialize)]
struct Header {
    schema_version: u32,
}

/// The schema-v2 file layout (current). `entities` reuses [`RuntimeEntity`].
#[derive(Serialize, Deserialize)]
struct SceneFileV2 {
    schema_version: u32,
    title: String,
    entities: Vec<RuntimeEntity>,
}

/// A frozen schema-v1 entity record (pre-P8.2b, 3D only). Never changes.
#[derive(Deserialize)]
struct EntityRecordV1 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<Material>,
    light: Option<Light>,
    camera: Option<Camera>,
}

/// A frozen schema-v1 file layout.
#[derive(Deserialize)]
struct SceneFileV1 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV1>,
}

impl EntityRecordV1 {
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material,
            light: self.light,
            camera: self.camera,
            sprite: None,
            tilemap: None,
            nine_slice: None,
            text2d: None,
            light_2d: None,
        }
    }
}

/// Decode a `.inf_lvl` payload, lifting older schemas to [`RuntimeLevel`].
pub fn decode(bytes: &[u8]) -> Result<RuntimeLevel> {
    let (header, _): (Header, usize) = bincode::serde::decode_from_slice(bytes, bincode_config())
        .map_err(|e| SceneError::Decode(format!("header: {e}")))?;
    match header.schema_version {
        0 | 1 => {
            let (v1, _): (SceneFileV1, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v1: {e}")))?;
            Ok(RuntimeLevel {
                title: v1.title,
                entities: v1
                    .entities
                    .into_iter()
                    .map(EntityRecordV1::into_runtime)
                    .collect(),
            })
        }
        2 => {
            let (v2, _): (SceneFileV2, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v2: {e}")))?;
            Ok(RuntimeLevel {
                title: v2.title,
                entities: v2.entities,
            })
        }
        found => Err(SceneError::SchemaTooNew {
            found,
            current: SCHEMA_VERSION,
        }),
    }
}

/// Encode a level to the current schema (v2) as a deterministic bincode payload.
pub fn encode(level: &RuntimeLevel) -> Result<Vec<u8>> {
    let file = SceneFileV2 {
        schema_version: SCHEMA_VERSION,
        title: level.title.clone(),
        entities: level.entities.clone(),
    };
    bincode::serde::encode_to_vec(&file, bincode_config())
        .map_err(|e| SceneError::Encode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// The workspace root, reachable from this crate at `crates/inf-scene`.
    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
    }

    fn read_committed(rel: &str) -> Vec<u8> {
        let p = workspace_root().join(rel);
        std::fs::read(&p).unwrap_or_else(|e| panic!("read committed {}: {e}", p.display()))
    }

    #[test]
    fn decodes_the_committed_platformer_level() {
        let level =
            RuntimeLevel::decode(&read_committed("samples/platformer-2d/Platformer.inf_lvl"))
                .expect("platformer decodes");
        assert_eq!(level.title, "Platformer 2D");
        // 5 entities (see the committed sidecar's entity_count).
        assert_eq!(level.len(), 5);
        // The platformer is a 2D scene: at least one entity carries a tilemap.
        assert!(
            level.entities.iter().any(|e| e.tilemap.is_some()),
            "platformer has a tilemap ground"
        );
        // Every entity has a name + a finite transform we can read.
        for e in &level.entities {
            assert!(!e.name.is_empty());
            let _ = e.transform.translation;
        }
    }

    #[test]
    fn decodes_the_committed_hybrid_template() {
        let level = RuntimeLevel::decode(&read_committed("templates/hybrid-2.5d/Hybrid.inf_lvl"))
            .expect("hybrid decodes");
        assert_eq!(level.title, "Hybrid 2.5D");
        assert_eq!(level.len(), 5);
    }

    #[test]
    fn decodes_the_frozen_v1_fixture_and_defaults_2d_slots() {
        // The editor's forever-load v1 fixture, copied into this crate.
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v1.inf_lvl"),
        )
        .expect("committed v1 fixture present");
        assert_eq!(bytes[0], 1, "fixture is a genuine schema-v1 payload");

        let level = RuntimeLevel::decode(&bytes).expect("v1 fixture decodes");
        assert_eq!(level.title, "Fixture Level");
        assert_eq!(level.len(), 4);

        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();
        // Legacy 3D data preserved through the lift.
        assert!(by_name("Ground").mesh.is_some());
        assert!(by_name("Ground").material.is_some());
        assert!(by_name("Sun").light.is_some());
        assert!(by_name("Cam").camera.is_some());
        assert!(!by_name("Cam").visible);
        // Every 2D slot defaulted to None on the old payload.
        for e in &level.entities {
            assert!(e.sprite.is_none());
            assert!(e.tilemap.is_none());
            assert!(e.nine_slice.is_none());
            assert!(e.text2d.is_none());
            assert!(e.light_2d.is_none());
        }
    }

    #[test]
    fn v2_decode_encode_is_byte_identical_for_committed_bytes() {
        // A committed v2 level re-encodes to the exact same bytes — decode/encode
        // is a lossless identity on current-schema content (so the cook's runtime
        // rewrite of an already-v2 level is a no-op, and deterministic).
        let original = read_committed("samples/platformer-2d/Platformer.inf_lvl");
        let level = RuntimeLevel::decode(&original).unwrap();
        let reencoded = level.encode().unwrap();
        assert_eq!(original, reencoded, "v2 round trip must be byte-identical");
    }

    #[test]
    fn v1_reencodes_to_v2() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v1.inf_lvl"),
        )
        .unwrap();
        let level = RuntimeLevel::decode(&bytes).unwrap();
        let out = level.encode().unwrap();
        // The rewritten payload is genuine schema v2, and re-decodes equal.
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    #[test]
    fn rejects_a_future_schema() {
        // Hand-forge a bincode payload whose leading varint is a huge version.
        let level = RuntimeLevel {
            title: "x".into(),
            entities: vec![],
        };
        let mut bytes = level.encode().unwrap();
        // schema_version is the first byte for small values; bump it well past us.
        bytes[0] = 99;
        assert!(matches!(
            RuntimeLevel::decode(&bytes),
            Err(SceneError::SchemaTooNew { .. })
        ));
    }
}
