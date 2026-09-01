//! Asset system: `.inf_*` schemas, bincode payload + TOML sidecars, GUIDs, the
//! asset database, and importers.
//!
//! This Ring-0 crate is the backbone of Phase 4. It owns:
//!   * **identity + hashing** — stable [`AssetId`] GUIDs and content-addressed
//!     [`ContentHash`]es (xxh3-128);
//!   * **the dual-format rule** — [`payload`] (deterministic bincode) beside
//!     [`AssetSidecar`] (git-diffable TOML);
//!   * **the database** — [`AssetDb`], a GUID-keyed registry with a live
//!     dependency graph (forward *and* reverse edges), a content-hash index,
//!     directory scanning, and the "who references this?" query behind
//!     delete-with-references warnings;
//!   * **change detection** — [`AssetWatcher`], a debounced recursive watcher;
//!   * **the import cache** — [`ImportCache`], content-addressed so unchanged
//!     reimports are a hash lookup.
//!
//! Concrete payload schemas + importers live in the domain crates (`inf-mesh`,
//! `inf-material`) and the editor's import orchestrator, which depend on this
//! crate for identity, the sidecar, and [`AssetPayload`].

pub mod atomic;
pub mod block;
pub mod data;
pub mod db;
pub mod derived_material;
pub mod error;
pub mod hash;
pub mod id;
pub mod import_cache;
pub mod kind;
pub mod pack;
pub mod payload;
pub mod sidecar;
// The `notify`-backed file watcher is host-only — there is no browser file
// watcher, and the wasm player streams its pack over HTTP instead (P14.2).
#[cfg(not(target_arch = "wasm32"))]
pub mod watch;

pub use atomic::{temp_sibling, write_atomically};
pub use block::{decode_block, encode_block, BlockCodec};
pub use data::{CellValue, EnumAsset, FieldDef, FieldType, StructAsset, TableAsset};
pub use db::{AssetDb, AssetEntry, IdCollision};
pub use derived_material::{
    derived_material_id, DerivedBlend, DerivedMaterial, DERIVED_MATERIAL_ID_SALT,
};
pub use error::{AssetError, Result};
pub use hash::ContentHash;
pub use id::AssetId;
pub use import_cache::{ImportCache, ImportKey};
pub use kind::{importable_source_kind, AssetKind};
pub use pack::{
    AlignedBytes, EntryPolicy, PackEntry, PackReader, PackWriter, BLOB_ALIGN, PACK_FORMAT_VERSION,
    PACK_MAGIC, PACK_MIN_READ_VERSION,
};
// `bincode_config` is exported so a container format that carries a bincode
// *section* beside its raw pages (`.inf_terrain`'s tile blobs, `.inf_vmesh`'s DAG
// groups) encodes it with exactly the codec the framed path uses — one config,
// one set of bytes, no second place to drift.
pub use payload::{
    bincode_config, decode, decode_shape, encode, encode_shape, peek_schema_version, AssetPayload,
};
pub use sidecar::{is_sidecar, sidecar_path, AssetSidecar, SIDECAR_SCHEMA_VERSION};
#[cfg(not(target_arch = "wasm32"))]
pub use watch::{AssetChange, AssetWatcher};
