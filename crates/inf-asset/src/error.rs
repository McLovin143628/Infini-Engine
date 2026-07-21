//! The crate-wide error type.

use crate::id::AssetId;

/// Errors from asset I/O, decoding, and database queries.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("bincode encode: {0}")]
    Encode(String),

    #[error("bincode decode: {0}")]
    Decode(String),

    #[error("toml serialize: {0}")]
    TomlSer(#[from] toml::ser::Error),

    #[error("toml parse: {0}")]
    TomlDe(#[from] toml::de::Error),

    /// A payload's `schema_version` is newer than this build understands.
    #[error("{kind} schema v{found} is newer than this build (v{current})")]
    SchemaTooNew {
        kind: &'static str,
        found: u32,
        current: u32,
    },

    /// A lookup or operation referenced an id not in the database.
    #[error("no asset with id {0}")]
    UnknownAsset(AssetId),

    #[error("import: {0}")]
    Import(String),

    /// A pack could not be written, parsed, or verified (`.inf_pack`).
    #[error("pack: {0}")]
    Pack(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, AssetError>;
