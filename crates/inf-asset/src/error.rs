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

    /// A payload was written by an **older** build and this format cannot read
    /// it back.
    ///
    /// Not the mirror of [`SchemaTooNew`](Self::SchemaTooNew): that one is caught
    /// by `migrate()`, *after* a successful decode. This one is caught **instead
    /// of** a decode, because bincode is positional — a field appended at the
    /// tail since the file was written is a short read, and the decoder gives up
    /// long before anything looks at `schema_version`. Without this variant the
    /// caller sees `Decode("UnexpectedEnd")`, which names neither the cause nor
    /// the cure.
    ///
    /// `remedy` is the payload type's own [`AssetPayload::UPGRADE_REMEDY`], so the
    /// message a user reads tells them which door to walk through.
    #[error(
        "{kind} schema v{found} was written by an older build and this one speaks \
         v{current}; bincode is positional, so it cannot be read back — {remedy}"
    )]
    SchemaTooOld {
        kind: &'static str,
        found: u32,
        current: u32,
        remedy: &'static str,
    },

    /// A lookup or operation referenced an id not in the database.
    #[error("no asset with id {0}")]
    UnknownAsset(AssetId),

    #[error("import: {0}")]
    Import(String),

    /// A pack could not be written, parsed, or verified (`.inf_pack`).
    #[error("pack: {0}")]
    Pack(String),

    /// A sidecar is on disk beside its payload but could not be parsed, so the
    /// database is holding a synthesized stand-in for it — and refuses to write
    /// that stand-in over the real file (C4-39).
    #[error(
        "the sidecar {path} exists but cannot be read, so this asset's real guid, source, \
         import settings, tags and dependencies are unknown; the file is left untouched — \
         repair or delete it, then rescan"
    )]
    SidecarUnreadable { path: String },
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, AssetError>;
