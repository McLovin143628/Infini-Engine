//! Stable asset identity.
//!
//! Every asset carries a GUID minted once at import/create time and never
//! reused. GUIDs — not paths — are what references store, so moving or renaming
//! an asset on disk never breaks a reference (the [`AssetDb`](crate::AssetDb)
//! re-indexes the path, the GUID is untouched).

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A globally-unique, stable asset identifier.
///
/// Wraps a v4 [`Uuid`]. Serializes transparently (as the UUID) so sidecars stay
/// readable and bincode payloads stay compact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AssetId(pub Uuid);

impl AssetId {
    /// Mint a fresh random id.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// The nil id (`0000…`), used as a sentinel for "no asset".
    pub const NIL: Self = Self(Uuid::nil());

    /// True for the nil sentinel.
    pub fn is_nil(&self) -> bool {
        self.0.is_nil()
    }

    /// The underlying UUID.
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

/// **`AssetId::default()` is [`NIL`](AssetId::NIL), not a fresh id** — the
/// sentinel for "no asset", so `#[derive(Default)]` on a struct holding an
/// optional reference means *unset* rather than *unnamed*.
///
/// It is therefore never the right way to mint one. `AssetProject::
/// register_written_asset` wrote `reuse.unwrap_or_default()` for two phases and
/// gave every asset it created the same NIL guid, so the second one replaced the
/// first in the database and orphaned its payload (P26.1 audit). Use
/// [`AssetId::new`], and note that `clippy::unwrap_or_default` will offer to
/// undo that — the lint assumes `new()` and `default()` agree, and here they
/// deliberately do not.
impl Default for AssetId {
    fn default() -> Self {
        Self::NIL
    }
}

impl From<Uuid> for AssetId {
    fn from(u: Uuid) -> Self {
        Self(u)
    }
}

impl std::fmt::Display for AssetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for AssetId {
    type Err = uuid::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nil_round_trips_through_string() {
        assert!(AssetId::NIL.is_nil());
        let parsed: AssetId = AssetId::NIL.to_string().parse().unwrap();
        assert_eq!(parsed, AssetId::NIL);
    }

    #[test]
    fn fresh_ids_are_unique_and_non_nil() {
        let a = AssetId::new();
        let b = AssetId::new();
        assert_ne!(a, b);
        assert!(!a.is_nil());
    }
}
