//! **The licence row on an asset's sidecar** (wave VEH3f.2b, moved to Ring 0
//! from the Unreal bridge) -- three keys of the sidecar's `import` table, named
//! ONCE here so the importer that writes them and the cook that refuses on them
//! read one spelling.
//!
//! # Why it lives in `inf-asset`
//!
//! The bridge (`inf_editor_core::assets::ue_import`) has written the row since
//! carried item 96, and until this wave nothing READ it but a gate grepping the
//! text: the cook had no reader of `licence_may_ship`, so a pack marked "local
//! reference only" would have been packed into a shipped `.inf_pack` exactly
//! like one that may ship. The cook depends on this crate and not on the
//! editor's, so the key names, and the one question the cook asks of them, are
//! here.

use crate::AssetSidecar;

/// The licence text of the pack an asset came from.
pub const LICENCE_KEY: &str = "licence";
/// Whether that licence permits SHIPPING the asset in a cooked build.
pub const LICENCE_SHIP_KEY: &str = "licence_may_ship";
/// Which pack the asset came from.
pub const LICENCE_PACK_KEY: &str = "licence_pack";

/// The three keys, together -- a reader that has to CARRY the row (a
/// derivation, a rebind) copies all three rather than the one it remembered.
pub const LICENCE_KEYS: [&str; 3] = [LICENCE_KEY, LICENCE_SHIP_KEY, LICENCE_PACK_KEY];

/// **May this asset ship?** `Some(false)` only for a sidecar whose licence row
/// says so in as many words (`licence_may_ship = false`); `Some(true)` for one
/// that says it may; `None` for every asset with no row -- this repository's
/// own content, which is licence-free and is never refused for want of a row.
pub fn may_ship(side: &AssetSidecar) -> Option<bool> {
    side.import
        .as_ref()?
        .get(LICENCE_SHIP_KEY)
        .and_then(|v| v.as_bool())
}

/// The pack a sidecar's licence row names, if any.
pub fn pack_of(side: &AssetSidecar) -> Option<String> {
    side.import
        .as_ref()?
        .get(LICENCE_PACK_KEY)
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetId, AssetKind, ContentHash};

    /// The three answers: no row, a row that may ship, a row that may not.
    #[test]
    fn a_sidecar_answers_whether_it_may_ship() {
        let mut s = AssetSidecar::new(AssetId::new(), AssetKind::Mesh, ContentHash::of(b"x"));
        assert_eq!(may_ship(&s), None);
        let mut t = toml::Table::new();
        t.insert(LICENCE_SHIP_KEY.into(), true.into());
        t.insert(LICENCE_PACK_KEY.into(), "P".into());
        s.import = Some(t.clone());
        assert_eq!(may_ship(&s), Some(true));
        assert_eq!(pack_of(&s).as_deref(), Some("P"));
        t.insert(LICENCE_SHIP_KEY.into(), false.into());
        s.import = Some(t);
        assert_eq!(may_ship(&s), Some(false));
    }
}
