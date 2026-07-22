//! Entity references for reflected components (E-P1).
//!
//! A component field that points at *another entity* (e.g. a physics joint's
//! `other` body) stores that link as a stable [`Uuid`] — the same GUID the
//! Outliner and `.inf_lvl` use. Bare `Option<Uuid>` cannot participate in
//! reflection (`Uuid` is not `Reflect`), so joint `other` fields were
//! historically `#[reflect(ignore)]` and invisible to the Details panel.
//!
//! [`EntityRef`] wraps that `Option<Uuid>` as an **opaque** reflected value:
//! `bevy_reflect` treats it as a leaf (never introspecting the `Uuid` inside),
//! while the Details walker recognises it and surfaces an entity-picker widget.
//! It is `#[serde(transparent)]` so the on-disk byte stream is *identical* to
//! the old `Option<Uuid>` — existing `.inf_lvl` fixtures keep round-tripping.
//!
//! Only `EntityRef` is needed this wave: asset fields (mesh / material GUIDs)
//! stay bare `Uuid` + `#[reflect(ignore)]`; a future `AssetRef` would follow the
//! same opaque pattern when an asset-picker widget lands.

use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A reflected, opaque reference to another entity by its stable [`Guid`].
///
/// `None` → unbound. Serializes transparently as `Option<Uuid>` (byte-identical
/// to the pre-`EntityRef` representation).
///
/// [`Guid`]: crate::components::Guid
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(opaque)]
#[reflect(Default, PartialEq)]
#[serde(transparent)]
pub struct EntityRef(pub Option<Uuid>);

impl EntityRef {
    /// An unbound reference.
    pub const NONE: Self = EntityRef(None);

    /// Wrap a concrete target GUID.
    pub fn new(guid: Uuid) -> Self {
        EntityRef(Some(guid))
    }

    /// The target GUID, if bound.
    pub fn get(self) -> Option<Uuid> {
        self.0
    }
}

impl From<Option<Uuid>> for EntityRef {
    fn from(v: Option<Uuid>) -> Self {
        EntityRef(v)
    }
}

impl From<EntityRef> for Option<Uuid> {
    fn from(v: EntityRef) -> Self {
        v.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_reflect::{GetPath, PartialReflect, Reflect};

    // ── SPIKE 1: opaque EntityRef participates in reflection ────────────────
    #[test]
    fn entity_ref_is_reflectable_opaque() {
        let r = EntityRef::new(Uuid::from_u128(7));
        let dynamic: &dyn Reflect = &r;
        // Opaque: downcast round-trips through reflection.
        let back = dynamic.downcast_ref::<EntityRef>().unwrap();
        assert_eq!(back.get(), Some(Uuid::from_u128(7)));

        // try_apply from a like value updates in place.
        let mut dst = EntityRef::NONE;
        let src = EntityRef::new(Uuid::from_u128(99));
        dst.apply(&src);
        assert_eq!(dst.get(), Some(Uuid::from_u128(99)));
    }

    // ── SPIKE 2: byte-identity with Option<Uuid> under bincode ──────────────
    #[test]
    fn bincode_transparent_matches_option_uuid() {
        let cfg = bincode::config::standard();
        for opt in [None, Some(Uuid::from_u128(0xdead_beef))] {
            let raw = bincode::serde::encode_to_vec(opt, cfg).unwrap();
            let wrapped = bincode::serde::encode_to_vec(EntityRef(opt), cfg).unwrap();
            assert_eq!(
                raw, wrapped,
                "EntityRef must be byte-identical to Option<Uuid>"
            );
            // And decode back through the wrapper.
            let (dec, _): (EntityRef, _) = bincode::serde::decode_from_slice(&raw, cfg).unwrap();
            assert_eq!(dec.0, opt);
        }
    }

    // ── SPIKE 3: GetPath / reflect_path_mut exists and walks nested paths ───
    #[test]
    fn reflect_path_mut_walks_nested() {
        use crate::components::Spline;
        use crate::math::Vec3d;
        let mut s = Spline {
            points: vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(1.0, 2.0, 3.0)],
            closed: false,
            interp: crate::components::SplineInterp::Linear,
        };
        // Path into a Vec<Vec3d> element field.
        let field = s.reflect_path_mut("points[1].y").unwrap();
        let y = field.try_downcast_mut::<f64>().unwrap();
        *y = 42.0;
        assert_eq!(s.points[1].y, 42.0);

        // Read path too.
        let read = s.reflect_path("points[0].x").unwrap();
        assert_eq!(*read.try_downcast_ref::<f64>().unwrap(), 0.0);
    }
}
