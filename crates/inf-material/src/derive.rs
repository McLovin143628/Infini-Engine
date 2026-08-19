//! **The one door for material resolution** (P26.3b): a `.inf_mat` → the
//! [`DerivedMaterial`] record every runtime path reads.
//!
//! # Why there is exactly one of these
//!
//! The P22.2 ruling, verbatim in shape: *"the cook, the PIE payload builder and
//! the editor's Simulate seeder all derive fractures; two of them were separate
//! code carrying comments claiming they agreed by construction. They did not."*
//! One walked ECS archetype order while the others walked document order, and
//! one skipped a refusal entirely.
//!
//! Material resolution has the same trap. **Two producers call this door today**
//! — that is the measured count, not three (P26.3b audit; the first draft of this
//! table named `inf_editor_core::render_assets`, which does not call it and does
//! not resolve a material at all):
//!
//! | path | who calls | what it produces |
//! |---|---|---|
//! | PIE payload | `inf_editor_core::pie` | `ScenePayload::materials` bytes |
//! | cook | `inf_packager::cook` | the `.inf_matd` entries of a pack, and the `.inf_mat` → `.inf_tex` closure edge |
//!
//! If those were two flattenings, "PIE == shipping" would be a claim about two
//! functions agreeing rather than about one function running twice. So they call
//! [`derive_material`], and the phase gate compares the **bytes** out of a real
//! pack against a real editor derivation — the shape `fracture_equivalence.rs`
//! established.
//!
//! **The third path is the open one.** The editor viewport and the shipped
//! player's render projector both still fill `vt: Default::default()` and neither
//! registers a texture outside tests, so nothing SAMPLES a material on either
//! host yet — P26.3b is the wire, and P26.4 is what makes this a door for three.
//! When it lands it must arrive through here rather than beside it; the grep that
//! finds a second spelling is for a read of `MaterialAsset`'s three texture slots
//! outside this crate, and today it finds only WRITERS (glTF import, the
//! photogrammetry finish, this crate's own tests).
//!
//! # Why the record's type is not in this crate
//!
//! Because the player must read it and must not link this crate: `inf-material`
//! owns the `image` decoders and the naga-validating material compiler, and the
//! P26.2 dependency inversion exists precisely to keep both out of a shipped
//! build. So [`DerivedMaterial`] lives in `inf-asset`, which the player already
//! links, and the door lives here, where `MaterialAsset` is. The question stays
//! in the authoring ring; the answer crosses.

use inf_asset::{DerivedBlend, DerivedMaterial};

use crate::material::{MatBlend, MaterialAsset};

/// Flatten a decoded `.inf_mat` into the record a runtime binds from.
///
/// Total and infallible: every field maps, and a material naming no texture
/// yields a record whose three slots are `None` — the scalars-only surface,
/// which is the permanent no-texture path rather than an error state.
pub fn derive_material(mat: &MaterialAsset) -> DerivedMaterial {
    DerivedMaterial {
        schema_version: DerivedMaterial::CURRENT_VERSION,
        albedo: mat.base_color_texture,
        normal: mat.normal_texture,
        orm: mat.metallic_roughness_texture,
        base_color: mat.base_color,
        metallic: mat.metallic,
        roughness: mat.roughness,
        emissive: mat.emissive,
        blend: match mat.blend {
            MatBlend::Opaque => DerivedBlend::Opaque,
            MatBlend::Masked => DerivedBlend::Masked,
            MatBlend::Translucent => DerivedBlend::Translucent,
        },
        alpha_cutoff: mat.alpha_cutoff,
        // Wave G: the detail slot crosses as a pair. An inert authoring state (a
        // texture with a zero scale) is carried across AS inert rather than
        // normalised away here, so the two records agree about what the material
        // says — the runtime's own `detail_scale_q8` is the single place that
        // turns "inert" into the renderer's zero.
        detail: mat.detail_texture,
        detail_scale_m: mat.detail_scale_m,
    }
}

/// [`derive_material`] from raw `.inf_mat` bytes — **the door the three paths
/// actually call**, because none of them holds a decoded `MaterialAsset`: the
/// cook reads a file, the payload builder is handed a resolver closure, and the
/// editor store reads its content root.
///
/// Taking bytes rather than a decoded value is what makes the agreement
/// mechanical: a path that decoded on its own could decode differently (a stale
/// build's positional bincode, a migration skipped), and then the three would
/// diverge *inside* a function that looks shared.
pub fn derive_material_bytes(bytes: &[u8]) -> Result<DerivedMaterial, inf_asset::AssetError> {
    let mat: MaterialAsset = inf_asset::decode(bytes)?;
    Ok(derive_material(&mat))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::AssetId;

    /// Three distinct ids, minted rather than spelled: `inf-material` has no
    /// `uuid` dependency and this batch is not the reason to add one. Bound to
    /// locals so each comparison below is against the same value it was built
    /// from — `AssetId::new()` is a fresh UUID on every call, which is exactly
    /// the trap `AssetId::default()` set for P26.1.
    fn three_ids() -> (AssetId, AssetId, AssetId) {
        let (a, b, c) = (AssetId::new(), AssetId::new(), AssetId::new());
        assert!(a != b && b != c && a != c, "three ids must be three");
        (a, b, c)
    }

    /// Every field crosses, and the three texture slots land in the three
    /// **named** slots — a mapping that swapped normal and ORM would round-trip
    /// perfectly and light every surface wrong (the P26.3 audit's own
    /// `InstanceRaw::pack` finding, one layer up).
    #[test]
    fn every_field_crosses_into_its_own_slot() {
        let (albedo, normal, orm) = three_ids();
        let detail = AssetId::new();
        assert!(detail != albedo && detail != normal && detail != orm);
        let mat = MaterialAsset {
            schema_version: MaterialAsset::CURRENT_VERSION,
            base_color: [0.1, 0.2, 0.3, 0.4],
            metallic: 0.6,
            roughness: 0.7,
            emissive: [0.8, 0.9, 1.0],
            base_color_texture: Some(albedo),
            normal_texture: Some(normal),
            metallic_roughness_texture: Some(orm),
            blend: MatBlend::Masked,
            alpha_cutoff: 0.375,
            detail_texture: Some(detail),
            detail_scale_m: 0.75,
        };
        let d = derive_material(&mat);
        assert_eq!(d.albedo, Some(albedo));
        assert_eq!(d.normal, Some(normal));
        assert_eq!(d.orm, Some(orm));
        assert_eq!(d.base_color, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(d.metallic, 0.6);
        assert_eq!(d.roughness, 0.7);
        assert_eq!(d.emissive, [0.8, 0.9, 1.0]);
        assert_eq!(d.blend, DerivedBlend::Masked);
        assert_eq!(d.alpha_cutoff, 0.375);
        assert_eq!(d.schema_version, DerivedMaterial::CURRENT_VERSION);
        // Wave G: the detail pair crosses into its own two slots, and the
        // dependency order APPENDS it.
        assert_eq!(d.detail, Some(detail));
        assert_eq!(d.detail_scale_m, 0.75);
        assert_eq!(
            d.texture_dependencies(),
            vec![albedo, normal, orm, detail],
            "the residency floor is a pure function of this order"
        );
        assert!(d.has_textures());
    }

    /// **The metres → 8.8 conversion, in the one place that owns it.**
    ///
    /// Both hosts read `detail_scale_q8()` off the record rather than converting
    /// themselves, so this is the only implementation there is — and the
    /// renderer's "zero disables the blend" encoding has to come out of it for
    /// every inert authoring state, or a material an artist thinks is off will
    /// render with a detail layer at an arbitrary scale.
    #[test]
    fn the_detail_scale_converts_once_and_encodes_inert_as_zero() {
        let detail = AssetId::new();
        let q = |scale: f32, tex: Option<AssetId>| {
            derive_material(&MaterialAsset {
                detail_texture: tex,
                detail_scale_m: scale,
                ..Default::default()
            })
            .detail_scale_q8()
        };
        // 8.8 fixed point: the integer part in the high byte.
        assert_eq!(q(1.0, Some(detail)), 256);
        assert_eq!(q(0.5, Some(detail)), 128);
        assert_eq!(q(0.25, Some(detail)), 64);
        assert_eq!(q(2.0, Some(detail)), 512);
        // Every inert state encodes as the renderer's own "disabled" zero.
        assert_eq!(q(0.5, None), 0, "no texture, no blend");
        assert_eq!(q(0.0, Some(detail)), 0);
        assert_eq!(q(-1.0, Some(detail)), 0);
        assert_eq!(q(f32::NAN, Some(detail)), 0);
        // A nonsense scale SATURATES rather than wrapping. Wrapping 300 m would
        // give 44 m — a wrong value that looks plausible, which is worse than an
        // obviously-clamped one.
        assert_eq!(q(1e6, Some(detail)), u16::MAX);
        // …and a tiny positive scale never rounds down INTO the disabled
        // encoding, which would silently turn "very fine detail" into "off".
        assert!(q(0.001, Some(detail)) >= 1);
    }

    /// All three blend modes map — and to three *distinct* values, so a `match`
    /// that collapsed two arms fails here rather than in a frame.
    #[test]
    fn the_three_blend_modes_map_to_three_distinct_values() {
        let of = |b: MatBlend| {
            derive_material(&MaterialAsset {
                blend: b,
                ..Default::default()
            })
            .blend
        };
        assert_eq!(of(MatBlend::Opaque), DerivedBlend::Opaque);
        assert_eq!(of(MatBlend::Masked), DerivedBlend::Masked);
        assert_eq!(of(MatBlend::Translucent), DerivedBlend::Translucent);
        let all = [
            of(MatBlend::Opaque),
            of(MatBlend::Masked),
            of(MatBlend::Translucent),
        ];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "two blend modes collapsed onto one");
            }
        }
    }

    /// A textureless material derives a textureless record — the no-texture
    /// fallback, which must stay structural rather than become an error.
    #[test]
    fn a_textureless_material_derives_a_textureless_record() {
        let d = derive_material(&MaterialAsset::default());
        assert!(!d.has_textures());
        assert!(d.texture_dependencies().is_empty());
        // …and its scalars are the material's, not a second set of defaults.
        assert_eq!(d.base_color, MaterialAsset::default().base_color);
        assert_eq!(d.roughness, MaterialAsset::default().roughness);
    }

    /// The bytes door and the value door agree, and the bytes door reports a
    /// decode failure rather than substituting a default — the class the P24.1
    /// audit's B1 finding is about (a payload that will not decode must not
    /// silently become a working default on one path and a refusal on another).
    #[test]
    fn the_bytes_door_agrees_with_the_value_door_and_refuses_garbage() {
        let mat = MaterialAsset {
            base_color_texture: Some(AssetId::new()),
            blend: MatBlend::Translucent,
            ..Default::default()
        };
        let bytes = inf_asset::encode(&mat).unwrap();
        assert_eq!(
            derive_material_bytes(&bytes).unwrap(),
            derive_material(&mat)
        );
        assert!(derive_material_bytes(&[0xFF, 0xFF, 0xFF]).is_err());
    }

    /// `texture_dependencies` on the derived record is the same SET the
    /// authored material reports — the cook walks one and the runtime the other,
    /// and a slot dropped in the flattening would ship a pack whose textures the
    /// player asks for and cannot find.
    #[test]
    fn the_dependency_sets_agree_across_the_door() {
        let (albedo, _, orm) = three_ids();
        let mat = MaterialAsset {
            base_color_texture: Some(albedo),
            normal_texture: None,
            metallic_roughness_texture: Some(orm),
            ..Default::default()
        };
        assert_eq!(
            derive_material(&mat).texture_dependencies(),
            mat.texture_dependencies(),
            "the derived record names different textures than the material it \
             came from — a cooked pack would carry the wrong set"
        );
    }
}
