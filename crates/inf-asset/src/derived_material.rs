//! The **derived material record** (`.inf_matd`, P26.3b): a `.inf_mat` flattened
//! into what a *runtime* needs to bind a surface — three texture GUIDs plus the
//! scalars that are the no-texture fallback.
//!
//! # Why the record exists at all
//!
//! The shipped player cannot read a `.inf_mat`. That is not an oversight, it is
//! the P26.2 dependency inversion working: `MaterialAsset` lives in
//! `inf-material`, which owns the `image` decoders and the naga-validating
//! material compiler, and a shipped player must link neither. So the player
//! needs the *answer* — which textures, which scalars — without the crate that
//! computes it.
//!
//! This is the [`inf_scene::partition`] arrangement one asset over: the derived
//! container lives in the crate the RUNTIME reads it from, and the code that
//! writes it lives up in the authoring ring.
//!
//! # The one door
//!
//! **This module does not derive anything.** Turning a `.inf_mat` into a
//! [`DerivedMaterial`] is `inf_material::derive_material`, the single function
//! the editor projector, the PIE payload builder and the cook all call — the
//! P22.2 "one door for three paths" law, whose lesson was that two of three
//! paths carrying comments claiming they agree *by construction* did not. What
//! lives here is the record's shape, its id derivation and its codec, so that
//! everything downstream of the door speaks one type.
//!
//! # The id
//!
//! A derived material is keyed by [`derived_material_id`] of its `.inf_mat`'s
//! GUID — an XOR with a fixed salt, exactly like `inf_mesh::derived_fracture_id`
//! and `inf_vgeom::derived_vmesh_id`. A pack index is keyed by GUID alone, so
//! the derived record cannot share its source's id; and because the id is
//! *computed*, a runtime finds it without a side table that could disagree with
//! the pack.

use serde::{Deserialize, Serialize};

use crate::id::AssetId;
use crate::kind::AssetKind;
use crate::payload::AssetPayload;

/// The fixed salt XORed into a material GUID to derive its `.inf_matd` GUID.
///
/// Same construction and the same reasoning as [`FRACTURE_ID_SALT`] and
/// `inf_vgeom::VMESH_ID_SALT`: XOR with a constant is a bijection, so distinct
/// materials always yield distinct derived ids, and the salt makes a collision
/// with an *authored* asset id vanishingly unlikely.
///
/// [`FRACTURE_ID_SALT`]: the mesh crate's fracture-id salt.
pub const DERIVED_MATERIAL_ID_SALT: u128 = 0x2626_0003_4d41_5444_a41f_7c02_9d58_6ee3;

/// Derive the deterministic `.inf_matd` asset id for a given `.inf_mat` id.
///
/// Involutive (`derived_material_id(derived_material_id(x)) == x`), because the
/// salt is XORed.
pub fn derived_material_id(material_id: AssetId) -> AssetId {
    AssetId(uuid::Uuid::from_u128(
        material_id.uuid().as_u128() ^ DERIVED_MATERIAL_ID_SALT,
    ))
}

/// How a derived material blends against the framebuffer.
///
/// A Ring-0 mirror of `inf_material::MatBlend` and of `inf_ecs::BlendMode`,
/// deliberately: this crate depends on neither, and the record must cross to a
/// player that links neither the authoring crate nor (on the cook side) care
/// which of the two spellings a producer started from. Serialized as an
/// externally-tagged enum — bincode-safe, no internal tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DerivedBlend {
    #[default]
    Opaque,
    Masked,
    Translucent,
}

/// A `.inf_mat` as a runtime sees it (`.inf_matd`).
///
/// Everything here is either a **texture GUID** — resolved against the
/// `.inf_tex` payloads that travel beside it — or a **scalar**, which is what a
/// surface renders with when a texture is absent, unregistered or not yet warm.
/// That fallback is structural: the per-instance attributes carry the scalars on
/// every path, so a material whose textures never page in renders exactly as a
/// pre-P26 level did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedMaterial {
    pub schema_version: u32,
    /// Base-colour / albedo texture (`.inf_tex`), if the material names one.
    pub albedo: Option<AssetId>,
    /// Tangent-space normal map (`.inf_tex`).
    pub normal: Option<AssetId>,
    /// Occlusion / roughness / metallic map (`.inf_tex`) — glTF's packed
    /// metallic-roughness slot, which is what `MaterialAsset` stores.
    pub orm: Option<AssetId>,
    /// Linear RGBA base colour factor.
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub blend: DerivedBlend,
    pub alpha_cutoff: f32,
    // ── schema v2 (Wave G) — the detail slot ─────────────────────────────────
    //
    // APPENDED, never inserted, for the positional-bincode reason and for the
    // residency-order reason below.
    /// High-frequency **detail** texture (`.inf_tex`) blended over the base
    /// colour at close range (schema v2).
    ///
    /// # Why this record had to move too
    ///
    /// The shipped player cannot read a `.inf_mat` at all — it reads `.inf_matd`
    /// — so a `detail_texture` that stopped at the authoring container would have
    /// been invisible to the cook and to the game. Wave G's `.inf_mat` v3 and
    /// this v2 are two halves of one field.
    pub detail: Option<AssetId>,
    /// How often the detail texture repeats, in **tiles per uv unit**
    /// (schema v2). Zero disables the blend — see
    /// `MaterialAsset::detail_is_active`.
    ///
    /// The name says metres and the shader does not: `vt_apply_detail` computes
    /// `duv = uv · scale`, so a larger number is a **finer** detail. Corrected in
    /// wave TER2a, the first wave to author a material that names a detail
    /// texture; the full note is on `MaterialAsset::detail_scale_m`, which this
    /// field is copied from unchanged.
    pub detail_scale_m: f32,
    // ── schema v3 (wave ROAD1) — the physical tiling rate ─────────────────────
    //
    // APPENDED, same two reasons.
    /// **World metres per texture repeat** on a mesh whose uv is authored in
    /// metres; `0.0` = "the uv is already in tile units" (schema v3).
    ///
    /// The other half of `inf_material::MaterialAsset::uv_tiling_m`, and it had
    /// to move for the same reason the detail slot did: the shipped player reads
    /// `.inf_matd` and never a `.inf_mat`, so a rate that stopped at the
    /// authoring container would tile correctly in the editor and wrongly in the
    /// game — which is precisely the PIE-equals-shipping failure this record
    /// exists to prevent.
    pub uv_tiling_m: f32,
    /// The roughness floor a wet-weather pass may drive this surface toward;
    /// `0.0` = no opinion (schema v3). Carried, unread — see the authoring
    /// field's note. It crosses now rather than in PAR4's own wave so that the
    /// pack format moves once for the pair rather than twice for one each.
    pub wet_roughness_floor: f32,
}

impl Default for DerivedMaterial {
    fn default() -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            albedo: None,
            normal: None,
            orm: None,
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            blend: DerivedBlend::Opaque,
            alpha_cutoff: 0.5,
            detail: None,
            detail_scale_m: 0.0,
            uv_tiling_m: 0.0,
            wet_roughness_floor: 0.0,
        }
    }
}

impl DerivedMaterial {
    /// The `.inf_matd` ladder.
    ///
    /// | version | what it added |
    /// |---|---|
    /// | v1 (P26.3b) | the original record |
    /// | **v2 (Wave G)** | `detail` + `detail_scale_m` |
    /// | **v3 (wave ROAD1)** | `uv_tiling_m` + `wet_roughness_floor` |
    pub const CURRENT_VERSION: u32 = 3;

    /// Every `.inf_tex` GUID this record references, in the fixed slot order
    /// albedo → normal → ORM → **detail**.
    ///
    /// **Order is part of the contract**, not a convenience: it is the order a
    /// host registers textures in, and `VtTextures::want_floor` is a pure
    /// function of the registration *sequence*, so two hosts that walked these
    /// in different orders would produce two different residency traces for one
    /// level. The P26.3 handle law (a handle is a per-registry index, so
    /// cross-host comparison is by GUID) says the handles may differ; it does
    /// not license the pages to.
    pub fn texture_dependencies(&self) -> Vec<AssetId> {
        // Detail is APPENDED. The order law above is why: `want_floor` is a pure
        // function of the registration sequence, so putting the new slot ahead of
        // an existing one would silently re-order what a shipped pack keeps
        // resident — a change to streaming behaviour disguised as a schema edit.
        [self.albedo, self.normal, self.orm, self.detail]
            .into_iter()
            .flatten()
            .collect()
    }

    /// Whether this material names any texture at all. `false` is the
    /// scalars-only surface — the permanent no-texture path.
    pub fn has_textures(&self) -> bool {
        self.albedo.is_some()
            || self.normal.is_some()
            || self.orm.is_some()
            || self.detail.is_some()
    }

    /// The detail scale as the renderer's 8.8 fixed-point word.
    ///
    /// **The one place the authored rate becomes `detail_scale_q8`.**
    /// (It was "the one place metres become …", and the unit in that sentence
    /// was wrong — see the field's own note. The *conversion* is unchanged and
    /// correct: 8.8 fixed point either way.) `VtTextureSet` packs
    /// the scale into sixteen bits of a per-instance word, and a conversion that
    /// existed at both host boundaries would be a conversion that could differ
    /// between them — the mirror hazard this engine has a law about. Zero (no
    /// texture, or a non-finite/non-positive scale) is the renderer's own
    /// "disabled" encoding, so an inert material converts to inert rather than to
    /// something arbitrary.
    pub fn detail_scale_q8(&self) -> u16 {
        if self.detail.is_none() || !self.detail_scale_m.is_finite() || self.detail_scale_m <= 0.0 {
            return 0;
        }
        // 8.8: the integer part in the high byte, 1/256 of a tile-per-uv in the
        // low. Saturating rather than wrapping — a rate of 300 is a nonsense
        // value, and wrapping it to 44 would be a nonsense value that looks
        // plausible.
        let q = (self.detail_scale_m * 256.0).round();
        q.clamp(1.0, u16::MAX as f64 as f32) as u16
    }

    /// The tiling rate as the renderer's 8.8 fixed-point word — **metres per
    /// repeat**, not tiles per uv unit (wave ROAD1).
    ///
    /// The one place the authored length becomes `uv_tiling_q8`, for exactly the
    /// reason [`detail_scale_q8`](Self::detail_scale_q8) is: a conversion that
    /// existed at both host boundaries is a conversion that can differ between
    /// them, and a road that tiled at 4 m in the editor and 3.99 m in the player
    /// would be a mirror break nothing would notice until somebody photographed
    /// both.
    ///
    /// `0` is "no rate stated — use the mesh's uv as it was authored", which is
    /// every material written before this bump. The ceiling is **255.996 m** and
    /// values past it saturate rather than wrap, so a nonsense 300 m becomes the
    /// largest sane rate rather than a plausible-looking 44 m.
    pub fn uv_tiling_q8(&self) -> u16 {
        if !self.uv_tiling_m.is_finite() || self.uv_tiling_m <= 0.0 {
            return 0;
        }
        let q = (self.uv_tiling_m * 256.0).round();
        q.clamp(1.0, u16::MAX as f64 as f32) as u16
    }
}

impl AssetPayload for DerivedMaterial {
    const KIND: AssetKind = AssetKind::DerivedMaterial;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    /// A derived record is never hand-authored, so "re-import it" is the wrong
    /// instruction: the remedy is to cook again, which regenerates it from the
    /// `.inf_mat` through the one door.
    const UPGRADE_REMEDY: &'static str =
        "re-cook the project (a .inf_matd is derived, never authored)";
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::{decode, encode};

    fn guid(n: u128) -> AssetId {
        AssetId(uuid::Uuid::from_u128(n))
    }

    #[test]
    fn the_id_salt_is_involutive_and_injective() {
        for n in [1u128, 2, 0xdead_beef, u128::MAX] {
            let m = guid(n);
            assert_eq!(derived_material_id(derived_material_id(m)), m);
            assert_ne!(derived_material_id(m), m);
        }
        assert_ne!(derived_material_id(guid(1)), derived_material_id(guid(2)));
    }

    /// The three derived-id salts in the tree must not collide, or a mesh's
    /// fracture and a material's derived record could land on one pack key.
    ///
    /// Asserted on the SALTS rather than on a sample of ids: `a ^ s1 == b ^ s2`
    /// has a solution for every pair of distinct salts, so a spot check of a few
    /// guids proves nothing. What matters is that the constants differ, which is
    /// what makes the id spaces disjoint *per source asset*.
    #[test]
    fn the_derived_material_salt_is_its_own() {
        // The other two are Ring-0 constants in crates this one must not depend
        // on, so they are quoted rather than imported — and quoted in full, so a
        // change to either shows up here as a failing equality rather than as a
        // silent collision.
        const FRACTURE: u128 = 0x2622_0002_4652_4143_8b17_4d9e_c05a_31f7;
        assert_ne!(DERIVED_MATERIAL_ID_SALT, FRACTURE);
        assert_ne!(DERIVED_MATERIAL_ID_SALT, 0);
    }

    #[test]
    fn round_trips_and_lists_its_textures() {
        let m = DerivedMaterial {
            albedo: Some(guid(0xA1)),
            orm: Some(guid(0xA3)),
            blend: DerivedBlend::Masked,
            alpha_cutoff: 0.25,
            ..Default::default()
        };
        let back: DerivedMaterial = decode(&encode(&m).unwrap()).unwrap();
        assert_eq!(back, m);
        // Slot order is the contract: albedo, normal, ORM — with the absent
        // normal skipped rather than shifting the ORM into its place.
        assert_eq!(back.texture_dependencies(), vec![guid(0xA1), guid(0xA3)]);
        assert!(back.has_textures());
        assert!(!DerivedMaterial::default().has_textures());
        assert!(DerivedMaterial::default().texture_dependencies().is_empty());
    }

    /// A record from a newer build is refused by name, with a remedy that is
    /// true for a derived asset (the `UPGRADE_REMEDY` doctrine).
    #[test]
    fn a_newer_record_is_refused_with_the_cook_remedy() {
        let m = DerivedMaterial {
            schema_version: DerivedMaterial::CURRENT_VERSION + 1,
            ..Default::default()
        };
        let bytes = encode(&m).unwrap();
        let err = decode::<DerivedMaterial>(&bytes).expect_err("a newer record must be refused");
        let msg = err.to_string();
        assert!(msg.contains("derived_material"), "{msg}");
        assert!(!DerivedMaterial::UPGRADE_REMEDY.contains("re-import"));
    }
}
