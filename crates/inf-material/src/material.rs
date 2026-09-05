//! The `.inf_mat` material payload.
//!
//! Phase 4 ships the material *model* (a PBR metallic-roughness parameter block
//! with texture references by GUID). The node-graph editor and WGSL codegen are
//! Phase 7 — this is the data those build on, and what an imported glTF material
//! becomes.

use inf_asset::{AssetId, AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

/// How a [`MaterialAsset`] blends against the framebuffer (R-P5). Ring-0 mirror of
/// the ECS `inf_ecs::BlendMode` (inf-material must NOT depend on inf-ecs — the
/// editor glue maps between the two): `Opaque` is the pre-R-P5 behaviour;
/// `Masked` alpha-tests against [`MaterialAsset::alpha_cutoff`]; `Translucent`
/// alpha-blends. Serialized as an externally-tagged enum (bincode-safe — no
/// internal tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MatBlend {
    #[default]
    Opaque,
    Masked,
    Translucent,
}

/// A PBR metallic-roughness material. Texture slots hold asset GUIDs (the
/// material's dependency edges); `None` means "use the factor alone".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialAsset {
    pub schema_version: u32,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    #[serde(default)]
    pub base_color_texture: Option<AssetId>,
    #[serde(default)]
    pub normal_texture: Option<AssetId>,
    #[serde(default)]
    pub metallic_roughness_texture: Option<AssetId>,
    /// Blend / transparency mode (schema v2). Additive: `#[serde(default)]` →
    /// [`MatBlend::Opaque`], the pre-v2 behaviour.
    #[serde(default)]
    pub blend: MatBlend,
    /// Alpha-test threshold used when `blend == MatBlend::Masked` (schema v2).
    #[serde(default = "default_alpha_cutoff")]
    pub alpha_cutoff: f32,
    // ── schema v3 (Wave G) — the detail slot's authoring half ────────────────
    //
    // APPENDED, never inserted. bincode is positional: `#[serde(default)]` buys
    // nothing, and a field placed above `blend` would re-interpret every byte
    // after it in every committed material.
    /// A high-frequency **detail** texture blended over the base colour at close
    /// range (schema v3).
    ///
    /// # The half that was missing
    ///
    /// Wave T shipped the renderer's whole detail path — `VtMaterialMaps::detail`,
    /// the 8.8 fixed-point scale, the shader's `vt_apply_detail`, the mip-derived
    /// fade — and then had to spell `..Default::default()` at both host
    /// boundaries, because there was no `.inf_mat` field for an artist to put a
    /// texture in. It was a capability with no way to reach it. This is the field
    /// that turns it into a feature, and it is why Wave T's own disposition memo
    /// asked for it first in the consolidated schema wave.
    #[serde(default)]
    pub detail_texture: Option<AssetId>,
    /// How often [`detail_texture`](Self::detail_texture) repeats, in **tiles
    /// per uv unit** (schema v3).
    ///
    /// # The name says metres and the renderer does not (wave TER2a)
    ///
    /// `vt_apply_detail` computes `duv = uv · scale`, so this is a *rate*, not a
    /// length: a larger number makes the detail **finer**. The field is named
    /// `_m` and was documented as "world metres per tile", and that reading is
    /// backwards — which nothing noticed because until TER2a **no material in
    /// this repository named a detail texture at all**, so the field had no
    /// consumer to be wrong for.
    ///
    /// On a mesh there are no metres to be had: a mesh uv is whatever the
    /// author unwrapped, so "tiles per uv unit" is the only thing the number can
    /// mean and the behaviour is right. On **terrain**, where
    /// `uv = world.xz / TerrainLayer::tex_scale`, one detail tile covers
    /// `tex_scale / detail_scale_m` metres — so a layer tiling every 2 m with a
    /// detail scale of 16 gets a 12.5 cm detail tile.
    ///
    /// It is the *documentation* that is corrected rather than the arithmetic,
    /// on the VIS1b chromatic-aberration precedent: rescaling would change what
    /// every future authored number means, and the behaviour is correct for the
    /// surface type the field was designed against. Renaming the field would
    /// move the wire format for a comment.
    ///
    /// **Consequence worth stating**: [`DEFAULT_DETAIL_SCALE_M`] is `0.5`, which
    /// under the real semantics makes a detail layer **twice as coarse** as its
    /// base rather than ten times finer. See its own note.
    ///
    /// # Why the reference alone would have been inert
    ///
    /// The renderer's slot and its scale are **one decision**: `VtMaterialMaps`
    /// carries `detail_scale_q8`, and a scale of zero disables the blend even
    /// when a detail texture is bound — there is a Wave T test asserting exactly
    /// that. So shipping `detail_texture` on its own would have produced a field
    /// an artist can fill in that changes nothing on screen, which is a worse
    /// outcome than not shipping it. The two land together.
    ///
    /// [`DEFAULT_DETAIL_SCALE_M`] is the default so a material that names a
    /// detail texture and says nothing else gets a visible blend — though see
    /// the unit note above for what "sensible" turned out to mean.
    #[serde(default = "default_detail_scale")]
    pub detail_scale_m: f32,
    // ── schema v4 (wave ROAD1) — the physical tiling rate ────────────────────
    //
    // APPENDED, for the same bincode-positional reason v3's pair was.
    /// **World metres per texture repeat** — how big this surface's texture is
    /// in the world (schema v4). `0.0` means *the mesh's uv is already in tile
    /// units and is used as it was authored*, which is every material written
    /// before this bump and every material that says nothing.
    ///
    /// # The number that made the road read three and a half times life size
    ///
    /// `inf_gis`'s road ribbon used to write `uv = (u across the carriageway,
    /// arc / width_m)`, so **one uv unit was the road's own width** — 14.0 m on
    /// the island, because the road layer carried no lane count and
    /// `RoadKind::default_lanes` gives an arterial four. The committed asphalt
    /// set is authored at 2 m and quotes 4 m
    /// ([`crate::ground::GroundKind::tex_scale_m`]), so the street tiled
    /// every 14 m: aggregate the size of paving slabs. The ASSET0 audit measured
    /// it and named the missing field; this is that field.
    ///
    /// # The contract, and it is a contract
    ///
    /// A material that states a metres-per-repeat is a material whose **mesh's
    /// uv is in METRES**. The renderer divides: `uv_sampled = uv / uv_tiling_m`.
    /// A mesh unwrapped in tile units must leave this at `0.0` or it will tile
    /// at the wrong rate — which is why `0.0` and not `1.0` is the default, and
    /// why the two states are distinguishable rather than one being a special
    /// case of the other.
    ///
    /// **Terrain never sets it.** A terrain layer's uv is already
    /// `world.xz / TerrainLayer::tex_scale`, so the layer's own rate is the door
    /// there and a second one would divide twice; `passes::terrain`'s slot
    /// builder zeroes this deliberately.
    ///
    /// Carried to the GPU as unsigned **8.8 fixed point** in the spare top half
    /// of the third instance word (`crate::scene::VtTextureSet::slots`) — the
    /// same free 16 bits Wave T's detail scale rides in, so it costs no instance
    /// byte, no vertex attribute and no bind group. The ceiling that follows
    /// from 8.8 is **255.996 m** per repeat, which is past every surface an
    /// author tiles.
    ///
    /// # The wet-road hook (routed to PAR4, unread today)
    ///
    /// See [`wet_roughness_floor`](Self::wet_roughness_floor), appended beside
    /// it in the same bump so the road's two authored numbers land together.
    #[serde(default)]
    pub uv_tiling_m: f32,
    /// **How smooth this surface goes when it is wet** — a roughness floor the
    /// wetness pass may drive toward, `0.0` = "no opinion" (schema v4).
    ///
    /// # A field with no reader, stated as such
    ///
    /// Nothing reads this today. P20.3's shoreline wetness darkens albedo and
    /// lowers roughness by one global rule for every surface it touches
    /// (`mesh.wgsl`'s `wet_apply`), and a road is the one surface whose wet
    /// behaviour is materially different from a rock's: rain sits in the voids
    /// between the aggregate and the polished chips turn the whole carriageway
    /// into a mirror, which is what `frames/steal-car/0028` shows and what a
    /// uniformly-damped roughness does not.
    ///
    /// It is authored now rather than later because the alternative is a second
    /// `.inf_mat` window for one `f32` — the schema-move law's own arithmetic.
    /// PAR4 owns the pass that reads it. Until then it round-trips and is
    /// otherwise inert, and this doc is the record that it is inert on purpose.
    #[serde(default)]
    pub wet_roughness_floor: f32,
}

fn default_alpha_cutoff() -> f32 {
    0.5
}

/// The default [`MaterialAsset::detail_scale_m`] — **tiles per uv unit**.
///
/// # This number is wrong for terrain and is kept anyway (wave TER2a)
///
/// It was chosen as "half a metre per tile", reasoning from the terrain splat's
/// own `tex_scale` defaults (4–10 m) "divided by the ~10× frequency ratio a
/// detail layer is for". The renderer's `duv = uv · scale` means the opposite:
/// `0.5` makes the detail repeat **half as often** as the base uv, so on a
/// terrain layer tiling every 4 m the "detail" tile is 8 m — twice as coarse as
/// the thing it details.
///
/// It is not changed, for the reason a default is a default: it is the value
/// every `.inf_mat` written before TER2a decodes with (the field is
/// `#[serde(default)]`), and moving it would silently re-scale content nobody
/// has authored yet in a direction no author asked for. What is changed is that
/// the number now says what it does. Content that wants a detail layer states a
/// rate: `inf_material::ground` authors 16 and 20.
///
/// The honest fix is a field whose name and unit agree, which is a schema move
/// and therefore a wave rather than a comment.
pub const DEFAULT_DETAIL_SCALE_M: f32 = 0.5;

fn default_detail_scale() -> f32 {
    DEFAULT_DETAIL_SCALE_M
}

impl Default for MaterialAsset {
    fn default() -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            blend: MatBlend::Opaque,
            alpha_cutoff: default_alpha_cutoff(),
            detail_texture: None,
            detail_scale_m: default_detail_scale(),
            uv_tiling_m: 0.0,
            wet_roughness_floor: 0.0,
        }
    }
}

impl MaterialAsset {
    /// The `.inf_mat` ladder.
    ///
    /// | version | what it added |
    /// |---|---|
    /// | v1 | the original PBR block + three texture slots |
    /// | v2 (R-P5) | `blend` + `alpha_cutoff` |
    /// | **v3 (Wave G)** | `detail_texture` + `detail_scale_m` |
    /// | **v4 (wave ROAD1)** | `uv_tiling_m` + `wet_roughness_floor` |
    ///
    /// This container bumps **once** a wave, and v4 is ROAD1's bump. Both new
    /// fields default to `0.0` — "no opinion" — so every committed `.inf_mat`
    /// decodes to exactly the surface it decoded to before, and the wave's own
    /// two materials are the only ones that state a rate.
    pub const CURRENT_VERSION: u32 = 4;

    /// **Whether this material states a physical tiling rate** — see
    /// [`uv_tiling_m`](Self::uv_tiling_m).
    ///
    /// The predicate rather than a bare `> 0.0` at each call site, so "the uv is
    /// authored in tile units" is asked one way in every host. Non-finite and
    /// negative rates answer `false`: a NaN divisor would put a NaN in a uv, and
    /// the quiet failure that follows is a surface sampling mip 0 everywhere.
    pub fn tiles_physically(&self) -> bool {
        self.uv_tiling_m.is_finite() && self.uv_tiling_m > 0.0
    }

    /// Every texture GUID this material references, for building the asset
    /// dependency edges.
    ///
    /// **Order is part of the contract**, and the detail slot is APPENDED. The
    /// same law `DerivedMaterial::texture_dependencies` states out loud: the
    /// streaming residency floor is a pure function of the registration
    /// sequence, so inserting a slot ahead of an existing one would silently
    /// re-order what a shipped pack keeps resident.
    pub fn texture_dependencies(&self) -> Vec<AssetId> {
        [
            self.base_color_texture,
            self.normal_texture,
            self.metallic_roughness_texture,
            self.detail_texture,
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// Whether the detail blend will actually do anything.
    ///
    /// A bound texture with a zero (or non-finite) scale is **inert** — the
    /// renderer disables the blend, by design. This is the predicate that lets a
    /// caller tell "no detail" from "detail that will not show", which are
    /// different authoring mistakes.
    pub fn detail_is_active(&self) -> bool {
        self.detail_texture.is_some()
            && self.detail_scale_m.is_finite()
            && self.detail_scale_m > 0.0
    }
}

impl AssetPayload for MaterialAsset {
    const KIND: AssetKind = AssetKind::Material;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    const UPGRADE_REMEDY: &'static str =
        "re-import it from its source file, or re-save it from the material editor";
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

// ── the frozen ladder ───────────────────────────────────────────────────────

/// The frozen **schema-v2** `.inf_mat` layout, as it shipped before Wave G.
///
/// # Why a real shadow struct rather than a trimmed encoding
///
/// The same reason `SkeletonAssetV1` is one: a hand-trimmed byte sequence
/// asserts what the author *believes* v2 looked like, whereas a struct the
/// compiler serializes asserts what serde *actually* produced for that field
/// list. Only the second can fail when somebody edits the live struct and
/// forgets the ladder.
#[cfg(test)]
#[derive(Serialize)]
struct MaterialAssetV2 {
    schema_version: u32,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    base_color_texture: Option<AssetId>,
    normal_texture: Option<AssetId>,
    metallic_roughness_texture: Option<AssetId>,
    blend: MatBlend,
    alpha_cutoff: f32,
}

/// The frozen **schema-v3** `.inf_mat` layout, as it shipped before ROAD1.
///
/// A `Serialize` twin, for the same reason [`MaterialAssetV2`] is one: it lets
/// the downgrade arm feed genuinely v3-shaped bytes through the live decoder and
/// watch it run off the end, rather than asserting what the author believes v3
/// looked like.
#[cfg(test)]
#[derive(Serialize)]
struct MaterialAssetV3 {
    schema_version: u32,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    base_color_texture: Option<AssetId>,
    normal_texture: Option<AssetId>,
    metallic_roughness_texture: Option<AssetId>,
    blend: MatBlend,
    alpha_cutoff: f32,
    detail_texture: Option<AssetId>,
    detail_scale_m: f32,
}

/// The v4 wire shape, **re-declared independently** of the live struct.
///
/// Deserializing a real encoding through this and asserting it consumed *every*
/// byte is what catches a future tail field appended without a version bump —
/// the failure mode that leaves old materials decoding into the wrong fields.
#[cfg(test)]
#[derive(Deserialize)]
// A wire pin exists to be DECODED THROUGH: a few fields are read by name below,
// and the rest are the SHAPE, which is the point. Dropping one to satisfy
// dead-code analysis would stop the pin noticing a field appended without a
// bump — the exact failure it exists to catch.
#[allow(dead_code)]
struct MaterialAssetV4Wire {
    schema_version: u32,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    base_color_texture: Option<AssetId>,
    normal_texture: Option<AssetId>,
    metallic_roughness_texture: Option<AssetId>,
    blend: MatBlend,
    alpha_cutoff: f32,
    detail_texture: Option<AssetId>,
    detail_scale_m: f32,
    uv_tiling_m: f32,
    wet_roughness_floor: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::{decode, encode};

    #[test]
    fn dependencies_list_present_textures() {
        let tex = AssetId::new();
        let m = MaterialAsset {
            base_color_texture: Some(tex),
            ..Default::default()
        };
        assert_eq!(m.texture_dependencies(), vec![tex]);
        assert!(MaterialAsset::default().texture_dependencies().is_empty());
    }

    #[test]
    fn round_trips() {
        let m = MaterialAsset::default();
        assert_eq!(decode::<MaterialAsset>(&encode(&m).unwrap()).unwrap(), m);
    }

    #[test]
    fn default_blend_is_opaque_v3() {
        let m = MaterialAsset::default();
        assert_eq!(m.blend, MatBlend::Opaque);
        assert_eq!(m.alpha_cutoff, 0.5);
        assert_eq!(m.schema_version, 4);
        assert_eq!(MaterialAsset::CURRENT_VERSION, 4);
        // The detail slot is off by default, so no existing material changes.
        assert_eq!(m.detail_texture, None);
        assert_eq!(m.detail_scale_m, DEFAULT_DETAIL_SCALE_M);
        assert!(!m.detail_is_active(), "no texture, no detail");
        // v4's pair is off by default too: a material that says nothing about
        // its tiling keeps the uv its mesh was authored with.
        assert_eq!(m.uv_tiling_m, 0.0);
        assert_eq!(m.wet_roughness_floor, 0.0);
        assert!(!m.tiles_physically(), "0 m per tile is `use the uv as-is`");
    }

    /// **The tiling rate refuses the values that would put a NaN in a uv.**
    ///
    /// `tiles_physically` is the one door, and it is spelled as a positive
    /// finite test rather than as `!(x <= 0.0)`: every ordering comparison a NaN
    /// takes part in is false, so the negated form lets one through — the same
    /// shape `scene::detail_scale_q8` spells out one crate over.
    #[test]
    fn a_tiling_rate_is_a_positive_finite_length_or_it_is_no_opinion() {
        for bad in [0.0, -0.0, -4.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let m = MaterialAsset {
                uv_tiling_m: bad,
                ..Default::default()
            };
            assert!(
                !m.tiles_physically(),
                "{bad} m per tile must not be taken as a rate"
            );
        }
        for good in [0.25_f32, 1.0, 2.0, 4.0, 255.0] {
            let m = MaterialAsset {
                uv_tiling_m: good,
                ..Default::default()
            };
            assert!(m.tiles_physically(), "{good} m per tile is a rate");
        }
    }

    /// **A v2 payload is refused BY NAME**, and the refusal names the remedy.
    ///
    /// This is the arm that proves the bump is real: encoded through the frozen
    /// v2 shadow struct, a v2 payload is genuinely two fields short, so decoding
    /// it as v3 runs off the end.
    ///
    /// Un-fix mutation: leave `CURRENT_VERSION` at 2 and this stops failing for
    /// the right reason.
    #[test]
    fn a_v2_payload_is_refused_by_name_with_its_remedy() {
        let tex = AssetId::new();
        let v2 = MaterialAssetV2 {
            schema_version: 2,
            base_color: [0.1, 0.2, 0.3, 1.0],
            metallic: 0.25,
            roughness: 0.75,
            emissive: [1.0, 0.0, 0.0],
            base_color_texture: Some(tex),
            normal_texture: None,
            metallic_roughness_texture: None,
            blend: MatBlend::Masked,
            alpha_cutoff: 0.4,
        };
        let bytes = bincode::serde::encode_to_vec(&v2, inf_asset::bincode_config())
            .expect("encode the frozen v2 record");
        let err = decode::<MaterialAsset>(&bytes).unwrap_err().to_string();
        assert!(
            err.contains("too old") || err.contains("2"),
            "a v2 material must be refused by name: {err}"
        );
        assert!(
            err.contains("re-import") || err.contains("re-save"),
            "…and the refusal must name the remedy: {err}"
        );
    }

    /// A payload from a FUTURE build is refused too, in the other direction.
    #[test]
    fn a_future_payload_is_refused_as_too_new() {
        let m = MaterialAsset {
            schema_version: MaterialAsset::CURRENT_VERSION + 1,
            ..Default::default()
        };
        let err = m.migrate().unwrap_err().to_string();
        assert!(err.contains("5") && err.contains("4"), "{err}");
    }

    /// **A v3 payload is refused BY NAME, and the refusal names the remedy** —
    /// the arm that proves ROAD1's bump is real rather than a constant.
    ///
    /// Encoded through the frozen v3 shadow struct, a v3 payload is genuinely
    /// two `f32`s short, so decoding it as v4 runs off the end of the buffer.
    /// This is the **downgrade bless**: the wave states that pre-ROAD1 `.inf_mat`
    /// bytes do not load, and the remedy the refusal names is the one the asset
    /// kind already carries ("re-import it … or re-save it from the material
    /// editor").
    ///
    /// Un-fix mutation: leave `CURRENT_VERSION` at 3 and this stops failing for
    /// the right reason.
    #[test]
    fn a_v3_payload_is_refused_by_name_with_its_remedy() {
        let v3 = MaterialAssetV3 {
            schema_version: 3,
            base_color: [0.1, 0.2, 0.3, 1.0],
            metallic: 0.25,
            roughness: 0.75,
            emissive: [1.0, 0.0, 0.0],
            base_color_texture: Some(AssetId::new()),
            normal_texture: None,
            metallic_roughness_texture: None,
            blend: MatBlend::Masked,
            alpha_cutoff: 0.4,
            detail_texture: None,
            detail_scale_m: DEFAULT_DETAIL_SCALE_M,
        };
        let bytes = bincode::serde::encode_to_vec(&v3, inf_asset::bincode_config())
            .expect("encode the frozen v3 record");
        let err = decode::<MaterialAsset>(&bytes).unwrap_err().to_string();
        assert!(
            err.contains("too old") || err.contains("3"),
            "a v3 material must be refused by name: {err}"
        );
        assert!(
            err.contains("re-import") || err.contains("re-save"),
            "…and the refusal must name the remedy: {err}"
        );
    }

    /// **The wire-shape pin.** The v4 encoding is decoded through an
    /// independently-declared twin and must consume every byte — so a tail field
    /// appended without a bump fails here rather than in somebody's project.
    #[test]
    fn the_v4_wire_shape_is_pinned_against_an_independent_declaration() {
        let m = MaterialAsset {
            schema_version: MaterialAsset::CURRENT_VERSION,
            base_color: [0.4, 0.5, 0.6, 0.7],
            metallic: 0.125,
            roughness: 0.875,
            emissive: [0.25, 0.5, 0.75],
            base_color_texture: Some(AssetId::new()),
            normal_texture: Some(AssetId::new()),
            metallic_roughness_texture: Some(AssetId::new()),
            blend: MatBlend::Translucent,
            alpha_cutoff: 0.33,
            detail_texture: Some(AssetId::new()),
            detail_scale_m: 0.25,
            uv_tiling_m: 4.0,
            wet_roughness_floor: 0.18,
        };
        let bytes = encode(&m).unwrap();
        let (wire, consumed): (MaterialAssetV4Wire, usize) =
            bincode::serde::decode_from_slice(&bytes, inf_asset::bincode_config())
                .expect("the v4 wire twin decodes");
        assert_eq!(
            consumed,
            bytes.len(),
            "the v4 encoding has {} bytes the pinned shape does not account for — \
             a field was appended without a schema bump",
            bytes.len() - consumed
        );
        // Each tail landed in its OWN named slot, not merely somewhere.
        assert_eq!(wire.schema_version, 4);
        assert_eq!(wire.alpha_cutoff, 0.33);
        assert_eq!(wire.detail_texture, m.detail_texture);
        assert_eq!(wire.detail_scale_m, 0.25);
        assert_eq!(wire.blend, MatBlend::Translucent);
        assert_eq!(wire.uv_tiling_m, 4.0);
        assert_eq!(wire.wet_roughness_floor, 0.18);
    }

    /// The detail slot round-trips, is listed as a dependency **last**, and a
    /// zero scale is recognised as inert rather than as "no detail".
    #[test]
    fn the_detail_slot_round_trips_and_reports_when_it_is_inert() {
        let base = AssetId::new();
        let detail = AssetId::new();
        let m = MaterialAsset {
            base_color_texture: Some(base),
            detail_texture: Some(detail),
            detail_scale_m: 0.25,
            ..Default::default()
        };
        let back = decode::<MaterialAsset>(&encode(&m).unwrap()).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.detail_texture, Some(detail));
        assert_eq!(back.detail_scale_m, 0.25);
        assert!(back.detail_is_active());

        // APPENDED: the detail slot comes last in the dependency order, which the
        // residency floor is a pure function of.
        assert_eq!(m.texture_dependencies(), vec![base, detail]);

        // A bound texture with a zero scale is INERT — the renderer disables the
        // blend — and that is a different state from "no detail texture".
        let inert = MaterialAsset {
            detail_texture: Some(detail),
            detail_scale_m: 0.0,
            ..Default::default()
        };
        assert!(!inert.detail_is_active(), "a zero scale disables the blend");
        assert_eq!(
            inert.texture_dependencies(),
            vec![detail],
            "…but it is still a real dependency edge: the asset is referenced"
        );
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            let m = MaterialAsset {
                detail_texture: Some(detail),
                detail_scale_m: bad,
                ..Default::default()
            };
            assert!(
                !m.detail_is_active(),
                "scale {bad} must not count as active"
            );
        }
    }

    /// A translucent/masked material round-trips its blend + cutoff through
    /// bincode (schema v2). The default `migrate` accepts an equal-or-older
    /// version and leaves the additive defaults (old → Opaque) untouched.
    #[test]
    fn blend_and_cutoff_round_trip_and_migrate() {
        let m = MaterialAsset {
            blend: MatBlend::Translucent,
            alpha_cutoff: 0.3,
            ..Default::default()
        };
        let back = decode::<MaterialAsset>(&encode(&m).unwrap()).unwrap();
        assert_eq!(back.blend, MatBlend::Translucent);
        assert_eq!(back.alpha_cutoff, 0.3);
        // A value stamped with the older schema still migrates cleanly (version
        // guard only — additive fields keep their Opaque defaults). Note this
        // exercises the guard, NOT a real short read: the bytes here are already
        // v3-shaped. `a_v2_payload_is_refused_by_name_with_its_remedy` is the arm
        // that feeds genuinely v2 bytes through the door.
        let old = MaterialAsset {
            schema_version: 1,
            ..Default::default()
        };
        let migrated = old.migrate().unwrap();
        assert_eq!(migrated.blend, MatBlend::Opaque);
    }
}
