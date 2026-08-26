//! **The committed ground library** (wave TER2a, clause 3) — five PBR ground
//! sets written into `samples/ground/` as engine content.
//!
//! # What this is for
//!
//! `inf_material::ground` synthesises the texels; this writes them into the
//! dual-format assets the rest of the engine reads — a `.inf_tex` v2 tiled
//! container plus a TOML sidecar for each map, and a `.inf_mat` plus sidecar
//! that names them. The island's four `TerrainLayer`s bind four of the five, and
//! any project scaffolded from a template can bind all five.
//!
//! # Why the GUIDs are constants
//!
//! Because a committed level names them. The island's `.inf_lvl` carries
//! `TerrainLayer::material` for four of these, so an id that moved would leave
//! the island's ground bound to nothing — the same reason the starter
//! character's ids are constants (`samples::starter_character_ids`) rather than
//! minted. They are laid out as one block so a sixth set appends rather than
//! renumbers.
//!
//! # The byte lock
//!
//! Everything here is a pure function of [`inf_material::ground`], which is a
//! pure function of nothing at all. `samples`'s lock test compares the bytes on
//! disk against a fresh generation on **every** CI leg — which is the whole
//! reason the synthesis is CPU-side and transcendental-free, and it is what
//! stops this library going stale the way a hand-exported texture would.

use std::path::Path;

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_material::ground::{albedo_settings, data_settings, GroundKind};
use inf_material::material::{MatBlend, MaterialAsset};
use uuid::Uuid;

/// The folder under `samples/` this library lives in.
pub const GROUND_FOLDER: &str = "ground";

/// The base of the ground library's GUID block.
///
/// Six ids per set (five used, one reserved) so a set's slots stay together and
/// a sixth ground appends at `+ 6 · 5`. **Frozen** — see the module note.
const GROUND_GUID_BASE: u128 = 0x9E20_0000;

/// The four `.inf_tex` slots and the `.inf_mat` one ground set is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroundIds {
    pub material: Uuid,
    pub albedo: Uuid,
    pub normal: Uuid,
    pub orm: Uuid,
    /// `Some` exactly when [`GroundKind::has_detail`].
    pub detail: Option<Uuid>,
}

/// The ids for one ground set. A pure function of the kind's position in
/// [`GroundKind::ALL`], which is frozen.
pub fn ground_ids(kind: GroundKind) -> GroundIds {
    let slot = GroundKind::ALL
        .iter()
        .position(|k| *k == kind)
        .expect("every kind is in ALL") as u128;
    let base = GROUND_GUID_BASE + slot * 6;
    GroundIds {
        material: Uuid::from_u128(base),
        albedo: Uuid::from_u128(base + 1),
        normal: Uuid::from_u128(base + 2),
        orm: Uuid::from_u128(base + 3),
        detail: kind.has_detail().then(|| Uuid::from_u128(base + 4)),
    }
}

/// The `.inf_mat` GUID a `TerrainLayer::material` binds for this ground.
pub fn ground_material_guid(kind: GroundKind) -> Uuid {
    ground_ids(kind).material
}

/// Every file this library writes, as basenames — the list a recipe's
/// `[content]` copies and the lock test walks.
pub fn ground_files() -> Vec<String> {
    let mut out = Vec::new();
    for kind in GroundKind::ALL {
        let stem = kind.stem();
        out.push(format!("{stem}.inf_mat"));
        out.push(format!("{stem}.inf_mat.toml"));
        for slot in ["Albedo", "Normal", "ORM"] {
            out.push(format!("{stem}_{slot}.inf_tex"));
            out.push(format!("{stem}_{slot}.inf_tex.toml"));
        }
        if kind.has_detail() {
            out.push(format!("{stem}_Detail.inf_tex"));
            out.push(format!("{stem}_Detail.inf_tex.toml"));
        }
    }
    out.sort();
    out
}

/// One asset's bytes and the name it is written under.
pub struct GroundFile {
    pub name: String,
    pub payload: Vec<u8>,
    pub sidecar: AssetSidecar,
}

/// **Generate the whole library in memory**, in a frozen order.
///
/// Split from the writing half so the lock test can compare bytes without
/// touching the tree, which is the arrangement the character lock already uses.
pub fn ground_library() -> Result<Vec<GroundFile>, String> {
    let mut out = Vec::new();
    for kind in GroundKind::ALL {
        let ids = ground_ids(kind);
        let maps = inf_material::ground::synthesize(kind);
        let stem = kind.stem();

        let mut tex =
            |slot: &str, guid: Uuid, rgba: Vec<u8>, extent: u32, settings| -> Result<(), String> {
                let image =
                    inf_material::tiles::build_tiled_texture(rgba, extent, extent, settings)
                        .map_err(|e| format!("{stem}_{slot}: {e}"))?;
                let payload = image.into_bytes();
                let name = format!("{stem}_{slot}.inf_tex");
                let mut sidecar =
                    AssetSidecar::new(AssetId(guid), AssetKind::Texture, ContentHash::of(&payload));
                sidecar.tags = vec!["ground".into(), kind.label().replace(' ', "-")];
                out.push(GroundFile {
                    name,
                    payload,
                    sidecar,
                });
                Ok(())
            };

        tex(
            "Albedo",
            ids.albedo,
            maps.albedo,
            inf_material::ground::GROUND_ALBEDO_EXTENT,
            albedo_settings(),
        )?;
        tex(
            "Normal",
            ids.normal,
            maps.normal,
            inf_material::ground::GROUND_MAP_EXTENT,
            data_settings(),
        )?;
        tex(
            "ORM",
            ids.orm,
            maps.orm,
            inf_material::ground::GROUND_MAP_EXTENT,
            data_settings(),
        )?;
        if let (Some(guid), Some(rgba)) = (ids.detail, maps.detail) {
            tex(
                "Detail",
                guid,
                rgba,
                inf_material::ground::GROUND_MAP_EXTENT,
                data_settings(),
            )?;
        }

        let mat = MaterialAsset {
            schema_version: MaterialAsset::CURRENT_VERSION,
            base_color: kind.base_color(),
            metallic: 0.0,
            roughness: kind.roughness(),
            emissive: [0.0; 3],
            base_color_texture: Some(AssetId(ids.albedo)),
            normal_texture: Some(AssetId(ids.normal)),
            metallic_roughness_texture: Some(AssetId(ids.orm)),
            blend: MatBlend::Opaque,
            alpha_cutoff: 0.5,
            detail_texture: ids.detail.map(AssetId),
            // Tiles per uv unit, NOT metres — see
            // `MaterialAsset::detail_scale_m`'s own note, which this wave wrote.
            detail_scale_m: kind.detail_scale(),
        };
        let payload = inf_asset::encode(&mat).map_err(|e| format!("{stem}.inf_mat: {e}"))?;
        let mut sidecar = AssetSidecar::new(
            AssetId(ids.material),
            AssetKind::Material,
            ContentHash::of(&payload),
        );
        sidecar.dependencies = mat.texture_dependencies();
        sidecar.tags = vec!["ground".into(), kind.label().replace(' ', "-")];
        out.push(GroundFile {
            name: format!("{stem}.inf_mat"),
            payload,
            sidecar,
        });
    }
    Ok(out)
}

/// Write the library into `dir`, creating it if needed.
pub fn write_ground_library(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    for f in ground_library()? {
        let path = dir.join(&f.name);
        inf_asset::write_atomically(&path, &f.payload)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        f.sidecar
            .save(&path)
            .map_err(|e| format!("sidecar {}: {e}", path.display()))?;
    }
    std::fs::write(dir.join("README.md"), README)
        .map_err(|e| format!("write ground README: {e}"))?;
    Ok(())
}

const README: &str = concat!(
    "# The engine's ground library (wave TER2a)\n",
    "\n",
    "Five PBR ground sets — grass, rock, forest floor, sand and soil — and the\n",
    "first `.inf_tex` files this repository has ever committed. Before TER2a the\n",
    "whole virtual-texture stack had no content that reached it and the 51 km²\n",
    "island's ground was one flat colour.\n",
    "\n",
    "Each set is a `.inf_mat` naming three or four `.inf_tex` v2 tiled\n",
    "containers: a 1 024² albedo, a 512² tangent-space normal, a 512² ORM, and —\n",
    "for grass and rock — a 512² high-frequency detail normal.\n",
    "\n",
    "| set | tiles every | albedo texel | detail tile |\n",
    "|---|---|---|---|\n",
    "| `Ground_Grass` | 2.0 m | 1.95 mm | 12.5 cm |\n",
    "| `Ground_Rock` | 3.0 m | 2.93 mm | 15.0 cm |\n",
    "| `Ground_ForestFloor` | 2.5 m | 2.44 mm | — |\n",
    "| `Ground_Sand` | 1.5 m | 1.46 mm | — |\n",
    "| `Ground_Soil` | 2.2 m | 2.15 mm | — |\n",
    "\n",
    "**Every map is BC1**, including the normals. That is a measurement, not a\n",
    "preference: `inf_render::build_vt_level` picks the atlas format from the\n",
    "stored formats of the textures a level binds, and a MIXED set demotes the\n",
    "whole pool to RGBA8 at eight times the page bytes. Wave T's `PageFormat::Bc5`\n",
    "is the right normal-map format and cannot be used beside a BC1 albedo until\n",
    "the atlas can hold two formats.\n",
    "\n",
    "Nothing here is hand-painted or imported. `inf_material::ground` synthesises\n",
    "every texel from an integer hash with no transcendental in the path, so the\n",
    "bytes are identical on every platform and the lock test below compares them\n",
    "on every CI leg. Regenerate with:\n",
    "\n",
    "```\n",
    "INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples\n",
    "```\n",
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every id in the library is distinct — a block-arithmetic slip would give
    /// two maps one GUID, and the second registration would silently replace
    /// the first.
    #[test]
    fn every_id_in_the_library_is_its_own() {
        let mut all: Vec<Uuid> = Vec::new();
        for kind in GroundKind::ALL {
            let i = ground_ids(kind);
            all.extend([i.material, i.albedo, i.normal, i.orm]);
            all.extend(i.detail);
        }
        let n = all.len();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), n, "the ground library shares a GUID with itself");
        assert_eq!(n, 5 * 4 + 2, "the library's asset count moved");
    }

    /// The library builds, and every material's sidecar names exactly the
    /// textures its material references — the edge the cook walks to drag a
    /// `.inf_tex` into a pack.
    #[test]
    fn every_material_declares_the_textures_it_names() {
        let files = ground_library().expect("the ground library builds");
        assert_eq!(files.len(), ground_files().len() / 2);
        let by_name: std::collections::BTreeMap<&str, &GroundFile> =
            files.iter().map(|f| (f.name.as_str(), f)).collect();
        for kind in GroundKind::ALL {
            let ids = ground_ids(kind);
            let name = format!("{}.inf_mat", kind.stem());
            let f = by_name.get(name.as_str()).expect("the material is written");
            let mat: MaterialAsset = inf_asset::decode(&f.payload).expect("the material decodes");
            assert_eq!(mat.base_color_texture, Some(AssetId(ids.albedo)));
            assert_eq!(mat.normal_texture, Some(AssetId(ids.normal)));
            assert_eq!(mat.metallic_roughness_texture, Some(AssetId(ids.orm)));
            assert_eq!(mat.detail_texture, ids.detail.map(AssetId));
            assert_eq!(
                f.sidecar.dependencies,
                mat.texture_dependencies(),
                "{name}'s sidecar and its payload name different textures"
            );
            // The detail pair is live or inert together — a texture with a zero
            // scale renders nothing and is the state `detail_scale_q8` encodes
            // as disabled.
            let derived = inf_material::derive_material(&mat);
            assert_eq!(
                derived.detail_scale_q8() > 0,
                kind.has_detail(),
                "{name}'s detail pair is half-authored"
            );
        }
    }

    /// **Every `.inf_tex` is a readable v2 container, and every one of them is
    /// BC1.** The second half is the one that matters: a single map in another
    /// format demotes the whole level's atlas to RGBA8 (8× the page bytes), so
    /// this is a property of the library rather than of each file.
    #[test]
    fn every_texture_is_a_readable_bc1_v2_container() {
        let files = ground_library().expect("the ground library builds");
        let mut textures = 0;
        let mut bytes = 0usize;
        for f in &files {
            if !f.name.ends_with(".inf_tex") {
                continue;
            }
            textures += 1;
            bytes += f.payload.len();
            let reader = inf_material::tiles::TiledTextureReader::new(f.payload.as_slice())
                .unwrap_or_else(|e| panic!("{} is not a v2 container: {e}", f.name));
            let format = reader.header().format;
            assert_eq!(
                format,
                inf_render::PageFormat::Bc1,
                "{} is {format:?}; a mixed-format level demotes its atlas to RGBA8",
                f.name
            );
            reader
                .vt_desc()
                .validate()
                .unwrap_or_else(|e| panic!("{} is not a registrable pyramid: {e}", f.name));
            // The sidecar's hash is of the bytes actually written.
            assert_eq!(f.sidecar.content_hash, ContentHash::of(&f.payload));
        }
        assert_eq!(textures, 17, "the library's texture count moved");
        println!(
            "GROUND LIBRARY: {textures} textures, {} materials, {:.2} MB committed",
            files.len() - textures,
            bytes as f64 / 1.0e6
        );
    }

    /// **The whole library fits its share of the pool.** The deterministic
    /// residency floor is a pure function of the registration set, and a floor
    /// that does not fit the budget is a level that cannot page its ground at
    /// all. Measured against the shipped 24 MiB / BC1 numbers.
    #[test]
    fn the_librarys_residency_floor_fits_the_shipped_budget() {
        let files = ground_library().expect("the ground library builds");
        let mut floor_pages = 0u64;
        for f in &files {
            if !f.name.ends_with(".inf_tex") {
                continue;
            }
            let reader = inf_material::tiles::TiledTextureReader::new(f.payload.as_slice())
                .expect("the container reads");
            let desc = reader.vt_desc();
            let coarsest = desc.coarsest_mip();
            let lowest = coarsest.saturating_sub(inf_render::VT_FLOOR_LEVELS - 1);
            for mip in lowest..=coarsest {
                let m = &desc.mips[mip as usize];
                floor_pages += u64::from(m.tiles_x * m.tiles_y);
            }
        }
        let stored = inf_render::STORED_TILE_SIZE;
        let budget = inf_render::DEFAULT_VT_BUDGET_BYTES;
        let slots = budget / inf_render::PageFormat::Bc1.page_bytes(stored);
        let rgba8_slots = budget / inf_render::PageFormat::Rgba8.page_bytes(stored);
        println!(
            "GROUND FLOOR: {floor_pages} pages against {slots} BC1 slots \
             ({:.1} % of the 24 MiB pool); the same content in a demoted RGBA8 \
             pool would have {rgba8_slots} slots",
            floor_pages as f64 / slots as f64 * 100.0
        );
        assert!(
            floor_pages < slots / 4,
            "the ground library's floor is {floor_pages} pages of a {slots}-slot \
             pool — there would be nothing left for anything else in the level"
        );
    }
}
