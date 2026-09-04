//! **The committed ground library** (wave TER2a, clause 3) — five PBR ground
//! sets written into `samples/ground/` as engine content, and since wave
//! ASSET0 a sixth: `Road_Asphalt`, the surface the island's road mesh binds.
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
use inf_material::ground::{albedo_settings, data_settings, normal_settings, GroundKind};
use inf_material::material::{MatBlend, MaterialAsset};
use uuid::Uuid;

/// The folder under `samples/` this library lives in.
pub const GROUND_FOLDER: &str = "ground";

/// The base of the ground library's GUID block.
///
/// Six ids per set (five used, one reserved) so a set's slots stay together.
/// **Frozen** — see the module note.
const GROUND_GUID_BASE: u128 = 0x9E20_0000;

/// How many sets the original block holds before [`COVER_GUID_BASE`] takes over.
///
/// [`GroundKind::ALL`] is `#[non_exhaustive]` in spirit — a set appends — and the
/// comment above this line used to say a sixth ground appends at `+ 6 · 5`.
/// **It does not, and wave ASSET0 found out by doing it.** `cover.rs` minted
/// `COVER_GUID_BASE = 0x9E20_0000 + 5 * 6` — the very next id — for the three
/// scatter meshes in the same folder, and those three are named by a committed
/// `.inf_pcg`, so they cannot move. Appending asphalt at the stated place gave
/// `Road_Asphalt.inf_mat` the same GUID as `Cover_GrassTuft.inf_mesh`, and the
/// second registration would have silently replaced the first.
///
/// `the_three_kinds_are_three_of_everything` in `cover.rs` is the arm that said
/// so, which is why it is written over `GroundKind::ALL` rather than over a
/// literal five.
const GROUND_ORIGINAL_SETS: u128 = 5;

/// Where the ground block CONTINUES, past the cover library's three ids.
///
/// A second base rather than a moved first one: the five original sets are
/// named by a committed `.inf_lvl`'s `TerrainLayer::material` and the three
/// cover meshes by a committed `.inf_pcg`, so the only free direction is
/// forward. Six ids per set here too, so a seventh set appends at `+ 6`.
/// **Frozen** — the island's `Roads` entity names the first material in it.
const GROUND_CONT_BASE: u128 = 0x9E21_0000;

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
    let base = if slot < GROUND_ORIGINAL_SETS {
        GROUND_GUID_BASE + slot * 6
    } else {
        GROUND_CONT_BASE + (slot - GROUND_ORIGINAL_SETS) * 6
    };
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
        // **BC5 since wave IASSET2** — the one slot whose format moved, and the
        // only reason it could is that a residency now holds one arm per stored
        // format. See `inf_material::ground`'s module table: BC1's worst error
        // on a normal's two channels is 122 of 255 on this content.
        tex(
            "Normal",
            ids.normal,
            maps.normal,
            inf_material::ground::GROUND_MAP_EXTENT,
            normal_settings(),
        )?;
        tex(
            "ORM",
            ids.orm,
            maps.orm,
            inf_material::ground::GROUND_MAP_EXTENT,
            data_settings(),
        )?;
        if let (Some(guid), Some(rgba)) = (ids.detail, maps.detail) {
            // A detail map is a high-frequency NORMAL (plus a roughness lane),
            // so it takes the normal route for the same measured reason.
            tex(
                "Detail",
                guid,
                rgba,
                inf_material::ground::GROUND_MAP_EXTENT,
                normal_settings(),
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
    "**Wave ASSET0 appended a sixth, `Road_Asphalt`** - not a splat layer\n",
    "but the same kind of object, and for the same reason one level up: the\n",
    "island's `Roads` entity carried a `MeshRef` and no `Material` at\n",
    "all, so the street the editor opens standing on drew\n",
    "`Material::default().base_color` - 0.8 linear, the engine's debug\n",
    "grey, in both hosts.\n",
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
    "| `Road_Asphalt` | 4.0 m | 3.91 mm | 15.4 cm |\n",
    "\n",
    "**Albedo and ORM are BC1; normals and detail maps are BC5** (wave\n",
    "IASSET2). Every map used to be BC1, including the normals, and that was a\n",
    "limitation rather than a choice: one atlas held one format, so a MIXED set\n",
    "demoted the whole pool to RGBA8 at eight times the page bytes. A residency\n",
    "now holds one arm per stored format, and the three slots were then measured\n",
    "on these exact bytes (worst error per channel, out of 255):\n",
    "\n",
    "| slot | format | BC1 | the alternative |\n",
    "|---|---|---|---|\n",
    "| normal, detail | **BC5** | worst **122** | BC5 worst **17** — taken |\n",
    "| albedo | BC1 | worst 11 | BC7 worst 5, at twice the page — declined |\n",
    "| ORM | BC1 | worst 45 | BC7 worst 33, at twice the page — declined |\n",
    "\n",
    "A BC1 normal map on this content has texels whose X or Y is off by 122 of\n",
    "255: the surface normal points somewhere else. The albedo's four per cent is\n",
    "not worth halving what the atlas holds, which is what a 16-byte block costs.\n",
    "\n",
    "## And the three things that stand on it\n",
    "\n",
    "`Cover_GrassTuft`, `Cover_Shrub` and `Cover_Rock` are the meshes the\n",
    "islands `.inf_pcg` scatters (wave TER2a, clause 5) — 32, 20 and 128\n",
    "triangles, generated by `inf_editor_core::cover`. All three scatter kinds\n",
    "carried `mesh: None` before it, which the scatter evaluated, the biome\n",
    "binding restricted and the frame counted, and which drew nothing. Each\n",
    "shares one of the ground materials above rather than carrying textures of\n",
    "its own, which is the storage argument virtual texturing rests on.\n",
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
        assert_eq!(n, 6 * 4 + 3, "the library's asset count moved");
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

    /// **Every `.inf_tex` is a readable v2 container, and each slot is in the
    /// format its measurement chose** (wave IASSET2).
    ///
    /// This arm used to assert *"every one of them is BC1"*, and the reason it
    /// could was a limitation rather than a property: a single map in another
    /// format demoted the whole level's atlas to RGBA8 at 8× the page bytes.
    /// Wave IASSET2 gave a residency one arm per stored format, so the library
    /// may now be mixed — and it is, in exactly one slot:
    ///
    /// * **normal and detail → BC5.** BC1's worst per-channel error on this
    ///   content's two normal channels is 122 of 255; BC5's is 17.
    /// * **albedo and ORM → BC1**, unchanged and measured: BC7 would halve the
    ///   pages a 24 MiB arm holds to take the albedo's worst from 11 to 5 and
    ///   the ORM's from 45 to 33.
    ///
    /// So the assertion is per SLOT, and the count of formats present is
    /// asserted too — the library must be two arms and not three, because a
    /// third would take a third of the atlas budget for one map kind.
    #[test]
    fn every_texture_is_a_readable_v2_container_in_its_measured_format() {
        let files = ground_library().expect("the ground library builds");
        let mut textures = 0;
        let mut bytes = 0usize;
        let mut formats: Vec<inf_render::PageFormat> = Vec::new();
        for f in &files {
            if !f.name.ends_with(".inf_tex") {
                continue;
            }
            textures += 1;
            bytes += f.payload.len();
            let reader = inf_material::tiles::TiledTextureReader::new(f.payload.as_slice())
                .unwrap_or_else(|e| panic!("{} is not a v2 container: {e}", f.name));
            let format = reader.header().format;
            let want = if f.name.contains("_Normal") || f.name.contains("_Detail") {
                inf_render::PageFormat::Bc5
            } else {
                inf_render::PageFormat::Bc1
            };
            assert_eq!(
                format, want,
                "{} is {format:?} and its slot's measurement chose {want:?}",
                f.name
            );
            if !formats.contains(&format) {
                formats.push(format);
            }
            reader
                .vt_desc()
                .validate()
                .unwrap_or_else(|e| panic!("{} is not a registrable pyramid: {e}", f.name));
            // The sidecar's hash is of the bytes actually written.
            assert_eq!(f.sidecar.content_hash, ContentHash::of(&f.payload));
        }
        assert_eq!(textures, 21, "the library's texture count moved");
        assert_eq!(
            formats.len(),
            2,
            "the library binds {formats:?}; each distinct format is an atlas arm \
             and there are only {} of them",
            inf_render::vt::VT_MAX_POOLS
        );
        println!(
            "GROUND LIBRARY: {textures} textures in {:?}, {} materials, {:.2} MB committed",
            formats,
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
