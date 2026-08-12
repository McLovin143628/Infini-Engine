//! The asset taxonomy: every `.inf_*` type the database understands.
//!
//! [`AssetKind`] is the discriminator that drives the Content Drawer type
//! column/filters, thumbnail rendering, and importer routing. The `.inf_*`
//! extension ↔ kind mapping is the single source of truth for both directions.

use serde::{Deserialize, Serialize};

/// A recognized asset type. `Unknown` covers files under the content root that
/// aren't (yet) one of our formats — surfaced but not editable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    /// Level / scene (`.inf_lvl`).
    Level,
    /// Static or skinned mesh (`.inf_mesh`).
    Mesh,
    /// Meshlet LOD DAG for virtualized geometry (`.inf_vmesh`) — derived at cook
    /// time from a [`Mesh`](AssetKind::Mesh) (P13.1).
    MeshletMesh,
    /// 2D texture (`.inf_tex`) — since P26.1 a **tiled** container: a header +
    /// mip/tile directories + 16-byte-aligned 136² tile blobs, the unit streaming
    /// virtual texturing pages. A **streaming-class** kind for the same reason
    /// [`Terrain`](AssetKind::Terrain) is (`PackWriter::compresses_kind`).
    /// v1 (bincode `inf_material::TextureAsset`) payloads keep loading forever,
    /// sniffed on the magic by `TextureAsset::from_payload`.
    Texture,
    /// Material graph (`.inf_mat`).
    Material,
    /// Material instance overriding a parent material's parameters (`.inf_mati`).
    MaterialInstance,
    /// Actor assembly / blueprint class (`.inf_act`).
    Blueprint,
    /// Function library (`.inf_fn`).
    FunctionLib,
    /// Strongly-typed struct definition (`.inf_struct`).
    Struct,
    /// Strongly-typed enum definition (`.inf_enum`).
    Enum,
    /// Tabular data (`.inf_table`).
    Table,
    /// Audio clip (`.inf_audio`).
    Audio,
    /// Procedural-content-generation graph (`.inf_pcg`) — scatter rules + samplers.
    Pcg,
    /// Skinning skeleton (`.inf_skel`) — a joint hierarchy + inverse binds (P11.1).
    Skeleton,
    /// Skeletal animation clip (`.inf_anim`) — per-joint keyframe tracks (P11.1).
    AnimClip,
    /// Animation state machine (`.inf_sm`) — states + transitions + blend spaces
    /// (P11.2).
    StateMachine,
    /// Streamable terrain tiles (`.inf_terrain`) — a header + tile directory +
    /// 16-byte-aligned per-tile blobs across an LOD pyramid (P16.3). A
    /// **streaming-class** kind: it cooks uncompressed so a runtime can page
    /// individual tiles straight out of an mmap'd pack
    /// (`PackWriter::compresses_kind`).
    Terrain,
    /// A level's cook-derived **world partition** (`.inf_part`) — a header + cell
    /// directory + 16-byte-aligned per-cell entity-record blobs (P16.5). Like
    /// [`MeshletMesh`](AssetKind::MeshletMesh) it is *derived at cook time*, never
    /// authored: its GUID is a pure function of its level's
    /// (`inf_packager::derived_partition_id`), so the runtime finds it without an
    /// index. A **streaming-class** kind: it cooks uncompressed so the player can
    /// slice one cell straight out of an mmap'd pack
    /// (`PackWriter::compresses_kind`).
    Partition,
    /// A level's named **biomes** (`.inf_biomes`) — id, display colour, splat
    /// mapping, PCG-graph reference and water/structure hints per biome (P19.2,
    /// [`inf_terrain::BiomeSet`]). Small and text-like, so it compresses like
    /// every other authored payload (`PackWriter::compresses_kind`).
    ///
    /// [`inf_terrain::BiomeSet`]: the terrain crate's `BiomeSet` payload.
    BiomeSet,
    /// A **sparse SDF voxel volume** (`.inf_voxel`) — a header + chunk directory +
    /// 16-byte-aligned per-chunk blobs, holding the caves/tunnels/excavations that
    /// locally extend the heightfield terrain (P21.1,
    /// [`inf_voxel::VoxelAsset`]). Authored, not derived: an author places a
    /// volume and carves it, exactly as they author a `.inf_terrain`.
    ///
    /// A **streaming-class** kind, for the same reason [`Terrain`](AssetKind::Terrain)
    /// is: it cooks uncompressed so a runtime can page individual chunks straight
    /// out of an mmap'd pack as borrowed slices (`PackWriter::compresses_kind`).
    ///
    /// [`inf_voxel::VoxelAsset`]: the voxel crate's `.inf_voxel` container.
    VoxelVolume,
    /// A mesh's **pre-fractured chunk set** (`.inf_fracture`) — the Voronoi
    /// chunk hierarchy P22.3 swaps in when the asset breaks (P22.2,
    /// [`inf_mesh::fracture::FractureAsset`]). Like
    /// [`MeshletMesh`](AssetKind::MeshletMesh) it is *derived at cook time*,
    /// never authored: its GUID is a pure function of its mesh's
    /// (`inf_mesh::fracture::derived_fracture_id`), so a runtime finds it
    /// without an index.
    ///
    /// **Not** a streaming-class kind, unlike the other derived containers: a
    /// fracture is loaded *whole*, at the instant one asset breaks, and there is
    /// no useful partial residency (a chunk set with half its chunks is a hole).
    /// So it compresses like every other bincode payload — see
    /// `PackWriter::compresses_kind`.
    ///
    /// [`inf_mesh::fracture::FractureAsset`]: the mesh crate's `.inf_fracture`
    /// payload.
    Fracture,
    /// A **simulated garment** (`.inf_cloth`) — the XPBD particle set, constraint
    /// lists, material and collision capsules a [`ClothSim`] component wears
    /// (P24.4, [`inf_anim::cloth::ClothAsset`]).
    ///
    /// Authored (derived from a garment `.inf_mesh` by the Model Editor's cloth
    /// door, then saved), not cook-derived: an author tunes stiffness and pins
    /// hems, and re-deriving those at cook would throw the tuning away.
    ///
    /// **Not** a streaming-class kind: a garment is loaded whole, at the instant
    /// its wearer spawns, and half a constraint set is not a cheaper load but a
    /// coat that tears. So it compresses like every other bincode payload — see
    /// `PackWriter::compresses_kind`.
    ///
    /// [`ClothSim`]: the ECS component (scene v21).
    /// [`inf_anim::cloth::ClothAsset`]: the animation crate's `.inf_cloth` payload.
    Cloth,
    /// **Strand hair** (`.inf_hair`) — guide strands rooted on a scalp mesh, their
    /// segment lists, ribbon parameters and collision capsules (P24.4,
    /// [`inf_anim::hair::HairAsset`]).
    ///
    /// The [`Cloth`](AssetKind::Cloth) shape for the [`Cloth`](AssetKind::Cloth)
    /// reasons: authored, loaded whole, compressed.
    ///
    /// [`inf_anim::hair::HairAsset`]: the animation crate's `.inf_hair` payload.
    Hair,
    /// A `.inf_mat` **flattened for a runtime** (`.inf_matd`) — three texture
    /// GUIDs plus the scalars that are the no-texture fallback (P26.3b,
    /// [`crate::DerivedMaterial`]).
    ///
    /// Like [`MeshletMesh`](AssetKind::MeshletMesh) and
    /// [`Fracture`](AssetKind::Fracture) it is *derived at cook time*, never
    /// authored: its GUID is a pure function of its material's
    /// ([`crate::derived_material_id`]), so a runtime finds it without an index.
    ///
    /// **It exists because a shipped player cannot read a `.inf_mat`** — the
    /// P26.2 dependency inversion, which keeps `image` and the naga-validating
    /// material compiler out of the player, also keeps `MaterialAsset` out. The
    /// answer crosses instead of the question.
    ///
    /// **Not** a streaming-class kind: it is a few dozen bytes read whole when a
    /// level loads, and there is nothing to page. So it compresses like every
    /// other bincode payload — see `PackWriter::compresses_kind`.
    DerivedMaterial,
    /// Anything else living under the content root.
    Unknown,
}

impl AssetKind {
    /// The canonical file extension (without the dot), or `None` for `Unknown`.
    pub fn extension(self) -> Option<&'static str> {
        Some(match self {
            AssetKind::Level => "inf_lvl",
            AssetKind::Mesh => "inf_mesh",
            AssetKind::MeshletMesh => "inf_vmesh",
            AssetKind::Texture => "inf_tex",
            AssetKind::Material => "inf_mat",
            AssetKind::MaterialInstance => "inf_mati",
            AssetKind::Blueprint => "inf_act",
            AssetKind::FunctionLib => "inf_fn",
            AssetKind::Struct => "inf_struct",
            AssetKind::Enum => "inf_enum",
            AssetKind::Table => "inf_table",
            AssetKind::Audio => "inf_audio",
            AssetKind::Pcg => "inf_pcg",
            AssetKind::Skeleton => "inf_skel",
            AssetKind::AnimClip => "inf_anim",
            AssetKind::StateMachine => "inf_sm",
            AssetKind::Terrain => "inf_terrain",
            AssetKind::Partition => "inf_part",
            AssetKind::BiomeSet => "inf_biomes",
            AssetKind::VoxelVolume => "inf_voxel",
            AssetKind::Fracture => "inf_fracture",
            AssetKind::Cloth => "inf_cloth",
            AssetKind::Hair => "inf_hair",
            AssetKind::DerivedMaterial => "inf_matd",
            AssetKind::Unknown => return None,
        })
    }

    /// Map a file extension (case-insensitive, no dot) to a kind.
    pub fn from_extension(ext: &str) -> AssetKind {
        match ext.to_ascii_lowercase().as_str() {
            "inf_lvl" => AssetKind::Level,
            "inf_mesh" => AssetKind::Mesh,
            "inf_vmesh" => AssetKind::MeshletMesh,
            "inf_tex" => AssetKind::Texture,
            "inf_mati" => AssetKind::MaterialInstance,
            "inf_mat" => AssetKind::Material,
            "inf_act" => AssetKind::Blueprint,
            "inf_fn" => AssetKind::FunctionLib,
            "inf_struct" => AssetKind::Struct,
            "inf_enum" => AssetKind::Enum,
            "inf_table" => AssetKind::Table,
            "inf_audio" => AssetKind::Audio,
            "inf_pcg" => AssetKind::Pcg,
            "inf_skel" => AssetKind::Skeleton,
            "inf_anim" => AssetKind::AnimClip,
            "inf_sm" => AssetKind::StateMachine,
            "inf_terrain" => AssetKind::Terrain,
            "inf_part" => AssetKind::Partition,
            "inf_biomes" => AssetKind::BiomeSet,
            "inf_voxel" => AssetKind::VoxelVolume,
            "inf_fracture" => AssetKind::Fracture,
            "inf_cloth" => AssetKind::Cloth,
            "inf_hair" => AssetKind::Hair,
            "inf_matd" => AssetKind::DerivedMaterial,
            _ => AssetKind::Unknown,
        }
    }

    /// Classify a path by its extension.
    pub fn from_path(path: &std::path::Path) -> AssetKind {
        path.extension()
            .and_then(|e| e.to_str())
            .map(AssetKind::from_extension)
            .unwrap_or(AssetKind::Unknown)
    }

    /// A stable lowercase slug for UI/filters (`"mesh"`, `"texture"`, …).
    pub fn slug(self) -> &'static str {
        match self {
            AssetKind::Level => "level",
            AssetKind::Mesh => "mesh",
            AssetKind::MeshletMesh => "meshlet_mesh",
            AssetKind::Texture => "texture",
            AssetKind::Material => "material",
            AssetKind::MaterialInstance => "material_instance",
            AssetKind::Blueprint => "blueprint",
            AssetKind::FunctionLib => "function",
            AssetKind::Struct => "struct",
            AssetKind::Enum => "enum",
            AssetKind::Table => "table",
            AssetKind::Audio => "audio",
            AssetKind::Pcg => "pcg",
            AssetKind::Skeleton => "skeleton",
            AssetKind::AnimClip => "anim_clip",
            AssetKind::StateMachine => "state_machine",
            AssetKind::Terrain => "terrain",
            AssetKind::Partition => "partition",
            AssetKind::BiomeSet => "biome_set",
            AssetKind::VoxelVolume => "voxel_volume",
            AssetKind::Fracture => "fracture",
            AssetKind::Cloth => "cloth",
            AssetKind::Hair => "hair",
            AssetKind::DerivedMaterial => "derived_material",
            AssetKind::Unknown => "unknown",
        }
    }

    /// Human-friendly display label.
    pub fn label(self) -> &'static str {
        match self {
            AssetKind::Level => "Level",
            AssetKind::Mesh => "Static Mesh",
            AssetKind::MeshletMesh => "Meshlet Mesh",
            AssetKind::Texture => "Texture",
            AssetKind::Material => "Material",
            AssetKind::MaterialInstance => "Material Instance",
            AssetKind::Blueprint => "Blueprint",
            AssetKind::FunctionLib => "Function Library",
            AssetKind::Struct => "Struct",
            AssetKind::Enum => "Enum",
            AssetKind::Table => "Data Table",
            AssetKind::Audio => "Audio",
            AssetKind::Pcg => "PCG Graph",
            AssetKind::Skeleton => "Skeleton",
            AssetKind::AnimClip => "Animation",
            AssetKind::StateMachine => "State Machine",
            AssetKind::Terrain => "Terrain",
            AssetKind::Partition => "World Partition",
            AssetKind::BiomeSet => "Biome Set",
            AssetKind::VoxelVolume => "Voxel Volume",
            AssetKind::Fracture => "Fracture",
            AssetKind::Cloth => "Cloth",
            AssetKind::Hair => "Hair",
            AssetKind::DerivedMaterial => "Derived Material",
            AssetKind::Unknown => "File",
        }
    }

    /// Every kind that represents an editable `.inf_*` asset (excludes
    /// `Unknown`), for enumerating type chips and "create new" menus.
    pub fn all() -> &'static [AssetKind] {
        &[
            AssetKind::Level,
            AssetKind::Mesh,
            AssetKind::MeshletMesh,
            AssetKind::Texture,
            AssetKind::Material,
            AssetKind::MaterialInstance,
            AssetKind::Blueprint,
            AssetKind::FunctionLib,
            AssetKind::Struct,
            AssetKind::Enum,
            AssetKind::Table,
            AssetKind::Audio,
            AssetKind::Pcg,
            AssetKind::Skeleton,
            AssetKind::AnimClip,
            AssetKind::StateMachine,
            AssetKind::Terrain,
            AssetKind::Partition,
            AssetKind::BiomeSet,
            AssetKind::VoxelVolume,
            AssetKind::Fracture,
            AssetKind::Cloth,
            AssetKind::Hair,
            AssetKind::DerivedMaterial,
        ]
    }
}

/// Source (importable) file extensions we know how to turn into assets, mapped
/// to the [`AssetKind`] they produce as their primary output.
pub fn importable_source_kind(ext: &str) -> Option<AssetKind> {
    match ext.to_ascii_lowercase().as_str() {
        "gltf" | "glb" | "obj" => Some(AssetKind::Mesh),
        "png" | "jpg" | "jpeg" | "tga" | "bmp" | "hdr" | "exr" => Some(AssetKind::Texture),
        "wav" | "ogg" | "mp3" | "flac" => Some(AssetKind::Audio),
        "csv" | "json" => Some(AssetKind::Table),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// **THE WIRE FREEZE** (P24.4): every kind's serde token, extension and slug,
    /// written down once.
    ///
    /// [`AssetKind`] is `Serialize`/`Deserialize` and rides **two** persisted
    /// surfaces: the TOML sidecar every asset carries next to it
    /// ([`crate::AssetSidecar`]) and the `.inf_pack` index a shipped build reads
    /// ([`crate::PackReader`]). So a variant renamed — or reordered under a codec
    /// that is positional — is a content-compatibility break in files that are
    /// already on users' disks, and nothing else in the tree would notice.
    ///
    /// The rule this pins is therefore **append-only**: a new kind goes at the
    /// tail of the enum and gets a new row here; an existing row never moves and
    /// never changes. P24.4 appended `Cloth` and `Hair` under exactly that rule.
    ///
    /// It is a table of *literals* on purpose. Deriving the expected token from
    /// the variant (`format!("{k:?}").to_snake_case()`) would agree with any
    /// rename by construction, which is the shape of gate this repository keeps
    /// having to repair.
    const FROZEN_WIRE: [(AssetKind, &str, &str, &str); 24] = [
        (AssetKind::Level, "level", "inf_lvl", "level"),
        (AssetKind::Mesh, "mesh", "inf_mesh", "mesh"),
        (
            AssetKind::MeshletMesh,
            "meshlet_mesh",
            "inf_vmesh",
            "meshlet_mesh",
        ),
        (AssetKind::Texture, "texture", "inf_tex", "texture"),
        (AssetKind::Material, "material", "inf_mat", "material"),
        (
            AssetKind::MaterialInstance,
            "material_instance",
            "inf_mati",
            "material_instance",
        ),
        (AssetKind::Blueprint, "blueprint", "inf_act", "blueprint"),
        (AssetKind::FunctionLib, "function_lib", "inf_fn", "function"),
        (AssetKind::Struct, "struct", "inf_struct", "struct"),
        (AssetKind::Enum, "enum", "inf_enum", "enum"),
        (AssetKind::Table, "table", "inf_table", "table"),
        (AssetKind::Audio, "audio", "inf_audio", "audio"),
        (AssetKind::Pcg, "pcg", "inf_pcg", "pcg"),
        (AssetKind::Skeleton, "skeleton", "inf_skel", "skeleton"),
        (AssetKind::AnimClip, "anim_clip", "inf_anim", "anim_clip"),
        (
            AssetKind::StateMachine,
            "state_machine",
            "inf_sm",
            "state_machine",
        ),
        (AssetKind::Terrain, "terrain", "inf_terrain", "terrain"),
        (AssetKind::Partition, "partition", "inf_part", "partition"),
        (AssetKind::BiomeSet, "biome_set", "inf_biomes", "biome_set"),
        (
            AssetKind::VoxelVolume,
            "voxel_volume",
            "inf_voxel",
            "voxel_volume",
        ),
        (AssetKind::Fracture, "fracture", "inf_fracture", "fracture"),
        // ── P24.4 append ────────────────────────────────────────────────────
        (AssetKind::Cloth, "cloth", "inf_cloth", "cloth"),
        (AssetKind::Hair, "hair", "inf_hair", "hair"),
        // ── P26.3b append ───────────────────────────────────────────────────
        (
            AssetKind::DerivedMaterial,
            "derived_material",
            "inf_matd",
            "derived_material",
        ),
    ];

    #[test]
    fn the_wire_tokens_are_frozen_and_append_only() {
        // `all()` is what the Content Drawer enumerates, so the table and the
        // list must be the same set in the same order — otherwise a kind could be
        // appended to one and not the other and this test would still pass.
        assert_eq!(
            AssetKind::all().len(),
            FROZEN_WIRE.len(),
            "a kind was added to `all()` without a frozen wire row (or the other \
             way round)"
        );
        for (i, (kind, token, ext, slug)) in FROZEN_WIRE.into_iter().enumerate() {
            assert_eq!(AssetKind::all()[i], kind, "row {i} moved");
            assert_eq!(kind.extension(), Some(ext), "{kind:?}");
            assert_eq!(AssetKind::from_extension(ext), kind, "{kind:?}");
            assert_eq!(kind.slug(), slug, "{kind:?}");
            // The serde token is what a sidecar on a user's disk actually holds.
            let json = serde_json::to_string(&kind).expect("kinds serialize");
            assert_eq!(json, format!("\"{token}\""), "{kind:?} serde token moved");
            let back: AssetKind = serde_json::from_str(&json).expect("and decode");
            assert_eq!(back, kind);
        }
        // ANTI-VACUITY: the table would notice a rename. `Unknown` is deliberately
        // absent above (it has no extension), so it is checked separately here
        // rather than being the one variant nothing looks at.
        assert_eq!(AssetKind::Unknown.extension(), None);
        assert_eq!(
            serde_json::to_string(&AssetKind::Unknown).unwrap(),
            "\"unknown\""
        );
        assert!(
            FROZEN_WIRE.iter().all(|(_, t, _, _)| *t != "unknown"),
            "`Unknown` must not be in the editable-kind table"
        );
    }

    #[test]
    fn extension_round_trips_for_all_known_kinds() {
        for &k in AssetKind::all() {
            let ext = k.extension().expect("known kinds have extensions");
            assert_eq!(AssetKind::from_extension(ext), k, "{k:?}");
            // Case-insensitive.
            assert_eq!(AssetKind::from_extension(&ext.to_uppercase()), k, "{k:?}");
        }
    }

    #[test]
    fn from_path_classifies_by_extension() {
        assert_eq!(
            AssetKind::from_path(Path::new("a/b/Hero.inf_mesh")),
            AssetKind::Mesh
        );
        assert_eq!(
            AssetKind::from_path(Path::new("notes.txt")),
            AssetKind::Unknown
        );
    }

    #[test]
    fn source_extensions_route_to_kinds() {
        assert_eq!(importable_source_kind("GLB"), Some(AssetKind::Mesh));
        assert_eq!(importable_source_kind("obj"), Some(AssetKind::Mesh));
        assert_eq!(importable_source_kind("OBJ"), Some(AssetKind::Mesh));
        assert_eq!(importable_source_kind("png"), Some(AssetKind::Texture));
        assert_eq!(importable_source_kind("csv"), Some(AssetKind::Table));
        assert_eq!(importable_source_kind("xyz"), None);
    }
}
