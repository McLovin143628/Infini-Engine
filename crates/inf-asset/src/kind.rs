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
    /// 2D texture (`.inf_tex`).
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
    /// Anything else living under the content root.
    Unknown,
}

impl AssetKind {
    /// The canonical file extension (without the dot), or `None` for `Unknown`.
    pub fn extension(self) -> Option<&'static str> {
        Some(match self {
            AssetKind::Level => "inf_lvl",
            AssetKind::Mesh => "inf_mesh",
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
            AssetKind::Unknown => return None,
        })
    }

    /// Map a file extension (case-insensitive, no dot) to a kind.
    pub fn from_extension(ext: &str) -> AssetKind {
        match ext.to_ascii_lowercase().as_str() {
            "inf_lvl" => AssetKind::Level,
            "inf_mesh" => AssetKind::Mesh,
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
            AssetKind::Unknown => "unknown",
        }
    }

    /// Human-friendly display label.
    pub fn label(self) -> &'static str {
        match self {
            AssetKind::Level => "Level",
            AssetKind::Mesh => "Static Mesh",
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
            AssetKind::Unknown => "File",
        }
    }

    /// Every kind that represents an editable `.inf_*` asset (excludes
    /// `Unknown`), for enumerating type chips and "create new" menus.
    pub fn all() -> &'static [AssetKind] {
        &[
            AssetKind::Level,
            AssetKind::Mesh,
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
        ]
    }
}

/// Source (importable) file extensions we know how to turn into assets, mapped
/// to the [`AssetKind`] they produce as their primary output.
pub fn importable_source_kind(ext: &str) -> Option<AssetKind> {
    match ext.to_ascii_lowercase().as_str() {
        "gltf" | "glb" => Some(AssetKind::Mesh),
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
        assert_eq!(importable_source_kind("png"), Some(AssetKind::Texture));
        assert_eq!(importable_source_kind("csv"), Some(AssetKind::Table));
        assert_eq!(importable_source_kind("xyz"), None);
    }
}
