//! Component registration (P3.1.1).
//!
//! One place registers every reflected component + value type into a
//! `bevy_reflect::TypeRegistry` and records which components are user-editable
//! (shown in the Details grid / "Add Component" menu). The Details reflection
//! walker (P3.3) and undo (P3.4) look components up here by their stable reflect
//! `type_path`, so this list is the single source of truth for "what can the
//! editor see and change".

use bevy_ecs::reflect::ReflectComponent;
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::{TypePath, TypeRegistry};

use crate::components::{
    AnimPlayer, AnimStateMachine, AtlasRect, AudioListener, AudioSource, BillboardMode, BlendMode,
    BodyKind2D, BodyKind3D, Camera, CharacterController2D, CharacterController3D, Collider2D,
    Collider3D, ColliderShape2DKind, ColliderShape3DKind, CombineRule, Decal, DistanceModel,
    Foliage, FoliagePaletteEntry, Joint2D, Joint3D, JointKind2D, JointKind3D, Light, Light2D,
    LightKind, Material, MeshRef, Name, NineSlice, PcgVolume, Primitive, RigidBody2D, RigidBody3D,
    SkeletalMesh, Spline, SplineInterp, Sprite, Terrain, Text2D, TextAlign, Tilemap, Transform,
    Visibility, Volume, VolumeKind,
};
use crate::math::{Color, Vec2d, Vec3d};

/// Metadata for one editable component type.
#[derive(Clone, Copy, Debug)]
pub struct ComponentInfo {
    /// Short display name (Details section header, Add-Component menu).
    pub display: &'static str,
    /// Stable reflect type path — the IPC key for this component.
    pub type_path: &'static str,
}

/// The engine's reflected type registry plus the editable-component list.
pub struct ComponentRegistry {
    types: TypeRegistry,
    editable: Vec<ComponentInfo>,
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentRegistry {
    /// Build the registry, registering every core component + value type.
    pub fn new() -> Self {
        let mut types = TypeRegistry::new();
        let mut editable = Vec::new();

        // Nested value / enum types: registered so reflect + serde can resolve
        // them, but never listed as editable components on their own.
        types.register::<Vec3d>();
        types.register::<Vec2d>();
        types.register::<Color>();
        // These value types derive `Reflect` without `#[reflect(Default)]` (their
        // `Default` lives in `math.rs`). E-P1 list editing rebuilds a collection
        // from the element type's `ReflectDefault`, so register that type-data
        // here (without touching `math.rs`) for the value types that can be list
        // elements — e.g. `Spline::points: Vec<Vec3d>`.
        types.register_type_data::<Vec3d, ReflectDefault>();
        types.register_type_data::<Vec2d, ReflectDefault>();
        types.register_type_data::<Color, ReflectDefault>();
        types.register::<Primitive>();
        types.register::<LightKind>();
        types.register::<TextAlign>();
        types.register::<BodyKind2D>();
        types.register::<ColliderShape2DKind>();
        types.register::<BodyKind3D>();
        types.register::<ColliderShape3DKind>();
        types.register::<CombineRule>();
        types.register::<JointKind2D>();
        types.register::<JointKind3D>();
        types.register::<BillboardMode>();
        types.register::<AtlasRect>();
        types.register::<DistanceModel>();
        types.register::<BlendMode>();
        types.register::<VolumeKind>();
        types.register::<SplineInterp>();
        types.register::<FoliagePaletteEntry>();
        // Opaque entity reference (E-P1) — joint `other` fields; surfaced by the
        // Details walker as an entity-picker widget.
        types.register::<crate::refs::EntityRef>();
        types.register::<String>();

        // `Name` is reflected (for completeness) but edited via a dedicated
        // rename command, so it is not in the generic Details grid.
        types.register::<Name>();

        macro_rules! editable {
            ($( $t:ty => $name:literal ),+ $(,)?) => {
                $(
                    types.register::<$t>();
                    editable.push(ComponentInfo {
                        display: $name,
                        type_path: <$t as TypePath>::type_path(),
                    });
                )+
            };
        }
        editable! {
            Transform => "Transform",
            Visibility => "Visibility",
            MeshRef => "Mesh",
            SkeletalMesh => "Skeletal Mesh",
            AnimPlayer => "Anim Player",
            AnimStateMachine => "Anim State Machine",
            Sprite => "Sprite",
            NineSlice => "Nine Slice",
            Text2D => "Text",
            Tilemap => "Tilemap",
            Material => "Material",
            Terrain => "Terrain",
            PcgVolume => "PCG Volume",
            Light => "Light",
            Light2D => "Light 2D",
            Camera => "Camera",
            Decal => "Decal",
            Volume => "Volume",
            Spline => "Spline",
            Foliage => "Foliage",
            RigidBody2D => "Rigid Body 2D",
            Collider2D => "Collider 2D",
            CharacterController2D => "Character Controller 2D",
            RigidBody3D => "Rigid Body 3D",
            Collider3D => "Collider 3D",
            CharacterController3D => "Character Controller 3D",
            Joint2D => "Joint 2D",
            Joint3D => "Joint 3D",
            AudioSource => "Audio Source",
            AudioListener => "Audio Listener",
        }

        Self { types, editable }
    }

    pub fn types(&self) -> &TypeRegistry {
        &self.types
    }

    /// The reflect `type_path` of an editable component by its Details display
    /// name (e.g. `"Material"`), for editor code that mutates components by
    /// field without naming `bevy_reflect` itself.
    pub fn type_path_for(&self, display: &str) -> Option<&'static str> {
        self.editable
            .iter()
            .find(|c| c.display == display)
            .map(|c| c.type_path)
    }

    /// The editable components, in canonical Details order.
    pub fn editable(&self) -> &[ComponentInfo] {
        &self.editable
    }

    /// The components the user may **add** to an entity via "+ Add Component"
    /// (E-P1), in canonical order. This is the editable list minus the components
    /// every entity structurally owns and cannot gain or lose:
    ///
    /// * `Transform` — spawned on every entity; the hierarchy + gizmo assume it.
    /// * `Name` — already excluded from `editable` (edited via rename).
    ///
    /// Computed components (`GlobalTransform`, `ComputedVisibility`) are not in
    /// `editable` at all, so they are already absent here. `Visibility` stays
    /// addable/removable (it is a plain optional flag).
    pub fn addable(&self) -> impl Iterator<Item = &ComponentInfo> {
        let transform = <Transform as TypePath>::type_path();
        self.editable
            .iter()
            .filter(move |c| c.type_path != transform)
    }

    /// The `ReflectDefault` handle for a registered `type_path`, if it carries
    /// that type-data — powers add-component (insert a `Default` instance) and
    /// per-property reset. Kept behind the facade like `reflect_component`.
    pub fn reflect_default(
        &self,
        type_path: &str,
    ) -> Option<&bevy_reflect::std_traits::ReflectDefault> {
        self.types
            .get_with_type_path(type_path)?
            .data::<bevy_reflect::std_traits::ReflectDefault>()
    }

    /// The `ReflectComponent` handle for a registered component `type_path`.
    pub fn reflect_component(&self, type_path: &str) -> Option<&ReflectComponent> {
        self.types
            .get_with_type_path(type_path)?
            .data::<ReflectComponent>()
    }

    /// Display name for a component `type_path`, if editable.
    pub fn display_name(&self, type_path: &str) -> Option<&'static str> {
        self.editable
            .iter()
            .find(|c| c.type_path == type_path)
            .map(|c| c.display)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_components_are_registered() {
        let reg = ComponentRegistry::new();
        assert_eq!(reg.editable().len(), 30);
        // Every editable component resolves a ReflectComponent handle.
        for info in reg.editable() {
            assert!(
                reg.reflect_component(info.type_path).is_some(),
                "{} missing ReflectComponent",
                info.display
            );
        }
        assert_eq!(reg.display_name(Transform::type_path()), Some("Transform"));
    }

    #[test]
    fn addable_excludes_transform_and_every_entry_has_reflect_default() {
        let reg = ComponentRegistry::new();
        let addable: Vec<_> = reg.addable().collect();
        // Transform is excluded; everything else editable stays.
        assert_eq!(addable.len(), reg.editable().len() - 1);
        assert!(addable
            .iter()
            .all(|c| c.type_path != Transform::type_path()));
        // Add-component inserts a Default instance, so every addable component
        // must carry ReflectDefault.
        for c in &addable {
            assert!(
                reg.reflect_default(c.type_path).is_some(),
                "{} missing ReflectDefault",
                c.display
            );
        }
    }
}
