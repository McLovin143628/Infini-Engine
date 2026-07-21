//! Component registration (P3.1.1).
//!
//! One place registers every reflected component + value type into a
//! `bevy_reflect::TypeRegistry` and records which components are user-editable
//! (shown in the Details grid / "Add Component" menu). The Details reflection
//! walker (P3.3) and undo (P3.4) look components up here by their stable reflect
//! `type_path`, so this list is the single source of truth for "what can the
//! editor see and change".

use bevy_ecs::reflect::ReflectComponent;
use bevy_reflect::{TypePath, TypeRegistry};

use crate::components::{
    AnimPlayer, AnimStateMachine, AtlasRect, BillboardMode, BodyKind2D, BodyKind3D, Camera,
    CharacterController2D, CharacterController3D, Collider2D, Collider3D, ColliderShape2DKind,
    ColliderShape3DKind, Light, Light2D, LightKind, Material, MeshRef, Name, NineSlice, PcgVolume,
    Primitive, RigidBody2D, RigidBody3D, SkeletalMesh, Sprite, Terrain, Text2D, TextAlign, Tilemap,
    Transform, Visibility,
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
        types.register::<Primitive>();
        types.register::<LightKind>();
        types.register::<TextAlign>();
        types.register::<BodyKind2D>();
        types.register::<ColliderShape2DKind>();
        types.register::<BodyKind3D>();
        types.register::<ColliderShape3DKind>();
        types.register::<BillboardMode>();
        types.register::<AtlasRect>();
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
            RigidBody2D => "Rigid Body 2D",
            Collider2D => "Collider 2D",
            CharacterController2D => "Character Controller 2D",
            RigidBody3D => "Rigid Body 3D",
            Collider3D => "Collider 3D",
            CharacterController3D => "Character Controller 3D",
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
        assert_eq!(reg.editable().len(), 22);
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
}
