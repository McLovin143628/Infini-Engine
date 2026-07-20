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
    Camera, Light, LightKind, Material, MeshRef, Name, Primitive, Transform, Visibility,
};
use crate::math::{Color, Vec3d};

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
        types.register::<Color>();
        types.register::<Primitive>();
        types.register::<LightKind>();
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
            Material => "Material",
            Light => "Light",
            Camera => "Camera",
        }

        Self { types, editable }
    }

    pub fn types(&self) -> &TypeRegistry {
        &self.types
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
        assert_eq!(reg.editable().len(), 6);
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
