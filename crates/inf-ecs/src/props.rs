//! Reflection-driven property access (P3.3), kept behind the facade.
//!
//! The Details panel must read and write component fields generically. Rather
//! than leak `bevy_reflect` types past this crate (the facade rule), we walk a
//! component's reflected fields here and hand back plain, serde-friendly
//! descriptors ([`ComponentProps`] / [`PropValue`]); writes come back in the
//! same vocabulary. The editor never names `dyn Reflect`.

use bevy_ecs::prelude::{Entity, World};
use bevy_reflect::enums::{DynamicEnum, DynamicVariant};
use bevy_reflect::{PartialReflect, ReflectMut, ReflectRef, TypeInfo, TypePath};

use crate::components::Name;
use crate::math::{Color, Vec2d, Vec3d};
use crate::registry::ComponentRegistry;

/// A typed property value — the widget kind is implied by the variant.
#[derive(Clone, Debug, PartialEq)]
pub enum PropValue {
    Bool(bool),
    Number(f64),
    Text(String),
    Vec3([f64; 3]),
    Color([f32; 4]),
    Enum { value: String, options: Vec<String> },
}

/// One editable field of a component.
#[derive(Clone, Debug, PartialEq)]
pub struct PropField {
    /// Reflect field name (write key), e.g. `base_color`.
    pub name: String,
    /// Human label, e.g. `Base Color`.
    pub label: String,
    pub value: PropValue,
}

/// One component's editable fields.
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentProps {
    pub type_path: String,
    pub display: String,
    pub fields: Vec<PropField>,
}

/// Read every editable component on `entity` as property descriptors, in the
/// registry's canonical order. `Name` is excluded (edited via rename).
pub fn read_entity(
    world: &World,
    registry: &ComponentRegistry,
    entity: Entity,
) -> Vec<ComponentProps> {
    let mut out = Vec::new();
    for info in registry.editable() {
        let Some(reflect) = registry.reflect_component(info.type_path) else {
            continue;
        };
        let entity_ref = world.entity(entity);
        let Some(comp) = reflect.reflect(entity_ref) else {
            continue; // component not present on this entity
        };
        if comp.reflect_type_path() == <Name as TypePath>::type_path() {
            continue;
        }
        if let Some(fields) = read_struct_fields(comp.as_partial_reflect()) {
            out.push(ComponentProps {
                type_path: info.type_path.to_string(),
                display: info.display.to_string(),
                fields,
            });
        }
    }
    out
}

fn read_struct_fields(pr: &dyn PartialReflect) -> Option<Vec<PropField>> {
    let ReflectRef::Struct(s) = pr.reflect_ref() else {
        return None;
    };
    let mut fields = Vec::new();
    for i in 0..s.field_len() {
        let name = s.name_at(i).unwrap_or("").to_string();
        let Some(field) = s.field_at(i) else { continue };
        if let Some(value) = read_value(field) {
            fields.push(PropField {
                label: prettify(&name),
                name,
                value,
            });
        }
    }
    Some(fields)
}

fn read_value(pr: &dyn PartialReflect) -> Option<PropValue> {
    if let Some(v) = pr.try_downcast_ref::<Vec3d>() {
        return Some(PropValue::Vec3([v.x, v.y, v.z]));
    }
    if let Some(v) = pr.try_downcast_ref::<Vec2d>() {
        // Surfaced on the existing vector widget; the third slot is unused for a
        // 2D value (reads back 0, write-back below keeps only x/y).
        return Some(PropValue::Vec3([v.x, v.y, 0.0]));
    }
    if let Some(v) = pr.try_downcast_ref::<Color>() {
        return Some(PropValue::Color([v.r, v.g, v.b, v.a]));
    }
    if let Some(v) = pr.try_downcast_ref::<f64>() {
        return Some(PropValue::Number(*v));
    }
    if let Some(v) = pr.try_downcast_ref::<f32>() {
        return Some(PropValue::Number(*v as f64));
    }
    if let Some(v) = pr.try_downcast_ref::<i32>() {
        return Some(PropValue::Number(*v as f64));
    }
    if let Some(v) = pr.try_downcast_ref::<bool>() {
        return Some(PropValue::Bool(*v));
    }
    if let Some(v) = pr.try_downcast_ref::<String>() {
        return Some(PropValue::Text(v.clone()));
    }
    if let ReflectRef::Enum(e) = pr.reflect_ref() {
        return Some(PropValue::Enum {
            value: e.variant_name().to_string(),
            options: enum_options(pr),
        });
    }
    None
}

fn enum_options(pr: &dyn PartialReflect) -> Vec<String> {
    match pr.get_represented_type_info() {
        Some(TypeInfo::Enum(info)) => info.variant_names().iter().map(|s| s.to_string()).collect(),
        _ => Vec::new(),
    }
}

/// Write a single field of `type_path`'s component on `entity`. Returns whether
/// the write applied.
pub fn write_field(
    world: &mut World,
    registry: &ComponentRegistry,
    entity: Entity,
    type_path: &str,
    field: &str,
    value: &PropValue,
) -> bool {
    let Some(reflect) = registry.reflect_component(type_path) else {
        return false;
    };
    // ReflectComponent uses the registry-held type data; clone the handle out so
    // we can borrow the world mutably below.
    let reflect = reflect.clone();
    let mut entity_mut = world.entity_mut(entity);
    let Some(mut comp) = reflect.reflect_mut(&mut entity_mut) else {
        return false;
    };
    let ReflectMut::Struct(s) = comp.reflect_mut() else {
        return false;
    };
    let Some(target) = s.field_mut(field) else {
        return false;
    };
    apply_value(target, value)
}

/// The default value of `type_path`'s `field` (from the component's
/// `ReflectDefault`) — powers per-property "reset to default" (P3.3.4).
pub fn default_field(
    registry: &ComponentRegistry,
    type_path: &str,
    field: &str,
) -> Option<PropValue> {
    use bevy_reflect::std_traits::ReflectDefault;
    let registration = registry.types().get_with_type_path(type_path)?;
    let default = registration.data::<ReflectDefault>()?.default();
    let ReflectRef::Struct(s) = default.reflect_ref() else {
        return None;
    };
    read_value(s.field(field)?)
}

fn apply_value(field: &mut dyn PartialReflect, value: &PropValue) -> bool {
    match value {
        PropValue::Number(n) => {
            if let Some(x) = field.try_downcast_mut::<f64>() {
                *x = *n;
                true
            } else if let Some(x) = field.try_downcast_mut::<f32>() {
                *x = *n as f32;
                true
            } else if let Some(x) = field.try_downcast_mut::<i32>() {
                *x = n.round() as i32;
                true
            } else {
                false
            }
        }
        PropValue::Bool(b) => field.try_downcast_mut::<bool>().map(|x| *x = *b).is_some(),
        PropValue::Text(t) => field
            .try_downcast_mut::<String>()
            .map(|x| *x = t.clone())
            .is_some(),
        PropValue::Vec3(a) => {
            if let Some(v) = field.try_downcast_mut::<Vec3d>() {
                *v = Vec3d::new(a[0], a[1], a[2]);
                true
            } else if let Some(v) = field.try_downcast_mut::<Vec2d>() {
                *v = Vec2d::new(a[0], a[1]);
                true
            } else {
                false
            }
        }
        PropValue::Color(c) => field
            .try_downcast_mut::<Color>()
            .map(|v| *v = Color::new(c[0], c[1], c[2], c[3]))
            .is_some(),
        PropValue::Enum { value, .. } => field
            .try_apply(&DynamicEnum::new(value.clone(), DynamicVariant::Unit))
            .is_ok(),
    }
}

/// `base_color` → `Base Color`.
fn prettify(field: &str) -> String {
    field
        .split('_')
        .filter(|s| !s.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().chain(c).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Light, MeshRef, Transform};

    use crate::EcsWorld;

    fn world_with_cube() -> (EcsWorld, ComponentRegistry, Entity) {
        let mut w = EcsWorld::new();
        let e = w.spawn("Cube", None);
        w.world_mut().entity_mut(e).insert((
            Transform::IDENTITY,
            MeshRef::default(),
            Light::default(),
        ));
        let reg = ComponentRegistry::new();
        (w, reg, e)
    }

    #[test]
    fn reads_transform_and_enum_fields() {
        let (w, reg, e) = world_with_cube();
        let props = read_entity(w.world(), &reg, e);
        let t = props.iter().find(|p| p.display == "Transform").unwrap();
        let translation = t.fields.iter().find(|f| f.name == "translation").unwrap();
        assert!(matches!(translation.value, PropValue::Vec3(_)));
        assert_eq!(translation.label, "Translation");

        let light = props.iter().find(|p| p.display == "Light").unwrap();
        let kind = light.fields.iter().find(|f| f.name == "kind").unwrap();
        match &kind.value {
            PropValue::Enum { value, options } => {
                assert_eq!(value, "Point");
                assert!(options.contains(&"Directional".to_string()));
            }
            other => panic!("expected enum, got {other:?}"),
        }
    }

    #[test]
    fn writes_number_vec3_and_enum() {
        let (mut w, reg, e) = world_with_cube();
        assert!(write_field(
            w.world_mut(),
            &reg,
            e,
            <Transform as TypePath>::type_path(),
            "translation",
            &PropValue::Vec3([1.0, 2.0, 3.0]),
        ));
        let props = read_entity(w.world(), &reg, e);
        let t = props.iter().find(|p| p.display == "Transform").unwrap();
        let translation = &t
            .fields
            .iter()
            .find(|f| f.name == "translation")
            .unwrap()
            .value;
        assert_eq!(*translation, PropValue::Vec3([1.0, 2.0, 3.0]));

        assert!(write_field(
            w.world_mut(),
            &reg,
            e,
            <Light as TypePath>::type_path(),
            "kind",
            &PropValue::Enum {
                value: "Spot".into(),
                options: vec![],
            },
        ));
        let props = read_entity(w.world(), &reg, e);
        let light = props.iter().find(|p| p.display == "Light").unwrap();
        let kind = &light
            .fields
            .iter()
            .find(|f| f.name == "kind")
            .unwrap()
            .value;
        assert!(matches!(kind, PropValue::Enum { value, .. } if value == "Spot"));
    }
}
