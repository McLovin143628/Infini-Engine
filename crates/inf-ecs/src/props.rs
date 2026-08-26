//! Reflection-driven property access (P3.3), kept behind the facade.
//!
//! The Details panel must read and write component fields generically. Rather
//! than leak `bevy_reflect` types past this crate (the facade rule), we walk a
//! component's reflected fields here and hand back plain, serde-friendly
//! descriptors ([`ComponentProps`] / [`PropValue`]); writes come back in the
//! same vocabulary. The editor never names `dyn Reflect`.

use bevy_ecs::prelude::{Entity, World};
use bevy_reflect::enums::{DynamicEnum, DynamicVariant};
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::{GetPath, PartialReflect, Reflect, ReflectMut, ReflectRef, TypeInfo, TypePath};
use uuid::Uuid;

use crate::components::Name;
use crate::math::{Color, Vec2d, Vec3d};
use crate::refs::EntityRef;
use crate::registry::ComponentRegistry;

/// How deep the reflection walker recurses into nested structs before treating a
/// value as opaque (top-level component fields are depth 0). Keeps the Details
/// grid bounded and avoids cycles.
const MAX_STRUCT_DEPTH: usize = 3;
/// The largest editable list. Longer lists (e.g. `Foliage::instances`, already
/// `#[reflect(ignore)]`) are omitted from the grid so a write-back can never
/// silently truncate bulk data.
const MAX_LIST_LEN: usize = 256;

/// A typed property value — the widget kind is implied by the variant.
///
/// The scalar variants (`Bool`..`Enum`) are leaves. `List`/`Struct` carry nested
/// property trees (E-P1 deep editing); `EntityRef` is an opaque entity link
/// surfaced by an entity-picker widget.
#[derive(Clone, Debug, PartialEq)]
pub enum PropValue {
    Bool(bool),
    Number(f64),
    Text(String),
    Vec3([f64; 3]),
    Color([f32; 4]),
    Enum {
        value: String,
        options: Vec<String>,
    },
    /// A homogeneous list. Editing sends the WHOLE list back through the write
    /// path (the List write arm rebuilds the reflected collection).
    List(Vec<PropValue>),
    /// A nested struct's `(field_name, value)` pairs, in declaration order.
    Struct(Vec<(String, PropValue)>),
    /// A reference to another entity by stable GUID (`None` → unbound).
    EntityRef(Option<Uuid>),
}

/// A numeric field's **UI range** — the hint that turns a bare number box into a
/// slider (wave VIS1b).
///
/// A *hint*, not a constraint: [`apply_value`] does not clamp to it, because a
/// number an author deliberately typed is not the widget's business. What the
/// range does is give the drag a scale, which is the whole difference between
/// "roughness" as a step-1 spinner (three clicks from 0 to useless) and as a
/// slider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropRange {
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

/// The per-field range table, keyed by `(type_path, field)`.
///
/// It lives in Ring 0 rather than in the panel for the reason
/// `ResolvedSky::cloud_time_s` does: the same four `min=0 max=1 step=0.01`
/// triples are currently retyped by hand in `MaterialInstanceEditor`,
/// `AudioMixerPanel`, `WorldSettingsPanel` and `BlendSpacePanel`, and a number
/// restated in four places is four chances to disagree.
///
/// Deliberately **short**. Every entry has to be a fact about the field's units,
/// and a table of guesses would be worse than no table — the reflection grid's
/// step-1 spinner is at least honest about knowing nothing.
const PROP_RANGES: &[(&str, &str, PropRange)] = &[
    // `Material`, the four numeric fields it has. Metallic/roughness/alpha are
    // `[0,1]` by the PBR model's own definition; the intensity's ceiling is a
    // *display* range rather than a limit — see `Material::emissive_intensity`.
    (
        MATERIAL_TYPE_PATH,
        "metallic",
        PropRange {
            min: 0.0,
            max: 1.0,
            step: 0.01,
        },
    ),
    (
        MATERIAL_TYPE_PATH,
        "roughness",
        PropRange {
            min: 0.0,
            max: 1.0,
            step: 0.01,
        },
    ),
    (
        MATERIAL_TYPE_PATH,
        "alpha_cutoff",
        PropRange {
            min: 0.0,
            max: 1.0,
            step: 0.01,
        },
    ),
    (
        MATERIAL_TYPE_PATH,
        "emissive_intensity",
        PropRange {
            min: 0.0,
            max: 64.0,
            step: 0.1,
        },
    ),
];

/// `Material`'s reflect type path, pinned against the type itself by
/// `the_range_table_names_types_that_exist`.
const MATERIAL_TYPE_PATH: &str = "inf_ecs::components::Material";

/// The UI range for `type_path`'s `field`, if the table names one.
pub fn prop_range(type_path: &str, field: &str) -> Option<PropRange> {
    PROP_RANGES
        .iter()
        .find(|(t, f, _)| *t == type_path && *f == field)
        .map(|(_, _, r)| *r)
}

/// One editable field of a component.
#[derive(Clone, Debug, PartialEq)]
pub struct PropField {
    /// Reflect field name (write key), e.g. `base_color`.
    pub name: String,
    /// Human label, e.g. `Base Color`.
    pub label: String,
    pub value: PropValue,
    /// The widget's numeric range, when the field is one the engine knows the
    /// units of (wave VIS1b). `None` ⇒ a plain number box, which is every field
    /// this tree has ever drawn.
    pub range: Option<PropRange>,
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
        if let Some(fields) = read_struct_fields(comp.as_partial_reflect(), 0, info.type_path) {
            out.push(ComponentProps {
                type_path: info.type_path.to_string(),
                display: info.display.to_string(),
                fields,
            });
        }
    }
    out
}

/// `owner` is the **component's** type path, so the range table can be consulted;
/// it is `""` for the nested levels, where a field name alone would not identify
/// anything.
fn read_struct_fields(
    pr: &dyn PartialReflect,
    depth: usize,
    owner: &str,
) -> Option<Vec<PropField>> {
    let ReflectRef::Struct(s) = pr.reflect_ref() else {
        return None;
    };
    let mut fields = Vec::new();
    for i in 0..s.field_len() {
        let name = s.name_at(i).unwrap_or("").to_string();
        let Some(field) = s.field_at(i) else { continue };
        if let Some(value) = read_value(field, depth) {
            fields.push(PropField {
                label: prettify(&name),
                range: prop_range(owner, &name),
                name,
                value,
            });
        }
    }
    Some(fields)
}

fn read_value(pr: &dyn PartialReflect, depth: usize) -> Option<PropValue> {
    // Opaque entity reference (E-P1) — checked before the generic struct arm.
    if let Some(v) = pr.try_downcast_ref::<EntityRef>() {
        return Some(PropValue::EntityRef(v.0));
    }
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
    if let Some(v) = pr.try_downcast_ref::<u32>() {
        return Some(PropValue::Number(*v as f64));
    }
    if let Some(v) = pr.try_downcast_ref::<bool>() {
        return Some(PropValue::Bool(*v));
    }
    if let Some(v) = pr.try_downcast_ref::<String>() {
        return Some(PropValue::Text(v.clone()));
    }
    match pr.reflect_ref() {
        ReflectRef::Enum(e) => {
            return Some(PropValue::Enum {
                value: e.variant_name().to_string(),
                options: enum_options(pr),
            });
        }
        // A homogeneous list (e.g. `Spline::points: Vec<Vec3d>`). Bounded to
        // MAX_LIST_LEN so a write-back never truncates bulk data — longer lists
        // are omitted (read-only by absence).
        ReflectRef::List(list) => {
            if depth >= MAX_STRUCT_DEPTH || list.len() > MAX_LIST_LEN {
                return None;
            }
            let mut elems = Vec::with_capacity(list.len());
            for i in 0..list.len() {
                let e = list.get(i)?;
                // Every element must be readable for the list to be editable as a
                // whole (writes replace the entire list).
                elems.push(read_value(e, depth + 1)?);
            }
            return Some(PropValue::List(elems));
        }
        // A nested struct — recurse, depth-capped.
        ReflectRef::Struct(_) => {
            if depth >= MAX_STRUCT_DEPTH {
                return None;
            }
            // No owner below the top level: a nested struct's field name alone
            // does not identify anything the range table could key on, and
            // `PropValue::Struct` carries `(name, value)` pairs rather than
            // `PropField`s anyway.
            let fields = read_struct_fields(pr, depth + 1, "")?;
            let pairs = fields.into_iter().map(|f| (f.name, f.value)).collect();
            return Some(PropValue::Struct(pairs));
        }
        _ => {}
    }
    None
}

fn enum_options(pr: &dyn PartialReflect) -> Vec<String> {
    match pr.get_represented_type_info() {
        Some(TypeInfo::Enum(info)) => info.variant_names().iter().map(|s| s.to_string()).collect(),
        _ => Vec::new(),
    }
}

/// Write a single field of `type_path`'s component on `entity`. `path` is a
/// `bevy_reflect` access path — a bare field name (`"translation"`) for a
/// top-level field, or a nested path (`"points[1].y"`, `"nested.field"`) for
/// deep edits (E-P1). Returns whether the write applied.
pub fn write_field(
    world: &mut World,
    registry: &ComponentRegistry,
    entity: Entity,
    type_path: &str,
    path: &str,
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
    let Ok(target) = comp.reflect_path_mut(path) else {
        return false;
    };
    apply_value(registry, target, value)
}

/// Insert a `Default` instance of `type_path`'s component onto `entity` (E-P1
/// "+ Add Component"). Returns whether it applied. Builds the value from the
/// component's `ReflectDefault` and inserts it via `ReflectComponent`, keeping
/// `bevy_reflect` behind the facade.
pub fn insert_default_component(
    world: &mut World,
    registry: &ComponentRegistry,
    entity: Entity,
    type_path: &str,
) -> bool {
    let Some(reflect) = registry.reflect_component(type_path) else {
        return false;
    };
    let reflect = reflect.clone();
    let Some(default) = registry.reflect_default(type_path) else {
        return false;
    };
    let value = default.default();
    let mut entity_mut = world.entity_mut(entity);
    reflect.insert(
        &mut entity_mut,
        value.as_partial_reflect(),
        registry.types(),
    );
    true
}

/// Remove `type_path`'s component from `entity` (E-P1 "Remove Component").
/// Returns whether the component type is known.
pub fn remove_component(
    world: &mut World,
    registry: &ComponentRegistry,
    entity: Entity,
    type_path: &str,
) -> bool {
    let Some(reflect) = registry.reflect_component(type_path) else {
        return false;
    };
    let reflect = reflect.clone();
    let mut entity_mut = world.entity_mut(entity);
    reflect.remove(&mut entity_mut);
    true
}

/// The default value of `type_path`'s field at `path` (from the component's
/// `ReflectDefault`) — powers per-property "reset to default" (P3.3.4). `path`
/// may be nested (E-P1).
pub fn default_field(
    registry: &ComponentRegistry,
    type_path: &str,
    path: &str,
) -> Option<PropValue> {
    let registration = registry.types().get_with_type_path(type_path)?;
    let default = registration.data::<ReflectDefault>()?.default();
    let field = default.reflect_path(path).ok()?;
    read_value(field, 0)
}

/// The default value of a single **element** of the list at `path` on
/// `type_path`'s component — powers the ListField "add element" button (the new
/// row starts from the element type's `Default`). E-P1.
pub fn default_list_element(
    registry: &ComponentRegistry,
    type_path: &str,
    path: &str,
) -> Option<PropValue> {
    let registration = registry.types().get_with_type_path(type_path)?;
    let default = registration.data::<ReflectDefault>()?.default();
    let list_field = default.reflect_path(path).ok()?;
    let elem = list_item_default(registry, list_field)?;
    read_value(elem.as_partial_reflect(), 0)
}

/// A fresh default element for the reflected list `list_field`, resolved from the
/// element type's `ReflectDefault` in the registry.
fn list_item_default(
    registry: &ComponentRegistry,
    list_field: &dyn PartialReflect,
) -> Option<Box<dyn Reflect>> {
    let TypeInfo::List(info) = list_field.get_represented_type_info()? else {
        return None;
    };
    let item_id = info.item_ty().id();
    let registration = registry.types().get(item_id)?;
    Some(registration.data::<ReflectDefault>()?.default())
}

fn apply_value(
    registry: &ComponentRegistry,
    field: &mut dyn PartialReflect,
    value: &PropValue,
) -> bool {
    match value {
        PropValue::Number(n) => {
            // **The finite door** (wave VIS1b), and it is here rather than in the
            // React number field because this is the one place EVERY numeric write
            // passes through: the Details panel, `tuning::apply_tune`'s live
            // preview, the sequencer's scrub, and `edit_apply_material`. The
            // frontend's `Number.isFinite` guard covers exactly one of the four,
            // and it cannot cover the case that actually reaches the GPU anyway —
            // `1e300` is a perfectly finite `f64` and is `inf` the instant it is
            // cast to `f32`, after which `Material::emissive_linear`'s
            // `0.0 * inf` is a NaN in the instance buffer.
            //
            // A refusal, not a failure: the write does not apply, the undo step is
            // not recorded, and the panel re-reads the value it already had. That
            // is the P21 ruling — an erroring edit takes its whole handler down.
            if !n.is_finite() {
                return false;
            }
            if let Some(x) = field.try_downcast_mut::<f64>() {
                *x = *n;
                true
            } else if let Some(x) = field.try_downcast_mut::<f32>() {
                let v = *n as f32;
                if !v.is_finite() {
                    return false;
                }
                *x = v;
                true
            } else if let Some(x) = field.try_downcast_mut::<i32>() {
                *x = n.round() as i32;
                true
            } else if let Some(x) = field.try_downcast_mut::<u32>() {
                *x = n.round().max(0.0) as u32;
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
        PropValue::EntityRef(guid) => field
            .try_downcast_mut::<EntityRef>()
            .map(|v| v.0 = *guid)
            .is_some(),
        PropValue::Enum { value, .. } => field
            .try_apply(&DynamicEnum::new(value.clone(), DynamicVariant::Unit))
            .is_ok(),
        // **Scratch-build, then swap** (C4-32). Apply into a *copy* of the
        // struct and commit it only when every field took; a partial apply
        // leaves the authored value exactly as it was.
        //
        // The old shape mutated in place and reported `all == false` afterwards,
        // which is the worst of both: the caller reads the verdict as "nothing
        // happened" — `world.rs`'s `dirty = true` is skipped and `doc.rs` does
        // not record the undo command — while the entity is already holding the
        // half-applied value, and *that* is what the next save writes. Data
        // changed, document unaware, no undo step.
        PropValue::Struct(pairs) => {
            let mut scratch = field.to_dynamic();
            {
                let ReflectMut::Struct(s) = scratch.reflect_mut() else {
                    return false;
                };
                for (name, pv) in pairs {
                    match s.field_mut(name) {
                        Some(child) => {
                            if !apply_value(registry, child, pv) {
                                return false;
                            }
                        }
                        None => return false,
                    }
                }
            }
            field.try_apply(scratch.as_ref()).is_ok()
        }
        // Replace the whole list: build a fresh default element per entry, apply
        // the incoming value into it, then swap the collection's contents. This
        // is length-changing (add/remove) safe, unlike element-wise `try_apply`.
        //
        // **Every element is built before any of the authored ones are
        // destroyed** (C4-32). `while list.pop().is_some() {}` used to run
        // *first*, so a value the Details ListField (or a malformed `PropValue`
        // over the IPC) could not apply left the user's authored list replaced
        // by defaults and partials — while the function returned `false`, i.e.
        // "nothing happened", so no undo step was recorded and the document did
        // not know it was dirty. The user's data was gone with no way back and
        // no indication anything had occurred.
        PropValue::List(elems) => {
            // Resolve the element `Default` factory up front — `item_ty` comes
            // from the field's `'static` type info (no lingering field borrow),
            // and the factory borrows only the registry, so it survives the
            // mutable list borrow below and yields a fresh element each call.
            let Some(TypeInfo::List(info)) = field.get_represented_type_info() else {
                return false;
            };
            let Some(item_default) = registry
                .types()
                .get(info.item_ty().id())
                .and_then(|r| r.data::<ReflectDefault>())
            else {
                return false;
            };
            let mut built = Vec::with_capacity(elems.len());
            for pv in elems {
                let mut elem = item_default.default();
                if !apply_value(registry, elem.as_partial_reflect_mut(), pv) {
                    return false; // the authored list is untouched
                }
                built.push(elem.into_partial_reflect());
            }
            let ReflectMut::List(list) = field.reflect_mut() else {
                return false;
            };
            while list.pop().is_some() {}
            for elem in built {
                list.push(elem);
            }
            true
        }
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
    use crate::components::{Joint3D, Light, MeshRef, Spline, SplineInterp, Tilemap, Transform};

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
    fn reads_and_writes_u32_atlas_dims() {
        // The tilemap's atlas grid dims are `u32` — they must surface on the
        // number widget and round-trip a write (chunk map / texture are ignored).
        let mut w = EcsWorld::new();
        let e = w.spawn("Tiles", None);
        w.world_mut().entity_mut(e).insert(Tilemap {
            atlas_cols: 4,
            ..Default::default()
        });
        let reg = ComponentRegistry::new();

        let props = read_entity(w.world(), &reg, e);
        let tm = props.iter().find(|p| p.display == "Tilemap").unwrap();
        let cols = tm.fields.iter().find(|f| f.name == "atlas_cols").unwrap();
        assert_eq!(cols.value, PropValue::Number(4.0));
        // The ignored chunk map / texture never appear as editable fields.
        assert!(tm
            .fields
            .iter()
            .all(|f| f.name != "chunks" && f.name != "texture"));

        assert!(write_field(
            w.world_mut(),
            &reg,
            e,
            <Tilemap as TypePath>::type_path(),
            "atlas_cols",
            &PropValue::Number(8.0),
        ));
        assert_eq!(w.world().entity(e).get::<Tilemap>().unwrap().atlas_cols, 8);
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

    // ── E-P1 deep editing ───────────────────────────────────────────────────

    fn world_with_spline() -> (EcsWorld, ComponentRegistry, Entity) {
        let mut w = EcsWorld::new();
        let e = w.spawn("Path", None);
        w.world_mut().entity_mut(e).insert(Spline {
            points: vec![
                Vec3d::new(0.0, 0.0, 0.0),
                Vec3d::new(1.0, 2.0, 3.0),
                Vec3d::new(4.0, 5.0, 6.0),
            ],
            closed: false,
            interp: SplineInterp::Linear,
        });
        (w, ComponentRegistry::new(), e)
    }

    #[test]
    fn reads_spline_points_as_a_list_of_vec3() {
        let (w, reg, e) = world_with_spline();
        let props = read_entity(w.world(), &reg, e);
        let sp = props.iter().find(|p| p.display == "Spline").unwrap();
        let points = &sp.fields.iter().find(|f| f.name == "points").unwrap().value;
        match points {
            PropValue::List(elems) => {
                assert_eq!(elems.len(), 3);
                assert_eq!(elems[1], PropValue::Vec3([1.0, 2.0, 3.0]));
            }
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn writes_nested_list_element_via_path() {
        // Element edit via a reflect path (the frontend also supports whole-list
        // replace, exercised below).
        let (mut w, reg, e) = world_with_spline();
        assert!(write_field(
            w.world_mut(),
            &reg,
            e,
            <Spline as TypePath>::type_path(),
            "points[1].y",
            &PropValue::Number(42.0),
        ));
        assert_eq!(
            w.world().entity(e).get::<Spline>().unwrap().points[1].y,
            42.0
        );
    }

    #[test]
    fn replaces_whole_list_add_and_remove() {
        let (mut w, reg, e) = world_with_spline();
        let tp = <Spline as TypePath>::type_path();
        // Grow to 4 (append) and change element 0.
        let new_list = PropValue::List(vec![
            PropValue::Vec3([9.0, 9.0, 9.0]),
            PropValue::Vec3([1.0, 2.0, 3.0]),
            PropValue::Vec3([4.0, 5.0, 6.0]),
            PropValue::Vec3([7.0, 7.0, 7.0]),
        ]);
        assert!(write_field(w.world_mut(), &reg, e, tp, "points", &new_list));
        {
            let pts = &w.world().entity(e).get::<Spline>().unwrap().points;
            assert_eq!(pts.len(), 4);
            assert_eq!(pts[0], Vec3d::new(9.0, 9.0, 9.0));
            assert_eq!(pts[3], Vec3d::new(7.0, 7.0, 7.0));
        }
        // Shrink to 1 (remove).
        let shrink = PropValue::List(vec![PropValue::Vec3([0.0, 0.0, 0.0])]);
        assert!(write_field(w.world_mut(), &reg, e, tp, "points", &shrink));
        assert_eq!(w.world().entity(e).get::<Spline>().unwrap().points.len(), 1);
    }

    /// **C4-32 — a rejected list edit must leave the authored list alone.**
    ///
    /// `while list.pop().is_some() {}` used to run *before* anything was
    /// validated, and each element was then rebuilt as a default, partially
    /// applied, and pushed regardless. So a `PropValue` the elements cannot take
    /// — a Details ListField in a mixed state, or a malformed value over the IPC
    /// — replaced the author's data with defaults and partials **and** returned
    /// `false`. That verdict is what `world.rs` reads to decide `dirty = true`
    /// and `doc.rs` reads to decide whether to record the undo command, so the
    /// data was gone, the document did not know it had changed, there was no
    /// undo step, and the mutated state is what the next save would write.
    ///
    /// Un-fix mutation: move the `pop` loop back above the build loop and this
    /// fails on the surviving-points assertion.
    #[test]
    fn a_rejected_list_edit_leaves_the_authored_list_untouched() {
        let (mut w, reg, e) = world_with_spline();
        let tp = <Spline as TypePath>::type_path();
        let before = w.world().entity(e).get::<Spline>().unwrap().points.clone();
        assert_eq!(before.len(), 3, "the fixture must have something to lose");

        // A list whose second element is the wrong shape for `Vec3d`.
        let bad = PropValue::List(vec![
            PropValue::Vec3([9.0, 9.0, 9.0]),
            PropValue::Text("not a point".into()),
            PropValue::Vec3([4.0, 5.0, 6.0]),
        ]);
        assert!(
            !write_field(w.world_mut(), &reg, e, tp, "points", &bad),
            "the write reported success for a value it could not apply"
        );
        assert_eq!(
            w.world().entity(e).get::<Spline>().unwrap().points,
            before,
            "a refused edit destroyed the authored list"
        );

        // The same rule for a struct: a partially-applicable struct value must
        // not half-land. `interp` is an enum, so a Number cannot take.
        let before_closed = w.world().entity(e).get::<Spline>().unwrap().closed;
        let bad_struct = PropValue::Struct(vec![
            ("closed".into(), PropValue::Bool(!before_closed)),
            ("interp".into(), PropValue::Number(3.0)),
        ]);
        assert!(!write_field(w.world_mut(), &reg, e, tp, "", &bad_struct));
        assert_eq!(
            w.world().entity(e).get::<Spline>().unwrap().closed,
            before_closed,
            "a refused struct edit half-landed"
        );

        // The control: a good edit still applies, so the refusals above are not
        // passing because nothing applies.
        let good = PropValue::List(vec![PropValue::Vec3([1.0, 1.0, 1.0])]);
        assert!(write_field(w.world_mut(), &reg, e, tp, "points", &good));
        assert_eq!(w.world().entity(e).get::<Spline>().unwrap().points.len(), 1);
    }

    #[test]
    fn entity_ref_round_trips_through_read_and_write() {
        use crate::refs::EntityRef;
        use uuid::Uuid;
        let mut w = EcsWorld::new();
        let e = w.spawn("Body", None);
        w.world_mut().entity_mut(e).insert(Joint3D::default());
        let reg = ComponentRegistry::new();
        let tp = <Joint3D as TypePath>::type_path();

        // Reads as an unbound EntityRef.
        let props = read_entity(w.world(), &reg, e);
        let joint = props.iter().find(|p| p.display == "Joint 3D").unwrap();
        let other = &joint
            .fields
            .iter()
            .find(|f| f.name == "other")
            .unwrap()
            .value;
        assert_eq!(*other, PropValue::EntityRef(None));

        // Write a target guid.
        let g = Uuid::from_u128(0xABC);
        assert!(write_field(
            w.world_mut(),
            &reg,
            e,
            tp,
            "other",
            &PropValue::EntityRef(Some(g)),
        ));
        assert_eq!(
            w.world().entity(e).get::<Joint3D>().unwrap().other,
            EntityRef::new(g)
        );
    }

    #[test]
    fn list_default_element_is_a_default_vec3() {
        let (_w, reg, _e) = world_with_spline();
        let d = default_list_element(&reg, <Spline as TypePath>::type_path(), "points").unwrap();
        assert_eq!(d, PropValue::Vec3([0.0, 0.0, 0.0]));
    }

    /// **The range table names types and fields that exist** (wave VIS1b).
    ///
    /// A table keyed by strings is a table that silently stops applying the day a
    /// field is renamed, and the symptom is a widget quietly reverting to a
    /// step-1 spinner -- nobody files that bug. So the key is checked against the
    /// reflected type rather than trusted.
    #[test]
    fn the_range_table_names_types_that_exist() {
        use crate::components::Material;
        assert_eq!(MATERIAL_TYPE_PATH, <Material as TypePath>::type_path());

        let mut w = EcsWorld::new();
        let e = w.spawn("Lamp", None);
        w.world_mut().entity_mut(e).insert(Material::default());
        let reg = ComponentRegistry::new();
        let props = read_entity(w.world(), &reg, e);
        let mat = props.iter().find(|p| p.display == "Material").unwrap();

        for (tp, field, want) in PROP_RANGES {
            assert_eq!(*tp, MATERIAL_TYPE_PATH);
            let row = mat
                .fields
                .iter()
                .find(|f| f.name == *field)
                .unwrap_or_else(|| {
                    panic!("the range table names `{field}`, which Material has not")
                });
            assert!(
                matches!(row.value, PropValue::Number(_)),
                "`{field}` carries a range and is not a number"
            );
            assert_eq!(row.range.as_ref(), Some(want));
            assert!(
                want.min < want.max && want.step > 0.0,
                "`{field}` range is empty"
            );
        }
        // And a field the table does NOT name still draws as a bare number.
        let colour = mat.fields.iter().find(|f| f.name == "emissive").unwrap();
        assert_eq!(colour.range, None);
    }

    /// **A write that is not finite is refused at the door** (wave VIS1b).
    ///
    /// The door is here rather than in the React number field because this is the
    /// one place all four writers pass through -- Details, the live tuner, the
    /// sequencer's scrub and apply-material -- and because the case that actually
    /// reaches the GPU is invisible to a `Number.isFinite` guard: `1e300` is a
    /// perfectly finite `f64` and is `inf` the moment it is cast to `f32`, after
    /// which `Material::emissive_linear`'s `0.0 * inf` is a NaN packed straight
    /// into the instance buffer.
    #[test]
    fn a_non_finite_number_is_refused_and_leaves_the_value_alone() {
        use crate::components::Material;
        let mut w = EcsWorld::new();
        let e = w.spawn("Lamp", None);
        w.world_mut().entity_mut(e).insert(Material {
            emissive_intensity: 9.0,
            ..Material::default()
        });
        let reg = ComponentRegistry::new();
        let tp = <Material as TypePath>::type_path();

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e300, -1e300] {
            assert!(
                !write_field(
                    w.world_mut(),
                    &reg,
                    e,
                    tp,
                    "emissive_intensity",
                    &PropValue::Number(bad)
                ),
                "{bad} was accepted"
            );
            let m = w.world().entity(e).get::<Material>().unwrap();
            assert_eq!(m.emissive_intensity, 9.0, "{bad} moved the value");
            assert!(m.emissive_linear().iter().all(|c| c.is_finite()));
        }
        // A large but representable value still gets through: the door refuses
        // non-finite numbers, not ambitious ones.
        assert!(write_field(
            w.world_mut(),
            &reg,
            e,
            tp,
            "emissive_intensity",
            &PropValue::Number(1.0e30)
        ));
        assert_eq!(
            w.world()
                .entity(e)
                .get::<Material>()
                .unwrap()
                .emissive_intensity,
            1.0e30
        );
    }

    #[test]
    fn depth_and_len_caps_omit_oversized_data() {
        // A list longer than MAX_LIST_LEN is omitted (read-only by absence) so a
        // write-back can never truncate it.
        let mut w = EcsWorld::new();
        let e = w.spawn("Path", None);
        let big: Vec<Vec3d> = (0..(MAX_LIST_LEN + 1) as u32)
            .map(|i| Vec3d::new(i as f64, 0.0, 0.0))
            .collect();
        w.world_mut().entity_mut(e).insert(Spline {
            points: big,
            closed: false,
            interp: SplineInterp::Linear,
        });
        let reg = ComponentRegistry::new();
        let props = read_entity(w.world(), &reg, e);
        let sp = props.iter().find(|p| p.display == "Spline").unwrap();
        assert!(sp.fields.iter().all(|f| f.name != "points"));
    }
}
