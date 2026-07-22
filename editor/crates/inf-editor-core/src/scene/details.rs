//! Details-panel projection (P3.3): turn the selection's reflected component
//! properties into the frontend [`DetailsDto`], and translate edits back.
//!
//! Multi-object edit (P3.3.3): the panel shows the component sections shared by
//! *every* selected object; each field is flagged `same` when all selected
//! share its value (the UI shows "—" otherwise). Writes apply to the whole
//! selection.

use inf_ecs::PropValue;

use crate::ipc::{ComponentDto, DetailsDto, PropFieldDto, PropValueDto};
use crate::scene::SceneDoc;

/// Build the Details view for the current selection.
pub fn build(doc: &SceneDoc) -> DetailsDto {
    let sel = doc.selection();
    if sel.is_empty() {
        return DetailsDto {
            selection: Vec::new(),
            name: String::new(),
            kind: String::new(),
            components: Vec::new(),
            multi: false,
        };
    }

    let primary = sel[0];
    let primary_props = doc.entity_props(primary);

    // Component type_paths present on every selected object.
    let others: Vec<Vec<inf_ecs::ComponentProps>> =
        sel.iter().skip(1).map(|g| doc.entity_props(*g)).collect();
    let shared = |type_path: &str| {
        others
            .iter()
            .all(|props| props.iter().any(|c| c.type_path == type_path))
    };

    let components: Vec<ComponentDto> = primary_props
        .iter()
        .filter(|c| shared(&c.type_path))
        .map(|c| ComponentDto {
            type_path: c.type_path.clone(),
            display: c.display.clone(),
            fields: c
                .fields
                .iter()
                .map(|f| {
                    let same = others.iter().all(|props| {
                        props
                            .iter()
                            .find(|oc| oc.type_path == c.type_path)
                            .and_then(|oc| oc.fields.iter().find(|of| of.name == f.name))
                            .map(|of| of.value == f.value)
                            .unwrap_or(false)
                    });
                    PropFieldDto {
                        name: f.name.clone(),
                        label: f.label.clone(),
                        value: to_dto(&f.value),
                        same,
                    }
                })
                .collect(),
        })
        .collect();

    let (name, kind) = if sel.len() == 1 {
        (doc.display_name(primary), doc.kind_of_guid(primary))
    } else {
        (format!("{} selected", sel.len()), String::new())
    };

    DetailsDto {
        selection: sel.iter().map(|g| g.to_string()).collect(),
        name,
        kind,
        components,
        multi: sel.len() > 1,
    }
}

pub fn to_dto(v: &PropValue) -> PropValueDto {
    match v {
        PropValue::Bool(b) => PropValueDto::Bool { value: *b },
        PropValue::Number(n) => PropValueDto::Number { value: *n },
        PropValue::Text(t) => PropValueDto::Text { value: t.clone() },
        PropValue::Vec3(a) => PropValueDto::Vec3 { value: a.to_vec() },
        PropValue::Color(c) => PropValueDto::Color { value: c.to_vec() },
        PropValue::Enum { value, options } => PropValueDto::Enum {
            value: value.clone(),
            options: options.clone(),
        },
        // E-P1 deep editing.
        PropValue::List(elems) => PropValueDto::List {
            value: elems.iter().map(to_dto).collect(),
        },
        PropValue::Struct(pairs) => PropValueDto::Struct {
            fields: pairs
                .iter()
                .map(|(name, value)| PropFieldDto {
                    label: prettify(name),
                    name: name.clone(),
                    value: to_dto(value),
                    // Nested struct rows are single-value; multi-select complex
                    // rows render read-only, so `same` is not consulted here.
                    same: true,
                })
                .collect(),
        },
        PropValue::EntityRef(guid) => PropValueDto::EntityRef {
            value: guid.map(|g| g.to_string()),
        },
    }
}

pub fn from_dto(v: &PropValueDto) -> PropValue {
    let at = |s: &[f64], i: usize| s.get(i).copied().unwrap_or(0.0);
    let atf = |s: &[f32], i: usize| s.get(i).copied().unwrap_or(0.0);
    match v {
        PropValueDto::Bool { value } => PropValue::Bool(*value),
        PropValueDto::Number { value } => PropValue::Number(*value),
        PropValueDto::Text { value } => PropValue::Text(value.clone()),
        PropValueDto::Vec3 { value } => PropValue::Vec3([at(value, 0), at(value, 1), at(value, 2)]),
        PropValueDto::Color { value } => {
            PropValue::Color([atf(value, 0), atf(value, 1), atf(value, 2), atf(value, 3)])
        }
        PropValueDto::Enum { value, .. } => PropValue::Enum {
            value: value.clone(),
            options: Vec::new(),
        },
        PropValueDto::List { value } => PropValue::List(value.iter().map(from_dto).collect()),
        PropValueDto::Struct { fields } => PropValue::Struct(
            fields
                .iter()
                .map(|f| (f.name.clone(), from_dto(&f.value)))
                .collect(),
        ),
        PropValueDto::EntityRef { value } => {
            PropValue::EntityRef(value.as_deref().and_then(|s| s.parse().ok()))
        }
    }
}

/// `base_color` → `Base Color` (nested-struct child labels).
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
    use crate::ipc::SpawnKind;

    #[test]
    fn multi_select_shows_shared_components() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Cube, "A", None);
        let b = doc.edit_create(SpawnKind::Sphere, "B", None);
        doc.select(&[a, b], false);
        let d = build(&doc);
        assert!(d.multi);
        // Both are meshes → Transform, Mesh, Material shared; Light is not.
        let displays: Vec<&str> = d.components.iter().map(|c| c.display.as_str()).collect();
        assert!(displays.contains(&"Transform"));
        assert!(displays.contains(&"Material"));
        assert!(!displays.contains(&"Light"));
    }

    #[test]
    fn dto_round_trips_value_kinds() {
        for v in [
            PropValue::Bool(true),
            PropValue::Number(1.5),
            PropValue::Vec3([1.0, 2.0, 3.0]),
            PropValue::Color([0.1, 0.2, 0.3, 1.0]),
        ] {
            assert_eq!(from_dto(&to_dto(&v)), v);
        }
    }

    #[test]
    fn dto_round_trips_deep_editing_kinds() {
        let guid = uuid::Uuid::from_u128(0xBEEF);
        for v in [
            PropValue::EntityRef(Some(guid)),
            PropValue::EntityRef(None),
            PropValue::List(vec![
                PropValue::Vec3([1.0, 2.0, 3.0]),
                PropValue::Vec3([4.0, 5.0, 6.0]),
            ]),
            PropValue::Struct(vec![
                ("base_color".into(), PropValue::Color([1.0, 0.0, 0.0, 1.0])),
                ("metallic".into(), PropValue::Number(0.5)),
            ]),
        ] {
            assert_eq!(from_dto(&to_dto(&v)), v);
        }
    }

    #[test]
    fn struct_dto_prettifies_child_labels() {
        let dto = to_dto(&PropValue::Struct(vec![(
            "base_color".into(),
            PropValue::Number(1.0),
        )]));
        match dto {
            PropValueDto::Struct { fields } => {
                assert_eq!(fields[0].name, "base_color");
                assert_eq!(fields[0].label, "Base Color");
            }
            other => panic!("expected struct dto, got {other:?}"),
        }
    }
}
