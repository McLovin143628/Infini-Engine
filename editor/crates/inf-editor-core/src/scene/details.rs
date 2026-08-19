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

/// The reflect type path of the ECS `Material` component — the section the
/// binding row is appended to.
const MATERIAL_TYPE_PATH: &str = "inf_ecs::components::Material";

/// The synthetic row key for `Material::asset` (P26.3b). Named the same as the
/// field so a future picker's write path is the obvious one.
const MATERIAL_ASSET_FIELD: &str = "asset";

/// The reflect type paths of the two mesh components whose asset links are
/// `#[reflect(ignore)]` and therefore invisible to the walker (Wave E).
const MESH_TYPE_PATH: &str = "inf_ecs::components::MeshRef";
const SKELETAL_MESH_TYPE_PATH: &str = "inf_ecs::components::SkeletalMesh";
/// `ActorClass` is not reflected at all, so its section is synthesized whole.
/// The path is spelled like a component path so the frontend's section keying
/// (and any future collapse-state persistence) treats it like every other one.
const ACTOR_CLASS_TYPE_PATH: &str = "inf_ecs::components::ActorClass";
/// The synthetic row key for `MeshRef::asset`.
const MESH_ASSET_FIELD: &str = "asset";

/// One read-only asset-reference row.
///
/// `same` is computed the way the walked fields compute it — across the WHOLE
/// selection, so a multi-select of two props on different meshes shows "—"
/// rather than the first one's, which is the bug a hand-rolled second copy of
/// this logic would eventually have.
fn asset_row(
    name: &str,
    label: &str,
    asset_kind: &str,
    value: Option<uuid::Uuid>,
    sel: &[uuid::Uuid],
    read: impl Fn(uuid::Uuid) -> Option<uuid::Uuid>,
) -> PropFieldDto {
    PropFieldDto {
        name: name.into(),
        label: label.into(),
        value: PropValueDto::AssetRef {
            value: value.map(|g| g.to_string()),
            asset_kind: asset_kind.into(),
        },
        same: sel.iter().skip(1).all(|g| read(*g) == value),
    }
}

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

    let mut components: Vec<ComponentDto> = primary_props
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

    // ── P26.3b: the persisted `.inf_mat` binding ─────────────────────────────
    //
    // Appended rather than walked, because the reflection walker cannot see it:
    // `Material::asset` is `#[reflect(ignore)]`, exactly as `MeshRef::asset` is,
    // so a level could carry a binding the Details panel had no way to show. It
    // is read-only this batch — the picker is the standing gap named on
    // [`PropValueDto::AssetRef`] — and the row is what makes the binding
    // *visible* rather than a fact only the cook knows.
    if let Some(mat) = components
        .iter_mut()
        .find(|c| c.type_path == MATERIAL_TYPE_PATH)
    {
        let binding = doc.material_asset_of(primary);
        let same = sel
            .iter()
            .skip(1)
            .all(|g| doc.material_asset_of(*g) == binding);
        mat.fields.push(PropFieldDto {
            name: MATERIAL_ASSET_FIELD.into(),
            label: "Material Asset".into(),
            value: PropValueDto::AssetRef {
                value: binding.map(|g| g.to_string()),
                asset_kind: "material".into(),
            },
            same,
        });
    }

    // ── Wave E: the mesh / rig / class bindings, on the same terms ───────────
    //
    // The P26.3b row above was the ONLY escape hatch from `#[reflect(ignore)]`
    // in the whole Details projection, so a user could see which material an
    // actor wore and had no way to see which *mesh* it was — the fact the
    // "Edit in Model Editor" routing is built on. Same mechanism, same
    // read-only scope, one helper so the four rows cannot drift apart.
    if let Some(mesh) = components
        .iter_mut()
        .find(|c| c.type_path == MESH_TYPE_PATH)
    {
        mesh.fields.push(asset_row(
            MESH_ASSET_FIELD,
            "Mesh Asset",
            "mesh",
            doc.mesh_asset_of(primary),
            sel,
            |g| doc.mesh_asset_of(g),
        ));
    }
    if let Some(skel) = components
        .iter_mut()
        .find(|c| c.type_path == SKELETAL_MESH_TYPE_PATH)
    {
        skel.fields.push(asset_row(
            "mesh",
            "Skeletal Mesh Asset",
            "mesh",
            doc.skeletal_mesh_of(primary).and_then(|(m, _)| m),
            sel,
            |g| doc.skeletal_mesh_of(g).and_then(|(m, _)| m),
        ));
        skel.fields.push(asset_row(
            "skeleton",
            "Skeleton Asset",
            "skeleton",
            doc.skeletal_mesh_of(primary).and_then(|(_, s)| s),
            sel,
            |g| doc.skeletal_mesh_of(g).and_then(|(_, s)| s),
        ));
    }

    // `ActorClass` is not reflected AT ALL, so unlike the three above there is no
    // component section to append to — the whole section is synthesized, and only
    // when a class is bound. Nothing else in Details shows a level's blueprint
    // bindings today.
    let actor_class = doc.actor_class_of(primary);
    if actor_class.is_some() || sel.iter().any(|g| doc.actor_class_of(*g).is_some()) {
        components.push(ComponentDto {
            type_path: ACTOR_CLASS_TYPE_PATH.into(),
            display: "Blueprint Class".into(),
            fields: vec![asset_row(
                "class",
                "Blueprint",
                "blueprint",
                actor_class,
                sel,
                |g| doc.actor_class_of(g),
            )],
        });
    }

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
        // An asset-ref row is a READ of a `#[reflect(ignore)]` field, so there is
        // no reflected field on the other side of a write: `edit_set_prop` finds
        // nothing named `asset` and no-ops. Mapped to text rather than given a
        // `PropValue` variant of its own for exactly that reason — a variant
        // would advertise a write path that does not exist.
        PropValueDto::AssetRef { value, .. } => PropValue::Text(value.clone().unwrap_or_default()),
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
