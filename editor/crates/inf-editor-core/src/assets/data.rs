//! Data-asset editing glue (P4.5): convert between the Ring-0 payloads
//! (`StructAsset`/`EnumAsset`/`TableAsset`) and the flat [`DataAssetDto`] the
//! editor speaks, create defaults, save edits, import tables, and expose the
//! Rust codegen.

use std::path::Path;

use inf_asset::data::{CellValue, EnumAsset, FieldDef, FieldType, StructAsset, TableAsset};
use inf_asset::{AssetError, AssetId, AssetKind, Result};

use crate::assets::{table_import, AssetProject};
use crate::ipc::{DataAssetDto, DataFieldDto};

// ── FieldType ↔ DTO ───────────────────────────────────────────────────────

fn field_to_dto(f: &FieldDef) -> DataFieldDto {
    let (ty, enum_id, enum_name) = match &f.ty {
        FieldType::Bool => ("bool", None, None),
        FieldType::Int => ("int", None, None),
        FieldType::Float => ("float", None, None),
        FieldType::Text => ("text", None, None),
        FieldType::Vec3 => ("vec3", None, None),
        FieldType::Color => ("color", None, None),
        FieldType::AssetRef => ("asset_ref", None, None),
        FieldType::Enum { enum_id, name } => {
            ("enum", Some(enum_id.to_string()), Some(name.clone()))
        }
    };
    DataFieldDto {
        name: f.name.clone(),
        ty: ty.to_string(),
        enum_id,
        enum_name,
    }
}

fn field_from_dto(d: &DataFieldDto) -> FieldDef {
    let ty = match d.ty.as_str() {
        "bool" => FieldType::Bool,
        "int" => FieldType::Int,
        "float" => FieldType::Float,
        "vec3" => FieldType::Vec3,
        "color" => FieldType::Color,
        "asset_ref" => FieldType::AssetRef,
        "enum" => match (
            d.enum_id.as_deref().and_then(|s| s.parse().ok()),
            &d.enum_name,
        ) {
            (Some(enum_id), Some(name)) => FieldType::Enum {
                enum_id,
                name: name.clone(),
            },
            _ => FieldType::Text,
        },
        _ => FieldType::Text,
    };
    FieldDef {
        name: d.name.clone(),
        ty,
    }
}

// ── load → DTO ─────────────────────────────────────────────────────────────

/// Build the editor DTO for a data asset, or `None` if `id` isn't a data kind.
pub fn to_dto(project: &AssetProject, id: AssetId) -> Result<Option<DataAssetDto>> {
    let kind = match project.db().get(id) {
        Some(e) => e.kind(),
        None => return Err(AssetError::UnknownAsset(id)),
    };
    let dto = match kind {
        AssetKind::Struct => {
            let s: StructAsset = project.load_payload(id)?;
            DataAssetDto {
                id: id.to_string(),
                kind: "struct".into(),
                name: s.name,
                fields: s.fields.iter().map(field_to_dto).collect(),
                variants: vec![],
                rows: vec![],
            }
        }
        AssetKind::Enum => {
            let e: EnumAsset = project.load_payload(id)?;
            DataAssetDto {
                id: id.to_string(),
                kind: "enum".into(),
                name: e.name,
                fields: vec![],
                variants: e.variants,
                rows: vec![],
            }
        }
        AssetKind::Table => {
            let t: TableAsset = project.load_payload(id)?;
            DataAssetDto {
                id: id.to_string(),
                kind: "table".into(),
                name: t.name,
                fields: t.columns.iter().map(field_to_dto).collect(),
                variants: vec![],
                rows: t
                    .rows
                    .iter()
                    .map(|r| r.iter().map(CellValue::as_display).collect())
                    .collect(),
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(dto))
}

// ── DTO → save ─────────────────────────────────────────────────────────────

/// Persist an edited data asset back to disk.
pub fn save_dto(project: &mut AssetProject, dto: &DataAssetDto) -> Result<()> {
    let id: AssetId = dto
        .id
        .parse()
        .map_err(|e| AssetError::Import(format!("bad id: {e}")))?;
    match dto.kind.as_str() {
        "struct" => {
            let mut s = StructAsset::new(&dto.name);
            s.fields = dto.fields.iter().map(field_from_dto).collect();
            let deps = s.dependencies();
            project.rewrite_payload(id, &s, deps)
        }
        "enum" => {
            let mut e = EnumAsset::new(&dto.name);
            e.variants = dto.variants.clone();
            project.rewrite_payload(id, &e, vec![])
        }
        "table" => {
            let mut t = TableAsset::new(&dto.name);
            t.columns = dto.fields.iter().map(field_from_dto).collect();
            for row in &dto.rows {
                t.push_row_raw(row);
            }
            let deps = t.dependencies();
            project.rewrite_payload(id, &t, deps)
        }
        other => Err(AssetError::Import(format!("not a data kind: {other}"))),
    }
}

/// Create a default data asset of `kind` ("struct"/"enum"/"table") in `dir`.
pub fn create_default(
    project: &mut AssetProject,
    kind: &str,
    dir: &Path,
    name: &str,
) -> Result<AssetId> {
    match kind {
        "struct" => project.write_asset(dir, name, &StructAsset::new(name), None, vec![], None),
        "enum" => {
            let mut e = EnumAsset::new(name);
            e.variants = vec!["VariantA".into(), "VariantB".into()];
            project.write_asset(dir, name, &e, None, vec![], None)
        }
        "table" => {
            let mut t = TableAsset::new(name);
            t.columns = vec![FieldDef {
                name: "Column1".into(),
                ty: FieldType::Text,
            }];
            project.write_asset(dir, name, &t, None, vec![], None)
        }
        other => Err(AssetError::Import(format!(
            "cannot create data kind: {other}"
        ))),
    }
}

/// Import a CSV/JSON file into an existing table asset, replacing its contents.
pub fn import_table_into(project: &mut AssetProject, id: AssetId, source: &Path) -> Result<()> {
    let bytes = std::fs::read(source)?;
    let ext = source.extension().and_then(|e| e.to_str()).unwrap_or("csv");
    let name = project
        .db()
        .get(id)
        .map(|e| e.name.clone())
        .unwrap_or_else(|| "Table".into());
    let table = table_import::import_table(&bytes, &name, ext)?;
    let deps = table.dependencies();
    project.rewrite_payload(id, &table, deps)
}

/// The generated Rust source for a struct/enum (codegen preview / export).
pub fn rust_source(project: &AssetProject, id: AssetId) -> Result<String> {
    let kind = project
        .db()
        .get(id)
        .map(|e| e.kind())
        .ok_or(AssetError::UnknownAsset(id))?;
    Ok(match kind {
        AssetKind::Struct => project.load_payload::<StructAsset>(id)?.to_rust(),
        AssetKind::Enum => project.load_payload::<EnumAsset>(id)?.to_rust(),
        _ => String::from("// Rust generation applies to struct and enum assets."),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_dto_round_trips_through_save() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let d = proj.content_dir("data").unwrap();
        let id = create_default(&mut proj, "struct", &d, "Stats").unwrap();

        let mut dto = to_dto(&proj, id).unwrap().unwrap();
        assert_eq!(dto.kind, "struct");
        dto.fields.push(DataFieldDto {
            name: "Health".into(),
            ty: "float".into(),
            enum_id: None,
            enum_name: None,
        });
        save_dto(&mut proj, &dto).unwrap();

        let back = to_dto(&proj, id).unwrap().unwrap();
        assert_eq!(back.fields.len(), 1);
        assert_eq!(back.fields[0].name, "Health");
        assert!(rust_source(&proj, id).unwrap().contains("pub health: f32,"));
    }

    #[test]
    fn table_import_replaces_contents() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let d = proj.content_dir("data").unwrap();
        let id = create_default(&mut proj, "table", &d, "Enemies").unwrap();

        let csv = d.join("src.csv");
        std::fs::write(&csv, b"name,hp\nGoblin,30\n").unwrap();
        import_table_into(&mut proj, id, &csv).unwrap();

        let dto = to_dto(&proj, id).unwrap().unwrap();
        assert_eq!(dto.fields.len(), 2);
        assert_eq!(dto.rows.len(), 1);
        assert_eq!(dto.rows[0][0], "Goblin");
    }

    #[test]
    fn enum_dto_saves_variants() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let d = proj.content_dir("data").unwrap();
        let id = create_default(&mut proj, "enum", &d, "Element").unwrap();
        let mut dto = to_dto(&proj, id).unwrap().unwrap();
        dto.variants = vec!["Fire".into(), "Ice".into()];
        save_dto(&mut proj, &dto).unwrap();
        assert!(rust_source(&proj, id).unwrap().contains("Fire,"));
    }
}
