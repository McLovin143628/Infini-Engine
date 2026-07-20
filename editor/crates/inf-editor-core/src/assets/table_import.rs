//! CSV / JSON import into a [`TableAsset`] (P4.5).
//!
//! CSV: the header row becomes the columns (all `Text` — types are refined in
//! the editor), each subsequent row a table row. JSON: an array of objects, the
//! union of keys (first-seen order) becomes the columns.

use inf_asset::data::{CellValue, FieldDef, FieldType, TableAsset};

use crate::assets::AssetError;

/// Import CSV bytes into a fresh table named `name`.
pub fn import_csv(bytes: &[u8], name: &str) -> Result<TableAsset, AssetError> {
    let mut rdr = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let headers = rdr
        .headers()
        .map_err(|e| AssetError::Import(format!("csv header: {e}")))?
        .iter()
        .map(|h| h.to_string())
        .collect::<Vec<_>>();

    let mut table = TableAsset::new(name);
    table.columns = headers
        .iter()
        .map(|h| FieldDef {
            name: h.clone(),
            ty: FieldType::Text,
        })
        .collect();

    for rec in rdr.records() {
        let rec = rec.map_err(|e| AssetError::Import(format!("csv row: {e}")))?;
        let raw: Vec<String> = rec.iter().map(|s| s.to_string()).collect();
        table.push_row_raw(&raw);
    }
    Ok(table)
}

/// Import a JSON array-of-objects into a fresh table named `name`.
pub fn import_json(bytes: &[u8], name: &str) -> Result<TableAsset, AssetError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| AssetError::Import(format!("json: {e}")))?;
    let arr = value
        .as_array()
        .ok_or_else(|| AssetError::Import("json table must be an array of objects".into()))?;

    // Columns = union of object keys, in first-seen order.
    let mut columns: Vec<String> = Vec::new();
    for obj in arr {
        if let Some(map) = obj.as_object() {
            for k in map.keys() {
                if !columns.contains(k) {
                    columns.push(k.clone());
                }
            }
        }
    }

    let mut table = TableAsset::new(name);
    table.columns = columns
        .iter()
        .map(|c| FieldDef {
            name: c.clone(),
            ty: FieldType::Text,
        })
        .collect();

    for obj in arr {
        let map = obj.as_object();
        let row = columns
            .iter()
            .map(|c| {
                let cell = map.and_then(|m| m.get(c));
                CellValue::Text {
                    value: match cell {
                        Some(serde_json::Value::String(s)) => s.clone(),
                        Some(v) if !v.is_null() => v.to_string(),
                        _ => String::new(),
                    },
                }
            })
            .collect();
        table.rows.push(row);
    }
    Ok(table)
}

/// Route by file extension.
pub fn import_table(bytes: &[u8], name: &str, ext: &str) -> Result<TableAsset, AssetError> {
    match ext.to_ascii_lowercase().as_str() {
        "csv" => import_csv(bytes, name),
        "json" => import_json(bytes, name),
        other => Err(AssetError::Import(format!(
            "no table importer for .{other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_header_becomes_columns_and_rows_import() {
        let csv = b"name,hp,speed\nGoblin,30,5\nOrc,80,3\n";
        let t = import_csv(csv, "Enemies").unwrap();
        assert_eq!(t.column_count(), 3);
        assert_eq!(t.columns[1].name, "hp");
        assert_eq!(t.row_count(), 2);
        assert_eq!(
            t.rows[0][0],
            CellValue::Text {
                value: "Goblin".into()
            }
        );
        assert_eq!(t.rows[1][1], CellValue::Text { value: "80".into() });
    }

    #[test]
    fn json_union_of_keys_becomes_columns() {
        let json = br#"[{"name":"A","hp":10},{"name":"B","armor":2}]"#;
        let t = import_json(json, "T").unwrap();
        // serde_json sorts object keys → columns are alphabetical across the
        // union: hp, name (from the first object), then armor (from the second).
        let names: Vec<&str> = t.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["hp", "name", "armor"]);
        assert_eq!(t.row_count(), 2);
        // Row A: hp=10, name=A, armor="" ; Row B: hp="", name=B, armor=2.
        assert_eq!(t.rows[0][0], CellValue::Text { value: "10".into() });
        assert_eq!(
            t.rows[0][2],
            CellValue::Text {
                value: String::new()
            }
        );
        assert_eq!(
            t.rows[1][0],
            CellValue::Text {
                value: String::new()
            }
        );
        assert_eq!(t.rows[1][2], CellValue::Text { value: "2".into() });
    }

    #[test]
    fn json_rejects_non_array() {
        assert!(import_json(b"{\"a\":1}", "T").is_err());
    }
}
