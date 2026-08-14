//! CSV / JSON import into a [`TableAsset`] (P4.5).
//!
//! CSV: the header row becomes the columns (all `Text` — types are refined in
//! the editor), each subsequent row a table row. JSON: an array of objects, the
//! union of keys (first-seen order) becomes the columns.

use inf_asset::data::{CellValue, FieldDef, FieldType, TableAsset};

use crate::assets::AssetError;

/// What an import had to do to the source to make a table of it (C4-42, C4-45).
///
/// A `.inf_table` is gameplay and balance data, and every entry here is a place
/// where the file the author handed over and the asset the engine holds differ.
/// None of these are errors — a spreadsheet with a ragged row is still worth
/// importing — but a silent one is content corruption the author cannot see,
/// which is exactly what "a cell that would not parse becomes 0" was.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TableImportReport {
    /// Human-readable notes, in source order, capped at [`MAX_ADVISORIES`].
    pub advisories: Vec<String>,
    /// How many notes were produced in total, including any past the cap.
    pub total: usize,
}

/// The advisory cap. A malformed 100 k-row CSV must not produce 100 k strings —
/// the sibling advisory caps in this crate use the same shape.
const MAX_ADVISORIES: usize = 32;

impl TableImportReport {
    fn note(&mut self, msg: String) {
        self.total += 1;
        if self.advisories.len() < MAX_ADVISORIES {
            self.advisories.push(msg);
        }
    }
}

/// Import CSV bytes into a fresh table named `name`.
pub fn import_csv(bytes: &[u8], name: &str) -> Result<TableAsset, AssetError> {
    Ok(import_csv_reported(bytes, name)?.0)
}

/// Import CSV bytes, and say what the import had to do to the source.
///
/// # `flexible(true)` is a decision, and it needs to be a visible one (C4-45)
///
/// The reader accepts rows longer or shorter than the header. That is right for
/// a hand-maintained spreadsheet — refusing the whole file over one trailing
/// comma would be worse — but a long row's extra cells are **dropped** and a
/// short row's missing ones are filled with the column type's zero, and neither
/// used to leave a trace anywhere.
pub fn import_csv_reported(
    bytes: &[u8],
    name: &str,
) -> Result<(TableAsset, TableImportReport), AssetError> {
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

    let mut report = TableImportReport::default();
    for (r, rec) in rdr.records().enumerate() {
        let rec = rec.map_err(|e| AssetError::Import(format!("csv row: {e}")))?;
        let raw: Vec<String> = rec.iter().map(|s| s.to_string()).collect();
        // Row 1 is the header, so the first data row is line 2 — the number the
        // author's spreadsheet shows them.
        let line = r + 2;
        if raw.len() != headers.len() {
            report.note(format!(
                "line {line} has {} cells against {} columns; {}",
                raw.len(),
                headers.len(),
                if raw.len() > headers.len() {
                    "the extra cells were dropped"
                } else {
                    "the missing cells were filled with the column's empty value"
                }
            ));
        }
        for (col, issue) in table.push_row_raw_checked(&raw) {
            report.note(format!(
                "line {line}, column {:?}: {:?} is not {} — imported as the column's zero",
                headers.get(col).map(String::as_str).unwrap_or("?"),
                issue.raw,
                issue.wanted
            ));
        }
    }
    Ok((table, report))
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
    Ok(import_table_reported(bytes, name, ext)?.0)
}

/// Route by file extension, keeping the import's advisories (C4-42, C4-45).
pub fn import_table_reported(
    bytes: &[u8],
    name: &str,
    ext: &str,
) -> Result<(TableAsset, TableImportReport), AssetError> {
    match ext.to_ascii_lowercase().as_str() {
        "csv" => import_csv_reported(bytes, name),
        // JSON columns are all `Text` and every value is stringified, so nothing
        // can fail to parse and no row can be ragged — the union of keys IS the
        // column set. An empty report here is a fact, not a gap.
        "json" => Ok((import_json(bytes, name)?, TableImportReport::default())),
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

    /// **C4-42/C4-45 — the CSV importer says what it had to do to the source.**
    ///
    /// `flexible(true)` silently drops a long row's extra cells and pads a short
    /// one, and a cell that would not parse became the column's zero. Neither
    /// left a trace, so a `.inf_table` of gameplay data could differ from the
    /// spreadsheet the author handed over with nothing anywhere to say so.
    #[test]
    fn a_ragged_or_unparseable_csv_reports_what_it_did() {
        // Every column is `Text` on import, so nothing can fail to PARSE here —
        // the raggedness is what this half is about.
        let csv = b"name,hp,speed
Goblin,30,5,extra
Orc,80
Troll,120,2
";
        let (t, report) = import_csv_reported(csv, "Enemies").unwrap();
        assert_eq!(t.row_count(), 3);
        assert_eq!(report.total, 2, "two ragged rows, two notes: {report:?}");
        assert!(report.advisories[0].contains("line 2"), "{report:?}");
        assert!(
            report.advisories[0].contains("dropped"),
            "a long row drops cells: {report:?}"
        );
        assert!(report.advisories[1].contains("line 3"), "{report:?}");
        assert!(
            report.advisories[1].contains("filled"),
            "a short row is padded: {report:?}"
        );

        // A well-formed file produces NOTHING — the control that stops the
        // assertions above passing because everything is reported.
        let (_, clean) = import_csv_reported(
            b"a,b
1,2
3,4
",
            "T",
        )
        .unwrap();
        assert_eq!(clean.total, 0);
        assert!(clean.advisories.is_empty());

        // And the typed half: retype a column, re-import, and the unparseable
        // cells are named with their line and column.
        let mut t = import_csv(
            b"name,hp
Goblin,1 234
Orc,80
",
            "E",
        )
        .unwrap();
        t.columns[1].ty = FieldType::Int;
        let issues = t.push_row_raw_checked(&["Troll".into(), "3,000".into()]);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].0, 1);
        assert_eq!(issues[0].1.raw, "3,000");
    }
}
