//! Data assets: `.inf_struct`, `.inf_enum`, `.inf_table` (Phase 4.5).
//!
//! These are the strongly-typed, editor-authored data shapes that Blueprints
//! and gameplay read. A struct/enum additionally **generates Rust source** so
//! hand-written and graph code can name the same type (the generated code lands
//! in the user's cargo workspace in P5; here we own the schema + the codegen).
//! Tables hold tabular rows (CSV/JSON-importable — the import lives in the
//! editor layer, which has the parsers).

use serde::{Deserialize, Serialize};

use crate::id::AssetId;
use crate::kind::AssetKind;
use crate::payload::AssetPayload;

/// The type of a struct field / table column. `AssetRef` and `EnumRef` carry the
/// referenced asset's GUID (a dependency edge); the rest are primitives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldType {
    Bool,
    Int,
    Float,
    Text,
    Vec3,
    Color,
    /// A reference to any asset (rendered as an asset-ref picker).
    AssetRef,
    /// A value of a named `.inf_enum` (the referenced enum asset).
    Enum {
        enum_id: AssetId,
        name: String,
    },
}

impl FieldType {
    /// The Rust type this field generates to.
    pub fn rust_type(&self) -> String {
        match self {
            FieldType::Bool => "bool".into(),
            FieldType::Int => "i64".into(),
            FieldType::Float => "f32".into(),
            FieldType::Text => "String".into(),
            FieldType::Vec3 => "Vec3d".into(),
            FieldType::Color => "Color".into(),
            FieldType::AssetRef => "AssetId".into(),
            FieldType::Enum { name, .. } => sanitize_ident(name, "Enum"),
        }
    }

    /// The enum this field references, if any (for dependency edges).
    pub fn enum_ref(&self) -> Option<AssetId> {
        match self {
            FieldType::Enum { enum_id, .. } => Some(*enum_id),
            _ => None,
        }
    }
}

/// One struct field / table column definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: FieldType,
}

/// A `.inf_struct` — a named record of typed fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructAsset {
    pub schema_version: u32,
    pub name: String,
    pub fields: Vec<FieldDef>,
}

impl StructAsset {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            name: name.into(),
            fields: Vec::new(),
        }
    }

    /// GUIDs this struct references (enum fields) — dependency edges.
    pub fn dependencies(&self) -> Vec<AssetId> {
        self.fields.iter().filter_map(|f| f.ty.enum_ref()).collect()
    }

    /// Generate the Rust source for this struct (Reflect + serde derives, to
    /// match the engine's editable-component convention).
    pub fn to_rust(&self) -> String {
        let ty = sanitize_ident(&self.name, "Struct");
        let mut out = String::new();
        out.push_str("#[derive(Debug, Clone, PartialEq, Reflect, Serialize, Deserialize)]\n");
        out.push_str(&format!("pub struct {ty} {{\n"));
        for f in &self.fields {
            out.push_str(&format!(
                "    pub {}: {},\n",
                sanitize_field(&f.name),
                f.ty.rust_type()
            ));
        }
        out.push_str("}\n");
        out
    }
}

impl AssetPayload for StructAsset {
    const KIND: AssetKind = AssetKind::Struct;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

/// A `.inf_enum` — a named set of variants with an editor dropdown binding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumAsset {
    pub schema_version: u32,
    pub name: String,
    pub variants: Vec<String>,
}

impl EnumAsset {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            name: name.into(),
            variants: Vec::new(),
        }
    }

    /// Generate the Rust source for this enum.
    pub fn to_rust(&self) -> String {
        let ty = sanitize_ident(&self.name, "Enum");
        let mut out = String::new();
        out.push_str(
            "#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect, Serialize, Deserialize)]\n",
        );
        out.push_str(&format!("pub enum {ty} {{\n"));
        if self.variants.is_empty() {
            out.push_str("    Unset,\n");
        }
        for v in &self.variants {
            out.push_str(&format!("    {},\n", sanitize_ident(v, "Variant")));
        }
        out.push_str("}\n");
        out
    }
}

impl AssetPayload for EnumAsset {
    const KIND: AssetKind = AssetKind::Enum;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

/// A single table cell value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CellValue {
    Bool {
        value: bool,
    },
    Int {
        value: i64,
    },
    Float {
        value: f64,
    },
    Text {
        value: String,
    },
    /// Asset/enum reference stored as its GUID string (or "" for none).
    Ref {
        value: String,
    },
}

impl CellValue {
    /// Parse a raw string into a cell of the given column type (CSV/JSON import).
    pub fn parse(raw: &str, ty: &FieldType) -> CellValue {
        match ty {
            FieldType::Bool => CellValue::Bool {
                value: matches!(
                    raw.trim().to_ascii_lowercase().as_str(),
                    "true" | "1" | "yes"
                ),
            },
            FieldType::Int => CellValue::Int {
                value: raw.trim().parse().unwrap_or(0),
            },
            FieldType::Float => CellValue::Float {
                value: raw.trim().parse().unwrap_or(0.0),
            },
            FieldType::AssetRef | FieldType::Enum { .. } => CellValue::Ref {
                value: raw.trim().to_string(),
            },
            _ => CellValue::Text {
                value: raw.to_string(),
            },
        }
    }

    /// Render for display / CSV export.
    pub fn as_display(&self) -> String {
        match self {
            CellValue::Bool { value } => value.to_string(),
            CellValue::Int { value } => value.to_string(),
            CellValue::Float { value } => value.to_string(),
            CellValue::Text { value } | CellValue::Ref { value } => value.clone(),
        }
    }
}

/// A `.inf_table` — typed columns + rows of [`CellValue`]s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableAsset {
    pub schema_version: u32,
    pub name: String,
    pub columns: Vec<FieldDef>,
    pub rows: Vec<Vec<CellValue>>,
}

impl TableAsset {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            name: name.into(),
            columns: Vec::new(),
            rows: Vec::new(),
        }
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Add a row from raw strings, coercing each to its column type. Extra values
    /// are ignored; missing ones default.
    pub fn push_row_raw(&mut self, raw: &[String]) {
        let row = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| CellValue::parse(raw.get(i).map(|s| s.as_str()).unwrap_or(""), &c.ty))
            .collect();
        self.rows.push(row);
    }

    /// Every asset GUID this table references (ref/enum cells) — dependency edges.
    pub fn dependencies(&self) -> Vec<AssetId> {
        let mut deps: Vec<AssetId> = self
            .rows
            .iter()
            .flatten()
            .filter_map(|c| match c {
                CellValue::Ref { value } => value.parse().ok(),
                _ => None,
            })
            .collect();
        deps.sort_unstable();
        deps.dedup();
        deps
    }
}

impl AssetPayload for TableAsset {
    const KIND: AssetKind = AssetKind::Table;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

/// Turn an arbitrary string into a `PascalCase` Rust type/variant identifier,
/// falling back to `fallback` if nothing usable remains.
pub fn sanitize_ident(s: &str, fallback: &str) -> String {
    let mut out = String::new();
    let mut upper_next = true;
    for c in s.chars() {
        if c.is_alphanumeric() {
            if upper_next {
                out.extend(c.to_uppercase());
                upper_next = false;
            } else {
                out.push(c);
            }
        } else {
            upper_next = true;
        }
    }
    if out.is_empty() || out.chars().next().unwrap().is_numeric() {
        format!("{fallback}{out}")
    } else {
        out
    }
}

/// Turn a string into a `snake_case` Rust field identifier.
pub fn sanitize_field(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() || trimmed.chars().next().unwrap().is_numeric() {
        format!("field_{trimmed}")
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode, encode};

    #[test]
    fn struct_generates_rust_and_tracks_enum_deps() {
        let enum_id = AssetId::new();
        let mut s = StructAsset::new("Weapon Stats");
        s.fields = vec![
            FieldDef {
                name: "Damage".into(),
                ty: FieldType::Float,
            },
            FieldDef {
                name: "ammo count".into(),
                ty: FieldType::Int,
            },
            FieldDef {
                name: "Element".into(),
                ty: FieldType::Enum {
                    enum_id,
                    name: "Element".into(),
                },
            },
        ];
        let rust = s.to_rust();
        assert!(rust.contains("pub struct WeaponStats"));
        assert!(rust.contains("pub damage: f32,"));
        assert!(rust.contains("pub ammo_count: i64,"));
        assert!(rust.contains("pub element: Element,"));
        assert_eq!(s.dependencies(), vec![enum_id]);
    }

    #[test]
    fn enum_generates_rust() {
        let mut e = EnumAsset::new("Damage Element");
        e.variants = vec!["Fire".into(), "ice cold".into()];
        let rust = e.to_rust();
        assert!(rust.contains("pub enum DamageElement"));
        assert!(rust.contains("Fire,"));
        assert!(rust.contains("IceCold,"));
    }

    #[test]
    fn empty_enum_gets_unset_variant() {
        assert!(EnumAsset::new("E").to_rust().contains("Unset,"));
    }

    #[test]
    fn table_coerces_raw_rows_by_column_type() {
        let mut t = TableAsset::new("Loot");
        t.columns = vec![
            FieldDef {
                name: "name".into(),
                ty: FieldType::Text,
            },
            FieldDef {
                name: "weight".into(),
                ty: FieldType::Float,
            },
            FieldDef {
                name: "rare".into(),
                ty: FieldType::Bool,
            },
        ];
        t.push_row_raw(&["Sword".into(), "2.5".into(), "true".into()]);
        assert_eq!(t.row_count(), 1);
        assert_eq!(t.rows[0][1], CellValue::Float { value: 2.5 });
        assert_eq!(t.rows[0][2], CellValue::Bool { value: true });
    }

    #[test]
    fn data_assets_round_trip() {
        let s = {
            let mut s = StructAsset::new("S");
            s.fields.push(FieldDef {
                name: "x".into(),
                ty: FieldType::Vec3,
            });
            s
        };
        assert_eq!(decode::<StructAsset>(&encode(&s).unwrap()).unwrap(), s);
        let t = TableAsset::new("T");
        assert_eq!(decode::<TableAsset>(&encode(&t).unwrap()).unwrap(), t);
    }

    #[test]
    fn sanitizers_handle_junk() {
        assert_eq!(sanitize_ident("123abc", "T"), "T123abc");
        assert_eq!(sanitize_ident("!!!", "Fallback"), "Fallback");
        assert_eq!(sanitize_field("Some Field!"), "some_field");
        assert_eq!(sanitize_field("9lives"), "field_9lives");
    }
}
