//! Curated per-parameter help + numeric bounds — an optional side table.
//!
//! Generalized from GeoCanvas `nodegraph::param_meta`. When a domain's op
//! descriptors don't carry description/min/max (and editing every call site is
//! impractical), a [`ParamMetaTable`] supplies them by `(op_id, param)` and is
//! merged in at derive time. `min`/`max` here are *live clamps* — only
//! mathematically-invariant bounds belong.

/// Metadata for one parameter.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParamMeta {
    pub description: Option<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl ParamMeta {
    pub fn desc(description: impl Into<String>) -> Self {
        Self {
            description: Some(description.into()),
            ..Default::default()
        }
    }

    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }
}

/// A `(op_id, param_name) → ParamMeta` lookup table.
#[derive(Debug, Clone, Default)]
pub struct ParamMetaTable {
    entries: Vec<(String, String, ParamMeta)>,
}

impl ParamMetaTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, op_id: impl Into<String>, param: impl Into<String>, meta: ParamMeta) {
        self.entries.push((op_id.into(), param.into(), meta));
    }

    pub fn lookup(&self, op_id: &str, param: &str) -> Option<&ParamMeta> {
        self.entries
            .iter()
            .find(|(o, p, _)| o == op_id && p == param)
            .map(|(_, _, m)| m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_by_op_and_param() {
        let mut t = ParamMetaTable::new();
        t.insert(
            "op.buffer",
            "distance",
            ParamMeta::desc("in meters").range(0.0, 1e6),
        );
        let m = t.lookup("op.buffer", "distance").unwrap();
        assert_eq!(m.min, Some(0.0));
        assert_eq!(m.description.as_deref(), Some("in meters"));
        assert!(t.lookup("op.buffer", "missing").is_none());
    }
}
