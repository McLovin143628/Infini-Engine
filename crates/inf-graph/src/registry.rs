//! Node-type definitions and the lookup table that drives everything.
//!
//! De-geo'd port of GeoCanvas `nodegraph::registry`. The type *shapes*
//! ([`PortDef`], [`ParamDef`], [`NodeDef`], [`NodeRegistry`]) are generic; the
//! concrete node kit is supplied by the domain (blueprints, materials, …) and
//! registered at startup. A node's execution is *not* named here — the domain
//! maps `type_id` to a [`NodeExecutor`](crate::exec::NodeExecutor) separately,
//! keeping this crate free of any behavior.

use serde::{Deserialize, Serialize};

use crate::model::{ParamMap, ParamValue};

/// What travels on a wire. De-geo'd: `Exec` is the blueprint execution-flow
/// pin (carries no data), the four scalars are universal, `Named` is any
/// domain type by name (`.inf_enum`/`.inf_struct` types, engine handles like
/// `Entity`/`Transform`, or downstream-domain types such as `Material`), and
/// `Wildcard` is a generic pin resolved to a concrete type by its first
/// connection. Compatibility is by structural match (see [`PortType::compatible_with`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub enum PortType {
    /// Execution-flow pin (white; no data). Blueprint exec wiring.
    Exec,
    Bool,
    Int,
    Float,
    Str,
    /// A named domain type; compatibility is exact-name.
    Named(String),
    /// Generic pin; compatible with anything, resolved on connect.
    Wildcard,
}

impl PortType {
    pub fn is_exec(&self) -> bool {
        matches!(self, PortType::Exec)
    }

    /// Whether a wire may run from an output of `self` into an input of
    /// `other`. Exec only connects to exec; `Wildcard` connects to any
    /// non-exec; scalars and `Named` match structurally.
    pub fn compatible_with(&self, other: &PortType) -> bool {
        match (self, other) {
            (PortType::Exec, PortType::Exec) => true,
            (PortType::Exec, _) | (_, PortType::Exec) => false,
            (PortType::Wildcard, _) | (_, PortType::Wildcard) => true,
            (a, b) => a == b,
        }
    }

    /// A stable string key for theming and hashing (never user-facing prose).
    pub fn key(&self) -> &str {
        match self {
            PortType::Exec => "exec",
            PortType::Bool => "bool",
            PortType::Int => "int",
            PortType::Float => "float",
            PortType::Str => "str",
            PortType::Named(n) => n,
            PortType::Wildcard => "wildcard",
        }
    }
}

/// Which inspector control renders for a parameter.
///
/// # This enum is safe to grow, and here is why
///
/// It is serialized **as its variant name** (`"number"`, `"multiline"`, …), not
/// as a positional index, and it appears only on a [`NodeDef`] — a registry DTO
/// the backend serves fresh on every editor session. Nothing persists a
/// `UiHint`: a saved graph stores [`ParamValue`]s, which are
/// unchanged. So a new variant costs one arm in the frontend's param-editor
/// switch and nothing else — none of the append-only discriminant reasoning that
/// governs `SamplerDef` or a component's bincode layout applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UiHint {
    Number,
    Text,
    Toggle,
    Choice,
    /// A multi-line text area (P19.4). The value is still a
    /// [`ParamValue::Text`]; only the control differs.
    /// Added for the PCG grammar's rule text, which is an authored *document*
    /// living on its node rather than in a panel of its own.
    Multiline,
}

/// One declared port on a node type. Ports live on the *type*, never on the
/// node instance, so adding a port is a zero-migration change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortDef {
    pub name: String,
    pub label: String,
    pub ty: PortType,
    #[serde(default)]
    pub required: bool,
    /// An input pin that overrides a same-named scalar param when wired.
    #[serde(default)]
    pub param_pin: bool,
}

impl PortDef {
    pub fn new(name: impl Into<String>, ty: PortType) -> Self {
        let name = name.into();
        Self {
            label: title_case(&name),
            name,
            ty,
            required: false,
            param_pin: false,
        }
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    pub fn param_pin(mut self) -> Self {
        self.param_pin = true;
        self
    }

    pub fn labeled(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
}

/// Parameter metadata: the default (that the sparse [`ParamMap`] deviates from),
/// live numeric clamps, `Choice` options, and the inspector control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParamDef {
    pub name: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub default: ParamValue,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
    pub ui: UiHint,
}

impl ParamDef {
    pub fn number(name: impl Into<String>, default: f64) -> Self {
        Self::scalar(name, ParamValue::Float(default), UiHint::Number)
    }

    pub fn int(name: impl Into<String>, default: i64) -> Self {
        Self::scalar(name, ParamValue::Int(default), UiHint::Number)
    }

    pub fn toggle(name: impl Into<String>, default: bool) -> Self {
        Self::scalar(name, ParamValue::Bool(default), UiHint::Toggle)
    }

    pub fn text(name: impl Into<String>, default: impl Into<String>) -> Self {
        Self::scalar(name, ParamValue::Text(default.into()), UiHint::Text)
    }

    /// A multi-line text parameter: same [`ParamValue::Text`] storage as
    /// [`text`](Self::text), rendered as a text area (see [`UiHint::Multiline`]).
    pub fn multiline(name: impl Into<String>, default: impl Into<String>) -> Self {
        Self::scalar(name, ParamValue::Text(default.into()), UiHint::Multiline)
    }

    pub fn choice(
        name: impl Into<String>,
        options: Vec<String>,
        default: impl Into<String>,
    ) -> Self {
        let mut p = Self::scalar(name, ParamValue::Enum(default.into()), UiHint::Choice);
        p.options = options;
        p
    }

    fn scalar(name: impl Into<String>, default: ParamValue, ui: UiHint) -> Self {
        let name = name.into();
        Self {
            label: title_case(&name),
            name,
            description: None,
            default,
            min: None,
            max: None,
            options: Vec::new(),
            ui,
        }
    }

    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Behavior/visibility flag bits packed into [`NodeDef::flags`].
pub type NodeFlags = u8;
/// A live-set root: data flows toward these (blueprint output / event terminus).
pub const SINK: NodeFlags = 1 << 0;
/// Hidden from the add-node palette (still valid if present in a document).
pub const HIDDEN: NodeFlags = 1 << 1;
/// Canvas furniture (comment box); never compiled or executed.
pub const COMMENT: NodeFlags = 1 << 2;
/// Forwards its first type-matching input unchanged; spliced out at compile
/// exactly like a disabled node (reroute pins).
pub const PASSTHROUGH: NodeFlags = 1 << 3;

/// The full specification of a node type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeDef {
    pub type_id: String,
    pub display: String,
    pub category: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub inputs: Vec<PortDef>,
    #[serde(default)]
    pub outputs: Vec<PortDef>,
    #[serde(default)]
    pub params: Vec<ParamDef>,
    #[serde(default)]
    pub flags: NodeFlags,
}

impl NodeDef {
    pub fn new(
        type_id: impl Into<String>,
        display: impl Into<String>,
        category: impl Into<String>,
    ) -> Self {
        Self {
            type_id: type_id.into(),
            display: display.into(),
            category: category.into(),
            description: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            params: Vec::new(),
            flags: 0,
        }
    }

    pub fn with_inputs(mut self, inputs: Vec<PortDef>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn with_outputs(mut self, outputs: Vec<PortDef>) -> Self {
        self.outputs = outputs;
        self
    }

    pub fn with_params(mut self, params: Vec<ParamDef>) -> Self {
        self.params = params;
        self
    }

    pub fn with_flags(mut self, flags: NodeFlags) -> Self {
        self.flags |= flags;
        self
    }

    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn has(&self, flag: NodeFlags) -> bool {
        self.flags & flag != 0
    }

    pub fn input(&self, name: &str) -> Option<&PortDef> {
        self.inputs.iter().find(|p| p.name == name)
    }

    pub fn output(&self, name: &str) -> Option<&PortDef> {
        self.outputs.iter().find(|p| p.name == name)
    }

    pub fn param(&self, name: &str) -> Option<&ParamDef> {
        self.params.iter().find(|p| p.name == name)
    }

    /// The input a disabled / passthrough node forwards: the first non-param-pin
    /// input whose type matches the first output's type.
    pub fn first_forwardable_input(&self) -> Option<&PortDef> {
        let out_ty = &self.outputs.first()?.ty;
        self.inputs
            .iter()
            .find(|p| !p.param_pin && p.ty.compatible_with(out_ty))
    }

    /// Fill registry defaults over a sparse [`ParamMap`], producing the dense
    /// effective params a compiled node runs with.
    pub fn resolve(&self, sparse: &ParamMap) -> ParamMap {
        let mut out = ParamMap::new();
        for pd in &self.params {
            let v = sparse
                .get(&pd.name)
                .cloned()
                .unwrap_or_else(|| pd.default.clone());
            out.insert(pd.name.clone(), v);
        }
        out
    }

    /// Drop unknown keys, clamp `Number` params to `[min,max]` (resetting
    /// non-finite to the default), and reset out-of-set `Enum` choices. Keeps
    /// a document self-consistent after registry changes or hostile edits.
    pub fn sanitize(&self, params: &mut ParamMap) {
        params.retain(|k, _| self.param(k).is_some());
        for pd in &self.params {
            let Some(v) = params.get_mut(&pd.name) else {
                continue;
            };
            match v {
                ParamValue::Float(f) => {
                    if !f.is_finite() {
                        *v = pd.default.clone();
                    } else {
                        if let Some(lo) = pd.min {
                            *f = f.max(lo);
                        }
                        if let Some(hi) = pd.max {
                            *f = f.min(hi);
                        }
                    }
                }
                ParamValue::Int(i) => {
                    if let Some(lo) = pd.min {
                        *i = (*i).max(lo as i64);
                    }
                    if let Some(hi) = pd.max {
                        *i = (*i).min(hi as i64);
                    }
                }
                ParamValue::Enum(choice)
                    if !pd.options.is_empty() && !pd.options.contains(choice) =>
                {
                    *v = pd.default.clone();
                }
                _ => {}
            }
        }
    }
}

/// The node-type table. Insertion order is preserved in `ordered` (palette
/// order); lookup is by `type_id`.
#[derive(Debug, Clone, Default)]
pub struct NodeRegistry {
    defs: std::collections::BTreeMap<String, NodeDef>,
    ordered: Vec<String>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or replace) a node type. New ids append to palette order.
    pub fn register(&mut self, def: NodeDef) {
        if !self.defs.contains_key(&def.type_id) {
            self.ordered.push(def.type_id.clone());
        }
        self.defs.insert(def.type_id.clone(), def);
    }

    pub fn register_all(&mut self, defs: impl IntoIterator<Item = NodeDef>) {
        for d in defs {
            self.register(d);
        }
    }

    pub fn get(&self, type_id: &str) -> Option<&NodeDef> {
        self.defs.get(type_id)
    }

    pub fn contains(&self, type_id: &str) -> bool {
        self.defs.contains_key(type_id)
    }

    pub fn len(&self) -> usize {
        self.defs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// Node types in palette (insertion) order.
    pub fn ordered(&self) -> impl Iterator<Item = &NodeDef> {
        self.ordered.iter().filter_map(|id| self.defs.get(id))
    }

    /// The palette/inspector payload for the frontend, in palette order.
    pub fn dtos(&self) -> Vec<NodeDef> {
        self.ordered().cloned().collect()
    }
}

/// `snake_or-kebab case` → `Title Case` for a default label.
fn title_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut cap = true;
    for ch in name.chars() {
        if ch == '_' || ch == '-' || ch == '.' {
            out.push(' ');
            cap = true;
        } else if cap {
            out.extend(ch.to_uppercase());
            cap = false;
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_compatibility() {
        assert!(PortType::Exec.compatible_with(&PortType::Exec));
        assert!(!PortType::Exec.compatible_with(&PortType::Float));
        assert!(PortType::Float.compatible_with(&PortType::Float));
        assert!(!PortType::Float.compatible_with(&PortType::Int));
        assert!(PortType::Wildcard.compatible_with(&PortType::Int));
        assert!(!PortType::Wildcard.compatible_with(&PortType::Exec));
        assert!(PortType::Named("Foo".into()).compatible_with(&PortType::Named("Foo".into())));
        assert!(!PortType::Named("Foo".into()).compatible_with(&PortType::Named("Bar".into())));
    }

    #[test]
    fn resolve_fills_defaults() {
        let def = NodeDef::new("math.add", "Add", "math").with_params(vec![
            ParamDef::number("bias", 1.0),
            ParamDef::number("scale", 2.0),
        ]);
        let mut sparse = ParamMap::new();
        sparse.insert("bias".into(), ParamValue::Float(9.0));
        let resolved = def.resolve(&sparse);
        assert_eq!(resolved.get("bias"), Some(&ParamValue::Float(9.0)));
        assert_eq!(resolved.get("scale"), Some(&ParamValue::Float(2.0)));
    }

    #[test]
    fn sanitize_clamps_and_drops() {
        let def = NodeDef::new("n", "N", "c").with_params(vec![
            ParamDef::number("x", 0.0).range(-1.0, 1.0),
            ParamDef::choice("mode", vec!["a".into(), "b".into()], "a"),
        ]);
        let mut p = ParamMap::new();
        p.insert("x".into(), ParamValue::Float(5.0));
        p.insert("mode".into(), ParamValue::Enum("z".into()));
        p.insert("ghost".into(), ParamValue::Int(1));
        def.sanitize(&mut p);
        assert_eq!(p.get("x"), Some(&ParamValue::Float(1.0)));
        assert_eq!(p.get("mode"), Some(&ParamValue::Enum("a".into())));
        assert!(!p.contains_key("ghost"));
    }

    #[test]
    fn forwardable_input_matches_output_type() {
        let def = NodeDef::new("reroute", "Reroute", "flow")
            .with_inputs(vec![PortDef::new("in", PortType::Float)])
            .with_outputs(vec![PortDef::new("out", PortType::Float)])
            .with_flags(PASSTHROUGH);
        assert_eq!(
            def.first_forwardable_input().map(|p| p.name.as_str()),
            Some("in")
        );
    }

    #[test]
    fn registry_preserves_order() {
        let mut reg = NodeRegistry::new();
        reg.register(NodeDef::new("b", "B", "c"));
        reg.register(NodeDef::new("a", "A", "c"));
        let ids: Vec<_> = reg.ordered().map(|d| d.type_id.as_str()).collect();
        assert_eq!(ids, ["b", "a"]);
    }
}
