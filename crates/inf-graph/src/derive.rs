//! Auto-derive a [`NodeDef`] from an abstract operation spec.
//!
//! Generalized from GeoCanvas `nodegraph::derive` (which turned every toolbox
//! op into a node for free). A domain that has its own op/function registry —
//! blueprint function libraries, material shader functions — describes each op
//! as an [`OpSpec`] and gets a full node type without hand-writing ports. The
//! scalar-mapping rule is the reusable core: a scalar param becomes a
//! [`ParamDef`] *and* an optional same-named param-pin input, so a wired value
//! overrides the stored default.

use crate::model::ParamValue;
use crate::registry::{NodeDef, ParamDef, PortDef, PortType, UiHint, SINK};

/// One parameter of an [`OpSpec`].
#[derive(Debug, Clone)]
pub enum OpParam {
    /// A scalar value: a param plus (if `pinnable`) an override input pin.
    Scalar {
        name: String,
        ty: PortType,
        default: ParamValue,
        pinnable: bool,
        description: Option<String>,
        min: Option<f64>,
        max: Option<f64>,
    },
    /// A closed choice: a param, never a pin.
    Choice {
        name: String,
        options: Vec<String>,
        default: String,
    },
    /// A pure data input pin (no stored param).
    Input {
        name: String,
        ty: PortType,
        required: bool,
    },
}

impl OpParam {
    pub fn scalar(name: impl Into<String>, ty: PortType, default: ParamValue) -> Self {
        OpParam::Scalar {
            name: name.into(),
            ty,
            default,
            pinnable: true,
            description: None,
            min: None,
            max: None,
        }
    }

    pub fn input(name: impl Into<String>, ty: PortType, required: bool) -> Self {
        OpParam::Input {
            name: name.into(),
            ty,
            required,
        }
    }
}

/// An abstract operation to derive a node from.
#[derive(Debug, Clone)]
pub struct OpSpec {
    pub id: String,
    pub label: String,
    pub category: String,
    pub description: String,
    pub params: Vec<OpParam>,
    pub outputs: Vec<(String, PortType)>,
    /// A terminal op (writes/consumes rather than produces) → `SINK`, no outputs.
    pub sink: bool,
}

/// Derive a [`NodeDef`] from an [`OpSpec`]. `type_id` is `op.<id>`.
pub fn derive_node_def(spec: &OpSpec) -> NodeDef {
    let mut inputs = Vec::new();
    let mut params = Vec::new();

    for p in &spec.params {
        match p {
            OpParam::Scalar {
                name,
                ty,
                default,
                pinnable,
                description,
                min,
                max,
            } => {
                let ui = match default {
                    ParamValue::Bool(_) => UiHint::Toggle,
                    ParamValue::Int(_) | ParamValue::Float(_) => UiHint::Number,
                    _ => UiHint::Text,
                };
                let mut pd = ParamDef {
                    label: crate::registry::PortDef::new(name.clone(), ty.clone()).label,
                    name: name.clone(),
                    description: description.clone(),
                    default: default.clone(),
                    min: *min,
                    max: *max,
                    options: Vec::new(),
                    ui,
                };
                // reuse the registry's title-casing via PortDef; keep the name.
                pd.label = title_from(name);
                params.push(pd);
                if *pinnable {
                    inputs.push(PortDef::new(name.clone(), ty.clone()).param_pin());
                }
            }
            OpParam::Choice {
                name,
                options,
                default,
            } => {
                params.push(ParamDef::choice(
                    name.clone(),
                    options.clone(),
                    default.clone(),
                ));
            }
            OpParam::Input { name, ty, required } => {
                let mut port = PortDef::new(name.clone(), ty.clone());
                port.required = *required;
                inputs.push(port);
            }
        }
    }

    let outputs = if spec.sink {
        Vec::new()
    } else {
        spec.outputs
            .iter()
            .map(|(n, ty)| PortDef::new(n.clone(), ty.clone()))
            .collect()
    };

    let mut def = NodeDef::new(
        format!("op.{}", spec.id),
        spec.label.clone(),
        spec.category.clone(),
    )
    .described(spec.description.clone())
    .with_inputs(inputs)
    .with_outputs(outputs)
    .with_params(params);
    if spec.sink {
        def = def.with_flags(SINK);
    }
    def
}

fn title_from(name: &str) -> String {
    // Same title-casing PortDef uses, reproduced to avoid leaking a private fn.
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
    fn scalar_becomes_param_and_pin() {
        let spec = OpSpec {
            id: "scale".into(),
            label: "Scale".into(),
            category: "math".into(),
            description: "multiply".into(),
            params: vec![OpParam::scalar(
                "factor",
                PortType::Float,
                ParamValue::Float(1.0),
            )],
            outputs: vec![("out".into(), PortType::Float)],
            sink: false,
        };
        let def = derive_node_def(&spec);
        assert_eq!(def.type_id, "op.scale");
        assert!(def.param("factor").is_some());
        // A same-named param-pin input exists.
        let pin = def.input("factor").unwrap();
        assert!(pin.param_pin);
        assert_eq!(def.outputs.len(), 1);
    }

    #[test]
    fn sink_op_has_no_outputs() {
        let spec = OpSpec {
            id: "print".into(),
            label: "Print".into(),
            category: "debug".into(),
            description: String::new(),
            params: vec![],
            outputs: vec![("out".into(), PortType::Str)],
            sink: true,
        };
        let def = derive_node_def(&spec);
        assert!(def.outputs.is_empty());
        assert!(def.has(SINK));
    }
}
