//! The blueprint node kit (ROADMAP P6.5): the palette of node types the visual
//! editor offers, expressed as generic [`inf_graph::NodeDef`]s.
//!
//! Node `type_id`s follow a uniform `namespace.name` convention that the
//! lowerer ([`crate::lower`]) maps straight to IR:
//!
//! - `event.*` — exec entry points (begin_play, tick); their data outputs
//!   become the handler's [`Param`](crate::Param)s.
//! - `lit.*` — literal value producers (a `value` param → data out).
//! - `math.*` / `cmp.*` / `logic.*` — pure data operators → [`Expr`](crate::Expr)
//!   trees (`Binary`/`Unary`).
//! - `var.get` / `var.set` — member-variable access → `vars::get/set` host calls.
//! - `flow.*` — control flow (branch, sequence, return) → `If`/nesting/`Return`.
//! - everything else with exec pins (`engine.*`, `debug.*`) → an `ExprStmt`
//!   host [`Call`](crate::Expr::Call) whose path is the `type_id`'s segments.
//!
//! The exec pin convention: the single exec input is `exec`; the default exec
//! output is `then`; `flow.branch` uses `true`/`false`; `flow.sequence` uses
//! `then0`/`then1`.

use inf_graph::{NodeDef, NodeRegistry, ParamDef, ParamValue, PortDef, PortType, SINK};

/// The exec input port every action/flow node carries.
pub const EXEC_IN: &str = "exec";
/// The default single exec output.
pub const EXEC_THEN: &str = "then";

fn exec_in() -> PortDef {
    PortDef::new(EXEC_IN, PortType::Exec)
}
fn exec_out(name: &str) -> PortDef {
    PortDef::new(name, PortType::Exec)
}

/// Build the standard blueprint node registry.
pub fn blueprint_registry() -> NodeRegistry {
    let mut reg = NodeRegistry::new();
    reg.register_all(event_nodes());
    reg.register_all(literal_nodes());
    reg.register_all(math_nodes());
    reg.register_all(compare_nodes());
    reg.register_all(logic_nodes());
    reg.register_all(variable_nodes());
    reg.register_all(flow_nodes());
    reg.register_all(action_nodes());
    reg
}

fn event_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("event.begin_play", "Begin Play", "events")
            .described("Fires once when the actor enters play.")
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("event.tick", "Tick", "events")
            .described("Fires every frame; `dt` is the delta time in seconds.")
            .with_outputs(vec![
                exec_out(EXEC_THEN),
                PortDef::new("dt", PortType::Float),
            ]),
    ]
}

fn literal_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("lit.float", "Float", "literals")
            .with_outputs(vec![PortDef::new("value", PortType::Float)])
            .with_params(vec![ParamDef::number("value", 0.0)]),
        NodeDef::new("lit.int", "Integer", "literals")
            .with_outputs(vec![PortDef::new("value", PortType::Int)])
            .with_params(vec![ParamDef::int("value", 0)]),
        NodeDef::new("lit.bool", "Boolean", "literals")
            .with_outputs(vec![PortDef::new("value", PortType::Bool)])
            .with_params(vec![ParamDef::toggle("value", false)]),
        NodeDef::new("lit.str", "String", "literals")
            .with_outputs(vec![PortDef::new("value", PortType::Str)])
            .with_params(vec![ParamDef::text("value", "")]),
    ]
}

fn binary_math(id: &str, display: &str) -> NodeDef {
    NodeDef::new(id, display, "math")
        .with_inputs(vec![
            PortDef::new("a", PortType::Float),
            PortDef::new("b", PortType::Float),
        ])
        .with_outputs(vec![PortDef::new("out", PortType::Float)])
}

fn math_nodes() -> Vec<NodeDef> {
    vec![
        binary_math("math.add", "Add (+)"),
        binary_math("math.sub", "Subtract (−)"),
        binary_math("math.mul", "Multiply (×)"),
        binary_math("math.div", "Divide (÷)"),
        binary_math("math.rem", "Remainder (%)"),
    ]
}

fn compare(id: &str, display: &str) -> NodeDef {
    NodeDef::new(id, display, "compare")
        .with_inputs(vec![
            PortDef::new("a", PortType::Wildcard),
            PortDef::new("b", PortType::Wildcard),
        ])
        .with_outputs(vec![PortDef::new("out", PortType::Bool)])
}

fn compare_nodes() -> Vec<NodeDef> {
    vec![
        compare("cmp.eq", "Equal (==)"),
        compare("cmp.ne", "Not Equal (≠)"),
        compare("cmp.lt", "Less (<)"),
        compare("cmp.le", "Less or Equal (≤)"),
        compare("cmp.gt", "Greater (>)"),
        compare("cmp.ge", "Greater or Equal (≥)"),
    ]
}

fn logic_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("logic.and", "And (&&)", "logic")
            .with_inputs(vec![
                PortDef::new("a", PortType::Bool),
                PortDef::new("b", PortType::Bool),
            ])
            .with_outputs(vec![PortDef::new("out", PortType::Bool)]),
        NodeDef::new("logic.or", "Or (||)", "logic")
            .with_inputs(vec![
                PortDef::new("a", PortType::Bool),
                PortDef::new("b", PortType::Bool),
            ])
            .with_outputs(vec![PortDef::new("out", PortType::Bool)]),
        NodeDef::new("logic.not", "Not (!)", "logic")
            .with_inputs(vec![PortDef::new("a", PortType::Bool)])
            .with_outputs(vec![PortDef::new("out", PortType::Bool)]),
    ]
}

fn variable_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("var.get", "Get Variable", "variables")
            .described("Reads a member variable by name.")
            .with_outputs(vec![PortDef::new("value", PortType::Wildcard)])
            .with_params(vec![ParamDef::text("name", "")]),
        NodeDef::new("var.set", "Set Variable", "variables")
            .described("Writes a member variable by name.")
            .with_inputs(vec![exec_in(), PortDef::new("value", PortType::Wildcard)])
            .with_outputs(vec![exec_out(EXEC_THEN)])
            .with_params(vec![ParamDef::text("name", "")]),
    ]
}

fn flow_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("flow.branch", "Branch (if)", "flow")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("condition", PortType::Bool).required(),
            ])
            .with_outputs(vec![exec_out("true"), exec_out("false")]),
        NodeDef::new("flow.sequence", "Sequence", "flow")
            .with_inputs(vec![exec_in()])
            .with_outputs(vec![exec_out("then0"), exec_out("then1")]),
        NodeDef::new("flow.return", "Return", "flow")
            .with_inputs(vec![exec_in(), PortDef::new("value", PortType::Wildcard)])
            .with_flags(SINK),
    ]
}

fn action_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("engine.set_rotation", "Set Rotation", "engine")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("angle", PortType::Float).required(),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("engine.spawn", "Spawn Prefab", "engine")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("prefab", PortType::Str).required(),
            ])
            .with_outputs(vec![
                exec_out(EXEC_THEN),
                PortDef::new("entity", PortType::Int),
            ]),
        NodeDef::new("engine.destroy", "Destroy Entity", "engine")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("entity", PortType::Int).required(),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("debug.print", "Print", "debug")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("message", PortType::Str).required(),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
    ]
}

/// Classify a node `type_id` for the lowerer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    Event,
    Literal,
    /// Pure binary op: `math.*` (arith) or `cmp.*` (comparison) or `logic.and/or`.
    BinaryOp,
    /// `logic.not` — unary.
    NotOp,
    VarGet,
    VarSet,
    Branch,
    Sequence,
    Return,
    /// A pure data-producing call (has outputs, no exec).
    PureCall,
    /// An impure exec action → `ExprStmt(Call)`.
    Action,
}

/// The [`ParamValue`] a literal node stores (helper for lowering).
pub fn literal_value(def_type: &str, params: &inf_graph::ParamMap) -> Option<ParamValue> {
    if def_type.starts_with("lit.") {
        params.get("value").cloned()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_the_kit() {
        let reg = blueprint_registry();
        assert!(reg.get("event.tick").is_some());
        assert!(reg.get("math.add").is_some());
        assert!(reg.get("flow.branch").is_some());
        assert!(reg.get("var.set").is_some());
        assert!(reg.get("engine.set_rotation").is_some());
        // tick exposes an exec output and a dt data output.
        let tick = reg.get("event.tick").unwrap();
        assert!(tick.output(EXEC_THEN).is_some());
        assert_eq!(tick.output("dt").unwrap().ty, PortType::Float);
    }

    #[test]
    fn branch_has_two_exec_outs() {
        let reg = blueprint_registry();
        let b = reg.get("flow.branch").unwrap();
        assert!(b.output("true").is_some() && b.output("false").is_some());
        assert!(b.input("condition").unwrap().required);
    }
}
