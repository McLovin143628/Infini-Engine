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
    reg.register_all(physics_nodes());
    reg.register_all(input_nodes());
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

/// The 2D character-controller / physics kit (P8.3b) — the nodes the platformer
/// uses. Execution reaches the physics world through the interpreter
/// [`Physics2dHost`](crate::interp::Physics2dHost); the transpiler emits the
/// matching `physics2d::*` free calls. Entities are `Int` pins; 2D vectors are
/// **split `_x`/`_y` `Float` pins** because the blueprint IR has no first-class
/// `Vec2` value type yet (the P6 value set is scalar-only) — promoting `Vec2` to
/// an IR value (with round-trip + parity coverage) is a documented follow-up.
///
/// Exec actions (`move_and_slide`, `set_velocity`, `apply_impulse`) lower to an
/// `ExprStmt`/`Let` host call; pure queries (`is_grounded`, `raycast`,
/// `get_velocity`) lower to data-pin calls, with multi-component results
/// (`raycast`, `get_velocity`) fanning each output pin to its own
/// `physics2d::<op>::<field>` call (see [`crate::lower`]).
fn physics_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("physics2d.move_and_slide", "Move and Slide", "physics 2d")
            .described("Slide an entity by a motion vector, resolving collisions.")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("entity", PortType::Int).required(),
                PortDef::new("motion_x", PortType::Float),
                PortDef::new("motion_y", PortType::Float),
            ])
            .with_outputs(vec![
                exec_out(EXEC_THEN),
                PortDef::new("grounded", PortType::Bool),
            ]),
        NodeDef::new("physics2d.is_grounded", "Is Grounded", "physics 2d")
            .described("Whether the entity is currently touching the ground.")
            .with_inputs(vec![PortDef::new("entity", PortType::Int).required()])
            .with_outputs(vec![PortDef::new("grounded", PortType::Bool)]),
        NodeDef::new("physics2d.raycast", "Raycast", "physics 2d")
            .described("Cast a ray; reports hit + world point + surface normal.")
            .with_inputs(vec![
                PortDef::new("origin_x", PortType::Float),
                PortDef::new("origin_y", PortType::Float),
                PortDef::new("dir_x", PortType::Float),
                PortDef::new("dir_y", PortType::Float),
                PortDef::new("max", PortType::Float),
            ])
            .with_outputs(vec![
                PortDef::new("hit", PortType::Bool),
                PortDef::new("point_x", PortType::Float),
                PortDef::new("point_y", PortType::Float),
                PortDef::new("normal_x", PortType::Float),
                PortDef::new("normal_y", PortType::Float),
            ]),
        NodeDef::new("physics2d.set_velocity", "Set Velocity", "physics 2d")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("entity", PortType::Int).required(),
                PortDef::new("vx", PortType::Float),
                PortDef::new("vy", PortType::Float),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("physics2d.get_velocity", "Get Velocity", "physics 2d")
            .with_inputs(vec![PortDef::new("entity", PortType::Int).required()])
            .with_outputs(vec![
                PortDef::new("x", PortType::Float),
                PortDef::new("y", PortType::Float),
            ]),
        NodeDef::new("physics2d.apply_impulse", "Apply Impulse", "physics 2d")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("entity", PortType::Int).required(),
                PortDef::new("vx", PortType::Float),
                PortDef::new("vy", PortType::Float),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
    ]
}

/// The input-state kit (P8.4): pure Bool queries the Simulate loop answers from
/// the focused viewport's keyboard state. Both take the action/key **as a `Str`
/// data input** (wire a `lit.str`) rather than a node param, so they lower with
/// zero special-casing — `PureCall` `build_call` turns the wired key into the
/// call's single argument, emitting `input::is_down("left")` /
/// `input::just_pressed("jump")`. `is_down` is the held state (polled every
/// tick for movement); `just_pressed` is the rising edge (one tick only — the
/// jump trigger). The interpreter routes both through the ordinary
/// [`Host`](crate::interp::Host) boundary (the `input.*` namespace), exactly like
/// `engine.*`; the transpiler emits matching `input::*` free calls a P9 game-loop
/// shim binds.
fn input_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("input.is_down", "Is Key Down", "input")
            .described("True while the named action/key is held.")
            .with_inputs(vec![PortDef::new("key", PortType::Str).required()])
            .with_outputs(vec![PortDef::new("down", PortType::Bool)]),
        NodeDef::new("input.just_pressed", "Was Key Pressed", "input")
            .described("True only on the tick the named action/key went down (rising edge).")
            .with_inputs(vec![PortDef::new("key", PortType::Str).required()])
            .with_outputs(vec![PortDef::new("pressed", PortType::Bool)]),
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
    fn physics_kit_is_registered() {
        let reg = blueprint_registry();
        for id in [
            "physics2d.move_and_slide",
            "physics2d.is_grounded",
            "physics2d.raycast",
            "physics2d.set_velocity",
            "physics2d.get_velocity",
            "physics2d.apply_impulse",
        ] {
            assert!(reg.get(id).is_some(), "missing physics node {id}");
        }
        // move_and_slide is an exec action with a grounded data output.
        let mas = reg.get("physics2d.move_and_slide").unwrap();
        assert!(mas.input(EXEC_IN).is_some());
        assert_eq!(mas.output("grounded").unwrap().ty, PortType::Bool);
        // raycast is a pure query (no exec) with the split vector outputs.
        let rc = reg.get("physics2d.raycast").unwrap();
        assert!(rc.input(EXEC_IN).is_none());
        assert_eq!(rc.output("point_x").unwrap().ty, PortType::Float);
        assert_eq!(rc.output("normal_y").unwrap().ty, PortType::Float);
    }

    #[test]
    fn input_kit_is_registered() {
        let reg = blueprint_registry();
        let down = reg.get("input.is_down").expect("input.is_down");
        assert!(down.input(EXEC_IN).is_none(), "is_down is a pure query");
        assert_eq!(down.input("key").unwrap().ty, PortType::Str);
        assert_eq!(down.output("down").unwrap().ty, PortType::Bool);
        let jp = reg.get("input.just_pressed").expect("input.just_pressed");
        assert_eq!(jp.output("pressed").unwrap().ty, PortType::Bool);
    }

    #[test]
    fn branch_has_two_exec_outs() {
        let reg = blueprint_registry();
        let b = reg.get("flow.branch").unwrap();
        assert!(b.output("true").is_some() && b.output("false").is_some());
        assert!(b.input("condition").unwrap().required);
    }
}
