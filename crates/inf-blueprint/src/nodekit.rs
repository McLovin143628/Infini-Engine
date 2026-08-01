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
    reg.register_all(dispatch_nodes());
    reg.register_all(physics_nodes());
    reg.register_all(physics3d_nodes());
    reg.register_all(input_nodes());
    reg.register_all(audio_nodes());
    reg.register_all(sky_nodes());
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
        // Wave 3 event entry points. `event.input` fires on the rising/falling
        // edge of the named action (`pressed` = true on press, false on release);
        // `event.collision` fires when a solid contact (or a sensor overlap)
        // begins, carrying the other entity's id; `event.custom` is invoked
        // explicitly (by name) via the dispatcher nodes below. Their `action` /
        // `name` params drive the lowerer's [`EventKind`](crate::EventKind).
        NodeDef::new("event.input", "Input Action", "events")
            .described("Fires on the named input action's edge; `pressed` is true on press, false on release.")
            .with_outputs(vec![
                exec_out(EXEC_THEN),
                PortDef::new("pressed", PortType::Bool),
            ])
            .with_params(vec![ParamDef::text("action", "")]),
        NodeDef::new("event.collision", "On Collision", "events")
            .described("Fires when a contact/overlap begins; `other` is the other entity's id (0 if none).")
            .with_outputs(vec![
                exec_out(EXEC_THEN),
                PortDef::new("other", PortType::Int),
            ]),
        NodeDef::new("event.custom", "Custom Event", "events")
            .described("A user-named event, invoked explicitly through the dispatcher nodes.")
            .with_outputs(vec![exec_out(EXEC_THEN)])
            .with_params(vec![ParamDef::text("name", "")]),
    ]
}

/// The event-dispatcher palette (Wave 3, B-P3). These are impure exec actions the
/// lowerer routes — via the generic [`NodeRole::Action`] path — to the
/// `event::dispatch` / `event::bind` / `event::unbind` host calls the sim
/// implements. `name`/`handler` are **`Str` data input pins** (wire a `lit.str`),
/// mirroring the `input.is_down` key precedent, so they lower with no special
/// casing beyond the `dispatch.* → event::*` path remap in [`crate::lower`].
///
/// - `dispatch.call(target, name)` — announce that entity `target` fired the
///   custom event `name`; the target's own `Custom(name)` handler runs and every
///   bound listener's handler is invoked.
/// - `dispatch.bind(source, name, handler)` — the calling actor subscribes: when
///   `source` fires `name`, run the calling actor's `Custom(handler)` event.
/// - `dispatch.unbind(source, name, handler)` — remove that subscription.
///
/// Dispatchers are **raise-excluded** (like the stateful `flow.*` nodes): they
/// have no single-node inverse in the round-trip image, so hand-edited Rust in
/// that shape stays a snippet. See [`crate::raise`].
fn dispatch_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("dispatch.call", "Call Event", "dispatch")
            .described("Fire the custom event `name` on `target` (and its bound listeners).")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("target", PortType::Int).required(),
                PortDef::new("name", PortType::Str).required(),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("dispatch.bind", "Bind Event", "dispatch")
            .described("Subscribe this actor's `handler` custom event to `source`'s `name` event.")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("source", PortType::Int).required(),
                PortDef::new("name", PortType::Str).required(),
                PortDef::new("handler", PortType::Str).required(),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("dispatch.unbind", "Unbind Event", "dispatch")
            .described("Remove this actor's `handler` subscription to `source`'s `name` event.")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("source", PortType::Int).required(),
                PortDef::new("name", PortType::Str).required(),
                PortDef::new("handler", PortType::Str).required(),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
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

/// A single-input `math.*` op (`a` → `out`, both `Float`).
fn unary_math(id: &str, display: &str) -> NodeDef {
    NodeDef::new(id, display, "math")
        .with_inputs(vec![PortDef::new("a", PortType::Float)])
        .with_outputs(vec![PortDef::new("out", PortType::Float)])
}

/// The math palette (ROADMAP B-P1). The five arithmetic ops + the six unary
/// functions + `min`/`max`/`pow`/`clamp`/`lerp` lower onto the IR: `math.add…rem`
/// and `math.min/max/pow…` that the lowerer recognizes as binary/unary IR nodes
/// stay [`Expr::Binary`](crate::Expr)/[`Expr::Unary`](crate::Expr); everything
/// else lowers to a pure `math::<name>(..)` [`Call`](crate::Expr::Call) backed by
/// [`crate::math_builtins`] (the interpreter and the transpiled Rust share that
/// one implementation — parity by construction). `math.neg` is the unary-negate
/// IR node (`-a`); `to_int`/`to_float` are the scalar type converters.
fn math_nodes() -> Vec<NodeDef> {
    vec![
        // Arithmetic (IR `BinOp`).
        binary_math("math.add", "Add (+)"),
        binary_math("math.sub", "Subtract (−)"),
        binary_math("math.mul", "Multiply (×)"),
        binary_math("math.div", "Divide (÷)"),
        binary_math("math.rem", "Remainder (%)"),
        // Unary functions.
        unary_math("math.neg", "Negate (−)"),
        unary_math("math.abs", "Absolute"),
        unary_math("math.floor", "Floor"),
        unary_math("math.ceil", "Ceil"),
        unary_math("math.round", "Round"),
        unary_math("math.sqrt", "Square Root"),
        unary_math("math.sin", "Sine"),
        unary_math("math.cos", "Cosine"),
        // Binary functions.
        binary_math("math.min", "Min"),
        binary_math("math.max", "Max"),
        binary_math("math.pow", "Power (xʸ)"),
        // Ternary helpers.
        NodeDef::new("math.clamp", "Clamp", "math")
            .described("Constrain x to [min, max] (non-panicking; inverted range yields max).")
            .with_inputs(vec![
                PortDef::new("x", PortType::Float),
                PortDef::new("min", PortType::Float),
                PortDef::new("max", PortType::Float),
            ])
            .with_outputs(vec![PortDef::new("out", PortType::Float)]),
        NodeDef::new("math.lerp", "Lerp", "math")
            .described("Linear interpolation a + (b − a) · t (unclamped t).")
            .with_inputs(vec![
                PortDef::new("a", PortType::Float),
                PortDef::new("b", PortType::Float),
                PortDef::new("t", PortType::Float),
            ])
            .with_outputs(vec![PortDef::new("out", PortType::Float)]),
        // Scalar type converters.
        NodeDef::new("math.to_int", "To Integer", "math")
            .described("Truncate a Float toward zero to an Integer (saturating).")
            .with_inputs(vec![PortDef::new("a", PortType::Float)])
            .with_outputs(vec![PortDef::new("out", PortType::Int)]),
        NodeDef::new("math.to_float", "To Float", "math")
            .described("Widen an Integer to a Float.")
            .with_inputs(vec![PortDef::new("a", PortType::Int)])
            .with_outputs(vec![PortDef::new("out", PortType::Float)]),
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

/// The control-flow palette (branch/sequence/return + the B-P2 loops and
/// stateful gates). Loops lower with a hard iteration cap
/// ([`LOOP_GUARD_MAX`](crate::LOOP_GUARD_MAX)) baked into the IR guard, so a
/// runaway blueprint loop can never hang the interpreter or the shipped game.
/// The stateful nodes (`do_once`/`flip_flop`/`gate`) persist across invocations
/// via the reserved `nodestate::*` host namespace, keyed `__bp_<kind>_<NodeId>`.
///
/// **Delay is out of scope here** (a time-based `flow.delay` needs the sim
/// scheduler's timer wheel + a suspend/resume exec model the pure IR does not
/// yet express) — see ROADMAP for the deferred latent-action follow-up.
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
        NodeDef::new("flow.while", "While Loop", "flow")
            .described("Run `loop_body` while `condition` holds, then `completed` (guarded).")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("condition", PortType::Bool).required(),
            ])
            .with_outputs(vec![exec_out("loop_body"), exec_out("completed")]),
        NodeDef::new("flow.for", "For Loop", "flow")
            .described("Run `loop_body` for `index` in first..=last, then `completed` (guarded).")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("first", PortType::Int),
                PortDef::new("last", PortType::Int),
            ])
            .with_outputs(vec![
                exec_out("loop_body"),
                PortDef::new("index", PortType::Int),
                exec_out("completed"),
            ]),
        NodeDef::new("flow.do_once", "Do Once", "flow")
            .described("Fire `then` on the first entry only; `reset` re-arms it.")
            .with_inputs(vec![exec_in(), exec_out("reset")])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("flow.flip_flop", "Flip Flop", "flow")
            .described("Alternate between `a` and `b` on each entry; `is_a` is the branch taken.")
            .with_inputs(vec![exec_in()])
            .with_outputs(vec![
                exec_out("a"),
                exec_out("b"),
                PortDef::new("is_a", PortType::Bool),
            ]),
        NodeDef::new("flow.gate", "Gate", "flow")
            .described("`enter` reaches `exit` only while open; open/close/toggle set the state.")
            .with_inputs(vec![
                exec_out("enter"),
                exec_out("open"),
                exec_out("close"),
                exec_out("toggle"),
            ])
            .with_outputs(vec![exec_out("exit")])
            .with_params(vec![ParamDef::toggle("start_open", true)]),
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

/// The 3D character-controller / physics kit (P11.3) — the exact `d3` mirror of
/// [`physics_nodes`]. Execution reaches the 3D physics world through the
/// interpreter [`Physics3dHost`](crate::interp::Physics3dHost); the transpiler
/// emits the matching `physics3d::*` free calls. Entities are `Int` pins; 3D
/// vectors are **split `_x`/`_y`/`_z` `Float` pins** (the IR still has no
/// first-class `Vec3` value — the same documented follow-up as the 2D kit).
///
/// Exec actions (`move_and_slide`, `set_velocity`, `apply_impulse`) lower to an
/// `ExprStmt`/`Let` host call; pure queries (`is_grounded`, `raycast`,
/// `get_velocity`) lower to data-pin calls, with multi-component results
/// (`raycast`, `get_velocity`) fanning each output pin to its own
/// `physics3d::<op>::<field>` call — all through the **generic** lowerer
/// ([`crate::lower::role_of`]), no physics3d-specific lowering code.
fn physics3d_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new(
            "physics3d.move_and_slide",
            "Move and Slide (3D)",
            "physics 3d",
        )
        .described("Slide an entity by a 3D motion vector, resolving collisions.")
        .with_inputs(vec![
            exec_in(),
            PortDef::new("entity", PortType::Int).required(),
            PortDef::new("motion_x", PortType::Float),
            PortDef::new("motion_y", PortType::Float),
            PortDef::new("motion_z", PortType::Float),
        ])
        .with_outputs(vec![
            exec_out(EXEC_THEN),
            PortDef::new("grounded", PortType::Bool),
        ]),
        NodeDef::new("physics3d.is_grounded", "Is Grounded (3D)", "physics 3d")
            .described("Whether the entity is currently touching the ground.")
            .with_inputs(vec![PortDef::new("entity", PortType::Int).required()])
            .with_outputs(vec![PortDef::new("grounded", PortType::Bool)]),
        NodeDef::new("physics3d.raycast", "Raycast (3D)", "physics 3d")
            .described("Cast a 3D ray; reports hit + world point + surface normal.")
            .with_inputs(vec![
                PortDef::new("origin_x", PortType::Float),
                PortDef::new("origin_y", PortType::Float),
                PortDef::new("origin_z", PortType::Float),
                PortDef::new("dir_x", PortType::Float),
                PortDef::new("dir_y", PortType::Float),
                PortDef::new("dir_z", PortType::Float),
                PortDef::new("max", PortType::Float),
            ])
            .with_outputs(vec![
                PortDef::new("hit", PortType::Bool),
                PortDef::new("point_x", PortType::Float),
                PortDef::new("point_y", PortType::Float),
                PortDef::new("point_z", PortType::Float),
                PortDef::new("normal_x", PortType::Float),
                PortDef::new("normal_y", PortType::Float),
                PortDef::new("normal_z", PortType::Float),
            ]),
        NodeDef::new("physics3d.set_velocity", "Set Velocity (3D)", "physics 3d")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("entity", PortType::Int).required(),
                PortDef::new("vx", PortType::Float),
                PortDef::new("vy", PortType::Float),
                PortDef::new("vz", PortType::Float),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("physics3d.get_velocity", "Get Velocity (3D)", "physics 3d")
            .with_inputs(vec![PortDef::new("entity", PortType::Int).required()])
            .with_outputs(vec![
                PortDef::new("x", PortType::Float),
                PortDef::new("y", PortType::Float),
                PortDef::new("z", PortType::Float),
            ]),
        NodeDef::new(
            "physics3d.apply_impulse",
            "Apply Impulse (3D)",
            "physics 3d",
        )
        .with_inputs(vec![
            exec_in(),
            PortDef::new("entity", PortType::Int).required(),
            PortDef::new("vx", PortType::Float),
            PortDef::new("vy", PortType::Float),
            PortDef::new("vz", PortType::Float),
        ])
        .with_outputs(vec![exec_out(EXEC_THEN)]),
    ]
}

/// The time-of-day kit (P17.1): read and drive the level clock the sun and moon
/// are a pure function of.
///
/// Four **single-purpose** nodes rather than one getter with two outputs, on
/// purpose: a `PureCall` with more than one data output fans each pin into its
/// own `sky::get::<field>` call (see [`crate::lower`]), which would force
/// three-segment match arms in both hosts. One output each keeps the emitted
/// path a plain `sky::get_time_of_day(…)`.
///
/// Execution reaches the level clock through the ordinary
/// [`Host`](crate::interp::Host) boundary — the `sky.*` namespace routes to a
/// pair of `inf_ecs::sky` seams shared verbatim by the editor's Simulate host and
/// the shipped runtime host — so there is **no IR change**, exactly like
/// `terrain.height_at`. The transpiler emits the matching `sky::*` free calls.
///
/// Units per architecture rule 6: `seconds` is UTC seconds since midnight
/// (`0..86400`, wrapped by the setter); `rate` is simulated clock-seconds per
/// simulated second — dimensionless, `0` = frozen, negative runs time backwards.
fn sky_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("sky.get_time_of_day", "Get Time of Day", "sky")
            .described("The level clock, in UTC seconds since midnight (0..86400).")
            .with_outputs(vec![PortDef::new("seconds", PortType::Float)]),
        NodeDef::new("sky.set_time_of_day", "Set Time of Day", "sky")
            .described("Set the level clock, in seconds since midnight (wraps at 86400).")
            .with_inputs(vec![exec_in(), PortDef::new("seconds", PortType::Float)])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("sky.get_rate", "Get Time Rate", "sky")
            .described("How fast the clock runs: simulated seconds per second (0 = frozen).")
            .with_outputs(vec![PortDef::new("rate", PortType::Float)]),
        NodeDef::new("sky.set_rate", "Set Time Rate", "sky")
            .described("Set how fast the clock runs (0 freezes it; negative runs it backwards).")
            .with_inputs(vec![exec_in(), PortDef::new("rate", PortType::Float)])
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

/// The audio kit (P12.3): entity-based exec actions the Simulate/runtime audio
/// step routes through the `audio.*` accessor into the host `AudioEngine`'s
/// command queue. Each takes the emitter **entity as an `Int` data input** (wire
/// an `engine.self`/entity ref), mirroring the physics kit — so they lower with
/// zero special-casing (`build_call` turns the wired entity + params into the
/// call args, emitting `audio::play(e)` / `audio::set_volume(e, 0.5)` etc.).
///
/// v1 is **entity-based only**: `play`/`stop`/`set_volume`/`set_pitch` act on an
/// entity that carries an `AudioSource` component (clip/bus/params live there).
/// A name-based `audio.play_oneshot(clip: Str)` is **deferred** — it needs an
/// asset-DB name→GUID lookup in the host, which the entity-based path avoids
/// (documented follow-up).
fn audio_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("audio.play", "Play Audio", "audio")
            .described("Start (or restart) the entity's AudioSource clip.")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("entity", PortType::Int).required(),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("audio.stop", "Stop Audio", "audio")
            .described("Stop the entity's currently-playing AudioSource voice.")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("entity", PortType::Int).required(),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("audio.set_volume", "Set Audio Volume", "audio")
            .described("Set the entity's AudioSource base volume (linear).")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("entity", PortType::Int).required(),
                PortDef::new("volume", PortType::Float),
            ])
            .with_outputs(vec![exec_out(EXEC_THEN)]),
        NodeDef::new("audio.set_pitch", "Set Audio Pitch", "audio")
            .described("Set the entity's AudioSource pitch (playback-rate factor).")
            .with_inputs(vec![
                exec_in(),
                PortDef::new("entity", PortType::Int).required(),
                PortDef::new("pitch", PortType::Float),
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
    /// `math.neg` — unary negate (`-a`).
    NegOp,
    VarGet,
    VarSet,
    Branch,
    Sequence,
    Return,
    /// `flow.while` — guarded condition loop.
    WhileLoop,
    /// `flow.for` — guarded `index in first..=last` loop.
    ForLoop,
    /// `flow.do_once` — fire once until reset (state in `nodestate::*`).
    DoOnce,
    /// `flow.flip_flop` — alternate `a`/`b` (state in `nodestate::*`).
    FlipFlop,
    /// `flow.gate` — open/closed exec gate (state in `nodestate::*`).
    Gate,
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
    fn audio_kit_is_registered() {
        let reg = blueprint_registry();
        for id in [
            "audio.play",
            "audio.stop",
            "audio.set_volume",
            "audio.set_pitch",
        ] {
            assert!(reg.get(id).is_some(), "missing audio node {id}");
        }
        // Every audio node is an exec action on an entity.
        let play = reg.get("audio.play").unwrap();
        assert!(play.input(EXEC_IN).is_some());
        assert_eq!(play.input("entity").unwrap().ty, PortType::Int);
        assert!(play.output(EXEC_THEN).is_some());
        // set_volume carries the float value pin.
        let sv = reg.get("audio.set_volume").unwrap();
        assert_eq!(sv.input("volume").unwrap().ty, PortType::Float);
    }

    #[test]
    fn physics3d_kit_is_registered() {
        let reg = blueprint_registry();
        for id in [
            "physics3d.move_and_slide",
            "physics3d.is_grounded",
            "physics3d.raycast",
            "physics3d.set_velocity",
            "physics3d.get_velocity",
            "physics3d.apply_impulse",
        ] {
            assert!(reg.get(id).is_some(), "missing physics3d node {id}");
        }
        // move_and_slide is an exec action with x/y/z motion + a grounded output.
        let mas = reg.get("physics3d.move_and_slide").unwrap();
        assert!(mas.input(EXEC_IN).is_some());
        assert_eq!(mas.input("motion_z").unwrap().ty, PortType::Float);
        assert_eq!(mas.output("grounded").unwrap().ty, PortType::Bool);
        // raycast is a pure query (no exec) with the split 3-component outputs.
        let rc = reg.get("physics3d.raycast").unwrap();
        assert!(rc.input(EXEC_IN).is_none());
        assert_eq!(rc.output("point_z").unwrap().ty, PortType::Float);
        assert_eq!(rc.output("normal_z").unwrap().ty, PortType::Float);
        // get_velocity fans to x/y/z.
        let gv = reg.get("physics3d.get_velocity").unwrap();
        assert_eq!(gv.output("z").unwrap().ty, PortType::Float);
    }

    #[test]
    fn sky_kit_is_registered() {
        let reg = blueprint_registry();
        for id in [
            "sky.get_time_of_day",
            "sky.set_time_of_day",
            "sky.get_rate",
            "sky.set_rate",
        ] {
            assert!(reg.get(id).is_some(), "missing sky node {id}");
        }
        // The getters are pure single-output queries — which is what keeps their
        // lowered call path a bare `sky::get_time_of_day(…)` instead of fanning
        // into `sky::get::<field>` (see `sky_nodes`' docs).
        for id in ["sky.get_time_of_day", "sky.get_rate"] {
            let d = reg.get(id).unwrap();
            assert!(d.input(EXEC_IN).is_none(), "{id} must be pure");
            assert_eq!(d.outputs.len(), 1, "{id} must have one data output");
        }
        assert_eq!(
            reg.get("sky.get_time_of_day")
                .unwrap()
                .output("seconds")
                .unwrap()
                .ty,
            PortType::Float
        );
        // The setters are exec actions taking one Float.
        for id in ["sky.set_time_of_day", "sky.set_rate"] {
            let d = reg.get(id).unwrap();
            assert!(d.input(EXEC_IN).is_some(), "{id} must be an exec action");
            assert!(d.output(EXEC_THEN).is_some());
        }
        assert_eq!(
            reg.get("sky.set_rate").unwrap().input("rate").unwrap().ty,
            PortType::Float
        );
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
    fn wave3_event_and_dispatch_nodes_registered() {
        let reg = blueprint_registry();
        // New event entry points with their param + data-output shape.
        let input = reg.get("event.input").expect("event.input");
        assert!(
            input.param("action").is_some(),
            "input carries an action param"
        );
        assert_eq!(input.output("pressed").unwrap().ty, PortType::Bool);
        let coll = reg.get("event.collision").expect("event.collision");
        assert_eq!(coll.output("other").unwrap().ty, PortType::Int);
        let custom = reg.get("event.custom").expect("event.custom");
        assert!(custom.param("name").is_some());
        // Dispatchers are exec actions whose name/handler are Str data input pins.
        let call = reg.get("dispatch.call").expect("dispatch.call");
        assert!(call.input(EXEC_IN).is_some());
        assert_eq!(call.input("target").unwrap().ty, PortType::Int);
        assert_eq!(call.input("name").unwrap().ty, PortType::Str);
        assert!(call.input("name").unwrap().required);
        let bind = reg.get("dispatch.bind").expect("dispatch.bind");
        assert_eq!(bind.input("handler").unwrap().ty, PortType::Str);
        assert!(reg.get("dispatch.unbind").is_some());
    }

    #[test]
    fn branch_has_two_exec_outs() {
        let reg = blueprint_registry();
        let b = reg.get("flow.branch").unwrap();
        assert!(b.output("true").is_some() && b.output("false").is_some());
        assert!(b.input("condition").unwrap().required);
    }

    #[test]
    fn math_palette_is_registered() {
        let reg = blueprint_registry();
        for id in [
            "math.neg",
            "math.abs",
            "math.floor",
            "math.ceil",
            "math.round",
            "math.sqrt",
            "math.sin",
            "math.cos",
            "math.min",
            "math.max",
            "math.pow",
            "math.clamp",
            "math.lerp",
            "math.to_int",
            "math.to_float",
        ] {
            assert!(reg.get(id).is_some(), "missing math node {id}");
        }
        // Converters carry the right pin types.
        assert_eq!(
            reg.get("math.to_int").unwrap().output("out").unwrap().ty,
            PortType::Int
        );
        assert_eq!(
            reg.get("math.to_float").unwrap().input("a").unwrap().ty,
            PortType::Int
        );
        // Ternaries expose their three inputs.
        let clamp = reg.get("math.clamp").unwrap();
        assert!(
            clamp.input("x").is_some()
                && clamp.input("min").is_some()
                && clamp.input("max").is_some()
        );
    }

    #[test]
    fn flow_palette_is_registered() {
        let reg = blueprint_registry();
        let wh = reg.get("flow.while").expect("flow.while");
        assert!(wh.input("condition").unwrap().required);
        assert!(wh.output("loop_body").is_some() && wh.output("completed").is_some());

        let forn = reg.get("flow.for").expect("flow.for");
        assert_eq!(forn.output("index").unwrap().ty, PortType::Int);
        assert!(forn.output("loop_body").is_some() && forn.output("completed").is_some());

        let once = reg.get("flow.do_once").expect("flow.do_once");
        assert!(once.input("reset").is_some());

        let ff = reg.get("flow.flip_flop").expect("flow.flip_flop");
        assert_eq!(ff.output("is_a").unwrap().ty, PortType::Bool);

        // The gate is entirely entry-port driven (enter/open/close/toggle → exit).
        let gate = reg.get("flow.gate").expect("flow.gate");
        for p in ["enter", "open", "close", "toggle"] {
            assert!(gate.input(p).is_some(), "gate missing input {p}");
        }
        assert!(gate.output("exit").is_some());
        assert!(gate.param("start_open").is_some());
    }
}
