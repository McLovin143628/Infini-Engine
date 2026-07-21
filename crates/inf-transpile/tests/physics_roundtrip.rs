//! Transpiler round-trip coverage for the P8.3b 2D character/physics node kit
//! (ROADMAP §8: every new node type ships with its round-trip case).
//!
//! Two angles:
//! * `character_kit_graph_round_trips` — a real graph using the exec actions +
//!   a split-output query is lowered, then survives generate → lift unchanged.
//! * `every_physics_node_ir_round_trips` — a hand-built IR that exercises every
//!   `physics2d.*` call shape (including the `raycast`/`get_velocity` per-field
//!   paths) round-trips generate → lift byte-stable.

use inf_blueprint::lower::lower_graph;
use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
use inf_blueprint::{Binding, Param};
use inf_blueprint::{BlueprintFn, Expr, Lit, LocalId, Stmt, Ty};
use inf_graph::{Graph, Link, NodeId, NodeUi, ParamValue};
use inf_transpile::{generate_fn, lift_file, FileEntry};

fn wire(g: &mut Graph, from: NodeId, fp: &str, to: NodeId, tp: &str) {
    g.links.push(Link {
        from,
        from_port: fp.into(),
        to,
        to_port: tp.into(),
    });
}

fn int(g: &mut Graph, v: i64) -> NodeId {
    let n = g.insert("lit.int", NodeUi::default());
    g.node_mut(n)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Int(v));
    n
}

fn float(g: &mut Graph, v: f64) -> NodeId {
    let n = g.insert("lit.float", NodeUi::default());
    g.node_mut(n)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Float(v));
    n
}

fn text(g: &mut Graph, v: &str) -> NodeId {
    let n = g.insert("lit.str", NodeUi::default());
    g.node_mut(n)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Text(v.into()));
    n
}

/// Generate → lift, returning the single lifted blueprint fn.
fn relift(f: &BlueprintFn) -> BlueprintFn {
    let src = generate_fn(f).expect("generate");
    let lifted = lift_file(&src).expect("lift");
    lifted
        .file
        .entries
        .into_iter()
        .find_map(|e| match e {
            FileEntry::Blueprint(bp) => Some(bp),
            FileEntry::Verbatim(_) => None,
        })
        .expect("one blueprint fn")
}

#[test]
fn character_kit_graph_round_trips() {
    // begin_play → set_velocity(e, 4, -2); move_and_slide(e, get_velocity(e).x,
    //                                                     get_velocity(e).y)
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let ent = int(&mut g, 1);
    let vx = float(&mut g, 4.0);
    let vy = float(&mut g, -2.0);
    let sv = g.insert("physics2d.set_velocity", NodeUi::default());
    let getv = g.insert("physics2d.get_velocity", NodeUi::default());
    let mas = g.insert("physics2d.move_and_slide", NodeUi::default());

    wire(&mut g, ent, "value", sv, "entity");
    wire(&mut g, vx, "value", sv, "vx");
    wire(&mut g, vy, "value", sv, "vy");
    wire(&mut g, ent, "value", getv, "entity");
    wire(&mut g, ent, "value", mas, "entity");
    wire(&mut g, getv, "x", mas, "motion_x");
    wire(&mut g, getv, "y", mas, "motion_y");
    wire(&mut g, bp, EXEC_THEN, sv, "exec");
    wire(&mut g, sv, EXEC_THEN, mas, "exec");

    let lowered = lower_graph(&g, &reg).expect("lower").pop().unwrap();

    // Round-trips through the transpiler unchanged, and regeneration is idempotent.
    let lifted = relift(&lowered);
    assert_eq!(
        &lifted, &lowered,
        "lowered physics IR must survive transpile"
    );
    assert_eq!(
        generate_fn(&lowered).unwrap(),
        generate_fn(&lifted).unwrap(),
        "regeneration must be idempotent"
    );

    // The generated Rust targets the `physics2d::*` runtime surface, with the
    // split-vector query fanned to per-field calls.
    let src = generate_fn(&lowered).unwrap();
    assert!(src.contains("physics2d::set_velocity("), "src:\n{src}");
    assert!(src.contains("physics2d::move_and_slide("), "src:\n{src}");
    assert!(src.contains("physics2d::get_velocity::x("), "src:\n{src}");
    assert!(src.contains("physics2d::get_velocity::y("), "src:\n{src}");
}

/// The P8.4 input-state kit (`input.is_down`, `input.just_pressed`): a graph
/// gating a jump on the `jump` action, plus a `left`/`right` `is_down` read,
/// lowers and survives generate → lift, emitting `input::*` free calls.
#[test]
fn input_kit_graph_round_trips() {
    let reg = blueprint_registry();
    let mut g = Graph::empty();

    let tick = g.insert("event.tick", NodeUi::default());
    let ent = int(&mut g, 1);
    // input.just_pressed("jump") → gate a move_and_slide with a small vertical.
    let jump_key = text(&mut g, "jump");
    let jp = g.insert("input.just_pressed", NodeUi::default());
    let right_key = text(&mut g, "right");
    let down = g.insert("input.is_down", NodeUi::default());
    let br = g.insert("flow.branch", NodeUi::default());
    let mas = g.insert("physics2d.move_and_slide", NodeUi::default());

    wire(&mut g, jump_key, "value", jp, "key");
    wire(&mut g, right_key, "value", down, "key");
    // Branch on just_pressed("jump").
    wire(&mut g, jp, "pressed", br, "condition");
    wire(&mut g, tick, EXEC_THEN, br, "exec");
    // True arm: move. `down` feeds motion_x (bool→the transpiler emits the call).
    wire(&mut g, ent, "value", mas, "entity");
    wire(&mut g, down, "down", mas, "motion_x");
    wire(&mut g, br, "true", mas, "exec");

    let lowered = lower_graph(&g, &reg).expect("lower").pop().unwrap();
    let lifted = relift(&lowered);
    assert_eq!(&lifted, &lowered, "input-kit IR must survive transpile");

    let src = generate_fn(&lowered).unwrap();
    assert!(src.contains("input::just_pressed(\"jump\")"), "src:\n{src}");
    assert!(src.contains("input::is_down(\"right\")"), "src:\n{src}");
}

#[test]
fn every_physics_node_ir_round_trips() {
    // A body that uses every physics2d call shape at least once.
    let e = || Expr::Lit(Lit::Int(1));
    let f = Expr::Lit(Lit::Float(0.0));
    let call = |segs: &[&str], args: Vec<Expr>| Expr::Call {
        path: segs.iter().map(|s| s.to_string()).collect(),
        args,
    };
    let bind = |id: u32, value: Expr| Stmt::Let {
        id: LocalId(id),
        binding: Binding::Anon,
        ty: None,
        mutable: false,
        value,
    };
    let ray_args = || {
        vec![
            f.clone(),
            f.clone(),
            f.clone(),
            f.clone(),
            Expr::Lit(Lit::Float(10.0)),
        ]
    };

    let func = BlueprintFn {
        id: "begin_play".into(),
        name: "begin_play".into(),
        params: vec![],
        ret: Ty::Unit,
        body: vec![
            // exec actions
            Stmt::ExprStmt(call(
                &["physics2d", "set_velocity"],
                vec![e(), f.clone(), f.clone()],
            )),
            Stmt::ExprStmt(call(
                &["physics2d", "apply_impulse"],
                vec![e(), f.clone(), f.clone()],
            )),
            // move_and_slide with its grounded output bound to a local
            bind(
                1,
                call(
                    &["physics2d", "move_and_slide"],
                    vec![e(), f.clone(), f.clone()],
                ),
            ),
            // pure single-output query
            bind(2, call(&["physics2d", "is_grounded"], vec![e()])),
            // split get_velocity
            bind(3, call(&["physics2d", "get_velocity", "x"], vec![e()])),
            bind(4, call(&["physics2d", "get_velocity", "y"], vec![e()])),
            // split raycast (all five fields)
            bind(5, call(&["physics2d", "raycast", "hit"], ray_args())),
            bind(6, call(&["physics2d", "raycast", "point_x"], ray_args())),
            bind(7, call(&["physics2d", "raycast", "point_y"], ray_args())),
            bind(8, call(&["physics2d", "raycast", "normal_x"], ray_args())),
            bind(9, call(&["physics2d", "raycast", "normal_y"], ray_args())),
        ],
    };

    let _ = Param {
        name: "x".into(),
        ty: Ty::Float,
    };
    let lifted = relift(&func);
    assert_eq!(lifted, func, "every physics2d call shape must round-trip");
}
