//! Transpiler round-trip coverage for the P11.3 3D character/physics node kit
//! (ROADMAP: every new node type ships with its round-trip case). The exact `d3`
//! mirror of `physics_roundtrip.rs` — because both the lowerer and the transpiler
//! treat a `namespace.name` node generically (path segments → a `::`-joined free
//! call), the `physics3d.*` kit round-trips with **zero** transpiler code beyond
//! this test.
//!
//! Two angles:
//! * `character_kit_3d_graph_round_trips` — a real graph using the exec actions +
//!   the split-output query is lowered, then survives generate → lift unchanged.
//! * `every_physics3d_node_ir_round_trips` — a hand-built IR exercising every
//!   `physics3d.*` call shape (including the `raycast`/`get_velocity` per-field
//!   paths) round-trips generate → lift byte-stable.

use inf_blueprint::lower::lower_graph;
use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
use inf_blueprint::{Binding, BlueprintFn, Expr, Lit, LocalId, Stmt, Ty};
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
fn character_kit_3d_graph_round_trips() {
    // begin_play → set_velocity(e, 4, -2, 3);
    //   move_and_slide(e, get_velocity(e).x, .y, .z)
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let ent = int(&mut g, 1);
    let vx = float(&mut g, 4.0);
    let vy = float(&mut g, -2.0);
    let vz = float(&mut g, 3.0);
    let sv = g.insert("physics3d.set_velocity", NodeUi::default());
    let getv = g.insert("physics3d.get_velocity", NodeUi::default());
    let mas = g.insert("physics3d.move_and_slide", NodeUi::default());

    wire(&mut g, ent, "value", sv, "entity");
    wire(&mut g, vx, "value", sv, "vx");
    wire(&mut g, vy, "value", sv, "vy");
    wire(&mut g, vz, "value", sv, "vz");
    wire(&mut g, ent, "value", getv, "entity");
    wire(&mut g, ent, "value", mas, "entity");
    wire(&mut g, getv, "x", mas, "motion_x");
    wire(&mut g, getv, "y", mas, "motion_y");
    wire(&mut g, getv, "z", mas, "motion_z");
    wire(&mut g, bp, EXEC_THEN, sv, "exec");
    wire(&mut g, sv, EXEC_THEN, mas, "exec");

    let lowered = lower_graph(&g, &reg).expect("lower").pop().unwrap();

    let lifted = relift(&lowered);
    assert_eq!(
        &lifted, &lowered,
        "lowered physics3d IR must survive transpile"
    );
    assert_eq!(
        generate_fn(&lowered).unwrap(),
        generate_fn(&lifted).unwrap(),
        "regeneration must be idempotent"
    );

    let src = generate_fn(&lowered).unwrap();
    assert!(src.contains("physics3d::set_velocity("), "src:\n{src}");
    assert!(src.contains("physics3d::move_and_slide("), "src:\n{src}");
    assert!(src.contains("physics3d::get_velocity::x("), "src:\n{src}");
    assert!(src.contains("physics3d::get_velocity::z("), "src:\n{src}");
}

#[test]
fn every_physics3d_node_ir_round_trips() {
    // A body that uses every physics3d call shape at least once.
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
                &["physics3d", "set_velocity"],
                vec![e(), f.clone(), f.clone(), f.clone()],
            )),
            Stmt::ExprStmt(call(
                &["physics3d", "apply_impulse"],
                vec![e(), f.clone(), f.clone(), f.clone()],
            )),
            // move_and_slide with its grounded output bound to a local
            bind(
                1,
                call(
                    &["physics3d", "move_and_slide"],
                    vec![e(), f.clone(), f.clone(), f.clone()],
                ),
            ),
            // pure single-output query
            bind(2, call(&["physics3d", "is_grounded"], vec![e()])),
            // split get_velocity (x/y/z)
            bind(3, call(&["physics3d", "get_velocity", "x"], vec![e()])),
            bind(4, call(&["physics3d", "get_velocity", "y"], vec![e()])),
            bind(5, call(&["physics3d", "get_velocity", "z"], vec![e()])),
            // split raycast (all seven fields)
            bind(6, call(&["physics3d", "raycast", "hit"], ray_args())),
            bind(7, call(&["physics3d", "raycast", "point_x"], ray_args())),
            bind(8, call(&["physics3d", "raycast", "point_y"], ray_args())),
            bind(9, call(&["physics3d", "raycast", "point_z"], ray_args())),
            bind(10, call(&["physics3d", "raycast", "normal_x"], ray_args())),
            bind(11, call(&["physics3d", "raycast", "normal_y"], ray_args())),
            bind(12, call(&["physics3d", "raycast", "normal_z"], ray_args())),
        ],
    };

    let lifted = relift(&func);
    assert_eq!(lifted, func, "every physics3d call shape must round-trip");
}
