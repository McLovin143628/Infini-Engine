//! The full forward chain: a visual blueprint graph → lowered `BlueprintFn` IR
//! → generated Rust → lifted IR, asserting the IR survives the transpiler
//! (ROADMAP P6.5). This closes the loop the interpreter and the compiled Rust
//! both depend on: the lowered IR is exactly what ships.

use inf_blueprint::lower::lower_graph;
use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
use inf_blueprint::{BlueprintFn, Stmt};
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

fn text(g: &mut Graph, n: NodeId, key: &str, val: &str) {
    g.node_mut(n)
        .unwrap()
        .params
        .insert(key.into(), ParamValue::Text(val.into()));
}

/// Lift the single fn back out of a generated projection.
fn relift(f: &BlueprintFn) -> BlueprintFn {
    let src = generate_fn(f).expect("generate");
    let lifted = lift_file(&src).expect("lift");
    let fns: Vec<_> = lifted
        .file
        .entries
        .into_iter()
        .filter_map(|e| match e {
            FileEntry::Blueprint(bp) => Some(bp),
            FileEntry::Verbatim(_) => None,
        })
        .collect();
    assert_eq!(fns.len(), 1, "expected exactly one lifted blueprint fn");
    fns.into_iter().next().unwrap()
}

#[test]
fn rotate_on_tick_graph_lowers_and_round_trips() {
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let tick = g.insert("event.tick", NodeUi::default());
    let speed = g.insert("var.get", NodeUi::default());
    text(&mut g, speed, "name", "speed");
    let mul = g.insert("math.mul", NodeUi::default());
    let angle = g.insert("var.get", NodeUi::default());
    text(&mut g, angle, "name", "angle");
    let add = g.insert("math.add", NodeUi::default());
    let setv = g.insert("var.set", NodeUi::default());
    text(&mut g, setv, "name", "angle");
    let rot = g.insert("engine.set_rotation", NodeUi::default());

    wire(&mut g, speed, "value", mul, "a");
    wire(&mut g, tick, "dt", mul, "b");
    wire(&mut g, angle, "value", add, "a");
    wire(&mut g, mul, "out", add, "b");
    wire(&mut g, add, "out", setv, "value");
    wire(&mut g, add, "out", rot, "angle");
    wire(&mut g, tick, EXEC_THEN, setv, "exec");
    wire(&mut g, setv, EXEC_THEN, rot, "exec");

    let fns = lower_graph(&g, &reg).expect("lower");
    assert_eq!(fns.len(), 1);
    let lowered = &fns[0];

    // The lowered IR round-trips through generate → lift byte-for-byte-stable.
    let lifted = relift(lowered);
    assert_eq!(&lifted, lowered, "lowered IR must survive the transpiler");

    // And regeneration is idempotent (watch-loop convergence).
    assert_eq!(
        generate_fn(lowered).unwrap(),
        generate_fn(&lifted).unwrap(),
        "regeneration must be idempotent"
    );

    // Sanity: the generated Rust mentions the engine call and the member vars.
    let src = generate_fn(lowered).unwrap();
    assert!(src.contains("engine::set_rotation"), "src:\n{src}");
    assert!(src.contains("vars::set"), "src:\n{src}");
    assert!(src.contains("fn tick"), "src:\n{src}");
}

#[test]
fn branch_graph_lowers_to_if() {
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let a = g.insert("lit.float", NodeUi::default());
    g.node_mut(a)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Float(2.0));
    let b = g.insert("lit.float", NodeUi::default());
    g.node_mut(b)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Float(1.0));
    let gt = g.insert("cmp.gt", NodeUi::default());
    let br = g.insert("flow.branch", NodeUi::default());
    let pt = g.insert("debug.print", NodeUi::default());
    text(&mut g, pt, "message", "hi");

    wire(&mut g, a, "value", gt, "a");
    wire(&mut g, b, "value", gt, "b");
    wire(&mut g, gt, "out", br, "condition");
    wire(&mut g, bp, EXEC_THEN, br, "exec");
    wire(&mut g, br, "true", pt, "exec");

    let fns = lower_graph(&g, &reg).unwrap();
    let lowered = &fns[0];
    assert!(matches!(lowered.body[0], Stmt::If { .. }));

    let lifted = relift(lowered);
    assert_eq!(&lifted, lowered);
    let src = generate_fn(lowered).unwrap();
    assert!(src.contains("if "), "src:\n{src}");
    assert!(src.contains("debug::print"), "src:\n{src}");
}
