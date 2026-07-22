//! Wave-3 emit coverage: event handlers whose stable key contains a `:` (the
//! `input:<action>` / `custom:<name>` shapes) must lower to a `BlueprintFn` whose
//! `name` is a **valid Rust identifier** — the lowerer sanitizes `:`/`.` to `_`
//! (`input:jump → input_jump`), keeping the raw key only in `id`. Without that,
//! `generate_fn`'s `emit::ident` would reject the name and codegen would fail.

use inf_blueprint::lower::lower_graph;
use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
use inf_graph::{Graph, Link, NodeId, NodeUi, ParamValue};
use inf_transpile::generate_fn;

fn wire(g: &mut Graph, from: NodeId, fp: &str, to: NodeId, tp: &str) {
    g.links.push(Link {
        from,
        from_port: fp.into(),
        to,
        to_port: tp.into(),
    });
}

fn text(g: &mut Graph, node: NodeId, key: &str, val: &str) {
    g.node_mut(node)
        .unwrap()
        .params
        .insert(key.into(), ParamValue::Text(val.into()));
}

/// Lower a single-`event.*`-node graph (the node fires `debug.print("x")`) and
/// return the generated Rust for its one handler.
fn generate_event(type_id: &str, param: Option<(&str, &str)>) -> String {
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let ev = g.insert(type_id, NodeUi::default());
    if let Some((k, v)) = param {
        text(&mut g, ev, k, v);
    }
    let pr = g.insert("debug.print", NodeUi::default());
    text(&mut g, pr, "message", "x");
    wire(&mut g, ev, EXEC_THEN, pr, "exec");
    let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
    generate_fn(&f).expect("generate_fn must accept the sanitized name")
}

#[test]
fn input_event_generates_a_valid_fn_ident() {
    let src = generate_event("event.input", Some(("action", "jump")));
    // The colon in `input:jump` became `_`, and the raw key survives in the attr.
    assert!(src.contains("fn input_jump"), "src:\n{src}");
    assert!(src.contains("id = \"input:jump\""), "src:\n{src}");
    assert!(
        src.contains("pressed"),
        "the pressed param is present\n{src}"
    );
}

#[test]
fn collision_event_generates_a_valid_fn_ident() {
    let src = generate_event("event.collision", None);
    assert!(src.contains("fn collision"), "src:\n{src}");
    assert!(src.contains("other"), "the other param is present\n{src}");
}

#[test]
fn custom_event_generates_a_valid_fn_ident() {
    let src = generate_event("event.custom", Some(("name", "ping")));
    assert!(src.contains("fn custom_ping"), "src:\n{src}");
    assert!(src.contains("id = \"custom:ping\""), "src:\n{src}");
}
