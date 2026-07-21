//! Transpiler round-trip coverage for the P12.3 `audio.*` node kit (ROADMAP §12:
//! every new node type ships with its round-trip case).
//!
//! The audio kit is entity-based exec actions (`play`/`stop`/`set_volume`/
//! `set_pitch`). Like every namespace, lowering + emission are registry/path
//! driven, so this exercises the generic machinery end-to-end:
//! * `audio_kit_graph_round_trips` — a real graph lowers, survives generate →
//!   lift unchanged, and emits `audio::*` free calls;
//! * `every_audio_node_ir_round_trips` — a hand-built IR using every `audio.*`
//!   call shape round-trips generate → lift byte-stable.

use inf_blueprint::lower::lower_graph;
use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
use inf_blueprint::{BlueprintFn, Expr, Lit, Stmt, Ty};
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
fn audio_kit_graph_round_trips() {
    // begin_play → audio.play(e); audio.set_volume(e, 0.5); audio.set_pitch(e, 2);
    //              audio.stop(e)
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let ent = int(&mut g, 1);
    let vol = float(&mut g, 0.5);
    let pit = float(&mut g, 2.0);
    let play = g.insert("audio.play", NodeUi::default());
    let setv = g.insert("audio.set_volume", NodeUi::default());
    let setp = g.insert("audio.set_pitch", NodeUi::default());
    let stop = g.insert("audio.stop", NodeUi::default());

    wire(&mut g, ent, "value", play, "entity");
    wire(&mut g, ent, "value", setv, "entity");
    wire(&mut g, vol, "value", setv, "volume");
    wire(&mut g, ent, "value", setp, "entity");
    wire(&mut g, pit, "value", setp, "pitch");
    wire(&mut g, ent, "value", stop, "entity");
    wire(&mut g, bp, EXEC_THEN, play, "exec");
    wire(&mut g, play, EXEC_THEN, setv, "exec");
    wire(&mut g, setv, EXEC_THEN, setp, "exec");
    wire(&mut g, setp, EXEC_THEN, stop, "exec");

    let lowered = lower_graph(&g, &reg).expect("lower").pop().unwrap();

    // Round-trips through the transpiler unchanged, and regeneration is idempotent.
    let lifted = relift(&lowered);
    assert_eq!(&lifted, &lowered, "lowered audio IR must survive transpile");
    assert_eq!(
        generate_fn(&lowered).unwrap(),
        generate_fn(&lifted).unwrap(),
        "regeneration must be idempotent"
    );

    // The generated Rust targets the `audio::*` runtime surface.
    let src = generate_fn(&lowered).unwrap();
    assert!(src.contains("audio::play("), "src:\n{src}");
    assert!(src.contains("audio::set_volume("), "src:\n{src}");
    assert!(src.contains("audio::set_pitch("), "src:\n{src}");
    assert!(src.contains("audio::stop("), "src:\n{src}");
}

#[test]
fn every_audio_node_ir_round_trips() {
    // A body that uses every audio call shape at least once.
    let e = || Expr::Lit(Lit::Int(1));
    let f = || Expr::Lit(Lit::Float(0.5));
    let call = |segs: &[&str], args: Vec<Expr>| Expr::Call {
        path: segs.iter().map(|s| s.to_string()).collect(),
        args,
    };

    let func = BlueprintFn {
        id: "begin_play".into(),
        name: "begin_play".into(),
        params: vec![],
        ret: Ty::Unit,
        body: vec![
            Stmt::ExprStmt(call(&["audio", "play"], vec![e()])),
            Stmt::ExprStmt(call(&["audio", "set_volume"], vec![e(), f()])),
            Stmt::ExprStmt(call(&["audio", "set_pitch"], vec![e(), f()])),
            Stmt::ExprStmt(call(&["audio", "stop"], vec![e()])),
        ],
    };

    let lifted = relift(&func);
    assert_eq!(lifted, func, "every audio call shape must round-trip");
}
