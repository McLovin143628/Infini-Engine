//! **P22.3: the `destruct.*` kit and the `Destroyed` event survive the
//! transpiler round trip — and their identifiers are frozen.**
//!
//! The `water_roundtrip` module's argument, one phase on: the lowerer and the
//! generator treat any `namespace.name` node generically (path segments joined
//! with `::`), and an event node's handler id comes from
//! [`EventKind::key`](inf_blueprint::EventKind::key). So a new kit needs **no
//! transpiler code at all** — and this file is what makes that a *claim* rather
//! than an assumption.
//!
//! The frozen-id half matters more here than the round trip. A `.inf_act` is
//! pretty JSON, so `EventKind::Destroyed` is written to disk as the string
//! `"Destroyed"` and its handler is generated as a Rust fn named after
//! `key() == "destroyed"`. Renaming either would silently orphan every handler
//! authored against the old name — and the four node ids are the same contract
//! one layer down, because the lowerer maps `destruct.apply_damage` onto exactly
//! the `destruct::apply_damage` both hosts' two-segment match arms read.

use inf_blueprint::lower::lower_graph;
use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
use inf_blueprint::raise::{event_kind_of, raise_fn};
use inf_blueprint::{Binding, BlueprintFn, EventKind, Expr, Lit, LocalId, Param, Stmt, Ty};
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

/// A `Tick` handler that damages an actor, blasts a radius and branches on
/// whether the wall is still standing — one graph exercising all four nodes,
/// with both queries consumed so neither is dead code on the way through.
#[test]
fn the_destruct_kit_graph_round_trips() {
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let ev = g.insert("event.tick", NodeUi::default());
    let wall = int(&mut g, 7);
    let energy = float(&mut g, 2.0e4); // 20 kJ
    let zero = float(&mut g, 0.0);
    let one = float(&mut g, 1.0);
    let blast = float(&mut g, 5.0e4); // 50 kN·s at 1 m
    let radius = float(&mut g, 6.0);

    let damage = g.insert("destruct.apply_damage", NodeUi::default());
    let impulse = g.insert("destruct.radial_impulse", NodeUi::default());
    let intact = g.insert("destruct.is_intact", NodeUi::default());
    let chunks = g.insert("destruct.chunk_count", NodeUi::default());
    let sink = g.insert("physics3d.set_velocity", NodeUi::default());
    let branch = g.insert("flow.branch", NodeUi::default());

    wire(&mut g, ev, EXEC_THEN, damage, "exec");
    wire(&mut g, wall, "value", damage, "entity");
    wire(&mut g, energy, "value", damage, "energy_j");
    wire(&mut g, damage, EXEC_THEN, impulse, "exec");
    wire(&mut g, zero, "value", impulse, "x");
    wire(&mut g, one, "value", impulse, "y");
    wire(&mut g, zero, "value", impulse, "z");
    wire(&mut g, blast, "value", impulse, "impulse_ns");
    wire(&mut g, radius, "value", impulse, "radius_m");
    // The Int query feeds an exec action's Float pin…
    wire(&mut g, wall, "value", chunks, "entity");
    wire(&mut g, impulse, EXEC_THEN, sink, "exec");
    wire(&mut g, wall, "value", sink, "entity");
    wire(&mut g, chunks, "chunks", sink, "vx");
    // …and the Bool one gates a branch.
    wire(&mut g, wall, "value", intact, "entity");
    wire(&mut g, intact, "intact", branch, "condition");
    wire(&mut g, sink, EXEC_THEN, branch, "exec");

    let lowered = lower_graph(&g, &reg).expect("lower").pop().unwrap();
    let lifted = relift(&lowered);
    assert_eq!(
        &lifted, &lowered,
        "lowered destruct IR must survive transpile"
    );
    assert_eq!(
        generate_fn(&lowered).unwrap(),
        generate_fn(&lifted).unwrap(),
        "regeneration must be idempotent"
    );

    let src = generate_fn(&lowered).unwrap();
    for fragment in [
        "destruct::apply_damage(",
        "destruct::radial_impulse(",
        "destruct::is_intact(",
        "destruct::chunk_count(",
    ] {
        assert!(src.contains(fragment), "missing {fragment} in:\n{src}");
    }
}

/// Each call shape survives the IR round trip on its own — arities 2, 5, 1, 1.
#[test]
fn every_destruct_call_ir_round_trips() {
    let e = || Expr::Lit(Lit::Int(7));
    let f = |v: f64| Expr::Lit(Lit::Float(v));
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
    let func = BlueprintFn {
        id: "tick".into(),
        name: "tick".into(),
        params: vec![Param {
            name: "dt".into(),
            ty: Ty::Float,
        }],
        ret: Ty::Unit,
        body: vec![
            bind(1, call(&["destruct", "apply_damage"], vec![e(), f(2.0e4)])),
            bind(
                2,
                call(
                    &["destruct", "radial_impulse"],
                    vec![f(0.0), f(1.0), f(0.0), f(5.0e4), f(6.0)],
                ),
            ),
            bind(3, call(&["destruct", "is_intact"], vec![e()])),
            bind(4, call(&["destruct", "chunk_count"], vec![e()])),
        ],
    };
    let lifted = relift(&func);
    assert_eq!(lifted, func, "every destruct call shape must round-trip");
}

/// The `Destroyed` event round-trips as a handler, and its data pins and
/// `signature()` are two halves of one contract.
#[test]
fn the_destroyed_event_round_trips() {
    let kind = EventKind::Destroyed;
    let id = kind.key();
    let params = kind.signature();
    assert_eq!(
        params.len(),
        1,
        "{id} must carry `chunks` — the event node's data pins and `signature()` \
         are two halves of one contract"
    );
    assert_eq!(params[0].name, "chunks");
    assert_eq!(params[0].ty, Ty::Int);

    let func = BlueprintFn {
        id: id.clone(),
        name: id.clone(),
        params: params.clone(),
        ret: Ty::Unit,
        body: vec![Stmt::ExprStmt(Expr::Call {
            path: vec!["debug".into(), "print".into()],
            args: vec![Expr::Lit(Lit::Str("gone".into()))],
        })],
    };
    // The IR survives generate → lift…
    assert_eq!(relift(&func), func, "{id} must survive transpile");
    // …and the graph half inverts too: raise → lower is the identity on it.
    let reg = blueprint_registry();
    let graph = raise_fn(&func).expect("raise");
    let relowered = lower_graph(&graph, &reg).expect("lower").pop().unwrap();
    assert_eq!(relowered.id, func.id, "{id} raised to the wrong event node");
    assert_eq!(relowered.params, params, "{id} lost its params on the way");
    assert_eq!(event_kind_of(&id), kind, "{id} must name its own kind");
}

/// **THE FROZEN-ID PIN.** These strings are what a committed `.inf_act` stores,
/// what a generated Rust `fn` is named, and what both hosts' match arms read.
/// Spelled out here so a rename has to be a deliberate act.
#[test]
fn the_destruct_ids_are_frozen() {
    assert_eq!(EventKind::Destroyed.key(), "destroyed");
    let ids = [
        EventKind::BeginPlay.key(),
        EventKind::Tick.key(),
        EventKind::Collision.key(),
        EventKind::WaterEnter.key(),
        EventKind::WaterExit.key(),
        EventKind::WaterSplash.key(),
        EventKind::Destroyed.key(),
    ];
    let mut sorted = ids.to_vec();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        ids.len(),
        "handler ids must be unique: {ids:?}"
    );

    // The node ids are the same contract one layer down.
    let reg = blueprint_registry();
    for id in [
        "destruct.apply_damage",
        "destruct.radial_impulse",
        "destruct.is_intact",
        "destruct.chunk_count",
        "event.destroyed",
    ] {
        assert!(
            reg.get(id).is_some(),
            "the frozen node id `{id}` is not registered"
        );
    }
}
