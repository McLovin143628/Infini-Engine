//! Transpiler round-trip coverage for the P20.2 water kit — the permanent
//! discipline (every new node type ships with its round-trip case), applied to
//! the `water.*` queries **and** to the three new `event.water_*` entry points.
//!
//! Like `physics3d_roundtrip.rs`, this needs **zero** transpiler code: the
//! lowerer and the generator both treat a `namespace.name` node generically (path
//! segments → a `::`-joined free call), and an event node's handler id is carried
//! by [`inf_blueprint::EventKind::key`]. The test is what makes that a *claim*
//! rather than an assumption.
//!
//! Three angles:
//! * `the_water_kit_graph_round_trips` — a real graph wiring all three queries
//!   survives lower → generate → lift unchanged.
//! * `every_water_event_round_trips` — each of `water_enter` / `water_exit` /
//!   `water_splash` survives with **both** of its params, which is the half that
//!   would break silently if `signature()` and the node's output pins ever
//!   disagreed.
//! * `the_water_event_ids_are_frozen` — the handler ids that end up in committed
//!   `.inf_act` files, spelled out.

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

#[test]
fn the_water_kit_graph_round_trips() {
    // On Enter Water(water, speed) → debug.print(is_in_water(self)
    //                                            + submerged_fraction(self)
    //                                            + surface_height(x, z))
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let ev = g.insert("event.water_enter", NodeUi::default());
    let ent = int(&mut g, 1);
    let x = float(&mut g, 12.5);
    let z = float(&mut g, -7.25);
    let is_in = g.insert("water.is_in_water", NodeUi::default());
    let frac = g.insert("water.submerged_fraction", NodeUi::default());
    let height = g.insert("water.surface_height", NodeUi::default());
    // An exec action that consumes two Floats, so neither query is dead code on
    // the way through.
    let sink = g.insert("physics3d.set_velocity", NodeUi::default());
    let branch = g.insert("flow.branch", NodeUi::default());

    wire(&mut g, ent, "value", is_in, "entity");
    wire(&mut g, ent, "value", frac, "entity");
    wire(&mut g, x, "value", height, "x");
    wire(&mut g, z, "value", height, "z");
    wire(&mut g, ent, "value", sink, "entity");
    wire(&mut g, frac, "fraction", sink, "vx");
    wire(&mut g, height, "height", sink, "vy");
    wire(&mut g, ev, EXEC_THEN, sink, "exec");
    // …and the Bool one gates a branch.
    wire(&mut g, is_in, "in_water", branch, "condition");
    wire(&mut g, sink, EXEC_THEN, branch, "exec");

    let lowered = lower_graph(&g, &reg).expect("lower").pop().unwrap();
    let lifted = relift(&lowered);
    assert_eq!(&lifted, &lowered, "lowered water IR must survive transpile");
    assert_eq!(
        generate_fn(&lowered).unwrap(),
        generate_fn(&lifted).unwrap(),
        "regeneration must be idempotent"
    );

    let src = generate_fn(&lowered).unwrap();
    for fragment in [
        "water::is_in_water(",
        "water::submerged_fraction(",
        "water::surface_height(",
    ] {
        assert!(src.contains(fragment), "missing {fragment} in:\n{src}");
    }
}

#[test]
fn every_water_query_ir_round_trips() {
    let e = || Expr::Lit(Lit::Int(1));
    let f = || Expr::Lit(Lit::Float(3.5));
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
            bind(1, call(&["water", "is_in_water"], vec![e()])),
            bind(2, call(&["water", "submerged_fraction"], vec![e()])),
            bind(3, call(&["water", "surface_height"], vec![f(), f()])),
        ],
    };
    let lifted = relift(&func);
    assert_eq!(lifted, func, "every water call shape must round-trip");
}

#[test]
fn every_water_event_round_trips() {
    for kind in [
        EventKind::WaterEnter,
        EventKind::WaterExit,
        EventKind::WaterSplash,
    ] {
        let id = kind.key();
        let params = kind.signature();
        assert_eq!(
            params.len(),
            2,
            "{id} must carry both `water` and `speed` — the event node's data \
             pins and `signature()` are two halves of one contract"
        );
        let func = BlueprintFn {
            id: id.clone(),
            name: id.clone(),
            params: params.clone(),
            ret: Ty::Unit,
            body: vec![Stmt::ExprStmt(Expr::Call {
                path: vec!["debug".into(), "print".into()],
                args: vec![Expr::Lit(Lit::Str("wet".into()))],
            })],
        };
        // The IR survives generate → lift…
        let lifted = relift(&func);
        assert_eq!(lifted, func, "{id} must survive transpile");
        // …and the graph half inverts too: raise → lower is the identity on it.
        let reg = blueprint_registry();
        let graph = raise_fn(&func).expect("raise");
        let relowered = lower_graph(&graph, &reg).expect("lower").pop().unwrap();
        assert_eq!(relowered.id, func.id, "{id} raised to the wrong event node");
        assert_eq!(relowered.params, params, "{id} lost its params on the way");
        assert_eq!(event_kind_of(&id), kind, "{id} must name its own kind");
    }
}

/// The handler ids are a **wire contract** — they are what a committed `.inf_act`
/// stores and what a generated Rust `fn` is named. Spelled out here so a rename
/// has to be a deliberate act.
#[test]
fn the_water_event_ids_are_frozen() {
    assert_eq!(EventKind::WaterEnter.key(), "water_enter");
    assert_eq!(EventKind::WaterExit.key(), "water_exit");
    assert_eq!(EventKind::WaterSplash.key(), "water_splash");
    // And they are distinct from every id that came before.
    let ids = [
        EventKind::BeginPlay.key(),
        EventKind::Tick.key(),
        EventKind::Collision.key(),
        EventKind::WaterEnter.key(),
        EventKind::WaterExit.key(),
        EventKind::WaterSplash.key(),
    ];
    let mut sorted = ids.to_vec();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        ids.len(),
        "handler ids must be unique: {ids:?}"
    );
}
