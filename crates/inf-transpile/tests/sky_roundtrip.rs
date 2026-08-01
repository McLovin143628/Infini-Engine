//! Transpiler round-trip coverage for the P17.1 `sky.*` node kit (ROADMAP §12:
//! every new node type ships with its round-trip case).
//!
//! The sky kit is two pure getters and two exec setters over the level clock.
//! Like every namespace, lowering + emission are registry/path driven, so this
//! exercises the generic machinery end-to-end and proves the kit needed **zero**
//! transpiler code:
//! * `sky_kit_graph_round_trips` — a real graph (read the clock, add an hour,
//!   write it back, then freeze time) lowers, survives generate → lift unchanged,
//!   and emits `sky::*` free calls;
//! * `every_sky_node_ir_round_trips` — a hand-built IR using every `sky.*` call
//!   shape round-trips generate → lift byte-stable;
//! * `pure_sky_getters_keep_a_bare_call_path` — the pin-count invariant the kit's
//!   design rests on: a single-output pure node must NOT fan into
//!   `sky::get::<field>`, or both engine hosts would need three-segment arms;
//! * `sky_weather_graph_round_trips` (P17.4) — the weather half, including the
//!   kit's only **two-argument** action, whose argument ORDER is the thing a
//!   generic lowerer could get wrong in a way nothing else would notice.

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
fn sky_kit_graph_round_trips() {
    // begin_play → sky.set_time_of_day(sky.get_time_of_day() + 3600);
    //              sky.set_rate(0)
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let now = g.insert("sky.get_time_of_day", NodeUi::default());
    let hour = float(&mut g, 3600.0);
    let add = g.insert("math.add", NodeUi::default());
    let set_tod = g.insert("sky.set_time_of_day", NodeUi::default());
    let frozen = float(&mut g, 0.0);
    let set_rate = g.insert("sky.set_rate", NodeUi::default());

    wire(&mut g, now, "seconds", add, "a");
    wire(&mut g, hour, "value", add, "b");
    wire(&mut g, add, "out", set_tod, "seconds");
    wire(&mut g, frozen, "value", set_rate, "rate");
    wire(&mut g, bp, EXEC_THEN, set_tod, "exec");
    wire(&mut g, set_tod, EXEC_THEN, set_rate, "exec");

    let lowered = lower_graph(&g, &reg).expect("lower").pop().unwrap();

    // Round-trips through the transpiler unchanged, and regeneration is idempotent.
    let lifted = relift(&lowered);
    assert_eq!(&lifted, &lowered, "lowered sky IR must survive transpile");
    assert_eq!(
        generate_fn(&lowered).unwrap(),
        generate_fn(&lifted).unwrap(),
        "regeneration must be idempotent"
    );

    // The generated Rust targets the `sky::*` runtime surface with bare paths.
    let src = generate_fn(&lowered).unwrap();
    assert!(src.contains("sky::get_time_of_day("), "src:\n{src}");
    assert!(src.contains("sky::set_time_of_day("), "src:\n{src}");
    assert!(src.contains("sky::set_rate("), "src:\n{src}");
    assert!(
        !src.contains("sky::get::"),
        "a single-output getter must not fan a field segment:\n{src}"
    );
}

#[test]
fn every_sky_node_ir_round_trips() {
    // A body that uses every sky call shape at least once.
    let f = || Expr::Lit(Lit::Float(36_000.0));
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
            Stmt::ExprStmt(call(&["sky", "set_time_of_day"], vec![f()])),
            Stmt::ExprStmt(call(
                &["sky", "set_rate"],
                vec![Expr::Lit(Lit::Float(60.0))],
            )),
            Stmt::ExprStmt(call(
                &["debug", "print"],
                vec![call(&["sky", "get_time_of_day"], vec![])],
            )),
            Stmt::ExprStmt(call(
                &["debug", "print"],
                vec![call(&["sky", "get_rate"], vec![])],
            )),
        ],
    };

    let lifted = relift(&func);
    assert_eq!(lifted, func, "every sky call shape must round-trip");
}

#[test]
fn pure_sky_getters_keep_a_bare_call_path() {
    // The pin-count invariant `sky_nodes` is designed around: `lower` appends the
    // output-port name to the call path only when a pure node has MORE than one
    // data output. Both sky getters have exactly one, so the emitted path stays
    // two segments — which is what lets each engine host answer with a single
    // `(Some("sky"), Some("get_time_of_day"))` arm.
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let now = g.insert("sky.get_time_of_day", NodeUi::default());
    let rate = g.insert("sky.get_rate", NodeUi::default());
    let add = g.insert("math.add", NodeUi::default());
    let set_tod = g.insert("sky.set_time_of_day", NodeUi::default());

    wire(&mut g, now, "seconds", add, "a");
    wire(&mut g, rate, "rate", add, "b");
    wire(&mut g, add, "out", set_tod, "seconds");
    wire(&mut g, bp, EXEC_THEN, set_tod, "exec");

    let lowered = lower_graph(&g, &reg).expect("lower").pop().unwrap();
    let mut paths = Vec::new();
    collect_call_paths(&lowered, &mut paths);
    assert!(
        paths.contains(&"sky::get_time_of_day".to_string()),
        "{paths:?}"
    );
    assert!(paths.contains(&"sky::get_rate".to_string()), "{paths:?}");
    assert!(
        paths.iter().all(|p| p.matches("::").count() <= 1),
        "no sky call may grow a third segment: {paths:?}"
    );
}

/// Every `Expr::Call` path in a function body, `::`-joined.
fn collect_call_paths(f: &BlueprintFn, out: &mut Vec<String>) {
    fn walk_expr(e: &Expr, out: &mut Vec<String>) {
        match e {
            Expr::Call { path, args } => {
                out.push(path.join("::"));
                for a in args {
                    walk_expr(a, out);
                }
            }
            Expr::Binary(_, lhs, rhs) => {
                walk_expr(lhs, out);
                walk_expr(rhs, out);
            }
            Expr::Unary(_, e) => walk_expr(e, out),
            _ => {}
        }
    }
    fn walk_stmt(s: &Stmt, out: &mut Vec<String>) {
        match s {
            Stmt::ExprStmt(e) | Stmt::Return(Some(e)) => walk_expr(e, out),
            Stmt::Let { value, .. } => walk_expr(value, out),
            Stmt::Assign { value, .. } => walk_expr(value, out),
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                walk_expr(cond, out);
                for s in then_body.iter().chain(else_body.iter()) {
                    walk_stmt(s, out);
                }
            }
            Stmt::While { cond, body } => {
                walk_expr(cond, out);
                for s in body {
                    walk_stmt(s, out);
                }
            }
            _ => {}
        }
    }
    for s in &f.body {
        walk_stmt(s, out);
    }
}

// ── P17.4 weather ────────────────────────────────────────────────────────────

fn string(g: &mut Graph, v: &str) -> NodeId {
    let n = g.insert("lit.str", NodeUi::default());
    g.node_mut(n)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Text(v.into()));
    n
}

/// A graph that reads the weather, reacts to it and sets it, then survives the
/// transpiler unchanged — the P17.4 half of the kit's round-trip case.
#[test]
fn sky_weather_graph_round_trips() {
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let preset = string(&mut g, "storm");
    let blend = float(&mut g, 6.0);
    let set = g.insert("sky.set_weather", NodeUi::default());
    let rain = g.insert("sky.get_precipitation", NodeUi::default());
    let wind = g.insert("sky.get_wind_speed", NodeUi::default());
    let sum = g.insert("math.add", NodeUi::default());
    let rate = g.insert("sky.set_rate", NodeUi::default());

    wire(&mut g, preset, "value", set, "preset");
    wire(&mut g, blend, "value", set, "blend_seconds");
    wire(&mut g, rain, "intensity", sum, "a");
    wire(&mut g, wind, "speed", sum, "b");
    wire(&mut g, sum, "out", rate, "rate");
    wire(&mut g, bp, EXEC_THEN, set, "exec");
    wire(&mut g, set, EXEC_THEN, rate, "exec");

    let lowered = lower_graph(&g, &reg).expect("lower").pop().unwrap();
    let lifted = relift(&lowered);
    assert_eq!(
        &lifted, &lowered,
        "lowered weather IR must survive transpile"
    );
    assert_eq!(
        generate_fn(&lowered).unwrap(),
        generate_fn(&lifted).unwrap(),
        "regeneration must be idempotent"
    );

    let src = generate_fn(&lowered).unwrap();
    assert!(
        src.contains("sky::set_weather("),
        "src:
{src}"
    );
    assert!(
        src.contains("sky::get_precipitation("),
        "src:
{src}"
    );
    assert!(
        src.contains("sky::get_wind_speed("),
        "src:
{src}"
    );
    assert!(
        !src.contains("sky::get::"),
        "single-output getters must not fan a field segment:
{src}"
    );

    // ARGUMENT ORDER is the claim that matters here, and it is not visible in the
    // round-trip: `lower::data_input_ports` walks the NodeDef's declared inputs,
    // so a reordered declaration would emit `set_weather(6.0, "storm")` and both
    // hosts would read a blend time as a preset name — parse it as nothing, and
    // silently do nothing at all.
    let mut paths = Vec::new();
    collect_call_paths(&lowered, &mut paths);
    assert!(paths.contains(&"sky::set_weather".to_string()), "{paths:?}");
    let call = find_call(&lowered, "sky::set_weather").expect("the set_weather call");
    match &call[..] {
        [Expr::Lit(Lit::Str(p)), Expr::Lit(Lit::Float(b))] => {
            assert_eq!(p, "storm", "the preset must be argument 0");
            assert_eq!(*b, 6.0, "the blend time must be argument 1");
        }
        other => panic!("unexpected set_weather args: {other:?}"),
    }
}

/// Every weather call shape round-trips generate → lift byte-stable, including
/// the getters used as data expressions.
#[test]
fn every_weather_node_ir_round_trips() {
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
            Stmt::ExprStmt(call(
                &["sky", "set_weather"],
                vec![
                    Expr::Lit(Lit::Str("snow".into())),
                    Expr::Lit(Lit::Float(12.0)),
                ],
            )),
            Stmt::ExprStmt(call(
                &["debug", "print"],
                vec![call(&["sky", "get_weather"], vec![])],
            )),
            Stmt::ExprStmt(call(
                &["debug", "print"],
                vec![call(&["sky", "get_precipitation"], vec![])],
            )),
            Stmt::ExprStmt(call(
                &["debug", "print"],
                vec![call(&["sky", "get_wind_speed"], vec![])],
            )),
        ],
    };
    assert_eq!(relift(&func), func, "every weather call shape round-trips");
}

/// The argument list of the first call to `path` in a function body.
fn find_call(f: &BlueprintFn, path: &str) -> Option<Vec<Expr>> {
    fn walk_expr(e: &Expr, path: &str, out: &mut Option<Vec<Expr>>) {
        if out.is_some() {
            return;
        }
        match e {
            Expr::Call { path: p, args } => {
                if p.join("::") == path {
                    *out = Some(args.clone());
                    return;
                }
                for a in args {
                    walk_expr(a, path, out);
                }
            }
            Expr::Binary(_, lhs, rhs) => {
                walk_expr(lhs, path, out);
                walk_expr(rhs, path, out);
            }
            Expr::Unary(_, e) => walk_expr(e, path, out),
            _ => {}
        }
    }
    fn walk_stmt(s: &Stmt, path: &str, out: &mut Option<Vec<Expr>>) {
        match s {
            Stmt::ExprStmt(e) | Stmt::Return(Some(e)) => walk_expr(e, path, out),
            Stmt::Let { value, .. } => walk_expr(value, path, out),
            Stmt::Assign { value, .. } => walk_expr(value, path, out),
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                walk_expr(cond, path, out);
                for s in then_body.iter().chain(else_body.iter()) {
                    walk_stmt(s, path, out);
                }
            }
            Stmt::While { cond, body } => {
                walk_expr(cond, path, out);
                for s in body {
                    walk_stmt(s, path, out);
                }
            }
            _ => {}
        }
    }
    let mut out = None;
    for s in &f.body {
        walk_stmt(s, path, &mut out);
    }
    out
}
