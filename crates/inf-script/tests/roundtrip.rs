//! **The round trip, in both directions and over the whole IR.**
//!
//! Three laws, stated in `lib.rs` and executed here:
//!
//! 1. `parse(emit(f)) == f`, exactly, for every `f` the parser produces.
//! 2. `emit(parse(emit(f))) == emit(f)`, exactly, for every `f` — including IR
//!    that a *graph* produced, where the parser's local ids need not agree with
//!    the lowerer's.
//! 3. `parse(emit(f))` **runs** identically to `f` — the same host calls in the
//!    same order with the same arguments, and the same wire trace. This is the
//!    honest replacement for the equality law 1 cannot make about graph-lowered
//!    IR, and it asserts the program rather than the report.
//!
//! # Why the corpus is checked for coverage before it is trusted
//!
//! "A round-trip test that round-trips the empty program proves nothing"
//! (P22/P23's law). So [`every_ir_construct_appears_in_the_corpus`] walks the
//! parsed corpus and fails unless every `Stmt` variant, every `Expr` variant,
//! every `BinOp`, both `UnOp`s and all three `Binding` kinds are present — and
//! [`the_corpus_is_not_trivial`] pins its size. A construct added to the IR with
//! no `.infini` spelling turns this file red on the day it lands.

use std::collections::{BTreeSet, HashMap};

use inf_blueprint::interp::{Host, RunError, Trace, Value};
use inf_blueprint::{BinOp, Binding, BlueprintClass, BlueprintFn, Expr, Lit, Stmt, UnOp};
use inf_script::{compile, emit_class, emit_fn, parse_fn, render};

/// The corpus: one `.infini` source per construct family. Hand-written, because
/// the law being checked is about text somebody would actually type.
const CORPUS: &[(&str, &str)] = &[
    (
        "the phase-6 gate handler",
        r#"actor "Spinner"

var angle: float = 0.0 exposed
var speed: float = 90.0 exposed

on tick(dt)
    local a = angle + dt * speed
    angle = a
    engine.set_rotation(a)
end
"#,
    ),
    (
        "branch, elseif and else",
        r#"actor "Gate"

var open: bool = false

on begin_play()
    if open then
        debug.print("open")
    elseif 1 < 2 then
        debug.print("shut")
    else
        debug.print("neither")
    end
end
"#,
    ),
    (
        "a branch that is NOT the last statement — the shape raise refuses",
        r#"actor "NonLinear"

on begin_play()
    if 1 < 2 then
        debug.print("first")
    end
    debug.print("second")
end
"#,
    ),
    (
        "the guarded while",
        r#"actor "Countdown"

var n: int = 3

on begin_play()
    while n > 0 do
        n = n - 1
    end
    debug.print("done")
end
"#,
    ),
    (
        "the counted for, keeping its loop variable's name",
        r#"actor "Sum"

var total: int = 0

on begin_play()
    for i = 0, 9 do
        total = total + i
    end
end
"#,
    ),
    (
        "nested loops and a branch inside one",
        r#"actor "Grid"

on begin_play()
    for row = 0, 2 do
        for col = 0, 2 do
            if row == col then
                debug.print("diagonal")
            end
        end
    end
end
"#,
    ),
    (
        "the math builtins seam",
        r#"actor "Maths"

on begin_play()
    local x = math.sqrt(16.0)
    local y = math.clamp(math.lerp(x, 10.0, 0.5), 0.0, 6.0)
    local z = math.sin(y) + math.cos(y) * math.pow(2.0, 3.0)
    local w = math.to_float(math.to_int(z))
    debug.print("ok")
end
"#,
    ),
    (
        "every binary operator and both unary ones",
        r#"actor "Ops"

on begin_play()
    local a = 1 + 2 - 3 * 4 / 5 % 6
    local b = a == 1.0 or a ~= 2.0
    local c = a < 1.0 and a <= 2.0
    local d = a > 1.0 and a >= 2.0
    local e = not b
    local f = -a
    local g = -(1)
    local h = -1
    debug.print("ops")
end
"#,
    ),
    (
        "precedence that must survive the reprint",
        r#"actor "Precedence"

on begin_play()
    local a = (1 + 2) * 3
    local b = 1 + 2 * 3
    local c = (1 or 0) and 1
    local d = 1 - (2 - 3)
    local e = 1 - 2 - 3
    local f = -(-(1.5))
    debug.print("prec")
end
"#,
    ),
    (
        "an action bound to a local, and a query in value position",
        r#"actor "Spawner"

on begin_play()
    local e = engine.spawn("enemy")
    engine.destroy(e)
end

on tick(dt)
    local me = 1
    if physics2d.is_grounded(me) then
        physics2d.set_velocity(me, 0.0, 0.0)
    end
    local hit = physics2d.raycast.hit(0.0, 0.0, 1.0, 0.0, 10.0)
    if hit then
        debug.print("hit")
    end
end
"#,
    ),
    (
        "every event header, including the named ones",
        r#"actor "Events"

on begin_play()
    debug.print("begin")
end

on tick(dt)
    debug.print("tick")
end

on collision(other)
    debug.print("bump")
end

on input "jump"(pressed)
    if pressed then
        debug.print("jump")
    end
end

on custom "ping"()
    dispatch.call(0, "pong")
end

on water_enter(water, speed)
    debug.print("splash")
end

on water_exit(water, speed)
    debug.print("dry")
end

on water_splash(water, speed)
    debug.print("plop")
end

on destroyed(chunks)
    debug.print("gone")
end
"#,
    ),
    (
        "a function with a return, and one that returns nothing",
        r#"actor "Library"

function double(x: float) -> float
    return x * 2.0
end

function shout(message: string)
    debug.print(message)
    return
end
"#,
    ),
    (
        "type annotations on locals, and every literal kind",
        r#"actor "Literals"

var flag: bool = true
var label: string = "hello"
var count: int = -7
var ratio: float = 0.30000000000000004

on begin_play()
    local a: float = 1.5
    local b: int = 2
    local c: bool = false
    local d: string = "with \"quotes\" and a\nnewline"
    local e: float = 2.2250738585072014e-308
    local f: int = -9223372036854775808
    debug.print(d)
end
"#,
    ),
    (
        // The content contains `]]` — `vec![[1, 2][0]]` really does — so the
        // emitter steps the bracket up a level and the fixture is written at
        // the level the emitter would choose. That is the whole point of the
        // long-bracket rule, exercised rather than described.
        "a rust escape block, which no graph can hold",
        r#"actor "Escape"

on begin_play()
    debug.print("before")
    rust [=[
    let _v = vec![[1, 2][0]];
]=]
    debug.print("after")
end
"#,
    ),
    (
        "a member variable whose name is not an identifier",
        r#"actor "Awkward"

on begin_play()
    var.set("hit count", var.get("hit count") + 1)
    debug.print("counted")
end
"#,
    ),
    (
        "the state cells a do_once graph lowers to",
        r#"actor "Once"

on tick(dt)
    if not nodestate.get_or("__bp_once_3", false) then
        nodestate.set("__bp_once_3", true)
        debug.print("first tick only")
    end
end
"#,
    ),
];

fn parsed_corpus() -> Vec<(&'static str, BlueprintClass)> {
    CORPUS
        .iter()
        .map(|(what, src)| {
            let (class, warnings) = compile(src, "act:test")
                .unwrap_or_else(|d| panic!("{what} did not compile:\n{}", render(&d)));
            // Every corpus entry declares what it reads, so nothing warns. A
            // warning here means the fixture drifted from its own declarations.
            assert!(warnings.is_empty(), "{what} warns:\n{}", render(&warnings));
            (*what, class)
        })
        .collect()
}

/// **Law 1** — `parse(emit(f)) == f`, exactly, over the corpus.
#[test]
fn the_parsers_own_output_round_trips_to_the_identical_ir() {
    for (what, class) in parsed_corpus() {
        let text = emit_class(&class).unwrap_or_else(|e| panic!("{what}: emit: {e}"));
        let (again, _) = compile(&text, "act:test").unwrap_or_else(|d| {
            panic!(
                "{what}: re-parse failed:\n{}\n--- text ---\n{text}",
                render(&d)
            )
        });
        assert_eq!(
            again, class,
            "{what}: the IR moved across a round trip\n--- text ---\n{text}"
        );
    }
}

/// **Law 2, the text half** — the emitted text is a fixed point.
#[test]
fn the_emitted_text_is_idempotent() {
    for (what, class) in parsed_corpus() {
        let once = emit_class(&class).unwrap();
        let (reparsed, _) = compile(&once, "act:test").unwrap();
        let twice = emit_class(&reparsed).unwrap();
        assert_eq!(once, twice, "{what}: the text moved on a second pass");
    }
}

/// …and for the corpus specifically, the emitted text **is the source**, modulo
/// the leading blank line the fixtures carry. That is stronger than idempotence
/// and it is what a designer experiences: open a script, save it, no diff.
#[test]
fn the_corpus_reprints_as_itself() {
    for (what, src) in CORPUS {
        let (class, _) = compile(src, "act:test").unwrap();
        let text = emit_class(&class).unwrap();
        assert_eq!(
            text.trim_end(),
            src.trim(),
            "{what}: the reprint differs from the source"
        );
    }
}

/// **Law 3** — the round trip preserves the *program*, not merely the bytes.
///
/// Every handler in the corpus is run twice: once from the parsed IR and once
/// from the IR that came back through the text. The recorded host calls — paths
/// and argument values — and the wire trace must match exactly.
#[test]
fn a_round_tripped_handler_runs_identically() {
    let mut compared = 0;
    for (what, class) in parsed_corpus() {
        let text = emit_class(&class).unwrap();
        let (again, _) = compile(&text, "act:test").unwrap();
        for (a, b) in class.events.iter().zip(&again.events) {
            let ta = run(&a.body, &class);
            let tb = run(&b.body, &again);
            assert_eq!(ta, tb, "{what}: `{}` ran differently", a.event.key());
            compared += 1;
        }
    }
    assert!(
        compared >= 20,
        "only {compared} handlers were actually run — a trace gate that runs \
         nothing passes for ever"
    );
}

/// **Law 2 over graph-lowered IR** — the half the corpus cannot reach.
///
/// A graph's IR has anonymous locals numbered by the lowerer's walk, which the
/// parser need not reproduce. The *text* still has to be a fixed point, and the
/// program still has to be the same one, so both are asserted here over the IR
/// `inf-blueprint`'s own round-trip fixtures build.
#[test]
fn graph_lowered_ir_reaches_a_text_fixed_point_and_keeps_its_program() {
    for (what, f) in ir_fixtures() {
        let once = emit_fn(&f).unwrap_or_else(|e| panic!("{what}: emit: {e}"));
        let back = parse_fn(&once)
            .unwrap_or_else(|d| panic!("{what}: re-parse:\n{}\n---\n{once}", render(&d)));
        let twice = emit_fn(&back).unwrap();
        assert_eq!(once, twice, "{what}: the text moved on a second pass");
        assert_eq!(
            run_bare(&f),
            run_bare(&back),
            "{what}: the round trip changed the program"
        );
    }
}

/// IR the parser is **not** the author of — real graphs, lowered, plus the one
/// binding kind only `lift` produces. These are the shapes law 2 and law 3 exist
/// for: their local ids are the lowerer's, not the parser's.
fn ir_fixtures() -> Vec<(&'static str, BlueprintFn)> {
    use inf_blueprint::lower::lower_graph;
    use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
    use inf_graph::{Graph, Link, NodeId, NodeUi, ParamValue};

    fn wire(g: &mut Graph, from: NodeId, fp: &str, to: NodeId, tp: &str) {
        g.links.push(Link {
            from,
            from_port: fp.into(),
            to,
            to_port: tp.into(),
        });
    }
    fn text(g: &mut Graph, n: NodeId, k: &str, v: &str) {
        g.node_mut(n)
            .unwrap()
            .params
            .insert(k.into(), ParamValue::Text(v.into()));
    }

    let reg = blueprint_registry();
    let mut out = Vec::new();

    // `do_once` — the `nodestate::*` shape, which `raise` refuses outright.
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let once = g.insert("flow.do_once", NodeUi::default());
    let p = g.insert("debug.print", NodeUi::default());
    text(&mut g, p, "message", "once");
    wire(&mut g, bp, EXEC_THEN, once, "exec");
    wire(&mut g, once, EXEC_THEN, p, "exec");
    out.push((
        "flow.do_once, lowered",
        lower_graph(&g, &reg).unwrap().pop().unwrap(),
    ));

    // `flip_flop` — a state read materialised into a `let`, then a branch on it.
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let ff = g.insert("flow.flip_flop", NodeUi::default());
    let a = g.insert("debug.print", NodeUi::default());
    text(&mut g, a, "message", "a");
    let b = g.insert("debug.print", NodeUi::default());
    text(&mut g, b, "message", "b");
    wire(&mut g, bp, EXEC_THEN, ff, "exec");
    wire(&mut g, ff, "a", a, "exec");
    wire(&mut g, ff, "b", b, "exec");
    out.push((
        "flow.flip_flop, lowered",
        lower_graph(&g, &reg).unwrap().pop().unwrap(),
    ));

    // `flow.sequence` — flattened at lowering, so the text shows the flattening.
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let seq = g.insert("flow.sequence", NodeUi::default());
    let a = g.insert("debug.print", NodeUi::default());
    text(&mut g, a, "message", "first");
    let b = g.insert("debug.print", NodeUi::default());
    text(&mut g, b, "message", "second");
    wire(&mut g, bp, EXEC_THEN, seq, "exec");
    wire(&mut g, seq, "then0", a, "exec");
    wire(&mut g, seq, "then1", b, "exec");
    out.push((
        "flow.sequence, flattened",
        lower_graph(&g, &reg).unwrap().pop().unwrap(),
    ));

    // A fanned-out pure value, which the lowerer materialises into an anonymous
    // `let` — the `Binding::Anon` case text never writes by hand.
    let mut g = Graph::empty();
    let tick = g.insert("event.tick", NodeUi::default());
    let v = g.insert("var.get", NodeUi::default());
    text(&mut g, v, "name", "speed");
    let mul = g.insert("math.mul", NodeUi::default());
    let set = g.insert("var.set", NodeUi::default());
    text(&mut g, set, "name", "angle");
    let rot = g.insert("engine.set_rotation", NodeUi::default());
    wire(&mut g, v, "value", mul, "a");
    wire(&mut g, tick, "dt", mul, "b");
    wire(&mut g, mul, "out", set, "value");
    wire(&mut g, mul, "out", rot, "angle");
    wire(&mut g, tick, EXEC_THEN, set, "exec");
    wire(&mut g, set, EXEC_THEN, rot, "exec");
    out.push((
        "a fanned-out pure value, materialised",
        lower_graph(&g, &reg).unwrap().pop().unwrap(),
    ));

    // A graph `for`, whose index is anonymous.
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let lo = g.insert("lit.int", NodeUi::default());
    g.node_mut(lo)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Int(0));
    let hi = g.insert("lit.int", NodeUi::default());
    g.node_mut(hi)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Int(4));
    let fo = g.insert("flow.for", NodeUi::default());
    let set = g.insert("var.set", NodeUi::default());
    text(&mut g, set, "name", "i");
    wire(&mut g, lo, "value", fo, "first");
    wire(&mut g, hi, "value", fo, "last");
    wire(&mut g, fo, "index", set, "value");
    wire(&mut g, bp, EXEC_THEN, fo, "exec");
    wire(&mut g, fo, "loop_body", set, "exec");
    out.push((
        "flow.for with an anonymous index",
        lower_graph(&g, &reg).unwrap().pop().unwrap(),
    ));

    // A `Binding::Named` — which `lower` never produces and `inf_transpile::lift`
    // does, when a graph value it recovered from hand-edited Rust carried a
    // display name. Built by hand because that is the only producer, and the
    // round trip has to cover the kind rather than the producer.
    out.push((
        "a Named binder, as lift recovers one",
        BlueprintFn {
            id: "begin_play".into(),
            name: "begin_play".into(),
            params: vec![],
            ret: inf_blueprint::Ty::Unit,
            body: vec![
                Stmt::Let {
                    id: inf_blueprint::LocalId(9),
                    binding: Binding::Named("speed".into()),
                    ty: Some(inf_blueprint::Ty::Float),
                    mutable: false,
                    value: Expr::Lit(Lit::Float(2.5)),
                },
                Stmt::ExprStmt(Expr::Call {
                    path: vec!["engine".into(), "set_rotation".into()],
                    args: vec![Expr::Local(inf_blueprint::LocalId(9))],
                }),
            ],
        },
    ));

    out
}

/// A host that writes every call down, so two runs can be compared as data.
#[derive(Default)]
struct Recorder {
    vars: HashMap<String, Value>,
    log: Vec<String>,
}

impl Host for Recorder {
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError> {
        let key = path.join("::");
        self.log.push(format!("{key}({args:?})"));
        match key.as_str() {
            "vars::get" => {
                let n = args[0].as_str()?.to_string();
                Ok(self.vars.get(&n).cloned().unwrap_or(Value::Float(1.0)))
            }
            "vars::set" => {
                let n = args[0].as_str()?.to_string();
                self.vars.insert(n, args[1].clone());
                Ok(Value::Unit)
            }
            "nodestate::get_or" => {
                let n = args[0].as_str()?.to_string();
                Ok(self
                    .vars
                    .get(&n)
                    .cloned()
                    .unwrap_or_else(|| args[1].clone()))
            }
            "nodestate::set" => {
                let n = args[0].as_str()?.to_string();
                self.vars.insert(n, args[1].clone());
                Ok(Value::Unit)
            }
            "engine::spawn" => Ok(Value::Int(7)),
            _ => Ok(Value::Float(0.0)),
        }
    }
}

/// Everything one run of a handler is observed to do: the host calls in order,
/// the computed wire **values** in order, and the verdict.
///
/// Wire values rather than `(LocalId, Value)` pairs, deliberately: a
/// graph-lowered handler's anonymous locals may renumber into the parser's walk
/// order across a round trip, and the law being asserted is that the *program*
/// is unchanged — not that two id counters agree.
type Observed = (Vec<String>, Vec<String>, String);

/// Run a handler against a recording host seeded from the class's variables.
fn run(f: &BlueprintFn, class: &BlueprintClass) -> Observed {
    let mut host = Recorder {
        vars: class
            .variables
            .iter()
            .map(|v| (v.name.clone(), Value::from(&v.default)))
            .collect(),
        log: Vec::new(),
    };
    run_on(f, &mut host)
}

fn run_bare(f: &BlueprintFn) -> Observed {
    let mut host = Recorder::default();
    run_on(f, &mut host)
}

fn run_on(f: &BlueprintFn, host: &mut Recorder) -> Observed {
    let args: HashMap<String, Value> = f
        .params
        .iter()
        .map(|p| {
            (
                p.name.clone(),
                match p.ty {
                    inf_blueprint::Ty::Bool => Value::Bool(true),
                    inf_blueprint::Ty::Int => Value::Int(3),
                    inf_blueprint::Ty::Str => Value::Str("s".into()),
                    _ => Value::Float(0.5),
                },
            )
        })
        .collect();
    let wires = |t: Trace| t.wires.into_iter().map(|(_, v)| format!("{v:?}")).collect();
    match inf_blueprint::eval_fn_traced(f, &args, host, &inf_blueprint::InterpDebug::default()) {
        // A `rust` block cannot be interpreted (only transpiled), which is a
        // *value* here and part of what the two runs must agree about.
        Ok((v, trace)) => (
            std::mem::take(&mut host.log),
            wires(trace),
            format!("{v:?}"),
        ),
        Err(e) => (std::mem::take(&mut host.log), Vec::new(), e.to_string()),
    }
}

/// **The corpus is not trivial.** A round-trip suite over an empty program is
/// green for ever and means nothing.
#[test]
fn the_corpus_is_not_trivial() {
    let mut statements = 0;
    let mut handlers = 0;
    for (_, class) in parsed_corpus() {
        for b in &class.events {
            handlers += 1;
            statements += count(&b.body.body);
        }
        for f in &class.functions {
            statements += count(&f.body);
        }
    }
    assert!(
        handlers >= 20 && statements >= 100,
        "the corpus holds {handlers} handlers and {statements} statements"
    );
}

fn count(body: &[Stmt]) -> usize {
    body.iter()
        .map(|s| {
            1 + match s {
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => count(then_body) + count(else_body),
                Stmt::While { body, .. } => count(body),
                _ => 0,
            }
        })
        .sum()
}

/// **Every IR construct has a `.infini` spelling, and the corpus writes it.**
///
/// The meta-arm that keeps the corpus honest: a `Stmt` or `Expr` variant, an
/// operator or a binding kind that nothing in the corpus produces is one the
/// round trip has never been tested on.
#[test]
fn every_ir_construct_appears_in_the_corpus() {
    let mut stmts = BTreeSet::new();
    let mut exprs = BTreeSet::new();
    let mut binops = BTreeSet::new();
    let mut unops = BTreeSet::new();
    let mut bindings = BTreeSet::new();
    let mut lits = BTreeSet::new();

    fn walk_expr(
        e: &Expr,
        exprs: &mut BTreeSet<&'static str>,
        binops: &mut BTreeSet<String>,
        unops: &mut BTreeSet<String>,
        lits: &mut BTreeSet<&'static str>,
    ) {
        match e {
            Expr::Lit(l) => {
                exprs.insert("Lit");
                lits.insert(match l {
                    Lit::Float(_) => "Float",
                    Lit::Int(_) => "Int",
                    Lit::Bool(_) => "Bool",
                    Lit::Str(_) => "Str",
                });
            }
            Expr::Param(_) => {
                exprs.insert("Param");
            }
            Expr::Local(_) => {
                exprs.insert("Local");
            }
            Expr::Unary(op, i) => {
                exprs.insert("Unary");
                unops.insert(format!("{op:?}"));
                walk_expr(i, exprs, binops, unops, lits);
            }
            Expr::Binary(op, a, b) => {
                exprs.insert("Binary");
                binops.insert(format!("{op:?}"));
                walk_expr(a, exprs, binops, unops, lits);
                walk_expr(b, exprs, binops, unops, lits);
            }
            Expr::Call { args, .. } => {
                exprs.insert("Call");
                for a in args {
                    walk_expr(a, exprs, binops, unops, lits);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        body: &[Stmt],
        stmts: &mut BTreeSet<&'static str>,
        exprs: &mut BTreeSet<&'static str>,
        binops: &mut BTreeSet<String>,
        unops: &mut BTreeSet<String>,
        bindings: &mut BTreeSet<&'static str>,
        lits: &mut BTreeSet<&'static str>,
    ) {
        for s in body {
            match s {
                Stmt::Let { binding, value, .. } => {
                    stmts.insert("Let");
                    bindings.insert(match binding {
                        Binding::Anon => "Anon",
                        Binding::Named(_) => "Named",
                        Binding::Raw(_) => "Raw",
                    });
                    walk_expr(value, exprs, binops, unops, lits);
                }
                Stmt::Assign { value, .. } => {
                    stmts.insert("Assign");
                    walk_expr(value, exprs, binops, unops, lits);
                }
                Stmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    stmts.insert("If");
                    walk_expr(cond, exprs, binops, unops, lits);
                    walk(then_body, stmts, exprs, binops, unops, bindings, lits);
                    walk(else_body, stmts, exprs, binops, unops, bindings, lits);
                }
                Stmt::While { cond, body } => {
                    stmts.insert("While");
                    walk_expr(cond, exprs, binops, unops, lits);
                    walk(body, stmts, exprs, binops, unops, bindings, lits);
                }
                Stmt::Return(v) => {
                    stmts.insert("Return");
                    if let Some(v) = v {
                        walk_expr(v, exprs, binops, unops, lits);
                    }
                }
                Stmt::ExprStmt(e) => {
                    stmts.insert("ExprStmt");
                    walk_expr(e, exprs, binops, unops, lits);
                }
                Stmt::Snippet(_) => {
                    stmts.insert("Snippet");
                }
            }
        }
    }

    for (_, class) in parsed_corpus() {
        for b in &class.events {
            walk(
                &b.body.body,
                &mut stmts,
                &mut exprs,
                &mut binops,
                &mut unops,
                &mut bindings,
                &mut lits,
            );
        }
        for f in &class.functions {
            walk(
                &f.body,
                &mut stmts,
                &mut exprs,
                &mut binops,
                &mut unops,
                &mut bindings,
                &mut lits,
            );
        }
    }
    // `Binding::Named` and `Binding::Anon` come from graphs, not from hand
    // writing, so the graph fixtures carry them.
    for (_, f) in ir_fixtures() {
        walk(
            &f.body,
            &mut stmts,
            &mut exprs,
            &mut binops,
            &mut unops,
            &mut bindings,
            &mut lits,
        );
    }

    let want_stmts: BTreeSet<&str> = [
        "Let", "Assign", "If", "While", "Return", "ExprStmt", "Snippet",
    ]
    .into();
    assert_eq!(
        stmts, want_stmts,
        "a `Stmt` variant is missing from the round-trip corpus"
    );

    let want_exprs: BTreeSet<&str> = ["Lit", "Param", "Local", "Unary", "Binary", "Call"].into();
    assert_eq!(exprs, want_exprs, "an `Expr` variant is missing");

    let want_binops: BTreeSet<String> = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Rem,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Lt,
        BinOp::Le,
        BinOp::Gt,
        BinOp::Ge,
        BinOp::And,
        BinOp::Or,
    ]
    .iter()
    .map(|o| format!("{o:?}"))
    .collect();
    assert_eq!(binops, want_binops, "a `BinOp` is missing");

    let want_unops: BTreeSet<String> = [UnOp::Neg, UnOp::Not]
        .iter()
        .map(|o| format!("{o:?}"))
        .collect();
    assert_eq!(unops, want_unops, "a `UnOp` is missing");

    let want_bindings: BTreeSet<&str> = ["Anon", "Named", "Raw"].into();
    assert_eq!(
        bindings, want_bindings,
        "a `Binding` kind is missing — `Named` and `Anon` come from graph-lowered IR"
    );

    let want_lits: BTreeSet<&str> = ["Float", "Int", "Bool", "Str"].into();
    assert_eq!(lits, want_lits, "a `Lit` kind is missing");
}

/// **The falsification arm.** Break the emitter's one non-obvious rule — the
/// parentheses that separate a negative literal from a negation — and the round
/// trip must go red. Asserted here as the *difference* between the two IRs, so
/// the day somebody "simplifies" `-(1)` to `-1` this file names the reason.
#[test]
fn a_negative_literal_and_a_negation_are_different_programs() {
    let neg_lit = parse_fn("on begin_play()\n    local a = -1\nend\n").unwrap();
    let negation = parse_fn("on begin_play()\n    local a = -(1)\nend\n").unwrap();
    assert_ne!(neg_lit.body, negation.body);
    let Stmt::Let { value, .. } = &neg_lit.body[0] else {
        panic!()
    };
    assert_eq!(value, &Expr::Lit(Lit::Int(-1)));
    let Stmt::Let { value, .. } = &negation.body[0] else {
        panic!()
    };
    assert_eq!(
        value,
        &Expr::Unary(UnOp::Neg, Box::new(Expr::Lit(Lit::Int(1))))
    );
    // …and each prints as itself.
    assert!(emit_fn(&neg_lit).unwrap().contains("= -1\n"));
    assert!(emit_fn(&negation).unwrap().contains("= -(1)\n"));
}

/// A member variable shadowed by a local prints as `var.get("…")`, because a
/// bare name there would mean the local. The silent-capture arm.
#[test]
fn a_shadowed_member_variable_prints_explicitly() {
    // The IR: `let speed = vars::get("speed"); debug::print(vars::get("speed"))`
    // — a local named after the variable it reads, which is legal IR a graph can
    // produce and which text cannot spell with a bare name after the binding.
    let f = BlueprintFn {
        id: "begin_play".into(),
        name: "begin_play".into(),
        params: vec![],
        ret: inf_blueprint::Ty::Unit,
        body: vec![
            Stmt::Let {
                id: inf_blueprint::LocalId(0),
                binding: Binding::Raw("speed".into()),
                ty: None,
                mutable: false,
                value: Expr::Call {
                    path: vec!["vars".into(), "get".into()],
                    args: vec![Expr::Lit(Lit::Str("speed".into()))],
                },
            },
            Stmt::ExprStmt(Expr::Call {
                path: vec!["debug".into(), "print".into()],
                args: vec![Expr::Call {
                    path: vec!["vars".into(), "get".into()],
                    args: vec![Expr::Lit(Lit::Str("speed".into()))],
                }],
            }),
        ],
    };
    let text = emit_fn(&f).unwrap();
    // The binding's own initialiser is written before the name enters scope…
    assert!(text.contains("local speed = speed\n"), "{text}");
    // …and the later read, which the local now shadows, is explicit.
    assert!(text.contains("debug.print(var.get(\"speed\"))"), "{text}");
    assert_eq!(parse_fn(&text).unwrap(), f);
}
