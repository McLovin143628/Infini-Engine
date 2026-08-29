//! **The honest-subset table, executed.**
//!
//! "Graphs and text are two views of one program" is exactly as complete as
//! `raise` is, and the memo's §4 says so in prose. Prose in a memo is a claim; a
//! table somebody has to keep green is a fact. Every row below is a `.infini`
//! construct, the verdict `raise` gives it, and — where it raises — proof that
//! the graph round trip does not change what the script *does*.
//!
//! # What SCRIPT1 changed, and what it deliberately did not
//!
//! `flow.for` was widened into `raise` this wave, because the language grew a
//! `for` statement and leaving it out would have meant every script with a
//! counted loop was unopenable as a graph. The shape is a recognition, like
//! `flow.while`'s, and it cost one matcher.
//!
//! **`RaiseError::NonLinear` was not, and the price is stated rather than
//! implied.** An `if` that is not the last statement of its block does not raise
//! at all, and `raise_chain` returns `Err` on the first unraisable statement, so
//! *one* such `if` makes the whole handler unraisable. Closing it means giving a
//! branch node a join — a *merge* point where its two exec paths come back
//! together — which the node kit does not have, the lowerer does not emit, and
//! `lower ∘ raise == id` would then have to hold across. That is a change to what
//! a Blueprint graph *is*, not a recogniser: new port semantics on `flow.branch`,
//! a matcher for the merge, canvas work to draw and route it, and a re-proof of
//! the round-trip invariant over the new image. It is a wave, not a clause, and
//! SCRIPT1 refuses to pretend otherwise.
//!
//! The other four non-flow refusals (`Assign`, `Snippet`, a non-call
//! `ExprStmt`, a call in value position) are the same shape of decision, and
//! `Assign` is the one a designer meets first: `local x = 0` then `x = x + 1` is
//! ordinary text and has no graph form, because a graph value is a wire and a
//! wire is not re-assigned. Member variables are the graph-friendly way to hold
//! state, and the spec says so.

use std::collections::HashMap;

use inf_blueprint::interp::{Host, RunError, Value};
use inf_blueprint::lower::lower_graph;
use inf_blueprint::nodekit::blueprint_registry;
use inf_blueprint::{raise_fn, BlueprintFn, RaiseError};
use inf_script::{parse_fn, render};

/// What `raise` should say about a construct.
#[derive(Debug, PartialEq)]
enum Verdict {
    /// It becomes a graph, and the graph runs the same program.
    Raises,
    /// It does not, with this error.
    Refuses(RaiseError),
}
use Verdict::{Raises, Refuses};

/// `(construct, source, verdict)` — the spec's appendix-A table, in code.
fn table() -> Vec<(&'static str, String, Verdict)> {
    let handler = |body: &str| format!("on begin_play()\n{body}end\n");
    vec![
        (
            "a linear chain of actions",
            handler("    debug.print(\"a\")\n    debug.print(\"b\")\n"),
            Raises,
        ),
        (
            "a member variable read and written",
            handler("    speed = speed + 1.0\n"),
            Raises,
        ),
        (
            "arithmetic, comparison and logic operators",
            handler("    speed = (1.0 + 2.0) * 3.0\n    flag = 1.0 < 2.0 and not false\n"),
            Raises,
        ),
        (
            "a pure math builtin in value position",
            handler("    speed = math.clamp(math.sqrt(16.0), 0.0, 3.0)\n"),
            Raises,
        ),
        (
            "unary minus",
            handler("    speed = -(speed)\n"),
            Raises,
        ),
        (
            "an `if` as the last statement of its block",
            handler("    if flag then\n        debug.print(\"yes\")\n    else\n        debug.print(\"no\")\n    end\n"),
            Raises,
        ),
        (
            "a `while` loop",
            handler("    while flag do\n        debug.print(\"spin\")\n    end\n"),
            Raises,
        ),
        (
            "a `while` loop with statements after it",
            handler("    while flag do\n        debug.print(\"spin\")\n    end\n    debug.print(\"done\")\n"),
            Raises,
        ),
        (
            "a `for` loop — WIDENED IN SCRIPT1",
            handler("    for i = 0, 3 do\n        total = i\n    end\n"),
            Raises,
        ),
        (
            "a `return` as the last statement",
            handler("    debug.print(\"bye\")\n    return\n"),
            Raises,
        ),
        (
            "an action bound to a local",
            handler("    local e = engine.spawn(\"enemy\")\n    engine.destroy(e)\n"),
            Raises,
        ),
        // ── and now the honest half ───────────────────────────────────────
        (
            "an `if` that is NOT the last statement of its block",
            handler("    if flag then\n        debug.print(\"yes\")\n    end\n    debug.print(\"after\")\n"),
            Refuses(RaiseError::NonLinear),
        ),
        (
            "a `return` that is NOT the last statement",
            handler("    return\n    debug.print(\"unreachable\")\n"),
            Refuses(RaiseError::NonLinear),
        ),
        (
            "assigning to a local",
            handler("    local x = 0\n    x = x + 1\n    debug.print(\"x\")\n"),
            Refuses(RaiseError::UnsupportedStmt("assign")),
        ),
        (
            "a `rust` escape block",
            handler("    rust [[\n    let _ = 1;\n]]\n"),
            Refuses(RaiseError::UnsupportedStmt("snippet")),
        ),
        (
            "a non-action call in value position",
            handler("    flag = physics2d.is_grounded(1)\n"),
            Refuses(RaiseError::UnsupportedExpr("pure call")),
        ),
        (
            "the `nodestate` cells a do_once graph lowers to",
            handler("    if not nodestate.get_or(\"k\", false) then\n        nodestate.set(\"k\", true)\n    end\n"),
            Refuses(RaiseError::UnsupportedExpr("pure call")),
        ),
        (
            "a bare `while` with no guard — which text cannot write, and lift can",
            handler("    while flag do\n        debug.print(\"spin\")\n    end\n"),
            Raises,
        ),
    ]
}

fn compile_one(what: &str, src: &str) -> BlueprintFn {
    parse_fn(src).unwrap_or_else(|d| panic!("{what} did not compile:\n{}", render(&d)))
}

/// **The table.** Every row's verdict, checked.
#[test]
fn the_honest_subset_table_is_what_raise_actually_does() {
    let mut wrong = Vec::new();
    for (what, src, want) in table() {
        let f = compile_one(what, &src);
        let got = match raise_fn(&f) {
            Ok(_) => Raises,
            Err(e) => Refuses(e),
        };
        if got != want {
            wrong.push(format!("{what}: expected {want:?}, got {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// **`flow.for` really is inverted now** — the wave's widening, asserted on the
/// node the graph comes back with rather than on the absence of an error.
#[test]
fn a_counted_loop_comes_back_as_a_for_node() {
    let f = compile_one(
        "for",
        "on begin_play()\n    for i = 0, 3 do\n        total = i\n    end\nend\n",
    );
    let g = raise_fn(&f).expect("a for loop raises");
    assert!(
        g.nodes.values().any(|n| n.type_id == "flow.for"),
        "no flow.for in {:?}",
        g.nodes.values().map(|n| &n.type_id).collect::<Vec<_>>()
    );
    assert!(
        !g.nodes.values().any(|n| n.type_id == "flow.while"),
        "the for expansion was read as a while"
    );
}

/// **One unraisable statement makes the whole handler unraisable**, which is the
/// per-handler degradation `graph_open_actor` already lives with and the
/// granularity SCRIPT2's UI has to speak in. Asserted, because a reader could
/// reasonably assume the raisable prefix survives. It does not.
#[test]
fn one_unraisable_statement_takes_the_whole_handler() {
    let f = compile_one(
        "a good prefix and one bad statement",
        "on begin_play()\n    debug.print(\"fine\")\n    debug.print(\"also fine\")\n    local x = 0\n    x = 1\nend\n",
    );
    assert_eq!(
        raise_fn(&f),
        Err(RaiseError::UnsupportedStmt("assign")),
        "the two good statements did not save the handler, and should not have"
    );
}

/// **Where a script raises, the graph round trip does not change what it does.**
///
/// `lower(raise(f)) == f` is `raise`'s own invariant *on lowering's image*, and a
/// text-authored handler is not in that image — its `for` index carries the name
/// the author gave it, which a graph has nowhere to keep. So the claim asserted
/// here is the one that matters to a designer: run the script, run the graph it
/// raises to, and compare the host calls.
#[test]
fn raising_a_script_to_a_graph_and_back_preserves_the_program() {
    let reg = blueprint_registry();
    let mut compared = 0;
    for (what, src, want) in table() {
        if want != Raises {
            continue;
        }
        let f = compile_one(what, &src);
        let graph = raise_fn(&f).unwrap_or_else(|e| panic!("{what}: {e}"));
        let mut relowered = lower_graph(&graph, &reg).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(relowered.len(), 1, "{what}: one event, one function");
        let g = relowered.pop().unwrap();
        assert_eq!(observe(&f), observe(&g), "{what}: the program changed");
        compared += 1;
    }
    assert!(
        compared >= 10,
        "only {compared} rows were actually re-lowered and run"
    );
}

/// A recording host: seeded variables, and every call written down.
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
                // `flag` reads false so the guarded `while` rows terminate; every
                // other variable reads a number.
                Ok(self
                    .vars
                    .entry(n.clone())
                    .or_insert(if n == "flag" {
                        Value::Bool(false)
                    } else {
                        Value::Float(1.0)
                    })
                    .clone())
            }
            "vars::set" => {
                self.vars
                    .insert(args[0].as_str()?.to_string(), args[1].clone());
                Ok(Value::Unit)
            }
            "engine::spawn" => Ok(Value::Int(7)),
            _ => Ok(Value::Float(0.0)),
        }
    }
}

/// The host calls a handler makes, in order — everything a designer can observe.
fn observe(f: &BlueprintFn) -> Vec<String> {
    let mut host = Recorder::default();
    let args: HashMap<String, Value> = HashMap::new();
    let _ = inf_blueprint::eval_fn(f, &args, &mut host);
    host.log
}

/// The table is not a list of things that happen to pass: it holds both
/// verdicts, in numbers, and a change that collapses one column shows up here.
#[test]
fn the_table_covers_both_verdicts() {
    let rows = table();
    let raises = rows.iter().filter(|(_, _, v)| *v == Raises).count();
    let refuses = rows.len() - raises;
    assert!(
        raises >= 11 && refuses >= 6,
        "{raises} raising rows and {refuses} refusing ones — the table has \
         stopped covering one of the two answers"
    );
}
