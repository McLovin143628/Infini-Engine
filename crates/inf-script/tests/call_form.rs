//! **The call form**: a handler calling its unit's own `function` declarations.
//!
//! SCRIPT1a's loudest carried item, and the first thing SCRIPT2's API surface
//! wanted. Its ledger entry read: *"the IR has no user-function call form;
//! `function` bodies parse, lower and transpile, and nothing invokes them yet.
//! Adding a call form is an IR change with the P6 vars-via-Host bar to clear."*
//!
//! # The pricing, which is the wave's first result
//!
//! **It is not an IR change, so it is not a wire question.** `Expr::Call` has a
//! `path: Vec<String>` and every registered verb is `namespace.verb` — two
//! segments, or three for a multi-result query naming its result. A call to a
//! unit-local function is the same variant with **one** segment. No new `Expr`
//! variant, no new `Stmt` variant, no `schema_version` move on `.inf_act`
//! (which is pretty JSON, not bincode: `load_actor_classes_from_dir` reads it
//! with `serde_json::from_slice`), no `FROZEN_WIRE` row, no pack format change.
//! The arc's one schema window is **not spent**, and this file's
//! [`the_call_form_moved_no_wire`] is the arm that says so.
//!
//! That is the P6 vars-via-Host precedent met a second time: member variables
//! needed no IR change because they became `vars::get`/`vars::set` calls
//! crossing the one `Host` seam, and a local function call needs none because it
//! becomes a one-segment call resolved on that same seam
//! (`inf_blueprint::interp::LocalFns`).
//!
//! # What every face had to learn, together
//!
//! | face | what it learned |
//! |---|---|
//! | the parser | a bare `name(args)` resolves against the unit's `function` headers (arity exact, value position needs a `->`) |
//! | the emitter | a one-segment path prints bare, through `ident` so it reads back |
//! | the interpreter | a new `LocalFns` decorator carries the unit's functions and a **depth budget**, stacked outside `ActorHost` so a callee's own `vars::*` still resolve |
//! | the transpiler | nothing — `emit` was already generic over `Expr::Call`, and a one-segment path renders as the Rust call to the sibling `pub fn` that `generate_file` puts in the same module. Proved by the crown gate, which compiles and runs it |
//! | `raise` | a **named refusal** — the node kit has no call-a-function node, and before this it built a graph node whose `type_id` no registry knows |
//!
//! # Recursion is refused, and the reason is parity rather than taste
//!
//! The interpreter bounds a call chain at `MAX_CALL_DEPTH` and answers with a
//! `RunError` (P21: a refusal is a value). The **transpiled Rust has no such
//! bound** — a recursive one-segment call is an ordinary recursive Rust call and
//! overflows the stack. Two faces of one program that disagree about what an
//! infinite one does is exactly what the parity gate exists to prevent, so the
//! *language* refuses the cycle statically, where a designer can see it, and the
//! interpreter's budget stays as the defence for IR that did not come through
//! this parser.

use std::collections::HashMap;

use inf_blueprint::interp::{Debug, Host, RunError, Value, MAX_CALL_DEPTH};
use inf_blueprint::semantics::{run_event, ActorInstance, EventKind};
use inf_blueprint::{
    raise::{raise_fn, RaiseError},
    Binding, BlueprintClass, BlueprintFn, Expr, Lit, LocalId, Param, Stmt, Ty,
};
use inf_script::{compile, emit_class, parse_unit, render};

/// A host that records what it was asked to do and answers `Unit`.
#[derive(Default)]
struct LogHost {
    log: Vec<String>,
}

impl Host for LogHost {
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError> {
        let rendered: Vec<String> = args.iter().map(|v| format!("{v:?}")).collect();
        self.log
            .push(format!("{}({})", path.join("::"), rendered.join(", ")));
        Ok(Value::Unit)
    }
}

fn class_of(src: &str) -> BlueprintClass {
    compile(src, "act:test")
        .unwrap_or_else(|d| panic!("{}", render(&d)))
        .0
}

/// Fire `BeginPlay` on a fresh instance of `src` and return the host log.
fn run(src: &str) -> Vec<String> {
    let class = class_of(src);
    let mut actor = ActorInstance::new(&class);
    let mut host = LogHost::default();
    run_event(
        &class,
        &mut actor,
        &EventKind::BeginPlay,
        &HashMap::new(),
        &mut host,
        &Debug::default(),
    )
    .unwrap_or_else(|e| panic!("the handler failed: {e}"));
    host.log
}

/// The wave's fixture, and it calls **forward**: the handler is written above
/// both functions, which is what the header prescan exists for and what a
/// designer does without thinking about it.
///
/// It is also written in the emitter's own order (variables, handlers,
/// functions) so that the strongest round-trip law — `emit(parse(src)) == src`
/// — can be asserted on it directly.
const DOUBLER: &str = "\
actor \"Doubler\"

var total: float = 0.0

on begin_play()
    local a = double(4.0)
    announce(\"doubled\", a)
    announce(\"again\", double(a))
end

function double(x: float) -> float
    return x * 2.0
end

function announce(label: string, value: float)
    debug.print(label)
    total = total + value
end
";

/// **The headline.** A handler calls its unit's own functions, by name, in both
/// positions — bound to a local, nested inside another call's argument, and as a
/// bare statement — and the effects land in order.
#[test]
fn a_handler_calls_its_units_own_functions() {
    let log = run(DOUBLER);
    assert_eq!(
        log,
        vec![
            "debug::print(Str(\"doubled\"))".to_string(),
            "debug::print(Str(\"again\"))".to_string(),
        ],
        "the two `announce` calls should reach the host in source order"
    );

    // And the member variable the callee wrote is the caller's: 8 + 16 = 24.
    let class = class_of(DOUBLER);
    let mut actor = ActorInstance::new(&class);
    let mut host = LogHost::default();
    run_event(
        &class,
        &mut actor,
        &EventKind::BeginPlay,
        &HashMap::new(),
        &mut host,
        &Debug::default(),
    )
    .expect("runs");
    assert_eq!(
        actor.get("total"),
        Some(&Value::Float(24.0)),
        "a called function writes the SAME actor's variables — it is one \
         `ActorInstance`, not a fresh scope"
    );
}

/// The IR is a one-segment [`Expr::Call`] and **nothing else**: no new variant,
/// no marker, no synthetic namespace. This is the pricing, asserted.
#[test]
fn the_call_form_moved_no_wire() {
    let class = class_of(DOUBLER);
    assert_eq!(
        class.schema_version,
        inf_blueprint::semantics::SCHEMA_VERSION
    );
    let begin = class
        .handler(&EventKind::BeginPlay)
        .expect("a begin_play handler");
    let Stmt::Let { value, .. } = &begin.body.body[0] else {
        panic!(
            "the first statement binds the call: {:?}",
            begin.body.body[0]
        );
    };
    let Expr::Call { path, args } = value else {
        panic!("a call: {value:?}");
    };
    assert_eq!(
        path,
        &["double".to_string()],
        "one segment, and that is all"
    );
    assert_eq!(args.len(), 1);

    // The whole class survives the `.inf_act` wire — which is JSON, so a
    // variant travels as its name and this one is a variant that already
    // existed. A byte-for-byte re-encode is the check that nothing about the
    // shape needed a migration.
    let json = serde_json::to_vec_pretty(&class).expect("encode");
    let back: BlueprintClass = serde_json::from_slice(&json).expect("decode");
    assert_eq!(back, class, "the call form round-trips the .inf_act wire");
    assert_eq!(
        serde_json::to_vec_pretty(&back).expect("re-encode"),
        json,
        "byte-identical re-encode"
    );
}

/// A function with no `->` produces no value, and using one in an expression is
/// a refusal that names the remedy rather than a type error three steps away.
#[test]
fn a_function_with_no_return_type_cannot_be_used_as_a_value() {
    let src = "\
function shout(m: string)
    debug.print(m)
end

on begin_play()
    local x = shout(\"hi\") + 1.0
end
";
    let diags = parse_unit(src).expect_err("refused");
    let text = render(&diags);
    assert!(
        text.contains("`shout` returns nothing")
            && text.contains("call it as a statement")
            && text.contains("-> float"),
        "{text}"
    );
}

/// Arity is **exact** for a local call, unlike a verb's (whose trailing inputs
/// have literal defaults the lowerer supplies). Both directions.
#[test]
fn a_local_call_checks_its_arity_exactly() {
    for (src, want) in [
        (
            "function two(a: float, b: float) -> float\n    return a + b\nend\n\
             on begin_play()\n    local x = two(1.0)\nend\n",
            "`two` takes 2 arguments (`a`, `b`), not 1",
        ),
        (
            "function none() -> float\n    return 1.0\nend\n\
             on begin_play()\n    local x = none(1.0)\nend\n",
            "`none` takes no arguments, not 1",
        ),
        (
            "function one(a: float) -> float\n    return a\nend\n\
             on begin_play()\n    local x = one(1.0, 2.0)\nend\n",
            "`one` takes 1 argument (`a`), not 2",
        ),
    ] {
        let diags = parse_unit(src).expect_err("refused");
        let text = render(&diags);
        assert!(text.contains(want), "wanted {want:?}, got {text}");
    }
}

/// A name that is **not** declared still gets the registry's refusal — and the
/// message now says how to make it a call rather than only that it is not one.
#[test]
fn an_undeclared_bare_call_is_still_refused() {
    let src = "on begin_play()\n    frobnicate(1.0)\nend\n";
    let diags = parse_unit(src).expect_err("refused");
    let text = render(&diags);
    assert!(
        text.contains("`frobnicate` is not a verb") && text.contains("function frobnicate("),
        "{text}"
    );
}

/// **Recursion is refused where a designer can see it**, direct and mutual, and
/// the message says why and what to write instead.
#[test]
fn recursion_is_refused_by_the_parser() {
    let direct = "\
function down(n: float) -> float
    if n > 0.0 then
        return down(n - 1.0)
    end
    return 0.0
end

on begin_play()
    local x = down(3.0)
end
";
    let text = render(&parse_unit(direct).expect_err("refused"));
    assert!(
        text.contains("`down` calls itself (down → down)")
            && text.contains("InfiniScript has no recursion")
            && text.contains("`while` or `for` loop"),
        "{text}"
    );

    let mutual = "\
function ping(n: float) -> float
    return pong(n)
end

function pong(n: float) -> float
    return ping(n)
end

on begin_play()
    local x = ping(1.0)
end
";
    let text = render(&parse_unit(mutual).expect_err("refused"));
    assert!(
        text.contains("`ping` calls itself (ping → pong → ping)"),
        "the message should name the route it found, not merely the fault: {text}"
    );

    // Anti-vacuity: the same shapes without the back-edge compile.
    let chain = "\
function a(n: float) -> float
    return b(n) + 1.0
end

function b(n: float) -> float
    return c(n) + 1.0
end

function c(n: float) -> float
    return n
end

on begin_play()
    local x = a(1.0)
end
";
    parse_unit(chain).unwrap_or_else(|d| panic!("a call DAG is legal: {}", render(&d)));
}

/// **The interpreter's own bound**, for IR that did not come through the parser.
///
/// A hand-built class whose function calls itself is exactly what a hand-edited
/// `.inf_act` or a lift of hand-written Rust can hold, and the parser's static
/// refusal cannot reach it. Without the budget this is `STATUS_STACK_OVERFLOW`
/// in whatever process is ticking; with it, it is a value with a sentence.
#[test]
fn hand_built_recursive_ir_refuses_instead_of_overflowing_the_stack() {
    let mut class = BlueprintClass::new("act:recurse", "Recurse");
    // function loop_forever(n: float) -> float { return loop_forever(n); }
    class.functions.push(BlueprintFn {
        id: "loop_forever".into(),
        name: "loop_forever".into(),
        params: vec![Param {
            name: "n".into(),
            ty: Ty::Float,
        }],
        ret: Ty::Float,
        body: vec![Stmt::Return(Some(Expr::Call {
            path: vec!["loop_forever".into()],
            args: vec![Expr::Param("n".into())],
        }))],
    });
    class.events.push(inf_blueprint::semantics::EventBinding {
        event: EventKind::BeginPlay,
        body: BlueprintFn {
            id: "begin_play".into(),
            name: "begin_play".into(),
            params: vec![],
            ret: Ty::Unit,
            body: vec![Stmt::Let {
                id: LocalId(0),
                binding: Binding::Named("x".into()),
                ty: None,
                mutable: false,
                value: Expr::Call {
                    path: vec!["loop_forever".into()],
                    args: vec![Expr::Lit(Lit::Float(1.0))],
                },
            }],
        },
    });

    let mut actor = ActorInstance::new(&class);
    let mut host = LogHost::default();
    let err = run_event(
        &class,
        &mut actor,
        &EventKind::BeginPlay,
        &HashMap::new(),
        &mut host,
        &Debug::default(),
    )
    .expect_err("the budget refuses");
    let text = err.to_string();
    assert!(
        text.contains(&MAX_CALL_DEPTH.to_string()) && text.contains("loop_forever"),
        "the refusal names its budget and the function it ran out on: {text}"
    );
}

/// A call chain **just inside** the budget runs. Without this the arm above is
/// satisfied by a budget of zero.
#[test]
fn a_chain_inside_the_budget_runs() {
    // f0 calls f1 calls … f_{n-1}; a DAG, so the parser is happy with it.
    let depth = 20;
    let mut src = String::from("actor \"Chain\"\n");
    for i in 0..depth {
        let body = if i + 1 == depth {
            "    return 1.0\n".to_string()
        } else {
            format!("    return f{}() + 1.0\n", i + 1)
        };
        src.push_str(&format!("\nfunction f{i}() -> float\n{body}end\n"));
    }
    src.push_str("\non begin_play()\n    debug.print(\"start\")\n    total = f0()\nend\n");
    let class = class_of(&src);
    let mut actor = ActorInstance::new(&class);
    let mut host = LogHost::default();
    run_event(
        &class,
        &mut actor,
        &EventKind::BeginPlay,
        &HashMap::new(),
        &mut host,
        &Debug::default(),
    )
    .expect("a 20-deep chain is inside the budget");
    assert_eq!(actor.get("total"), Some(&Value::Float(depth as f64)));
}

/// A one-segment call that names **no** declared function reaches the engine
/// host, where both real hosts log it and answer `Unit` — the same
/// "a partially-authored blueprint still runs" behaviour every unknown call
/// gets, rather than a new failure mode.
#[test]
fn an_unresolved_bare_call_falls_through_to_the_host() {
    let mut class = BlueprintClass::new("act:fallthrough", "Fallthrough");
    class.events.push(inf_blueprint::semantics::EventBinding {
        event: EventKind::BeginPlay,
        body: BlueprintFn {
            id: "begin_play".into(),
            name: "begin_play".into(),
            params: vec![],
            ret: Ty::Unit,
            body: vec![Stmt::ExprStmt(Expr::Call {
                path: vec!["nobody_declared_this".into()],
                args: vec![],
            })],
        },
    });
    let mut actor = ActorInstance::new(&class);
    let mut host = LogHost::default();
    run_event(
        &class,
        &mut actor,
        &EventKind::BeginPlay,
        &HashMap::new(),
        &mut host,
        &Debug::default(),
    )
    .expect("the handler survives");
    assert_eq!(host.log, vec!["nobody_declared_this()".to_string()]);
}

/// **The round trip**, over a unit that uses the form: the text is a fixed
/// point, and the IR survives.
#[test]
fn a_unit_using_the_call_form_round_trips() {
    let class = class_of(DOUBLER);
    let text = emit_class(&class).expect("emits");
    assert!(
        text.contains("local a = double(4.0)") && text.contains("announce(\"again\", double(a))"),
        "a local call prints bare, exactly as it was written:\n{text}"
    );
    let again = class_of(&text);
    assert_eq!(again, class, "parse(emit(f)) == f");
    assert_eq!(
        emit_class(&again).expect("re-emits"),
        text,
        "emit(parse(emit(f))) == emit(f)"
    );
    assert_eq!(
        text, DOUBLER,
        "and for this corpus the stronger law holds too: emit(parse(src)) == src"
    );
}

/// **`raise` learned it as a refusal**, in both positions, and the refusal names
/// the function.
///
/// Before this the statement form went to `raise_action`, which added a graph
/// node whose `type_id` no registry knows — a malformed graph rather than a
/// verdict. The value form was folded into the generic `pure call`, whose remedy
/// does not apply.
#[test]
fn raise_refuses_the_call_form_by_name() {
    let class = class_of(DOUBLER);
    let begin = class.handler(&EventKind::BeginPlay).expect("handler");
    match raise_fn(&begin.body) {
        Err(RaiseError::LocalFunctionCall(name)) => assert_eq!(name, "double"),
        other => panic!("expected a named refusal, got {other:?}"),
    }

    // Statement position, on its own.
    let stmt_only = class_of(
        "function tap()\n    debug.print(\"tap\")\nend\n\
         on begin_play()\n    tap()\nend\n",
    );
    let begin = stmt_only.handler(&EventKind::BeginPlay).expect("handler");
    match raise_fn(&begin.body) {
        Err(RaiseError::LocalFunctionCall(name)) => assert_eq!(name, "tap"),
        other => panic!("expected a named refusal, got {other:?}"),
    }
}
