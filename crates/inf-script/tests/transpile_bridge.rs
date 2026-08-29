//! **The third leg: text → IR → Rust → IR.**
//!
//! `roundtrip.rs` closes text↔IR and `raise_coverage.rs` closes IR↔graph. This
//! file closes IR↔Rust *for script-authored IR specifically*, which nothing else
//! does: `inf-transpile`'s own 38 proptests and its hand-edit corpus generate
//! their IR from graphs and from Rust, and neither producer makes a
//! `Binding::Raw` binder, a `Stmt::Assign`, a non-terminal `if`, or a `for` loop
//! whose index carries an author's name. A text face produces all four on the
//! first day somebody writes a script.
//!
//! Three claims, in increasing strength:
//!
//! 1. **The transpiler accepts everything the language can say.** Every handler
//!    in the corpus renders to Rust without an `EmitError`.
//! 2. **`lift` recovers it structurally**, with no verbatim fallback and no
//!    warning — so a designer's script is *editable Rust*, not an opaque blob
//!    inside a generated file.
//! 3. **Regeneration is byte-idempotent, and the program is unchanged** —
//!    `generate(lift(generate(f))) == generate(f)`, and the lifted IR runs the
//!    same host calls in the same order. Byte-idempotence rather than IR
//!    equality because `lift` re-derives the ids of `Raw` binders, exactly as
//!    `Binding::Raw`'s own doc says it does; the *program* is what must not move.
//!
//! # What this is NOT
//!
//! **It does not compile the generated Rust, and it does not run it.** No test
//! in this repository does — `parity.rs`'s own module doc calls itself "the
//! CI-cheap half of the parity story (no runtime `cargo build`)", and the four
//! parity families each run the interpreter against a *hand-written* Rust mirror
//! that a string pin ties to `generate_fn`'s output. Closing that is SCRIPT1b's
//! **crown gate**: cook a `.infini` script, build the generated crate, run it,
//! and diff the trace against the interpreter. This file narrows what that gate
//! has left to prove; it does not stand in for it, and saying otherwise would be
//! the exact over-claim the SCRIPT0 audit's finding 5 caught.

use std::collections::HashMap;

use inf_blueprint::interp::{Host, RunError, Value};
use inf_blueprint::BlueprintFn;
use inf_script::{compile, render};
use inf_transpile::{generate_fn, lift_file, FileEntry};

/// Scripts that between them produce every shape the transpiler has to accept
/// from a text author — including the four no graph and no lift ever makes.
const SCRIPTS: &[(&str, &str)] = &[
    (
        "the rotate-on-tick handler",
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
        "a NON-terminal branch and a local assignment — neither of which raises",
        r#"actor "Mixed"

var hits: int = 0

on collision(other)
    local n = hits
    if other > 0 then
        n = n + 1
    end
    hits = n
    debug.print("counted")
end
"#,
    ),
    (
        "both loop forms, nested, with a named index",
        r#"actor "Loops"

var total: int = 0
var budget: int = 3

on begin_play()
    for row = 0, 2 do
        while budget > 0 do
            budget = budget - 1
        end
        total = total + row
    end
end
"#,
    ),
    (
        "the math builtins, which parity holds for by construction",
        r#"actor "Maths"

var out: float = 0.0

on tick(dt)
    out = math.clamp(math.lerp(math.sqrt(dt), 10.0, 0.5), -(1.0), 6.0)
end
"#,
    ),
    (
        "a function with a return type, and an action bound to a local",
        r#"actor "Library"

on begin_play()
    local e = engine.spawn("enemy")
    engine.destroy(e)
end

function double(x: float) -> float
    return x * 2.0
end
"#,
    ),
    (
        "every event header",
        r#"actor "Events"

on begin_play()
    debug.print("a")
end

on input "jump"(pressed)
    if pressed then
        debug.print("b")
    end
end

on custom "ping"()
    dispatch.call(0, "pong")
end

on water_splash(water, speed)
    debug.print("c")
end

on destroyed(chunks)
    debug.print("d")
end
"#,
    ),
];

/// Every handler and function in the corpus.
fn handlers() -> Vec<(&'static str, BlueprintFn)> {
    let mut out = Vec::new();
    for (what, src) in SCRIPTS {
        let (class, warnings) = compile(src, "act:bridge")
            .unwrap_or_else(|d| panic!("{what} did not compile:\n{}", render(&d)));
        assert!(warnings.is_empty(), "{what} warns:\n{}", render(&warnings));
        for b in class.events {
            out.push((*what, b.body));
        }
        for f in class.functions {
            out.push((*what, f));
        }
    }
    out
}

/// **Claim 1 + 2**: the transpiler renders every script, and `lift` recovers it
/// structurally rather than keeping it verbatim.
#[test]
fn every_script_renders_to_rust_and_lifts_back_structurally() {
    let hs = handlers();
    assert!(
        hs.len() >= 10,
        "only {} handlers in the bridge corpus",
        hs.len()
    );
    for (what, f) in &hs {
        let rust = generate_fn(f).unwrap_or_else(|e| panic!("{what}/{}: {e}", f.id));
        assert!(
            rust.contains("infinity :: blueprint") || rust.contains("infinity::blueprint"),
            "{what}/{}: the generated fn lost its identity attribute:\n{rust}",
            f.id
        );
        let lifted = lift_file(&rust).unwrap_or_else(|e| panic!("{what}/{}: {e}", f.id));
        assert!(
            lifted.warnings.is_empty(),
            "{what}/{}: lift kept it verbatim: {:?}\n{rust}",
            f.id,
            lifted.warnings
        );
        let entries = &lifted.file.entries;
        assert_eq!(entries.len(), 1, "{what}/{}: {entries:#?}", f.id);
        assert!(
            matches!(entries[0], FileEntry::Blueprint(_)),
            "{what}/{}: not recovered as a blueprint:\n{rust}",
            f.id
        );
    }
}

/// **Claim 3**: regeneration is byte-idempotent and the program does not move.
#[test]
fn regeneration_is_byte_idempotent_and_the_program_is_unchanged() {
    for (what, f) in handlers() {
        let once = generate_fn(&f).unwrap();
        let lifted = lift_file(&once).unwrap();
        let FileEntry::Blueprint(back) = &lifted.file.entries[0] else {
            panic!("{what}/{}: not a blueprint", f.id)
        };
        let twice = generate_fn(back).unwrap();
        assert_eq!(
            once, twice,
            "{what}/{}: the generated Rust moved on a second pass",
            f.id
        );
        assert_eq!(
            observe(&f),
            observe(back),
            "{what}/{}: the lifted IR runs a different program",
            f.id
        );
        // The identity survives, which is what lets the Code tab find the
        // function again after a hand edit.
        assert_eq!(back.id, f.id, "{what}: the blueprint id moved");
        assert_eq!(back.name, f.name, "{what}: the fn name moved");
        assert_eq!(back.params, f.params, "{what}: the signature moved");
    }
}

/// **The one literal the language can write and the transpiler cannot render.**
///
/// `i64::MIN` has no Rust source spelling the lifter can fold back
/// (`-9223372036854775808` is `-(9223372036854775808)` and the magnitude
/// overflows), so `inf_transpile::emit` refuses it by name. It is **not** a hole
/// this crate opens — a `lit.int` node holds it just as happily — and the
/// interpreter computes with it perfectly, so a script using it previews and
/// does not cook.
///
/// Pinned rather than discovered: SCRIPT1b's cook has to report it as an
/// advisory a designer can act on, and this arm is what tells that wave the
/// bound is still there.
#[test]
fn the_one_literal_a_script_can_write_and_the_cook_cannot_render() {
    let (class, _) = compile(
        "on begin_play()\n    local floor: int = -9223372036854775808\n    debug.print(\"x\")\nend\n",
        "act:intmin",
    )
    .expect("i64::MIN parses, and the interpreter is happy with it");
    let f = &class.events[0].body;
    let err = generate_fn(f).expect_err("the transpiler refuses i64::MIN");
    assert!(
        err.to_string().contains("i64::MIN"),
        "the refusal should name the literal: {err}"
    );
    // …and the interpreter really does run it, which is what makes the bound a
    // *cook* bound rather than a language one.
    assert!(!observe(f).is_empty());
}

/// **The carried gap, armed.** The language no longer produces `Unary(Neg,
/// Lit)` — the parser folds it and the emitter prints the canonical negative
/// literal — but the **graph** lowerer still can: a `math.neg` node wired to a
/// `lit.float` is exactly that shape, and `inf_transpile::emit` refuses it, so
/// the Code tab cannot generate for such a graph.
///
/// SCRIPT1a carried that by name and the SCRIPT1a audit gives it a tripwire,
/// because a carried item with no arm is a sentence that goes stale silently.
/// This builds the graph, lowers it, and asserts the refusal — so the day
/// `inf-blueprint` closes the gap (fold at lowering, or teach `emit` the
/// canonicalisation and re-prove `generate → lift`) **this test fails**, and the
/// ledger item gets retired by the failure rather than by somebody remembering.
///
/// It also asserts the *text* face is unaffected, which is the asymmetry the
/// carry is about: the same IR prints as `-1.5` and reads back as a literal.
#[test]
fn a_math_neg_node_on_a_literal_still_lowers_to_ir_the_cook_refuses() {
    use inf_blueprint::lower::lower_graph;
    use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
    use inf_blueprint::{Expr, Lit, Stmt, UnOp};
    use inf_graph::{Graph, Link, NodeUi, ParamValue};

    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let lit = g.insert("lit.float", NodeUi::default());
    g.node_mut(lit)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Float(1.5));
    let neg = g.insert("math.neg", NodeUi::default());
    let set = g.insert("var.set", NodeUi::default());
    g.node_mut(set)
        .unwrap()
        .params
        .insert("name".into(), ParamValue::Text("angle".into()));
    for (from, fp, to, tp) in [
        (lit, "value", neg, "a"),
        (neg, "out", set, "value"),
        (bp, EXEC_THEN, set, "exec"),
    ] {
        g.links.push(Link {
            from,
            from_port: fp.into(),
            to,
            to_port: tp.into(),
        });
    }
    let f = lower_graph(&g, &reg).unwrap().pop().unwrap();

    // The shape really is the one the carry names.
    let Stmt::ExprStmt(Expr::Call { args, .. }) = &f.body[0] else {
        panic!("unexpected lowering: {:#?}", f.body)
    };
    assert!(
        matches!(&args[1], Expr::Unary(UnOp::Neg, inner) if matches!(inner.as_ref(), Expr::Lit(Lit::Float(_)))),
        "the graph no longer lowers to `Unary(Neg, Lit)` — if that is deliberate, \
         retire carried item 4: {:#?}",
        args[1]
    );

    // The cook refuses it. THIS is the gap.
    let err = generate_fn(&f).expect_err(
        "`inf_transpile::emit` accepted `Unary(Neg, Lit)` — the SCRIPT1a carried \
         item 4 is CLOSED and this arm plus the ledger entry should go",
    );
    assert!(
        format!("{err:?}").contains("NegatedLiteral"),
        "refused for a different reason: {err}"
    );

    // …and the text face is fine with it, which is the asymmetry.
    let text = inf_script::emit_fn(&f).expect("text prints it");
    assert!(text.contains("-1.5"), "{text}");
    inf_script::parse_fn(&text).expect("and reads it back");
}

/// The three shapes a graph author never produces, present and rendered.
///
/// Without this the corpus could quietly become "things a graph could also have
/// said", and the file would be re-testing what `inf-transpile` already covers.
#[test]
fn the_corpus_really_does_carry_the_shapes_only_text_produces() {
    use inf_blueprint::{Binding, Stmt};
    let (mut raw, mut assign, mut nonterminal_if) = (false, false, false);
    fn walk(body: &[Stmt], raw: &mut bool, assign: &mut bool, nonterminal_if: &mut bool) {
        for (i, s) in body.iter().enumerate() {
            match s {
                Stmt::Let {
                    binding: Binding::Raw(_),
                    ..
                } => *raw = true,
                Stmt::Assign { .. } => *assign = true,
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    if i + 1 != body.len() {
                        *nonterminal_if = true;
                    }
                    walk(then_body, raw, assign, nonterminal_if);
                    walk(else_body, raw, assign, nonterminal_if);
                }
                Stmt::While { body, .. } => walk(body, raw, assign, nonterminal_if),
                _ => {}
            }
        }
    }
    for (_, f) in handlers() {
        walk(&f.body, &mut raw, &mut assign, &mut nonterminal_if);
    }
    assert!(
        raw,
        "no `Binding::Raw` — a graph makes none, so the corpus drifted"
    );
    assert!(assign, "no `Stmt::Assign`");
    assert!(nonterminal_if, "no non-terminal `Stmt::If`");
}

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
                Ok(self.vars.get(&n).cloned().unwrap_or(Value::Int(2)))
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

fn observe(f: &BlueprintFn) -> Vec<String> {
    let mut host = Recorder::default();
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
    let _ = inf_blueprint::eval_fn(f, &args, &mut host);
    host.log
}
