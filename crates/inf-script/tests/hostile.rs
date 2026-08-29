//! **Hostile input.** A language is the largest attack surface a wave can ship,
//! and this file is the half of the gate that does not care what the grammar
//! says: it feeds the compiler things nobody would write and asserts that every
//! one of them comes back as a *value*.
//!
//! Three claims, each of which the SCRIPT1a audit found false before it was
//! true:
//!
//! 1. **Nesting is bounded.** A recursive-descent parser building a tree
//!    recurses once per level, and so does every consumer of that tree — the
//!    emitter, the interpreter, the transpiler and `Drop`. Unguarded,
//!    `((((…))))` was `STATUS_STACK_OVERFLOW` rather than a refusal, at about
//!    600 levels on a 1 MiB stack: a *crash*, in a library the editor calls on
//!    every keystroke. [`crate::MAX_NESTING`](inf_script::parse::MAX_NESTING)
//!    bounds it and the refusal names the remedy.
//! 2. **Anything the emitter writes, the parser reads back.** An emitter that
//!    produces a file its own parser rejects is worse than one that refuses,
//!    because the refusal is silent until somebody opens the result. Two shapes
//!    did exactly that — a comparison in a comparison's left operand, and an
//!    expression in statement position — and the first is reachable by drawing
//!    two nodes on the canvas.
//! 3. **Nothing panics.** Not on a truncated file, not on a mutated one, not on
//!    a hundred-megabyte line, not on `i64::MIN` written in decimal.

use inf_blueprint::lower::lower_graph;
use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
use inf_blueprint::{BinOp, Binding, BlueprintFn, Expr, Lit, LocalId, Stmt, Ty};
use inf_graph::{Graph, Link, NodeId, NodeUi, ParamValue};
use inf_script::parse::MAX_NESTING;
use inf_script::{compile, emit_class, emit_fn, parse_fn, EmitError};

// ── 1. the nesting bound ────────────────────────────────────────────────────

fn parens(n: usize) -> String {
    format!(
        "on tick(dt)\n    local a = {}1{}\nend\n",
        "(".repeat(n),
        ")".repeat(n)
    )
}

fn nested_ifs(n: usize) -> String {
    let mut s = String::from("on tick(dt)\n");
    for _ in 0..n {
        s.push_str("if dt > 0.0 then\n");
    }
    s.push_str("debug.print(\"x\")\n");
    for _ in 0..n {
        s.push_str("end\n");
    }
    s.push_str("end\n");
    s
}

fn unary_chain(word: &str, n: usize, tail: &str) -> String {
    format!("on tick(dt)\n    local a = {}{tail}\nend\n", word.repeat(n))
}

/// `1 + 1 + … + 1`. No syntactic nesting at all — the parser reads it in a loop
/// — and yet the tree it builds is `n` deep, which is what killed the *emitter*
/// at ten thousand terms. The bound is on the tree, not on the parser's stack.
fn flat_chain(n: usize) -> String {
    format!(
        "on tick(dt)\n    local a = {}\nend\n",
        vec!["1"; n].join(" + ")
    )
}

/// **A refusal, not a crash.** Every one of these overflowed the stack before
/// the SCRIPT1a audit; a stack overflow is not catchable and takes the process
/// that hosts the parser.
#[test]
fn nesting_past_the_bound_refuses_instead_of_overflowing_the_stack() {
    for (what, src) in [
        ("parentheses", parens(10_000)),
        ("nested `if`", nested_ifs(10_000)),
        ("`not` chain", unary_chain("not ", 10_000, "true")),
        ("unary minus chain", unary_chain("- ", 10_000, "dt")),
        ("a flat operator chain", flat_chain(10_000)),
    ] {
        let diags = compile(&src, "act:deep")
            .err()
            .unwrap_or_else(|| panic!("{what}: {} levels compiled", 10_000));
        let first = diags.first().expect("a refusal carries a diagnostic");
        assert!(
            first.message.contains("nested more than"),
            "{what}: refused for the wrong reason: {}",
            first.message
        );
        assert!(first.span.line >= 1, "{what}: {first}");
    }
}

/// The falsification half: the bound is a **bound**, not a ban. A construction
/// at exactly `MAX_NESTING` parses, and the whole pipeline survives it — parse,
/// emit, re-parse, identical IR. That is what proves the emitter's own bound is
/// loose enough that *anything the parser accepts, the emitter can write*.
#[test]
fn a_construction_at_the_bound_parses_and_round_trips() {
    let n = MAX_NESTING as usize - 4; // the handler, its block and the `local`
    for (what, src) in [
        ("parentheses", parens(n)),
        ("`not` chain", unary_chain("not ", n, "true")),
        ("a flat operator chain", flat_chain(n)),
    ] {
        let (class, _) = compile(&src, "act:deep")
            .unwrap_or_else(|d| panic!("{what} at {n}: {}", inf_script::render(&d)));
        let text = emit_class(&class).unwrap_or_else(|e| panic!("{what} at {n}: emit: {e}"));
        let (again, _) = compile(&text, "act:deep")
            .unwrap_or_else(|d| panic!("{what} at {n}: re-parse: {}", inf_script::render(&d)));
        assert_eq!(again, class, "{what} at {n}: the IR moved");
    }
}

/// A **graph** can chain operator nodes without limit, and the canvas is where
/// that comes from. The emitter refuses such IR by name rather than recursing
/// into the stack guard page.
#[test]
fn the_emitter_refuses_ir_deeper_than_any_text_could_hold() {
    let mut e = Expr::Lit(Lit::Int(1));
    for _ in 0..2_000 {
        e = Expr::Binary(BinOp::Add, Box::new(e), Box::new(Expr::Lit(Lit::Int(1))));
    }
    let f = BlueprintFn {
        id: "begin_play".into(),
        name: "begin_play".into(),
        params: vec![],
        ret: Ty::Unit,
        body: vec![Stmt::Let {
            id: LocalId(0),
            binding: Binding::Anon,
            ty: None,
            mutable: false,
            value: e,
        }],
    };
    assert_eq!(emit_fn(&f), Err(EmitError::TooDeep));
}

// ── 2. anything the emitter writes, the parser reads back ───────────────────

const OPS: [BinOp; 13] = [
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
];

fn let_of(e: Expr) -> BlueprintFn {
    BlueprintFn {
        id: "begin_play".into(),
        name: "begin_play".into(),
        params: vec![],
        ret: Ty::Unit,
        body: vec![Stmt::Let {
            id: LocalId(0),
            binding: Binding::Raw("a".into()),
            ty: None,
            mutable: false,
            value: e,
        }],
    }
}

/// **Every operator against every operator, on both sides.** 338 trees; each is
/// printed and read back and must be the same IR.
///
/// 36 of them failed before the fix, and they were all one shape: a comparison
/// nested in a comparison's **left** operand. The emitter parenthesised the
/// right operand of a non-associating operator and not the left, so
/// `Binary(Eq, Binary(Lt, …), …)` printed as `1 < 2 == 3` — which the grammar
/// refuses, on purpose, as a chained comparison.
#[test]
fn every_operator_pairing_the_emitter_writes_reads_back_identically() {
    let mut broken: Vec<String> = Vec::new();
    let mut checked = 0;
    for outer in OPS {
        for inner in OPS {
            for side in 0..2 {
                let sub = Expr::Binary(
                    inner,
                    Box::new(Expr::Lit(Lit::Int(1))),
                    Box::new(Expr::Lit(Lit::Int(2))),
                );
                let e = if side == 0 {
                    Expr::Binary(outer, Box::new(sub), Box::new(Expr::Lit(Lit::Int(3))))
                } else {
                    Expr::Binary(outer, Box::new(Expr::Lit(Lit::Int(3))), Box::new(sub))
                };
                let f = let_of(e);
                checked += 1;
                let text = match emit_fn(&f) {
                    Ok(t) => t,
                    Err(err) => {
                        broken.push(format!("{outer:?}/{inner:?}/{side}: emit refused: {err}"));
                        continue;
                    }
                };
                match parse_fn(&text) {
                    Ok(back) if back == f => {}
                    Ok(_) => broken.push(format!(
                        "{outer:?}/{inner:?}/{side}: the IR moved: {}",
                        text.replace('\n', " ⏎ ")
                    )),
                    Err(d) => broken.push(format!(
                        "{outer:?}/{inner:?}/{side}: emitted text does not parse ({}): {}",
                        d.first().map(|x| x.to_string()).unwrap_or_default(),
                        text.replace('\n', " ⏎ ")
                    )),
                }
            }
        }
    }
    assert_eq!(
        checked, 338,
        "the sweep stopped covering the operator table"
    );
    assert!(broken.is_empty(), "{}", broken.join("\n"));
}

fn wire(g: &mut Graph, from: NodeId, fp: &str, to: NodeId, tp: &str) {
    g.links.push(Link {
        from,
        from_port: fp.into(),
        to,
        to_port: tp.into(),
    });
}

/// …and the pairing above is not a hypothetical about the IR: **it is three
/// nodes on a canvas.** `cmp.lt` wired into `cmp.eq`'s `a` input, the result
/// driving a `flow.branch`. Built from a graph so the claim is about what a
/// designer can draw, not about a `Box::new` a test wrote.
#[test]
fn a_comparison_wired_into_a_comparison_opens_as_text_and_comes_back() {
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let lit = |g: &mut Graph, v: f64| {
        let n = g.insert("lit.float", NodeUi::default());
        g.node_mut(n)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Float(v));
        n
    };
    let a = lit(&mut g, 1.0);
    let b = lit(&mut g, 2.0);
    let lt = g.insert("cmp.lt", NodeUi::default());
    wire(&mut g, a, "value", lt, "a");
    wire(&mut g, b, "value", lt, "b");
    let t = g.insert("lit.bool", NodeUi::default());
    g.node_mut(t)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Bool(true));
    let eq = g.insert("cmp.eq", NodeUi::default());
    wire(&mut g, lt, "out", eq, "a");
    wire(&mut g, t, "value", eq, "b");
    let br = g.insert("flow.branch", NodeUi::default());
    wire(&mut g, eq, "out", br, "condition");
    wire(&mut g, bp, EXEC_THEN, br, "exec");
    let p = g.insert("debug.print", NodeUi::default());
    g.node_mut(p)
        .unwrap()
        .params
        .insert("message".into(), ParamValue::Text("yes".into()));
    wire(&mut g, br, "true", p, "exec");

    let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
    let text = emit_fn(&f).expect("the graph's IR prints");
    assert!(
        text.contains("if (1.0 < 2.0) == true then"),
        "the nested comparison lost its parentheses: {text}"
    );
    let back = parse_fn(&text)
        .unwrap_or_else(|d| panic!("{}\n--- text ---\n{text}", inf_script::render(&d)));
    assert_eq!(back.body, f.body, "the round trip changed the program");
}

/// **A statement position holds a call and nothing else.** Four `ExprStmt`
/// shapes printed happily and re-parsed not at all; each is now a named refusal,
/// which is the same verdict `raise` gives them
/// (`UnsupportedStmt("non-call expr stmt")`).
#[test]
fn an_expression_in_statement_position_is_refused_rather_than_misprinted() {
    for e in [
        Expr::Lit(Lit::Int(1)),
        Expr::Local(LocalId(0)),
        Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Lit(Lit::Int(1))),
            Box::new(Expr::Lit(Lit::Int(2))),
        ),
        // `vars::get` prints as a bare name, which is not a statement…
        Expr::Call {
            path: vec!["vars".into(), "get".into()],
            args: vec![Expr::Lit(Lit::Str("x".into()))],
        },
        // …and `nodestate::get_or` has a value spelling and no statement one.
        Expr::Call {
            path: vec!["nodestate".into(), "get_or".into()],
            args: vec![Expr::Lit(Lit::Str("k".into())), Expr::Lit(Lit::Bool(false))],
        },
    ] {
        let f = BlueprintFn {
            id: "begin_play".into(),
            name: "begin_play".into(),
            params: vec![],
            ret: Ty::Unit,
            body: vec![
                Stmt::Let {
                    id: LocalId(0),
                    binding: Binding::Raw("q".into()),
                    ty: None,
                    mutable: false,
                    value: Expr::Lit(Lit::Int(0)),
                },
                Stmt::ExprStmt(e.clone()),
            ],
        };
        assert!(
            matches!(emit_fn(&f), Err(EmitError::UnspellableStatement(_))),
            "{e:?} was written as a statement: {:?}",
            emit_fn(&f)
        );
    }
    // The falsifier: an ordinary action *is* a statement, and a pure builtin in
    // statement position is one too.
    for path in [
        vec!["debug".to_string(), "print".into()],
        vec!["math".to_string(), "sqrt".into()],
    ] {
        let f = BlueprintFn {
            id: "begin_play".into(),
            name: "begin_play".into(),
            params: vec![],
            ret: Ty::Unit,
            body: vec![Stmt::ExprStmt(Expr::Call {
                path,
                args: vec![Expr::Lit(Lit::Str("x".into()))],
            })],
        };
        let text = emit_fn(&f).expect("a call is a statement");
        assert!(parse_fn(&text).is_ok(), "{text}");
    }
}

// ── 3. nothing panics ───────────────────────────────────────────────────────

/// Pathological *lexical* input: things a fuzzer finds and a person occasionally
/// types. Every one must be a value — an `Ok` or a diagnostic with a line — and
/// none may panic, hang or allocate the machine to death.
#[test]
fn pathological_input_is_always_a_value() {
    let cases: Vec<(&str, String)> = vec![
        (
            "an unterminated string at end of file",
            "on tick(dt)\n    debug.print(\"x".into(),
        ),
        (
            "a backslash at end of file",
            "on tick(dt)\n    debug.print(\"x\\".into(),
        ),
        (
            "a hundred thousand digits",
            format!("on tick(dt)\n    local a = {}\nend\n", "9".repeat(100_000)),
        ),
        (
            "an exponent that overflows",
            "on tick(dt)\n    local a = 1e999999999\nend\n".into(),
        ),
        (
            "a long bracket opened at level 100000",
            format!("on tick(dt)\n    rust [{}[\n", "=".repeat(100_000)),
        ),
        (
            "a `\\u` escape that overflows u32",
            "on tick(dt)\n    debug.print(\"\\u{ffffffffffffffff}\")\nend\n".into(),
        ),
        (
            "an empty `\\u{}`",
            "on tick(dt)\n    debug.print(\"\\u{}\")\nend\n".into(),
        ),
        (
            "a lone surrogate",
            "on tick(dt)\n    debug.print(\"\\u{d800}\")\nend\n".into(),
        ),
        (
            "a ten-thousand-segment path",
            format!("on tick(dt)\n    a{}()\nend\n", ".b".repeat(10_000)),
        ),
        // **The SCRIPT2a audit's second finding, and the reason this row is a
        // SIZE rather than a shape.** The call graph gets two walks (no cycle,
        // no chain past `MAX_CALL_DEPTH`), and the first version of them looked
        // a callee up by SCANNING the name list — three linear scans nested
        // inside two walks. Measured, in release, on a file well inside
        // `source::MAX_SOURCE_BYTES`: 2 000 chained functions took **2.0 s** to
        // refuse and 10 000 took **359 s**. The watcher parses on every save and
        // the cook parses on every build, so that is an editor that stops
        // responding and a build that looks hung, from a 488 KiB script.
        //
        // Indices and a map make it linear (7 ms and 129 ms for the same two).
        // There is no wall-clock assertion here — this house does not make them
        // — and none is needed: the arm *completing* is the gate, because the
        // cost of the defect is measured in minutes.
        (
            "five thousand functions in one call chain",
            (0..5_000)
                .map(|i| {
                    let body = if i + 1 == 5_000 {
                        "    return 1.0\n".to_string()
                    } else {
                        format!("    return f{}()\n", i + 1)
                    };
                    format!("function f{i}() -> float\n{body}end\n")
                })
                .collect::<String>(),
        ),
        (
            "five thousand functions that call nobody",
            (0..5_000)
                .map(|i| format!("function f{i}() -> float\n    return 1.0\nend\n"))
                .collect::<String>(),
        ),
        ("nothing but dots", ".".repeat(10_000)),
        ("a file of end keywords", "end ".repeat(10_000)),
        ("an empty file", String::new()),
        ("one byte", "o".into()),
        (
            "a twenty-million-character line",
            format!(
                "on tick(dt)\n    debug.print(\"{}\")\nend\n",
                "x".repeat(20_000_000)
            ),
        ),
    ];
    for (what, src) in cases {
        match compile(&src, "act:hostile") {
            Ok(_) => {}
            Err(d) => {
                assert!(!d.is_empty(), "{what}: a refusal with no diagnostic");
                for diag in &d {
                    assert!(diag.span.line >= 1, "{what}: line 0 in `{diag}`");
                }
            }
        }
    }
}

/// The source a designer is halfway through typing, mutated one character at a
/// time — every position against a hostile alphabet, plus the deletion of every
/// position. **Nothing panics**, and every mutation that *parses* also emits and
/// re-parses to the identical IR, which is the totality claim in the only form
/// that can be tested: anything the parser accepts, the emitter can write, and
/// what it writes reads back.
#[test]
fn single_character_mutations_never_panic_and_always_round_trip() {
    // The **call form** is in the fixture, so every one of its characters is
    // mutated too: a name that stops matching a declaration, a `(` that closes
    // somewhere else, a `-` that turns `->` into a subtraction. Each is a source
    // a designer can be halfway through typing, and the sweep requires every one
    // that parses to emit and read back identically.
    const GOOD: &str = r#"actor "M"

var angle: float = 0.0

on tick(dt)
    local a = scaled(angle + dt * 2.0)
    if a > 1.0 then
        debug.print("x")
    end
    angle = a
end

function scaled(v: float) -> float
    return v * 0.5
end
"#;
    let chars: Vec<char> = GOOD.chars().collect();
    let alphabet: Vec<char> = "ab01 \n\t\"(),.:;=<>+-*/%~[]{}@\\".chars().collect();
    let mut fed = 0;
    let mut parsed = 0;
    let mut broken: Vec<String> = Vec::new();
    for i in 0..chars.len() {
        let mut variants: Vec<String> = alphabet
            .iter()
            .map(|c| {
                let mut m = chars.clone();
                m[i] = *c;
                m.into_iter().collect()
            })
            .collect();
        let mut deleted = chars.clone();
        deleted.remove(i);
        variants.push(deleted.into_iter().collect());
        for s in variants {
            fed += 1;
            let Ok((class, _)) = compile(&s, "act:m") else {
                continue;
            };
            parsed += 1;
            let text = match emit_class(&class) {
                Ok(t) => t,
                Err(e) => {
                    broken.push(format!("emit refused a source that parsed ({e}):\n{s}"));
                    continue;
                }
            };
            match compile(&text, "act:m") {
                Ok((again, _)) if again == class => {}
                Ok(_) => broken.push(format!("the IR moved:\n{s}\n=== reprint ===\n{text}")),
                Err(d) => broken.push(format!(
                    "the reprint does not parse ({}):\n{s}\n=== reprint ===\n{text}",
                    d.first().map(|x| x.to_string()).unwrap_or_default()
                )),
            }
        }
    }
    assert!(fed > 3_000, "only {fed} mutations were fed");
    assert!(
        parsed > 100,
        "only {parsed} of {fed} mutations parsed — the sweep has stopped \
         exercising the success path, so it proves nothing about the round trip"
    );
    assert!(broken.is_empty(), "{}", broken.join("\n---\n"));
}
