//! Spike B hand-edit corpus: real-world edits users make to generated
//! blueprint Rust, and what the lifter must do with each. Every case also
//! asserts convergence: regenerating, re-lifting, and regenerating again is
//! byte-stable (the watch-loop must never ping-pong).

use inf_blueprint::{BinOp, Binding, BlueprintFn, Expr, Lit, LocalId, Stmt, Ty};
use inf_transpile::{generate_file, lift_file, FileEntry};

const ATTR: &str = "#[infinity::blueprint(id = \"bp-1\")]";

/// Lift `src`, assert regen convergence, return (entries, first regen).
fn round(src: &str) -> (Vec<FileEntry>, String) {
    let lifted = lift_file(src).expect("corpus source parses");
    let regen1 = generate_file(&lifted.file).expect("regen 1");
    let lifted2 = lift_file(&regen1).expect("regen 1 parses");
    let regen2 = generate_file(&lifted2.file).expect("regen 2");
    assert_eq!(regen1, regen2, "regeneration did not converge for:\n{src}");
    (lifted.file.entries, regen1)
}

fn only_fn(entries: &[FileEntry]) -> &BlueprintFn {
    assert_eq!(entries.len(), 1, "expected exactly one entry");
    match &entries[0] {
        FileEntry::Blueprint(f) => f,
        FileEntry::Verbatim(v) => panic!("expected blueprint, got verbatim:\n{v}"),
    }
}

fn body_of(src_body: &str) -> String {
    format!("{ATTR}\npub fn tick(alpha: f64) -> f64 {{\n{src_body}\n}}\n")
}

fn snippet_count(stmts: &[Stmt]) -> usize {
    stmts
        .iter()
        .map(|s| match s {
            Stmt::Snippet(_) => 1,
            Stmt::If {
                then_body,
                else_body,
                ..
            } => snippet_count(then_body) + snippet_count(else_body),
            Stmt::While { body, .. } => snippet_count(body),
            _ => 0,
        })
        .sum()
}

// ── value edits ─────────────────────────────────────────────────────────────

#[test]
fn edited_literal_lifts() {
    let (e, _) = round(&body_of(
        "    let speed_n0: f64 = 2.5;\n    speed_n0 * alpha",
    ));
    let f = only_fn(&e);
    assert_eq!(
        f.body[0],
        Stmt::Let {
            id: LocalId(0),
            binding: Binding::Named("speed".into()),
            ty: Some(Ty::Float),
            mutable: false,
            value: Expr::Lit(Lit::Float(2.5)),
        }
    );
}

#[test]
fn renamed_local_keeps_node_id() {
    // User renamed `speed_n7` to `velocity_n7` — same node, new display name.
    let (e, _) = round(&body_of("    let velocity_n7 = 1.0;\n    velocity_n7"));
    let f = only_fn(&e);
    let Stmt::Let { id, binding, .. } = &f.body[0] else {
        panic!("expected let");
    };
    assert_eq!(*id, LocalId(7));
    assert_eq!(*binding, Binding::Named("velocity".into()));
}

#[test]
fn hand_written_let_gets_fresh_id_and_keeps_bare_name() {
    let (e, regen) = round(&body_of(
        "    let boost = 3.0;\n    let n2 = boost;\n    n2",
    ));
    let f = only_fn(&e);
    let Stmt::Let { id, binding, .. } = &f.body[0] else {
        panic!("expected let");
    };
    // Fresh id starts above the max explicit id (n2 → 3).
    assert_eq!(*id, LocalId(3));
    assert_eq!(*binding, Binding::Raw("boost".into()));
    // The bare name must survive regeneration (no forced rename).
    assert!(regen.contains("let boost = 3.0;"), "regen:\n{regen}");
}

#[test]
fn added_mut_and_assignment_lift() {
    let (e, _) = round(&body_of(
        "    let mut acc_n0 = 0.0;\n    acc_n0 = acc_n0 + alpha;\n    acc_n0",
    ));
    let f = only_fn(&e);
    assert!(matches!(f.body[0], Stmt::Let { mutable: true, .. }));
    assert_eq!(
        f.body[1],
        Stmt::Assign {
            target: LocalId(0),
            value: Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Local(LocalId(0))),
                Box::new(Expr::Param("alpha".into())),
            ),
        }
    );
}

#[test]
fn nested_arithmetic_lifts_as_tree() {
    let (e, _) = round(&body_of("    (alpha + 1.0) * (alpha - 2.0)"));
    let f = only_fn(&e);
    let Stmt::Return(Some(Expr::Binary(BinOp::Mul, l, r))) = &f.body[0] else {
        panic!("expected tail-expr multiply, got {:?}", f.body);
    };
    assert!(matches!(**l, Expr::Binary(BinOp::Add, _, _)));
    assert!(matches!(**r, Expr::Binary(BinOp::Sub, _, _)));
}

#[test]
fn redundant_parens_normalize_away() {
    let (_, regen) = round(&body_of("    ((alpha) + (1.0))"));
    assert!(regen.contains("alpha + 1.0"), "regen:\n{regen}");
}

#[test]
fn precedence_parens_are_preserved_in_regen() {
    let (_, regen) = round(&body_of("    (alpha + 1.0) * 2.0"));
    assert!(regen.contains("(alpha + 1.0) * 2.0"), "regen:\n{regen}");
}

#[test]
fn negative_literals_fold() {
    let (e, regen) = round(&body_of("    let x_n0 = -2.5;\n    x_n0"));
    let f = only_fn(&e);
    let Stmt::Let { value, .. } = &f.body[0] else {
        panic!("expected let");
    };
    assert_eq!(*value, Expr::Lit(Lit::Float(-2.5)));
    assert!(regen.contains("-2.5"), "regen:\n{regen}");
}

#[test]
fn int_suffix_normalizes() {
    let src = format!("{ATTR}\npub fn f() -> i64 {{\n    5i64\n}}\n");
    let (e, regen) = round(&src);
    let f = only_fn(&e);
    assert_eq!(f.body[0], Stmt::Return(Some(Expr::Lit(Lit::Int(5)))));
    assert!(regen.contains("    5\n"), "regen:\n{regen}");
}

#[test]
fn string_escapes_round_trip() {
    let src = format!("{ATTR}\npub fn f() -> String {{\n    api::name(\"a\\\"b\\n\\\\c\")\n}}\n");
    let (e, _) = round(&src);
    let f = only_fn(&e);
    let Stmt::Return(Some(Expr::Call { args, .. })) = &f.body[0] else {
        panic!("expected call");
    };
    assert_eq!(args[0], Expr::Lit(Lit::Str("a\"b\n\\c".into())));
}

// ── control flow ────────────────────────────────────────────────────────────

#[test]
fn if_else_lifts() {
    let (e, _) = round(&body_of(
        "    if alpha > 1.0 {\n        api::print_f64(alpha);\n    } else {\n        api::print_f64(0.0);\n    }\n    alpha",
    ));
    let f = only_fn(&e);
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = &f.body[0]
    else {
        panic!("expected if");
    };
    assert!(matches!(cond, Expr::Binary(BinOp::Gt, _, _)));
    assert_eq!(then_body.len(), 1);
    assert_eq!(else_body.len(), 1);
}

#[test]
fn else_if_chain_lifts_nested() {
    let (e, regen) = round(&body_of(
        "    if alpha > 2.0 {\n        api::a();\n    } else if alpha > 1.0 {\n        api::b();\n    } else {\n        api::c();\n    }\n    alpha",
    ));
    let f = only_fn(&e);
    let Stmt::If { else_body, .. } = &f.body[0] else {
        panic!("expected if");
    };
    assert!(matches!(else_body.as_slice(), [Stmt::If { .. }]));
    // Regen keeps the idiomatic `else if` spelling.
    assert!(regen.contains("} else if alpha > 1.0 {"), "regen:\n{regen}");
}

#[test]
fn while_loop_lifts() {
    let (e, _) = round(&body_of(
        "    let mut i_n0 = 0.0;\n    while i_n0 < alpha {\n        i_n0 = i_n0 + 1.0;\n    }\n    i_n0",
    ));
    let f = only_fn(&e);
    let Stmt::While { cond, body } = &f.body[1] else {
        panic!("expected while, got {:?}", f.body);
    };
    assert!(matches!(cond, Expr::Binary(BinOp::Lt, _, _)));
    assert!(matches!(body.as_slice(), [Stmt::Assign { .. }]));
}

#[test]
fn early_return_lifts() {
    let (e, _) = round(&body_of(
        "    if alpha < 0.0 {\n        return 0.0;\n    }\n    alpha",
    ));
    let f = only_fn(&e);
    let Stmt::If { then_body, .. } = &f.body[0] else {
        panic!("expected if");
    };
    assert_eq!(then_body[0], Stmt::Return(Some(Expr::Lit(Lit::Float(0.0)))));
}

#[test]
fn tail_expr_becomes_return_node() {
    let (e, _) = round(&body_of("    alpha * 2.0"));
    let f = only_fn(&e);
    assert!(matches!(f.body[0], Stmt::Return(Some(_))));
}

#[test]
fn explicit_tail_return_normalizes_to_tail_expr() {
    let (_, regen) = round(&body_of("    return alpha * 2.0;"));
    assert!(!regen.contains("return"), "regen:\n{regen}");
    assert!(regen.contains("    alpha * 2.0\n"), "regen:\n{regen}");
}

#[test]
fn bool_logic_lifts() {
    let src = format!("{ATTR}\npub fn f(a: bool, b: bool) -> bool {{\n    !a && (b || true)\n}}\n");
    let (e, _) = round(&src);
    let f = only_fn(&e);
    assert!(matches!(
        f.body[0],
        Stmt::Return(Some(Expr::Binary(BinOp::And, _, _)))
    ));
}

// ── unliftable shapes become snippets, never data loss ──────────────────────

#[test]
fn method_call_becomes_snippet() {
    let (e, regen) = round(&body_of("    let r = alpha.sqrt();\n    alpha"));
    let f = only_fn(&e);
    assert_eq!(snippet_count(&f.body), 1);
    assert!(regen.contains("let r = alpha.sqrt();"), "regen:\n{regen}");
}

#[test]
fn macro_becomes_snippet() {
    let (e, regen) = round(&body_of("    println!(\"alpha = {alpha}\");\n    alpha"));
    let f = only_fn(&e);
    assert_eq!(snippet_count(&f.body), 1);
    assert!(
        regen.contains("println!(\"alpha = {alpha}\");"),
        "regen:\n{regen}"
    );
}

#[test]
fn for_loop_becomes_snippet() {
    let (e, regen) = round(&body_of(
        "    for i in 0..10 {\n        api::print_f64(alpha);\n    }\n    alpha",
    ));
    let f = only_fn(&e);
    assert_eq!(snippet_count(&f.body), 1);
    assert!(regen.contains("for i in 0..10 {"), "regen:\n{regen}");
}

#[test]
fn match_becomes_snippet() {
    let (e, _) = round(&body_of(
        "    match alpha as i64 {\n        0 => api::a(),\n        _ => api::b(),\n    }\n    alpha",
    ));
    let f = only_fn(&e);
    assert_eq!(snippet_count(&f.body), 1);
}

#[test]
fn closure_becomes_snippet() {
    let (e, regen) = round(&body_of("    let f = |x: f64| x * 2.0;\n    alpha"));
    let f = only_fn(&e);
    assert_eq!(snippet_count(&f.body), 1);
    assert!(regen.contains("|x: f64| x * 2.0"), "regen:\n{regen}");
}

#[test]
fn unknown_ident_reference_becomes_snippet() {
    let (e, _) = round(&body_of("    let x_n0 = mystery + 1.0;\n    alpha"));
    let f = only_fn(&e);
    assert_eq!(snippet_count(&f.body), 1);
}

#[test]
fn snippet_bound_name_poisons_later_uses() {
    // `q` is bound by an unliftable statement; the statement that uses `q`
    // must ALSO stay a snippet, or regen would wire it to the wrong node.
    let (e, regen) = round(&body_of(
        "    let q = alpha.sqrt();\n    let y_n0 = q + 1.0;\n    y_n0",
    ));
    let f = only_fn(&e);
    // All three stay textual: `q` has no graph node, so `y_n0 = q + 1.0` can't
    // be an edge — and then neither can the tail reference to `y_n0`.
    assert_eq!(snippet_count(&f.body), 3);
    assert!(regen.contains("let q = alpha.sqrt();"), "regen:\n{regen}");
    assert!(regen.contains("let y_n0 = q + 1.0;"), "regen:\n{regen}");
}

#[test]
fn if_let_becomes_snippet() {
    let (e, _) = round(&body_of(
        "    if let Some(v) = api::probe() {\n        api::print_f64(v);\n    }\n    alpha",
    ));
    let f = only_fn(&e);
    assert_eq!(snippet_count(&f.body), 1);
}

#[test]
fn nested_block_tail_expr_becomes_snippet() {
    // A value-carrying tail expr in a nested block has semantics our exec
    // model can't represent — conservative snippet of the whole `if`.
    let (e, _) = round(&body_of(
        "    let x_n0 = 1.0;\n    if alpha > 0.0 {\n        x_n0\n    } else {\n        alpha\n    };\n    alpha",
    ));
    let f = only_fn(&e);
    assert!(snippet_count(&f.body) >= 1);
}

#[test]
fn duplicate_node_id_binder_stays_snippet() {
    let (e, _) = round(&body_of(
        "    let x_n0 = 1.0;\n    let y_n0 = 2.0;\n    alpha",
    ));
    let f = only_fn(&e);
    // Second binder reuses id 0 — conservatively snippeted.
    assert_eq!(snippet_count(&f.body), 1);
}

// ── item-level edits ────────────────────────────────────────────────────────

#[test]
fn helper_fn_is_preserved_verbatim() {
    let src = format!(
        "fn helper(x: f64) -> f64 {{\n    x * x\n}}\n{}\npub fn tick(alpha: f64) -> f64 {{\n    helper(alpha)\n}}\n",
        ATTR
    );
    let (e, regen) = round(&src);
    assert_eq!(e.len(), 2);
    assert!(matches!(&e[0], FileEntry::Verbatim(v) if v.contains("fn helper")));
    assert!(matches!(&e[1], FileEntry::Blueprint(_)));
    // Order preserved on regen.
    let helper_pos = regen.find("fn helper").unwrap();
    let bp_pos = regen.find("pub fn tick").unwrap();
    assert!(helper_pos < bp_pos);
}

#[test]
fn use_statement_is_preserved_verbatim() {
    let src = format!(
        "use std::f64::consts::PI;\n{}\npub fn tick(alpha: f64) -> f64 {{\n    alpha\n}}\n",
        ATTR
    );
    let (e, regen) = round(&src);
    assert!(matches!(&e[0], FileEntry::Verbatim(v) if v.contains("use std::f64::consts::PI;")));
    assert!(regen.contains("use std::f64::consts::PI;"));
}

#[test]
fn struct_item_is_preserved_verbatim() {
    let src = format!(
        "struct Config {{\n    max: f64,\n}}\n{}\npub fn tick(alpha: f64) -> f64 {{\n    alpha\n}}\n",
        ATTR
    );
    let (e, _) = round(&src);
    assert!(matches!(&e[0], FileEntry::Verbatim(v) if v.contains("struct Config")));
}

#[test]
fn doc_comment_on_blueprint_fn_keeps_it_verbatim() {
    // Doc comments are attributes; we refuse to lift rather than drop them.
    let src = format!(
        "/// Hand-written docs.\n{}\npub fn tick(alpha: f64) -> f64 {{\n    alpha\n}}\n",
        ATTR
    );
    let lifted = lift_file(&src).expect("parses");
    assert_eq!(lifted.warnings.len(), 1);
    assert!(
        matches!(&lifted.file.entries[0], FileEntry::Verbatim(v) if v.contains("Hand-written docs"))
    );
}

#[test]
fn unsupported_param_type_keeps_fn_verbatim_with_warning() {
    let src = format!(
        "{}\npub fn tick(v: Vec<f64>) -> f64 {{\n    0.0\n}}\n",
        ATTR
    );
    let lifted = lift_file(&src).expect("parses");
    assert_eq!(lifted.warnings.len(), 1);
    assert!(matches!(&lifted.file.entries[0], FileEntry::Verbatim(_)));
}

#[test]
fn two_blueprint_fns_in_one_file() {
    let src = "#[infinity::blueprint(id = \"bp-a\")]\npub fn a() {\n    api::a();\n}\n#[infinity::blueprint(id = \"bp-b\")]\npub fn b() {\n    api::b();\n}\n";
    let (e, _) = round(src);
    assert_eq!(e.len(), 2);
    let ids: Vec<_> = e
        .iter()
        .map(|entry| match entry {
            FileEntry::Blueprint(f) => f.id.clone(),
            _ => panic!("expected blueprint"),
        })
        .collect();
    assert_eq!(ids, vec!["bp-a".to_owned(), "bp-b".to_owned()]);
}

#[test]
fn empty_body_fn() {
    let src = format!("{ATTR}\npub fn noop() {{}}\n");
    let (e, _) = round(&src);
    assert!(only_fn(&e).body.is_empty());
}

#[test]
fn fn_without_attr_is_not_lifted() {
    let src = "pub fn plain(x: f64) -> f64 {\n    x\n}\n";
    let (e, _) = round(src);
    assert!(matches!(&e[0], FileEntry::Verbatim(_)));
}

#[test]
fn shadowing_let_lifts_both_nodes() {
    let (e, _) = round(&body_of("    let x = 1.0;\n    let x = 2.0;\n    x"));
    let f = only_fn(&e);
    // Two raw binders, distinct ids; tail resolves to the second.
    let Stmt::Let { id: id1, .. } = &f.body[0] else {
        panic!()
    };
    let Stmt::Let { id: id2, .. } = &f.body[1] else {
        panic!()
    };
    assert_ne!(id1, id2);
    assert_eq!(f.body[2], Stmt::Return(Some(Expr::Local(*id2))));
}

#[test]
fn call_statement_lifts() {
    let src = format!("{ATTR}\npub fn f(alpha: f64) {{\n    api::print_f64(alpha * 2.0);\n}}\n");
    let (e, _) = round(&src);
    let f = only_fn(&e);
    assert!(matches!(f.body[0], Stmt::ExprStmt(Expr::Call { .. })));
}

#[test]
fn multi_segment_call_paths_are_opaque_and_liftable() {
    let (e, _) = round(&body_of("    game::math::clamp01(alpha)"));
    let f = only_fn(&e);
    let Stmt::Return(Some(Expr::Call { path, .. })) = &f.body[0] else {
        panic!("expected call");
    };
    assert_eq!(path, &["game", "math", "clamp01"]);
}
