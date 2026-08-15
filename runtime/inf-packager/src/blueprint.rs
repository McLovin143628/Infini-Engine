//! Blueprint validation at cook time (P9.2, deliverable 2).
//!
//! A `.inf_act` stores a [`BlueprintClass`] (JSON) whose event/function bodies
//! are already the lowered [`BlueprintFn`] IR — the *same* IR the P8.4
//! `SimSession` interprets and `inf-transpile` renders to Rust. "Compile
//! blueprints" at cook therefore means: **decode, migrate, and statically
//! validate that IR**, failing the cook with a clear, handler-anchored message so
//! a broken blueprint never ships.
//!
//! The static checks are the ones that make an IR body ill-formed regardless of
//! host:
//!
//! * every `Local`/`Assign` reference resolves to a `Let` in scope (block-scoped:
//!   a child block sees its parents' locals, siblings don't leak);
//! * every float literal is finite (the IR contract — NaN/inf have no Rust
//!   literal spelling, so shipping one would fail the compiled path).

use std::collections::HashSet;

use inf_blueprint::{BlueprintClass, BlueprintFn, BlueprintLibrary, Expr, Lit, LocalId, Stmt};

/// A validation failure, anchored to the handler/function it was found in.
pub struct BlueprintIssue {
    /// The handler key (e.g. `tick`) or function name the problem is in.
    pub handler: String,
    /// A human-readable description.
    pub message: String,
}

/// Validate an actor class: migrate it, then check every event handler + member
/// function body. Returns the first issue found, if any.
pub fn validate_class(class: &mut BlueprintClass) -> Option<BlueprintIssue> {
    class.migrate();
    // C4-44: this module's doc says "every float literal is finite", and
    // `validate_expr` enforces it inside bodies — but the walk skipped
    // `class.variables`, whose `default: Lit::Float` is (per
    // `inf_blueprint::semantics`) "the other place a float persists". A
    // non-finite default escaped the cook and reached `inf-transpile`, which has
    // no Rust spelling for it: a **ship-time compile failure** instead of a cook
    // error. Checked first, because a variable is read before any handler runs.
    for v in &class.variables {
        if let Lit::Float(x) = v.default {
            if !x.is_finite() {
                return Some(BlueprintIssue {
                    handler: format!("variable {}", v.name),
                    message: format!(
                        "default is {x}, which is not finite; the IR contract is that every \
                         float literal has a Rust spelling, and this one does not"
                    ),
                });
            }
        }
    }
    for binding in &class.events {
        if let Some(msg) = validate_fn(&binding.body) {
            return Some(BlueprintIssue {
                handler: binding.event.key(),
                message: msg,
            });
        }
    }
    for f in &class.functions {
        if let Some(msg) = validate_fn(f) {
            return Some(BlueprintIssue {
                handler: format!("fn {}", f.name),
                message: msg,
            });
        }
    }
    None
}

/// Validate a function library: migrate + check every function body.
pub fn validate_library(lib: &mut BlueprintLibrary) -> Option<BlueprintIssue> {
    lib.migrate();
    for f in &lib.functions {
        if let Some(msg) = validate_fn(f) {
            return Some(BlueprintIssue {
                handler: format!("fn {}", f.name),
                message: msg,
            });
        }
    }
    None
}

/// Statically validate one [`BlueprintFn`] body. Returns a message on the first
/// problem, else `None`.
fn validate_fn(f: &BlueprintFn) -> Option<String> {
    let mut scope: HashSet<LocalId> = HashSet::new();
    validate_block(&f.body, &mut scope)
}

/// Walk a statement block with a mutable in-scope local set (block-scoped: a
/// clone is handed to child blocks so their locals don't leak to siblings).
fn validate_block(body: &[Stmt], scope: &mut HashSet<LocalId>) -> Option<String> {
    for stmt in body {
        match stmt {
            Stmt::Let { id, value, .. } => {
                if let Some(m) = validate_expr(value, scope) {
                    return Some(m);
                }
                scope.insert(*id);
            }
            Stmt::Assign { target, value } => {
                if !scope.contains(target) {
                    return Some(format!("assignment targets undefined local n{}", target.0));
                }
                if let Some(m) = validate_expr(value, scope) {
                    return Some(m);
                }
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                if let Some(m) = validate_expr(cond, scope) {
                    return Some(m);
                }
                if let Some(m) = validate_block(then_body, &mut scope.clone()) {
                    return Some(m);
                }
                if let Some(m) = validate_block(else_body, &mut scope.clone()) {
                    return Some(m);
                }
            }
            Stmt::While { cond, body } => {
                if let Some(m) = validate_expr(cond, scope) {
                    return Some(m);
                }
                if let Some(m) = validate_block(body, &mut scope.clone()) {
                    return Some(m);
                }
            }
            Stmt::Return(Some(e)) | Stmt::ExprStmt(e) => {
                if let Some(m) = validate_expr(e, scope) {
                    return Some(m);
                }
            }
            Stmt::Return(None) | Stmt::Snippet(_) => {}
        }
    }
    None
}

/// Recursively validate an expression tree against the in-scope locals.
fn validate_expr(expr: &Expr, scope: &HashSet<LocalId>) -> Option<String> {
    match expr {
        Expr::Lit(Lit::Float(v)) if !v.is_finite() => {
            Some(format!("non-finite float literal ({v})"))
        }
        Expr::Lit(_) | Expr::Param(_) => None,
        Expr::Local(id) => {
            if scope.contains(id) {
                None
            } else {
                Some(format!("references undefined local n{}", id.0))
            }
        }
        Expr::Unary(_, e) => validate_expr(e, scope),
        Expr::Binary(_, a, b) => validate_expr(a, scope).or_else(|| validate_expr(b, scope)),
        Expr::Call { args, .. } => args.iter().find_map(|a| validate_expr(a, scope)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_blueprint::{BinOp, Binding, EventBinding, EventKind, Param, Ty};

    fn fn_with_body(body: Vec<Stmt>) -> BlueprintFn {
        BlueprintFn {
            id: "f".into(),
            name: "f".into(),
            params: vec![Param {
                name: "dt".into(),
                ty: Ty::Float,
            }],
            ret: Ty::Unit,
            body,
        }
    }

    #[test]
    fn accepts_well_formed_ir() {
        let f = fn_with_body(vec![
            Stmt::Let {
                id: LocalId(1),
                binding: Binding::Named("a".into()),
                ty: None,
                mutable: false,
                value: Expr::Binary(
                    BinOp::Mul,
                    Box::new(Expr::Param("dt".into())),
                    Box::new(Expr::Lit(Lit::Float(2.0))),
                ),
            },
            Stmt::ExprStmt(Expr::Call {
                path: vec!["engine".into(), "log".into()],
                args: vec![Expr::Local(LocalId(1))],
            }),
        ]);
        assert!(validate_fn(&f).is_none());
    }

    #[test]
    fn rejects_undefined_local() {
        let f = fn_with_body(vec![Stmt::ExprStmt(Expr::Local(LocalId(99)))]);
        let msg = validate_fn(&f).unwrap();
        assert!(msg.contains("undefined local n99"), "{msg}");
    }

    #[test]
    fn rejects_non_finite_float() {
        let f = fn_with_body(vec![Stmt::ExprStmt(Expr::Lit(Lit::Float(f64::NAN)))]);
        assert!(validate_fn(&f).unwrap().contains("non-finite"));
    }

    #[test]
    fn sibling_scopes_do_not_leak() {
        // A local declared in the `then` branch is not visible in `else`.
        let f = fn_with_body(vec![Stmt::If {
            cond: Expr::Lit(Lit::Bool(true)),
            then_body: vec![Stmt::Let {
                id: LocalId(5),
                binding: Binding::Anon,
                ty: None,
                mutable: false,
                value: Expr::Lit(Lit::Float(1.0)),
            }],
            else_body: vec![Stmt::ExprStmt(Expr::Local(LocalId(5)))],
        }]);
        assert!(validate_fn(&f).unwrap().contains("undefined local n5"));
    }

    #[test]
    fn validate_class_anchors_to_handler() {
        let mut class = BlueprintClass::new("act:x", "X");
        class.events = vec![EventBinding {
            event: EventKind::Tick,
            body: fn_with_body(vec![Stmt::ExprStmt(Expr::Local(LocalId(7)))]),
        }];
        let issue = validate_class(&mut class).unwrap();
        assert_eq!(issue.handler, "tick");
        assert!(issue.message.contains("n7"));
    }

    /// C4-44: this module's docs say "every float literal is finite", and the
    /// walk covered `events` and `functions` and **not** `class.variables` —
    /// whose `default: Lit::Float` is the other place a float persists. A
    /// non-finite default escaped the cook and reached `inf-transpile`, which
    /// has no Rust spelling for it: a ship-time compile failure instead of a
    /// cook error, which is the whole point of validating at cook.
    #[test]
    fn a_variable_default_that_is_not_finite_fails_the_cook() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut class = BlueprintClass::new("act:x", "X");
            class.variables = vec![inf_blueprint::Variable {
                name: "speed".into(),
                ty: inf_blueprint::Ty::Float,
                default: Lit::Float(bad),
                exposed: true,
            }];
            let issue = validate_class(&mut class)
                .unwrap_or_else(|| panic!("a {bad} default must not ship"));
            assert_eq!(issue.handler, "variable speed", "anchored to the variable");
            assert!(issue.message.contains("not finite"), "{}", issue.message);
        }

        // The control: a finite default passes, so the check is not "refuse".
        let mut ok = BlueprintClass::new("act:x", "X");
        ok.variables = vec![inf_blueprint::Variable {
            name: "speed".into(),
            ty: inf_blueprint::Ty::Float,
            default: Lit::Float(3.5),
            exposed: true,
        }];
        assert!(validate_class(&mut ok).is_none());
    }
}
