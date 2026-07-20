//! Spike B property suite: for every well-formed graph,
//!   lift(generate(g)) == g          (isomorphism)
//!   generate(lift(generate(g)))     is byte-identical (idempotence)
//!
//! Strategy: proptest generates an abstract "recipe" (structurally random,
//! reference-free), which a deterministic resolver turns into a well-formed
//! `BlueprintFn` — references only to visible, unambiguous locals/params,
//! unique ids, canonical literals. This keeps shrinking useful while
//! guaranteeing the invariants the editor enforces at authoring time.

use inf_blueprint::{BinOp, Binding, BlueprintFn, Expr, Lit, LocalId, Param, Stmt, Ty, UnOp};
use inf_transpile::{fingerprint_fn, generate_fn, lift_file, FileEntry};
use proptest::prelude::*;

// ── recipes ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum RLit {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
}

#[derive(Debug, Clone)]
enum RExpr {
    Lit(RLit),
    Param(u8),
    Local(u8),
    Neg(Box<RExpr>),
    Not(Box<RExpr>),
    Binary(u8, Box<RExpr>, Box<RExpr>),
    Call(u8, Vec<RExpr>),
}

#[derive(Debug, Clone)]
enum RStmt {
    Let {
        style: u8,
        annotated: bool,
        mutable: bool,
        value: RExpr,
    },
    Assign {
        target: u8,
        value: RExpr,
    },
    If {
        cond: RExpr,
        then_body: Vec<RStmt>,
        else_body: Vec<RStmt>,
    },
    While {
        cond: RExpr,
        body: Vec<RStmt>,
    },
    Return(Option<RExpr>),
    CallStmt(u8, Vec<RExpr>),
    Snippet(u8),
}

fn lit_strategy() -> impl Strategy<Value = RLit> {
    prop_oneof![
        // Finite floats only; i64::MIN excluded — both documented model limits.
        proptest::num::f64::NORMAL.prop_map(RLit::Float),
        any::<i64>()
            .prop_filter("no i64::MIN", |v| *v != i64::MIN)
            .prop_map(RLit::Int),
        any::<bool>().prop_map(RLit::Bool),
        "[a-z ]{0,12}".prop_map(RLit::Str),
    ]
}

fn expr_strategy() -> impl Strategy<Value = RExpr> {
    let leaf = prop_oneof![
        lit_strategy().prop_map(RExpr::Lit),
        any::<u8>().prop_map(RExpr::Param),
        any::<u8>().prop_map(RExpr::Local),
    ];
    leaf.prop_recursive(3, 24, 3, |inner| {
        prop_oneof![
            inner.clone().prop_map(|e| RExpr::Neg(Box::new(e))),
            inner.clone().prop_map(|e| RExpr::Not(Box::new(e))),
            (any::<u8>(), inner.clone(), inner.clone()).prop_map(|(op, l, r)| RExpr::Binary(
                op,
                Box::new(l),
                Box::new(r)
            )),
            (any::<u8>(), proptest::collection::vec(inner, 0..3))
                .prop_map(|(p, args)| RExpr::Call(p, args)),
        ]
    })
}

fn stmt_strategy() -> impl Strategy<Value = RStmt> {
    let flat = prop_oneof![
        (any::<u8>(), any::<bool>(), any::<bool>(), expr_strategy()).prop_map(
            |(style, annotated, mutable, value)| RStmt::Let {
                style,
                annotated,
                mutable,
                value,
            }
        ),
        (any::<u8>(), expr_strategy()).prop_map(|(target, value)| RStmt::Assign { target, value }),
        proptest::option::of(expr_strategy()).prop_map(RStmt::Return),
        (
            any::<u8>(),
            proptest::collection::vec(expr_strategy(), 0..3)
        )
            .prop_map(|(p, args)| RStmt::CallStmt(p, args)),
        any::<u8>().prop_map(RStmt::Snippet),
    ];
    flat.prop_recursive(2, 16, 4, |inner| {
        prop_oneof![
            (
                expr_strategy(),
                proptest::collection::vec(inner.clone(), 0..4),
                proptest::collection::vec(inner.clone(), 0..4)
            )
                .prop_map(|(cond, then_body, else_body)| RStmt::If {
                    cond,
                    then_body,
                    else_body,
                }),
            (expr_strategy(), proptest::collection::vec(inner, 0..4))
                .prop_map(|(cond, body)| RStmt::While { cond, body }),
        ]
    })
}

fn fn_strategy() -> impl Strategy<Value = (Vec<(u8, u8)>, u8, Vec<RStmt>)> {
    (
        proptest::collection::vec((any::<u8>(), any::<u8>()), 0..4),
        any::<u8>(),
        proptest::collection::vec(stmt_strategy(), 0..8),
    )
}

// ── resolver: recipe → well-formed BlueprintFn ──────────────────────────────

const PARAM_POOL: &[&str] = &["alpha", "beta", "gamma", "delta"];
const LOCAL_NAME_POOL: &[&str] = &["speed", "accel", "mass", "dist", "score", "flag"];
const CALL_POOL: &[&[&str]] = &[
    &["api", "print_f64"],
    &["helper"],
    &["game", "math", "clamp01"],
    &["engine", "spawn_marker"],
];
// Snippets must be UNLIFTABLE (method calls, macros, loops — so they stay
// snippets) and reference no liftable locals; the resolver normalizes them
// through `normalize_snippet` so string equality holds after a round trip.
const SNIPPET_POOL: &[&str] = &[
    "let ext = timer.elapsed();",
    "println!(\"tick\");",
    "for _ in 0..3 {}",
    "let side = area.sqrt();",
];

/// Raw binders don't encode their id in the identifier; the lifter re-derives
/// them as `max_explicit_id + k` in encounter order. The resolver mints
/// temporary ids at this base and remaps afterwards to match.
const RAW_TEMP_BASE: u32 = 1_000_000;

fn ty_of(sel: u8) -> Ty {
    [Ty::Float, Ty::Int, Ty::Bool, Ty::Str, Ty::Unit][sel as usize % 5]
}

fn bin_op_of(sel: u8) -> BinOp {
    [
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
    ][sel as usize % 13]
}

struct Resolver {
    next_id: u32,
    next_raw_name: u32,
    params: Vec<String>,
}

#[derive(Clone, Default)]
struct Visible {
    /// (id) of locals referencable here; raw-named ones excluded (their
    /// rendered ident can be shadowed, making graph refs ambiguous).
    ids: Vec<LocalId>,
    mutable_ids: Vec<LocalId>,
}

impl Resolver {
    fn lit(&self, r: &RLit) -> Lit {
        match r {
            RLit::Float(v) => Lit::Float(*v),
            RLit::Int(v) => Lit::Int(*v),
            RLit::Bool(v) => Lit::Bool(*v),
            RLit::Str(s) => Lit::Str(s.clone()),
        }
    }

    fn expr(&self, r: &RExpr, vis: &Visible) -> Expr {
        match r {
            RExpr::Lit(l) => Expr::Lit(self.lit(l)),
            RExpr::Param(sel) => {
                if self.params.is_empty() {
                    Expr::Lit(Lit::Int(*sel as i64))
                } else {
                    Expr::Param(self.params[*sel as usize % self.params.len()].clone())
                }
            }
            RExpr::Local(sel) => {
                if vis.ids.is_empty() {
                    Expr::Lit(Lit::Float(*sel as f64))
                } else {
                    Expr::Local(vis.ids[*sel as usize % vis.ids.len()])
                }
            }
            RExpr::Neg(inner) => {
                let e = self.expr(inner, vis);
                match e {
                    // Canonical model: negation folds into numeric literals.
                    Expr::Lit(Lit::Int(v)) => Expr::Lit(Lit::Int(v.checked_neg().unwrap_or(0))),
                    Expr::Lit(Lit::Float(v)) => Expr::Lit(Lit::Float(-v)),
                    other => Expr::Unary(UnOp::Neg, Box::new(other)),
                }
            }
            RExpr::Not(inner) => Expr::Unary(UnOp::Not, Box::new(self.expr(inner, vis))),
            RExpr::Binary(op, l, r) => Expr::Binary(
                bin_op_of(*op),
                Box::new(self.expr(l, vis)),
                Box::new(self.expr(r, vis)),
            ),
            RExpr::Call(p, args) => {
                let path = CALL_POOL[*p as usize % CALL_POOL.len()];
                Expr::Call {
                    path: path.iter().map(|s| s.to_string()).collect(),
                    args: args.iter().map(|a| self.expr(a, vis)).collect(),
                }
            }
        }
    }

    fn stmts(&mut self, recipes: &[RStmt], vis: &mut Visible) -> Vec<Stmt> {
        let mut out = Vec::new();
        for r in recipes {
            match r {
                RStmt::Let {
                    style,
                    annotated,
                    mutable,
                    value,
                } => {
                    let value = self.expr(value, vis);
                    // style: 0 = Anon, 1 = Named, 2 = Raw (unique bare name).
                    let (binding, id) = match style % 3 {
                        0 => {
                            let id = LocalId(self.next_id);
                            self.next_id += 1;
                            (Binding::Anon, id)
                        }
                        1 => {
                            let id = LocalId(self.next_id);
                            self.next_id += 1;
                            (
                                Binding::Named(
                                    LOCAL_NAME_POOL[*style as usize % LOCAL_NAME_POOL.len()]
                                        .to_owned(),
                                ),
                                id,
                            )
                        }
                        _ => {
                            let n = self.next_raw_name;
                            self.next_raw_name += 1;
                            (Binding::Raw(format!("raw{n}")), LocalId(RAW_TEMP_BASE + n))
                        }
                    };
                    // Raw binders are excluded from graph references (see
                    // Visible docs); id-carrying ones are fair game.
                    if !matches!(binding, Binding::Raw(_)) {
                        vis.ids.push(id);
                        if *mutable {
                            vis.mutable_ids.push(id);
                        }
                    }
                    out.push(Stmt::Let {
                        id,
                        binding,
                        ty: annotated.then(|| ty_of(*style)),
                        mutable: *mutable,
                        value,
                    });
                }
                RStmt::Assign { target, value } => {
                    let value = self.expr(value, vis);
                    match vis.mutable_ids.is_empty() {
                        true => {
                            // No assignable local in scope — degrade to a let.
                            let id = LocalId(self.next_id);
                            self.next_id += 1;
                            vis.ids.push(id);
                            out.push(Stmt::Let {
                                id,
                                binding: Binding::Anon,
                                ty: None,
                                mutable: false,
                                value,
                            });
                        }
                        false => out.push(Stmt::Assign {
                            target: vis.mutable_ids[*target as usize % vis.mutable_ids.len()],
                            value,
                        }),
                    }
                }
                RStmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    let cond = self.expr(cond, vis);
                    let mut then_vis = vis.clone();
                    let then_body = self.stmts(then_body, &mut then_vis);
                    let mut else_vis = vis.clone();
                    let else_body = self.stmts(else_body, &mut else_vis);
                    out.push(Stmt::If {
                        cond,
                        then_body,
                        else_body,
                    });
                }
                RStmt::While { cond, body } => {
                    let cond = self.expr(cond, vis);
                    let mut body_vis = vis.clone();
                    let body = self.stmts(body, &mut body_vis);
                    out.push(Stmt::While { cond, body });
                }
                RStmt::Return(value) => {
                    out.push(Stmt::Return(value.as_ref().map(|v| self.expr(v, vis))));
                }
                RStmt::CallStmt(p, args) => {
                    let path = CALL_POOL[*p as usize % CALL_POOL.len()];
                    out.push(Stmt::ExprStmt(Expr::Call {
                        path: path.iter().map(|s| s.to_string()).collect(),
                        args: args.iter().map(|a| self.expr(a, vis)).collect(),
                    }));
                }
                RStmt::Snippet(sel) => {
                    let raw = SNIPPET_POOL[*sel as usize % SNIPPET_POOL.len()];
                    let normalized = inf_transpile::normalize_snippet(raw)
                        .expect("pool snippet parses")
                        .remove(0);
                    out.push(Stmt::Snippet(normalized));
                }
            }
        }
        out
    }
}

fn resolve(recipe: &(Vec<(u8, u8)>, u8, Vec<RStmt>)) -> BlueprintFn {
    let (params, ret, body) = recipe;
    let params: Vec<Param> = params
        .iter()
        .enumerate()
        .take(PARAM_POOL.len())
        .map(|(i, (_, ty))| Param {
            name: PARAM_POOL[i].to_owned(),
            ty: ty_of(*ty),
        })
        .collect();
    let mut resolver = Resolver {
        next_id: 0,
        next_raw_name: 0,
        params: params.iter().map(|p| p.name.clone()).collect(),
    };
    let mut vis = Visible::default();
    let mut body = resolver.stmts(body, &mut vis);
    // Canonical raw-binder ids: max explicit id + k in creation order (which
    // equals lexical encounter order — matching what the lifter re-derives).
    remap_raw_ids(&mut body, resolver.next_id);
    BlueprintFn {
        id: "01890a5d-ac96-774b-bcce-b302099a8057".to_owned(),
        name: "update_actor".to_owned(),
        params,
        ret: ty_of(*ret),
        body,
    }
}

fn remap_raw_ids(stmts: &mut [Stmt], explicit_next: u32) {
    for s in stmts {
        match s {
            Stmt::Let { id, .. } if id.0 >= RAW_TEMP_BASE => {
                *id = LocalId(explicit_next + (id.0 - RAW_TEMP_BASE));
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                remap_raw_ids(then_body, explicit_next);
                remap_raw_ids(else_body, explicit_next);
            }
            Stmt::While { body, .. } => remap_raw_ids(body, explicit_next),
            _ => {}
        }
    }
}

// ── properties ──────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn generate_then_lift_is_identity(recipe in fn_strategy()) {
        let f = resolve(&recipe);
        let code = generate_fn(&f).expect("well-formed graph generates");
        let lifted = lift_file(&code).expect("generated code parses");
        prop_assert!(lifted.warnings.is_empty(), "warnings: {:?}\ncode:\n{code}", lifted.warnings);
        prop_assert_eq!(
            lifted.file.entries.len(), 1,
            "expected a single entry\ncode:\n{}", code
        );
        match &lifted.file.entries[0] {
            FileEntry::Blueprint(back) => prop_assert_eq!(
                back, &f,
                "graph not preserved\ncode:\n{}", code
            ),
            FileEntry::Verbatim(v) => prop_assert!(false, "fell back to verbatim:\n{}", v),
        }
    }

    #[test]
    fn regeneration_is_idempotent(recipe in fn_strategy()) {
        let f = resolve(&recipe);
        let gen1 = generate_fn(&f).expect("generate");
        let lifted = lift_file(&gen1).expect("parse");
        let FileEntry::Blueprint(back) = &lifted.file.entries[0] else {
            panic!("verbatim fallback");
        };
        let gen2 = generate_fn(back).expect("regenerate");
        prop_assert_eq!(&gen1, &gen2, "regeneration drifted");
    }

    #[test]
    fn fingerprint_is_stable_across_round_trip(recipe in fn_strategy()) {
        let f = resolve(&recipe);
        let code = generate_fn(&f).expect("generate");
        let lifted = lift_file(&code).expect("parse");
        let FileEntry::Blueprint(back) = &lifted.file.entries[0] else {
            panic!("verbatim fallback");
        };
        prop_assert_eq!(fingerprint_fn(&f).unwrap(), fingerprint_fn(back).unwrap());
    }
}
