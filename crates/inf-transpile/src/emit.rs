//! Graph → Rust. Builds `syn` ASTs node-by-node (never token streams parsed
//! back, which could re-associate expressions by precedence) and prints with
//! `prettyplease`, which inserts whatever parentheses the concrete syntax
//! needs.

use std::collections::BTreeMap;

use inf_blueprint::{BinOp, BlueprintFn, Expr, Lit, LocalId, Stmt, Ty, UnOp};
use proc_macro2::Span;

use crate::{BlueprintFile, FileEntry};

#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    #[error("invalid identifier `{0}`")]
    BadIdent(String),
    #[error("duplicate local node id n{0}")]
    DuplicateLocal(u32),
    #[error("reference to undeclared local node id n{0}")]
    UnknownLocal(u32),
    #[error("non-finite float literal {0} has no source spelling")]
    NonFiniteFloat(f64),
    #[error("i64::MIN literal is not liftable; store it as a snippet")]
    IntMin,
    #[error("negation of a literal is non-canonical; store a negative literal instead")]
    NegatedLiteral,
    #[error("snippet does not parse as Rust statements: {0}")]
    SnippetParse(String),
    #[error("verbatim item does not parse: {0}")]
    VerbatimParse(String),
}

/// Render a whole blueprint file (blueprint fns + verbatim items, in order).
pub fn generate_file(file: &BlueprintFile) -> Result<String, EmitError> {
    let mut items = Vec::new();
    for entry in &file.entries {
        match entry {
            FileEntry::Blueprint(f) => items.push(syn::Item::Fn(gen_fn(f)?)),
            FileEntry::Verbatim(src) => {
                let parsed: syn::File =
                    syn::parse_file(src).map_err(|e| EmitError::VerbatimParse(e.to_string()))?;
                items.extend(parsed.items);
            }
        }
    }
    Ok(prettyplease::unparse(&syn::File {
        shebang: None,
        attrs: vec![],
        items,
    }))
}

/// Render a single blueprint function (also the fingerprint input).
pub fn generate_fn(f: &BlueprintFn) -> Result<String, EmitError> {
    let item = syn::Item::Fn(gen_fn(f)?);
    Ok(prettyplease::unparse(&syn::File {
        shebang: None,
        attrs: vec![],
        items: vec![item],
    }))
}

/// Normalize hand-entered snippet code into per-statement snippet bodies
/// (prettyplease-formatted, one string per statement). The editor calls this
/// when the user creates/edits a snippet node so that regeneration is
/// byte-stable.
pub fn normalize_snippet(code: &str) -> Result<Vec<String>, EmitError> {
    let block: syn::Block = syn::parse_str(&format!("{{ {code} }}"))
        .map_err(|e| EmitError::SnippetParse(e.to_string()))?;
    Ok(block.stmts.iter().map(stmt_to_string).collect())
}

/// Pretty-print one statement standalone (4-space-dedented body of a dummy
/// fn — prettyplease only prints whole files).
pub(crate) fn stmt_to_string(stmt: &syn::Stmt) -> String {
    let mut item: syn::ItemFn = syn::parse_quote! { fn __snippet__() {} };
    item.block.stmts = vec![stmt.clone()];
    let out = prettyplease::unparse(&syn::File {
        shebang: None,
        attrs: vec![],
        items: vec![syn::Item::Fn(item)],
    });
    let lines: Vec<&str> = out.lines().collect();
    // "fn __snippet__() {" … body … "}"
    lines[1..lines.len() - 1]
        .iter()
        .map(|l| l.strip_prefix("    ").unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Pretty-print one item standalone (trailing newline trimmed).
pub(crate) fn item_to_string(item: &syn::Item) -> String {
    prettyplease::unparse(&syn::File {
        shebang: None,
        attrs: vec![],
        items: vec![item.clone()],
    })
    .trim_end()
    .to_owned()
}

fn ident(s: &str) -> Result<syn::Ident, EmitError> {
    syn::parse_str::<syn::Ident>(s).map_err(|_| EmitError::BadIdent(s.to_owned()))
}

fn ty_syn(ty: Ty) -> syn::Type {
    let spelled = ty.rust_name().unwrap_or("()");
    syn::parse_str(spelled).expect("static type name parses")
}

fn gen_fn(f: &BlueprintFn) -> Result<syn::ItemFn, EmitError> {
    // Map every Let node id to its rendered identifier up front (references
    // may legally point anywhere the graph editor wired them).
    let mut locals = BTreeMap::new();
    collect_locals(&f.body, &mut locals)?;

    let id_lit = syn::LitStr::new(&f.id, Span::call_site());
    let attr: syn::Attribute = syn::parse_quote!(#[infinity::blueprint(id = #id_lit)]);

    let mut inputs = syn::punctuated::Punctuated::new();
    for p in &f.params {
        let name = ident(&p.name)?;
        let ty = ty_syn(p.ty);
        inputs.push(syn::FnArg::Typed(syn::PatType {
            attrs: vec![],
            pat: Box::new(syn::Pat::Ident(syn::PatIdent {
                attrs: vec![],
                by_ref: None,
                mutability: None,
                ident: name,
                subpat: None,
            })),
            colon_token: Default::default(),
            ty: Box::new(ty),
        }));
    }

    let output = match f.ret {
        Ty::Unit => syn::ReturnType::Default,
        t => syn::ReturnType::Type(Default::default(), Box::new(ty_syn(t))),
    };

    let mut sig_template: syn::Signature = syn::parse_quote! { fn __name__() };
    sig_template.ident = ident(&f.name)?;
    sig_template.inputs = inputs;
    sig_template.output = output;

    let stmts = gen_stmts(&f.body, &locals, true)?;

    Ok(syn::ItemFn {
        attrs: vec![attr],
        vis: syn::Visibility::Public(Default::default()),
        sig: sig_template,
        block: Box::new(syn::Block {
            brace_token: Default::default(),
            stmts,
        }),
    })
}

fn collect_locals(
    stmts: &[Stmt],
    out: &mut BTreeMap<LocalId, syn::Ident>,
) -> Result<(), EmitError> {
    for s in stmts {
        match s {
            Stmt::Let { id, binding, .. } => {
                let rendered = BlueprintFn::local_ident(binding, *id);
                if out.insert(*id, ident(&rendered)?).is_some() {
                    return Err(EmitError::DuplicateLocal(id.0));
                }
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_locals(then_body, out)?;
                collect_locals(else_body, out)?;
            }
            Stmt::While { body, .. } => collect_locals(body, out)?,
            _ => {}
        }
    }
    Ok(())
}

fn gen_stmts(
    stmts: &[Stmt],
    locals: &BTreeMap<LocalId, syn::Ident>,
    fn_tail: bool,
) -> Result<Vec<syn::Stmt>, EmitError> {
    let mut out = Vec::new();
    for (i, s) in stmts.iter().enumerate() {
        let is_last = i + 1 == stmts.len();
        match s {
            Stmt::Let {
                id,
                ty,
                mutable,
                value,
                ..
            } => {
                let name = locals[id].clone();
                let base = syn::Pat::Ident(syn::PatIdent {
                    attrs: vec![],
                    by_ref: None,
                    mutability: mutable.then(Default::default),
                    ident: name,
                    subpat: None,
                });
                let pat = match ty {
                    Some(t) => syn::Pat::Type(syn::PatType {
                        attrs: vec![],
                        pat: Box::new(base),
                        colon_token: Default::default(),
                        ty: Box::new(ty_syn(*t)),
                    }),
                    None => base,
                };
                out.push(syn::Stmt::Local(syn::Local {
                    attrs: vec![],
                    let_token: Default::default(),
                    pat,
                    init: Some(syn::LocalInit {
                        eq_token: Default::default(),
                        expr: Box::new(expr_syn(value, locals)?),
                        diverge: None,
                    }),
                    semi_token: Default::default(),
                }));
            }
            Stmt::Assign { target, value } => {
                let target_ident = locals
                    .get(target)
                    .ok_or(EmitError::UnknownLocal(target.0))?
                    .clone();
                out.push(syn::Stmt::Expr(
                    syn::Expr::Assign(syn::ExprAssign {
                        attrs: vec![],
                        left: Box::new(ident_expr(target_ident)),
                        eq_token: Default::default(),
                        right: Box::new(expr_syn(value, locals)?),
                    }),
                    Some(Default::default()),
                ));
            }
            Stmt::If { .. } => {
                out.push(syn::Stmt::Expr(gen_if(s, locals)?, None));
            }
            Stmt::While { cond, body } => {
                out.push(syn::Stmt::Expr(
                    syn::Expr::While(syn::ExprWhile {
                        attrs: vec![],
                        label: None,
                        while_token: Default::default(),
                        cond: Box::new(expr_syn(cond, locals)?),
                        body: syn::Block {
                            brace_token: Default::default(),
                            stmts: gen_stmts(body, locals, false)?,
                        },
                    }),
                    None,
                ));
            }
            Stmt::Return(value) => match value {
                // Idiomatic tail expression at the very end of the fn body.
                Some(e) if fn_tail && is_last => {
                    out.push(syn::Stmt::Expr(expr_syn(e, locals)?, None));
                }
                _ => {
                    out.push(syn::Stmt::Expr(
                        syn::Expr::Return(syn::ExprReturn {
                            attrs: vec![],
                            return_token: Default::default(),
                            expr: value
                                .as_ref()
                                .map(|e| expr_syn(e, locals).map(Box::new))
                                .transpose()?,
                        }),
                        Some(Default::default()),
                    ));
                }
            },
            Stmt::ExprStmt(e) => {
                out.push(syn::Stmt::Expr(
                    expr_syn(e, locals)?,
                    Some(Default::default()),
                ));
            }
            Stmt::Snippet(code) => {
                let block: syn::Block = syn::parse_str(&format!("{{ {code} }}"))
                    .map_err(|e| EmitError::SnippetParse(e.to_string()))?;
                out.extend(block.stmts);
            }
        }
    }
    Ok(out)
}

fn gen_if(stmt: &Stmt, locals: &BTreeMap<LocalId, syn::Ident>) -> Result<syn::Expr, EmitError> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        unreachable!("gen_if called on non-If");
    };
    let else_branch = if else_body.is_empty() {
        None
    } else if let [only @ Stmt::If { .. }] = else_body.as_slice() {
        // `else if` chain.
        Some((Default::default(), Box::new(gen_if(only, locals)?)))
    } else {
        Some((
            Default::default(),
            Box::new(syn::Expr::Block(syn::ExprBlock {
                attrs: vec![],
                label: None,
                block: syn::Block {
                    brace_token: Default::default(),
                    stmts: gen_stmts(else_body, locals, false)?,
                },
            })),
        ))
    };
    Ok(syn::Expr::If(syn::ExprIf {
        attrs: vec![],
        if_token: Default::default(),
        cond: Box::new(expr_syn(cond, locals)?),
        then_branch: syn::Block {
            brace_token: Default::default(),
            stmts: gen_stmts(then_body, locals, false)?,
        },
        else_branch,
    }))
}

fn ident_expr(ident: syn::Ident) -> syn::Expr {
    let mut segments = syn::punctuated::Punctuated::new();
    segments.push(syn::PathSegment {
        ident,
        arguments: syn::PathArguments::None,
    });
    syn::Expr::Path(syn::ExprPath {
        attrs: vec![],
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments,
        },
    })
}

fn lit_expr(lit: syn::Lit) -> syn::Expr {
    syn::Expr::Lit(syn::ExprLit { attrs: vec![], lit })
}

fn neg(e: syn::Expr) -> syn::Expr {
    syn::Expr::Unary(syn::ExprUnary {
        attrs: vec![],
        op: syn::UnOp::Neg(Default::default()),
        expr: Box::new(e),
    })
}

fn expr_syn(e: &Expr, locals: &BTreeMap<LocalId, syn::Ident>) -> Result<syn::Expr, EmitError> {
    Ok(match e {
        Expr::Lit(Lit::Float(v)) => {
            if !v.is_finite() {
                return Err(EmitError::NonFiniteFloat(*v));
            }
            // `{v:?}` is Rust's shortest round-trip float spelling and always
            // valid float-literal syntax; negatives render as unary minus.
            let abs = v.abs();
            let lit = lit_expr(syn::Lit::Float(syn::LitFloat::new(
                &format!("{abs:?}"),
                Span::call_site(),
            )));
            if v.is_sign_negative() {
                neg(lit)
            } else {
                lit
            }
        }
        Expr::Lit(Lit::Int(v)) => {
            if *v == i64::MIN {
                return Err(EmitError::IntMin);
            }
            let lit = lit_expr(syn::Lit::Int(syn::LitInt::new(
                &v.unsigned_abs().to_string(),
                Span::call_site(),
            )));
            if *v < 0 {
                neg(lit)
            } else {
                lit
            }
        }
        Expr::Lit(Lit::Bool(v)) => lit_expr(syn::Lit::Bool(syn::LitBool {
            value: *v,
            span: Span::call_site(),
        })),
        Expr::Lit(Lit::Str(s)) => lit_expr(syn::Lit::Str(syn::LitStr::new(s, Span::call_site()))),
        Expr::Param(name) => ident_expr(ident(name)?),
        Expr::Local(id) => ident_expr(locals.get(id).ok_or(EmitError::UnknownLocal(id.0))?.clone()),
        Expr::Unary(op, inner) => {
            // Canonical form: negative constants live in the literal itself
            // (the lifter folds `-lit`, so emitting Neg(lit) wouldn't
            // round-trip).
            if matches!(
                (op, inner.as_ref()),
                (UnOp::Neg, Expr::Lit(Lit::Int(_) | Lit::Float(_)))
            ) {
                return Err(EmitError::NegatedLiteral);
            }
            let inner = expr_syn(inner, locals)?;
            match op {
                UnOp::Neg => neg(inner),
                UnOp::Not => syn::Expr::Unary(syn::ExprUnary {
                    attrs: vec![],
                    op: syn::UnOp::Not(Default::default()),
                    expr: Box::new(inner),
                }),
            }
        }
        Expr::Binary(op, l, r) => syn::Expr::Binary(syn::ExprBinary {
            attrs: vec![],
            left: Box::new(expr_syn(l, locals)?),
            op: bin_op_syn(*op),
            right: Box::new(expr_syn(r, locals)?),
        }),
        Expr::Call { path, args } => {
            let mut segments = syn::punctuated::Punctuated::new();
            for seg in path {
                segments.push(syn::PathSegment {
                    ident: ident(seg)?,
                    arguments: syn::PathArguments::None,
                });
            }
            let func = syn::Expr::Path(syn::ExprPath {
                attrs: vec![],
                qself: None,
                path: syn::Path {
                    leading_colon: None,
                    segments,
                },
            });
            let mut call_args = syn::punctuated::Punctuated::new();
            for a in args {
                call_args.push(expr_syn(a, locals)?);
            }
            syn::Expr::Call(syn::ExprCall {
                attrs: vec![],
                func: Box::new(func),
                paren_token: Default::default(),
                args: call_args,
            })
        }
    })
}

fn bin_op_syn(op: BinOp) -> syn::BinOp {
    match op {
        BinOp::Add => syn::BinOp::Add(Default::default()),
        BinOp::Sub => syn::BinOp::Sub(Default::default()),
        BinOp::Mul => syn::BinOp::Mul(Default::default()),
        BinOp::Div => syn::BinOp::Div(Default::default()),
        BinOp::Rem => syn::BinOp::Rem(Default::default()),
        BinOp::Eq => syn::BinOp::Eq(Default::default()),
        BinOp::Ne => syn::BinOp::Ne(Default::default()),
        BinOp::Lt => syn::BinOp::Lt(Default::default()),
        BinOp::Le => syn::BinOp::Le(Default::default()),
        BinOp::Gt => syn::BinOp::Gt(Default::default()),
        BinOp::Ge => syn::BinOp::Ge(Default::default()),
        BinOp::And => syn::BinOp::And(Default::default()),
        BinOp::Or => syn::BinOp::Or(Default::default()),
    }
}
