//! Rust → graph. Lifts the liftable subset back into `inf_blueprint`
//! statements; anything else becomes a snippet (statement level) or a
//! verbatim item (item level). Lifting NEVER fails on parseable Rust.
//!
//! Identifier semantics: references are resolved lexically against the
//! scopes the lifter has seen (locals shadow params, inner scopes shadow
//! outer). A binder introduced by an *unliftable* statement poisons its
//! name — later liftable statements that reference it would otherwise be
//! wired to the wrong node, so they conservatively become snippets too.

use std::collections::{HashMap, HashSet};

use inf_blueprint::{BinOp, Binding, BlueprintFn, Expr, Lit, LocalId, Param, Stmt, Ty, UnOp};
use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;

use crate::emit::{item_to_string, stmt_to_string};
use crate::{BlueprintFile, FileEntry};

#[derive(Debug, thiserror::Error)]
pub enum LiftError {
    #[error("source does not parse as Rust: {0}")]
    Parse(String),
}

/// Result of lifting a source file. `warnings` lists blueprint-attributed
/// functions that could not be lifted structurally (kept verbatim instead).
#[derive(Debug)]
pub struct Lifted {
    pub file: BlueprintFile,
    pub warnings: Vec<String>,
}

pub fn lift_file(src: &str) -> Result<Lifted, LiftError> {
    let file = syn::parse_file(src).map_err(|e| LiftError::Parse(e.to_string()))?;
    let mut entries = Vec::new();
    let mut warnings = Vec::new();
    for item in &file.items {
        match item {
            syn::Item::Fn(f) if blueprint_id(f).is_some() => match lift_fn(f) {
                Ok(bp) => entries.push(FileEntry::Blueprint(bp)),
                Err(why) => {
                    warnings.push(format!("fn `{}` kept verbatim: {why}", f.sig.ident));
                    entries.push(FileEntry::Verbatim(item_to_string(item)));
                }
            },
            _ => entries.push(FileEntry::Verbatim(item_to_string(item))),
        }
    }
    Ok(Lifted {
        file: BlueprintFile { entries },
        warnings,
    })
}

/// Extract the id from `#[infinity::blueprint(id = "…")]`, if present.
fn blueprint_id(f: &syn::ItemFn) -> Option<String> {
    for attr in &f.attrs {
        let path = attr.path();
        let is_ours = path.segments.len() == 2
            && path.segments[0].ident == "infinity"
            && path.segments[1].ident == "blueprint";
        if !is_ours {
            continue;
        }
        let mut found = None;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("id") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                found = Some(lit.value());
            }
            Ok(())
        });
        return found;
    }
    None
}

// ── scope tracking ──────────────────────────────────────────────────────────

/// `Some(id)` = liftable local; `None` = poisoned (bound by a snippet).
type Scope = HashMap<String, Option<LocalId>>;

struct Env {
    scopes: Vec<Scope>,
    params: HashSet<String>,
    used_ids: HashSet<LocalId>,
    next_fresh: u32,
}

enum Resolved {
    Local(LocalId),
    Param,
    Poisoned,
    Unknown,
}

impl Env {
    fn resolve(&self, name: &str) -> Resolved {
        for scope in self.scopes.iter().rev() {
            match scope.get(name) {
                Some(Some(id)) => return Resolved::Local(*id),
                Some(None) => return Resolved::Poisoned,
                None => {}
            }
        }
        if self.params.contains(name) {
            Resolved::Param
        } else {
            Resolved::Unknown
        }
    }

    fn bind(&mut self, name: &str, id: LocalId) {
        self.scopes
            .last_mut()
            .expect("scope stack never empty")
            .insert(name.to_owned(), Some(id));
    }

    fn poison(&mut self, name: &str) {
        self.scopes
            .last_mut()
            .expect("scope stack never empty")
            .insert(name.to_owned(), None);
    }

    fn fresh(&mut self) -> LocalId {
        loop {
            let id = LocalId(self.next_fresh);
            self.next_fresh += 1;
            if self.used_ids.insert(id) {
                return id;
            }
        }
    }
}

/// Highest node id mentioned by any identifier anywhere in the token stream
/// (fresh ids start above it so snippet-referenced ids can never collide).
fn scan_max_id(ts: TokenStream, max: &mut u32) {
    for tt in ts {
        match tt {
            TokenTree::Ident(i) => {
                if let Some((_, id)) = BlueprintFn::parse_local_ident(&i.to_string()) {
                    *max = (*max).max(id.0 + 1);
                }
            }
            TokenTree::Group(g) => scan_max_id(g.stream(), max),
            _ => {}
        }
    }
}

// ── function lifting ────────────────────────────────────────────────────────

fn lift_fn(f: &syn::ItemFn) -> Result<BlueprintFn, String> {
    let id = blueprint_id(f).expect("caller checked the attribute");
    if f.attrs.len() != 1 {
        return Err("extra attributes on blueprint fn".into());
    }
    if !matches!(f.vis, syn::Visibility::Public(_)) {
        return Err("blueprint fns must be `pub`".into());
    }
    let sig = &f.sig;
    if sig.constness.is_some()
        || sig.asyncness.is_some()
        || sig.unsafety.is_some()
        || sig.abi.is_some()
        || sig.variadic.is_some()
        || !sig.generics.params.is_empty()
        || sig.generics.where_clause.is_some()
    {
        return Err("unsupported fn qualifiers/generics".into());
    }

    let mut params = Vec::new();
    for input in &sig.inputs {
        let syn::FnArg::Typed(pt) = input else {
            return Err("self parameter".into());
        };
        if !pt.attrs.is_empty() {
            return Err("parameter attributes".into());
        }
        let syn::Pat::Ident(pi) = pt.pat.as_ref() else {
            return Err("pattern parameter".into());
        };
        if pi.by_ref.is_some() || pi.mutability.is_some() || pi.subpat.is_some() {
            return Err("unsupported parameter binding mode".into());
        }
        let ty = lift_ty(&pt.ty).ok_or("unsupported parameter type")?;
        params.push(Param {
            name: pi.ident.to_string(),
            ty,
        });
    }

    let ret = match &sig.output {
        syn::ReturnType::Default => Ty::Unit,
        syn::ReturnType::Type(_, t) => lift_ty(t).ok_or("unsupported return type")?,
    };

    let mut max = 0;
    scan_max_id(f.to_token_stream(), &mut max);
    let mut env = Env {
        scopes: vec![Scope::new()],
        params: params.iter().map(|p| p.name.clone()).collect(),
        used_ids: HashSet::new(),
        next_fresh: max,
    };

    let body = lift_block(&f.block.stmts, &mut env, true);

    Ok(BlueprintFn {
        id,
        name: sig.ident.to_string(),
        params,
        ret,
        body,
    })
}

fn lift_block(stmts: &[syn::Stmt], env: &mut Env, fn_tail: bool) -> Vec<Stmt> {
    let mut out = Vec::new();
    for (i, s) in stmts.iter().enumerate() {
        let is_last = i + 1 == stmts.len();
        out.push(lift_stmt(s, env, fn_tail && is_last));
    }
    out
}

/// Lift one statement; on any unsupported shape, fall back to a snippet and
/// poison identifiers the statement would have bound.
fn lift_stmt(s: &syn::Stmt, env: &mut Env, at_fn_tail: bool) -> Stmt {
    match try_lift_stmt(s, env, at_fn_tail) {
        Some(stmt) => stmt,
        None => {
            if let syn::Stmt::Local(local) = s {
                poison_pat(&local.pat, env);
            }
            Stmt::Snippet(stmt_to_string(s))
        }
    }
}

fn poison_pat(pat: &syn::Pat, env: &mut Env) {
    struct V<'a> {
        env: &'a mut Env,
    }
    impl syn::visit::Visit<'_> for V<'_> {
        fn visit_pat_ident(&mut self, pi: &syn::PatIdent) {
            self.env.poison(&pi.ident.to_string());
            syn::visit::visit_pat_ident(self, pi);
        }
    }
    syn::visit::Visit::visit_pat(&mut V { env }, pat);
}

fn try_lift_stmt(s: &syn::Stmt, env: &mut Env, at_fn_tail: bool) -> Option<Stmt> {
    match s {
        syn::Stmt::Local(local) => {
            if !local.attrs.is_empty() {
                return None;
            }
            let init = local.init.as_ref()?;
            if init.diverge.is_some() {
                return None;
            }
            // `let [mut] ident` or `let [mut] ident: ty`
            let (pat_ident, ty) = match &local.pat {
                syn::Pat::Ident(pi) => (pi, None),
                syn::Pat::Type(pt) => {
                    let syn::Pat::Ident(pi) = pt.pat.as_ref() else {
                        return None;
                    };
                    (pi, Some(lift_ty(&pt.ty)?))
                }
                _ => return None,
            };
            if pat_ident.by_ref.is_some() || pat_ident.subpat.is_some() {
                return None;
            }
            // Value is lifted in the OUTER scope (shadowing: `let x = x + 1`).
            let value = lift_expr(&init.expr, env)?;

            let ident = pat_ident.ident.to_string();
            let (binding, id) = match BlueprintFn::parse_local_ident(&ident) {
                Some((name, id)) => {
                    if !env.used_ids.insert(id) {
                        return None; // duplicate id binder — keep verbatim
                    }
                    (name.map(Binding::Named).unwrap_or(Binding::Anon), id)
                }
                None => (Binding::Raw(ident.clone()), env.fresh()),
            };
            env.bind(&ident, id);
            Some(Stmt::Let {
                id,
                binding,
                ty,
                mutable: pat_ident.mutability.is_some(),
                value,
            })
        }

        syn::Stmt::Expr(e, semi) => try_lift_expr_stmt(e, semi.is_some(), env, at_fn_tail),

        // Macros and nested items are opaque by design.
        syn::Stmt::Macro(_) | syn::Stmt::Item(_) => None,
    }
}

fn try_lift_expr_stmt(
    e: &syn::Expr,
    has_semi: bool,
    env: &mut Env,
    at_fn_tail: bool,
) -> Option<Stmt> {
    match e {
        syn::Expr::Assign(a) if a.attrs.is_empty() => {
            let syn::Expr::Path(p) = a.left.as_ref() else {
                return None;
            };
            let name = single_ident(&p.path)?;
            let Resolved::Local(target) = env.resolve(&name) else {
                return None;
            };
            let value = lift_expr(&a.right, env)?;
            Some(Stmt::Assign { target, value })
        }

        syn::Expr::If(_) => lift_if(e, env),

        syn::Expr::While(w) if w.attrs.is_empty() && w.label.is_none() => {
            let cond = lift_expr(&w.cond, env)?;
            env.scopes.push(Scope::new());
            let body = lift_block(&w.body.stmts, env, false);
            env.scopes.pop();
            Some(Stmt::While { cond, body })
        }

        syn::Expr::Return(r) if r.attrs.is_empty() => {
            let value = match &r.expr {
                Some(e) => Some(lift_expr(e, env)?),
                None => None,
            };
            Some(Stmt::Return(value))
        }

        // Tail expression of the fn body = implicit return.
        _ if !has_semi && at_fn_tail => Some(Stmt::Return(Some(lift_expr(e, env)?))),

        // A tail expression of a NESTED block has value semantics we can't
        // model — snippet (handled by the caller via None).
        _ if !has_semi => None,

        // Plain effect statement, e.g. `api::fire(x);`
        _ => Some(Stmt::ExprStmt(lift_expr(e, env)?)),
    }
}

fn lift_if(e: &syn::Expr, env: &mut Env) -> Option<Stmt> {
    let syn::Expr::If(i) = e else { return None };
    if !i.attrs.is_empty() {
        return None;
    }
    // `if let` conditions contain `let`; reject anything but a plain expr.
    let cond = lift_expr(&i.cond, env)?;
    env.scopes.push(Scope::new());
    let then_body = lift_block(&i.then_branch.stmts, env, false);
    env.scopes.pop();
    let else_body = match &i.else_branch {
        None => Vec::new(),
        Some((_, else_expr)) => match else_expr.as_ref() {
            syn::Expr::Block(b) if b.attrs.is_empty() && b.label.is_none() => {
                env.scopes.push(Scope::new());
                let stmts = lift_block(&b.block.stmts, env, false);
                env.scopes.pop();
                stmts
            }
            nested @ syn::Expr::If(_) => vec![lift_if(nested, env)?],
            _ => return None,
        },
    };
    Some(Stmt::If {
        cond,
        then_body,
        else_body,
    })
}

fn lift_ty(t: &syn::Type) -> Option<Ty> {
    match t {
        syn::Type::Path(p) if p.qself.is_none() => match single_ident(&p.path)?.as_str() {
            "f64" => Some(Ty::Float),
            "i64" => Some(Ty::Int),
            "bool" => Some(Ty::Bool),
            "String" => Some(Ty::Str),
            _ => None,
        },
        syn::Type::Tuple(t) if t.elems.is_empty() => Some(Ty::Unit),
        _ => None,
    }
}

// ── expression lifting ──────────────────────────────────────────────────────

fn single_ident(path: &syn::Path) -> Option<String> {
    if path.leading_colon.is_some() || path.segments.len() != 1 {
        return None;
    }
    let seg = &path.segments[0];
    if !matches!(seg.arguments, syn::PathArguments::None) {
        return None;
    }
    Some(seg.ident.to_string())
}

fn path_segments(path: &syn::Path) -> Option<Vec<String>> {
    if path.leading_colon.is_some() {
        return None;
    }
    path.segments
        .iter()
        .map(|seg| matches!(seg.arguments, syn::PathArguments::None).then(|| seg.ident.to_string()))
        .collect()
}

fn lift_expr(e: &syn::Expr, env: &Env) -> Option<Expr> {
    match e {
        syn::Expr::Paren(p) if p.attrs.is_empty() => lift_expr(&p.expr, env),
        syn::Expr::Group(g) if g.attrs.is_empty() => lift_expr(&g.expr, env),

        syn::Expr::Lit(l) if l.attrs.is_empty() => lift_lit(&l.lit).map(Expr::Lit),

        syn::Expr::Path(p) if p.attrs.is_empty() && p.qself.is_none() => {
            let name = single_ident(&p.path)?;
            match env.resolve(&name) {
                Resolved::Local(id) => Some(Expr::Local(id)),
                Resolved::Param => Some(Expr::Param(name)),
                Resolved::Poisoned | Resolved::Unknown => None,
            }
        }

        syn::Expr::Unary(u) if u.attrs.is_empty() => {
            let inner = lift_expr(&u.expr, env)?;
            match u.op {
                // Fold `-lit` into the literal (canonical form).
                syn::UnOp::Neg(_) => Some(match inner {
                    Expr::Lit(Lit::Int(v)) => Expr::Lit(Lit::Int(v.checked_neg()?)),
                    Expr::Lit(Lit::Float(v)) => Expr::Lit(Lit::Float(-v)),
                    other => Expr::Unary(UnOp::Neg, Box::new(other)),
                }),
                syn::UnOp::Not(_) => Some(Expr::Unary(UnOp::Not, Box::new(inner))),
                _ => None,
            }
        }

        syn::Expr::Binary(b) if b.attrs.is_empty() => {
            let op = lift_bin_op(&b.op)?;
            let l = lift_expr(&b.left, env)?;
            let r = lift_expr(&b.right, env)?;
            Some(Expr::Binary(op, Box::new(l), Box::new(r)))
        }

        syn::Expr::Call(c) if c.attrs.is_empty() => {
            let syn::Expr::Path(p) = c.func.as_ref() else {
                return None;
            };
            if !p.attrs.is_empty() || p.qself.is_some() {
                return None;
            }
            let path = path_segments(&p.path)?;
            let args = c
                .args
                .iter()
                .map(|a| lift_expr(a, env))
                .collect::<Option<Vec<_>>>()?;
            Some(Expr::Call { path, args })
        }

        _ => None,
    }
}

fn lift_lit(l: &syn::Lit) -> Option<Lit> {
    match l {
        syn::Lit::Int(i) if matches!(i.suffix(), "" | "i64") => {
            i.base10_parse::<i64>().ok().map(Lit::Int)
        }
        syn::Lit::Float(f) if matches!(f.suffix(), "" | "f64") => {
            f.base10_parse::<f64>().ok().map(Lit::Float)
        }
        syn::Lit::Bool(b) => Some(Lit::Bool(b.value)),
        syn::Lit::Str(s) if s.suffix().is_empty() => Some(Lit::Str(s.value())),
        _ => None,
    }
}

fn lift_bin_op(op: &syn::BinOp) -> Option<BinOp> {
    Some(match op {
        syn::BinOp::Add(_) => BinOp::Add,
        syn::BinOp::Sub(_) => BinOp::Sub,
        syn::BinOp::Mul(_) => BinOp::Mul,
        syn::BinOp::Div(_) => BinOp::Div,
        syn::BinOp::Rem(_) => BinOp::Rem,
        syn::BinOp::Eq(_) => BinOp::Eq,
        syn::BinOp::Ne(_) => BinOp::Ne,
        syn::BinOp::Lt(_) => BinOp::Lt,
        syn::BinOp::Le(_) => BinOp::Le,
        syn::BinOp::Gt(_) => BinOp::Gt,
        syn::BinOp::Ge(_) => BinOp::Ge,
        syn::BinOp::And(_) => BinOp::And,
        syn::BinOp::Or(_) => BinOp::Or,
        _ => return None,
    })
}
