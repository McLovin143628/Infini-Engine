//! Infinity Blueprints: graph semantics + tree-walking interpreter for editor preview.
//!
//! Spike B model (docs/ROADMAP.md §5): the data structures that make the
//! graph the *source of truth* while real Rust stays a faithful, hand-editable
//! projection. `inf-transpile` renders this model to Rust (quote/prettyplease)
//! and lifts edited Rust back (syn). The contract that makes the round trip
//! lossless:
//!
//! - **Value nodes carry their id inside the generated identifier** —
//!   `n7`, or `speed_n7` when the node has a display name. Identifiers
//!   survive `syn`; comments do not, so nothing load-bearing lives in
//!   comments.
//! - **Statement nodes** (`if`/`while`/`return`/calls) are identified
//!   structurally by their position in the body.
//! - **Anything the lifter cannot understand becomes a [`Stmt::Snippet`]** —
//!   opaque, verbatim Rust that regenerates exactly. Hand edits are never
//!   lost and sync never fails; they just opt out of graph editing until
//!   rewritten in liftable shape.
//!
//! The editor's full node-graph UI model (P6) maps onto these statements and
//! expression trees; the P6 interpreter walks this same model for live
//! preview so that preview == shipped compiled code (parity CI gate).

use serde::{Deserialize, Serialize};

/// Pin/value types available in the Spike B subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ty {
    Float,
    Int,
    Bool,
    Str,
    Unit,
}

impl Ty {
    /// The Rust spelling of this type in generated code.
    pub fn rust_name(self) -> Option<&'static str> {
        match self {
            Ty::Float => Some("f64"),
            Ty::Int => Some("i64"),
            Ty::Bool => Some("bool"),
            Ty::Str => Some("String"),
            Ty::Unit => None,
        }
    }
}

/// Literal values. Floats must be finite (NaN/inf have no literal spelling in
/// Rust source; the editor rejects them at input time).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Lit {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
}

/// Identity of a value-producing node (a `let` binding in the projection).
/// Encoded in the generated identifier (`n{id}` / `{name}_n{id}`) so it
/// survives arbitrary reformatting and `syn` round trips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LocalId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnOp {
    /// `-x`
    Neg,
    /// `!x`
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// A data-pin expression tree. Trees (rather than per-op nodes) mirror Rust
/// expressions 1:1, so nested hand-written arithmetic lifts without loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Lit(Lit),
    /// Reference to a function parameter by name.
    Param(String),
    /// Reference to the output of a value node.
    Local(LocalId),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// Free-function call, e.g. `api::print_f64(x)`.
    Call {
        path: Vec<String>,
        args: Vec<Expr>,
    },
}

/// How a value node renders its binder identifier.
///
/// `Raw` exists for hand-written bindings: renaming `x` to `x_n7` on
/// regeneration would silently unbind snippet statements that still say `x`,
/// so raw binders keep their bare identifier and their node id lives only in
/// the graph (re-derived deterministically on each lift). The editor "adopts"
/// a raw binder into `Named` when the user first wires it in the graph UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Binding {
    /// Editor-created, unnamed: renders `n{id}`.
    Anon,
    /// Editor-named: renders `{name}_n{id}`.
    Named(String),
    /// Hand-written: renders `{name}` verbatim, id not encoded.
    Raw(String),
}

/// An exec-pin statement node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// A value node: `let [mut] <binding>[: ty] = value;`
    Let {
        id: LocalId,
        binding: Binding,
        /// Optional explicit type annotation in the projection.
        ty: Option<Ty>,
        mutable: bool,
        value: Expr,
    },
    /// `target = value;`
    Assign {
        target: LocalId,
        value: Expr,
    },
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    Return(Option<Expr>),
    /// An expression evaluated for effect, e.g. a call statement.
    ExprStmt(Expr),
    /// Opaque hand-written Rust statement(s), regenerated verbatim. The
    /// string is the prettyplease-normalized source of one or more
    /// statements.
    Snippet(String),
}

/// A function parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty: Ty,
}

/// One blueprint function: the unit of graph↔code sync. Rendered as a free
/// `pub fn` tagged `#[infinity::blueprint(id = "…")]` (a no-op passthrough
/// attribute macro shipped with the engine runtime).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlueprintFn {
    /// Stable identity across renames (GUID string in real projects).
    pub id: String,
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Ty,
    pub body: Vec<Stmt>,
}

impl BlueprintFn {
    /// The identifier a `Let` node renders to.
    pub fn local_ident(binding: &Binding, id: LocalId) -> String {
        match binding {
            Binding::Anon => format!("n{}", id.0),
            Binding::Named(n) => format!("{n}_n{}", id.0),
            Binding::Raw(n) => n.clone(),
        }
    }

    /// Parse an identifier produced by [`Self::local_ident`] back into
    /// `(display_name, id)`. Returns `None` for identifiers that don't carry
    /// a node id (pure hand-written bindings).
    pub fn parse_local_ident(ident: &str) -> Option<(Option<String>, LocalId)> {
        let (prefix, digits) = match ident.rsplit_once("_n") {
            Some((name, digits)) => (Some(name), digits),
            None => (None, ident.strip_prefix('n')?),
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let id = digits.parse().ok()?;
        match prefix {
            Some("") => None,
            Some(name) => Some((Some(name.to_owned()), LocalId(id))),
            None => Some((None, LocalId(id))),
        }
    }

    /// All `Let` ids appearing anywhere in the function, in declaration order.
    pub fn declared_locals(&self) -> Vec<LocalId> {
        fn walk(stmts: &[Stmt], out: &mut Vec<LocalId>) {
            for s in stmts {
                match s {
                    Stmt::Let { id, .. } => out.push(*id),
                    Stmt::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        walk(then_body, out);
                        walk(else_body, out);
                    }
                    Stmt::While { body, .. } => walk(body, out),
                    _ => {}
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.body, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ident_round_trip() {
        for (binding, id) in [
            (Binding::Anon, LocalId(0)),
            (Binding::Anon, LocalId(41)),
            (Binding::Named("speed".into()), LocalId(7)),
            (Binding::Named("hit_count".into()), LocalId(123)),
        ] {
            let ident = BlueprintFn::local_ident(&binding, id);
            let expected_name = match &binding {
                Binding::Named(n) => Some(n.clone()),
                _ => None,
            };
            assert_eq!(
                BlueprintFn::parse_local_ident(&ident),
                Some((expected_name, id)),
                "ident {ident}"
            );
        }
    }

    #[test]
    fn raw_binding_renders_bare() {
        assert_eq!(
            BlueprintFn::local_ident(&Binding::Raw("x".into()), LocalId(9)),
            "x"
        );
    }

    #[test]
    fn foreign_idents_rejected() {
        for ident in ["foo", "n", "nx7", "n7x", "_n7", "x_n", "x_n7y"] {
            assert_eq!(BlueprintFn::parse_local_ident(ident), None, "{ident}");
        }
    }
}
