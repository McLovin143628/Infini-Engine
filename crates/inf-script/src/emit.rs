//! The emitter: `BlueprintFn` IR → `.infini` text. `raise`'s text twin.
//!
//! # The text face is **total**; the graph face is not
//!
//! This is the wave's sharpest finding and it belongs at the top of the module
//! that proves it. `raise` — IR → *graph* — refuses ten shapes (five flow forms
//! and five that are not flow at all, `RaiseError::NonLinear` chief among them),
//! and a single unraisable statement makes a whole handler unraisable. This
//! emitter refuses **none of them**. Every `Stmt` and every `Expr` in the IR has
//! a `.infini` spelling, including the ones a canvas cannot draw:
//!
//! | IR shape | graph | text |
//! |---|---|---|
//! | `Stmt::If` that is not last in its block | `NonLinear` | `if … then … end`, followed by whatever follows |
//! | `Stmt::Assign` | `UnsupportedStmt("assign")` | `x = …` |
//! | `Stmt::Snippet` | `UnsupportedStmt("snippet")` | `rust [[ … ]]` |
//! | a call in value position | `UnsupportedExpr("pure call")` | an ordinary call |
//! | `flow.do_once` / `flip_flop` / `gate` state | refused | `nodestate.get_or(…)` / `nodestate.set(…)` |
//!
//! So "graphs and text are two views of one program" is true, and the *text*
//! view is the complete one. What the emitter does refuse is a handful of IR
//! states no producer in the tree makes and no text could denote — a non-finite
//! float literal, a binder whose name is not an identifier, two live binders with
//! the same name, a handler whose parameters are not its event's — and each is a
//! value with a reason ([`EmitError`]).
//!
//! # …and the converse, which is the half that was false
//!
//! "Total" was stated as a property of *this module alone*: nothing the IR can
//! hold is refused. Read that way it points the wrong direction. What a designer
//! needs is **a Blueprint that opens as text and comes back**, and that needs
//! the converse — *anything the emitter writes, the parser reads back* — which
//! nothing asserted and which was false in two places the SCRIPT1a audit found:
//!
//! * a **comparison in a comparison's left operand**. `cmp.lt` wired into
//!   `cmp.eq`'s `a` input is a graph anybody can draw; it printed
//!   `1.0 < 2.0 == true`, which the grammar refuses as a chained comparison. 36
//!   of the 338 operator pairings. Comparisons do not associate, so **both**
//!   operands need the parenthesis, not only the right one;
//! * a **`Stmt::ExprStmt` that is not a statement-callable call** — a literal, a
//!   local, a binary expression, a `vars::get` (which prints as a bare name), a
//!   `nodestate::get_or` (a value spelling with no statement one). InfiniScript
//!   has no evaluate-and-discard statement, so those are
//!   [`EmitError::UnspellableStatement`] — the verdict `raise` already gives the
//!   shape.
//!
//! An emitter that produces a file its own parser rejects is worse than one that
//! refuses, because the refusal is silent until somebody opens the result. Both
//! directions are gated now: `tests/hostile.rs` sweeps all 338 pairings and the
//! two-node graph itself, and mutates a real script character by character
//! requiring every mutation that parses to emit and re-parse identically.
//!
//! The sixth refusal is a resource bound rather than a shape:
//! [`EmitError::TooDeep`]. A graph can chain operator nodes without limit and
//! lowering makes the chain a left-deep `Expr`; printing one used to be a stack
//! overflow. `MAX_EMIT_NESTING` is twice the parser's budget, which is a proof
//! rather than a guess that anything the parser accepts, this module can write.
//!
//! # Two rules the round trip rests on
//!
//! **A negated literal prints as the negative literal it means.** `Unary(Neg,
//! Lit::Int(1))` is written `-1`, not `-(1)`. The first draft kept the two apart
//! so a graph holding the second could print and come back; `tests/
//! transpile_bridge.rs` then found what that cost — `inf_transpile::emit`
//! **refuses** `Unary(Neg, Lit)` outright, because the lifter folds `-lit` on the
//! way back and the shape cannot round-trip through Rust. A language able to
//! write it is a language able to write programs the cook refuses. So
//! [`crate::parse`] folds and this module prints the canonical form, and the
//! round trip survives because the two *agree* rather than because they disagree
//! carefully.
//!
//! **A member variable prints bare unless something shadows it.** `vars::get("x")`
//! is `x` — until a `local x` is live at that point, at which case it prints as
//! `var.get("x")`, because a bare `x` there would resolve to the local and mean a
//! different program. The emitter therefore tracks scope, which is not a nicety:
//! silently re-binding a name across a round trip is exactly the class of defect
//! this house writes gates for.

use std::collections::HashMap;
use std::fmt::Write as _;

use inf_blueprint::loopshape;
use inf_blueprint::lower::node_type_of_path;
use inf_blueprint::semantics::{EventBinding, EventKind, Variable};
use inf_blueprint::{
    BinOp, BlueprintClass, BlueprintFn, Expr, Lit, LocalId, Param, Stmt, Ty, UnOp,
};

use crate::parse::ty_name;

/// Why an IR could not be written as `.infini`. Each variant is an IR state no
/// producer in the tree makes; they are named rather than papered over so that
/// the day one appears it says which.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EmitError {
    #[error("a float literal is {0}, which has no written form (the IR's `Lit::Float` is documented finite)")]
    NonFiniteFloat(f64),
    #[error(
        "this IR nests more than {MAX_EMIT_NESTING} levels deep, which no `.infini` \
         file can hold — the parser refuses past {} (a graph can chain operator \
         nodes without limit; text cannot)",
        crate::parse::MAX_NESTING
    )]
    TooDeep,
    #[error(
        "`{0}` is an expression, not a statement — InfiniScript has no \
         evaluate-and-discard statement, so this IR has no written form"
    )]
    UnspellableStatement(String),
    #[error("`{0}` is not an identifier, so it cannot be written as a binder")]
    UnspellableName(String),
    #[error("two live bindings are both called `{0}`; the later one would capture the earlier's readers")]
    ShadowedBinder(String),
    #[error("local n{0} is read but never bound")]
    UnboundLocal(u32),
    #[error("`{0}` is not a call InfiniScript can write")]
    UnspellableCall(String),
    #[error("handler `{0}` takes {1}, which is not what that event delivers")]
    SignatureMismatch(String, String),
}

/// Write a whole actor class as a `.infini` file.
pub fn emit_class(class: &BlueprintClass) -> Result<String, EmitError> {
    let mut out = String::new();
    let _ = writeln!(out, "actor {}", quote(&class.name));
    if !class.variables.is_empty() {
        out.push('\n');
        for v in &class.variables {
            out.push_str(&emit_var(v)?);
        }
    }
    for binding in &class.events {
        out.push('\n');
        out.push_str(&emit_event(binding)?);
    }
    for f in &class.functions {
        out.push('\n');
        out.push_str(&emit_function(f)?);
    }
    Ok(out)
}

/// Write one handler or function on its own — the graph↔text bridge's Ring-0
/// half, and what `parse_fn` reads back.
///
/// Which of the two it is comes from the id: the event keys
/// (`begin_play`, `tick`, `collision`, `water_*`, `destroyed`, `input:…`,
/// `custom:…`) are handlers and everything else is a function. A user function
/// deliberately named `tick` would be written as a handler; the unit emitter,
/// which knows which list a body came from, has no such ambiguity.
pub fn emit_fn(f: &BlueprintFn) -> Result<String, EmitError> {
    match handler_event(&f.id) {
        Some(event) => emit_event(&EventBinding {
            event,
            body: f.clone(),
        }),
        None => emit_function(f),
    }
}

/// The [`EventKind`] an id names, or `None` when the id is a function's.
fn handler_event(id: &str) -> Option<EventKind> {
    Some(match id {
        "begin_play" => EventKind::BeginPlay,
        "tick" => EventKind::Tick,
        "collision" => EventKind::Collision,
        "water_enter" => EventKind::WaterEnter,
        "water_exit" => EventKind::WaterExit,
        "water_splash" => EventKind::WaterSplash,
        "destroyed" => EventKind::Destroyed,
        other => match (other.strip_prefix("input:"), other.strip_prefix("custom:")) {
            (Some(a), _) => EventKind::Input(a.to_string()),
            (_, Some(n)) => EventKind::Custom(n.to_string()),
            _ => return None,
        },
    })
}

fn emit_var(v: &Variable) -> Result<String, EmitError> {
    let name = ident(&v.name)?;
    let mut s = format!("var {name}: {} = {}", ty_name(v.ty), literal(&v.default)?);
    if v.exposed {
        s.push_str(" exposed");
    }
    s.push('\n');
    Ok(s)
}

fn emit_event(b: &EventBinding) -> Result<String, EmitError> {
    let signature = b.event.signature();
    if b.body.params != signature {
        return Err(EmitError::SignatureMismatch(
            b.event.key(),
            describe(&b.body.params),
        ));
    }
    let head = match &b.event {
        EventKind::Input(a) => format!("on input {}", quote(a)),
        EventKind::Custom(n) => format!("on custom {}", quote(n)),
        other => format!("on {}", other.key()),
    };
    let params: Vec<String> = signature.iter().map(|p| p.name.clone()).collect();
    let mut w = Writer::new(&b.body)?;
    let body = w.block(&b.body.body, 1)?;
    Ok(format!("{head}({})\n{body}end\n", params.join(", ")))
}

fn emit_function(f: &BlueprintFn) -> Result<String, EmitError> {
    let name = ident(&f.name)?;
    let params: Vec<String> = f
        .params
        .iter()
        .map(|p| Ok(format!("{}: {}", ident(&p.name)?, ty_name(p.ty))))
        .collect::<Result<_, EmitError>>()?;
    let ret = match f.ret {
        Ty::Unit => String::new(),
        t => format!(" -> {}", ty_name(t)),
    };
    let mut w = Writer::new(f)?;
    let body = w.block(&f.body, 1)?;
    Ok(format!(
        "function {name}({}){ret}\n{body}end\n",
        params.join(", ")
    ))
}

fn describe(params: &[Param]) -> String {
    format!(
        "({})",
        params
            .iter()
            .map(|p| format!("{}: {}", p.name, ty_name(p.ty)))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Walks one function, carrying the scope the bare-variable rule needs and the
/// **display numbering** that makes the emitted text a fixed point.
///
/// # Why the ids are renumbered
///
/// `Binding::Anon` prints `n7` and `Binding::Named("speed")` prints `speed_n7`,
/// and the number is the local's id. A graph's lowerer numbers locals in *its*
/// walk (materialised values, loop counters and all) and the parser numbers them
/// in the parser's — so printing the lowerer's numbers gives text that re-parses
/// to different numbers and prints differently the second time. The text would
/// not be a fixed point, and "open a Blueprint as text, save it, no diff" would
/// be false on the first save.
///
/// So the writer numbers as it goes, in **exactly the parser's allocation
/// order**: a `let`'s binder after its value, a `for`'s index, bound and counter
/// in that order after its bounds, a `while`'s counter after its condition — the
/// last two invisible in the text and numbered anyway, because the parser will
/// allocate them when it reads the loop back. The law-1 arm is the gate on that
/// mirror: the day the two orders disagree, a parsed script stops round-tripping
/// to itself.
struct Writer {
    /// Every local's written identifier, by id.
    idents: HashMap<LocalId, String>,
    /// Names currently live: parameters, plus locals of enclosing blocks.
    live: Vec<Vec<String>>,
    /// The next display number, in the parser's allocation order.
    next: u32,
    /// How deeply the walk is nested — see [`MAX_EMIT_NESTING`].
    depth: u32,
}

impl Writer {
    fn new(f: &BlueprintFn) -> Result<Self, EmitError> {
        let mut live = Vec::new();
        let mut params = Vec::new();
        for p in &f.params {
            params.push(ident(&p.name)?.to_string());
        }
        live.push(params);
        Ok(Self {
            idents: HashMap::new(),
            live,
            next: 0,
            depth: 0,
        })
    }

    /// Spend one level of the emitter's own bound, or refuse.
    ///
    /// The parser is bounded ([`crate::parse::MAX_NESTING`]) so text can never
    /// build a tree deeper than this; a *graph* can, because a canvas will chain
    /// as many `math.add` nodes as somebody drags out, and lowering turns that
    /// chain into a left-deep `Expr`. Unbounded, printing one is a stack
    /// overflow in whatever process opened the Blueprint as text.
    fn deeper(&mut self) -> Result<(), EmitError> {
        self.depth += 1;
        if self.depth > MAX_EMIT_NESTING {
            return Err(EmitError::TooDeep);
        }
        Ok(())
    }

    /// Claim the next display number for a local the text does not name (a
    /// loop's guard counter, a `for`'s snapshotted bound), so the numbering
    /// stays in step with the parser's.
    fn skip_hidden(&mut self) {
        self.next += 1;
    }

    /// The written identifier for a binder, taking the next display number.
    fn display(&mut self, binding: &inf_blueprint::Binding) -> String {
        let n = LocalId(self.next);
        self.next += 1;
        BlueprintFn::local_ident(binding, n)
    }

    fn is_live(&self, name: &str) -> bool {
        self.live.iter().any(|s| s.iter().any(|n| n == name))
    }

    fn bind(&mut self, id: LocalId, name: String) -> Result<(), EmitError> {
        if self.is_live(&name) {
            return Err(EmitError::ShadowedBinder(name));
        }
        self.live
            .last_mut()
            .expect("a scope is always open")
            .push(name.clone());
        self.idents.insert(id, name);
        Ok(())
    }

    fn scoped<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, EmitError>,
    ) -> Result<T, EmitError> {
        self.live.push(Vec::new());
        let r = f(self);
        self.live.pop();
        r
    }

    /// A statement list at `depth`, each line already indented and terminated.
    fn block(&mut self, body: &[Stmt], depth: usize) -> Result<String, EmitError> {
        self.deeper()?;
        let r = self.block_inner(body, depth);
        self.depth -= 1;
        r
    }

    fn block_inner(&mut self, body: &[Stmt], depth: usize) -> Result<String, EmitError> {
        let mut out = String::new();
        let mut i = 0;
        while i < body.len() {
            // `for` before `while`: a `for` expansion satisfies the while matcher
            // from its third statement (`loopshape`'s own arm pins that).
            if let Some(m) = loopshape::match_for(body, i) {
                let (index, first, last, inner, consumed) = (
                    m.index,
                    m.first.clone(),
                    m.last.clone(),
                    m.body.to_vec(),
                    m.consumed,
                );
                let (first, last) = (self.expr(&first, 0)?, self.expr(&last, 0)?);
                // A graph's index is `Anon` and prints `n1`; a script's keeps
                // the name the author gave it. The bound and the counter follow
                // in the parser's order even though neither is written.
                let name = {
                    let n = self.display(m.index_binding);
                    ident(&n)?.to_string()
                };
                self.skip_hidden();
                self.skip_hidden();
                let inner = self.scoped(|w| {
                    w.bind(index, name.clone())?;
                    w.block(&inner, depth + 1)
                })?;
                let pad = indent(depth);
                let _ = write!(
                    out,
                    "{pad}for {name} = {first}, {last} do\n{inner}{pad}end\n"
                );
                i += consumed;
                continue;
            }
            if let Some(m) = loopshape::match_while(body, i) {
                let (cond, inner, consumed) = (m.cond.clone(), m.body.to_vec(), m.consumed);
                let cond = self.expr(&cond, 0)?;
                // The guard counter is a local the text does not name.
                self.skip_hidden();
                let inner = self.scoped(|w| w.block(&inner, depth + 1))?;
                let pad = indent(depth);
                let _ = write!(out, "{pad}while {cond} do\n{inner}{pad}end\n");
                i += consumed;
                continue;
            }
            out.push_str(&self.stmt(&body[i], depth)?);
            i += 1;
        }
        Ok(out)
    }

    fn stmt(&mut self, s: &Stmt, depth: usize) -> Result<String, EmitError> {
        let pad = indent(depth);
        Ok(match s {
            Stmt::Let {
                id,
                binding,
                ty,
                value,
                ..
            } => {
                // The value is written *before* the binder enters scope, so
                // `local x = x` reads the member variable on the right, exactly
                // as the IR says.
                let value = self.expr(value, 0)?;
                let name = {
                    let n = self.display(binding);
                    ident(&n)?.to_string()
                };
                self.bind(*id, name.clone())?;
                let ty = match ty {
                    Some(t) => format!(": {}", ty_name(*t)),
                    None => String::new(),
                };
                format!("{pad}local {name}{ty} = {value}\n")
            }
            Stmt::Assign { target, value } => {
                let name = self.local(*target)?;
                format!("{pad}{name} = {}\n", self.expr(value, 0)?)
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let mut out = format!("{pad}if {} then\n", self.expr(cond, 0)?);
                out.push_str(&self.scoped(|w| w.block(then_body, depth + 1))?);
                out.push_str(&self.else_tail(else_body, depth)?);
                out
            }
            // A `While` that is not the guarded shape — the block walker tried
            // both matchers first. It is legal IR (a lift of hand-written Rust),
            // it runs, and it does not raise; written here as the loop it is.
            Stmt::While { cond, body } => {
                let cond = self.expr(cond, 0)?;
                let inner = self.scoped(|w| w.block(body, depth + 1))?;
                format!("{pad}while {cond} do\n{inner}{pad}end\n")
            }
            Stmt::Return(None) => format!("{pad}return\n"),
            Stmt::Return(Some(e)) => format!("{pad}return {}\n", self.expr(e, 0)?),
            Stmt::ExprStmt(Expr::Call { path, args }) if path == &vars_path("set") => {
                let (name, value) = var_set_parts(args)?;
                let value = self.expr(value, 0)?;
                if is_ident(name) && !self.is_live(name) {
                    format!("{pad}{name} = {value}\n")
                } else {
                    format!("{pad}var.set({}, {value})\n", quote(name))
                }
            }
            // **A statement position holds a call and nothing else.**
            // InfiniScript has no evaluate-and-discard statement — `1 + 2` on a
            // line of its own is a refusal in the grammar — and the two calls
            // that print as something *other* than a call (`vars::get`, which
            // prints as a bare name, and `nodestate::get_or`, whose namespace
            // resolves only in value position) cannot be written either. The
            // SCRIPT1a audit found all four printing happily and re-parsing not
            // at all. `raise` names the same shape (`UnsupportedStmt("non-call
            // expr stmt")`), so this is a state both faces agree they cannot
            // hold, rather than a new limit.
            Stmt::ExprStmt(e) => match e {
                Expr::Call { path, .. }
                    if *path != vars_path("get") && path.as_slice() != nodestate_get_or() =>
                {
                    format!("{pad}{}\n", self.expr(e, 0)?)
                }
                Expr::Call { path, .. } => {
                    return Err(EmitError::UnspellableStatement(path.join("::")))
                }
                _ => return Err(EmitError::UnspellableStatement(describe_expr(e).into())),
            },
            Stmt::Snippet(code) => {
                let (open, close) = long_brackets(code);
                format!("{pad}rust {open}\n{code}{close}\n")
            }
        })
    }

    fn else_tail(&mut self, else_body: &[Stmt], depth: usize) -> Result<String, EmitError> {
        let pad = indent(depth);
        // An `else` holding exactly one `if` is an `elseif`, which is what the
        // parser builds and so what round-trips.
        if let [Stmt::If {
            cond,
            then_body,
            else_body: inner,
        }] = else_body
        {
            let mut out = format!("{pad}elseif {} then\n", self.expr(cond, 0)?);
            out.push_str(&self.scoped(|w| w.block(then_body, depth + 1))?);
            out.push_str(&self.else_tail(inner, depth)?);
            return Ok(out);
        }
        if else_body.is_empty() {
            return Ok(format!("{pad}end\n"));
        }
        let mut out = format!("{pad}else\n");
        out.push_str(&self.scoped(|w| w.block(else_body, depth + 1))?);
        out.push_str(&format!("{pad}end\n"));
        Ok(out)
    }

    fn local(&self, id: LocalId) -> Result<String, EmitError> {
        self.idents
            .get(&id)
            .cloned()
            .ok_or(EmitError::UnboundLocal(id.0))
    }

    /// An expression, parenthesised when its precedence is below `min`.
    fn expr(&mut self, e: &Expr, min: u8) -> Result<String, EmitError> {
        self.deeper()?;
        let r = self.expr_inner(e, min);
        self.depth -= 1;
        r
    }

    fn expr_inner(&mut self, e: &Expr, min: u8) -> Result<String, EmitError> {
        let (text, prec) = match e {
            Expr::Lit(l) => (literal(l)?, ATOM),
            Expr::Param(name) => (ident(name)?.to_string(), ATOM),
            Expr::Local(id) => (self.local(*id)?, ATOM),
            Expr::Call { path, args } if path == &vars_path("get") => {
                let name = var_get_name(args)?;
                if is_ident(name) && !self.is_live(name) {
                    (name.to_string(), ATOM)
                } else {
                    (format!("var.get({})", quote(name)), ATOM)
                }
            }
            Expr::Call { path, args } => {
                let mut written = Vec::new();
                for a in args {
                    written.push(self.expr(a, 0)?);
                }
                (
                    format!("{}({})", call_spelling(path)?, written.join(", ")),
                    ATOM,
                )
            }
            Expr::Unary(UnOp::Not, inner) => (format!("not {}", self.expr(inner, UNARY)?), UNARY),
            // **A negated literal is written as the negative literal.** The
            // canonical IR for a negative constant is `Lit`, not `Unary(Neg,
            // Lit)` — `inf_transpile::emit` refuses the latter outright — and
            // `crate::parse` folds it on the way in, so printing the canonical
            // form is what makes the text a fixed point over an IR a *graph*
            // produced (a `math.neg` node wired to a `lit.float` is exactly
            // that shape, and it is why the graph path still has the gap).
            Expr::Unary(UnOp::Neg, inner) => match inner.as_ref() {
                Expr::Lit(Lit::Float(f)) => (literal(&Lit::Float(-f))?, ATOM),
                Expr::Lit(Lit::Int(i)) if i.checked_neg().is_some() => {
                    (literal(&Lit::Int(-i))?, ATOM)
                }
                _ => {
                    // A nested unary must be parenthesised or the two minus
                    // signs open a comment; `-i64::MIN` keeps the unary form
                    // and so does anything non-literal that binds looser.
                    let force = matches!(inner.as_ref(), Expr::Lit(_) | Expr::Unary(..));
                    let written = self.expr(inner, UNARY)?;
                    let written = if force && !written.starts_with('(') {
                        format!("({written})")
                    } else {
                        written
                    };
                    (format!("-{written}"), UNARY)
                }
            },
            Expr::Binary(op, l, r) => {
                let p = prec(*op);
                // Left-associative operators let their *left* operand sit at
                // their own level and force the right one up. A **comparison
                // does not associate at all** — the grammar refuses `a < b == c`
                // outright — so a comparison nested in *either* operand of a
                // comparison has to be parenthesised, the left included.
                //
                // The SCRIPT1a audit found the left half missing: `cmp.lt` wired
                // into `cmp.eq`'s `a` input is a graph anybody can draw, it
                // lowers to `Binary(Eq, Binary(Lt, …), …)`, and the emitter
                // printed `1.0 < 2.0 == true` — text the parser then refuses.
                // An emitter that writes a file its own parser rejects is worse
                // than one that refuses, because the refusal is silent until
                // somebody tries to open the result.
                let left_min = if associates(*op) { p } else { p + 1 };
                let lhs = self.expr(l, left_min)?;
                let rhs = self.expr(r, p + 1)?;
                (format!("{lhs} {} {rhs}", spelling(*op)), p)
            }
        };
        Ok(if prec < min {
            format!("({text})")
        } else {
            text
        })
    }
}

const ATOM: u8 = 7;
const UNARY: u8 = 6;

/// How deeply the emitter will walk before refusing.
///
/// Twice [`crate::parse::MAX_NESTING`], deliberately: the parser counts a
/// parenthesis and a chain step where the emitter counts a tree node, and at
/// most one uncounted comparison node can sit inside each counted level — so
/// twice is a proof rather than a guess that **anything the parser accepts, the
/// emitter can write**. The bound exists for the other producer: a graph.
pub const MAX_EMIT_NESTING: u32 = crate::parse::MAX_NESTING * 2;

/// Whether an operator associates, i.e. whether `a op b op c` is a legal
/// spelling of `(a op b) op c`. The comparisons do not — [`crate::parse`]
/// refuses a chained one — so they need a parenthesis on both sides.
fn associates(op: BinOp) -> bool {
    !matches!(
        op,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
    )
}

fn prec(op: BinOp) -> u8 {
    match op {
        BinOp::Or => 1,
        BinOp::And => 2,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 3,
        BinOp::Add | BinOp::Sub => 4,
        BinOp::Mul | BinOp::Div | BinOp::Rem => 5,
    }
}

fn spelling(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Eq => "==",
        BinOp::Ne => "~=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::And => "and",
        BinOp::Or => "or",
    }
}

fn vars_path(op: &str) -> Vec<String> {
    vec!["vars".to_string(), op.to_string()]
}

/// The lowerer's own state-cell read, which has a value spelling and no
/// statement one.
fn nodestate_get_or() -> [String; 2] {
    ["nodestate".to_string(), "get_or".to_string()]
}

/// What an expression is, for the one message that has to name a shape rather
/// than a path.
fn describe_expr(e: &Expr) -> &'static str {
    match e {
        Expr::Lit(_) => "a literal",
        Expr::Param(_) => "a parameter",
        Expr::Local(_) => "a local",
        Expr::Unary(..) => "a unary expression",
        Expr::Binary(..) => "a binary expression",
        Expr::Call { .. } => "a call",
    }
}

fn var_get_name(args: &[Expr]) -> Result<&str, EmitError> {
    match args {
        [Expr::Lit(Lit::Str(n))] => Ok(n),
        _ => Err(EmitError::UnspellableCall("vars::get".into())),
    }
}

fn var_set_parts(args: &[Expr]) -> Result<(&str, &Expr), EmitError> {
    match args {
        [Expr::Lit(Lit::Str(n)), value] => Ok((n, value)),
        _ => Err(EmitError::UnspellableCall("vars::set".into())),
    }
}

/// The text spelling of an IR call path.
///
/// Two segments are a node's `type_id` (with the three `dispatch.*` renames
/// undone); three are a multi-result query naming its result. `nodestate::*` has
/// no node and keeps its own name — the rule being that text uses the node id
/// where a node exists and the host path where none does.
fn call_spelling(path: &[String]) -> Result<String, EmitError> {
    match path.len() {
        2 => Ok(node_type_of_path(path)),
        3 => Ok(format!("{}.{}", node_type_of_path(&path[..2]), path[2])),
        _ => Err(EmitError::UnspellableCall(path.join("::"))),
    }
}

fn indent(depth: usize) -> String {
    "    ".repeat(depth)
}

/// A literal's written form. Floats use `{:?}`, which is the **shortest
/// representation that parses back to the same bits** — the property the round
/// trip needs, and the reason the `serde_json` `float_roundtrip` pin exists on
/// the other face of the same IR.
fn literal(l: &Lit) -> Result<String, EmitError> {
    Ok(match l {
        Lit::Int(i) => i.to_string(),
        Lit::Float(f) if f.is_finite() => {
            let s = format!("{f:?}");
            // `{:?}` writes `1e300`-style exponents without a decimal point and
            // integral values with one, both of which the lexer reads back.
            s
        }
        Lit::Float(f) => return Err(EmitError::NonFiniteFloat(*f)),
        Lit::Bool(b) => b.to_string(),
        Lit::Str(s) => quote(s),
    })
}

/// A `"…"` literal that survives the lexer: backslash and quote escaped,
/// every control character written as `\u{…}`, everything else verbatim (so a
/// script's own language stays readable).
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\u{{{:x}}}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && !crate::lex::KEYWORDS.contains(&s)
        && s.chars()
            .next()
            .is_some_and(|c| c == '_' || c.is_alphabetic())
        && s.chars().all(|c| c == '_' || c.is_alphanumeric())
}

fn ident(s: &str) -> Result<&str, EmitError> {
    if is_ident(s) {
        Ok(s)
    } else {
        Err(EmitError::UnspellableName(s.to_string()))
    }
}

/// The shortest long-bracket level whose closing delimiter the content does not
/// contain — Lua's own answer, and what makes `rust [[ … ]]` total over opaque
/// Rust (`v[[1, 2][0]]` is a real expression and it contains `]]`).
fn long_brackets(content: &str) -> (String, String) {
    for level in 0..64 {
        let close: String = std::iter::once(']')
            .chain(std::iter::repeat_n('=', level))
            .chain(std::iter::once(']'))
            .collect();
        if !content.contains(&close) {
            let open: String = std::iter::once('[')
                .chain(std::iter::repeat_n('=', level))
                .chain(std::iter::once('['))
                .collect();
            return (open, close);
        }
    }
    // 64 nested levels of `]=…=]` is not a program; fall back to the widest
    // rather than looping, and let the round-trip arm catch it if it ever ships.
    (
        "[".to_string() + &"=".repeat(64) + "[",
        "]".to_string() + &"=".repeat(64) + "]",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_brackets_step_up_past_their_content() {
        assert_eq!(long_brackets("plain"), ("[[".into(), "]]".into()));
        assert_eq!(long_brackets("v[[1,2][0]]"), ("[=[".into(), "]=]".into()));
        assert_eq!(long_brackets("a]] b]=]"), ("[==[".into(), "]==]".into()));
    }

    #[test]
    fn a_float_prints_its_shortest_round_tripping_form() {
        assert_eq!(
            literal(&Lit::Float(0.1 + 0.2)).unwrap(),
            "0.30000000000000004"
        );
        assert_eq!(literal(&Lit::Float(2.0)).unwrap(), "2.0");
        assert_eq!(
            literal(&Lit::Float(f64::MIN_POSITIVE)).unwrap(),
            "2.2250738585072014e-308"
        );
        // `NonFiniteFloat(NaN) == NonFiniteFloat(NaN)` is false — NaN is not
        // equal to itself — so the variant is matched rather than compared.
        assert!(matches!(
            literal(&Lit::Float(f64::NAN)),
            Err(EmitError::NonFiniteFloat(f)) if f.is_nan()
        ));
        assert_eq!(
            literal(&Lit::Float(f64::INFINITY)).unwrap_err(),
            EmitError::NonFiniteFloat(f64::INFINITY)
        );
    }

    #[test]
    fn a_keyword_is_not_an_identifier() {
        assert!(!is_ident("end"));
        assert!(!is_ident("2fast"));
        assert!(!is_ident(""));
        assert!(is_ident("_x9"));
    }
}
