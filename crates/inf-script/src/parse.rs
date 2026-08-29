//! The `.infini` parser: tokens → the `BlueprintFn` IR, in one pass.
//!
//! # There is no separate AST, and that is the thesis
//!
//! InfiniScript is not a language that *compiles to* the Blueprint IR; it is a
//! **text face on it**. So the parser's output type is `BlueprintFn` itself —
//! `Stmt` and `Expr` are the syntax tree. Nothing sits between the two that
//! could acquire semantics of its own, which is the whole reason this arc adds a
//! parser rather than a fourth execution model.
//!
//! # Names
//!
//! A bare identifier resolves in one order, innermost first:
//!
//! 1. a `local` in an enclosing block → [`Expr::Local`];
//! 2. a handler/function parameter → [`Expr::Param`];
//! 3. **anything else is a member variable** → `vars::get("name")`.
//!
//! Rule 3 is deliberately total rather than a lookup: it is what lets a single
//! handler be parsed on its own (the graph↔text bridge opens one function, with
//! no `var` declarations in sight) and still mean the same thing it means inside
//! its class. When the whole unit *is* in view, an undeclared name is reported
//! as a **warning** rather than an error, because the runtime already refuses it
//! by name — `vars::get` on an unknown variable is a clean `RunError`, which is
//! P21's law: a gameplay refusal is a value.
//!
//! # Mutability is derived, not declared
//!
//! There is no `local mut`. A local is `mutable` in the IR exactly when
//! something assigns to it, computed in a pass over the finished body. That is
//! what makes `parse(emit(f)) == f` hold for loop counters (which the lowerer
//! marks mutable and does assign) without a keyword no reader would want.
//!
//! # Local ids come from the walk, never from the text
//!
//! `n7` and `speed_n7` are how [`Binding::Anon`] and [`Binding::Named`] *print*;
//! the digits are display, and the parser assigns ids from its own counter as it
//! walks. That is what makes the round trip a property of the walk rather than
//! of an encoding, and it is why the emitter's output re-parses to the same ids.
//!
//! # Nesting is bounded, because a recursive-descent parser's real failure mode
//! is the stack
//!
//! This is a **recursive** descent parser and the IR it builds is a tree, so
//! both the parse and everything downstream (the emitter, the interpreter, the
//! transpiler, and `Drop` itself) recurse once per level. Without a bound,
//! `((((…1…))))` is not a refusal — it is `STATUS_STACK_OVERFLOW`, which takes
//! the *editor* rather than the script, and a parser is exactly the component an
//! editor calls on every keystroke. P19 paid for this law once already in the
//! PCG grammar's parser; the SCRIPT1a audit measured it here: on a **1 MiB**
//! stack (what a Windows main thread gets) the parser died somewhere between
//! 512 and 768 nested parentheses, and it crashed rather than refused.
//!
//! So [`MAX_NESTING`] bounds the total nesting of a declaration — blocks and
//! expressions share one budget, because they share one stack — and going past
//! it is a diagnostic with a line, a column and a remedy. The deepest
//! construction in the whole round-trip corpus is 4 levels.
//!
//! **A chain counts.** `1 + 1 + 1 + …` is parsed by a loop and costs no parser
//! frames at all, but it builds a *left-deep tree*, and the emitter recursed
//! itself to death on ten thousand of them. So each accumulation in the
//! `or`/`and`/`+`/`*` chains spends one level of the same budget: the bound is on
//! the tree, not on the parser's own stack.

use std::collections::{BTreeMap, HashMap, HashSet};

use inf_blueprint::loopshape;
use inf_blueprint::lower::sanitize_ident;
use inf_blueprint::semantics::{EventBinding, EventKind, Variable};
use inf_blueprint::{
    BinOp, Binding, BlueprintClass, BlueprintFn, Expr, Lit, LocalId, Param, Stmt, Ty, UnOp,
};

use crate::lex::{lex, Span, Spanned, Tok};
use crate::verbs::Verbs;
use crate::{Diagnostic, Severity};

/// One parsed `.infini` file, before it is given an asset id.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Unit {
    /// The `actor "…"` header's display name, if the file carries one.
    pub name: Option<String>,
    pub variables: Vec<Variable>,
    pub events: Vec<EventBinding>,
    pub functions: Vec<BlueprintFn>,
    /// Where each handler/function begins, by its `BlueprintFn::id`.
    ///
    /// The source map SCRIPT1a ships: enough for a failing handler to be pointed
    /// at in the editor, which is the granularity failure containment actually
    /// has (`run_on_guid` aborts *a handler, on an actor, for a tick*). A
    /// statement-level map is SCRIPT2's, when a panel consumes one.
    pub spans: BTreeMap<String, Span>,
}

impl Unit {
    /// Give the unit an identity and make it an actor class.
    pub fn into_class(self, id: impl Into<String>) -> BlueprintClass {
        let id = id.into();
        let name = self.name.unwrap_or_else(|| id.clone());
        let mut class = BlueprintClass::new(id, name);
        class.variables = self.variables;
        class.events = self.events;
        class.functions = self.functions;
        class
    }
}

/// Unwinding marker: a diagnostic has already been recorded.
struct Bail;
type P<T> = Result<T, Bail>;

/// How deeply one declaration may nest — blocks, parentheses, unary operators
/// and operator-chain steps together, because all four cost the same stack.
///
/// Chosen against a **measurement**, not a feeling: on a 1 MiB stack (a Windows
/// main thread) the unguarded parser overflowed between 512 and 768 nested
/// parentheses, which is the tightest shape (eight frames per level through
/// `expr → or → and → cmp → add → mul → unary → primary`). 128 leaves at least
/// a fourfold margin there and a much larger one everywhere else, and it is two
/// orders of magnitude past anything a person writes — the deepest construction
/// in the round-trip corpus is **4**.
pub const MAX_NESTING: u32 = 128;

/// Parse a whole `.infini` file.
///
/// Returns the unit plus every diagnostic — including warnings, which do not
/// prevent a unit from being produced. `Err` carries the diagnostics of a file
/// that could not be parsed at all.
pub fn parse_unit(source: &str) -> Result<(Unit, Vec<Diagnostic>), Vec<Diagnostic>> {
    let toks = match lex(source) {
        Ok(t) => t,
        Err(e) => {
            return Err(vec![Diagnostic {
                severity: Severity::Error,
                span: e.span,
                message: e.message,
            }])
        }
    };
    let declared = prescan_variables(&toks);
    let functions = prescan_functions(&toks);
    let mut p = Parser::new(&toks, declared, functions);
    let unit = p.unit();
    let mut diags = std::mem::take(&mut p.diags);
    check_call_graph(&unit, &mut diags);
    if diags.iter().any(|d| d.severity == Severity::Error) {
        return Err(diags);
    }
    Ok((unit, diags))
}

/// **The call graph must fit inside the interpreter's budget**, and both halves
/// of that sentence are refusals made here rather than at run time — because the
/// two faces of one program would otherwise disagree about what a call chain
/// *does*.
///
/// The interpreter bounds a chain at
/// [`MAX_CALL_DEPTH`](inf_blueprint::interp::MAX_CALL_DEPTH) and answers with a
/// `RunError` — a value, per P21. The **transpiled** Rust has no such bound:
/// `generate_fn` renders a one-segment call as an ordinary Rust call. So there
/// are two ways for a legal-looking program to run in one face and be refused by
/// the other, and this function closes both:
///
/// 1. **A cycle** — refused with the route it found, because a recursive Rust
///    call overflows the stack of whatever is running it and an interpreted one
///    answers with a value.
/// 2. **A chain longer than the budget** — refused with its length, because the
///    SCRIPT2a audit measured the hole the first half alone leaves: a unit of 65
///    functions each calling the next has **no cycle**, so the cycle check
///    passes, and then the interpreter refuses at 64 while `rustc` compiles the
///    same program and it prints `65`. One program, two answers. (The audit ran
///    both faces: 64 agrees on both sides, 65 diverges.) Nothing bounds how many
///    `function`s a unit may declare, so the parser's own `MAX_NESTING` — which
///    is a bound on one declaration's *tree* — never sees it.
///
/// Together they make [`MAX_CALL_DEPTH`]'s own doc comment true: **a `.infini`
/// file cannot reach that bound**, and the budget is left as the defence for IR
/// that did not come through this parser (a hand-edited `.inf_act`, a lift of
/// hand-written Rust).
///
/// A `while` loop is the spelling that survives both faces — it lowers to the
/// counter-guarded expansion, so the bound lives in the IR and the interpreter
/// and the compiled program share it (A.7).
///
/// The walk is over declaration order and, inside a body, over statement order,
/// so the route a file reports is a function of the file rather than of a hash.
fn check_call_graph(unit: &Unit, diags: &mut Vec<Diagnostic>) {
    let names: Vec<&str> = unit.functions.iter().map(|f| f.name.as_str()).collect();
    // **Resolved once, into indices.** Both walks below used to look a callee up
    // by scanning `names`, and `collect_local_calls` decided "is this one of
    // ours" the same way — three linear scans nested inside two walks, which the
    // SCRIPT2a audit measured as **cubic**: 2 000 chained functions took 2.0 s to
    // refuse and 10 000 took **359 s**, in release, on a 488 KiB file — well
    // inside `source::MAX_SOURCE_BYTES`. The watcher calls this on every save and
    // the cook calls it on every build, so that is a parse that hangs an editor.
    // A map plus `Vec<Vec<usize>>` edges makes the whole thing linear.
    let mut index: HashMap<&str, usize> = HashMap::with_capacity(names.len());
    for (i, n) in names.iter().enumerate() {
        // First wins, which is what a linear `position` scan did.
        index.entry(*n).or_insert(i);
    }
    let edges: Vec<Vec<usize>> = unit
        .functions
        .iter()
        .map(|f| {
            let mut out: Vec<usize> = Vec::new();
            let mut seen: HashSet<usize> = HashSet::new();
            collect_local_calls(&f.body, &index, &mut seen, &mut out);
            out
        })
        .collect();
    let before = diags.len();
    check_no_recursion(unit, &names, &edges, diags);
    // A cycle makes "how deep is the deepest chain" meaningless — and the walk
    // below assumes an acyclic graph — so the depth check runs only when the
    // graph is one.
    if diags.len() == before {
        check_call_depth(unit, &names, &edges, diags);
    }
}

/// **No `function` may reach itself**, directly or through another.
///
/// Split out of [`check_call_graph`] so the two refusals share one call graph
/// and one walk order.
///
/// A **white/grey/black** depth-first colouring: a `Grey` node is on the current
/// route, so an edge into one is a back edge and the route from it is the cycle.
/// `Black` is finished and never re-entered, which is what makes the whole sweep
/// `O(V + E)` — the first version restarted a fresh walk, with a fresh `seen`
/// buffer, from every function, and the audit measured what that cost (see
/// [`check_call_graph`]).
///
/// The reported cycle is unchanged by the rewrite, and that is checked rather
/// than assumed: roots are taken in **declaration order** and edges in statement
/// order, so `down → down`, `ping → pong → ping`, `a → b → c → a` and the
/// downstream `b → c → b` all still name the function they named before.
fn check_no_recursion(
    unit: &Unit,
    names: &[&str],
    edges: &[Vec<usize>],
    diags: &mut Vec<Diagnostic>,
) {
    #[derive(Clone, Copy, PartialEq)]
    enum Colour {
        White,
        Grey,
        Black,
    }
    let mut colour = vec![Colour::White; names.len()];
    // The grey nodes, in route order. `route.last()` is always the node whose
    // edges are being walked.
    let mut route: Vec<usize> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for root in 0..names.len() {
        if colour[root] != Colour::White {
            continue;
        }
        colour[root] = Colour::Grey;
        route.push(root);
        stack.push((root, 0));
        while let Some((node, next)) = stack.pop() {
            let Some(&to) = edges[node].get(next) else {
                colour[node] = Colour::Black;
                route.pop();
                continue;
            };
            stack.push((node, next + 1));
            match colour[to] {
                Colour::Grey => {
                    let at = route
                        .iter()
                        .position(|n| *n == to)
                        .expect("a grey node is on the route");
                    let mut chain: Vec<&str> = route[at..].iter().map(|i| names[*i]).collect();
                    chain.push(names[to]);
                    let span = unit
                        .functions
                        .get(to)
                        .and_then(|f| unit.spans.get(&f.id))
                        .copied()
                        .unwrap_or_default();
                    diags.push(Diagnostic {
                        severity: Severity::Error,
                        span,
                        message: format!(
                            "`{}` calls itself ({}) — InfiniScript has no recursion, because \
                             the interpreter bounds a call chain and the Rust the cook \
                             generates does not, so the two would not agree. Write the \
                             repetition as a `while` or `for` loop",
                            names[to],
                            chain.join(" → ")
                        ),
                    });
                    return;
                }
                Colour::Black => {}
                Colour::White => {
                    colour[to] = Colour::Grey;
                    route.push(to);
                    stack.push((to, 0));
                }
            }
        }
    }
}

/// **No call chain may be longer than the interpreter's budget** — the second
/// half of [`check_call_graph`], and the SCRIPT2a audit's finding.
///
/// `depth(f)` is the number of calls in the longest chain that *starts with a
/// call to `f`*, so a leaf is 1 and a chain of `n` is `n`. The interpreter
/// spends one unit of [`MAX_CALL_DEPTH`] per call and a handler is not a call,
/// so a chain of exactly `MAX_CALL_DEPTH` runs and one more does not — measured
/// on both faces, not inferred.
///
/// Every declared function is checked rather than only the ones a handler can
/// reach: an unreachable over-deep chain is dead code either way, and "the
/// parser accepted it" must not depend on which handler happens to exist today.
///
/// The graph is acyclic here (the caller guarantees it), so the memo terminates.
fn check_call_depth(
    unit: &Unit,
    names: &[&str],
    edges: &[Vec<usize>],
    diags: &mut Vec<Diagnostic>,
) {
    // `depth[i]`, plus the callee that achieves it, so the refusal can print the
    // route rather than only a number. Memoized, so the whole sweep is
    // `O(V + E)` however many roots it starts from.
    let mut depth: Vec<Option<(u32, Option<usize>)>> = vec![None; names.len()];
    for start in 0..names.len() {
        // Iterative post-order, so a unit with thousands of functions cannot
        // overflow the stack of the process doing the checking — which is the
        // whole class of defect this function exists to close.
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        while let Some((node, next)) = stack.pop() {
            if depth[node].is_some() {
                continue;
            }
            match edges[node].get(next) {
                Some(&to) if depth[to].is_none() => {
                    stack.push((node, next));
                    stack.push((to, 0));
                }
                Some(_) => stack.push((node, next + 1)),
                None => {
                    // Every callee is resolved: fold them.
                    let mut best: (u32, Option<usize>) = (0, None);
                    for &to in &edges[node] {
                        let d = depth[to].expect("callee resolved first").0;
                        if d > best.0 {
                            best = (d, Some(to));
                        }
                    }
                    depth[node] = Some((best.0 + 1, best.1));
                }
            }
        }
    }

    let budget = inf_blueprint::interp::MAX_CALL_DEPTH;
    for (start, f) in unit.functions.iter().enumerate() {
        let (d, _) = depth[start].expect("every function resolved");
        if d <= budget {
            continue;
        }
        let mut route: Vec<&str> = vec![names[start]];
        let mut at = depth[start].and_then(|(_, n)| n);
        while let Some(i) = at {
            route.push(names[i]);
            at = depth[i].and_then(|(_, n)| n);
        }
        let shown = if route.len() > 6 {
            format!(
                "{} → … → {}",
                route[..3].join(" → "),
                route[route.len() - 2..].join(" → ")
            )
        } else {
            route.join(" → ")
        };
        diags.push(Diagnostic {
            severity: Severity::Error,
            span: unit.spans.get(&f.id).copied().unwrap_or_default(),
            message: format!(
                "calling `{}` starts a chain of {d} calls ({shown}), and InfiniScript \
                 allows {budget} — the interpreter refuses past that and the Rust the \
                 cook generates does not, so the two would not agree. Flatten the chain, \
                 or write the repetition as a `while` or `for` loop",
                names[start]
            ),
        });
        return;
    }
}

/// Every one-segment call in `body` that names a declared `function`, resolved
/// to its index, in walk order and without repeats.
///
/// `known` and `seen` are both maps rather than scans: this runs once per call
/// expression in the unit, and doing either by walking a `Vec` is one of the
/// three linear scans the SCRIPT2a audit measured as a cubic parse.
fn collect_local_calls(
    body: &[Stmt],
    known: &HashMap<&str, usize>,
    seen: &mut HashSet<usize>,
    out: &mut Vec<usize>,
) {
    fn expr(
        e: &Expr,
        known: &HashMap<&str, usize>,
        seen: &mut HashSet<usize>,
        out: &mut Vec<usize>,
    ) {
        match e {
            Expr::Call { path, args } => {
                if let [name] = path.as_slice() {
                    if let Some(&i) = known.get(name.as_str()) {
                        if seen.insert(i) {
                            out.push(i);
                        }
                    }
                }
                for a in args {
                    expr(a, known, seen, out);
                }
            }
            Expr::Unary(_, a) => expr(a, known, seen, out),
            Expr::Binary(_, a, b) => {
                expr(a, known, seen, out);
                expr(b, known, seen, out);
            }
            Expr::Lit(_) | Expr::Param(_) | Expr::Local(_) => {}
        }
    }
    for s in body {
        match s {
            Stmt::Let { value, .. } | Stmt::Assign { value, .. } | Stmt::ExprStmt(value) => {
                expr(value, known, seen, out)
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                expr(cond, known, seen, out);
                collect_local_calls(then_body, known, seen, out);
                collect_local_calls(else_body, known, seen, out);
            }
            Stmt::While { cond, body } => {
                expr(cond, known, seen, out);
                collect_local_calls(body, known, seen, out);
            }
            Stmt::Return(Some(e)) => expr(e, known, seen, out),
            Stmt::Return(None) | Stmt::Snippet(_) => {}
        }
    }
}

/// Parse a single handler or function on its own — the Ring-0 half of the
/// graph↔text bridge, where one `.inf_act` function is opened as text.
///
/// Bare names are member variables here without complaint: a lone function has
/// no declarations in view, so there is nothing to check against and nothing to
/// warn about.
pub fn parse_fn(source: &str) -> Result<BlueprintFn, Vec<Diagnostic>> {
    let (unit, _) = parse_unit(source)?;
    let mut fns: Vec<BlueprintFn> = unit
        .events
        .into_iter()
        .map(|b| b.body)
        .chain(unit.functions)
        .collect();
    match fns.len() {
        1 => Ok(fns.remove(0)),
        0 => Err(vec![Diagnostic {
            severity: Severity::Error,
            span: Span::default(),
            message: "no handler or function in this source — expected one \
                      `on <event>(…) … end` or `function <name>(…) … end`"
                .into(),
        }]),
        n => Err(vec![Diagnostic {
            severity: Severity::Error,
            span: Span::default(),
            message: format!("{n} handlers in a source that must hold exactly one"),
        }]),
    }
}

/// The `var` names a file declares, found before parsing so that a handler
/// written above its declarations still resolves against them.
fn prescan_variables(toks: &[Spanned]) -> Vec<String> {
    let mut out = Vec::new();
    for w in toks.windows(2) {
        if w[0].tok == Tok::Ident("var".into()) {
            if let Tok::Ident(name) = &w[1].tok {
                out.push(name.clone());
            }
        }
    }
    out
}

/// A `function` declaration's header, learned before any body is parsed.
///
/// Only the header: a call needs the name, the argument count and whether there
/// is a value to use — never the body. Parameter *types* are deliberately not
/// carried, because nothing in this parser infers an expression's type and a
/// check it cannot make is a check it must not claim.
#[derive(Debug, Clone, PartialEq)]
struct LocalFn {
    name: String,
    /// Parameter names, in declaration order — argument order, and what a
    /// missing-argument refusal names.
    params: Vec<String>,
    /// The declared return type; `Unit` when the header carries no `->`.
    ret: Ty,
}

/// The `function` **headers** a file declares, found before parsing so that a
/// handler written above its declarations can still call them.
///
/// SCRIPT1a scanned for the names alone, for the opposite reason: the IR had no
/// user-function call form, so the prescan existed to make the *refusal* name
/// the real bound rather than send a designer hunting for a namespace. SCRIPT2
/// adds the form (a one-segment [`Expr::Call`]), so the same prescan now
/// resolves the call instead of explaining it — and it has to read the whole
/// header, because arity and "does this produce a value" are checked here.
///
/// It is a token scan rather than a pre-parse, and it is deliberately lenient:
/// a malformed header contributes what it can and the real parser produces the
/// diagnostic. Nothing here reports an error, so nothing here can report one
/// twice.
fn prescan_functions(toks: &[Spanned]) -> Vec<LocalFn> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i].tok != Tok::Kw("function") {
            i += 1;
            continue;
        }
        let Some(Tok::Ident(name)) = toks.get(i + 1).map(|s| &s.tok) else {
            i += 1;
            continue;
        };
        let name = name.clone();
        let mut j = i + 2;
        if toks.get(j).map(|s| &s.tok) != Some(&Tok::Sym("(")) {
            i += 1;
            continue;
        }
        j += 1;
        let mut params = Vec::new();
        // `a: float, b: int )` — read names, skip the `: type` annotations, and
        // stop at the first token that is none of those (a malformed header).
        loop {
            match toks.get(j).map(|s| &s.tok) {
                Some(Tok::Sym(")")) => {
                    j += 1;
                    break;
                }
                Some(Tok::Sym(",")) => j += 1,
                Some(Tok::Sym(":")) => {
                    j += 2;
                }
                Some(Tok::Ident(p)) => {
                    params.push(p.clone());
                    j += 1;
                }
                _ => break,
            }
        }
        let ret = match (toks.get(j).map(|s| &s.tok), toks.get(j + 1).map(|s| &s.tok)) {
            (Some(Tok::Sym("->")), Some(Tok::Ident(t))) => ty_of_name(t).unwrap_or(Ty::Unit),
            _ => Ty::Unit,
        };
        out.push(LocalFn { name, params, ret });
        i += 2;
    }
    out
}

/// A type keyword's [`Ty`], for the header prescan. The parser's own `ty()`
/// reports a diagnostic; this one only needs to recognise.
fn ty_of_name(name: &str) -> Option<Ty> {
    Some(match name {
        "float" => Ty::Float,
        "int" => Ty::Int,
        "bool" => Ty::Bool,
        "string" => Ty::Str,
        _ => return None,
    })
}

struct Parser<'a> {
    toks: &'a [Spanned],
    i: usize,
    diags: Vec<Diagnostic>,
    verbs: Verbs,
    declared: Vec<String>,
    /// The `function` headers this file declares — what a bare call resolves
    /// against (see [`prescan_functions`]).
    functions: Vec<LocalFn>,
    /// Per-function: the next local id.
    next_local: u32,
    /// Per-function: block scopes of `local` bindings, innermost last.
    scopes: Vec<Vec<(String, LocalId)>>,
    /// Per-function: parameter names.
    params: Vec<String>,
    /// Per-function: locals something assigns to.
    assigned: Vec<LocalId>,
    /// How deeply nested the walk currently is — see [`MAX_NESTING`].
    depth: u32,
}

impl<'a> Parser<'a> {
    fn new(toks: &'a [Spanned], declared: Vec<String>, functions: Vec<LocalFn>) -> Self {
        Self {
            toks,
            i: 0,
            diags: Vec::new(),
            verbs: Verbs::new(),
            declared,
            functions,
            next_local: 0,
            scopes: Vec::new(),
            params: Vec::new(),
            assigned: Vec::new(),
            depth: 0,
        }
    }

    // ── the nesting bound ────────────────────────────────────────────────

    /// Spend one level of [`MAX_NESTING`], or refuse.
    ///
    /// The refusal is a **value** with a line and a remedy, which is the whole
    /// point: unbounded, this is a stack overflow, and a stack overflow in a
    /// library an editor calls on every keystroke takes the editor.
    fn enter(&mut self) -> P<()> {
        self.depth += 1;
        if self.depth > MAX_NESTING {
            let span = self.span();
            return Err(self.error(
                span,
                format!(
                    "nested more than {MAX_NESTING} levels deep — split the \
                     expression across `local` bindings, or the block across \
                     handlers. (Blocks, parentheses, unary operators and the \
                     steps of an operator chain share one budget, because they \
                     share one stack.)"
                ),
            ));
        }
        Ok(())
    }

    /// Release levels claimed by [`Self::enter`].
    fn leave(&mut self, to: u32) {
        self.depth = to;
    }

    // ── token plumbing ───────────────────────────────────────────────────

    fn peek(&self) -> &Tok {
        &self.toks[self.i.min(self.toks.len() - 1)].tok
    }

    fn span(&self) -> Span {
        self.toks[self.i.min(self.toks.len() - 1)].span
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.i.min(self.toks.len() - 1)].tok.clone();
        if self.i < self.toks.len() - 1 {
            self.i += 1;
        }
        t
    }

    fn at(&self, t: &Tok) -> bool {
        self.peek() == t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.at(t) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn error(&mut self, span: Span, message: impl Into<String>) -> Bail {
        self.diags.push(Diagnostic {
            severity: Severity::Error,
            span,
            message: message.into(),
        });
        Bail
    }

    fn warn(&mut self, span: Span, message: impl Into<String>) {
        self.diags.push(Diagnostic {
            severity: Severity::Warning,
            span,
            message: message.into(),
        });
    }

    /// Consume `t` or refuse, naming what was expected and what was found.
    fn expect(&mut self, t: Tok) -> P<()> {
        if self.eat(&t) {
            return Ok(());
        }
        let (span, found) = (self.span(), self.peek().describe());
        Err(self.error(span, format!("expected {}, found {found}", t.describe())))
    }

    fn expect_ident(&mut self, what: &str) -> P<(String, Span)> {
        let span = self.span();
        match self.bump() {
            Tok::Ident(n) => Ok((n, span)),
            other => Err(self.error(span, format!("expected {what}, found {}", other.describe()))),
        }
    }

    fn expect_string(&mut self, what: &str) -> P<String> {
        let span = self.span();
        match self.bump() {
            Tok::Str(s) => Ok(s),
            other => Err(self.error(span, format!("expected {what}, found {}", other.describe()))),
        }
    }

    // ── the file ─────────────────────────────────────────────────────────

    fn unit(&mut self) -> Unit {
        let mut unit = Unit::default();
        if self.at(&Tok::Kw("actor")) {
            self.bump();
            match self.expect_string("the actor's display name, in quotes") {
                Ok(name) => unit.name = Some(name),
                Err(Bail) => self.recover(),
            }
        }
        loop {
            match self.peek().clone() {
                Tok::Eof => return unit,
                Tok::Ident(n) if n == "var" => {
                    if self.var_decl(&mut unit).is_err() {
                        self.recover();
                    }
                }
                Tok::Kw("on") => {
                    if self.event_decl(&mut unit).is_err() {
                        self.recover();
                    }
                }
                Tok::Kw("function") => {
                    if self.fn_decl(&mut unit).is_err() {
                        self.recover();
                    }
                }
                Tok::Kw("actor") => {
                    let span = self.span();
                    self.error(span, "the `actor` header must be the file's first line");
                    self.recover();
                }
                other => {
                    let span = self.span();
                    self.error(
                        span,
                        format!(
                            "expected `var`, `on` or `function` at the top level, found {}",
                            other.describe()
                        ),
                    );
                    self.recover();
                }
            }
        }
    }

    /// Skip to the next plausible declaration start so one bad handler does not
    /// hide the rest of the file's diagnostics. Always advances.
    fn recover(&mut self) {
        // A bail unwinds out of an arbitrary number of `enter`s, so the budget
        // is reset here rather than unwound — otherwise the *next* declaration
        // in a file whose first one nested too deeply would inherit a spent
        // budget and refuse for a reason that is not about it.
        self.depth = 0;
        loop {
            self.bump();
            match self.peek() {
                Tok::Eof => return,
                Tok::Kw(k) if matches!(*k, "on" | "function") && self.span().col == 1 => return,
                Tok::Ident(n) if n == "var" && self.span().col == 1 => return,
                _ => {}
            }
        }
    }

    /// `var name: type = literal [exposed]`
    fn var_decl(&mut self, unit: &mut Unit) -> P<()> {
        self.expect(Tok::Ident("var".into()))?;
        let (name, nspan) = self.expect_ident("a variable name")?;
        self.check_writable(&name, nspan)?;
        self.expect(Tok::Sym(":"))?;
        let ty = self.ty()?;
        self.expect(Tok::Sym("="))?;
        let span = self.span();
        let default = self.literal()?;
        if !lit_matches(&default, ty) {
            return Err(self.error(
                span,
                format!(
                    "`{name}` is declared `{}` but its default is {}",
                    ty_name(ty),
                    lit_kind(&default)
                ),
            ));
        }
        let exposed = self.eat(&Tok::Kw("exposed"));
        unit.variables.push(Variable {
            name,
            ty,
            default,
            exposed,
        });
        Ok(())
    }

    /// `on <event> [ "arg" ] ( params ) … end`
    fn event_decl(&mut self, unit: &mut Unit) -> P<()> {
        let header = self.span();
        self.expect(Tok::Kw("on"))?;
        let (word, wspan) = self.expect_ident("an event name")?;
        let arg = if matches!(self.peek(), Tok::Str(_)) {
            Some(self.expect_string("the event's name")?)
        } else {
            None
        };
        let Some(event) = event_kind(&word, arg.clone()) else {
            return Err(self.error(
                wspan,
                format!(
                    "`{word}` is not an event. The events are {}",
                    EVENT_NAMES.join(", ")
                ),
            ));
        };
        if matches!(word.as_str(), "input" | "custom") && arg.is_none() {
            return Err(self.error(
                wspan,
                format!("`on {word}` needs a name: `on {word} \"jump\"(…)`"),
            ));
        }
        let signature = event.signature();
        let params = self.params(Some(&signature))?;
        let key = event.key();
        let body = self.function_body(&params)?;
        if unit.events.iter().any(|b| b.event == event) {
            return Err(self.error(header, format!("`{key}` already has a handler")));
        }
        unit.spans.insert(key.clone(), header);
        unit.events.push(EventBinding {
            event,
            body: BlueprintFn {
                name: sanitize_ident(&key),
                id: key,
                params,
                ret: Ty::Unit,
                body,
            },
        });
        Ok(())
    }

    /// `function name(params) [-> type] … end`
    fn fn_decl(&mut self, unit: &mut Unit) -> P<()> {
        let header = self.span();
        self.expect(Tok::Kw("function"))?;
        let (name, nspan) = self.expect_ident("a function name")?;
        self.check_writable(&name, nspan)?;
        let params = self.params(None)?;
        let ret = if self.eat(&Tok::Sym("->")) {
            self.ty()?
        } else {
            Ty::Unit
        };
        let body = self.function_body(&params)?;
        if unit.functions.iter().any(|f| f.id == name) {
            return Err(self.error(nspan, format!("`{name}` is declared twice")));
        }
        unit.spans.insert(name.clone(), header);
        unit.functions.push(BlueprintFn {
            id: name.clone(),
            name,
            params,
            ret,
            body,
        });
        Ok(())
    }

    /// `( a: float, b )` — types are required for a `function`, and checked
    /// against the event's own signature for a handler.
    fn params(&mut self, signature: Option<&[Param]>) -> P<Vec<Param>> {
        let open = self.span();
        self.expect(Tok::Sym("("))?;
        let mut out: Vec<Param> = Vec::new();
        while !self.at(&Tok::Sym(")")) {
            if !out.is_empty() {
                self.expect(Tok::Sym(","))?;
            }
            let (name, nspan) = self.expect_ident("a parameter name")?;
            // `n7` and `speed_n7` are how an unnamed and a named graph value
            // print; a parameter wearing one of those spellings would collide
            // with a local's printed name on the way back out.
            if signature.is_none() {
                self.check_writable(&name, nspan)?;
            }
            if signature.is_none() && BlueprintFn::parse_local_ident(&name).is_some() {
                return Err(self.error(
                    nspan,
                    format!(
                        "`{name}` is reserved — `n<number>` and `<name>_n<number>` are how graph-authored values are written"
                    ),
                ));
            }
            let declared = if self.eat(&Tok::Sym(":")) {
                Some(self.ty()?)
            } else {
                None
            };
            let ty = match (declared, signature) {
                (Some(t), _) => t,
                (None, Some(sig)) => match sig.get(out.len()) {
                    Some(p) => p.ty,
                    None => Ty::Unit,
                },
                (None, None) => {
                    return Err(self.error(nspan, format!("`{name}` needs a type: `{name}: float`")))
                }
            };
            out.push(Param { name, ty });
        }
        self.expect(Tok::Sym(")"))?;
        if let Some(sig) = signature {
            if out.len() != sig.len()
                || out
                    .iter()
                    .zip(sig)
                    .any(|(a, b)| a.name != b.name || a.ty != b.ty)
            {
                return Err(self.error(
                    open,
                    format!("this event's handler takes {}", describe_signature(sig)),
                ));
            }
        }
        Ok(out)
    }

    /// A function body, in a fresh local scope, terminated by `end`.
    fn function_body(&mut self, params: &[Param]) -> P<Vec<Stmt>> {
        self.next_local = 0;
        self.scopes = vec![Vec::new()];
        self.params = params.iter().map(|p| p.name.clone()).collect();
        self.assigned.clear();
        self.depth = 0;
        let mut body = self.block(&["end"])?;
        self.expect(Tok::Kw("end"))?;
        let assigned = std::mem::take(&mut self.assigned);
        mark_mutable(&mut body, &assigned);
        self.scopes.clear();
        Ok(body)
    }

    /// Statements until one of `terminators` (not consumed). One level of the
    /// nesting budget: a block is reached only by recursing through a statement.
    fn block(&mut self, terminators: &[&str]) -> P<Vec<Stmt>> {
        let entry = self.depth;
        self.enter()?;
        let r = self.block_inner(terminators);
        self.leave(entry);
        r
    }

    fn block_inner(&mut self, terminators: &[&str]) -> P<Vec<Stmt>> {
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Tok::Eof => {
                    let span = self.span();
                    return Err(self.error(
                        span,
                        format!(
                            "unexpected end of file — expected `{}`",
                            terminators.join("` or `")
                        ),
                    ));
                }
                Tok::Kw(k) if terminators.contains(k) => return Ok(out),
                Tok::Sym(";") => {
                    self.bump();
                }
                _ => self.stmt(&mut out)?,
            }
        }
    }

    fn scoped_block(&mut self, terminators: &[&str]) -> P<Vec<Stmt>> {
        self.scopes.push(Vec::new());
        let r = self.block(terminators);
        self.scopes.pop();
        r
    }

    // ── statements ───────────────────────────────────────────────────────

    fn stmt(&mut self, out: &mut Vec<Stmt>) -> P<()> {
        let span = self.span();
        match self.peek().clone() {
            Tok::Kw("local") => self.local_decl(out),
            Tok::Kw("if") => self.if_stmt(out),
            Tok::Kw("while") => self.while_stmt(out),
            Tok::Kw("for") => self.for_stmt(out),
            Tok::Kw("return") => {
                self.bump();
                // **A returned value must be on the `return`'s own line.**
                // Without the rule `return` followed by a call on the next line
                // reads the call as the returned value — Lua avoids the
                // ambiguity by requiring `return` to end its block, and this
                // engine's IR has `Stmt::Return` in the middle of a body (it is
                // `raise` that refuses one, not the interpreter), so the line is
                // the cheaper rule and it matches what a reader sees.
                let same_line = self.span().line == span.line;
                let value = if same_line && self.starts_expr() {
                    Some(self.expr()?)
                } else {
                    None
                };
                out.push(Stmt::Return(value));
                Ok(())
            }
            Tok::Kw("rust") => {
                self.bump();
                let at = self.span();
                match self.bump() {
                    Tok::Long(body) => {
                        out.push(Stmt::Snippet(body));
                        Ok(())
                    }
                    other => Err(self.error(
                        at,
                        format!("`rust` takes a `[[…]]` block, found {}", other.describe()),
                    )),
                }
            }
            Tok::Ident(_) => self.ident_stmt(out, span),
            other => Err(self.error(
                span,
                format!("{} cannot start a statement", other.describe()),
            )),
        }
    }

    /// `local x[: ty] = expr`
    fn local_decl(&mut self, out: &mut Vec<Stmt>) -> P<()> {
        self.expect(Tok::Kw("local"))?;
        let (name, nspan) = self.expect_ident("a local's name")?;
        self.check_shadow(&name, nspan)?;
        let ty = if self.eat(&Tok::Sym(":")) {
            Some(self.ty()?)
        } else {
            None
        };
        self.expect(Tok::Sym("="))?;
        let value = self.expr()?;
        let id = self.alloc_local();
        out.push(Stmt::Let {
            id,
            binding: binding_of(&name),
            ty,
            mutable: false,
            value,
        });
        self.scopes
            .last_mut()
            .expect("a body always opens a scope")
            .push((name, id));
        Ok(())
    }

    /// An identifier at statement position: an assignment, or a call.
    fn ident_stmt(&mut self, out: &mut Vec<Stmt>, span: Span) -> P<()> {
        let path = self.dotted_path()?;
        if path.len() == 1 && self.at(&Tok::Sym("=")) {
            self.bump();
            let value = self.expr()?;
            let name = &path[0];
            if let Some(id) = self.lookup_local(name) {
                self.assigned.push(id);
                out.push(Stmt::Assign { target: id, value });
            } else if self.params.iter().any(|p| p == name) {
                return Err(self.error(
                    span,
                    format!(
                        "`{name}` is a handler parameter and cannot be assigned — \
                         copy it into a `local` first"
                    ),
                ));
            } else {
                self.check_declared(name, span);
                out.push(var_set(name, value));
            }
            return Ok(());
        }
        // `nodestate.set(key, value)` — the state cells `flow.do_once`,
        // `flip_flop` and `gate` lower to. A script rarely writes one by hand;
        // it exists so that opening such a graph as text is *readable*, and so
        // that the emitter is total over the IR.
        if path == ["nodestate", "set"] {
            let args = self.call_args()?;
            if args.len() != 2 {
                return Err(self.error(span, "`nodestate.set` takes a key and a value"));
            }
            out.push(Stmt::ExprStmt(Expr::Call {
                path: vec!["nodestate".into(), "set".into()],
                args,
            }));
            return Ok(());
        }
        // `var.set("name", value)` — the escape hatch for a variable whose name
        // is not an identifier.
        if path == ["var", "set"] {
            let args = self.call_args()?;
            let mut args = args.into_iter();
            let (Some(Expr::Lit(Lit::Str(name))), Some(value), None) =
                (args.next(), args.next(), args.next())
            else {
                return Err(self.error(span, "`var.set` takes a quoted variable name and a value"));
            };
            out.push(var_set(&name, value));
            return Ok(());
        }
        // **The call form**, in statement position. Checked before the registry
        // so a declaration in this file wins; a name that is not declared here
        // still gets the registry's refusal.
        if let [name] = path.as_slice() {
            if let Some(f) = self.local_fn(name) {
                let args = self.call_args()?;
                self.check_local_arity(&f, &args, span)?;
                out.push(Stmt::ExprStmt(Expr::Call {
                    path: vec![f.name],
                    args,
                }));
                return Ok(());
            }
        }
        let verb = self.resolve_call(&path, span)?;
        let args = self.call_args()?;
        self.check_arity(&verb, &args, span)?;
        let call = Expr::Call {
            path: verb.path.clone(),
            args,
        };
        // An action with a consumed data output binds it, exactly as the graph
        // lowerer does; here nothing consumes it, so it stays a bare statement.
        out.push(Stmt::ExprStmt(call));
        Ok(())
    }

    /// `if c then … [elseif c then …] [else …] end`
    fn if_stmt(&mut self, out: &mut Vec<Stmt>) -> P<()> {
        self.expect(Tok::Kw("if"))?;
        let cond = self.expr()?;
        self.expect(Tok::Kw("then"))?;
        let then_body = self.scoped_block(&["elseif", "else", "end"])?;
        let else_body = self.else_tail()?;
        out.push(Stmt::If {
            cond,
            then_body,
            else_body,
        });
        Ok(())
    }

    /// The `elseif`/`else`/`end` tail, as the nested `Stmt::If` an `elseif` is.
    fn else_tail(&mut self) -> P<Vec<Stmt>> {
        if self.eat(&Tok::Kw("elseif")) {
            let cond = self.expr()?;
            self.expect(Tok::Kw("then"))?;
            let then_body = self.scoped_block(&["elseif", "else", "end"])?;
            let else_body = self.else_tail()?;
            return Ok(vec![Stmt::If {
                cond,
                then_body,
                else_body,
            }]);
        }
        if self.eat(&Tok::Kw("else")) {
            let body = self.scoped_block(&["end"])?;
            self.expect(Tok::Kw("end"))?;
            return Ok(body);
        }
        self.expect(Tok::Kw("end"))?;
        Ok(Vec::new())
    }

    /// `while c do … end` — the counter-guarded expansion, so the shipped
    /// program cannot spin for ever and so the loop raises to a `flow.while`.
    fn while_stmt(&mut self, out: &mut Vec<Stmt>) -> P<()> {
        self.expect(Tok::Kw("while"))?;
        let cond = self.expr()?;
        self.expect(Tok::Kw("do"))?;
        // The counter is allocated before the body, matching `lower_while`.
        let counter = self.alloc_local();
        let mut body = self.scoped_block(&["end"])?;
        self.expect(Tok::Kw("end"))?;
        body.push(loopshape::increment(counter));
        self.assigned.push(counter);
        out.push(loopshape::counter_init(counter));
        out.push(Stmt::While {
            cond: Expr::Binary(
                BinOp::And,
                Box::new(cond),
                Box::new(loopshape::guard_lt(counter)),
            ),
            body,
        });
        out.push(loopshape::runaway_report(counter));
        Ok(())
    }

    /// `for i = first, last do … end` — inclusive of `last`, stepping by one,
    /// which is `flow.for`'s contract.
    fn for_stmt(&mut self, out: &mut Vec<Stmt>) -> P<()> {
        self.expect(Tok::Kw("for"))?;
        let (name, nspan) = self.expect_ident("the loop variable's name")?;
        self.expect(Tok::Sym("="))?;
        let first = self.expr()?;
        self.expect(Tok::Sym(","))?;
        let last = self.expr()?;
        self.expect(Tok::Kw("do"))?;
        // Allocation order mirrors `lower_for` exactly: index, bound, counter.
        let index = self.alloc_local();
        let bound = self.alloc_local();
        let counter = self.alloc_local();
        self.check_shadow(&name, nspan)?;
        let binding = binding_of(&name);
        self.scopes.push(vec![(name, index)]);
        let body = self.block(&["end"]);
        self.scopes.pop();
        let mut body = body?;
        self.expect(Tok::Kw("end"))?;
        body.push(loopshape::increment(index));
        body.push(loopshape::increment(counter));
        self.assigned.push(index);
        self.assigned.push(counter);
        out.push(loopshape::index_init(index, binding, first));
        out.push(loopshape::bound_init(bound, last));
        out.push(loopshape::counter_init(counter));
        out.push(Stmt::While {
            cond: Expr::Binary(
                BinOp::And,
                Box::new(loopshape::index_le(index, bound)),
                Box::new(loopshape::guard_lt(counter)),
            ),
            body,
        });
        out.push(loopshape::runaway_report(counter));
        Ok(())
    }

    // ── expressions ──────────────────────────────────────────────────────

    fn starts_expr(&self) -> bool {
        matches!(
            self.peek(),
            Tok::Ident(_)
                | Tok::Int(_)
                | Tok::Float(_)
                | Tok::Str(_)
                | Tok::Kw("true")
                | Tok::Kw("false")
                | Tok::Kw("not")
                | Tok::Sym("(")
                | Tok::Sym("-")
        )
    }

    fn expr(&mut self) -> P<Expr> {
        self.or_expr()
    }

    /// Each of the four chain parsers releases the levels its accumulation
    /// spent once the whole chain is built: the tree it produced is a sibling of
    /// whatever follows, not an ancestor of it. What it may **not** do is spend
    /// nothing, because the tree is left-deep and every consumer of the IR walks
    /// it recursively.
    fn or_expr(&mut self) -> P<Expr> {
        let entry = self.depth;
        let r = self.or_expr_inner();
        self.leave(entry);
        r
    }

    fn or_expr_inner(&mut self) -> P<Expr> {
        let mut lhs = self.and_expr()?;
        while self.eat(&Tok::Kw("or")) {
            self.enter()?;
            let rhs = self.and_expr()?;
            lhs = Expr::Binary(BinOp::Or, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn and_expr(&mut self) -> P<Expr> {
        let entry = self.depth;
        let r = self.and_expr_inner();
        self.leave(entry);
        r
    }

    fn and_expr_inner(&mut self) -> P<Expr> {
        let mut lhs = self.cmp_expr()?;
        while self.eat(&Tok::Kw("and")) {
            self.enter()?;
            let rhs = self.cmp_expr()?;
            lhs = Expr::Binary(BinOp::And, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    /// Comparisons do **not** chain (`a < b < c` is a refusal, not a surprise).
    fn cmp_expr(&mut self) -> P<Expr> {
        let lhs = self.add_expr()?;
        let op = match self.peek() {
            Tok::Sym("==") => BinOp::Eq,
            Tok::Sym("~=") => BinOp::Ne,
            Tok::Sym("<") => BinOp::Lt,
            Tok::Sym("<=") => BinOp::Le,
            Tok::Sym(">") => BinOp::Gt,
            Tok::Sym(">=") => BinOp::Ge,
            _ => return Ok(lhs),
        };
        self.bump();
        let rhs = self.add_expr()?;
        if matches!(
            self.peek(),
            Tok::Sym("==")
                | Tok::Sym("~=")
                | Tok::Sym("<")
                | Tok::Sym("<=")
                | Tok::Sym(">")
                | Tok::Sym(">=")
        ) {
            let span = self.span();
            return Err(self.error(span, "comparisons do not chain — write `a < b and b < c`"));
        }
        Ok(Expr::Binary(op, Box::new(lhs), Box::new(rhs)))
    }

    fn add_expr(&mut self) -> P<Expr> {
        let entry = self.depth;
        let r = self.add_expr_inner();
        self.leave(entry);
        r
    }

    fn add_expr_inner(&mut self) -> P<Expr> {
        let mut lhs = self.mul_expr()?;
        loop {
            let op = match self.peek() {
                Tok::Sym("+") => BinOp::Add,
                Tok::Sym("-") => BinOp::Sub,
                _ => return Ok(lhs),
            };
            self.bump();
            self.enter()?;
            let rhs = self.mul_expr()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
    }

    fn mul_expr(&mut self) -> P<Expr> {
        let entry = self.depth;
        let r = self.mul_expr_inner();
        self.leave(entry);
        r
    }

    fn mul_expr_inner(&mut self) -> P<Expr> {
        let mut lhs = self.unary()?;
        loop {
            let op = match self.peek() {
                Tok::Sym("*") => BinOp::Mul,
                Tok::Sym("/") => BinOp::Div,
                Tok::Sym("%") => BinOp::Rem,
                _ => return Ok(lhs),
            };
            self.bump();
            self.enter()?;
            let rhs = self.unary()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
    }

    /// `-x`, `not x`, and the one fold that makes the round trip work:
    /// **`-` directly against a numeric literal is part of the literal**, while
    /// `-(1)` is the negation of one. Without the distinction `Lit::Int(-1)` and
    /// `Unary(Neg, Lit::Int(1))` — two different IRs a graph can hold — would
    /// print the same text and could not both come back.
    fn unary(&mut self) -> P<Expr> {
        let entry = self.depth;
        self.enter()?;
        let r = self.unary_inner();
        self.leave(entry);
        r
    }

    fn unary_inner(&mut self) -> P<Expr> {
        if self.at(&Tok::Kw("not")) {
            self.bump();
            let inner = self.unary()?;
            return Ok(Expr::Unary(UnOp::Not, Box::new(inner)));
        }
        if self.at(&Tok::Sym("-")) {
            let span = self.span();
            self.bump();
            if let Tok::Int(digits) | Tok::Float(digits) = self.peek().clone() {
                let is_int = matches!(self.peek(), Tok::Int(_));
                self.bump();
                return self.number_lit(&format!("-{digits}"), is_int, span);
            }
            let inner = self.unary()?;
            // **A negated literal is folded into a negative one**, whatever
            // spelling it arrived in. `Unary(Neg, Lit)` is *non-canonical IR*:
            // `inf_transpile::emit` refuses it outright
            // (`EmitError::NegatedLiteral`), because the lifter folds `-lit` on
            // the way back and so the shape could not round-trip through Rust.
            // A language that could write it would be a language that can write
            // programs the cook refuses, which is not a trade this arc makes.
            // `-(1)` and `-1` are therefore one thing here, and the emitter
            // prints the canonical form.
            return Ok(match inner {
                Expr::Lit(Lit::Float(f)) => Expr::Lit(Lit::Float(-f)),
                // `-i64::MIN` has no `i64`; that one keeps the unary form, which
                // the transpiler refuses for a different reason it already names
                // (`EmitError::IntMin`).
                Expr::Lit(Lit::Int(i)) => match i.checked_neg() {
                    Some(n) => Expr::Lit(Lit::Int(n)),
                    None => Expr::Unary(UnOp::Neg, Box::new(Expr::Lit(Lit::Int(i)))),
                },
                Expr::Lit(other) => {
                    return Err(self.error(
                        span,
                        format!("`-` needs a number, not {}", lit_kind(&other)),
                    ))
                }
                other => Expr::Unary(UnOp::Neg, Box::new(other)),
            });
        }
        self.primary()
    }

    fn primary(&mut self) -> P<Expr> {
        let span = self.span();
        match self.peek().clone() {
            Tok::Int(d) => {
                self.bump();
                self.number_lit(&d, true, span)
            }
            Tok::Float(d) => {
                self.bump();
                self.number_lit(&d, false, span)
            }
            Tok::Str(s) => {
                self.bump();
                Ok(Expr::Lit(Lit::Str(s)))
            }
            Tok::Kw("true") => {
                self.bump();
                Ok(Expr::Lit(Lit::Bool(true)))
            }
            Tok::Kw("false") => {
                self.bump();
                Ok(Expr::Lit(Lit::Bool(false)))
            }
            Tok::Sym("(") => {
                self.bump();
                let e = self.expr()?;
                self.expect(Tok::Sym(")"))?;
                Ok(e)
            }
            Tok::Ident(_) => self.name_or_call(span),
            other => Err(self.error(
                span,
                format!("expected a value, found {}", other.describe()),
            )),
        }
    }

    fn number_lit(&mut self, text: &str, is_int: bool, span: Span) -> P<Expr> {
        if is_int {
            match text.parse::<i64>() {
                Ok(v) => Ok(Expr::Lit(Lit::Int(v))),
                Err(_) => Err(self.error(span, format!("`{text}` does not fit a 64-bit integer"))),
            }
        } else {
            match text.parse::<f64>() {
                Ok(v) if v.is_finite() => Ok(Expr::Lit(Lit::Float(v))),
                Ok(_) => {
                    Err(self.error(span, format!("`{text}` is out of range for a 64-bit float")))
                }
                Err(_) => Err(self.error(span, format!("`{text}` is not a number"))),
            }
        }
    }

    /// A bare name, a `var.get("…")`, or a call.
    fn name_or_call(&mut self, span: Span) -> P<Expr> {
        let path = self.dotted_path()?;
        if path.len() == 1 {
            let name = &path[0];
            if self.at(&Tok::Sym("(")) {
                let name = name.clone();
                // **The call form**, in value position: a bare call naming one
                // of this unit's own `function` declarations.
                if let Some(f) = self.local_fn(&name) {
                    let args = self.call_args()?;
                    self.check_local_arity(&f, &args, span)?;
                    if f.ret == Ty::Unit {
                        return Err(self.error(
                            span,
                            format!(
                                "`{name}` returns nothing, so it cannot appear inside an \
                                 expression — call it as a statement, or give it a return \
                                 type: `function {name}(…) -> float`"
                            ),
                        ));
                    }
                    return Ok(Expr::Call {
                        path: vec![name],
                        args,
                    });
                }
                let _ = self.call_args();
                let message = self.bare_call_message(&name);
                return Err(self.error(span, message));
            }
            if let Some(id) = self.lookup_local(name) {
                return Ok(Expr::Local(id));
            }
            if self.params.iter().any(|p| p == name) {
                return Ok(Expr::Param(name.clone()));
            }
            self.check_declared(name, span);
            return Ok(var_get(name));
        }
        if path == ["nodestate", "get_or"] {
            let args = self.call_args()?;
            if args.len() != 2 {
                return Err(self.error(
                    span,
                    "`nodestate.get_or` takes a key and the value a never-set cell reads as",
                ));
            }
            return Ok(Expr::Call {
                path: vec!["nodestate".into(), "get_or".into()],
                args,
            });
        }
        if path == ["var", "get"] {
            let args = self.call_args()?;
            let mut args = args.into_iter();
            let (Some(Expr::Lit(Lit::Str(name))), None) = (args.next(), args.next()) else {
                return Err(self.error(span, "`var.get` takes one quoted variable name"));
            };
            return Ok(var_get(&name));
        }
        let verb = self.resolve_call(&path, span)?;
        if !verb.produces_value {
            return Err(self.error(
                span,
                format!(
                    "`{}` produces no value, so it cannot appear inside an \
                     expression — call it as a statement",
                    verb.type_id
                ),
            ));
        }
        let args = self.call_args()?;
        self.check_arity(&verb, &args, span)?;
        Ok(Expr::Call {
            path: verb.path,
            args,
        })
    }

    fn dotted_path(&mut self) -> P<Vec<String>> {
        let (first, _) = self.expect_ident("a name")?;
        let mut out = vec![first];
        while self.eat(&Tok::Sym(".")) {
            let (next, _) = self.expect_ident("a verb name after `.`")?;
            out.push(next);
        }
        Ok(out)
    }

    fn call_args(&mut self) -> P<Vec<Expr>> {
        self.expect(Tok::Sym("("))?;
        let mut args = Vec::new();
        while !self.at(&Tok::Sym(")")) {
            if !args.is_empty() {
                self.expect(Tok::Sym(","))?;
            }
            args.push(self.expr()?);
        }
        self.expect(Tok::Sym(")"))?;
        Ok(args)
    }

    fn resolve_call(&mut self, path: &[String], span: Span) -> P<crate::verbs::Verb> {
        match self.verbs.resolve(path) {
            Ok(v) => Ok(v),
            // A one-segment call is the shape a designer writes when they mean
            // one of their own `function`s, so it gets the message that names
            // the real bound rather than the registry's generic one.
            Err(crate::verbs::VerbError::NotNamespaced(_)) if path.len() == 1 => {
                let message = self.bare_call_message(&path[0]);
                Err(self.error(span, message))
            }
            Err(e) => Err(self.error(span, e.message())),
        }
    }

    /// Why a bare `name(…)` is not a call, when the file declares no `function`
    /// by that name.
    ///
    /// A file that *does* declare one no longer reaches here: SCRIPT2's call
    /// form resolves it (see [`Self::local_fn`]). SCRIPT1a's version of this
    /// message explained why the call was impossible; the wave that made it
    /// possible retires the explanation and keeps the one refusal that is still
    /// true.
    fn bare_call_message(&self, name: &str) -> String {
        format!(
            "`{name}` is not a verb — a call names a namespace and a verb, like \
             `debug.print(\"hello\")`. To call one of this file's own functions, \
             declare it: `function {name}(…) … end`"
        )
    }

    /// The unit-local `function` a bare name resolves to, if any.
    fn local_fn(&self, name: &str) -> Option<LocalFn> {
        self.functions.iter().find(|f| f.name == name).cloned()
    }

    /// A unit-local call's arguments, checked against the declared parameters.
    ///
    /// **Exact**, unlike a verb's: a verb's trailing inputs have literal
    /// defaults the lowerer supplies for an unwired pin, and a function's
    /// parameters have none — a missing one is an `UnboundParam` at run time,
    /// which is a refusal a designer meets three steps from the mistake.
    fn check_local_arity(&mut self, f: &LocalFn, args: &[Expr], span: Span) -> P<()> {
        if args.len() == f.params.len() {
            return Ok(());
        }
        let want = if f.params.is_empty() {
            "no arguments".to_string()
        } else {
            format!(
                "{} ({})",
                plural(f.params.len(), "argument"),
                f.params
                    .iter()
                    .map(|p| format!("`{p}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        Err(self.error(
            span,
            format!("`{}` takes {want}, not {}", f.name, args.len()),
        ))
    }

    fn check_arity(&mut self, verb: &crate::verbs::Verb, args: &[Expr], span: Span) -> P<()> {
        if args.len() > verb.inputs.len() {
            let names: Vec<String> = verb.inputs.iter().map(|(n, _)| format!("`{n}`")).collect();
            return Err(self.error(
                span,
                format!(
                    "`{}` takes {} ({}), not {}",
                    verb.type_id,
                    plural(verb.inputs.len(), "argument"),
                    if names.is_empty() {
                        "none".to_string()
                    } else {
                        names.join(", ")
                    },
                    args.len()
                ),
            ));
        }
        if let Some((name, _)) = verb
            .inputs
            .iter()
            .skip(args.len())
            .find(|(_, required)| *required)
        {
            return Err(self.error(
                span,
                format!("`{}` needs its `{name}` argument", verb.type_id),
            ));
        }
        Ok(())
    }

    // ── scopes ───────────────────────────────────────────────────────────

    fn alloc_local(&mut self) -> LocalId {
        let id = LocalId(self.next_local);
        self.next_local += 1;
        id
    }

    /// **Shadowing is refused**, which is a deliberate divergence from Lua (and
    /// from Rust). Two live bindings of one name would print as one name, and a
    /// re-parse would bind every reader to the later one — a silent change of
    /// program across a round trip, which is the class of defect this house
    /// writes gates for. Renaming costs a designer one word; a capture costs a
    /// session.
    /// A name the emitter can write back bare.
    ///
    /// `var` and `nodestate` open the two explicit escape hatches
    /// (`var.get("…")`, `nodestate.set("…", …)`), so a binder wearing either
    /// would print bare and turn the next escape hatch in the same body into a
    /// field access on itself. Checked at every declaration site — variables,
    /// functions, function parameters, locals and loop indices — because
    /// **anything the parser accepts, the emitter must be able to write**, and a
    /// program that parses and then cannot be printed is a worse failure than a
    /// refusal with a line number.
    fn check_writable(&mut self, name: &str, span: Span) -> P<()> {
        if matches!(name, "var" | "nodestate") {
            return Err(self.error(
                span,
                format!("`{name}` is reserved — it opens the explicit `{name}.…` form"),
            ));
        }
        Ok(())
    }

    fn check_shadow(&mut self, name: &str, span: Span) -> P<()> {
        self.check_writable(name, span)?;
        if self.lookup_local(name).is_some() {
            return Err(self.error(
                span,
                format!("`{name}` is already a local here — pick another name (InfiniScript does not shadow, so text and graph cannot disagree about which one a reader meant)"),
            ));
        }
        if self.params.iter().any(|p| p == name) {
            return Err(self.error(
                span,
                format!("`{name}` is a parameter of this handler — pick another name"),
            ));
        }
        Ok(())
    }

    fn lookup_local(&self, name: &str) -> Option<LocalId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|s| s.iter().rev().find(|(n, _)| n == name).map(|(_, id)| *id))
    }

    /// A bare name that is neither a local nor a parameter is a member variable.
    /// When the whole file is in view and it declares none by that name, say so
    /// — as a **warning**, because the runtime refuses it by name anyway.
    fn check_declared(&mut self, name: &str, span: Span) {
        if self.declared.iter().any(|d| d == name) {
            return;
        }
        let known = if self.declared.is_empty() {
            "this file declares none".to_string()
        } else {
            format!(
                "it declares {}",
                self.declared
                    .iter()
                    .map(|d| format!("`{d}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        self.warn(
            span,
            format!(
                "`{name}` is read as a member variable and no `var {name}` is \
                 declared — {known}. The handler will refuse at run time"
            ),
        );
    }

    // ── small parsers ────────────────────────────────────────────────────

    fn ty(&mut self) -> P<Ty> {
        let span = self.span();
        let (name, _) = self.expect_ident("a type")?;
        match name.as_str() {
            "float" => Ok(Ty::Float),
            "int" => Ok(Ty::Int),
            "bool" => Ok(Ty::Bool),
            "string" => Ok(Ty::Str),
            other => Err(self.error(
                span,
                format!("`{other}` is not a type — the types are `float`, `int`, `bool`, `string`"),
            )),
        }
    }

    /// A literal, for a `var` default. Folds a leading `-` like [`Self::unary`].
    fn literal(&mut self) -> P<Lit> {
        let e = self.unary()?;
        let span = self.span();
        match e {
            Expr::Lit(l) => Ok(l),
            _ => Err(self.error(span, "a variable's default must be a literal value")),
        }
    }
}

/// `vars::get("name")`.
fn var_get(name: &str) -> Expr {
    Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str(name.to_string()))],
    }
}

/// `vars::set("name", value);`.
fn var_set(name: &str, value: Expr) -> Stmt {
    Stmt::ExprStmt(Expr::Call {
        path: vec!["vars".into(), "set".into()],
        args: vec![Expr::Lit(Lit::Str(name.to_string())), value],
    })
}

/// The binding kind an identifier denotes — the same encoding the Rust
/// projection uses, so a `speed_n7` that came from a graph keeps being one.
fn binding_of(name: &str) -> Binding {
    match BlueprintFn::parse_local_ident(name) {
        Some((Some(n), _)) => Binding::Named(n),
        Some((None, _)) => Binding::Anon,
        None => Binding::Raw(name.to_string()),
    }
}

/// Set `mutable` on every `Let` something assigns to. There is no `local mut`;
/// mutability is a fact about the body, and deriving it is what keeps the
/// emitted text free of a keyword the reader would have to maintain.
fn mark_mutable(body: &mut [Stmt], assigned: &[LocalId]) {
    for s in body.iter_mut() {
        match s {
            Stmt::Let { id, mutable, .. } => *mutable = assigned.contains(id),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                mark_mutable(then_body, assigned);
                mark_mutable(else_body, assigned);
            }
            Stmt::While { body, .. } => mark_mutable(body, assigned),
            _ => {}
        }
    }
}

/// The events a handler header may name.
const EVENT_NAMES: [&str; 9] = [
    "begin_play",
    "tick",
    "collision",
    "input \"…\"",
    "custom \"…\"",
    "water_enter",
    "water_exit",
    "water_splash",
    "destroyed",
];

fn event_kind(word: &str, arg: Option<String>) -> Option<EventKind> {
    Some(match word {
        "begin_play" => EventKind::BeginPlay,
        "tick" => EventKind::Tick,
        "collision" => EventKind::Collision,
        "water_enter" => EventKind::WaterEnter,
        "water_exit" => EventKind::WaterExit,
        "water_splash" => EventKind::WaterSplash,
        "destroyed" => EventKind::Destroyed,
        "input" => EventKind::Input(arg.unwrap_or_default()),
        "custom" => EventKind::Custom(arg.unwrap_or_default()),
        _ => return None,
    })
}

fn describe_signature(sig: &[Param]) -> String {
    if sig.is_empty() {
        return "no parameters: `()`".into();
    }
    format!(
        "`({})`",
        sig.iter()
            .map(|p| format!("{}: {}", p.name, ty_name(p.ty)))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The `.infini` spelling of a type.
pub fn ty_name(ty: Ty) -> &'static str {
    match ty {
        Ty::Float => "float",
        Ty::Int => "int",
        Ty::Bool => "bool",
        Ty::Str => "string",
        Ty::Unit => "nothing",
    }
}

fn lit_kind(l: &Lit) -> &'static str {
    match l {
        Lit::Float(_) => "a float",
        Lit::Int(_) => "an int",
        Lit::Bool(_) => "a bool",
        Lit::Str(_) => "a string",
    }
}

/// A `var`'s default must match its declared type. `int` accepts an int and
/// `float` accepts either, because `2` is a perfectly ordinary way to write a
/// float default and the IR's `Value` promotes it anyway.
fn lit_matches(l: &Lit, ty: Ty) -> bool {
    matches!(
        (l, ty),
        (Lit::Float(_), Ty::Float)
            | (Lit::Int(_), Ty::Float | Ty::Int)
            | (Lit::Bool(_), Ty::Bool)
            | (Lit::Str(_), Ty::Str)
    )
}

fn plural(n: usize, what: &str) -> String {
    if n == 1 {
        format!("1 {what}")
    } else {
        format!("{n} {what}s")
    }
}
