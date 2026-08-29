//! The guarded-loop **shape**: one definition of the statement pattern
//! `flow.while` and `flow.for` lower to, and one recogniser for reading it back.
//!
//! # Why this module exists
//!
//! The loop guard is written in three places and read in three more. Before
//! wave SCRIPT1 the builders lived in [`crate::lower`] as private helpers and the
//! recogniser lived in [`crate::raise`] as a private method, and they agreed
//! **by hand** — the exact arrangement the house has paid for repeatedly (the
//! scaffolded banner's three copies, the six diverged libm ban lists). SCRIPT1
//! adds a *third* consumer — `inf-script`, which parses `while`/`for` from
//! `.infini` text and prints them back — so the hand agreement would have become
//! a four-way one.
//!
//! So the shape is stated once, here, as a builder and a matcher per form, and
//! the three consumers are:
//!
//! | consumer | uses |
//! |---|---|
//! | [`crate::lower`] | [`counter_init`], [`guard_lt`], [`increment`], [`runaway_report`] |
//! | [`crate::raise`] | [`match_while`], [`match_for`] |
//! | `inf_script` (parse + emit) | both halves |
//!
//! [`the_matchers_invert_the_builders`](self) — the module's own test — builds
//! each form and matches it back, so a change to one half that forgets the other
//! fails here rather than in a consumer.
//!
//! # The shapes
//!
//! `flow.while`, three statements:
//!
//! ```text
//! let mut counter = 0;
//! while (user_cond && counter < LOOP_GUARD_MAX) { <body>; counter = counter + 1; }
//! if counter >= LOOP_GUARD_MAX { debug::print(RUNAWAY_MSG); }
//! ```
//!
//! `flow.for`, five statements — the same guard with an induction variable in
//! front, `first`/`last` snapshotted so each is evaluated exactly once:
//!
//! ```text
//! let mut index = <first>;
//! let last = <last>;
//! let mut counter = 0;
//! while (index <= last && counter < LOOP_GUARD_MAX) { <body>; index = index + 1; counter = counter + 1; }
//! if counter >= LOOP_GUARD_MAX { debug::print(RUNAWAY_MSG); }
//! ```
//!
//! The guard lives **in the IR**, not in the interpreter, so the interpreter and
//! the transpiled Rust share the exact same bound — and a `.infini` `while true`
//! cannot hang the editor.

use crate::{BinOp, Binding, Expr, Lit, LocalId, Stmt, LOOP_GUARD_MAX};

/// The exact message the loop guard prints when it trips.
///
/// A **constant**, deliberately node-agnostic: it is a synthetic statement
/// `raise` discards and re-lowering regenerates, so embedding a (raise-
/// renumbered, non-user-facing) `NodeId` would break `lower(raise(f)) == f`.
pub const RUNAWAY_MSG: &str = "Runaway loop stopped (blueprint loop guard exceeded)";

/// `let mut <id> = 0;` — the loop counter's initialiser.
pub fn counter_init(counter: LocalId) -> Stmt {
    Stmt::Let {
        id: counter,
        binding: Binding::Anon,
        ty: None,
        mutable: true,
        value: Expr::Lit(Lit::Int(0)),
    }
}

/// `let mut <binding> = <value>;` — the `for` index's initialiser.
///
/// The **binding is a parameter** rather than always [`Binding::Anon`], because
/// the two authors of a `for` loop name its variable differently and both are
/// right: a graph's `flow.for` has no name for its index and passes `Anon`,
/// while `.infini` text says `for i = 0, 9` and passes `Raw("i")`. The matcher
/// accepts either and hands the binding back, so a designer's loop variable
/// survives a text round trip. Raising a *text*-authored `for` to a graph and
/// lowering it again returns `Anon` — the name has nowhere to live in a graph —
/// which is the documented "up to binding kind" half of the graph round trip.
pub fn index_init(index: LocalId, binding: Binding, value: Expr) -> Stmt {
    Stmt::Let {
        id: index,
        binding,
        ty: None,
        mutable: true,
        value,
    }
}

/// `let <id> = <value>;` — the `for` upper bound, snapshotted so it is
/// evaluated once rather than every iteration.
pub fn bound_init(bound: LocalId, value: Expr) -> Stmt {
    Stmt::Let {
        id: bound,
        binding: Binding::Anon,
        ty: None,
        mutable: false,
        value,
    }
}

/// `counter < LOOP_GUARD_MAX` — the guard half of a loop condition.
pub fn guard_lt(counter: LocalId) -> Expr {
    Expr::Binary(
        BinOp::Lt,
        Box::new(Expr::Local(counter)),
        Box::new(Expr::Lit(Lit::Int(LOOP_GUARD_MAX))),
    )
}

/// `index <= last` — the `for` form's user-facing condition.
pub fn index_le(index: LocalId, last: LocalId) -> Expr {
    Expr::Binary(
        BinOp::Le,
        Box::new(Expr::Local(index)),
        Box::new(Expr::Local(last)),
    )
}

/// `<id> = <id> + 1;` — a loop counter or index increment.
pub fn increment(counter: LocalId) -> Stmt {
    Stmt::Assign {
        target: counter,
        value: Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Local(counter)),
            Box::new(Expr::Lit(Lit::Int(1))),
        ),
    }
}

/// `if counter >= LOOP_GUARD_MAX { debug::print(RUNAWAY_MSG); }` — the
/// after-loop report emitted when the guard, not the user condition, stopped
/// the loop.
pub fn runaway_report(counter: LocalId) -> Stmt {
    Stmt::If {
        cond: Expr::Binary(
            BinOp::Ge,
            Box::new(Expr::Local(counter)),
            Box::new(Expr::Lit(Lit::Int(LOOP_GUARD_MAX))),
        ),
        then_body: vec![Stmt::ExprStmt(Expr::Call {
            path: vec!["debug".into(), "print".into()],
            args: vec![Expr::Lit(Lit::Str(RUNAWAY_MSG.to_string()))],
        })],
        else_body: Vec::new(),
    }
}

/// A recognised `flow.while` expansion.
pub struct GuardedWhile<'a> {
    /// The counter local the guard counts on.
    pub counter: LocalId,
    /// The author's condition — the guard half stripped off.
    pub cond: &'a Expr,
    /// The loop body with the counter increment stripped off.
    pub body: &'a [Stmt],
    /// How many statements of the enclosing block the form occupies (3).
    pub consumed: usize,
}

/// A recognised `flow.for` expansion.
pub struct GuardedFor<'a> {
    /// The induction variable, and the local a body read of the index resolves to.
    pub index: LocalId,
    /// How the index was bound — `Anon` from a graph, `Raw("i")` from text.
    pub index_binding: &'a Binding,
    /// The snapshotted upper bound's local.
    pub bound: LocalId,
    /// The counter local the guard counts on.
    pub counter: LocalId,
    /// The lower bound expression.
    pub first: &'a Expr,
    /// The upper bound expression.
    pub last: &'a Expr,
    /// The loop body with both increments stripped off.
    pub body: &'a [Stmt],
    /// How many statements of the enclosing block the form occupies (5).
    pub consumed: usize,
}

/// Recognise the three-statement `flow.while` expansion at `body[i..]`.
///
/// Deliberately an **exact** match, not a heuristic: a hand-written raw `while`
/// is not this form and must stay unrecognised, so that the text face prints it
/// as what it is rather than inventing a guard the program does not have.
pub fn match_while(body: &[Stmt], i: usize) -> Option<GuardedWhile<'_>> {
    let (s0, s1, s2) = (body.get(i)?, body.get(i + 1)?, body.get(i + 2)?);
    let counter = anon_int_zero(s0, true)?;
    let Stmt::While { cond, body: wbody } = s1 else {
        return None;
    };
    let Expr::Binary(BinOp::And, user_cond, guard) = cond else {
        return None;
    };
    if !is_guard_lt(guard, counter) {
        return None;
    }
    let (last, inner) = wbody.split_last()?;
    if !is_increment(last, counter) {
        return None;
    }
    if !is_runaway_if(s2, counter) {
        return None;
    }
    Some(GuardedWhile {
        counter,
        cond: user_cond,
        body: inner,
        consumed: 3,
    })
}

/// Recognise the five-statement `flow.for` expansion at `body[i..]`.
///
/// Checked **before** [`match_while`] by every consumer, because a `for`
/// expansion contains a guarded `while` at its third statement: matched the
/// other way round, a `for` reads as a `while` whose body ends in an index
/// assignment, which is not a program either half meant to describe.
pub fn match_for(body: &[Stmt], i: usize) -> Option<GuardedFor<'_>> {
    let (s0, s1, s2, s3, s4) = (
        body.get(i)?,
        body.get(i + 1)?,
        body.get(i + 2)?,
        body.get(i + 3)?,
        body.get(i + 4)?,
    );
    // s0: `let mut index = <first>;` — any binding kind (see `index_init`).
    let Stmt::Let {
        id: index,
        binding: index_binding,
        ty: None,
        mutable: true,
        value: first,
    } = s0
    else {
        return None;
    };
    // s1: `let bound = <last>;`
    let Stmt::Let {
        id: bound,
        binding: Binding::Anon,
        ty: None,
        mutable: false,
        value: last,
    } = s1
    else {
        return None;
    };
    // s2: `let mut counter = 0;`
    let counter = anon_int_zero(s2, true)?;
    let Stmt::While { cond, body: wbody } = s3 else {
        return None;
    };
    let Expr::Binary(BinOp::And, user_cond, guard) = cond else {
        return None;
    };
    if !is_guard_lt(guard, counter) {
        return None;
    }
    // The condition is exactly `index <= bound`.
    let Expr::Binary(BinOp::Le, lhs, rhs) = user_cond.as_ref() else {
        return None;
    };
    if !matches!(lhs.as_ref(), Expr::Local(id) if id == index)
        || !matches!(rhs.as_ref(), Expr::Local(id) if id == bound)
    {
        return None;
    }
    // The body ends `index = index + 1; counter = counter + 1;`.
    let (inc_counter, rest) = wbody.split_last()?;
    let (inc_index, inner) = rest.split_last()?;
    if !is_increment(inc_counter, counter) || !is_increment(inc_index, *index) {
        return None;
    }
    if !is_runaway_if(s4, counter) {
        return None;
    }
    Some(GuardedFor {
        index: *index,
        index_binding,
        bound: *bound,
        counter,
        first,
        last,
        body: inner,
        consumed: 5,
    })
}

/// `let [mut] <anon> = 0;` → its id.
fn anon_int_zero(s: &Stmt, want_mut: bool) -> Option<LocalId> {
    match s {
        Stmt::Let {
            id,
            binding: Binding::Anon,
            ty: None,
            mutable,
            value: Expr::Lit(Lit::Int(0)),
        } if *mutable == want_mut => Some(*id),
        _ => None,
    }
}

/// `counter < LOOP_GUARD_MAX`.
fn is_guard_lt(e: &Expr, counter: LocalId) -> bool {
    matches!(e, Expr::Binary(BinOp::Lt, l, r)
        if matches!(l.as_ref(), Expr::Local(id) if *id == counter)
        && matches!(r.as_ref(), Expr::Lit(Lit::Int(v)) if *v == LOOP_GUARD_MAX))
}

/// `<id> = <id> + 1;`.
fn is_increment(s: &Stmt, id: LocalId) -> bool {
    let Stmt::Assign { target, value } = s else {
        return false;
    };
    *target == id
        && matches!(value, Expr::Binary(BinOp::Add, l, r)
            if matches!(l.as_ref(), Expr::Local(x) if *x == id)
            && matches!(r.as_ref(), Expr::Lit(Lit::Int(1))))
}

/// `if counter >= LOOP_GUARD_MAX { debug::print(_) }`.
fn is_runaway_if(s: &Stmt, counter: LocalId) -> bool {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = s
    else {
        return false;
    };
    else_body.is_empty()
        && matches!(cond, Expr::Binary(BinOp::Ge, l, r)
            if matches!(l.as_ref(), Expr::Local(id) if *id == counter)
            && matches!(r.as_ref(), Expr::Lit(Lit::Int(v)) if *v == LOOP_GUARD_MAX))
        && matches!(then_body.as_slice(),
            [Stmt::ExprStmt(Expr::Call { path, .. })]
                if path.as_slice() == ["debug".to_string(), "print".to_string()])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn while_form(counter: LocalId, cond: Expr, body: Vec<Stmt>) -> Vec<Stmt> {
        let mut wbody = body;
        wbody.push(increment(counter));
        vec![
            counter_init(counter),
            Stmt::While {
                cond: Expr::Binary(BinOp::And, Box::new(cond), Box::new(guard_lt(counter))),
                body: wbody,
            },
            runaway_report(counter),
        ]
    }

    fn for_form(
        index: LocalId,
        bound: LocalId,
        counter: LocalId,
        first: Expr,
        last: Expr,
        body: Vec<Stmt>,
    ) -> Vec<Stmt> {
        let mut wbody = body;
        wbody.push(increment(index));
        wbody.push(increment(counter));
        vec![
            index_init(index, Binding::Anon, first),
            bound_init(bound, last),
            counter_init(counter),
            Stmt::While {
                cond: Expr::Binary(
                    BinOp::And,
                    Box::new(index_le(index, bound)),
                    Box::new(guard_lt(counter)),
                ),
                body: wbody,
            },
            runaway_report(counter),
        ]
    }

    /// The point of the module: what the builders build, the matchers read back.
    #[test]
    fn the_matchers_invert_the_builders() {
        let body = vec![Stmt::ExprStmt(Expr::Call {
            path: vec!["debug".into(), "print".into()],
            args: vec![Expr::Lit(Lit::Str("tick".into()))],
        })];
        let w = while_form(LocalId(4), Expr::Lit(Lit::Bool(true)), body.clone());
        let m = match_while(&w, 0).expect("the while form matches");
        assert_eq!(m.counter, LocalId(4));
        assert_eq!(m.cond, &Expr::Lit(Lit::Bool(true)));
        assert_eq!(m.body, body.as_slice());
        assert_eq!(m.consumed, 3);

        let f = for_form(
            LocalId(1),
            LocalId(2),
            LocalId(3),
            Expr::Lit(Lit::Int(0)),
            Expr::Lit(Lit::Int(9)),
            body.clone(),
        );
        let m = match_for(&f, 0).expect("the for form matches");
        assert_eq!(
            (m.index, m.bound, m.counter),
            (LocalId(1), LocalId(2), LocalId(3))
        );
        assert_eq!(m.index_binding, &Binding::Anon);
        assert_eq!(m.first, &Expr::Lit(Lit::Int(0)));
        assert_eq!(m.last, &Expr::Lit(Lit::Int(9)));
        assert_eq!(m.body, body.as_slice());
        assert_eq!(m.consumed, 5);
    }

    /// **A `for` expansion also satisfies `match_while` at its third statement.**
    ///
    /// This is not a curiosity: it is the reason every consumer must try
    /// [`match_for`] first, and it is why `raise` reported `flow.for` as an
    /// *assign* refusal rather than the "while" one the direction memo predicted
    /// — the while matcher caught the tail and then choked on the index
    /// increment inside the body. Asserted so that a later reordering of the two
    /// matchers in any consumer is a red test rather than a silent misreading.
    #[test]
    fn a_for_expansion_looks_like_a_while_from_its_third_statement() {
        let f = for_form(
            LocalId(1),
            LocalId(2),
            LocalId(3),
            Expr::Lit(Lit::Int(0)),
            Expr::Lit(Lit::Int(9)),
            vec![],
        );
        assert!(match_while(&f, 0).is_none(), "not from the first statement");
        let tail = match_while(&f, 2).expect("but it does match from the third");
        // …and the body it hands back still carries the index increment, which is
        // a `Stmt::Assign` — precisely the thing `raise` refuses.
        assert!(matches!(tail.body, [Stmt::Assign { .. }]));
    }

    /// A raw `while` — no counter, no guard, no report — is **not** this shape.
    /// A matcher that accepted it would invent a bound the program lacks.
    #[test]
    fn a_bare_while_is_not_the_guarded_shape() {
        let raw = vec![Stmt::While {
            cond: Expr::Lit(Lit::Bool(true)),
            body: vec![],
        }];
        assert!(match_while(&raw, 0).is_none());
        assert!(match_for(&raw, 0).is_none());
    }

    /// The guard bound is read from [`LOOP_GUARD_MAX`], so a form built against a
    /// different cap does not match. Falsification arm: the matcher is a pin on
    /// the constant, not merely on the shape.
    #[test]
    fn a_different_cap_does_not_match() {
        let counter = LocalId(0);
        let mut w = while_form(counter, Expr::Lit(Lit::Bool(true)), vec![]);
        let Stmt::While { cond, .. } = &mut w[1] else {
            unreachable!()
        };
        *cond = Expr::Binary(
            BinOp::And,
            Box::new(Expr::Lit(Lit::Bool(true))),
            Box::new(Expr::Binary(
                BinOp::Lt,
                Box::new(Expr::Local(counter)),
                Box::new(Expr::Lit(Lit::Int(LOOP_GUARD_MAX - 1))),
            )),
        );
        assert!(match_while(&w, 0).is_none());
    }
}
