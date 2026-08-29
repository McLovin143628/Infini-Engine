//! Raising: a [`BlueprintFn`] IR → a visual blueprint [`Graph`] (ROADMAP P6.5,
//! the reverse of [`crate::lower`]). This is the "hand-edited Rust updates the
//! graph" half of bidirectional sync: edited source → `inf-transpile` lift →
//! IR → `raise` → graph.
//!
//! The round-trip pivot is the IR, exactly as the transpiler's pivot is Rust:
//! **`lower(raise(f)) == f`** for every `f` in lowering's image (linear exec
//! chains + terminal branches — the subset the node kit produces). Structural
//! sugar with no faithful graph form is out of scope and reported, mirroring the
//! transpiler's snippet fallback.
//!
//! Raise coverage of the B-P2 flow palette (SCRIPT1's table; the memo's §4 is
//! the prose version):
//! - **`flow.while` and `flow.for` are inverted** — `try_raise_for` and
//!   `try_raise_while` recognise, through the one shared
//!   [`crate::loopshape`] matcher, the exact counter-guarded expansions the
//!   lowerer emits, and rebuild the node, so `lower(raise(f)) == f` holds for
//!   both loop forms. `for` is tried **first**: a `for` expansion contains a
//!   guarded `while` at its third statement, and matched the other way round it
//!   reads as a while whose body ends in an index assignment — which is how
//!   `flow.for` came to report an *assign* refusal rather than a "while" one
//!   before SCRIPT1 widened it.
//! - **`flow.do_once` / `flow.flip_flop` / `flow.gate` stay raise-excluded**
//!   (like `flow.sequence`'s flattening): their lowered form wraps `nodestate::*`
//!   state in a `Stmt::If`, which has no unambiguous single-node inverse — the
//!   read is a pure call in value position and the wrapper is non-terminal.
//!   Hand-edited Rust in those shapes stays a `Stmt::Snippet` on lift rather
//!   than a graph node — the documented, lossless fallback.

use std::collections::HashMap;

use inf_graph::{Graph, Link, NodeId, NodeUi, ParamValue};

use crate::loopshape;
use crate::nodekit::EXEC_THEN;
use crate::semantics::EventKind;
use crate::{BinOp, BlueprintFn, Expr, Lit, LocalId, Stmt, UnOp};

/// A failure while raising IR to a graph (the IR is outside lowering's image).
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum RaiseError {
    #[error("statement kind cannot be raised to a node ({0})")]
    UnsupportedStmt(&'static str),
    #[error("expression cannot be raised to a node ({0})")]
    UnsupportedExpr(&'static str),
    #[error("statements follow a terminal branch/return — not a linear graph")]
    NonLinear,
    #[error("reference to unbound local n{0}")]
    UnboundLocal(u32),
    /// The statement walk made no progress — see the progress guard in
    /// `Raiser::raise_chain` (private, so this is a name rather than a link:
    /// a public doc that links a private item is a rustdoc warning, and this
    /// crate's count is a ratcheted number). Unreachable by construction; a
    /// **value** rather than a hang if the construction is ever broken again.
    #[error("the statement walk did not advance past statement {0} — this is a bug in `raise`")]
    NoProgress(usize),
}

/// Raise one handler function into a fresh graph rooted at an `event.*` node.
pub fn raise_fn(f: &BlueprintFn) -> Result<Graph, RaiseError> {
    let mut r = Raiser {
        graph: Graph::empty(),
        locals: HashMap::new(),
        params: HashMap::new(),
        y: 0.0,
    };
    let (event_type, param) = event_node_spec(&f.id);
    let event = r.add(&event_type, 0.0);
    // Wave 3: an `event.input`/`event.custom` node carries its action/name as a
    // param, restored here so a re-lower reproduces the same `EventKind`.
    if let Some((key, val)) = param {
        r.set_text(event, key, &val);
    }
    // The event node's data outputs are the handler params (e.g. `dt`, `pressed`,
    // `other`).
    r.params = f.params.iter().map(|p| (p.name.clone(), event)).collect();
    r.raise_chain(&f.body, Some((event, EXEC_THEN.to_string())))?;
    Ok(r.graph)
}

/// The `event.*` node type + optional (`param key`, value) a handler id raises to.
/// The inverse of the lowerer's `event_of`: `input:jump → (event.input, action=
/// jump)`, `custom:foo → (event.custom, name=foo)`, `collision → event.collision`.
fn event_node_spec(fn_id: &str) -> (String, Option<(&'static str, String)>) {
    match fn_id {
        "begin_play" => ("event.begin_play".into(), None),
        "tick" => ("event.tick".into(), None),
        "collision" => ("event.collision".into(), None),
        "water_enter" => ("event.water_enter".into(), None),
        "water_exit" => ("event.water_exit".into(), None),
        "water_splash" => ("event.water_splash".into(), None),
        other => {
            if let Some(action) = other.strip_prefix("input:") {
                ("event.input".into(), Some(("action", action.to_string())))
            } else if let Some(name) = other.strip_prefix("custom:") {
                ("event.custom".into(), Some(("name", name.to_string())))
            } else {
                // Legacy `event.<name>` custom-event fallback (param-less node).
                (format!("event.{other}"), None)
            }
        }
    }
}

/// The [`EventKind`] a handler id denotes (for pairing with a class). Wave 3
/// extends it to `input:`/`collision` ids.
pub fn event_kind_of(fn_id: &str) -> EventKind {
    match fn_id {
        "begin_play" => EventKind::BeginPlay,
        "tick" => EventKind::Tick,
        "collision" => EventKind::Collision,
        "water_enter" => EventKind::WaterEnter,
        "water_exit" => EventKind::WaterExit,
        "water_splash" => EventKind::WaterSplash,
        "destroyed" => EventKind::Destroyed,
        other => {
            if let Some(action) = other.strip_prefix("input:") {
                EventKind::Input(action.to_string())
            } else {
                EventKind::Custom(other.trim_start_matches("custom:").to_string())
            }
        }
    }
}

struct Raiser {
    graph: Graph,
    /// Local id → the (node, output-port) that produces it.
    locals: HashMap<LocalId, (NodeId, String)>,
    params: HashMap<String, NodeId>,
    y: f64,
}

impl Raiser {
    fn add(&mut self, type_id: &str, x: f64) -> NodeId {
        self.y += 90.0;
        let y = self.y;
        self.graph.insert(
            type_id,
            NodeUi {
                x,
                y,
                ..Default::default()
            },
        )
    }

    fn wire(&mut self, from: NodeId, fp: &str, to: NodeId, tp: &str) {
        self.graph.links.push(Link {
            from,
            from_port: fp.into(),
            to,
            to_port: tp.into(),
        });
    }

    fn set_text(&mut self, node: NodeId, key: &str, val: &str) {
        self.graph
            .node_mut(node)
            .unwrap()
            .params
            .insert(key.into(), ParamValue::Text(val.into()));
    }

    /// Raise a statement chain, threading the exec wire from `prev`
    /// (`Some((node, out_port))`) into each successive exec node.
    fn raise_chain(
        &mut self,
        body: &[Stmt],
        mut prev: Option<(NodeId, String)>,
    ) -> Result<(), RaiseError> {
        let mut i = 0;
        // **The progress guard.** SCRIPT1's spawn hang was a `continue` that
        // did not advance `i`, and the way it announced itself was a 7.5 GB
        // allocation failure after a minute of adding graph nodes. That is a
        // *detection*, but it is the worst kind: on a CI runner it is an
        // OOM-killed job with no line number, and in the editor it is a frozen
        // process. So non-termination is made a **value** instead — the loop
        // asserts its own advance, which costs one comparison per statement and
        // turns the same mutation into an immediate, named `Err`.
        let mut previous: Option<usize> = None;
        while i < body.len() {
            if previous == Some(i) {
                return Err(RaiseError::NoProgress(i));
            }
            previous = Some(i);
            // Recognize the canonical guarded loop expansions first; each spans
            // several statements and continues from the loop's `completed`.
            // **`for` before `while`** — a `for` expansion satisfies the while
            // matcher from its third statement (see `loopshape`'s own arm).
            if let Some(consumed) = self.try_raise_for(body, i, &mut prev)? {
                i += consumed;
                continue;
            }
            if let Some(consumed) = self.try_raise_while(body, i, &mut prev)? {
                i += consumed;
                continue;
            }
            let stmt = &body[i];
            let is_last = i + 1 == body.len();
            match stmt {
                Stmt::Let { id, value, .. } => {
                    // Either an action bound to a local (impure call) or a
                    // materialized shared pure expression.
                    if let Expr::Call { path, .. } = value {
                        if is_action_path(path) {
                            let node = self.raise_action(value, &prev)?;
                            // Bind the action's data output.
                            let out = self.first_data_output(node);
                            if let Some(port) = out {
                                self.locals.insert(*id, (node, port));
                            }
                            prev = Some((node, EXEC_THEN.to_string()));
                            // **`i += 1` before the `continue`.** This arm used
                            // to jump straight back to the loop head without
                            // advancing, so `let e = engine::spawn(…)` — the one
                            // action in the kit with a consumed data output —
                            // raised the same statement for ever, adding a node
                            // per pass until the process ran out of memory. It
                            // reached the editor: `graph_open_actor` on any
                            // blueprint holding a bound spawn would hang and
                            // then die. Found by SCRIPT1's honest-subset table,
                            // which is the first thing that ever asked `raise`
                            // about that shape.
                            i += 1;
                            continue;
                        }
                    }
                    // Pure materialization: build the data node, bind the local;
                    // it is not on the exec chain.
                    let (node, port) = self.raise_expr(value)?;
                    self.locals.insert(*id, (node, port));
                }
                Stmt::ExprStmt(Expr::Call { path, .. })
                    if path == &["vars".to_string(), "set".into()] =>
                {
                    let node = self.raise_var_set(stmt, &prev)?;
                    prev = Some((node, EXEC_THEN.to_string()));
                }
                Stmt::ExprStmt(call @ Expr::Call { .. }) => {
                    let node = self.raise_action(call, &prev)?;
                    prev = Some((node, EXEC_THEN.to_string()));
                }
                Stmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    if !is_last {
                        return Err(RaiseError::NonLinear);
                    }
                    let br = self.add("flow.branch", 0.0);
                    self.wire_prev(&prev, br);
                    let c = self.raise_expr(cond)?;
                    self.wire(c.0, &c.1, br, "condition");
                    self.raise_chain(then_body, Some((br, "true".into())))?;
                    self.raise_chain(else_body, Some((br, "false".into())))?;
                    return Ok(());
                }
                Stmt::Return(opt) => {
                    if !is_last {
                        return Err(RaiseError::NonLinear);
                    }
                    let ret = self.add("flow.return", 0.0);
                    self.wire_prev(&prev, ret);
                    if let Some(e) = opt {
                        let v = self.raise_expr(e)?;
                        self.wire(v.0, &v.1, ret, "value");
                    }
                    return Ok(());
                }
                Stmt::ExprStmt(_) => return Err(RaiseError::UnsupportedStmt("non-call expr stmt")),
                Stmt::Assign { .. } => return Err(RaiseError::UnsupportedStmt("assign")),
                // A `While` that isn't the recognized guarded pattern (e.g. a
                // hand-written raw loop) has no faithful single-node inverse.
                Stmt::While { .. } => return Err(RaiseError::UnsupportedStmt("while")),
                Stmt::Snippet(_) => return Err(RaiseError::UnsupportedStmt("snippet")),
            }
            i += 1;
        }
        Ok(())
    }

    /// Recognize the exact counter-guarded `flow.while` expansion the lowerer
    /// emits at `body[i..i+3]` and rebuild the node, threading the outer chain on
    /// from its `completed` output. Returns `Some(3)` (statements consumed) on a
    /// match, `None` otherwise (the statement is handled by the normal chain).
    ///
    /// The recognized shape is precisely [`crate::loopshape`]'s while form,
    /// which is what [`crate::lower`] builds — one definition, three readers.
    fn try_raise_while(
        &mut self,
        body: &[Stmt],
        i: usize,
        prev: &mut Option<(NodeId, String)>,
    ) -> Result<Option<usize>, RaiseError> {
        let Some(m) = loopshape::match_while(body, i) else {
            return Ok(None);
        };
        let (cond, inner, consumed) = (m.cond.clone(), m.body.to_vec(), m.consumed);
        // Rebuild the node.
        let node = self.add("flow.while", 0.0);
        self.wire_prev(prev, node);
        let c = self.raise_expr(&cond)?;
        self.wire(c.0, &c.1, node, "condition");
        self.raise_chain(&inner, Some((node, "loop_body".into())))?;
        *prev = Some((node, "completed".into()));
        Ok(Some(consumed))
    }

    /// Recognize the five-statement `flow.for` expansion at `body[i..i+5]` and
    /// rebuild the node — the [`Self::try_raise_while`] twin, over
    /// [`loopshape::match_for`].
    ///
    /// **Widened in SCRIPT1**, and for a reason the text face made concrete:
    /// `.infini` has a `for` statement, so leaving `flow.for` raise-excluded
    /// would have meant that every script containing a counted loop was
    /// unopenable as a graph — a hole the language itself would have dug. The
    /// shape is a recognition, exactly like the while form; the *expensive* half
    /// of the raise gap ([`RaiseError::NonLinear`]) is untouched and stays named
    /// in the honest-subset table.
    ///
    /// The index local is bound to the node's `index` output **before** the body
    /// is raised, mirroring `lower_for`'s own ordering, so a body read of the
    /// loop variable wires back to the loop rather than dangling.
    fn try_raise_for(
        &mut self,
        body: &[Stmt],
        i: usize,
        prev: &mut Option<(NodeId, String)>,
    ) -> Result<Option<usize>, RaiseError> {
        let Some(m) = loopshape::match_for(body, i) else {
            return Ok(None);
        };
        let (index, first, last, inner, consumed) = (
            m.index,
            m.first.clone(),
            m.last.clone(),
            m.body.to_vec(),
            m.consumed,
        );
        let node = self.add("flow.for", 0.0);
        self.wire_prev(prev, node);
        let f = self.raise_expr(&first)?;
        self.wire(f.0, &f.1, node, "first");
        let l = self.raise_expr(&last)?;
        self.wire(l.0, &l.1, node, "last");
        // The loop variable *is* the node's `index` output.
        self.locals.insert(index, (node, "index".into()));
        self.raise_chain(&inner, Some((node, "loop_body".into())))?;
        *prev = Some((node, "completed".into()));
        Ok(Some(consumed))
    }

    fn wire_prev(&mut self, prev: &Option<(NodeId, String)>, into: NodeId) {
        if let Some((pn, pp)) = prev {
            let (pn, pp) = (*pn, pp.clone());
            self.wire(pn, &pp, into, "exec");
        }
    }

    fn raise_var_set(
        &mut self,
        stmt: &Stmt,
        prev: &Option<(NodeId, String)>,
    ) -> Result<NodeId, RaiseError> {
        let Stmt::ExprStmt(Expr::Call { args, .. }) = stmt else {
            return Err(RaiseError::UnsupportedStmt("var.set shape"));
        };
        let name = str_arg(args.first())?;
        let node = self.add("var.set", 0.0);
        self.set_text(node, "name", &name);
        self.wire_prev(prev, node);
        let value = args
            .get(1)
            .ok_or(RaiseError::UnsupportedStmt("var.set value"))?;
        let v = self.raise_expr(value)?;
        self.wire(v.0, &v.1, node, "value");
        Ok(node)
    }

    fn raise_action(
        &mut self,
        call: &Expr,
        prev: &Option<(NodeId, String)>,
    ) -> Result<NodeId, RaiseError> {
        let Expr::Call { path, args } = call else {
            return Err(RaiseError::UnsupportedExpr("action shape"));
        };
        let type_id = path.join(".");
        let node = self.add(&type_id, 0.0);
        self.wire_prev(prev, node);
        // Wire each arg to the action's data input ports in order.
        let ports = self.data_input_ports(node);
        for (arg, port) in args.iter().zip(ports) {
            let v = self.raise_expr(arg)?;
            self.wire(v.0, &v.1, node, &port);
        }
        Ok(node)
    }

    /// Raise a data expression into a node subtree, returning its (node, port).
    fn raise_expr(&mut self, expr: &Expr) -> Result<(NodeId, String), RaiseError> {
        match expr {
            Expr::Lit(lit) => {
                let (type_id, val) = lit_node(lit);
                let node = self.add(type_id, -160.0);
                self.graph
                    .node_mut(node)
                    .unwrap()
                    .params
                    .insert("value".into(), val);
                Ok((node, "value".into()))
            }
            Expr::Param(name) => {
                let ev = *self
                    .params
                    .get(name)
                    .ok_or(RaiseError::UnsupportedExpr("unknown param"))?;
                Ok((ev, name.clone()))
            }
            Expr::Local(id) => self
                .locals
                .get(id)
                .cloned()
                .ok_or(RaiseError::UnboundLocal(id.0)),
            Expr::Unary(UnOp::Not, inner) => {
                let node = self.add("logic.not", -160.0);
                let a = self.raise_expr(inner)?;
                self.wire(a.0, &a.1, node, "a");
                Ok((node, "out".into()))
            }
            Expr::Unary(UnOp::Neg, inner) => {
                // `-x` is the `math.neg` unary node (round-trips with the lowerer's
                // NegOp), not the old `0 - x` sugar.
                let node = self.add("math.neg", -160.0);
                let a = self.raise_expr(inner)?;
                self.wire(a.0, &a.1, node, "a");
                Ok((node, "out".into()))
            }
            Expr::Binary(op, a, b) => {
                let type_id = binop_type(*op);
                let node = self.add(type_id, -160.0);
                let av = self.raise_expr(a)?;
                self.wire(av.0, &av.1, node, "a");
                let bv = self.raise_expr(b)?;
                self.wire(bv.0, &bv.1, node, "b");
                Ok((node, "out".into()))
            }
            Expr::Call { path, args } if path == &["vars".to_string(), "get".into()] => {
                let name = str_arg(args.first())?;
                let node = self.add("var.get", -160.0);
                self.set_text(node, "name", &name);
                Ok((node, "value".into()))
            }
            // Pure `math::<name>(..)` calls raise back to their `math.<name>`
            // node (the PureCall inverse). Data ports match the node kit's
            // declared input order so a re-lower reproduces the same call.
            Expr::Call { path, args } if path.first().map(String::as_str) == Some("math") => {
                let type_id = path.join(".");
                let node = self.add(&type_id, -160.0);
                for (arg, port) in args.iter().zip(math_data_ports(&type_id)) {
                    let v = self.raise_expr(arg)?;
                    self.wire(v.0, &v.1, node, &port);
                }
                Ok((node, "out".into()))
            }
            Expr::Call { .. } => Err(RaiseError::UnsupportedExpr("pure call")),
        }
    }

    fn data_input_ports(&self, node: NodeId) -> Vec<String> {
        // The kit's action inputs are exec + data; return the data ports in
        // declaration order (angle/prefab/entity/message/…).
        let type_id = self
            .graph
            .node(node)
            .map(|n| n.type_id.clone())
            .unwrap_or_default();
        match type_id.as_str() {
            "engine.set_rotation" => vec!["angle".into()],
            "engine.spawn" => vec!["prefab".into()],
            "engine.destroy" => vec!["entity".into()],
            "debug.print" => vec!["message".into()],
            _ => Vec::new(),
        }
    }

    fn first_data_output(&self, node: NodeId) -> Option<String> {
        let type_id = self.graph.node(node)?.type_id.clone();
        match type_id.as_str() {
            "engine.spawn" => Some("entity".into()),
            _ => None,
        }
    }
}

fn is_action_path(path: &[String]) -> bool {
    matches!(
        path.first().map(String::as_str),
        Some("engine") | Some("debug")
    )
}

/// The declared data-input ports of a `math.*` node, in the node kit's order —
/// so a raised call re-lowers to the identical argument list.
fn math_data_ports(type_id: &str) -> Vec<String> {
    match type_id {
        "math.clamp" => vec!["x".into(), "min".into(), "max".into()],
        "math.lerp" => vec!["a".into(), "b".into(), "t".into()],
        "math.min" | "math.max" | "math.pow" => vec!["a".into(), "b".into()],
        // abs/floor/ceil/round/sqrt/sin/cos/to_int/to_float are single-input.
        _ => vec!["a".into()],
    }
}

fn str_arg(arg: Option<&Expr>) -> Result<String, RaiseError> {
    match arg {
        Some(Expr::Lit(Lit::Str(s))) => Ok(s.clone()),
        _ => Err(RaiseError::UnsupportedExpr("expected string literal")),
    }
}

fn lit_node(lit: &Lit) -> (&'static str, ParamValue) {
    match lit {
        Lit::Float(f) => ("lit.float", ParamValue::Float(*f)),
        Lit::Int(i) => ("lit.int", ParamValue::Int(*i)),
        Lit::Bool(b) => ("lit.bool", ParamValue::Bool(*b)),
        Lit::Str(s) => ("lit.str", ParamValue::Text(s.clone())),
    }
}

fn binop_type(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "math.add",
        BinOp::Sub => "math.sub",
        BinOp::Mul => "math.mul",
        BinOp::Div => "math.div",
        BinOp::Rem => "math.rem",
        BinOp::Eq => "cmp.eq",
        BinOp::Ne => "cmp.ne",
        BinOp::Lt => "cmp.lt",
        BinOp::Le => "cmp.le",
        BinOp::Gt => "cmp.gt",
        BinOp::Ge => "cmp.ge",
        BinOp::And => "logic.and",
        BinOp::Or => "logic.or",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::lower_graph;
    use crate::nodekit::blueprint_registry;

    /// lower(raise(f)) == f for lowering's image — the round-trip invariant.
    fn assert_round_trips(f: &BlueprintFn) {
        let graph = raise_fn(f).expect("raise");
        let reg = blueprint_registry();
        let relowered = lower_graph(&graph, &reg).expect("re-lower");
        assert_eq!(relowered.len(), 1, "one event → one fn");
        assert_eq!(&relowered[0], f, "lower ∘ raise must be the identity on IR");
    }

    #[test]
    fn rotate_on_tick_round_trips() {
        // Build the IR by lowering a known graph, then check raise inverts it.
        use inf_graph::{Graph, Link, NodeId, NodeUi, ParamValue};
        fn wire(g: &mut Graph, from: NodeId, fp: &str, to: NodeId, tp: &str) {
            g.links.push(Link {
                from,
                from_port: fp.into(),
                to,
                to_port: tp.into(),
            });
        }
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let tick = g.insert("event.tick", NodeUi::default());
        let speed = g.insert("var.get", NodeUi::default());
        g.node_mut(speed)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("speed".into()));
        let mul = g.insert("math.mul", NodeUi::default());
        let angle = g.insert("var.get", NodeUi::default());
        g.node_mut(angle)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("angle".into()));
        let add = g.insert("math.add", NodeUi::default());
        let setv = g.insert("var.set", NodeUi::default());
        g.node_mut(setv)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("angle".into()));
        let rot = g.insert("engine.set_rotation", NodeUi::default());
        wire(&mut g, speed, "value", mul, "a");
        wire(&mut g, tick, "dt", mul, "b");
        wire(&mut g, angle, "value", add, "a");
        wire(&mut g, mul, "out", add, "b");
        wire(&mut g, add, "out", setv, "value");
        wire(&mut g, add, "out", rot, "angle");
        wire(&mut g, tick, EXEC_THEN, setv, "exec");
        wire(&mut g, setv, EXEC_THEN, rot, "exec");

        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        assert_round_trips(&f);
    }

    #[test]
    fn branch_round_trips() {
        use crate::{Binding, Param, Ty};
        // begin_play { if 2 > 1 { debug::print("hi"); } }
        let f = BlueprintFn {
            id: "begin_play".into(),
            name: "begin_play".into(),
            params: vec![],
            ret: Ty::Unit,
            body: vec![Stmt::If {
                cond: Expr::Binary(
                    BinOp::Gt,
                    Box::new(Expr::Lit(Lit::Float(2.0))),
                    Box::new(Expr::Lit(Lit::Float(1.0))),
                ),
                then_body: vec![Stmt::ExprStmt(Expr::Call {
                    path: vec!["debug".into(), "print".into()],
                    args: vec![Expr::Lit(Lit::Str("hi".into()))],
                })],
                else_body: vec![],
            }],
        };
        // Silence unused import warnings for Binding/Param in this focused test.
        let _ = (
            Binding::Anon,
            Param {
                name: "x".into(),
                ty: Ty::Float,
            },
        );
        assert_round_trips(&f);
    }

    #[test]
    fn input_event_round_trips() {
        use crate::{Param, Ty};
        // input:jump handler branching on `pressed` — round-trips through the
        // param-carrying `event.input` node (action restored on raise).
        let f = BlueprintFn {
            id: "input:jump".into(),
            name: "input_jump".into(),
            params: vec![Param {
                name: "pressed".into(),
                ty: Ty::Bool,
            }],
            ret: Ty::Unit,
            body: vec![Stmt::If {
                cond: Expr::Param("pressed".into()),
                then_body: vec![Stmt::ExprStmt(Expr::Call {
                    path: vec!["debug".into(), "print".into()],
                    args: vec![Expr::Lit(Lit::Str("jumped".into()))],
                })],
                else_body: vec![],
            }],
        };
        assert_round_trips(&f);
    }

    #[test]
    fn collision_event_round_trips() {
        use crate::{Param, Ty};
        let f = BlueprintFn {
            id: "collision".into(),
            name: "collision".into(),
            params: vec![Param {
                name: "other".into(),
                ty: Ty::Int,
            }],
            ret: Ty::Unit,
            body: vec![Stmt::ExprStmt(Expr::Call {
                path: vec!["debug".into(), "print".into()],
                args: vec![Expr::Lit(Lit::Str("hit".into()))],
            })],
        };
        assert_round_trips(&f);
    }

    #[test]
    fn custom_event_round_trips() {
        use crate::Ty;
        // custom:ping handler — round-trips through the `event.custom` node whose
        // `name` param is restored on raise.
        let f = BlueprintFn {
            id: "custom:ping".into(),
            name: "custom_ping".into(),
            params: vec![],
            ret: Ty::Unit,
            body: vec![Stmt::ExprStmt(Expr::Call {
                path: vec!["debug".into(), "print".into()],
                args: vec![Expr::Lit(Lit::Str("pong".into()))],
            })],
        };
        assert_round_trips(&f);
    }

    #[test]
    fn bare_while_is_still_unsupported() {
        use crate::Ty;
        // A raw `while` that is NOT the guarded pattern (no counter let / runaway
        // if) has no faithful node form — still reported, unchanged.
        let f = BlueprintFn {
            id: "begin_play".into(),
            name: "begin_play".into(),
            params: vec![],
            ret: Ty::Unit,
            body: vec![Stmt::While {
                cond: Expr::Lit(Lit::Bool(false)),
                body: vec![],
            }],
        };
        assert_eq!(raise_fn(&f), Err(RaiseError::UnsupportedStmt("while")));
    }

    fn wire(g: &mut Graph, from: NodeId, fp: &str, to: NodeId, tp: &str) {
        g.links.push(Link {
            from,
            from_port: fp.into(),
            to,
            to_port: tp.into(),
        });
    }

    #[test]
    fn math_calls_round_trip() {
        use inf_graph::{Graph, NodeUi, ParamValue};
        // begin_play → var.set("out", clamp(lerp(sqrt(x), 10, t), lo, hi))
        // exercises unary, ternary, and nested pure math calls.
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let mk = |g: &mut Graph, v: f64| {
            let n = g.insert("lit.float", NodeUi::default());
            g.node_mut(n)
                .unwrap()
                .params
                .insert("value".into(), ParamValue::Float(v));
            n
        };
        let x = mk(&mut g, 16.0);
        let t = mk(&mut g, 0.5);
        let ten = mk(&mut g, 10.0);
        let lo = mk(&mut g, 0.0);
        let hi = mk(&mut g, 6.0);
        let sqrt = g.insert("math.sqrt", NodeUi::default());
        let lerp = g.insert("math.lerp", NodeUi::default());
        let clamp = g.insert("math.clamp", NodeUi::default());
        let setv = g.insert("var.set", NodeUi::default());
        g.node_mut(setv)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("out".into()));
        wire(&mut g, x, "value", sqrt, "a");
        wire(&mut g, sqrt, "out", lerp, "a");
        wire(&mut g, ten, "value", lerp, "b");
        wire(&mut g, t, "value", lerp, "t");
        wire(&mut g, lerp, "out", clamp, "x");
        wire(&mut g, lo, "value", clamp, "min");
        wire(&mut g, hi, "value", clamp, "max");
        wire(&mut g, clamp, "out", setv, "value");
        wire(&mut g, bp, EXEC_THEN, setv, "exec");

        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        assert_round_trips(&f);
    }

    #[test]
    fn neg_round_trips() {
        use inf_graph::{Graph, NodeUi, ParamValue};
        // begin_play → var.set("out", neg(x))
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let x = g.insert("lit.float", NodeUi::default());
        g.node_mut(x)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Float(3.0));
        let neg = g.insert("math.neg", NodeUi::default());
        let setv = g.insert("var.set", NodeUi::default());
        g.node_mut(setv)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("out".into()));
        wire(&mut g, x, "value", neg, "a");
        wire(&mut g, neg, "out", setv, "value");
        wire(&mut g, bp, EXEC_THEN, setv, "exec");
        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        assert_round_trips(&f);
    }

    #[test]
    fn while_loop_round_trips() {
        use inf_graph::{Graph, NodeUi, ParamValue};
        // begin_play → while(n > 0) { n = n - 1 }; the guarded expansion must
        // raise back to a single flow.while and re-lower identically.
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let getn = g.insert("var.get", NodeUi::default());
        g.node_mut(getn)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("n".into()));
        let zero = g.insert("lit.int", NodeUi::default());
        let gt = g.insert("cmp.gt", NodeUi::default());
        let wh = g.insert("flow.while", NodeUi::default());
        let getn2 = g.insert("var.get", NodeUi::default());
        g.node_mut(getn2)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("n".into()));
        let one = g.insert("lit.int", NodeUi::default());
        g.node_mut(one)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Int(1));
        let sub = g.insert("math.sub", NodeUi::default());
        let setn = g.insert("var.set", NodeUi::default());
        g.node_mut(setn)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("n".into()));
        wire(&mut g, getn, "value", gt, "a");
        wire(&mut g, zero, "value", gt, "b");
        wire(&mut g, gt, "out", wh, "condition");
        wire(&mut g, getn2, "value", sub, "a");
        wire(&mut g, one, "value", sub, "b");
        wire(&mut g, sub, "out", setn, "value");
        wire(&mut g, bp, EXEC_THEN, wh, "exec");
        wire(&mut g, wh, "loop_body", setn, "exec");

        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        // Sanity: it really lowered to the guarded shape.
        assert!(matches!(
            f.body.as_slice(),
            [Stmt::Let { .. }, Stmt::While { .. }, Stmt::If { .. }]
        ));
        assert_round_trips(&f);
    }

    /// **The spawn hang.** `let e = engine::spawn("x"); engine::destroy(e);`
    /// looped for ever inside `raise_chain` — the action-bound-to-a-local arm
    /// `continue`d without advancing the statement index, so it raised the same
    /// `Let` on every pass, adding a graph node each time until the process ran
    /// out of memory.
    ///
    /// It was reachable from the editor: `graph_open_actor` raises every handler
    /// of a `.inf_act`, and `engine.spawn` is the one action in the kit with a
    /// consumed data output — the exact shape "spawn something and keep the
    /// entity" produces. Nothing had ever asked `raise` about it, because the
    /// round-trip fixtures all start from a *graph* and none of them wires a
    /// spawn's `entity` output onward.
    ///
    /// The arm asserts the graph rather than merely the absence of a hang: two
    /// action nodes, threaded, with the spawn's entity feeding the destroy.
    #[test]
    fn an_action_bound_to_a_local_raises_and_terminates() {
        use inf_graph::{Graph, NodeUi, ParamValue};
        // begin_play → spawn("enemy") → destroy(entity) — the shape "spawn
        // something and keep the entity", built from a graph so the round-trip
        // invariant applies on lowering's own image.
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let name = g.insert("lit.str", NodeUi::default());
        g.node_mut(name)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Text("enemy".into()));
        let spawn = g.insert("engine.spawn", NodeUi::default());
        let destroy = g.insert("engine.destroy", NodeUi::default());
        wire(&mut g, name, "value", spawn, "prefab");
        wire(&mut g, spawn, "entity", destroy, "entity");
        wire(&mut g, bp, EXEC_THEN, spawn, "exec");
        wire(&mut g, spawn, EXEC_THEN, destroy, "exec");
        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        assert!(
            matches!(f.body.as_slice(), [Stmt::Let { .. }, Stmt::ExprStmt(_)]),
            "{:#?}",
            f.body
        );

        let raised = raise_fn(&f).expect("a bound action raises");
        let types: Vec<&str> = raised.nodes.values().map(|n| n.type_id.as_str()).collect();
        assert!(types.contains(&"engine.spawn"), "{types:?}");
        assert!(types.contains(&"engine.destroy"), "{types:?}");
        // The event, the spawn, its string literal and the destroy — four
        // nodes, not the thousands the runaway produced before it died.
        assert_eq!(raised.nodes.len(), 4, "{types:?}");
        assert_round_trips(&f);
    }

    /// **The progress guard**, and why the arm above needed one beside it.
    ///
    /// Reverting the spawn hang's `i += 1` used to announce itself as a 7.5 GB
    /// allocation failure after a minute of adding graph nodes. That is a
    /// detection of the worst kind: on a CI runner it is an OOM-killed job with
    /// no line number, and in the editor it is a frozen process. `raise_chain`
    /// now asserts its own advance, so the same mutation is an immediate,
    /// named `Err` — a *value*, which is P21's law applied to a walk rather than
    /// to a gameplay node.
    ///
    /// The guard is unreachable by construction, so it is exercised the only way
    /// an unreachable branch can be: by calling the loop over a body whose walk
    /// cannot advance. There is no such body, so the arm pins the *mechanism* —
    /// that the error exists, that it names a statement index, and that an
    /// ordinary chain still raises without tripping it.
    #[test]
    fn the_statement_walk_asserts_its_own_advance() {
        // The mechanism: the variant is a value with a place in it.
        let e = RaiseError::NoProgress(3);
        assert!(e.to_string().contains("did not advance past statement 3"));
        // …and an ordinary multi-statement handler is not tripped by it. Seven
        // statements, so a guard that fired on any repeated index would show.
        let mut body = Vec::new();
        for k in 0..7 {
            body.push(Stmt::ExprStmt(Expr::Call {
                path: vec!["debug".into(), "print".into()],
                args: vec![Expr::Lit(Lit::Str(format!("{k}")))],
            }));
        }
        let f = BlueprintFn {
            id: "begin_play".into(),
            name: "begin_play".into(),
            params: vec![],
            ret: crate::Ty::Unit,
            body,
        };
        let g = raise_fn(&f).expect("a plain chain raises");
        assert_eq!(
            g.nodes
                .values()
                .filter(|n| n.type_id == "debug.print")
                .count(),
            7,
            "the walk visited each statement exactly once"
        );
    }

    /// **SCRIPT1's widening.** `flow.for` now inverts, so a counted loop is a
    /// graph and a script and the same program.
    ///
    /// Before this wave `raise` reported it as `UnsupportedStmt("assign")` — not
    /// the `"while"` the direction memo predicted, because `try_raise_while`
    /// matched the expansion's *tail* from its third statement and then met the
    /// index increment inside the body. `loopshape`'s
    /// `a_for_expansion_looks_like_a_while_from_its_third_statement` pins that
    /// overlap so the ordering here can never be reversed by accident.
    #[test]
    fn for_loop_round_trips() {
        use inf_graph::{Graph, NodeUi, ParamValue};
        // begin_play → for i in 0..=3 { total = total + i } → debug.print("done")
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let lit = |g: &mut Graph, v: i64| {
            let n = g.insert("lit.int", NodeUi::default());
            g.node_mut(n)
                .unwrap()
                .params
                .insert("value".into(), ParamValue::Int(v));
            n
        };
        let first = lit(&mut g, 0);
        let last = lit(&mut g, 3);
        let fo = g.insert("flow.for", NodeUi::default());
        let gettotal = g.insert("var.get", NodeUi::default());
        g.node_mut(gettotal)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("total".into()));
        let add = g.insert("math.add", NodeUi::default());
        let settotal = g.insert("var.set", NodeUi::default());
        g.node_mut(settotal)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("total".into()));
        let done = g.insert("debug.print", NodeUi::default());
        g.node_mut(done)
            .unwrap()
            .params
            .insert("message".into(), ParamValue::Text("done".into()));
        wire(&mut g, first, "value", fo, "first");
        wire(&mut g, last, "value", fo, "last");
        wire(&mut g, gettotal, "value", add, "a");
        // The loop variable feeds the accumulation — the `index` output is what
        // `try_raise_for` has to rebind, so this arm would fail if it did not.
        wire(&mut g, fo, "index", add, "b");
        wire(&mut g, add, "out", settotal, "value");
        wire(&mut g, bp, EXEC_THEN, fo, "exec");
        wire(&mut g, fo, "loop_body", settotal, "exec");
        wire(&mut g, fo, "completed", done, "exec");

        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        // Sanity: it really lowered to the five-statement guarded shape.
        assert!(
            matches!(
                f.body.as_slice(),
                [
                    Stmt::Let { .. },
                    Stmt::Let { .. },
                    Stmt::Let { .. },
                    Stmt::While { .. },
                    Stmt::If { .. },
                    Stmt::ExprStmt(_)
                ]
            ),
            "unexpected lowering: {:#?}",
            f.body
        );
        assert_round_trips(&f);
        // …and the rebuilt graph really carries a `flow.for`, not a while pair.
        let g2 = raise_fn(&f).unwrap();
        assert!(g2.nodes.values().any(|n| n.type_id == "flow.for"));
        assert!(!g2.nodes.values().any(|n| n.type_id == "flow.while"));
    }

    #[test]
    fn while_with_completed_continuation_round_trips() {
        use inf_graph::{Graph, NodeUi, ParamValue};
        // begin_play → while(n > 0) { n = n - 1 } → debug.print("done")
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let getn = g.insert("var.get", NodeUi::default());
        g.node_mut(getn)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("n".into()));
        let zero = g.insert("lit.int", NodeUi::default());
        let gt = g.insert("cmp.gt", NodeUi::default());
        let wh = g.insert("flow.while", NodeUi::default());
        let getn2 = g.insert("var.get", NodeUi::default());
        g.node_mut(getn2)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("n".into()));
        let one = g.insert("lit.int", NodeUi::default());
        g.node_mut(one)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Int(1));
        let sub = g.insert("math.sub", NodeUi::default());
        let setn = g.insert("var.set", NodeUi::default());
        g.node_mut(setn)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("n".into()));
        let done = g.insert("debug.print", NodeUi::default());
        g.node_mut(done)
            .unwrap()
            .params
            .insert("message".into(), ParamValue::Text("done".into()));
        wire(&mut g, getn, "value", gt, "a");
        wire(&mut g, zero, "value", gt, "b");
        wire(&mut g, gt, "out", wh, "condition");
        wire(&mut g, getn2, "value", sub, "a");
        wire(&mut g, one, "value", sub, "b");
        wire(&mut g, sub, "out", setn, "value");
        wire(&mut g, bp, EXEC_THEN, wh, "exec");
        wire(&mut g, wh, "loop_body", setn, "exec");
        wire(&mut g, wh, "completed", done, "exec");

        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        assert_round_trips(&f);
    }
}
