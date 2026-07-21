//! Raising: a [`BlueprintFn`] IR → a visual blueprint [`Graph`] (ROADMAP P6.5,
//! the reverse of [`crate::lower`]). This is the "hand-edited Rust updates the
//! graph" half of bidirectional sync: edited source → `inf-transpile` lift →
//! IR → `raise` → graph.
//!
//! The round-trip pivot is the IR, exactly as the transpiler's pivot is Rust:
//! **`lower(raise(f)) == f`** for every `f` in lowering's image (linear exec
//! chains + terminal branches — the subset the node kit produces). Structural
//! sugar with no faithful graph form (e.g. `while`, `flow.sequence` flattening)
//! is out of scope and reported, mirroring the transpiler's snippet fallback.

use std::collections::HashMap;

use inf_graph::{Graph, Link, NodeId, NodeUi, ParamValue};

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
}

/// Raise one handler function into a fresh graph rooted at an `event.*` node.
pub fn raise_fn(f: &BlueprintFn) -> Result<Graph, RaiseError> {
    let mut r = Raiser {
        graph: Graph::empty(),
        locals: HashMap::new(),
        params: HashMap::new(),
        y: 0.0,
    };
    let event_type = event_type_id(&f.id);
    let event = r.add(&event_type, 0.0);
    // The event node's data outputs are the handler params (e.g. `dt`).
    r.params = f.params.iter().map(|p| (p.name.clone(), event)).collect();
    r.raise_chain(&f.body, Some((event, EXEC_THEN.to_string())))?;
    Ok(r.graph)
}

fn event_type_id(fn_id: &str) -> String {
    match fn_id {
        "begin_play" => "event.begin_play".into(),
        "tick" => "event.tick".into(),
        other => format!("event.{}", other.trim_start_matches("custom:")),
    }
}

/// The [`EventKind`] a handler id denotes (for pairing with a class).
pub fn event_kind_of(fn_id: &str) -> EventKind {
    match fn_id {
        "begin_play" => EventKind::BeginPlay,
        "tick" => EventKind::Tick,
        other => EventKind::Custom(other.trim_start_matches("custom:").to_string()),
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
        for (i, stmt) in body.iter().enumerate() {
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
                Stmt::While { .. } => return Err(RaiseError::UnsupportedStmt("while")),
                Stmt::Snippet(_) => return Err(RaiseError::UnsupportedStmt("snippet")),
            }
        }
        Ok(())
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
                // -x is sugar for 0 - x in the node kit.
                let node = self.add("math.sub", -160.0);
                let zero = self.add("lit.float", -320.0);
                self.graph
                    .node_mut(zero)
                    .unwrap()
                    .params
                    .insert("value".into(), ParamValue::Float(0.0));
                self.wire(zero, "value", node, "a");
                let b = self.raise_expr(inner)?;
                self.wire(b.0, &b.1, node, "b");
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
    fn while_is_unsupported() {
        use crate::Ty;
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
}
