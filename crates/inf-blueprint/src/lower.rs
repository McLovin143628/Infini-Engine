//! Lowering: a visual blueprint [`Graph`] → the [`BlueprintFn`] IR (ROADMAP
//! P6.5). This is the bridge that lets one authored graph both **run** (via the
//! [`interp`](crate::interp)reter) and **generate Rust** (via `inf-transpile`)
//! — they consume the same lowered IR, so they can't drift.
//!
//! The walk starts at an `event.*` node and follows exec pins forward, emitting
//! a statement per exec node; each node's data inputs are pulled recursively
//! into [`Expr`] trees. Impure exec nodes that produce a *consumed* data output
//! (e.g. `engine.spawn` → `entity`) bind it to a local so the side effect runs
//! exactly once; unconsumed outputs stay bare `ExprStmt`s.

use std::collections::HashMap;

use inf_graph::{Graph, NodeId, NodeRegistry, ParamValue, PortType};

use crate::nodekit::{NodeRole, EXEC_THEN};
use crate::semantics::EventKind;
use crate::{
    BinOp, Binding, BlueprintFn, Expr, Lit, LocalId, Param, Stmt, Ty, UnOp, LOOP_GUARD_MAX,
};

/// A failure while lowering a graph to IR.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum LowerError {
    #[error("node {0} has unknown type `{1}`")]
    UnknownType(NodeId, String),
    #[error("node {0} (`{1}`) cannot be lowered here")]
    Unsupported(NodeId, String),
    #[error("required input `{1}` of node {0} is not connected")]
    MissingInput(NodeId, String),
    #[error("exec flow revisits node {0} (cycle)")]
    ExecCycle(NodeId),
}

/// Classify a `type_id` into a [`NodeRole`] for lowering.
pub fn role_of(type_id: &str, has_exec_input: bool) -> NodeRole {
    if type_id.starts_with("event.") {
        NodeRole::Event
    } else if type_id.starts_with("lit.") {
        NodeRole::Literal
    } else if type_id == "math.neg" {
        // `math.neg` is a unary IR node (`-a`), not a `math::neg` call.
        NodeRole::NegOp
    } else if binop_of(type_id).is_some() {
        // Exactly the arith `math.add…rem`, the `cmp.*`, and `logic.and/or`.
        // Every other `math.*` (abs/min/pow/clamp/…) falls through to PureCall.
        NodeRole::BinaryOp
    } else if type_id == "logic.not" {
        NodeRole::NotOp
    } else if type_id == "var.get" {
        NodeRole::VarGet
    } else if type_id == "var.set" {
        NodeRole::VarSet
    } else if type_id == "flow.branch" {
        NodeRole::Branch
    } else if type_id == "flow.sequence" {
        NodeRole::Sequence
    } else if type_id == "flow.return" {
        NodeRole::Return
    } else if type_id == "flow.while" {
        NodeRole::WhileLoop
    } else if type_id == "flow.for" {
        NodeRole::ForLoop
    } else if type_id == "flow.do_once" {
        NodeRole::DoOnce
    } else if type_id == "flow.flip_flop" {
        NodeRole::FlipFlop
    } else if type_id == "flow.gate" {
        NodeRole::Gate
    } else if has_exec_input {
        NodeRole::Action
    } else {
        NodeRole::PureCall
    }
}

fn binop_of(type_id: &str) -> Option<BinOp> {
    Some(match type_id {
        "math.add" => BinOp::Add,
        "math.sub" => BinOp::Sub,
        "math.mul" => BinOp::Mul,
        "math.div" => BinOp::Div,
        "math.rem" => BinOp::Rem,
        "cmp.eq" => BinOp::Eq,
        "cmp.ne" => BinOp::Ne,
        "cmp.lt" => BinOp::Lt,
        "cmp.le" => BinOp::Le,
        "cmp.gt" => BinOp::Gt,
        "cmp.ge" => BinOp::Ge,
        "logic.and" => BinOp::And,
        "logic.or" => BinOp::Or,
        _ => return None,
    })
}

/// Lower every `event.*` node in the graph into one [`BlueprintFn`] each, in
/// deterministic (`NodeId`) order.
pub fn lower_graph(graph: &Graph, reg: &NodeRegistry) -> Result<Vec<BlueprintFn>, LowerError> {
    let mut out = Vec::new();
    for (id, node) in &graph.nodes {
        if node.type_id.starts_with("event.") {
            let event = event_of(&node.type_id);
            out.push(lower_event(graph, reg, *id, &event)?);
        }
    }
    Ok(out)
}

fn event_of(type_id: &str) -> EventKind {
    match type_id {
        "event.begin_play" => EventKind::BeginPlay,
        "event.tick" => EventKind::Tick,
        other => EventKind::Custom(other.trim_start_matches("event.").to_string()),
    }
}

/// Lower a single event node's exec chain into a handler function.
pub fn lower_event(
    graph: &Graph,
    reg: &NodeRegistry,
    event_node: NodeId,
    event: &EventKind,
) -> Result<BlueprintFn, LowerError> {
    let mut lw = Lowerer {
        graph,
        reg,
        next_local: 1,
        locals: HashMap::new(),
        visiting: Vec::new(),
    };
    let body = lw.exec_from(event_node, EXEC_THEN)?;
    let params: Vec<Param> = event.signature();
    Ok(BlueprintFn {
        id: event.key(),
        name: event.key(),
        params,
        ret: Ty::Unit,
        body,
    })
}

struct Lowerer<'a> {
    graph: &'a Graph,
    reg: &'a NodeRegistry,
    next_local: u32,
    /// (producer node, output port) → the local it was bound to.
    locals: HashMap<(NodeId, String), LocalId>,
    visiting: Vec<NodeId>,
}

impl Lowerer<'_> {
    fn type_id(&self, node: NodeId) -> Result<&str, LowerError> {
        self.graph
            .node(node)
            .map(|n| n.type_id.as_str())
            .ok_or(LowerError::UnknownType(node, String::new()))
    }

    fn role(&self, node: NodeId) -> Result<NodeRole, LowerError> {
        let type_id = self.type_id(node)?;
        let has_exec_in = self
            .reg
            .get(type_id)
            .map(|d| d.inputs.iter().any(|p| p.ty.is_exec()))
            .unwrap_or(false);
        Ok(role_of(type_id, has_exec_in))
    }

    /// Follow the exec wire out of `(node, out_port)` and lower the chain it
    /// leads to. An unconnected exec output ends the chain. The **input** port
    /// the wire lands on is carried through as the next node's `entry_port`, so
    /// multi-entry nodes (`flow.gate`, `flow.do_once`) can lower per entry.
    fn exec_from(&mut self, node: NodeId, out_port: &str) -> Result<Vec<Stmt>, LowerError> {
        // Exec flow is single-threaded: take the one node this output leads to.
        match self.graph.links_from(node, out_port).next() {
            Some(link) => {
                // The consumed node is `link.to`, entered at `link.to_port`.
                let (to, entry_port) = (link.to, link.to_port.clone());
                self.exec_node(to, &entry_port)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Lower an exec node plus everything downstream of it, entered at
    /// `entry_port` (the input port the incoming exec wire landed on; `"exec"`
    /// for the single-entry nodes, which ignore it).
    fn exec_node(&mut self, node: NodeId, entry_port: &str) -> Result<Vec<Stmt>, LowerError> {
        if self.visiting.contains(&node) {
            return Err(LowerError::ExecCycle(node));
        }
        self.visiting.push(node);
        let result = self.exec_node_inner(node, entry_port);
        self.visiting.pop();
        result
    }

    fn exec_node_inner(&mut self, node: NodeId, entry_port: &str) -> Result<Vec<Stmt>, LowerError> {
        let type_id = self.type_id(node)?.to_string();
        // `out` collects any `let`s that materialize shared pure nodes (the
        // prelude) followed by this node's own statement(s).
        let mut out = Vec::new();
        match self.role(node)? {
            NodeRole::Branch => {
                let cond = self.resolve_input(node, "condition", &mut out)?;
                let then_body = self.exec_from(node, "true")?;
                let else_body = self.exec_from(node, "false")?;
                out.push(Stmt::If {
                    cond,
                    then_body,
                    else_body,
                });
                Ok(out)
            }
            NodeRole::Sequence => {
                let mut body = self.exec_from(node, "then0")?;
                body.extend(self.exec_from(node, "then1")?);
                Ok(body)
            }
            NodeRole::Return => {
                let value = match self.graph.link_into(node, "value") {
                    Some(_) => Some(self.resolve_input(node, "value", &mut out)?),
                    None => None,
                };
                out.push(Stmt::Return(value));
                Ok(out)
            }
            NodeRole::VarSet => {
                let name = self.string_param(node, "name");
                let value = self.resolve_input(node, "value", &mut out)?;
                out.push(Stmt::ExprStmt(Expr::Call {
                    path: vec!["vars".into(), "set".into()],
                    args: vec![Expr::Lit(Lit::Str(name)), value],
                }));
                out.extend(self.exec_from(node, EXEC_THEN)?);
                Ok(out)
            }
            NodeRole::Action => {
                let call = self.build_call(node, &type_id, &mut out)?;
                // If the action produces a consumed data output, bind it so the
                // side effect runs exactly once and downstream reads the local.
                match self.consumed_output(node) {
                    Some(port) => {
                        let id = self.alloc_local();
                        self.locals.insert((node, port), id);
                        out.push(Stmt::Let {
                            id,
                            binding: Binding::Anon,
                            ty: None,
                            mutable: false,
                            value: call,
                        });
                    }
                    None => out.push(Stmt::ExprStmt(call)),
                }
                out.extend(self.exec_from(node, EXEC_THEN)?);
                Ok(out)
            }
            NodeRole::WhileLoop => self.lower_while(node, &mut out).map(|()| out),
            NodeRole::ForLoop => self.lower_for(node, &mut out).map(|()| out),
            NodeRole::DoOnce => self.lower_do_once(node, entry_port, &mut out).map(|()| out),
            NodeRole::FlipFlop => self.lower_flip_flop(node, &mut out).map(|()| out),
            NodeRole::Gate => self.lower_gate(node, entry_port, &mut out).map(|()| out),
            other => Err(LowerError::Unsupported(
                node,
                format!("{type_id} ({other:?})"),
            )),
        }
    }

    /// `flow.while`: a counter-guarded loop.
    ///
    /// ```text
    /// let mut counter = 0;
    /// while (user_cond && counter < LOOP_GUARD_MAX) { <body>; counter = counter + 1; }
    /// if counter >= LOOP_GUARD_MAX { debug::print("Runaway loop stopped (<node>)"); }
    /// <completed continuation>
    /// ```
    ///
    /// The guard lives **in the IR** so the interpreter and the transpiled Rust
    /// share the exact same bound. NOTE: the user condition is resolved fresh so
    /// it re-evaluates each iteration; a condition node that *also* fans out
    /// (≥2 consumers) materializes into a `let` hoisted before the loop — an
    /// accepted edge-case limitation of the shared-value model, not the norm.
    fn lower_while(&mut self, node: NodeId, out: &mut Vec<Stmt>) -> Result<(), LowerError> {
        let counter = self.alloc_local();
        let user_cond = self.resolve_input(node, "condition", out)?;
        out.push(Stmt::Let {
            id: counter,
            binding: Binding::Anon,
            ty: None,
            mutable: true,
            value: Expr::Lit(Lit::Int(0)),
        });
        let cond = Expr::Binary(BinOp::And, Box::new(user_cond), Box::new(guard_lt(counter)));
        let mut body = self.exec_from(node, "loop_body")?;
        body.push(increment(counter));
        out.push(Stmt::While { cond, body });
        out.push(runaway_report(counter));
        out.extend(self.exec_from(node, "completed")?);
        Ok(())
    }

    /// `flow.for`: `index in first..=last`, counter-guarded. `first`/`last` are
    /// snapshotted into locals (evaluated once) and the `index` output is
    /// registered **before** the body is lowered so body reads resolve to it.
    fn lower_for(&mut self, node: NodeId, out: &mut Vec<Stmt>) -> Result<(), LowerError> {
        let idx = self.alloc_local();
        let last_local = self.alloc_local();
        let counter = self.alloc_local();
        // Register the index output up front so `loop_body` reads see the local.
        self.locals.insert((node, "index".to_string()), idx);
        let first = self.resolve_input(node, "first", out)?;
        let last = self.resolve_input(node, "last", out)?;
        out.push(Stmt::Let {
            id: idx,
            binding: Binding::Anon,
            ty: None,
            mutable: true,
            value: first,
        });
        out.push(Stmt::Let {
            id: last_local,
            binding: Binding::Anon,
            ty: None,
            mutable: false,
            value: last,
        });
        out.push(Stmt::Let {
            id: counter,
            binding: Binding::Anon,
            ty: None,
            mutable: true,
            value: Expr::Lit(Lit::Int(0)),
        });
        let cond = Expr::Binary(
            BinOp::And,
            Box::new(Expr::Binary(
                BinOp::Le,
                Box::new(Expr::Local(idx)),
                Box::new(Expr::Local(last_local)),
            )),
            Box::new(guard_lt(counter)),
        );
        let mut body = self.exec_from(node, "loop_body")?;
        body.push(increment(idx));
        body.push(increment(counter));
        out.push(Stmt::While { cond, body });
        out.push(runaway_report(counter));
        out.extend(self.exec_from(node, "completed")?);
        Ok(())
    }

    /// `flow.do_once`: entered at `exec` it fires `then` the first time only;
    /// entered at `reset` it re-arms. State is a `Bool` in `nodestate::*`.
    fn lower_do_once(
        &mut self,
        node: NodeId,
        entry_port: &str,
        out: &mut Vec<Stmt>,
    ) -> Result<(), LowerError> {
        let key = format!("__bp_once_{node}");
        if entry_port == "reset" {
            out.push(nodestate_set(&key, Expr::Lit(Lit::Bool(false))));
            return Ok(());
        }
        // exec entry: if !fired { fired = true; <then> }
        let mut then_body = vec![nodestate_set(&key, Expr::Lit(Lit::Bool(true)))];
        then_body.extend(self.exec_from(node, EXEC_THEN)?);
        out.push(Stmt::If {
            cond: Expr::Unary(
                UnOp::Not,
                Box::new(nodestate_get_or(&key, Expr::Lit(Lit::Bool(false)))),
            ),
            then_body,
            else_body: Vec::new(),
        });
        Ok(())
    }

    /// `flow.flip_flop`: alternate `a`/`b` on each entry. The state read is
    /// materialized into a `let` first (so `is_a` and the branch agree on *this*
    /// invocation), then the stored value is toggled for next time.
    fn lower_flip_flop(&mut self, node: NodeId, out: &mut Vec<Stmt>) -> Result<(), LowerError> {
        let key = format!("__bp_flip_{node}");
        let is_a = self.alloc_local();
        out.push(Stmt::Let {
            id: is_a,
            binding: Binding::Anon,
            ty: None,
            mutable: false,
            value: nodestate_get_or(&key, Expr::Lit(Lit::Bool(true))),
        });
        // Register the `is_a` data output before lowering the branches.
        self.locals.insert((node, "is_a".to_string()), is_a);
        out.push(nodestate_set(
            &key,
            Expr::Unary(UnOp::Not, Box::new(Expr::Local(is_a))),
        ));
        let then_body = self.exec_from(node, "a")?;
        let else_body = self.exec_from(node, "b")?;
        out.push(Stmt::If {
            cond: Expr::Local(is_a),
            then_body,
            else_body,
        });
        Ok(())
    }

    /// `flow.gate`: `enter` reaches `exit` only while open; `open`/`close`/
    /// `toggle` mutate the stored `Bool` (default = the `start_open` param).
    fn lower_gate(
        &mut self,
        node: NodeId,
        entry_port: &str,
        out: &mut Vec<Stmt>,
    ) -> Result<(), LowerError> {
        let key = format!("__bp_gate_{node}");
        let start_open = self.bool_param(node, "start_open");
        let default = || Expr::Lit(Lit::Bool(start_open));
        match entry_port {
            "open" => out.push(nodestate_set(&key, Expr::Lit(Lit::Bool(true)))),
            "close" => out.push(nodestate_set(&key, Expr::Lit(Lit::Bool(false)))),
            "toggle" => out.push(nodestate_set(
                &key,
                Expr::Unary(UnOp::Not, Box::new(nodestate_get_or(&key, default()))),
            )),
            _ => {
                // "enter": if open { <exit chain> }
                let then_body = self.exec_from(node, "exit")?;
                out.push(Stmt::If {
                    cond: nodestate_get_or(&key, default()),
                    then_body,
                    else_body: Vec::new(),
                });
            }
        }
        Ok(())
    }

    /// The first output data port of `node` that some link consumes, if any.
    fn consumed_output(&self, node: NodeId) -> Option<String> {
        let def = self.reg.get(self.graph.node(node)?.type_id.as_str())?;
        def.outputs
            .iter()
            .filter(|p| !p.ty.is_exec())
            .map(|p| p.name.clone())
            .find(|port| self.graph.links_from(node, port).next().is_some())
    }

    /// Build the host [`Expr::Call`] for an action node: path = type_id
    /// segments, args = its data inputs in declared order.
    fn build_call(
        &mut self,
        node: NodeId,
        type_id: &str,
        prelude: &mut Vec<Stmt>,
    ) -> Result<Expr, LowerError> {
        let path: Vec<String> = type_id.split('.').map(str::to_string).collect();
        let ports = self.data_input_ports(node);
        let mut args = Vec::new();
        for port in ports {
            args.push(self.resolve_input(node, &port, prelude)?);
        }
        Ok(Expr::Call { path, args })
    }

    /// How many non-exec data outputs `node` declares.
    fn data_output_count(&self, node: NodeId) -> usize {
        self.graph
            .node(node)
            .and_then(|n| self.reg.get(&n.type_id))
            .map(|def| def.outputs.iter().filter(|p| !p.ty.is_exec()).count())
            .unwrap_or(0)
    }

    /// The non-exec, non-param-pin input port names of `node`, in order.
    fn data_input_ports(&self, node: NodeId) -> Vec<String> {
        self.graph
            .node(node)
            .and_then(|n| self.reg.get(&n.type_id))
            .map(|def| {
                def.inputs
                    .iter()
                    .filter(|p| !p.ty.is_exec() && !p.param_pin)
                    .map(|p| p.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Resolve a data input to an [`Expr`]: follow its wire, or fall back to a
    /// type-appropriate literal default when unconnected.
    fn resolve_input(
        &mut self,
        node: NodeId,
        port: &str,
        prelude: &mut Vec<Stmt>,
    ) -> Result<Expr, LowerError> {
        if let Some(link) = self.graph.link_into(node, port) {
            let (src, src_port) = (link.from, link.from_port.clone());
            self.resolve_output(src, &src_port, prelude)
        } else {
            Ok(default_literal(self.input_type(node, port)))
        }
    }

    /// Resolve a data *output* of a producer node into an [`Expr`].
    ///
    /// A non-trivial pure node whose output fans out to **two or more**
    /// consumers is materialized into a `let` (pushed onto `prelude`) and read
    /// back as a [`Expr::Local`], so the wire's value is computed exactly once
    /// — matching the dataflow "evaluate a node once" model and, crucially,
    /// staying stable across a later member-variable mutation.
    fn resolve_output(
        &mut self,
        node: NodeId,
        port: &str,
        prelude: &mut Vec<Stmt>,
    ) -> Result<Expr, LowerError> {
        // A previously-bound output (materialized pure, or an impure action's
        // data output) reads from its local.
        if let Some(id) = self.locals.get(&(node, port.to_string())) {
            return Ok(Expr::Local(*id));
        }
        let type_id = self.type_id(node)?.to_string();
        let role = self.role(node)?;
        let expr = match role {
            NodeRole::Event => return Ok(Expr::Param(port.to_string())),
            NodeRole::Literal => Expr::Lit(self.literal(node)),
            NodeRole::BinaryOp => {
                let op = binop_of(&type_id)
                    .ok_or_else(|| LowerError::Unsupported(node, type_id.clone()))?;
                let a = self.resolve_input(node, "a", prelude)?;
                let b = self.resolve_input(node, "b", prelude)?;
                Expr::Binary(op, Box::new(a), Box::new(b))
            }
            NodeRole::NotOp => {
                let a = self.resolve_input(node, "a", prelude)?;
                Expr::Unary(UnOp::Not, Box::new(a))
            }
            NodeRole::NegOp => {
                let a = self.resolve_input(node, "a", prelude)?;
                Expr::Unary(UnOp::Neg, Box::new(a))
            }
            NodeRole::VarGet => {
                let name = self.string_param(node, "name");
                Expr::Call {
                    path: vec!["vars".into(), "get".into()],
                    args: vec![Expr::Lit(Lit::Str(name))],
                }
            }
            NodeRole::PureCall => {
                let mut call = self.build_call(node, &type_id, prelude)?;
                // A pure node with several data outputs (e.g. `physics2d.raycast`
                // → hit/point/normal, `physics2d.get_velocity` → x/y) fans each
                // output pin to its own `…::<field>` call, so a single wire
                // carries a single scalar. Single-output pure calls keep their
                // bare path (`vars::get`, `physics2d::is_grounded`).
                if self.data_output_count(node) > 1 {
                    if let Expr::Call { path, .. } = &mut call {
                        path.push(port.to_string());
                    }
                }
                call
            }
            // An action's data output must have been bound to a local when the
            // action ran; reaching here means it wasn't on the exec path.
            NodeRole::Action => {
                return Err(LowerError::Unsupported(
                    node,
                    format!("{type_id}: data output read but node is not on the exec path"),
                ))
            }
            other => {
                return Err(LowerError::Unsupported(
                    node,
                    format!("{type_id} ({other:?})"),
                ))
            }
        };

        // Literals are constants — safe to inline even when shared. Every other
        // pure node with fan-out ≥ 2 is materialized so it evaluates once.
        let shared = self.graph.links_from(node, port).count() >= 2;
        if shared && role != NodeRole::Literal {
            let id = self.alloc_local();
            self.locals.insert((node, port.to_string()), id);
            prelude.push(Stmt::Let {
                id,
                binding: Binding::Anon,
                ty: None,
                mutable: false,
                value: expr,
            });
            Ok(Expr::Local(id))
        } else {
            Ok(expr)
        }
    }

    fn literal(&self, node: NodeId) -> Lit {
        let params = self.graph.node(node).map(|n| &n.params);
        match params.and_then(|p| p.get("value")) {
            Some(ParamValue::Float(f)) => Lit::Float(*f),
            Some(ParamValue::Int(i)) => Lit::Int(*i),
            Some(ParamValue::Bool(b)) => Lit::Bool(*b),
            Some(ParamValue::Text(s)) | Some(ParamValue::Enum(s)) => Lit::Str(s.clone()),
            None => {
                // No stored override → the registry default for a `lit.*` node.
                match self.type_id(node).unwrap_or("") {
                    "lit.int" => Lit::Int(0),
                    "lit.bool" => Lit::Bool(false),
                    "lit.str" => Lit::Str(String::new()),
                    _ => Lit::Float(0.0),
                }
            }
        }
    }

    fn string_param(&self, node: NodeId, name: &str) -> String {
        match self.graph.node(node).and_then(|n| n.params.get(name)) {
            Some(ParamValue::Text(s)) | Some(ParamValue::Enum(s)) => s.clone(),
            _ => String::new(),
        }
    }

    /// A `Bool` param override, defaulting to the registry default (fetched from
    /// the def) or `false` when the node/param is unknown.
    fn bool_param(&self, node: NodeId, name: &str) -> bool {
        if let Some(ParamValue::Bool(b)) = self.graph.node(node).and_then(|n| n.params.get(name)) {
            return *b;
        }
        // No stored override → the registry default (e.g. gate.start_open = true).
        self.graph
            .node(node)
            .and_then(|n| self.reg.get(&n.type_id))
            .and_then(|def| def.param(name))
            .map(|pd| matches!(pd.default, ParamValue::Bool(true)))
            .unwrap_or(false)
    }

    fn input_type(&self, node: NodeId, port: &str) -> PortType {
        self.graph
            .node(node)
            .and_then(|n| self.reg.get(&n.type_id))
            .and_then(|def| def.input(port))
            .map(|p| p.ty.clone())
            .unwrap_or(PortType::Float)
    }

    fn alloc_local(&mut self) -> LocalId {
        let id = LocalId(self.next_local);
        self.next_local += 1;
        id
    }
}

fn default_literal(ty: PortType) -> Expr {
    Expr::Lit(match ty {
        PortType::Int => Lit::Int(0),
        PortType::Bool => Lit::Bool(false),
        PortType::Str => Lit::Str(String::new()),
        _ => Lit::Float(0.0),
    })
}

/// The loop-guard sub-condition `counter < LOOP_GUARD_MAX`.
fn guard_lt(counter: LocalId) -> Expr {
    Expr::Binary(
        BinOp::Lt,
        Box::new(Expr::Local(counter)),
        Box::new(Expr::Lit(Lit::Int(LOOP_GUARD_MAX))),
    )
}

/// `counter = counter + 1;`
fn increment(counter: LocalId) -> Stmt {
    Stmt::Assign {
        target: counter,
        value: Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Local(counter)),
            Box::new(Expr::Lit(Lit::Int(1))),
        ),
    }
}

/// `if counter >= LOOP_GUARD_MAX { debug::print(RUNAWAY_MSG); }` — the after-loop
/// report emitted when the guard, not the user condition, stopped the loop.
///
/// The message is a **constant**, deliberately node-agnostic: it is a synthetic
/// statement `raise` discards and re-lowering regenerates, so embedding the
/// (raise-renumbered, non-user-facing) `NodeId` would break `lower(raise(f)) ==
/// f`. A stable string keeps the loop guarded *and* round-trippable.
fn runaway_report(counter: LocalId) -> Stmt {
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

/// The exact message the loop guard prints when it trips (constant so it
/// round-trips through `raise`).
pub const RUNAWAY_MSG: &str = "Runaway loop stopped (blueprint loop guard exceeded)";

/// `nodestate::get_or(key, default)` — read persisted node state or a default.
fn nodestate_get_or(key: &str, default: Expr) -> Expr {
    Expr::Call {
        path: vec!["nodestate".into(), "get_or".into()],
        args: vec![Expr::Lit(Lit::Str(key.to_string())), default],
    }
}

/// `nodestate::set(key, value);` — persist node state across invocations.
fn nodestate_set(key: &str, value: Expr) -> Stmt {
    Stmt::ExprStmt(Expr::Call {
        path: vec!["nodestate".into(), "set".into()],
        args: vec![Expr::Lit(Lit::Str(key.to_string())), value],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::{eval_fn, FnHost, Value};
    use crate::nodekit::blueprint_registry;
    use inf_graph::{Link, NodeUi};
    use std::collections::HashMap;

    fn wire(g: &mut Graph, from: NodeId, fp: &str, to: NodeId, tp: &str) {
        g.links.push(Link {
            from,
            from_port: fp.into(),
            to,
            to_port: tp.into(),
        });
    }

    #[test]
    fn lowers_rotate_on_tick() {
        // tick.dt → mul.b, var.get(speed) → mul.a, var.get(angle)+mul → set angle,
        // then engine.set_rotation(angle).
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

        // data
        wire(&mut g, speed, "value", mul, "a");
        wire(&mut g, tick, "dt", mul, "b");
        wire(&mut g, angle, "value", add, "a");
        wire(&mut g, mul, "out", add, "b");
        wire(&mut g, add, "out", setv, "value");
        wire(&mut g, add, "out", rot, "angle");
        // exec
        wire(&mut g, tick, EXEC_THEN, setv, "exec");
        wire(&mut g, setv, EXEC_THEN, rot, "exec");

        let fns = lower_graph(&g, &reg).unwrap();
        assert_eq!(fns.len(), 1);
        let f = &fns[0];
        assert_eq!(f.name, "tick");
        assert_eq!(f.params[0].name, "dt");
        // add.out fans out to var.set + set_rotation → materialized into a
        // `let`, so: let n = angle + speed*dt; set angle=n; set_rotation(n).
        assert_eq!(f.body.len(), 3);
        assert!(matches!(f.body[0], Stmt::Let { .. }));

        // Interpret it: angle starts 0, speed 90, dt 0.5 → angle 45, rotation 45.
        let mut vars: HashMap<String, Value> = [
            ("angle".into(), Value::Float(0.0)),
            ("speed".into(), Value::Float(90.0)),
        ]
        .into();
        let mut last = None;
        let mut host = FnHost(|path: &[String], args: &[Value]| match path {
            p if p == ["vars", "get"] => Ok(vars.get(args[0].as_str().unwrap()).cloned().unwrap()),
            p if p == ["vars", "set"] => {
                vars.insert(args[0].as_str().unwrap().to_string(), args[1].clone());
                Ok(Value::Unit)
            }
            p if p == ["engine", "set_rotation"] => {
                last = Some(args[0].as_float().unwrap());
                Ok(Value::Unit)
            }
            _ => Ok(Value::Unit),
        });
        let args: HashMap<String, Value> = [("dt".into(), Value::Float(0.5))].into();
        eval_fn(f, &args, &mut host).unwrap();
        assert_eq!(last, Some(45.0));
    }

    /// A tiny physics host for exercising the character kit end-to-end: one
    /// entity with a velocity + position; `move_and_slide` integrates it.
    struct MockPhysics {
        vx: f64,
        vy: f64,
        x: f64,
        y: f64,
    }

    impl crate::interp::Host for MockPhysics {
        fn call(&mut self, path: &[String], _args: &[Value]) -> Result<Value, crate::RunError> {
            Err(crate::RunError::NoSuchHostFn(path.join("::")))
        }
        fn physics(&mut self) -> Option<&mut dyn crate::interp::Physics2dHost> {
            Some(self)
        }
    }

    impl crate::interp::Physics2dHost for MockPhysics {
        fn move_and_slide(
            &mut self,
            _entity: i64,
            motion: [f64; 2],
        ) -> Result<crate::interp::MoveResult2d, String> {
            self.x += motion[0];
            self.y += motion[1];
            Ok(crate::interp::MoveResult2d {
                applied: motion,
                grounded: self.y <= 0.0,
            })
        }
        fn is_grounded(&mut self, _entity: i64) -> Result<bool, String> {
            Ok(self.y <= 0.0)
        }
        fn raycast(
            &mut self,
            _o: [f64; 2],
            _d: [f64; 2],
            _m: f64,
        ) -> Result<Option<crate::interp::RayHit2d>, String> {
            Ok(None)
        }
        fn set_velocity(&mut self, _entity: i64, v: [f64; 2]) -> Result<(), String> {
            self.vx = v[0];
            self.vy = v[1];
            Ok(())
        }
        fn get_velocity(&mut self, _entity: i64) -> Result<[f64; 2], String> {
            Ok([self.vx, self.vy])
        }
        fn apply_impulse(&mut self, _entity: i64, v: [f64; 2]) -> Result<(), String> {
            self.vx += v[0];
            self.vy += v[1];
            Ok(())
        }
    }

    #[test]
    fn lowers_and_runs_character_kit() {
        // begin_play → set_velocity(e, 4, -2); move_and_slide(e, get_velocity(e).x,
        //                                                     get_velocity(e).y)
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let ent = g.insert("lit.int", NodeUi::default());
        g.node_mut(ent)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Int(1));
        let vx = g.insert("lit.float", NodeUi::default());
        g.node_mut(vx)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Float(4.0));
        let vy = g.insert("lit.float", NodeUi::default());
        g.node_mut(vy)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Float(-2.0));
        let sv = g.insert("physics2d.set_velocity", NodeUi::default());
        let getv = g.insert("physics2d.get_velocity", NodeUi::default());
        let mas = g.insert("physics2d.move_and_slide", NodeUi::default());

        wire(&mut g, ent, "value", sv, "entity");
        wire(&mut g, vx, "value", sv, "vx");
        wire(&mut g, vy, "value", sv, "vy");
        wire(&mut g, ent, "value", getv, "entity");
        wire(&mut g, ent, "value", mas, "entity");
        wire(&mut g, getv, "x", mas, "motion_x");
        wire(&mut g, getv, "y", mas, "motion_y");
        wire(&mut g, bp, EXEC_THEN, sv, "exec");
        wire(&mut g, sv, EXEC_THEN, mas, "exec");

        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();

        // The split get_velocity outputs lower to per-field calls.
        let src = format!("{:?}", f.body);
        assert!(src.contains("get_velocity"), "body: {src}");

        let mut host = MockPhysics {
            vx: 0.0,
            vy: 0.0,
            x: 0.0,
            y: 0.0,
        };
        eval_fn(&f, &HashMap::new(), &mut host).unwrap();
        // set_velocity(4,-2) → get_velocity=(4,-2) → move_and_slide adds it once.
        assert_eq!(host.vx, 4.0);
        assert_eq!(host.x, 4.0);
        assert_eq!(host.y, -2.0);
    }

    #[test]
    fn physics_node_without_a_world_errors_cleanly() {
        use crate::interp::PureHost;
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let ent = g.insert("lit.int", NodeUi::default());
        let sv = g.insert("physics2d.set_velocity", NodeUi::default());
        wire(&mut g, ent, "value", sv, "entity");
        wire(&mut g, bp, EXEC_THEN, sv, "exec");
        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        let err = eval_fn(&f, &HashMap::new(), &mut PureHost).unwrap_err();
        assert!(
            matches!(&err, crate::RunError::Host(_, m) if m.contains("no physics world")),
            "got {err:?}"
        );
    }

    /// An audio host that records the command stream (the mock the P12.3 headless
    /// command-stream test asserts against).
    #[derive(Default)]
    struct MockAudio {
        commands: Vec<String>,
    }

    impl crate::interp::Host for MockAudio {
        fn call(&mut self, path: &[String], _args: &[Value]) -> Result<Value, crate::RunError> {
            Err(crate::RunError::NoSuchHostFn(path.join("::")))
        }
        fn audio(&mut self) -> Option<&mut dyn crate::interp::AudioHost> {
            Some(self)
        }
    }

    impl crate::interp::AudioHost for MockAudio {
        fn play(&mut self, entity: i64) -> Result<(), String> {
            self.commands.push(format!("play {entity}"));
            Ok(())
        }
        fn stop(&mut self, entity: i64) -> Result<(), String> {
            self.commands.push(format!("stop {entity}"));
            Ok(())
        }
        fn set_volume(&mut self, entity: i64, volume: f64) -> Result<(), String> {
            self.commands.push(format!("set_volume {entity} {volume}"));
            Ok(())
        }
        fn set_pitch(&mut self, entity: i64, pitch: f64) -> Result<(), String> {
            self.commands.push(format!("set_pitch {entity} {pitch}"));
            Ok(())
        }
    }

    #[test]
    fn lowers_and_runs_audio_kit_recording_the_command_stream() {
        // begin_play → audio.play(e); audio.set_volume(e, 0.5); audio.stop(e)
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let ent = g.insert("lit.int", NodeUi::default());
        g.node_mut(ent)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Int(7));
        let vol = g.insert("lit.float", NodeUi::default());
        g.node_mut(vol)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Float(0.5));
        let play = g.insert("audio.play", NodeUi::default());
        let setv = g.insert("audio.set_volume", NodeUi::default());
        let stop = g.insert("audio.stop", NodeUi::default());

        wire(&mut g, ent, "value", play, "entity");
        wire(&mut g, ent, "value", setv, "entity");
        wire(&mut g, vol, "value", setv, "volume");
        wire(&mut g, ent, "value", stop, "entity");
        wire(&mut g, bp, EXEC_THEN, play, "exec");
        wire(&mut g, play, EXEC_THEN, setv, "exec");
        wire(&mut g, setv, EXEC_THEN, stop, "exec");

        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        let mut host = MockAudio::default();
        eval_fn(&f, &HashMap::new(), &mut host).unwrap();
        // The deterministic command stream, in exec order.
        assert_eq!(
            host.commands,
            vec![
                "play 7".to_string(),
                "set_volume 7 0.5".to_string(),
                "stop 7".to_string(),
            ]
        );
    }

    #[test]
    fn audio_node_without_a_host_errors_cleanly() {
        use crate::interp::PureHost;
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let ent = g.insert("lit.int", NodeUi::default());
        let play = g.insert("audio.play", NodeUi::default());
        wire(&mut g, ent, "value", play, "entity");
        wire(&mut g, bp, EXEC_THEN, play, "exec");
        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        let err = eval_fn(&f, &HashMap::new(), &mut PureHost).unwrap_err();
        assert!(
            matches!(&err, crate::RunError::Host(_, m) if m.contains("no audio host")),
            "got {err:?}"
        );
    }

    /// A tiny 3D physics host: one entity with velocity + position that
    /// `move_and_slide` integrates (the `d3` mirror of [`MockPhysics`]).
    struct MockPhysics3d {
        v: [f64; 3],
        p: [f64; 3],
    }

    impl crate::interp::Host for MockPhysics3d {
        fn call(&mut self, path: &[String], _args: &[Value]) -> Result<Value, crate::RunError> {
            Err(crate::RunError::NoSuchHostFn(path.join("::")))
        }
        fn physics3d(&mut self) -> Option<&mut dyn crate::interp::Physics3dHost> {
            Some(self)
        }
    }

    impl crate::interp::Physics3dHost for MockPhysics3d {
        fn move_and_slide(
            &mut self,
            _entity: i64,
            motion: [f64; 3],
        ) -> Result<crate::interp::MoveResult3d, String> {
            for (p, m) in self.p.iter_mut().zip(motion) {
                *p += m;
            }
            Ok(crate::interp::MoveResult3d {
                applied: motion,
                grounded: self.p[1] <= 0.0,
            })
        }
        fn is_grounded(&mut self, _entity: i64) -> Result<bool, String> {
            Ok(self.p[1] <= 0.0)
        }
        fn raycast(
            &mut self,
            _o: [f64; 3],
            _d: [f64; 3],
            _m: f64,
        ) -> Result<Option<crate::interp::RayHit3d>, String> {
            Ok(None)
        }
        fn set_velocity(&mut self, _entity: i64, v: [f64; 3]) -> Result<(), String> {
            self.v = v;
            Ok(())
        }
        fn get_velocity(&mut self, _entity: i64) -> Result<[f64; 3], String> {
            Ok(self.v)
        }
        fn apply_impulse(&mut self, _entity: i64, v: [f64; 3]) -> Result<(), String> {
            for (dst, src) in self.v.iter_mut().zip(v) {
                *dst += src;
            }
            Ok(())
        }
    }

    #[test]
    fn lowers_and_runs_character_kit_3d() {
        // begin_play → set_velocity(e, 4, -2, 3);
        //   move_and_slide(e, get_velocity(e).x, .y, .z)
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let ent = g.insert("lit.int", NodeUi::default());
        g.node_mut(ent)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Int(1));
        let mk_float = |g: &mut Graph, v: f64| {
            let n = g.insert("lit.float", NodeUi::default());
            g.node_mut(n)
                .unwrap()
                .params
                .insert("value".into(), ParamValue::Float(v));
            n
        };
        let vx = mk_float(&mut g, 4.0);
        let vy = mk_float(&mut g, -2.0);
        let vz = mk_float(&mut g, 3.0);
        let sv = g.insert("physics3d.set_velocity", NodeUi::default());
        let getv = g.insert("physics3d.get_velocity", NodeUi::default());
        let mas = g.insert("physics3d.move_and_slide", NodeUi::default());

        wire(&mut g, ent, "value", sv, "entity");
        wire(&mut g, vx, "value", sv, "vx");
        wire(&mut g, vy, "value", sv, "vy");
        wire(&mut g, vz, "value", sv, "vz");
        wire(&mut g, ent, "value", getv, "entity");
        wire(&mut g, ent, "value", mas, "entity");
        wire(&mut g, getv, "x", mas, "motion_x");
        wire(&mut g, getv, "y", mas, "motion_y");
        wire(&mut g, getv, "z", mas, "motion_z");
        wire(&mut g, bp, EXEC_THEN, sv, "exec");
        wire(&mut g, sv, EXEC_THEN, mas, "exec");

        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        let src = format!("{:?}", f.body);
        assert!(src.contains("get_velocity"), "body: {src}");

        let mut host = MockPhysics3d {
            v: [0.0; 3],
            p: [0.0; 3],
        };
        eval_fn(&f, &HashMap::new(), &mut host).unwrap();
        // set_velocity(4,-2,3) → get_velocity=(4,-2,3) → move_and_slide adds it once.
        assert_eq!(host.v, [4.0, -2.0, 3.0]);
        assert_eq!(host.p, [4.0, -2.0, 3.0]);
    }

    #[test]
    fn physics3d_node_without_a_world_errors_cleanly() {
        use crate::interp::PureHost;
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let ent = g.insert("lit.int", NodeUi::default());
        let sv = g.insert("physics3d.set_velocity", NodeUi::default());
        wire(&mut g, ent, "value", sv, "entity");
        wire(&mut g, bp, EXEC_THEN, sv, "exec");
        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        let err = eval_fn(&f, &HashMap::new(), &mut PureHost).unwrap_err();
        assert!(
            matches!(&err, crate::RunError::Host(_, m) if m.contains("no physics world")),
            "got {err:?}"
        );
    }

    #[test]
    fn lowers_branch() {
        // begin_play → branch(condition = 1 > 0) ? print("t") : print("f")
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let one = g.insert("lit.float", NodeUi::default());
        g.node_mut(one)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Float(1.0));
        let zero = g.insert("lit.float", NodeUi::default());
        let gt = g.insert("cmp.gt", NodeUi::default());
        let br = g.insert("flow.branch", NodeUi::default());
        let pt = g.insert("debug.print", NodeUi::default());
        g.node_mut(pt)
            .unwrap()
            .params
            .insert("message".into(), ParamValue::Text("t".into()));

        wire(&mut g, one, "value", gt, "a");
        wire(&mut g, zero, "value", gt, "b");
        wire(&mut g, gt, "out", br, "condition");
        wire(&mut g, bp, EXEC_THEN, br, "exec");
        wire(&mut g, br, "true", pt, "exec");

        let fns = lower_graph(&g, &reg).unwrap();
        let f = &fns[0];
        assert_eq!(f.body.len(), 1);
        assert!(matches!(f.body[0], Stmt::If { .. }));
    }

    /// A host backing `vars::*`, `nodestate::*`, and `debug.print` over maps —
    /// enough to run the math + flow palettes end to end.
    #[derive(Default)]
    struct MapHost {
        vars: HashMap<String, Value>,
        state: HashMap<String, Value>,
        logs: Vec<String>,
    }

    impl crate::interp::Host for MapHost {
        fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, crate::RunError> {
            match (
                path.first().map(String::as_str),
                path.get(1).map(String::as_str),
            ) {
                (Some("vars"), Some("get")) => Ok(self
                    .vars
                    .get(args[0].as_str().unwrap())
                    .cloned()
                    .unwrap_or(Value::Float(0.0))),
                (Some("vars"), Some("set")) => {
                    self.vars
                        .insert(args[0].as_str().unwrap().to_string(), args[1].clone());
                    Ok(Value::Unit)
                }
                (Some("nodestate"), Some("get_or")) => Ok(self
                    .state
                    .get(args[0].as_str().unwrap())
                    .cloned()
                    .unwrap_or_else(|| args[1].clone())),
                (Some("nodestate"), Some("set")) => {
                    self.state
                        .insert(args[0].as_str().unwrap().to_string(), args[1].clone());
                    Ok(Value::Unit)
                }
                (Some("debug"), Some("print")) => {
                    self.logs.push(args[0].as_str().unwrap().to_string());
                    Ok(Value::Unit)
                }
                _ => Ok(Value::Unit),
            }
        }
    }

    fn lit_float(g: &mut Graph, v: f64) -> NodeId {
        let n = g.insert("lit.float", NodeUi::default());
        g.node_mut(n)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Float(v));
        n
    }

    fn lit_int(g: &mut Graph, v: i64) -> NodeId {
        let n = g.insert("lit.int", NodeUi::default());
        g.node_mut(n)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Int(v));
        n
    }

    /// Lower `begin_play → var.set("out", <math node fed by `ins`>)` and run it,
    /// returning the stored `out` value. `ins` are (port, node) wires.
    fn run_math_node(type_id: &str, ins: &[(&str, NodeId)], g: &mut Graph) -> Value {
        let reg = blueprint_registry();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let op = g.insert(type_id, NodeUi::default());
        let setv = g.insert("var.set", NodeUi::default());
        g.node_mut(setv)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("out".into()));
        for (port, src) in ins {
            wire(g, *src, "value", op, port);
        }
        wire(g, op, "out", setv, "value");
        wire(g, bp, EXEC_THEN, setv, "exec");
        let f = lower_graph(g, &reg).unwrap().pop().unwrap();
        let mut host = MapHost::default();
        eval_fn(&f, &HashMap::new(), &mut host).unwrap();
        host.vars.get("out").cloned().unwrap()
    }

    #[test]
    fn lowers_and_runs_math_palette() {
        // A representative sweep across the unary/binary/ternary/convert families.
        let mut g = Graph::empty();
        let a = lit_float(&mut g, -9.0);
        assert_eq!(
            run_math_node("math.neg", &[("a", a)], &mut g),
            Value::Float(9.0)
        );

        let mut g = Graph::empty();
        let a = lit_float(&mut g, -3.5);
        assert_eq!(
            run_math_node("math.abs", &[("a", a)], &mut g),
            Value::Float(3.5)
        );

        let mut g = Graph::empty();
        let a = lit_float(&mut g, 16.0);
        assert_eq!(
            run_math_node("math.sqrt", &[("a", a)], &mut g),
            Value::Float(4.0)
        );

        let mut g = Graph::empty();
        let a = lit_float(&mut g, 2.0);
        let b = lit_float(&mut g, 10.0);
        assert_eq!(
            run_math_node("math.pow", &[("a", a), ("b", b)], &mut g),
            Value::Float(1024.0)
        );

        let mut g = Graph::empty();
        let a = lit_float(&mut g, 3.0);
        let b = lit_float(&mut g, 7.0);
        assert_eq!(
            run_math_node("math.min", &[("a", a), ("b", b)], &mut g),
            Value::Float(3.0)
        );

        let mut g = Graph::empty();
        let x = lit_float(&mut g, 99.0);
        let lo = lit_float(&mut g, 0.0);
        let hi = lit_float(&mut g, 10.0);
        assert_eq!(
            run_math_node("math.clamp", &[("x", x), ("min", lo), ("max", hi)], &mut g),
            Value::Float(10.0)
        );

        let mut g = Graph::empty();
        let a = lit_float(&mut g, 0.0);
        let b = lit_float(&mut g, 10.0);
        let t = lit_float(&mut g, 0.25);
        assert_eq!(
            run_math_node("math.lerp", &[("a", a), ("b", b), ("t", t)], &mut g),
            Value::Float(2.5)
        );

        let mut g = Graph::empty();
        let a = lit_float(&mut g, 3.9);
        assert_eq!(
            run_math_node("math.to_int", &[("a", a)], &mut g),
            Value::Int(3)
        );

        let mut g = Graph::empty();
        let a = lit_int(&mut g, 5);
        assert_eq!(
            run_math_node("math.to_float", &[("a", a)], &mut g),
            Value::Float(5.0)
        );
    }

    #[test]
    fn math_neg_lowers_to_unary() {
        // math.neg is a NegOp → Expr::Unary(Neg), not a math::neg call.
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let a = lit_float(&mut g, 2.0);
        let neg = g.insert("math.neg", NodeUi::default());
        let setv = g.insert("var.set", NodeUi::default());
        g.node_mut(setv)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("out".into()));
        wire(&mut g, a, "value", neg, "a");
        wire(&mut g, neg, "out", setv, "value");
        wire(&mut g, bp, EXEC_THEN, setv, "exec");
        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        let src = format!("{:?}", f.body);
        assert!(src.contains("Unary(Neg"), "body: {src}");
        assert!(!src.contains("math"), "neg must not be a math call: {src}");
    }

    #[test]
    fn lowers_and_runs_for_loop_sum() {
        // begin_play → for(0..=5) { sum = sum + index }
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let first = lit_int(&mut g, 0);
        let last = lit_int(&mut g, 5);
        let forn = g.insert("flow.for", NodeUi::default());
        let getsum = g.insert("var.get", NodeUi::default());
        g.node_mut(getsum)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("sum".into()));
        let add = g.insert("math.add", NodeUi::default());
        let setsum = g.insert("var.set", NodeUi::default());
        g.node_mut(setsum)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("sum".into()));

        wire(&mut g, first, "value", forn, "first");
        wire(&mut g, last, "value", forn, "last");
        wire(&mut g, getsum, "value", add, "a");
        wire(&mut g, forn, "index", add, "b");
        wire(&mut g, add, "out", setsum, "value");
        wire(&mut g, bp, EXEC_THEN, forn, "exec");
        wire(&mut g, forn, "loop_body", setsum, "exec");

        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        let mut host = MapHost::default();
        host.vars.insert("sum".into(), Value::Int(0));
        eval_fn(&f, &HashMap::new(), &mut host).unwrap();
        // 0+1+2+3+4+5 = 15.
        assert_eq!(host.vars.get("sum"), Some(&Value::Int(15)));
    }

    #[test]
    fn lowers_and_runs_while_countdown() {
        // begin_play → while(n > 0) { n = n - 1 }, n starts 5.
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let getn = g.insert("var.get", NodeUi::default());
        g.node_mut(getn)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("n".into()));
        let zero = lit_int(&mut g, 0);
        let gt = g.insert("cmp.gt", NodeUi::default());
        let wh = g.insert("flow.while", NodeUi::default());
        let getn2 = g.insert("var.get", NodeUi::default());
        g.node_mut(getn2)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("n".into()));
        let one = lit_int(&mut g, 1);
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
        let mut host = MapHost::default();
        host.vars.insert("n".into(), Value::Int(5));
        eval_fn(&f, &HashMap::new(), &mut host).unwrap();
        assert_eq!(host.vars.get("n"), Some(&Value::Int(0)));
        assert!(host.logs.is_empty(), "no runaway for a terminating loop");
    }

    #[test]
    fn lowers_and_runs_do_once() {
        // A custom event → do_once → var.set("hits", hits + 1). Fire 3×.
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let ev = g.insert("event.begin_play", NodeUi::default());
        let once = g.insert("flow.do_once", NodeUi::default());
        let geth = g.insert("var.get", NodeUi::default());
        g.node_mut(geth)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("hits".into()));
        let one = lit_int(&mut g, 1);
        let add = g.insert("math.add", NodeUi::default());
        let seth = g.insert("var.set", NodeUi::default());
        g.node_mut(seth)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("hits".into()));
        wire(&mut g, geth, "value", add, "a");
        wire(&mut g, one, "value", add, "b");
        wire(&mut g, add, "out", seth, "value");
        wire(&mut g, ev, EXEC_THEN, once, "exec");
        wire(&mut g, once, EXEC_THEN, seth, "exec");

        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        let mut host = MapHost::default();
        host.vars.insert("hits".into(), Value::Int(0));
        for _ in 0..3 {
            eval_fn(&f, &HashMap::new(), &mut host).unwrap();
        }
        // Fires exactly once despite three invocations.
        assert_eq!(host.vars.get("hits"), Some(&Value::Int(1)));
    }
}
