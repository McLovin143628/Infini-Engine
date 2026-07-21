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
use crate::{BinOp, Binding, BlueprintFn, Expr, Lit, LocalId, Param, Stmt, Ty, UnOp};

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
    } else if type_id.starts_with("math.")
        || type_id.starts_with("cmp.")
        || type_id == "logic.and"
        || type_id == "logic.or"
    {
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
    /// leads to. An unconnected exec output ends the chain.
    fn exec_from(&mut self, node: NodeId, out_port: &str) -> Result<Vec<Stmt>, LowerError> {
        // Exec flow is single-threaded: take the one node this output leads to.
        match self.graph.links_from(node, out_port).next().map(|l| l.to) {
            Some(next) => self.exec_node(next),
            None => Ok(Vec::new()),
        }
    }

    /// Lower an exec node plus everything downstream of it.
    fn exec_node(&mut self, node: NodeId) -> Result<Vec<Stmt>, LowerError> {
        if self.visiting.contains(&node) {
            return Err(LowerError::ExecCycle(node));
        }
        self.visiting.push(node);
        let result = self.exec_node_inner(node);
        self.visiting.pop();
        result
    }

    fn exec_node_inner(&mut self, node: NodeId) -> Result<Vec<Stmt>, LowerError> {
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
            other => Err(LowerError::Unsupported(
                node,
                format!("{type_id} ({other:?})"),
            )),
        }
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
            NodeRole::VarGet => {
                let name = self.string_param(node, "name");
                Expr::Call {
                    path: vec!["vars".into(), "get".into()],
                    args: vec![Expr::Lit(Lit::Str(name))],
                }
            }
            NodeRole::PureCall => self.build_call(node, &type_id, prelude)?,
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
}
