//! `Graph` → `CompiledGraph`: pure, no I/O.
//!
//! De-geo'd port of GeoCanvas `nodegraph::compile`. Produces a flat,
//! topologically-ordered vector of [`CompiledNode`]s with dense `u32` producer
//! indices, resolved params, and consumer refcounts — the artifact the
//! executor walks straight through. Determinism is load-bearing: the topo
//! sort breaks ties by `NodeId` (a `BTreeSet` ready-set), so recompiles are
//! byte-identical and cache hashes are stable.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::{Graph, NodeId, ParamMap};
use crate::registry::{NodeRegistry, PortType, COMMENT, PASSTHROUGH, SINK};

/// A fatal compile error — the graph cannot be run as given.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum GraphError {
    #[error("graph has a cycle through {nodes:?}")]
    Cycle { nodes: Vec<NodeId> },
    #[error("node {node} has unknown type `{type_id}`")]
    UnknownType { node: NodeId, type_id: String },
    #[error("link references missing node {node}")]
    UnknownNode { node: NodeId },
}

/// A non-fatal structural problem surfaced as an editor badge; a graph with
/// issues still compiles and runs (issues just warn).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum GraphIssue {
    MissingInput {
        node: NodeId,
        port: String,
    },
    UnknownType {
        node: NodeId,
        type_id: String,
    },
    /// Nothing downstream of this node reaches a sink.
    Orphan {
        node: NodeId,
    },
    /// The link at this index wires incompatible port types.
    TypeMismatch {
        link: usize,
    },
    /// No sink node, so nothing would run.
    NoSink,
}

/// One node in a [`CompiledGraph`], carrying everything exec needs and a
/// back-reference into the source model.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledNode {
    pub node_id: NodeId,
    pub type_id: String,
    /// Defaults resolved over the sparse model params.
    pub params: ParamMap,
    /// One entry per declared input port, in port order (param-pins included):
    /// `Some(dense_index)` of the producing compiled node, or `None` if the
    /// port is unconnected.
    pub inputs: Vec<Option<u32>>,
    /// How many live nodes consume this node's output — drives value release.
    pub consumers: u32,
}

/// The compiled artifact: nodes in topological order, dense-indexed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompiledGraph {
    pub nodes: Vec<CompiledNode>,
}

impl CompiledGraph {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

/// Kahn topological sort over *all* nodes. Ties break by `NodeId`.
pub fn topo_sort(g: &Graph) -> Result<Vec<NodeId>, GraphError> {
    let subset: BTreeSet<NodeId> = g.nodes.keys().copied().collect();
    topo_within(g, &subset)
}

/// Would adding an edge `from → to` create a cycle? Walk upstream from `from`
/// (following incoming links) and report whether `to` is reachable. The editor
/// calls this before every connect, so acyclicity is a document invariant.
pub fn would_cycle(g: &Graph, from: NodeId, to: NodeId) -> bool {
    if from == to {
        return true;
    }
    let mut stack = vec![from];
    let mut seen = BTreeSet::new();
    while let Some(n) = stack.pop() {
        if n == to {
            return true;
        }
        if !seen.insert(n) {
            continue;
        }
        for l in g.incoming(n) {
            stack.push(l.from);
        }
    }
    false
}

/// Collect non-fatal structural issues (editor badges). Never errors.
pub fn validate(g: &Graph, reg: &NodeRegistry) -> Vec<GraphIssue> {
    let mut issues = Vec::new();

    // Unknown types + missing required inputs.
    for node in g.nodes.values() {
        let Some(def) = reg.get(&node.type_id) else {
            issues.push(GraphIssue::UnknownType {
                node: node.id,
                type_id: node.type_id.clone(),
            });
            continue;
        };
        if def.has(COMMENT) {
            continue;
        }
        for port in &def.inputs {
            if port.required && g.link_into(node.id, &port.name).is_none() {
                issues.push(GraphIssue::MissingInput {
                    node: node.id,
                    port: port.name.clone(),
                });
            }
        }
    }

    // Per-link type mismatches.
    for (i, link) in g.links.iter().enumerate() {
        if let (Some(out), Some(inp)) = (port_type_out(g, reg, link), port_type_in(g, reg, link)) {
            if !out.compatible_with(&inp) {
                issues.push(GraphIssue::TypeMismatch { link: i });
            }
        }
    }

    // Sinks + orphans.
    let sinks: Vec<NodeId> = g
        .nodes
        .values()
        .filter(|n| reg.get(&n.type_id).is_some_and(|d| d.has(SINK)) && !n.disabled)
        .map(|n| n.id)
        .collect();
    if sinks.is_empty() {
        issues.push(GraphIssue::NoSink);
    } else {
        let live = live_nodes(g, reg, None);
        for node in g.nodes.values() {
            let Some(def) = reg.get(&node.type_id) else {
                continue;
            };
            if def.has(COMMENT) || def.has(PASSTHROUGH) || node.disabled {
                continue;
            }
            if !live.contains(&node.id) {
                issues.push(GraphIssue::Orphan { node: node.id });
            }
        }
    }

    issues
}

/// The set of nodes that must run: reverse-reachability from the roots,
/// following resolved producers (skipping disabled/passthrough) and restricted
/// to executable nodes. Roots are the enabled `SINK` nodes, or — for
/// "run to here" — just `skip_transparent(target)`.
pub fn live_nodes(g: &Graph, reg: &NodeRegistry, target: Option<NodeId>) -> BTreeSet<NodeId> {
    let mut exec_state = BTreeMap::new();
    let roots: Vec<NodeId> = match target {
        Some(t) => skip_transparent(g, reg, t).into_iter().collect(),
        None => g
            .nodes
            .values()
            .filter(|n| reg.get(&n.type_id).is_some_and(|d| d.has(SINK)) && !n.disabled)
            .map(|n| n.id)
            .filter(|id| executable(g, reg, *id, &mut exec_state))
            .collect(),
    };

    let mut live = BTreeSet::new();
    let mut stack = roots;
    while let Some(n) = stack.pop() {
        if !live.insert(n) {
            continue;
        }
        let Some(def) = reg.get(&g.nodes[&n].type_id) else {
            continue;
        };
        for port in &def.inputs {
            if let Some(src) = resolve_producer(g, reg, n, &port.name) {
                if executable(g, reg, src, &mut exec_state) {
                    stack.push(src);
                }
            }
        }
    }
    live
}

/// Compile `g` for the given run target (or all sinks). Empty / sinkless /
/// fully-pruned graphs compile to zero nodes — never an error.
pub fn compile_graph(
    g: &Graph,
    reg: &NodeRegistry,
    target: Option<NodeId>,
) -> Result<CompiledGraph, GraphError> {
    let live = live_nodes(g, reg, target);
    if live.is_empty() {
        return Ok(CompiledGraph::default());
    }
    let order = topo_within(g, &live)?;

    // Dense index for each live node.
    let index: BTreeMap<NodeId, u32> = order
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i as u32))
        .collect();

    let mut nodes: Vec<CompiledNode> = Vec::with_capacity(order.len());
    let mut consumer_tally: BTreeMap<NodeId, u32> = BTreeMap::new();

    for id in &order {
        let node = &g.nodes[id];
        let def = reg
            .get(&node.type_id)
            .ok_or_else(|| GraphError::UnknownType {
                node: *id,
                type_id: node.type_id.clone(),
            })?;
        let params = def.resolve(&node.params);
        let mut inputs = Vec::with_capacity(def.inputs.len());
        for port in &def.inputs {
            let producer = resolve_producer(g, reg, *id, &port.name).and_then(|p| {
                index.get(&p).copied().inspect(|_| {
                    *consumer_tally.entry(p).or_default() += 1;
                })
            });
            inputs.push(producer);
        }
        nodes.push(CompiledNode {
            node_id: *id,
            type_id: node.type_id.clone(),
            params,
            inputs,
            consumers: 0,
        });
    }

    for cn in &mut nodes {
        cn.consumers = consumer_tally.get(&cn.node_id).copied().unwrap_or(0);
    }

    Ok(CompiledGraph { nodes })
}

// --- private helpers ---------------------------------------------------------

/// Kahn's algorithm over `subset` only, ties broken by `NodeId` (BTreeSet
/// ready-set) for deterministic order.
fn topo_within(g: &Graph, subset: &BTreeSet<NodeId>) -> Result<Vec<NodeId>, GraphError> {
    let mut indeg: BTreeMap<NodeId, usize> = subset.iter().map(|id| (*id, 0)).collect();
    // Count in-edges that stay within the subset.
    for link in &g.links {
        if subset.contains(&link.from) && subset.contains(&link.to) {
            *indeg.get_mut(&link.to).unwrap() += 1;
        }
    }
    let mut ready: BTreeSet<NodeId> = indeg
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(id, _)| *id)
        .collect();
    let mut order = Vec::with_capacity(subset.len());
    while let Some(&n) = ready.iter().next() {
        ready.remove(&n);
        order.push(n);
        for link in g.outgoing(n) {
            if !subset.contains(&link.to) {
                continue;
            }
            let d = indeg.get_mut(&link.to).unwrap();
            *d -= 1;
            if *d == 0 {
                ready.insert(link.to);
            }
        }
    }
    if order.len() != subset.len() {
        let stuck: Vec<NodeId> = subset
            .iter()
            .filter(|id| !order.contains(id))
            .copied()
            .collect();
        return Err(GraphError::Cycle { nodes: stuck });
    }
    Ok(order)
}

/// The producing node feeding `(node, port)` after skipping any disabled /
/// passthrough nodes in between.
fn resolve_producer(g: &Graph, reg: &NodeRegistry, node: NodeId, port: &str) -> Option<NodeId> {
    let link = g.link_into(node, port)?;
    skip_transparent(g, reg, link.from)
}

/// Walk past disabled / `PASSTHROUGH` nodes via their forwardable input,
/// returning the first opaque producer. A `budget` guard makes corrupt cyclic
/// files terminate rather than loop.
fn skip_transparent(g: &Graph, reg: &NodeRegistry, start: NodeId) -> Option<NodeId> {
    let mut cur = start;
    let mut budget = g.nodes.len() + 1;
    loop {
        if budget == 0 {
            return Some(cur);
        }
        budget -= 1;
        let node = g.nodes.get(&cur)?;
        let def = reg.get(&node.type_id)?;
        let transparent = node.disabled || def.has(PASSTHROUGH);
        if !transparent {
            return Some(cur);
        }
        let fwd = def.first_forwardable_input()?;
        cur = g.link_into(cur, &fwd.name)?.from;
    }
}

/// Memoized: a node is executable iff every *required* input resolves to an
/// executable producer. Unexecutability propagates downstream; a provisional
/// `false` on the visiting node breaks cycles.
fn executable(
    g: &Graph,
    reg: &NodeRegistry,
    id: NodeId,
    state: &mut BTreeMap<NodeId, bool>,
) -> bool {
    if let Some(v) = state.get(&id) {
        return *v;
    }
    state.insert(id, false); // break cycles: provisional false while visiting
    let Some(node) = g.nodes.get(&id) else {
        return false;
    };
    let Some(def) = reg.get(&node.type_id) else {
        return false;
    };
    let mut ok = true;
    for port in &def.inputs {
        if !port.required {
            continue;
        }
        match resolve_producer(g, reg, id, &port.name) {
            Some(src) => {
                if !executable(g, reg, src, state) {
                    ok = false;
                    break;
                }
            }
            None => {
                ok = false;
                break;
            }
        }
    }
    state.insert(id, ok);
    ok
}

fn port_type_out(g: &Graph, reg: &NodeRegistry, link: &crate::model::Link) -> Option<PortType> {
    let def = reg.get(&g.nodes.get(&link.from)?.type_id)?;
    port_ty(def.output(&link.from_port))
}

fn port_type_in(g: &Graph, reg: &NodeRegistry, link: &crate::model::Link) -> Option<PortType> {
    let def = reg.get(&g.nodes.get(&link.to)?.type_id)?;
    port_ty(def.input(&link.to_port))
}

fn port_ty(port: Option<&crate::registry::PortDef>) -> Option<PortType> {
    port.map(|p| p.ty.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Link;
    use crate::registry::{NodeDef, PortDef};

    fn reg() -> NodeRegistry {
        let mut r = NodeRegistry::new();
        r.register(
            NodeDef::new("val", "Value", "v")
                .with_outputs(vec![PortDef::new("out", PortType::Float)]),
        );
        r.register(
            NodeDef::new("add", "Add", "m")
                .with_inputs(vec![
                    PortDef::new("a", PortType::Float).required(),
                    PortDef::new("b", PortType::Float).required(),
                ])
                .with_outputs(vec![PortDef::new("out", PortType::Float)]),
        );
        r.register(
            NodeDef::new("sink", "Sink", "o")
                .with_inputs(vec![PortDef::new("in", PortType::Float).required()])
                .with_flags(SINK),
        );
        r.register(
            NodeDef::new("reroute", "Reroute", "flow")
                .with_inputs(vec![PortDef::new("in", PortType::Float)])
                .with_outputs(vec![PortDef::new("out", PortType::Float)])
                .with_flags(PASSTHROUGH),
        );
        r
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
    fn compiles_in_topo_order_with_consumers() {
        let reg = reg();
        let mut g = Graph::empty();
        let a = g.insert("val", Default::default());
        let b = g.insert("val", Default::default());
        let add = g.insert("add", Default::default());
        let sink = g.insert("sink", Default::default());
        wire(&mut g, a, "out", add, "a");
        wire(&mut g, b, "out", add, "b");
        wire(&mut g, add, "out", sink, "in");

        let cg = compile_graph(&g, &reg, None).unwrap();
        assert_eq!(cg.len(), 4);
        // sink is last; add before sink; values before add.
        let pos = |id: NodeId| cg.nodes.iter().position(|n| n.node_id == id).unwrap();
        assert!(pos(a) < pos(add) && pos(b) < pos(add) && pos(add) < pos(sink));
        // add is consumed once; each value once.
        let addn = &cg.nodes[pos(add)];
        assert_eq!(addn.consumers, 1);
        assert_eq!(addn.inputs.iter().filter(|i| i.is_some()).count(), 2);
    }

    #[test]
    fn cycle_is_rejected_by_would_cycle() {
        let mut g = Graph::empty();
        let a = g.insert("add", Default::default());
        let b = g.insert("add", Default::default());
        wire(&mut g, a, "out", b, "a");
        assert!(would_cycle(&g, b, a));
        assert!(!would_cycle(&g, a, b));
    }

    #[test]
    fn passthrough_is_spliced_out() {
        let reg = reg();
        let mut g = Graph::empty();
        let v = g.insert("val", Default::default());
        let rr = g.insert("reroute", Default::default());
        let sink = g.insert("sink", Default::default());
        wire(&mut g, v, "out", rr, "in");
        wire(&mut g, rr, "out", sink, "in");
        let cg = compile_graph(&g, &reg, None).unwrap();
        // reroute is not compiled; sink's input resolves straight to the value.
        assert!(cg.nodes.iter().all(|n| n.type_id != "reroute"));
        let sink_cn = cg.nodes.iter().find(|n| n.type_id == "sink").unwrap();
        let vidx = cg.nodes.iter().position(|n| n.node_id == v).unwrap() as u32;
        assert_eq!(sink_cn.inputs[0], Some(vidx));
    }

    #[test]
    fn no_sink_is_empty_not_error() {
        let reg = reg();
        let mut g = Graph::empty();
        g.insert("val", Default::default());
        let cg = compile_graph(&g, &reg, None).unwrap();
        assert!(cg.is_empty());
        assert!(validate(&g, &reg).contains(&GraphIssue::NoSink));
    }

    #[test]
    fn missing_required_input_is_an_issue() {
        let reg = reg();
        let mut g = Graph::empty();
        let add = g.insert("add", Default::default());
        let sink = g.insert("sink", Default::default());
        wire(&mut g, add, "out", sink, "in");
        let issues = validate(&g, &reg);
        assert!(issues
            .iter()
            .any(|i| matches!(i, GraphIssue::MissingInput { node, .. } if *node == add)));
    }
}
