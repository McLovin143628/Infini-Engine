//! The generic dataflow executor.
//!
//! De-geo'd port of the GeoCanvas `nodegraph::exec` *skeleton* — the domain
//! dispatch body (GDAL/geo ops) is replaced by a [`NodeExecutor`] trait the
//! domain implements per node type. Evaluation is **push / eager, single
//! linear pass**: [`compile_graph`](crate::compile::compile_graph) already
//! linearized the DAG, so this just iterates the compiled vector, flowing
//! results forward through a `values` array. Memory is bounded by consumer
//! refcounts (a value is dropped the moment its last consumer runs), and a
//! failed node's error is contained to its downstream cone — siblings run on.
//!
//! This dataflow engine drives pure data-pin subgraphs (blueprint pure
//! functions, material/PCG graphs). Blueprint *exec-flow* sequencing is a
//! separate concern handled by the interpreter in `inf-blueprint`.

use std::collections::BTreeMap;

use crate::cache::{node_hash, NodeCache};
use crate::compile::CompiledGraph;
use crate::model::{NodeId, ParamMap};
use crate::value::GraphValue;

/// Per-node run status phase, streamed to the editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodePhase {
    Running,
    Done,
    Cached,
    Error,
    Skipped,
}

/// A single node status update.
#[derive(Debug, Clone)]
pub struct NodeRunStatus {
    pub node_id: NodeId,
    pub phase: NodePhase,
    pub message: Option<String>,
}

/// Receives per-node status as a run proceeds. The default no-op impl
/// ([`NullSink`]) suits tests and headless runs.
pub trait NodeSink {
    fn status(&self, status: &NodeRunStatus);
}

/// A sink that drops every status.
pub struct NullSink;
impl NodeSink for NullSink {
    fn status(&self, _status: &NodeRunStatus) {}
}

/// What a node executor sees: its resolved params and its input values, in
/// declared input-port order (`None` = unconnected).
pub struct EvalCtx<'a, V> {
    pub node_id: NodeId,
    pub type_id: &'a str,
    pub params: &'a ParamMap,
    pub inputs: &'a [Option<V>],
}

impl<V> EvalCtx<'_, V> {
    /// The value on input port index `i`, if connected.
    pub fn input(&self, i: usize) -> Option<&V> {
        self.inputs.get(i).and_then(|o| o.as_ref())
    }
}

/// An error from evaluating one node. Its message surfaces on the node badge;
/// its cone is skipped.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct ExecError(pub String);

impl ExecError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

/// How one node type computes its output from its inputs + params.
pub trait NodeExecutor<V>: Send + Sync {
    fn eval(&self, ctx: &EvalCtx<V>) -> Result<V, ExecError>;
}

/// A blanket impl lets a plain closure serve as an executor.
impl<V, F> NodeExecutor<V> for F
where
    F: Fn(&EvalCtx<V>) -> Result<V, ExecError> + Send + Sync,
{
    fn eval(&self, ctx: &EvalCtx<V>) -> Result<V, ExecError> {
        self(ctx)
    }
}

/// Maps `type_id` → executor. The domain populates this alongside the
/// [`NodeRegistry`](crate::registry::NodeRegistry).
#[derive(Default)]
pub struct ExecRegistry<V> {
    execs: BTreeMap<String, Box<dyn NodeExecutor<V>>>,
}

impl<V> ExecRegistry<V> {
    pub fn new() -> Self {
        Self {
            execs: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, type_id: impl Into<String>, exec: impl NodeExecutor<V> + 'static) {
        self.execs.insert(type_id.into(), Box::new(exec));
    }

    pub fn get(&self, type_id: &str) -> Option<&dyn NodeExecutor<V>> {
        self.execs.get(type_id).map(|b| b.as_ref())
    }
}

/// The result of a run: statuses in evaluation order, the outputs of every
/// node the caller asked to keep (sinks), and the content hash each node ran
/// under (for data-peek / debugging).
pub struct RunOutcome<V> {
    pub statuses: Vec<NodeRunStatus>,
    pub outputs: BTreeMap<NodeId, V>,
    pub hashes: BTreeMap<NodeId, u64>,
}

/// Run a compiled graph. `keep(node_id)` decides which node outputs land in
/// [`RunOutcome::outputs`] (typically the sinks). Values not kept and past
/// their last consumer are dropped to bound memory. Cache hits skip evaluation.
pub fn run_graph<V: GraphValue>(
    cg: &CompiledGraph,
    execs: &ExecRegistry<V>,
    cache: &NodeCache<V>,
    sink: &dyn NodeSink,
    keep: impl Fn(NodeId) -> bool,
) -> RunOutcome<V> {
    let n = cg.nodes.len();
    let mut values: Vec<Option<V>> = (0..n).map(|_| None).collect();
    let mut remaining: Vec<u32> = cg.nodes.iter().map(|c| c.consumers).collect();
    let mut failed = vec![false; n];
    let mut hashes = vec![0u64; n];

    let mut statuses = Vec::with_capacity(n);
    let mut outputs = BTreeMap::new();

    let emit = |statuses: &mut Vec<NodeRunStatus>,
                node_id: NodeId,
                phase: NodePhase,
                message: Option<String>,
                sink: &dyn NodeSink| {
        let s = NodeRunStatus {
            node_id,
            phase,
            message,
        };
        sink.status(&s);
        statuses.push(s);
    };

    for i in 0..n {
        let cn = &cg.nodes[i];

        // Upstream-failure containment: any wired producer that failed (or
        // whose value was released before us — impossible in topo order unless
        // it failed) skips this node's whole cone.
        let skip = cn.inputs.iter().flatten().any(|src| failed[*src as usize]);
        if skip {
            failed[i] = true;
            emit(&mut statuses, cn.node_id, NodePhase::Skipped, None, sink);
            continue;
        }

        // Gather input values (cloned; cheap per GraphValue contract).
        let inputs: Vec<Option<V>> = cn
            .inputs
            .iter()
            .map(|src| src.and_then(|s| values[s as usize].clone()))
            .collect();

        // Content hash folds in upstream hashes in port order.
        let upstream: Vec<u64> = cn
            .inputs
            .iter()
            .filter_map(|src| src.map(|s| hashes[s as usize]))
            .collect();
        let key = node_hash(&cn.type_id, &cn.params, &upstream);
        hashes[i] = key;

        // Cache lookup around dispatch.
        if let Some(hit) = cache.get(key) {
            emit(&mut statuses, cn.node_id, NodePhase::Cached, None, sink);
            store_result(i, cn.node_id, hit, &mut values, &mut outputs, &keep);
            release_consumed(cn, &mut remaining, &mut values);
            continue;
        }

        let Some(exec) = execs.get(&cn.type_id) else {
            failed[i] = true;
            emit(
                &mut statuses,
                cn.node_id,
                NodePhase::Error,
                Some(format!("no executor for `{}`", cn.type_id)),
                sink,
            );
            continue;
        };

        emit(&mut statuses, cn.node_id, NodePhase::Running, None, sink);
        let ctx = EvalCtx {
            node_id: cn.node_id,
            type_id: &cn.type_id,
            params: &cn.params,
            inputs: &inputs,
        };
        match exec.eval(&ctx) {
            Ok(v) => {
                cache.insert(key, v.clone());
                store_result(i, cn.node_id, v, &mut values, &mut outputs, &keep);
                emit(&mut statuses, cn.node_id, NodePhase::Done, None, sink);
                release_consumed(cn, &mut remaining, &mut values);
            }
            Err(e) => {
                failed[i] = true;
                emit(&mut statuses, cn.node_id, NodePhase::Error, Some(e.0), sink);
            }
        }
    }

    RunOutcome {
        statuses,
        outputs,
        hashes: cg
            .nodes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.node_id, hashes[i]))
            .collect(),
    }
}

fn store_result<V: Clone>(
    idx: usize,
    node_id: NodeId,
    v: V,
    values: &mut [Option<V>],
    outputs: &mut BTreeMap<NodeId, V>,
    keep: &impl Fn(NodeId) -> bool,
) {
    if keep(node_id) {
        outputs.insert(node_id, v.clone());
    }
    values[idx] = Some(v);
}

/// Decrement each upstream producer's remaining-consumer count; drop a value
/// once its last consumer has run.
fn release_consumed<V>(
    cn: &crate::compile::CompiledNode,
    remaining: &mut [u32],
    values: &mut [Option<V>],
) {
    for src in cn.inputs.iter().flatten() {
        let s = *src as usize;
        if remaining[s] > 0 {
            remaining[s] -= 1;
            if remaining[s] == 0 {
                values[s] = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::compile_graph;
    use crate::model::{Graph, Link, ParamValue};
    use crate::registry::{NodeDef, NodeRegistry, PortDef, PortType, SINK};

    // A tiny arithmetic domain: values are f64.
    impl GraphValue for f64 {}

    fn setup() -> (NodeRegistry, ExecRegistry<f64>) {
        let mut reg = NodeRegistry::new();
        reg.register(
            NodeDef::new("k", "Const", "v")
                .with_outputs(vec![PortDef::new("out", PortType::Float)])
                .with_params(vec![crate::registry::ParamDef::number("v", 0.0)]),
        );
        reg.register(
            NodeDef::new("add", "Add", "m")
                .with_inputs(vec![
                    PortDef::new("a", PortType::Float).required(),
                    PortDef::new("b", PortType::Float).required(),
                ])
                .with_outputs(vec![PortDef::new("out", PortType::Float)]),
        );
        reg.register(
            NodeDef::new("out", "Out", "o")
                .with_inputs(vec![PortDef::new("in", PortType::Float).required()])
                .with_flags(SINK),
        );

        let mut execs: ExecRegistry<f64> = ExecRegistry::new();
        execs.register("k", |ctx: &EvalCtx<f64>| {
            Ok(match ctx.params.get("v") {
                Some(ParamValue::Float(f)) => *f,
                _ => 0.0,
            })
        });
        execs.register("add", |ctx: &EvalCtx<f64>| {
            Ok(ctx.input(0).copied().unwrap_or(0.0) + ctx.input(1).copied().unwrap_or(0.0))
        });
        execs.register("out", |ctx: &EvalCtx<f64>| {
            Ok(ctx.input(0).copied().unwrap_or(0.0))
        });
        (reg, execs)
    }

    fn konst(g: &mut Graph, v: f64) -> NodeId {
        let id = g.insert("k", Default::default());
        g.node_mut(id)
            .unwrap()
            .params
            .insert("v".into(), ParamValue::Float(v));
        id
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
    fn evaluates_arithmetic() {
        let (reg, execs) = setup();
        let mut g = Graph::empty();
        let a = konst(&mut g, 2.0);
        let b = konst(&mut g, 3.0);
        let add = g.insert("add", Default::default());
        let out = g.insert("out", Default::default());
        wire(&mut g, a, "out", add, "a");
        wire(&mut g, b, "out", add, "b");
        wire(&mut g, add, "out", out, "in");

        let cg = compile_graph(&g, &reg, None).unwrap();
        let cache = NodeCache::new(1 << 20);
        let outcome = run_graph(&cg, &execs, &cache, &NullSink, |id| id == out);
        assert_eq!(outcome.outputs.get(&out), Some(&5.0));
        assert!(outcome
            .statuses
            .iter()
            .any(|s| s.node_id == out && s.phase == NodePhase::Done));
    }

    #[test]
    fn cache_hit_second_run() {
        let (reg, execs) = setup();
        let mut g = Graph::empty();
        let a = konst(&mut g, 4.0);
        let out = g.insert("out", Default::default());
        wire(&mut g, a, "out", out, "in");
        let cg = compile_graph(&g, &reg, None).unwrap();
        let cache = NodeCache::new(1 << 20);
        let _ = run_graph(&cg, &execs, &cache, &NullSink, |id| id == out);
        let second = run_graph(&cg, &execs, &cache, &NullSink, |id| id == out);
        assert!(second.statuses.iter().any(|s| s.phase == NodePhase::Cached));
    }

    #[test]
    fn error_is_contained_to_downstream_cone() {
        let (reg, mut execs) = setup();
        execs.register("boom", |_: &EvalCtx<f64>| Err(ExecError::new("kaboom")));
        let mut reg = reg;
        reg.register(
            NodeDef::new("boom", "Boom", "m")
                .with_outputs(vec![PortDef::new("out", PortType::Float)]),
        );

        let mut g = Graph::empty();
        let bad = g.insert("boom", Default::default());
        let good = konst(&mut g, 1.0);
        let add = g.insert("add", Default::default());
        let out = g.insert("out", Default::default());
        wire(&mut g, bad, "out", add, "a");
        wire(&mut g, good, "out", add, "b");
        wire(&mut g, add, "out", out, "in");

        let cg = compile_graph(&g, &reg, None).unwrap();
        let cache = NodeCache::new(1 << 20);
        let outcome = run_graph(&cg, &execs, &cache, &NullSink, |id| id == out);
        // boom errored; add + out are skipped; nothing kept.
        assert!(!outcome.outputs.contains_key(&out));
        assert!(outcome
            .statuses
            .iter()
            .any(|s| s.node_id == bad && s.phase == NodePhase::Error));
        assert!(outcome
            .statuses
            .iter()
            .any(|s| s.node_id == out && s.phase == NodePhase::Skipped));
        // The healthy sibling still ran.
        assert!(outcome
            .statuses
            .iter()
            .any(|s| s.node_id == good && s.phase == NodePhase::Done));
    }
}
