//! Generic node-graph DAG: model, registry, compile, exec, derive, cache.
//!
//! A domain-agnostic ("de-geo'd") port of the GeoCanvas node-graph engine.
//! `inf-graph` owns the *substrate* every visual graph in Infinity Engine
//! shares — Blueprints (P6), material graphs (P7), texture/PCG graphs (P10),
//! animation state machines (P11) — while each domain supplies its own node
//! kit and value type on top.
//!
//! The pieces, and the invariants that make them composable:
//!
//! - [`model`] — the pure, serializable topology. A [`Graph`] is a
//!   `BTreeMap<NodeId, Node>` + `Vec<Link>`; a [`Node`] stores only a sparse
//!   [`ParamMap`] and canvas UI, its behavior keyed entirely by `type_id`.
//!   `BTreeMap` everywhere makes serialization and cache hashes deterministic.
//! - [`registry`] — [`NodeDef`]s keyed by `type_id`: ports ([`PortType`]-typed,
//!   including the blueprint `Exec` pin and `Named` domain types), params with
//!   defaults/clamps, and behavior flags. Ports live on the *type*, so adding
//!   one never migrates a document.
//! - [`compile`] — `Graph` → [`CompiledGraph`]: a topologically-ordered vector
//!   (Kahn, `NodeId`-tie-broken for determinism) with dense producer indices
//!   and consumer refcounts. Disabled / passthrough nodes are spliced out.
//! - [`exec`] — a push/eager dataflow walk over the compiled vector, dispatch
//!   supplied by a [`NodeExecutor`] trait, values released at their last
//!   consumer, failures contained to the downstream cone.
//! - [`cache`] — content-addressed xxh3 caching where each node's hash folds
//!   in all upstream hashes, so subgraph caching is emergent (no dirty bits).
//! - [`edits`] — the mutation door ([`GraphEdit`]) that enforces acyclicity +
//!   single-input-link at entry, plus a snapshot [`GraphJournal`] for undo/redo.
//! - [`derive`] — turn an abstract [`OpSpec`] into a [`NodeDef`] for free.
//! - [`param_meta`] — an optional side table of curated help + numeric bounds.
//!
//! Ring 0: serde-only, no Tauri, no ts-rs. The editor's Ring-1 layer projects
//! the serde DTOs to the frontend.

pub mod cache;
pub mod compile;
pub mod derive;
pub mod edits;
pub mod exec;
pub mod model;
pub mod param_meta;
pub mod registry;
pub mod value;

pub use cache::{content_hash, file_identity_hash, node_hash, NodeCache};
pub use compile::{
    compile_graph, live_nodes, topo_sort, validate, would_cycle, CompiledGraph, CompiledNode,
    GraphError, GraphIssue,
};
pub use derive::{derive_node_def, OpParam, OpSpec};
pub use edits::{apply_edit, apply_edits, GraphEdit, GraphJournal};
pub use exec::{
    run_graph, EvalCtx, ExecError, ExecRegistry, NodeExecutor, NodePhase, NodeRunStatus, NodeSink,
    NullSink, RunOutcome,
};
pub use model::{Graph, GraphDoc, GraphViewport, Link, Node, NodeId, NodeUi, ParamMap, ParamValue};
pub use param_meta::{ParamMeta, ParamMetaTable};
pub use registry::{
    NodeDef, NodeFlags, NodeRegistry, ParamDef, PortDef, PortType, UiHint, COMMENT, HIDDEN,
    PASSTHROUGH, SINK,
};
pub use value::GraphValue;
