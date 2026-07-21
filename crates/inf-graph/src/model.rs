//! The pure, serializable graph topology — free of execution semantics.
//!
//! De-geo'd port of GeoCanvas `nodegraph::model`. A [`Graph`] is a
//! `BTreeMap<NodeId, Node>` plus a `Vec<Link>`; a [`Node`] carries only its
//! sparse [`ParamMap`] and canvas [`NodeUi`] — all behavior (ports, defaults,
//! typing) lives in the registry keyed by `type_id`. Determinism is
//! architectural: `BTreeMap` everywhere gives key-sorted, byte-identical
//! serialization and stable cache hashes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Identity of a node, unique **within one graph** (not global). Allocated
/// monotonically by [`Graph::alloc`] and never reused, so a stale reference is
/// always a miss rather than a silent alias.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct NodeId(pub u32);

impl std::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "N{}", self.0)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "N{}", self.0)
    }
}

/// A stored scalar parameter value. Closed set — the universal scalars plus a
/// named `Enum` choice (a selection from a [`ParamDef`](crate::registry::ParamDef)'s
/// `options`). Anything richer travels on a wire, not in a param.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ParamValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Enum(String),
}

impl ParamValue {
    /// JSON projection for the frontend inspector. Non-finite floats collapse
    /// to `null` (they have no literal spelling and are rejected upstream).
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            ParamValue::Bool(b) => serde_json::Value::Bool(*b),
            ParamValue::Int(i) => serde_json::Value::Number((*i).into()),
            ParamValue::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            ParamValue::Text(s) | ParamValue::Enum(s) => serde_json::Value::String(s.clone()),
        }
    }
}

/// A node's parameter overrides. **Sparse**: only values differing from the
/// registry default are stored, so adding a param to a node type never
/// invalidates existing documents — [`resolve`](crate::registry::NodeDef::resolve)
/// fills defaults at compile time. `BTreeMap` for deterministic iteration.
pub type ParamMap = BTreeMap<String, ParamValue>;

/// Presentation-only node state: canvas position, comment-box sizing, tint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct NodeUi {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub w: f64,
    #[serde(default)]
    pub h: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

/// A graph node. Its behavior is entirely determined by `type_id` (a registry
/// key like `"flow.branch"` or `"math.add"`); this struct stores only identity,
/// param overrides, the A/B `disabled` toggle, and canvas UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub type_id: String,
    #[serde(default)]
    pub params: ParamMap,
    /// A/B toggle: a disabled node is spliced out at compile, forwarding its
    /// first type-matching input to consumers.
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub ui: NodeUi,
}

/// A directed edge from one node's output port to another's input port, both
/// referenced by name. An input port holds at most one link.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    pub from: NodeId,
    pub from_port: String,
    pub to: NodeId,
    pub to_port: String,
}

/// The graph topology: nodes keyed by id, a flat link list, and the monotonic
/// id allocator. No single `output` field — a graph may have many sinks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: BTreeMap<NodeId, Node>,
    pub links: Vec<Link>,
    pub next_id: u32,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            nodes: BTreeMap::new(),
            links: Vec::new(),
            next_id: 1,
        }
    }
}

impl Graph {
    /// A fresh empty graph (`next_id == 1`; `NodeId(0)` stays a sentinel).
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Allocate the next node id. Monotonic; ids are never reused.
    pub fn alloc(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Insert a node with a freshly allocated id, returning that id.
    pub fn insert(&mut self, type_id: impl Into<String>, ui: NodeUi) -> NodeId {
        let id = self.alloc();
        self.nodes.insert(
            id,
            Node {
                id,
                type_id: type_id.into(),
                params: ParamMap::new(),
                disabled: false,
                ui,
            },
        );
        id
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Links terminating at `node` (any input port).
    pub fn incoming(&self, node: NodeId) -> impl Iterator<Item = &Link> {
        self.links.iter().filter(move |l| l.to == node)
    }

    /// Links originating at `node` (any output port).
    pub fn outgoing(&self, node: NodeId) -> impl Iterator<Item = &Link> {
        self.links.iter().filter(move |l| l.from == node)
    }

    /// The single link feeding `(node, port)`, if any. Input ports are
    /// single-assignment, so this returns the first (and only) match.
    pub fn link_into(&self, node: NodeId, port: &str) -> Option<&Link> {
        self.links
            .iter()
            .find(|l| l.to == node && l.to_port == port)
    }

    /// Every link feeding out of `(node, port)` (output ports fan out freely).
    pub fn links_from(&self, node: NodeId, port: &str) -> impl Iterator<Item = &Link> {
        let port = port.to_owned();
        self.links
            .iter()
            .filter(move |l| l.from == node && l.from_port == port)
    }
}

/// Saved canvas pan/zoom.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct GraphViewport {
    pub x: f64,
    pub y: f64,
    pub zoom: f64,
}

/// A named, runnable graph document — the on-disk unit (one "sheet").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphDoc {
    pub id: String,
    pub name: String,
    pub graph: Graph,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub viewport: Option<GraphViewport>,
    #[serde(default)]
    pub modified_ms: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_is_monotonic() {
        let mut g = Graph::empty();
        let a = g.alloc();
        let b = g.alloc();
        assert_eq!((a, b), (NodeId(1), NodeId(2)));
        assert_eq!(g.next_id, 3);
    }

    #[test]
    fn incoming_outgoing_and_link_into() {
        let mut g = Graph::empty();
        let a = g.insert("math.value", NodeUi::default());
        let b = g.insert("math.add", NodeUi::default());
        g.links.push(Link {
            from: a,
            from_port: "out".into(),
            to: b,
            to_port: "lhs".into(),
        });
        assert_eq!(g.incoming(b).count(), 1);
        assert_eq!(g.outgoing(a).count(), 1);
        assert!(g.link_into(b, "lhs").is_some());
        assert!(g.link_into(b, "rhs").is_none());
    }

    #[test]
    fn param_value_json() {
        assert_eq!(ParamValue::Bool(true).to_json(), serde_json::json!(true));
        assert_eq!(ParamValue::Int(7).to_json(), serde_json::json!(7));
        assert_eq!(
            ParamValue::Float(f64::NAN).to_json(),
            serde_json::Value::Null
        );
        assert_eq!(
            ParamValue::Enum("A".into()).to_json(),
            serde_json::json!("A")
        );
    }

    #[test]
    fn graph_round_trips_through_json() {
        let mut g = Graph::empty();
        let n = g.insert(
            "v",
            NodeUi {
                x: 1.0,
                y: 2.0,
                ..Default::default()
            },
        );
        g.node_mut(n)
            .unwrap()
            .params
            .insert("k".into(), ParamValue::Int(3));
        let doc = GraphDoc {
            id: "ng:1".into(),
            name: "test".into(),
            graph: g,
            viewport: Some(GraphViewport {
                x: 0.0,
                y: 0.0,
                zoom: 1.5,
            }),
            modified_ms: 42,
        };
        let json = serde_json::to_string(&doc).unwrap();
        let back: GraphDoc = serde_json::from_str(&json).unwrap();
        assert_eq!(doc, back);
    }
}
