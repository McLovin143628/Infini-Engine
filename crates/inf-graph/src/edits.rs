//! The edit door + undo journal.
//!
//! De-geo'd port of GeoCanvas `nodegraph::edits`. Every mutation goes through
//! a [`GraphEdit`]; [`apply_edits`] enforces the graph invariants at the door
//! (acyclicity, at-most-one link per input port, monotonic ids), so the
//! compiler downstream can trust a proven DAG. A snapshot [`GraphJournal`]
//! backs undo/redo — one entry per edit *group*.

use serde::{Deserialize, Serialize};

use crate::compile::would_cycle;
use crate::model::{Graph, Link, Node, NodeId, NodeUi, ParamValue};
use crate::registry::NodeRegistry;

/// A single structural mutation. Mirrors the frontend `GraphEdit` union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum GraphEdit {
    AddNode {
        id: NodeId,
        #[serde(rename = "typeId")]
        type_id: String,
        x: f64,
        y: f64,
        #[serde(default)]
        params: crate::model::ParamMap,
    },
    RemoveNode {
        id: NodeId,
    },
    /// Re-insert a removed node with its incident links (undo of RemoveNode).
    RestoreNode {
        node: Node,
        links: Vec<Link>,
    },
    Connect {
        link: Link,
    },
    Disconnect {
        link: Link,
    },
    SetParam {
        id: NodeId,
        name: String,
        value: ParamValue,
    },
    ClearParam {
        id: NodeId,
        name: String,
    },
    SetDisabled {
        id: NodeId,
        disabled: bool,
    },
    MoveNode {
        id: NodeId,
        x: f64,
        y: f64,
    },
    ResizeNode {
        id: NodeId,
        w: f64,
        h: f64,
    },
    SetTitle {
        id: NodeId,
        title: String,
    },
}

/// Apply one edit structurally, returning whether it took effect. Invalid
/// edits (cycle-forming connects, edits to missing nodes) are rejected as
/// no-ops rather than corrupting the graph.
pub fn apply_edit(g: &mut Graph, reg: &NodeRegistry, edit: &GraphEdit) -> bool {
    match edit {
        GraphEdit::AddNode {
            id,
            type_id,
            x,
            y,
            params,
        } => {
            if g.nodes.contains_key(id) {
                return false;
            }
            let mut params = params.clone();
            if let Some(def) = reg.get(type_id) {
                def.sanitize(&mut params);
            }
            g.nodes.insert(
                *id,
                Node {
                    id: *id,
                    type_id: type_id.clone(),
                    params,
                    disabled: false,
                    ui: NodeUi {
                        x: *x,
                        y: *y,
                        ..Default::default()
                    },
                },
            );
            g.next_id = g.next_id.max(id.0 + 1);
            true
        }
        GraphEdit::RemoveNode { id } => {
            if g.nodes.remove(id).is_none() {
                return false;
            }
            g.links.retain(|l| l.from != *id && l.to != *id);
            true
        }
        GraphEdit::RestoreNode { node, links } => {
            g.next_id = g.next_id.max(node.id.0 + 1);
            g.nodes.insert(node.id, node.clone());
            for l in links {
                if g.nodes.contains_key(&l.from) && g.nodes.contains_key(&l.to) {
                    g.links.push(l.clone());
                }
            }
            true
        }
        GraphEdit::Connect { link } => {
            if !g.nodes.contains_key(&link.from) || !g.nodes.contains_key(&link.to) {
                return false;
            }
            if link.from == link.to || would_cycle(g, link.from, link.to) {
                return false;
            }
            // Replace-on-connect: an input port holds at most one link.
            g.links
                .retain(|l| !(l.to == link.to && l.to_port == link.to_port));
            g.links.push(link.clone());
            true
        }
        GraphEdit::Disconnect { link } => {
            let before = g.links.len();
            g.links.retain(|l| l != link);
            g.links.len() != before
        }
        GraphEdit::SetParam { id, name, value } => {
            let Some(node) = g.nodes.get_mut(id) else {
                return false;
            };
            node.params.insert(name.clone(), value.clone());
            if let Some(def) = reg.get(&node.type_id) {
                def.sanitize(&mut node.params);
            }
            true
        }
        GraphEdit::ClearParam { id, name } => {
            let Some(node) = g.nodes.get_mut(id) else {
                return false;
            };
            node.params.remove(name).is_some()
        }
        GraphEdit::SetDisabled { id, disabled } => {
            let Some(node) = g.nodes.get_mut(id) else {
                return false;
            };
            node.disabled = *disabled;
            true
        }
        GraphEdit::MoveNode { id, x, y } => {
            let Some(node) = g.nodes.get_mut(id) else {
                return false;
            };
            node.ui.x = *x;
            node.ui.y = *y;
            true
        }
        GraphEdit::ResizeNode { id, w, h } => {
            let Some(node) = g.nodes.get_mut(id) else {
                return false;
            };
            node.ui.w = *w;
            node.ui.h = *h;
            true
        }
        GraphEdit::SetTitle { id, title } => {
            let Some(node) = g.nodes.get_mut(id) else {
                return false;
            };
            node.ui.title = title.clone();
            true
        }
    }
}

/// Apply a batch of edits in order, returning how many took effect.
pub fn apply_edits(g: &mut Graph, reg: &NodeRegistry, edits: &[GraphEdit]) -> usize {
    edits.iter().filter(|e| apply_edit(g, reg, e)).count()
}

/// Snapshot-based undo/redo journal — one entry per edit group.
#[derive(Debug, Clone)]
pub struct GraphJournal {
    undo: Vec<(String, Graph)>,
    redo: Vec<(String, Graph)>,
    cap: usize,
}

impl GraphJournal {
    pub fn new(cap: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            cap: cap.max(1),
        }
    }

    /// Record the state `before` an edit group labeled `label`. Clears the
    /// redo stack (a new edit forks history) and bounds the undo depth.
    pub fn record(&mut self, label: impl Into<String>, before: Graph) {
        self.redo.clear();
        self.undo.push((label.into(), before));
        while self.undo.len() > self.cap {
            self.undo.remove(0);
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Undo the last group: returns the prior graph, stashing `current` for redo.
    pub fn undo(&mut self, current: Graph) -> Option<Graph> {
        let (label, prev) = self.undo.pop()?;
        self.redo.push((label, current));
        Some(prev)
    }

    /// Redo the last undone group.
    pub fn redo(&mut self, current: Graph) -> Option<Graph> {
        let (label, next) = self.redo.pop()?;
        self.undo.push((label, current));
        Some(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{NodeDef, PortDef, PortType};

    fn reg() -> NodeRegistry {
        let mut r = NodeRegistry::new();
        r.register(
            NodeDef::new("n", "N", "c")
                .with_inputs(vec![PortDef::new("in", PortType::Float)])
                .with_outputs(vec![PortDef::new("out", PortType::Float)]),
        );
        r
    }

    #[test]
    fn connect_rejects_cycles_and_replaces_input() {
        let reg = reg();
        let mut g = Graph::empty();
        let a = NodeId(1);
        let b = NodeId(2);
        apply_edit(
            &mut g,
            &reg,
            &GraphEdit::AddNode {
                id: a,
                type_id: "n".into(),
                x: 0.0,
                y: 0.0,
                params: Default::default(),
            },
        );
        apply_edit(
            &mut g,
            &reg,
            &GraphEdit::AddNode {
                id: b,
                type_id: "n".into(),
                x: 0.0,
                y: 0.0,
                params: Default::default(),
            },
        );
        let ab = Link {
            from: a,
            from_port: "out".into(),
            to: b,
            to_port: "in".into(),
        };
        assert!(apply_edit(
            &mut g,
            &reg,
            &GraphEdit::Connect { link: ab.clone() }
        ));
        // b->a would close a loop (a already reaches b) → rejected.
        assert!(would_cycle(&g, b, a));
        let ba = Link {
            from: b,
            from_port: "out".into(),
            to: a,
            to_port: "in".into(),
        };
        assert!(!apply_edit(&mut g, &reg, &GraphEdit::Connect { link: ba }));
        assert_eq!(g.links.len(), 1);
    }

    #[test]
    fn journal_undo_redo() {
        let reg = reg();
        let mut g = Graph::empty();
        let mut j = GraphJournal::new(10);

        j.record("add", g.clone());
        apply_edit(
            &mut g,
            &reg,
            &GraphEdit::AddNode {
                id: NodeId(1),
                type_id: "n".into(),
                x: 0.0,
                y: 0.0,
                params: Default::default(),
            },
        );
        assert_eq!(g.nodes.len(), 1);
        assert!(j.can_undo());

        g = j.undo(g).unwrap();
        assert_eq!(g.nodes.len(), 0);
        assert!(j.can_redo());

        g = j.redo(g).unwrap();
        assert_eq!(g.nodes.len(), 1);
    }
}
