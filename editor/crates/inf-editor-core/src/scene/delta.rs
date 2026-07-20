//! Snapshot diffing (P3.2.1) — "diff-based, not full dumps".
//!
//! Structural changes ship as added/removed/updated node sets; the small
//! whole-tree bits (root order, selection, doc meta) ride along so the frontend
//! reducer stays trivially correct. Two equal snapshots diff to an empty delta.

use std::collections::HashMap;

use crate::ipc::{SceneDelta, SceneNode, SceneSnapshot};

/// Compute the change from `prev` to `next`.
pub fn diff(prev: &SceneSnapshot, next: &SceneSnapshot) -> SceneDelta {
    let prev_map: HashMap<&str, &SceneNode> =
        prev.nodes.iter().map(|n| (n.guid.as_str(), n)).collect();
    let next_map: HashMap<&str, &SceneNode> =
        next.nodes.iter().map(|n| (n.guid.as_str(), n)).collect();

    let mut added = Vec::new();
    let mut updated = Vec::new();
    for n in &next.nodes {
        match prev_map.get(n.guid.as_str()) {
            None => added.push(n.clone()),
            Some(p) if **p != *n => updated.push(n.clone()),
            Some(_) => {}
        }
    }
    let removed: Vec<String> = prev
        .nodes
        .iter()
        .filter(|n| !next_map.contains_key(n.guid.as_str()))
        .map(|n| n.guid.clone())
        .collect();

    SceneDelta {
        version: next.version,
        added,
        removed,
        updated,
        roots: next.roots.clone(),
        selection: next.selection.clone(),
        dirty: next.dirty,
        title: next.title.clone(),
        can_undo: next.can_undo,
        can_redo: next.can_redo,
        undo_label: next.undo_label.clone(),
        redo_label: next.redo_label.clone(),
    }
}

/// Whether a delta carries any change at all (lets Ring 2 skip empty emits).
pub fn is_empty(d: &SceneDelta) -> bool {
    d.added.is_empty() && d.removed.is_empty() && d.updated.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::SpawnKind;
    use crate::scene::SceneDoc;

    #[test]
    fn diff_detects_add_update_remove() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let prev = doc.snapshot();

        let b = doc.create(SpawnKind::Cube, "B", None);
        doc.rename(a, "A2");
        let next = doc.snapshot();

        let d = diff(&prev, &next);
        assert_eq!(d.added.len(), 1);
        assert_eq!(d.added[0].guid, b.to_string());
        assert_eq!(d.updated.len(), 1);
        assert_eq!(d.updated[0].name, "A2");
        assert!(d.removed.is_empty());

        // Delete B → removed.
        let prev = next;
        doc.delete(&[b]);
        let next = doc.snapshot();
        let d = diff(&prev, &next);
        assert_eq!(d.removed, vec![b.to_string()]);
    }

    #[test]
    fn identical_snapshots_diff_empty() {
        let mut doc = SceneDoc::new();
        doc.create(SpawnKind::Empty, "A", None);
        let s = doc.snapshot();
        assert!(is_empty(&diff(&s, &s)));
    }
}
