//! Snapshot diffing (P3.2.1) — "diff-based, not full dumps".
//!
//! Structural changes ship as added/removed/updated node sets; the small
//! whole-tree bits (selection, doc meta) ride along so the frontend reducer
//! stays trivially correct. Two equal snapshots diff to an empty delta.
//!
//! # This is no longer the production path (IB-13)
//!
//! [`diff`] compares two whole `SceneSnapshot`s, so a `world://delta` cost one
//! full projection plus one full comparison — **52.857 ms at 100 000 entities**,
//! on every gizmo-drag mouse-move. `SceneDoc::project_delta` replaced it: the
//! document knows which entities a mutation named and which guid set it last
//! published, so a select frame is **0.0006 ms** at the same size and a drag
//! frame is 8.105 ms, of which the projection is a rounding error and the rest
//! is `EcsWorld::propagate`.
//!
//! `diff` stays because it is the **oracle**: it is a different computation over
//! the same inputs, and
//! `the_scoped_projection_never_misses_a_change_and_never_states_a_stale_one`
//! runs both over one script and requires the scoped answer to contain the full
//! one. A fast projection with no independent statement of what it should have
//! said would be unfalsifiable.
//!
//! The root list is the one whole-tree bit that stopped riding along; see
//! [`SceneDelta::roots`](crate::ipc::SceneDelta::roots).

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
        roots: Some(next.roots.clone()),
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
    use uuid::Uuid;

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

    /// **`project_delta` never omits a change, and never states a stale one**
    /// (IB-13).
    ///
    /// The scoped projection is a *different computation* from the one it
    /// replaces — it never builds the other 99 999 nodes — so the gate runs both
    /// over the same script and compares them. The relation is a *containment*,
    /// not an equality, and the asymmetry is the design rather than a weakness:
    ///
    /// * every node the full `snapshot` + `diff` would have shipped **is** in
    ///   the scoped delta (nothing is missed — the failure that matters);
    /// * every node the scoped delta ships **equals the current snapshot's**
    ///   (nothing is stale — the other failure that matters);
    /// * the scoped delta may ship *extra* nodes, because a named entity is
    ///   re-projected whether or not the write changed anything a `SceneNode`
    ///   carries. A transform write is exactly that case, and the redundant
    ///   restatement is one node against a hundred thousand.
    ///
    /// Both documents are built with the SAME guids, because two documents
    /// minting their own would make every node compare unequal for a reason that
    /// has nothing to do with the claim.
    #[test]
    fn the_scoped_projection_never_misses_a_change_and_never_states_a_stale_one() {
        let mut scoped = SceneDoc::new();
        let mut full = SceneDoc::new();
        let mut prev = full.snapshot();
        let _ = scoped.snapshot();
        let g = |i: u128| Uuid::from_u128(0x13_0000 + i);

        let mut step = |scoped: &mut SceneDoc,
                        full: &mut SceneDoc,
                        prev: &mut crate::ipc::SceneSnapshot,
                        label: &str,
                        f: &dyn Fn(&mut SceneDoc)| {
            let prev_roots = prev.roots.clone();
            f(scoped);
            f(full);
            let a = scoped.project_delta();
            let next = full.snapshot();
            let b = diff(prev, &next);

            let a_nodes: Vec<&SceneNode> = a.added.iter().chain(&a.updated).collect();
            for want in b.added.iter().chain(&b.updated) {
                assert!(
                    a_nodes.iter().any(|n| **n == *want),
                    "[{label}] the scoped delta MISSED `{}` ({}), which the full \
                     diff shipped",
                    want.name,
                    want.guid
                );
            }
            for got in &a_nodes {
                let truth = next
                    .nodes
                    .iter()
                    .find(|n| n.guid == got.guid)
                    .unwrap_or_else(|| panic!("[{label}] shipped a dead node {}", got.guid));
                assert_eq!(**got, *truth, "[{label}] the scoped delta is stale");
            }
            let (mut ar, mut br) = (a.removed.clone(), b.removed.clone());
            ar.sort();
            br.sort();
            assert_eq!(ar, br, "[{label}] the removals differ");
            // An omitted root list is legal **only** when the roots did not
            // move. That is a stronger claim than the equality it replaces: it
            // fails both if the shipped list is wrong AND if a list that changed
            // was not shipped, which is the failure the `Option` introduces.
            match &a.roots {
                Some(r) => assert_eq!(*r, next.roots, "[{label}] the root list is wrong"),
                None => assert_eq!(
                    prev_roots, next.roots,
                    "[{label}] the delta omitted a root list that CHANGED — the \
                     Outliner would keep drawing the old tree"
                ),
            }
            assert_eq!(a.selection, b.selection, "[{label}] the selection differs");
            assert_eq!(a.dirty, b.dirty, "[{label}] the dirty flag differs");
            *prev = next;
        };

        for (i, kind) in [SpawnKind::Empty, SpawnKind::Cube, SpawnKind::Sphere]
            .into_iter()
            .enumerate()
        {
            let name = format!("N{i}");
            let id = g(i as u128);
            step(&mut scoped, &mut full, &mut prev, "create", &|d| {
                d.create_with_guid(id, kind, &name, None);
            });
        }

        step(&mut scoped, &mut full, &mut prev, "rename", &|d| {
            d.rename(g(1), "renamed");
        });
        step(&mut scoped, &mut full, &mut prev, "hide", &|d| {
            d.set_visible(g(1), false);
        });
        step(&mut scoped, &mut full, &mut prev, "select", &|d| {
            d.select(&[g(2)], false);
        });
        step(&mut scoped, &mut full, &mut prev, "reparent", &|d| {
            d.reparent(g(2), Some(g(0)));
        });
        // …and now the parent is hidden, which must reach the CHILD's
        // `effective_visible` — the case a per-entity scope gets wrong, and the
        // reason `set_visible` uses `touch_subtree`.
        step(&mut scoped, &mut full, &mut prev, "hide a parent", &|d| {
            d.set_visible(g(0), false);
        });
        step(&mut scoped, &mut full, &mut prev, "delete", &|d| {
            d.delete(&[g(1)]);
        });
        step(&mut scoped, &mut full, &mut prev, "unparent", &|d| {
            d.reparent(g(2), None);
        });
    }

    /// A reparent moves the child, its old parent and its new parent — three
    /// nodes, of which a per-entity scope names one. It is a `touch`, and
    /// `touch` means everything, and this is what says so.
    #[test]
    fn a_reparent_reaches_the_delta_through_every_node_it_moves() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let b = doc.create(SpawnKind::Empty, "B", None);
        let c = doc.create(SpawnKind::Cube, "C", Some(a));
        let _ = doc.snapshot();

        assert!(doc.reparent(c, Some(b)));
        let d = doc.project_delta();
        let touched: std::collections::BTreeSet<String> = d
            .added
            .iter()
            .chain(&d.updated)
            .map(|n| n.guid.clone())
            .collect();
        for (label, g) in [
            ("the child", c),
            ("the old parent", a),
            ("the new parent", b),
        ] {
            assert!(
                touched.contains(&g.to_string()),
                "{label} is missing from the delta — the Outliner would draw it \
                 under both parents"
            );
        }
        // And the node contents really are right, not merely present.
        let node = |g: Uuid| {
            d.added
                .iter()
                .chain(&d.updated)
                .find(|n| n.guid == g.to_string())
                .expect("in the delta")
        };
        assert!(node(a).children.is_empty(), "the old parent kept the child");
        assert_eq!(node(b).children, vec![c.to_string()]);
        assert_eq!(node(c).parent, Some(b.to_string()));
    }

    /// **A gizmo drag ships ONE node, not the whole tree** — the frame this
    /// whole item exists to make cheap.
    ///
    /// One rather than zero: a `SceneNode` carries no transform, so the node is
    /// a redundant restatement of what the frontend already holds. Suppressing it
    /// would mean retaining the previously published node to compare against,
    /// which is the retained-snapshot memory this item just removed — one node is
    /// the cheaper side of that trade by a factor of the level's size.
    #[test]
    fn a_transform_write_ships_one_node_not_the_whole_tree() {
        use inf_ecs::components::Transform;
        let mut doc = SceneDoc::new();
        let g = doc.create(SpawnKind::Cube, "Prop", None);
        for i in 0..8 {
            doc.create(SpawnKind::Cube, &format!("Other {i}"), None);
        }
        let _ = doc.snapshot();

        let before = doc.version();
        doc.raw_set_transform(
            g,
            Transform {
                translation: inf_ecs::Vec3d::new(1.0, 2.0, 3.0),
                ..Transform::IDENTITY
            },
        );
        let d = doc.project_delta();
        assert!(d.version > before, "the viewport re-syncs on the version");
        assert_eq!(
            d.added.len() + d.updated.len(),
            1,
            "a drag frame shipped {} nodes of 9",
            d.added.len() + d.updated.len()
        );
        assert_eq!(d.updated[0].guid, g.to_string());
        assert!(d.removed.is_empty());
        assert!(d.dirty, "…but the document IS unsaved");
        assert!(
            d.roots.is_none(),
            "a drag frame re-shipped the whole root list — at 100 000 entities \
             that clone alone measured 3.496 ms"
        );

        // ANTI-VACUITY: the same document under a `touch()` — what every
        // unconverted call site still does — ships all nine.
        doc.rename(g, "Prop");
        let _ = doc.project_delta();
        doc.delete(&[]); // a no-subject delete still calls `touch()`
        let wide = doc.project_delta();
        assert_eq!(
            wide.added.len() + wide.updated.len(),
            9,
            "an unscoped touch must still project everything"
        );
    }
}
