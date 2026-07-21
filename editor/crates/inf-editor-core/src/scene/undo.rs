//! Undo / redo (P3.4).
//!
//! Every editor mutation is an [`EditCommand`] with an `apply` (redo) and a
//! `revert` (undo) that both go through `SceneDoc`'s **raw** (non-recording)
//! mutations — so undo/redo never re-enter the recorder. Commands group into a
//! [`Transaction`]: a gizmo drag opens one, streams transform edits into it, and
//! commits a single undo entry (P3.4.2). Structural inverses (create/delete)
//! reuse the P3.5 [`EntityRecord`] so a deleted subtree round-trips exactly.

use inf_ecs::components::{Sprite, Transform};
use inf_ecs::PropValue;
use inf_terrain::{HeightDelta, SplatDelta};
use uuid::Uuid;

use crate::scene::serialize::EntityRecord;
use crate::scene::SceneDoc;

/// The default history depth (well past the phase gate's 50 steps).
pub const HISTORY_LIMIT: usize = 256;

pub(crate) enum EditCommand {
    /// `at` is the entity's slot in the creation-order list. The record is boxed
    /// so this variant doesn't bloat the whole enum (an `EntityRecord` grew with
    /// the P8.2b 2D component slots).
    Create {
        at: usize,
        record: Box<EntityRecord>,
    },
    /// A deleted subtree (records with their original order slots) + the top
    /// GUIDs actually removed.
    Delete {
        items: Vec<(usize, EntityRecord)>,
        tops: Vec<Uuid>,
    },
    Rename {
        guid: Uuid,
        before: String,
        after: String,
    },
    Reparent {
        guid: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
    },
    SetVisible {
        guid: Uuid,
        before: bool,
        after: bool,
    },
    SetTransform {
        guid: Uuid,
        before: Transform,
        after: Transform,
    },
    SetProp {
        guid: Uuid,
        type_path: String,
        field: String,
        before: PropValue,
        after: PropValue,
    },
    /// Whole-component `Sprite` swap (P8.2a). The `Sprite` fields the slicer
    /// writes (`texture`, `atlas_rect`) aren't reflection-addressable, so the
    /// component round-trips as a value; `None` means "no `Sprite` component".
    SetSprite {
        guid: Uuid,
        before: Option<Sprite>,
        after: Option<Sprite>,
    },
    /// One tile-painting stroke (P8.2b). Stores the pre/post index of **only the
    /// touched cells** — never a whole chunk map — so a stroke over a large map
    /// stays cheap. `cells` is `(x, y, before, after)`.
    SetTiles {
        guid: Uuid,
        cells: Vec<(i32, i32, u32, u32)>,
    },
    /// Whole-component `ActorClass` swap (P9.5): the blueprint-class binding GUID
    /// (a non-reflected identity link) round-trips as a value; `None` = unbound.
    SetActor {
        guid: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
    },
    /// One terrain sculpt stroke (P10.2b): a sparse before/after height-sample
    /// record ([`HeightDelta`]). The live stroke already mutated the terrain when
    /// this is recorded, so `apply`/`revert` here are pure redo/undo — redo
    /// replays the `after` samples, undo replays `before` (and drops any tiles the
    /// stroke authored from nothing). Boxed so the (potentially large) delta
    /// doesn't bloat every other command variant.
    SculptTerrain { guid: Uuid, delta: Box<HeightDelta> },
    /// One terrain splat-paint stroke (P10.4): a sparse before/after weight-sample
    /// record ([`SplatDelta`]). Like [`SculptTerrain`](EditCommand::SculptTerrain)
    /// the live stroke already mutated the weights, so `apply`/`revert` are pure
    /// redo/undo — redo replays `after` weights, undo replays `before` (and drops
    /// any weight buffers the stroke materialized from the sparse default). Boxed.
    PaintSplat { guid: Uuid, delta: Box<SplatDelta> },
}

impl EditCommand {
    /// Do (or redo) the edit.
    pub(crate) fn apply(&self, doc: &mut SceneDoc) {
        match self {
            EditCommand::Create { at, record } => doc.raw_spawn_record(record, *at),
            EditCommand::Delete { tops, .. } => doc.raw_delete(tops),
            EditCommand::Rename { guid, after, .. } => doc.raw_rename(*guid, after),
            EditCommand::Reparent { guid, after, .. } => {
                doc.raw_reparent(*guid, *after);
            }
            EditCommand::SetVisible { guid, after, .. } => doc.raw_set_visible(*guid, *after),
            EditCommand::SetTransform { guid, after, .. } => doc.raw_set_transform(*guid, *after),
            EditCommand::SetProp {
                guid,
                type_path,
                field,
                after,
                ..
            } => {
                doc.raw_write_prop(*guid, type_path, field, after);
            }
            EditCommand::SetSprite { guid, after, .. } => {
                doc.raw_set_sprite(*guid, after.clone());
            }
            EditCommand::SetTiles { guid, cells } => {
                let after: Vec<(i32, i32, u32)> =
                    cells.iter().map(|&(x, y, _, a)| (x, y, a)).collect();
                doc.raw_set_tiles(*guid, &after);
            }
            EditCommand::SetActor { guid, after, .. } => {
                doc.raw_set_actor(*guid, *after);
            }
            EditCommand::SculptTerrain { guid, delta } => {
                doc.raw_apply_terrain_delta(*guid, delta);
            }
            EditCommand::PaintSplat { guid, delta } => {
                doc.raw_apply_splat_delta(*guid, delta);
            }
        }
    }

    /// Undo the edit.
    pub(crate) fn revert(&self, doc: &mut SceneDoc) {
        match self {
            EditCommand::Create { record, .. } => doc.raw_delete(&[record.guid]),
            EditCommand::Delete { items, .. } => {
                // Recreate in ascending original-slot order so parents precede
                // children (creation order guarantees that).
                let mut items: Vec<&(usize, EntityRecord)> = items.iter().collect();
                items.sort_by_key(|(at, _)| *at);
                for (at, record) in items {
                    doc.raw_spawn_record(record, *at);
                }
            }
            EditCommand::Rename { guid, before, .. } => doc.raw_rename(*guid, before),
            EditCommand::Reparent { guid, before, .. } => {
                doc.raw_reparent(*guid, *before);
            }
            EditCommand::SetVisible { guid, before, .. } => doc.raw_set_visible(*guid, *before),
            EditCommand::SetTransform { guid, before, .. } => doc.raw_set_transform(*guid, *before),
            EditCommand::SetProp {
                guid,
                type_path,
                field,
                before,
                ..
            } => {
                doc.raw_write_prop(*guid, type_path, field, before);
            }
            EditCommand::SetSprite { guid, before, .. } => {
                doc.raw_set_sprite(*guid, before.clone());
            }
            EditCommand::SetTiles { guid, cells } => {
                let before: Vec<(i32, i32, u32)> =
                    cells.iter().map(|&(x, y, b, _)| (x, y, b)).collect();
                doc.raw_set_tiles(*guid, &before);
            }
            EditCommand::SetActor { guid, before, .. } => {
                doc.raw_set_actor(*guid, *before);
            }
            EditCommand::SculptTerrain { guid, delta } => {
                doc.raw_revert_terrain_delta(*guid, delta);
            }
            EditCommand::PaintSplat { guid, delta } => {
                doc.raw_revert_splat_delta(*guid, delta);
            }
        }
    }
}

pub(crate) struct Transaction {
    pub label: String,
    pub commands: Vec<EditCommand>,
}

/// The undo/redo stacks plus the currently-open transaction.
pub struct EditHistory {
    undo: Vec<Transaction>,
    redo: Vec<Transaction>,
    open: Option<Transaction>,
    limit: usize,
}

impl Default for EditHistory {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            open: None,
            limit: HISTORY_LIMIT,
        }
    }
}

impl EditHistory {
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Number of undo entries currently on the stack (bounded by [`HISTORY_LIMIT`]).
    /// Surfaced by the memory diagnostics (P15) — the "undo stack depth" budget.
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// Number of redo entries currently on the stack.
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Label of the next undo/redo entry (for the Edit menu: "Undo Rename").
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|t| t.label.as_str())
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|t| t.label.as_str())
    }

    pub(crate) fn begin(&mut self, label: &str) {
        // Nested begins fold into the outer transaction.
        if self.open.is_none() {
            self.open = Some(Transaction {
                label: label.to_string(),
                commands: Vec::new(),
            });
        }
    }

    /// Record a command: append to the open transaction, or commit it as a
    /// standalone entry labelled `label`. Clears the redo stack (a new edit
    /// forks history).
    pub(crate) fn record(&mut self, label: &str, cmd: EditCommand) {
        if let Some(open) = self.open.as_mut() {
            open.commands.push(cmd);
        } else {
            self.push(Transaction {
                label: label.to_string(),
                commands: vec![cmd],
            });
        }
    }

    pub(crate) fn commit(&mut self) {
        if let Some(txn) = self.open.take() {
            if !txn.commands.is_empty() {
                self.push(txn);
            }
        }
    }

    fn push(&mut self, txn: Transaction) {
        self.redo.clear();
        self.undo.push(txn);
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
    }

    pub(crate) fn take_undo(&mut self) -> Option<Transaction> {
        self.undo.pop()
    }

    pub(crate) fn take_redo(&mut self) -> Option<Transaction> {
        self.redo.pop()
    }

    pub(crate) fn push_redo(&mut self, txn: Transaction) {
        self.redo.push(txn);
    }

    pub(crate) fn push_undo(&mut self, txn: Transaction) {
        self.undo.push(txn);
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.open = None;
    }
}

#[cfg(test)]
mod tests {
    use crate::ipc::{SceneNode, SpawnKind};
    use crate::scene::SceneDoc;
    use inf_ecs::components::Transform;
    use inf_ecs::math::Vec3d;

    /// Order-independent scene comparison (name/kind/parent/visibility).
    fn fingerprint(doc: &mut SceneDoc) -> Vec<SceneNode> {
        let mut nodes = doc.snapshot().nodes;
        nodes.sort_by(|a, b| a.guid.cmp(&b.guid));
        for n in &mut nodes {
            n.children.sort();
        }
        nodes
    }

    #[test]
    fn undo_redo_restores_each_mutation() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Empty, "A", None);
        let before = fingerprint(&mut doc);

        // A rename, a child create, a reparent, a visibility toggle.
        doc.edit_rename(a, "Renamed");
        let b = doc.edit_create(SpawnKind::Cube, "B", Some(a));
        doc.edit_reparent(b, None);
        doc.edit_set_visible(a, false);

        // Four independent edits → four undo steps back to `before`.
        for _ in 0..4 {
            assert!(doc.undo());
        }
        assert_eq!(fingerprint(&mut doc), before, "undo did not restore state");

        // Redo them all forward again.
        for _ in 0..4 {
            assert!(doc.redo());
        }
        assert!(!doc.can_redo());
    }

    #[test]
    fn fifty_step_undo_redo_is_clean() {
        let mut doc = SceneDoc::new();
        let base = fingerprint(&mut doc);
        let mut guids = Vec::new();
        for i in 0..50 {
            guids.push(doc.edit_create(SpawnKind::Cube, &format!("C{i}"), None));
        }
        let full = fingerprint(&mut doc);

        for _ in 0..50 {
            assert!(doc.undo());
        }
        assert_eq!(fingerprint(&mut doc), base, "50 undos must reach the start");
        assert!(!doc.can_undo());

        for _ in 0..50 {
            assert!(doc.redo());
        }
        assert_eq!(
            fingerprint(&mut doc),
            full,
            "50 redos must restore everything"
        );
    }

    fn translation_x(doc: &SceneDoc, guid: uuid::Uuid) -> f64 {
        let props = doc.entity_props(guid);
        let t = props.iter().find(|p| p.display == "Transform").unwrap();
        match &t
            .fields
            .iter()
            .find(|f| f.name == "translation")
            .unwrap()
            .value
        {
            inf_ecs::PropValue::Vec3(v) => v[0],
            _ => panic!("translation not a vec3"),
        }
    }

    #[test]
    fn transaction_groups_into_one_step() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Cube, "A", None);

        // A gizmo drag: many transform edits stream into one transaction.
        doc.begin_transaction("drag");
        for i in 1..=10 {
            doc.edit_set_transform(
                a,
                Transform {
                    translation: Vec3d::new(i as f64, 0.0, 0.0),
                    ..Transform::IDENTITY
                },
            );
        }
        doc.commit_transaction();
        assert_eq!(translation_x(&doc, a), 10.0);

        // A single undo reverts the whole drag back to the origin (not just the
        // last of the ten edits).
        assert!(doc.undo());
        assert_eq!(translation_x(&doc, a), 0.0, "the drag is one undo step");
        // The remaining undo entry is the create.
        assert!(doc.undo());
        assert!(!doc.can_undo());
    }

    #[test]
    fn delete_undo_restores_subtree() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Empty, "A", None);
        let _b = doc.edit_create(SpawnKind::Cube, "B", Some(a));
        let _c = doc.edit_create(SpawnKind::Sphere, "C", Some(a));
        let full = fingerprint(&mut doc);

        doc.edit_delete(&[a]);
        assert!(doc.snapshot().nodes.is_empty());

        assert!(doc.undo());
        assert_eq!(
            fingerprint(&mut doc),
            full,
            "deleted subtree must round-trip"
        );
    }
}
