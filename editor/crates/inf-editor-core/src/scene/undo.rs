//! Undo / redo (P3.4).
//!
//! Every editor mutation is an [`EditCommand`] with an `apply` (redo) and a
//! `revert` (undo) that both go through `SceneDoc`'s **raw** (non-recording)
//! mutations — so undo/redo never re-enter the recorder. Commands group into a
//! [`Transaction`]: a gizmo drag opens one, streams transform edits into it, and
//! commits a single undo entry (P3.4.2). Structural inverses (create/delete)
//! reuse the P3.5 [`EntityRecord`] so a deleted subtree round-trips exactly.

use inf_ecs::components::{FoliageInstance, Sprite, Transform};
use inf_ecs::PropValue;
use inf_terrain::{BiomeDelta, DataMapDelta, HeightDelta, SplatDelta};
use uuid::Uuid;

use crate::scene::serialize::{EntityRecord, LevelSettings};
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
    /// A whole-file level-settings swap (R-P4): gravity / sim rate / the persisted
    /// render (post/exposure/lighting) block round-trip as one value. `Copy` +
    /// small, so this variant is stored inline (no boxing).
    SetLevelSettings {
        old: LevelSettings,
        new: LevelSettings,
    },
    /// One terrain splat-paint stroke (P10.4): a sparse before/after weight-sample
    /// record ([`SplatDelta`]). Like [`SculptTerrain`](EditCommand::SculptTerrain)
    /// the live stroke already mutated the weights, so `apply`/`revert` are pure
    /// redo/undo — redo replays `after` weights, undo replays `before` (and drops
    /// any weight buffers the stroke materialized from the sparse default). Boxed.
    PaintSplat { guid: Uuid, delta: Box<SplatDelta> },
    /// One terrain **biome-paint** stroke (P19.2): a sparse before/after record of
    /// the per-sample biome ids it wrote ([`BiomeDelta`]).
    ///
    /// A sibling of [`PaintSplat`](EditCommand::PaintSplat), not a field on it,
    /// for the same reason `PaintSplat` is a sibling of
    /// [`SculptTerrain`](EditCommand::SculptTerrain): the payloads are genuinely
    /// different layers with different producers, and merging them would put an
    /// always-empty buffer inside every stroke on the undo stack. Like both, the
    /// live stroke already mutated the terrain, so `apply`/`revert` are pure
    /// redo/undo — undo replays `before` and drops any id buffers the stroke
    /// materialized from the sparse default. Boxed.
    PaintBiome { guid: Uuid, delta: Box<BiomeDelta> },
    /// One erosion bake's **data-map** half (P19.1): a sparse before/after record
    /// of the flow / deposition / wear accumulators it moved ([`DataMapDelta`]).
    ///
    /// Recorded *beside* the bake's [`SculptTerrain`](EditCommand::SculptTerrain)
    /// inside a single transaction, so one Ctrl+Z restores both layers. It is a
    /// separate command rather than a field on the height delta because the two
    /// layers have different producers: every sculpt brush writes heights, only
    /// erosion writes data maps, and folding them together would put an
    /// always-empty map buffer inside every stroke on the undo stack. Boxed.
    WriteDataMaps {
        guid: Uuid,
        delta: Box<DataMapDelta>,
    },
    /// Add / remove a whole component (E-P1). `before`/`after` are full entity
    /// component snapshots ([`EntityRecord`] via `record_of`) — the record is the
    /// complete truth, so replaying either side re-inserts what it holds and
    /// removes every optional component it leaves `None`
    /// (`raw_apply_record_components`). Boxed so the (large) record doesn't bloat
    /// the other variants. Covers add (`before` lacks the component, `after` has
    /// it) and remove (the reverse) with one code path.
    SwapComponents {
        guid: Uuid,
        before: Box<EntityRecord>,
        after: Box<EntityRecord>,
    },
    /// One foliage scatter stroke (E-P6): the instances added and/or the
    /// `(index, instance)` pairs removed. Like the terrain deltas the live stroke
    /// already mutated the component, so `apply`/`revert` are pure redo/undo — redo
    /// removes the `removed` indices (descending) then pushes `added`; undo pops
    /// `added` off the end then re-inserts `removed` at their original indices.
    /// A stroke is add-XOR-erase, so one vector is always empty.
    PaintFoliage {
        guid: Uuid,
        added: Vec<FoliageInstance>,
        removed: Vec<(usize, FoliageInstance)>,
    },
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
            EditCommand::SetLevelSettings { new, .. } => {
                doc.raw_set_settings(*new);
            }
            EditCommand::SculptTerrain { guid, delta } => {
                doc.raw_apply_terrain_delta(*guid, delta);
            }
            EditCommand::PaintSplat { guid, delta } => {
                doc.raw_apply_splat_delta(*guid, delta);
            }
            EditCommand::PaintBiome { guid, delta } => {
                doc.raw_apply_biome_delta(*guid, delta);
            }
            EditCommand::WriteDataMaps { guid, delta } => {
                doc.raw_apply_data_map_delta(*guid, delta);
            }
            EditCommand::SwapComponents { guid, after, .. } => {
                doc.raw_apply_record_components(*guid, after);
            }
            EditCommand::PaintFoliage {
                guid,
                added,
                removed,
            } => {
                doc.raw_apply_foliage(*guid, added, removed);
            }
        }
    }

    /// Undo the edit.
    pub(crate) fn revert(&self, doc: &mut SceneDoc) {
        match self {
            EditCommand::Create { record, .. } => doc.raw_delete(&[record.guid]),
            EditCommand::Delete { items, .. } => {
                // Two passes so the hierarchy survives regardless of the order
                // slots: (1) respawn every record at its slot — a record whose
                // parent sits at a LATER slot (a reparent under a later-created
                // node) spawns to the root because its parent isn't back yet;
                // (2) fix up every parent link now that all GUIDs exist again.
                // The second pass is a no-op for the common parents-precede-
                // children ordering.
                let mut items: Vec<&(usize, EntityRecord)> = items.iter().collect();
                items.sort_by_key(|(at, _)| *at);
                for (at, record) in &items {
                    doc.raw_spawn_record(record, *at);
                }
                for (_, record) in &items {
                    doc.raw_fixup_parent(record.guid, record.parent);
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
            EditCommand::SetLevelSettings { old, .. } => {
                doc.raw_set_settings(*old);
            }
            EditCommand::SculptTerrain { guid, delta } => {
                doc.raw_revert_terrain_delta(*guid, delta);
            }
            EditCommand::PaintSplat { guid, delta } => {
                doc.raw_revert_splat_delta(*guid, delta);
            }
            EditCommand::PaintBiome { guid, delta } => {
                doc.raw_revert_biome_delta(*guid, delta);
            }
            EditCommand::WriteDataMaps { guid, delta } => {
                doc.raw_revert_data_map_delta(*guid, delta);
            }
            EditCommand::SwapComponents { guid, before, .. } => {
                doc.raw_apply_record_components(*guid, before);
            }
            EditCommand::PaintFoliage {
                guid,
                added,
                removed,
            } => {
                doc.raw_revert_foliage(*guid, added, removed);
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
    /// Open-transaction nesting depth. `begin` increments it, `commit`
    /// decrements it; the transaction closes only when the OUTERMOST commit
    /// brings this back to zero, so begin/begin/commit/commit nests correctly.
    depth: u32,
    limit: usize,
}

impl Default for EditHistory {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            open: None,
            depth: 0,
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
        // Nested begins fold into the outer transaction; the depth counter tracks
        // the nesting so only the matching outermost commit closes it.
        if self.open.is_none() {
            self.open = Some(Transaction {
                label: label.to_string(),
                commands: Vec::new(),
            });
        }
        self.depth += 1;
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
        // Only the outermost commit closes the transaction; inner commits just
        // unwind one level of nesting.
        if self.depth > 1 {
            self.depth -= 1;
            return;
        }
        self.depth = 0;
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
        self.depth = 0;
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

    /// Deleting a reparented pair together and undoing restores the hierarchy
    /// even when the child sits at an EARLIER order slot than its parent (A
    /// created before B, then A reparented under B). The two-pass respawn
    /// (spawn-all, then fix-up-parents) re-attaches A under B instead of
    /// silently rooting it.
    #[test]
    fn delete_undo_restores_reparent_under_later_node() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Cube, "A", None);
        let b = doc.edit_create(SpawnKind::Empty, "B", None); // later order slot
        assert!(doc.edit_reparent(a, Some(b)));

        doc.edit_delete(&[a, b]);
        assert!(doc.snapshot().nodes.is_empty(), "both deleted");
        assert!(doc.undo());

        // A is restored UNDER B (not as a root).
        let ea = doc.entity_of(a).expect("A restored by undo");
        let parent = doc
            .world()
            .parent_of(ea)
            .and_then(|p| doc.world().guid_of(p));
        assert_eq!(parent, Some(b), "A stays under B after delete→undo");
        assert!(doc.entity_of(b).is_some(), "B restored by undo");
    }

    /// Editing the level settings dirties + bumps the version, and undo/redo
    /// round-trip the whole settings block (gravity + render/post) exactly (R-P4).
    #[test]
    fn level_settings_edit_undo_redo_round_trips() {
        use crate::scene::serialize::LevelSettings;

        let mut doc = SceneDoc::new();
        let base = doc.settings();
        let v0 = doc.version();
        assert!(!doc.is_dirty());

        let mut edited = base;
        edited.render.exposure = 2.5;
        edited.render.bloom_enabled = true;
        edited.sim_hz = 120.0;
        doc.edit_settings(edited);

        assert!(doc.is_dirty(), "a settings edit dirties the document");
        assert!(doc.version() > v0, "a settings edit bumps the version");
        assert_eq!(doc.settings(), edited);

        // Undo restores the original settings exactly …
        assert!(doc.undo());
        assert_eq!(doc.settings(), base);
        // … and redo re-applies the edited block.
        assert!(doc.redo());
        assert_eq!(doc.settings(), edited);

        // An idempotent edit (same value) records nothing.
        let before_len = doc.undo_len();
        doc.edit_settings(edited);
        assert_eq!(doc.undo_len(), before_len, "no-op edit records nothing");
        assert_eq!(doc.settings(), LevelSettings { ..edited });
    }

    /// P17.1: editing the time of day from World Settings is **one** undo step,
    /// and the first edit creates the sky authority inside that same step — so a
    /// single Ctrl+Z takes the whole opt-in back out, entity and all.
    #[test]
    fn time_of_day_edit_creates_the_authority_in_one_undo_step() {
        use inf_ecs::components::TimeOfDay;

        let mut doc = SceneDoc::new();
        let entities = doc.order().len();
        let undos = doc.undo_len();
        assert!(doc.time_of_day().is_none(), "a new level has no clock");

        let authored = TimeOfDay {
            seconds: 43_200.0,
            day_of_year: 355,
            latitude_deg: -33.9,
            longitude_deg: 151.2,
            rate: 60.0,
        };
        let guid = doc.edit_time_of_day(authored, true).expect("created");

        assert_eq!(doc.time_of_day(), Some(authored));
        assert_eq!(doc.sky_authority(), Some(guid));
        assert_eq!(doc.order().len(), entities + 1, "one Sky actor appeared");
        assert_eq!(
            doc.undo_len(),
            undos + 1,
            "create + two components + five fields collapse into ONE step"
        );
        assert!(doc.is_dirty());

        // Undo removes the whole opt-in …
        assert!(doc.undo());
        assert!(doc.time_of_day().is_none());
        assert_eq!(doc.order().len(), entities);
        // … and redo restores it exactly.
        assert!(doc.redo());
        assert_eq!(doc.time_of_day(), Some(authored));

        // A second edit re-uses the authority (no new entity) and is still one step.
        let undos = doc.undo_len();
        let later = TimeOfDay {
            seconds: 0.0,
            ..authored
        };
        assert_eq!(doc.edit_time_of_day(later, false), Some(guid));
        assert_eq!(doc.order().len(), entities + 1);
        assert_eq!(doc.undo_len(), undos + 1);
        assert_eq!(doc.time_of_day(), Some(later));

        // An idempotent edit records nothing.
        let undos = doc.undo_len();
        doc.edit_time_of_day(later, true);
        assert_eq!(doc.undo_len(), undos, "no-op edit records nothing");
    }

    /// P17.1 finding-3 guard: `create: false` on a level with **no** clock must be
    /// a total no-op — no entity, no undo entry, no version bump, no dirty flag.
    ///
    /// This is what stops an unrelated World Settings write (gravity, sim rate, the
    /// partition block) from conjuring a `Sky` actor out of the time-of-day
    /// *preview* values the panel round-trips. The flag used to live only in the
    /// Ring-2 command, where nothing could test it.
    #[test]
    fn time_of_day_without_create_never_conjures_an_authority() {
        use inf_ecs::components::TimeOfDay;

        let mut doc = SceneDoc::new();
        let entities = doc.order().len();
        let undos = doc.undo_len();
        let version = doc.version();
        assert!(doc.time_of_day().is_none());

        let previewed = TimeOfDay {
            seconds: 1234.0,
            ..TimeOfDay::default()
        };
        assert_eq!(doc.edit_time_of_day(previewed, false), None);

        assert!(doc.time_of_day().is_none(), "no clock was created");
        assert!(doc.sky_authority().is_none());
        assert_eq!(doc.order().len(), entities, "no entity appeared");
        assert_eq!(doc.undo_len(), undos, "nothing was recorded");
        assert_eq!(doc.version(), version, "the version did not bump");
        assert!(!doc.is_dirty(), "the document was not dirtied");

        // …and `create: true` on the same doc does the whole opt-in in one step.
        let guid = doc.edit_time_of_day(previewed, true).expect("created");
        assert_eq!(doc.sky_authority(), Some(guid));
        assert_eq!(doc.time_of_day(), Some(previewed));
        assert_eq!(doc.order().len(), entities + 1);
        assert_eq!(doc.undo_len(), undos + 1);
        assert!(doc.is_dirty());

        // Once an authority exists, `create: false` still writes it — the flag
        // governs creation only.
        let later = TimeOfDay {
            seconds: 4321.0,
            ..previewed
        };
        assert_eq!(doc.edit_time_of_day(later, false), Some(guid));
        assert_eq!(doc.time_of_day(), Some(later));
        assert_eq!(
            doc.order().len(),
            entities + 1,
            "still exactly one Sky actor"
        );
    }

    /// Nested `begin`/`commit` collapse into ONE undo step: an inner commit must
    /// not close the outer transaction (only the outermost commit does).
    #[test]
    fn nested_transactions_close_on_outermost_commit() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Cube, "A", None);

        let at = |x: f64| Transform {
            translation: Vec3d::new(x, 0.0, 0.0),
            ..Transform::IDENTITY
        };
        doc.begin_transaction("outer");
        doc.edit_set_transform(a, at(1.0));
        doc.begin_transaction("inner");
        doc.edit_set_transform(a, at(2.0));
        doc.commit_transaction(); // inner: must NOT close the transaction
        doc.edit_set_transform(a, at(3.0));
        doc.commit_transaction(); // outer: closes now
        assert_eq!(translation_x(&doc, a), 3.0);

        // One undo reverts all three edits (a single grouped step), not just the
        // last — the discriminator against the old fold-on-first-commit bug.
        assert!(doc.undo());
        assert_eq!(
            translation_x(&doc, a),
            0.0,
            "nested begins collapse into one undo step"
        );
        // The remaining entry is the create.
        assert!(doc.undo());
        assert!(!doc.can_undo());
    }

    /// P17.2: the atmosphere is edited through the same one-step door as the
    /// clock, on the same authority — and the first edit on a clockless level
    /// creates BOTH components inside that single step, because an atmosphere
    /// with no clock beside it is the inert shape `inf_ecs::sky` warns about.
    #[test]
    fn sky_atmosphere_edit_creates_the_authority_in_one_undo_step() {
        use inf_ecs::components::SkyAtmosphere;

        let mut doc = SceneDoc::new();
        let entities = doc.order().len();
        let undos = doc.undo_len();
        assert!(doc.sky_atmosphere().is_none(), "a new level has no sky");

        let authored = SkyAtmosphere {
            physical: true,
            turbidity: 2.5,
            fog_density: 6.0e-4,
            fog_height: 120.0,
            star_intensity: 0.4,
            ..SkyAtmosphere::default()
        };
        let guid = doc.edit_sky_atmosphere(authored, true).expect("created");

        assert_eq!(doc.sky_atmosphere(), Some(authored));
        assert_eq!(doc.sky_authority(), Some(guid));
        // The clock came with it, at its defaults — the level is now a complete
        // sky authority, not half of one.
        assert_eq!(
            doc.time_of_day(),
            Some(inf_ecs::components::TimeOfDay::default())
        );
        assert_eq!(doc.order().len(), entities + 1, "one Sky actor appeared");
        assert_eq!(
            doc.undo_len(),
            undos + 1,
            "create + both components + every field collapse into ONE step"
        );

        // One Ctrl+Z takes the whole opt-in back out; redo restores it exactly.
        assert!(doc.undo());
        assert!(doc.sky_atmosphere().is_none());
        assert_eq!(doc.order().len(), entities);
        assert!(doc.redo());
        assert_eq!(doc.sky_atmosphere(), Some(authored));

        // An idempotent edit records nothing.
        let undos = doc.undo_len();
        doc.edit_sky_atmosphere(authored, true);
        assert_eq!(doc.undo_len(), undos, "no-op edit records nothing");
    }

    /// The same finding-3 guard the clock has: `create: false` on a level with no
    /// authority must be a total no-op, so a World Settings write that merely
    /// echoes the previewed atmosphere defaults back cannot conjure a `Sky` actor
    /// while somebody is editing gravity.
    #[test]
    fn sky_atmosphere_without_create_never_conjures_an_authority() {
        use inf_ecs::components::SkyAtmosphere;

        let mut doc = SceneDoc::new();
        let entities = doc.order().len();
        let undos = doc.undo_len();
        let version = doc.version();

        assert_eq!(
            doc.edit_sky_atmosphere(SkyAtmosphere::default(), false),
            None
        );
        assert_eq!(doc.order().len(), entities);
        assert_eq!(doc.undo_len(), undos);
        assert_eq!(doc.version(), version, "a no-op must not bump the version");
        assert!(!doc.is_dirty());

        // With `create`, the same call opts in …
        let guid = doc
            .edit_sky_atmosphere(SkyAtmosphere::default(), true)
            .expect("opted in");
        assert_eq!(doc.sky_authority(), Some(guid));

        // … and afterwards `create: false` still writes the existing authority.
        let hazy = SkyAtmosphere {
            turbidity: 4.0,
            ..SkyAtmosphere::default()
        };
        assert_eq!(doc.edit_sky_atmosphere(hazy, false), Some(guid));
        assert_eq!(doc.sky_atmosphere(), Some(hazy));
    }

    /// P17.4: a weather edit is **one** undo step, labelled for what the user did,
    /// and it opts a clockless level in exactly like the other two entry points.
    #[test]
    fn weather_edit_is_one_undo_step_and_creates_the_authority() {
        use inf_ecs::components::{SkyAtmosphere, WeatherPreset};

        let mut doc = SceneDoc::new();
        let entities = doc.order().len();
        let undos = doc.undo_len();

        // Without `create` on a clockless level: a total no-op.
        assert_eq!(doc.edit_weather(SkyAtmosphere::default(), false), None);
        assert_eq!(doc.order().len(), entities);
        assert_eq!(doc.undo_len(), undos);
        assert!(
            !doc.is_dirty(),
            "a refused write must not dirty the document"
        );

        // With `create`: the Sky actor + both components + eleven field writes
        // collapse into ONE entry.
        let storm = SkyAtmosphere {
            weather_enabled: true,
            weather_target: WeatherPreset::Storm,
            ..SkyAtmosphere::default()
        };
        let guid = doc.edit_weather(storm, true).expect("opted in");
        assert_eq!(doc.sky_authority(), Some(guid));
        assert_eq!(doc.undo_len(), undos + 1, "one undo entry, not thirteen");
        assert_eq!(doc.order().len(), entities + 1);
        let a = doc.sky_atmosphere().expect("atmosphere");
        assert!(a.weather_enabled);
        assert_eq!(a.weather_target, WeatherPreset::Storm);

        // A single Ctrl+Z takes the whole opt-in back out …
        assert!(doc.undo());
        assert_eq!(doc.order().len(), entities);
        assert_eq!(doc.sky_authority(), None);
        // … and redo restores it, enum field included (the reflect variant name
        // really applied — a wrong spelling would silently leave `Clear`).
        assert!(doc.redo());
        assert_eq!(
            doc.sky_atmosphere().unwrap().weather_target,
            WeatherPreset::Storm
        );

        // Re-writing the same values records nothing.
        let before = doc.undo_len();
        assert_eq!(doc.edit_weather(storm, false), Some(guid));
        assert_eq!(doc.undo_len(), before, "a no-op must record nothing");
    }

    /// The two blocks share one component, so writing either must leave the other
    /// alone — the composition the Ring-2 command depends on when it posts the
    /// whole settings DTO on every edit.
    #[test]
    fn weather_and_atmosphere_edits_compose_rather_than_overwrite() {
        use crate::ipc::{WeatherDto, WeatherPresetDto};
        use inf_ecs::components::{SkyAtmosphere, WeatherPreset};

        let mut doc = SceneDoc::new();
        doc.edit_sky_atmosphere(
            SkyAtmosphere {
                turbidity: 4.0,
                fog_height: 120.0,
                ..SkyAtmosphere::default()
            },
            true,
        )
        .expect("created");

        // A weather write through the DTO overlay path the command uses.
        let base = doc.sky_atmosphere().unwrap();
        let dto = WeatherDto::from_doc(&doc).snapped_to(WeatherPresetDto::Snow);
        doc.edit_weather(dto.to_component(base), true);

        let a = doc.sky_atmosphere().unwrap();
        assert_eq!(a.turbidity, 4.0, "the atmosphere half must survive");
        assert_eq!(a.fog_height, 120.0);
        assert!(a.weather_enabled);
        assert_eq!(a.weather_target, WeatherPreset::Snow);
        assert_eq!(a.weather_params(), WeatherPreset::Snow.params());
        assert_eq!(a.weather_blend_remaining, 0.0, "a preset button snaps");

        // …and the reverse order: an atmosphere write must not reset the weather.
        let base = doc.sky_atmosphere().unwrap();
        doc.edit_sky_atmosphere(
            SkyAtmosphere {
                turbidity: 1.5,
                ..base
            },
            false,
        );
        let a = doc.sky_atmosphere().unwrap();
        assert_eq!(a.turbidity, 1.5);
        assert_eq!(a.weather_target, WeatherPreset::Snow);
        assert_eq!(
            a.weather_precipitation,
            WeatherPreset::Snow.params().precipitation
        );
    }

    /// Hostile DTO input is clamped, never stored — the same contract
    /// `atmosphere_dto_clamps_hostile_input` holds one block over.
    #[test]
    fn weather_dto_clamps_hostile_input() {
        use crate::ipc::{WeatherDto, WeatherPresetDto};
        use inf_ecs::components::SkyAtmosphere;

        let hostile = WeatherDto {
            present: true,
            enabled: true,
            preset: WeatherPresetDto::Fog,
            blend_seconds: f32::NAN,
            blend_remaining: -5.0,
            coverage: 9.0,
            cloud_type: -3.0,
            wind_x: f32::INFINITY,
            wind_z: -1e9,
            fog_density: -1.0,
            precipitation: 7.5,
            snowiness: f32::NAN,
        };
        let a = hostile.to_component(SkyAtmosphere::default());
        assert_eq!(
            a.weather_blend_seconds, 8.0,
            "NaN falls back to the default"
        );
        assert_eq!(a.weather_blend_remaining, 0.0, "negative clamps to settled");
        // The blend bound is the SHARED Ring-0 constant, not a repeated literal:
        // `sky::set_weather` is the other door into these two fields and clamps to
        // the same value, and the bound is arithmetic (past it the f32 countdown
        // stops making progress and the blend never settles), so two copies of the
        // number would be two chances to arm the blender forever.
        let huge = WeatherDto {
            blend_seconds: 5e6,
            blend_remaining: 5e6,
            ..hostile
        };
        let b = huge.to_component(SkyAtmosphere::default());
        assert_eq!(b.weather_blend_seconds, inf_ecs::sky::MAX_WEATHER_BLEND_S);
        assert_eq!(b.weather_blend_remaining, inf_ecs::sky::MAX_WEATHER_BLEND_S);
        assert_eq!(a.weather_coverage, 1.0);
        assert_eq!(a.weather_cloud_type, 0.0);
        assert_eq!(a.weather_wind_x, 0.0, "infinite wind falls back to still");
        assert_eq!(a.weather_wind_z, -200.0);
        assert_eq!(a.weather_fog_density, 0.0);
        assert_eq!(a.weather_precipitation, 1.0);
        assert_eq!(a.weather_snowiness, 0.0);
        // The atmosphere half of the base is untouched.
        assert_eq!(a.cloud_coverage, SkyAtmosphere::default().cloud_coverage);
    }

    /// The DTO deliberately carries no colours, so a World Settings write must
    /// leave a Details-authored sun/moon/gradient colour alone. This is the
    /// overlay contract that `SkyAtmosphereDto::to_component` implements and the
    /// Ring-2 command relies on.
    #[test]
    fn atmosphere_dto_overlay_preserves_authored_colours() {
        use crate::ipc::SkyAtmosphereDto;
        use inf_ecs::components::SkyAtmosphere;
        use inf_ecs::math::Color;

        let mut doc = SceneDoc::new();
        let authored_sun = Color::new(1.0, 0.4, 0.1, 1.0);
        doc.edit_sky_atmosphere(
            SkyAtmosphere {
                sun_color: authored_sun,
                ..SkyAtmosphere::default()
            },
            true,
        )
        .expect("created");
        // …except the DTO cannot carry that colour, so write it through the
        // component path the Details grid uses.
        let guid = doc.sky_authority().unwrap();
        let tp = doc
            .world()
            .registry()
            .type_path_for("SkyAtmosphere")
            .unwrap();
        doc.edit_set_prop(
            guid,
            tp,
            "sun_color",
            &inf_ecs::props::PropValue::Color([1.0, 0.4, 0.1, 1.0]),
        );
        assert_eq!(doc.sky_atmosphere().unwrap().sun_color, authored_sun);

        // Now do what the panel does: project, change one number, write back.
        let mut dto = SkyAtmosphereDto::from_doc(&doc);
        assert!(dto.present);
        dto.fog_density = 5.0e-4;
        let base = doc.sky_atmosphere().unwrap();
        doc.edit_sky_atmosphere(dto.to_component(base), dto.present);

        let after = doc.sky_atmosphere().unwrap();
        assert_eq!(after.fog_density, 5.0e-4);
        assert_eq!(
            after.sun_color, authored_sun,
            "a fog edit erased the authored sun colour"
        );
    }

    /// Nonsense from a hand-crafted IPC payload is clamped, never stored — a
    /// `NaN` turbidity would blank the sky and a negative fog density would make
    /// the exponential integral blow up.
    #[test]
    fn atmosphere_dto_clamps_hostile_input() {
        use crate::ipc::SkyAtmosphereDto;
        use inf_ecs::components::SkyAtmosphere;

        let hostile = SkyAtmosphereDto {
            present: true,
            enabled: true,
            physical: true,
            sky_intensity: f32::NAN,
            turbidity: -5.0,
            mie_anisotropy: 12.0,
            sun_disc_deg: -1.0,
            moon_disc_deg: 1e9,
            star_intensity: f32::INFINITY,
            tint_strength: 4.0,
            aerial_perspective: -2.0,
            fog_density: -1.0,
            fog_falloff: f32::NAN,
            fog_height: 0.0,
            clouds_enabled: true,
            cloud_coverage: 9.0,
            cloud_type: -3.0,
            cloud_bottom: f32::NAN,
            cloud_top: 1e12,
            cloud_density: -1.0,
            cloud_detail: f32::NAN,
            cloud_seed: u32::MAX,
            cloud_wind_x: 1e9,
            cloud_wind_z: f32::NAN,
            cloud_phase_g: 5.0,
            cloud_shadow: -1.0,
            cloud_ambient: 1e9,
        };
        let a = hostile.to_component(SkyAtmosphere::default());
        assert_eq!(a.sky_intensity, 1.0, "NaN fell back to the default");
        assert_eq!(a.turbidity, 0.0);
        assert_eq!(a.mie_anisotropy, 0.95);
        assert_eq!(a.sun_disc_deg, 0.0);
        assert_eq!(a.moon_disc_deg, 90.0);
        assert_eq!(a.star_intensity, 1.0);
        assert_eq!(a.tint_strength, 1.0);
        assert_eq!(a.aerial_perspective, 0.0);
        assert_eq!(a.fog_density, 0.0);
        assert_eq!(a.fog_falloff, 0.002);
        // ── clouds (P17.3) ──
        assert_eq!(a.cloud_coverage, 1.0);
        assert_eq!(a.cloud_type, 0.0);
        assert_eq!(a.cloud_bottom, 1500.0, "NaN fell back to the default");
        assert_eq!(a.cloud_top, 50_000.0);
        assert_eq!(a.cloud_density, 0.0);
        assert_eq!(a.cloud_detail, 0.6, "NaN fell back to the default");
        // The seed is MASKED, not clamped: only the low 24 bits survive the f32
        // uniform, so anything else would display one sky and render another.
        assert_eq!(a.cloud_seed, 0x00ff_ffff);
        assert_eq!(a.cloud_wind_x, 200.0);
        assert_eq!(a.cloud_wind_z, 2.0, "NaN fell back to the default");
        assert_eq!(a.cloud_phase_g, 0.95);
        assert_eq!(a.cloud_shadow, 0.0);
        assert_eq!(a.cloud_ambient, 4.0);
        // Everything the DTO does not carry is untouched — including the cloud
        // block's own `Color`, which stays in the Details grid with the other five.
        assert_eq!(a.zenith, SkyAtmosphere::default().zenith);
        assert_eq!(a.cloud_color, SkyAtmosphere::default().cloud_color);
    }

    /// The Phase-17 "done when": a **new** level's sky is the physical one. The
    /// default scene carries the authority pair, and — the part that is easy to
    /// get wrong — it does NOT also carry a hand-authored directional sun, which
    /// would light the level twice from two directions.
    #[test]
    fn the_default_scene_has_a_physical_sky_and_only_one_sun() {
        use inf_ecs::components::{Light, LightKind};

        let doc = SceneDoc::with_demo();
        let sky = doc.sky_atmosphere().expect("the default level has a sky");
        assert!(sky.physical, "the default sky is the physical one");
        assert!(sky.enabled, "the default sun lights the scene");
        assert_eq!(
            doc.time_of_day(),
            Some(inf_ecs::components::TimeOfDay::default()),
            "the default clock is the documented 10:00 default"
        );
        assert_eq!(
            doc.time_of_day().unwrap().rate,
            0.0,
            "an idle editor must not move the sun"
        );

        // No authored directional light: the time of day IS the sun.
        let world = doc.world().world();
        let directional = world
            .iter_entities()
            .filter_map(|e| e.get::<Light>())
            .filter(|l| l.kind == LightKind::Directional)
            .count();
        assert_eq!(
            directional, 0,
            "the default scene has both a sky sun and an authored one — it would be lit twice"
        );

        // The authority is the actor World Settings would have created.
        let guid = doc.sky_authority().expect("authority");
        assert_eq!(doc.display_name(guid), "Sky");
    }

    /// P17.3: a **new** level boots with clouds, while the component default —
    /// which is also what every v12 level lifts to — stays off.
    ///
    /// The two must disagree in exactly this direction and nowhere else. If the
    /// component default ever flipped true, every existing level would silently
    /// grow a sky it was never authored against on its next load, which is the
    /// one thing the frozen-record scheme cannot undo. If the *demo* default
    /// flipped false, the `editor_default` golden's one P17.3 re-bless would have
    /// been for nothing.
    #[test]
    fn the_default_scene_opts_into_clouds_while_the_component_does_not() {
        use inf_ecs::components::SkyAtmosphere;

        let doc = SceneDoc::with_demo();
        let sky = doc.sky_atmosphere().expect("the default level has a sky");
        assert!(
            sky.clouds_enabled,
            "a new level should boot with clouds — that is the P17.3 look"
        );
        assert!(
            !SkyAtmosphere::default().clouds_enabled,
            "the COMPONENT default must stay off, or every v12 level grows clouds \
             it was never authored against when it is lifted"
        );
        // ...and nothing else in the block was privately tuned: the demo diverges
        // by exactly one boolean, so what the golden pictures is the documented
        // defaults.
        let defaults = SkyAtmosphere::default();
        assert_eq!(
            sky,
            SkyAtmosphere {
                clouds_enabled: true,
                ..defaults
            },
            "the default scene tuned something other than `clouds_enabled`"
        );
    }
}
