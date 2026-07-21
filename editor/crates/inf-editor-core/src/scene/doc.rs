//! The authoritative scene document (P3.2).
//!
//! [`SceneDoc`] wraps an [`inf_ecs::EcsWorld`] with editor state — selection,
//! a monotonic version, a dirty flag, an explicit creation-order list (so the
//! Outliner and the serialized file are deterministic) — and exposes the
//! primitive mutations (create / rename / delete / reparent / visibility /
//! select). Undo (P3.4) wraps these; the snapshot layer projects them to the
//! frontend DTOs; the viewport reads through `world()` to render + pick.
//!
//! GUIDs, not bevy `Entity` ids, are the identity that crosses every boundary:
//! entity ids are reused across despawn and never serialized.

use inf_ecs::components::{
    AtlasRect, Camera, Light, LightKind, Material, MeshRef, Primitive, Sprite, Transform,
    Visibility,
};
use inf_ecs::{ComputedVisibility, EcsWorld, Entity, PropValue, Vec2d};
use uuid::Uuid;

use crate::ipc::{SceneNode, SceneSnapshot, SpawnKind};
use crate::scene::serialize::EntityRecord;
use crate::scene::undo::{EditCommand, EditHistory};

pub struct SceneDoc {
    world: EcsWorld,
    /// Creation order of every live entity — the single ordering source for the
    /// Outliner tree and the serialized file (bevy iteration order is unstable).
    order: Vec<Uuid>,
    selection: Vec<Uuid>,
    version: u64,
    dirty: bool,
    title: String,
    history: EditHistory,
}

impl Default for SceneDoc {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneDoc {
    pub fn new() -> Self {
        Self {
            world: EcsWorld::new(),
            order: Vec::new(),
            selection: Vec::new(),
            version: 0,
            dirty: false,
            title: "Untitled".to_string(),
            history: EditHistory::default(),
        }
    }

    // ── accessors ────────────────────────────────────────────────────────

    pub fn world(&self) -> &EcsWorld {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut EcsWorld {
        &mut self.world
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = title.into();
    }

    pub fn selection(&self) -> &[Uuid] {
        &self.selection
    }

    /// Live entities in creation order.
    pub fn order(&self) -> &[Uuid] {
        &self.order
    }

    pub fn entity_of(&self, guid: Uuid) -> Option<Entity> {
        self.world.entity_of(guid)
    }

    /// Bump the version; mark unsaved. Every mutation funnels through here.
    pub(crate) fn touch(&mut self) {
        self.version += 1;
        self.dirty = true;
    }

    /// Clear the dirty flag (after a successful save) without bumping version.
    pub fn mark_saved(&mut self) {
        self.dirty = false;
    }

    // ── mutations ────────────────────────────────────────────────────────

    /// Create an entity of `kind` under `parent`, returning its GUID. When
    /// `name` is empty a kind-appropriate default is used.
    pub fn create(&mut self, kind: SpawnKind, name: &str, parent: Option<Uuid>) -> Uuid {
        let guid = Uuid::new_v4();
        self.create_with_guid(guid, kind, name, parent);
        guid
    }

    /// Create with an explicit GUID (load + deterministic tests + undo redo).
    pub fn create_with_guid(
        &mut self,
        guid: Uuid,
        kind: SpawnKind,
        name: &str,
        parent: Option<Uuid>,
    ) -> Entity {
        let label = if name.is_empty() {
            default_name(kind)
        } else {
            name.to_string()
        };
        let parent_entity = parent.and_then(|p| self.world.entity_of(p));
        let entity = self.world.spawn_with_guid(guid, &label, parent_entity);
        attach_kind(&mut self.world, entity, kind);
        self.order.push(guid);
        self.touch();
        entity
    }

    /// Low-level spawn for the loader: explicit identity, no kind components
    /// (the caller inserts concrete components). Does not dirty the document.
    pub(crate) fn spawn_bare(&mut self, guid: Uuid, name: &str, parent: Option<Uuid>) -> Entity {
        let parent_entity = parent.and_then(|p| self.world.entity_of(p));
        let entity = self.world.spawn_with_guid(guid, name, parent_entity);
        self.order.push(guid);
        entity
    }

    /// Empty the document (before a load). Keeps title/version bookkeeping to
    /// the caller.
    pub(crate) fn reset(&mut self) {
        self.world.clear();
        self.order.clear();
        self.selection.clear();
    }

    /// Read one entity's editable component properties (Details, P3.3).
    pub fn entity_props(&self, guid: Uuid) -> Vec<inf_ecs::ComponentProps> {
        match self.world.entity_of(guid) {
            Some(e) => self.world.read_props(e),
            None => Vec::new(),
        }
    }

    /// The Outliner label for `guid`.
    pub fn display_name(&self, guid: Uuid) -> String {
        self.world
            .entity_of(guid)
            .and_then(|e| self.world.name_of(e))
            .unwrap_or("")
            .to_string()
    }

    /// The UE-style type label for `guid`.
    pub fn kind_of_guid(&self, guid: Uuid) -> String {
        self.world
            .entity_of(guid)
            .map(|e| kind_of(&self.world, e))
            .unwrap_or_default()
    }

    /// The Details view of the current selection (P3.3).
    pub fn details(&self) -> crate::ipc::DetailsDto {
        crate::scene::details::build(self)
    }

    /// Reset one component field to its `Default` value (P3.3.4), recorded for
    /// undo. Returns whether it changed.
    pub fn edit_reset_prop(&mut self, guid: Uuid, type_path: &str, field: &str) -> bool {
        match inf_ecs::default_field(self.world.registry(), type_path, field) {
            Some(default) => self.edit_set_prop(guid, type_path, field, &default),
            None => false,
        }
    }

    /// Write one component field on `guid`. Returns whether it applied.
    pub fn write_prop(
        &mut self,
        guid: Uuid,
        type_path: &str,
        field: &str,
        value: &inf_ecs::PropValue,
    ) -> bool {
        let Some(e) = self.world.entity_of(guid) else {
            return false;
        };
        let ok = self.world.write_prop(e, type_path, field, value);
        if ok {
            self.touch();
        }
        ok
    }

    pub fn rename(&mut self, guid: Uuid, name: &str) {
        if let Some(e) = self.world.entity_of(guid) {
            self.world.rename(e, name);
            self.touch();
        }
    }

    pub fn set_visible(&mut self, guid: Uuid, visible: bool) {
        if let Some(e) = self.world.entity_of(guid) {
            self.world.set_visible(e, visible);
            self.touch();
        }
    }

    /// Re-parent `guid` under `parent` (`None` → root). Returns false if the
    /// move would create a cycle or the guid is unknown.
    pub fn reparent(&mut self, guid: Uuid, parent: Option<Uuid>) -> bool {
        let Some(child) = self.world.entity_of(guid) else {
            return false;
        };
        let parent_entity = match parent {
            Some(p) => match self.world.entity_of(p) {
                Some(e) => Some(e),
                None => return false,
            },
            None => None,
        };
        let ok = self.world.reparent(child, parent_entity);
        if ok {
            self.touch();
        }
        ok
    }

    /// Delete each guid (and its descendants). Prunes selection + order.
    pub fn delete(&mut self, guids: &[Uuid]) {
        for &guid in guids {
            if let Some(e) = self.world.entity_of(guid) {
                let subtree: Vec<Uuid> = self
                    .world
                    .subtree(e)
                    .into_iter()
                    .filter_map(|se| self.world.guid_of(se))
                    .collect();
                self.world.despawn(e);
                self.order.retain(|g| !subtree.contains(g));
                self.selection.retain(|g| !subtree.contains(g));
            }
        }
        self.touch();
    }

    // ── selection ────────────────────────────────────────────────────────

    /// Set (or extend, when `additive`) the selection. Unknown guids are
    /// dropped. Selection changes bump the version (so the UI re-syncs) but do
    /// **not** dirty the document.
    pub fn select(&mut self, guids: &[Uuid], additive: bool) {
        let valid: Vec<Uuid> = guids
            .iter()
            .copied()
            .filter(|g| self.world.entity_of(*g).is_some())
            .collect();
        if additive {
            for g in valid {
                if let Some(pos) = self.selection.iter().position(|s| *s == g) {
                    self.selection.remove(pos); // toggle off
                } else {
                    self.selection.push(g);
                }
            }
        } else {
            self.selection = valid;
        }
        self.version += 1; // resync UI, but selection is not an unsaved edit
    }

    pub fn clear_selection(&mut self) {
        if !self.selection.is_empty() {
            self.selection.clear();
            self.version += 1;
        }
    }

    // ── raw mutations for undo (non-recording) ───────────────────────────
    //
    // These are the same primitives the public methods use, exposed so
    // `EditCommand::apply`/`revert` can drive the world without re-entering the
    // recorder. They still `touch()` (a revert is a real change).

    pub(crate) fn raw_rename(&mut self, guid: Uuid, name: &str) {
        self.rename(guid, name);
    }

    pub(crate) fn raw_reparent(&mut self, guid: Uuid, parent: Option<Uuid>) -> bool {
        self.reparent(guid, parent)
    }

    pub(crate) fn raw_set_visible(&mut self, guid: Uuid, visible: bool) {
        self.set_visible(guid, visible);
    }

    pub(crate) fn raw_delete(&mut self, guids: &[Uuid]) {
        self.delete(guids);
    }

    pub(crate) fn raw_write_prop(
        &mut self,
        guid: Uuid,
        type_path: &str,
        field: &str,
        value: &PropValue,
    ) -> bool {
        self.write_prop(guid, type_path, field, value)
    }

    pub(crate) fn raw_set_transform(&mut self, guid: Uuid, t: Transform) {
        if let Some(e) = self.world.entity_of(guid) {
            self.world.world_mut().entity_mut(e).insert(t);
            self.world.mark_dirty();
            self.touch();
        }
    }

    /// Read an entity's [`Sprite`] component (the fields — `texture`,
    /// `atlas_rect` — that the reflection Details grid can't reach).
    pub(crate) fn raw_get_sprite(&self, guid: Uuid) -> Option<Sprite> {
        let e = self.world.entity_of(guid)?;
        self.world.world().get::<Sprite>(e).cloned()
    }

    /// Insert (`Some`) or remove (`None`) an entity's [`Sprite`] component.
    pub(crate) fn raw_set_sprite(&mut self, guid: Uuid, sprite: Option<Sprite>) {
        if let Some(e) = self.world.entity_of(guid) {
            match sprite {
                Some(s) => {
                    self.world.world_mut().entity_mut(e).insert(s);
                }
                None => {
                    self.world.world_mut().entity_mut(e).remove::<Sprite>();
                }
            }
            self.world.mark_dirty();
            self.touch();
        }
    }

    /// Recreate an entity from a serialized record at order slot `at`.
    pub(crate) fn raw_spawn_record(&mut self, rec: &EntityRecord, at: usize) {
        let e = self.spawn_bare(rec.guid, &rec.name, rec.parent);
        // `spawn_bare` appended the guid; move it to its original slot.
        if let Some(pos) = self.order.iter().position(|g| *g == rec.guid) {
            let g = self.order.remove(pos);
            let at = at.min(self.order.len());
            self.order.insert(at, g);
        }
        let w = self.world.world_mut();
        w.entity_mut(e).insert((
            rec.transform,
            Visibility {
                visible: rec.visible,
            },
        ));
        if let Some(m) = rec.mesh {
            w.entity_mut(e).insert(m);
        }
        if let Some(m) = rec.material {
            w.entity_mut(e).insert(m);
        }
        if let Some(l) = rec.light {
            w.entity_mut(e).insert(l);
        }
        if let Some(c) = rec.camera {
            w.entity_mut(e).insert(c);
        }
        self.touch();
    }

    fn prop_value(&self, guid: Uuid, type_path: &str, field: &str) -> Option<PropValue> {
        let comps = self.entity_props(guid);
        let c = comps.iter().find(|c| c.type_path == type_path)?;
        c.fields
            .iter()
            .find(|f| f.name == field)
            .map(|f| f.value.clone())
    }

    // ── recorded mutations (Ring 2 / gizmo call these) ───────────────────

    /// Create + record for undo. Returns the new GUID.
    pub fn edit_create(&mut self, kind: SpawnKind, name: &str, parent: Option<Uuid>) -> Uuid {
        let guid = self.create(kind, name, parent);
        let at = self.order.iter().position(|g| *g == guid).unwrap_or(0);
        if let Some(record) = crate::scene::serialize::record_of(self, guid) {
            self.history
                .record("Create", EditCommand::Create { at, record });
        }
        guid
    }

    /// Delete + record for undo (the whole subtree round-trips on undo).
    pub fn edit_delete(&mut self, guids: &[Uuid]) {
        use std::collections::HashSet;
        let mut set: HashSet<Uuid> = HashSet::new();
        for &g in guids {
            if let Some(e) = self.world.entity_of(g) {
                for se in self.world.subtree(e) {
                    if let Some(sg) = self.world.guid_of(se) {
                        set.insert(sg);
                    }
                }
            }
        }
        let items: Vec<(usize, EntityRecord)> = self
            .order
            .iter()
            .enumerate()
            .filter(|(_, g)| set.contains(g))
            .filter_map(|(i, g)| crate::scene::serialize::record_of(self, *g).map(|r| (i, r)))
            .collect();
        let tops: Vec<Uuid> = guids
            .iter()
            .copied()
            .filter(|g| self.world.entity_of(*g).is_some())
            .collect();
        if items.is_empty() {
            return;
        }
        self.delete(&tops);
        self.history
            .record("Delete", EditCommand::Delete { items, tops });
    }

    pub fn edit_rename(&mut self, guid: Uuid, name: &str) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        let before = self.world.name_of(e).unwrap_or("").to_string();
        if before == name {
            return;
        }
        self.rename(guid, name);
        self.history.record(
            "Rename",
            EditCommand::Rename {
                guid,
                before,
                after: name.to_string(),
            },
        );
    }

    pub fn edit_reparent(&mut self, guid: Uuid, parent: Option<Uuid>) -> bool {
        let before = self
            .world
            .entity_of(guid)
            .and_then(|e| self.world.parent_of(e))
            .and_then(|p| self.world.guid_of(p));
        if before == parent {
            return true;
        }
        let ok = self.reparent(guid, parent);
        if ok {
            self.history.record(
                "Reparent",
                EditCommand::Reparent {
                    guid,
                    before,
                    after: parent,
                },
            );
        }
        ok
    }

    pub fn edit_set_visible(&mut self, guid: Uuid, visible: bool) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        let before = self
            .world
            .world()
            .get::<Visibility>(e)
            .map(|v| v.visible)
            .unwrap_or(true);
        if before == visible {
            return;
        }
        self.set_visible(guid, visible);
        self.history.record(
            "Set Visibility",
            EditCommand::SetVisible {
                guid,
                before,
                after: visible,
            },
        );
    }

    pub fn edit_set_transform(&mut self, guid: Uuid, t: Transform) {
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        let before = self
            .world
            .world()
            .get::<Transform>(e)
            .copied()
            .unwrap_or(Transform::IDENTITY);
        if before == t {
            return;
        }
        self.raw_set_transform(guid, t);
        self.history.record(
            "Move",
            EditCommand::SetTransform {
                guid,
                before,
                after: t,
            },
        );
    }

    pub fn edit_set_prop(
        &mut self,
        guid: Uuid,
        type_path: &str,
        field: &str,
        value: &PropValue,
    ) -> bool {
        let Some(before) = self.prop_value(guid, type_path, field) else {
            return false;
        };
        if before == *value {
            return true;
        }
        let ok = self.write_prop(guid, type_path, field, value);
        if ok {
            let label = format!("Edit {field}");
            self.history.record(
                &label,
                EditCommand::SetProp {
                    guid,
                    type_path: type_path.to_string(),
                    field: field.to_string(),
                    before,
                    after: value.clone(),
                },
            );
        }
        ok
    }

    /// Apply a material's PBR parameters to each target entity's `Material`
    /// component as one undo step (Content-Drawer apply-by-drag / "Apply to
    /// Selection", P7.1). Targets without a `Material` are skipped. Returns how
    /// many entities were updated.
    pub fn edit_apply_material(
        &mut self,
        targets: &[Uuid],
        base_color: [f32; 4],
        metallic: f32,
        roughness: f32,
        emissive: [f32; 3],
    ) -> usize {
        let Some(tp) = self.world.registry().type_path_for("Material") else {
            return 0;
        };
        self.begin_transaction("Apply Material");
        let mut applied = 0;
        for &g in targets {
            // Only entities that already carry a Material component.
            if self.prop_value(g, tp, "base_color").is_none() {
                continue;
            }
            self.edit_set_prop(g, tp, "base_color", &PropValue::Color(base_color));
            self.edit_set_prop(g, tp, "metallic", &PropValue::Number(metallic as f64));
            self.edit_set_prop(g, tp, "roughness", &PropValue::Number(roughness as f64));
            self.edit_set_prop(
                g,
                tp,
                "emissive",
                &PropValue::Color([emissive[0], emissive[1], emissive[2], 1.0]),
            );
            applied += 1;
        }
        self.commit_transaction();
        applied
    }

    /// Apply a sprite-sheet slice to each target's [`Sprite`] component as one
    /// undo step (P8.2a "Apply to Selection"). A target without a `Sprite` gets
    /// one inserted (defaults + the slice); an existing one keeps its other
    /// fields (pivot, color, sorting layer, flips). `size`, when `Some`, sets the
    /// quad extent from the slice's pixel aspect. Returns how many were updated.
    pub fn edit_apply_sprite_slice(
        &mut self,
        targets: &[Uuid],
        texture: Option<uuid::Uuid>,
        uv_min: [f64; 2],
        uv_max: [f64; 2],
        size: Option<[f64; 2]>,
    ) -> usize {
        let atlas_rect = AtlasRect {
            min: Vec2d::new(uv_min[0], uv_min[1]),
            max: Vec2d::new(uv_max[0], uv_max[1]),
        };
        self.begin_transaction("Apply Sprite Slice");
        let mut applied = 0;
        for &g in targets {
            if self.world.entity_of(g).is_none() {
                continue;
            }
            let before = self.raw_get_sprite(g);
            let mut sprite = before.clone().unwrap_or_default();
            sprite.texture = texture;
            sprite.atlas_rect = atlas_rect;
            if let Some(sz) = size {
                sprite.size = Vec2d::new(sz[0], sz[1]);
            }
            let after = Some(sprite);
            if before == after {
                continue;
            }
            self.raw_set_sprite(g, after.clone());
            self.history.record(
                "Apply Sprite Slice",
                EditCommand::SetSprite {
                    guid: g,
                    before,
                    after,
                },
            );
            applied += 1;
        }
        self.commit_transaction();
        applied
    }

    // ── history control ──────────────────────────────────────────────────

    /// Open an undo transaction; every recorded edit until [`Self::commit_transaction`]
    /// collapses into one entry (a gizmo drag is one undo step, P3.4.2).
    pub fn begin_transaction(&mut self, label: &str) {
        self.history.begin(label);
    }

    pub fn commit_transaction(&mut self) {
        self.history.commit();
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Undo the most recent transaction. Returns whether anything was undone.
    pub fn undo(&mut self) -> bool {
        self.history.commit();
        if let Some(txn) = self.history.take_undo() {
            for cmd in txn.commands.iter().rev() {
                cmd.revert(self);
            }
            self.history.push_redo(txn);
            self.version += 1;
            true
        } else {
            false
        }
    }

    /// Redo the most recently undone transaction.
    pub fn redo(&mut self) -> bool {
        if let Some(txn) = self.history.take_redo() {
            for cmd in txn.commands.iter() {
                cmd.apply(self);
            }
            self.history.push_undo(txn);
            self.version += 1;
            true
        } else {
            false
        }
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    // ── snapshot ─────────────────────────────────────────────────────────

    /// Project the world to a full [`SceneSnapshot`] (propagates first so
    /// effective visibility + world transforms are current).
    pub fn snapshot(&mut self) -> SceneSnapshot {
        self.world.propagate();

        let nodes: Vec<SceneNode> = self
            .order
            .iter()
            .filter_map(|&guid| self.node_of(guid))
            .collect();
        let roots: Vec<String> = self
            .order
            .iter()
            .filter(|&&g| {
                self.world
                    .entity_of(g)
                    .map(|e| self.world.parent_of(e).is_none())
                    .unwrap_or(false)
            })
            .map(|g| g.to_string())
            .collect();

        SceneSnapshot {
            version: self.version,
            roots,
            nodes,
            selection: self.selection.iter().map(|g| g.to_string()).collect(),
            dirty: self.dirty,
            title: self.title.clone(),
            can_undo: self.history.can_undo(),
            can_redo: self.history.can_redo(),
            undo_label: self.history.undo_label().map(str::to_string),
            redo_label: self.history.redo_label().map(str::to_string),
        }
    }

    fn node_of(&self, guid: Uuid) -> Option<SceneNode> {
        let e = self.world.entity_of(guid)?;
        let name = self.world.name_of(e).unwrap_or("").to_string();
        let visible = self
            .world
            .world()
            .get::<Visibility>(e)
            .map(|v| v.visible)
            .unwrap_or(true);
        let effective_visible = self
            .world
            .world()
            .get::<ComputedVisibility>(e)
            .map(|c| c.0)
            .unwrap_or(true);
        let parent = self
            .world
            .parent_of(e)
            .and_then(|p| self.world.guid_of(p))
            .map(|g| g.to_string());
        // Children in creation order (scan the order list).
        let children: Vec<String> = self
            .order
            .iter()
            .filter(|&&g| {
                self.world
                    .entity_of(g)
                    .and_then(|ce| self.world.parent_of(ce))
                    .and_then(|pe| self.world.guid_of(pe))
                    == Some(guid)
            })
            .map(|g| g.to_string())
            .collect();

        Some(SceneNode {
            guid: guid.to_string(),
            name,
            kind: kind_of(&self.world, e),
            visible,
            effective_visible,
            parent,
            children,
        })
    }
}

/// UE-style type-column label from the components an entity carries.
fn kind_of(world: &EcsWorld, e: Entity) -> String {
    let w = world.world();
    if let Some(light) = w.get::<Light>(e) {
        return match light.kind {
            LightKind::Directional => "Directional Light",
            LightKind::Point => "Point Light",
            LightKind::Spot => "Spot Light",
        }
        .to_string();
    }
    if w.get::<Camera>(e).is_some() {
        return "Camera".to_string();
    }
    if w.get::<MeshRef>(e).is_some() {
        return "Static Mesh".to_string();
    }
    if w.get::<Sprite>(e).is_some() {
        return "Sprite".to_string();
    }
    // No renderable payload: a folder if it has children, else a plain actor.
    if !world.children_of(e).is_empty() {
        "Folder".to_string()
    } else {
        "Actor".to_string()
    }
}

fn default_name(kind: SpawnKind) -> String {
    match kind {
        SpawnKind::Empty => "Empty",
        SpawnKind::Cube => "Cube",
        SpawnKind::Sphere => "Sphere",
        SpawnKind::Plane => "Plane",
        SpawnKind::Cylinder => "Cylinder",
        SpawnKind::Cone => "Cone",
        SpawnKind::DirectionalLight => "DirectionalLight",
        SpawnKind::PointLight => "PointLight",
        SpawnKind::SpotLight => "SpotLight",
        SpawnKind::Camera => "Camera",
    }
    .to_string()
}

/// Insert the components that make an entity the requested kind.
fn attach_kind(world: &mut EcsWorld, entity: Entity, kind: SpawnKind) {
    let w = world.world_mut();
    let primitive = match kind {
        SpawnKind::Cube => Some(Primitive::Cube),
        SpawnKind::Sphere => Some(Primitive::Sphere),
        SpawnKind::Plane => Some(Primitive::Plane),
        SpawnKind::Cylinder => Some(Primitive::Cylinder),
        SpawnKind::Cone => Some(Primitive::Cone),
        _ => None,
    };
    if let Some(primitive) = primitive {
        w.entity_mut(entity)
            .insert((MeshRef { primitive }, Material::default()));
        return;
    }
    match kind {
        SpawnKind::DirectionalLight => {
            w.entity_mut(entity).insert(Light {
                kind: LightKind::Directional,
                ..Light::default()
            });
        }
        SpawnKind::PointLight => {
            w.entity_mut(entity).insert(Light {
                kind: LightKind::Point,
                ..Light::default()
            });
        }
        SpawnKind::SpotLight => {
            w.entity_mut(entity).insert(Light {
                kind: LightKind::Spot,
                ..Light::default()
            });
        }
        SpawnKind::Camera => {
            w.entity_mut(entity).insert(Camera::default());
        }
        SpawnKind::Empty
        | SpawnKind::Cube
        | SpawnKind::Sphere
        | SpawnKind::Plane
        | SpawnKind::Cylinder
        | SpawnKind::Cone => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_rename_reparent_delete() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let b = doc.create(SpawnKind::Cube, "", Some(a));
        // default name for a cube
        let snap = doc.snapshot();
        let bn = snap.nodes.iter().find(|n| n.guid == b.to_string()).unwrap();
        assert_eq!(bn.name, "Cube");
        assert_eq!(bn.kind, "Static Mesh");
        assert_eq!(bn.parent.as_deref(), Some(a.to_string().as_str()));

        doc.rename(b, "Crate");
        // Reparent b to root.
        assert!(doc.reparent(b, None));
        let snap = doc.snapshot();
        assert_eq!(snap.roots.len(), 2);

        doc.delete(&[a]);
        let snap = doc.snapshot();
        // a gone, b (now a root) survives.
        assert!(snap.nodes.iter().all(|n| n.guid != a.to_string()));
        assert_eq!(snap.nodes.len(), 1);
    }

    #[test]
    fn apply_material_sets_pbr_and_undoes_as_one_step() {
        let mut doc = SceneDoc::new();
        let cube = doc.create(SpawnKind::Cube, "Cube", None);
        let tp = doc.world().registry().type_path_for("Material").unwrap();

        let applied =
            doc.edit_apply_material(&[cube], [1.0, 0.0, 0.0, 1.0], 1.0, 0.2, [0.5, 0.0, 0.0]);
        assert_eq!(applied, 1);
        assert_eq!(
            doc.prop_value(cube, tp, "metallic"),
            Some(PropValue::Number(1.0))
        );
        assert_eq!(
            doc.prop_value(cube, tp, "base_color"),
            Some(PropValue::Color([1.0, 0.0, 0.0, 1.0]))
        );

        // The four field writes collapse into one undo step (back to defaults).
        assert!(doc.undo());
        assert_eq!(
            doc.prop_value(cube, tp, "metallic"),
            Some(PropValue::Number(0.0))
        );
    }

    #[test]
    fn apply_sprite_slice_inserts_and_undoes_as_one_step() {
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Empty, "Sprite", None);
        assert!(doc.raw_get_sprite(e).is_none(), "no sprite to start");

        let tex = uuid::Uuid::from_u128(0xABCD);
        let applied =
            doc.edit_apply_sprite_slice(&[e], Some(tex), [0.25, 0.5], [0.5, 1.0], Some([2.0, 1.0]));
        assert_eq!(applied, 1);
        let s = doc.raw_get_sprite(e).expect("sprite inserted");
        assert_eq!(s.texture, Some(tex));
        assert_eq!(
            s.atlas_rect,
            AtlasRect {
                min: Vec2d::new(0.25, 0.5),
                max: Vec2d::new(0.5, 1.0),
            }
        );
        assert_eq!(s.size, Vec2d::new(2.0, 1.0));
        assert_eq!(doc.kind_of_guid(e), "Sprite");

        // One undo removes the whole Sprite (it didn't exist before).
        assert!(doc.undo());
        assert!(doc.raw_get_sprite(e).is_none(), "undo removes the sprite");
        assert!(doc.redo());
        assert_eq!(doc.raw_get_sprite(e).unwrap().texture, Some(tex));
    }

    #[test]
    fn deleting_parent_removes_children() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let _b = doc.create(SpawnKind::Cube, "B", Some(a));
        doc.delete(&[a]);
        assert!(doc.snapshot().nodes.is_empty());
    }

    #[test]
    fn selection_additive_toggles() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let b = doc.create(SpawnKind::Empty, "B", None);
        doc.select(&[a], false);
        doc.select(&[b], true);
        assert_eq!(doc.selection(), &[a, b]);
        doc.select(&[a], true); // toggle a off
        assert_eq!(doc.selection(), &[b]);
    }

    #[test]
    fn effective_visibility_follows_ancestors() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Empty, "A", None);
        let b = doc.create(SpawnKind::Cube, "B", Some(a));
        doc.set_visible(a, false);
        let snap = doc.snapshot();
        let bn = snap.nodes.iter().find(|n| n.guid == b.to_string()).unwrap();
        assert!(bn.visible, "b's own toggle is still on");
        assert!(!bn.effective_visible, "b hidden because A is hidden");
    }
}
