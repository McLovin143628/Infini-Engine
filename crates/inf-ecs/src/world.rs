//! The editor world facade (P3.1.4).
//!
//! [`EcsWorld`] wraps a bevy `World` with the engine's component registry, a
//! stable `Guid → Entity` index, and a dirty flag that gates transform
//! propagation. Spawn/despawn/reparent/rename are the primitive editor
//! mutations — undo (P3.4) records inverses around exactly these, and the
//! snapshot layer (P3.2) reads through them.

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use glam::DVec3;
use uuid::Uuid;

use crate::components::{ComputedVisibility, GlobalTransform, Guid, Name, Transform, Visibility};
use crate::hierarchy::{children_of, parent_of, set_parent};
use crate::registry::ComponentRegistry;
use crate::transform;

pub struct EcsWorld {
    world: World,
    registry: ComponentRegistry,
    guids: HashMap<Uuid, Entity>,
    dirty: bool,
}

impl Default for EcsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl EcsWorld {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            registry: ComponentRegistry::new(),
            guids: HashMap::new(),
            dirty: true,
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    /// Raw mutable access to the underlying bevy `World`.
    ///
    /// INVARIANT: this does **not** set the dirty flag. Any mutation made through
    /// the returned `&mut World` that changes a transform, visibility, or the
    /// hierarchy leaves [`propagate`](Self::propagate) believing nothing changed —
    /// stale `GlobalTransform`/`ComputedVisibility` until something else dirties the
    /// world. Callers that mutate such state MUST call [`mark_dirty`](Self::mark_dirty)
    /// afterwards. (The typed mutators — `spawn`/`despawn`/`reparent`/`set_visible`/
    /// `write_prop` — set it for you; this escape hatch cannot.)
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn registry(&self) -> &ComponentRegistry {
        &self.registry
    }

    /// **Does any live entity carry `T`?** — answered from the archetype table,
    /// without visiting a single entity (Hardening Wave G).
    ///
    /// The door for the "this whole pass has nothing to do" early-out. Several
    /// per-fixed-step passes exist to find a component that most levels do not
    /// have at all — the 2D physics bridge on a pure-3D level, the `PcgVolume`
    /// solid gather on a hand-authored one, the `Terrain` tile gather on an
    /// interior — and each of them proved it by walking every entity in the
    /// world with a component lookup apiece, sixty times a second, to find
    /// nothing. This is O(archetypes), and an archetype count is a function of
    /// how many distinct component *combinations* a level uses (tens), not of
    /// how many entities it has (tens of thousands).
    ///
    /// Empty archetypes are skipped: bevy keeps an archetype alive after its
    /// last entity leaves, so "an archetype mentions `T`" is not the same claim
    /// as "something has a `T`", and a pass that early-outed on the first is
    /// exactly as wrong as one that never early-outed at all — in the other
    /// direction, and silently.
    ///
    /// A `false` here is a guarantee, not a hint: a caller may skip work
    /// entirely. A `true` says only that the walk is worth doing.
    pub fn has_component<T: bevy_ecs::component::Component>(&self) -> bool {
        let Some(id) = self.world.component_id::<T>() else {
            // The type has never been registered in this world, so nothing can
            // be carrying it.
            return false;
        };
        self.world
            .archetypes()
            .iter()
            .any(|a| !a.is_empty() && a.contains(id))
    }

    // ── reflection-driven properties (Details, P3.3) ─────────────────────

    /// Editable component properties for `entity` (registry order).
    pub fn read_props(&self, entity: Entity) -> Vec<crate::props::ComponentProps> {
        crate::props::read_entity(&self.world, &self.registry, entity)
    }

    /// Write one component field on `entity`. Returns whether it applied.
    pub fn write_prop(
        &mut self,
        entity: Entity,
        type_path: &str,
        field: &str,
        value: &crate::props::PropValue,
    ) -> bool {
        // Disjoint borrow of the two fields (world + registry).
        let ok = crate::props::write_field(
            &mut self.world,
            &self.registry,
            entity,
            type_path,
            field,
            value,
        );
        if ok {
            self.dirty = true;
        }
        ok
    }

    /// Insert a `Default` instance of `type_path`'s component onto `entity`
    /// (E-P1 add-component). Returns whether it applied. Sets the dirty flag.
    pub fn insert_default_component(&mut self, entity: Entity, type_path: &str) -> bool {
        let ok = crate::props::insert_default_component(
            &mut self.world,
            &self.registry,
            entity,
            type_path,
        );
        if ok {
            self.dirty = true;
        }
        ok
    }

    /// Remove `type_path`'s component from `entity` (E-P1 remove-component).
    /// Returns whether the component type is known. Sets the dirty flag.
    pub fn remove_component(&mut self, entity: Entity, type_path: &str) -> bool {
        let ok = crate::props::remove_component(&mut self.world, &self.registry, entity, type_path);
        if ok {
            self.dirty = true;
        }
        ok
    }

    // ── spawn / despawn ──────────────────────────────────────────────────

    /// Spawn an empty editor entity (Guid + Name + identity Transform +
    /// Visibility) under `parent`, with a fresh random Guid.
    pub fn spawn(&mut self, name: &str, parent: Option<Entity>) -> Entity {
        self.spawn_with_guid(Uuid::new_v4(), name, parent)
    }

    /// Spawn with an explicit Guid (used by load + deterministic tests).
    pub fn spawn_with_guid(&mut self, guid: Uuid, name: &str, parent: Option<Entity>) -> Entity {
        let entity = self
            .world
            .spawn((
                Guid(guid),
                Name(name.to_string()),
                Transform::IDENTITY,
                Visibility::default(),
                GlobalTransform::default(),
                ComputedVisibility::default(),
            ))
            .id();
        self.guids.insert(guid, entity);
        if let Some(p) = parent {
            set_parent(&mut self.world, entity, Some(p));
        }
        self.dirty = true;
        entity
    }

    /// Despawn `entity` and all descendants; purges their Guids from the index.
    pub fn despawn(&mut self, entity: Entity) {
        for e in self.subtree(entity) {
            if let Some(g) = self.guid_of(e) {
                self.guids.remove(&g);
            }
        }
        if let Ok(e) = self.world.get_entity_mut(entity) {
            e.despawn();
        }
        self.dirty = true;
    }

    // ── hierarchy / naming ───────────────────────────────────────────────

    /// Re-parent `child` under `parent` (`None` → root). Returns false if the
    /// move would create a cycle.
    pub fn reparent(&mut self, child: Entity, parent: Option<Entity>) -> bool {
        let ok = set_parent(&mut self.world, child, parent);
        if ok {
            self.dirty = true;
        }
        ok
    }

    pub fn rename(&mut self, entity: Entity, name: &str) {
        if let Some(mut n) = self.world.get_mut::<Name>(entity) {
            n.0 = name.to_string();
        }
    }

    pub fn set_visible(&mut self, entity: Entity, visible: bool) {
        if let Some(mut v) = self.world.get_mut::<Visibility>(entity) {
            v.visible = visible;
            self.dirty = true;
        }
    }

    // ── lookups ──────────────────────────────────────────────────────────

    pub fn entity_of(&self, guid: Uuid) -> Option<Entity> {
        self.guids.get(&guid).copied()
    }

    pub fn guid_of(&self, entity: Entity) -> Option<Uuid> {
        self.world.get::<Guid>(entity).map(|g| g.0)
    }

    pub fn name_of(&self, entity: Entity) -> Option<&str> {
        self.world.get::<Name>(entity).map(|n| n.as_str())
    }

    pub fn parent_of(&self, entity: Entity) -> Option<Entity> {
        parent_of(&self.world, entity)
    }

    pub fn children_of(&self, entity: Entity) -> Vec<Entity> {
        children_of(&self.world, entity)
    }

    /// `entity` and all its descendants, depth-first.
    pub fn subtree(&self, entity: Entity) -> Vec<Entity> {
        let mut out = Vec::new();
        let mut stack = vec![entity];
        while let Some(e) = stack.pop() {
            out.push(e);
            for c in children_of(&self.world, e) {
                stack.push(c);
            }
        }
        out
    }

    /// Every editor entity (has a Guid), in Guid order for determinism.
    pub fn entities(&mut self) -> Vec<Entity> {
        let mut v: Vec<(Uuid, Entity)> = self
            .world
            .query::<(&Guid, Entity)>()
            .iter(&self.world)
            .map(|(g, e)| (g.0, e))
            .collect();
        v.sort_by_key(|(g, _)| *g);
        v.into_iter().map(|(_, e)| e).collect()
    }

    /// World-space position of an entity (post-propagation).
    pub fn world_translation(&self, entity: Entity) -> Option<DVec3> {
        self.world
            .get::<GlobalTransform>(entity)
            .map(|g| g.translation())
    }

    // ── propagation / dirty tracking ─────────────────────────────────────

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Recompute world transforms + effective visibility if anything changed.
    pub fn propagate(&mut self) {
        if self.dirty {
            transform::propagate(&mut self.world);
            self.dirty = false;
        }
    }

    /// Rebuild the Guid index from scratch (after a bulk load / clear).
    pub fn reindex_guids(&mut self) {
        self.guids.clear();
        let pairs: Vec<(Uuid, Entity)> = self
            .world
            .query::<(&Guid, Entity)>()
            .iter(&self.world)
            .map(|(g, e)| (g.0, e))
            .collect();
        for (g, e) in pairs {
            self.guids.insert(g, e);
        }
        self.dirty = true;
    }

    /// Remove every editor entity (fresh document / before load).
    pub fn clear(&mut self) {
        let roots: Vec<Entity> = crate::hierarchy::roots(&mut self.world);
        for r in roots {
            if let Ok(e) = self.world.get_entity_mut(r) {
                e.despawn();
            }
        }
        self.guids.clear();
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_indexes_guid_and_parents() {
        let mut w = EcsWorld::new();
        let a = w.spawn("A", None);
        let b = w.spawn("B", Some(a));
        let ga = w.guid_of(a).unwrap();
        assert_eq!(w.entity_of(ga), Some(a));
        assert_eq!(w.parent_of(b), Some(a));
        assert_eq!(w.children_of(a), vec![b]);
    }

    #[test]
    fn despawn_removes_subtree_and_guids() {
        let mut w = EcsWorld::new();
        let a = w.spawn("A", None);
        let b = w.spawn("B", Some(a));
        let gb = w.guid_of(b).unwrap();
        w.despawn(a);
        assert_eq!(w.entity_of(gb), None);
        assert!(w.entities().is_empty());
    }

    #[test]
    fn reparent_rejects_cycles() {
        let mut w = EcsWorld::new();
        let a = w.spawn("A", None);
        let b = w.spawn("B", Some(a));
        // Making A a child of its own descendant B must fail.
        assert!(!w.reparent(a, Some(b)));
        assert_eq!(w.parent_of(a), None);
    }

    #[test]
    fn propagate_clears_dirty() {
        let mut w = EcsWorld::new();
        w.spawn("A", None);
        assert!(w.is_dirty());
        w.propagate();
        assert!(!w.is_dirty());
    }
}
