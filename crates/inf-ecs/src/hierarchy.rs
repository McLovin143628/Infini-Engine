//! Transform hierarchy (P3.1.2), built on bevy_ecs relationships.
//!
//! We reuse bevy's `ChildOf` / `Children` relationship pair: inserting
//! `ChildOf(parent)` auto-maintains the parent's `Children`, re-inserting it
//! moves the child (removing it from the old parent), and `despawn()` is
//! recursive over `Children`. These thin helpers give the editor a stable
//! vocabulary (`reparent`, `set_parent`, `roots`) without leaking bevy details
//! past the facade.

pub use bevy_ecs::hierarchy::{ChildOf, Children};
use bevy_ecs::prelude::*;

use crate::components::Guid;

/// The parent of `entity`, or `None` if it is a root.
pub fn parent_of(world: &World, entity: Entity) -> Option<Entity> {
    world.get::<ChildOf>(entity).map(|c| c.parent())
}

/// The ordered children of `entity`.
pub fn children_of(world: &World, entity: Entity) -> Vec<Entity> {
    world
        .get::<Children>(entity)
        .map(|c| c.iter().collect())
        .unwrap_or_default()
}

/// Re-parent `child` under `parent` (`None` makes it a root). Cycles are
/// rejected (a node may not become a descendant of itself); returns whether the
/// change was applied.
pub fn set_parent(world: &mut World, child: Entity, parent: Option<Entity>) -> bool {
    if let Some(p) = parent {
        if p == child || is_ancestor(world, child, p) {
            return false; // would create a cycle
        }
        world.entity_mut(child).insert(ChildOf(p));
    } else {
        world.entity_mut(child).remove::<ChildOf>();
    }
    true
}

/// Is `maybe_ancestor` an ancestor of (or equal to) `entity`?
pub fn is_ancestor(world: &World, maybe_ancestor: Entity, entity: Entity) -> bool {
    let mut cur = Some(entity);
    while let Some(e) = cur {
        if e == maybe_ancestor {
            return true;
        }
        cur = parent_of(world, e);
    }
    false
}

/// Editor roots: entities that carry a [`Guid`] and have no parent, in a
/// deterministic order (bevy iteration order is unspecified, so callers that
/// need stable ordering should sort by `Guid`).
pub fn roots(world: &mut World) -> Vec<Entity> {
    world
        .query_filtered::<Entity, (With<Guid>, Without<ChildOf>)>()
        .iter(world)
        .collect()
}
