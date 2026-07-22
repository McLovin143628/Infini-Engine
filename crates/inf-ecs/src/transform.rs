//! Transform + visibility propagation (P3.1.2).
//!
//! A depth-first walk from every root recomputes `GlobalTransform` (parent
//! affine × local affine) and `ComputedVisibility` (AND of the ancestor chain).
//! Editor scenes are small, so propagation is a full walk rather than an
//! incremental system — but it is gated by a dirty flag on [`crate::EcsWorld`]
//! so idle frames pay nothing (the "dirty tracking" batch item). Every
//! structural or transform edit sets the flag; [`EcsWorld::propagate`] clears
//! it.

use bevy_ecs::prelude::*;
use glam::DAffine3;

use crate::components::{ComputedVisibility, GlobalTransform, Transform, Visibility};
use crate::hierarchy::{children_of, roots};

/// Recompute every entity's world transform and effective visibility.
///
/// Pre-order DFS over an **explicit stack** (not recursion — a pathologically deep
/// hierarchy must not overflow the thread stack). Order is identical to a recursive
/// walk: roots in `roots()` order, each node before its children, siblings in
/// `children_of()` order (children are pushed reversed so they pop in that order).
pub fn propagate(world: &mut World) {
    // Snapshot roots first (the queries below borrow the world mutably). Stack
    // frames carry the inherited (parent) global + visibility.
    let mut stack: Vec<(Entity, DAffine3, bool)> = Vec::new();
    for root in roots(world).into_iter().rev() {
        stack.push((root, DAffine3::IDENTITY, true));
    }

    while let Some((entity, parent_global, parent_visible)) = stack.pop() {
        let local = world
            .get::<Transform>(entity)
            .copied()
            .unwrap_or(Transform::IDENTITY);
        let self_visible = world
            .get::<Visibility>(entity)
            .map(|v| v.visible)
            .unwrap_or(true);

        let global = parent_global * local.affine();
        let visible = parent_visible && self_visible;

        // Insert-or-update the computed components in ONE bundle insert → one
        // archetype move (two separate inserts moved the entity's archetype twice).
        world
            .entity_mut(entity)
            .insert((GlobalTransform(global), ComputedVisibility(visible)));

        // Push children reversed so they pop in `children_of()` order (pre-order).
        for child in children_of(world, entity).into_iter().rev() {
            stack.push((child, global, visible));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Guid, Name};
    use crate::hierarchy::ChildOf;
    use crate::math::Vec3d;
    use glam::DVec3;
    use uuid::Uuid;

    fn node(world: &mut World, t: Transform, vis: bool) -> Entity {
        world
            .spawn((
                Guid(Uuid::from_u128(world.entities().len() as u128 + 1)),
                Name("n".into()),
                t,
                Visibility { visible: vis },
            ))
            .id()
    }

    #[test]
    fn child_world_pos_is_parent_plus_local() {
        let mut world = World::new();
        let parent = node(
            &mut world,
            Transform::from_translation(DVec3::new(10.0, 0.0, 0.0)),
            true,
        );
        let child = node(
            &mut world,
            Transform::from_translation(DVec3::new(0.0, 5.0, 0.0)),
            true,
        );
        world.entity_mut(child).insert(ChildOf(parent));
        propagate(&mut world);
        let g = world.get::<GlobalTransform>(child).unwrap();
        assert!((g.translation() - DVec3::new(10.0, 5.0, 0.0)).length() < 1e-9);
    }

    #[test]
    fn rotation_composes_child_offset() {
        let mut world = World::new();
        // Parent yawed 90° about Y; child offset +X local → world +Z (RH).
        let parent = node(
            &mut world,
            Transform {
                translation: Vec3d::ZERO,
                rotation: Vec3d::new(0.0, 90.0, 0.0),
                scale: Vec3d::ONE,
            },
            true,
        );
        let child = node(
            &mut world,
            Transform::from_translation(DVec3::new(1.0, 0.0, 0.0)),
            true,
        );
        world.entity_mut(child).insert(ChildOf(parent));
        propagate(&mut world);
        let p = world.get::<GlobalTransform>(child).unwrap().translation();
        assert!(p.z.abs() > 0.99, "expected child rotated onto Z, got {p:?}");
        assert!(p.x.abs() < 1e-6);
    }

    #[test]
    fn deep_chain_propagates_without_stack_overflow() {
        // Order guarantee under test: a pre-order DFS (parent before child) over an
        // explicit stack, so a chain far deeper than a recursive walk could survive
        // still propagates. The old recursion would overflow the thread stack here;
        // correctness is checked by folding one +X translation per link.
        let mut world = World::new();
        const N: usize = 60_000;
        let mut prev: Option<Entity> = None;
        for _ in 0..N {
            let e = node(
                &mut world,
                Transform::from_translation(DVec3::new(1.0, 0.0, 0.0)),
                true,
            );
            if let Some(p) = prev {
                // Direct ChildOf insert keeps the build O(n) (no ancestor walk).
                world.entity_mut(e).insert(ChildOf(p));
            }
            prev = Some(e);
        }
        let leaf = prev.unwrap();
        propagate(&mut world);
        let g = world.get::<GlobalTransform>(leaf).unwrap();
        assert!(
            (g.translation().x - N as f64).abs() < 1e-6,
            "leaf world X == chain depth ({N})"
        );
    }

    #[test]
    fn siblings_propagate_in_children_of_order() {
        // Pre-order DFS visits each root, then its children in `children_of()` order.
        // Both siblings must fold in the parent's global (proves parent-before-child
        // and that neither sibling is skipped by the reversed-push stack order).
        let mut world = World::new();
        let parent = node(
            &mut world,
            Transform::from_translation(DVec3::new(10.0, 0.0, 0.0)),
            true,
        );
        let a = node(
            &mut world,
            Transform::from_translation(DVec3::new(1.0, 0.0, 0.0)),
            true,
        );
        let b = node(
            &mut world,
            Transform::from_translation(DVec3::new(0.0, 2.0, 0.0)),
            true,
        );
        world.entity_mut(a).insert(ChildOf(parent));
        world.entity_mut(b).insert(ChildOf(parent));
        propagate(&mut world);
        let ga = world.get::<GlobalTransform>(a).unwrap().translation();
        let gb = world.get::<GlobalTransform>(b).unwrap().translation();
        assert!((ga - DVec3::new(11.0, 0.0, 0.0)).length() < 1e-9);
        assert!((gb - DVec3::new(10.0, 2.0, 0.0)).length() < 1e-9);
    }

    #[test]
    fn hidden_ancestor_hides_descendant() {
        let mut world = World::new();
        let parent = node(&mut world, Transform::IDENTITY, false);
        let child = node(&mut world, Transform::IDENTITY, true);
        world.entity_mut(child).insert(ChildOf(parent));
        propagate(&mut world);
        assert!(!world.get::<ComputedVisibility>(child).unwrap().0);
    }
}
