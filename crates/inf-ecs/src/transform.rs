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
pub fn propagate(world: &mut World) {
    // Snapshot roots first (the queries below borrow the world mutably).
    let roots = roots(world);
    for root in roots {
        propagate_from(world, root, DAffine3::IDENTITY, true);
    }
}

fn propagate_from(
    world: &mut World,
    entity: Entity,
    parent_global: DAffine3,
    parent_visible: bool,
) {
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

    // Insert-or-update the computed components.
    world.entity_mut(entity).insert(GlobalTransform(global));
    world.entity_mut(entity).insert(ComputedVisibility(visible));

    for child in children_of(world, entity) {
        propagate_from(world, child, global, visible);
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
    fn hidden_ancestor_hides_descendant() {
        let mut world = World::new();
        let parent = node(&mut world, Transform::IDENTITY, false);
        let child = node(&mut world, Transform::IDENTITY, true);
        world.entity_mut(child).insert(ChildOf(parent));
        propagate(&mut world);
        assert!(!world.get::<ComputedVisibility>(child).unwrap().0);
    }
}
