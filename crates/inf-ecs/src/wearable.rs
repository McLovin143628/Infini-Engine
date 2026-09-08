//! **WEARABLES** (wave OUTFIT1): the derived rule that lets a character wear a
//! second skinned mesh — an outfit, a head of hair cards, a pair of eyes —
//! without a schema byte moving.
//!
//! # The engine draws ONE `SkeletalMesh` per entity, and that is not changing
//!
//! Both projectors' skeletal branch resolves *the* mesh on *the* entity. A
//! MetaHuman's clothes are a second skeletal mesh on the same rig, and its hair
//! cards a third; a body that had to carry them in one asset would be a body
//! whose shirt cannot be taken off. So a garment is a **child entity** that names
//! its own mesh and its own material, and this module is the one rule that says
//! whose pose it wears.
//!
//! # THE RULE, and why it is derived rather than persisted
//!
//! > A [`SkeletalMesh`] entity whose **wearer** — its hierarchy parent, or the
//! > target of its [`AttachedTo`] — carries a [`SkeletalMesh`] bound to the
//! > **same skeleton GUID** is a WEARABLE of that wearer.
//!
//! Nothing is stored. There is no `Wearable` component, no new field on
//! `SkeletalMesh`, no payload bump and no downgrade bless — the fact is already
//! on disk in two things that have been persisted since scene v5 (an entity's
//! `parent`, and `SkeletalMesh.skeleton`), and this reads it back. The price of
//! the alternative was written down before this door was chosen: a persisted
//! `Wearable { of: Uuid }` is 16 bytes an entity plus its `Option` tag, both
//! `apply_record` mirrors, a `ScenePayload` bump, a downgrade bless and 24
//! levels re-cooked — for a fact the document already states.
//!
//! # What follows from it, and what does NOT
//!
//! A wearable is drawn with the **wearer's** evaluated pose, palette,
//! model-space lift, interpolated position, crowd tier and near fade, and with
//! its **own** mesh, material, sections and pick id. Everything else it gets for
//! free from being a child entity rather than from anything here:
//!
//! * it **rides the wearer's transform** — `propagate()` composes parent × local;
//! * it **culls with the wearer** — `ComputedVisibility` is the AND of the
//!   ancestor chain, so hiding a character hides its clothes;
//! * it **is cooked with the wearer** — the cook's dependency closure walks
//!   `SkeletalMesh` refs entity by entity and a child is an entity;
//! * it **follows the ragdoll** — the wearer's evaluated pose *is* the ragdoll's
//!   while it is one ([`crate::pose`]'s read-back), and the wearable reads that
//!   same pose.
//!
//! What does not follow: a wearable is **not** separately simulated, not
//! separately posed and not a second pose evaluation. It costs a draw and a
//! palette reference, which is what [`crate::pose::posed_count`] deliberately
//! keeps *not* counting.
//!
//! # The `AttachedTo` half
//!
//! Both parentages are accepted because both already exist and each is right for
//! a different author. A level's dressed character is a hierarchy child (free
//! culling, free transform, persisted). A garment put on at RUNTIME — the
//! equipped-weapon precedent in `inf_physics::d3::gameplay` — has no document to
//! be a child of, and `AttachedTo` is the door that spawns one. The rule is the
//! same for both, so a gate that measures one measures the other.
//!
//! [`SkeletalMesh`]: crate::components::SkeletalMesh
//! [`AttachedTo`]: crate::components::AttachedTo

use bevy_ecs::entity::Entity;
use uuid::Uuid;

use crate::components::{AttachedTo, Guid, SkeletalMesh};
use crate::world::EcsWorld;

/// How many wearer hops [`wearer_of`] will follow before giving up.
///
/// A hat on a hood on a body is two, and nothing this engine authors is deeper.
/// The bound is not a performance guard — it is what makes a malformed document
/// (a cycle an editor should never write, or an `AttachedTo` ring) terminate with
/// an answer instead of hanging a projector.
const MAX_HOPS: usize = 4;

/// The entity a wearable takes its pose from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wearer {
    /// The wearer's ECS entity, for the component reads a projector does.
    pub entity: Entity,
    /// The wearer's stable identity, for the pose store and the camera subject.
    pub guid: Uuid,
}

/// **The rule.** The wearer `entity` is dressing, or `None` if it is not a
/// wearable.
///
/// Walks up at most [`MAX_HOPS`] links so a wearable worn by a wearable (a hat on
/// a hood) resolves to the body at the bottom rather than to the middle, which is
/// the only entity in such a chain that has a pose of its own. A link whose
/// skeleton differs stops the walk: two skeletons is two characters standing in
/// the same place, not one wearing the other.
pub fn wearer_of(world: &EcsWorld, entity: Entity) -> Option<Wearer> {
    let w = world.world();
    let skeleton = w.get::<SkeletalMesh>(entity)?.skeleton?;
    let mut cur = entity;
    let mut found: Option<Wearer> = None;
    for _ in 0..MAX_HOPS {
        let Some(up) = crate::hierarchy::parent_of(w, cur).or_else(|| {
            w.get::<AttachedTo>(cur)
                .and_then(|a| world.entity_of(a.target))
        }) else {
            break;
        };
        if up == cur || up == entity {
            break;
        }
        if w.get::<SkeletalMesh>(up).and_then(|s| s.skeleton) != Some(skeleton) {
            break;
        }
        let Some(guid) = w.get::<Guid>(up).map(|g| g.0) else {
            break;
        };
        found = Some(Wearer { entity: up, guid });
        cur = up;
    }
    found
}

/// **The door both projectors go through**: whose pose, position, tier and fade
/// this entity draws with.
///
/// `(entity, guid)` unchanged for everything that is not a wearable — which is
/// every character, every crowd agent and every prop in this tree before this
/// wave — so the two projectors' skeletal branches compute exactly what they
/// computed before on every entity that already existed.
pub fn pose_source(world: &EcsWorld, entity: Entity, guid: Uuid) -> (Entity, Uuid) {
    match wearer_of(world, entity) {
        Some(wearer) => (wearer.entity, wearer.guid),
        None => (entity, guid),
    }
}

/// Every wearable `wearer` is wearing, by GUID, sorted.
///
/// Sorted because the callers are gates, an editor Details panel and a cook
/// census, and bevy's iteration order is unspecified: an answer that changed
/// between two reads of one world would make all three lie differently.
pub fn wearables_of(world: &EcsWorld, wearer: Uuid) -> Vec<Uuid> {
    let mut out: Vec<Uuid> = Vec::new();
    let w = world.world();
    for e in w.iter_entities() {
        let Some(guid) = e.get::<Guid>().map(|g| g.0) else {
            continue;
        };
        if wearer_of(world, e.id()).map(|x| x.guid) == Some(wearer) {
            out.push(guid);
        }
    }
    out.sort();
    out
}

/// How many wearables the world holds — the census a perf ledger quotes.
pub fn worn_count(world: &EcsWorld) -> usize {
    let w = world.world();
    w.iter_entities()
        .filter(|e| wearer_of(world, e.id()).is_some())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Transform;

    const RIG: Uuid = Uuid::from_u128(0x0FF1_7000_0000_0001);
    const OTHER_RIG: Uuid = Uuid::from_u128(0x0FF1_7000_0000_0002);
    const BODY: Uuid = Uuid::from_u128(0x0FF1_7000_0000_0010);
    const SHIRT: Uuid = Uuid::from_u128(0x0FF1_7000_0000_0011);

    fn skeletal(world: &mut EcsWorld, guid: Uuid, name: &str, skeleton: Uuid) -> Entity {
        let e = world.spawn_with_guid(guid, name, None);
        world.world_mut().entity_mut(e).insert((
            SkeletalMesh {
                mesh: Some(Uuid::from_u128(guid.as_u128() ^ 0xABCD)),
                skeleton: Some(skeleton),
            },
            Transform::IDENTITY,
        ));
        e
    }

    /// The rule, on the shape a level writes: a child on the same rig.
    #[test]
    fn a_child_on_the_same_rig_is_a_wearable_and_one_on_another_rig_is_not() {
        let mut world = EcsWorld::new();
        let body = skeletal(&mut world, BODY, "Body", RIG);
        let shirt = skeletal(&mut world, SHIRT, "Shirt", RIG);
        crate::hierarchy::set_parent(world.world_mut(), shirt, Some(body));
        assert_eq!(
            wearer_of(&world, shirt).map(|x| x.guid),
            Some(BODY),
            "a child on the wearer's own rig is not read as a wearable"
        );
        assert_eq!(wearer_of(&world, body), None, "the body is wearing itself");
        assert_eq!(pose_source(&world, shirt, SHIRT), (body, BODY));
        assert_eq!(pose_source(&world, body, BODY), (body, BODY));
        assert_eq!(wearables_of(&world, BODY), vec![SHIRT]);
        assert_eq!(worn_count(&world), 1);

        // The MUTATION: re-bind the shirt to a different rig. It is then a
        // character standing inside another character, not a garment, and the
        // rule has to say so — otherwise every second body parented for
        // convenience would inherit somebody else's pose.
        world.world_mut().entity_mut(shirt).insert(SkeletalMesh {
            mesh: Some(Uuid::from_u128(1)),
            skeleton: Some(OTHER_RIG),
        });
        assert_eq!(
            wearer_of(&world, shirt),
            None,
            "a child on ANOTHER rig is being drawn with this body's pose"
        );
        assert_eq!(worn_count(&world), 0);
    }

    /// The runtime shape: `AttachedTo`, the equipped-weapon precedent.
    #[test]
    fn an_attached_wearable_resolves_the_same_way_a_parented_one_does() {
        let mut world = EcsWorld::new();
        let body = skeletal(&mut world, BODY, "Body", RIG);
        let shirt = skeletal(&mut world, SHIRT, "Shirt", RIG);
        world.world_mut().entity_mut(shirt).insert(AttachedTo::new(
            BODY,
            "",
            crate::math::Vec3d::ZERO,
        ));
        assert_eq!(wearer_of(&world, shirt).map(|x| x.entity), Some(body));
        assert_eq!(wearer_of(&world, shirt).map(|x| x.guid), Some(BODY));
    }

    /// A hat on a hood resolves to the body — the only entity with a pose.
    #[test]
    fn a_wearable_worn_by_a_wearable_resolves_to_the_body_at_the_bottom() {
        let mut world = EcsWorld::new();
        let body = skeletal(&mut world, BODY, "Body", RIG);
        let hood = skeletal(&mut world, SHIRT, "Hood", RIG);
        let hat = skeletal(
            &mut world,
            Uuid::from_u128(0x0FF1_7000_0000_0012),
            "Hat",
            RIG,
        );
        crate::hierarchy::set_parent(world.world_mut(), hood, Some(body));
        crate::hierarchy::set_parent(world.world_mut(), hat, Some(hood));
        assert_eq!(wearer_of(&world, hat).map(|x| x.guid), Some(BODY));
        assert_eq!(worn_count(&world), 2);
    }

    /// An `AttachedTo` ring terminates with an answer rather than hanging the
    /// projector that asked.
    #[test]
    fn an_attachment_ring_terminates() {
        let mut world = EcsWorld::new();
        let a = skeletal(&mut world, BODY, "A", RIG);
        let b = skeletal(&mut world, SHIRT, "B", RIG);
        world.world_mut().entity_mut(a).insert(AttachedTo::new(
            SHIRT,
            "",
            crate::math::Vec3d::ZERO,
        ));
        world
            .world_mut()
            .entity_mut(b)
            .insert(AttachedTo::new(BODY, "", crate::math::Vec3d::ZERO));
        // Whatever it answers, it answers: the test is that it returns at all.
        let _ = wearer_of(&world, a);
        let _ = wearer_of(&world, b);
    }
}
