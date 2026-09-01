//! **The `engine.*` kit's one rule** (wave SCRIPT3) — spawn, destroy and turn.
//!
//! Three verbs sat in the node kit from Phase 6 with nothing behind them.
//! SCRIPT2a's generalized registry gate measured it: `engine.set_rotation`,
//! `engine.spawn` and `engine.destroy` were registered, callable from `.infini`
//! text, and **implemented by neither host** — both ended their `match` on the
//! unknown-call arm, logged the path and answered `Value::Unit`. A heist that
//! cannot put anything in the world is not a heist, so SCRIPT3 owed them.
//!
//! They live here, in Ring 0, for the reason the `door.*` and `item.*` kits do:
//! **one rule, two hosts.** `inf_editor_core::simulate` and
//! `inf_player::runtime_sim` are mirrors, and a spawn implemented twice is two
//! implementations that agree until they do not (P22's *one door for three
//! paths*). Each host arm is a call to a function here and nothing else, so the
//! two hosts' added lines diff to nothing — and since the SCRIPT3 audit that is
//! an **arm** rather than a sentence: `nodekit`'s
//! `both_hosts_reach_the_same_ring_0_rule_for_the_engine_kit` extracts the three
//! arms out of each host's source and compares them as code. Reaching a shared
//! rule was already checked and is not sufficient — a host could wrap the call
//! in a guard or a clamp, name the rule on the line in the middle, and diverge
//! on the first program that took the wrapped path.
//!
//! # The identity of a spawned thing is CONTENT, never a counter
//!
//! [`authored_spawn_guid`] folds the prefab name and the place into a `Uuid`,
//! exactly as [`crate::item::authored_pickup_guid`] does and for exactly its
//! reason: *a spawn keyed on how many times a graph had run would put two hosts'
//! worlds out of step the first time one of them ran a handler twice.* PIE and
//! the shipped player run the same program over the same world, so they fold the
//! same bytes and name the same entity.
//!
//! The consequence a designer has to be told rather than discover: **two spawns
//! of one prefab at one point are one entity.** `spawn_with_guid` would
//! otherwise register a second entity under a GUID the index already holds, so
//! [`spawn_prefab`] answers the existing entity instead, which is the pickup
//! kit's ruling met a second time.
//!
//! **And the honest half of that sentence** (the SCRIPT3 audit's correction —
//! this paragraph used to say "an author who wants two puts them in two
//! places", which is not a thing a script can do): **`engine.spawn` takes no
//! point.** The node has exactly one input and it is the prefab, so a spawn
//! lands at the *acting entity's* own position. An author who wants two of one
//! prefab needs two spawner actors, or a spawner that has moved between the two
//! calls, or a second name — they cannot offset the second one from inside the
//! expression. Where the *point* is the thing being authored,
//! [`crate::item::spawn_pickup`] is the verb that takes one. Giving the node a
//! second input is a kit change, and it is priced rather than half-taken.
//!
//! # The handle is folded from the identity, not handed out
//!
//! The IR has no `Guid` value: a Blueprint addresses an entity as an opaque
//! `i64`, and both hosts keep an `i64 → Guid` map seeded in `Guid` order at
//! entry. A spawn has to add to that map, and the id it adds must be the same
//! number in both hosts — so it is folded from the GUID
//! ([`spawn_entity_id`]) rather than taken from a counter. Two hosts that agree
//! about the entity agree about its handle, whatever order they got there in.
//!
//! # What a prefab name means today, and the bound that is stated not hidden
//!
//! `engine.spawn`'s `prefab` is the node kit's **only** `StrRole::Asset` port
//! (`inf_blueprint::assetrefs::STR_PORTS`), and the cook resolves it:
//! a name that matches a committed asset's file stem, or a GUID, pulls that
//! asset into the pack's dependency closure. At **run time** the two are not
//! symmetric, and the reason is measurable rather than an oversight:
//!
//! * a **GUID** string binds — the spawned entity gets `MeshRef { asset:
//!   Some(guid) }`, so it draws whatever the pack (or the content directory)
//!   holds under that id;
//! * a **name** does not — an `.ipack` entry carries a GUID, a kind and a
//!   content hash and **no name** (`inf_asset::PackEntry`), so a shipped player
//!   has nothing to resolve a stem against. Binding by name would need a name
//!   index in the pack, which is a pack-format move; it is priced here rather
//!   than half-taken.
//!
//! Either way the entity is real, named, transformed, collider-free and
//! **placeholder-cubed**, which is the P4 drag-and-drop ruling ("a real,
//! selectable, saveable placeholder primitive named after the asset") and the
//! `item::spawn_pickup` ruling ("a cube, because the engine has no item
//! geometry") met a third time.

use uuid::Uuid;

use crate::components::{MeshRef, Primitive, Transform, Visibility};
use crate::math::Vec3d;
use crate::world::EcsWorld;

/// The salt that carves **script-spawned** entities' GUID space out of the
/// scene's own, the [`crate::item::authored_pickup_guid`] pattern with a
/// different constant so a spawn and a pickup of the same name at the same point
/// are different entities.
const SPAWN_SALT: u128 = 0x5350_4157_4e5f_5052_4546_4142_5f53_4331;

/// **The identity of an entity a script spawned**, folded from the prefab name
/// and the place it was asked for.
///
/// A pure function of what the program asked for and **not** of a counter — see
/// the module header. Two spawns of one prefab at one point name one entity.
pub fn authored_spawn_guid(prefab: &str, at: Vec3d) -> Uuid {
    let mut x = SPAWN_SALT;
    for b in prefab.as_bytes() {
        x ^= u128::from(*b);
        x = x
            .rotate_left(11)
            .wrapping_mul(0x0100_0000_01b3_0100_0000_01b3_0100_0001);
    }
    for v in [at.x, at.y, at.z] {
        x ^= u128::from(v.to_bits());
        x = x.rotate_left(29) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    }
    Uuid::from_u128(x)
}

/// The **blueprint handle** of a spawned entity: its GUID folded into the `i64`
/// an IR expression can hold.
///
/// Three properties, and all three are load-bearing:
///
/// 1. **It is a function of the identity alone**, so two hosts that spawned the
///    same thing hand the same number to the same program.
/// 2. **It cannot collide with an authored actor's id.** Both hosts number their
///    actors `1..=n` from a `Guid`-ordered walk at entry, and this sets bit 52,
///    so every spawned handle is at least `2^52` and no actor's is.
/// 3. **It survives the language's only numeric widening.** InfiniScript's
///    arithmetic is float-first and `math.to_float` is the way a handle reaches
///    a `float` member variable, so a handle above `2^53` would be rounded on the
///    way in and name a different entity on the way out — `engine.destroy` would
///    silently miss. The fold is therefore 52 bits, which is inside the range
///    where every integer is an exact `f64`. A handle a script can hold must
///    survive being held.
///
/// Two *different* GUIDs colliding needs their low 52 bits to agree — 1 in
/// 4.5 × 10^15. Stated rather than guarded, because a guard would need a table
/// the second host does not have.
pub fn spawn_entity_id(guid: Uuid) -> i64 {
    let low = guid.as_u128() as u64;
    ((low & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000) as i64
}

/// **`engine.spawn`.** Put a copy of `prefab` in the world at `at` and answer
/// `(guid, handle)`.
///
/// The entity is `Guid + Name + Transform + Visibility + MeshRef` — the shape
/// [`EcsWorld::spawn_with_guid`] makes plus the two components anything drawn
/// needs. `MeshRef::asset` is `Some` exactly when `prefab` parses as a `Uuid`
/// (the module header states why a *name* cannot bind at run time).
///
/// **Idempotent by identity**: called twice with the same name and place it
/// answers the same pair the second time and spawns nothing, because the GUID is
/// already in the world's index. That is the same answer both hosts give and the
/// same answer a replay gives.
pub fn spawn_prefab(world: &mut EcsWorld, prefab: &str, at: Vec3d) -> (Uuid, i64) {
    let guid = authored_spawn_guid(prefab, at);
    let handle = spawn_entity_id(guid);
    if world.entity_of(guid).is_some() {
        return (guid, handle);
    }
    let name = if prefab.is_empty() { "Prefab" } else { prefab };
    let entity = world.spawn_with_guid(guid, name, None);
    let mut t = Transform::IDENTITY;
    t.translation = at;
    world.world_mut().entity_mut(entity).insert((
        t,
        MeshRef {
            primitive: Primitive::Cube,
            asset: prefab.parse::<Uuid>().ok(),
        },
        Visibility::default(),
    ));
    world.mark_dirty();
    (guid, handle)
}

/// **`engine.destroy`.** Remove `guid` and everything under it, and answer
/// **every guid that left the world**; empty when the world does not have it.
///
/// Through [`EcsWorld::despawn`], which is the door `cell_stream` deactivates a
/// cell with — the guid index is purged with the entity, so a handle that named
/// it stops resolving rather than dangling. A runtime-spawned entity is in no
/// cell's list, so streaming will never despawn it and this is the only way it
/// goes (`inf_player::cell_stream`'s own header states that half).
///
/// # Why it answers the SUBTREE and not a `bool`
///
/// The SCRIPT3 audit's finding, and it is the shape a `bool` hides. The verb's
/// own description promises *"an entity, **and everything parented under it**"*,
/// and both hosts use the answer to decide whose handlers stop. Told only that
/// "something was destroyed", a host removes the **root** from its actor map and
/// leaves every destroyed *child* actor in it — ticking, every step, against a
/// world that no longer has its entity. That is exactly the ghost
/// `an_actor_that_destroys_itself_finishes_the_handler_and_stops` exists to
/// forbid, one level down.
///
/// The list is [`EcsWorld::subtree`] order, which is a deterministic DFS, so the
/// two hosts drain the same guids in the same order. An entity in the subtree
/// with no [`crate::components::Guid`] contributes nothing — it was addressable
/// by nobody.
pub fn destroy_entity(world: &mut EcsWorld, guid: Uuid) -> Vec<Uuid> {
    let Some(entity) = world.entity_of(guid) else {
        return Vec::new();
    };
    let purged: Vec<Uuid> = world
        .subtree(entity)
        .into_iter()
        .filter_map(|e| world.guid_of(e))
        .collect();
    world.despawn(entity);
    purged
}

/// **`engine.set_rotation`.** Turn `guid` to an absolute yaw in **degrees**,
/// leaving pitch and roll alone; `false` when the entity has no `Transform`.
///
/// Degrees and not radians because [`Transform::rotation`] is euler **degrees**
/// (the units doctrine's one UI-facing exception, and what the Details panel
/// shows), so a script and the inspector say the same number about one entity.
/// The value is taken as written — `370` is `370`, not `10` — because the
/// component holds what an author typed and `quat()` reduces it.
///
/// "Absolute" is **in the entity's own frame**: `Transform` is LOCAL, so on a
/// parented entity this is a yaw relative to its parent rather than to the
/// world. Everything [`spawn_prefab`] makes is a root, so for a spawned subject
/// the two are the same number; a script turning an authored *child* is the case
/// where they are not.
pub fn set_yaw_degrees(world: &mut EcsWorld, guid: Uuid, degrees: f64) -> bool {
    let Some(entity) = world.entity_of(guid) else {
        return false;
    };
    let Some(mut t) = world.world_mut().get_mut::<Transform>(entity) else {
        return false;
    };
    t.rotation.y = degrees;
    world.mark_dirty();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f64) -> Vec3d {
        Vec3d::new(x, 1.0, 2.0)
    }

    #[test]
    fn an_identity_is_a_function_of_the_name_and_the_place() {
        assert_eq!(
            authored_spawn_guid("Crate", at(3.0)),
            authored_spawn_guid("Crate", at(3.0))
        );
        assert_ne!(
            authored_spawn_guid("Crate", at(3.0)),
            authored_spawn_guid("Crate", at(4.0))
        );
        assert_ne!(
            authored_spawn_guid("Crate", at(3.0)),
            authored_spawn_guid("Crat", at(3.0))
        );
        // …and it is NOT the pickup kit's identity for the same inputs, which is
        // the whole reason the salt differs.
        assert_ne!(
            authored_spawn_guid("Crate", at(3.0)),
            crate::item::authored_pickup_guid("Crate", at(3.0))
        );
    }

    /// The handle's two properties, measured: above every actor id, and a pure
    /// function of the GUID.
    #[test]
    fn a_handle_cannot_be_mistaken_for_an_authored_actor() {
        let g = authored_spawn_guid("Crate", at(3.0));
        assert_eq!(spawn_entity_id(g), spawn_entity_id(g));
        assert!(
            spawn_entity_id(g) >= 0x0010_0000_0000_0000,
            "a spawned handle must sit above every 1..=n actor id"
        );
        // A thousand different prefabs, no collisions, none of them small — and
        // **every one of them exactly representable as an `f64`**, which is what
        // a `float` member variable holding a handle depends on.
        let ids: std::collections::BTreeSet<i64> = (0..1000)
            .map(|i| spawn_entity_id(authored_spawn_guid(&format!("p{i}"), at(0.0))))
            .collect();
        assert_eq!(ids.len(), 1000);
        assert!(ids.iter().all(|&i| i >= 0x0010_0000_0000_0000));
        assert!(
            ids.iter().all(|&i| i as f64 as i64 == i),
            "a handle that does not survive `math.to_float` names a different \
             entity on the way back"
        );
    }

    #[test]
    fn a_spawn_puts_a_named_entity_in_the_world() {
        let mut w = EcsWorld::new();
        let (guid, handle) = spawn_prefab(&mut w, "Crate", at(3.0));
        assert_eq!(handle, spawn_entity_id(guid));
        let e = w.entity_of(guid).expect("the entity is in the index");
        assert_eq!(w.name_of(e), Some("Crate"));
        assert_eq!(
            w.world()
                .get::<Transform>(e)
                .expect("a transform")
                .translation,
            at(3.0)
        );
        assert!(w.world().get::<MeshRef>(e).expect("a mesh").asset.is_none());
    }

    /// A GUID-spelled prefab BINDS; a name does not, and the bound is measured
    /// rather than described (see the module header for the pack reason).
    #[test]
    fn a_guid_spelled_prefab_binds_its_asset_and_a_name_does_not() {
        let mut w = EcsWorld::new();
        let asset = Uuid::from_u128(0x5C15_3000);
        let (by_guid, _) = spawn_prefab(&mut w, &asset.to_string(), at(1.0));
        let (by_name, _) = spawn_prefab(&mut w, "Crate", at(2.0));
        let mesh = |g: Uuid| {
            w.world()
                .get::<MeshRef>(w.entity_of(g).expect("spawned"))
                .expect("a mesh")
                .asset
        };
        assert_eq!(mesh(by_guid), Some(asset));
        assert_eq!(mesh(by_name), None);
    }

    #[test]
    fn spawning_the_same_thing_twice_is_one_entity() {
        let mut w = EcsWorld::new();
        let before = w.entities().len();
        let a = spawn_prefab(&mut w, "Crate", at(3.0));
        let b = spawn_prefab(&mut w, "Crate", at(3.0));
        assert_eq!(a, b);
        assert_eq!(w.entities().len(), before + 1);
    }

    #[test]
    fn destroy_takes_the_entity_and_its_handle_with_it() {
        let mut w = EcsWorld::new();
        let (guid, _) = spawn_prefab(&mut w, "Crate", at(3.0));
        assert_eq!(destroy_entity(&mut w, guid), vec![guid]);
        assert!(w.entity_of(guid).is_none());
        // …and a second destroy is a refusal with a value, not a panic.
        assert!(destroy_entity(&mut w, guid).is_empty());
    }

    /// **A destroy answers its whole subtree**, because a host uses the answer
    /// to decide whose handlers stop — and a destroyed CHILD actor the host was
    /// never told about keeps ticking against a world that has no entity for it.
    #[test]
    fn destroy_answers_every_guid_that_left_the_world() {
        let mut w = EcsWorld::new();
        let parent = Uuid::from_u128(0x5C13_A001);
        let child = Uuid::from_u128(0x5C13_A002);
        let grandchild = Uuid::from_u128(0x5C13_A003);
        let pe = w.spawn_with_guid(parent, "Parent", None);
        let ce = w.spawn_with_guid(child, "Child", Some(pe));
        w.spawn_with_guid(grandchild, "Grandchild", Some(ce));
        // A sibling of the parent, so the answer is the subtree and not the world.
        let bystander = Uuid::from_u128(0x5C13_A004);
        w.spawn_with_guid(bystander, "Bystander", None);

        let purged = destroy_entity(&mut w, parent);
        assert_eq!(
            purged
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            [parent, child, grandchild].into_iter().collect(),
            "a destroy must name every guid that left, or a host stops the root's \
             handlers and leaves its children's running"
        );
        for g in [parent, child, grandchild] {
            assert!(w.entity_of(g).is_none());
        }
        assert!(
            w.entity_of(bystander).is_some(),
            "the answer is the SUBTREE, not the world"
        );
    }

    #[test]
    fn set_rotation_writes_yaw_and_leaves_the_other_two() {
        let mut w = EcsWorld::new();
        let (guid, _) = spawn_prefab(&mut w, "Crate", at(3.0));
        let e = w.entity_of(guid).expect("spawned");
        w.world_mut()
            .get_mut::<Transform>(e)
            .expect("a transform")
            .rotation = Vec3d::new(5.0, 0.0, 7.0);
        assert!(set_yaw_degrees(&mut w, guid, 90.0));
        assert_eq!(
            w.world().get::<Transform>(e).expect("a transform").rotation,
            Vec3d::new(5.0, 90.0, 7.0)
        );
        assert!(!set_yaw_degrees(&mut w, Uuid::nil(), 90.0));
    }
}
