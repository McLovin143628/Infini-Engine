//! **What a collider IS** (wave COV1) — the runtime side table that turns a
//! bridge handle back into a place a person can name.
//!
//! # Why this exists, and it is a carried finding
//!
//! The CHAR1c audit's carried 162: *"Harbour City's buildings have no ECS
//! collider. A census within 60 m of the spawn finds twenty dynamic boxes and
//! four kinematic capsules and nothing else; the façades' colliders exist only
//! in the physics bridge, with no entity and therefore no `Name`. Every
//! offender the audit's route caught is reported as `entity-less guid …`, which
//! makes a camera failure a GUID rather than a place."*
//!
//! Six kinds of collider in this engine have **no entity at all**: a PCG
//! structure's solid, a group's far shell, a terrain tile, a voxel chunk, a
//! kerb slab, a fracture chunk and a door leaf. Each one is minted by a pure
//! function of `(owner, index)` — [`super::pcg_structure_guid`] and its five
//! siblings — and each of those functions is a **one-way mix**, so a guid that
//! comes back out of [`super::PhysicsBridge3D::guid_of_collider`] cannot be
//! turned back into what it names.
//!
//! So the mint sites record what they minted. That is all this is: a
//! `BTreeMap<Uuid, ColliderLabel>` filled where the descriptor is built,
//! pruned beside [`super::PhysicsBridge3D`]'s own collider map, and read by
//! [`super::PhysicsBridge3D::describe_collider`].
//!
//! # No schema moves and nothing is persisted
//!
//! A label is derived state about a derived collider. It is rebuilt whenever the
//! thing it names is re-described, it is dropped when the thing goes away, and
//! nothing ever writes one to a file — the [`inf_ecs::dispatch::RespondersRes`]
//! shape, one ring down.
//!
//! # Who reads it
//!
//! Wave COV1's cover probe, whose whole job is to answer *what am I behind* —
//! a wall, a parked car, a container, a kerb — and whose failure messages are
//! useless without it. The camera's own offender report (carried 162) is the
//! second reader the day somebody writes it.

use uuid::Uuid;

use inf_ecs::world::EcsWorld;

/// What FAMILY of thing a collider belongs to.
///
/// Ordered as the bridge's own gather passes run, so a reader meeting the enum
/// meets it in the order the world is assembled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ColliderFamily {
    /// A real entity in the document — it has a `Guid`, and usually a `Name`.
    /// The default, because an unlabelled collider is one the bridge attached
    /// for an entity it was handed.
    #[default]
    Entity,
    /// One solid of a `PcgVolume`'s structural population: a wall, a floor
    /// slab, a container, a piece of a building module.
    Structure,
    /// The single oriented box that stands in for a whole building group past
    /// the collider band's near radius.
    StructureShell,
    /// A terrain tile's height field.
    Terrain,
    /// A voxel chunk's trimesh.
    Voxel,
    /// One slab of a street's kerb (wave ROAD1b).
    Kerb,
    /// One chunk of a fractured actor.
    Fracture,
    /// A door leaf.
    DoorLeaf,
}

impl ColliderFamily {
    /// The word a failure message uses.
    pub fn noun(self) -> &'static str {
        match self {
            ColliderFamily::Entity => "entity",
            ColliderFamily::Structure => "structure",
            ColliderFamily::StructureShell => "building shell",
            ColliderFamily::Terrain => "terrain tile",
            ColliderFamily::Voxel => "voxel chunk",
            ColliderFamily::Kerb => "kerb slab",
            ColliderFamily::Fracture => "fracture chunk",
            ColliderFamily::DoorLeaf => "door leaf",
        }
    }

    /// Whether a surface of this family is **worth taking cover behind** at all
    /// (wave COV1).
    ///
    /// The three that are not: the terrain (the ground the character is
    /// standing on is never the wall in front of it — the walkable test already
    /// refuses it, and this is the second, cheaper refusal), a kerb (it is
    /// 12 cm tall and the class floor refuses it on height anyway — this says
    /// so by NAME so the refusal message reads "a kerb is not cover" rather
    /// than "0.12 m is below 0.55 m"), and a door leaf (a thing that swings
    /// open is not a thing to put your back against).
    pub fn is_coverable(self) -> bool {
        !matches!(
            self,
            ColliderFamily::Terrain | ColliderFamily::Kerb | ColliderFamily::DoorLeaf
        )
    }
}

/// **What one collider is**: its family, the thing that owns it, and which one
/// of that owner's parts it is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColliderLabel {
    /// What kind of thing it is.
    pub family: ColliderFamily,
    /// The **block** — the entity whose evaluation produced it: the `PcgVolume`
    /// for a structure, the `Terrain` for a tile, the destructible actor for a
    /// chunk, the entity itself for [`ColliderFamily::Entity`].
    pub owner: Uuid,
    /// Which of the owner's parts. `0` for a family that has only one.
    pub index: u32,
}

impl ColliderLabel {
    /// A label for a part of `owner`.
    pub fn part(family: ColliderFamily, owner: Uuid, index: u32) -> Self {
        Self {
            family,
            owner,
            index,
        }
    }

    /// The entity label for `guid` — what an unlabelled collider means.
    pub fn entity(guid: Uuid) -> Self {
        Self {
            family: ColliderFamily::Entity,
            owner: guid,
            index: 0,
        }
    }

    /// **Say what this is, in words**, resolving the owner's `Name` out of
    /// `world` when it has one.
    ///
    /// The whole point of the door: `structure #412 of block `HarbourCity``
    /// rather than `entity-less guid 7f3a…`.
    pub fn describe(&self, world: &EcsWorld) -> String {
        // A **nil** owner means the family has no owning entity at all: a kerb
        // slab is derived from the traffic RESOURCE, which is a fact about the
        // level rather than about a row in it. Saying "the level" is the honest
        // answer and a nil uuid printed out is not.
        let block = if self.owner.is_nil() {
            "the level".to_string()
        } else {
            world
                .entity_of(self.owner)
                .and_then(|e| world.world().get::<inf_ecs::components::Name>(e))
                .map(|n| format!("`{}`", n.0))
                .unwrap_or_else(|| format!("{}", self.owner))
        };
        match self.family {
            ColliderFamily::Entity => block,
            f => format!("{} #{} of {}", f.noun(), self.index, block),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three families a cover surface may **not** be, by name — the refusal
    /// a height threshold cannot express.
    #[test]
    fn the_ground_a_kerb_and_a_door_leaf_are_not_cover() {
        for f in [
            ColliderFamily::Terrain,
            ColliderFamily::Kerb,
            ColliderFamily::DoorLeaf,
        ] {
            assert!(!f.is_coverable(), "{f:?}");
        }
        for f in [
            ColliderFamily::Entity,
            ColliderFamily::Structure,
            ColliderFamily::StructureShell,
            ColliderFamily::Voxel,
            ColliderFamily::Fracture,
        ] {
            assert!(f.is_coverable(), "{f:?}");
        }
    }

    /// **A label says what a guid could not.** An entity with a `Name` reads as
    /// its name; a structure with no entity at all reads as its family, its
    /// ordinal and the block that produced it.
    #[test]
    fn a_label_names_a_place_and_a_bare_guid_does_not() {
        let mut w = EcsWorld::new();
        let block = w.spawn("HarbourCity", None);
        let block_guid = w
            .world()
            .get::<inf_ecs::components::Guid>(block)
            .expect("a spawned entity has a guid")
            .0;
        let wall = ColliderLabel::part(ColliderFamily::Structure, block_guid, 412);
        let said = wall.describe(&w);
        assert_eq!(said, "structure #412 of `HarbourCity`", "{said}");
        // The same label with no entity behind it still names the family and the
        // ordinal — which is the half a raw guid never had.
        let orphan = ColliderLabel::part(ColliderFamily::Structure, uuid::Uuid::from_u128(7), 3);
        assert!(
            orphan.describe(&w).starts_with("structure #3 of "),
            "{}",
            orphan.describe(&w)
        );
        // An entity is its own name and carries no ordinal.
        assert_eq!(
            ColliderLabel::entity(block_guid).describe(&w),
            "`HarbourCity`"
        );
        // A kerb belongs to the level rather than to a row in it.
        let kerb = ColliderLabel::part(ColliderFamily::Kerb, uuid::Uuid::nil(), 38);
        assert_eq!(kerb.describe(&w), "kerb slab #38 of the level");
    }
}
