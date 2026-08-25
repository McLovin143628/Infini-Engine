//! **Items and inventories** (island wave I6): a name-keyed catalogue, a
//! slotted [`Inventory`] on a character, and the pick-up verb I5's one
//! interaction site calls.
//!
//! # THE PERSISTENCE ANSWER, written where the next reader will look
//!
//! [`Inventory`] is a **runtime** component with no scene slot —
//! `crate::interact::Interactable`'s shape and for its reason. That is not a
//! shortcut, and here is the exact accounting, because the question "does an
//! inventory need a wire field" is one a later wave will ask again:
//!
//! * A **runtime** inventory — what a character picks up, carries, equips and
//!   drops during a session — needs no wire field at all. It is derived, it is
//!   mutated by gameplay, and like every other runtime mutation in this engine
//!   (a broken wall, a carved cave, a footprint) it is **not persisted**,
//!   because `.inf_lvl` is the author's document and this engine has no
//!   save-game container.
//! * An **authored per-entity starting inventory** — "this crate contains three
//!   bandages and that one contains a key", set in the Details panel and
//!   surviving Ctrl+S — *is* a wire field. There is no generic
//!   component-reflection save path in this tree (`props`/`registry` exist for
//!   the Details panel and have no writer), so it would be
//!   `RuntimeEntityGen::inventory: Option<Inventory>` at the record's tail,
//!   scene **v26**, its editor mirror, a frozen `EntityRecordV25`, a committed
//!   downgrade fixture, and `SCENE_PAYLOAD_VERSION` **12** by the envelope's own
//!   doctrine.
//!
//! I6 does not need the second, so I6 does not take it. What content authors
//! instead is the **catalogue** ([`ItemDefs`]) and the **placements**, both
//! through the Blueprint kit, which rides `.inf_act` bytes that a cooked pack
//! and a PIE payload already both carry.
//!
//! # Why the catalogue is not an asset
//!
//! A `.inf_item` asset kind is additive and cheap on its own. What is not cheap
//! is *reaching the player*: an entity would need a `Uuid` field to name one
//! (a scene bump) and the PIE envelope would need a `Vec<(Uuid, Vec<u8>)>` to
//! carry one (a payload bump). Both are refused here and both are priced above.
//! An `items.toml` beside the level is refused for a different reason and it is
//! measured: `input.toml` reaches exactly **one** of the three boot paths (the
//! I5 audit's finding A6), so a catalogue that lived there would be present in a
//! dev run and absent in the build.
//!
//! The Blueprint class is the one authoring surface that reaches all three, and
//! it is the surface `destruct.*` and `voxel.*` already are.

use std::collections::BTreeMap;

use bevy_ecs::prelude::{Component, Resource, With};
use uuid::Uuid;

use crate::components::{GlobalTransform, Guid, MeshRef, Primitive, Transform, Visibility};
use crate::interact::{InteractVerb, Interactable};
use crate::math::Vec3d;
use crate::world::EcsWorld;

/// How many characters an item id may have. A bound, so a hostile catalogue
/// cannot make a `BTreeMap` key out of a megabyte.
pub const MAX_ITEM_ID_LEN: usize = 64;
/// How many entries a catalogue may hold.
pub const MAX_ITEM_DEFS: usize = 4096;
/// How many slots an inventory has unless something says otherwise.
pub const DEFAULT_INVENTORY_SLOTS: usize = 20;
/// The most slots an inventory may have.
pub const MAX_INVENTORY_SLOTS: usize = 240;
/// How many of one thing fits in a slot unless the definition says otherwise.
pub const DEFAULT_STACK_MAX: u32 = 20;
/// How far in front of the character a dropped item lands, metres.
pub const DROP_REACH_M: f64 = 1.1;
/// How far above the character's feet a dropped item lands, metres.
pub const DROP_HEIGHT_M: f64 = 0.4;
/// A dropped item's half extent, metres — the cube a pickup draws as.
pub const PICKUP_HALF_M: f64 = 0.12;

/// **What an item IS** — the name-keyed definition.
///
/// Runtime data, never on the wire. `id` is the key and it is a string for the
/// same reason `CharacterMovement::overlay` is (Ruling 4): a studio must be able
/// to add a fourteenth item without an engine schema bump.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemDef {
    /// The key. Lower-cased and trimmed by [`ItemDefs::insert`].
    pub id: String,
    /// What a prompt calls it.
    pub label: String,
    /// How many fit in one slot.
    pub stack_max: u32,
    /// What one weighs, kg. Carried for a future encumbrance rule and for the
    /// panel's own readout; nothing in I6 refuses on it, and the panel says so
    /// rather than implying a limit that does not exist.
    pub mass_kg: f64,
    /// The weapon this item is, if it is one (I6, the weapons half).
    pub weapon: Option<crate::weapon::WeaponDef>,
}

impl Default for ItemDef {
    fn default() -> Self {
        Self {
            id: String::new(),
            label: String::new(),
            stack_max: DEFAULT_STACK_MAX,
            mass_kg: 1.0,
            weapon: None,
        }
    }
}

impl ItemDef {
    /// Whether this definition can be used. A refusal is a value everywhere it
    /// is asked.
    pub fn is_usable(&self) -> bool {
        !self.id.is_empty()
            && self.id.len() <= MAX_ITEM_ID_LEN
            && self.stack_max > 0
            && self.mass_kg.is_finite()
            && self.mass_kg >= 0.0
    }

    /// Whether this item is a weapon.
    pub fn is_weapon(&self) -> bool {
        self.weapon.is_some()
    }
}

/// **The catalogue** — every item this session knows, by name.
///
/// A resource, for `DeformField`'s reason: it is derived at load, nothing can
/// save it, and no schema moves.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct ItemDefs(pub BTreeMap<String, ItemDef>);

/// The canonical form of an item id: trimmed and lower-cased.
///
/// One function, because the id crosses the catalogue, the inventory, the
/// pickup component and the Blueprint kit — and four spellings of "the same
/// item" is how a level ends up with `Rifle` and `rifle` in two slots.
pub fn canonical_id(id: &str) -> String {
    id.trim().to_ascii_lowercase()
}

impl ItemDefs {
    /// Add or replace a definition. `false` when it is unusable or the
    /// catalogue is full — a refusal, not a panic.
    pub fn insert(&mut self, mut def: ItemDef) -> bool {
        def.id = canonical_id(&def.id);
        if def.label.trim().is_empty() {
            def.label = def.id.clone();
        }
        if !def.is_usable() {
            return false;
        }
        if !self.0.contains_key(&def.id) && self.0.len() >= MAX_ITEM_DEFS {
            return false;
        }
        self.0.insert(def.id.clone(), def);
        true
    }

    /// This item's definition, if the catalogue has it.
    pub fn get(&self, id: &str) -> Option<&ItemDef> {
        self.0.get(&canonical_id(id))
    }

    /// How many definitions there are.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the catalogue is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// **Parse a name-keyed TOML catalogue.**
    ///
    /// ```text
    /// [rifle]
    /// label = "Rifle"
    /// stack_max = 1
    /// mass_kg = 3.6
    /// ```
    ///
    /// The editor's own doctrine, restated where content reaches it: absent is
    /// the default, a malformed document is an **error** with nothing applied,
    /// and every numeric is guarded — a non-finite mass takes the default rather
    /// than becoming a weight nothing can compare.
    ///
    /// Returns how many definitions were taken, which is what a Blueprint node
    /// hands back so an author can see their catalogue arrive.
    pub fn merge_toml(&mut self, text: &str) -> Result<usize, String> {
        let doc: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
        let table = doc
            .as_table()
            .ok_or_else(|| "an item catalogue is a table of named items".to_string())?;
        let mut taken = 0usize;
        for (id, value) in table {
            let t = value
                .as_table()
                .ok_or_else(|| format!("item {id} is not a table"))?;
            let label = t
                .get("label")
                .and_then(|v| v.as_str())
                .unwrap_or(id.as_str())
                .to_string();
            let stack_max = t
                .get("stack_max")
                .and_then(|v| v.as_integer())
                .filter(|n| *n > 0)
                .map(|n| n.min(i64::from(u32::MAX)) as u32)
                .unwrap_or(DEFAULT_STACK_MAX);
            let mass_kg = t
                .get("mass_kg")
                .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|n| n as f64)))
                .filter(|m| m.is_finite() && *m >= 0.0)
                .unwrap_or(1.0);
            let weapon = crate::weapon::WeaponDef::from_toml_table(t)?;
            if self.insert(ItemDef {
                id: id.clone(),
                label,
                stack_max,
                mass_kg,
                weapon,
            }) {
                taken += 1;
            }
        }
        Ok(taken)
    }
}

/// The catalogue, if this world has one.
pub fn item_defs(world: &EcsWorld) -> Option<&ItemDefs> {
    world.world().get_resource::<ItemDefs>()
}

/// The catalogue for writing, created on first use.
pub fn item_defs_mut(world: &mut EcsWorld) -> &mut ItemDefs {
    let w = world.world_mut();
    if !w.contains_resource::<ItemDefs>() {
        w.insert_resource(ItemDefs::default());
    }
    w.get_resource_mut::<ItemDefs>()
        .expect("just inserted")
        .into_inner()
}

/// **One slot's contents.**
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItemStack {
    /// The item's canonical id.
    pub id: String,
    /// How many.
    pub count: u32,
}

/// **What a character is carrying** — a runtime component. See the module
/// header for the persistence accounting.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Inventory {
    /// The slots, in grid order. A `None` slot is empty; the vector's length is
    /// the inventory's size and never changes.
    pub slots: Vec<Option<ItemStack>>,
    /// Which slot is equipped, if any.
    pub equipped: Option<usize>,
    /// How many things this inventory has dropped.
    ///
    /// **A counter, and it is load-bearing**: a dropped item becomes an entity,
    /// an entity needs a `Guid`, and a `Guid` that came from anywhere but the
    /// simulation's own state would make two hosts running the same trace
    /// disagree about what is in their worlds. Folding this into the guid makes
    /// the identity a pure function of what the character has done.
    pub drops: u64,
}

impl Default for Inventory {
    fn default() -> Self {
        Self::with_slots(DEFAULT_INVENTORY_SLOTS)
    }
}

impl Inventory {
    /// An empty inventory of `n` slots, bounded by [`MAX_INVENTORY_SLOTS`].
    pub fn with_slots(n: usize) -> Self {
        Self {
            slots: vec![None; n.clamp(1, MAX_INVENTORY_SLOTS)],
            equipped: None,
            drops: 0,
        }
    }

    /// How many slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether every slot is empty.
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.is_none())
    }

    /// How many of `id` are carried, across every slot.
    pub fn count_of(&self, id: &str) -> u32 {
        let id = canonical_id(id);
        self.slots
            .iter()
            .flatten()
            .filter(|s| s.id == id)
            .fold(0u32, |a, s| a.saturating_add(s.count))
    }

    /// The equipped item's id, if anything is equipped and the slot is not
    /// empty.
    pub fn equipped_id(&self) -> Option<&str> {
        let i = self.equipped?;
        self.slots.get(i)?.as_ref().map(|s| s.id.as_str())
    }

    /// **Put `count` of `id` in**, stacking into partial slots first and then
    /// into empty ones.
    ///
    /// Returns how many did **not** fit — a refusal expressed as a number, so a
    /// caller can put the remainder back on the floor rather than losing it.
    /// An unknown id does not fit at all: an inventory that accepted items the
    /// catalogue has never heard of would carry things nothing could describe.
    pub fn add(&mut self, defs: &ItemDefs, id: &str, count: u32) -> u32 {
        let id = canonical_id(id);
        let Some(def) = defs.get(&id) else {
            return count;
        };
        let stack_max = def.stack_max.max(1);
        let mut left = count;
        // Partial stacks first, in slot order.
        for slot in self.slots.iter_mut() {
            if left == 0 {
                break;
            }
            if let Some(s) = slot {
                if s.id == id && s.count < stack_max {
                    let room = stack_max - s.count;
                    let take = room.min(left);
                    s.count += take;
                    left -= take;
                }
            }
        }
        // Then empty ones.
        for slot in self.slots.iter_mut() {
            if left == 0 {
                break;
            }
            if slot.is_none() {
                let take = stack_max.min(left);
                *slot = Some(ItemStack {
                    id: id.clone(),
                    count: take,
                });
                left -= take;
            }
        }
        left
    }

    /// **Take up to `count` out of slot `index`.** Returns how many came out.
    ///
    /// Emptying the equipped slot un-equips it, because an equipped slot with
    /// nothing in it is a weapon a character is holding that does not exist.
    pub fn take_at(&mut self, index: usize, count: u32) -> Option<ItemStack> {
        let slot = self.slots.get_mut(index)?;
        let s = slot.as_mut()?;
        let take = count.min(s.count);
        if take == 0 {
            return None;
        }
        let id = s.id.clone();
        s.count -= take;
        if s.count == 0 {
            *slot = None;
            if self.equipped == Some(index) {
                self.equipped = None;
            }
        }
        Some(ItemStack { id, count: take })
    }

    /// **Move a slot's contents onto another slot**, merging when they are the
    /// same item and swapping when they are not.
    ///
    /// `false` for an index that is not there or a move onto itself — a refusal,
    /// and the equipped index follows whatever moved so an equipped weapon stays
    /// equipped when a player tidies their grid.
    pub fn move_slot(&mut self, defs: &ItemDefs, from: usize, to: usize) -> bool {
        if from == to || from >= self.slots.len() || to >= self.slots.len() {
            return false;
        }
        if self.slots[from].is_none() {
            return false;
        }
        let same = match (&self.slots[from], &self.slots[to]) {
            (Some(a), Some(b)) => a.id == b.id,
            _ => false,
        };
        if same {
            let stack_max = self.slots[from]
                .as_ref()
                .and_then(|s| defs.get(&s.id))
                .map(|d| d.stack_max.max(1))
                .unwrap_or(DEFAULT_STACK_MAX);
            let moving = self.slots[from].as_ref().map(|s| s.count).unwrap_or(0);
            let there = self.slots[to].as_ref().map(|s| s.count).unwrap_or(0);
            let take = moving.min(stack_max.saturating_sub(there));
            if take == 0 {
                return false;
            }
            if let Some(s) = self.slots[to].as_mut() {
                s.count += take;
            }
            let emptied = {
                let s = self.slots[from].as_mut().expect("checked above");
                s.count -= take;
                s.count == 0
            };
            if emptied {
                self.slots[from] = None;
                if self.equipped == Some(from) {
                    self.equipped = None;
                }
            }
            return true;
        }
        self.slots.swap(from, to);
        // The equipped index names a SLOT, so a swap has to carry it or the
        // character silently equips whatever landed under the old number.
        self.equipped = match self.equipped {
            Some(i) if i == from => Some(to),
            Some(i) if i == to => Some(from),
            other => other,
        };
        true
    }

    /// **Equip slot `index`.** `false` for an empty or absent slot.
    pub fn equip(&mut self, index: usize) -> bool {
        if self.slots.get(index).and_then(|s| s.as_ref()).is_none() {
            return false;
        }
        self.equipped = Some(index);
        true
    }

    /// Un-equip whatever is equipped.
    pub fn unequip(&mut self) {
        self.equipped = None;
    }

    /// **The scroll wheel**: the next equippable slot in `dir`'s direction.
    ///
    /// `filter` decides what counts as equippable — the weapons half passes
    /// "is it a weapon", and a caller that wants any item passes everything.
    /// Wraps, skips empty slots, and answers `None` when there is nothing to
    /// equip, which is what stops a wheel with an empty inventory from spinning
    /// through a hundred slots per notch.
    ///
    /// `dir` is a **sign**, because the wheel reaches this engine as a rate and
    /// not as a notch count (the I5 remainder): the consumer reads the sign,
    /// which is exactly what this takes.
    pub fn cycle_equipped(
        &mut self,
        defs: &ItemDefs,
        dir: i32,
        filter: impl Fn(&ItemDef) -> bool,
    ) -> Option<usize> {
        if dir == 0 || self.slots.is_empty() {
            return None;
        }
        let n = self.slots.len();
        let step: usize = if dir > 0 { 1 } else { n - 1 };
        let start = self.equipped.unwrap_or(0);
        let mut i = start;
        for _ in 0..n {
            i = (i + step) % n;
            let ok = self
                .slots
                .get(i)
                .and_then(|s| s.as_ref())
                .and_then(|s| defs.get(&s.id))
                .is_some_and(&filter);
            if ok {
                self.equipped = Some(i);
                return Some(i);
            }
        }
        None
    }

    /// The bytes this inventory contributes to the sim trace.
    fn state_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.slots.len() as u32).to_le_bytes());
        for slot in &self.slots {
            match slot {
                Some(s) => {
                    out.push(1);
                    out.extend_from_slice(&(s.id.len() as u32).to_le_bytes());
                    out.extend_from_slice(s.id.as_bytes());
                    out.extend_from_slice(&s.count.to_le_bytes());
                }
                None => out.push(0),
            }
        }
        match self.equipped {
            Some(i) => {
                out.push(1);
                out.extend_from_slice(&(i as u32).to_le_bytes());
            }
            None => out.push(0),
        }
        out.extend_from_slice(&self.drops.to_le_bytes());
    }
}

/// **An entity is an item lying on the ground** — a runtime component.
///
/// Paired with an [`Interactable`] carrying [`InteractVerb::PickUp`], which is
/// what puts it in front of the one interaction rule. The two are inserted
/// together by [`spawn_pickup`] so a pickup cannot exist that the E key cannot
/// see.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct ItemPickup {
    /// The item's canonical id.
    pub id: String,
    /// How many are lying there.
    pub count: u32,
}

/// The salt that carves dropped items' GUID space out of the scene's own.
const DROPPED_ITEM_SALT: u128 = 0x6006_0200_4954_454d_4452_4f50_5045_4421;

/// **The identity of the `n`-th thing `owner` has dropped.**
///
/// A pure function of sim state — the character's own guid and its own drop
/// count — so two hosts running one trace put the same entity in the same place
/// with the same name. The `inf_physics::d3::pcg_structure_guid` rule with its
/// own salt; a 128-bit mix rather than a XOR, because XORing a small counter
/// into a guid makes two characters whose ids differ in the low bits alias each
/// other's drops.
pub fn dropped_item_guid(owner: Uuid, n: u64) -> Uuid {
    let mut x = owner.as_u128() ^ DROPPED_ITEM_SALT;
    x ^= u128::from(n).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5cec_c5c3);
    x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

/// **Put an item on the ground**, as an entity the E key can see.
///
/// Returns the guid it used, or `None` when the id is not in the catalogue —
/// which is the same refusal [`Inventory::add`] makes and for the same reason.
pub fn spawn_pickup(
    world: &mut EcsWorld,
    guid: Uuid,
    id: &str,
    count: u32,
    at: crate::math::Vec3d,
) -> Option<Uuid> {
    let id = canonical_id(id);
    let label = item_defs(world)?.get(&id)?.label.clone();
    if count == 0 {
        return None;
    }
    let entity = world.spawn_with_guid(guid, &label, None);
    let mut t = Transform::IDENTITY;
    t.translation = at;
    t.scale = Vec3d::new(
        PICKUP_HALF_M * 2.0,
        PICKUP_HALF_M * 2.0,
        PICKUP_HALF_M * 2.0,
    );
    world.world_mut().entity_mut(entity).insert((
        t,
        // A cube, because the engine has no item geometry and a pickup nobody
        // can see is a pickup nobody picks up. A definition that names a mesh
        // asset is the obvious next field and is not in I6.
        MeshRef {
            primitive: Primitive::Cube,
            asset: None,
        },
        Visibility::default(),
        ItemPickup {
            id: id.clone(),
            count,
        },
        Interactable {
            verb: InteractVerb::PickUp,
            label,
            // **A dropped thing is picked up with a hand** (SK1c). `prop` is the
            // widest grip a rig's catalogue carries -- a 9 cm ball, every finger
            // on it -- which is right for a cube-shaped pickup and is the same
            // affordance the grip gate measures a thrown prop with.
            grip: Some(inf_anim::GRIP_PROP.to_string()),
            ..Default::default()
        },
    ));
    world.mark_dirty();
    Some(guid)
}

/// The salt that carves **authored** pickups' GUID space out of the scene's own.
const AUTHORED_PICKUP_SALT: u128 = 0x6006_0400_4954_454d_5350_4157_4e21_2121;

/// **The identity of a pickup a Blueprint spawned**, folded from its id and its
/// place.
///
/// A pure function of what the author asked for, and **not** of a counter: a
/// spawn keyed on how many times a graph had run would put two hosts' worlds out
/// of step the first time one of them ran a handler twice. Two pickups of the
/// same item at the same point are the same entity, which is the right answer —
/// an author who wants two puts them in two places.
pub fn authored_pickup_guid(id: &str, at: Vec3d) -> Uuid {
    let mut x = AUTHORED_PICKUP_SALT;
    for b in canonical_id(id).as_bytes() {
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

/// **Put items straight into a character's bag**, creating one if it has none.
///
/// The Blueprint kit's `item.give`, and the one door that door goes through.
/// Returns how many did **not** fit, so a refusal is a number a graph can read.
///
/// Creating the inventory here rather than requiring one first is deliberate:
/// an author who says "give this actor a rifle" has said everything needed, and
/// a second node called *Give Inventory* would be a step nobody would remember
/// and a failure nothing would explain.
pub fn give(world: &mut EcsWorld, character: Uuid, id: &str, count: u32) -> u32 {
    let Some(entity) = world.entity_of(character) else {
        return count;
    };
    if world.world().get::<Inventory>(entity).is_none() {
        world
            .world_mut()
            .entity_mut(entity)
            .insert(Inventory::default());
    }
    let defs = item_defs(world).cloned().unwrap_or_default();
    let w = world.world_mut();
    match w.get_mut::<Inventory>(entity) {
        Some(mut inv) => inv.add(&defs, id, count),
        None => count,
    }
}

/// What a pick-up did. **A refusal is a value.**
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickUpVerdict {
    /// It all went in and the pickup is gone.
    Taken(u32),
    /// Some went in; what is left is still on the floor.
    Partial(u32),
    /// The character has nowhere to put it.
    Full,
    /// The character has no inventory at all.
    NoInventory,
    /// That entity is not a pickup.
    NotAnItem,
}

/// **The E key on an item** — the consumer the one interaction site calls when
/// the hit it resolved is a pick-up.
///
/// A partial pick-up leaves the remainder on the floor rather than deleting it,
/// which is the only outcome that does not lose a player's things.
pub fn pick_up(world: &mut EcsWorld, character: Uuid, target: Uuid) -> PickUpVerdict {
    let Some(pickup_entity) = world.entity_of(target) else {
        return PickUpVerdict::NotAnItem;
    };
    let Some(pickup) = world.world().get::<ItemPickup>(pickup_entity).cloned() else {
        return PickUpVerdict::NotAnItem;
    };
    let Some(character_entity) = world.entity_of(character) else {
        return PickUpVerdict::NoInventory;
    };
    if world.world().get::<Inventory>(character_entity).is_none() {
        return PickUpVerdict::NoInventory;
    }
    let defs = item_defs(world).cloned().unwrap_or_default();
    let left = {
        let w = world.world_mut();
        let mut inv = w
            .get_mut::<Inventory>(character_entity)
            .expect("checked above");
        inv.add(&defs, &pickup.id, pickup.count)
    };
    if left == pickup.count {
        return PickUpVerdict::Full;
    }
    if left == 0 {
        world.despawn(pickup_entity);
        return PickUpVerdict::Taken(pickup.count);
    }
    if let Some(mut p) = world.world_mut().get_mut::<ItemPickup>(pickup_entity) {
        p.count = left;
    }
    PickUpVerdict::Partial(pickup.count - left)
}

/// **Drop slot `index`** — the inventory panel's own verb.
///
/// Puts the stack in front of the character, one metre out and knee high, as an
/// entity the E key can pick up again. Returns the new entity's guid.
pub fn drop_slot(world: &mut EcsWorld, character: Uuid, index: usize, count: u32) -> Option<Uuid> {
    let entity = world.entity_of(character)?;
    let at = {
        let w = world.world();
        let t = w
            .get::<GlobalTransform>(entity)
            .map(|g| g.translation())
            .or_else(|| w.get::<Transform>(entity).map(|t| t.translation.to_dvec3()))?;
        let yaw = w
            .get::<Transform>(entity)
            .map(|t| t.rotation.y)
            .unwrap_or(0.0);
        let r = yaw.to_radians();
        let fwd = glam::DVec3::new(inf_math::psin64(r), 0.0, inf_math::pcos64(r));
        t + fwd * DROP_REACH_M + glam::DVec3::Y * DROP_HEIGHT_M
    };
    let (stack, n) = {
        let w = world.world_mut();
        let mut inv = w.get_mut::<Inventory>(entity)?;
        let stack = inv.take_at(index, count)?;
        inv.drops = inv.drops.saturating_add(1);
        (stack, inv.drops)
    };
    let guid = dropped_item_guid(character, n);
    spawn_pickup(world, guid, &stack.id, stack.count, Vec3d::from_dvec3(at))
}

/// **The trace bytes for inventories**, in `Guid` order, appended to
/// `state_bytes` by both hosts.
///
/// Empty on a level with no inventory, which is what keeps every pre-I6 trace
/// byte-identical.
pub fn item_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let w = world.world();
    let Some(mut q) = w.try_query_filtered::<(&Guid, &Inventory), With<Inventory>>() else {
        return Vec::new();
    };
    let mut rows: Vec<(Uuid, &Inventory)> = q.iter(w).map(|(g, i)| (g.0, i)).collect();
    if rows.is_empty() {
        return Vec::new();
    }
    rows.sort_by_key(|(g, _)| *g);
    let mut out = Vec::new();
    for (guid, inv) in rows {
        out.extend_from_slice(guid.as_bytes());
        inv.state_bytes(&mut out);
    }
    out
}

/// Give `character` an inventory of `slots` slots, replacing any it had.
///
/// The one insertion door, so a character cannot end up with an inventory the
/// panel can see and the pick-up cannot, or the other way round.
pub fn give_inventory(world: &mut EcsWorld, character: Uuid, slots: usize) -> bool {
    let Some(entity) = world.entity_of(character) else {
        return false;
    };
    world
        .world_mut()
        .entity_mut(entity)
        .insert(Inventory::with_slots(slots));
    true
}

/// Read one character's inventory.
pub fn inventory_of(world: &EcsWorld, character: Uuid) -> Option<&Inventory> {
    let entity = world.entity_of(character)?;
    world.world().get::<Inventory>(entity)
}

/// Every entity carrying an inventory, in `Guid` order. `O(inventories)`.
pub fn inventory_holders(world: &EcsWorld) -> Vec<Uuid> {
    let w = world.world();
    let Some(mut q) = w.try_query_filtered::<&Guid, With<Inventory>>() else {
        return Vec::new();
    };
    let mut out: Vec<Uuid> = q.iter(w).map(|g| g.0).collect();
    out.sort_unstable();
    out
}

/// A name for the pickup entity's label, kept out of `spawn_pickup`'s body so a
/// test can name the same thing. Unused elsewhere on purpose: `Name` is what the
/// outliner shows and the label is what the prompt says.
pub fn pickup_name(def: &ItemDef) -> &str {
    &def.label
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defs() -> ItemDefs {
        let mut d = ItemDefs::default();
        assert!(d.insert(ItemDef {
            id: "Bandage".into(),
            label: "Bandage".into(),
            stack_max: 5,
            mass_kg: 0.1,
            ..Default::default()
        }));
        assert!(d.insert(ItemDef {
            id: "rifle".into(),
            label: "Rifle".into(),
            stack_max: 1,
            mass_kg: 3.6,
            ..Default::default()
        }));
        d
    }

    /// **An id is one thing however it is spelled.**
    #[test]
    fn an_item_id_is_canonical_wherever_it_is_written() {
        let d = defs();
        assert!(d.get("bandage").is_some());
        assert!(d.get("BANDAGE").is_some());
        assert!(d.get("  Bandage  ").is_some());
        assert_eq!(d.get("bandage").expect("there").id, "bandage");
        assert!(d.get("nothing").is_none());
        // …and an unusable definition is refused rather than stored.
        let mut d = d;
        assert!(!d.insert(ItemDef {
            id: "  ".into(),
            ..Default::default()
        }));
        assert!(!d.insert(ItemDef {
            id: "x".into(),
            stack_max: 0,
            ..Default::default()
        }));
        assert!(!d.insert(ItemDef {
            id: "y".into(),
            mass_kg: f64::NAN,
            ..Default::default()
        }));
        assert_eq!(d.len(), 2);
    }

    /// **Stacking**, and the remainder a full inventory hands back.
    #[test]
    fn adding_fills_partial_stacks_first_and_says_what_did_not_fit() {
        let d = defs();
        let mut inv = Inventory::with_slots(2);
        assert_eq!(inv.add(&d, "bandage", 3), 0);
        assert_eq!(inv.count_of("bandage"), 3);
        assert_eq!(inv.slots.iter().flatten().count(), 1, "it made two stacks");
        // Two more top the first stack up to five and start a second.
        assert_eq!(inv.add(&d, "bandage", 4), 0);
        assert_eq!(inv.count_of("bandage"), 7);
        assert_eq!(inv.slots.iter().flatten().count(), 2);
        // Now it is full: five plus two of five, so three fit and the rest does
        // not.
        assert_eq!(inv.add(&d, "bandage", 10), 7);
        assert_eq!(inv.count_of("bandage"), 10);
        // An unknown item does not fit at all, however much room there is.
        let mut roomy = Inventory::with_slots(9);
        assert_eq!(roomy.add(&d, "grenade", 1), 1);
        assert!(roomy.is_empty());
    }

    /// **Taking**, and the equipped slot that empties under the character.
    #[test]
    fn emptying_the_equipped_slot_unequips_it() {
        let d = defs();
        let mut inv = Inventory::with_slots(4);
        inv.add(&d, "rifle", 1);
        assert!(inv.equip(0));
        assert_eq!(inv.equipped_id(), Some("rifle"));
        let got = inv.take_at(0, 1).expect("a rifle");
        assert_eq!(got.count, 1);
        assert_eq!(inv.equipped, None, "the character is holding nothing");
        assert_eq!(inv.equipped_id(), None);
        // Equipping an empty slot is refused.
        assert!(!inv.equip(0));
        assert!(!inv.equip(99));
        assert!(inv.take_at(0, 1).is_none());
        assert!(inv.take_at(99, 1).is_none());
    }

    /// **A move carries the equipped index**, which is the defect a swap makes
    /// if it does not: the character silently equips whatever landed under the
    /// old number.
    #[test]
    fn moving_a_slot_carries_what_is_equipped_and_merges_like_items() {
        let d = defs();
        let mut inv = Inventory::with_slots(4);
        inv.add(&d, "rifle", 1);
        inv.slots[2] = Some(ItemStack {
            id: "bandage".into(),
            count: 2,
        });
        assert!(inv.equip(0));
        assert!(inv.move_slot(&d, 0, 3));
        assert_eq!(inv.equipped, Some(3), "the equipped index did not follow");
        assert_eq!(inv.equipped_id(), Some("rifle"));
        // Merging: two bandages onto three makes five in one slot.
        inv.slots[0] = Some(ItemStack {
            id: "bandage".into(),
            count: 3,
        });
        assert!(inv.move_slot(&d, 2, 0));
        assert_eq!(inv.count_of("bandage"), 5);
        assert!(inv.slots[2].is_none());
        // A merge that has no room is a refusal, not a silent loss.
        inv.slots[1] = Some(ItemStack {
            id: "bandage".into(),
            count: 4,
        });
        assert!(!inv.move_slot(&d, 1, 0), "it merged past the stack max");
        assert_eq!(inv.count_of("bandage"), 9);
        // Degenerate moves are refusals.
        assert!(!inv.move_slot(&d, 0, 0));
        assert!(!inv.move_slot(&d, 2, 0), "an empty slot moved");
        assert!(!inv.move_slot(&d, 99, 0));
    }

    /// **The wheel cycles, wraps and gives up.**
    #[test]
    fn the_wheel_wraps_over_what_it_is_allowed_to_equip() {
        let d = defs();
        let mut inv = Inventory::with_slots(5);
        inv.slots[1] = Some(ItemStack {
            id: "rifle".into(),
            count: 1,
        });
        inv.slots[3] = Some(ItemStack {
            id: "bandage".into(),
            count: 1,
        });
        assert_eq!(inv.cycle_equipped(&d, 1, |_| true), Some(1));
        assert_eq!(inv.cycle_equipped(&d, 1, |_| true), Some(3));
        assert_eq!(
            inv.cycle_equipped(&d, 1, |_| true),
            Some(1),
            "it did not wrap"
        );
        assert_eq!(
            inv.cycle_equipped(&d, -1, |_| true),
            Some(3),
            "it did not go back"
        );
        // A filter that admits nothing answers None rather than spinning.
        assert_eq!(inv.cycle_equipped(&d, 1, |_| false), None);
        assert_eq!(inv.equipped, Some(3), "a refused cycle moved the equip");
        // A zero delta is not a notch.
        assert_eq!(inv.cycle_equipped(&d, 0, |_| true), None);
        // An empty inventory answers None.
        let mut empty = Inventory::with_slots(4);
        assert_eq!(empty.cycle_equipped(&d, 1, |_| true), None);
    }

    /// **The catalogue parses name-keyed TOML**, guards its numerics and
    /// refuses a malformed document whole.
    #[test]
    fn a_toml_catalogue_is_name_keyed_and_its_numbers_are_guarded() {
        let mut d = ItemDefs::default();
        let taken = d
            .merge_toml(
                r#"
[rifle]
label = "Rifle"
stack_max = 1
mass_kg = 3.6

[bandage]
stack_max = 5
"#,
            )
            .expect("a catalogue");
        assert_eq!(taken, 2);
        assert_eq!(d.get("rifle").expect("rifle").label, "Rifle");
        assert_eq!(d.get("rifle").expect("rifle").stack_max, 1);
        // An absent label defaults to the id, an absent number to the default.
        assert_eq!(d.get("bandage").expect("bandage").label, "bandage");
        assert!((d.get("bandage").expect("bandage").mass_kg - 1.0).abs() < 1e-12);
        // A hostile numeric takes the default rather than becoming a weight
        // nothing can compare — the settings doctrine, restated at content.
        let mut e = ItemDefs::default();
        e.merge_toml("[x]\nmass_kg = nan\nstack_max = -3\n")
            .expect("a catalogue");
        assert!(e.get("x").expect("x").mass_kg.is_finite());
        assert_eq!(e.get("x").expect("x").stack_max, DEFAULT_STACK_MAX);
        // A malformed document is an error with NOTHING applied.
        let mut f = ItemDefs::default();
        assert!(f.merge_toml("[x").is_err());
        assert!(f.is_empty());
        assert!(f.merge_toml("x = 1").is_err());
        assert!(f.is_empty());
    }

    /// **A dropped item's identity is a pure function of sim state**, which is
    /// what lets two hosts running one trace agree about what is in the world.
    #[test]
    fn a_drop_is_named_by_who_dropped_it_and_how_many_they_have_dropped() {
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        assert_ne!(dropped_item_guid(a, 0), dropped_item_guid(a, 1));
        assert_ne!(dropped_item_guid(a, 0), dropped_item_guid(b, 0));
        // …and the low-bit aliasing a XOR would produce does not happen.
        assert_ne!(dropped_item_guid(a, 1), dropped_item_guid(b, 0));
        assert_eq!(dropped_item_guid(a, 7), dropped_item_guid(a, 7));
    }

    /// **The round trip**: a pickup on the floor, into the inventory, back out,
    /// and pickable again — asserted against the WORLD each time.
    #[test]
    fn an_item_goes_from_the_floor_into_a_bag_and_back_onto_the_floor() {
        let mut w = EcsWorld::new();
        *item_defs_mut(&mut w) = defs();
        let hero = Uuid::from_u128(100);
        w.spawn_with_guid(hero, "Hero", None);
        assert!(give_inventory(&mut w, hero, 6));
        let on_floor = Uuid::from_u128(200);
        assert_eq!(
            spawn_pickup(&mut w, on_floor, "rifle", 1, Vec3d::new(1.0, 0.0, 0.0)),
            Some(on_floor)
        );
        // It is in the world, and the E key can see it: it carries the PickUp
        // verb, so it is a candidate for the one interaction rule.
        let e = w.entity_of(on_floor).expect("the pickup exists");
        assert_eq!(
            w.world()
                .get::<Interactable>(e)
                .expect("a pickup is interactable")
                .verb,
            InteractVerb::PickUp
        );
        assert_eq!(pick_up(&mut w, hero, on_floor), PickUpVerdict::Taken(1));
        assert!(w.entity_of(on_floor).is_none(), "the floor still has it");
        assert_eq!(inventory_of(&w, hero).expect("a bag").count_of("rifle"), 1);
        // Back out: a new entity, at a deterministic guid, in front of the hero.
        let dropped = drop_slot(&mut w, hero, 0, 1).expect("it dropped");
        assert_eq!(dropped, dropped_item_guid(hero, 1));
        assert_eq!(inventory_of(&w, hero).expect("a bag").count_of("rifle"), 0);
        let de = w.entity_of(dropped).expect("the drop exists");
        let at = w
            .world()
            .get::<Transform>(de)
            .expect("a transform")
            .translation;
        println!("the rifle came back out at {at:?}");
        assert!(
            (at.z - DROP_REACH_M).abs() < 1e-6,
            "it did not land in front"
        );
        assert!((at.y - DROP_HEIGHT_M).abs() < 1e-12);
        // …and it can be picked up again.
        assert_eq!(pick_up(&mut w, hero, dropped), PickUpVerdict::Taken(1));
        assert_eq!(inventory_of(&w, hero).expect("a bag").count_of("rifle"), 1);
    }

    /// Every refusal on the pick-up path is a value.
    #[test]
    fn every_pick_up_refusal_is_a_value() {
        let mut w = EcsWorld::new();
        *item_defs_mut(&mut w) = defs();
        let hero = Uuid::from_u128(100);
        w.spawn_with_guid(hero, "Hero", None);
        let nothing = Uuid::from_u128(999);
        assert_eq!(pick_up(&mut w, hero, nothing), PickUpVerdict::NotAnItem);
        let item = Uuid::from_u128(200);
        spawn_pickup(&mut w, item, "rifle", 1, Vec3d::ZERO);
        // No inventory yet.
        assert_eq!(pick_up(&mut w, hero, item), PickUpVerdict::NoInventory);
        // A one-slot bag with a rifle already in it is full for a second one.
        assert!(give_inventory(&mut w, hero, 1));
        assert_eq!(pick_up(&mut w, hero, item), PickUpVerdict::Taken(1));
        let second = Uuid::from_u128(201);
        spawn_pickup(&mut w, second, "rifle", 1, Vec3d::ZERO);
        assert_eq!(pick_up(&mut w, hero, second), PickUpVerdict::Full);
        assert!(w.entity_of(second).is_some(), "a refused pick-up ate it");
        // A partial pick-up leaves the remainder on the floor.
        let mut w2 = EcsWorld::new();
        *item_defs_mut(&mut w2) = defs();
        w2.spawn_with_guid(hero, "Hero", None);
        give_inventory(&mut w2, hero, 1);
        let pile = Uuid::from_u128(300);
        spawn_pickup(&mut w2, pile, "bandage", 8, Vec3d::ZERO);
        assert_eq!(pick_up(&mut w2, hero, pile), PickUpVerdict::Partial(5));
        let pe = w2.entity_of(pile).expect("the remainder is still there");
        assert_eq!(w2.world().get::<ItemPickup>(pe).expect("a pickup").count, 3);
        // …and a pickup of nothing is refused at the spawn.
        assert!(spawn_pickup(&mut w2, Uuid::from_u128(400), "rifle", 0, Vec3d::ZERO).is_none());
        assert!(spawn_pickup(&mut w2, Uuid::from_u128(401), "ghost", 1, Vec3d::ZERO).is_none());
    }

    /// **An untouched world costs the trace nothing**, and a bag's bytes move
    /// when the bag does.
    #[test]
    fn the_item_trace_is_empty_without_inventories_and_moves_with_them() {
        let mut w = EcsWorld::new();
        assert!(item_state_bytes(&w).is_empty());
        *item_defs_mut(&mut w) = defs();
        assert!(item_state_bytes(&w).is_empty(), "a catalogue is not state");
        let hero = Uuid::from_u128(100);
        w.spawn_with_guid(hero, "Hero", None);
        give_inventory(&mut w, hero, 3);
        let empty = item_state_bytes(&w);
        assert!(!empty.is_empty());
        let e = w.entity_of(hero).expect("the hero");
        let defs = defs();
        w.world_mut()
            .get_mut::<Inventory>(e)
            .expect("a bag")
            .add(&defs, "rifle", 1);
        let full = item_state_bytes(&w);
        assert_ne!(empty, full, "the trace did not see the rifle");
        assert!(
            item_state_bytes(&w) == full,
            "the trace is not a function of state"
        );
    }
}
