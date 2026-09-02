//! **Who a building holds** (island wave NPC1d) — the population a
//! [`BuildingPlan`] implies, derived from its own rooms.
//!
//! A settlement generator that plans bedrooms and offices has already decided
//! how many people live and work in a town; nobody had ever asked it. This
//! module asks, and the answer is a pure function of the plan — which is a pure
//! function of the block's archetype, footprint and seed — so a level's
//! population is a property of its *design* rather than of a spawner somebody
//! placed.
//!
//! # The derivation table
//!
//! | room | role | rule | why that number |
//! |---|---|---|---|
//! | [`RoomType::Bedroom`] | [`SlotRole::Home`] | **one a room** | a bedroom is a person's place to sleep, whatever its size |
//! | [`RoomType::Guest`] | [`SlotRole::Home`] | **one a room** | a hotel room holds a guest, and a guest sleeps there |
//! | [`RoomType::Office`] | [`SlotRole::Work`] | one per [`OFFICE_M2_PER_WORKER`] m² | a desk plus its share of the circulation it does not have |
//! | [`RoomType::Workshop`] | [`SlotRole::Work`] | one per [`WORKSHOP_M2_PER_WORKER`] m² | a bench is wider than a desk |
//! | [`RoomType::Retail`] | [`SlotRole::Work`] | one per [`RETAIL_M2_PER_WORKER`] m² | a shop floor is mostly for its customers |
//! | [`RoomType::Retail`] | [`SlotRole::Errand`] | **one a room** | somewhere to go that is neither home nor work |
//! | [`RoomType::BarRoom`] | [`SlotRole::Work`] | one per [`BAR_M2_PER_KEEPER`] m² | a bar room is mostly in front of the counter |
//! | [`RoomType::Stage`] | [`SlotRole::Work`] | **one a room** | one act on a stage, however big it is |
//! | [`RoomType::BarRoom`], [`RoomType::DanceFloor`] | [`SlotRole::Errand`] | **one a room** | a venue is somewhere the town goes |
//!
//! The other nine room types hold nobody. A corridor, a stair, a lobby, a
//! service riser, a living room, a kitchen, a bath, a store room and a dance
//! floor's *work* count are all places a person passes through rather than
//! places a person *is* at an hour of the day, and this wave's schedule is
//! about hours. (A dance floor is still an errand destination — nobody works
//! one and everybody visits one, which is the case the two arms of
//! [`slots_of`] disagreeing would lose entirely.)
//!
//! # What the table means per archetype, and why it is not written per archetype
//!
//! A `House` plan draws bedrooms on its upper floors, so a house holds a
//! household; an `Apartment` draws them on every dwelling floor, so it holds one
//! per dwelling; an `Office` or a `Shop` draws no bedroom at all and holds only
//! workers, by area. Those three sentences are *consequences* of the table above
//! rather than entries in it — the palette decides which rooms a building gets
//! and this module decides what a room is worth, and neither has to know the
//! other's business. A ninth archetype needs no line here.
//!
//! # A slot is a place, not a person
//!
//! [`PcgSlot`] names a room and a role. Turning slots into agents — pairing a
//! home with a work, giving each a schedule and a route — is
//! `inf_ecs::society`'s, because it needs the whole *level* and this crate can
//! only see one building. That split is why a slot carries its room index rather
//! than a nav node id: the id needs a per-building salt only a level can mint.
//!
//! # Portable math
//!
//! An area is `(max − min).x · (max − min).z` and a count is a `ceil` of a
//! division. No trigonometry, no transcendental, nothing that varies between
//! two implementations of libm — which matters because a slot's position lands
//! on an NPC's `Transform` and therefore in the replay trace.

use super::{BuildingPlan, RoomType};
use glam::DVec3;
use uuid::Uuid;

/// **The nav-namespace salt of one building** — which building this is, in a
/// level's whole network (NPC1d).
///
/// `(volume, ordinal)` is the level's own name for a building: a `PcgVolume`'s
/// `Guid` and the position `plans_of` returned the plan in. Both halves are
/// needed — one volume grows many buildings and one level holds many volumes —
/// and neither is dense across a level, so the pair is *hashed* into the
/// thirty-nine-bit field rather than counted into it. A counter would have to be
/// assigned by something that has seen the whole level at once, and in a
/// streaming world nothing has.
///
/// A volume with no `Guid` (`None`) salts on the ordinal alone, which is right
/// for the one-volume case a unit fixture is.
///
/// Collisions are the hazard this width exists for: two buildings sharing a salt
/// weld a bedroom to a bedroom, silently. At 2³⁹ over the island's own thousand
/// buildings that is about one time in a million, and
/// `inf_ecs::society` counts them anyway.
pub fn building_salt(volume: Option<Uuid>, ordinal: u32) -> u64 {
    let (lo, hi) = match volume {
        Some(v) => {
            let b = v.as_u128();
            (b as u64, (b >> 64) as u64)
        }
        None => (0, 0),
    };
    crate::hash::Hash64::new(lo)
        .mix_u64(hi)
        .mix_u64(u64::from(ordinal))
        .finish()
        & super::NAV_SALT_MASK
}

/// Square metres of office floor a worker takes.
///
/// Twelve is a desk (about 4 m²) plus its share of the meeting rooms, risers and
/// circulation the partition already puts in *separate* rooms — so it is
/// deliberately generous against the usual 8–10 m² per-desk figure, because
/// those rooms are counted here as `Meeting` and `Service` and hold nobody.
pub const OFFICE_M2_PER_WORKER: f64 = 12.0;

/// Square metres of workshop floor a worker takes. A bench, its machine and the
/// aisle to walk round it.
pub const WORKSHOP_M2_PER_WORKER: f64 = 20.0;

/// Square metres of shop floor a worker takes.
///
/// The largest of the three, and for the reason a shop exists: most of a retail
/// room is standing room for the people who do **not** work there. Those arrive
/// as [`SlotRole::Errand`] visitors instead.
pub const RETAIL_M2_PER_WORKER: f64 = 30.0;

/// Square metres of bar-room floor a keeper takes (wave VEN1a).
///
/// The most generous of the four, and deliberately: the floor area of a bar
/// room is overwhelmingly *in front of* the counter, and one keeper serves a
/// long run of it. At `RETAIL_M2_PER_WORKER` a 120 m2 bar would be staffed by
/// four, which is a shift and not a bar.
pub const BAR_M2_PER_KEEPER: f64 = 45.0;

/// The most people one room may hold, whatever its area.
///
/// A ceiling rather than a hope: `plan` clamps a building at
/// [`MAX_FLOORS`](super::plan::MAX_FLOORS) floors and a partition can hand back
/// one enormous undivided room, and `area / 12` on a warehouse-sized office is a
/// crowd nobody authored. Sixty-four is far past any room this generator draws
/// (measured: the largest office room on the island's own zone library is under
/// 200 m², i.e. 17 workers) and small enough that a degenerate plan cannot mint
/// a thousand agents from one rectangle.
pub const MAX_SLOTS_PER_ROOM: usize = 64;

/// **What a room offers a person** — the vocabulary a schedule is written in.
///
/// Three roles rather than fourteen room types, because a schedule asks "where
/// does this agent sleep, work and go" and not "what kind of room is this". The
/// mapping from the one to the other is the table in this module's docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SlotRole {
    /// Somewhere to sleep. An agent has exactly one, and it is what makes it a
    /// resident of this building rather than a visitor.
    Home,
    /// Somewhere to work.
    Work,
    /// Somewhere to go that is neither — a shop.
    Errand,
}

impl SlotRole {
    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            SlotRole::Home => "home",
            SlotRole::Work => "work",
            SlotRole::Errand => "errand",
        }
    }
}

/// **One place one person can be**, in the world, on a storey, in a named room.
///
/// Located in world metres *and* by room index: the metres are what a schedule
/// walks to and the index is what a caller turns into an
/// [`inf_nav`] node once it has minted the building's own salt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgSlot {
    /// What this place is for.
    pub role: SlotRole,
    /// The room's centre on its own walking surface, world metres — the same
    /// point [`BuildingPlan::interior_nav`] puts that room's node at.
    pub at: DVec3,
    /// Index into the plan's [`rooms`](BuildingPlan::rooms).
    pub room: u32,
    /// Which of the volume's own buildings this belongs to — assigned by the
    /// caller that evaluated them, in the order [`plans_of`](super::plans_of)
    /// returned them.
    pub building: u32,
    /// The storey, 0-based.
    pub floor: u32,
    /// Which of the room's own slots this is, `0..n`. Two workers in one office
    /// stand on the same node and differ only here, which is what makes a slot's
    /// identity stable when the room's area changes by a rounding.
    pub index: u32,
    /// **The nav node this slot stands on**, in the level's own namespace —
    /// `room_node_id_in(building_salt(volume, building), room)`.
    ///
    /// Carried rather than re-derived because the salt is the *level's* word and
    /// a consumer holding a slot may not hold the volume it came from.
    pub node: inf_nav::NavNodeId,
}

/// **How many people a room of this kind and area holds**, and in which role.
///
/// The one place the table in this module's docs is written as code. Returns
/// `(role, count)`; a count of zero is the answer for the eight room types that
/// hold nobody, and `Errand` slots are added by [`slots_of`] rather than here
/// because a shop is one destination however many people staff it.
pub fn occupancy(kind: RoomType, area_m2: f64) -> (SlotRole, usize) {
    let per = |m2_each: f64| -> usize {
        if !(area_m2.is_finite() && area_m2 > 0.0) {
            return 0;
        }
        ((area_m2 / m2_each).ceil() as usize).clamp(1, MAX_SLOTS_PER_ROOM)
    };
    match kind {
        RoomType::Bedroom | RoomType::Guest => (SlotRole::Home, 1),
        RoomType::Office => (SlotRole::Work, per(OFFICE_M2_PER_WORKER)),
        RoomType::Workshop => (SlotRole::Work, per(WORKSHOP_M2_PER_WORKER)),
        RoomType::Retail => (SlotRole::Work, per(RETAIL_M2_PER_WORKER)),
        // **A bar is staffed like a shop counter, not like a shop** (wave
        // VEN1a): the floor area behind a bar is small and one keeper serves a
        // long run of it, so the metres-per-worker is generous.
        RoomType::BarRoom => (SlotRole::Work, per(BAR_M2_PER_KEEPER)),
        // **One act on a stage, however big it is** — the same argument that
        // makes a shop one errand however many people staff it. A 40 m2 stage
        // and a 12 m2 one both hold a routine, and a per-area count would put
        // four dancers on one pole.
        RoomType::Stage => (SlotRole::Work, 1),
        // A dance floor is nobody's WORKPLACE. It is an errand destination and
        // gets its visit slot from `slots_of`, which is why it appears in the
        // zero arm and is still somewhere a person can be sent.
        RoomType::DanceFloor => (SlotRole::Home, 0),
        RoomType::Corridor
        | RoomType::Stair
        | RoomType::Lobby
        | RoomType::Meeting
        | RoomType::Service
        | RoomType::Living
        | RoomType::Kitchen
        | RoomType::Bath
        | RoomType::Storage => (SlotRole::Home, 0),
    }
}

/// **Every place a person can be in this building**, in room order then slot
/// order.
///
/// `building` is the ordinal the caller gives this plan inside its own volume;
/// it is copied onto every slot and never interpreted here.
///
/// Deterministic by construction: it walks [`BuildingPlan::rooms`], which is
/// floor-major then partition order, and appends in that order. No sort, no map,
/// no seed.
pub fn slots_of(plan: &BuildingPlan, building: u32, salt: u64) -> Vec<PcgSlot> {
    let mut out = Vec::new();
    for (i, room) in plan.rooms.iter().enumerate() {
        let (role, n) = occupancy(room.kind, room.rect.area());
        if n == 0 && !room.kind.is_errand_destination() {
            continue;
        }
        let c = plan.frame.to_world(room.rect.center());
        let at = DVec3::new(c.x, plan.floor_y(room.floor), c.y);
        if !at.is_finite() {
            continue;
        }
        let node = super::room_node_id_in(salt, i);
        for k in 0..n {
            out.push(PcgSlot {
                role,
                at,
                room: i as u32,
                building,
                floor: room.floor,
                index: k as u32,
                node,
            });
        }
        // **A shop is one errand however many people staff it.** The visit slot
        // is emitted after the workers so a room's slots stay in role order, and
        // it takes the next `index` so no two slots of one room collide.
        //
        // Wave VEN1a widened the test from `== Retail` to the rule
        // `RoomType::is_errand_destination`, so a bar and a dance floor are
        // places the town already walks to -- and a venue therefore has slots,
        // which is what keeps `pass.rs` from handing it an EMPTY interior nav
        // graph and orphaning every room in it.
        if room.kind.is_errand_destination() {
            out.push(PcgSlot {
                role: SlotRole::Errand,
                at,
                room: i as u32,
                building,
                floor: room.floor,
                index: n as u32,
                node,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::{plan_building, ArchetypeId, BuildingParams, Rect2};
    use glam::DVec2;

    fn plan_of(archetype: ArchetypeId, floors: u32) -> BuildingPlan {
        let footprint = Rect2::new(DVec2::new(-9.0, -7.0), DVec2::new(9.0, 7.0));
        let mut params = BuildingParams::new(archetype, footprint, 0.0, 0x51075);
        params.floors = floors;
        plan_building(&params)
    }

    #[test]
    fn a_bedroom_holds_one_person_whatever_its_area() {
        for area in [4.0_f64, 12.0, 40.0, 400.0] {
            assert_eq!(occupancy(RoomType::Bedroom, area), (SlotRole::Home, 1));
            assert_eq!(occupancy(RoomType::Guest, area), (SlotRole::Home, 1));
        }
    }

    #[test]
    fn a_workplace_holds_people_by_area_and_the_three_rates_differ() {
        // 60 m2: an office at 12 is 5, a workshop at 20 is 3, a shop at 30 is 2.
        assert_eq!(occupancy(RoomType::Office, 60.0), (SlotRole::Work, 5));
        assert_eq!(occupancy(RoomType::Workshop, 60.0), (SlotRole::Work, 3));
        assert_eq!(occupancy(RoomType::Retail, 60.0), (SlotRole::Work, 2));
        // The rounding is UP and the floor is one: half a desk is still a desk.
        assert_eq!(occupancy(RoomType::Office, 1.0), (SlotRole::Work, 1));
        assert_eq!(occupancy(RoomType::Office, 13.0), (SlotRole::Work, 2));
    }

    #[test]
    fn a_room_that_holds_nobody_holds_nobody() {
        for kind in [
            RoomType::Corridor,
            RoomType::Stair,
            RoomType::Lobby,
            RoomType::Meeting,
            RoomType::Service,
            RoomType::Living,
            RoomType::Kitchen,
            RoomType::Bath,
            RoomType::Storage,
        ] {
            assert_eq!(
                occupancy(kind, 100.0).1,
                0,
                "{} holds somebody",
                kind.name()
            );
        }
    }

    #[test]
    fn a_degenerate_area_cannot_mint_a_crowd() {
        // The ceiling, and the non-finite guard beside it.
        assert_eq!(
            occupancy(RoomType::Office, 1.0e9).1,
            MAX_SLOTS_PER_ROOM,
            "an enormous room is capped"
        );
        assert_eq!(occupancy(RoomType::Office, f64::NAN).1, 0);
        assert_eq!(occupancy(RoomType::Office, f64::INFINITY).1, 0);
        assert_eq!(occupancy(RoomType::Office, -5.0).1, 0);
    }

    #[test]
    fn a_house_holds_a_household_and_an_office_holds_workers() {
        let house = plan_of(ArchetypeId::House, 2);
        let hs = slots_of(&house, 0, 0);
        let homes = hs.iter().filter(|s| s.role == SlotRole::Home).count();
        let works = hs.iter().filter(|s| s.role == SlotRole::Work).count();
        assert!(
            homes > 0,
            "a two-storey House holds nobody: {} room(s), kinds {:?}",
            house.rooms.len(),
            house
                .rooms
                .iter()
                .map(|r| r.kind.name())
                .collect::<Vec<_>>()
        );
        assert_eq!(works, 0, "a House holds workers");

        let office = plan_of(ArchetypeId::Office, 3);
        let os = slots_of(&office, 0, 0);
        assert_eq!(
            os.iter().filter(|s| s.role == SlotRole::Home).count(),
            0,
            "an Office holds residents"
        );
        assert!(
            os.iter().filter(|s| s.role == SlotRole::Work).count() > 0,
            "a three-storey Office holds no workers"
        );
    }

    #[test]
    fn an_apartment_holds_more_people_than_a_house_on_the_same_lot() {
        let house = slots_of(&plan_of(ArchetypeId::House, 2), 0, 0)
            .iter()
            .filter(|s| s.role == SlotRole::Home)
            .count();
        let flats = slots_of(&plan_of(ArchetypeId::Apartment, 4), 0, 0)
            .iter()
            .filter(|s| s.role == SlotRole::Home)
            .count();
        assert!(
            flats > house,
            "a four-storey Apartment ({flats}) holds no more people than a \
             two-storey House ({house}) on the same footprint"
        );
    }

    #[test]
    fn a_shop_offers_an_errand_as_well_as_its_jobs() {
        let shop = plan_of(ArchetypeId::Shop, 2);
        let ss = slots_of(&shop, 7, 0);
        let errands: Vec<_> = ss.iter().filter(|s| s.role == SlotRole::Errand).collect();
        let retail = shop
            .rooms
            .iter()
            .filter(|r| r.kind == RoomType::Retail)
            .count();
        assert_eq!(
            errands.len(),
            retail,
            "a Shop's {retail} retail room(s) offer {} errand(s)",
            errands.len()
        );
        assert!(retail > 0, "a two-storey Shop plans no retail room at all");
        assert!(
            ss.iter().all(|s| s.building == 7),
            "the caller's building ordinal is not carried onto every slot"
        );
    }

    #[test]
    fn a_slot_stands_where_its_own_rooms_nav_node_stands() {
        // The claim that lets a caller turn `room` into a node id and get the
        // same metres back: a slot's `at` IS `interior_nav`'s room node.
        let plan = plan_of(ArchetypeId::Apartment, 3);
        let g = plan.interior_nav();
        let mut checked = 0usize;
        for s in slots_of(&plan, 0, 0) {
            let node = g
                .node(crate::building::room_node_id(s.room as usize))
                .expect("every slot names a room the interior graph has");
            assert_eq!(
                node.position.to_array().map(f64::to_bits),
                s.at.to_array().map(f64::to_bits),
                "slot in room {} stands at {:?} and its node at {:?}",
                s.room,
                s.at,
                node.position
            );
            checked += 1;
        }
        assert!(checked > 0, "the plan offered no slot to check");
    }

    /// The slot's node id is the level's word for that room, and two buildings
    /// salted apart never share one.
    #[test]
    fn a_slots_node_is_its_own_buildings_room_in_the_levels_namespace() {
        let plan = plan_of(ArchetypeId::House, 2);
        let v = Uuid::from_u128(0x50C1_E7A0);
        let (s0, s1) = (building_salt(Some(v), 0), building_salt(Some(v), 1));
        assert_ne!(s0, s1, "two buildings of one volume share a salt");
        assert_ne!(
            building_salt(Some(v), 0),
            building_salt(Some(Uuid::from_u128(0x50C1_E7A1)), 0),
            "two volumes' first buildings share a salt"
        );
        assert_eq!(building_salt(Some(v), 0), building_salt(Some(v), 0));
        assert!(s0 <= super::super::NAV_SALT_MASK);
        for slot in slots_of(&plan, 0, s0) {
            assert_eq!(
                slot.node,
                crate::building::room_node_id_in(s0, slot.room as usize)
            );
            assert_eq!(crate::building::node_salt(slot.node), s0);
        }
    }

    /// **The derivation table, over all seven archetypes** — printed so the
    /// ledger quotes a measurement rather than an intention, and asserted where
    /// the table's own sentences say something a reader can check.
    #[test]
    fn every_archetype_holds_what_its_rooms_imply() {
        println!(
            "{:<10} {:>6} {:>6} {:>6} {:>7} {:>7}",
            "archetype", "floors", "rooms", "homes", "workers", "errands"
        );
        let mut homes_of = std::collections::BTreeMap::new();
        for id in ArchetypeId::ALL {
            let floors = 3;
            let plan = plan_of(id, floors);
            let slots = slots_of(&plan, 0, 0);
            let h = slots.iter().filter(|s| s.role == SlotRole::Home).count();
            let w = slots.iter().filter(|s| s.role == SlotRole::Work).count();
            let e = slots.iter().filter(|s| s.role == SlotRole::Errand).count();
            println!(
                "{:<10} {:>6} {:>6} {:>6} {:>7} {:>7}",
                id.name(),
                floors,
                plan.rooms.len(),
                h,
                w,
                e
            );
            homes_of.insert(id.name(), h);
            assert!(
                h + w + e > 0,
                "{} holds nobody at all on {floors} storeys of {} room(s)",
                id.name(),
                plan.rooms.len()
            );
        }
        // The three sentences the module docs make about archetypes, each one a
        // consequence of the ROOM table rather than an entry in it.
        assert!(
            homes_of["House"] > 0 && homes_of["Apartment"] > 0,
            "a House or an Apartment holds nobody"
        );
        assert_eq!(
            homes_of["Office"], 0,
            "an Office holds residents, so the room table has gained a bedroom \
             somewhere it should not have"
        );
        assert_eq!(homes_of["Shop"], 0, "a Shop holds residents");
    }

    /// **A slot-bearing building offers exactly one FRONT DOOR to a level
    /// network**, and it is the doorway node with a single edge.
    ///
    /// `inf_ecs::society` finds a building's join to the street that way and by
    /// no other means, so the day it stops being true a whole settlement stops
    /// being routable and nothing else says so.
    #[test]
    fn a_building_offers_exactly_one_single_edge_doorway() {
        for id in ArchetypeId::ALL {
            for floors in [1u32, 2, 3] {
                let plan = plan_of(id, floors);
                if slots_of(&plan, 0, 7).is_empty() {
                    continue;
                }
                let g = plan.interior_nav_in(7);
                let leaves: Vec<_> = g
                    .nodes()
                    .filter(|n| {
                        n.kind == inf_nav::NavKind::Doorway && g.edges_from(n.id).len() == 1
                    })
                    .map(|n| n.id)
                    .collect();
                let doorways = g
                    .nodes()
                    .filter(|n| n.kind == inf_nav::NavKind::Doorway)
                    .count();
                println!(
                    "{:<10} {floors}F: {} rooms, {doorways} doorway node(s), {} leaf/leaves, entrance {:?}",
                    id.name(),
                    plan.rooms.len(),
                    leaves.len(),
                    plan.entrance
                );
                assert_eq!(
                    leaves.len(),
                    1,
                    "{} at {floors} storeys offers {} of its {doorways} doorways as a single-edge leaf",
                    id.name(),
                    leaves.len()
                );
            }
        }
    }

    #[test]
    fn slots_are_a_pure_function_of_the_plan() {
        let plan = plan_of(ArchetypeId::Hotel, 4);
        assert_eq!(slots_of(&plan, 3, 5), slots_of(&plan, 3, 5));
        // The ordinal is the only thing a caller can change.
        let a = slots_of(&plan, 0, 0);
        let b = slots_of(&plan, 1, 0);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.role == y.role
            && x.room == y.room
            && x.at == y.at
            && x.building + 1 == y.building));
    }
}
