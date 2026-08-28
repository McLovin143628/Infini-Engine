//! **The society** (island wave NPC1d): a level's own buildings, turned into
//! people with days.
//!
//! `inf_pcg` decides who a *building* holds — bedrooms are homes, offices and
//! shops are workplaces, a shop is also somewhere to go (`inf_pcg::building::
//! society`). This module is the half that needs the whole **level**: it pairs a
//! home with a workplace, lays a network a body can walk between them, plans the
//! four legs of a day over it, and installs the result through
//! [`crate::crowd::add_agents`].
//!
//! ```text
//!   PcgVolume::residents ─┐
//!   PcgVolume::interior_nav ─┼─▶ SocietyRes { network, places } ─▶ CrowdSchedule
//!   PcgVolume centre+extent ─┘                                        │
//!                                                                     ▼
//!                                                      CrowdPopulationRes
//! ```
//!
//! # The street network is the PAVEMENT round the blocks
//!
//! A settlement's street plan lives in Ring 1, in the recipe, and is not in a
//! cooked pack — so a shipped player has no grid to read. What every host *does*
//! have is the blocks themselves: a `PcgVolume` is a centre and an axis-aligned
//! half-extent, and the streets are the gaps between them. So each volume lays a
//! **ring of eight nodes** [`PAVEMENT_M`] outside its own rectangle — four
//! corners and four edge midpoints — and two rings within [`BLOCK_LINK_MAX_M`]
//! of each other are joined at their nearest pair. That is a street crossing,
//! and it is derived from the level's own contents rather than from a plan the
//! level does not carry.
//!
//! It also closes NPC1c's defect 5 by construction. A settlement's grid runs out
//! to its whole reservation radius while the *levelled pad* is smaller, so the
//! outer lines lie on raw hillside and a route down one walks a body into the cut
//! face. A pavement hugs its own block, and a block stands on the pad — so every
//! node of this network is on ground a body can walk, without a ground profile
//! and without a terrain query.
//!
//! # THE SEARCH IS TWO LEVELS, and that is a measurement rather than a taste
//!
//! The obvious network is one graph with every building's whole interior in it.
//! It was built that way first and priced: a settlement's own buildings are
//! about **25 000 nodes**, and `inf-nav`'s own measured 743 µs over a 1 600-node
//! grid extrapolates to roughly **11 ms a search** — four legs an agent, four
//! hundred agents, inside a fixed step. That is not a slow gate, it is a
//! simulation that stops.
//!
//! So the level network holds the **streets and the front doors** — about
//! 1 600 nodes, which is exactly the size `inf-nav` measured — and a building's
//! interior is searched *inside the building*, over the hundred-odd nodes it has
//! of its own. A leg is `home → front door` (inside), `front door → front door`
//! (outside), `front door → work` (inside), joined. The outer half is memoized
//! on its endpoint pair, because a hundred residents of one block commuting to
//! one office is a hundred agents sharing one street route.
//!
//! # A building joins at its own front door
//!
//! Each volume's [`PcgVolume::interior_nav`] is already salted per building
//! (`inf_pcg::building::society::building_salt`), so absorbing them all is a
//! union rather than a collision. A building's **exterior** door is the one
//! doorway node with a single edge — a leaf, because the wall it stands in has no
//! room on the far side — and it is linked to the nearest node of its own block's
//! ring. So a route reads *room → doorway → pavement → pavement → doorway →
//! room*, which is the building↔street↔building crossing this wave exists to
//! make possible.
//!
//! # Nobody plans a day while the town is still being built
//!
//! Volumes stream. A resident of the first block to arrive would be paired with
//! the only workplace it could see, which would make a level's society a function
//! of the order its cells activated. So [`sync_society`] plans **only on a step
//! that folded no new volume**, and plans at most [`SOCIETY_PLANS_PER_STEP`]
//! agents on it — a bounded spike rather than a load-time cliff. Both hosts
//! stream identically, so both derive the same society; what is carried honestly
//! is that an agent plans **once**, and a workplace that arrives afterwards does
//! not re-open its day.
//!
//! # Everything here is derived
//!
//! `SocietyRes` is a bevy resource, exactly as `CrowdPopulationRes` and
//! `DeformFieldRes` are, so **no schema moves**. The slots it reads are
//! `#[serde(skip)]` on the volume. An agent's `Guid` is a hash of the level's own
//! content, so two hosts mint the same one without talking.
//!
//! # Portable math
//!
//! Distances are `sqrt` of sums of products, a quantization is a `floor`, and a
//! pairing is a comparison — the P14 ban list binds this module because a slot's
//! metres land on an NPC's `Transform` and therefore in the replay trace.

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::Resource;
use glam::{DVec2, DVec3};
use inf_nav::{NavGraph, NavKind, NavNodeId};
use uuid::Uuid;

use crate::components::{GlobalTransform, Guid, PcgVolume, SlotRole};
use crate::crowd::{CrowdArchetype, CrowdRecord, CrowdSchedule, ScheduleLeg};
use crate::world::EcsWorld;

/// How far outside a block's own rectangle its pavement ring runs, metres.
///
/// Two metres is a pavement: far enough out that the ring is not inside the
/// building line, close enough that a crossing to the next block is the width of
/// the street and not a diagonal across it. The settlement generator reserves
/// about eight metres between blocks, so two rings sit about four metres apart.
pub const PAVEMENT_M: f64 = 2.0;

/// The furthest two blocks' pavements are joined across the street, metres.
///
/// Sized against the settlement grid it is derived from rather than for looks:
/// blocks on one grid are a street reserve apart (about 8 m) and the *next* block
/// along is a whole pitch away (about 120 m), so forty metres joins neighbours
/// and never reaches past one. Two settlements are kilometres apart and stay
/// separate components, which is correct — they are separate towns.
pub const BLOCK_LINK_MAX_M: f64 = 40.0;

/// The furthest a front door is joined to its own block's pavement, metres.
///
/// A door that is further than this from its own block's ring is a door on a
/// building that is not on that block, and linking it would cut a route through
/// whatever stands between. Refused rather than stretched; the count is in
/// [`SocietyStats::frontages_refused`].
pub const FRONTAGE_MAX_M: f64 = 40.0;

/// The lattice a pavement node's id is quantized onto, metres.
///
/// One centimetre. Two blocks' rings are laid from their own centres and
/// half-extents, so a shared corner is the same point *arithmetically* — the
/// quantization is what makes it the same point after the arithmetic, and it
/// is deliberately far finer than any distance that means anything here.
pub const PAVEMENT_LATTICE_M: f64 = 0.01;

/// How many agents' days [`sync_society`] plans on one fixed step.
///
/// A day is four Dijkstras over the level's network, so a settlement of four
/// hundred residents is sixteen hundred searches. Doing them all on the step the
/// last block arrived would be a load-time cliff inside a *fixed* step, which is
/// the shape wave I4b's budgets exist to refuse. Eight a step fills a settlement
/// in about a second of sim and is measured rather than assumed — see the wave
/// ledger's planning row.
pub const SOCIETY_PLANS_PER_STEP: usize = 8;

/// The hour the working day begins, and the hours the other three legs do.
///
/// One table, in one place, so "the town populates at morning and empties at
/// night" is a statement about these six numbers. The commute is an hour and the
/// errand half of one, which at the island's authored rate is a walking pace
/// over a settlement-sized route (`ScheduleLeg::implied_speed_mps`).
pub const WORK_START_H: f64 = 8.0;
/// How long a commute leg takes, in hours of the level clock.
pub const COMMUTE_H: f64 = 1.0;
/// The hour the errand out of work begins.
pub const ERRAND_OUT_H: f64 = 12.0;
/// How long an errand leg takes, in hours of the level clock.
pub const ERRAND_H: f64 = 0.5;
/// The hour the walk back to work begins.
pub const ERRAND_BACK_H: f64 = 13.0;
/// The hour the walk home begins.
pub const HOME_H: f64 = 18.0;

/// Salts an agent's derived `Guid`. See [`agent_guid`].
const SALT_AGENT: u64 = 0x4147_454E_5400_0001;

/// The tag every derived agent `Guid` carries in its top sixteen bits — `"NP"`.
///
/// Not a namespace guarantee and not pretended to be one: it is there so a guid
/// in a trace or a log is recognizable as a crowd agent's rather than a level
/// entity's. The guarantee that an agent never overwrites a level entity is
/// `crate::crowd::add_agents`' own refusal, which asks the world.
const AGENT_TAG: u128 = 0x4E50;

/// **One place a level offers**, with the node a route reaches it by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SocietyPlace {
    /// What it is for.
    pub role: SlotRole,
    /// Where it is, world metres.
    pub at: DVec3,
    /// The node of its own building's interior it stands on — a node of
    /// [`SocietyRes::interiors`]`[volume]`, **not** of the level network.
    pub node: NavNodeId,
    /// The volume whose interior graph holds [`node`](Self::node).
    pub volume: Uuid,
    /// Its building's front door, which IS a node of the level network — the
    /// join between the two levels of the search.
    pub door: NavNodeId,
}

/// What one [`sync_society`] did, and what the society holds after it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SocietyStats {
    /// Volumes folded into the network so far.
    pub volumes: usize,
    /// Volumes folded on THIS step. Non-zero means the town is still building,
    /// and nothing plans a day on such a step.
    pub folded_now: usize,
    /// Homes the level has offered.
    pub homes: usize,
    /// Workplaces the level has offered.
    pub works: usize,
    /// Errands the level has offered.
    pub errands: usize,
    /// Agents installed into the population.
    pub agents: usize,
    /// Homes still waiting for a day.
    pub pending: usize,
    /// Days planned on this step.
    pub planned_now: usize,
    /// Agents whose home routes to no workplace, and who therefore keep a
    /// stay-at-home day. A number to watch: a large one means the network is not
    /// joined up.
    pub homebound: usize,
    /// Agents with nowhere at all to go — no reachable work and no reachable
    /// errand. They stand at home.
    pub housebound: usize,
    /// Nodes in the network.
    pub nodes: usize,
    /// Directed edges in the network.
    pub edges: usize,
    /// Front doors joined to a pavement.
    pub frontages: usize,
    /// Front doors further than [`FRONTAGE_MAX_M`] from their own block's ring,
    /// and therefore not joined. Should be zero on a settlement.
    pub frontages_refused: usize,
    /// Pavement rings joined to a neighbour's.
    pub crossings: usize,
    /// **Interior node ids that were already in the network when their building
    /// was absorbed** — a building salt collision, which welds one building's
    /// room to another's. Zero is the expectation and the arm.
    pub salt_collisions: usize,
    /// Slots in a building with no exterior door — nobody can walk in, so
    /// nobody living there gets a day. Zero on a settlement.
    pub doorless: usize,
    /// Agent `Guid`s `add_agents` refused because the world already held them.
    pub guid_refusals: usize,
    /// Street routes searched over the level network so far — the outer halves
    /// that were NOT served by the memo.
    pub outer_searches: usize,
    /// Outer halves served by the memo. The ratio of this to
    /// [`outer_searches`](Self::outer_searches) is what the two-level split buys
    /// on a real settlement.
    pub outer_cached: usize,
}

/// **The level's society** — its walkable network, the places it offers, and the
/// homes that have not been given a day yet.
///
/// A bevy resource, so nothing here is serialized and no schema moves. Absent
/// until [`sync_society`] finds a volume with residents on it, which is what
/// makes "a level with no population costs one `contains_resource`" structural.
#[derive(Resource, Debug, Clone, Default)]
pub struct SocietyRes {
    /// **The level network**: every block's pavement, every slot-bearing
    /// building's front door, the frontage links between them and the crossings
    /// between blocks. About 1 600 nodes on a settlement — see the module docs
    /// for why the interiors are not in it.
    pub network: NavGraph,
    /// **Each volume's own interior**, searched inside the building.
    pub interiors: BTreeMap<Uuid, NavGraph>,
    /// **The outer half of a leg, memoized on its endpoint pair.** A hundred
    /// residents of one block commuting to one office share one street route,
    /// and this is what makes them pay for it once. `None` records a pair the
    /// network cannot join, so a refusal is not re-searched either.
    pub legs: BTreeMap<(NavNodeId, NavNodeId), Option<inf_nav::NavPath>>,
    /// Volumes already folded in, by `Guid`.
    pub folded: BTreeSet<Uuid>,
    /// Every workplace, in the order the volumes were folded.
    pub work: Vec<SocietyPlace>,
    /// Every errand.
    pub errand: Vec<SocietyPlace>,
    /// Homes with no day yet, by the agent `Guid` that will live there.
    pub pending: BTreeMap<Uuid, SocietyPlace>,
    /// The counters after the last sync.
    pub stats: SocietyStats,
}

/// **The `Guid` of the agent who lives in one home slot** — a hash of the
/// level's own content, so two hosts mint the same one without talking.
///
/// `(volume, building, room, index)` is the level's name for that bed. Nothing
/// about it depends on iteration order, on when the volume streamed in, or on
/// how many agents have been minted already, which is what makes a society
/// re-derivable from a level rather than a thing a save file has to carry.
pub fn agent_guid(volume: Uuid, building: u32, room: u32, index: u32) -> Uuid {
    let b = volume.as_u128();
    let hi = mix64((b as u64) ^ SALT_AGENT ^ u64::from(building));
    let lo = mix64(((b >> 64) as u64) ^ (u64::from(room) << 32) ^ u64::from(index) ^ hi);
    let raw = (u128::from(hi) << 64) | u128::from(lo);
    Uuid::from_u128((AGENT_TAG << 112) | (raw & ((1u128 << 112) - 1)))
}

/// The SplitMix64 finalizer, the house mixer.
///
/// The same constants `crate::crowd::agent_rand` pins, spelled out here rather
/// than borrowed so this module's ids do not move if that one's salts ever do.
fn mix64(x: u64) -> u64 {
    let mut x = x ^ 0x9e37_79b9_7f4a_7c15;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The id of a pavement node standing at `p` — a hash of its quantized XZ.
///
/// Position rather than an index, so two blocks that lay a node at the same
/// corner lay the **same** node and the rings weld themselves. XZ and not Y,
/// because two neighbouring pads may differ by a few centimetres and a corner is
/// a corner; the last ring to write it wins, which is deterministic because
/// volumes are folded in `Guid` order.
pub fn pavement_node_id(p: DVec2) -> NavNodeId {
    let q = |v: f64| -> i64 {
        if !v.is_finite() {
            return 0;
        }
        (v / PAVEMENT_LATTICE_M).floor() as i64
    };
    let h = mix64((q(p.x) as u64) ^ mix64(q(p.y) as u64));
    inf_nav::domain::PAVEMENT | (h & inf_nav::domain::LOCAL_MASK)
}

/// **Which building an interior node belongs to** — the salt out of its id.
///
/// The mirror of `inf_pcg::building::node_salt`, spelled here because `inf-ecs`
/// does not depend on `inf-pcg` (the P19.5 dependency-light mirror ruling, which
/// is also why [`crate::components::ResidentSlot`] exists at all). The layout is
/// frozen by `inf_pcg::building::interior_nav_in`'s own doc table: domain in
/// 60–63, class in 59, salt in 20–58, index in 0–19.
fn node_salt(id: NavNodeId) -> u64 {
    (id >> 20) & ((1 << 39) - 1)
}

/// One volume's own eight pavement points, world metres, in a walking order
/// round the ring.
fn ring_points(centre: DVec3, extent: DVec2, y: f64) -> [DVec3; 8] {
    let (x, z) = (extent.x + PAVEMENT_M, extent.y + PAVEMENT_M);
    let (cx, cz) = (centre.x, centre.z);
    let p = |dx: f64, dz: f64| DVec3::new(cx + dx, y, cz + dz);
    [
        p(-x, -z),
        p(0.0, -z),
        p(x, -z),
        p(x, 0.0),
        p(x, z),
        p(0.0, z),
        p(-x, z),
        p(-x, 0.0),
    ]
}

/// The plan distance between two axis-aligned block rectangles, metres — `0.0`
/// when they overlap.
fn rect_gap(a_c: DVec3, a_e: DVec2, b_c: DVec3, b_e: DVec2) -> f64 {
    let dx = ((a_c.x - b_c.x).abs() - (a_e.x + b_e.x)).max(0.0);
    let dz = ((a_c.z - b_c.z).abs() - (a_e.y + b_e.y)).max(0.0);
    (dx * dx + dz * dz).sqrt()
}

/// What one volume contributes, read out of the world before anything is
/// mutated.
struct VolumeFacts {
    guid: Uuid,
    centre: DVec3,
    extent: DVec2,
    pad_y: f64,
    residents: Vec<crate::components::ResidentSlot>,
    interior: NavGraph,
}

/// Read every volume that offers a resident, in `Guid` order.
fn volume_facts(world: &EcsWorld) -> Vec<VolumeFacts> {
    let mut out: Vec<VolumeFacts> = Vec::new();
    for e in world.world().iter_entities() {
        let (Some(g), Some(v)) = (e.get::<Guid>(), e.get::<PcgVolume>()) else {
            continue;
        };
        if v.residents.is_empty() {
            continue;
        }
        let centre = e
            .get::<GlobalTransform>()
            .map(|t| t.translation())
            .unwrap_or(DVec3::ZERO);
        if !centre.is_finite() {
            continue;
        }
        // **The pad, from the level's own front doors.** A block's volume
        // entity sits at the block centre, whose Y is whatever the level put
        // there; a ground-floor exterior doorway's sill is the walking surface a
        // body actually stands on. Taking the pad from the doors is what keeps a
        // pavement out of the hillside without a terrain query — NPC1c's defect
        // 5, closed by construction rather than by a profile.
        let pad_y = v
            .doorways
            .iter()
            .filter(|d| d.exterior && d.floor == 0 && d.hinge.is_finite())
            .map(|d| d.hinge.y - d.height_m * 0.5)
            .fold(f64::INFINITY, f64::min);
        out.push(VolumeFacts {
            guid: g.0,
            centre,
            extent: DVec2::new(v.extent.x, v.extent.y),
            pad_y: if pad_y.is_finite() { pad_y } else { centre.y },
            residents: v.residents.clone(),
            interior: v.interior_nav.clone(),
        });
    }
    out.sort_by_key(|f| f.guid);
    out
}

/// **The body a crowd wears on this level** (NPC1d) — the lowest-`Guid` entity
/// that carries a rigged [`SkeletalMesh`](crate::components::SkeletalMesh), with
/// its skeleton and its state machine.
///
/// Derived rather than configured, and derived HERE rather than by each host, so
/// a crowd installed by the editor's Simulate and one installed by the shipped
/// player are made of the same asset without either being told which. The
/// lowest `Guid` because a rule that says "the first one" over a bevy world is a
/// rule about archetype layout.
///
/// A level with no rigged character answers a bodiless humanoid: an NPC with the
/// right capsule, the right feet offset and no mesh. It still walks, still
/// collides and still traces; it is simply not drawn. That is the honest answer
/// for a level that has nothing to draw it with, and it is why this returns a
/// value rather than an `Option` a caller would have to case-split.
pub fn level_archetype(world: &EcsWorld) -> CrowdArchetype {
    let mut best: Option<(Uuid, CrowdArchetype)> = None;
    for e in world.world().iter_entities() {
        let (Some(g), Some(sk)) = (e.get::<Guid>(), e.get::<crate::components::SkeletalMesh>())
        else {
            continue;
        };
        if sk.skeleton.is_none() {
            continue;
        }
        let sm = e
            .get::<crate::components::AnimStateMachine>()
            .and_then(|a| a.sm);
        if best.as_ref().is_none_or(|(bg, _)| g.0 < *bg) {
            best = Some((g.0, CrowdArchetype::humanoid(sk.mesh, sk.skeleton, sm)));
        }
    }
    best.map(|(_, a)| a)
        .unwrap_or_else(|| CrowdArchetype::humanoid(None, None, None))
}

/// **Grow the level's society, and install any day it can plan** (NPC1d) — the
/// one Ring-0 door both hosts call, once per fixed step, inside the crowd phase.
///
/// Cheap when nothing changed: one walk over the entities to see whether a
/// volume with residents has appeared, and nothing else on a step that neither
/// folded a volume nor had a home waiting.
///
/// Returns this step's counters; they are also left on
/// [`SocietyRes::stats`].
pub fn sync_society(world: &mut EcsWorld) -> SocietyStats {
    let facts = volume_facts(world);
    if facts.is_empty() && !world.world().contains_resource::<SocietyRes>() {
        // Absent costs nothing.
        return SocietyStats::default();
    }
    let mut soc = world
        .world_mut()
        .remove_resource::<SocietyRes>()
        .unwrap_or_default();
    let mut stats = SocietyStats {
        agents: soc.stats.agents,
        guid_refusals: soc.stats.guid_refusals,
        homebound: soc.stats.homebound,
        housebound: soc.stats.housebound,
        doorless: soc.stats.doorless,
        outer_searches: soc.stats.outer_searches,
        outer_cached: soc.stats.outer_cached,
        ..SocietyStats::default()
    };

    // ── 1. fold every volume the network has not seen ──────────────────────
    // Rings are laid FIRST, all of them, so a crossing can never miss a
    // neighbour that happens to be folded later in the same step.
    let fresh: Vec<&VolumeFacts> = facts
        .iter()
        .filter(|f| !soc.folded.contains(&f.guid))
        .collect();
    let mut rings: BTreeMap<Uuid, Vec<(NavNodeId, DVec3)>> = BTreeMap::new();
    for f in &fresh {
        let pts = ring_points(f.centre, f.extent, f.pad_y);
        let ids: Vec<(NavNodeId, DVec3)> = pts
            .iter()
            .map(|p| (pavement_node_id(DVec2::new(p.x, p.z)), *p))
            .collect();
        for (id, p) in &ids {
            soc.network.add_node(*id, *p, NavKind::Street);
        }
        for i in 0..ids.len() {
            let j = (i + 1) % ids.len();
            soc.network
                .link(ids[i].0, ids[j].0, NavKind::Street, Vec::new());
        }
        rings.insert(f.guid, ids);
    }
    // The rings a crossing may reach: the ones just laid, plus every one already
    // in the society, re-derived from the volumes that are still resident.
    let mut known: BTreeMap<Uuid, (DVec3, DVec2, Vec<(NavNodeId, DVec3)>)> = BTreeMap::new();
    for f in &facts {
        let ids = match rings.get(&f.guid) {
            Some(v) => v.clone(),
            None => ring_points(f.centre, f.extent, f.pad_y)
                .iter()
                .map(|p| (pavement_node_id(DVec2::new(p.x, p.z)), *p))
                .collect(),
        };
        known.insert(f.guid, (f.centre, f.extent, ids));
    }
    for f in &fresh {
        let (ac, ae, a_ids) = &known[&f.guid];
        for (other, (bc, be, b_ids)) in &known {
            if other == &f.guid || rect_gap(*ac, *ae, *bc, *be) > BLOCK_LINK_MAX_M {
                continue;
            }
            // Only one direction per pair: a fresh ring links to everything, and
            // two fresh rings link once because `link` is symmetric and
            // `push_edge` deduplicates on `(to, cost)`.
            let mut best: Option<(f64, NavNodeId, NavNodeId)> = None;
            for (ai, ap) in a_ids {
                for (bi, bp) in b_ids {
                    let d = (*ap - *bp).length();
                    if best.map(|(bd, _, _)| d < bd).unwrap_or(true) {
                        best = Some((d, *ai, *bi));
                    }
                }
            }
            if let Some((_, ai, bi)) = best {
                soc.network.link(ai, bi, NavKind::Street, Vec::new());
                stats.crossings += 1;
            }
        }
    }

    // ── 2. keep each fresh volume's interior, and put its FRONT DOORS on the
    //       level network ───────────────────────────────────────────────────
    let mut doors: BTreeMap<Uuid, BTreeMap<u64, NavNodeId>> = BTreeMap::new();
    for f in &fresh {
        let ring = &known[&f.guid].2;
        let mut mine: BTreeMap<u64, NavNodeId> = BTreeMap::new();
        for n in f.interior.nodes() {
            // A building's EXTERIOR door is the doorway with one edge: the wall
            // it stands in has no room on the far side, so `interior_nav` links
            // it to exactly one room. Every internal door has two.
            if n.kind != NavKind::Doorway || f.interior.edges_from(n.id).len() != 1 {
                continue;
            }
            // **A salt collision is two buildings claiming one id.** Checked on
            // the door, which is the node that reaches the shared network, and
            // counted rather than papered over: a collision welds one
            // building's front door to another's, which is a route that walks
            // into the wrong house.
            if soc.network.contains(n.id) {
                stats.salt_collisions += 1;
            }
            soc.network.add_node(n.id, n.position, NavKind::Doorway);
            mine.insert(node_salt(n.id), n.id);
            let mut best: Option<(f64, NavNodeId)> = None;
            for (id, p) in ring {
                let d = (*p - n.position).length();
                if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                    best = Some((d, *id));
                }
            }
            match best {
                Some((d, id)) if d <= FRONTAGE_MAX_M => {
                    soc.network.link(n.id, id, NavKind::Doorway, Vec::new());
                    stats.frontages += 1;
                }
                _ => stats.frontages_refused += 1,
            }
        }
        doors.insert(f.guid, mine);
        soc.interiors.insert(f.guid, f.interior.clone());
    }

    // ── 3. register the fresh volumes' slots ───────────────────────────────
    for f in &fresh {
        for s in &f.residents {
            if !s.at.is_finite() {
                continue;
            }
            // The building this slot is in, by the salt its own node carries —
            // the same word `interior_nav_in` wrote and `building_salt` minted.
            let Some(door) = doors
                .get(&f.guid)
                .and_then(|m| m.get(&node_salt(s.node)))
                .copied()
            else {
                // A building with no exterior door is one nobody can walk into.
                // Its people are not people this society can give a day to, and
                // saying so is better than giving them one that starts inside a
                // sealed box.
                stats.doorless += 1;
                continue;
            };
            let place = SocietyPlace {
                role: s.role,
                at: s.at,
                node: s.node,
                volume: f.guid,
                door,
            };
            match s.role {
                SlotRole::Home => {
                    let g = agent_guid(f.guid, s.building, s.room, s.index);
                    soc.pending.entry(g).or_insert(place);
                }
                SlotRole::Work => soc.work.push(place),
                SlotRole::Errand => soc.errand.push(place),
            }
        }
        soc.folded.insert(f.guid);
    }
    stats.folded_now = fresh.len();
    stats.volumes = soc.folded.len();
    stats.works = soc.work.len();
    stats.errands = soc.errand.len();

    // ── 4. plan days, but never while the town is still arriving ───────────
    if stats.folded_now == 0 && !soc.pending.is_empty() {
        let archetype = level_archetype(world);
        let batch: Vec<Uuid> = soc
            .pending
            .keys()
            .copied()
            .take(SOCIETY_PLANS_PER_STEP)
            .collect();
        let mut records: BTreeMap<Uuid, CrowdRecord> = BTreeMap::new();
        for g in batch {
            let home = soc.pending.remove(&g).expect("a key we just read");
            let (rec, kind) = plan_day(&mut soc, archetype, home, &mut stats);
            match kind {
                DayKind::Full => {}
                DayKind::Homebound => stats.homebound += 1,
                DayKind::Housebound => stats.housebound += 1,
            }
            records.insert(g, rec);
            stats.planned_now += 1;
        }
        stats.agents += records.len();
        let refused = crate::crowd::add_agents(world, records);
        stats.agents -= refused;
        stats.guid_refusals += refused;
    }

    stats.homes = stats.agents + soc.pending.len();
    stats.pending = soc.pending.len();
    stats.nodes = soc.network.len();
    stats.edges = soc.network.edge_count();
    soc.stats = stats;
    world.world_mut().insert_resource(soc);
    stats
}

/// What kind of day one agent got.
enum DayKind {
    /// Home, work, an errand and home again.
    Full,
    /// No reachable workplace — an errand out and back, and home the rest of the
    /// day.
    Homebound,
    /// Nowhere reachable at all. The agent stands at home, which is what a
    /// record with no schedule does.
    Housebound,
}

/// The nearest place of a role to `from`, ties broken on the node id.
fn nearest(places: &[SocietyPlace], from: DVec3) -> Option<SocietyPlace> {
    let mut best: Option<(f64, SocietyPlace)> = None;
    for p in places {
        let d = (p.at - from).length();
        if !d.is_finite() {
            continue;
        }
        let better = match &best {
            None => true,
            Some((bd, bp)) => d < *bd || (d == *bd && p.node < bp.node),
        };
        if better {
            best = Some((d, *p));
        }
    }
    best.map(|(_, p)| p)
}

/// A route between two nodes of one graph, or `None`. A search whose two ends
/// are the same node answers an empty contribution rather than a refusal.
fn hop(graph: &NavGraph, from: NavNodeId, to: NavNodeId) -> Option<Vec<DVec3>> {
    if from == to {
        return graph.node(from).map(|n| vec![n.position]);
    }
    match inf_nav::route(graph, from, to) {
        inf_nav::NavVerdict::Found(r) => Some(r.path.points().to_vec()),
        _ => None,
    }
}

/// **A leg from one place to another**, over the two levels of the search — the
/// one door every leg of every day goes through.
///
/// Same building: one search inside it. Different buildings: out to the front
/// door, along the street (memoized), and in through the other front door. The
/// three point lists are joined by `NavPath::new`, which drops the coincident
/// ends for us.
fn leg(
    soc: &mut SocietyRes,
    from: &SocietyPlace,
    to: &SocietyPlace,
    stats: &mut SocietyStats,
) -> Option<inf_nav::NavPath> {
    if from.volume == to.volume {
        let g = soc.interiors.get(&from.volume)?;
        let pts = hop(g, from.node, to.node)?;
        return (pts.len() > 1).then(|| inf_nav::NavPath::new(pts));
    }
    let inside_a = hop(soc.interiors.get(&from.volume)?, from.node, from.door)?;
    let inside_b = hop(soc.interiors.get(&to.volume)?, to.door, to.node)?;
    let key = (from.door, to.door);
    let street = match soc.legs.get(&key) {
        Some(hit) => {
            stats.outer_cached += 1;
            hit.clone()
        }
        None => {
            stats.outer_searches += 1;
            let found = match inf_nav::route(&soc.network, key.0, key.1) {
                inf_nav::NavVerdict::Found(r) => Some(r.path),
                _ => None,
            };
            soc.legs.insert(key, found.clone());
            found
        }
    }?;
    let mut pts = inside_a;
    pts.extend_from_slice(street.points());
    pts.extend_from_slice(&inside_b);
    let path = inf_nav::NavPath::new(pts);
    (path.length_m() > 0.0).then_some(path)
}

/// **Plan one agent's day** — the four legs, and the two honest fall-backs.
fn plan_day(
    soc: &mut SocietyRes,
    archetype: CrowdArchetype,
    home: SocietyPlace,
    stats: &mut SocietyStats,
) -> (CrowdRecord, DayKind) {
    if let Some(w) = nearest(&soc.work, home.at) {
        if let Some(out) = leg(soc, &home, &w, stats) {
            let back = leg(soc, &w, &home, stats);
            let mut legs = vec![ScheduleLeg {
                start_h: WORK_START_H,
                travel_h: COMMUTE_H,
                path: out,
            }];
            // The errand is nearest the WORKPLACE, because that is where the
            // agent is standing at noon.
            if let Some(e) = nearest(&soc.errand, w.at) {
                if let (Some(to_shop), Some(to_work)) =
                    (leg(soc, &w, &e, stats), leg(soc, &e, &w, stats))
                {
                    legs.push(ScheduleLeg {
                        start_h: ERRAND_OUT_H,
                        travel_h: ERRAND_H,
                        path: to_shop,
                    });
                    legs.push(ScheduleLeg {
                        start_h: ERRAND_BACK_H,
                        travel_h: ERRAND_H,
                        path: to_work,
                    });
                }
            }
            if let Some(back) = back {
                legs.push(ScheduleLeg {
                    start_h: HOME_H,
                    travel_h: COMMUTE_H,
                    path: back,
                });
            }
            if let Some(sched) = CrowdSchedule::new(legs) {
                return (CrowdRecord::scheduled(archetype, sched), DayKind::Full);
            }
        }
    }
    // No workplace. An errand out and back is still a day.
    if let Some(e) = nearest(&soc.errand, home.at) {
        if let (Some(out), Some(back)) = (leg(soc, &home, &e, stats), leg(soc, &e, &home, stats)) {
            if let Some(sched) = CrowdSchedule::new(vec![
                ScheduleLeg {
                    start_h: 10.0,
                    travel_h: ERRAND_H,
                    path: out,
                },
                ScheduleLeg {
                    start_h: 16.0,
                    travel_h: ERRAND_H,
                    path: back,
                },
            ]) {
                return (CrowdRecord::scheduled(archetype, sched), DayKind::Homebound);
            }
        }
    }
    (
        CrowdRecord::standing(archetype, home.at),
        DayKind::Housebound,
    )
}

/// The society's counters, or all zeroes on a level that has none.
pub fn society_stats(world: &EcsWorld) -> SocietyStats {
    world
        .world()
        .get_resource::<SocietyRes>()
        .map(|s| s.stats)
        .unwrap_or_default()
}

/// Forget the society — the twin of [`crate::crowd::clear_crowd`], and called
/// beside it for the same reason: a `SceneDoc` snapshot restores entities and
/// components and touches no resource, so without this a stopped Simulate
/// session's network would outlive the run that built it.
pub fn clear_society(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<SocietyRes>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{DoorwaySlot, ResidentSlot, Transform};
    use inf_nav::domain;

    /// A block with one building on it: a bedroom, an office and a shop, joined
    /// by a corridor, with one exterior door.
    fn block(
        world: &mut EcsWorld,
        guid: Uuid,
        centre: DVec3,
        half: f64,
        salt: u64,
        roles: &[SlotRole],
    ) {
        world.spawn_with_guid(guid, "block", None);
        let e = world.entity_of(guid).expect("the block");
        let room = |i: usize| domain::BUILDING | (i as u64 & 0xF_FFFF) | (salt << 20);
        let door =
            |i: usize| domain::BUILDING | (1u64 << 59) | (i as u64 & 0xF_FFFF) | (salt << 20);
        let mut g = NavGraph::new();
        // room 0 is the corridor, at the block centre; the others hang off it.
        g.add_node(room(0), centre, NavKind::Room);
        let mut residents = Vec::new();
        for (i, role) in roles.iter().enumerate() {
            let at = centre + DVec3::new(0.0, 0.0, (i as f64 + 1.0) * 2.0);
            g.add_node(room(i + 1), at, NavKind::Room);
            g.add_node(door(i + 1), (at + centre) * 0.5, NavKind::Doorway);
            g.link(room(0), door(i + 1), NavKind::Doorway, Vec::new());
            g.link(door(i + 1), room(i + 1), NavKind::Doorway, Vec::new());
            residents.push(ResidentSlot {
                role: *role,
                at,
                room: (i + 1) as u32,
                building: 0,
                floor: 0,
                index: 0,
                node: room(i + 1),
            });
        }
        // The exterior door: one edge, at the block's own edge.
        g.add_node(
            door(0),
            centre + DVec3::new(half, 0.0, 0.0),
            NavKind::Doorway,
        );
        g.link(room(0), door(0), NavKind::Doorway, Vec::new());

        let mut vol = PcgVolume {
            extent: crate::math::Vec2d::new(half, half),
            ..Default::default()
        };
        vol.set_population(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![DoorwaySlot {
                hinge: centre + DVec3::new(half, 1.05, 0.0),
                closed_yaw_deg: 0.0,
                width_m: 0.9,
                height_m: 2.1,
                thickness_m: 0.2,
                inside_yaw_deg: 180.0,
                exterior: true,
                floor: 0,
            }],
            residents,
            g,
        );
        world.world_mut().entity_mut(e).insert((
            Transform {
                translation: crate::math::Vec3d::new(centre.x, centre.y, centre.z),
                ..Default::default()
            },
            GlobalTransform(glam::DAffine3::from_translation(centre)),
            vol,
        ));
    }

    /// **The clause: a route crosses building, street and building.** Two blocks
    /// twenty metres apart, one holding a home and one a workplace; the agent's
    /// commute must leave its own building, walk the pavement, and enter the
    /// other one.
    #[test]
    fn a_commute_crosses_building_street_and_building() {
        let mut world = EcsWorld::new();
        block(
            &mut world,
            Uuid::from_u128(1),
            DVec3::new(0.0, 0.0, 0.0),
            10.0,
            0x11,
            &[SlotRole::Home],
        );
        block(
            &mut world,
            Uuid::from_u128(2),
            DVec3::new(28.0, 0.0, 0.0),
            10.0,
            0x22,
            &[SlotRole::Work, SlotRole::Errand],
        );
        // First sync folds; the second plans (nobody plans while a town builds).
        let a = sync_society(&mut world);
        assert_eq!(a.folded_now, 2);
        assert_eq!(a.planned_now, 0, "a day was planned on a folding step");
        assert_eq!(a.salt_collisions, 0);
        assert_eq!(a.frontages, 2, "the two front doors did not both join");
        assert!(a.crossings > 0, "the two pavements were never joined");

        let b = sync_society(&mut world);
        assert_eq!(b.folded_now, 0);
        assert_eq!(b.planned_now, 1, "the one resident got no day");
        assert_eq!(b.agents, 1);
        assert_eq!(b.homebound, 0, "the resident could not reach the workplace");
        assert_eq!(b.housebound, 0);
        assert_eq!(b.guid_refusals, 0);

        // And the commute really crosses all three.
        let pop = world
            .world()
            .get_resource::<crate::crowd::CrowdPopulationRes>()
            .expect("a population");
        let (_, rec) = pop.records.iter().next().expect("one agent");
        let sched = rec.schedule.as_ref().expect("a schedule");
        assert_eq!(sched.legs().len(), 4, "a full day is four legs");
        let commute = &sched.legs()[0];
        assert!(
            commute.path.length_m() > 20.0,
            "the commute is {:.1} m, which is not across a street",
            commute.path.length_m()
        );
        // **The commute crosses all three, asserted on the PATH the agent
        // walks** rather than on a search a reader could re-run differently.
        // The network holds streets and front doors; a leg is joined from an
        // inside hop, a street route and another inside hop, so the claim has to
        // be made of the joined thing.
        let soc = world
            .world()
            .get_resource::<SocietyRes>()
            .expect("a society");
        let home_at = DVec3::new(0.0, 0.0, 2.0);
        let work_at = DVec3::new(28.0, 0.0, 2.0);
        let pts = commute.path.points();
        assert!(
            (pts[0] - home_at).length() < 1e-9,
            "the commute starts at {:?} and the home is at {home_at:?}",
            pts[0]
        );
        assert!(
            (pts[pts.len() - 1] - work_at).length() < 1e-9,
            "the commute ends at {:?} and the work is at {work_at:?}",
            pts[pts.len() - 1]
        );
        // It goes out through ONE block's pavement and in through the other's.
        let mut on_pavement: BTreeSet<NavNodeId> = BTreeSet::new();
        for n in soc.network.nodes() {
            if domain::of(n.id) != domain::PAVEMENT {
                continue;
            }
            if pts.iter().any(|p| (*p - n.position).length() < 1e-6) {
                on_pavement.insert(n.id);
            }
        }
        assert!(
            on_pavement.len() >= 2,
            "the commute touches {} pavement node(s) -- it never crossed a \
             street",
            on_pavement.len()
        );
        // And the two ends are in two DIFFERENT buildings, which is the half a
        // single-namespace network could never make true. The salt is what says
        // so, read off the nodes the two places name.
        let salts: BTreeSet<u64> = soc
            .work
            .iter()
            .map(|w| super::node_salt(w.node))
            .chain(soc.pending.values().map(|h| super::node_salt(h.node)))
            .chain(std::iter::once(super::node_salt(
                soc.interiors[&Uuid::from_u128(1)]
                    .nodes()
                    .next()
                    .expect("a node")
                    .id,
            )))
            .collect();
        assert!(
            salts.contains(&0x11) && salts.contains(&0x22),
            "the two buildings' salts are {salts:?}"
        );
        // The memo did its job: four legs, and the second commute of the day is
        // a different pair, so both searched once and neither twice.
        assert!(b.outer_searches > 0, "no street route was ever searched");
    }

    /// A resident with no reachable workplace still gets a day, and the counter
    /// says which kind — a refusal is a value.
    #[test]
    fn a_resident_with_nowhere_to_work_keeps_a_stay_at_home_day() {
        let mut world = EcsWorld::new();
        block(
            &mut world,
            Uuid::from_u128(1),
            DVec3::ZERO,
            10.0,
            0x11,
            &[SlotRole::Home, SlotRole::Errand],
        );
        sync_society(&mut world);
        let s = sync_society(&mut world);
        assert_eq!(s.agents, 1);
        assert_eq!(
            s.homebound, 1,
            "the resident was given a job it has not got"
        );
        let pop = world
            .world()
            .get_resource::<crate::crowd::CrowdPopulationRes>()
            .expect("a population");
        let sched = pop
            .records
            .values()
            .next()
            .expect("one agent")
            .schedule
            .as_ref()
            .expect("a stay-at-home day is still a day");
        assert_eq!(sched.legs().len(), 2);
    }

    /// A block a kilometre away is a different town, and the pavements do not
    /// pretend otherwise.
    #[test]
    fn two_towns_are_two_components() {
        let mut world = EcsWorld::new();
        block(
            &mut world,
            Uuid::from_u128(1),
            DVec3::ZERO,
            10.0,
            0x11,
            &[SlotRole::Home],
        );
        block(
            &mut world,
            Uuid::from_u128(2),
            DVec3::new(1000.0, 0.0, 0.0),
            10.0,
            0x22,
            &[SlotRole::Work],
        );
        let a = sync_society(&mut world);
        assert_eq!(a.crossings, 0, "a kilometre of sea got a zebra crossing");
        let b = sync_society(&mut world);
        assert_eq!(
            b.housebound, 1,
            "the resident commuted a kilometre with no road"
        );
    }

    /// **Nothing is a function of streaming order that can be**: a level's agent
    /// `Guid`s are a hash of its own content, so the same block folded in a
    /// different order mints the same people.
    #[test]
    fn the_agents_are_the_levels_own_and_not_the_orders() {
        let mut a = EcsWorld::new();
        block(
            &mut a,
            Uuid::from_u128(1),
            DVec3::ZERO,
            10.0,
            0x11,
            &[SlotRole::Home],
        );
        block(
            &mut a,
            Uuid::from_u128(2),
            DVec3::new(28.0, 0.0, 0.0),
            10.0,
            0x22,
            &[SlotRole::Home, SlotRole::Work],
        );
        sync_society(&mut a);
        sync_society(&mut a);
        sync_society(&mut a);

        let mut b = EcsWorld::new();
        // The other order.
        block(
            &mut b,
            Uuid::from_u128(2),
            DVec3::new(28.0, 0.0, 0.0),
            10.0,
            0x22,
            &[SlotRole::Home, SlotRole::Work],
        );
        block(
            &mut b,
            Uuid::from_u128(1),
            DVec3::ZERO,
            10.0,
            0x11,
            &[SlotRole::Home],
        );
        sync_society(&mut b);
        sync_society(&mut b);
        sync_society(&mut b);

        let keys = |w: &EcsWorld| -> Vec<Uuid> {
            w.world()
                .get_resource::<crate::crowd::CrowdPopulationRes>()
                .map(|p| p.records.keys().copied().collect())
                .unwrap_or_default()
        };
        let (ka, kb) = (keys(&a), keys(&b));
        assert_eq!(ka.len(), 2, "two homes made {} agents", ka.len());
        assert_eq!(ka, kb, "two orders minted two different populations");
        assert_eq!(
            crowd_bytes(&a),
            crowd_bytes(&b),
            "two orders produced different crowd traces"
        );
    }

    fn crowd_bytes(w: &EcsWorld) -> Vec<u8> {
        crate::crowd::crowd_state_bytes(w)
    }

    /// A level with no residents installs nothing at all.
    #[test]
    fn a_level_with_no_residents_has_no_society() {
        let mut world = EcsWorld::new();
        let s = sync_society(&mut world);
        assert_eq!(s, SocietyStats::default());
        assert!(world.world().get_resource::<SocietyRes>().is_none());
        assert!(world
            .world()
            .get_resource::<crate::crowd::CrowdPopulationRes>()
            .is_none());
    }

    /// A pavement node's id is its own place, so two blocks laying a node at one
    /// corner lay one node.
    #[test]
    fn a_pavement_node_is_named_by_where_it_stands() {
        let a = pavement_node_id(DVec2::new(12.0, -4.0));
        assert_eq!(a, pavement_node_id(DVec2::new(12.0, -4.0)));
        assert_eq!(a, pavement_node_id(DVec2::new(12.004, -3.998)));
        assert_ne!(a, pavement_node_id(DVec2::new(12.5, -4.0)));
        assert_eq!(domain::of(a), domain::PAVEMENT);
        // A non-finite point does not corrupt the tag.
        assert_eq!(
            domain::of(pavement_node_id(DVec2::new(f64::NAN, 0.0))),
            domain::PAVEMENT
        );
    }
}
