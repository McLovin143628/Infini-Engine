//! **The settlements** (island wave I8a) — the generator that turns seven empty
//! levelled sites into two cities and five towns.
//!
//! # What wave I7 left, and what this is
//!
//! Wave I7's own carried remainder 12 reads: *"the site pads are terraces, not
//! settlements. Seven sites, 2.34 km² of levelled ground and 5.8 % of the land
//! reserved urban — and nothing standing on any of it."* This module is the
//! thing that stands on it.
//!
//! # It reads the COMMITTED DESIGN and nothing else
//!
//! Exactly the rule [`crate::island`] states at length: the level is a committed
//! document, so everything in it has to come from the recipe, the committed
//! layers and a GUID derived from the island's own name. A settlement plan is
//! therefore a **pure function of `IslandDesign`** — the sites, the road routes
//! and the coastline rings — with no elevation anywhere in it. The building
//! datum is resolved at *evaluation* time from the terrain under each lot's own
//! centre (`Ground::Terrain`), which is the one place the ground is allowed to
//! be asked.
//!
//! `the_settlement_generator_is_authored_from_committed_design_alone` runs the
//! same allowlist scan over this file that
//! `the_level_is_authored_from_committed_design_alone` runs over `island.rs`.
//!
//! # The shape
//!
//! ```text
//!   site (centre, radius) ─▶ street GRID on the world axes, a line through the
//!                            centre on both axes
//!                         ─▶ BLOCKS between the lines, refused when they leave
//!                            the reservation circle or the coastline
//!                         ─▶ ZONE per block (distance from the centre + the
//!                            highway cluster)
//!                         ─▶ one PcgVolume naming that zone's `.inf_pcg`
//!                              ─▶ `building.lots` cuts the block into lots
//!                              ─▶ `building.plan` stands one building on each
//! ```
//!
//! Nothing here is one building: the whole reason IB-2c's
//! [`subdivide_block`](inf_pcg::subdivide_block) exists is that a block is the
//! unit an author draws and the lots inside it are a *rule*. A settlement is
//! ~52 (city) or ~12-16 (town) `PcgVolume` records in the committed level, and
//! each one becomes a dozen buildings when its cell activates.
//!
//! # Why the grid runs on the world axes
//!
//! A `PcgVolume` is a **centre and an axis-aligned half-extent**
//! (`GrammarContext` carries `center` and `extent` and no rotation), so a block
//! that is a volume's own box is axis-aligned by construction. Giving a
//! settlement its own bearing would mean either an oriented volume — an engine
//! schema move, ruled against for this wave — or one committed block polygon per
//! block, which is committing *geometry* where the small-committed-folder law
//! says commit *rules*. The bearing is therefore stated as a bound rather than
//! taken, and it is named in the wave's routed list.
//!
//! # How the grid JOINS the island's road network
//!
//! It does not stitch anything: [`plan_network`](inf_island::plan_network) routes
//! **centre to centre**, so every highway and arterial that reaches a settlement
//! terminates at the site's own `(x, z)`. The grid puts a street line through
//! that point on both axes, so the arriving route lands on the local network's
//! central crossroads. `the_settlement_grid_meets_the_island_road_network`
//! measures the distance from every route endpoint to the nearest street
//! centreline and prints it.
//!
//! **The streets are a plan, not a surface, in this slice.** They decide where
//! the blocks are and they carry the join; they are not drawn. The island's road
//! mesh is built by draping the committed road layer at the *terrain's* own
//! pitch (1 m), because a road that follows real ground has a chord error at any
//! coarser step — and `roads::build_mesh` takes **one** `ground_step_m` for one
//! layer. Settlement streets sit on a levelled pad, where the chord error is
//! zero at any step (the `phase30-city` measurement: 0.000000 m), so drawing
//! them honestly wants a *second* surface at a coarse step. At the island's 1 m
//! pitch the grid below is [`Settlement::street_km`] of centreline, and that is
//! what the wave's routed list carries beside I8b's sidewalk/kerb item.

use glam::{DVec2, DVec3};
use inf_pcg::hash::Hash64;
use inf_pcg::{ArchetypeId, LotRules};
use uuid::Uuid;

use inf_island::{IslandDesign, Route, Site, SiteKind};

/// A city block's own side, metres — the space between two street centrelines,
/// less the street.
///
/// One hundred metres is the North-American downtown block the `phase30-city`
/// fixture already measures its thousand buildings on
/// (`CITY_BLOCK_M` there is 108 × 78), and it is what the lot rules below are
/// sized against: a 100 m block at 30 m of office frontage is three lots across.
pub const CITY_BLOCK_M: f64 = 100.0;

/// The street reserve between two city blocks, metres. Two 3.5 m lanes each way
/// plus parking and footway either side.
pub const CITY_STREET_M: f64 = 20.0;

/// A town block's own side, metres. Smaller than a city's because a town of
/// 210 m radius has room for exactly one ring of blocks at the city's pitch,
/// which is a crossroads rather than a town.
pub const TOWN_BLOCK_M: f64 = 60.0;

/// The street reserve between two town blocks, metres.
pub const TOWN_STREET_M: f64 = 16.0;

/// How many blocks a city's one industrial cluster holds.
///
/// **One cluster per city, near the highway** — the zoning table's own sentence.
/// Four blocks is 4 × 100 m² of yard, which at the industrial lot rules below is
/// sixteen sheds: a works, not a single warehouse and not a district.
pub const CITY_INDUSTRIAL_BLOCKS: usize = 4;

/// The innermost ring an industrial block may take, for a settlement whose
/// outermost ring is `max_ring`.
///
/// **The outer half, never the core**: a city's centre is its offices and its
/// hotels, and putting the works on the main crossroads would be a zoning table
/// that says one thing and does another. Stated as a fraction of the
/// settlement's own depth rather than as the constant `2` it started as, because
/// a constant makes the rule unreachable in any settlement with fewer than three
/// rings — which is every settlement the CI fixture has, i.e. the one place the
/// cluster would have been gated.
///
/// Harbour City is four rings deep, so this is 2 either way; the fixture's city
/// is two, so it is 1.
pub fn industrial_min_ring(max_ring: u32) -> u32 {
    max_ring.div_ceil(2).max(1)
}

/// Grid lines a settlement may lay each way from its own centre.
///
/// A ceiling rather than a preference, on `subdivide_block`'s own
/// `MAX_LOTS_PER_AXIS` precedent: `Site::radius_m` is author-supplied and the
/// recipe checks it only for finiteness, so a mis-typed `600000` would ask the
/// cell loop for a hundred million candidates before the pad test refused every
/// one of them. Sixty-four lines is 7.7 km at the city pitch — wider than any
/// world this recipe format can describe.
pub const MAX_GRID_LINES: u32 = 64;

/// The furthest grid line [`Settlement::street_graph`] will mint an index for,
/// either way from the centre.
///
/// One past [`MAX_GRID_LINES`], which is what `plan_site`'s own `reach` is at
/// its widest. A ceiling on an id field rather than a preference — the same role
/// the line count's own clamp plays one level down, and for the same reason:
/// `Site::radius_m` is author input.
const MAX_LINE_INDEX: i64 = MAX_GRID_LINES as i64 + 1;

/// What a signed grid line index is shifted by to become a non-negative id
/// field.
///
/// One past [`MAX_LINE_INDEX`], so index `0` sits free below the range of real
/// lines and [`AXIS_MAX_INDEX`] sits free above it — which is where a street's
/// two **ends** go, since a line stops at `±buildable_m` and that is a whole
/// number of blocks from the centre only by coincidence.
const LATTICE_BIAS: i64 = MAX_LINE_INDEX + 1;

/// The index of a street's low end — below every grid line's own index.
const AXIS_MIN_INDEX: u64 = 0;

/// The index of a street's high end — above every grid line's own index.
const AXIS_MAX_INDEX: u64 = (2 * LATTICE_BIAS + 1) as u64;

/// One field of a street node's id: twenty bits. See
/// [`Settlement::street_graph`] for the layout.
const NODE_FIELD_MASK: u64 = (1 << 20) - 1;

/// The most blocks one city's industrial cluster may take, as a share of the
/// blocks it is allowed to sit on.
///
/// A quarter, capped at [`CITY_INDUSTRIAL_BLOCKS`]. Without the share a small
/// settlement's whole outer ring becomes a works: the fixture's city has eight
/// eligible blocks and four of them would be half the town.
pub const INDUSTRIAL_SHARE: usize = 4;

/// **A settlement's venues, in the order the strip assigns them** (wave VEN1a).
///
/// A city gets all three; a town gets a bar. That ordering is not decorative —
/// the blocks are handed out nearest-first, so the bar is the one closest to
/// the centre and the strip club is the one furthest out, which is where they
/// are in every town that has all three.
///
/// The list is per SITE KIND rather than per radius because it is a statement
/// about what a settlement *is*, not about how much room it has: a hamlet with
/// a nightclub is stranger than a city without one.
fn venue_strip(kind: SiteKind) -> &'static [ArchetypeId] {
    const CITY: &[ArchetypeId] = &[
        ArchetypeId::Bar,
        ArchetypeId::Nightclub,
        ArchetypeId::StripClub,
    ];
    const TOWN: &[ArchetypeId] = &[ArchetypeId::Bar];
    match kind {
        SiteKind::City => CITY,
        _ => TOWN,
    }
}

/// The innermost ring a venue may take (wave VEN1a).
///
/// A city's ring 0 is its office core — the most expensive ground in the
/// settlement and the one the zone table gives to offices and hotels — so the
/// strip sits one ring out. A town has no such core: its ring 0 *is* its high
/// street, which is exactly where its bar belongs.
fn venue_min_ring(kind: SiteKind) -> u32 {
    match kind {
        SiteKind::City => 1,
        _ => 0,
    }
}

/// At most one block in this many becomes a venue (wave VEN1a) — the
/// industrial cluster's `INDUSTRIAL_SHARE` guard, for its reason.
///
/// A settlement of four blocks that spent three of them on nightlife is not a
/// settlement. The share bites only on the very smallest reservations, where it
/// costs the strip its club and then its bar in that order.
const VENUE_SHARE: usize = 3;

/// **A settlement's institutions, in the order the strip assigns them** (wave
/// EMS1).
///
/// A city gets all four; a town gets a fire hall and a clinic. That is the
/// answer to *what does a settlement of this size actually have*, and the order
/// is not decorative for [`venue_strip`]'s reason exactly — the blocks are
/// handed out nearest-first, so the first entry is the one closest to the
/// centre.
///
/// # Why a town gets these two and not the other two
///
/// A **fire hall** is the one civic building a settlement of any size has,
/// because the alternative is that it burns down; it is first so it is central,
/// which is where a hall goes when the thing that matters about it is how long
/// it takes to leave. A **clinic** is a hospital a town can afford: one or two
/// storeys, a waiting room and consulting rooms, on a high-street lot — which is
/// exactly what `zone_lots` gives it and why it is the institution that scales
/// down.
///
/// A town gets no **hospital** because a hospital is a 52 × 50 m lot and five
/// storeys and a town's whole grid is 76 m on a side; and no **police station**
/// because a station without a city to police is a building with a cell block
/// and nobody in it. Both are city facts, and this list is where they are
/// stated rather than emerging from a weight nobody can read.
fn civic_strip(kind: SiteKind) -> &'static [ArchetypeId] {
    const CITY: &[ArchetypeId] = &[
        ArchetypeId::FireHall,
        ArchetypeId::Hospital,
        ArchetypeId::PoliceStation,
        ArchetypeId::Clinic,
    ];
    const TOWN: &[ArchetypeId] = &[ArchetypeId::FireHall, ArchetypeId::Clinic];
    match kind {
        SiteKind::City => CITY,
        _ => TOWN,
    }
}

/// The innermost ring an institution may take (wave EMS1).
///
/// **One ring out of a city, ring 0 of a town — [`venue_min_ring`] exactly.**
///
/// The first draft of this rule was `0` everywhere, on the argument that a fire
/// hall does not care what the ground costs, it cares how long the appliance
/// takes to reach the far side of the city. That argument is right about the
/// appliance and wrong about the city: ring 0 IS a city's office core, it is the
/// most expensive ground in the settlement, and a hall on the crossroads is a
/// hall that has bought the one block a downtown is made of. Halls and emergency
/// rooms sit a block off the main street in every city that has them, and a ring
/// in a city is 120 m.
///
/// **The island gate is what made this get looked at**, and the way it failed is
/// worth keeping: taking the block nearest the crossroads took the settlement's
/// only guaranteed-multi-storey block — the subject
/// `pie_equals_shipping_when_an_npc_walks_across_town` walks into — and the
/// arm reported it as *"no street line is on the same storey as its own
/// buildings"*, which is a true sentence about a building nobody meant to
/// choose. A rule that moves another wave's subject is a rule worth re-reading.
///
/// A town keeps ring 0: its ring 0 is its high street, not a core, which is
/// `venue_min_ring`'s own reasoning for the same number.
fn civic_min_ring(kind: SiteKind) -> u32 {
    match kind {
        SiteKind::City => 1,
        _ => 0,
    }
}

/// At most one block in this many becomes an institution (wave EMS1) — the
/// `VENUE_SHARE` guard, at `VENUE_SHARE`'s own value.
///
/// Three, and not the four the first draft carried. The extra rung was there to
/// stop the civic strip and the nightlife strip between them spending too much
/// of a small settlement — and with [`civic_min_ring`] corrected to keep a
/// city's core, it costs the CI fixture its only institution instead: a
/// four-block camp with one bar has three eligible blocks, and three over four
/// is none. A share is a fraction of a settlement, and a quarter and a third are
/// the same answer everywhere the two rules can both bite.
const CIVIC_SHARE: usize = 3;

/// **Which archetypes are furnished** (island wave I8a, ruling 3).
///
/// The orchestrator's ruling was *measure, then decide, default ON*, and the
/// measurement is `the_furnish_battery_prices_a_city_block_at_island_scale` in
/// `runtime/inf-player/tests/island_gate.rs`. What it found is in the wave's
/// ledger; what it decided is here, in one place, read by every zone
/// document so a reader cannot find two answers.
///
/// The split is not a compromise for its own sake — it is the shape the
/// measurement implies. Furniture is per **room**, so its cost scales with
/// storeys, and the archetypes a player walks into on foot (a house, a shop, an
/// estate) are the one- to four-storey ones. A ten-storey hotel is 5 × the rooms
/// of a house for a lobby nobody has walked past yet.
pub fn furnishes(a: ArchetypeId) -> bool {
    // **A venue is always furnished** (wave VEN1a), and it is not an exception
    // to the rule above -- it is the rule. The measurement's argument is that
    // furniture costs per room and is worth it for the archetypes a player
    // walks into on foot. A club's fittings ARE the club: an unfurnished
    // nightclub is an empty concrete box with a sign on it, which is strictly
    // worse than not placing one. The venues are also one to two storeys, which
    // is the other half of the same argument.
    //
    // **An institution is always furnished** (wave EMS1), and it is not a third
    // exception either — it is the strongest case of the rule. A venue's
    // fittings are the venue; an institution's fittings are its *staff*. The
    // front counter is a `Placement::Run` and a `Tend` station is derived where
    // a run is PLACED, so an unfurnished police station has no desk, and a
    // station with no desk has nobody behind it, and a hospital with no beds is
    // an office block with a white sign on it. Turning furniture off for these
    // four would not make them cheaper — it would make them empty.
    //
    // The cost argument holds too: a `Clinic` is one to two storeys, a
    // `PoliceStation` and a `FireHall` two to three, and only the `Hospital`
    // reaches five. Its price is on the record in the furnish battery's table
    // beside the other thirteen.
    a.is_venue()
        || a.is_institution()
        || matches!(
            a,
            ArchetypeId::House | ArchetypeId::Shop | ArchetypeId::Estate
        )
}

/// One street centreline of a settlement's local grid, in world XZ.
///
/// A plan, not a surface — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Street {
    pub a: DVec2,
    pub b: DVec2,
    /// `true` for the two lines through the site's own centre — the pair every
    /// arriving island route lands on.
    pub main: bool,
}

impl Street {
    /// Length in metres.
    pub fn length_m(&self) -> f64 {
        (self.b - self.a).length()
    }

    /// The shortest distance from `p` to this centreline, metres.
    pub fn distance_to(&self, p: DVec2) -> f64 {
        let d = self.b - self.a;
        let len2 = d.length_squared();
        if len2 <= 0.0 {
            return (p - self.a).length();
        }
        let t = ((p - self.a).dot(d) / len2).clamp(0.0, 1.0);
        (p - (self.a + d * t)).length()
    }
}

/// One block of a settlement: an axis-aligned rectangle, a zone, and the seed
/// the buildings on it draw from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Block {
    /// Index into `IslandRecipe::sites`.
    pub site: usize,
    /// Grid indices. Block `col` spans `[centre.x + col·pitch + half_street,
    /// centre.x + (col+1)·pitch − half_street]`, so `col = 0` and `col = −1` are
    /// the two blocks touching the central crossroads.
    pub col: i32,
    pub row: i32,
    /// Chebyshev ring: `0` for the four blocks on the central crossroads.
    pub ring: u32,
    /// The block's centre in world XZ.
    pub centre: DVec2,
    /// Half-extent in world XZ — the `PcgVolume::extent` this block becomes.
    pub half: DVec2,
    /// The palette this block builds from, and therefore which zone `.inf_pcg`
    /// its volume names.
    pub archetype: ArchetypeId,
    /// The volume's own seed — what makes a hundred and seventy volumes sharing
    /// seven graphs a hundred and seventy different blocks.
    pub seed: u32,
}

impl Block {
    /// The block's four corners in world XZ.
    pub fn corners(&self) -> [DVec2; 4] {
        let (a, b) = (self.centre - self.half, self.centre + self.half);
        [a, DVec2::new(b.x, a.y), b, DVec2::new(a.x, b.y)]
    }

    /// Floor area in m².
    pub fn area_m2(&self) -> f64 {
        4.0 * self.half.x * self.half.y
    }
}

/// One settlement's whole plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Settlement {
    /// Index into `IslandRecipe::sites`.
    pub site: usize,
    pub name: String,
    pub kind: SiteKind,
    pub centre: DVec2,
    /// The site's own reserved radius — the pad the carve levels **and** the
    /// circle the biome map paints urban. One number, two jobs, and that is why
    /// a block inside it stands on bare ground rather than in a forest.
    pub radius_m: f64,
    /// The radius blocks are actually allowed inside: one street width in from
    /// the reservation, so the outermost street still has ground under it.
    pub buildable_m: f64,
    /// Centre-to-centre street spacing, metres.
    pub pitch_m: f64,
    /// The street reserve, metres.
    pub street_m: f64,
    pub streets: Vec<Street>,
    /// Blocks in ascending `(row, col)` — the order the level writes its
    /// entities in, so a committed document is a function of the design and not
    /// of an iteration order.
    pub blocks: Vec<Block>,
    /// Grid cells refused because a corner left the reservation circle.
    pub refused_off_pad: usize,
    /// Grid cells refused because the coastline crosses them or they are at sea.
    pub refused_off_land: usize,
}

impl Settlement {
    /// Total street centreline, kilometres.
    pub fn street_km(&self) -> f64 {
        self.streets.iter().map(|s| s.length_m()).sum::<f64>() / 1000.0
    }

    /// Blocks that took `a`.
    pub fn blocks_of(&self, a: ArchetypeId) -> usize {
        self.blocks.iter().filter(|b| b.archetype == a).count()
    }

    /// The node id of the grid crossing at `(col, row)`. The layout table is on
    /// [`street_graph`](Self::street_graph) and this is the one place it is
    /// written.
    fn street_node_id(&self, col: u64, row: u64) -> inf_nav::NavNodeId {
        inf_nav::domain::STREET
            | ((self.site as u64 & NODE_FIELD_MASK) << 40)
            | ((col & NODE_FIELD_MASK) << 20)
            | (row & NODE_FIELD_MASK)
    }

    /// **The id of this settlement's central crossroads** — the node every
    /// arriving island route lands on.
    ///
    /// `plan_network` routes centre to centre, so a highway that reaches this
    /// town terminates on the site's own `(x, z)`, and the grid puts a line
    /// through that point on both axes (the module docs say so and
    /// `the_settlement_grid_meets_the_island_road_network` measures how close to
    /// it the routes really end). This is the id of the node there, so a caller
    /// folding the island's road graph and this one into one network can join
    /// them by **name** rather than by a nearest-node query.
    ///
    /// It is an id, not a promise that the node exists: a reservation too small
    /// for one block plans no streets at all and
    /// [`street_graph`](Self::street_graph) is empty for it. Ask the graph.
    pub fn centre_node_id(&self) -> inf_nav::NavNodeId {
        let mid = LATTICE_BIAS as u64;
        self.street_node_id(mid, mid)
    }

    /// **The street grid as an [`inf_nav::NavGraph`]** (NPC1c) — the plan that
    /// decided where the blocks are, exposed as something a body can walk.
    ///
    /// Until this wave the streets were *consumed*: they cut the ground into
    /// blocks, they carried the join to the island's road network, and then they
    /// were a `Vec<Street>` nothing asked a question of. They are still a plan
    /// and not a surface (see the module docs for why the drawing is routed
    /// rather than taken), but a plan is exactly what a route wants.
    ///
    /// # The shape
    ///
    /// The grid is orthogonal and axis-aligned — a bound this module already
    /// states and takes on purpose, because a `PcgVolume` is a centre and an
    /// axis-aligned half-extent — so every line that runs along world X crosses
    /// every line that runs along world Z, and the crossings are the junctions.
    /// A node stands at each of them, plus one at each street's two ends (the
    /// outermost line stops at `buildable_m`, which is not a crossing), and
    /// consecutive nodes along one line are linked. That is `L² + 4L` nodes for
    /// `L` lines each way, which
    /// `a_street_grids_node_count_is_its_crossing_arithmetic` asserts rather
    /// than assumes.
    ///
    /// # The id layout
    ///
    /// | bits | meaning |
    /// |---|---|
    /// | 60–63 | [`inf_nav::domain::STREET`] — who minted the id |
    /// | 40–59 | the site index: **which settlement** |
    /// | 20–39 | the column: the world-X grid line |
    /// | 0–19 | the row: the world-Z grid line |
    ///
    /// The column and row are lattice indices — `round(offset / pitch)` shifted
    /// non-negative, the same rounding [`Block::col`]/[`Block::row`]
    /// state — so a node's id is a function of the **design** and not of the
    /// order `plan_site` happened to push its streets. That matters because
    /// `NavGraph::absorb` joins on id equality: two towns folded into one island
    /// network must not share a crossroads, and the site field is what keeps
    /// them apart.
    ///
    /// # Every node sits at `y = 0`, and that is the honest answer here
    ///
    /// A settlement plan carries no elevation at all — the module's own first
    /// law, enforced by
    /// `the_settlement_generator_is_authored_from_committed_design_alone` — and
    /// the pad's real datum is resolved at *evaluation* time from the terrain
    /// under each lot. So the grid is planar, and a caller that wants a route on
    /// the ground puts it there once with `NavPath::snapped`, which is where
    /// that query belongs anyway: `snapped`'s own doctrine is that a per-step
    /// ground query makes a position depend on streaming residency.
    pub fn street_graph(&self) -> inf_nav::NavGraph {
        let mut g = inf_nav::NavGraph::new();
        if !(self.pitch_m.is_finite() && self.pitch_m > 0.0) {
            // A reservation too small for one block of the finest grid planned
            // no streets, and an empty graph is the value that says so.
            return g;
        }
        // The two families of centreline, keyed by their own lattice index: the
        // world X of every line running along Z, the world Z of every line
        // running along X. `BTreeMap`, so the walk below is a function of the
        // indices and not of the plan's push order.
        let mut cols: std::collections::BTreeMap<u64, f64> = Default::default();
        let mut rows: std::collections::BTreeMap<u64, f64> = Default::default();
        for s in &self.streets {
            if runs_along_z(s) {
                cols.insert(lattice_index(s.a.x - self.centre.x, self.pitch_m), s.a.x);
            } else if runs_along_x(s) {
                rows.insert(lattice_index(s.a.y - self.centre.y, self.pitch_m), s.a.y);
            }
        }

        for s in &self.streets {
            let z_run = runs_along_z(s);
            if !(z_run || runs_along_x(s)) {
                // A street with no length is not a run: there are no two ends to
                // link and no crossing it can carry.
                continue;
            }
            let (lo, hi, fixed, origin) = if z_run {
                (s.a.y.min(s.b.y), s.a.y.max(s.b.y), s.a.x, self.centre.x)
            } else {
                (s.a.x.min(s.b.x), s.a.x.max(s.b.x), s.a.y, self.centre.y)
            };
            let fixed_index = lattice_index(fixed - origin, self.pitch_m);
            // The crossings this line carries, and then its two ends — an end
            // being its own node only when no crossing already stands there.
            // The equality is exact and can be: both coordinates are the site's
            // centre plus an offset, so two that name one point are one `f64`,
            // and two that do not are a whole block apart.
            let crossings = if z_run { &rows } else { &cols };
            let mut on: Vec<(f64, u64)> = crossings
                .iter()
                .filter(|(_, v)| **v >= lo && **v <= hi)
                .map(|(i, v)| (*v, *i))
                .collect();
            for (end, index) in [(lo, AXIS_MIN_INDEX), (hi, AXIS_MAX_INDEX)] {
                if !on.iter().any(|(v, _)| *v == end) {
                    on.push((end, index));
                }
            }
            on.sort_by(|a, b| a.0.total_cmp(&b.0));

            let mut prev: Option<inf_nav::NavNodeId> = None;
            for (along, index) in on {
                let (id, p) = if z_run {
                    (
                        self.street_node_id(fixed_index, index),
                        DVec3::new(fixed, 0.0, along),
                    )
                } else {
                    (
                        self.street_node_id(index, fixed_index),
                        DVec3::new(along, 0.0, fixed),
                    )
                };
                // Re-adding a crossing the other family already placed is the
                // same node, by id — which is exactly how the two families
                // become one grid.
                g.add_node(id, p, inf_nav::NavKind::Street);
                if let Some(a) = prev {
                    g.link(a, id, inf_nav::NavKind::Street, Vec::new());
                }
                prev = Some(id);
            }
        }
        g
    }
}

/// A street whose world X is constant — it runs along Z.
///
/// The grid is axis-aligned by the module's own bound, so this and
/// [`runs_along_x`] are exhaustive over everything `plan_site` plans, and an
/// exact comparison is the right test: a line's two ends are written from one
/// coordinate.
fn runs_along_z(s: &Street) -> bool {
    s.a.x == s.b.x && s.a.y != s.b.y
}

/// A street whose world Z is constant — it runs along X.
fn runs_along_x(s: &Street) -> bool {
    s.a.y == s.b.y && s.a.x != s.b.x
}

/// **Which grid line a coordinate belongs to**, as the non-negative id field
/// [`Settlement::street_graph`] writes.
///
/// `off_m` is the signed distance from the site's centre along one world axis
/// and `pitch_m` is the centre-to-centre street spacing, so a line sits at an
/// exact multiple of the pitch and its index is `round(off / pitch)` shifted by
/// [`LATTICE_BIAS`] — the same rounding [`Block::col`] and [`Block::row`] state.
///
/// **Rounded rather than compared exactly, on purpose.** A line's world
/// coordinate is `centre + k · pitch`, and subtracting a five-figure easting
/// back out of it does not return `k · pitch` to the bit. Rounding to the
/// nearest line is immune to that — the residual is picometres against a
/// sixty-metre pitch — and an equality test would have quietly given a whole
/// family of lines the wrong index on any site whose centre is not a small
/// number.
///
/// Clamped, because `Site::radius_m` is author input: a nonsense reservation
/// must corrupt nothing above the twenty bits this is written into.
fn lattice_index(off_m: f64, pitch_m: f64) -> u64 {
    let k = off_m / pitch_m;
    if !k.is_finite() {
        return LATTICE_BIAS as u64;
    }
    let max = MAX_LINE_INDEX as f64;
    (k.round().clamp(-max, max) as i64 + LATTICE_BIAS) as u64
}

/// **Every settlement the committed design implies.**
///
/// Waypoint sites build nothing — `SiteKind::reserves_urban` is the same
/// predicate the biome map's reservation uses, so a site that reserves no ground
/// gets no buildings on it and the two cannot disagree.
pub fn settlements(design: &IslandDesign) -> Vec<Settlement> {
    let land = Land::of(&design.coast);
    design
        .recipe
        .sites
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kind.reserves_urban())
        .map(|(i, s)| plan_site(design, i, s, &land))
        .collect()
}

/// Total blocks over every settlement — the number the level's entity count
/// grows by.
pub fn block_count(plans: &[Settlement]) -> usize {
    plans.iter().map(|s| s.blocks.len()).sum()
}

/// **The grid a site of this kind gets inside a `radius_m` reservation**, as
/// `(pitch, street, buildable)` — or `None` when not one block of the finest
/// grid fits.
///
/// `buildable` is the radius blocks are allowed inside: one street reserve in
/// from the reservation, so the outermost street still has levelled, urban-zoned
/// ground under it. It is part of the answer rather than a caller's arithmetic
/// because it depends on which rung of the ladder was taken.
///
/// # Why a ladder and not one grid per kind
///
/// The first draft was one grid per kind, and it made a settlement's existence a
/// step function of its radius: the first city block's far corner is 155.6 m
/// from the centre, so a 150 m city reservation held **zero** blocks and built
/// nothing at all — silently, because nothing else in the plan depends on a
/// block existing. The CI fixture's own city is 120 m (its radius is pinned by
/// its lake and its road grades — see `samples/island-fixture/island.toml`), so
/// the one settlement CI ever builds was the one that fell off the step.
///
/// A settlement therefore takes the **coarsest grid of its kind's ladder that
/// fits**, and a reservation that fits none builds nothing *and says so*
/// ([`Settlement::refused_off_pad`] counts every cell it refused). A city on a
/// town's grid is still zoned by the city's table — the ladder decides how
/// finely the ground is cut, the kind decides what stands on it.
///
/// "Fits" is the first block's own far corner against the radius, in **squared**
/// metres so no square root reaches a committed block position.
pub fn grid_for(kind: SiteKind, radius_m: f64) -> Option<(f64, f64, f64)> {
    let ladder: &[(f64, f64)] = match kind {
        SiteKind::City => &[
            (CITY_BLOCK_M + CITY_STREET_M, CITY_STREET_M),
            (TOWN_BLOCK_M + TOWN_STREET_M, TOWN_STREET_M),
        ],
        // A waypoint never reaches here (see `settlements`); answering the
        // town's ladder rather than panicking keeps this total.
        SiteKind::Town | SiteKind::Waypoint => &[(TOWN_BLOCK_M + TOWN_STREET_M, TOWN_STREET_M)],
    };
    ladder.iter().copied().find_map(|(pitch, street)| {
        let buildable = (radius_m - street).max(0.0);
        let far = street * 0.5 + (pitch - street);
        (2.0 * far * far <= buildable * buildable).then_some((pitch, street, buildable))
    })
}

/// The Chebyshev ring a grid index sits in: `0` for the two indices either side
/// of the centre line, growing outward symmetrically.
fn ring_of(i: i32) -> u32 {
    if i >= 0 {
        i as u32
    } else {
        (-i - 1) as u32
    }
}

fn plan_site(design: &IslandDesign, site: usize, s: &Site, land: &Land) -> Settlement {
    let centre = DVec2::new(s.x, s.z);
    // A reservation too small for one block of the finest grid builds nothing,
    // and the empty plan is the value that says so.
    let Some((pitch, street, buildable)) = grid_for(s.kind, s.radius_m) else {
        return Settlement {
            site,
            name: s.name.clone(),
            kind: s.kind,
            centre,
            radius_m: s.radius_m,
            buildable_m: 0.0,
            pitch_m: 0.0,
            street_m: 0.0,
            streets: Vec::new(),
            blocks: Vec::new(),
            refused_off_pad: 0,
            refused_off_land: 0,
        };
    };
    let half_street = street * 0.5;
    let half = DVec2::splat((pitch - street) * 0.5);

    // How many grid lines fit each way. A line at `k·pitch` from the centre is
    // useful while the block inside it can still fit, so the line count is the
    // block count plus one.
    //
    // **Bounded, on `subdivide_block`'s own `MAX_LOTS_PER_AXIS` precedent.**
    // `radius_m` is an author-supplied number that the recipe checks only for
    // finiteness, and a mis-typed `600000` would ask this loop for a hundred
    // million cells before the pad test refused every one of them. The clamp is
    // a *ceiling* rather than a preference: the widest reservation any committed
    // island holds is 600 m, which is five lines at the city pitch.
    let reach = ((buildable / pitch)
        .floor()
        .clamp(0.0, f64::from(MAX_GRID_LINES)) as i32)
        + 1;

    // The lines, and both ends of every one, stop at the buildable radius: a
    // street plan that ran on past the reservation would be a plan for ground
    // the pad never levelled and the biome map never reserved.
    let mut streets = Vec::new();
    for k in -reach..=reach {
        let off = f64::from(k) * pitch;
        if off.abs() > buildable {
            continue;
        }
        streets.push(Street {
            a: DVec2::new(centre.x + off, centre.y - buildable),
            b: DVec2::new(centre.x + off, centre.y + buildable),
            main: k == 0,
        });
        streets.push(Street {
            a: DVec2::new(centre.x - buildable, centre.y + off),
            b: DVec2::new(centre.x + buildable, centre.y + off),
            main: k == 0,
        });
    }

    // Every candidate cell, before zoning: the refusals are counted here so the
    // plan can say WHY a settlement is smaller than its circle (the standing law
    // — a refusal is a value).
    let mut cells: Vec<(i32, i32, u32, DVec2)> = Vec::new();
    let mut refused_off_pad = 0usize;
    let mut refused_off_land = 0usize;
    for row in -reach..reach {
        for col in -reach..reach {
            let c = centre
                + DVec2::new(
                    f64::from(col) * pitch + half_street + half.x,
                    f64::from(row) * pitch + half_street + half.y,
                );
            let rect = (c - half, c + half);
            // **The pad test is a PROOF, not a sample.** A disc is convex and a
            // rectangle is the convex hull of its corners, so four corners
            // inside the circle put every point of the block — and therefore
            // every lot the subdivider cuts out of it — inside it too. That is
            // the same argument `subdivide_block`'s own containment test makes
            // one level down.
            let corners = [
                rect.0,
                DVec2::new(rect.1.x, rect.0.y),
                rect.1,
                DVec2::new(rect.0.x, rect.1.y),
            ];
            if corners.iter().any(|p| (*p - centre).length() > buildable) {
                refused_off_pad += 1;
                continue;
            }
            if !land.contains_rect(rect.0, rect.1) {
                refused_off_land += 1;
                continue;
            }
            cells.push((col, row, ring_of(col).max(ring_of(row)), c));
        }
    }
    cells.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    // The industrial cluster: the blocks nearest the point the highway leaves
    // this city, outside the core.
    let max_ring = cells.iter().map(|(_, _, r, _)| *r).max().unwrap_or(0);
    let industrial: Vec<(i32, i32)> =
        match (s.kind, highway_exit(&design.routes, centre, buildable)) {
            (SiteKind::City, Some(exit)) => {
                let floor = industrial_min_ring(max_ring);
                let mut near: Vec<(u64, i32, i32)> = cells
                    .iter()
                    .filter(|(_, _, ring, _)| *ring >= floor)
                    .map(|(col, row, _, c)| ((*c - exit).length_squared().to_bits(), *col, *row))
                    .collect();
                near.sort();
                near.truncate(CITY_INDUSTRIAL_BLOCKS.min(near.len() / INDUSTRIAL_SHARE));
                near.into_iter().map(|(_, col, row)| (col, row)).collect()
            }
            _ => Vec::new(),
        };

    // **THE NIGHTLIFE STRIP** (wave VEN1a): a settlement's venues, on the
    // industrial cluster's own pattern.
    //
    // A weighted zone table cannot say "one of these per settlement" -- it is a
    // per-block draw, so a table carrying a nightclub at weight 1 would give a
    // city eleven of them and a town none. `zone_table`'s own arm asserts that
    // `Industrial` never comes out of the ladder for exactly this reason ("the
    // cluster is the only door onto it"), and a venue is the same shape of
    // thing: a fact about the settlement, not a probability per block.
    //
    // Where: nearest the centre, one ring OUT of it for a city (a strip is just
    // off the main street, not on it -- and ring 0 is the office core that pays
    // for the ground), and ring 0 for a town, whose high street IS its strip.
    // Sorted by squared distance `to_bits` then `(col, row)`, which is the
    // industrial cluster's own portable comparison: no float ordering reaches a
    // committed decision.
    //
    // Blocks the industrial cluster has already claimed are skipped rather than
    // overwritten, so the two rules cannot silently disagree about one block.
    let venues: Vec<((i32, i32), ArchetypeId)> = {
        let want = venue_strip(s.kind);
        let floor = venue_min_ring(s.kind);
        let mut near: Vec<(u64, i32, i32)> = cells
            .iter()
            .filter(|(col, row, ring, _)| *ring >= floor && !industrial.contains(&(*col, *row)))
            .map(|(col, row, _, c)| ((*c - centre).length_squared().to_bits(), *col, *row))
            .collect();
        near.sort();
        near.truncate(want.len().min(near.len() / VENUE_SHARE));
        near.into_iter()
            .zip(want.iter().copied())
            .map(|((_, col, row), a)| ((col, row), a))
            .collect()
    };

    // **THE CIVIC STRIP** (wave EMS1): the settlement's institutions, on the
    // venue strip's pattern exactly and for its argument -- "a city has a
    // hospital" is a fact about the settlement and not a probability per block,
    // so it cannot live in `zone_table` any more than `Industrial` or a
    // nightclub can.
    //
    // Where: nearest the centre, from `civic_min_ring` -- which is
    // `venue_min_ring` exactly, one ring out of a city and ring 0 of a town.
    // The rule is stated once, on that function, and NOT restated here: the
    // first draft of this comment carried the ring-0-everywhere ruling that
    // `civic_min_ring` itself retired later in the same wave, which is the A14
    // restated-rule defect this file already carries a scar from (`walk_door`).
    //
    // **AFTER the venues, and that ordering is load-bearing.** Both strips take
    // the blocks nearest the centre, so evaluating civics first would shrink the
    // venue strip's candidate list -- and on the small fixture that costs a city
    // its strip club, which `the_nightlife_strip_is_one_per_settlement_and_
    // three_kinds_per_city` asserts by name. Two gate arms also pick their
    // target venue with `find(|b| b.archetype.is_venue())` over block order, so
    // a venue that moved would move a whole PIE-versus-shipping trace.
    //
    // Blocks either earlier rule has claimed are skipped rather than
    // overwritten, so no two rules can silently disagree about one block.
    let civics: Vec<((i32, i32), ArchetypeId)> = {
        let want = civic_strip(s.kind);
        let floor = civic_min_ring(s.kind);
        let mut near: Vec<(u64, i32, i32)> = cells
            .iter()
            .filter(|(col, row, ring, _)| {
                *ring >= floor
                    && !industrial.contains(&(*col, *row))
                    && !venues.iter().any(|(k, _)| *k == (*col, *row))
            })
            .map(|(col, row, _, c)| ((*c - centre).length_squared().to_bits(), *col, *row))
            .collect();
        near.sort();
        near.truncate(want.len().min(near.len() / CIVIC_SHARE));
        near.into_iter()
            .zip(want.iter().copied())
            .map(|((_, col, row), a)| ((col, row), a))
            .collect()
    };

    let name_seed = Hash64::new(design.recipe.seed)
        .mix_u64(SETTLEMENT_SALT)
        .mix_u64(site as u64);
    let blocks = cells
        .iter()
        .map(|(col, row, ring, c)| {
            let h = name_seed.mix_i64(i64::from(*col)).mix_i64(i64::from(*row));
            let archetype = if industrial.contains(&(*col, *row)) {
                ArchetypeId::Industrial
            } else if let Some((_, a)) = venues.iter().find(|(k, _)| *k == (*col, *row)) {
                *a
            } else if let Some((_, a)) = civics.iter().find(|(k, _)| *k == (*col, *row)) {
                *a
            } else {
                pick(zone_table(s.kind, *ring), h.mix_u64(ZONE_SALT).unit())
            };
            Block {
                site,
                col: *col,
                row: *row,
                ring: *ring,
                centre: *c,
                half,
                archetype,
                seed: (h.mix_u64(BLOCK_SEED_SALT).finish() >> 32) as u32,
            }
        })
        .collect();

    Settlement {
        site,
        name: s.name.clone(),
        kind: s.kind,
        centre,
        radius_m: s.radius_m,
        buildable_m: buildable,
        pitch_m: pitch,
        street_m: street,
        streets,
        blocks,
        refused_off_pad,
        refused_off_land,
    }
}

/// Separates the settlement plan's draws from every other draw the island makes.
const SETTLEMENT_SALT: u64 = 0x0073_6574_746C_6531; // "settle1"
/// The zoning pick.
const ZONE_SALT: u64 = 0x5A4F_4E45_5F5F_5F5F; // "ZONE____"
/// The block's own volume seed.
const BLOCK_SEED_SALT: u64 = 0x424C_4F43_4B5F_5F5F; // "BLOCK___"

/// **THE ZONING TABLE** — what a block builds, from its ring and its site kind.
///
/// "Districts" are deterministic rules, not a new type (the wave's own ruling).
/// A zone is a weighted archetype list; the pick is a counter hash of the
/// block's own grid position, so it is a pure function of the design.
///
/// | site | ring | zone | archetypes (weights) |
/// |---|---|---|---|
/// | city | 0 | core | Office 5, Hotel 2, Shop 3 |
/// | city | 1 | inner | Office 3, Apartment 3, Shop 2, Hotel 1 |
/// | city | 2 | ring | Apartment 4, Shop 1 |
/// | city | ≥ 3 | edge | House 3, Estate 1 |
/// | city | (cluster) | industrial | Industrial — see [`CITY_INDUSTRIAL_BLOCKS`] |
/// | town | 0 | high street | Shop 3, House 1 |
/// | town | ≥ 1 | outskirt | House 4, Estate 1 |
///
/// The city rings are three deep because that is what the geometry allows: a
/// 600 m reservation at a 120 m pitch is four rings of blocks, of which the
/// outermost loses its corners to the circle. Deepening the ladder needs a
/// bigger city, not a bigger table.
pub fn zone_table(kind: SiteKind, ring: u32) -> &'static [(ArchetypeId, f64)] {
    const CORE: &[(ArchetypeId, f64)] = &[
        (ArchetypeId::Office, 5.0),
        (ArchetypeId::Hotel, 2.0),
        (ArchetypeId::Shop, 3.0),
    ];
    const INNER: &[(ArchetypeId, f64)] = &[
        (ArchetypeId::Office, 3.0),
        (ArchetypeId::Apartment, 3.0),
        (ArchetypeId::Shop, 2.0),
        (ArchetypeId::Hotel, 1.0),
    ];
    const RING: &[(ArchetypeId, f64)] = &[(ArchetypeId::Apartment, 4.0), (ArchetypeId::Shop, 1.0)];
    const EDGE: &[(ArchetypeId, f64)] = &[(ArchetypeId::House, 3.0), (ArchetypeId::Estate, 1.0)];
    const HIGH_STREET: &[(ArchetypeId, f64)] =
        &[(ArchetypeId::Shop, 3.0), (ArchetypeId::House, 1.0)];
    const OUTSKIRT: &[(ArchetypeId, f64)] =
        &[(ArchetypeId::House, 4.0), (ArchetypeId::Estate, 1.0)];
    match (kind, ring) {
        (SiteKind::City, 0) => CORE,
        (SiteKind::City, 1) => INNER,
        (SiteKind::City, 2) => RING,
        (SiteKind::City, _) => EDGE,
        (_, 0) => HIGH_STREET,
        (_, _) => OUTSKIRT,
    }
}

/// A weighted pick from `u ∈ [0, 1)`. Deterministic, and it never falls off the
/// end: a table whose weights do not add up still answers a zone.
///
/// The guard is `!is_finite() || <= 0.0` rather than `!(total > 0.0)` — the
/// negated comparison clippy refuses on a partially-ordered type — and the two
/// are NOT interchangeable: `total <= 0.0` alone is **false** for a NaN, so a
/// table carrying one would fall through into `u * NaN < acc` and answer the
/// last entry. Both halves are needed and both are what the arm below drives.
fn pick(table: &[(ArchetypeId, f64)], u: f64) -> ArchetypeId {
    let total: f64 = table.iter().map(|(_, w)| w.max(0.0)).sum();
    if !total.is_finite() || total <= 0.0 {
        return ArchetypeId::House;
    }
    let mut acc = 0.0;
    for (a, w) in table {
        acc += w.max(0.0);
        if u * total < acc {
            return *a;
        }
    }
    table[table.len() - 1].0
}

/// Where the **highway** leaves this city, if one passes through it.
///
/// The route runs city to city, so the exit is the first vertex past the
/// buildable radius walking outward from the vertex nearest the centre. Walking
/// outward — rather than taking an endpoint — is what makes this the *near*
/// exit rather than the far city.
fn highway_exit(routes: &[Route], centre: DVec2, radius: f64) -> Option<DVec2> {
    let mut best: Option<(u64, DVec2)> = None;
    for r in routes.iter().filter(|r| r.class == "highway") {
        let pts: Vec<DVec2> = r.points.iter().map(|p| DVec2::new(p.x, p.z)).collect();
        let Some(near) = pts
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (*a.1 - centre)
                    .length_squared()
                    .total_cmp(&(*b.1 - centre).length_squared())
            })
            .map(|(i, _)| i)
        else {
            continue;
        };
        if (pts[near] - centre).length() > radius {
            continue;
        }
        // Outward is whichever neighbour is farther from the centre; ties go to
        // increasing index so the answer is a function of the layer's own order.
        let forward = pts
            .get(near + 1)
            .map(|p| (*p - centre).length())
            .unwrap_or(f64::NEG_INFINITY);
        let back = near
            .checked_sub(1)
            .and_then(|i| pts.get(i))
            .map(|p| (*p - centre).length())
            .unwrap_or(f64::NEG_INFINITY);
        let mut exit = None;
        if forward >= back {
            for p in pts.iter().skip(near) {
                if (*p - centre).length() >= radius {
                    exit = Some(*p);
                    break;
                }
            }
        } else {
            for p in pts.iter().take(near + 1).rev() {
                if (*p - centre).length() >= radius {
                    exit = Some(*p);
                    break;
                }
            }
        }
        if let Some(e) = exit {
            let d = (e - centre).length_squared().to_bits();
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, e));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// The coastline, as a containment test.
///
/// # Why a rectangle's containment is a PROOF here
///
/// A ring is not convex, so four corners inside it do not put the interior
/// inside it. What does: **if no ring edge crosses the rectangle and the
/// rectangle's centre is inside, the whole rectangle is inside** — a connected
/// region whose boundary misses a connected set is either wholly in or wholly
/// out of it. Both halves are segment arithmetic and neither needs a
/// transcendental, which is what a committed block position requires.
struct Land {
    rings: Vec<Vec<DVec2>>,
}

impl Land {
    fn of(coast: &[Vec<DVec2>]) -> Self {
        Self {
            rings: coast.to_vec(),
        }
    }

    /// Is `p` inside any ring? Even-odd crossing count, on a ray along `+X`.
    fn contains_point(&self, p: DVec2) -> bool {
        let mut inside = false;
        for ring in &self.rings {
            let n = ring.len();
            if n < 3 {
                continue;
            }
            let mut hit = false;
            for i in 0..n {
                let (a, b) = (ring[i], ring[(i + 1) % n]);
                if (a.y > p.y) != (b.y > p.y) {
                    let t = (p.y - a.y) / (b.y - a.y);
                    if a.x + t * (b.x - a.x) > p.x {
                        hit = !hit;
                    }
                }
            }
            inside ^= hit;
        }
        inside
    }

    /// Is the whole axis-aligned rectangle `[lo, hi]` on land?
    fn contains_rect(&self, lo: DVec2, hi: DVec2) -> bool {
        if self.rings.is_empty() {
            // An island with no committed shore: everything is land, which is
            // what the fixture's own degenerate case should answer rather than
            // "no settlement anywhere".
            return true;
        }
        for ring in &self.rings {
            let n = ring.len();
            if n < 2 {
                continue;
            }
            for i in 0..n {
                if segment_hits_rect(ring[i], ring[(i + 1) % n], lo, hi) {
                    return false;
                }
            }
        }
        self.contains_point((lo + hi) * 0.5)
    }
}

/// Does the segment `a..b` touch the axis-aligned rectangle `[lo, hi]`?
///
/// Slab clipping (Liang-Barsky), so it is four divides and no trigonometry.
fn segment_hits_rect(a: DVec2, b: DVec2, lo: DVec2, hi: DVec2) -> bool {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [
        (-d.x, a.x - lo.x),
        (d.x, hi.x - a.x),
        (-d.y, a.y - lo.y),
        (d.y, hi.y - a.y),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return false;
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            if r > t1 {
                return false;
            }
            t0 = t0.max(r);
        } else {
            if r < t0 {
                return false;
            }
            t1 = t1.min(r);
        }
    }
    t0 <= t1
}

// ── the zone library ────────────────────────────────────────────────────────
//
// Seven `.inf_pcg` documents, one per archetype, shared by BOTH islands and by
// anything else that wants a district. They live with the ENGINE
// (`samples/settlement/`) rather than in one island's folder, for the reason
// `samples/ground` and `samples/starter-character` already live there: one copy,
// byte-locked against its generator, referenced by every recipe that wants it.
//
// Each is a few hundred bytes of RULES — a footprint, a subdivision and a
// palette choice — never a block position and never a street. That is the
// small-committed-folder law read at settlement scale: `phase30-city`'s 221 KB
// committed road mesh is the anti-pattern, and 172 volume records naming seven
// documents is the shape that is not.

/// The zone document's stable GUID, derived from the archetype's own name.
///
/// Island-independent on purpose: the library is engine content, so the same
/// seven GUIDs resolve in every project that copies it.
pub fn zone_guid(a: ArchetypeId) -> Uuid {
    crate::island::derived(a.name(), "settlement.zone")
}

/// One block's entity GUID in a committed island level.
pub fn block_guid(island: &str, site: usize, col: i32, row: i32) -> Uuid {
    crate::island::derived(&format!("{island}/{site}/{col}/{row}"), "settlement.block")
}

/// The `Zone_<Archetype>.inf_pcg` file name.
pub fn zone_file_name(a: ArchetypeId) -> String {
    format!("Zone_{}.inf_pcg", a.name())
}

/// **How each archetype cuts its block into lots.**
///
/// | archetype | frontage × depth | setback | jitter | min area | lots on a 100 m block |
/// |---|---|---|---|---|---|
/// | Office | 30 × 34 | 2.0 | 0.10 | 120 | 3 × 3 = 9 |
/// | Hotel | 36 × 36 | 2.5 | 0.08 | 200 | 3 × 3 = 9 |
/// | Shop | 20 × 28 | 1.0 | 0.14 | 50 | 5 × 4 = 20 |
/// | Apartment | 26 × 30 | 2.5 | 0.10 | 100 | 4 × 3 = 12 |
/// | House | 18 × 26 | 3.0 | 0.16 | 60 | 6 × 4 = 24 |
/// | Estate | 40 × 44 | 6.0 | 0.10 | 300 | 3 × 2 = 6 |
/// | Industrial | 48 × 48 | 4.0 | 0.06 | 400 | 2 × 2 = 4 |
/// | Bar | 22 × 30 | 1.0 | 0.12 | 90 | 5 × 3 = 15 |
/// | Nightclub | 32 × 36 | 2.0 | 0.08 | 200 | 3 × 3 = 9 |
/// | StripClub | 28 × 34 | 2.0 | 0.08 | 170 | 4 × 3 = 12 |
/// | PoliceStation | 40 × 42 | 3.0 | 0.06 | 300 | 3 × 2 = 6 |
/// | FireHall | 44 × 40 | 3.0 | 0.06 | 320 | 2 × 3 = 6 |
/// | Hospital | 52 × 50 | 4.0 | 0.05 | 500 | 2 × 2 = 4 |
/// | Clinic | 22 × 28 | 1.5 | 0.12 | 80 | 5 × 4 = 20 |
///
/// The counts are `subdivide_block`'s own rounding rule — *round to the nearest
/// whole number of lots, never below one* — so a 60 m town block takes fewer of
/// the same lots rather than a different rule (a house block in a town is
/// 3 × 2 = 6).
pub fn zone_lots(a: ArchetypeId) -> LotRules {
    let (frontage_m, depth_m, jitter, setback_m, min_area_m2) = match a {
        ArchetypeId::Office => (30.0, 34.0, 0.10, 2.0, 120.0),
        ArchetypeId::Hotel => (36.0, 36.0, 0.08, 2.5, 200.0),
        ArchetypeId::Shop => (20.0, 28.0, 0.14, 1.0, 50.0),
        ArchetypeId::Apartment => (26.0, 30.0, 0.10, 2.5, 100.0),
        ArchetypeId::House => (18.0, 26.0, 0.16, 3.0, 60.0),
        ArchetypeId::Estate => (40.0, 44.0, 0.10, 6.0, 300.0),
        ArchetypeId::Industrial => (48.0, 48.0, 0.06, 4.0, 400.0),
        // **The venues** (wave VEN1a). A bar is a shopfront with a deeper
        // back-of-house; a club is a box, and it has to be one -- a nightclub's
        // `max_room_area` is 260 m2 and a 20 m frontage would never give it a
        // room that big to anchor its dance floor in.
        //
        // A WHOLE-BLOCK lot was tried and withdrawn. It made one venue per
        // venue block, which reads better and holds the light budget by
        // construction -- and it made the gate arm **11 s -> 539 s**, because a
        // single 54 x 54 m building is a different shape of problem from six
        // 20 x 30 ones with the same total floor area. The budget is held at the
        // source instead, by `inf_pcg::VOLUME_LIGHT_CAP`, which is a rule about
        // the thing that is scarce rather than about the thing that is not.
        ArchetypeId::Bar => (22.0, 30.0, 0.12, 1.0, 90.0),
        ArchetypeId::Nightclub => (32.0, 36.0, 0.08, 2.0, 200.0),
        ArchetypeId::StripClub => (28.0, 34.0, 0.08, 2.0, 170.0),
        // **The institutions** (wave EMS1). Big lots and very little jitter,
        // which is what a civic building looks like: three of them are set back
        // squarely off the street on a plot of their own, and a row of them
        // wandering by 16% the way a street of houses does would read as a
        // suburb rather than as a campus.
        //
        // The sizes are driven by the room the archetype anchors on, exactly as
        // the nightclub's are. A `PoliceStation` and a `FireHall` have to give
        // an `ApparatusBay` 150–240 m² of undivided floor to anchor in, and a
        // 22 m frontage never would. A `Hospital` is the largest lot in the
        // tree because it is the only five-storey civic building and its
        // corridor is 2.8 m of every floor.
        //
        // The `Clinic` is the exception and is deliberately a SHOP's lot: it
        // sits in a high-street parade, it is one or two storeys, and its
        // biggest room is a waiting room. That is what makes it the institution
        // a town can have when it cannot have a hospital.
        ArchetypeId::PoliceStation => (40.0, 42.0, 0.06, 3.0, 300.0),
        ArchetypeId::FireHall => (44.0, 40.0, 0.06, 3.0, 320.0),
        ArchetypeId::Hospital => (52.0, 50.0, 0.05, 4.0, 500.0),
        ArchetypeId::Clinic => (22.0, 28.0, 0.12, 1.5, 80.0),
    };
    LotRules {
        frontage_m,
        depth_m,
        jitter,
        setback_m,
        min_area_m2,
    }
}

/// The zone document's graph: **the `phase30-city` chain, per archetype.**
///
/// `grammar.footprint` (size 0 → the volume's own extent) → `building.lots`
/// (this zone's rules) → `building.plan`, with a `building.archetype` on the
/// side. `floors` is **0**, which draws from the archetype's own storey range
/// per lot seed — so a street of houses is one to three storeys rather than a
/// terrace of identical boxes.
///
/// `ground` is `Terrain`, not `Span`: the island has real ground and a building
/// takes its datum from under its own footprint centre. That also makes the
/// refusal free — no ground under a lot is no building, rather than a building
/// at `y = 0` (`jobs_of`'s fail-closed rule).
pub fn zone_graph(a: ArchetypeId) -> inf_graph::Graph {
    use inf_graph::ParamValue as P;
    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    let add = |g: &mut inf_graph::Graph,
               n: u32,
               type_id: &str,
               params: &[(&str, inf_graph::ParamValue)]| {
        let node = inf_graph::NodeId(n);
        let mut m = inf_graph::ParamMap::new();
        for (k, v) in params {
            m.insert((*k).to_string(), v.clone());
        }
        inf_graph::apply_edits(
            g,
            &reg,
            &[inf_graph::GraphEdit::AddNode {
                id: node,
                type_id: type_id.into(),
                x: 0.0,
                y: 0.0,
                params: m,
            }],
        );
        node
    };
    let lots = zone_lots(a);
    let block = add(
        &mut g,
        1,
        "grammar.footprint",
        &[("size_x", P::Float(0.0)), ("size_z", P::Float(0.0))],
    );
    let cut = add(
        &mut g,
        2,
        "building.lots",
        &[
            ("frontage", P::Float(lots.frontage_m)),
            ("depth", P::Float(lots.depth_m)),
            ("jitter", P::Float(lots.jitter)),
            ("setback", P::Float(lots.setback_m)),
            ("min_area", P::Float(lots.min_area_m2)),
        ],
    );
    let arch = add(
        &mut g,
        3,
        "building.archetype",
        &[
            ("archetype", P::Enum(a.name().into())),
            ("floors", P::Int(0)),
            ("furnish", P::Bool(furnishes(a))),
        ],
    );
    let plan = add(
        &mut g,
        4,
        "building.plan",
        &[
            ("name", P::Text(a.name().to_lowercase())),
            // Distinct per zone so two zone documents evaluated on ONE volume
            // would still be two districts. Nothing does that today; a seed
            // shared by construction is how it would stop being noticed.
            (
                "seed",
                P::Int(
                    1 + ArchetypeId::ALL
                        .iter()
                        .position(|x| *x == a)
                        .expect("every archetype is in ALL") as i64,
                ),
            ),
            ("ground", P::Enum("Terrain".into())),
        ],
    );
    let out = add(&mut g, 5, "output.pcg", &[]);
    for (from, fp, to, tp) in [
        (block, "out", cut, "block"),
        (cut, "out", plan, "lots"),
        (arch, "out", plan, "archetype"),
        (plan, "out", out, "scatter"),
    ] {
        inf_graph::apply_edits(
            &mut g,
            &reg,
            &[inf_graph::GraphEdit::Connect {
                link: inf_graph::Link {
                    from,
                    from_port: fp.into(),
                    to,
                    to_port: tp.into(),
                },
            }],
        );
    }
    g
}

/// The zone document's `.inf_pcg` payload.
pub fn zone_payload(a: ArchetypeId) -> Result<inf_pcg::PcgAssetPayload, String> {
    let graph = zone_graph(a);
    let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
    if !lowered.ok {
        return Err(format!(
            "the {} zone graph does not lower: {:?}",
            a.name(),
            lowered.issues
        ));
    }
    Ok(inf_pcg::PcgAssetPayload::from_graph(
        &graph,
        lowered.document,
    ))
}

/// `samples/settlement/` — the committed zone library.
pub fn settlement_dir() -> std::path::PathBuf {
    crate::island::repo_root().join("samples/settlement")
}

/// Every file the library commits, in the order a recipe's `[content]` list
/// names them: payload then sidecar, archetype by archetype.
pub fn settlement_files() -> Vec<String> {
    let mut out = Vec::new();
    for a in ArchetypeId::ALL {
        let n = zone_file_name(a);
        out.push(format!("{n}.toml"));
        out.push(n);
    }
    // VEN1b: and the club loop the venue archetypes' own music emitter names.
    out.push(VENUE_MUSIC_FILE.to_string());
    out.push(format!("{VENUE_MUSIC_FILE}.toml"));
    out.sort();
    out
}

/// **The file a venue's music loop lives in**, beside the zone documents.
///
/// The settlement library is where a settlement's own content goes, and a club
/// loop is settlement content: it is named by `inf_ecs::venue::VENUE_MUSIC_CLIP`
/// from the engine and by every island recipe's `[content]` list, exactly as a
/// zone document is.
pub const VENUE_MUSIC_FILE: &str = "Venue_Music.inf_audio";

/// **The committed club loop**, as an [`inf_audio::AudioAsset`].
///
/// A short deterministic tone, generated rather than recorded, on
/// `crate::samples::playground_audio_asset`'s own terms: a committed `.inf_audio`
/// needs no binary fixture, and a clip a test can regenerate is a clip a
/// reviewer can diff. Four thousand samples at 8 kHz is half a second, looped —
/// which is not music and is not pretending to be. What it makes true is the
/// thing the wave needs true: the `Play` a venue issues **names a clip that
/// resolves**, so the doorway model is attenuating a real voice rather than a
/// command into the void.
pub fn venue_music_asset() -> inf_audio::AudioAsset {
    crate::samples::playground_audio_asset()
}

/// Write every archetype zone document and the README.
pub fn write_settlement_library(dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    for a in ArchetypeId::ALL {
        let bytes = inf_asset::encode(&zone_payload(a)?)
            .map_err(|e| format!("encode the {} zone: {e}", a.name()))?;
        let p = dir.join(zone_file_name(a));
        std::fs::write(&p, &bytes).map_err(|e| format!("write {}: {e}", p.display()))?;
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(zone_guid(a)),
            inf_asset::AssetKind::Pcg,
            inf_asset::ContentHash::of(&bytes),
        )
        .save(&p)
        .map_err(|e| format!("write the {} zone sidecar: {e}", a.name()))?;
    }
    // **The club loop** (VEN1b), with the GUID `inf_ecs::venue` names it by —
    // an asset a level references by id has to have the same id every time or
    // the committed bytes are a different set of files on every build.
    let audio = venue_music_asset();
    let bytes = inf_asset::encode(&audio).map_err(|e| format!("encode the venue music: {e}"))?;
    let p = dir.join(VENUE_MUSIC_FILE);
    std::fs::write(&p, &bytes).map_err(|e| format!("write {}: {e}", p.display()))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(inf_ecs::venue::VENUE_MUSIC_CLIP),
        inf_asset::AssetKind::Audio,
        inf_asset::ContentHash::of(&bytes),
    )
    .save(&p)
    .map_err(|e| format!("write the venue music sidecar: {e}"))?;
    std::fs::write(dir.join("README.md"), SETTLEMENT_README)
        .map_err(|e| format!("write readme: {e}"))
}

/// The library's own README.
pub const SETTLEMENT_README: &str = "# The settlement zone library (island wave I8a)\n\n\
Generated by `inf_editor_core::settlement::write_settlement_library`. **One\n\
`.inf_pcg` document per building archetype** -- fourteen of them since wave\n\
EMS1 appended the police station, the fire hall, the hospital and the clinic\n\
behind VEN1a's three venues -- each a few hundred bytes\n\
of RULES: a footprint span over the evaluating volume's own extent, IB-2c's\n\
`building.lots` subdivision, and a `building.plan` on the archetype's palette.\n\
Beside them, since wave VEN1b, **one `.inf_audio`**: the club loop every\n\
venue's music emitter names by GUID (`inf_ecs::venue::VENUE_MUSIC_CLIP`).\n\n\
They are ENGINE content, like `samples/ground` and `samples/starter-character`:\n\
one copy, byte-locked against the generator, named by both islands' recipes\n\
under `[content]` and by every settlement block's `PcgVolume.graph`.\n\n\
## Why one document per ARCHETYPE and not one per block\n\n\
A committed level carries one `PcgVolume` record per block -- a centre, a\n\
half-extent, a seed and a graph GUID. What makes a hundred and seventy volumes a\n\
hundred and seventy different blocks is the **seed**, not a hundred and seventy\n\
graphs. Committing a document per block would be committing geometry where the\n\
small-committed-folder law says commit rules.\n\n\
## The zoning table lives in the generator, not here\n\n\
Which archetype a block takes is `inf_editor_core::settlement::zone_table` --\n\
a deterministic function of the block's ring and its site's kind, plus one\n\
industrial cluster per city near the highway. A zone document knows nothing\n\
about where it is used.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

#[cfg(test)]
mod tests {
    use super::*;

    /// The module's own **non-test, non-comment** source — the same scan
    /// `island.rs` reads, through the same door.
    fn module_code() -> Vec<(usize, String)> {
        crate::island::scan::code_lines(include_str!("settlement.rs"))
    }

    /// **The settlement plan reads the committed design and no elevation.**
    ///
    /// The same allowlist `island.rs` carries, over the module that stands the
    /// buildings up — because the plan reaches the committed `.inf_lvl` exactly
    /// as the level's own numbers do, and an allowlist that stops at one file
    /// stops at the file somebody happened to write first.
    #[test]
    fn the_settlement_generator_is_authored_from_committed_design_alone() {
        const ALLOWED: &[&str] = &["IslandDesign", "Route", "Site", "SiteKind"];
        let code = module_code();
        assert!(
            crate::island::scan::aliases(&code).is_empty(),
            "settlement.rs imports `inf_island` under another name at line(s) \
             {:?} — the scan follows the literal `inf_island::` and an alias \
             walks past it (island wave I8a audit)",
            crate::island::scan::aliases(&code)
        );
        let used = crate::island::scan::island_doors(&code);
        println!("settlement.rs reaches inf_island::{{{:?}}}", used.keys());
        for (name, line) in &used {
            assert!(
                ALLOWED.contains(&name.as_str()),
                "settlement.rs:{line} names `inf_island::{name}`, which is not on \
                 the committed-design allowlist"
            );
        }
        assert!(
            used.contains_key("IslandDesign"),
            "the module no longer reads the committed design at all"
        );
        for (n, line) in &code {
            assert!(
                !line.contains("inf_terrain::"),
                "settlement.rs:{n} names `inf_terrain::` — every door onto an \
                 elevation is in that crate: {}",
                line.trim()
            );
        }
        // …and the scan can fail, which is the anti-vacuity half. The probe is
        // the BRACE form on purpose: it is the one this module actually writes,
        // and it is the one the first extractor could not read.
        let probe = vec![(
            1usize,
            "use inf_island::{IslandDesign, sample_terrain};".to_string(),
        )];
        let found = crate::island::scan::island_doors(&probe);
        assert!(found.contains_key("sample_terrain"));
        assert!(!ALLOWED.contains(&"sample_terrain"));
        // **And the WRAPPED form** (island wave I8a audit), which is what this
        // module's own import becomes the day a fifth door joins it: four names
        // is 48 characters and `rustfmt` wraps at a hundred.
        let wrapped = vec![
            (1usize, "use inf_island::{".to_string()),
            (2, "IslandDesign, Route, Site, SiteKind,".to_string()),
            (3, "sample_terrain,".to_string()),
            (4, "};".to_string()),
        ];
        let found = crate::island::scan::island_doors(&wrapped);
        assert!(
            found.contains_key("sample_terrain"),
            "a wrapped brace import scanned clean: {:?}",
            found.keys()
        );
    }

    /// **THE LEDGER LINE**: what the committed design actually plans, printed
    /// per settlement, with the refusals beside the blocks.
    ///
    /// Asserted loosely and printed exactly — the numbers move with the recipe's
    /// own radii, and a settlement plan that is a function of the design is
    /// supposed to.
    #[test]
    fn the_committed_island_plans_two_cities_and_five_towns() {
        let Some(d) = crate::island::committed_design(crate::island::ISLAND_RECIPES[0]) else {
            eprintln!("SKIP: no committed island design");
            return;
        };
        let plans = settlements(&d);
        assert_eq!(plans.len(), 5 + 2, "seven sites reserve urban ground");
        let mut blocks = 0usize;
        let mut km = 0.0f64;
        for s in &plans {
            let by: Vec<String> = ArchetypeId::ALL
                .iter()
                .filter(|a| s.blocks_of(**a) > 0)
                .map(|a| format!("{} {}", s.blocks_of(*a), a.name()))
                .collect();
            println!(
                "SETTLEMENT {:>13} ({:>4}) r={:.0} m buildable={:.0} m pitch={:.0} m: \
                 {} blocks [{}], {} refused off-pad, {} refused off-land, {:.2} km of street",
                s.name,
                s.kind.label(),
                s.radius_m,
                s.buildable_m,
                s.pitch_m,
                s.blocks.len(),
                by.join(", "),
                s.refused_off_pad,
                s.refused_off_land,
                s.street_km()
            );
            blocks += s.blocks.len();
            km += s.street_km();
            assert!(
                !s.blocks.is_empty(),
                "{} planned no block at all inside a {:.0} m reservation",
                s.name,
                s.radius_m
            );
            // Every block is inside the reservation, corner by corner — the
            // property that makes every LOT inside it too, by convexity.
            for b in &s.blocks {
                for c in b.corners() {
                    assert!(
                        (c - s.centre).length() <= s.buildable_m + 1e-9,
                        "{}'s block {:?} has a corner {:.1} m out",
                        s.name,
                        (b.col, b.row),
                        (c - s.centre).length()
                    );
                }
            }
        }
        println!("SETTLEMENTS TOTAL: {blocks} blocks, {km:.2} km of street centreline");
        assert_eq!(blocks, block_count(&plans));
        // Two seeds are two blocks: a shared seed is a district of identical
        // buildings, which is what `plan.rs` says a shared seed produces.
        let mut seeds: Vec<u32> = plans
            .iter()
            .flat_map(|s| s.blocks.iter().map(|b| b.seed))
            .collect();
        let n = seeds.len();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(seeds.len(), n, "two blocks drew the same volume seed");
    }

    /// **The grid joins the island's road network**, measured rather than
    /// asserted by construction: `plan_network` routes centre to centre, so
    /// every arriving route's near endpoint has to land on a street centreline.
    #[test]
    fn the_settlement_grid_meets_the_island_road_network() {
        let Some(d) = crate::island::committed_design(crate::island::ISLAND_RECIPES[0]) else {
            eprintln!("SKIP: no committed island design");
            return;
        };
        let mut worst = 0.0f64;
        let mut worst_at = String::new();
        let mut checked = 0usize;
        let mut reached: std::collections::BTreeSet<String> = Default::default();
        let mut narrowest = f64::INFINITY;
        for s in settlements(&d) {
            narrowest = narrowest.min(s.street_m);
            for r in &d.routes {
                for end in [r.points.first(), r.points.last()].into_iter().flatten() {
                    let p = DVec2::new(end.x, end.z);
                    if (p - s.centre).length() > s.radius_m {
                        continue;
                    }
                    let near = s
                        .streets
                        .iter()
                        .map(|st| st.distance_to(p))
                        .fold(f64::INFINITY, f64::min);
                    if near > worst {
                        worst = near;
                        worst_at = format!("{} ({})", s.name, r.name);
                    }
                    reached.insert(s.name.clone());
                    checked += 1;
                }
            }
        }
        println!(
            "ROAD JOIN: {checked} route endpoints inside {} of the 7 settlements, \
             worst {worst:.3} m from a street centreline (at {worst_at}) against a \
             narrowest street reserve of {narrowest:.1} m",
            reached.len()
        );
        assert_eq!(
            reached.len(),
            7,
            "the island's road network reaches only {reached:?}"
        );
        // **On the carriageway, not on the centreline.** The routes are planned
        // on the derivation lattice (8 m) and drape onto the 1 m grid, so an
        // endpoint lands within a lattice cell of the site's own `(x, z)` rather
        // than exactly on it. What "joins" means is that the arriving road ends
        // INSIDE the street it hands over to, which is the half-reserve.
        assert!(
            worst <= narrowest * 0.5,
            "a route ends {worst:.3} m from the nearest street centreline at \
             {worst_at}, outside the {narrowest:.1} m street's own half-reserve — \
             the grid does not join the island's road network"
        );
    }

    #[test]
    fn a_ring_grows_symmetrically_out_of_the_crossroads() {
        assert_eq!(ring_of(0), 0);
        assert_eq!(ring_of(-1), 0);
        assert_eq!(ring_of(1), 1);
        assert_eq!(ring_of(-2), 1);
        assert_eq!(ring_of(3), 3);
        assert_eq!(ring_of(-4), 3);
    }

    /// The industrial floor is the settlement's outer half and never its core —
    /// including in a settlement too shallow for the constant it replaced.
    /// **The grid ladder**, and the step it exists to remove.
    #[test]
    fn a_reservation_takes_the_coarsest_grid_that_fits_it() {
        let city = CITY_BLOCK_M + CITY_STREET_M;
        let town = TOWN_BLOCK_M + TOWN_STREET_M;
        // A full city reservation takes the city's own grid.
        assert_eq!(
            grid_for(SiteKind::City, 600.0),
            Some((city, CITY_STREET_M, 580.0))
        );
        // The fixture's 120 m city falls to the town's grid — and WITHOUT the
        // ladder it would have built nothing at all, which is the step.
        let (pitch, street, buildable) =
            grid_for(SiteKind::City, 120.0).expect("a 120 m city still gets a grid");
        assert_eq!((pitch, street), (town, TOWN_STREET_M));
        assert_eq!(buildable, 104.0);
        let far = TOWN_STREET_M * 0.5 + TOWN_BLOCK_M;
        assert!(2.0 * far * far <= buildable * buildable);
        // The first city block's far corner is 155.6 m out, which is what makes
        // 120 m too small for the city's own grid — the arithmetic behind the
        // ladder, stated rather than trusted.
        let city_far = CITY_STREET_M * 0.5 + CITY_BLOCK_M;
        let city_corner = (2.0 * city_far * city_far).sqrt();
        println!(
            "GRID LADDER: a city block's far corner is {city_corner:.1} m, a town \
             block's {:.1} m",
            (2.0 * far * far).sqrt()
        );
        assert!((city_corner - 155.563).abs() < 0.01);
        // …and a reservation too small for either builds nothing.
        assert_eq!(grid_for(SiteKind::City, 90.0), None);
        assert_eq!(grid_for(SiteKind::Town, 90.0), None);
        assert_eq!(
            grid_for(SiteKind::Town, 130.0),
            Some((town, TOWN_STREET_M, 114.0))
        );
        assert_eq!(grid_for(SiteKind::Town, 0.0), None);
        assert_eq!(grid_for(SiteKind::Town, f64::NAN), None);
    }

    /// Author input cannot make the planner misbehave — the subdivider's own
    /// `hostile_rules_resolve_rather_than_propagate`, one level up.
    #[test]
    fn a_hostile_radius_resolves_rather_than_propagating() {
        assert_eq!(
            grid_for(SiteKind::City, f64::INFINITY).map(|g| g.0),
            Some(120.0)
        );
        assert_eq!(grid_for(SiteKind::City, f64::NAN), None);
        assert_eq!(grid_for(SiteKind::Town, -5.0), None);
        // …and the cell loop is bounded whatever the radius: the reach is
        // `floor(buildable / pitch)` clamped to `MAX_GRID_LINES`, so an absurd
        // reservation asks for `(2 * 64)^2` candidates rather than for a
        // hundred million.
        let huge = 600_000.0f64;
        let (pitch, street, buildable) = grid_for(SiteKind::City, huge).expect("a grid");
        assert_eq!(buildable, huge - street);
        let raw = (buildable / pitch).floor();
        assert!(raw > f64::from(MAX_GRID_LINES), "the fixture is not absurd");
        let reach = (raw.clamp(0.0, f64::from(MAX_GRID_LINES)) as i32) + 1;
        assert_eq!(reach, MAX_GRID_LINES as i32 + 1);
        println!(
            "GRID CLAMP: a {huge:.0} m reservation asks for {raw:.0} lines and \
             takes {}, i.e. {} candidate cells",
            MAX_GRID_LINES,
            (2 * reach) * (2 * reach)
        );
    }

    #[test]
    fn the_industrial_floor_is_the_outer_half_and_never_the_core() {
        assert_eq!(industrial_min_ring(3), 2, "a four-ring city");
        assert_eq!(industrial_min_ring(2), 1);
        assert_eq!(industrial_min_ring(1), 1, "the fixture's two-ring city");
        assert_eq!(
            industrial_min_ring(0),
            1,
            "a one-ring settlement has no room for a works, and 0 would put it on \
             the crossroads"
        );
        for r in 0..12u32 {
            assert!(industrial_min_ring(r) >= 1);
            assert!(industrial_min_ring(r) <= r.max(1));
        }
    }

    #[test]
    fn the_zoning_table_is_the_one_in_the_doc_comment() {
        use ArchetypeId::*;
        // A city's ladder really is core → inner → ring → edge, and every
        // archetype the memo names is reachable.
        let mut seen: std::collections::BTreeSet<&'static str> = Default::default();
        for ring in 0..6u32 {
            for k in 0..64 {
                let a = pick(zone_table(SiteKind::City, ring), f64::from(k) / 64.0);
                seen.insert(a.name());
            }
        }
        for a in [Office, Hotel, Shop, Apartment, House, Estate] {
            assert!(seen.contains(a.name()), "a city never zones {}", a.name());
        }
        // Industrial is the cluster's, never the ladder's.
        assert!(
            !seen.contains(Industrial.name()),
            "the ring ladder handed out an industrial block — the cluster is the \
             only door onto it"
        );
        let mut town: std::collections::BTreeSet<&'static str> = Default::default();
        for ring in 0..4u32 {
            for k in 0..64 {
                town.insert(pick(zone_table(SiteKind::Town, ring), f64::from(k) / 64.0).name());
            }
        }
        assert_eq!(
            town,
            [House.name(), Shop.name(), Estate.name()]
                .into_iter()
                .collect(),
            "a town zones {town:?}"
        );
    }

    /// A weighted pick really is weighted, and it is a pure function of `u`.
    #[test]
    fn the_pick_is_weighted_and_total() {
        let t: &[(ArchetypeId, f64)] = &[(ArchetypeId::Office, 3.0), (ArchetypeId::Shop, 1.0)];
        let n = 10_000;
        let offices = (0..n)
            .filter(|k| pick(t, f64::from(*k) / f64::from(n)) == ArchetypeId::Office)
            .count();
        let share = offices as f64 / f64::from(n);
        assert!((share - 0.75).abs() < 0.01, "office share {share}");
        // Degenerate tables answer rather than panic.
        assert_eq!(pick(&[], 0.5), ArchetypeId::House);
        assert_eq!(
            pick(&[(ArchetypeId::Hotel, 0.0)], 0.5),
            ArchetypeId::House,
            "a table of zero weights has no pick to make"
        );
        // …and a NaN weight, which `total <= 0.0` alone would wave through.
        assert_eq!(
            pick(&[(ArchetypeId::Hotel, f64::NAN)], 0.5),
            ArchetypeId::House,
            "a NaN weight fell through the guard and picked the last entry"
        );
        assert_eq!(
            pick(&[(ArchetypeId::Hotel, f64::INFINITY)], 0.5),
            ArchetypeId::House
        );
        assert_eq!(pick(t, 0.999_999_999), ArchetypeId::Shop);
    }

    #[test]
    fn a_segment_meets_a_rectangle_only_when_it_really_does() {
        let (lo, hi) = (DVec2::new(-1.0, -1.0), DVec2::new(1.0, 1.0));
        assert!(segment_hits_rect(
            DVec2::new(-5.0, 0.0),
            DVec2::new(5.0, 0.0),
            lo,
            hi
        ));
        assert!(segment_hits_rect(DVec2::ZERO, DVec2::new(0.5, 0.5), lo, hi));
        assert!(!segment_hits_rect(
            DVec2::new(-5.0, 2.0),
            DVec2::new(5.0, 2.0),
            lo,
            hi
        ));
        assert!(!segment_hits_rect(
            DVec2::new(2.0, -5.0),
            DVec2::new(2.0, 5.0),
            lo,
            hi
        ));
        // A degenerate segment is a point test.
        assert!(segment_hits_rect(DVec2::ZERO, DVec2::ZERO, lo, hi));
        assert!(!segment_hits_rect(
            DVec2::new(9.0, 9.0),
            DVec2::new(9.0, 9.0),
            lo,
            hi
        ));
    }

    /// **A block that straddles the shore is refused**, and the refusal is a
    /// proof rather than a sample: a corner test alone would admit a rectangle
    /// with an inlet cut through the middle of it.
    #[test]
    fn a_block_that_straddles_the_shore_is_refused() {
        // A square island with a deep inlet biting into its north edge.
        let land = Land::of(&[vec![
            DVec2::new(-100.0, -100.0),
            DVec2::new(100.0, -100.0),
            DVec2::new(100.0, 100.0),
            DVec2::new(10.0, 100.0),
            DVec2::new(10.0, 10.0),
            DVec2::new(-10.0, 10.0),
            DVec2::new(-10.0, 100.0),
            DVec2::new(-100.0, 100.0),
        ]]);
        // Wholly inland.
        assert!(land.contains_rect(DVec2::new(-50.0, -50.0), DVec2::new(-20.0, -20.0)));
        // Wholly at sea.
        assert!(!land.contains_rect(DVec2::new(200.0, 200.0), DVec2::new(220.0, 220.0)));
        // **The corner trap**: all four corners are on land and the inlet runs
        // straight through it.
        let (lo, hi) = (DVec2::new(-40.0, 20.0), DVec2::new(40.0, 60.0));
        for c in [lo, DVec2::new(hi.x, lo.y), hi, DVec2::new(lo.x, hi.y)] {
            assert!(
                land.contains_point(c),
                "the fixture's own corner {c:?} must be on land or this arm \
                 proves nothing"
            );
        }
        assert!(
            !land.contains_rect(lo, hi),
            "a block with a sea inlet through the middle of it was admitted by \
             its corners"
        );
        // An island with no committed shore admits everything rather than
        // refusing every settlement.
        assert!(Land::of(&[]).contains_rect(lo, hi));
    }

    #[test]
    fn a_street_measures_the_distance_to_itself() {
        let s = Street {
            a: DVec2::new(-10.0, 0.0),
            b: DVec2::new(10.0, 0.0),
            main: true,
        };
        assert_eq!(s.length_m(), 20.0);
        assert_eq!(s.distance_to(DVec2::ZERO), 0.0);
        assert_eq!(s.distance_to(DVec2::new(0.0, 3.0)), 3.0);
        assert_eq!(s.distance_to(DVec2::new(14.0, 0.0)), 4.0);
        // A degenerate street answers rather than dividing by zero.
        let p = Street { b: s.a, ..s };
        assert_eq!(
            p.distance_to(DVec2::new(0.0, 5.0)),
            (100.0f64 + 25.0).sqrt()
        );
    }

    // ── the grid as a route (NPC1c) ─────────────────────────────────────────

    /// A settlement written by hand, so the grid arithmetic is measured even in
    /// a tree whose samples have not been blessed.
    ///
    /// The lines are laid out the way `plan_site` lays its own out — a line
    /// every `pitch` from the centre, both ends stopping at `buildable` — at the
    /// CI fixture city's own numbers. It is a fixture and not a second planner:
    /// [`check_grid_arithmetic`] is run over it *and* over every committed
    /// settlement, so the claim it pins is the claim the real ones satisfy.
    fn hand_grid() -> Settlement {
        let centre = DVec2::new(1000.0, -500.0);
        let (pitch, street, buildable) = (TOWN_BLOCK_M + TOWN_STREET_M, TOWN_STREET_M, 104.0);
        let mut streets = Vec::new();
        for k in -1..=1i32 {
            let off = f64::from(k) * pitch;
            streets.push(Street {
                a: DVec2::new(centre.x + off, centre.y - buildable),
                b: DVec2::new(centre.x + off, centre.y + buildable),
                main: k == 0,
            });
            streets.push(Street {
                a: DVec2::new(centre.x - buildable, centre.y + off),
                b: DVec2::new(centre.x + buildable, centre.y + off),
                main: k == 0,
            });
        }
        Settlement {
            site: 3,
            name: "Handgrid".into(),
            kind: SiteKind::Town,
            centre,
            radius_m: buildable + street,
            buildable_m: buildable,
            pitch_m: pitch,
            street_m: street,
            streets,
            blocks: Vec::new(),
            refused_off_pad: 0,
            refused_off_land: 0,
        }
    }

    /// **The node and edge counts are the crossing arithmetic**: `L` lines each
    /// way cross at `L²` junctions, stop at `4L` ends, and chain `L + 1` links
    /// along each of `2L` lines.
    fn check_grid_arithmetic(plan: &Settlement) {
        let g = plan.street_graph();
        let lines = plan.streets.iter().filter(|s| runs_along_z(s)).count();
        assert_eq!(
            lines,
            plan.streets.iter().filter(|s| runs_along_x(s)).count(),
            "{}: the grid is not square",
            plan.name
        );
        // A reservation that is a whole number of blocks deep would put the
        // outermost line exactly ON the end, and then the ends ARE crossings and
        // there are fewer nodes than the formula says. No committed settlement
        // is one; the assert is here so that the day one is, this arm fails
        // loudly rather than quietly measuring something else.
        assert!(
            !plan
                .streets
                .iter()
                .any(|s| runs_along_z(s) && (s.a.x - plan.centre.x).abs() == plan.buildable_m),
            "{}: its outermost street line lands exactly on the buildable radius, \
             so this arm's node count no longer holds",
            plan.name
        );
        assert_eq!(
            g.len(),
            lines * lines + 4 * lines,
            "{}: {lines} lines each way",
            plan.name
        );
        assert_eq!(
            g.edge_count(),
            4 * lines * (lines + 1),
            "{}: {lines} lines each way",
            plan.name
        );
        for n in g.nodes() {
            assert_eq!(inf_nav::domain::of(n.id), inf_nav::domain::STREET);
            assert_eq!(
                (n.id >> 40) & NODE_FIELD_MASK,
                plan.site as u64,
                "a node of {} carries another site's id field",
                plan.name
            );
            assert_eq!(n.kind, inf_nav::NavKind::Street);
            // The plan carries no elevation — the module's own first law.
            assert_eq!(n.position.y, 0.0);
            assert!(n.position.is_finite());
        }
    }

    /// **A hand-laid grid is exactly its arithmetic**, and its ids are the
    /// documented layout rather than an iteration order.
    #[test]
    fn a_street_grids_node_count_is_its_crossing_arithmetic() {
        let plan = hand_grid();
        check_grid_arithmetic(&plan);
        let g = plan.street_graph();
        assert_eq!(g.len(), 3 * 3 + 4 * 3);
        // The crossroads is at the site's own centre, and its id is the one the
        // layout table describes: site 3, both fields on the bias.
        let mid = LATTICE_BIAS as u64;
        assert_eq!(
            plan.centre_node_id(),
            inf_nav::domain::STREET | (3 << 40) | (mid << 20) | mid
        );
        let node = g.node(plan.centre_node_id()).expect("the crossroads");
        assert_eq!(node.position, DVec3::new(plan.centre.x, 0.0, plan.centre.y));
        // …and the id really is a function of the design: the same plan built
        // with its streets in the reverse order is the same graph, node for node
        // and edge for edge.
        let mut reversed = plan.clone();
        reversed.streets.reverse();
        assert_eq!(reversed.street_graph(), g);
        // A settlement whose reservation planned no grid answers an empty graph
        // rather than dividing by a zero pitch.
        let empty = Settlement {
            pitch_m: 0.0,
            streets: Vec::new(),
            ..plan.clone()
        };
        assert!(empty.street_graph().is_empty());
        println!(
            "NPC1c streets: a 3-line town grid is {} nodes / {} directed edges, \
             crossroads {:#x}",
            g.len(),
            g.edge_count(),
            plan.centre_node_id()
        );
    }

    /// **A settlement's grid is one connected network**, its centre really is
    /// the site's own, and an arriving island road **welds** onto it — the join
    /// the module docs claim, as a route rather than as a distance.
    #[test]
    fn a_settlements_street_grid_is_one_network_the_island_road_welds_onto() {
        let Some(d) = crate::island::committed_design(crate::island::ISLAND_RECIPES[0]) else {
            eprintln!("SKIP: no committed island design");
            return;
        };
        let plans = settlements(&d);
        assert!(!plans.is_empty());
        let mut nodes = 0usize;
        let mut edges = 0usize;
        for plan in &plans {
            check_grid_arithmetic(plan);
            let g = plan.street_graph();
            let centre = plan.centre_node_id();
            let at = g
                .node(centre)
                .unwrap_or_else(|| panic!("{} has no central crossroads", plan.name));
            let off = (DVec2::new(at.position.x, at.position.z) - plan.centre).length();
            assert!(
                off <= 1.0,
                "{}'s crossroads sits {off:.3} m off the site's own centre — an \
                 island route lands there and would hand over to nothing",
                plan.name
            );

            // Connected, the strong way: every node is reachable from the
            // crossroads. `L² + 4L` searches over a graph of the same size, which
            // for the widest committed settlement is a few hundred microseconds.
            let ids: Vec<(inf_nav::NavNodeId, DVec3)> =
                g.nodes().map(|n| (n.id, n.position)).collect();
            for (id, _) in &ids {
                let v = inf_nav::route(&g, centre, *id);
                assert!(
                    v.is_found(),
                    "{}: {} from the crossroads to node {id:#x}",
                    plan.name,
                    v.reason()
                );
            }
            // …and the two most distant nodes really do join, which is the
            // corner-to-corner claim rather than the spoke-to-hub one.
            let mut far = (0.0f64, ids[0].0, ids[0].0);
            for (a, pa) in &ids {
                for (b, pb) in &ids {
                    let d = (*pb - *pa).length();
                    if d > far.0 {
                        far = (d, *a, *b);
                    }
                }
            }
            let v = inf_nav::route(&g, far.1, far.2);
            let r = v
                .route()
                .unwrap_or_else(|| panic!("{}: its two most distant nodes do not join", plan.name));
            assert!(
                r.cost_m >= far.0 - 1e-9,
                "a route shorter than its own chord"
            );

            // **THE JOIN.** An island highway terminates on the site's own
            // (x, z); dropping a road-domain node there and welding must make
            // one network. 2 m is `inf_gis::SNAP_TOLERANCE_M`, the tolerance the
            // road layer already derives its own junctions at.
            let mut joined = g.clone();
            let arrival = inf_nav::domain::ROAD | plan.site as u64;
            joined.add_node(
                arrival,
                DVec3::new(plan.centre.x, 0.0, plan.centre.y),
                inf_nav::NavKind::Road,
            );
            let welded = joined.weld(2.0, 1.0);
            assert_eq!(
                welded, 1,
                "{}: an arriving road welded to {welded} street nodes rather \
                 than to the one crossroads",
                plan.name
            );
            // …and welding again adds nothing, so a caller may fold the same two
            // graphs twice without doubling the frontier.
            assert_eq!(joined.weld(2.0, 1.0), 0);
            let v = inf_nav::route(&joined, arrival, far.2);
            assert!(
                v.is_found(),
                "{}: an arriving island road cannot reach the far corner of the \
                 grid it landed on ({})",
                plan.name,
                v.reason()
            );
            println!(
                "NPC1c streets: {:>13} ({:>4}) {} nodes / {} edges, crossroads \
                 {off:.3} m off centre, corner-to-corner {:.1} m of route over a \
                 {:.1} m chord, 1 weld onto the arriving road",
                plan.name,
                plan.kind.label(),
                g.len(),
                g.edge_count(),
                r.cost_m,
                far.0
            );
            nodes += g.len();
            edges += g.edge_count();
        }
        println!(
            "NPC1c streets TOTAL: {} settlements, {nodes} nodes, {edges} directed \
             edges",
            plans.len()
        );
    }
    /// **THE NIGHTLIFE STRIP, on both committed recipes** (wave VEN1a).
    ///
    /// Two claims, and the second is the one that costs something to state:
    ///
    /// * the shipped island's every settlement gets a strip, and its cities get
    ///   all three kinds — so "at least one venue of each kind is placed" is a
    ///   measurement and not a hope;
    /// * the **CI fixture** gets exactly one bar, in its camp and not in its
    ///   town, and that is correct rather than a bug. `Fixture Town` is a
    ///   four-block reservation wearing a `city` label: a city's strip starts at
    ///   ring 1 (ring 0 is the office core) and `VENUE_SHARE` refuses to spend
    ///   more than a third of a settlement on nightlife, so four blocks buy
    ///   none. A hamlet with a nightclub is stranger than a hamlet without one.
    ///
    /// Stated here because a gate that walks into a club has to know which
    /// recipe has one, and because a later wave that grows the fixture will
    /// change this number and should be made to say so.
    #[test]
    fn the_nightlife_strip_is_one_per_settlement_and_three_kinds_per_city() {
        let mut kinds: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for (which, recipe) in crate::island::ISLAND_RECIPES.iter().enumerate() {
            let Some(design) = crate::island::committed_design(recipe) else {
                eprintln!("SKIP: no committed island design for {recipe}");
                return;
            };
            let plans = settlements(&design);
            assert!(!plans.is_empty(), "{recipe} reserves no urban ground");
            for plan in &plans {
                let mut v: Vec<String> = plan
                    .blocks
                    .iter()
                    .filter(|b| b.archetype.is_venue())
                    .map(|b| b.archetype.name().to_string())
                    .collect();
                v.sort();
                println!(
                    "VEN1a strip: {recipe} {} ({} blocks) -> {v:?}",
                    plan.name,
                    plan.blocks.len()
                );
                for n in &v {
                    kinds.insert(match n.as_str() {
                        "Bar" => "Bar",
                        "Nightclub" => "Nightclub",
                        _ => "StripClub",
                    });
                }
                // A settlement never spends more than a third of itself on
                // nightlife — the `VENUE_SHARE` guard, asserted rather than
                // trusted.
                assert!(
                    v.len() * VENUE_SHARE <= plan.blocks.len().max(VENUE_SHARE),
                    "{}: {} venues on {} blocks",
                    plan.name,
                    v.len(),
                    plan.blocks.len()
                );
                // Every settlement on the SHIPPED island has somewhere to go
                // at night; the fixture's four-block town is the documented
                // exception and is checked by the count below instead.
                if which == 0 {
                    assert!(!v.is_empty(), "{} has no nightlife at all", plan.name);
                }
            }
        }
        assert_eq!(
            kinds.into_iter().collect::<Vec<_>>(),
            vec!["Bar", "Nightclub", "StripClub"],
            "the committed recipes do not place one venue of each kind"
        );
    }

    /// **THE CIVIC STRIP, on both committed recipes** (wave EMS1) — the
    /// nightlife strip's arm, one wave later, plus the three claims that are
    /// only true of institutions.
    ///
    /// * every settlement on the shipped island has a **fire hall**, because
    ///   the alternative is that it burns down and because it is the first
    ///   entry of both strips;
    /// * the cities and only the cities have a **hospital** and a **police
    ///   station** — a town's grid is 76 m on a side and a hospital's lot is
    ///   52 × 50 m, and a station with no city to police is a cell block with
    ///   nobody in it;
    /// * no settlement spends more than a quarter of itself on civic buildings,
    ///   the `CIVIC_SHARE` guard asserted rather than trusted.
    ///
    /// And the ordering claim the whole placement rests on: **the venue strip
    /// is unchanged**. Both strips take blocks nearest the centre, so a civic
    /// strip evaluated first would eat a city's strip club — and two island-gate
    /// arms pick their target venue by block order, so a venue that moved would
    /// move a whole PIE-versus-shipping trace. The arm above is the pin on that;
    /// this one records the count that would have changed.
    #[test]
    fn the_civic_strip_gives_every_settlement_a_hall_and_the_cities_a_hospital() {
        let mut kinds: std::collections::BTreeSet<String> = Default::default();
        for (which, recipe) in crate::island::ISLAND_RECIPES.iter().enumerate() {
            let Some(design) = crate::island::committed_design(recipe) else {
                eprintln!("SKIP: no committed island design for {recipe}");
                return;
            };
            for plan in settlements(&design) {
                let mut v: Vec<String> = plan
                    .blocks
                    .iter()
                    .filter(|b| b.archetype.is_institution())
                    .map(|b| b.archetype.name().to_string())
                    .collect();
                v.sort();
                println!(
                    "EMS1 civic strip: {recipe} {} ({:?}, {} blocks) -> {v:?}",
                    plan.name,
                    plan.kind,
                    plan.blocks.len()
                );
                for n in &v {
                    kinds.insert(n.clone());
                }
                // **A LITERAL, NOT `CIVIC_SHARE`** (EMS1 audit). Written with
                // the constant on both sides this was satisfied for *any* value
                // of it — the guard could never see itself loosen. Measured:
                // `CIVIC_SHARE = 1` left this arm green with a four-block
                // settlement spending all four blocks on institutions. A
                // quarter is the claim about the WORLD, so a quarter is what is
                // written; if the rule is ever meant to change, this number
                // changes with it, deliberately.
                const MOST: usize = 4;
                assert!(
                    v.len() * MOST <= plan.blocks.len().max(MOST),
                    "{}: {} institution(s) on {} blocks — a settlement is mostly \
                     the places people live and work in",
                    plan.name,
                    v.len(),
                    plan.blocks.len()
                );
                // A hospital and a station are CITY facts. Asserted as an
                // implication rather than as a count, because "a town has no
                // hospital" is the half a widened `civic_strip` would silently
                // break.
                if plan.kind != SiteKind::City {
                    for c in ["Hospital", "PoliceStation"] {
                        assert!(
                            !v.iter().any(|n| n == c),
                            "{} is a {:?} and has a {c}",
                            plan.name,
                            plan.kind
                        );
                    }
                }
                if which == 0 {
                    assert!(
                        v.iter().any(|n| n == "FireHall"),
                        "{} has no fire hall — every settlement has one, or the \
                         strip's first entry is not reaching the smallest \
                         reservations",
                        plan.name
                    );
                }
            }
        }
        assert_eq!(
            kinds.into_iter().collect::<Vec<_>>(),
            vec!["Clinic", "FireHall", "Hospital", "PoliceStation"],
            "the committed recipes do not place one institution of each kind"
        );
    }

    /// **A CITY KEEPS ITS CORE** (wave EMS1) — the civic floor, pinned.
    ///
    /// Written as an arm rather than as a comment for
    /// `the_industrial_floor_is_the_outer_half_and_never_the_core`'s reason: a
    /// min-ring rule that quietly went back to `0` would put a fire hall on a
    /// city's crossroads, and the only thing in the tree that noticed last time
    /// was another wave's walk gate, reporting it as a building with no street
    /// on its storey.
    #[test]
    fn a_citys_civic_strip_starts_one_ring_out_and_a_towns_does_not() {
        assert_eq!(
            civic_min_ring(SiteKind::City),
            1,
            "a fire hall took a city's office core"
        );
        for kind in [SiteKind::Town, SiteKind::Waypoint] {
            assert_eq!(
                civic_min_ring(kind),
                0,
                "{kind:?}: a town's ring 0 is its high street, not a core"
            );
        }
        // …and it is the venue floor exactly, which is the point: the two
        // strips want the same ground for the same reason.
        for kind in [SiteKind::City, SiteKind::Town, SiteKind::Waypoint] {
            assert_eq!(civic_min_ring(kind), venue_min_ring(kind), "{kind:?}");
        }
        assert_eq!(venue_min_ring(SiteKind::City), 1, "the venue floor moved");
        // …and the two strips never name the same archetype, or one block would
        // be claimed by whichever rule ran first.
        for kind in [SiteKind::City, SiteKind::Town] {
            for a in civic_strip(kind) {
                assert!(a.is_institution(), "{a:?} is on the civic strip");
                assert!(!a.is_venue(), "{a:?} is on both strips");
            }
            for a in venue_strip(kind) {
                assert!(!a.is_institution(), "{a:?} is on both strips");
            }
        }
        // A town has the two that scale down and neither of the two that do
        // not, which is the whole of `civic_strip`'s doc as an assertion.
        let town: Vec<&str> = civic_strip(SiteKind::Town)
            .iter()
            .map(|a| a.name())
            .collect();
        assert_eq!(town, vec!["FireHall", "Clinic"]);
        assert_eq!(civic_strip(SiteKind::City).len(), 4);
    }
}
