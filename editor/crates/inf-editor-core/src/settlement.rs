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

use glam::DVec2;
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

/// The most blocks one city's industrial cluster may take, as a share of the
/// blocks it is allowed to sit on.
///
/// A quarter, capped at [`CITY_INDUSTRIAL_BLOCKS`]. Without the share a small
/// settlement's whole outer ring becomes a works: the fixture's city has eight
/// eligible blocks and four of them would be half the town.
pub const INDUSTRIAL_SHARE: usize = 4;

/// **Which archetypes are furnished** (island wave I8a, ruling 3).
///
/// The orchestrator's ruling was *measure, then decide, default ON*, and the
/// measurement is `the_furnish_battery_prices_a_city_block_at_island_scale` in
/// `runtime/inf-player/tests/island_gate.rs`. What it found is in the wave's
/// ledger; what it decided is here, in one place, read by all seven zone
/// documents so a reader cannot find two answers.
///
/// The split is not a compromise for its own sake — it is the shape the
/// measurement implies. Furniture is per **room**, so its cost scales with
/// storeys, and the archetypes a player walks into on foot (a house, a shop, an
/// estate) are the one- to four-storey ones. A ten-storey hotel is 5 × the rooms
/// of a house for a lobby nobody has walked past yet.
pub fn furnishes(a: ArchetypeId) -> bool {
    matches!(
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

    let name_seed = Hash64::new(design.recipe.seed)
        .mix_u64(SETTLEMENT_SALT)
        .mix_u64(site as u64);
    let blocks = cells
        .iter()
        .map(|(col, row, ring, c)| {
            let h = name_seed.mix_i64(i64::from(*col)).mix_i64(i64::from(*row));
            let archetype = if industrial.contains(&(*col, *row)) {
                ArchetypeId::Industrial
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
    out.sort();
    out
}

/// Write the seven zone documents and the README.
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
    std::fs::write(dir.join("README.md"), SETTLEMENT_README)
        .map_err(|e| format!("write readme: {e}"))
}

/// The library's own README.
pub const SETTLEMENT_README: &str = "# The settlement zone library (island wave I8a)\n\n\
Generated by `inf_editor_core::settlement::write_settlement_library`. **Seven\n\
`.inf_pcg` documents, one per building archetype**, each a few hundred bytes of\n\
RULES: a footprint span over the evaluating volume's own extent, IB-2c's\n\
`building.lots` subdivision, and a `building.plan` on the archetype's palette.\n\n\
They are ENGINE content, like `samples/ground` and `samples/starter-character`:\n\
one copy, byte-locked against the generator, named by both islands' recipes\n\
under `[content]` and by every settlement block's `PcgVolume.graph`.\n\n\
## Why seven documents and not one per block\n\n\
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
}
