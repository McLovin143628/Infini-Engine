//! **Traffic** (wave VEH2b): the streets a level re-derives for itself, and the
//! driver that holds a car on one.
//!
//! # Two halves, and why they are in one file
//!
//! The first half is a *map*: [`carriageway`] reads a level's own block
//! rectangles and answers the street grid between them, as an
//! [`inf_nav::LaneNetwork`]. The second is a *driver*:
//! [`drive_intent`] reads where a car is against a lane and answers the same
//! `Vec2d` a player's stick produces. They are together because the second is
//! meaningless without the first and the first exists only for the second.
//!
//! Neither of them touches rapier, and that is the split this crate already
//! makes twice — `inf_ecs::vehicle` decides and `inf_physics::d3::vehicle`
//! applies; `inf_ecs::movement` decides and `inf_physics::d3::movement` applies.
//! [`inf_physics::d3::traffic`] is the third.
//!
//! # THE STREETS ARE RE-DERIVED, and this is the honest reason
//!
//! `inf_editor_core::settlement::Settlement::street_graph` already knows every
//! street of every settlement, exactly, from the plan that placed the blocks.
//! The shipped player cannot have it: that is Ring 1, the plan is not a
//! component, and wave VEH2a spent this arc's schema window on
//! `VehicleClass` — so there is nowhere in a committed `.inf_lvl` for a street
//! centreline to live.
//!
//! What a committed level *does* carry is the blocks: a `PcgVolume` per block,
//! with a centre and an axis-aligned extent, which is the same data
//! [`crate::society`] already reads every fixed step to lay its pavements. A
//! street is the **gap between two of them**, and its centreline is the middle
//! of that gap. So the grid is recovered from the ground rather than from the
//! plan, both hosts recover it identically because they read the same world, and
//! `the_derived_carriageway_is_the_settlements_own_street_grid` in the island
//! gate holds the recovery against the plan it is recovering.
//!
//! The price is stated: only the streets **between** blocks are recovered. The
//! outermost grid line of a settlement has ground on one side and no block to
//! bound it, so it is not a gap and it is not derived. That is a smaller network
//! than the plan drew, and it is a network entirely inside the town.
//!
//! # No schema moves
//!
//! [`TrafficRes`] is a bevy **resource** and every traffic body is spawned at
//! runtime, exactly as [`crate::crowd::CrowdPopulationRes`] is: the `.inf_lvl`
//! walk writes `RuntimeEntity` fields and never a resource, so nothing here can
//! be saved and **scene v27 does not move**.
//!
//! [`inf_physics::d3::traffic`]: https://docs.rs/inf-physics

use std::collections::{BTreeMap, BTreeSet};

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_nav::lane::{right_of, LaneNetwork, LaneSpec};
use inf_nav::{NavGraph, NavKind, NavNodeId, NavPath};

use crate::math::Vec2d;
pub use inf_nav::lane::DEFAULT_LANE_WIDTH_M;
pub use inf_nav::NavPath as LanePath;

use crate::crowd::CrowdTier;
use crate::society::{mix64, rect_gap_2d, volume_sites, PAVEMENT_LATTICE_M, PAVEMENT_M};
use crate::world::EcsWorld;

// ── the map ─────────────────────────────────────────────────────────────────

/// The widest gap between two blocks that is still a **street** and not
/// countryside, metres.
///
/// `inf_editor_core::settlement` reserves 20 m for a city street and 16 m for a
/// town's, so 32 is comfortably over the widest thing this engine plans and
/// comfortably under the hundreds of metres between one settlement's blocks and
/// the next settlement's. It is what makes the derivation need no clustering
/// pass of its own beyond the one it does: two towns are not neighbours.
pub const MAX_STREET_GAP_M: f64 = 32.0;

/// The narrowest gap that is a street, metres.
///
/// Under six metres two blocks are an alley or a modelling accident, and a
/// carriageway laid in one would put opposing traffic 3.5 m apart with no
/// pavement either side. A gap this small is left as ground.
pub const MIN_STREET_GAP_M: f64 = 6.0;

/// How near a street line a block has to be to say what its ground height is,
/// metres.
///
/// A block and a half of a city grid. Wider and a street takes its height from
/// blocks on the far side of the town; narrower and a line that happens to run
/// along the edge of a settlement finds nothing and falls back to the cluster.
pub const PAD_REACH_M: f64 = 150.0;

/// The sign on a settlement street, km/h.
///
/// Thirty, because these are the streets a town's own residents walk across:
/// `inf_ecs::society` links two blocks' pavements straight over the gap
/// ([`crate::society::BLOCK_LINK_MAX_M`]), so every crossing in a settlement is
/// a place a pedestrian steps into the road. The island's *circuit* is a
/// different class and carries `inf_gis::default_speed_kmh`'s own numbers.
pub const STREET_SPEED_KMH: u32 = 30;

/// [`STREET_SPEED_KMH`] as a speed, m/s — the conversion at the point of use.
///
/// A function rather than a `const` because the units doctrine's carve-out is
/// that a *sign* is km/h and a *physical quantity* is SI, and this is the one
/// place the crossing happens for a settlement street. Every lane the runtime
/// derivation lays carries the same sign today, which is why the controller can
/// be handed a constant; the day a level has a fast road in it, this is the
/// call site that becomes `lane.speed_limit_mps()`.
pub fn street_speed_mps() -> f64 {
    inf_nav::lane::kmh_to_mps(f64::from(STREET_SPEED_KMH))
}

/// **A kerb stone's width, metres** — pinned by value to `inf_gis::KERB_WIDTH_M`,
/// which is the concrete that actually gets drawn (wave ROAD1b).
///
/// One stone. It is stated twice because `inf-ecs` and `inf-gis` cannot name
/// each other, and `inf-editor-core` — which links both — asserts the equality
/// in `road_authority`, exactly as it already does for [`PAVEMENT_M`] and
/// [`DEFAULT_LANE_WIDTH_M`]. Before this wave nothing here needed it, because
/// nothing here knew a street had a kerb at all.
pub const KERB_WIDTH_M: f64 = 0.30;

/// **A kerb's upstand, metres** — pinned by value to `inf_gis::KERB_HEIGHT_M`,
/// the height the footway is actually drawn at (wave ROAD1b).
///
/// The standard a highway authority specifies: it stops a wheel, holds a
/// gutter, and a person steps up it without thinking. It is stated here because
/// `inf_physics` builds the **collider** that makes that step real and cannot
/// name `inf-gis`; `road_authority` asserts the equality.
pub const KERB_HEIGHT_M: f64 = 0.15;

/// **How wide a settlement street's carriageway is**, half-width in metres, for
/// a street recovered from a `gap_m` reserve (wave ROAD1b).
///
/// # The rule, and the two things it has to satisfy at once
///
/// A street's reserve runs from block edge to block edge. What has to fit in it,
/// outward from the crown: the carriageway, a [`KERB_WIDTH_M`] stone, and
/// [`PAVEMENT_M`] of footway whose **back edge is the block's own frontage** —
/// because that is where `crate::society`'s pavement ring is, `PAVEMENT_M`
/// outside the block rectangle, and a crowd routed onto concrete that is not
/// there is a crowd walking beside its own pavement.
///
/// So the carriageway is everything the reserve has left:
/// `gap/2 − PAVEMENT_M − KERB_WIDTH_M`. On the two reserves this engine plans
/// that is **5.700 m** on a 16 m town street and **7.700 m** on a 20 m city one,
/// and in both cases the ring lands exactly `KERB_WIDTH_M` outside the kerb
/// face — on the footway, by construction rather than by coincidence.
///
/// A gap under [`MIN_STREET_GAP_M`] never reaches here; the derivation refuses
/// it. The `max` is the guard for one that somehow does.
pub fn street_carriageway_half_m(gap_m: f64) -> f64 {
    (gap_m * 0.5 - PAVEMENT_M - KERB_WIDTH_M).max(DEFAULT_LANE_WIDTH_M)
}

/// **The lane count a settlement street's reserve implies** (wave ROAD1b).
///
/// The paving is driven by a lane count, because that is what a road layer
/// carries and what `inf_gis::RoadKind::width_m` multiplies — so the width in
/// [`street_carriageway_half_m`] has to be expressed as lanes to be drawn at
/// all. Rounded rather than floored: a 16 m reserve wants 11.400 m and gets
/// **3** lanes (10.500 m), a 20 m reserve wants 15.400 m and gets **4**
/// (14.000 m), and in both the footway still reaches the block frontage with a
/// few tens of centimetres of verge rather than overhanging it.
///
/// Three is not a mistake and it is not a compromise: a three-lane city street
/// is one running lane each way and a shared middle, which is what a North
/// American town street of that width is.
pub fn street_lanes(gap_m: f64) -> u32 {
    let want = street_carriageway_half_m(gap_m) * 2.0 / DEFAULT_LANE_WIDTH_M;
    if !want.is_finite() {
        return 2;
    }
    (want.round() as i64).clamp(2, 8) as u32
}

/// **Where a settlement street's kerb face actually is**, metres from the
/// centreline (wave ROAD1b).
///
/// [`street_carriageway_half_m`] is what the reserve *wants*;
/// [`street_lanes`] rounds it to a whole lane because a road layer states lanes
/// and `inf_gis::RoadKind::width_m` multiplies them. This is the number that
/// comes back out — **the kerb the paving draws** — and it is what anything
/// measuring against a kerb must read.
///
/// 5.250 m on a 16 m reserve (3 lanes) and 7.000 m on a 20 m one (4 lanes),
/// against the 5.700 and 7.700 the reserve wanted. The residual is verge
/// between the footway's back and the block frontage, which is what a setback
/// is; it is the *rounding* that must not be re-derived twice, and this is the
/// one place it is undone.
pub fn street_kerb_offset_m(gap_m: f64) -> f64 {
    f64::from(street_lanes(gap_m)) * DEFAULT_LANE_WIDTH_M * 0.5
}

/// **Half a parked car's width, metres.**
///
/// 0.9 m is a saloon's 1.8 m body. It is the figure that was already inside
/// [`KERB_PARK_OFFSET_M`]'s arithmetic and is now spelled, because
/// [`kerb_park_offset_m`] has to subtract it from a kerb whose position depends
/// on the street.
pub const PARKED_CAR_HALF_W_M: f64 = 0.9;

/// **Where a car SHOULD park on a street of this reserve** — its centre's
/// offset from the centreline, metres, so its flank is at the kerb the paving
/// draws (wave ROAD1b).
///
/// # It is the right number and `kerb_slots` does not use it yet
///
/// [`KERB_PARK_OFFSET_M`] is 5.0 m on every street and was derived against a
/// carriageway no settlement street ever had. Measured against the kerb wave
/// ROAD1b actually draws ([`street_kerb_offset_m`]), that constant parks a car
/// **0.650 m onto the footway** on a 16 m town street and **1.100 m out into
/// the road** on a 20 m city one. This function is what puts the flank on the
/// kerb: 4.350 m and 6.100 m respectively.
///
/// **Moving the lattice to it deadlocks the emergency service, measured.**
/// Wiring this into [`kerb_slots`] turns `inf-physics`'s `dispatch_3d` red on
/// three arms, and the middle one is not a re-sample: with the row 1.1 m
/// further out on a 20 m street the ambulance in
/// `a_collapse_brings_the_ambulance_and_sends_it_home_again` arrives (step
/// 1 799) and resolves (2 159) and then **never goes home in 30 000 steps** —
/// five hundred seconds of simulation, against a test budget of 6 000. It is a
/// stuck vehicle, not a slow one.
///
/// So the number is stated here, the discrepancy is pinned by
/// `the_parked_car_lattice_is_not_yet_on_the_kerb_the_paving_draws` in
/// `inf-editor-core`'s `road_authority`, and the move is carried for the wave
/// that can also answer why a unit cannot return to a kerb 1.1 m further out.
/// Landing it here would have been trading a car in the right place for an
/// ambulance that never comes back.
pub fn kerb_park_offset_m(gap_m: f64) -> f64 {
    (street_kerb_offset_m(gap_m) - PARKED_CAR_HALF_W_M).max(0.0)
}

/// How far from a street's centreline a car parks at the kerb, metres.
///
/// Half the carriageway (3.5 m — one lane each way at
/// [`DEFAULT_LANE_WIDTH_M`]) plus a car's own half width and a hand's clearance.
/// It has to leave [`PAVEMENT_M`] free at the kerb, which is what
/// [`kerb_fits`] checks: on the narrowest street this engine plans (16 m) there
/// are 6.0 m between the centreline and the pavement and this uses 5.0 of them.
///
/// **It is not the kerb the paving draws** — see [`kerb_park_offset_m`] for the
/// measurement and for what moving to it costs.
pub const KERB_PARK_OFFSET_M: f64 = 5.0;

/// Metres of kerb one parked car occupies.
///
/// A saloon is 4.4 m long and this engine's longest catalogue row is a 5.4 m
/// van, so fourteen metres is a car plus enough room to get out of the space —
/// which is what makes a kerb look parked rather than shunted. It also bounds
/// the population: with a slot's clearance at each end, a 100 m run holds
/// **six** a side before the occupancy draw.
pub const KERB_SLOT_M: f64 = 14.0;

/// Whether a street of this width has room to park at the kerb.
///
/// Half the gap, less the pavement, has to hold [`KERB_PARK_OFFSET_M`] plus a
/// metre for the car's own body. A narrower street gets lanes and no parking,
/// which is the honest answer rather than a car on the pavement.
pub fn kerb_fits(gap_m: f64) -> bool {
    gap_m * 0.5 - PAVEMENT_M >= KERB_PARK_OFFSET_M + 1.0
}

/// **A crossing's node id** — the domain tag over a hash of its own position.
///
/// [`crate::society::pavement_node_id`]'s layout and its reasons, one domain
/// along: a level that has been committed, cooked and paged back in has no
/// settlement plan to name a crossing `(site, column, row)` by, so the only key
/// it has is where the crossing is.
pub fn carriageway_node_id(p: DVec2) -> NavNodeId {
    let q = |v: f64| {
        if v.is_finite() {
            (v / PAVEMENT_LATTICE_M).round() as i64
        } else {
            0
        }
    };
    let h = mix64((q(p.x) as u64) ^ mix64(q(p.y) as u64));
    inf_nav::domain::CARRIAGEWAY | (h & inf_nav::domain::LOCAL_MASK)
}

/// One recovered street: a centreline and the gap it was found in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Street {
    /// One end, world XZ.
    pub a: DVec2,
    /// The other, world XZ.
    pub b: DVec2,
    /// The walking surface the blocks either side stand on, world Y.
    pub y: f64,
    /// The reserve it was recovered from, metres — the distance between the two
    /// block edges it runs down the middle of.
    pub gap_m: f64,
}

impl Street {
    /// Whether this line runs along world X (a constant Z).
    pub fn along_x(&self) -> bool {
        self.a.y == self.b.y
    }
}

/// **The streets a level's blocks imply** — [`streets_of_blocks`] over the
/// `PcgVolume`s of a world.
///
/// # What "a level's blocks" means here, exactly
///
/// `volume_sites` answers the volumes that offer a **resident**, which a
/// `PcgVolume` only does once it has been *evaluated*. So this reads the blocks
/// a host has populated, not the blocks a level carries — and under cell
/// streaming those are not the same set. That is why wave ROAD1b paves from the
/// island's **authored** plan through [`streets_of_blocks`] rather than from
/// this: a street that appeared and moved as blocks paged in would be a street
/// the editor and the shipped player disagreed about. The disagreement is
/// measured in `inf-editor-core`'s `road_authority` battery.
///
/// It runs when the block set **changes**, never per step — see
/// [`TrafficRes::stamp`].
pub fn streets_of(world: &EcsWorld) -> Vec<Street> {
    streets_of_blocks(
        &volume_sites(world)
            .into_iter()
            .map(|s| BlockRect {
                guid: s.guid,
                centre: DVec2::new(s.centre.x, s.centre.z),
                half: s.extent,
                pad_y: s.pad_y,
            })
            .collect::<Vec<_>>(),
    )
}

/// **One block, as the street derivation needs it** (wave ROAD1b) — a
/// rectangle on the ground, its identity, and the surface it stands on.
///
/// # Why it is a type and not four arguments
///
/// [`streets_of`] reads these out of an [`EcsWorld`]'s `PcgVolume`s, and until
/// this wave that was the only way to ask the question. Wave ROAD1b needs the
/// same answer at **island build time**, where the blocks are a plan and there
/// is no world yet — and a settlement whose streets are PAVED in one place and
/// DERIVED in another is two answers to where a street is. So the derivation
/// takes the rectangles and nothing else, and both callers reach it through
/// [`streets_of_blocks`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockRect {
    /// The block's stable identity. It decides the grouping's representative
    /// and therefore the order the whole answer is built in — see
    /// [`streets_of_blocks`].
    pub guid: Uuid,
    /// Its centre, world XZ.
    pub centre: DVec2,
    /// Its half-extent, world XZ — a `PcgVolume::extent`.
    pub half: DVec2,
    /// The walking surface it stands on, world Y.
    pub pad_y: f64,
}

/// **THE derivation** (wave ROAD1b) — the block rectangles in, the street
/// centrelines out, and the one implementation both the traffic sim and the
/// paving read.
///
/// [`streets_of`] is this over a world's `PcgVolume`s and
/// `inf_editor_core::island` is this over the island's authored blocks. There
/// is deliberately no second copy: the defect wave ROAD1b was called for is
/// that the island had TWO road networks, and a paving that re-derived these
/// lines its own way would have made it three.
///
/// The blocks are grouped by proximity ([`MAX_STREET_GAP_M`]) so two
/// settlements a kilometre apart do not imply a kilometre-wide street between
/// them; within a group each axis is reduced to its occupied intervals and the
/// gaps between consecutive intervals are the streets.
///
/// `O(blocks²)` for the grouping and `O(blocks log blocks)` for the rest.
///
/// The blocks are sorted by `Guid` on entry rather than assumed sorted, so the
/// answer is a function of the SET and not of the order a caller happened to
/// collect it in — `volume_sites` already sorts and the island's plan walk does
/// not.
pub fn streets_of_blocks(blocks: &[BlockRect]) -> Vec<Street> {
    let mut sites: Vec<BlockRect> = blocks.to_vec();
    sites.sort_by_key(|b| b.guid);
    if sites.is_empty() {
        return Vec::new();
    }
    // ── the grouping. Union-find over "close enough to have a street between
    //    them", walked in `Guid` order, so the group a block lands in is a
    //    function of the level and not of an iteration.
    let n = sites.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..n {
        for j in i + 1..n {
            let gap = rect_gap_2d(
                sites[i].centre,
                sites[i].half,
                sites[j].centre,
                sites[j].half,
            );
            if gap <= MAX_STREET_GAP_M {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    // Union toward the LOWER index, which is the lower `Guid`,
                    // so a group's representative is a function of the level.
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    parent[hi] = lo;
                }
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }

    let mut out: Vec<Street> = Vec::new();
    for members in groups.values() {
        // A single block has no gap to anything and therefore no street.
        if members.len() < 2 {
            continue;
        }
        // **The pad is per STREET, not per settlement.** A cluster's mean is the
        // same number to a centimetre on a levelled pad and is metres out on a
        // town that climbs — and a street laid metres under its own ground is a
        // row of cars inside a heightfield, which rapier answers by launching
        // them. (Measured: one car left the island fixture at 72 m/s.) So the
        // mean is taken over the blocks that actually BOUND each line, below.
        let pad_of = |a: DVec2, b: DVec2| -> f64 {
            let mut near: Vec<f64> = Vec::new();
            for &i in members {
                let c = sites[i].centre;
                let d = DVec2::new(
                    ((c.x - a.x).min(c.x - b.x)).max(0.0) + ((a.x - c.x).min(b.x - c.x)).max(0.0),
                    ((c.y - a.y).min(c.y - b.y)).max(0.0) + ((a.y - c.y).min(b.y - c.y)).max(0.0),
                );
                if (d.x * d.x + d.y * d.y).sqrt() <= PAD_REACH_M {
                    near.push(sites[i].pad_y);
                }
            }
            if near.is_empty() {
                near = members.iter().map(|&i| sites[i].pad_y).collect();
            }
            // **The MEDIAN, not the mean.** `volume_sites` falls back to a
            // volume's entity `y` when it offers no exterior ground-floor
            // doorway, and the island authors that as zero — so a settlement on
            // a pad at 130 m whose blocks are a mix answers 86 to a mean and
            // 130 to a median. Both are only a first guess (the ray in
            // `inf_physics::d3::traffic` is the authority), but a first guess
            // forty-five metres out is one a body can be built on before the
            // ray has anything to hit.
            near.sort_by(f64::total_cmp);
            near[near.len() / 2]
        };
        let span = |f: &dyn Fn(usize) -> (f64, f64)| -> (f64, f64) {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &i in members {
                let (a, b) = f(i);
                lo = lo.min(a);
                hi = hi.max(b);
            }
            (lo, hi)
        };
        let x_span = span(&|i| {
            (
                sites[i].centre.x - sites[i].half.x,
                sites[i].centre.x + sites[i].half.x,
            )
        });
        let z_span = span(&|i| {
            (
                sites[i].centre.y - sites[i].half.y,
                sites[i].centre.y + sites[i].half.y,
            )
        });
        if !(x_span.0.is_finite() && z_span.0.is_finite()) {
            continue;
        }
        // ── each axis's occupied intervals, merged.
        for axis_x in [true, false] {
            let mut iv: Vec<(f64, f64)> = members
                .iter()
                .map(|&i| {
                    let (c, e) = if axis_x {
                        (sites[i].centre.x, sites[i].half.x)
                    } else {
                        (sites[i].centre.y, sites[i].half.y)
                    };
                    (c - e, c + e)
                })
                .collect();
            iv.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
            let mut merged: Vec<(f64, f64)> = Vec::with_capacity(iv.len());
            for (lo, hi) in iv {
                match merged.last_mut() {
                    Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
                    _ => merged.push((lo, hi)),
                }
            }
            for w in merged.windows(2) {
                let gap = w[1].0 - w[0].1;
                if !(MIN_STREET_GAP_M..=MAX_STREET_GAP_M).contains(&gap) {
                    continue;
                }
                let mid = (w[0].1 + w[1].0) * 0.5;
                // A line found in the X intervals is a constant X, so it RUNS
                // along Z; the one found in the Z intervals runs along X.
                let (a, b) = if axis_x {
                    (DVec2::new(mid, z_span.0), DVec2::new(mid, z_span.1))
                } else {
                    (DVec2::new(x_span.0, mid), DVec2::new(x_span.1, mid))
                };
                out.push(Street {
                    a,
                    b,
                    y: pad_of(a, b),
                    gap_m: gap,
                });
            }
        }
    }
    // One order for the whole answer, so two hosts that grouped identically also
    // list identically. `total_cmp` because a NaN would otherwise make the sort
    // itself order-dependent.
    out.sort_by(|p, q| {
        p.a.x
            .total_cmp(&q.a.x)
            .then(p.a.y.total_cmp(&q.a.y))
            .then(p.b.x.total_cmp(&q.b.x))
            .then(p.b.y.total_cmp(&q.b.y))
    });
    out
}

/// **The street grid as a routable graph** — a node at every crossing and at
/// every line's two ends, consecutive nodes linked.
///
/// `Settlement::street_graph`'s shape, re-derived: the grid is orthogonal, so
/// every line along X crosses every line along Z that its span covers, and the
/// crossings are the junctions. What is different is the id — see
/// [`carriageway_node_id`] — and that the outermost line of a settlement is
/// missing, because it is not a gap.
pub fn carriageway_graph(streets: &[Street]) -> NavGraph {
    let mut g = NavGraph::new();
    let along_x: Vec<&Street> = streets.iter().filter(|s| s.along_x()).collect();
    let along_z: Vec<&Street> = streets.iter().filter(|s| !s.along_x()).collect();
    let add = |g: &mut NavGraph, p: DVec2, y: f64| -> NavNodeId {
        let id = carriageway_node_id(p);
        // **First writer wins.** `NavGraph::add_node` overwrites, and a crossing
        // is added once per street that carries it — so the last line walked
        // would silently decide the junction's height while the first line's
        // edge cost had already been measured against the other one. The walk is
        // in `streets_of`'s sorted order, so "first" is a function of the level.
        if !g.contains(id) {
            g.add_node(id, DVec3::new(p.x, y, p.y), NavKind::Street);
        }
        id
    };
    for s in streets {
        // The crossings this line carries, plus its two ends.
        let mut on: Vec<(f64, DVec2)> = Vec::new();
        let others = if s.along_x() { &along_z } else { &along_x };
        for o in others {
            let (cross, along) = if s.along_x() {
                // `o` is a constant X; this line is a constant Z.
                (DVec2::new(o.a.x, s.a.y), o.a.x)
            } else {
                (DVec2::new(s.a.x, o.a.y), o.a.y)
            };
            let (lo, hi, oa, ob) = if s.along_x() {
                (
                    s.a.x.min(s.b.x),
                    s.a.x.max(s.b.x),
                    o.a.y.min(o.b.y),
                    o.a.y.max(o.b.y),
                )
            } else {
                (
                    s.a.y.min(s.b.y),
                    s.a.y.max(s.b.y),
                    o.a.x.min(o.b.x),
                    o.a.x.max(o.b.x),
                )
            };
            let perp = if s.along_x() { s.a.y } else { s.a.x };
            if along >= lo && along <= hi && perp >= oa && perp <= ob {
                on.push((along, cross));
            }
        }
        for end in [s.a, s.b] {
            let along = if s.along_x() { end.x } else { end.y };
            if !on.iter().any(|(v, _)| *v == along) {
                on.push((along, end));
            }
        }
        on.sort_by(|p, q| p.0.total_cmp(&q.0));
        let mut prev: Option<NavNodeId> = None;
        for (_, p) in on {
            let id = add(&mut g, p, s.y);
            if let Some(a) = prev {
                g.link(a, id, NavKind::Street, Vec::new());
            }
            prev = Some(id);
        }
    }
    g
}

/// **The lanes of a level's own streets** — the whole derivation, end to end.
///
/// Two lanes on every street, one each way, at [`DEFAULT_LANE_WIDTH_M`] — not
/// at half the gap. A twenty-metre city street is not a twenty-metre
/// carriageway: it is seven metres of tarmac with six and a half of verge,
/// kerb and parking either side, which is what leaves [`KERB_PARK_OFFSET_M`]
/// somewhere to put a car.
pub fn carriageway(streets: &[Street]) -> LaneNetwork {
    let graph = carriageway_graph(streets);
    LaneNetwork::from_graph(&graph, |_, _| {
        Some(LaneSpec {
            lane_count: 2,
            width_m: DEFAULT_LANE_WIDTH_M,
            speed_limit_kmh: STREET_SPEED_KMH,
        })
    })
}

/// **A fold of the block set** — what says the derivation is stale.
///
/// FNV-1a over each site's `Guid` and its quantized rectangle, in `Guid` order.
/// It is a *membership* hash and the only legal operation on it is `==`, which
/// is `crate::band::SimBand`'s own rule about a stamp, restated.
pub fn block_stamp(world: &EcsWorld) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut fold = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    let q = |v: f64| {
        if v.is_finite() {
            (v / PAVEMENT_LATTICE_M).round() as i64 as u64
        } else {
            0
        }
    };
    for s in volume_sites(world) {
        let (hi, lo) = s.guid.as_u64_pair();
        fold(hi);
        fold(lo);
        fold(q(s.centre.x));
        fold(q(s.centre.z));
        fold(q(s.extent.x));
        fold(q(s.extent.y));
        fold(q(s.pad_y));
    }
    h
}

// ── the driver ──────────────────────────────────────────────────────────────

/// How far ahead a traffic car aims, in seconds of its own travel.
///
/// One second, the crowd's `PURSUIT_LOOKAHEAD_M` argument scaled by the fact
/// that a car cannot turn on the spot: aim at where the car will be next second
/// and the steering leads the corner instead of chasing it. Shorter and the
/// pursuit oscillates about the lane (the target is inside the car's own turn
/// radius); longer and it cuts corners, which on a street grid means driving
/// over the pavement on the inside of a turn.
pub const LOOKAHEAD_S: f64 = 1.0;

/// The shortest lookahead, metres — what a stopped car aims at.
///
/// Six metres is a car and a half. Below it the aim point is inside the
/// vehicle's own wheelbase and the steer command is decided by centimetres of
/// lateral error, which is a car that shimmies at a red light.
pub const LOOKAHEAD_MIN_M: f64 = 6.0;

/// The longest, metres. Thirty is a city block's worth of look and is where a
/// 30 km/h street's lookahead would land at four times the limit.
pub const LOOKAHEAD_MAX_M: f64 = 30.0;

/// The nearest the aim point may be measured as being *ahead*, metres.
///
/// The steer is `lateral / ahead`, which is a tangent, and an aim point beside
/// or behind the car would divide by zero or change sign. Clamping the
/// denominator turns both into full lock toward the lateral error, which is
/// what a driver does when the road is beside them.
pub const MIN_AHEAD_M: f64 = 1.0;

/// The tangent of the aim angle at which the wheel goes to full lock.
///
/// `0.5` is about 26.6°, so a car whose aim point is half as far to the side as
/// it is ahead is asking for everything the steering has. Beyond it the request
/// is clamped rather than growing, which matters because
/// `steer_limit_deg` narrows the actual lock with speed and a request that grew
/// without bound would just sit on the clamp.
pub const STEER_FULL_TAN: f64 = 0.5;

/// The speed error, m/s, over which the throttle goes from nothing to
/// everything.
///
/// Three metres a second is about 11 km/h. Narrower and traffic pumps the
/// throttle at cruise; wider and a car joining a 90 km/h highway crawls up to
/// it.
pub const SPEED_BAND_MPS: f64 = 3.0;

/// How hard traffic is willing to brake, m/s².
///
/// 3.5 is a firm but unremarkable stop — about a third of a g. It sizes the
/// following distance, and it is deliberately well under what the tyre model
/// can actually deliver: a traffic car that braked at the limit of adhesion
/// every time the car ahead slowed would be a queue of cars nose-diving.
pub const COMFORT_DECEL_MPS2: f64 = 3.5;

/// The gap a stopped car keeps to the one in front, metres.
///
/// Six metres from origin to origin is about a metre and a half of clear air
/// between a saloon's bumper and the next one's, which is what a queue looks
/// like. It is measured origin to origin because that is what the step can
/// measure without asking every car for its own length.
pub const STANDING_GAP_M: f64 = 6.0;

/// How hard traffic is willing to corner, m/s².
///
/// 2.5 is a quarter of a g — a comfortable town corner, and again well under
/// the tyre. It is what makes a car slow for a junction turn instead of
/// arriving at it at the speed limit and understeering into the far kerb.
pub const CORNER_LATERAL_MPS2: f64 = 2.5;

/// Below this speed a car asking for a stop holds the **handbrake** instead of
/// asking for reverse, m/s.
///
/// `VehicleControls::from_intent` reads a negative stick at rest as REVERSE —
/// its `rolling_forward` test is `> 0.5` — so a controller that kept asking for
/// `-1` at a red light would roll backwards through the queue. Matched to that
/// same 0.5 on purpose: the two numbers are one decision about when a car is
/// stopped, and they must not disagree.
pub const STOPPED_MPS: f64 = 0.5;

/// **What the driver can see** — everything [`drive_intent`] reads, gathered by
/// the caller that has a physics world.
#[derive(Clone, Copy, Debug)]
pub struct DriveView<'a> {
    /// The chassis origin, world metres.
    pub at: DVec3,
    /// Unit chassis forward (its local `+Z`), world.
    pub forward: DVec3,
    /// Speed along [`forward`](Self::forward), m/s. Negative is reversing.
    pub forward_mps: f64,
    /// The lane chain being followed.
    pub path: &'a NavPath,
    /// Arc length of the car's own projection onto it, metres.
    pub s_m: f64,
    /// The sign, m/s.
    pub speed_limit_mps: f64,
    /// Metres to the car in front along this path, or `None` for clear road.
    pub gap_m: Option<f64>,
    /// **How far right of its lane this driver should aim**, metres (wave EMS2).
    ///
    /// `0.0` for every car that is not getting out of somebody's way, which is
    /// every car in every level committed before this wave — and the term below
    /// is *guarded* on exactly that, so a zero bias is bit-identical to the
    /// arithmetic that had no bias at all. (Not merely "adds nothing": `x + 0.0`
    /// turns a `-0.0` into a `+0.0`, and a sign of zero that reached
    /// `move_input` would move a trace byte.)
    ///
    /// The sign is [`right_of`]'s, so positive is the kerb on a right-hand-drive
    /// carriageway — which is the side this engine's `lanes_of_spine` lays its
    /// forward lanes on.
    pub lateral_bias_m: f64,
    /// **Whether the path closes on itself** — a [`Circuit`]'s loop.
    ///
    /// Two things turn on it and both are wrong without it: the aim point wraps
    /// past the end instead of clamping to it (or a car brakes at the seam every
    /// lap), and the end-of-road stop is skipped (or it brakes to a *halt*
    /// there, for ever, because the seam never goes away).
    pub loops: bool,
}

/// **What the driver asks for** — a stick, and the three numbers that decided
/// it.
///
/// The stick and nothing else, because
/// [`crate::vehicle::VehicleControls::from_intent`] is the one place a stick
/// becomes a throttle and a brake in this engine and an AI that produced
/// controls directly would be the second. That is not a stylistic preference:
/// `from_intent`'s "back into forward motion is a BRAKE, not reverse" rule is a
/// control decision, and a traffic car that did not obey it would stop
/// differently from the way the player's car stops.
///
/// The three diagnostics are published for the reason `VehicleOutcome`'s are:
/// an arm that can only see the world cannot tell a car holding its lane from a
/// car that happens to be pointing the right way.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DriveIntent {
    /// `x` is steer, `y` is throttle/brake — a `MovementIntent::move_input`.
    pub move_input: Vec2d,
    /// Held at a standstill; never otherwise (see [`STOPPED_MPS`]).
    pub handbrake: bool,
    /// The speed this step asked for, m/s — the lowest of the limit, the bend
    /// and the gap.
    pub target_mps: f64,
    /// Signed metres the car is to the **right** of its lane. The sign is
    /// [`right_of`]'s.
    pub lateral_m: f64,
    /// The lookahead used, metres.
    pub lookahead_m: f64,
    /// How far along its path the car is, metres — the projection the whole
    /// decision was taken at.
    pub s_m: f64,
    /// How much road is left, metres. `INFINITY` on a loop.
    pub remaining_m: f64,
}

/// **The speed a bend allows**, m/s — `sqrt(a_lat / curvature)`.
///
/// Curvature is measured as the heading change over the lookahead, and the
/// heading change is the magnitude of the **cross product** of two unit
/// directions rather than an angle: `|a x b|` is `sin` of the turn, which for
/// the angles a road bends through is the angle, and it costs no trigonometry
/// (the P14 law).
///
/// A straight has no curvature and no limit, which is `f64::INFINITY` — a value
/// the caller's `min` handles without a branch.
pub fn corner_speed_mps(path: &NavPath, s_m: f64, lookahead_m: f64, loops: bool) -> f64 {
    if !(lookahead_m.is_finite() && lookahead_m > 0.0) || !s_m.is_finite() {
        return f64::INFINITY;
    }
    let a = path.direction_at(s_m);
    // The lookahead wraps on a loop, exactly as the aim point does.
    // `direction_at` CLAMPS, so without this a circuit car is blind to the
    // curvature across its own seam — it steers round the corner and does not
    // slow for it, which is half of `DriveView::loops`' claim applied and half
    // forgotten.
    let ahead = if loops {
        let len = path.length_m();
        if len > 0.0 {
            (s_m + lookahead_m).rem_euclid(len)
        } else {
            0.0
        }
    } else {
        s_m + lookahead_m
    };
    let b = path.direction_at(ahead);
    // The ground-plane component of `a x b`, which is the Y one.
    let sin = (a.z * b.x - a.x * b.z).abs();
    if !(sin.is_finite() && sin > 1.0e-6) {
        return f64::INFINITY;
    }
    let curvature = sin / lookahead_m;
    (CORNER_LATERAL_MPS2 / curvature).sqrt()
}

/// **The stick a lane-following driver would hold** — the whole controller, as
/// a pure function.
///
/// Four decisions, in order:
///
/// 1. **How far to look**: one second of travel, clamped into
///    `[LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M]`.
/// 2. **How fast to go**: the lowest of the sign, what the bend ahead allows
///    and what the gap in front allows. The gap rule is the stopping-distance
///    one — `v = sqrt(2 a (gap - standing))` — so a car closing on a queue
///    slows continuously rather than arriving and slamming on.
/// 3. **The pedal**: the speed error over [`SPEED_BAND_MPS`], as a stick, so
///    `from_intent` decides whether that is a throttle or a brake. A car asking
///    for a stop below [`STOPPED_MPS`] asks for **nothing and the handbrake**
///    instead of a negative stick, which `from_intent` would read as reverse.
/// 4. **The wheel**: pure pursuit of the point `lookahead` further along the
///    lane, as `lateral / ahead` — a tangent, clamped at
///    [`STEER_FULL_TAN`] — and no trigonometry anywhere in it.
///
/// # What this does NOT do, in one sentence
///
/// **It never changes lane and never overtakes.** A car behind a slower one
/// slows to its speed and stays there for as long as it is there; a car behind a
/// stopped one stops. That is the wave's stated v1 bound, and it is expressed as
/// *the absence of a rule* rather than as a rule that refuses — there is no lane
/// choice in this function at all, because [`LaneNetwork::lane_route`] picks one
/// index for a whole journey.
///
/// [`LaneNetwork::lane_route`]: inf_nav::LaneNetwork::lane_route
pub fn drive_intent(view: &DriveView<'_>) -> DriveIntent {
    let v = if view.forward_mps.is_finite() {
        view.forward_mps
    } else {
        0.0
    };
    let lookahead = (v.abs() * LOOKAHEAD_S).clamp(LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M);

    // ── 2. the speed.
    let limit = if view.speed_limit_mps.is_finite() && view.speed_limit_mps > 0.0 {
        view.speed_limit_mps
    } else {
        0.0
    };
    let bend = corner_speed_mps(view.path, view.s_m, lookahead, view.loops);
    let ahead_of_us = match view.gap_m {
        Some(g) if g.is_finite() => {
            let clear = (g - STANDING_GAP_M).max(0.0);
            (2.0 * COMFORT_DECEL_MPS2 * clear).sqrt()
        }
        _ => f64::INFINITY,
    };
    // …and the end of the road. A car that has run out of lane stops at it
    // rather than driving off the end, on the same stopping-distance rule — and
    // a LOOP has no end, so it is not slowed at its own seam.
    let endstop = if view.loops {
        f64::INFINITY
    } else {
        let to_end = (view.path.length_m() - view.s_m).max(0.0);
        (2.0 * COMFORT_DECEL_MPS2 * to_end).sqrt()
    };
    let target = limit.min(bend).min(ahead_of_us).min(endstop);
    // ── EMS2 the yield, second half. A car being asked to move over must keep
    //    rolling, or the handbrake below pins it exactly where it is in the way.
    //    Guarded on the same non-zero bias the wheel term is, so an ordinary
    //    street produces the bits it always produced. See
    //    `crate::dispatch::YIELD_CREEP_MPS` for the deadlock this closes.
    let target = if view.lateral_bias_m != 0.0 && view.lateral_bias_m.is_finite() {
        target.max(crate::dispatch::YIELD_CREEP_MPS)
    } else {
        target
    };

    // ── 3. the pedal.
    let (fwd, handbrake) = if target <= STOPPED_MPS && v.abs() <= STOPPED_MPS {
        (0.0, true)
    } else {
        (((target - v) / SPEED_BAND_MPS).clamp(-1.0, 1.0), false)
    };

    // ── 4. the wheel.
    let ahead_s = if view.loops {
        let len = view.path.length_m();
        if len > 0.0 {
            (view.s_m + lookahead).rem_euclid(len)
        } else {
            0.0
        }
    } else {
        view.s_m + lookahead
    };
    let aim = view.path.position_at(ahead_s);
    // ── EMS2 the yield. One term, guarded, on an aim point the pure pursuit was
    //    going to chase anyway: a car told to get out of the way aims a lane's
    //    half-width to its right and the existing wheel rule takes it there,
    //    slowing for the corner and stopping for the queue exactly as before.
    //
    //    Guarded rather than unconditional so a level with no siren in it
    //    produces the same bits it always did — see `DriveView::lateral_bias_m`.
    let aim = if view.lateral_bias_m != 0.0 && view.lateral_bias_m.is_finite() {
        let f0 = {
            let len = (view.forward.x * view.forward.x + view.forward.z * view.forward.z).sqrt();
            if len > 0.0 {
                DVec3::new(view.forward.x / len, 0.0, view.forward.z / len)
            } else {
                DVec3::Z
            }
        };
        aim + right_of(f0) * view.lateral_bias_m
    } else {
        aim
    };
    let to = aim - view.at;
    let f = {
        let len = (view.forward.x * view.forward.x + view.forward.z * view.forward.z).sqrt();
        if len > 0.0 {
            DVec3::new(view.forward.x / len, 0.0, view.forward.z / len)
        } else {
            DVec3::Z
        }
    };
    let r = right_of(f);
    let ahead = (to.x * f.x + to.z * f.z).max(MIN_AHEAD_M);
    let lateral = to.x * r.x + to.z * r.z;
    let tan = lateral / ahead;
    let steer = if tan.is_finite() {
        (tan / STEER_FULL_TAN).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    DriveIntent {
        move_input: Vec2d::new(steer, fwd),
        handbrake,
        target_mps: target,
        lateral_m: {
            let off = view.path.position_at(view.s_m) - view.at;
            -(off.x * r.x + off.z * r.z)
        },
        lookahead_m: lookahead,
        s_m: view.s_m,
        remaining_m: if view.loops {
            f64::INFINITY
        } else {
            (view.path.length_m() - view.s_m).max(0.0)
        },
    }
}

// ── the resource ────────────────────────────────────────────────────────────

/// **The level's own carriageway**, derived once and rebuilt when the blocks
/// change.
///
/// A resource, so no schema moves (see the module docs). Absent until something
/// asks, so a level with no blocks pays exactly one `contains_resource` per
/// fixed step and allocates nothing — the "absent costs nothing" discipline
/// [`crate::society`] and [`crate::crowd`] already follow.
#[derive(bevy_ecs::prelude::Resource, Debug, Clone, Default, PartialEq)]
pub struct TrafficRes {
    /// The recovered street lines, in the order [`streets_of`] answers them.
    pub streets: Vec<Street>,
    /// The lanes over them.
    pub lanes: LaneNetwork,
    /// The block-set fold this was derived from. `0` before the first
    /// derivation, which is a value no non-empty level produces.
    pub stamp: u64,
    /// How many times the derivation has actually run — a counter a gate can
    /// assert is **one** over a settled level, which is what says the cache is
    /// a cache.
    pub derivations: u64,
}

/// **Derive the level's streets if its blocks have moved** — the one door both
/// hosts call.
///
/// Returns `true` when it rebuilt. Cheap when nothing changed: one
/// [`block_stamp`] walk, which is the same entity walk
/// [`crate::society::sync_society`] already makes.
pub fn sync_carriageway(world: &mut EcsWorld) -> bool {
    let stamp = block_stamp(world);
    if let Some(res) = world.world().get_resource::<TrafficRes>() {
        if res.stamp == stamp {
            return false;
        }
    }
    let streets = streets_of(world);
    let lanes = carriageway(&streets);
    let derivations = world
        .world()
        .get_resource::<TrafficRes>()
        .map(|r| r.derivations)
        .unwrap_or(0);
    world.world_mut().insert_resource(TrafficRes {
        streets,
        lanes,
        stamp,
        derivations: derivations + 1,
    });
    true
}

/// The level's carriageway, or an empty one — the read side, for a caller that
/// must not derive.
pub fn carriageway_of(world: &EcsWorld) -> Option<&TrafficRes> {
    world.world().get_resource::<TrafficRes>()
}

/// Forget the derivation, so a world is byte-for-byte one that never had it.
///
/// The editor calls this at both ends of a Simulate session for
/// [`crate::crowd::clear_crowd`]'s reason: a `SceneDoc` snapshot carries
/// entities and `EcsWorld::clear` despawns them, and neither touches a resource.
pub fn clear_carriageway(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<TrafficRes>();
}

/// **Every kerb parking slot the level's streets offer**, in a deterministic
/// order — where a parked car goes.
///
/// One slot every [`KERB_SLOT_M`] along each side of every street wide enough
/// to park on, offset [`KERB_PARK_OFFSET_M`] from the centreline, facing the
/// way traffic runs on that side (so a kerbside row all points one way, which is
/// what `frames/steal-car/0016` shows and what a row of randomly-yawed cars
/// does not).
///
/// The count is geometry and not a setting: a hundred-metre run holds **six** a
/// side, and the slots are on a **world lattice** rather than measured from the
/// line's own end, so a line that grows keeps the slots it had. What decides
/// whether a slot is *taken* is the caller's own draw — see
/// `inf_physics::d3::traffic`.
pub fn kerb_slots(streets: &[Street]) -> Vec<(DVec3, f64)> {
    let mut out = Vec::new();
    for (i, s) in streets.iter().enumerate() {
        if !kerb_fits(s.gap_m) {
            continue;
        }
        let d = DVec2::new(s.b.x - s.a.x, s.b.y - s.a.y);
        let len = (d.x * d.x + d.y * d.y).sqrt();
        if !(len.is_finite() && len > KERB_SLOT_M) {
            continue;
        }
        let dir = DVec3::new(d.x / len, 0.0, d.y / len);
        let r = right_of(dir);
        // **The slots sit on a GLOBAL lattice, not on this line's own end.**
        //
        // A street's `a` is the group's bounding-box corner, so one block
        // arriving anywhere in a settlement moves it — and with it every slot
        // on every line, and with those every `parked_car_guid`, which
        // quantizes at a centimetre. `derive_parked`'s whole carry-forward
        // ("the guid is a pure function of the space, so the cars that did not
        // move keep their records") was therefore dead in exactly the case it
        // was written for. Measured from the world origin instead: a line that
        // grows keeps the slots it had and gains new ones at the end.
        //
        // A full slot of clearance at each end, so a row does not run into the
        // junction it ends at. (The comment here used to say "half", and the
        // arithmetic has always said whole.)
        let origin = if s.along_x() { s.a.x } else { s.a.y };
        let first = (origin / KERB_SLOT_M).floor() + 1.0;
        let n = ((len - KERB_SLOT_M) / KERB_SLOT_M).max(0.0) as usize;
        for side in [1.0f64, -1.0] {
            // The near side faces the way this line runs; the far side faces
            // back, because that is the direction traffic runs on it.
            let heading = if side > 0.0 { dir } else { -dir };
            let yaw = yaw_of_dir(heading);
            for k in 0..=n {
                let lattice = (first + k as f64) * KERB_SLOT_M;
                let along = lattice - origin;
                if !(along >= KERB_SLOT_M * 0.5 && along <= len - KERB_SLOT_M * 0.5) {
                    continue;
                }
                // **Still the constant, and `kerb_park_offset_m` says why**
                // (wave ROAD1b): the offset the drawn kerb implies is the right
                // number and moving to it deadlocks the EMS return, measured.
                let p =
                    DVec3::new(s.a.x, s.y, s.a.y) + dir * along + r * (side * KERB_PARK_OFFSET_M);
                if !clear_of_junctions(streets, i, DVec2::new(p.x, p.z)) {
                    continue;
                }
                out.push((p, yaw));
            }
        }
    }
    out
}

/// How much room a parked car leaves beside a **crossing** street, on top of
/// that street's own half-width, metres.
///
/// Without it a slot laid down the length of one line lands inside the
/// carriageway of every line that crosses it: the first draft parked a car two
/// metres from the middle of a four-way junction. Two metres past the far kerb
/// of the crossing street is a car that has left the junction clear, which is
/// what a kerb looks like and what a car pulling out of one needs.
pub const JUNCTION_CLEAR_M: f64 = 2.0;

/// Whether a kerb slot is far enough from every street but its own.
///
/// Measured to the crossing street's **segment**, clamped at its ends, so a
/// line that stops short of this one does not reserve room it does not occupy.
fn clear_of_junctions(streets: &[Street], own: usize, p: DVec2) -> bool {
    for (j, o) in streets.iter().enumerate() {
        if j == own {
            continue;
        }
        let d = DVec2::new(o.b.x - o.a.x, o.b.y - o.a.y);
        let len2 = d.x * d.x + d.y * d.y;
        let q = if len2 > 0.0 {
            let t = (((p.x - o.a.x) * d.x + (p.y - o.a.y) * d.y) / len2).clamp(0.0, 1.0);
            DVec2::new(o.a.x + d.x * t, o.a.y + d.y * t)
        } else {
            o.a
        };
        let (dx, dz) = (p.x - q.x, p.y - q.y);
        if (dx * dx + dz * dz).sqrt() < o.gap_m * 0.5 + JUNCTION_CLEAR_M {
            return false;
        }
    }
    true
}

/// The compass yaw of a ground heading, degrees — `0` faces `+Z`.
///
/// `inf_math::patan2_64` and never `f64::atan2`: this number becomes a
/// `Transform::rotation` and therefore the replay trace, which is the P14 law's
/// first class exactly.
pub fn yaw_of_dir(d: DVec3) -> f64 {
    inf_math::patan2_64(d.x, d.z).to_degrees()
}

/// **Every guid this module mints for a parked car**, so two of them cannot
/// collide and a re-derivation names the same car.
///
/// A content-derived guid on the P22.3 precedent: the slot's own quantized
/// position is the identity, so a car parked at a kerb keeps its guid across a
/// re-derivation, across a save and across two hosts.
pub fn parked_car_guid(p: DVec3) -> Uuid {
    let q = |v: f64| {
        if v.is_finite() {
            (v / PAVEMENT_LATTICE_M).round() as i64 as u64
        } else {
            0
        }
    };
    // **The PLAN position only.** The height a slot is derived at is a median
    // over the blocks that bound its street, and that median moves when any of
    // them does — so folding Y in would re-mint every guid in a settlement the
    // first time one block paged in, which is the carry-forward defect this
    // wave's adversarial read found. Two cars at one `(x, z)` and two heights
    // is an overpass, and this engine's settlements do not build one.
    let hi = mix64(q(p.x) ^ mix64(q(p.z)));
    let lo = mix64(q(p.x).rotate_left(17) ^ mix64(PARKED_SALT));
    Uuid::from_u64_pair(hi, lo)
}

/// Salts [`parked_car_guid`], so a parked car and a pavement node at the same
/// place are different numbers.
pub const PARKED_SALT: u64 = 0x5041_524b_4544_0001;

/// The guids this module has minted in a world, so a caller can tell a derived
/// car from an authored one.
pub fn derived_guids(streets: &[Street]) -> BTreeSet<Uuid> {
    kerb_slots(streets)
        .into_iter()
        .map(|(p, _)| parked_car_guid(p))
        .collect()
}

// ── the population ──────────────────────────────────────────────────────────

/// Metres inside which a traffic car is [`Full`](CrowdTier::Full) — a real rig,
/// with physics, a driver if it is moving, and a seat the hero can reach.
///
/// **Twice the crowd's 32 m**, chosen against the same thing the crowd chose
/// against: [`crate::band::DEFAULT_COLLIDER_NEAR_M`] is 64 m, the radius inside
/// which a building is solid, and a car you can drive into has to be inside a
/// world you can drive into. It is also 21 times `ENTER_REACH_M`, which is what
/// makes the promotion rule of clause 4 structural rather than a second check:
/// **any car the hero can reach the seat of is Full**, because 3 m is deep
/// inside 64.
pub const TRAFFIC_FULL_M: f64 = 64.0;

/// Metres inside which a traffic car exists at all — [`Near`](CrowdTier::Near),
/// drawn and solid but moved by its own clock.
///
/// # A CAR HAS THREE RUNGS, NOT FOUR, and this is the pricing of clause 4
///
/// The crowd's ladder has a fourth rung because a crowd agent can be made
/// cheap: `Far` drops the body and the pose and keeps one entity, and NPC1b's
/// impostors draw a thousand of them out of one instanced batch. **A car cannot
/// go through that path.** A vehicle body is a union of built-in primitives
/// derived from a Ring-0 table with no `.inf_mesh` at all — which is wave
/// VEH1a's whole argument for why the fleet costs no committed content — and
/// `PcgKind::mesh` is a mesh GUID. Scattering parked cars would mean committing
/// a car mesh, which is exactly the content that argument avoided.
///
/// So a car is entities all the way down: fourteen of them at
/// [`RigDetail::Full`] and five or six at [`RigDetail::Body`]. The ladder
/// therefore stops one rung earlier and the radius is where a car stops being
/// readable rather than where it stops being cheap. `Far` is unreachable by
/// construction — [`TRAFFIC_RADII`] sets `near == far`, and `CrowdBand::tier`'s
/// `d <= far_m` branch cannot fire after `d <= near_m` has.
pub const TRAFFIC_NEAR_M: f64 = 128.0;

/// The three radii a car is banded by. `near == far` — see [`TRAFFIC_NEAR_M`].
pub const TRAFFIC_RADII: (f64, f64, f64) = (TRAFFIC_FULL_M, TRAFFIC_NEAR_M, TRAFFIC_NEAR_M);

/// The share of kerb slots that actually hold a car.
///
/// Not one: `frames/steal-car/0016` and `frames/driving/0014` both show kerbs
/// with gaps in them, and a row with no gaps in it reads as a wall rather than
/// as parking. Drawn per slot from [`SALT_PARK`], so it is a function of the
/// level and not of a spawn order, and it is what bounds the population against
/// the geometry: a hundred-metre run offers six slots a side and holds about
/// three cars.
pub const KERB_OCCUPANCY: f64 = 0.45;

/// **What kind of day a car has**, as one draw split into four bands.
///
/// ONE draw, from [`SALT_DAY`], because the alternative — a draw for "does it
/// commute" and a second for "is it a circuit instead" — makes the second share
/// a share *of the first*, and the first cut of this wave did exactly that: a
/// three-per-cent night shift of a thirty-five-per-cent commuting population is
/// **one per cent of the street**, which on a forty-nine-car town is zero cars
/// and an empty three in the morning. The arm caught it
/// (`the_street_is_busy_at_eight_and_sparse_at_three_and_never_empty`), and the
/// fix is that the bands are all shares of the same thing.
///
/// | draw | day |
/// |---|---|
/// | `[0, 0.06)` | a night circuit |
/// | `[0.06, 0.20)` | a day circuit |
/// | `[0.20, 0.50)` | a commute |
/// | `[0.50, 1)` | parked, for ever |
///
/// Half the kerb never moves, which is what a kerb is.
pub const COMMUTER_SHARE: f64 = 0.30;

/// Salts the parked/empty draw.
pub const SALT_PARK: u64 = 0x5041_524b_0000_0011;
/// Salts the one draw that decides a car's whole day — see [`COMMUTER_SHARE`].
pub const SALT_DAY: u64 = 0x434f_4d4d_0000_0012;
/// Salts which catalogue row a slot's car is.
pub const SALT_CLASS: u64 = 0x434c_4153_0000_0013;
/// Salts a car's paint.
pub const SALT_PAINT: u64 = 0x5041_494e_0000_0014;
/// Salts a commuter's destination.
pub const SALT_DEST: u64 = 0x4445_5354_0000_0015;

/// The most cars one level's traffic may hold.
///
/// [`MAX_VEHICLE_DEFS`]'s rule at a city's scale: a population derived from
/// authored geometry needs a bound, so a recipe with a thousand blocks in it is
/// an error message rather than a memory problem. Four thousand is an order
/// over what the island's seven settlements offer.
///
/// [`MAX_VEHICLE_DEFS`]: crate::vehicle::MAX_VEHICLE_DEFS
pub const MAX_TRAFFIC_CARS: usize = 4096;

/// The most cars that may have a day.
///
/// A commuter costs a Dijkstra and a lane route to derive, where a parked car
/// costs a hash. This is the bound on the expensive half, and it is a hundred
/// and twenty-eight because that is far more cars than
/// [`TRAFFIC_NEAR_M`] can hold at once and therefore more than a viewer can be
/// looking at.
///
/// Counted over the **cars that have one**, not over one derivation's queue.
/// The first cut checked the freshly built queue alone, so every re-derivation
/// opened another hundred and twenty-eight slots on top of the days already
/// planned and the bound did not bound what its own name says.
pub const MAX_COMMUTERS: usize = 128;

/// How many commuter routes are planned per fixed step.
///
/// [`crate::society::SOCIETY_PLANS_PER_STEP`]'s shape and its reason: a
/// derivation that ran a hundred and twenty-eight Dijkstras in one step would
/// be a hitch on the step a settlement pages in. Four a step spreads it over
/// half a second and every one of them is a pure function of the level, so a
/// host that took longer to get there arrives at the same population.
pub const TRAFFIC_PLANS_PER_STEP: usize = 4;

/// How far before a parking slot the approach point is laid, metres.
///
/// A leg's path ends `[.., slot - kerb_heading x this, slot]`, so the last
/// segment of a drive runs *along the kerb* and the car ends up pointing the
/// way the row points. Without it the final heading is the diagonal from the
/// lane to the space, and a car finishes its commute parked at forty degrees to
/// the pavement.
pub const APPROACH_M: f64 = 6.0;

/// **How much of a car is built** — the tier's other axis.
///
/// A crowd agent's tiers differ in what *runs*; a car's differ in what
/// *exists*, because there is no impostor path for one (see
/// [`TRAFFIC_NEAR_M`]). `Body` is the chassis and its panels: no wheels, no
/// tyres, no `VehicleClass`, no sensor colliders — so `rig_of` finds no wheels,
/// answers `None`, and **`step_vehicles` cannot see the car at all**. That is
/// the watched-car honesty sentence made structural rather than promised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RigDetail {
    /// Nothing at all.
    #[default]
    None,
    /// The chassis and its body panels, drawn and solid, moved by a clock.
    Body,
    /// Everything: wheels, tyres, the class, and a rig `step_vehicles` drives.
    Full,
}

impl RigDetail {
    /// What a tier builds.
    pub fn of(tier: CrowdTier) -> Self {
        match tier {
            CrowdTier::Full => RigDetail::Full,
            CrowdTier::Near | CrowdTier::Far => RigDetail::Body,
            CrowdTier::Dormant => RigDetail::None,
        }
    }

    /// The byte the trace folds. Frozen: compared between two hosts.
    pub fn as_u8(self) -> u8 {
        match self {
            RigDetail::None => 0,
            RigDetail::Body => 1,
            RigDetail::Full => 2,
        }
    }
}

/// **A loop, and the hours it runs** — the other kind of day a car has.
///
/// A commute is a `CrowdSchedule`: two legs, and the position is a fraction of
/// each leg's *window*. That is exactly right for a person, and it has a
/// property a car cannot live with — the implied speed is `length / window`, so
/// a four-hundred-metre commute over an hour of a level clock running at
/// eighteen is **two metres a second**. A pedestrian's pace, in a car.
///
/// A circuit is a [`crate::crowd::CrowdRoute`] in
/// [`RouteMode::Loop`](crate::crowd::RouteMode::Loop), whose position is
/// `speed x t + phase` folded by the loop's length: it has a **stated speed**
/// and no window at all, so a car on one does town speed whatever rate the
/// level's day runs at. It is what a delivery van, a taxi and somebody running
/// errands all are, and it is the only thing that puts cars on a street at three
/// in the morning.
///
/// Outside its hours the car does not exist — see
/// [`TrafficRecord::alive`]. Not "parked at home": a day-shift van that
/// teleported back to its space at ten at night would be a teleport somebody
/// watched.
#[derive(Debug, Clone, PartialEq)]
pub struct Circuit {
    /// The loop, at a real speed.
    pub route: crate::crowd::CrowdRoute,
    /// The local hour it starts, `[0, 24)`.
    pub from_h: f64,
    /// The local hour it stops. May be **before** `from_h`, which is a night
    /// shift — a day is a circle and the modulo is the circle
    /// ([`crate::crowd::CrowdSchedule::at`]'s own sentence).
    pub to_h: f64,
}

impl Circuit {
    /// Whether this circuit is running at `hour`.
    pub fn running(&self, hour: f64) -> bool {
        if !hour.is_finite() {
            return false;
        }
        let span = (self.to_h - self.from_h).rem_euclid(24.0);
        let since = (hour - self.from_h).rem_euclid(24.0);
        span > 0.0 && since < span
    }
}

/// The hours a day-shift circuit runs.
pub const DAY_CIRCUIT_H: (f64, f64) = (6.0, 22.0);
/// The hours a night-shift one does — `frames/steal-car/0028`'s reference: a
/// street at night is not empty, it is *sparse*.
pub const NIGHT_CIRCUIT_H: (f64, f64) = (22.0, 6.0);
/// The share of parked cars that are out on a day circuit.
pub const DAY_CIRCUIT_SHARE: f64 = 0.14;
/// The share that are out at night — a third as many, which is what makes the
/// small hours read as quiet rather than as broken.
pub const NIGHT_CIRCUIT_SHARE: f64 = 0.06;
/// What a circuit car actually holds, as a share of the sign.
///
/// Eight tenths, because traffic does not sit on the limit and a clock-tier car
/// that did would arrive at every junction faster than the steered one beside
/// it.
pub const CIRCUIT_SPEED_FRAC: f64 = 0.8;

/// **One traffic car**: a body, a kerb space, and possibly a day.
///
/// The schedule is [`crate::crowd::CrowdSchedule`] — the *same* type a resident
/// carries, resolved by the *same* [`crate::crowd::CrowdClock`] against the
/// *same* hours ([`crate::society::WORK_START_H`],
/// [`crate::society::HOME_H`], [`crate::society::COMMUTE_H`]). That is what
/// "commuter cars join the society schedule" means here, precisely: one
/// vocabulary, one clock, one set of hours. What it does **not** mean is that a
/// named resident owns a named car — see the wave's carried list.
///
/// A car with **no** schedule never moves. A car with one is parked at its own
/// kerb slot overnight, drives to a second slot over the morning window, stands
/// there all day and drives back in the evening — which is a real street's rush
/// hour, expressed as `CrowdSchedule::at`'s existing "the walk is a fraction of
/// its WINDOW, then you stand at the far end".
#[derive(Debug, Clone, PartialEq)]
pub struct TrafficRecord {
    /// Which catalogue row it is.
    pub def: crate::vehicle::VehicleDef,
    /// Its paint.
    pub paint: crate::math::Color,
    /// Its own kerb space, world metres (the chassis origin, already lifted to
    /// its resting height).
    pub home: DVec3,
    /// The way the row it is parked in points, degrees.
    pub home_yaw_deg: f64,
    /// Its commute, or `None`. A record has at most one of this and
    /// [`circuit`](Self::circuit), by construction in `plan_batch`.
    pub schedule: Option<crate::crowd::CrowdSchedule>,
    /// Its loop, or `None` — see [`Circuit`].
    pub circuit: Option<Circuit>,
    /// The tier it took on the last step.
    pub tier: CrowdTier,
    /// How much of it is built right now.
    pub detail: RigDetail,
    /// Where it was then.
    pub last: DVec3,
    /// Which way it was pointing then, degrees.
    pub yaw_deg: f64,
    /// Which leg of its day it was on. `0` for a car with no schedule.
    pub leg: u8,
    /// **What the steered tier did to the clock**, metres of phase — the
    /// [`crate::crowd::CrowdRecord::rephase_m`] of a car, with that field's
    /// whole argument: a body that has been driven by a controller is not where
    /// its clock says, and snapping it back would teleport it.
    pub rephase_m: f64,
    /// **The ground under its own space**, world Y, once something has measured
    /// it — see `inf_physics::d3::traffic::settle_on_the_ground`.
    ///
    /// `None` until the first time the car has a body, because measuring it
    /// needs a collision world and the derivation has none: the street's own
    /// `y` is the mean pad of the blocks that bound it, which is right on a
    /// levelled town and metres out on one that climbs. A car placed metres
    /// under a heightfield is a car rapier launches, and the island fixture
    /// launched one at **72 m/s** before this field existed.
    ///
    /// # It IS a streaming-dependent measurement, latched
    ///
    /// The honest statement, which an earlier draft of this doc got backwards.
    /// The ray reads the live collision world, so *which step it first ran on*
    /// decides the answer — and it is kept rather than re-measured precisely so
    /// that the answer stops moving afterwards. `NavPath::snapped`'s doctrine is
    /// satisfied in the only way it can be here: the dependency is paid **once**
    /// and never again, so a car does not change height because a tile paged
    /// out. Both hosts pay it on the same step because both build the car on the
    /// same step — which is a property `traffic_state_bytes` folds `last` for,
    /// and the rush-hour gate compares.
    ///
    /// # One height for a whole journey
    ///
    /// [`place`](Self::place) applies this single Y to every point of the car's
    /// path, so a `Near` car crossing a graded town holds the height of the
    /// space it started in. Right on a levelled settlement pad, which is what
    /// this engine carves; wrong on a route that climbs, and carried.
    pub ground_y: Option<f64>,
    /// **Somebody has interfered with this car** — its driver was pulled out of
    /// it, or a character who is not its driver is sitting in it.
    ///
    /// One flag for both, because they are one rule: *a car the player has
    /// touched is no longer traffic's*. Once it is set, the traffic step never
    /// steers this car again, never gives it another driver, and never takes
    /// its body down — so a stolen car keeps its rig wherever it is left, which
    /// makes it an ordinary vehicle exactly like the seven the island authors.
    ///
    /// It is never unset. A car that reverted to traffic control the moment the
    /// player got out would drive itself away from under them, and a car that
    /// grew a second driver while the first was still lying in the road would
    /// be two people in one seat.
    ///
    /// Genuine sim state, and folded into [`traffic_state_bytes`] for
    /// [`crate::crowd::CrowdRecord::rephase_m`]'s reason: it is produced by the
    /// simulation, and two hosts that disagreed about it have diverged.
    pub taken: bool,
}

impl TrafficRecord {
    /// A car that never moves, parked at `home`.
    pub fn parked(
        def: crate::vehicle::VehicleDef,
        paint: crate::math::Color,
        home: DVec3,
        yaw: f64,
    ) -> Self {
        Self {
            def,
            paint,
            home,
            home_yaw_deg: yaw,
            schedule: None,
            circuit: None,
            tier: CrowdTier::Dormant,
            detail: RigDetail::None,
            last: home,
            yaw_deg: yaw,
            leg: 0,
            rephase_m: 0.0,
            ground_y: None,
            taken: false,
        }
    }

    /// Whether this car goes anywhere at all.
    pub fn commutes(&self) -> bool {
        self.schedule.is_some() || self.circuit.is_some()
    }

    /// **Whether this car is on the road at all at this hour.**
    ///
    /// A parked car and a commuter are always there — a commuter that is not
    /// driving is standing in a space somebody could steal it out of. A circuit
    /// car outside its hours is **not there**, which is the honest answer for a
    /// van that is not working: parking it back at its space would be a teleport
    /// (see [`Circuit`]).
    pub fn alive(&self, hour: f64) -> bool {
        match &self.circuit {
            Some(c) => c.running(hour),
            None => true,
        }
    }

    /// The metres of head start this car has round its own loop.
    ///
    /// [`crate::crowd::CrowdRecord::phase_of`]'s draw, scaled to the loop rather
    /// than to eight metres: a dozen vans that all started at the same point
    /// would be a convoy.
    pub fn phase_of(&self, guid: Uuid) -> f64 {
        let Some(c) = self.circuit.as_ref() else {
            return 0.0;
        };
        crate::crowd::agent_unit(guid, 0, crate::crowd::SALT_PHASE) * c.route.length_m()
            + self.rephase_m
    }

    /// **The path this car is driving right now**, whichever kind of day it has.
    pub fn active_path(
        &self,
        clock: crate::crowd::CrowdClock,
        leg: crate::crowd::ActiveLeg,
    ) -> Option<&NavPath> {
        match &self.circuit {
            Some(c) => c.running(clock.hour).then_some(&c.route.path),
            None => self.path_on(leg),
        }
    }

    /// **Where the clock says it is and which way it points** — one door over
    /// both kinds of day, so the tier that draws a car and the tier that steers
    /// one cannot disagree about either.
    pub fn place(
        &self,
        guid: Uuid,
        clock: crate::crowd::CrowdClock,
        leg: crate::crowd::ActiveLeg,
    ) -> (DVec3, f64) {
        let (p, yaw) = match self.circuit.as_ref() {
            None => self.place_on(leg),
            Some(c) if !c.running(clock.hour) => (self.home, self.home_yaw_deg),
            Some(c) => {
                let s = c.route.progress_at(clock.t_s, self.phase_of(guid)).s_m;
                (
                    c.route.path.position_at(s),
                    yaw_of_dir(c.route.path.direction_at(s)),
                )
            }
        };
        // **The measured ground wins over the derived one.** The street's `y`
        // is a mean of the blocks that bound it; `ground_y` is a ray. Applied
        // here, in the one place a car's place is decided, so the tier that
        // draws a car and the tier that steers one are lifted by the same
        // number.
        match self.ground_y {
            Some(g) => (
                DVec3::new(p.x, crate::vehicle::resting_origin_y(&self.def, g), p.z),
                yaw,
            ),
            None => (p, yaw),
        }
    }

    /// The phase change that puts the clock on the metre the body reached, over
    /// both kinds of day.
    pub fn rephase_delta(
        &self,
        guid: Uuid,
        clock: crate::crowd::CrowdClock,
        leg: crate::crowd::ActiveLeg,
        s_m: f64,
    ) -> f64 {
        let Some(c) = self.circuit.as_ref() else {
            return self.rephase_delta_on(leg, s_m);
        };
        let travelled = c.route.travelled_at(clock.t_s, self.phase_of(guid));
        c.route.rephase_delta(travelled, s_m)
    }

    /// Which leg it is on and how far through, or `None` for a car with no day.
    ///
    /// The jitter is [`crate::crowd::CrowdRecord::hour_of`]'s, drawn on the
    /// car's own guid: a town whose eighty commuters all pull out at exactly
    /// eight o'clock is a tide rather than a street.
    pub fn leg_at(&self, guid: Uuid, clock: crate::crowd::CrowdClock) -> crate::crowd::ActiveLeg {
        let s = self.schedule.as_ref()?;
        let jitter = crate::crowd::SCHEDULE_JITTER_H
            * (2.0 * crate::crowd::agent_unit(guid, 0, crate::crowd::SALT_SCHEDULE) - 1.0);
        Some(s.at((clock.hour - jitter).rem_euclid(24.0)))
    }

    /// The path it is driving right now, or `None` when it is standing.
    pub fn path_on(&self, leg: crate::crowd::ActiveLeg) -> Option<&NavPath> {
        match (&self.schedule, leg) {
            (Some(s), Some((i, _))) => Some(&s.legs()[i].path),
            _ => None,
        }
    }

    /// **Where the clock says it is, and which way it is pointing** — the one
    /// place a traffic car's place is decided, so the tier that draws it and the
    /// tier that steers it cannot disagree.
    ///
    /// A car with no leg stands in its own space. A car on a leg is
    /// `length x u + rephase`, clamped, with its heading the lane's own
    /// direction there — so a car standing at the end of a leg is parked facing
    /// the way the row faces (see [`APPROACH_M`]).
    pub fn place_on(&self, leg: crate::crowd::ActiveLeg) -> (DVec3, f64) {
        let (Some(path), Some((i, u))) = (self.path_on(leg), leg) else {
            return (self.home, self.home_yaw_deg);
        };
        let len = self
            .schedule
            .as_ref()
            .map(|s| s.legs()[i].path.length_m())
            .unwrap_or(0.0);
        let raw = len * u + self.rephase_m;
        let s_m = if raw.is_finite() {
            raw.clamp(0.0, len)
        } else {
            0.0
        };
        (path.position_at(s_m), yaw_of_dir(path.direction_at(s_m)))
    }

    /// How far along its current leg the clock says it is, metres.
    pub fn progress_on(&self, leg: crate::crowd::ActiveLeg) -> f64 {
        let (Some(_), Some((i, u))) = (self.path_on(leg), leg) else {
            return 0.0;
        };
        let len = self
            .schedule
            .as_ref()
            .map(|s| s.legs()[i].path.length_m())
            .unwrap_or(0.0);
        let raw = len * u + self.rephase_m;
        if raw.is_finite() {
            raw.clamp(0.0, len)
        } else {
            0.0
        }
    }

    /// **The phase change that puts the clock on the metre the BODY reached** —
    /// [`crate::crowd::CrowdRecord::rephase_delta_on`]'s schedule branch, for a
    /// car.
    pub fn rephase_delta_on(&self, leg: crate::crowd::ActiveLeg, s_m: f64) -> f64 {
        let (Some(_), Some((i, u))) = (self.path_on(leg), leg) else {
            return 0.0;
        };
        let len = self
            .schedule
            .as_ref()
            .map(|s| s.legs()[i].path.length_m())
            .unwrap_or(0.0);
        if !(len.is_finite() && s_m.is_finite() && self.rephase_m.is_finite()) {
            return 0.0;
        }
        (s_m.clamp(0.0, len) - len * u) - self.rephase_m
    }

    /// Whether the car is **driving** right now — on a leg, and not yet at the
    /// far end of it.
    ///
    /// This is what decides whether it carries a driver, which is what decides
    /// whether it can be carjacked rather than merely stolen.
    pub fn is_driving(
        &self,
        clock: crate::crowd::CrowdClock,
        leg: crate::crowd::ActiveLeg,
    ) -> bool {
        match &self.circuit {
            Some(c) => c.running(clock.hour),
            None => matches!(leg, Some((_, u)) if u < 1.0) && self.schedule.is_some(),
        }
    }
}

/// **A traffic car's driver**, as a guid derived from the car's.
///
/// Derived rather than stored for `CrowdRecord::speed_of`'s reason: a guid
/// written into the record at spawn time is a second copy of a pure function.
pub fn driver_guid(chassis: Uuid) -> Uuid {
    let (hi, lo) = chassis.as_u64_pair();
    Uuid::from_u64_pair(mix64(hi ^ DRIVER_SALT), mix64(lo ^ mix64(DRIVER_SALT)))
}

/// Salts [`driver_guid`].
pub const DRIVER_SALT: u64 = 0x4452_4956_4552_0001;

/// **The traffic population** — every car a level has, whether or not it
/// currently has a body.
#[derive(bevy_ecs::prelude::Resource, Debug, Clone, Default, PartialEq)]
pub struct TrafficPopulationRes {
    /// The records, in `Guid` order.
    pub records: BTreeMap<Uuid, TrafficRecord>,
    /// Kerb slots whose commuter route has not been planned yet, in `Guid`
    /// order — the batch queue [`TRAFFIC_PLANS_PER_STEP`] drains.
    pub pending: BTreeMap<Uuid, DVec3>,
    /// Fixed steps since the population was installed.
    pub steps: u64,
    /// The carriageway stamp this population was derived from.
    pub stamp: u64,
    /// Somebody installed this population by hand, so nothing derives one.
    /// [`CrowdPopulationRes::hand_installed`]'s rule, verbatim.
    ///
    /// [`CrowdPopulationRes::hand_installed`]: crate::crowd::CrowdPopulationRes::hand_installed
    pub hand_installed: bool,
}

/// What one traffic step did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrafficStats {
    /// How many records there are.
    pub cars: usize,
    /// How many of them have a day.
    pub commuters: usize,
    /// How many are on a leg and not yet at the end of it, this step.
    pub driving: usize,
    /// Cars per tier, indexed by [`CrowdTier::as_u8`].
    pub per_tier: [usize; 4],
    /// How many were built this step.
    pub built: usize,
    /// How many were taken down.
    pub removed: usize,
    /// How many changed tier.
    pub retiered: usize,
    /// How many handed their clock back to their body on the way down.
    pub rephased: usize,
    /// How many carry an NPC driver right now.
    pub drivers: usize,
    /// How many the traffic has let go of — see [`TrafficRecord::taken`].
    pub taken: usize,
    /// How many measured the ground under themselves this step.
    pub settled: usize,
    /// How many are waiting for something to say what they are standing on —
    /// a slot whose terrain has not paged in, or one over a hole.
    pub groundless: usize,
    /// How many commuter routes were planned this step.
    pub planned_now: usize,
    /// How many slots are still waiting for a route.
    pub pending: usize,
    /// The band's membership stamp.
    pub band_stamp: u64,
}

/// **Grow the level's traffic** — the derivation, one batch per step.
///
/// Parked cars are derived **whole** on the step the carriageway changes: a
/// kerb slot costs a hash and there is nothing to plan. Commuter routes are
/// planned [`TRAFFIC_PLANS_PER_STEP`] at a time, because each one is a Dijkstra
/// over the street graph and a hundred and twenty-eight of them in one step is
/// the hitch [`crate::society::SOCIETY_PLANS_PER_STEP`] exists to avoid.
///
/// Returns how many routes were planned this step.
pub fn sync_traffic(world: &mut EcsWorld) -> usize {
    let Some(stamp) = carriageway_of(world).map(|r| r.stamp) else {
        return 0;
    };
    if world
        .world()
        .get_resource::<TrafficPopulationRes>()
        .is_some_and(|p| p.hand_installed)
    {
        return 0;
    }
    let stale = world
        .world()
        .get_resource::<TrafficPopulationRes>()
        .is_none_or(|p| p.stamp != stamp);
    if stale {
        derive_parked(world, stamp);
    }
    plan_batch(world)
}

/// Every kerb slot's car, in one pass — the cheap half of the derivation.
fn derive_parked(world: &mut EcsWorld, stamp: u64) {
    let Some(res) = carriageway_of(world) else {
        return;
    };
    let slots = kerb_slots(&res.streets);
    // **What the level already had.** A re-derivation is not a fresh start: a
    // settlement paging in one block across town changes the block stamp, and a
    // rebuild that dropped every record would reset the phase of every car on
    // the road AND un-steal the one the player is sitting in. Every guid this
    // derivation produces again keeps the record it already had — the guid is a
    // pure function of the space, so "again" is exactly the cars that did not
    // move.
    let kept: BTreeMap<Uuid, TrafficRecord> = world
        .world()
        .get_resource::<TrafficPopulationRes>()
        .map(|p| p.records.clone())
        .unwrap_or_default();
    let planned: BTreeMap<Uuid, DVec3> = world
        .world()
        .get_resource::<TrafficPopulationRes>()
        .map(|p| p.pending.clone())
        .unwrap_or_default();
    let mut records: BTreeMap<Uuid, TrafficRecord> = BTreeMap::new();
    let mut pending: BTreeMap<Uuid, DVec3> = BTreeMap::new();
    // How many cars already HAVE a day -- see `MAX_COMMUTERS`.
    let mut with_a_day = 0usize;
    // A car the player has touched is kept whatever the geometry did: it is not
    // the traffic's any more, so a derivation that no longer names its slot has
    // no business forgetting it.
    for (g, r) in &kept {
        if r.taken {
            records.insert(*g, r.clone());
        }
    }
    for (p, yaw) in slots {
        if records.len() >= MAX_TRAFFIC_CARS {
            break;
        }
        let guid = parked_car_guid(p);
        if crate::crowd::agent_unit(guid, 0, SALT_PARK) >= KERB_OCCUPANCY {
            continue;
        }
        let def = catalogue_row(guid);
        // The record's own space is LIFTED to the car's ride height; the route
        // planner is handed the slot as it is on the ground (see `drive_path`).
        let at = DVec3::new(p.x, crate::vehicle::resting_origin_y(&def, p.y), p.z);
        let flat = p;
        match kept.get(&guid) {
            Some(old) => {
                let has_day = old.schedule.is_some() || old.circuit.is_some();
                with_a_day += usize::from(has_day);
                records.insert(guid, old.clone());
                // A car whose route never got planned is still waiting for one.
                if day_of(guid) != TrafficDay::Parked
                    && !has_day
                    && planned.contains_key(&guid)
                    && with_a_day + pending.len() < MAX_COMMUTERS
                {
                    pending.insert(guid, flat);
                }
            }
            None => {
                records.insert(guid, TrafficRecord::parked(def, car_paint(guid), at, yaw));
                if day_of(guid) != TrafficDay::Parked && with_a_day + pending.len() < MAX_COMMUTERS
                {
                    pending.insert(guid, flat);
                }
            }
        }
    }
    let steps = world
        .world()
        .get_resource::<TrafficPopulationRes>()
        .map(|p| p.steps)
        .unwrap_or(0);
    // Any car that is no longer derived loses its body first: a record dropped
    // while its entities stand is the crowd's own two-opinions defect
    // (`set_population` despawns before it replaces).
    let gone: Vec<(Uuid, crate::vehicle::VehicleDef)> = world
        .world()
        .get_resource::<TrafficPopulationRes>()
        .map(|p| {
            p.records
                .iter()
                // **Never a car the player has touched.** `taken`'s own doc
                // promises the traffic never takes its body down, and a block
                // paging in across town would otherwise despawn the chassis the
                // player is sitting in.
                .filter(|(g, r)| {
                    r.detail != RigDetail::None && !r.taken && !records.contains_key(g)
                })
                .map(|(g, r)| (*g, r.def))
                .collect()
        })
        .unwrap_or_default();
    for (g, def) in gone {
        crate::vehicle::despawn_rig(world, g, &def);
    }
    world.world_mut().insert_resource(TrafficPopulationRes {
        records,
        pending,
        steps,
        stamp,
        hand_installed: false,
    });
}

/// Plan up to [`TRAFFIC_PLANS_PER_STEP`] commuter routes.
fn plan_batch(world: &mut EcsWorld) -> usize {
    // **Nothing to plan is nothing to build.** The queue is the guard, and it
    // has to be checked BEFORE the two derivations below: a settled level would
    // otherwise rebuild its whole street graph and its whole slot list every
    // fixed step for the life of the session, to plan zero routes. Found by
    // this wave's own adversarial read, and it falsified three doc claims at
    // once — including `TrafficRes::derivations`' *"a counter a gate can assert
    // is one, which is what says the cache is a cache"*, which stayed at one
    // while the expensive products were recomputed outside it.
    if world
        .world()
        .get_resource::<TrafficPopulationRes>()
        .is_none_or(|p| p.pending.is_empty())
    {
        return 0;
    }
    let Some(res) = world.world().get_resource::<TrafficRes>().cloned() else {
        return 0;
    };
    let Some(mut pop) = world.world_mut().remove_resource::<TrafficPopulationRes>() else {
        return 0;
    };
    let graph = carriageway_graph(&res.streets);
    let slots = kerb_slots(&res.streets);
    let mut planned = 0;
    while planned < TRAFFIC_PLANS_PER_STEP {
        let Some((&guid, &home)) = pop.pending.iter().next() else {
            break;
        };
        pop.pending.remove(&guid);
        planned += 1;
        // **Which kind of day**, from the car's own seed and nothing else — so a
        // level derives the same mix on both hosts, a re-derivation does not
        // reshuffle the street, and the two fields are exclusive by
        // construction: this is the only place either is written.
        match day_of(guid) {
            TrafficDay::Parked => {}
            TrafficDay::NightCircuit | TrafficDay::DayCircuit => {
                let hours = if day_of(guid) == TrafficDay::NightCircuit {
                    NIGHT_CIRCUIT_H
                } else {
                    DAY_CIRCUIT_H
                };
                if let Some(c) = plan_circuit(&graph, &res.lanes, &slots, guid, home, hours) {
                    if let Some(rec) = pop.records.get_mut(&guid) {
                        rec.circuit = Some(c);
                    }
                }
            }
            TrafficDay::Commute => {
                if let Some(sched) = plan_commute(&graph, &res.lanes, &slots, guid, home) {
                    if let Some(rec) = pop.records.get_mut(&guid) {
                        rec.schedule = Some(sched);
                    }
                }
            }
        }
    }
    world.world_mut().insert_resource(pop);
    planned
}

/// **One commuter's day**, as two legs over the carriageway.
///
/// Home is the car's own kerb space; work is another slot, drawn from
/// [`SALT_DEST`] and taken far enough away to be a journey. The path is
/// `[home, approach, lane route…, approach, work]`, so a drive leaves its space,
/// runs the lanes and pulls in facing the way the destination row faces.
///
/// `None` when the two ends are not connected, which is a refusal as a value:
/// the car stays parked, for ever, which is a thing cars do.
pub fn plan_commute(
    graph: &NavGraph,
    lanes: &LaneNetwork,
    slots: &[(DVec3, f64)],
    guid: Uuid,
    home: DVec3,
) -> Option<crate::crowd::CrowdSchedule> {
    let (dest_p, dest_yaw) = pick_destination(slots, guid, home)?;
    let home_yaw = home_yaw_at(slots, home);
    let out = drive_path(graph, lanes, home, home_yaw, dest_p, dest_yaw)?;
    let back = drive_path(graph, lanes, dest_p, dest_yaw, home, home_yaw)?;
    crate::crowd::CrowdSchedule::new(vec![
        crate::crowd::ScheduleLeg {
            start_h: crate::society::WORK_START_H,
            travel_h: crate::society::COMMUTE_H,
            path: out,
            arrival: crate::crowd::SlotArrival::STANDING,
        },
        crate::crowd::ScheduleLeg {
            start_h: crate::society::HOME_H,
            travel_h: crate::society::COMMUTE_H,
            path: back,
            arrival: crate::crowd::SlotArrival::STANDING,
        },
    ])
}

/// What kind of day one car has — [`COMMUTER_SHARE`]'s table, as a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TrafficDay {
    /// It never moves.
    Parked,
    /// A loop through the small hours.
    NightCircuit,
    /// A loop through the working day.
    DayCircuit,
    /// Out at eight, home at six.
    Commute,
}

/// **The one draw** that decides a car's day. See [`COMMUTER_SHARE`].
pub fn day_of(guid: Uuid) -> TrafficDay {
    let d = crate::crowd::agent_unit(guid, 0, SALT_DAY);
    if d < NIGHT_CIRCUIT_SHARE {
        TrafficDay::NightCircuit
    } else if d < NIGHT_CIRCUIT_SHARE + DAY_CIRCUIT_SHARE {
        TrafficDay::DayCircuit
    } else if d < NIGHT_CIRCUIT_SHARE + DAY_CIRCUIT_SHARE + COMMUTER_SHARE {
        TrafficDay::Commute
    } else {
        TrafficDay::Parked
    }
}

/// **One car's loop**, out along the lanes and back down the other side.
///
/// The two halves are two [`drive_path`]s in opposite directions, so a circuit
/// runs out on one carriageway and home on the other — a real loop rather than a
/// car reversing down its own lane. Its speed is stated
/// ([`CIRCUIT_SPEED_FRAC`] of the sign) and not implied by a window, which is
/// the whole difference between this and a commute.
///
/// `None` when the two ends are not connected. A refusal is a value: the car
/// stays parked.
pub fn plan_circuit(
    graph: &NavGraph,
    lanes: &LaneNetwork,
    slots: &[(DVec3, f64)],
    guid: Uuid,
    home: DVec3,
    hours: (f64, f64),
) -> Option<Circuit> {
    let (dest_p, dest_yaw) = pick_destination(slots, guid, home)?;
    let home_yaw = home_yaw_at(slots, home);
    let out = drive_path(graph, lanes, home, home_yaw, dest_p, dest_yaw)?;
    let back = drive_path(graph, lanes, dest_p, dest_yaw, home, home_yaw)?;
    let mut pts: Vec<DVec3> = out.points().to_vec();
    pts.extend_from_slice(back.points());
    // The ends coincide, which is what makes `RouteMode::Loop` a loop rather
    // than a jump: `NavPath::new` drops the duplicate and the fold closes it.
    let path = NavPath::new(pts);
    if path.is_stand() {
        return None;
    }
    Some(Circuit {
        route: crate::crowd::CrowdRoute::along(
            path,
            street_speed_mps() * CIRCUIT_SPEED_FRAC,
            crate::crowd::RouteMode::Loop,
        ),
        from_h: hours.0,
        to_h: hours.1,
    })
}

/// The yaw of the slot at `home`, or zero — so a loop pulls back into its own
/// space facing the way the row faces.
fn home_yaw_at(slots: &[(DVec3, f64)], home: DVec3) -> f64 {
    slots
        .iter()
        .find(|(p, _)| {
            let d = *p - home;
            (d.x * d.x + d.z * d.z).sqrt() < 0.5
        })
        .map(|(_, y)| *y)
        .unwrap_or(0.0)
}

/// **A destination slot, drawn from this car's own seed** — at least
/// [`COMMUTE_MIN_M`] away, walked forward until one qualifies so the draw cannot
/// fail on a small town.
fn pick_destination(slots: &[(DVec3, f64)], guid: Uuid, home: DVec3) -> Option<(DVec3, f64)> {
    if slots.len() < 2 {
        return None;
    }
    let start = (crate::crowd::agent_unit(guid, 0, SALT_DEST) * slots.len() as f64) as usize;
    let mut furthest: Option<(f64, DVec3, f64)> = None;
    for k in 0..slots.len() {
        let (p, yaw) = slots[(start + k) % slots.len()];
        let d = p - home;
        let m = (d.x * d.x + d.z * d.z).sqrt();
        if m >= COMMUTE_MIN_M {
            return Some((p, yaw));
        }
        if furthest.is_none_or(|(b, _, _)| m > b) {
            furthest = Some((m, p, yaw));
        }
    }
    // **A small town still has traffic.** The first cut answered `None` when no
    // slot was a hundred metres off, and on the CI island's own two-street
    // fixture that is EVERY slot — fifteen cars, none of them with a day, and a
    // rush-hour gate reporting an empty street. A journey shorter than a city
    // block is still a journey; what it must not be is a car pulling out of one
    // space and into the next, which is what the walk above prefers and this
    // falls back from.
    furthest.and_then(|(m, p, yaw)| (m > KERB_SLOT_M * 2.0).then_some((p, yaw)))
}

/// How far a commute has to be to be worth driving, metres.
///
/// A hundred metres is one city block: shorter than that and the "commute" is a
/// car pulling out of one space and into the next, which reads as a glitch
/// rather than as traffic.
pub const COMMUTE_MIN_M: f64 = 100.0;

/// **One drive, as a path** — out of a space, along the lanes, into a space.
///
/// The two ends are kerb slots and the middle is a lane route, joined by the
/// nearest carriageway node at each end. `from_yaw` and `to_yaw` are the rows
/// the two spaces belong to, so the first two points and the last two run
/// *along* the kerb rather than diagonally out of it (see [`APPROACH_M`]).
///
/// Both ends are **unlifted** — the slot's own plan position at its street's
/// height. A car's ride height is applied once, in
/// [`TrafficRecord::place`], from the ground it measured; putting it into the
/// path as well would make a leg start a wheel-radius above the road and drop
/// to the pad for the whole drive.
pub fn drive_path(
    graph: &NavGraph,
    lanes: &LaneNetwork,
    from: DVec3,
    from_yaw: f64,
    to: DVec3,
    to_yaw: f64,
) -> Option<NavPath> {
    let a = graph.nearest_planar(from, f64::INFINITY)?;
    let b = graph.nearest_planar(to, f64::INFINITY)?;
    if a == b {
        return None;
    }
    let route = inf_nav::route(graph, a, b).route()?;
    let ids = lanes.lane_route(&route.nodes, 0);
    if ids.is_empty() {
        return None;
    }
    let mid = lanes.path_of(&ids);
    let mut pts: Vec<DVec3> = Vec::with_capacity(mid.points().len() + 4);
    // **An approach at BOTH ends.** The doc has always said `[home, approach,
    // lanes…, approach, work]` and the first cut only laid the far one, so a
    // car left its space on a diagonal straight into the lane — the exact
    // symptom `APPROACH_M` exists to prevent, applied to the arrival and not to
    // the departure.
    let out_h = heading_of_yaw(from_yaw);
    pts.push(from);
    pts.push(from + out_h * APPROACH_M);
    pts.extend_from_slice(mid.points());
    let in_h = heading_of_yaw(to_yaw);
    pts.push(to - in_h * APPROACH_M);
    pts.push(to);
    let path = NavPath::new(pts);
    (!path.is_stand()).then_some(path)
}

/// The unit ground heading a compass yaw names — [`yaw_of_dir`]'s inverse.
///
/// `psin64`/`pcos64` and never `f64::sin`/`cos`: this vector decides a metre a
/// committed drive passes through (the P14 law).
pub fn heading_of_yaw(yaw_deg: f64) -> DVec3 {
    let r = yaw_deg.to_radians();
    DVec3::new(inf_math::psin64(r), 0.0, inf_math::pcos64(r))
}

/// **Which catalogue row a slot's car is** — drawn from the car's own guid.
///
/// The rows are the five the island's own fleet declares, named here rather
/// than read from it because `inf-ecs` is Ring 0 and the catalogue is authoring
/// input in Ring 1. What Ring 0 owns is the *geometry and tuning* of a class
/// ([`crate::vehicle::VehicleDef`]), and these are that type's own defaults
/// wearing five different silhouettes — a traffic stream of five shapes rather
/// than of one, which is what `frames/driving/0014` shows (a van, a saloon and
/// a pickup in three consecutive lanes).
pub fn catalogue_row(guid: Uuid) -> crate::vehicle::VehicleDef {
    use crate::vehicle::VehicleBody;
    // **`CIVILIAN`, not `ALL`** — the kerb trap, sprung and closed in the same
    // commit that could have sprung it (wave VEH2c). This draw is UNIFORM over
    // whatever list it is handed, so the day `ALL` grew a launch and a
    // helicopter was the day every sixth and seventh kerb slot in every town on
    // the island would have held a boat. `VehicleBody::CIVILIAN`'s own doc
    // carries the rest of the argument; the size table below stays exhaustive
    // over `VehicleBody`, so that the NEXT family is a compile error here
    // rather than a boat on a pavement.
    let bodies = VehicleBody::CIVILIAN;
    let i = (crate::crowd::agent_unit(guid, 0, SALT_CLASS) * bodies.len() as f64) as usize;
    let body = bodies[i.min(bodies.len() - 1)];
    let mut def = crate::vehicle::VehicleDef {
        body,
        ..Default::default()
    };
    // The silhouettes differ in size as well as in shape, or five families draw
    // as one box in five costumes.
    let (l, w, h) = match body {
        VehicleBody::Sports => (2.10, 0.90, 0.60),
        VehicleBody::Sedan => (2.25, 0.92, 0.72),
        VehicleBody::Suv => (2.35, 0.98, 0.88),
        VehicleBody::Truck => (2.70, 1.02, 0.85),
        VehicleBody::Van => (2.70, 1.00, 1.05),
        // Unreachable from the draw above and deliberately still written:
        // this match is the tripwire that makes a new family visible HERE,
        // and an arm that answered with a wildcard would have let a launch
        // through wearing a saloon's dimensions.
        VehicleBody::Launch => (2.60, 1.05, 0.95),
        VehicleBody::Rotorcraft => (2.40, 1.20, 1.00),
    };
    def.half_extents = crate::math::Vec3d::new(w, h, l);
    def.half_track_m = w - 0.12;
    def.half_wheelbase_m = l - 0.55;
    size_the_suspension(&mut def);
    def
}

/// The share of a strut's travel a parked car sits at.
///
/// Forty-five per cent, which is a road car: enough left to absorb a kerb, and
/// enough used that the spring is doing something at rest.
pub const STATIC_SAG_FRAC: f64 = 0.45;

/// How close to critically damped a traffic car's strut is.
///
/// 0.4 — under-damped, like a road car, so it settles in about a second instead
/// of arriving dead and instead of pogoing.
pub const DAMPING_RATIO: f64 = 0.4;

/// Tractive effort a class is given, as a multiple of its own mass — so the
/// number is an acceleration.
///
/// 3.5 N a kilogram is 0.36 g, which is a brisk road car and is what the
/// driveline ceiling `max_engine_force_n` is for.
pub const DRIVE_N_PER_KG: f64 = 3.5;

/// Braking effort, same units. 9.0 is 0.92 g — the tyre runs out first, which
/// is what `abs_slip` is there to manage.
pub const BRAKE_N_PER_KG: f64 = 9.0;

/// How much air a parked car has under its hull, metres.
///
/// Twenty centimetres, measured at the STATIC SAG rather than at full
/// extension — which is the whole point of the number. A hull that clears the
/// road only with its springs unloaded is a hull that lands on the road the
/// moment anybody sits in it.
pub const GROUND_CLEARANCE_M: f64 = 0.20;

/// **Give a class springs that can hold up the body it is bolted to.**
///
/// # The defect this exists to fix, in numbers
///
/// `VehicleDef::default()`'s suspension is the P29.7 test rig's: 20 kN/m over
/// 0.25 m of travel, sized for a 1.2-tonne box. [`catalogue_row`] gives a Van a
/// 2 x 2.1 x 5.4 m hull, which at the 150 kg/m³ hollow-shell convention is
/// **3 402 kg** — and 20 kN/m over four corners carries 5 kN of a 33 kN car. The
/// struts bottom out, the chassis collider lands on the road, and the van drives
/// on its BELLY: the wheels see a tenth of the load they should, the tyres make
/// almost no force, and a player who steals it holds full throttle and goes
/// nowhere. Measured before this function existed: 4 104 N of suspension load
/// under 33 373 N of weight, `slip_ratio` 0.0 and 3.5 m travelled in ten
/// seconds under full throttle.
///
/// So the geometry decides the mass and the **mass decides the springs**, in one
/// place, rather than a table of numbers a silhouette can outgrow.
/// `every_catalogue_row_sits_inside_its_own_travel` is the arm.
pub fn size_the_suspension(def: &mut crate::vehicle::VehicleDef) {
    let mass =
        8.0 * def.half_extents.x * def.half_extents.y * def.half_extents.z * def.density_kg_m3;
    if !(mass.is_finite() && mass > 0.0) {
        return;
    }
    let corner = mass * 0.25;
    let travel = if def.class.travel_m.is_finite() && def.class.travel_m > 0.0 {
        def.class.travel_m
    } else {
        0.25
    };
    // **The wheels hang below the BODY, whatever the body is.**
    //
    // `wheel_drop_m` is the default rig's -0.75, which suits a hull half a metre
    // tall and buries one that is a metre. Solving for the hull's bottom edge at
    // static sag: `origin = -drop + radius - sag`, `bottom = origin - h`, and
    // `bottom = GROUND_CLEARANCE_M` gives the drop below. Without it a Van's
    // hull sits 6 cm UNDER the road, the chassis collider carries the car
    // instead of the tyres, and full throttle produces 2.7 m in ten seconds.
    let sag = travel * STATIC_SAG_FRAC;
    def.wheel_drop_m = -(def.half_extents.y + GROUND_CLEARANCE_M + sag - def.wheel_radius_m);
    // `k x sag = corner x g`, with the sag a fixed share of the travel.
    let k = corner * 9.81 / (travel * STATIC_SAG_FRAC);
    def.class.stiffness_n_per_m = k;
    def.class.damping_ns_per_m = 2.0 * DAMPING_RATIO * (k * corner).sqrt();
    def.class.max_engine_force_n = mass * DRIVE_N_PER_KG;
    def.class.brake_force_n = mass * BRAKE_N_PER_KG;
    def.class.handbrake_force_n = mass * BRAKE_N_PER_KG * 0.7;
}

/// **A traffic car's paint** — eight body colours, drawn per car.
///
/// A street of one colour is a car park of clones; eight is enough that a row of
/// seven at a kerb is very unlikely to repeat, and it is
/// [`crate::crowd::CROWD_LOOKS`]'s own argument one system over.
pub fn car_paint(guid: Uuid) -> crate::math::Color {
    const PAINT: [[f32; 3]; 8] = [
        [0.72, 0.73, 0.75],
        [0.10, 0.11, 0.13],
        [0.62, 0.14, 0.13],
        [0.14, 0.24, 0.46],
        [0.90, 0.90, 0.88],
        [0.20, 0.36, 0.26],
        [0.55, 0.47, 0.30],
        [0.35, 0.36, 0.40],
    ];
    let i = (crate::crowd::agent_unit(guid, 0, SALT_PAINT) * PAINT.len() as f64) as usize;
    let c = PAINT[i.min(PAINT.len() - 1)];
    crate::math::Color::new(c[0], c[1], c[2], 1.0)
}

/// The traffic population, if a level has one.
pub fn traffic_of(world: &EcsWorld) -> Option<&TrafficPopulationRes> {
    world.world().get_resource::<TrafficPopulationRes>()
}

/// **Install a traffic population by hand** — the instrument's door, and the
/// one that stops the derivation.
///
/// [`crate::crowd::set_population`]'s rule verbatim: a caller that installs a
/// population by hand owns it, so [`sync_traffic`] stops deriving one. Without
/// it a gate that installed a measured fleet would find the level's own kerbs
/// filling back up on the next step.
pub fn set_traffic(world: &mut EcsWorld, mut records: BTreeMap<Uuid, TrafficRecord>) {
    clear_traffic(world);
    // **Every installed record arrives bodiless.** `clear_traffic` above has
    // just despawned every rig in the world, so a record handed in carrying a
    // `detail` from a previous life would tell the step it already had a body
    // and the step would never build one — a car that exists in the table and
    // nowhere else. A caller cannot pre-decide a tier, which is
    // `set_crowd_population`'s own rule.
    for rec in records.values_mut() {
        rec.detail = RigDetail::None;
        rec.tier = CrowdTier::Dormant;
    }
    let stamp = carriageway_of(world).map(|r| r.stamp).unwrap_or(0);
    world.world_mut().insert_resource(TrafficPopulationRes {
        records,
        pending: BTreeMap::new(),
        steps: 0,
        stamp,
        hand_installed: true,
    });
}

/// Forget the traffic: take down every body and remove the resource.
pub fn clear_traffic(world: &mut EcsWorld) {
    let built: Vec<(Uuid, crate::vehicle::VehicleDef)> = world
        .world()
        .get_resource::<TrafficPopulationRes>()
        .map(|p| {
            p.records
                .iter()
                .filter(|(_, r)| r.detail != RigDetail::None)
                .map(|(g, r)| (*g, r.def))
                .collect()
        })
        .unwrap_or_default();
    for (g, def) in built {
        crate::vehicle::despawn_rig(world, g, &def);
    }
    world.world_mut().remove_resource::<TrafficPopulationRes>();
}

/// **The bytes two hosts compare** — one row per car, in `Guid` order.
///
/// [`crate::crowd::crowd_state_bytes`]'s shape and its reasons: the guid, the
/// tier, the detail, the place, the heading and the phase, because those are
/// the things the *simulation* decides. The schedule is not folded — it is
/// derived from the level and a host that had a different one would already
/// differ in every place below it.
pub fn traffic_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(pop) = traffic_of(world) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(pop.records.len() * CAR_TRACE_BYTES);
    for (guid, rec) in &pop.records {
        out.extend_from_slice(guid.as_bytes());
        out.push(rec.tier.as_u8());
        out.push(rec.detail.as_u8());
        out.push(rec.leg);
        out.extend_from_slice(&rec.last.x.to_le_bytes());
        out.extend_from_slice(&rec.last.y.to_le_bytes());
        out.extend_from_slice(&rec.last.z.to_le_bytes());
        out.extend_from_slice(&rec.yaw_deg.to_le_bytes());
        out.extend_from_slice(&rec.rephase_m.to_le_bytes());
        out.push(u8::from(rec.taken));
    }
    out
}

/// How many bytes one car folds into [`traffic_state_bytes`].
pub const CAR_TRACE_BYTES: usize = 16 + 1 + 1 + 1 + 24 + 8 + 8 + 1;

/// **Mark a car as no longer traffic's** — the one door both interferences go
/// through.
///
/// Returns whether the flag moved. Called by the carjack (which pulls a driver
/// out before anybody is sitting in the seat) and by the traffic step itself
/// (which sees a seat taken by somebody who is not this car's own driver). Two
/// callers, one rule, and neither of them owns a second copy of it.
pub fn mark_taken(world: &mut EcsWorld, chassis: Uuid) -> bool {
    let Some(mut pop) = world.world_mut().get_resource_mut::<TrafficPopulationRes>() else {
        return false;
    };
    match pop.records.get_mut(&chassis) {
        Some(rec) if !rec.taken => {
            rec.taken = true;
            true
        }
        _ => false,
    }
}

/// **How many fixed steps the traffic has run** — the tick a per-attempt draw
/// is taken on.
///
/// `0` on a level with no traffic, which is a value both hosts agree about.
pub fn steps(world: &EcsWorld) -> u64 {
    traffic_of(world).map(|p| p.steps).unwrap_or(0)
}

/// Whether this car is one the traffic has let go of.
pub fn is_taken(world: &EcsWorld, chassis: Uuid) -> bool {
    traffic_of(world).is_some_and(|p| p.records.get(&chassis).is_some_and(|r| r.taken))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(pts: &[(f64, f64)]) -> NavPath {
        NavPath::new(pts.iter().map(|&(x, z)| DVec3::new(x, 0.0, z)))
    }

    fn view<'a>(p: &'a NavPath, at: DVec3, fwd: DVec3, v: f64) -> DriveView<'a> {
        let s = p.project(at).s_m;
        DriveView {
            at,
            forward: fwd,
            forward_mps: v,
            path: p,
            s_m: s,
            speed_limit_mps: 14.0,
            gap_m: None,
            lateral_bias_m: 0.0,
            loops: false,
        }
    }

    /// **THE TWO YIELDS, PRICED** (wave EMS2) — and the cheaper one deadlocks.
    ///
    /// # The choice this arm exists to settle
    ///
    /// A siren coming up behind a civilian car can be answered two ways:
    ///
    /// * **stop in lane** — the responder is injected into the civilian's own
    ///   `gap_m` as a phantom obstacle and the *existing* stopping-distance rule
    ///   brings it to a halt. **Zero new fields**, zero new terms, and it reuses
    ///   a rule already proven by `a_car_behind_a_queue_slows_to_a_stop`;
    /// * **pull over** — one field on [`DriveView`] and one guarded term in
    ///   [`drive_intent`]'s wheel step.
    ///
    /// The first is cheaper by every measure of *edits*. It is also **wrong**,
    /// and the reason is arithmetic rather than taste: `drive_intent` "never
    /// changes lane and never overtakes" — its own stated v1 bound — so a
    /// civilian stopped in the lane stays inside
    /// `inf_physics::d3::traffic::CORRIDOR_HALF_M` for ever, `gap_ahead` keeps
    /// answering [`STANDING_GAP_M`], and the responder's own stopping-distance
    /// rule holds it at **exactly zero** behind it. The siren arrives at the
    /// back of the queue it created and stops there.
    ///
    /// The numbers below are the two `target_mps` a responder is left with, and
    /// they are what settled the clause. [`YIELD_BIAS_M`] is 2.6 m against a
    /// 2.5 m corridor half-width, which is the whole of why the pull-over works:
    /// the yielding car leaves the corridor, so the gap re-opens.
    ///
    /// [`YIELD_BIAS_M`]: crate::dispatch::YIELD_BIAS_M
    /// [`STANDING_GAP_M`]: STANDING_GAP_M
    #[test]
    fn a_car_that_stops_in_lane_stops_the_ambulance_behind_it() {
        let p = path(&[(0.0, 0.0), (300.0, 0.0)]);
        let limit = 11.7;
        let responder = |gap: Option<f64>| -> DriveIntent {
            let mut v = view(&p, DVec3::new(50.0, 0.0, 0.0), DVec3::X, 8.0);
            v.speed_limit_mps = limit;
            v.gap_m = gap;
            drive_intent(&v)
        };
        // (a) the cheap design: the civilian stops IN the lane, so it stays an
        //     obstacle at the standing gap for ever.
        let stopped = responder(Some(STANDING_GAP_M));
        // (b) the shipped design: the civilian has left the corridor, so
        //     `gap_ahead` no longer sees it at all.
        let cleared = responder(None);
        println!(
            "EMS2 yield pricing: stop-in-lane leaves the responder at \
             {:.3} m/s; pull-over leaves it at {:.3} m/s (limit {limit})",
            stopped.target_mps, cleared.target_mps
        );
        assert_eq!(
            stopped.target_mps, 0.0,
            "a civilian stopped at the standing gap left the responder \
             {:.3} m/s — if this is ever non-zero the pricing below has changed \
             and the ruling should be revisited",
            stopped.target_mps
        );
        assert_eq!(
            cleared.target_mps, limit,
            "a cleared lane did not give the responder its limit back"
        );
        // …and the 2.6 m of bias really does clear the 2.5 m corridor, which is
        // the load-bearing inequality of the whole clause.
        // Bound to locals rather than compared as constants, and not only to
        // satisfy a lint: `CORRIDOR_HALF_M` lives in `inf_physics::d3::traffic`,
        // which this crate may not name (the split's own wall), so the 2.5 is a
        // literal here and the pair reads as the measurement it is.
        let bias = crate::dispatch::YIELD_BIAS_M;
        let corridor_half_m = 2.5_f64;
        assert!(
            bias > corridor_half_m,
            "the pull-over ({bias} m) no longer clears the following rule's \
             {corridor_half_m} m corridor half-width, so a yielding car stays \
             an obstacle and the design silently degenerates into the \
             stop-in-lane this arm rejects"
        );
    }

    /// **THE BIAS MOVES THE CAR, AND ZERO MOVES NOTHING AT ALL.**
    ///
    /// The second half is the one worth having: the term is *guarded* so a level
    /// with no siren in it steers the bits it always steered, and a guard that
    /// was quietly dropped would still pass an "it steers right" assertion.
    /// Compared as `==` on the whole intent, because "bit-identical" is the
    /// claim.
    #[test]
    fn a_yielding_car_aims_at_the_kerb_and_a_zero_bias_changes_nothing() {
        let p = path(&[(0.0, 0.0), (300.0, 0.0)]);
        let base = view(&p, DVec3::new(50.0, 0.0, 0.0), DVec3::X, 8.0);
        let plain = drive_intent(&base);
        let mut zero = base;
        zero.lateral_bias_m = 0.0;
        assert_eq!(
            plain,
            drive_intent(&zero),
            "a zero bias changed the intent — the guard is gone and every level \
             committed before this wave steers different bits"
        );
        let mut over = base;
        over.lateral_bias_m = crate::dispatch::YIELD_BIAS_M;
        let yielded = drive_intent(&over);
        println!(
            "EMS2 yield: steer {:.4} -> {:.4}",
            plain.move_input.x, yielded.move_input.x
        );
        assert!(
            plain.move_input.x.abs() < 1e-12,
            "the control is not straight to begin with: {plain:?}"
        );
        assert!(
            yielded.move_input.x > 0.05,
            "a car told to pull {} m over steered {:.4} — `right_of`'s sign is \
             positive, so this must be a right-hand turn onto the kerb",
            crate::dispatch::YIELD_BIAS_M,
            yielded.move_input.x
        );
        // …and it does not become a hard turn: the pull-over is a lane's
        // half-width over a lookahead, not a swerve.
        assert!(
            yielded.move_input.x < 0.9,
            "the yield asks for {:.4} of full lock — that is a swerve, not a \
             pull-over",
            yielded.move_input.x
        );
    }

    /// **THE RULE FIRES FOR A SIREN BEHIND AND FOR NOTHING ELSE.**
    ///
    /// Four negatives, and each is a street this rule must not stop: a unit in
    /// front, a unit on the next street over, a unit three blocks back, and an
    /// empty list.
    #[test]
    fn only_a_siren_behind_and_in_line_makes_a_car_pull_over() {
        use crate::dispatch::{yield_bias_m, YIELD_BIAS_M};
        let at = DVec3::new(100.0, 0.0, 0.0);
        let fwd = DVec3::X;
        let unit = |x: f64, z: f64| vec![(Uuid::from_u128(1), DVec3::new(x, 0.0, z))];
        assert_eq!(yield_bias_m(at, fwd, &unit(70.0, 0.0)), YIELD_BIAS_M);
        assert_eq!(
            yield_bias_m(at, fwd, &unit(130.0, 0.0)),
            0.0,
            "a car pulled over for a unit coming the other way — it would be \
             moving INTO it"
        );
        assert_eq!(
            yield_bias_m(at, fwd, &unit(70.0, 20.0)),
            0.0,
            "a car pulled over for a unit on the next street"
        );
        assert_eq!(
            yield_bias_m(at, fwd, &unit(-100.0, 0.0)),
            0.0,
            "a car pulled over for a unit 200 m back"
        );
        assert_eq!(yield_bias_m(at, fwd, &[]), 0.0);
    }

    /// A car on its lane, pointing down it, at the limit, asks for nothing.
    #[test]
    fn a_car_on_the_line_at_the_limit_holds_both_controls_still() {
        let p = path(&[(0.0, 0.0), (200.0, 0.0)]);
        let i = drive_intent(&view(&p, DVec3::new(50.0, 0.0, 0.0), DVec3::X, 14.0));
        assert!(i.move_input.x.abs() < 1e-12, "{i:?}");
        assert!(i.move_input.y.abs() < 1e-12, "{i:?}");
        assert!(!i.handbrake);
        assert_eq!(i.target_mps, 14.0);
        assert!(i.lateral_m.abs() < 1e-12);
    }

    /// The two halves of "hold a speed": under it the throttle goes down, over
    /// it the stick goes negative — which `from_intent` reads as a BRAKE.
    #[test]
    fn the_pedal_closes_a_speed_error_from_either_side() {
        let p = path(&[(0.0, 0.0), (400.0, 0.0)]);
        let slow = drive_intent(&view(&p, DVec3::new(50.0, 0.0, 0.0), DVec3::X, 2.0));
        assert_eq!(slow.move_input.y, 1.0, "{slow:?}");
        let fast = drive_intent(&view(&p, DVec3::new(50.0, 0.0, 0.0), DVec3::X, 30.0));
        assert_eq!(fast.move_input.y, -1.0, "{fast:?}");
        // …and inside the band it is proportional, not a bang-bang.
        let near = drive_intent(&view(&p, DVec3::new(50.0, 0.0, 0.0), DVec3::X, 13.0));
        assert!(
            (near.move_input.y - 1.0 / SPEED_BAND_MPS).abs() < 1e-12,
            "{near:?}"
        );
    }

    /// Wave VEH2b's own hazard, pinned: a negative stick at rest is REVERSE, so
    /// a stopped traffic car must ask for the handbrake instead.
    #[test]
    fn a_stopped_car_holds_the_handbrake_and_never_asks_for_reverse() {
        let p = path(&[(0.0, 0.0), (200.0, 0.0)]);
        let mut v = view(&p, DVec3::new(50.0, 0.0, 0.0), DVec3::X, 0.0);
        v.gap_m = Some(STANDING_GAP_M);
        let i = drive_intent(&v);
        assert_eq!(i.target_mps, 0.0);
        assert_eq!(i.move_input.y, 0.0, "{i:?}");
        assert!(i.handbrake);
        // And what `from_intent` makes of it: nothing at all, rather than a
        // reverse gear.
        let c = crate::vehicle::VehicleControls::from_intent(i.move_input, 0.0, i.handbrake, 0.0);
        assert_eq!(c.throttle, 0.0);
        assert_eq!(c.brake, 0.0);
        assert!(c.handbrake);
    }

    /// The steering: a car parked 2 m to the left of its lane steers right, and
    /// the sign is the one `right_of` fixed.
    #[test]
    fn a_car_off_its_lane_steers_back_onto_it() {
        let p = path(&[(0.0, 0.0), (200.0, 0.0)]);
        // Heading +X, "right" is -Z, so a car at z = +2 is 2 m to the LEFT.
        let i = drive_intent(&view(&p, DVec3::new(50.0, 0.0, 2.0), DVec3::X, 10.0));
        assert!(i.move_input.x > 0.0, "should steer right: {i:?}");
        assert!((i.lateral_m + 2.0).abs() < 1e-9, "{i:?}");
        let j = drive_intent(&view(&p, DVec3::new(50.0, 0.0, -2.0), DVec3::X, 10.0));
        assert!(j.move_input.x < 0.0, "should steer left: {j:?}");
        assert!((j.lateral_m - 2.0).abs() < 1e-9, "{j:?}");
    }

    /// The clamp: a car pointing the wrong way asks for everything the steering
    /// has, rather than for a tangent that ran to infinity.
    #[test]
    fn a_car_facing_away_asks_for_full_lock_and_a_finite_number() {
        let p = path(&[(0.0, 0.0), (200.0, 0.0)]);
        let i = drive_intent(&view(
            &p,
            DVec3::new(50.0, 0.0, 4.0),
            DVec3::new(-1.0, 0.0, 0.0),
            1.0,
        ));
        assert_eq!(i.move_input.x.abs(), 1.0, "{i:?}");
        assert!(i.move_input.x.is_finite());
    }

    /// The bend rule: a right-angle corner is slower than a straight, and it is
    /// slower by the lateral-acceleration arithmetic rather than by a guess.
    #[test]
    fn a_bend_is_taken_slower_than_a_straight() {
        let straight = path(&[(0.0, 0.0), (400.0, 0.0)]);
        assert_eq!(
            corner_speed_mps(&straight, 10.0, 20.0, false),
            f64::INFINITY,
            "a straight has no bend"
        );
        let bend = path(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]);
        let v = corner_speed_mps(&bend, 95.0, 20.0, false);
        assert!(v.is_finite() && v > 0.0, "{v}");
        // A 90 degrees turn over 20 m of lookahead: curvature is 1/20, so the
        // speed is sqrt(2.5 * 20) = 7.07 m/s. The cross product is sin(90) = 1.
        assert!(
            (v - (CORNER_LATERAL_MPS2 * 20.0).sqrt()).abs() < 1e-9,
            "{v}"
        );
        // …and the controller actually uses it.
        let i = drive_intent(&DriveView {
            at: DVec3::new(95.0, 0.0, 0.0),
            forward: DVec3::X,
            forward_mps: 20.0,
            path: &bend,
            s_m: 95.0,
            speed_limit_mps: 25.0,
            gap_m: None,
            lateral_bias_m: 0.0,
            loops: false,
        });
        assert!(i.target_mps < 25.0, "{i:?}");
    }

    /// The stop-and-wait rule: closing on a queue slows continuously, and the
    /// standing gap is where it reaches zero.
    #[test]
    fn a_car_behind_a_stopped_one_slows_continuously_and_stops_behind_it() {
        let p = path(&[(0.0, 0.0), (400.0, 0.0)]);
        let ask = |gap: f64| {
            let mut v = view(&p, DVec3::new(50.0, 0.0, 0.0), DVec3::X, 14.0);
            v.gap_m = Some(gap);
            drive_intent(&v).target_mps
        };
        let far = ask(100.0);
        let mid = ask(30.0);
        let near = ask(10.0);
        let touching = ask(STANDING_GAP_M);
        assert!(
            far > mid && mid > near && near > touching,
            "{far} {mid} {near} {touching}"
        );
        assert_eq!(touching, 0.0);
        // The limit still binds on an open road far behind a queue.
        assert_eq!(far, 14.0);
        // …and the arithmetic is the stopping distance, not a curve somebody
        // liked: at 30 m the clear road is 24 m and sqrt(2*3.5*24) = 12.96.
        assert!((mid - (2.0 * COMFORT_DECEL_MPS2 * 24.0).sqrt()).abs() < 1e-12);
    }

    /// A car that has run out of lane stops at the end of it rather than
    /// driving off.
    #[test]
    fn the_end_of_the_road_is_a_stop() {
        let p = path(&[(0.0, 0.0), (100.0, 0.0)]);
        // Ten metres out it is already braking for the end rather than holding
        // the limit…
        let near = drive_intent(&view(&p, DVec3::new(90.0, 0.0, 0.0), DVec3::X, 14.0));
        assert!(near.target_mps < 14.0, "{near:?}");
        assert!(near.move_input.y < 0.0, "{near:?}");
        // …and at it, it is stopped with the handbrake on.
        let at_it = drive_intent(&view(&p, DVec3::new(100.0, 0.0, 0.0), DVec3::X, 0.2));
        assert_eq!(at_it.target_mps, 0.0, "{at_it:?}");
        assert!(at_it.handbrake, "{at_it:?}");
    }

    /// Nonsense in, a usable stick out — the refusal-is-a-value rule, at the
    /// one function a physics world hands its numbers to.
    #[test]
    fn a_non_finite_view_still_answers_a_finite_stick() {
        let p = path(&[(0.0, 0.0), (100.0, 0.0)]);
        let i = drive_intent(&DriveView {
            at: DVec3::new(f64::NAN, 0.0, 0.0),
            forward: DVec3::ZERO,
            forward_mps: f64::NAN,
            path: &p,
            s_m: 0.0,
            speed_limit_mps: f64::NAN,
            lateral_bias_m: 0.0,
            gap_m: Some(f64::NAN),
            loops: false,
        });
        assert!(
            i.move_input.x.is_finite() && i.move_input.y.is_finite(),
            "{i:?}"
        );
        assert!(i.move_input.x.abs() <= 1.0 && i.move_input.y.abs() <= 1.0);
    }

    // ── the map ─────────────────────────────────────────────────────────────

    /// A world with a `col x row` grid of blocks on a `pitch` lattice, each one
    /// `pitch - street` across — the shape `inf_editor_core::settlement` plans,
    /// and the only input the derivation is allowed to read.
    fn block_grid(cols: i32, rows: i32, pitch: f64, street: f64, pad_y: f64) -> EcsWorld {
        use crate::components::{PcgVolume, ResidentSlot, Transform};
        use crate::math::Vec3d;
        let mut w = EcsWorld::new();
        let half = (pitch - street) * 0.5;
        for row in 0..rows {
            for col in 0..cols {
                let c = DVec2::new(f64::from(col) * pitch, f64::from(row) * pitch);
                let guid = Uuid::from_u64_pair(0x51, (row as u64) << 32 | col as u64);
                let e = w.spawn_with_guid(guid, "block", None);
                w.world_mut().entity_mut(e).insert(Transform {
                    translation: Vec3d::new(c.x, pad_y, c.y),
                    rotation: Vec3d::ZERO,
                    scale: Vec3d::ONE,
                });
                let mut v = PcgVolume {
                    extent: Vec2d::new(half, half),
                    ..Default::default()
                };
                // A volume with no residents is invisible to `volume_sites`,
                // which is the door this derivation shares with the society.
                v.residents = vec![ResidentSlot {
                    role: crate::components::SlotRole::Home,
                    at: DVec3::new(c.x, pad_y, c.y),
                    room: 0,
                    building: 0,
                    floor: 0,
                    index: 0,
                    node: 0,
                    posture: crate::components::SlotPosture::Stand,
                    shift: crate::components::SlotShift::Day,
                    face: DVec3::ZERO,
                }];
                w.world_mut().entity_mut(e).insert(v);
            }
        }
        w.propagate();
        w
    }

    fn grid_streets() -> Vec<Street> {
        // Two lines each way, 20 m apart, over a 200 m span: the shape a 2x2
        // block settlement leaves behind.
        vec![
            Street {
                a: DVec2::new(0.0, -100.0),
                b: DVec2::new(0.0, 100.0),
                y: 3.0,
                gap_m: 20.0,
            },
            Street {
                a: DVec2::new(-100.0, 0.0),
                b: DVec2::new(100.0, 0.0),
                y: 3.0,
                gap_m: 20.0,
            },
        ]
    }

    #[test]
    fn a_crossing_of_two_streets_is_one_node_and_four_arms() {
        let g = carriageway_graph(&grid_streets());
        // Four ends and one crossing.
        assert_eq!(g.len(), 5);
        let centre = carriageway_node_id(DVec2::new(0.0, 0.0));
        assert!(g.contains(centre));
        assert_eq!(g.edges_from(centre).len(), 4);
        // …and the crossing sits on the pad the blocks stand on.
        assert_eq!(g.node(centre).unwrap().position.y, 3.0);
    }

    #[test]
    fn the_derived_grid_carries_a_lane_each_way_on_every_arm() {
        let net = carriageway(&grid_streets());
        // Four arms, both ways, one lane each.
        assert_eq!(net.len(), 8);
        assert_eq!(net.worst_fold_m(), 0.0);
        for lane in net.lanes() {
            assert_eq!(lane.speed_limit_kmh, STREET_SPEED_KMH);
            assert_eq!(lane.width_m, DEFAULT_LANE_WIDTH_M);
        }
        // A car can get from one arm to any other but its own.
        let east = net
            .lanes()
            .find(|l| l.exit().x > 50.0)
            .expect("an eastbound arm");
        let arriving = inf_nav::LaneId {
            from: east.id.to,
            to: east.id.from,
            index: 0,
        };
        assert_eq!(net.successors(arriving).len(), 3);
    }

    /// A twenty-metre street is not a twenty-metre carriageway: the lanes are
    /// 3.5 m and the rest is verge, kerb and somewhere to park.
    #[test]
    fn a_street_keeps_room_at_the_kerb_for_the_cars_that_park_there() {
        assert!(kerb_fits(20.0));
        assert!(kerb_fits(16.0));
        assert!(!kerb_fits(10.0));
        let streets = grid_streets();
        let slots = kerb_slots(&streets);
        // 200 m of line holds thirteen whole 14 m slots a side; the one that
        // lands within 12 m of the crossing is refused, so twelve survive, on
        // two sides of two lines.
        assert_eq!(kerb_slots(&streets[..1]).len(), 2 * 13);
        assert_eq!(slots.len(), 2 * 2 * 12);
        // The fixture's streets are all 20 m, so every slot is at the same
        // offset. It is still `KERB_PARK_OFFSET_M` and not the kerb the paving
        // draws — see `kerb_park_offset_m` for the 1.100 m that separates them
        // and for the ambulance that does not come home if this moves.
        let want = KERB_PARK_OFFSET_M;
        for (p, _) in &slots {
            // Every slot is exactly `kerb_park_offset_m` from its own line —
            // which for this fixture is the axis it is NOT running along, so
            // one of the two coordinates is that number to the bit.
            let off = if (p.x.abs() - want).abs() < 1e-9 {
                p.x.abs()
            } else {
                p.z.abs()
            };
            assert!((off - want).abs() < 1e-9, "{p:?} is at {off}");
            // Off the running lanes…
            assert!(off > DEFAULT_LANE_WIDTH_M, "{p:?} is in a lane");
            // …on the carriageway rather than on the footway, which is what
            // parking at a kerb IS: the flank touches the kerb face.
            // …and off the pavement, on a 20 m street.
            assert!(off <= 10.0 - PAVEMENT_M, "{p:?} is on the pavement");
            // …and out of the junction: 12 m from the crossing at the origin.
            assert!(
                p.x.abs().max(p.z.abs()) >= 12.0 - 1e-9,
                "{p:?} is parked in the crossing"
            );
        }
    }

    /// The two sides of a street face opposite ways, so a kerb reads as parked
    /// rather than as a scatter of yaws.
    #[test]
    fn the_two_kerbs_of_one_street_face_opposite_ways() {
        let slots = kerb_slots(&grid_streets()[1..]);
        let yaws: BTreeSet<i64> = slots.iter().map(|(_, y)| y.round() as i64).collect();
        assert_eq!(yaws.len(), 2, "{yaws:?}");
        let v: Vec<i64> = yaws.into_iter().collect();
        assert_eq!((v[1] - v[0]).abs(), 180);
    }

    /// **The recovery, against the shape it is recovering.** A 3x3 grid of
    /// blocks on a 100 m pitch with a 20 m street leaves two gaps each way, and
    /// the derivation finds exactly those two, in the middle of each.
    #[test]
    fn the_streets_are_recovered_from_the_gaps_between_blocks() {
        let w = block_grid(3, 3, 100.0, 20.0, 7.5);
        let streets = streets_of(&w);
        assert_eq!(streets.len(), 4, "{streets:?}");
        let mut x_lines: Vec<i64> = streets
            .iter()
            .filter(|s| !s.along_x())
            .map(|s| s.a.x.round() as i64)
            .collect();
        x_lines.sort();
        assert_eq!(x_lines, vec![50, 150]);
        for s in &streets {
            assert!((s.gap_m - 20.0).abs() < 1e-9, "{s:?}");
            assert_eq!(s.y, 7.5);
        }
        // …and the lanes over them: four crossings' worth of arms.
        let net = carriageway(&streets);
        assert!(net.len() >= 8, "{}", net.len());
        assert_eq!(net.worst_fold_m(), 0.0);
    }

    /// Two settlements a kilometre apart do not imply a kilometre-wide street
    /// between them — the whole reason the derivation groups first.
    #[test]
    fn two_towns_are_not_one_street() {
        use crate::components::{PcgVolume, ResidentSlot, Transform};
        use crate::math::Vec3d;
        let mut w = block_grid(2, 1, 100.0, 20.0, 0.0);
        // A lone block a kilometre east.
        let e = w.spawn_with_guid(Uuid::from_u64_pair(0x99, 1), "far", None);
        w.world_mut().entity_mut(e).insert(Transform {
            translation: Vec3d::new(1_200.0, 0.0, 0.0),
            rotation: Vec3d::ZERO,
            scale: Vec3d::ONE,
        });
        let mut v = PcgVolume {
            extent: Vec2d::new(40.0, 40.0),
            ..Default::default()
        };
        v.residents = vec![ResidentSlot {
            role: crate::components::SlotRole::Home,
            at: DVec3::new(1_200.0, 0.0, 0.0),
            room: 0,
            building: 0,
            floor: 0,
            index: 0,
            node: 0,
            posture: crate::components::SlotPosture::Stand,
            shift: crate::components::SlotShift::Day,
            face: DVec3::ZERO,
        }];
        w.world_mut().entity_mut(e).insert(v);
        w.propagate();
        let streets = streets_of(&w);
        // The 2x1 pair still has its one street; the outlier is its own group
        // of one and contributes nothing.
        assert_eq!(streets.len(), 1, "{streets:?}");
        assert!((streets[0].a.x - 50.0).abs() < 1e-9);
    }

    /// The cache: derived once for a settled level, and never again.
    #[test]
    fn the_derivation_runs_when_the_blocks_move_and_not_otherwise() {
        let mut w = block_grid(2, 2, 100.0, 20.0, 0.0);
        assert!(sync_carriageway(&mut w));
        assert_eq!(carriageway_of(&w).unwrap().derivations, 1);
        for _ in 0..10 {
            assert!(!sync_carriageway(&mut w), "it re-derived a settled level");
        }
        assert_eq!(carriageway_of(&w).unwrap().derivations, 1);
        // A block appearing is a new level.
        let before = carriageway_of(&w).unwrap().stamp;
        let mut w2 = block_grid(3, 2, 100.0, 20.0, 0.0);
        assert!(sync_carriageway(&mut w2));
        assert_ne!(carriageway_of(&w2).unwrap().stamp, before);
        // …and clearing leaves a world that never had one.
        clear_carriageway(&mut w);
        assert!(carriageway_of(&w).is_none());
    }

    /// A level with nothing in it costs one walk and produces no traffic — the
    /// "absent costs nothing" discipline, asserted.
    #[test]
    fn a_level_with_no_blocks_has_no_streets() {
        let mut w = EcsWorld::new();
        assert!(streets_of(&w).is_empty());
        assert!(sync_carriageway(&mut w));
        assert!(carriageway_of(&w).unwrap().lanes.is_empty());
        assert!(kerb_slots(&[]).is_empty());
    }

    /// **Every catalogue row sits INSIDE its own suspension travel.**
    ///
    /// The falsifier for `size_the_suspension`: a class whose static sag is past
    /// its travel is a car resting on its chassis collider, which is a car that
    /// does not steer, does not brake and cannot be driven away.
    ///
    /// **`audit:` VEH2b — and it covers every silhouette by NAME.** Forty draws
    /// off one seed are whatever `SALT_CLASS` happened to produce; a sweep that
    /// silently stopped covering the Van — the heaviest row and the one the
    /// belly defect was measured on — would still be forty green assertions.
    #[test]
    fn every_catalogue_row_sits_inside_its_own_travel() {
        let mut seen: Vec<crate::vehicle::VehicleBody> = Vec::new();
        for k in 0..40u64 {
            let def = catalogue_row(Uuid::from_u64_pair(0xC0FFEE, k));
            if !seen.contains(&def.body) {
                seen.push(def.body);
            }
            let mass = 8.0
                * def.half_extents.x
                * def.half_extents.y
                * def.half_extents.z
                * def.density_kg_m3;
            let sag = mass * 0.25 * 9.81 / def.class.stiffness_n_per_m;
            assert!(
                sag < def.class.travel_m,
                "{:?} at {mass:.0} kg sags {sag:.3} m into {:.3} m of travel",
                def.body,
                def.class.travel_m
            );
            assert!(
                (sag / def.class.travel_m - STATIC_SAG_FRAC).abs() < 1e-6,
                "{:?} sags {:.3} of its travel",
                def.body,
                sag / def.class.travel_m
            );
            // …and it has the effort to move its own mass.
            assert!(def.class.max_engine_force_n > mass * 2.0);
            assert!(def.class.brake_force_n > def.class.max_engine_force_n);
            // …and its hull is off the road WITH the springs loaded, which is
            // the case the default rig's `wheel_drop_m` did not survive.
            let origin = crate::vehicle::resting_origin_y(&def, 0.0) - sag;
            let clearance = origin - def.half_extents.y;
            assert!(
                (clearance - GROUND_CLEARANCE_M).abs() < 1e-9,
                "{:?} clears the road by {clearance:.3} m at static sag",
                def.body
            );
        }
        // `CIVILIAN`, not `ALL` (wave VEH2c): this arm's subject is what
        // `catalogue_row` can produce, and a boat has no springs to check. The
        // two craft answer `suspension_rest_m() == 0.0`, which is asserted
        // where they are.
        assert_eq!(
            seen.len(),
            crate::vehicle::VehicleBody::CIVILIAN.len(),
            "the sweep only met {seen:?} — a silhouette this arm does not draw is a \
             silhouette nothing checks the springs of"
        );
    }

    /// **Every catalogue row can actually be driven.** The wave's own arms found
    /// a traffic car that would not move under full throttle, and the question a
    /// world-level test cannot answer is whether the fault is the world or the
    /// ROW: `catalogue_row` overrides four of a `VehicleDef`'s geometry fields,
    /// and a rig whose wheels the recogniser cannot tell apart, or whose gearbox
    /// **NO KERB IN ANY TOWN EVER HOLDS A BOAT** — the kerb trap, armed
    /// (wave VEH2c).
    ///
    /// The trap was written down twice in this repository before it could
    /// spring: `catalogue_row` draws a parked car's silhouette UNIFORMLY over a
    /// list, so the day `VehicleBody::ALL` grew a launch and a helicopter was
    /// the day every sixth and seventh kerb slot on the island would have held
    /// one. Wave EMS1 avoided it by borrowing bodies and left the remedy
    /// written: a named CIVILIAN sub-list, in the same commit.
    ///
    /// This arm is what makes the remedy a fact rather than a comment. Pointing
    /// the draw back at `ALL` reds it in one line with the offending family
    /// named.
    #[test]
    fn no_kerb_in_any_town_ever_holds_a_boat_or_a_helicopter() {
        use crate::vehicle::VehicleBody;
        // The list itself: every civilian body is a road vehicle, and no craft
        // is in it.
        for b in VehicleBody::CIVILIAN {
            assert!(b.wheeled(), "{:?} is in CIVILIAN and has mounts", b.name());
        }
        assert_eq!(
            VehicleBody::ALL.len() - VehicleBody::CIVILIAN.len(),
            2,
            "a family was added to ALL without a decision about the kerb"
        );

        // …and the DRAW, over two thousand kerb slots, which is far more than
        // the island has. A uniform draw over seven would put roughly 570 boats
        // in here.
        let mut seen = std::collections::BTreeSet::new();
        for k in 0..2_000u64 {
            let guid = Uuid::from_u64_pair(0x4B33, k);
            let body = catalogue_row(guid).body;
            assert!(
                VehicleBody::CIVILIAN.contains(&body),
                "kerb slot {k} drew a {}",
                body.name()
            );
            seen.insert(body.name());
        }
        // Vacuity guard: the draw really does reach all five, so "every one was
        // civilian" is not a statement about a constant.
        assert_eq!(seen.len(), 5, "the draw only reached {seen:?}");
    }

    /// hands out no ratio, is a car nothing can drive anywhere.
    #[test]
    fn every_catalogue_row_makes_torque_at_a_standstill() {
        use crate::vehicle::{ChassisState, RaycastVehicle, Vehicle, VehicleControls, WheelForce};
        for k in 0..40u64 {
            let guid = Uuid::from_u64_pair(0xCA7, k);
            let def = catalogue_row(guid);
            let rig = crate::vehicle::VehicleRig {
                chassis: guid,
                seat_local: crate::math::Vec3d::new(0.0, def.half_extents.y, 0.0),
                wheels: def
                    .wheel_mounts()
                    .into_iter()
                    .enumerate()
                    .map(|(i, m)| crate::vehicle::WheelMount {
                        guid: Uuid::from_u64_pair(0x7777, i as u64),
                        mount_local: m,
                        radius_m: def.wheel_radius_m,
                    })
                    .collect(),
                parts: Vec::new(),
            };
            let mut v = RaycastVehicle::new(rig);
            def.class.install(&mut v);
            let mass = 8.0
                * def.half_extents.x
                * def.half_extents.y
                * def.half_extents.z
                * def.density_kg_m3;
            let state = ChassisState {
                position: DVec3::new(0.0, crate::vehicle::resting_origin_y(&def, 0.0), 0.0),
                rotation: glam::DQuat::IDENTITY,
                linvel: DVec3::ZERO,
                angvel: DVec3::ZERO,
                mass_kg: mass,
                water_y: None,
            };
            let mut out: Vec<WheelForce> = Vec::new();
            // Ground every wheel at its own rest, so the model has a contact to
            // push against.
            let rest = v.suspension_rest_m();
            // **First, ninety steps of being PARKED** — which is what the
            // traffic step does to a `Full` car nobody is driving: the
            // handbrake, held. A model that latched anything over that would be
            // a car the player gets into and cannot drive away.
            v.control(VehicleControls {
                handbrake: true,
                ..Default::default()
            });
            for _ in 0..90 {
                for w in v.wheels_mut().iter_mut() {
                    w.contact = Some(crate::vehicle::WheelContact {
                        point: DVec3::ZERO,
                        normal: DVec3::Y,
                        distance_m: rest + def.wheel_radius_m - 0.08,
                    });
                }
                out.clear();
                v.solve(state, 1.0 / 60.0, &mut out);
            }
            v.control(VehicleControls {
                throttle: 1.0,
                ..Default::default()
            });
            for _ in 0..40 {
                for w in v.wheels_mut().iter_mut() {
                    w.contact = Some(crate::vehicle::WheelContact {
                        point: DVec3::ZERO,
                        normal: DVec3::Y,
                        distance_m: rest + def.wheel_radius_m - 0.08,
                    });
                }
                out.clear();
                v.solve(state, 1.0 / 60.0, &mut out);
            }
            let drive: f64 = out.iter().map(|f| f.force.x.abs() + f.force.z.abs()).sum();
            let load: f64 = v.wheels().iter().map(|w| w.load_n).sum();
            let omega: f64 = v.wheels().iter().map(|w| w.omega_rad_s.abs()).sum();
            assert!(
                drive > 1.0,
                "row {k} ({body:?}) makes {drive:.4} N at full throttle -- mass {mass:.0} kg, load {load:.0} N, omega {omega:.4}, gear {gear}, rest {rest:.3}",
                body = def.body,
                gear = v.gear(),
            );
        }
    }

    #[test]
    fn a_parked_cars_guid_is_its_own_place() {
        let a = parked_car_guid(DVec3::new(12.0, 3.0, -40.0));
        let b = parked_car_guid(DVec3::new(12.0, 3.0, -40.0));
        let c = parked_car_guid(DVec3::new(12.0, 3.0, -41.0));
        assert_eq!(a, b);
        assert_ne!(a, c);
        // **`audit:` VEH2b — and the HEIGHT is not in it.** A street's `y` is a
        // median over the blocks that bound it, and that median moves when any
        // one of them pages in — so a guid that folded Y in would re-mint every
        // car in a settlement in exactly the case `derive_parked`'s
        // carry-forward exists for. The doc has said so since the fix; this is
        // the arm that would notice it coming back.
        assert_eq!(a, parked_car_guid(DVec3::new(12.0, 131.5, -40.0)));
        assert_eq!(a, parked_car_guid(DVec3::new(12.0, f64::NAN, -40.0)));
        assert_eq!(derived_guids(&grid_streets()).len(), 2 * 2 * 12);
    }
}
