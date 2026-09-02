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

use inf_nav::lane::{right_of, LaneNetwork, LaneSpec, DEFAULT_LANE_WIDTH_M};
use inf_nav::{NavGraph, NavKind, NavNodeId, NavPath};

use crate::math::Vec2d;
use crate::society::{mix64, rect_gap, volume_sites, PAVEMENT_LATTICE_M, PAVEMENT_M};
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

/// The sign on a settlement street, km/h.
///
/// Thirty, because these are the streets a town's own residents walk across:
/// `inf_ecs::society` links two blocks' pavements straight over the gap
/// ([`crate::society::BLOCK_LINK_MAX_M`]), so every crossing in a settlement is
/// a place a pedestrian steps into the road. The island's *circuit* is a
/// different class and carries `inf_gis::default_speed_kmh`'s own numbers.
pub const STREET_SPEED_KMH: u32 = 30;

/// How far from a street's centreline a car parks at the kerb, metres.
///
/// Half the carriageway (3.5 m — one lane each way at
/// [`DEFAULT_LANE_WIDTH_M`]) plus a car's own half width and a hand's clearance.
/// It has to leave [`PAVEMENT_M`] free at the kerb, which is what
/// [`kerb_fits`] checks: on the narrowest street this engine plans (16 m) there
/// are 6.0 m between the centreline and the pavement and this uses 5.0 of them.
pub const KERB_PARK_OFFSET_M: f64 = 5.0;

/// Metres of kerb one parked car occupies.
///
/// A saloon is 4.4 m long and this engine's longest catalogue row is a 5.4 m
/// van, so fourteen metres is a car plus enough room to get out of the space —
/// which is what makes a kerb look parked rather than shunted. It also bounds
/// the population: a 100 m city block edge holds seven a side.
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

/// **The streets a level's blocks imply** — the derivation, as a value, so it
/// can be tested without a lane network and asserted against a plan.
///
/// The blocks are grouped by proximity ([`MAX_STREET_GAP_M`]) so two
/// settlements a kilometre apart do not imply a kilometre-wide street between
/// them; within a group each axis is reduced to its occupied intervals and the
/// gaps between consecutive intervals are the streets.
///
/// `O(blocks²)` for the grouping and `O(blocks log blocks)` for the rest, walked
/// in `Guid` order. It runs when the block set **changes**, never per step —
/// see [`TrafficRes::stamp`].
pub fn streets_of(world: &EcsWorld) -> Vec<Street> {
    let sites = volume_sites(world);
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
            let gap = rect_gap(
                sites[i].centre,
                sites[i].extent,
                sites[j].centre,
                sites[j].extent,
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
        let pad_y = {
            // The median would need a sort of its own; the mean of a levelled
            // pad is the same number to a centimetre and is one pass.
            let sum: f64 = members.iter().map(|&i| sites[i].pad_y).sum();
            sum / members.len() as f64
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
                sites[i].centre.x - sites[i].extent.x,
                sites[i].centre.x + sites[i].extent.x,
            )
        });
        let z_span = span(&|i| {
            (
                sites[i].centre.z - sites[i].extent.y,
                sites[i].centre.z + sites[i].extent.y,
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
                        (sites[i].centre.x, sites[i].extent.x)
                    } else {
                        (sites[i].centre.z, sites[i].extent.y)
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
                if !(gap >= MIN_STREET_GAP_M && gap <= MAX_STREET_GAP_M) {
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
                    y: pad_y,
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
        g.add_node(id, DVec3::new(p.x, y, p.y), NavKind::Street);
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
pub fn corner_speed_mps(path: &NavPath, s_m: f64, lookahead_m: f64) -> f64 {
    if !(lookahead_m > 0.0) || !s_m.is_finite() {
        return f64::INFINITY;
    }
    let a = path.direction_at(s_m);
    let b = path.direction_at(s_m + lookahead_m);
    // The ground-plane component of `a x b`, which is the Y one.
    let sin = (a.z * b.x - a.x * b.z).abs();
    if !(sin > 1.0e-6) {
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
    let bend = corner_speed_mps(view.path, view.s_m, lookahead);
    let ahead_of_us = match view.gap_m {
        Some(g) if g.is_finite() => {
            let clear = (g - STANDING_GAP_M).max(0.0);
            (2.0 * COMFORT_DECEL_MPS2 * clear).sqrt()
        }
        _ => f64::INFINITY,
    };
    // …and the end of the road. A car that has run out of lane stops at it
    // rather than driving off the end, on the same stopping-distance rule.
    let to_end = (view.path.length_m() - view.s_m).max(0.0);
    let endstop = (2.0 * COMFORT_DECEL_MPS2 * to_end).sqrt();
    let target = limit.min(bend).min(ahead_of_us).min(endstop);

    // ── 3. the pedal.
    let (fwd, handbrake) = if target <= STOPPED_MPS && v.abs() <= STOPPED_MPS {
        (0.0, true)
    } else {
        (((target - v) / SPEED_BAND_MPS).clamp(-1.0, 1.0), false)
    };

    // ── 4. the wheel.
    let aim = view.path.position_at(view.s_m + lookahead);
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
/// The count is geometry and not a setting: a hundred-metre block edge holds
/// seven a side. What decides whether a slot is *taken* is the caller's own
/// draw — see `inf_physics::d3::traffic`.
pub fn kerb_slots(streets: &[Street]) -> Vec<(DVec3, f64)> {
    let mut out = Vec::new();
    for (i, s) in streets.iter().enumerate() {
        if !kerb_fits(s.gap_m) {
            continue;
        }
        let d = DVec2::new(s.b.x - s.a.x, s.b.y - s.a.y);
        let len = (d.x * d.x + d.y * d.y).sqrt();
        if !(len > KERB_SLOT_M) {
            continue;
        }
        let dir = DVec3::new(d.x / len, 0.0, d.y / len);
        let r = right_of(dir);
        // Whole slots only, and a half-slot of clearance at each end, so a row
        // does not run into the junction it ends at.
        let n = ((len - KERB_SLOT_M) / KERB_SLOT_M) as usize;
        for side in [1.0f64, -1.0] {
            // The near side faces the way this line runs; the far side faces
            // back, because that is the direction traffic runs on it.
            let heading = if side > 0.0 { dir } else { -dir };
            let yaw = yaw_of_dir(heading);
            for k in 0..n {
                let along = KERB_SLOT_M * (k as f64 + 1.0);
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
    let hi = mix64(q(p.x) ^ mix64(q(p.z)));
    let lo = mix64(q(p.y) ^ mix64(PARKED_SALT));
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
        }
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
        let c = crate::vehicle::VehicleControls::from_intent(i.move_input, 0.0, i.handbrake);
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
            corner_speed_mps(&straight, 10.0, 20.0),
            f64::INFINITY,
            "a straight has no bend"
        );
        let bend = path(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]);
        let v = corner_speed_mps(&bend, 95.0, 20.0);
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
            gap_m: Some(f64::NAN),
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
        for (p, _) in &slots {
            // Every slot is exactly `KERB_PARK_OFFSET_M` from its own line —
            // which for this fixture is the axis it is NOT running along, so
            // one of the two coordinates is 5.0 to the bit.
            let off = if (p.x.abs() - KERB_PARK_OFFSET_M).abs() < 1e-9 {
                p.x.abs()
            } else {
                p.z.abs()
            };
            assert!((off - KERB_PARK_OFFSET_M).abs() < 1e-9, "{p:?} is at {off}");
            // Off the carriageway…
            assert!(off > DEFAULT_LANE_WIDTH_M, "{p:?} is in a lane");
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

    #[test]
    fn a_parked_cars_guid_is_its_own_place() {
        let a = parked_car_guid(DVec3::new(12.0, 3.0, -40.0));
        let b = parked_car_guid(DVec3::new(12.0, 3.0, -40.0));
        let c = parked_car_guid(DVec3::new(12.0, 3.0, -41.0));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(derived_guids(&grid_streets()).len(), 2 * 2 * 12);
    }
}
