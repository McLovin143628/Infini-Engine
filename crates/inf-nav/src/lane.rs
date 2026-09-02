//! The lane: **where a car drives**, as distinct from where a road *is*.
//!
//! # A spine is not a lane, and this is the only place that difference lives
//!
//! Every road producer in this tree hands over a **centreline**: `RoadGraph`'s
//! `RoadSegment::spine`, a settlement's street lines, the ribbon builder's
//! polyline. A centreline is the middle of the carriageway — the paint, if the
//! road has any — and a vehicle that drove it would sit astride the crown of the
//! road with its offside wheels in the oncoming traffic. Wave VEH1a parked seven
//! cars on that line and nothing moved, so nothing noticed.
//!
//! A lane is the centreline **offset to one side by half a lane width times an
//! odd number**, walked in one direction, with a speed limit on it. That is the
//! whole of it, and it is thirty lines of arithmetic — but it is thirty lines
//! that three producers would otherwise each write, and the day two of them
//! disagreed about which side of the road this engine drives on, the cars in one
//! settlement would meet the cars in another head on.
//!
//! # Right-hand traffic, once, here
//!
//! [`offset_path`] is signed **positive to the RIGHT of travel**, and a lane's
//! own offset is positive. So this engine drives on the right, the decision is
//! one sign in one function, and a left-hand-drive world is that sign flipped in
//! [`lanes_of_spine`] rather than a search through three crates. Named as a
//! bound rather than as a setting: nothing reads a handedness flag today, and a
//! flag nothing reads is a second opinion waiting to happen.
//!
//! # Why the id is a struct and not a packed integer
//!
//! [`NavNodeId`] is packed because a *producer* mints it and the domain tag has
//! to survive [`NavGraph::absorb`]'s id-equality join. A lane is not minted by a
//! producer — it is **derived from a directed edge**, and the edge already has
//! two ids. So [`LaneId`] is `(from, to, index)`: no hash, no collision, no
//! twenty-bit field to overflow, and `BTreeMap<LaneId, _>` orders by the source
//! data exactly as this crate's other walks do.
//!
//! # Determinism
//!
//! `+ - * / sqrt` only, `BTreeMap` order throughout, and successors listed in
//! [`NavGraph::edges_from`]'s own `(to, cost)` order. There is no trigonometry:
//! the lateral normal is a *rotation of the direction vector by ninety degrees*,
//! which is two negations, and the mitre is a reciprocal of a dot product. The
//! P14 law binds here for the reason it binds on [`crate::path`] — a lane's
//! metres become a car's `Transform` and therefore the replay trace.
//!
//! [`NavNodeId`]: crate::NavNodeId
//! [`NavGraph::absorb`]: crate::NavGraph::absorb
//! [`NavGraph::edges_from`]: crate::NavGraph::edges_from

use std::collections::BTreeMap;

use glam::DVec3;

use crate::graph::{NavEdge, NavGraph, NavKind, NavNodeId};
use crate::path::NavPath;

/// One lane's width, metres.
///
/// 3.5 m, which is `inf_gis::LANE_WIDTH_M` — the number the road *surface* is
/// already built at, so a lane derived here fits inside the ribbon drawn there.
/// It is restated rather than imported because `inf-nav` depends on nothing but
/// `glam` on purpose (see the crate docs), and
/// `the_lane_width_agrees_with_the_surface_it_is_drawn_on` in `inf-gis` holds
/// the two together.
pub const DEFAULT_LANE_WIDTH_M: f64 = 3.5;

/// How far a mitred offset may stretch a corner, as a multiple of the offset.
///
/// The same 4.0 the road ribbon mitres at, and for the same reason: at a sharp
/// enough bend the exact offset of two segments meets arbitrarily far from the
/// corner, so the stretch is **clamped**. A lane offset is 1.75 m where a
/// ribbon's is 7, so this clamp bites four times later here than it does there.
///
/// Clamped, and not bevelled: the corner keeps its one vertex, pushed out along
/// the bisector by at most this multiple. A clamped corner is therefore *inside*
/// the exact offset — the two lanes meeting there fall short of their true
/// intersection — which on a hairpin is a metre and on anything this engine
/// plans is nothing.
pub const MITER_LIMIT: f64 = 4.0;

/// The default speed limit a lane takes when its producer names none, km/h.
///
/// **Kilometres per hour, and converted at the point of use** — the units
/// doctrine's own carve-out, restated by `RoadSegment::speed_limit_kmh`: a
/// speed limit is a *sign*, not a physical quantity, and a sign says 50.
pub const DEFAULT_SPEED_LIMIT_KMH: u32 = 50;

/// Kilometres per hour to metres per second.
///
/// One division, in one place, so a controller and a gate cannot disagree about
/// what a fifty means.
#[inline]
pub fn kmh_to_mps(kmh: f64) -> f64 {
    kmh / 3.6
}

/// **Which lane** — a directed edge of the source graph, and an index across it.
///
/// Ordered by `(from, to, index)`, which is the source graph's own order, so
/// every walk over a [`LaneNetwork`] is a function of the network and not of a
/// hash seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaneId {
    /// The node this lane leaves.
    pub from: NavNodeId,
    /// The node it arrives at.
    pub to: NavNodeId,
    /// `0` is the lane nearest the centreline; `1` is one width further right.
    pub index: u8,
}

impl LaneId {
    /// The lane running the other way along the same edge, at the same index.
    ///
    /// Not necessarily a lane the network holds: a one-lane track has no
    /// opposing carriageway, and an edge whose reverse half was filtered out has
    /// no reverse lanes at all. Ask the network.
    pub fn opposing(self) -> Self {
        Self {
            from: self.to,
            to: self.from,
            index: self.index,
        }
    }
}

/// **One lane**: a path, walked one way, with a limit on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Lane {
    /// Which lane this is.
    pub id: LaneId,
    /// Its centreline, in world metres, in travel order.
    pub path: NavPath,
    /// The sign at the roadside, km/h. Converted where it is used.
    pub speed_limit_kmh: u32,
    /// How wide the lane is, metres — what a car needs to stay inside, and what
    /// the next lane out is offset by.
    pub width_m: f64,
    /// What the source edge was, so a caller can price a dirt track differently
    /// from a highway without a second lookup.
    pub kind: NavKind,
}

impl Lane {
    /// The limit as a speed, m/s.
    pub fn speed_limit_mps(&self) -> f64 {
        kmh_to_mps(f64::from(self.speed_limit_kmh))
    }

    /// Where the lane begins.
    pub fn entry(&self) -> DVec3 {
        self.path.points()[0]
    }

    /// Where it ends.
    pub fn exit(&self) -> DVec3 {
        let pts = self.path.points();
        pts[pts.len() - 1]
    }
}

/// **How many lanes a road carries, and how they are shared out** — the half of
/// a lane derivation the producer knows and this crate does not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaneSpec {
    /// Total lanes across the carriageway, both directions together. `0` is
    /// read as `1`.
    pub lane_count: u32,
    /// One lane's width, metres. Non-finite or non-positive takes
    /// [`DEFAULT_LANE_WIDTH_M`].
    pub width_m: f64,
    /// The sign, km/h. `0` takes [`DEFAULT_SPEED_LIMIT_KMH`].
    pub speed_limit_kmh: u32,
}

impl Default for LaneSpec {
    fn default() -> Self {
        Self {
            lane_count: 2,
            width_m: DEFAULT_LANE_WIDTH_M,
            speed_limit_kmh: DEFAULT_SPEED_LIMIT_KMH,
        }
    }
}

impl LaneSpec {
    /// The spec with every field sanitized — the one place a nonsense input
    /// becomes a usable number, so nothing downstream has to check twice.
    fn sane(self) -> (u32, f64, u32) {
        let lanes = self.lane_count.clamp(1, MAX_LANES_PER_EDGE);
        let width = if self.width_m.is_finite() && self.width_m > 0.0 {
            self.width_m
        } else {
            DEFAULT_LANE_WIDTH_M
        };
        let limit = if self.speed_limit_kmh == 0 {
            DEFAULT_SPEED_LIMIT_KMH
        } else {
            self.speed_limit_kmh
        };
        (lanes, width, limit)
    }

    /// **How many of `lane_count` run in the spine's own direction.**
    ///
    /// Half, rounded **up**, so a three-lane road is two out and one back — the
    /// asymmetry a real three-lane road has, resolved one way rather than
    /// refused. A single-lane road is one lane and it runs **on the spine**
    /// (see [`lanes_of_spine`]): a forestry track is not half a carriageway.
    pub fn forward_lanes(self) -> u32 {
        let (lanes, _, _) = self.sane();
        lanes.div_ceil(2)
    }
}

/// The most lanes one edge may carry.
///
/// Eight is two more than the widest carriageway this engine's road classes
/// name (`RoadKind::Highway` defaults to four), and it keeps
/// [`LaneId::index`] — a `u8` — comfortably inside its own range with the
/// per-direction split applied.
pub const MAX_LANES_PER_EDGE: u32 = 8;

/// **The unit vector ninety degrees to the RIGHT of a heading**, in the ground
/// plane.
///
/// `right = up × forward` with `up = +Y`, which for a heading `d` is
/// `(d.z, 0, -d.x)` — two component swaps and one negation, no trigonometry, and
/// the same handedness the vehicle model's own `ChassisState::basis` uses (a car
/// facing `+Z` has its right along `+X`, which is the side its driver gets out
/// of).
///
/// A vertical or degenerate heading has no right, and answers `+X` — the answer
/// a heading of `+Z` gives, so a caller that meets one gets a usable frame
/// rather than a zero it would divide by.
#[inline]
pub fn right_of(d: DVec3) -> DVec3 {
    let (x, z) = (d.x, d.z);
    let len = (x * x + z * z).sqrt();
    if !(len > 0.0) || !len.is_finite() {
        return DVec3::X;
    }
    DVec3::new(z / len, 0.0, -x / len)
}

/// **How much of an offset ran BACKWARDS**, metres — the fold measure.
///
/// # It is not "how much shorter it came out"
///
/// The first cut of this measured `spine.length - offset.length`, which is a
/// number every inside bend produces by construction: offsetting a curve toward
/// its centre shortens it, and that is correct geometry rather than a defect.
/// On a straight producer it reads zero and looks right; on `inf-gis`'s graded
/// road spines it would have read positive on every right-hand bend in the
/// island and called each one a fold. (Found by this wave's own adversarial
/// read, and the arms could not see it because they only offset straights.)
///
/// A **fold** is the polyline crossing itself, and its signature is a segment
/// whose direction has *reversed* against the spine segment it came from: the
/// mitre pushed the corner past its neighbour. So that is what is measured —
/// the summed length of the offset segments whose heading opposes their own
/// spine's. Zero for any offset that merely tightened.
pub fn fold_of(spine: &NavPath, offset: &NavPath) -> f64 {
    let (a, b) = (spine.points(), offset.points());
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }
    let mut folded = 0.0;
    for i in 0..a.len() - 1 {
        let (sx, sz) = (a[i + 1].x - a[i].x, a[i + 1].z - a[i].z);
        let (ox, oz) = (b[i + 1].x - b[i].x, b[i + 1].z - b[i].z);
        if sx * ox + sz * oz < 0.0 {
            folded += (ox * ox + oz * oz).sqrt();
        }
    }
    folded
}

/// **The path shifted `offset_m` metres to its own right**, corners mitred.
///
/// Positive is to the right of travel; negative is to the left. Y is carried
/// through untouched: a lane is beside the road, not above it, and the ground
/// under it is the ground the spine was snapped to.
///
/// # The mitre, and where it gives up
///
/// At an interior vertex the two adjacent segments have two different normals,
/// and the exact offset of both meets at the *mitre point* — the bisector,
/// stretched by `1 / cos(half the turn)`. That factor grows without bound as the
/// bend tightens, so it is clamped at [`MITER_LIMIT`] and the corner is cut
/// instead. Written as a reciprocal of a dot product rather than as a cosine,
/// because this crate has no trigonometry in it.
///
/// **A bend tighter than the offset is not offset correctly and cannot be.** An
/// inside corner whose turn radius is smaller than `offset_m` folds the polyline
/// through itself, which is a fact about offsetting polylines and not about this
/// implementation. Both producers in this tree are safe from it by construction
/// — a settlement grid turns at ninety degrees over a sixty-metre pitch, and a
/// road spine is planned on an eight-metre lattice with a graded turn radius —
/// and the day one is not, the symptom is a lane that doubles back on itself for
/// a metre or two, which [`LaneNetwork::worst_fold_m`] reports rather than hides.
pub fn offset_path(path: &NavPath, offset_m: f64) -> NavPath {
    let pts = path.points();
    if !offset_m.is_finite() || offset_m == 0.0 || pts.len() < 2 {
        return path.clone();
    }
    let n = pts.len();
    // Per-segment right normals, computed once: an interior vertex reads the two
    // that meet at it, so computing them inside the vertex loop would compute
    // every one of them twice.
    let normals: Vec<DVec3> = (0..n - 1).map(|i| right_of(pts[i + 1] - pts[i])).collect();
    let mut out: Vec<DVec3> = Vec::with_capacity(n);
    for (i, p) in pts.iter().enumerate() {
        let shift = if i == 0 {
            normals[0] * offset_m
        } else if i == n - 1 {
            normals[n - 2] * offset_m
        } else {
            let (a, b) = (normals[i - 1], normals[i]);
            let bisector = a + b;
            let len = (bisector.x * bisector.x + bisector.z * bisector.z).sqrt();
            if len > 0.0 {
                let unit = DVec3::new(bisector.x / len, 0.0, bisector.z / len);
                // `unit · a` is the cosine of half the turn. A U-turn drives it
                // to zero, which is what the clamp is for.
                let cos_half = unit.x * a.x + unit.z * a.z;
                let stretch = if cos_half > 1.0 / MITER_LIMIT {
                    1.0 / cos_half
                } else {
                    MITER_LIMIT
                };
                unit * (offset_m * stretch)
            } else {
                // The two segments double back exactly: there is no bisector, so
                // the corner keeps the incoming normal and the offset cuts it.
                a * offset_m
            }
        };
        out.push(DVec3::new(p.x + shift.x, p.y, p.z + shift.z));
    }
    NavPath::new(out)
}

/// **Every lane of one spine, in the spine's own direction.**
///
/// `spine` is the centreline in travel order. The answer is
/// [`LaneSpec::forward_lanes`] paths, index `0` nearest the crown of the road
/// and each subsequent one a width further right.
///
/// A **single-lane** road is the exception the doc on `forward_lanes` names: one
/// lane, offset by nothing, running down the spine itself. A forestry track, a
/// farm lane and a one-car alley are one carriageway and not half of one, and
/// offsetting them by 1.75 m would put every vehicle in the ditch.
pub fn lanes_of_spine(spine: &NavPath, spec: LaneSpec) -> Vec<(u8, NavPath)> {
    let (lanes, width, _) = spec.sane();
    if lanes <= 1 {
        return vec![(0, spine.clone())];
    }
    (0..spec.forward_lanes())
        .map(|k| {
            let offset = width * (f64::from(k) + 0.5);
            (k as u8, offset_path(spine, offset))
        })
        .collect()
}

/// **The carriageway network**: every lane, and what each one leads to.
///
/// Built from a [`NavGraph`], because a junction is a *node two edges share* and
/// no amount of geometry recovers that reliably: two lanes that meet at a T are
/// offset to two different sides of two different headings, so their endpoints
/// are metres apart even though the roads touch exactly. The graph already knows
/// they touch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LaneNetwork {
    lanes: BTreeMap<LaneId, Lane>,
    next: BTreeMap<LaneId, Vec<LaneId>>,
    worst_fold_m: f64,
}

impl LaneNetwork {
    /// **Derive the lanes of a graph**, one [`LaneSpec`] per directed edge.
    ///
    /// The spec closure is handed the edge and the node it leaves, so a producer
    /// can widen a highway and narrow an alley from whatever it knows. Both
    /// halves of an undirected link are visited, which is what gives a two-way
    /// road its opposing carriageway: the `b → a` half's own "right" is the
    /// `a → b` half's left, so nothing here has to know that a road has two
    /// sides.
    ///
    /// A spec answering `None` **omits that direction entirely** — the door a
    /// one-way street goes through, and the reason a `Rail` edge carries no
    /// lanes at all.
    ///
    /// `O(edges × lanes × points)`, walked in `BTreeMap` order.
    pub fn from_graph(
        graph: &NavGraph,
        mut spec_of: impl FnMut(NavNodeId, &NavEdge) -> Option<LaneSpec>,
    ) -> Self {
        let mut out = Self::default();
        let ids: Vec<NavNodeId> = graph.nodes().map(|n| n.id).collect();
        for from in &ids {
            let Some(a) = graph.node(*from).map(|n| n.position) else {
                continue;
            };
            for edge in graph.edges_from(*from) {
                let Some(spec) = spec_of(*from, edge) else {
                    continue;
                };
                let Some(b) = graph.node(edge.to).map(|n| n.position) else {
                    continue;
                };
                let mut chain = Vec::with_capacity(edge.via.len() + 2);
                chain.push(a);
                chain.extend_from_slice(&edge.via);
                chain.push(b);
                let spine = NavPath::new(chain);
                if spine.is_stand() {
                    continue;
                }
                let (_, width, limit) = spec.sane();
                for (index, path) in lanes_of_spine(&spine, spec) {
                    let id = LaneId {
                        from: *from,
                        to: edge.to,
                        index,
                    };
                    let fold = fold_of(&spine, &path);
                    if fold > out.worst_fold_m {
                        out.worst_fold_m = fold;
                    }
                    out.lanes.insert(
                        id,
                        Lane {
                            id,
                            path,
                            speed_limit_kmh: limit,
                            width_m: width,
                            kind: edge.kind,
                        },
                    );
                }
            }
        }
        out.join(graph);
        out
    }

    /// **What each lane leads to** — the junction half, run once at build time.
    ///
    /// A lane arriving at `b` from `a` may leave `b` on any edge that is not the
    /// one it came in on, at any index that edge carries. The U-turn is refused
    /// **except at a dead end**, where it is the only way out and refusing it
    /// would strand every car that ever drove down a cul-de-sac.
    ///
    /// Successors are listed in [`NavGraph::edges_from`]'s `(to, cost)` order
    /// then by index, so a chooser that takes "the first" takes the same one on
    /// two hosts.
    ///
    /// [`NavGraph::edges_from`]: crate::NavGraph::edges_from
    fn join(&mut self, graph: &NavGraph) {
        let ids: Vec<LaneId> = self.lanes.keys().copied().collect();
        for id in ids {
            let out_edges = graph.edges_from(id.to);
            let dead_end = out_edges.len() <= 1;
            let mut succ: Vec<LaneId> = Vec::new();
            for edge in out_edges {
                if edge.to == id.from && !dead_end {
                    continue;
                }
                for index in 0..=u8::MAX {
                    let candidate = LaneId {
                        from: id.to,
                        to: edge.to,
                        index,
                    };
                    if self.lanes.contains_key(&candidate) {
                        succ.push(candidate);
                    } else {
                        break;
                    }
                }
            }
            if !succ.is_empty() {
                self.next.insert(id, succ);
            }
        }
    }

    /// How many lanes.
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Whether there is nowhere at all to drive.
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    /// Every lane, in [`LaneId`] order.
    pub fn lanes(&self) -> impl Iterator<Item = &Lane> {
        self.lanes.values()
    }

    /// One lane.
    pub fn lane(&self, id: LaneId) -> Option<&Lane> {
        self.lanes.get(&id)
    }

    /// The lanes this one leads into, in the order [`join`](Self::join)
    /// documents. Empty for a lane that leads nowhere.
    pub fn successors(&self, id: LaneId) -> &[LaneId] {
        self.next.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// **The worst fold any offset produced**, metres — a lane that came out
    /// shorter than its spine because the polyline crossed itself.
    ///
    /// Zero on both producers in this tree. Reported so a level whose roads bend
    /// tighter than a lane is wide says so out loud, rather than shipping a lane
    /// with a kink a car will steer into.
    pub fn worst_fold_m(&self) -> f64 {
        self.worst_fold_m
    }

    /// **The lane nearest `p`, and how far along it that is** — how a car that
    /// has just been placed finds the lane it is standing in.
    ///
    /// Measured to the lane's own centreline in three dimensions, ties broken by
    /// [`LaneId`] order (a strict `<`, so the `BTreeMap` walk decides). `O(lanes
    /// × points)`: this is a placement query, not a per-step one.
    pub fn nearest(&self, p: DVec3) -> Option<(LaneId, f64, f64)> {
        let mut best: Option<(LaneId, f64, f64)> = None;
        for lane in self.lanes.values() {
            let proj = lane.path.project(p);
            // A NaN distance is skipped rather than accepted. The obvious
            // spelling — `None => true` and then `proj.distance_m < d` — LATCHES
            // one: a NaN taken as the first candidate compares false against
            // every later lane and wins for ever, which turns the documented
            // `LaneId` tie-break into "whichever lane sorted first".
            if !proj.distance_m.is_finite() {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, _, d)) => proj.distance_m < d,
            };
            if better {
                best = Some((lane.id, proj.s_m, proj.distance_m));
            }
        }
        best
    }

    /// **The lane route along a node route** — a chain of lanes, one per hop, at
    /// one index the whole way.
    ///
    /// `index` is clamped to whatever each hop actually carries, so a route that
    /// leaves a dual carriageway for a single-track lane keeps going instead of
    /// stopping at the join. **There is no lane change**: the car stays as far
    /// right as `index` puts it for the whole journey, which is wave VEH2b's
    /// stated v1 bound (no overtaking) expressed as geometry rather than as a
    /// rule in the controller.
    ///
    /// Hops the network has no lane for are **skipped**, and the chain closes
    /// over the gap — the same "a refusal is a value" the rest of this crate
    /// takes. A caller that needs to know asks for the length it got back.
    pub fn lane_route(&self, nodes: &[NavNodeId], index: u8) -> Vec<LaneId> {
        let mut out = Vec::new();
        for w in nodes.windows(2) {
            let mut chosen = None;
            for k in (0..=index).rev() {
                let id = LaneId {
                    from: w[0],
                    to: w[1],
                    index: k,
                };
                if self.lanes.contains_key(&id) {
                    chosen = Some(id);
                    break;
                }
            }
            if let Some(id) = chosen {
                out.push(id);
            }
        }
        out
    }

    /// **One drivable path out of a lane route** — the lanes' own polylines,
    /// end to end.
    ///
    /// The junction between two lanes is a **chord**, not a fillet: lane A ends
    /// offset to the right of its own heading and lane B starts offset to the
    /// right of B's, so on a ninety-degree turn the two ends are about
    /// `offset x sqrt(2)` apart and the chain cuts straight across. That is
    /// deliberate for v1 and it is what a real car does anyway — the steering
    /// controller aims at a point *ahead* on this path, so a corner it cannot
    /// take squarely it rounds. A proper turn geometry (an arc struck on the
    /// kerb radius, per leg pair) is the same road-modelling project
    /// `inf_gis::roads` names as out of scope for its junction fans.
    pub fn path_of(&self, route: &[LaneId]) -> NavPath {
        let mut pts: Vec<DVec3> = Vec::new();
        for id in route {
            let Some(lane) = self.lanes.get(id) else {
                continue;
            };
            pts.extend_from_slice(lane.path.points());
        }
        NavPath::new(pts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, z: f64) -> DVec3 {
        DVec3::new(x, 0.0, z)
    }

    /// A straight run east, so "right" is unambiguous: heading `+X`, right is
    /// `-Z`.
    #[test]
    fn right_of_a_heading_is_the_side_a_driver_gets_out_of() {
        assert_eq!(right_of(DVec3::Z), DVec3::X);
        assert_eq!(right_of(DVec3::X), DVec3::new(0.0, 0.0, -1.0));
        assert_eq!(right_of(DVec3::new(0.0, 5.0, 0.0)), DVec3::X);
        assert_eq!(right_of(DVec3::ZERO), DVec3::X);
    }

    #[test]
    fn a_straight_offset_moves_the_whole_line_and_keeps_its_length() {
        let path = NavPath::new([p(0.0, 0.0), p(100.0, 0.0)]);
        let right = offset_path(&path, 1.75);
        assert_eq!(right.points()[0], DVec3::new(0.0, 0.0, -1.75));
        assert_eq!(right.points()[1], DVec3::new(100.0, 0.0, -1.75));
        assert_eq!(right.length_m(), 100.0);
    }

    /// The Y is the ground's, not the offset's: a lane is beside the road.
    #[test]
    fn an_offset_carries_the_height_through() {
        let path = NavPath::new([DVec3::new(0.0, 12.0, 0.0), DVec3::new(10.0, 18.0, 0.0)]);
        let out = offset_path(&path, 2.0);
        assert_eq!(out.points()[0].y, 12.0);
        assert_eq!(out.points()[1].y, 18.0);
    }

    /// The whole point of the mitre: the corner of an offset right angle sits on
    /// the bisector at `offset x sqrt(2)`, not at `offset`.
    #[test]
    fn a_right_angle_mitres_onto_its_bisector() {
        // East then north(-Z): a LEFT turn. Offsetting right by 2 m puts the
        // corner further out.
        let path = NavPath::new([p(0.0, 0.0), p(10.0, 0.0), p(10.0, -10.0)]);
        let out = offset_path(&path, 2.0);
        let corner = out.points()[1];
        let want = 2.0 * 2.0f64.sqrt();
        let d = ((corner.x - 10.0).powi(2) + (corner.z - 0.0f64).powi(2)).sqrt();
        assert!(
            (d - want).abs() < 1e-9,
            "corner {corner:?} is {d} from the vertex, wanted {want}"
        );
    }

    /// A doubling-back polyline is where the exact offset runs to infinity; the
    /// clamp is what keeps the answer finite.
    #[test]
    fn a_hairpin_is_clamped_rather_than_sent_to_infinity() {
        let path = NavPath::new([p(0.0, 0.0), p(10.0, 0.0), p(0.0, 0.001)]);
        let out = offset_path(&path, 2.0);
        for q in out.points() {
            assert!(q.is_finite(), "{q:?}");
            assert!(q.length() < 100.0, "{q:?} ran away");
        }
    }

    #[test]
    fn a_single_lane_road_runs_down_its_own_spine() {
        let spine = NavPath::new([p(0.0, 0.0), p(50.0, 0.0)]);
        let lanes = lanes_of_spine(
            &spine,
            LaneSpec {
                lane_count: 1,
                ..Default::default()
            },
        );
        assert_eq!(lanes.len(), 1);
        assert_eq!(lanes[0].1.points(), spine.points());
    }

    #[test]
    fn a_four_lane_road_gives_two_each_way_at_odd_half_widths() {
        let spine = NavPath::new([p(0.0, 0.0), p(50.0, 0.0)]);
        let spec = LaneSpec {
            lane_count: 4,
            width_m: 3.5,
            speed_limit_kmh: 80,
        };
        assert_eq!(spec.forward_lanes(), 2);
        let lanes = lanes_of_spine(&spine, spec);
        assert_eq!(lanes.len(), 2);
        // Heading +X, right is -Z.
        assert_eq!(lanes[0].1.points()[0].z, -1.75);
        assert_eq!(lanes[1].1.points()[0].z, -5.25);
    }

    /// Three lanes is two out and one back, decided rather than refused.
    #[test]
    fn an_odd_lane_count_puts_the_spare_lane_outbound() {
        let spec = LaneSpec {
            lane_count: 3,
            ..Default::default()
        };
        assert_eq!(spec.forward_lanes(), 2);
    }

    fn cross() -> NavGraph {
        // A plus: centre at the origin, four arms 100 m out.
        let mut g = NavGraph::new();
        g.add_node(0, p(0.0, 0.0), NavKind::Street);
        g.add_node(1, p(100.0, 0.0), NavKind::Street);
        g.add_node(2, p(-100.0, 0.0), NavKind::Street);
        g.add_node(3, p(0.0, 100.0), NavKind::Street);
        g.add_node(4, p(0.0, -100.0), NavKind::Street);
        for arm in 1..=4 {
            g.link(0, arm, NavKind::Street, vec![]);
        }
        g
    }

    /// Both halves of every undirected link become lanes, so a two-way street
    /// has an opposing carriageway without this crate knowing what "opposing"
    /// means.
    #[test]
    fn every_directed_half_carries_its_own_lane() {
        let net = LaneNetwork::from_graph(&cross(), |_, _| Some(LaneSpec::default()));
        // Four arms, both ways, one forward lane each = 8.
        assert_eq!(net.len(), 8);
        assert_eq!(net.worst_fold_m(), 0.0);
        let out = net
            .lane(LaneId {
                from: 0,
                to: 1,
                index: 0,
            })
            .expect("the east lane");
        let back = net
            .lane(LaneId {
                from: 1,
                to: 0,
                index: 0,
            })
            .expect("the west lane");
        // Heading +X the lane is at z = -1.75; heading -X it is at z = +1.75.
        assert_eq!(out.entry().z, -1.75);
        assert_eq!(back.entry().z, 1.75);
        assert_eq!(out.speed_limit_kmh, DEFAULT_SPEED_LIMIT_KMH);
        assert!((out.speed_limit_mps() - 50.0 / 3.6).abs() < 1e-12);
    }

    /// The junction rule, and the whole reason the network is built from a graph.
    #[test]
    fn a_lane_leads_everywhere_but_back_the_way_it_came() {
        let net = LaneNetwork::from_graph(&cross(), |_, _| Some(LaneSpec::default()));
        let arriving = LaneId {
            from: 1,
            to: 0,
            index: 0,
        };
        let succ = net.successors(arriving);
        assert_eq!(succ.len(), 3, "{succ:?}");
        assert!(
            !succ.iter().any(|s| s.to == 1),
            "a U-turn at a four-way: {succ:?}"
        );
    }

    /// …except at a dead end, where turning round is the only way out.
    #[test]
    fn a_cul_de_sac_permits_the_only_turn_it_has() {
        let mut g = NavGraph::new();
        g.add_node(0, p(0.0, 0.0), NavKind::Street);
        g.add_node(1, p(60.0, 0.0), NavKind::Street);
        g.link(0, 1, NavKind::Street, vec![]);
        let net = LaneNetwork::from_graph(&g, |_, _| Some(LaneSpec::default()));
        let succ = net.successors(LaneId {
            from: 0,
            to: 1,
            index: 0,
        });
        assert_eq!(
            succ,
            [LaneId {
                from: 1,
                to: 0,
                index: 0
            }]
        );
    }

    /// A producer that refuses a direction gets no lane there — the one-way door.
    #[test]
    fn a_refused_direction_has_no_carriageway() {
        let net = LaneNetwork::from_graph(&cross(), |from, edge| {
            (from == 0 || edge.to != 0).then(LaneSpec::default)
        });
        assert_eq!(net.len(), 4);
        assert!(net
            .lane(LaneId {
                from: 1,
                to: 0,
                index: 0
            })
            .is_none());
    }

    #[test]
    fn a_route_over_nodes_becomes_a_route_over_lanes_and_one_path() {
        let net = LaneNetwork::from_graph(&cross(), |_, _| Some(LaneSpec::default()));
        let route = net.lane_route(&[2, 0, 1], 0);
        assert_eq!(
            route,
            [
                LaneId {
                    from: 2,
                    to: 0,
                    index: 0
                },
                LaneId {
                    from: 0,
                    to: 1,
                    index: 0
                },
            ]
        );
        let path = net.path_of(&route);
        // 100 m in, 100 m out, plus the chord across the junction.
        assert!(path.length_m() >= 200.0, "{}", path.length_m());
        assert!(path.length_m() < 205.0, "{}", path.length_m());
    }

    /// A lane index the hop does not carry falls back rather than dropping the
    /// hop, so a dual carriageway meeting a lane keeps its route.
    #[test]
    fn an_index_wider_than_the_road_falls_back_to_one_that_fits() {
        let mut g = NavGraph::new();
        g.add_node(0, p(0.0, 0.0), NavKind::Road);
        g.add_node(1, p(100.0, 0.0), NavKind::Road);
        g.add_node(2, p(200.0, 0.0), NavKind::Road);
        g.link(0, 1, NavKind::Road, vec![]);
        g.link(1, 2, NavKind::Road, vec![]);
        let net = LaneNetwork::from_graph(&g, |from, _| {
            Some(LaneSpec {
                lane_count: if from == 0 { 4 } else { 2 },
                ..Default::default()
            })
        });
        let route = net.lane_route(&[0, 1, 2], 1);
        assert_eq!(route.len(), 2);
        assert_eq!(route[0].index, 1);
        assert_eq!(route[1].index, 0);
    }

    /// The network is a function of the graph, not of the order it was walked.
    #[test]
    fn the_network_is_a_function_of_the_graph() {
        let a = LaneNetwork::from_graph(&cross(), |_, _| Some(LaneSpec::default()));
        let b = LaneNetwork::from_graph(&cross(), |_, _| Some(LaneSpec::default()));
        assert_eq!(a, b);
        let ids: Vec<LaneId> = a.lanes().map(|l| l.id).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted, "the walk is not in id order");
    }

    #[test]
    fn nearest_finds_the_carriageway_a_car_is_standing_in() {
        let net = LaneNetwork::from_graph(&cross(), |_, _| Some(LaneSpec::default()));
        // 50 m east, 1.75 m to the south — dead centre of the eastbound lane.
        let (id, s, d) = net.nearest(p(50.0, -1.75)).expect("a lane");
        assert_eq!(
            id,
            LaneId {
                from: 0,
                to: 1,
                index: 0
            }
        );
        assert!((s - 50.0).abs() < 1e-9, "{s}");
        assert!(d < 1e-9, "{d}");
    }

    /// **An inside bend is not a fold.** The measure the first cut used —
    /// length lost — reads positive on every curve offset toward its centre,
    /// which is correct geometry, and would have called every bend on the
    /// island a defect.
    #[test]
    fn offsetting_a_bend_inwards_is_not_a_fold() {
        // A right turn: offsetting to the right shortens the inside line.
        let bend = NavPath::new([p(0.0, 0.0), p(50.0, 0.0), p(50.0, 50.0)]);
        let inside = offset_path(&bend, -3.0);
        assert!(
            inside.length_m() < bend.length_m(),
            "the inside of a bend is shorter, or this fixture bends the other way"
        );
        assert_eq!(
            fold_of(&bend, &inside),
            0.0,
            "an inside bend read as a fold"
        );
        assert_eq!(fold_of(&bend, &offset_path(&bend, 3.0)), 0.0);
    }

    /// …and a real fold does read. A dogleg tighter than the offset pushes one
    /// segment past its neighbour and it comes out pointing backwards.
    #[test]
    fn a_connector_shorter_than_the_offset_folds_and_says_so() {
        // A U-turn with a ONE-METRE connector, turned so that the right-hand
        // offset is the INSIDE of it. Offset six metres, the connector's own
        // segment comes out pointing the other way — which is the polyline
        // crossing itself, and is the only thing `fold_of` calls a fold.
        let u = NavPath::new([p(0.0, 0.0), p(20.0, 0.0), p(20.0, -1.0), p(0.0, -1.0)]);
        let out = offset_path(&u, 6.0);
        assert!(
            fold_of(&u, &out) > 0.0,
            "a one-metre connector offset by six metres did not fold: {:?}",
            out.points()
        );
        // …and the network reports it rather than hiding it.
        let mut g = NavGraph::new();
        g.add_node(0, p(0.0, 0.0), NavKind::Road);
        g.add_node(1, p(0.0, -1.0), NavKind::Road);
        g.link(0, 1, NavKind::Road, vec![p(20.0, 0.0), p(20.0, -1.0)]);
        let net = LaneNetwork::from_graph(&g, |_, _| {
            Some(LaneSpec {
                lane_count: 8,
                width_m: 3.0,
                ..Default::default()
            })
        });
        assert!(net.worst_fold_m() > 0.0, "{}", net.worst_fold_m());
    }

    /// A NaN projection must not win the nearest query for ever.
    #[test]
    fn a_non_finite_query_point_finds_nothing_rather_than_the_first_lane() {
        let net = LaneNetwork::from_graph(&cross(), |_, _| Some(LaneSpec::default()));
        assert!(net.nearest(DVec3::new(f64::NAN, 0.0, 0.0)).is_none());
        // …and a finite one still answers.
        assert!(net.nearest(p(50.0, -1.75)).is_some());
    }

    #[test]
    fn a_nonsense_spec_is_sanitized_rather_than_refused() {
        let spine = NavPath::new([p(0.0, 0.0), p(10.0, 0.0)]);
        let lanes = lanes_of_spine(
            &spine,
            LaneSpec {
                lane_count: 0,
                width_m: f64::NAN,
                speed_limit_kmh: 0,
            },
        );
        assert_eq!(lanes.len(), 1);
        let spec = LaneSpec {
            lane_count: 9_999,
            width_m: -1.0,
            speed_limit_kmh: 0,
        };
        assert_eq!(spec.forward_lanes(), MAX_LANES_PER_EDGE / 2);
        let (_, w, l) = spec.sane();
        assert_eq!(w, DEFAULT_LANE_WIDTH_M);
        assert_eq!(l, DEFAULT_SPEED_LIMIT_KMH);
    }
}
